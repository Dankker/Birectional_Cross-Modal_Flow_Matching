#!/usr/bin/env python
"""Train paper-faithful variational BVFM on a released FlowTok checkpoint.

There is one canonically oriented field v(z_t, t, z_v), trained on
text -> image trajectories.  T2I integrates it from 0 to 1 and I2T integrates
the exact same field from 1 to 0.  Direction is never passed to the model.

The BVFM objective follows Eq. (17): one posterior-sampled FM reconstruction
plus beta KL(q||p_text) + (1-beta) KL(q||p_image).  An inference-matched
reverse rollout supplies the caption objective analogous to the paper's CTC
term.  The released field is frozen and augmented by shared zero-initialized
z_v residuals; a released-field distillation loss protects T2I quality.
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch.utils.data import DataLoader, DistributedSampler


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from bvfm_image.common import (  # noqa: E402
    CocoImageCaptionDataset,
    ar_caption_loss,
    barrier,
    bleu4,
    clip_text_features,
    encode_image_latents,
    flow_interp,
    ids_to_text,
    is_dist,
    load_config,
    load_val_images,
    logit_normal_time,
    setup_dist,
)
from bvfm_image.model_factory import build_field  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--resume", default="auto", choices=["auto", "none"])
    return parser.parse_args()


def build_models(config, device):
    """Load released FlowTok with only the new BVFM residual keys missing."""
    import open_clip
    from libs.flowtitok import FlowTiTok

    nnet = build_field(config)
    state = torch.load(
        config.nnet_path, map_location="cpu", weights_only=True, mmap=True)
    incompatible = nnet.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        key for key in missing
        if key.startswith("bvfm_variation_projector.")
        or key.startswith("bvfm_adapters.")
        or key.startswith("bvfm_output_adapter.")
    }
    if missing != allowed_missing or unexpected:
        raise RuntimeError(
            "Released FlowTok load mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}")
    del state
    if not nnet.use_bvfm_condition:
        raise RuntimeError("Config did not enable BVFM variation conditioning")
    if nnet.use_task_condition or nnet.use_task_adapters:
        raise RuntimeError("BVFM shared field must not use a direction/task id")
    nnet.to(device)

    try:
        clip_encoder, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336-quickgelu", pretrained="openai")
        clip_tokenizer = open_clip.get_tokenizer(
            "ViT-L-14-336-quickgelu")
    except Exception:
        clip_encoder, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336", pretrained="openai")
        clip_tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
    del clip_encoder.visual
    clip_encoder.transformer.batch_first = False
    clip_encoder.eval().requires_grad_(False).to(device)

    autoencoder = FlowTiTok(config)
    tokenizer_state = torch.load(
        config.tokenizer_checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True)
    autoencoder.load_state_dict(tokenizer_state)
    del tokenizer_state
    autoencoder.eval().requires_grad_(False).to(device)
    return nnet, clip_encoder, clip_tokenizer, autoencoder


def make_released_reference(nnet):
    """Immutable BF16 copy of the released, non-variational field."""
    reference = copy.deepcopy(nnet)
    reference.context_encoder = nn.Identity()
    reference.context_projector = nn.Identity()
    reference.t2t_temperature = None
    reference.bvfm_variation_projector = None
    reference.bvfm_adapters = nn.ModuleList()
    reference.bvfm_output_adapter = None
    reference.use_bvfm_condition = False
    reference.eval().requires_grad_(False)
    reference.to(dtype=torch.bfloat16)
    for block in reference.blocks:
        block.forward = block._forward
    return reference


def integrate_shared(field, source, variation_latent, steps, reverse=False):
    """Euler integration of the same conditional field in either interval."""
    batch = source.shape[0]
    device = source.device
    if reverse:
        grid = torch.linspace(1.0 - 1e-5, 0.0, steps + 1, device=device)
    else:
        grid = torch.linspace(0.0, 1.0 - 1e-5, steps + 1, device=device)
    state = source
    indicator = torch.zeros(batch, dtype=torch.bool, device=device)
    for index in range(steps):
        velocity = field(
            state,
            t=grid[index].expand(batch),
            null_indicator=indicator,
            variation_latent=variation_latent)[0]
        state = state + velocity.float() * (
            grid[index + 1] - grid[index])
    return state


def integrate_shared_train(
        field, source, variation_latent, steps, reverse=False):
    """Differentiable inference-matched shared-field rollout."""
    batch = source.shape[0]
    device = source.device
    if reverse:
        grid = torch.linspace(1.0 - 1e-5, 0.0, steps + 1, device=device)
    else:
        grid = torch.linspace(0.0, 1.0 - 1e-5, steps + 1, device=device)
    state = source
    indicator = torch.zeros(batch, dtype=torch.bool, device=device)

    def field_velocity(current, time, z_v):
        return field(
            current,
            t=time,
            null_indicator=indicator,
            variation_latent=z_v)[0]

    for index in range(steps):
        velocity = gradient_checkpoint(
            field_velocity,
            state,
            grid[index].expand(batch),
            variation_latent,
            use_reentrant=False,
            preserve_rng_state=False)
        state = state + velocity.float() * (
            grid[index + 1] - grid[index])
    return state


@torch.no_grad()
def integrate_t2i_cfg(
        field, source, variation_latent, steps, cfg):
    batch = source.shape[0]
    device = source.device
    grid = torch.linspace(0.0, 1.0 - 1e-5, steps + 1, device=device)
    state = source
    cond = torch.zeros(batch, dtype=torch.bool, device=device)
    uncond = torch.ones(batch, dtype=torch.bool, device=device)
    for index in range(steps):
        time = grid[index].expand(batch)
        v_cond = field(
            state,
            t=time,
            null_indicator=cond,
            variation_latent=variation_latent)[0]
        v_uncond = field(
            state,
            t=time,
            null_indicator=uncond,
            variation_latent=variation_latent)[0]
        velocity = v_uncond + float(cfg) * (v_cond - v_uncond)
        state = state + velocity.float() * (
            grid[index + 1] - grid[index])
    return state


@torch.no_grad()
def integrate_released_t2i(field, source, steps, cfg):
    batch = source.shape[0]
    device = source.device
    grid = torch.linspace(0.0, 1.0 - 1e-5, steps + 1, device=device)
    state = source
    cond = torch.zeros(batch, dtype=torch.bool, device=device)
    uncond = torch.ones(batch, dtype=torch.bool, device=device)
    for index in range(steps):
        time = grid[index].expand(batch)
        v_cond = field(
            state, t=time, null_indicator=cond)[0]
        v_uncond = field(
            state, t=time, null_indicator=uncond)[0]
        velocity = v_uncond + float(cfg) * (v_cond - v_uncond)
        state = state + velocity.float() * (
            grid[index + 1] - grid[index])
    return state


@torch.no_grad()
def collect_shared_trajectory(
        field, source, variation_latent, steps, reverse=False, cfg=None):
    """Return every Euler state for PCA diagnostics.

    ``cfg`` is used only for the forward T2I arm. Passing ``None`` follows
    the ordinary conditional field, which is the inference path used by I2T.
    The model still receives no direction/task id.
    """
    batch = source.shape[0]
    device = source.device
    if reverse:
        grid = torch.linspace(1.0 - 1e-5, 0.0, steps + 1, device=device)
    else:
        grid = torch.linspace(0.0, 1.0 - 1e-5, steps + 1, device=device)
    state = source.float()
    states = [state.detach()]
    cond = torch.zeros(batch, dtype=torch.bool, device=device)
    uncond = torch.ones(batch, dtype=torch.bool, device=device)
    for index in range(steps):
        time = grid[index].expand(batch)
        if cfg is None:
            velocity = field(
                state,
                t=time,
                null_indicator=cond,
                variation_latent=variation_latent)[0]
        else:
            v_cond = field(
                state,
                t=time,
                null_indicator=cond,
                variation_latent=variation_latent)[0]
            v_uncond = field(
                state,
                t=time,
                null_indicator=uncond,
                variation_latent=variation_latent)[0]
            velocity = v_uncond + float(cfg) * (v_cond - v_uncond)
        state = state + velocity.float() * (
            grid[index + 1] - grid[index])
        states.append(state.detach())
    return torch.stack(states, dim=0)


def _fit_shared_pca(*batches):
    """Fit a deterministic two-dimensional PCA basis on flattened samples."""
    matrix = torch.cat([
        batch.detach().float().reshape(batch.shape[0], -1).cpu()
        for batch in batches
    ], dim=0)
    center = matrix.mean(dim=0, keepdim=True)
    _, _, components = torch.linalg.svd(
        matrix - center, full_matrices=False)
    components = components[:2].transpose(0, 1).contiguous()
    # SVD signs are arbitrary. Fix each sign so plots remain comparable when
    # the validation endpoints and principal axes have not changed.
    for axis in range(components.shape[1]):
        pivot = components[:, axis].abs().argmax()
        if components[pivot, axis] < 0:
            components[:, axis].mul_(-1.0)
    return center, components


def _project_shared_pca(states, center, components):
    leading_shape = states.shape[:-2]
    flat = states.detach().float().reshape(-1, center.shape[1]).cpu()
    projected = (flat - center).matmul(components)
    return projected.reshape(*leading_shape, 2).numpy()


def _batch_variance_energy(states):
    flat = states.detach().float().reshape(states.shape[0], -1)
    return flat.var(dim=0, unbiased=False).mean()


@torch.no_grad()
def save_variational_pca_demos(
        step, config, field, variational, text_latent, image_latent,
        output_dir, text_target_latent=None):
    """Visualize repeated z_v samples for one fixed paired example.

    The endpoint bank is used only to define a stable shared PCA basis and a
    faint target-manifold background. Every colored rollout in the four main
    panels starts from the same example; only z_v changes between samples.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    count = min(
        int(config.diag.get("pca_pairs", 16)),
        text_latent.shape[0], image_latent.shape[0])
    if count < 2:
        return {}
    sample_count = int(config.diag.get("pca_samples", 32))
    if sample_count < 2:
        raise ValueError("diag.pca_samples must be at least 2")
    example_index = min(
        int(config.diag.get("pca_example_index", 0)), count - 1)
    steps = int(config.diag.get("pca_steps", config.diag.i2t_steps))
    cfg = float(config.diag.get("pca_t2i_cfg", config.diag.t2i_cfg))
    temperature = float(config.diag.get("pca_temperature", 1.0))
    text = text_latent[:count].float()
    if text_target_latent is None:
        text_target_bank = text
    else:
        text_target_bank = text_target_latent[:count].float()
    image = image_latent[:count].float()

    distributions = variational.all_distributions(text, image)
    text_anchor = text[example_index:example_index + 1]
    text_target_anchor = text_target_bank[
        example_index:example_index + 1]
    image_anchor = image[example_index:example_index + 1]
    text_source = text_anchor.expand(sample_count, -1, -1).contiguous()
    image_source = image_anchor.expand(sample_count, -1, -1).contiguous()

    def sample_fixed_gaussian(mu, logvar, seed_offset):
        mu = mu[example_index:example_index + 1].float().expand(
            sample_count, -1)
        logvar = logvar[example_index:example_index + 1].float().expand(
            sample_count, -1)
        generator = torch.Generator(device=mu.device)
        generator.manual_seed(
            int(config.train.seed) + 104729 * int(step) + seed_offset)
        noise = torch.randn(
            mu.shape, device=mu.device, dtype=mu.dtype,
            generator=generator)
        return mu + temperature * torch.exp(0.5 * logvar) * noise

    z_q_samples = sample_fixed_gaussian(
        distributions["q_mu"], distributions["q_logvar"], 11)
    z_text_samples = sample_fixed_gaussian(
        distributions["text_mu"], distributions["text_logvar"], 23)
    z_image_samples = sample_fixed_gaussian(
        distributions["image_mu"], distributions["image_logvar"], 37)

    # ``None`` bypasses the entire BVFM residual branch. Feeding an all-zero
    # vector would not be a valid no-z_v ablation because the projector and
    # conditional adapters contain biases.
    t2i_no_zv = collect_shared_trajectory(
        field, text_source, None, steps, reverse=False, cfg=cfg)
    t2i_prior = collect_shared_trajectory(
        field, text_source, z_text_samples, steps,
        reverse=False, cfg=cfg)
    i2t_no_zv = collect_shared_trajectory(
        field, image_source, None, steps, reverse=True, cfg=None)
    i2t_prior = collect_shared_trajectory(
        field, image_source, z_image_samples, steps,
        reverse=True, cfg=None)

    eps = 1e-8
    text_variance = _batch_variance_energy(
        text_target_bank).clamp_min(eps)
    image_variance = _batch_variance_energy(image).clamp_min(eps)
    t2i_task_displacement = F.mse_loss(
        image_anchor, text_anchor).sqrt().clamp_min(eps)
    i2t_task_displacement = F.mse_loss(
        image_anchor, text_target_anchor).sqrt().clamp_min(eps)
    image_target = image_anchor.expand_as(t2i_prior[-1])
    text_target = text_target_anchor.expand_as(i2t_prior[-1])

    def endpoint_cosine(endpoint, target):
        return F.cosine_similarity(
            endpoint.float().flatten(1), target.float().flatten(1),
            dim=1).mean()

    metrics = {
        "pca_t2i_no_zv_spread_ratio": (
            _batch_variance_energy(t2i_no_zv[-1])
            / image_variance).item(),
        "pca_t2i_prior_spread_ratio": (
            _batch_variance_energy(t2i_prior[-1])
            / image_variance).item(),
        "pca_i2t_no_zv_spread_ratio": (
            _batch_variance_energy(i2t_no_zv[-1])
            / text_variance).item(),
        "pca_i2t_prior_spread_ratio": (
            _batch_variance_energy(i2t_prior[-1])
            / text_variance).item(),
        "pca_t2i_zv_effect_rel": (
            F.mse_loss(t2i_prior[-1], t2i_no_zv[-1]).sqrt()
            / t2i_task_displacement).item(),
        "pca_i2t_zv_effect_rel": (
            F.mse_loss(i2t_prior[-1], i2t_no_zv[-1]).sqrt()
            / i2t_task_displacement).item(),
        "pca_t2i_no_zv_target_mse": F.mse_loss(
            t2i_no_zv[-1], image_target).item(),
        "pca_t2i_prior_target_mse": F.mse_loss(
            t2i_prior[-1], image_target).item(),
        "pca_i2t_no_zv_target_mse": F.mse_loss(
            i2t_no_zv[-1], text_target).item(),
        "pca_i2t_prior_target_mse": F.mse_loss(
            i2t_prior[-1], text_target).item(),
        "pca_t2i_no_zv_target_cosine": endpoint_cosine(
            t2i_no_zv[-1], image_target).item(),
        "pca_t2i_prior_target_cosine": endpoint_cosine(
            t2i_prior[-1], image_target).item(),
        "pca_i2t_no_zv_target_cosine": endpoint_cosine(
            i2t_no_zv[-1], text_target).item(),
        "pca_i2t_prior_target_cosine": endpoint_cosine(
            i2t_prior[-1], text_target).item(),
    }

    flow_center, flow_components = _fit_shared_pca(
        text, text_target_bank, image)
    text_source_xy = _project_shared_pca(
        text, flow_center, flow_components)
    text_target_xy = _project_shared_pca(
        text_target_bank, flow_center, flow_components)
    image_xy = _project_shared_pca(image, flow_center, flow_components)
    trajectories = [
        _project_shared_pca(t2i_no_zv, flow_center, flow_components),
        _project_shared_pca(t2i_prior, flow_center, flow_components),
        _project_shared_pca(i2t_no_zv, flow_center, flow_components),
        _project_shared_pca(i2t_prior, flow_center, flow_components),
    ]
    titles = [
        f"(a) T2I 0->1, without z_v ({sample_count} repeats)",
        f"(b) T2I 0->1, z_v~p_text ({sample_count} samples)",
        f"(c) I2T 1->0, without z_v ({sample_count} repeats)",
        f"(d) I2T 1->0, z_v~p_image ({sample_count} samples)",
    ]
    colors = ["#d97706", "#16a34a", "#737373", "#0f766e"]
    spread_values = [
        metrics["pca_t2i_no_zv_spread_ratio"],
        metrics["pca_t2i_prior_spread_ratio"],
        metrics["pca_i2t_no_zv_spread_ratio"],
        metrics["pca_i2t_prior_spread_ratio"],
    ]
    target_mse_values = [
        metrics["pca_t2i_no_zv_target_mse"],
        metrics["pca_t2i_prior_target_mse"],
        metrics["pca_i2t_no_zv_target_mse"],
        metrics["pca_i2t_prior_target_mse"],
    ]
    target_cosine_values = [
        metrics["pca_t2i_no_zv_target_cosine"],
        metrics["pca_t2i_prior_target_cosine"],
        metrics["pca_i2t_no_zv_target_cosine"],
        metrics["pca_i2t_prior_target_cosine"],
    ]

    figure, axes = plt.subplots(
        1, 4, figsize=(20, 5), sharex=True, sharey=True,
        constrained_layout=True)
    for panel, (axis, path, title, color, spread, target_mse,
                target_cosine) in enumerate(zip(
                    axes, trajectories, titles, colors, spread_values,
                    target_mse_values, target_cosine_values)):
        for sample_index in range(sample_count):
            axis.plot(
                path[:, sample_index, 0], path[:, sample_index, 1],
                color=color, alpha=0.20, linewidth=0.8)
        source_xy = (
            text_source_xy[example_index] if panel < 2
            else image_xy[example_index])
        target_xy = image_xy if panel < 2 else text_target_xy
        paired_target = target_xy[example_index]
        axis.scatter(
            target_xy[:, 0], target_xy[:, 1], s=18,
            color="#737373", alpha=0.25,
            label="reference target bank", zorder=2)
        axis.scatter(
            source_xy[0], source_xy[1], s=52,
            color="#2563eb", marker="o", label="fixed source", zorder=4)
        axis.scatter(
            paired_target[0], paired_target[1], s=70,
            color="#111827", marker="*", label="paired target", zorder=5)
        axis.scatter(
            path[-1, :, 0], path[-1, :, 1], s=22,
            color=color, label="rollout endpoint", zorder=4)
        axis.set_title(
            f"{title}\npaired MSE={target_mse:.3f}, "
            f"cos={target_cosine:.3f}; sample spread={spread:.4f}")
        axis.set_xlabel("shared endpoint PCA 1")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("shared endpoint PCA 2")
    axes[0].legend(loc="best", fontsize=8)
    figure.suptitle(
        f"Step {step}: fixed pair #{example_index}, "
        f"temperature={temperature:g}; "
        f"z_v effect T2I={metrics['pca_t2i_zv_effect_rel']:.3f}, "
        f"I2T={metrics['pca_i2t_zv_effect_rel']:.3f}")
    os.makedirs(output_dir, exist_ok=True)
    flow_path = os.path.join(
        output_dir, f"step{step}_flow_pca_trajectories.png")
    figure.savefig(flow_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    q_mu = distributions["q_mu"].float()
    text_mu = distributions["text_mu"].float()
    image_mu = distributions["image_mu"].float()
    zv_center, zv_components = _fit_shared_pca(
        z_q_samples, z_text_samples, z_image_samples)
    q_xy = _project_shared_pca(
        z_q_samples.unsqueeze(1), zv_center, zv_components).reshape(
            sample_count, 2)
    text_prior_xy = _project_shared_pca(
        z_text_samples.unsqueeze(1), zv_center, zv_components).reshape(
            sample_count, 2)
    image_prior_xy = _project_shared_pca(
        z_image_samples.unsqueeze(1), zv_center, zv_components).reshape(
            sample_count, 2)
    q_signal = q_mu.std(dim=0, unbiased=False).mean()
    text_signal = text_mu.std(dim=0, unbiased=False).mean()
    image_signal = image_mu.std(dim=0, unbiased=False).mean()
    q_noise = torch.exp(0.5 * distributions["q_logvar"]).mean()
    text_noise = torch.exp(
        0.5 * distributions["text_logvar"]).mean()
    image_noise = torch.exp(
        0.5 * distributions["image_logvar"]).mean()
    metrics.update({
        "pca_q_signal_to_noise": (q_signal / q_noise.clamp_min(eps)).item(),
        "pca_text_prior_signal_to_noise": (
            text_signal / text_noise.clamp_min(eps)).item(),
        "pca_image_prior_signal_to_noise": (
            image_signal / image_noise.clamp_min(eps)).item(),
    })

    figure, axis = plt.subplots(
        1, 1, figsize=(7, 6), constrained_layout=True)
    axis.scatter(
        q_xy[:, 0], q_xy[:, 1], s=26, color="#111827",
        alpha=0.65, label="q pair posterior samples", zorder=4)
    axis.scatter(
        text_prior_xy[:, 0], text_prior_xy[:, 1], s=24,
        color="#16a34a", alpha=0.65,
        label="p_text samples", zorder=3)
    axis.scatter(
        image_prior_xy[:, 0], image_prior_xy[:, 1], s=24,
        color="#d97706", alpha=0.65,
        label="p_image samples", zorder=3)
    axis.set_title(
        f"Step {step}: fixed pair #{example_index}, "
        f"{sample_count} samples, temperature={temperature:g}\n"
        f"cross-data mean-spread/std  q={metrics['pca_q_signal_to_noise']:.4f}, "
        f"p_text={metrics['pca_text_prior_signal_to_noise']:.4f}, "
        f"p_image={metrics['pca_image_prior_signal_to_noise']:.4f}")
    axis.set_xlabel("z_v PCA 1")
    axis.set_ylabel("z_v PCA 2")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    zv_path = os.path.join(
        output_dir, f"step{step}_zv_pca_distributions.png")
    figure.savefig(zv_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return metrics


class SharedVariationalBVFM(nn.Module):
    def __init__(self, nnet, decoder, variational, train_config):
        super().__init__()
        self.nnet = nnet
        self.decoder = decoder
        self.variational = variational
        self.config = train_config

    def decoder_loss(self, latent, text_ids):
        logits = self.decoder(latent, text_ids[:, :-1])
        return ar_caption_loss(logits, text_ids)

    def forward(
            self,
            x_t,
            time_t,
            target_velocity,
            text_condition,
            text_target,
            image_latent,
            image_source,
            text_ids,
            reference_velocity,
            do_endpoint_ce,
            do_endpoint_latent,
            kl_weight):
        batch = x_t.shape[0]
        device = x_t.device
        distributions = self.variational.all_distributions(
            text_condition, image_latent)
        z_q = self.variational.sample(
            distributions["q_mu"],
            distributions["q_logvar"],
            float(self.config.posterior_temperature))

        prediction = self.nnet(
            x_t,
            t=time_t,
            null_indicator=torch.zeros(
                batch, dtype=torch.bool, device=device),
            variation_latent=z_q)[0]
        loss_fm = F.mse_loss(
            prediction.float(), target_velocity.float())
        loss_kl_text = self.variational.kl(
            distributions["q_mu"], distributions["q_logvar"],
            distributions["text_mu"], distributions["text_logvar"])
        loss_kl_image = self.variational.kl(
            distributions["q_mu"], distributions["q_logvar"],
            distributions["image_mu"], distributions["image_logvar"])
        beta = float(self.config.bvfm_beta)
        loss_kl = beta * loss_kl_text + (1.0 - beta) * loss_kl_image

        loss_distill = torch.zeros((), device=device)
        distill_relative = torch.zeros((), device=device)
        distill_cosine = torch.ones((), device=device)
        if reference_velocity.numel() > 0:
            if reference_velocity.shape[0] % 2 != 0:
                raise ValueError(
                    "reference_velocity must contain matched cond/uncond rows")
            count = reference_velocity.shape[0] // 2
            # T2I inference has only z_text, hence its one-sided text prior is
            # used here rather than posterior z_q. Sampling at the
            # configured temperature protects the actual variational T2I path,
            # not only the prior mean. Both CFG arms are anchored because either
            # arm drifting can ruin T2I.
            z_text_distill = self.variational.sample(
                distributions["text_mu"][:count],
                distributions["text_logvar"][:count],
                float(self.config.distill_prior_temperature))
            current_t2i = self.nnet(
                x_t[:count].repeat(2, 1, 1),
                t=time_t[:count].repeat(2),
                null_indicator=torch.cat([
                    torch.zeros(count, dtype=torch.bool, device=device),
                    torch.ones(count, dtype=torch.bool, device=device),
                ]),
                variation_latent=z_text_distill.repeat(2, 1))[0]
            reference_float = reference_velocity.float()
            loss_distill = F.mse_loss(
                current_t2i.float(), reference_float)
            distill_relative = loss_distill / reference_float.pow(
                2).mean().clamp_min(1e-8)
            distill_cosine = F.cosine_similarity(
                current_t2i.float().flatten(1),
                reference_float.flatten(1), dim=1).mean()

        loss_teacher, teacher_acc = self.decoder_loss(
            text_target.detach(), text_ids)

        loss_endpoint = torch.zeros((), device=device)
        endpoint_acc = torch.zeros((), device=device)
        loss_endpoint_latent = torch.zeros((), device=device)
        endpoint_cosine = torch.zeros((), device=device)
        if do_endpoint_latent:
            count = min(int(self.config.endpoint_latent_batch), batch)
            z_image = self.variational.sample(
                distributions["image_mu"][:count],
                distributions["image_logvar"][:count],
                float(self.config.rollout_prior_temperature))
            endpoint = integrate_shared_train(
                self.nnet,
                image_source[:count].detach(),
                z_image,
                int(self.config.endpoint_steps),
                reverse=True)
            target = text_target[:count].detach()
            endpoint_mse = F.mse_loss(
                endpoint.float(), target.float())
            endpoint_cosine = F.cosine_similarity(
                endpoint.float().flatten(1),
                target.float().flatten(1), dim=1).mean()
            loss_endpoint_latent = endpoint_mse + float(
                self.config.endpoint_cosine_weight) * (
                    1.0 - endpoint_cosine)
            # The caption objective reaches decoder, shared VF and image prior.
            loss_endpoint, endpoint_acc = self.decoder_loss(
                endpoint.float(), text_ids[:count])
        elif do_endpoint_ce:
            count = min(int(self.config.endpoint_ce_batch), batch)
            with torch.no_grad():
                z_image = self.variational.sample(
                    distributions["image_mu"][:count],
                    distributions["image_logvar"][:count],
                    float(self.config.rollout_prior_temperature))
                endpoint = integrate_shared(
                    self.nnet,
                    image_source[:count].detach(),
                    z_image,
                    int(self.config.endpoint_steps),
                    reverse=True)
            loss_endpoint, endpoint_acc = self.decoder_loss(
                endpoint.float(), text_ids[:count])

        total = (
            float(self.config.w_bvfm) * (
                loss_fm + float(kl_weight) * loss_kl)
            + float(self.config.w_t2i_distill) * loss_distill
            + float(self.config.w_teacher_ce) * loss_teacher
            + float(self.config.w_endpoint_ce) * loss_endpoint
            + float(self.config.w_endpoint_latent) * loss_endpoint_latent
        )
        metrics = {
            "loss_fm": loss_fm.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_kl_text": loss_kl_text.detach(),
            "loss_kl_image": loss_kl_image.detach(),
            "loss_distill": loss_distill.detach(),
            "distill_relative": distill_relative.detach(),
            "distill_cosine": distill_cosine.detach(),
            "loss_teacher": loss_teacher.detach(),
            "teacher_acc": teacher_acc.detach(),
            "loss_endpoint": loss_endpoint.detach(),
            "endpoint_acc": endpoint_acc.detach(),
            "loss_endpoint_latent": loss_endpoint_latent.detach(),
            "endpoint_cosine": endpoint_cosine.detach(),
            "q_std": torch.exp(
                0.5 * distributions["q_logvar"]).mean().detach(),
            "text_prior_std": torch.exp(
                0.5 * distributions["text_logvar"]).mean().detach(),
            "image_prior_std": torch.exp(
                0.5 * distributions["image_logvar"]).mean().detach(),
            "q_mu_batch_std": distributions[
                "q_mu"].float().std(dim=0, unbiased=False).mean().detach(),
            "text_mu_batch_std": distributions[
                "text_mu"].float().std(
                    dim=0, unbiased=False).mean().detach(),
            "image_mu_batch_std": distributions[
                "image_mu"].float().std(
                    dim=0, unbiased=False).mean().detach(),
        }
        return total, metrics


@torch.no_grad()
def save_t2i_comparison(
        step, config, model, reference, autoencoder,
        clip_encoder, clip_tokenizer, prompts, device):
    from torchvision.transforms import functional as vision_functional

    text_ids = clip_tokenizer(prompts).to(device)
    text_features = clip_text_features(clip_encoder, text_ids)
    text_sample, _, _ = model.nnet(text_features, text_encoder=True)
    source = text_sample.float()
    text_mu, text_logvar = model.variational.prior_from_text(source)
    z_text = model.variational.sample(
        text_mu, text_logvar, float(config.diag.prior_temperature))
    current_tokens = integrate_t2i_cfg(
        model.nnet, source.clone(), z_text,
        int(config.diag.t2i_steps), float(config.diag.t2i_cfg))
    released_tokens = integrate_released_t2i(
        reference, source.clone(),
        int(config.diag.t2i_steps), float(config.diag.t2i_cfg))

    def decode(tokens):
        tokens = (
            tokens.permute(0, 2, 1).unsqueeze(2)
            / float(config.vq_model.scale_factor))
        images = autoencoder.decode_tokens(
            tokens, text_guidance=text_features)
        return images.float().mul(255).add_(0.5).clamp_(0, 255)

    current_images = decode(current_tokens)
    released_images = decode(released_tokens)
    demo_dir = os.path.join(config.train.output_dir, "demos")
    os.makedirs(demo_dir, exist_ok=True)
    for index in range(len(prompts)):
        vision_functional.to_pil_image(
            current_images[index].cpu().byte()).save(os.path.join(
                demo_dir, f"step{step}_t2i_bvfm_{index}.png"))
        vision_functional.to_pil_image(
            released_images[index].cpu().byte()).save(os.path.join(
                demo_dir, f"step{step}_t2i_released_{index}.png"))


@torch.no_grad()
def run_diagnostics(
        step, config, model, reference, clip_encoder, clip_tokenizer,
        autoencoder, val_captions, val_images, val_field_batch, device,
        log_fn):
    model.eval()
    model.nnet.context_encoder.eval()
    metrics = {}

    teacher_losses = []
    teacher_accs = []
    for start in range(0, len(val_captions), 128):
        captions = val_captions[start:start + 128]
        text_ids = clip_tokenizer(captions).to(device)
        text_features = clip_text_features(clip_encoder, text_ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, text_mean, _ = model.nnet(
                text_features, text_encoder=True)
            loss, acc = model.decoder_loss(text_mean.float(), text_ids)
        teacher_losses.append(loss.item())
        teacher_accs.append(acc.item())
    metrics["val_teacher_ce"] = sum(teacher_losses) / len(teacher_losses)
    metrics["val_teacher_acc"] = sum(teacher_accs) / len(teacher_accs)

    hypotheses = []
    teacher_hypotheses = []
    references = []
    endpoint_mse = []
    endpoint_cosine = []
    records = []
    printed = 0
    for start in range(0, len(val_images), 8):
        chunk = val_images[start:start + 8]
        images = torch.stack([item[0] for item in chunk]).to(device)
        first_refs = [item[1][0] for item in chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, image_mean = encode_image_latents(
                autoencoder, images, config.vq_model.scale_factor)
            image_mu, image_logvar = model.variational.prior_from_image(
                image_mean)
            z_image = model.variational.sample(
                image_mu, image_logvar,
                float(config.diag.prior_temperature))
            endpoint = integrate_shared(
                model.nnet, image_mean, z_image,
                int(config.diag.i2t_steps), reverse=True)
            predicted_ids = model.decoder.generate(endpoint.float())

            text_ids = clip_tokenizer(first_refs).to(device)
            text_features = clip_text_features(clip_encoder, text_ids)
            _, text_mean, _ = model.nnet(
                text_features, text_encoder=True)
            teacher_ids = model.decoder.generate(text_mean.float())

        endpoint_mse.append(F.mse_loss(
            endpoint.float(), text_mean.float()).item())
        endpoint_cosine.append(F.cosine_similarity(
            endpoint.float().flatten(1),
            text_mean.float().flatten(1), dim=1).mean().item())
        for index, (_, refs, image_id) in enumerate(chunk):
            hypothesis = ids_to_text(
                predicted_ids[index], clip_tokenizer)
            teacher_hypothesis = ids_to_text(
                teacher_ids[index], clip_tokenizer)
            hypotheses.append(hypothesis)
            teacher_hypotheses.append(teacher_hypothesis)
            references.append(refs)
            records.append({
                "image_id": image_id,
                "prediction": hypothesis,
                "teacher_prediction": teacher_hypothesis,
                "references": refs,
            })
            if printed < int(config.diag.n_print_samples):
                log_fn(
                    f"[DIAG] i2t sample | pred: {hypothesis!r} "
                    f"| ref: {refs[0]!r}")
                printed += 1

    metrics["val_i2t_bleu4"] = bleu4(hypotheses, references)
    metrics["val_teacher_decode_bleu4"] = bleu4(
        teacher_hypotheses, references)
    metrics["val_i2t_endpoint_mse"] = sum(endpoint_mse) / len(endpoint_mse)
    metrics["val_i2t_endpoint_cosine"] = (
        sum(endpoint_cosine) / len(endpoint_cosine))

    if val_field_batch is not None:
        (text_condition, text_target, image_latent,
         x_t, time_t, velocity) = val_field_batch
        # The immutable reference is stored in BF16 to avoid another full FP32
        # FlowTok-XL copy.  Autocast is required here just as in the training
        # reference call; otherwise FP32 cached validation latents reach BF16
        # linear weights directly.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            distributions = model.variational.all_distributions(
                text_condition, image_latent)
            z_q = distributions["q_mu"]
            current = model.nnet(
                x_t, t=time_t,
                null_indicator=torch.zeros(
                    x_t.shape[0], dtype=torch.bool, device=device),
                variation_latent=z_q)[0]
            current_cond = model.nnet(
                x_t, t=time_t,
                null_indicator=torch.zeros(
                    x_t.shape[0], dtype=torch.bool, device=device),
                variation_latent=distributions["text_mu"])[0]
            current_uncond = model.nnet(
                x_t, t=time_t,
                null_indicator=torch.ones(
                    x_t.shape[0], dtype=torch.bool, device=device),
                variation_latent=distributions["text_mu"])[0]
            released_cond = reference(
                x_t, t=time_t,
                null_indicator=torch.zeros(
                    x_t.shape[0], dtype=torch.bool, device=device))[0]
            released_uncond = reference(
                x_t, t=time_t,
                null_indicator=torch.ones(
                    x_t.shape[0], dtype=torch.bool, device=device))[0]
        cfg = float(config.diag.t2i_cfg)
        current_t2i = current_uncond + cfg * (
            current_cond - current_uncond)
        released = released_uncond + cfg * (
            released_cond - released_uncond)
        drift_mse = F.mse_loss(
            current_t2i.float(), released.float()).item()
        released_energy = released.float().pow(2).mean().item()
        metrics["val_bvfm_fm"] = F.mse_loss(
            current.float(), velocity.float()).item()
        metrics["val_kl_text"] = model.variational.kl(
            distributions["q_mu"], distributions["q_logvar"],
            distributions["text_mu"], distributions["text_logvar"]
        ).item()
        metrics["val_kl_image"] = model.variational.kl(
            distributions["q_mu"], distributions["q_logvar"],
            distributions["image_mu"], distributions["image_logvar"]
        ).item()
        metrics["val_t2i_velocity_drift_mse"] = drift_mse
        metrics["val_t2i_relative_velocity_drift"] = (
            drift_mse / max(released_energy, 1e-8))
        metrics["val_t2i_velocity_cosine"] = F.cosine_similarity(
            current_t2i.float().flatten(1),
            released.float().flatten(1), dim=1).mean().item()

        if step % int(config.diag.get("pca_every", 1_000)) == 0:
            demo_dir = os.path.join(config.train.output_dir, "demos")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pca_metrics = save_variational_pca_demos(
                    step, config, model.nnet, model.variational,
                    text_condition, image_latent, demo_dir,
                    text_target_latent=text_target)
            metrics.update(pca_metrics)
            log_fn(
                "[DIAG] PCA demos saved: "
                f"{os.path.join(demo_dir, f'step{step}_flow_pca_trajectories.png')} "
                f"and {os.path.join(demo_dir, f'step{step}_zv_pca_distributions.png')}")

    if step % int(config.diag.t2i_every) == 0 and val_images:
        prompts = [
            item[1][0]
            for item in val_images[:int(config.diag.t2i_prompts)]
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            save_t2i_comparison(
                step, config, model, reference, autoencoder,
                clip_encoder, clip_tokenizer, prompts, device)

    demo_dir = os.path.join(config.train.output_dir, "demos")
    os.makedirs(demo_dir, exist_ok=True)
    with open(
            os.path.join(demo_dir, f"step{step}_i2t.json"),
            "w") as handle:
        json.dump(
            {"step": step, "metrics": metrics, "samples": records},
            handle, indent=2, ensure_ascii=False)
    log_fn(
        f"[DIAG] step {step} | "
        + " | ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    model.train()
    model.nnet.context_encoder.eval()
    reference.eval()
    return metrics


def checkpoint_payload(
        step, nnet, decoder, variational, optimizer,
        best_bleu, best_step, include_optimizer):
    payload = {
        "format": "flowtok_bvfm_shared_variational_v2",
        "step": step,
        "field_semantics": {
            "orientation": "text_at_t0_to_image_at_t1",
            "t2i": "integrate_same_field_0_to_1_with_text_prior",
            "i2t": "integrate_same_field_1_to_0_with_image_prior",
            "direction_condition": False,
            "interpolant": "flowtok_sigma_min_1e-5",
        },
        "nnet": nnet.state_dict(),
        "variational": variational.state_dict(),
        "variational_config": {
            "token_dim": 16,
            "latent_dim": variational.latent_dim,
            "posterior": "q(z_v|z_text,z_image)",
            "forward_prior": "p_text(z_v|z_text)",
            "reverse_prior": "p_image(z_v|z_image)",
            "kl_reduction": "mean_over_batch_and_latent_dim",
        },
        "decoder": decoder.state_dict(),
        "decoder_type": "ar",
        "decoder_config": {
            "latent_dim": decoder.mem_proj.in_features,
            "d_model": decoder.mem_proj.out_features,
            "depth": len(decoder.blocks),
            "num_heads": decoder.blocks[0].cross_attn.num_heads,
            "d_ff": decoder.blocks[0].mlp[0].out_features,
            "vocab_size": decoder.vocab_size,
            "seq_len": decoder.seq_len,
            "dropout": decoder.blocks[0].dropout.p,
        },
        "best_i2t_bleu4": best_bleu,
        "best_step": best_step,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.batch_size_per_gpu is not None:
        config.train.batch_size_per_gpu = args.batch_size_per_gpu
    if args.n_steps is not None:
        config.train.n_steps = args.n_steps
    train_config = config.train

    rank, world, local_rank = setup_dist()
    device = f"cuda:{local_rank}"
    torch.manual_seed(int(train_config.seed) + rank)

    def log(message):
        if rank == 0:
            print(message, flush=True)

    os.makedirs(train_config.output_dir, exist_ok=True)
    metrics_path = os.path.join(
        train_config.output_dir, "train_metrics.jsonl")
    latest_path = os.path.join(train_config.output_dir, "latest.pt")
    # This gate uses a cheap T2I velocity-drift proxy, not FID/CLIP.  Calling
    # it "best_joint" previously encouraged selecting the wrong checkpoint.
    best_path = os.path.join(train_config.output_dir, "best_proxy.pt")
    final_path = os.path.join(train_config.output_dir, "final.pt")

    nnet, clip_encoder, clip_tokenizer, autoencoder = build_models(
        config, device)
    reference = make_released_reference(nnet)
    for block in nnet.blocks:
        block.forward = block._forward

    # The released backbone is immutable.  Only shared z_v-conditioned
    # residuals are trained, and those residuals are used in both directions.
    nnet.requires_grad_(False)
    for name, parameter in nnet.named_parameters():
        if (
                name.startswith("bvfm_variation_projector.")
                or name.startswith("bvfm_adapters.")
                or name.startswith("bvfm_output_adapter.")):
            parameter.requires_grad_(True)
    unexpected_trainable = [
        name for name, parameter in nnet.named_parameters()
        if parameter.requires_grad
        and not name.startswith("bvfm_variation_projector.")
        and not name.startswith("bvfm_adapters.")
        and not name.startswith("bvfm_output_adapter.")
    ]
    if unexpected_trainable:
        raise RuntimeError(
            f"Released field was not frozen: {unexpected_trainable[:8]}")
    nnet.train()
    nnet.context_encoder.eval()

    from libs.model.bvfm_variational import BVFMVariationalHeads
    variational = BVFMVariationalHeads(
        token_dim=int(config.nnet.model_args.channels),
        latent_dim=int(config.bvfm.latent_dim),
        hidden_dim=int(config.bvfm.hidden_dim),
        dropout=float(config.bvfm.dropout),
        logvar_bias=float(config.bvfm.logvar_bias)).to(device)

    decoder_checkpoint = torch.load(
        config.decoder_init,
        map_location="cpu",
        weights_only=True,
        mmap=True)
    from libs.model.text_decoder_ar import ARTextDecoder
    decoder_config = decoder_checkpoint.get(
        "decoder_config", dict(config.decoder))
    decoder = ARTextDecoder(**decoder_config).to(device)
    decoder.load_state_dict(decoder_checkpoint["decoder"])
    decoder_init_step = decoder_checkpoint.get("step", "?")
    del decoder_checkpoint

    model = SharedVariationalBVFM(
        nnet, decoder, variational, train_config).to(device)
    model_ddp = DDP(
        model, device_ids=[local_rank], find_unused_parameters=False
    ) if world > 1 else model

    vf_parameters = [
        parameter for parameter in nnet.parameters()
        if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": vf_parameters, "lr": train_config.lr_vf},
            {"params": variational.parameters(),
             "lr": train_config.lr_variational},
            {"params": decoder.parameters(), "lr": train_config.lr_decoder},
        ],
        weight_decay=float(train_config.weight_decay),
        betas=tuple(train_config.betas))
    base_lrs = [
        float(train_config.lr_vf),
        float(train_config.lr_variational),
        float(train_config.lr_decoder),
    ]

    def learning_rate_scale(step):
        if step < int(train_config.warmup_steps):
            return (step + 1) / max(1, int(train_config.warmup_steps))
        progress = (
            (step - int(train_config.warmup_steps))
            / max(1, int(train_config.n_steps) - int(train_config.warmup_steps)))
        return 0.1 + 0.9 * 0.5 * (
            1.0 + math.cos(math.pi * min(progress, 1.0)))

    def kl_weight_at(step):
        if step < int(train_config.kl_start):
            return 0.0
        return min(
            1.0,
            (step - int(train_config.kl_start) + 1)
            / max(1, int(train_config.kl_anneal_steps)))

    start_step = 0
    best_bleu = 0.0
    best_step = 0
    if args.resume == "auto" and os.path.isfile(latest_path):
        checkpoint = torch.load(
            latest_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "flowtok_bvfm_shared_variational_v2":
            raise RuntimeError(
                f"Refusing incompatible resume checkpoint: {latest_path}")
        nnet.load_state_dict(checkpoint["nnet"])
        decoder.load_state_dict(checkpoint["decoder"])
        variational.load_state_dict(checkpoint["variational"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_bleu = float(checkpoint.get("best_i2t_bleu4", 0.0))
        best_step = int(checkpoint.get("best_step", 0))
        log(f"[INFO] resumed {latest_path} at step {start_step}")

    dataset = CocoImageCaptionDataset(
        config.data.train_images_dir,
        config.data.train_captions,
        int(config.dataset.crop_size))
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True
    ) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=int(train_config.batch_size_per_gpu),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(train_config.num_workers),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True)

    val_captions = []
    val_images = []
    val_field_batch = None
    if rank == 0:
        with open(config.data.val_captions) as handle:
            annotations = json.load(handle)["annotations"]
        val_captions = [
            item["caption"].strip()
            for item in annotations[:int(config.diag.n_val_captions)]]
        val_images = load_val_images(config, clip_tokenizer)
        count = min(int(config.diag.n_val_fm_pairs), len(val_images))
        images = torch.stack(
            [item[0] for item in val_images[:count]]).to(device)
        captions = [item[1][0] for item in val_images[:count]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_sample, image_mean = encode_image_latents(
                autoencoder, images, config.vq_model.scale_factor)
            text_ids = clip_tokenizer(captions).to(device)
            text_features = clip_text_features(clip_encoder, text_ids)
            text_sample, text_mean, _ = nnet(
                text_features, text_encoder=True)
            time_grid = torch.linspace(0.05, 0.95, count, device=device)
            x_t, velocity = flow_interp(
                time_grid, text_sample.float(), image_sample)
        val_field_batch = (
            text_sample.float(), text_mean.float(), image_mean,
            x_t, time_grid, velocity)

    log(
        f"[INFO] shared variational BVFM world={world} "
        f"batch/gpu={train_config.batch_size_per_gpu} "
        f"decoder_init_step={decoder_init_step}")
    log(
        f"[INFO] one VF: T2I=0->1, I2T=1->0, direction_condition=false "
        f"zv_dim={config.bvfm.latent_dim} beta={train_config.bvfm_beta}")
    log(
        "[INFO] canonical orientation uses FlowTok sigma_min=1e-5 "
        "interpolant for released-checkpoint compatibility")
    log(
        f"[INFO] released_backbone_frozen=true "
        f"trainable_shared_vf_params={sum(p.numel() for p in vf_parameters):,} "
        f"trainable_variational_params="
        f"{sum(p.numel() for p in variational.parameters()):,}")
    log(
        f"[INFO] output={train_config.output_dir} "
        f"existing task adapter baseline is not loaded or overwritten")

    step = start_step
    epoch = 0
    log_time = time.time()
    last_endpoint_latent = 0.0
    last_endpoint_cosine = 0.0
    while step < int(train_config.n_steps):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for images, captions in loader:
            if step >= int(train_config.n_steps):
                break

            lr_scale = learning_rate_scale(step)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * lr_scale

            images = images.to(device, non_blocking=True)
            captions = list(captions)
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16):
                image_sample, image_mean = encode_image_latents(
                    autoencoder, images, config.vq_model.scale_factor)
                text_ids = clip_tokenizer(captions).to(device)
                text_features = clip_text_features(
                    clip_encoder, text_ids)
                text_sample, text_mean, _ = nnet(
                    text_features, text_encoder=True)
                text_sample = text_sample.float()
                text_mean = text_mean.float()
                batch = image_sample.shape[0]
                time_t = logit_normal_time(batch, device)
                x_t, target_velocity = flow_interp(
                    time_t, text_sample, image_sample)

                reference_velocity = torch.empty(0, device=device)
                if step % int(train_config.distill_every) == 0:
                    count = min(int(train_config.distill_batch), batch)
                    reference_cond = reference(
                        x_t[:count],
                        t=time_t[:count],
                        null_indicator=torch.zeros(
                            count, dtype=torch.bool, device=device))[0]
                    reference_uncond = reference(
                        x_t[:count],
                        t=time_t[:count],
                        null_indicator=torch.ones(
                            count, dtype=torch.bool, device=device))[0]
                    reference_velocity = torch.cat(
                        [reference_cond, reference_uncond], dim=0)

            do_endpoint_ce = (
                step >= int(train_config.endpoint_ce_start)
                and step % int(train_config.endpoint_ce_every) == 0)
            do_endpoint_latent = (
                step >= int(train_config.endpoint_latent_start)
                and step % int(train_config.endpoint_latent_every) == 0)
            kl_weight = kl_weight_at(step)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, train_metrics = model_ddp(
                    x_t,
                    time_t,
                    target_velocity,
                    text_sample,
                    text_mean,
                    image_mean,
                    image_mean,
                    text_ids,
                    reference_velocity,
                    do_endpoint_ce,
                    do_endpoint_latent,
                    kl_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0 and start_step == 0:
                vf_grad = sum(
                    parameter.grad.float().norm()
                    for parameter in vf_parameters
                    if parameter.grad is not None)
                decoder_grad = sum(
                    parameter.grad.float().norm()
                    for parameter in decoder.parameters()
                    if parameter.grad is not None)
                checks = {
                    "shared_vf_grad": float(vf_grad),
                    "decoder_grad": float(decoder_grad),
                    "released_relative_drift": float(
                        train_metrics["distill_relative"]),
                    "direction_parameter_count": 0,
                }
                if (
                        any(not math.isfinite(value) or value <= 0
                            for value in (
                                checks["shared_vf_grad"],
                                checks["decoder_grad"]))
                        or not math.isfinite(
                            checks["released_relative_drift"])
                        or checks["released_relative_drift"] > float(
                            train_config.startup_max_relative_drift)):
                    raise RuntimeError(f"startup invariant failed: {checks}")
                log(f"[CHECK] startup invariants passed: {checks}")

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad],
                float(train_config.grad_clip))
            optimizer.step()
            if do_endpoint_latent:
                last_endpoint_latent = float(
                    train_metrics["loss_endpoint_latent"])
                last_endpoint_cosine = float(
                    train_metrics["endpoint_cosine"])
            step += 1

            if step % int(train_config.log_every) == 0:
                steps_per_second = int(train_config.log_every) / max(
                    time.time() - log_time, 1e-6)
                log_time = time.time()
                log(
                    f"[TRAIN] step {step}/{train_config.n_steps} "
                    f"loss={loss.item():.4f} "
                    f"fm={train_metrics['loss_fm'].item():.4f} "
                    f"kl={train_metrics['loss_kl'].item():.4f} "
                    f"klT={train_metrics['loss_kl_text'].item():.4f} "
                    f"klI={train_metrics['loss_kl_image'].item():.4f} "
                    f"klw={kl_weight:.3f} "
                    f"dist_rel={train_metrics['distill_relative'].item():.5f} "
                    f"ep_ce={train_metrics['loss_endpoint'].item():.4f} "
                    f"ep_lat_last={last_endpoint_latent:.4f} "
                    f"ep_cos_last={last_endpoint_cosine:.4f} "
                    f"teacher={train_metrics['loss_teacher'].item():.4f} "
                    f"qstd={train_metrics['q_std'].item():.3f} "
                    f"pTstd={train_metrics['text_prior_std'].item():.3f} "
                    f"pIstd={train_metrics['image_prior_std'].item():.3f} "
                    f"qmu_var={train_metrics['q_mu_batch_std'].item():.4f} "
                    f"pTmu_var="
                    f"{train_metrics['text_mu_batch_std'].item():.4f} "
                    f"pImu_var="
                    f"{train_metrics['image_mu_batch_std'].item():.4f} "
                    f"grad={float(gradient_norm):.3f} "
                    f"steps/s={steps_per_second:.2f}")
                if rank == 0:
                    record = {
                        "step": step,
                        "loss": loss.item(),
                        "kl_weight": kl_weight,
                        "grad_norm": float(gradient_norm),
                        "lr_vf": optimizer.param_groups[0]["lr"],
                        "lr_variational": optimizer.param_groups[1]["lr"],
                        "lr_decoder": optimizer.param_groups[2]["lr"],
                        "last_endpoint_latent": last_endpoint_latent,
                        "last_endpoint_cosine": last_endpoint_cosine,
                    }
                    record.update({
                        key: value.item()
                        for key, value in train_metrics.items()})
                    with open(metrics_path, "a") as handle:
                        handle.write(json.dumps(record) + "\n")

            if step % int(train_config.eval_every) == 0:
                if rank == 0:
                    # Persist optimizer state before diagnostics so a reporting
                    # failure never discards the preceding training interval.
                    torch.save(checkpoint_payload(
                        step, nnet, decoder, variational, optimizer,
                        best_bleu, best_step, include_optimizer=True),
                        latest_path)
                    log(
                        f"[INFO] saved pre-diagnostic recovery step={step}")
                    diagnostic_metrics = run_diagnostics(
                        step, config, model, reference,
                        clip_encoder, clip_tokenizer, autoencoder,
                        val_captions, val_images, val_field_batch,
                        device, log)
                    with open(metrics_path, "a") as handle:
                        handle.write(json.dumps({
                            "step": step, **diagnostic_metrics}) + "\n")
                    drift = diagnostic_metrics[
                        "val_t2i_relative_velocity_drift"]
                    bleu = diagnostic_metrics["val_i2t_bleu4"]
                    safe = drift <= float(
                        config.diag.max_t2i_relative_velocity_drift)
                    eligible = safe and bleu >= float(
                        config.diag.min_i2t_bleu4_for_best)
                    if eligible and bleu > best_bleu:
                        best_bleu = bleu
                        best_step = step
                        torch.save(checkpoint_payload(
                            step, nnet, decoder, variational, optimizer,
                            best_bleu, best_step, include_optimizer=False),
                            best_path)
                        log(
                            f"[GATE] new best_proxy step={step} "
                            f"bleu4={bleu:.4f} drift={drift:.5f}")
                    elif not safe:
                        log(
                            f"[GATE] reject step={step}: T2I relative drift "
                            f"{drift:.5f} > "
                            f"{config.diag.max_t2i_relative_velocity_drift}")
                barrier()

            if step % int(train_config.save_every) == 0 and rank == 0:
                torch.save(checkpoint_payload(
                    step, nnet, decoder, variational, optimizer,
                    best_bleu, best_step, include_optimizer=True), latest_path)
                if step % int(train_config.snapshot_every) == 0:
                    torch.save(checkpoint_payload(
                        step, nnet, decoder, variational, optimizer,
                        best_bleu, best_step, include_optimizer=False),
                        os.path.join(
                            train_config.output_dir, f"step{step}.pt"))
                log(
                    f"[INFO] saved latest step={step}; "
                    f"best_proxy_step={best_step}")
        epoch += 1

    if rank == 0:
        torch.save(checkpoint_payload(
            step, nnet, decoder, variational, optimizer,
            best_bleu, best_step, include_optimizer=True), latest_path)
        torch.save(checkpoint_payload(
            step, nnet, decoder, variational, optimizer,
            best_bleu, best_step, include_optimizer=False), final_path)
        log(
            f"[INFO] training complete step={step} "
            f"best_proxy_step={best_step} best_i2t_bleu4={best_bleu:.4f} "
            f"deploy_candidate={final_path}")
    barrier()
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
