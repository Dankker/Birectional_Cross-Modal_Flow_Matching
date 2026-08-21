#!/usr/bin/env python
"""Export FlowTok-BVFM T2I-to-I2T generated-image round-trip trajectories."""

import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def default_checkpoint():
    weights_root = os.environ.get("BVFM_WEIGHTS_ROOT")
    if weights_root:
        return os.path.join(
            weights_root, "image", "bvfm_image_step40000.pt")
    return os.path.join(REPO_ROOT, "checkpoints", "bvfm_image_step40000.pt")

from bvfm_image import training as TRAIN  # noqa: E402
from bvfm_image.common import (  # noqa: E402
    clip_text_features,
    encode_image_latents,
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=os.path.join(
            REPO_ROOT, "configs", "bvfm_image_xl.py"))
    parser.add_argument(
        "--checkpoint", default=default_checkpoint())
    parser.add_argument("--coco-captions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--text-temperature", type=float, default=0.5)
    parser.add_argument("--zv-temperature", type=float, default=1.0)
    return parser.parse_args()


def load_coco_prompts(captions_json, count):
    with open(captions_json) as handle:
        annotations = json.load(handle)["annotations"]
    references = {}
    for item in annotations:
        references.setdefault(
            int(item["image_id"]), item["caption"].strip())
    prompts = [references[image_id] for image_id in sorted(references)[:count]]
    if len(prompts) < count:
        raise RuntimeError(f"Need {count} captions, found {len(prompts)}")
    return prompts


def sample_gaussian(mu, logvar, temperature, seed):
    generator = torch.Generator(device=mu.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(
        mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + float(temperature) * torch.exp(0.5 * logvar) * noise


def decode_image_tokens(autoencoder, tokens, text_features, scale_factor):
    token_grid = (
        tokens.permute(0, 2, 1).unsqueeze(2) / float(scale_factor))
    return autoencoder.decode_tokens(
        token_grid, text_guidance=text_features).float().clamp(0.0, 1.0)


def endpoint_spread(endpoint):
    flat = endpoint.detach().float().flatten(1)
    return flat.var(dim=0, unbiased=False).mean().sqrt().item()


def per_sample_metrics(endpoint, target):
    mse = (endpoint.float() - target.float()).pow(2).flatten(1).mean(1)
    cosine = F.cosine_similarity(
        endpoint.float().flatten(1), target.float().flatten(1), dim=1)
    return mse.cpu().tolist(), cosine.cpu().tolist()


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_generated_images(
        output_dir, no_zv_images, with_zv_images, prompts):
    from torchvision.transforms import functional as vision_functional
    from torchvision.utils import make_grid

    image_dir = os.path.join(output_dir, "generated_images")
    os.makedirs(image_dir, exist_ok=True)
    manifest = []
    interleaved = []
    for index, prompt in enumerate(prompts):
        no_zv_path = os.path.join(
            image_dir, f"sample{index:02d}_t2i_no_zv.png")
        with_zv_path = os.path.join(
            image_dir, f"sample{index:02d}_t2i_with_zv.png")
        vision_functional.to_pil_image(
            no_zv_images[index].cpu()).save(no_zv_path)
        vision_functional.to_pil_image(
            with_zv_images[index].cpu()).save(with_zv_path)
        interleaved.extend([no_zv_images[index].cpu(),
                            with_zv_images[index].cpu()])
        manifest.append({
            "sample_id": index,
            "prompt": prompt,
            "no_zv_image": no_zv_path,
            "with_zv_image": with_zv_path,
            "used_as_i2t_source": with_zv_path,
        })
    grid = make_grid(torch.stack(interleaved), nrow=4, padding=4)
    grid_path = os.path.join(output_dir, "generated_comparison_grid.png")
    vision_functional.to_pil_image(grid).save(grid_path)
    with open(os.path.join(output_dir, "generated_images.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest, grid_path


def plot_roundtrip(path, projected, text_targets, summaries):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    text_color = "#1774C4"
    image_color = "#555555"
    no_zv_color = "#E18118"
    with_zv_color = "#1A9850"
    residual_color = "#9A9A9A"
    specs = [
        ("forward_no_zv", r"(a) Forward T2I flow w/o $\mathbf{z}_{\mathrm{v}}$"),
        ("forward_text_prior", r"(b) Forward T2I flow w/ $\mathbf{z}_{\mathrm{v}}\!\sim\!p_{\mathrm{text}}$"),
        ("backward_no_zv", r"(c) Backward I2T flow w/o $\mathbf{z}_{\mathrm{v}}$"),
        ("backward_image_prior", r"(d) Backward I2T flow w/ $\mathbf{z}_{\mathrm{v}}\!\sim\!p_{\mathrm{image}}$"),
    ]
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })
    figure, axes = plt.subplots(
        1, 4, figsize=(14.4, 3.45), sharex=True, sharey=True)
    for axis, (arm, title) in zip(axes, specs):
        trajectory = projected[arm]
        is_forward = arm.startswith("forward")
        has_zv = arm not in ("forward_no_zv", "backward_no_zv")
        path_color = with_zv_color if has_zv else no_zv_color
        source_color = text_color if is_forward else image_color
        for sample in range(trajectory.shape[1]):
            axis.plot(
                trajectory[:, sample, 0], trajectory[:, sample, 1],
                color=path_color, alpha=0.28, linewidth=0.78, zorder=1)
            axis.scatter(
                trajectory[0, sample, 0], trajectory[0, sample, 1],
                s=20, color=source_color, marker="o", edgecolor="white",
                linewidth=0.25, zorder=3)
            axis.scatter(
                trajectory[-1, sample, 0], trajectory[-1, sample, 1],
                s=22, color=path_color, marker="o", edgecolor="white",
                linewidth=0.25, zorder=4)
            if not is_forward:
                target = text_targets[sample]
                axis.scatter(
                    target[0], target[1], s=28, color=text_color,
                    marker="o", edgecolor="white", linewidth=0.3,
                    alpha=0.88, zorder=2)
                axis.plot(
                    [trajectory[-1, sample, 0], target[0]],
                    [trajectory[-1, sample, 1], target[1]],
                    color=residual_color, alpha=0.18, linewidth=0.55,
                    linestyle="--", zorder=1)
        if is_forward:
            axis.set_title(title, pad=5)
        else:
            summary = summaries[arm]
            axis.set_title(
                f"{title}\nMSE={summary['mean_endpoint_mse']:.3f}, "
                f"cos={summary['mean_endpoint_cosine']:.3f}", pad=5)
        axis.set_xlabel("PCA 1")
        axis.grid(color="#B8B8B8", alpha=0.38, linewidth=0.55)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=3, width=0.7)
    axes[0].set_ylabel("PCA 2")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=text_color, markeredgecolor="white",
               markersize=6, label="text source / target"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=image_color, markeredgecolor="white",
               markersize=6, label="generated-image source"),
        Line2D([0], [0], color=no_zv_color, linewidth=1.2, marker="o",
               markerfacecolor=no_zv_color, markersize=4,
               label=r"endpoint w/o $\mathbf{z}_{\mathrm{v}}$"),
        Line2D([0], [0], color=with_zv_color, linewidth=1.2, marker="o",
               markerfacecolor=with_zv_color, markersize=4,
               label=r"endpoint w/ $\mathbf{z}_{\mathrm{v}}$"),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", ncol=4, frameon=True,
        bbox_to_anchor=(0.5, -0.015), handlelength=1.5,
        columnspacing=1.25)
    figure.subplots_adjust(
        left=0.055, right=0.995, top=0.82, bottom=0.22, wspace=0.10)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.text_temperature < 0.0 or args.zv_temperature < 0.0:
        raise ValueError("sampling temperatures must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("Round-trip export must run in the CUDA Slurm job")
    device = "cuda:0"
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    nnet, clip_encoder, clip_tokenizer, autoencoder = TRAIN.build_models(
        config, device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("format") != "flowtok_bvfm_shared_variational_v2":
        raise RuntimeError(f"Not a shared BVFM checkpoint: {args.checkpoint}")
    nnet.load_state_dict(checkpoint["nnet"])
    for block in nnet.blocks:
        block.forward = block._forward
    nnet.eval().requires_grad_(False)

    from libs.model.bvfm_variational import BVFMVariationalHeads
    variational = BVFMVariationalHeads(
        token_dim=int(config.nnet.model_args.channels),
        latent_dim=int(config.bvfm.latent_dim),
        hidden_dim=int(config.bvfm.hidden_dim),
        dropout=float(config.bvfm.dropout),
        logvar_bias=float(config.bvfm.logvar_bias)).to(device)
    variational.load_state_dict(checkpoint["variational"])
    variational.eval().requires_grad_(False)
    checkpoint_step = int(checkpoint["step"])
    del checkpoint

    prompts = load_coco_prompts(args.coco_captions, args.samples)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        text_ids = clip_tokenizer(prompts).to(device)
        text_features = clip_text_features(clip_encoder, text_ids)
        text_sample, text_mean, _ = nnet(
            text_features, text_encoder=True)
        text_sample = text_mean + float(args.text_temperature) * (
            text_sample - text_mean)
        text_sample = text_sample.float()
        text_mean = text_mean.float()

        text_prior_mu, text_prior_logvar = variational.prior_from_text(
            text_sample)
        z_text = sample_gaussian(
            text_prior_mu.float(), text_prior_logvar.float(),
            args.zv_temperature, args.seed + 101)
        forward_no_zv = TRAIN.collect_shared_trajectory(
            nnet, text_sample, None, args.steps,
            reverse=False, cfg=args.cfg)
        forward_text_prior = TRAIN.collect_shared_trajectory(
            nnet, text_sample, z_text, args.steps,
            reverse=False, cfg=args.cfg)

        no_zv_images = decode_image_tokens(
            autoencoder, forward_no_zv[-1], text_features,
            config.vq_model.scale_factor)
        with_zv_images = decode_image_tokens(
            autoencoder, forward_text_prior[-1], text_features,
            config.vq_model.scale_factor)
        _, generated_image_mean = encode_image_latents(
            autoencoder, with_zv_images, config.vq_model.scale_factor)
        generated_image_mean = generated_image_mean.float()

        image_prior_mu, image_prior_logvar = variational.prior_from_image(
            generated_image_mean)
        z_image = sample_gaussian(
            image_prior_mu.float(), image_prior_logvar.float(),
            args.zv_temperature, args.seed + 211)
        backward_no_zv = TRAIN.collect_shared_trajectory(
            nnet, generated_image_mean, None, args.steps,
            reverse=True, cfg=None)
        backward_image_prior = TRAIN.collect_shared_trajectory(
            nnet, generated_image_mean, z_image, args.steps,
            reverse=True, cfg=None)

    manifest, grid_path = save_generated_images(
        args.output_dir, no_zv_images, with_zv_images, prompts)
    arms = {
        "forward_no_zv": forward_no_zv,
        "forward_text_prior": forward_text_prior,
        "backward_no_zv": backward_no_zv,
        "backward_image_prior": backward_image_prior,
    }
    center, components = TRAIN._fit_shared_pca(
        text_sample, text_mean, generated_image_mean,
        forward_no_zv[-1], forward_text_prior[-1])
    projected = {
        arm: TRAIN._project_shared_pca(states, center, components)
        for arm, states in arms.items()
    }
    text_targets = TRAIN._project_shared_pca(
        text_mean, center, components)

    summaries = {}
    for arm, states in arms.items():
        summary = {"endpoint_spread": endpoint_spread(states[-1])}
        if arm.startswith("backward"):
            mse, cosine = per_sample_metrics(states[-1], text_mean)
            summary.update({
                "mean_endpoint_mse": sum(mse) / len(mse),
                "mean_endpoint_cosine": sum(cosine) / len(cosine),
                "per_sample_mse": mse,
                "per_sample_cosine": cosine,
            })
        summaries[arm] = summary

    plot_path = os.path.join(
        args.output_dir, "trajectory_generated_roundtrip.png")
    plot_roundtrip(plot_path, projected, text_targets, summaries)

    # TikZ-ready plot data only.  Prompts, file paths, latent metrics, and
    # redundant source/endpoint flags belong in metadata rather than the CSV.
    fields = [
        "panel", "sample_id", "point_type", "step_index", "time",
        "pc1", "pc2"]
    panels = {
        "forward_no_zv": "a",
        "forward_text_prior": "b",
        "backward_no_zv": "c",
        "backward_image_prior": "d",
    }
    rows = []
    for arm, states in arms.items():
        is_forward = arm.startswith("forward")
        for step_index in range(args.steps + 1):
            time_value = (
                (1.0 - 1e-5) * step_index / args.steps if is_forward
                else (1.0 - 1e-5) * (1.0 - step_index / args.steps))
            for sample_id in range(args.samples):
                rows.append({
                    "panel": panels[arm],
                    "sample_id": sample_id,
                    "point_type": "trajectory",
                    "step_index": step_index,
                    "time": f"{time_value:.8f}",
                    "pc1": f"{projected[arm][step_index, sample_id, 0]:.8f}",
                    "pc2": f"{projected[arm][step_index, sample_id, 1]:.8f}",
                })
        if not is_forward:
            for sample_id in range(args.samples):
                rows.append({
                    "panel": panels[arm],
                    "sample_id": sample_id,
                    "point_type": "target",
                    "step_index": "",
                    "time": "",
                    "pc1": f"{text_targets[sample_id, 0]:.8f}",
                    "pc2": f"{text_targets[sample_id, 1]:.8f}",
                })
    write_csv(
        os.path.join(args.output_dir, "trajectories_all.csv"), rows, fields)
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as handle:
        json.dump({
            "checkpoint": args.checkpoint,
            "checkpoint_step": checkpoint_step,
            "samples": args.samples,
            "steps": args.steps,
            "cfg": args.cfg,
            "seed": args.seed,
            "text_sample_temperature": args.text_temperature,
            "zv_temperature": args.zv_temperature,
            "i2t_source": "decoded and re-encoded with-z_v T2I images",
            "t2i_paired_targets_shown": False,
            "prompts": prompts,
            "generated_grid": grid_path,
            "summaries": summaries,
        }, handle, indent=2, ensure_ascii=False)
    print(
        "[RESULT] "
        f"t2i_no_zv_spread={summaries['forward_no_zv']['endpoint_spread']:.6f} "
        f"t2i_with_zv_spread={summaries['forward_text_prior']['endpoint_spread']:.6f} | "
        f"i2t_no_zv_mse={summaries['backward_no_zv']['mean_endpoint_mse']:.6f} "
        f"i2t_no_zv_cos={summaries['backward_no_zv']['mean_endpoint_cosine']:.6f} | "
        f"i2t_with_zv_mse={summaries['backward_image_prior']['mean_endpoint_mse']:.6f} "
        f"i2t_with_zv_cos={summaries['backward_image_prior']['mean_endpoint_cosine']:.6f}",
        flush=True)
    print(f"[INFO] outputs={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
