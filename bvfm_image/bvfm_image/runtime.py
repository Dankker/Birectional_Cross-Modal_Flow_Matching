"""Checkpoint loading and shared-field inference primitives."""

from dataclasses import dataclass

import torch

from bvfm_image.common import clip_text_features, encode_image_latents
from bvfm_image.model_factory import build_field


@dataclass
class ModelBundle:
    field: torch.nn.Module
    variational: torch.nn.Module
    decoder: torch.nn.Module
    autoencoder: torch.nn.Module
    clip_encoder: torch.nn.Module
    clip_tokenizer: object
    checkpoint_step: int


def build_clip(device):
    import open_clip

    try:
        encoder, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336-quickgelu", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-L-14-336-quickgelu")
    except Exception:
        encoder, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14-336", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-L-14-336")
    del encoder.visual
    encoder.transformer.batch_first = False
    encoder.eval().requires_grad_(False).to(device)
    return encoder, tokenizer


def load_bundle(config, checkpoint_path, device):
    from libs.flowtitok import FlowTiTok
    from libs.model.bvfm_variational import BVFMVariationalHeads
    from libs.model.text_decoder_ar import ARTextDecoder

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("format") != "flowtok_bvfm_shared_variational_v2":
        raise RuntimeError(f"Not a BVFM-image checkpoint: {checkpoint_path}")

    field = build_field(config)
    field.load_state_dict(checkpoint["nnet"])
    for block in field.blocks:
        block.forward = block._forward
    field.eval().requires_grad_(False).to(device)

    variational = BVFMVariationalHeads(
        token_dim=int(config.nnet.model_args.channels),
        latent_dim=int(config.bvfm.latent_dim),
        hidden_dim=int(config.bvfm.hidden_dim),
        dropout=float(config.bvfm.dropout),
        logvar_bias=float(config.bvfm.logvar_bias)).to(device)
    variational.load_state_dict(checkpoint["variational"])
    variational.eval().requires_grad_(False)

    decoder = ARTextDecoder(**checkpoint["decoder_config"]).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval().requires_grad_(False)

    autoencoder = FlowTiTok(config)
    tokenizer_state = torch.load(
        config.tokenizer_checkpoint, map_location="cpu",
        weights_only=True, mmap=True)
    autoencoder.load_state_dict(tokenizer_state)
    del tokenizer_state
    autoencoder.eval().requires_grad_(False).to(device)

    clip_encoder, clip_tokenizer = build_clip(device)
    step = int(checkpoint["step"])
    del checkpoint
    return ModelBundle(
        field=field,
        variational=variational,
        decoder=decoder,
        autoencoder=autoencoder,
        clip_encoder=clip_encoder,
        clip_tokenizer=clip_tokenizer,
        checkpoint_step=step)


def sample_gaussian(mu, logvar, temperature, generator=None):
    if float(temperature) <= 0.0:
        return mu
    noise = torch.randn(
        mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + float(temperature) * torch.exp(0.5 * logvar) * noise


@torch.no_grad()
def encode_text(bundle, prompts, temperature=1.0):
    token_ids = bundle.clip_tokenizer(prompts).to(
        next(bundle.field.parameters()).device)
    features = clip_text_features(bundle.clip_encoder, token_ids)
    sample, mean, logvar = bundle.field(features, text_encoder=True)
    sample = mean + float(temperature) * (sample - mean)
    return sample.float(), mean.float(), logvar.float(), features, token_ids


@torch.no_grad()
def integrate(field, source, variation_latent, steps, reverse=False, cfg=None):
    batch = source.shape[0]
    device = source.device
    grid = torch.linspace(
        1.0 - 1e-5 if reverse else 0.0,
        0.0 if reverse else 1.0 - 1e-5,
        int(steps) + 1, device=device)
    state = source.float()
    conditional = torch.zeros(batch, dtype=torch.bool, device=device)
    unconditional = torch.ones(batch, dtype=torch.bool, device=device)
    for index in range(int(steps)):
        time = grid[index].expand(batch)
        conditional_velocity = field(
            state, t=time, null_indicator=conditional,
            variation_latent=variation_latent)[0]
        if cfg is None:
            velocity = conditional_velocity
        else:
            unconditional_velocity = field(
                state, t=time, null_indicator=unconditional,
                variation_latent=variation_latent)[0]
            velocity = unconditional_velocity + float(cfg) * (
                conditional_velocity - unconditional_velocity)
        state = state + velocity.float() * (grid[index + 1] - grid[index])
    return state


@torch.no_grad()
def decode_image_tokens(bundle, tokens, text_features):
    token_grid = (
        tokens.permute(0, 2, 1).unsqueeze(2)
        / float(bundle.autoencoder.config.vq_model.scale_factor))
    return bundle.autoencoder.decode_tokens(
        token_grid, text_guidance=text_features).float().clamp(0.0, 1.0)


@torch.no_grad()
def encode_images(bundle, images, scale_factor):
    return encode_image_latents(bundle.autoencoder, images, scale_factor)
