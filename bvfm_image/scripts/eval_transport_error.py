#!/usr/bin/env python
"""Evaluate paired normalized latent transport error on COCO validation pairs."""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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
    load_val_refs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "configs", "bvfm_image_xl.py"),
    )
    parser.add_argument(
        "--checkpoint",
        default=default_checkpoint(),
    )
    parser.add_argument("--images-dir")
    parser.add_argument("--captions")
    parser.add_argument(
        "--output",
        default=os.path.join(
            REPO_ROOT, "runs", "transport_error", "image_transport.npz"),
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--text-temperature", type=float, default=0.0)
    parser.add_argument("--zv-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    return parser.parse_args()


class PairedCocoDataset(Dataset):
    """One deterministic caption (the first annotation) per COCO image."""

    def __init__(self, images_dir, captions_json, crop_size, samples):
        from torchvision import transforms

        references = load_val_refs(captions_json)
        items = []
        for image_id in sorted(references):
            path = os.path.join(images_dir, f"{image_id:012d}.jpg")
            if os.path.isfile(path):
                items.append((int(image_id), path, references[image_id][0]))
            if len(items) >= int(samples):
                break
        if len(items) < int(samples):
            raise RuntimeError(
                f"Requested {samples} paired COCO samples, found {len(items)} "
                f"under {images_dir}"
            )
        self.items = items
        self.transform = transforms.Compose([
            transforms.Resize(
                int(crop_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(int(crop_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        from PIL import Image

        image_id, path, caption = self.items[index]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, caption, image_id


def sample_gaussian(mu, logvar, temperature, generator):
    if float(temperature) <= 0.0:
        return mu
    noise = torch.randn(
        mu.shape,
        device=mu.device,
        dtype=mu.dtype,
        generator=generator,
    )
    return mu + float(temperature) * torch.exp(0.5 * logvar) * noise


def normalized_error(states, source, target, epsilon):
    """Return per-sample D_i(s), shaped [batch, steps + 1]."""
    states = states.float()
    source = source.float()
    target = target.float()
    numerator = (states - target.unsqueeze(0)).flatten(2).norm(dim=-1)
    denominator = (source - target).flatten(1).norm(dim=-1).clamp_min(
        float(epsilon)
    )
    return (numerator / denominator.unsqueeze(0)).transpose(0, 1).cpu()


def standard_error(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Image transport evaluation must run on CUDA")
    if args.samples <= 0 or args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("samples, batch-size, and steps must be positive")
    if args.text_temperature < 0.0 or args.zv_temperature < 0.0:
        raise ValueError("sampling temperatures must be non-negative")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = "cuda:0"
    config = load_config(args.config)
    images_dir = os.path.abspath(args.images_dir or config.data.val_images_dir)
    captions_json = os.path.abspath(args.captions or config.data.val_captions)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"COCO validation image directory not found: {images_dir}")
    if not os.path.isfile(captions_json):
        raise FileNotFoundError(f"COCO captions file not found: {captions_json}")

    dataset = PairedCocoDataset(
        images_dir,
        captions_json,
        config.dataset.crop_size,
        args.samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True,
    )

    nnet, clip_encoder, clip_tokenizer, autoencoder = TRAIN.build_models(
        config, device
    )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
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
        logvar_bias=float(config.bvfm.logvar_bias),
    ).to(device)
    variational.load_state_dict(checkpoint["variational"])
    variational.eval().requires_grad_(False)
    checkpoint_step = int(checkpoint["step"])
    del checkpoint

    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed) + 104729)
    task_values = {
        "t2i_without_zv": [],
        "t2i_with_zv": [],
        "i2t_without_zv": [],
        "i2t_with_zv": [],
    }
    image_ids = []
    captions = []

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_index, (images, batch_captions, batch_ids) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            batch_captions = list(batch_captions)
            text_ids = clip_tokenizer(batch_captions).to(device)
            text_features = clip_text_features(clip_encoder, text_ids)
            text_sample, text_mean, _ = nnet(text_features, text_encoder=True)
            text_source = text_mean + float(args.text_temperature) * (
                text_sample - text_mean
            )
            text_source = text_source.float()
            text_target = text_mean.float()
            _, image_mean = encode_image_latents(
                autoencoder, images, config.vq_model.scale_factor
            )
            image_target = image_mean.float()

            text_mu, text_logvar = variational.prior_from_text(text_source)
            image_mu, image_logvar = variational.prior_from_image(image_target)
            z_text = sample_gaussian(
                text_mu.float(),
                text_logvar.float(),
                args.zv_temperature,
                generator,
            )
            z_image = sample_gaussian(
                image_mu.float(),
                image_logvar.float(),
                args.zv_temperature,
                generator,
            )

            trajectories = {
                "t2i_without_zv": TRAIN.collect_shared_trajectory(
                    nnet,
                    text_source,
                    None,
                    args.steps,
                    reverse=False,
                    cfg=args.cfg,
                ),
                "t2i_with_zv": TRAIN.collect_shared_trajectory(
                    nnet,
                    text_source,
                    z_text,
                    args.steps,
                    reverse=False,
                    cfg=args.cfg,
                ),
                "i2t_without_zv": TRAIN.collect_shared_trajectory(
                    nnet,
                    image_target,
                    None,
                    args.steps,
                    reverse=True,
                    cfg=None,
                ),
                "i2t_with_zv": TRAIN.collect_shared_trajectory(
                    nnet,
                    image_target,
                    z_image,
                    args.steps,
                    reverse=True,
                    cfg=None,
                ),
            }
            task_values["t2i_without_zv"].append(
                normalized_error(
                    trajectories["t2i_without_zv"],
                    text_source,
                    image_target,
                    args.epsilon,
                )
            )
            task_values["t2i_with_zv"].append(
                normalized_error(
                    trajectories["t2i_with_zv"],
                    text_source,
                    image_target,
                    args.epsilon,
                )
            )
            task_values["i2t_without_zv"].append(
                normalized_error(
                    trajectories["i2t_without_zv"],
                    image_target,
                    text_target,
                    args.epsilon,
                )
            )
            task_values["i2t_with_zv"].append(
                normalized_error(
                    trajectories["i2t_with_zv"],
                    image_target,
                    text_target,
                    args.epsilon,
                )
            )
            image_ids.extend(int(value) for value in batch_ids.tolist())
            captions.extend(batch_captions)
            print(
                f"[BATCH] {batch_index + 1}/{len(loader)} "
                f"samples={len(image_ids)}/{len(dataset)}",
                flush=True,
            )

    arrays = {
        key: torch.cat(chunks, dim=0).numpy().astype(np.float32)
        for key, chunks in task_values.items()
    }
    progress = np.linspace(0.0, 1.0, int(args.steps) + 1, dtype=np.float32)
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite normalized error in {key}")
        if not np.allclose(values[:, 0], 1.0, atol=2e-4, rtol=2e-4):
            raise RuntimeError(
                f"Normalization invariant failed for {key}: "
                f"D(0) range=({values[:, 0].min()}, {values[:, 0].max()})"
            )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(1, dtype=np.int64),
        domain=np.asarray("image"),
        progress=progress,
        image_ids=np.asarray(image_ids, dtype=np.int64),
        captions=np.asarray(captions, dtype=np.str_),
        **arrays,
    )
    summary = {
        "schema_version": 1,
        "domain": "image",
        "output": output_path,
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "config": os.path.abspath(args.config),
        "images_dir": images_dir,
        "captions": captions_json,
        "samples": len(image_ids),
        "steps": int(args.steps),
        "solver": "euler",
        "cfg": float(args.cfg),
        "text_temperature": float(args.text_temperature),
        "zv_temperature": float(args.zv_temperature),
        "seed": int(args.seed),
        "epsilon": float(args.epsilon),
        "paired_target": True,
        "caption_policy": "first COCO annotation for each sorted image id",
        "normalization": "||z_hat(s)-z_target||_F/(||z_source-z_target||_F+epsilon)",
        "tasks": {},
    }
    for task in ("t2i", "i2t"):
        without = arrays[f"{task}_without_zv"][:, -1]
        with_zv = arrays[f"{task}_with_zv"][:, -1]
        summary["tasks"][task] = {
            "without_zv_endpoint_mean": float(without.mean()),
            "without_zv_endpoint_se": standard_error(without),
            "with_zv_endpoint_mean": float(with_zv.mean()),
            "with_zv_endpoint_se": standard_error(with_zv),
            "paired_endpoint_delta_mean": float((without - with_zv).mean()),
        }
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
