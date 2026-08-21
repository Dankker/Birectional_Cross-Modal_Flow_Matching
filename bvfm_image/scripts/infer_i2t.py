#!/usr/bin/env python
"""Caption images with reverse integration of the same BVFM field."""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def default_checkpoint():
    weights_root = os.environ.get("BVFM_WEIGHTS_ROOT")
    if weights_root:
        return os.path.join(
            weights_root, "image", "bvfm_image_step40000.pt")
    return os.path.join(REPO_ROOT, "checkpoints", "bvfm_image_step40000.pt")

from bvfm_image.common import ImagePathDataset, ids_to_text, load_config  # noqa: E402
from bvfm_image.runtime import (  # noqa: E402
    encode_images,
    integrate,
    load_bundle,
    sample_gaussian,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=os.path.join(
            REPO_ROOT, "configs", "bvfm_image_xl.py"))
    parser.add_argument(
        "--checkpoint", default=default_checkpoint())
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--image-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--zv-mode", choices=["none", "mean", "sample"], default="mean")
    parser.add_argument("--zv-temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def collect_paths(args):
    paths = [os.path.abspath(path) for path in args.image]
    if args.image_dir:
        paths.extend(
            os.path.join(os.path.abspath(args.image_dir), name)
            for name in sorted(os.listdir(args.image_dir))
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    paths = list(dict.fromkeys(paths))
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing image(s): {missing[:4]}")
    if not paths:
        raise ValueError("Provide --image PATH or --image-dir DIR")
    return paths


def decode_one(decoder, endpoint, tokenizer, beam):
    if int(beam) > 1:
        ids = decoder.generate_beam(endpoint.float(), beam=int(beam))[0]
    else:
        ids = decoder.generate(endpoint.float())[0]
    return ids_to_text(ids, tokenizer)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("I2T inference must run in a CUDA Slurm job")
    device = "cuda:0"
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    bundle = load_bundle(config, args.checkpoint, device)
    paths = collect_paths(args)
    loader = DataLoader(
        ImagePathDataset(paths, int(config.dataset.crop_size)),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True)

    outputs = []
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 2027)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for images, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            _, image_mean = encode_images(
                bundle, images, config.vq_model.scale_factor)
            prior_mu, prior_logvar = bundle.variational.prior_from_image(
                image_mean)
            if args.zv_mode == "none":
                z_v = None
            elif args.zv_mode == "mean":
                z_v = prior_mu.float()
            else:
                z_v = sample_gaussian(
                    prior_mu.float(), prior_logvar.float(),
                    args.zv_temperature, generator)
            endpoints = integrate(
                bundle.field, image_mean, z_v, args.steps,
                reverse=True, cfg=None)
            for index, path in enumerate(batch_paths):
                outputs.append({
                    "image": path,
                    "caption": decode_one(
                        bundle.decoder, endpoints[index:index + 1],
                        bundle.clip_tokenizer, args.beam),
                })

    result = {
        "task": "i2t",
        "checkpoint": args.checkpoint,
        "checkpoint_step": bundle.checkpoint_step,
        "zv_mode": args.zv_mode,
        "zv_temperature": args.zv_temperature,
        "steps": args.steps,
        "beam": args.beam,
        "seed": args.seed,
        "outputs": outputs,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"[RESULT] captioned={len(outputs)} output={output_path}")
    for item in outputs[:8]:
        print(f"[CAPTION] {item['image']} :: {item['caption']}")


if __name__ == "__main__":
    main()
