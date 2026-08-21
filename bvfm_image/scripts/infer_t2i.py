#!/usr/bin/env python
"""Generate images with the selected bidirectional BVFM checkpoint."""

import argparse
import json
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def default_checkpoint():
    weights_root = os.environ.get("BVFM_WEIGHTS_ROOT")
    if weights_root:
        return os.path.join(
            weights_root, "image", "bvfm_image_step40000.pt")
    return os.path.join(REPO_ROOT, "checkpoints", "bvfm_image_step40000.pt")

from bvfm_image.common import load_config  # noqa: E402
from bvfm_image.runtime import (  # noqa: E402
    decode_image_tokens,
    encode_text,
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
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument(
        "--zv-mode", choices=["none", "mean", "sample"], default="mean")
    parser.add_argument("--zv-temperature", type=float, default=1.0)
    parser.add_argument("--text-temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_prompts(args):
    prompts = list(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file) as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    if not prompts:
        prompts = ["A corgi wearing sunglasses on a tropical beach at sunset"]
    return prompts


def main():
    from torchvision.transforms import functional as vision_functional

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("T2I inference must run in a CUDA Slurm job")
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be positive")
    device = "cuda:0"
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    bundle = load_bundle(config, args.checkpoint, device)
    prompts = load_prompts(args)
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for prompt_index, prompt in enumerate(prompts):
            source, _, _, features, _ = encode_text(
                bundle, [prompt], args.text_temperature)
            count = int(args.samples_per_prompt)
            source = source.expand(count, -1, -1).contiguous()
            features = features.expand(count, -1, -1).contiguous()
            prior_mu, prior_logvar = bundle.variational.prior_from_text(
                source[:1])
            if args.zv_mode == "none":
                z_v = None
            elif args.zv_mode == "mean":
                z_v = prior_mu.float().expand(count, -1).contiguous()
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + 1009 * prompt_index)
                z_v = sample_gaussian(
                    prior_mu.float().expand(count, -1),
                    prior_logvar.float().expand(count, -1),
                    args.zv_temperature, generator)
            endpoint = integrate(
                bundle.field, source, z_v, args.steps,
                reverse=False, cfg=args.cfg)
            images = decode_image_tokens(bundle, endpoint, features)
            for sample_index in range(count):
                filename = (
                    f"prompt{prompt_index:03d}_sample{sample_index:03d}.png")
                path = os.path.join(args.output_dir, filename)
                vision_functional.to_pil_image(
                    images[sample_index].cpu()).save(path)
                records.append({
                    "prompt_id": prompt_index,
                    "sample_id": sample_index,
                    "prompt": prompt,
                    "image": path,
                })

    manifest = {
        "task": "t2i",
        "checkpoint": args.checkpoint,
        "checkpoint_step": bundle.checkpoint_step,
        "zv_mode": args.zv_mode,
        "zv_temperature": args.zv_temperature,
        "text_temperature": args.text_temperature,
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "outputs": records,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"[RESULT] generated={len(records)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
