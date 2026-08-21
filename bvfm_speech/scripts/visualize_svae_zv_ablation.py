#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_tts_test_topk import TTSEvaluator
from infer_tts_one import load_json, resolve_checkpoint, set_seed, synthesize_one
from plot_svae_latent_heatmap_norm import (
    frame_norm,
    load_latent,
    plot_comparison,
    plot_one,
    summarize,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and visualize matched TTS samples with z_v from its prior and with z_v=0."
    )
    parser.add_argument(
        "--ckpt-dir",
        default=str(REPO_ROOT / "checkpoints" / "ckpt_joint_svae_zeroshot_norm"),
    )
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument(
        "--text",
        default=(
            "THE MAGICIAN IS VERY BUSY AS I SAID BUT IF YOU WILL PROMISE NOT TO "
            "DISTURB HIM YOU MAY COME INTO HIS WORKSHOP AND WATCH HIM PREPARE A WONDERFUL CHARM"
        ),
    )
    parser.add_argument(
        "--ref-wav",
        default=(
            "/work/dankker0900/dataset/libritts_r/LibriTTS_R/test-clean/237/126133/"
            "237_126133_000002_000003.wav"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "svae_latent_visualizations" / "zv_ablation"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solver", choices=["heun", "euler"], default="heun")
    parser.add_argument("--nfe", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--prior-temp", type=float, default=0.0)
    parser.add_argument("--style-temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--percentile", type=float, default=99.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_path = resolve_checkpoint(ckpt_dir, args.checkpoint)
    model_cfg_path = ckpt_dir / "merged_config.json"
    if not model_cfg_path.exists():
        model_cfg_path = REPO_ROOT / "configs" / "cutmanifest_svae_latent.json"
    model_cfg = load_json(model_cfg_path)

    print(f"[INFO] checkpoint={ckpt_path}")
    print(f"[INFO] model_config={model_cfg_path}")
    print(f"[INFO] output={out_dir}")
    evaluator = TTSEvaluator(model_cfg, ckpt_path, out_dir, args.device)

    results = {}
    for name, style_mode in (("with_zv", "prior"), ("without_zv", "zero")):
        set_seed(args.seed)
        results[name] = synthesize_one(
            evaluator,
            text=args.text,
            ref_wav=args.ref_wav,
            solver=args.solver,
            nfe=args.nfe,
            cfg_scale=args.cfg_scale,
            prior_temp=args.prior_temp,
            style_temp=args.style_temp,
            style_mode=style_mode,
            out_wav=out_dir / f"{name}.wav",
            out_mel=out_dir / f"{name}_svae_latent.npy",
        )

    with_zv = load_latent(out_dir / "with_zv_svae_latent.npy")
    without_zv = load_latent(out_dir / "without_zv_svae_latent.npy")
    values = np.concatenate([with_zv.reshape(-1), without_zv.reshape(-1)])
    vmax = max(float(np.percentile(np.abs(values), args.percentile)), 1e-6)
    norm_ylim = 1.05 * max(frame_norm(with_zv).max(), frame_norm(without_zv).max(), 1e-6)

    with_plot = out_dir / "with_zv_heatmap_norm.png"
    without_plot = out_dir / "without_zv_heatmap_norm.png"
    comparison_plot = out_dir / "zv_ablation_comparison_heatmap_norm.png"
    plot_one(with_zv, "Semantic-VAE latent with $z_v$", with_plot, vmax, norm_ylim)
    plot_one(without_zv, "Semantic-VAE latent without $z_v$", without_plot, vmax, norm_ylim)
    plot_comparison(with_zv, without_zv, comparison_plot, vmax, norm_ylim)

    summary = {
        "checkpoint": os.path.abspath(ckpt_path),
        "model_config": str(model_cfg_path.resolve()),
        "text": args.text,
        "ref_wav": os.path.abspath(args.ref_wav),
        "seed": args.seed,
        "solver": args.solver,
        "nfe": args.nfe,
        "prior_temp": args.prior_temp,
        "style_temp": args.style_temp,
        "shared_abs_percentile": args.percentile,
        "shared_color_limit": vmax,
        "shared_norm_ylim": float(norm_ylim),
        "with_zv": {**results["with_zv"], "latent_stats": summarize(with_zv)},
        "without_zv": {**results["without_zv"], "latent_stats": summarize(without_zv)},
        "plots": {
            "with_zv": str(with_plot),
            "without_zv": str(without_plot),
            "comparison": str(comparison_plot),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
