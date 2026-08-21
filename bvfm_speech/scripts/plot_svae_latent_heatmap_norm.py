#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot comparable SVAE latent heatmaps and frame-wise L2 norm curves."
    )
    parser.add_argument("--with-zv", required=True, help="SVAE latent .npy generated with z_v.")
    parser.add_argument("--without-zv", required=True, help="SVAE latent .npy generated with z_v=0.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="svae_zv_ablation")
    parser.add_argument("--percentile", type=float, default=99.0)
    return parser.parse_args()


def load_latent(path):
    latent = np.asarray(np.load(path), dtype=np.float32)
    latent = np.squeeze(latent)
    if latent.ndim != 2:
        raise ValueError(f"Expected a 2-D [T, D] latent in {path}, got {latent.shape}")
    if latent.shape[0] <= 128 and latent.shape[1] > latent.shape[0]:
        latent = latent.T
    return latent


def frame_norm(latent):
    return np.linalg.norm(latent, axis=1)


def plot_one(latent, title, out_path, vmax, norm_ylim):
    frames = np.arange(latent.shape[0])
    norms = frame_norm(latent)
    fig, (ax_hm, ax_norm) = plt.subplots(
        2,
        1,
        figsize=(12, 6.5),
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    image = ax_hm.imshow(
        latent.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax_hm.set_title(title)
    ax_hm.set_ylabel("SVAE channel")
    ax_hm.set_xlabel("Frame")
    fig.colorbar(image, ax=ax_hm, label="Latent value", pad=0.01)

    ax_norm.plot(frames, norms, color="#1665a7", linewidth=1.6)
    ax_norm.fill_between(frames, norms, color="#6baed6", alpha=0.22)
    ax_norm.set_xlim(0, max(latent.shape[0] - 1, 1))
    ax_norm.set_ylim(0, norm_ylim)
    ax_norm.set_ylabel("L2 norm")
    ax_norm.set_xlabel("Frame")
    ax_norm.grid(alpha=0.22)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(with_zv, without_zv, out_path, vmax, norm_ylim):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 8),
        gridspec_kw={"height_ratios": [3.2, 1.0]},
        constrained_layout=True,
    )
    image = None
    variants = (
        (with_zv, "With $z_v$ (TTS prior)", "#1665a7"),
        (without_zv, "Without $z_v$ ($z_v=0$)", "#d95f02"),
    )
    for column, (latent, label, color) in enumerate(variants):
        image = axes[0, column].imshow(
            latent.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[0, column].set_title(label)
        axes[0, column].set_xlabel("Frame")
        axes[0, column].set_ylabel("SVAE channel")

        norms = frame_norm(latent)
        frames = np.arange(latent.shape[0])
        axes[1, column].plot(frames, norms, color=color, linewidth=1.6)
        axes[1, column].fill_between(frames, norms, color=color, alpha=0.18)
        axes[1, column].set_xlim(0, max(latent.shape[0] - 1, 1))
        axes[1, column].set_ylim(0, norm_ylim)
        axes[1, column].set_xlabel("Frame")
        axes[1, column].set_ylabel("L2 norm")
        axes[1, column].grid(alpha=0.22)

    fig.colorbar(image, ax=axes[0, :], label="Latent value", pad=0.01, shrink=0.92)
    fig.suptitle("Semantic-VAE speech latent: $z_v$ ablation", fontsize=16)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(latent):
    norms = frame_norm(latent)
    return {
        "shape": list(latent.shape),
        "mean": float(latent.mean()),
        "std": float(latent.std()),
        "abs_mean": float(np.abs(latent).mean()),
        "frame_norm_mean": float(norms.mean()),
        "frame_norm_std": float(norms.std()),
        "frame_norm_min": float(norms.min()),
        "frame_norm_max": float(norms.max()),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with_zv = load_latent(args.with_zv)
    without_zv = load_latent(args.without_zv)

    values = np.concatenate([with_zv.reshape(-1), without_zv.reshape(-1)])
    vmax = max(float(np.percentile(np.abs(values), args.percentile)), 1e-6)
    norm_ylim = 1.05 * max(frame_norm(with_zv).max(), frame_norm(without_zv).max(), 1e-6)

    with_path = out_dir / f"{args.prefix}_with_zv_heatmap_norm.png"
    without_path = out_dir / f"{args.prefix}_without_zv_heatmap_norm.png"
    compare_path = out_dir / f"{args.prefix}_comparison_heatmap_norm.png"
    stats_path = out_dir / f"{args.prefix}_stats.json"

    plot_one(with_zv, "Semantic-VAE latent with $z_v$", with_path, vmax, norm_ylim)
    plot_one(without_zv, "Semantic-VAE latent without $z_v$", without_path, vmax, norm_ylim)
    plot_comparison(with_zv, without_zv, compare_path, vmax, norm_ylim)

    stats = {
        "shared_abs_percentile": float(args.percentile),
        "shared_color_limit": vmax,
        "shared_norm_ylim": float(norm_ylim),
        "with_zv": summarize(with_zv),
        "without_zv": summarize(without_zv),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "with_zv_plot": str(with_path),
        "without_zv_plot": str(without_path),
        "comparison_plot": str(compare_path),
        "stats": str(stats_path),
    }, indent=2))


if __name__ == "__main__":
    main()
