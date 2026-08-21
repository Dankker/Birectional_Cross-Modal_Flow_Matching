#!/usr/bin/env python3
"""Combine speech and image transport exports into the four-panel paper plot."""

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speech",
        default=os.path.join(
            REPO_ROOT,
            "bvfm_speech",
            "runs",
            "transport_error",
            "speech_transport.npz",
        ),
    )
    parser.add_argument(
        "--image",
        default=os.path.join(
            REPO_ROOT,
            "bvfm_image",
            "runs",
            "transport_error",
            "image_transport.npz",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "runs", "transport_error"),
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def scalar_text(value):
    array = np.asarray(value)
    return str(array.item())


def load_export(path, expected_domain, tasks):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    if int(np.asarray(data["schema_version"]).item()) != 1:
        raise RuntimeError(f"Unsupported schema in {path}")
    domain = scalar_text(data["domain"])
    if domain != expected_domain:
        raise RuntimeError(
            f"Expected domain={expected_domain!r}, found {domain!r} in {path}"
        )
    progress = np.asarray(data["progress"], dtype=np.float64)
    if progress.ndim != 1 or len(progress) < 2:
        raise RuntimeError(f"Bad progress array in {path}: {progress.shape}")
    if not np.allclose(progress[[0, -1]], [0.0, 1.0], atol=1e-6):
        raise RuntimeError(f"Progress must span [0,1] in {path}")
    curves = {}
    for task in tasks:
        curves[task] = {}
        for condition in ("without_zv", "with_zv"):
            key = f"{task}_{condition}"
            values = np.asarray(data[key], dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != len(progress):
                raise RuntimeError(f"Bad {key} shape in {path}: {values.shape}")
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite values in {path}:{key}")
            if not np.allclose(values[:, 0], 1.0, atol=2e-4, rtol=2e-4):
                raise RuntimeError(f"D(0) != 1 in {path}:{key}")
            curves[task][condition] = values
    id_key = "utterance_ids" if expected_domain == "speech" else "image_ids"
    sample_ids = np.asarray(data[id_key]).astype(str).tolist()
    return {
        "path": path,
        "domain": domain,
        "progress": progress,
        "curves": curves,
        "sample_ids": sample_ids,
    }


def curve_stats(values):
    count = int(values.shape[0])
    mean = values.mean(axis=0)
    if count <= 1:
        se = np.zeros_like(mean)
    else:
        se = values.std(axis=0, ddof=1) / math.sqrt(count)
    return mean, se


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    speech = load_export(args.speech, "speech", ("tts", "asr"))
    image = load_export(args.image, "image", ("t2i", "i2t"))
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    task_sources = {
        "tts": speech,
        "asr": speech,
        "t2i": image,
        "i2t": image,
    }
    task_specs = [
        ("tts", "(a) TTS"),
        ("asr", "(b) ASR"),
        ("t2i", "(c) T2I"),
        ("i2t", "(d) I2T"),
    ]
    conditions = [
        ("without_zv", r"w/o $\mathbf{z}_{\mathrm{v}}$", "#D97706"),
        ("with_zv", r"w/ $\mathbf{z}_{\mathrm{v}}$", "#16803C"),
    ]

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
    })
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 6.1), sharex=True, sharey=True)
    aggregate_rows = []
    sample_rows = []
    summary = {
        "schema_version": 1,
        "speech_input": speech["path"],
        "image_input": image["path"],
        "shading": "mean +/- standard error across paired samples",
        "tasks": {},
    }
    global_upper = 1.0
    for axis, (task, title) in zip(axes.flat, task_specs):
        source = task_sources[task]
        progress = source["progress"]
        task_summary = {"samples": len(source["sample_ids"])}
        for condition, label, color in conditions:
            values = source["curves"][task][condition]
            mean, se = curve_stats(values)
            global_upper = max(global_upper, float(np.max(mean + se)))
            axis.plot(progress, mean, color=color, linewidth=2.0, label=label)
            axis.fill_between(
                progress,
                np.maximum(0.0, mean - se),
                mean + se,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            task_summary[condition] = {
                "endpoint_mean": float(mean[-1]),
                "endpoint_se": float(se[-1]),
            }
            for step_index, (s_value, mean_value, se_value) in enumerate(
                zip(progress, mean, se)
            ):
                aggregate_rows.append({
                    "task": task.upper(),
                    "condition": condition,
                    "step_index": step_index,
                    "progress": f"{s_value:.8f}",
                    "mean": f"{mean_value:.8f}",
                    "standard_error": f"{se_value:.8f}",
                    "samples": values.shape[0],
                })
            for sample_index, sample_id in enumerate(source["sample_ids"]):
                for step_index, (s_value, error_value) in enumerate(
                    zip(progress, values[sample_index])
                ):
                    sample_rows.append({
                        "task": task.upper(),
                        "condition": condition,
                        "sample_id": sample_id,
                        "step_index": step_index,
                        "progress": f"{s_value:.8f}",
                        "normalized_error": f"{error_value:.8f}",
                    })
        without_endpoint = source["curves"][task]["without_zv"][:, -1]
        with_endpoint = source["curves"][task]["with_zv"][:, -1]
        paired_delta = without_endpoint - with_endpoint
        delta_se = (
            float(paired_delta.std(ddof=1) / math.sqrt(len(paired_delta)))
            if len(paired_delta) > 1
            else 0.0
        )
        task_summary["paired_endpoint_delta_mean"] = float(paired_delta.mean())
        task_summary["paired_endpoint_delta_se"] = delta_se
        summary["tasks"][task] = task_summary

        axis.axhline(1.0, color="#777777", linewidth=0.7, linestyle=":", alpha=0.65)
        axis.set_title(
            f"{title}\n"
            + r"$\Delta D(1)=$"
            + f"{paired_delta.mean():.3f}"
        )
        axis.grid(color="#B8B8B8", alpha=0.35, linewidth=0.55)
        axis.set_axisbelow(True)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Normalized integration progress")

    y_upper = max(1.05, 1.04 * global_upper)
    for axis in axes.flat:
        axis.set_ylim(0.0, y_upper)
    axes[0, 0].set_ylabel("Normalized transport error")
    axes[1, 0].set_ylabel("Normalized transport error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.subplots_adjust(
        left=0.105, right=0.985, top=0.92, bottom=0.13, hspace=0.34, wspace=0.13
    )
    figure_path = os.path.join(output_dir, "normalized_transport_error.png")
    figure.savefig(figure_path, dpi=int(args.dpi), bbox_inches="tight")
    figure.savefig(
        os.path.join(output_dir, "normalized_transport_error.pdf"),
        bbox_inches="tight",
    )
    plt.close(figure)

    aggregate_path = os.path.join(output_dir, "transport_error_curves.csv")
    sample_path = os.path.join(output_dir, "transport_error_samples.csv")
    write_csv(
        aggregate_path,
        [
            "task",
            "condition",
            "step_index",
            "progress",
            "mean",
            "standard_error",
            "samples",
        ],
        aggregate_rows,
    )
    write_csv(
        sample_path,
        [
            "task",
            "condition",
            "sample_id",
            "step_index",
            "progress",
            "normalized_error",
        ],
        sample_rows,
    )
    summary.update({
        "figure_png": figure_path,
        "figure_pdf": os.path.join(output_dir, "normalized_transport_error.pdf"),
        "aggregate_csv": aggregate_path,
        "sample_csv": sample_path,
    })
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
