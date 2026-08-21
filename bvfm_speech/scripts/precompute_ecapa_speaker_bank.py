#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biflow.utils import extract_speaker_id_from_path, read_jsonl_rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute one averaged ECAPA/x-vector speaker embedding per training speaker."
    )
    p.add_argument(
        "--manifest",
        type=str,
        default="/work/dankker0900/dataset/processed_svae_unified/full_manifest_clean.jsonl",
    )
    p.add_argument(
        "--out",
        type=str,
        default="/work/dankker0900/dataset/processed_svae_unified/speaker_ecapa_avg.pt",
    )
    p.add_argument("--model", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument(
        "--savedir",
        type=str,
        default="/work/dankker0900/speechbrain_pretrained/spkrec-ecapa-voxceleb",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--top-k-spk", type=int, default=1000)
    p.add_argument("--target-spks", nargs="*", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-utts-per-spk", type=int, default=80)
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-utt-l2-normalize", action="store_true")
    p.add_argument("--no-spk-l2-normalize", action="store_true")
    return p.parse_args()


def row_speaker(row):
    spk = row.get("speaker")
    if spk is not None:
        return str(spk)
    wav = row.get("wav", row.get("parent_wav", ""))
    return str(extract_speaker_id_from_path(wav))


def row_wav(row):
    wav = row.get("wav") or row.get("parent_wav")
    return str(wav) if wav else ""


def select_speakers(rows, target_spks, top_k_spk):
    counts = Counter()
    for row in rows:
        spk = row_speaker(row)
        if spk:
            counts[spk] += 1
    if target_spks:
        return [str(spk) for spk in target_spks]
    return [spk for spk, _ in counts.most_common(int(top_k_spk))]


def load_audio(path, sample_rate):
    import librosa
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if int(sr) != int(sample_rate):
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sample_rate))
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size > 0:
        peak = float(np.max(np.abs(wav)))
        if peak > 1.0:
            wav = wav / peak
    return wav


def main():
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    rows = read_jsonl_rows(args.manifest, max_rows=args.max_rows)
    if not rows:
        raise RuntimeError(f"Empty manifest: {args.manifest}")

    selected = select_speakers(rows, args.target_spks, args.top_k_spk)
    selected_set = set(selected)
    buckets = defaultdict(list)
    for row in rows:
        spk = row_speaker(row)
        wav = row_wav(row)
        if spk in selected_set and wav:
            buckets[spk].append(row)

    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    kwargs = {
        "source": args.model,
        "run_opts": {"device": device},
    }
    if args.savedir:
        kwargs["savedir"] = args.savedir
    classifier = EncoderClassifier.from_hparams(**kwargs)
    if hasattr(classifier, "eval"):
        classifier.eval()

    sums = {}
    counts = {}
    skipped = Counter()
    max_utts = int(args.max_utts_per_spk)
    min_samples = int(float(args.min_sec) * int(args.sample_rate))
    max_samples = int(float(args.max_sec) * int(args.sample_rate)) if args.max_sec else None

    for spk_idx, spk in enumerate(selected, start=1):
        candidates = list(buckets.get(spk, []))
        random.shuffle(candidates)
        if max_utts > 0:
            candidates = candidates[:max_utts]
        for row in candidates:
            wav_path = row_wav(row)
            try:
                wav_np = load_audio(wav_path, int(args.sample_rate))
            except Exception:
                skipped["load_error"] += 1
                continue
            if wav_np.size < min_samples:
                skipped["too_short"] += 1
                continue
            if max_samples is not None and wav_np.size > max_samples:
                wav_np = wav_np[:max_samples]
            wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                emb = classifier.encode_batch(wav).squeeze().detach().cpu().float().reshape(-1)
            if not args.no_utt_l2_normalize:
                emb = F.normalize(emb, p=2, dim=0)
            if spk not in sums:
                sums[spk] = emb.clone()
                counts[spk] = 1
            else:
                sums[spk] += emb
                counts[spk] += 1
        print(f"[{spk_idx:04d}/{len(selected):04d}] spk={spk} used={counts.get(spk, 0)}")

    embeddings = {}
    for spk in selected:
        if spk not in sums:
            continue
        emb = sums[spk] / float(max(1, counts[spk]))
        if not args.no_spk_l2_normalize:
            emb = F.normalize(emb, p=2, dim=0)
        embeddings[spk] = emb.cpu()

    if not embeddings:
        raise RuntimeError("No speaker embeddings were computed.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "speaker_embedding_bank_v1",
        "embedding_type": "ecapa",
        "model": args.model,
        "sample_rate": int(args.sample_rate),
        "source_manifest": os.path.abspath(args.manifest),
        "speaker_ids": list(embeddings.keys()),
        "embeddings": embeddings,
        "counts": {spk: int(counts[spk]) for spk in embeddings},
        "utt_l2_normalize": not args.no_utt_l2_normalize,
        "spk_l2_normalize": not args.no_spk_l2_normalize,
        "skipped": dict(skipped),
    }
    torch.save(payload, out_path)

    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "out": str(out_path),
                "num_speakers": len(embeddings),
                "embedding_dim": int(next(iter(embeddings.values())).numel()),
                "min_utts": int(min(counts[spk] for spk in embeddings)),
                "max_utts": int(max(counts[spk] for spk in embeddings)),
                "skipped": dict(skipped),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    print(f"[OK] saved {out_path}")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
