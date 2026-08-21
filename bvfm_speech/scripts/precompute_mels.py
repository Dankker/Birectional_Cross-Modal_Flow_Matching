#!/usr/bin/env python3

import argparse
import hashlib
import os
import sys
from collections import OrderedDict

import librosa
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIGVGAN_ROOT = os.path.join(REPO_ROOT, "BigVGAN")
if BIGVGAN_ROOT not in sys.path:
    sys.path.insert(0, BIGVGAN_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bigvgan
from biflow.config import load_config
from biflow.utils import read_jsonl_rows
from meldataset import get_mel_spectrogram


def sha1_key(text: str):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def select_target_speakers(rows, target_spks, top_k_spk):
    spk_count = {}
    for row in rows:
        spk = str(row.get("speaker", ""))
        if spk:
            spk_count[spk] = spk_count.get(spk, 0) + 1
    if target_spks is not None:
        return list(target_spks)
    spk_sorted = sorted(spk_count.items(), key=lambda kv: kv[1], reverse=True)
    return [s for (s, _) in spk_sorted[:top_k_spk]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "cutmanifest_singlevf_stable.json"))
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg["cache"]["mel_dir"]
    assert output_dir, "cache.mel_dir or --output-dir is required"
    os.makedirs(output_dir, exist_ok=True)

    cut_rows_all = read_jsonl_rows(cfg["paths"]["cut_manifest"], max_rows=cfg["paths"]["max_cut_rows"])
    target_spks = select_target_speakers(
        cut_rows_all,
        cfg["data"]["target_spks"],
        int(cfg["data"]["top_k_spk"]),
    )
    target_spk_set = set(target_spks)
    cut_rows = [r for r in cut_rows_all if str(r.get("speaker", "")) in target_spk_set]
    unique_wavs = list(OrderedDict.fromkeys([r["parent_wav"] for r in cut_rows]).keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = bigvgan.BigVGAN.from_pretrained(cfg["model"]["bigvgan_name"], use_cuda_kernel=False).to(device).eval()
    h = model.h
    sr = int(h.sampling_rate)
    hop_size = int(h.hop_size)

    print(f"[MEL-CACHE] device={device} wavs={len(unique_wavs)} out={output_dir}")
    done = 0
    skipped = 0
    for idx, wav_path in enumerate(unique_wavs, start=1):
        out_path = os.path.join(output_dir, f"{sha1_key(os.path.abspath(wav_path))}.pt")
        if os.path.exists(out_path):
            skipped += 1
            continue
        wav_np, _ = librosa.load(wav_path, sr=sr, mono=True)
        rem = wav_np.shape[0] % hop_size
        if rem != 0:
            wav_np = wav_np[:-rem]
        wav_np = librosa.util.normalize(wav_np) * 0.95
        wav = torch.tensor(wav_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mel = get_mel_spectrogram(wav, h)
        mel_cpu = mel[0].detach().cpu().float().transpose(0, 1).contiguous()
        torch.save(mel_cpu, out_path)
        done += 1
        if idx % 100 == 0 or idx == len(unique_wavs):
            print(f"[MEL-CACHE] progress {idx}/{len(unique_wavs)} wrote={done} skipped={skipped}")


if __name__ == "__main__":
    main()
