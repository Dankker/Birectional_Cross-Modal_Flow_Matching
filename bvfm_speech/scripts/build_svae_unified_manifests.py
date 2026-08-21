#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build unified ASR/TTS manifests for the Semantic-VAE speech space.

This script replaces the old 100-bin mel target with a 40 Hz, 64-dim
Semantic-VAE latent target. It intentionally preserves the old processed
manifest field names such as mel_len/start_mel/end_mel so the training code can
reuse the same duration/chunk logic; those fields now mean "speech frames" in
the Semantic-VAE latent timeline.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biflow.utils import normalize_text_basic


PUNCT_CHARS = set(",.;:?!-\"'()[]{}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Semantic-VAE latents and build 40Hz unified manifests."
    )
    p.add_argument("--manifest-tsv", default="/work/dankker0900/dataset/manifest.tsv")
    p.add_argument("--aligned-jsonl", default="/work/dankker0900/dataset/align/train_manifest_aligned.jsonl")
    p.add_argument("--out-dir", default="/work/dankker0900/dataset/processed_svae_unified")
    p.add_argument("--semantic-vae-root", default="/work/dankker0900/bvfm/bvfm_speech/Semantic-VAE")
    p.add_argument("--semantic-vae-ckpt", default="/work/dankker0900/bvfm/bvfm_speech/Semantic-VAE/ckpts/semantic_vae_1000k")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-use-ema", action="store_false", dest="use_ema")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--latent-subdir", default="svae_latents")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--latent-fps", type=float, default=40.0)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--chunk-core-frames", type=int, default=192)
    p.add_argument("--chunk-ctx-frames", type=int, default=64)
    p.add_argument("--core-words", type=int, default=12)
    p.add_argument("--ctx-words", type=int, default=2)
    p.add_argument("--stride-words", type=int, default=6)
    p.add_argument("--max-ctx-frames", type=int, default=384)
    p.add_argument("--min-audio-sec", type=float, default=0.5)
    p.add_argument("--max-tts-audio-sec", type=float, default=20.0)
    p.add_argument("--min-ctc-frames", type=int, default=20)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--min-speaker-utts", type=int, default=1)
    p.add_argument(
        "--allow-no-alignment",
        action="store_true",
        help=(
            "Keep rows that have text/audio but no word alignment. Use this for "
            "test/demo manifests only; duration and FM cut outputs remain disabled "
            "for such rows."
        ),
    )
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--skip-extract", action="store_true", help="Only rebuild manifests from existing .npy latents.")
    p.add_argument("--overwrite-latents", action="store_true", help="Re-extract latent .npy files even when they already exist.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_manifest_tsv(path):
    wav_to_raw = {}
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                bad_lines += 1
                continue
            wav, text = parts
            wav_to_raw[os.path.abspath(wav)] = text.strip()
    return wav_to_raw, bad_lines


def iter_jsonl(path, max_rows=None):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_librispeech_path(path):
    p = os.path.abspath(path).replace("\\", "/")
    parts = p.split("/")
    subset = None
    speaker = None
    chapter = None
    for i, part in enumerate(parts):
        if part.startswith(("train-", "dev-", "test-")):
            subset = part
            if i + 1 < len(parts):
                speaker = parts[i + 1]
            if i + 2 < len(parts):
                chapter = parts[i + 2]
            break
    utt_id = os.path.splitext(os.path.basename(p))[0]
    if speaker is None:
        pieces = utt_id.split("_")
        speaker = pieces[0] if pieces else None
        chapter = pieces[1] if len(pieces) > 1 else None
    return {"wav": p, "utt_id": utt_id, "subset": subset, "speaker": speaker, "chapter": chapter}


def sec_to_frame(sec, fps, total_len):
    if total_len <= 0:
        return 0
    return max(0, min(int(round(float(sec) * float(fps))), total_len))


def extract_punct_after_words(text_raw, words_norm):
    tokens = []
    for match in re.finditer(r"([A-Za-z0-9]+(?:['’`][A-Za-z0-9]+)?)([^A-Za-z0-9]*)", text_raw):
        raw_word = match.group(1)
        suffix = match.group(2) or ""
        norm_word = normalize_text_basic(raw_word)
        punct = "".join(ch for ch in suffix if ch in PUNCT_CHARS)
        tokens.append((norm_word, punct))

    punct_after = []
    j = 0
    for word in words_norm:
        target = normalize_text_basic(word)
        found = ""
        while j < len(tokens):
            tok_word, tok_punct = tokens[j]
            j += 1
            if tok_word == target:
                found = tok_punct
                break
        punct_after.append(found)
    if len(punct_after) < len(words_norm):
        punct_after += [""] * (len(words_norm) - len(punct_after))
    return punct_after[: len(words_norm)]


def latent_path_for_row(row, out_dir, latent_subdir):
    meta = parse_librispeech_path(row["wav"])
    subset = meta["subset"] or "unknown_subset"
    speaker = meta["speaker"] or "unknown_spk"
    chapter = meta["chapter"] or "unknown_chapter"
    return Path(out_dir) / latent_subdir / subset / speaker / chapter / f"{meta['utt_id']}.npy"


def load_svae_model(args):
    semantic_root = Path(args.semantic_vae_root).resolve()
    sys.path.insert(0, str(semantic_root))
    from dac.model.dac import DAC
    from dac.model.utils import read_json_file

    ckpt_dir = Path(args.semantic_vae_ckpt).resolve()
    metainfo = read_json_file(ckpt_dir / "metainfo.json")
    bigvgan_conf = Path(metainfo["DAC"]["bigvgan_conf"])
    if not bigvgan_conf.is_absolute():
        metainfo["DAC"]["bigvgan_conf"] = str(semantic_root / bigvgan_conf)
    ckpt_name = "ema_state_dict.pth" if args.use_ema else "weights.pth"
    ckpt = torch.load(ckpt_dir / "dac" / ckpt_name, map_location="cpu")
    if args.use_ema:
        ckpt = {k.replace("ema_model.", ""): v for k, v in ckpt.items()}
    else:
        ckpt = ckpt["state_dict"]
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("projectors")}
    model = DAC(**metainfo["DAC"])
    if hasattr(model, "projectors"):
        del model.projectors
    model.load_state_dict(ckpt, strict=False)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_one_latent(model, wav_path, device):
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != model.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = model.preprocess(wav, model.sample_rate).to(device)
    z_hat, _, _, _ = model.encode(wav.unsqueeze(0))
    return z_hat.squeeze(0).detach().cpu().float().numpy()  # [T, 64]


def maybe_extract_latents(rows, args, out_dir):
    if args.skip_extract:
        return
    model = load_svae_model(args)
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        out_path = latent_path_for_row(row, out_dir, args.latent_subdir)
        if out_path.exists() and not args.overwrite_latents:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        latent = extract_one_latent(model, row["wav"], args.device)
        if latent.ndim != 2 or latent.shape[-1] != int(args.latent_dim):
            raise RuntimeError(f"Unexpected latent shape for {row['wav']}: {latent.shape}")
        np.save(out_path, latent)
        if idx % 500 == 0 or idx == total:
            print(f"[SVAE] extracted {idx}/{total}: {out_path}")


def make_word_records(row, speech_len, fps):
    records = []
    for span in row.get("word_spans") or []:
        start_ctc = int(span["start_frame"])
        end_ctc = int(span["end_frame"])
        start_sec = float(span.get("start_sec", 0.0))
        end_sec = float(span.get("end_sec", start_sec))
        start_frame = sec_to_frame(start_sec, fps, speech_len)
        end_frame = sec_to_frame(end_sec, fps, speech_len)
        if end_frame <= start_frame:
            end_frame = min(speech_len, start_frame + 1)
        records.append(
            {
                "word": str(span.get("word", "")),
                "start_ctc": start_ctc,
                "end_ctc": end_ctc,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_mel": start_frame,
                "end_mel": end_frame,
                "start_svae": start_frame,
                "end_svae": end_frame,
                "dur_ctc": max(1, end_ctc - start_ctc + 1),
                "dur_mel": max(1, end_frame - start_frame),
                "dur_svae": max(1, end_frame - start_frame),
            }
        )
    return records


def add_pause_fields(full_row):
    words = full_row["words"]
    ctc_num_frames = int(full_row["ctc_num_frames"])
    speech_len = int(full_row["mel_len"])
    if not words:
        full_row["leading_sil_ctc_frames"] = 0
        full_row["trailing_sil_ctc_frames"] = 0
        full_row["leading_sil_mel_frames"] = 0
        full_row["trailing_sil_mel_frames"] = 0
        full_row["leading_sil_svae_frames"] = 0
        full_row["trailing_sil_svae_frames"] = 0
        full_row["pause_after_word_ctc_frames"] = []
        full_row["pause_after_word_mel_frames"] = []
        full_row["pause_after_word_svae_frames"] = []
        return full_row
    full_row["leading_sil_ctc_frames"] = max(0, int(words[0]["start_ctc"]))
    full_row["trailing_sil_ctc_frames"] = max(0, ctc_num_frames - 1 - int(words[-1]["end_ctc"]))
    full_row["leading_sil_mel_frames"] = max(0, int(words[0]["start_mel"]))
    full_row["trailing_sil_mel_frames"] = max(0, speech_len - int(words[-1]["end_mel"]))
    full_row["leading_sil_svae_frames"] = full_row["leading_sil_mel_frames"]
    full_row["trailing_sil_svae_frames"] = full_row["trailing_sil_mel_frames"]
    pause_ctc = []
    pause_frames = []
    for i, word in enumerate(words):
        if i + 1 < len(words):
            pause_ctc.append(max(0, int(words[i + 1]["start_ctc"]) - int(word["end_ctc"]) - 1))
            pause_frames.append(max(0, int(words[i + 1]["start_mel"]) - int(word["end_mel"])))
        else:
            pause_ctc.append(full_row["trailing_sil_ctc_frames"])
            pause_frames.append(full_row["trailing_sil_mel_frames"])
    full_row["pause_after_word_ctc_frames"] = pause_ctc
    full_row["pause_after_word_mel_frames"] = pause_frames
    full_row["pause_after_word_svae_frames"] = pause_frames
    return full_row


def make_full_row(row, wav_to_raw, args, out_dir):
    meta = parse_librispeech_path(row["wav"])
    wav_abs = meta["wav"]
    latent_path = latent_path_for_row(row, out_dir, args.latent_subdir)
    if not latent_path.exists():
        raise FileNotFoundError(f"Missing Semantic-VAE latent: {latent_path}")
    latent = np.load(latent_path, mmap_mode="r")
    speech_len = int(latent.shape[0])
    if int(latent.shape[-1]) != int(args.latent_dim):
        raise RuntimeError(f"Bad latent dim for {latent_path}: {latent.shape}")
    text_raw_aligned = str(row.get("text_raw", "")).strip()
    text_raw_manifest = wav_to_raw.get(wav_abs, "")
    text_raw = text_raw_aligned or text_raw_manifest
    text_norm_source = text_raw or str(row.get("text_norm", "")).strip()
    text_norm = normalize_text_basic(text_norm_source)
    audio_sec = float(row.get("audio_sec", speech_len / float(args.latent_fps)))
    ctc_num_frames = int(row.get("ctc_num_frames", 0))
    if ctc_num_frames <= 0 and bool(args.allow_no_alignment):
        # Demo/test manifests may only provide wav + text. Keep a reasonable
        # frame count for reporting/chunk metadata; no duration teacher uses it.
        ctc_num_frames = max(1, int(round(float(audio_sec) / 0.02)))
    words = make_word_records(row, speech_len=speech_len, fps=args.latent_fps)
    word_texts = [w["word"] for w in words]
    punct_after = extract_punct_after_words(text_raw, word_texts)
    valid_alignment = bool(words)
    if words:
        valid_alignment = (
            words[0]["start_ctc"] >= 0
            and words[-1]["end_ctc"] < max(1, ctc_num_frames)
            and words[0]["start_mel"] >= 0
            and words[-1]["end_mel"] <= max(1, speech_len)
        )
    elif bool(args.allow_no_alignment):
        valid_alignment = True
    require_alignment_for_training = not bool(args.allow_no_alignment)
    quality = {
        "valid_alignment": bool(valid_alignment),
        "has_word_alignment": bool(words),
        "too_short": bool(
            audio_sec < args.min_audio_sec
            or (require_alignment_for_training and ctc_num_frames < args.min_ctc_frames)
            or (require_alignment_for_training and len(words) < args.min_words)
            or len(text_norm) == 0
            or speech_len < 4
        ),
        "too_long_for_tts": bool(audio_sec > args.max_tts_audio_sec),
        "raw_text_manifest_mismatch": bool(
            text_raw_manifest and text_raw_aligned and text_raw_manifest.strip() != text_raw_aligned.strip()
        ),
    }
    full_row = {
        "utt_id": meta["utt_id"],
        "wav": wav_abs,
        "speaker": meta["speaker"],
        "chapter": meta["chapter"],
        "subset": meta["subset"],
        "text_raw": text_raw,
        "text_norm": text_norm,
        "unit_text": str(row.get("unit_text", "")).strip() or "|".join(word_texts),
        "audio_sec": audio_sec,
        "mel_len": speech_len,
        "svae_len": speech_len,
        "ctc_num_frames": ctc_num_frames,
        "ctc_sec_per_frame": float(row.get("ctc_sec_per_frame", 0.0)),
        "sample_rate": int(args.sample_rate),
        "hop_size": int(round(float(args.sample_rate) / float(args.latent_fps))),
        "speech_rep": "semantic_vae",
        "speech_dim": int(args.latent_dim),
        "speech_fps": float(args.latent_fps),
        "svae_latent_path": str(latent_path),
        "words": words,
        "punct_after_word": punct_after,
        "npz_path": row.get("npz_path"),
        "quality": quality,
    }
    return add_pause_fields(full_row)


def make_duration_row(row):
    words = row["words"]
    return {
        "utt_id": row["utt_id"],
        "wav": row["wav"],
        "speaker": row["speaker"],
        "chapter": row["chapter"],
        "subset": row["subset"],
        "text_raw": row["text_raw"],
        "text_norm": row["text_norm"],
        "unit_text": row["unit_text"],
        "mel_len": row["mel_len"],
        "svae_len": row["svae_len"],
        "svae_latent_path": row["svae_latent_path"],
        "ctc_num_frames": row["ctc_num_frames"],
        "word_text": [w["word"] for w in words],
        "word_dur_ctc_frames": [w["dur_ctc"] for w in words],
        "word_dur_mel_frames": [w["dur_mel"] for w in words],
        "word_dur_svae_frames": [w["dur_svae"] for w in words],
        "pause_after_word_ctc_frames": row["pause_after_word_ctc_frames"],
        "pause_after_word_mel_frames": row["pause_after_word_mel_frames"],
        "pause_after_word_svae_frames": row["pause_after_word_svae_frames"],
        "leading_sil_ctc_frames": row["leading_sil_ctc_frames"],
        "trailing_sil_ctc_frames": row["trailing_sil_ctc_frames"],
        "leading_sil_mel_frames": row["leading_sil_mel_frames"],
        "trailing_sil_mel_frames": row["trailing_sil_mel_frames"],
        "punct_after_word": row["punct_after_word"],
    }


def make_asr_chunks(row, args):
    K = int(row["mel_len"])
    core = max(1, int(args.chunk_core_frames))
    ctx = max(0, int(args.chunk_ctx_frames))
    chunks = []
    pos = 0
    while pos < K:
        core_start = pos
        core_end = min(K, pos + core)
        ctx_start = max(0, core_start - ctx)
        ctx_end = min(K, core_end + ctx)
        chunks.append(
            {
                "ctx_mel_start": ctx_start,
                "ctx_mel_end": ctx_end,
                "core_mel_start": core_start,
                "core_mel_end": core_end,
                "core_start_in_ctx": core_start - ctx_start,
                "core_end_in_ctx": core_end - ctx_start,
            }
        )
        pos = core_end
    return {
        "utt_id": row["utt_id"],
        "wav": row["wav"],
        "speaker": row["speaker"],
        "chapter": row["chapter"],
        "subset": row["subset"],
        "text_norm": row["text_norm"],
        "mel_len": row["mel_len"],
        "svae_len": row["svae_len"],
        "svae_latent_path": row["svae_latent_path"],
        "ctc_num_frames": row["ctc_num_frames"],
        "chunk_core_mel": core,
        "chunk_ctx_mel": ctx,
        "chunks": chunks,
    }


def make_fm_cuts(row, args):
    words = row["words"]
    n = len(words)
    if n == 0:
        return []
    core_words = max(1, int(args.core_words))
    ctx_words = max(0, int(args.ctx_words))
    stride_words = max(1, int(args.stride_words))
    max_ctx = max(16, int(args.max_ctx_frames))
    cuts = []
    start = 0
    cut_idx = 0
    while start < n:
        core_start = start
        core_end = min(n, core_start + core_words)
        while core_end > core_start + 1:
            ctx_start = max(0, core_start - ctx_words)
            ctx_end = min(n, core_end + ctx_words)
            ctx_s = 0 if ctx_start == 0 else int(words[ctx_start]["start_mel"])
            ctx_e = int(row["mel_len"]) if ctx_end == n else int(words[ctx_end - 1]["end_mel"])
            if ctx_e - ctx_s <= max_ctx:
                break
            core_end -= 1
        ctx_start = max(0, core_start - ctx_words)
        ctx_end = min(n, core_end + ctx_words)
        core_s = int(words[core_start]["start_mel"])
        core_e = int(words[core_end - 1]["end_mel"])
        ctx_s = 0 if ctx_start == 0 else int(words[ctx_start]["start_mel"])
        ctx_e = int(row["mel_len"]) if ctx_end == n else int(words[ctx_end - 1]["end_mel"])
        if core_e > core_s and ctx_e > ctx_s:
            words_core = [w["word"] for w in words[core_start:core_end]]
            words_ctx = [w["word"] for w in words[ctx_start:ctx_end]]
            cuts.append(
                {
                    "cut_id": f"{row['utt_id']}__svaefm__{cut_idx:04d}",
                    "utt_id": row["utt_id"],
                    "wav": row["wav"],
                    "speaker": row["speaker"],
                    "chapter": row["chapter"],
                    "subset": row["subset"],
                    "svae_latent_path": row["svae_latent_path"],
                    "text_norm_full": row["text_norm"],
                    "text_raw_full": row["text_raw"],
                    "text_norm_core": " ".join(words_core),
                    "text_norm_ctx": " ".join(words_ctx),
                    "words_core": words_core,
                    "words_ctx": words_ctx,
                    "punct_after_word_ctx": row["punct_after_word"][ctx_start:ctx_end],
                    "core_word_start": core_start,
                    "core_word_end": core_end,
                    "ctx_word_start": ctx_start,
                    "ctx_word_end": ctx_end,
                    "ctx_mel_start": ctx_s,
                    "ctx_mel_end": ctx_e,
                    "core_mel_start": core_s,
                    "core_mel_end": core_e,
                    "core_start_in_ctx": core_s - ctx_s,
                    "core_end_in_ctx": core_e - ctx_s,
                    "ctx_mel_len": ctx_e - ctx_s,
                    "core_mel_len": core_e - core_s,
                    "is_full_utt": bool(ctx_s == 0 and ctx_e == int(row["mel_len"])),
                }
            )
            cut_idx += 1
        if core_end >= n:
            break
        start += stride_words
    return cuts


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def pct(p):
        return values[min(len(values) - 1, int((len(values) - 1) * p))]
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "min": values[0],
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": values[-1],
    }


def ensure_writable_outputs(out_dir, overwrite):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "full_manifest_clean.jsonl",
        "tts_duration_full.jsonl",
        "asr_full_chunks.jsonl",
        "fm_core_context_cuts.jsonl",
        "speaker_stats.json",
        "preprocessing_report.json",
    ]
    existing = [str(out_dir / name) for name in names if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError("Output files already exist. Use --overwrite:\n" + "\n".join(existing))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_writable_outputs(out_dir, args.overwrite)
    wav_to_raw, manifest_bad_lines = read_manifest_tsv(args.manifest_tsv)
    aligned_rows = list(iter_jsonl(args.aligned_jsonl, max_rows=args.max_rows))
    maybe_extract_latents(aligned_rows, args, out_dir)

    full_rows = []
    counters = Counter(aligned_rows_seen=len(aligned_rows))
    speaker_counter = Counter()
    speaker_sec = defaultdict(float)
    speaker_chapters = defaultdict(set)
    for aligned_row in aligned_rows:
        try:
            full_row = make_full_row(aligned_row, wav_to_raw, args, out_dir)
        except Exception as exc:
            counters["failed_rows"] += 1
            if counters["failed_rows"] <= 20:
                print(f"[WARN] skip row wav={aligned_row.get('wav')} error={repr(exc)}")
            continue
        q = full_row["quality"]
        if q["raw_text_manifest_mismatch"]:
            counters["raw_text_manifest_mismatch"] += 1
        if not q["valid_alignment"]:
            counters["invalid_alignment"] += 1
            continue
        full_rows.append(full_row)
        spk = full_row["speaker"]
        speaker_counter[spk] += 1
        speaker_sec[spk] += float(full_row["audio_sec"])
        speaker_chapters[spk].add(full_row["chapter"])

    allowed_speakers = {spk for spk, count in speaker_counter.items() if count >= int(args.min_speaker_utts)}
    clean_rows = []
    for row in full_rows:
        q = row["quality"]
        speaker_ok = row["speaker"] in allowed_speakers
        has_words = bool(row["words"])
        q["speaker_min_count_ok"] = bool(speaker_ok)
        row["include_asr"] = bool(speaker_ok and not q["too_short"])
        row["include_tts_duration"] = bool(
            speaker_ok and has_words and not q["too_short"] and not q["too_long_for_tts"]
        )
        row["include_fm_cut"] = bool(speaker_ok and has_words and not q["too_short"])
        if speaker_ok:
            clean_rows.append(row)
        else:
            counters["speaker_too_few_utts"] += 1

    duration_rows = [make_duration_row(row) for row in clean_rows if row["include_tts_duration"]]
    asr_rows = [make_asr_chunks(row, args) for row in clean_rows if row["include_asr"]]
    fm_cuts = []
    for row in clean_rows:
        if row["include_fm_cut"]:
            fm_cuts.extend(make_fm_cuts(row, args))

    speaker_stats = {
        spk: {
            "num_utts": int(speaker_counter[spk]),
            "total_sec": float(speaker_sec[spk]),
            "num_chapters": len(speaker_chapters[spk]),
            "chapters": sorted(ch for ch in speaker_chapters[spk] if ch is not None),
            "allowed": spk in allowed_speakers,
        }
        for spk in sorted(speaker_counter, key=lambda x: str(x))
    }
    report = {
        "inputs": {
            "manifest_tsv": os.path.abspath(args.manifest_tsv),
            "aligned_jsonl": os.path.abspath(args.aligned_jsonl),
            "manifest_rows": len(wav_to_raw),
            "manifest_bad_lines": manifest_bad_lines,
            "semantic_vae_ckpt": os.path.abspath(args.semantic_vae_ckpt),
        },
        "outputs": {
            "full_manifest_clean": len(clean_rows),
            "tts_duration_full": len(duration_rows),
            "asr_full_chunks": len(asr_rows),
            "fm_core_context_cuts": len(fm_cuts),
            "speakers": len(speaker_stats),
            "allowed_speakers": len(allowed_speakers),
        },
        "counters": dict(counters),
        "config": vars(args),
        "stats": {
            "audio_sec": quantiles([float(r["audio_sec"]) for r in clean_rows]),
            "svae_len": quantiles([int(r["svae_len"]) for r in clean_rows]),
            "ctc_num_frames": quantiles([int(r["ctc_num_frames"]) for r in clean_rows]),
            "word_count": quantiles([len(r["words"]) for r in clean_rows]),
            "speaker_utts": quantiles([int(v["num_utts"]) for v in speaker_stats.values()]),
            "fm_ctx_svae_len": quantiles([int(c["ctx_mel_len"]) for c in fm_cuts]),
            "fm_core_svae_len": quantiles([int(c["core_mel_len"]) for c in fm_cuts]),
        },
    }
    write_jsonl(out_dir / "full_manifest_clean.jsonl", clean_rows)
    write_jsonl(out_dir / "tts_duration_full.jsonl", duration_rows)
    write_jsonl(out_dir / "asr_full_chunks.jsonl", asr_rows)
    write_jsonl(out_dir / "fm_core_context_cuts.jsonl", fm_cuts)
    write_json(out_dir / "speaker_stats.json", speaker_stats)
    write_json(out_dir / "preprocessing_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
