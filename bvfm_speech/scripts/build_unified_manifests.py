#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biflow.utils import normalize_text_basic


PUNCT_CHARS = set(",.;:?!-\"'()[]{}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build unified ASR-TTS manifests from raw TSV and aligned JSONL."
    )
    p.add_argument("--manifest-tsv", default="/work/dankker0900/dataset/manifest.tsv")
    p.add_argument("--aligned-jsonl", default="/work/dankker0900/dataset/align/train_manifest_aligned.jsonl")
    p.add_argument("--out-dir", default="/work/dankker0900/dataset/processed_unified")
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--hop-size", type=int, default=256)
    p.add_argument("--chunk-core-mel", type=int, default=256)
    p.add_argument("--chunk-ctx-mel", type=int, default=96)
    p.add_argument("--core-words", type=int, default=12)
    p.add_argument("--ctx-words", type=int, default=2)
    p.add_argument("--stride-words", type=int, default=6)
    p.add_argument("--max-ctx-mel", type=int, default=512)
    p.add_argument("--min-audio-sec", type=float, default=0.5)
    p.add_argument("--max-tts-audio-sec", type=float, default=20.0)
    p.add_argument("--min-ctc-frames", type=int, default=20)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--min-speaker-utts", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=None, help="Debug only: process first N aligned rows.")
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
        if pieces:
            speaker = pieces[0]
        if len(pieces) > 1:
            chapter = pieces[1]
    return {
        "wav": p,
        "utt_id": utt_id,
        "subset": subset,
        "speaker": speaker,
        "chapter": chapter,
    }


def sec_to_mel(sec, mel_fps, mel_len):
    if mel_len <= 0:
        return 0
    return max(0, min(int(round(float(sec) * mel_fps)), mel_len))


def make_word_records(row, mel_len, mel_fps):
    records = []
    for span in row.get("word_spans") or []:
        start_ctc = int(span["start_frame"])
        end_ctc = int(span["end_frame"])
        start_sec = float(span.get("start_sec", 0.0))
        end_sec = float(span.get("end_sec", start_sec))
        start_mel = sec_to_mel(start_sec, mel_fps, mel_len)
        end_mel = sec_to_mel(end_sec, mel_fps, mel_len)
        if end_mel <= start_mel:
            end_mel = min(mel_len, start_mel + 1)
        records.append(
            {
                "word": str(span.get("word", "")),
                "start_ctc": start_ctc,
                "end_ctc": end_ctc,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_mel": start_mel,
                "end_mel": end_mel,
                "dur_ctc": max(1, end_ctc - start_ctc + 1),
                "dur_mel": max(1, end_mel - start_mel),
            }
        )
    return records


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


def add_pause_fields(full_row):
    words = full_row["words"]
    ctc_num_frames = int(full_row["ctc_num_frames"])
    mel_len = int(full_row["mel_len"])
    if not words:
        full_row["leading_sil_ctc_frames"] = 0
        full_row["trailing_sil_ctc_frames"] = 0
        full_row["leading_sil_mel_frames"] = 0
        full_row["trailing_sil_mel_frames"] = 0
        full_row["pause_after_word_ctc_frames"] = []
        full_row["pause_after_word_mel_frames"] = []
        return full_row

    full_row["leading_sil_ctc_frames"] = max(0, int(words[0]["start_ctc"]))
    full_row["trailing_sil_ctc_frames"] = max(0, ctc_num_frames - 1 - int(words[-1]["end_ctc"]))
    full_row["leading_sil_mel_frames"] = max(0, int(words[0]["start_mel"]))
    full_row["trailing_sil_mel_frames"] = max(0, mel_len - int(words[-1]["end_mel"]))

    pause_ctc = []
    pause_mel = []
    for i, word in enumerate(words):
        if i + 1 < len(words):
            pause_ctc.append(max(0, int(words[i + 1]["start_ctc"]) - int(word["end_ctc"]) - 1))
            pause_mel.append(max(0, int(words[i + 1]["start_mel"]) - int(word["end_mel"])))
        else:
            pause_ctc.append(full_row["trailing_sil_ctc_frames"])
            pause_mel.append(full_row["trailing_sil_mel_frames"])
    full_row["pause_after_word_ctc_frames"] = pause_ctc
    full_row["pause_after_word_mel_frames"] = pause_mel
    return full_row


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


def make_full_row(row, wav_to_raw, args):
    meta = parse_librispeech_path(row["wav"])
    wav_abs = meta["wav"]
    text_raw_aligned = str(row.get("text_raw", "")).strip()
    text_raw_manifest = wav_to_raw.get(wav_abs, "")
    text_raw = text_raw_aligned or text_raw_manifest
    text_norm_source = text_raw or str(row.get("text_norm", "")).strip()
    text_norm = normalize_text_basic(text_norm_source)

    audio_sec = float(row.get("audio_sec", 0.0))
    mel_fps = float(args.sample_rate) / float(args.hop_size)
    mel_len = int(round(audio_sec * mel_fps))
    ctc_num_frames = int(row.get("ctc_num_frames", 0))
    words = make_word_records(row, mel_len=mel_len, mel_fps=mel_fps)
    word_texts = [w["word"] for w in words]
    punct_after = extract_punct_after_words(text_raw, word_texts)

    valid_alignment = bool(words)
    if words:
        valid_alignment = (
            words[0]["start_ctc"] >= 0
            and words[-1]["end_ctc"] < max(1, ctc_num_frames)
            and words[0]["start_mel"] >= 0
            and words[-1]["end_mel"] <= max(1, mel_len)
        )

    quality = {
        "valid_alignment": bool(valid_alignment),
        "too_short": bool(
            audio_sec < args.min_audio_sec
            or ctc_num_frames < args.min_ctc_frames
            or len(words) < args.min_words
            or len(text_norm) == 0
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
        "mel_len": mel_len,
        "ctc_num_frames": ctc_num_frames,
        "ctc_sec_per_frame": float(row.get("ctc_sec_per_frame", 0.0)),
        "sample_rate": int(args.sample_rate),
        "hop_size": int(args.hop_size),
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
        "ctc_num_frames": row["ctc_num_frames"],
        "word_text": [w["word"] for w in words],
        "word_dur_ctc_frames": [w["dur_ctc"] for w in words],
        "word_dur_mel_frames": [w["dur_mel"] for w in words],
        "pause_after_word_ctc_frames": row["pause_after_word_ctc_frames"],
        "pause_after_word_mel_frames": row["pause_after_word_mel_frames"],
        "leading_sil_ctc_frames": row["leading_sil_ctc_frames"],
        "trailing_sil_ctc_frames": row["trailing_sil_ctc_frames"],
        "leading_sil_mel_frames": row["leading_sil_mel_frames"],
        "trailing_sil_mel_frames": row["trailing_sil_mel_frames"],
        "punct_after_word": row["punct_after_word"],
    }


def make_asr_chunks(row, args):
    mel_len = int(row["mel_len"])
    core = max(1, int(args.chunk_core_mel))
    ctx = max(0, int(args.chunk_ctx_mel))
    chunks = []
    pos = 0
    while pos < mel_len:
        core_start = pos
        core_end = min(mel_len, pos + core)
        ctx_start = max(0, core_start - ctx)
        ctx_end = min(mel_len, core_end + ctx)
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
    max_ctx_mel = max(16, int(args.max_ctx_mel))
    cuts = []
    start = 0
    cut_idx = 0
    while start < n:
        core_start = start
        core_end = min(n, core_start + core_words)

        while core_end > core_start + 1:
            ctx_start = max(0, core_start - ctx_words)
            ctx_end = min(n, core_end + ctx_words)
            ctx_mel_start = 0 if ctx_start == 0 else int(words[ctx_start]["start_mel"])
            ctx_mel_end = int(row["mel_len"]) if ctx_end == n else int(words[ctx_end - 1]["end_mel"])
            if ctx_mel_end - ctx_mel_start <= max_ctx_mel:
                break
            core_end -= 1

        ctx_start = max(0, core_start - ctx_words)
        ctx_end = min(n, core_end + ctx_words)
        core_mel_start = int(words[core_start]["start_mel"])
        core_mel_end = int(words[core_end - 1]["end_mel"])
        ctx_mel_start = 0 if ctx_start == 0 else int(words[ctx_start]["start_mel"])
        ctx_mel_end = int(row["mel_len"]) if ctx_end == n else int(words[ctx_end - 1]["end_mel"])
        if core_mel_end > core_mel_start and ctx_mel_end > ctx_mel_start:
            words_core = [w["word"] for w in words[core_start:core_end]]
            words_ctx = [w["word"] for w in words[ctx_start:ctx_end]]
            cuts.append(
                {
                    "cut_id": f"{row['utt_id']}__fmctx__{cut_idx:04d}",
                    "utt_id": row["utt_id"],
                    "wav": row["wav"],
                    "speaker": row["speaker"],
                    "chapter": row["chapter"],
                    "subset": row["subset"],
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
                    "ctx_mel_start": ctx_mel_start,
                    "ctx_mel_end": ctx_mel_end,
                    "core_mel_start": core_mel_start,
                    "core_mel_end": core_mel_end,
                    "core_start_in_ctx": core_mel_start - ctx_mel_start,
                    "core_end_in_ctx": core_mel_end - ctx_mel_start,
                    "ctx_mel_len": ctx_mel_end - ctx_mel_start,
                    "core_mel_len": core_mel_end - core_mel_start,
                    "is_full_utt": bool(ctx_mel_start == 0 and ctx_mel_end == int(row["mel_len"])),
                }
            )
            cut_idx += 1
        if core_end >= n:
            break
        start += stride_words
    return cuts


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
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            + "\n".join(existing)
        )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_writable_outputs(out_dir, args.overwrite)

    wav_to_raw, manifest_bad_lines = read_manifest_tsv(args.manifest_tsv)
    full_rows = []
    counters = Counter()
    speaker_counter = Counter()
    speaker_sec = defaultdict(float)
    speaker_chapters = defaultdict(set)

    for aligned_row in iter_jsonl(args.aligned_jsonl, max_rows=args.max_rows):
        counters["aligned_rows_seen"] += 1
        full_row = make_full_row(aligned_row, wav_to_raw, args)
        q = full_row["quality"]
        if q["raw_text_manifest_mismatch"]:
            counters["raw_text_manifest_mismatch"] += 1
        if not q["valid_alignment"]:
            counters["invalid_alignment"] += 1
            continue
        full_rows.append(full_row)
        speaker = full_row["speaker"]
        speaker_counter[speaker] += 1
        speaker_sec[speaker] += float(full_row["audio_sec"])
        speaker_chapters[speaker].add(full_row["chapter"])

    allowed_speakers = {
        speaker for speaker, count in speaker_counter.items()
        if count >= int(args.min_speaker_utts)
    }

    clean_rows = []
    for row in full_rows:
        q = row["quality"]
        speaker_ok = row["speaker"] in allowed_speakers
        q["speaker_min_count_ok"] = bool(speaker_ok)
        row["include_asr"] = bool(speaker_ok and not q["too_short"])
        row["include_tts_duration"] = bool(speaker_ok and not q["too_short"] and not q["too_long_for_tts"])
        row["include_fm_cut"] = bool(speaker_ok and not q["too_short"])
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
        speaker: {
            "num_utts": int(speaker_counter[speaker]),
            "total_sec": float(speaker_sec[speaker]),
            "num_chapters": len(speaker_chapters[speaker]),
            "chapters": sorted(ch for ch in speaker_chapters[speaker] if ch is not None),
            "allowed": speaker in allowed_speakers,
        }
        for speaker in sorted(speaker_counter, key=lambda x: (str(x)))
    }

    report = {
        "inputs": {
            "manifest_tsv": os.path.abspath(args.manifest_tsv),
            "aligned_jsonl": os.path.abspath(args.aligned_jsonl),
            "manifest_rows": len(wav_to_raw),
            "manifest_bad_lines": manifest_bad_lines,
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
            "mel_len": quantiles([int(r["mel_len"]) for r in clean_rows]),
            "ctc_num_frames": quantiles([int(r["ctc_num_frames"]) for r in clean_rows]),
            "word_count": quantiles([len(r["words"]) for r in clean_rows]),
            "leading_sil_mel_frames": quantiles([int(r["leading_sil_mel_frames"]) for r in clean_rows]),
            "trailing_sil_mel_frames": quantiles([int(r["trailing_sil_mel_frames"]) for r in clean_rows]),
            "speaker_utts": quantiles([int(v["num_utts"]) for v in speaker_stats.values()]),
            "fm_ctx_mel_len": quantiles([int(c["ctx_mel_len"]) for c in fm_cuts]),
            "fm_core_mel_len": quantiles([int(c["core_mel_len"]) for c in fm_cuts]),
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
