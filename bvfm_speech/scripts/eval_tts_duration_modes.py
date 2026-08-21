#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from biflow.alignment import durations_to_int_and_fixsum, gaussian_mas_score, monotonic_alignment_search
from biflow.models import euler_integrate, heun_integrate
from biflow.utils import normalize_text_basic, save_wav, set_seed
from eval_tts_test_topk import TTSEvaluator, load_utmos, load_whisper, score_utmos, transcribe_whisper


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_text(row):
    for key in ("text_norm", "normalized_text", "text", "transcript", "sentence"):
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return ""


def row_wav(row):
    for key in ("wav", "wav_path", "audio_path", "source_wav", "parent_wav"):
        val = row.get(key)
        if val is not None and str(val).strip():
            return os.path.abspath(str(val))
    return ""


def row_speaker(row):
    for key in ("speaker", "spk", "speaker_id", "spk_id"):
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val)
    wav = row_wav(row)
    if wav:
        parts = Path(wav).parts
        for part in reversed(parts):
            if str(part).isdigit():
                return str(part)
    return ""


def normalize_for_wer(text):
    text = normalize_text_basic(str(text))
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = cur
    return prev[-1]


def word_error_rate(ref, hyp):
    ref_words = normalize_for_wer(ref).split()
    hyp_words = normalize_for_wer(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return float(edit_distance(ref_words, hyp_words)) / float(len(ref_words))


def mean_or_none(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.mean(vals)) if vals else None


def resolve_checkpoint(ckpt_dir, checkpoint):
    checkpoint = str(checkpoint)
    if os.path.isabs(checkpoint):
        return checkpoint
    return str(Path(ckpt_dir) / checkpoint)


def default_manifest_from_cfg(cfg):
    processed = cfg.get("paths", {}).get("processed_unified_dir")
    if not processed:
        return None
    return os.path.join(processed, "full_manifest_clean.jsonl")


class SpeakerEncoder:
    def __init__(self, model_name, savedir, device, sampling_rate, max_sec=12.0):
        self.model_name = str(model_name)
        if savedir is None:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name).strip("_")
            savedir = str(REPO_ROOT / "pretrained_models" / safe_name)
        self.savedir = savedir
        self.device = str(device)
        self.sampling_rate = int(sampling_rate)
        self.max_sec = float(max_sec)
        self.classifier = None

    def _load(self):
        if self.classifier is not None:
            return self.classifier
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            from speechbrain.pretrained import EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_name,
            savedir=self.savedir,
            run_opts={"device": self.device},
        )
        self.classifier.eval()
        return self.classifier

    @torch.no_grad()
    def encode_wav(self, wav_path):
        import librosa

        wav_np, _ = librosa.load(str(wav_path), sr=self.sampling_rate, mono=True)
        if self.max_sec > 0:
            wav_np = wav_np[: int(round(self.max_sec * self.sampling_rate))]
        wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).unsqueeze(0).to(self.device)
        emb = self._load().encode_batch(wav).detach().float().reshape(1, -1)
        return F.normalize(emb, p=2, dim=-1)

    @torch.no_grad()
    def similarity(self, ref_wav, gen_wav):
        e_ref = self.encode_wav(ref_wav)
        e_gen = self.encode_wav(gen_wav)
        return float(F.cosine_similarity(e_ref, e_gen, dim=-1).mean().detach().cpu().item())


def maybe_get_speaker_condition(evaluator, row, speaker_encoder):
    wav = row_wav(row)
    spk = row_speaker(row)
    if hasattr(evaluator.spk_table, "from_pretrained_embedding") and speaker_encoder is not None:
        raw = speaker_encoder.encode_wav(wav).to(evaluator.device)
        spk_e = evaluator.spk_table.from_pretrained_embedding(raw).to(evaluator.device)
        return spk_e, {"speaker_cond": "reference", "speaker": spk}
    if spk not in evaluator.spk2id:
        raise KeyError(f"speaker {spk!r} is not in checkpoint speaker table")
    spk_id = torch.tensor([evaluator.spk2id[spk]], device=evaluator.device, dtype=torch.long)
    return evaluator.spk_table(spk_id), {"speaker_cond": "table", "speaker": spk}


def select_rows(rows, evaluator, max_items, speaker=None):
    selected = []
    for row in rows:
        text = row_text(row)
        wav = row_wav(row)
        spk = row_speaker(row)
        if not text or not wav or not os.path.exists(wav):
            continue
        if speaker is not None and str(spk) != str(speaker):
            continue
        if not hasattr(evaluator.spk_table, "from_pretrained_embedding") and spk not in evaluator.spk2id:
            continue
        selected.append(row)
        if max_items and len(selected) >= int(max_items):
            break
    return selected


@torch.no_grad()
def oracle_mas_durations(evaluator, text, wav_path):
    h_enc, maskL, mu_tok, logvar_tok, align_mu_tok, align_logvar_tok = evaluator.encode_text_batch([text])
    z_s = evaluator.load_svae_latent(wav_path).to(evaluator.device)
    z_s = (z_s.unsqueeze(0) - evaluator.mu_b) / evaluator.std_b
    k_mas = int(z_s.shape[1])
    l_valid = int(maskL.long().sum().item())
    if k_mas < l_valid:
        raise RuntimeError(f"speech frames K={k_mas} shorter than text tokens L={l_valid}")

    mas_mu_tok = align_mu_tok if align_mu_tok is not None else mu_tok
    mas_logvar_tok = align_logvar_tok if align_logvar_tok is not None else logvar_tok
    if int(mas_mu_tok.shape[-1]) != int(z_s.shape[-1]):
        raise RuntimeError(
            f"MAS prior dim {mas_mu_tok.shape[-1]} does not match speech dim {z_s.shape[-1]}; "
            "this script expects align_mu_tok to be in speech-latent space."
        )

    maskK = torch.ones(1, k_mas, device=evaluator.device, dtype=torch.bool)
    score = gaussian_mas_score(z_s, mas_mu_tok, mas_logvar_tok, maskK, maskL)
    attn = monotonic_alignment_search(score, maskK, maskL, neg_inf=-1e4).to(dtype=mu_tok.dtype)
    dur_mas = attn[0].sum(dim=0).long()
    zc_mean = torch.bmm(attn, mu_tok)
    zc_logvar = torch.bmm(attn, logvar_tok)
    return {
        "h_enc": h_enc,
        "maskL": maskL,
        "mu_tok": mu_tok,
        "logvar_tok": logvar_tok,
        "dur_mas": dur_mas,
        "zc_mean_mas": zc_mean,
        "zc_logvar_mas": zc_logvar,
        "k_mas": k_mas,
        "l_valid": l_valid,
    }


@torch.no_grad()
def predicted_durations(evaluator, h_enc, maskL, k_mas):
    l_valid = int(maskL.long().sum().item())
    limit = int(getattr(evaluator.vf.rope, "max_seq_len", 4096))
    log_dur = evaluator.dur_pred(h_enc, maskL)
    dur_float = (torch.exp(log_dur) - 1.0) * maskL.float()
    if evaluator.len_pred is not None:
        k_raw = int(max(16, round(float(evaluator.len_pred(h_enc, maskL).item()))))
        k_pred = min(max(k_raw, l_valid), limit)
    else:
        k_pred = min(max(int(round(float(dur_float.sum().item()))), l_valid, 16), limit)
    dur_pred_gen, _ = durations_to_int_and_fixsum(dur_float, maskL, k_pred)
    dur_pred_mas_len, _ = durations_to_int_and_fixsum(dur_float, maskL, k_mas)
    return {
        "dur_float": dur_float,
        "dur_pred_gen": dur_pred_gen.long(),
        "dur_pred_mas_len": dur_pred_mas_len.long(),
        "k_pred": int(k_pred),
    }


def expand_by_duration(mu_tok, logvar_tok, dur_int, l_valid):
    mu_feats = []
    lv_feats = []
    for i in range(l_valid):
        d = int(dur_int[i].item())
        if d <= 0:
            continue
        mu_feats.append(mu_tok[0, i : i + 1].repeat(d, 1))
        lv_feats.append(logvar_tok[0, i : i + 1].repeat(d, 1))
    if not mu_feats:
        raise RuntimeError("duration expansion produced no frames")
    return torch.cat(mu_feats, dim=0).unsqueeze(0), torch.cat(lv_feats, dim=0).unsqueeze(0)


@torch.no_grad()
def synthesize_from_source(
    evaluator,
    *,
    h_enc,
    maskL,
    zc_mean,
    zc_logvar,
    spk_e,
    out_wav,
    solver,
    ode_steps,
    cfg_scale,
    prior_temp,
    style_temp,
):
    if prior_temp > 0:
        zT0 = zc_mean + float(prior_temp) * torch.exp(0.5 * zc_logvar) * torch.randn_like(zc_mean)
    else:
        zT0 = zc_mean

    maskK = torch.ones(1, zT0.shape[1], device=evaluator.device, dtype=torch.bool)
    style_e = None
    if evaluator.use_tts_style_latent:
        if evaluator.tts_style_prior is not None:
            if getattr(evaluator, "tts_style_prior_type", "") == "canonical_speaker":
                u_mu, u_logvar = evaluator.tts_style_prior(
                    zT0,
                    maskK,
                    spk_e.to(device=zT0.device, dtype=zT0.dtype),
                )
            else:
                u_mu, u_logvar = evaluator.tts_style_prior(spk_e.to(dtype=zT0.dtype))
        else:
            u_mu = torch.zeros(1, evaluator.tts_style_dim, device=evaluator.device, dtype=zT0.dtype)
            u_logvar = torch.zeros_like(u_mu)
        if style_temp > 0.0:
            style_e = u_mu + float(style_temp) * torch.exp(0.5 * u_logvar) * torch.randn_like(u_mu)
        else:
            style_e = u_mu

    text_cond = None
    if evaluator.canonical_to_source is not None:
        zT0_source = evaluator.canonical_to_source(
            zT0,
            maskK,
            spk_e=spk_e.to(dtype=zT0.dtype),
            style_e=style_e if evaluator.tts_style_into_source else None,
        )
        zT_mean_source = evaluator.canonical_to_source(
            zc_mean,
            maskK,
            spk_e=spk_e.to(dtype=zc_mean.dtype),
            style_e=(
                style_e.to(dtype=zc_mean.dtype)
                if (evaluator.tts_style_into_source and style_e is not None)
                else None
            ),
        )
        if evaluator.use_vf_canonical_text_cond:
            text_cond = zc_mean
    else:
        zT0_source = zT0
        zT_mean_source = zc_mean
        if evaluator.tts_style_to_source is not None and style_e is not None:
            style_bias = evaluator.tts_style_to_source(style_e, spk_e.to(dtype=zT0.dtype)).to(dtype=zT0.dtype)
            zT0_source = zT0_source + evaluator.tts_style_source_scale * style_bias.unsqueeze(1)
            zT_mean_source = zT_mean_source + evaluator.tts_style_source_scale * style_bias.unsqueeze(1)
        if evaluator.use_vf_canonical_text_cond:
            text_cond = zc_mean

    spk_e_vf = spk_e.to(dtype=zT0_source.dtype)
    if evaluator.tts_source_cond is not None and prior_temp > 0 and evaluator.canonical_to_source is None:
        source_delta = zT0_source - zT_mean_source
        spk_e_vf = spk_e_vf + evaluator.tts_source_cond_scale * evaluator.tts_source_cond(
            source_delta,
            maskK,
        ).to(dtype=spk_e_vf.dtype)

    t0 = time.perf_counter()
    solver = str(solver).lower()
    if solver == "heun":
        zS_pred = heun_integrate(
            evaluator.vf,
            zT0_source,
            maskK,
            steps=int(ode_steps),
            direction=+1,
            cfg_scale=float(cfg_scale),
            spk_e=spk_e_vf,
            style_e=style_e,
            text_cond=text_cond,
        )
    elif solver == "euler":
        zS_pred = euler_integrate(
            evaluator.vf,
            zT0_source,
            maskK,
            steps=int(ode_steps),
            direction=+1,
            cfg_flag_value=1,
            spk_e=spk_e_vf,
            style_e=style_e,
            text_cond=text_cond,
        )
    else:
        raise ValueError(f"unsupported solver={solver!r}")

    zS_ref = evaluator.mel_refiner(zS_pred, cond=h_enc, cond_mask=maskL) if evaluator.mel_refiner is not None else zS_pred
    mel = (zS_ref * evaluator.std_b + evaluator.mu_b).float()
    wav = evaluator.svae_model.decode(mel).squeeze(0).float().detach().cpu()
    infer_sec = time.perf_counter() - t0

    os.makedirs(os.path.dirname(str(out_wav)), exist_ok=True)
    save_wav(str(out_wav), wav, sr=evaluator.sampling_rate)
    audio_sec = float(wav.numel()) / float(evaluator.sampling_rate)
    return {
        "wav_path": str(out_wav),
        "frames": int(mel.shape[1]),
        "audio_seconds": audio_sec,
        "infer_seconds": float(infer_sec),
        "rtf": float(infer_sec / max(audio_sec, 1e-8)),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare TTS synthesis using oracle MAS durations or predicted durations.")
    p.add_argument("--duration-mode", choices=["mas", "pred"], default=None)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="latest.pt")
    p.add_argument("--config", type=str, default=None, help="Fallback config if ckpt-dir/merged_config.json is absent.")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-items", type=int, default=50)
    p.add_argument("--speaker", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--solver", choices=["heun", "euler"], default="heun")
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--prior-temp", type=float, default=0.0)
    p.add_argument("--style-temp", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--skip-whisper", action="store_true")
    p.add_argument("--whisper-model", type=str, default="medium.en")
    p.add_argument("--skip-utmos", action="store_true")
    p.add_argument("--utmos-repo", type=str, default="tarepan/SpeechMOS:v1.2.0")
    p.add_argument("--utmos-model", type=str, default="utmos22_strong")
    p.add_argument("--skip-spk-sim", action="store_true")
    p.add_argument("--spk-sim-model", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument("--spk-sim-savedir", type=str, default=None)
    p.add_argument("--spk-sim-max-sec", type=float, default=12.0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.duration_mode is None:
        raise ValueError("Missing --duration-mode; use test_tts_duration_mas.py or test_tts_duration_pred.py.")
    set_seed(int(args.seed))

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = resolve_checkpoint(ckpt_dir, args.checkpoint)
    cfg_path = ckpt_dir / "merged_config.json"
    if not cfg_path.exists():
        cfg_path = Path(args.config) if args.config else REPO_ROOT / "configs" / "cutmanifest_svae_latent.json"
    cfg = load_json(cfg_path)

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / f"duration_eval_{args.duration_mode}"
    wav_dir = out_dir / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    if results_path.exists():
        results_path.unlink()

    evaluator = TTSEvaluator(cfg, ckpt_path, out_dir, args.device)
    manifest = args.manifest or default_manifest_from_cfg(cfg)
    if not manifest or not os.path.exists(manifest):
        raise FileNotFoundError(f"manifest not found: {manifest}")
    rows = select_rows(read_jsonl(manifest), evaluator, args.max_items, speaker=args.speaker)
    if not rows:
        raise RuntimeError("no usable rows selected")

    whisper_state = None
    if not args.skip_whisper:
        try:
            whisper_state = load_whisper(args, args.device)
        except Exception as exc:
            print(f"[WARN] disabling Whisper metric: {exc}")
    utmos_model = None
    if not args.skip_utmos:
        try:
            utmos_model = load_utmos(args, args.device)
        except Exception as exc:
            print(f"[WARN] disabling UTMOS metric: {exc}")
    speaker_encoder = None
    if (not args.skip_spk_sim) or hasattr(evaluator.spk_table, "from_pretrained_embedding"):
        try:
            speaker_encoder = SpeakerEncoder(
                args.spk_sim_model,
                args.spk_sim_savedir,
                args.device,
                evaluator.sampling_rate,
                max_sec=args.spk_sim_max_sec,
            )
        except Exception as exc:
            print(f"[WARN] speaker encoder construction failed: {exc}")

    records = []
    for idx, row in enumerate(rows):
        text_raw = row_text(row)
        text = evaluator.canonicalize_text(text_raw)
        wav = row_wav(row)
        spk = row_speaker(row)
        utt_id = str(row.get("utt_id") or row.get("id") or f"item{idx:05d}")
        try:
            mas = oracle_mas_durations(evaluator, text, wav)
            pred = predicted_durations(evaluator, mas["h_enc"], mas["maskL"], mas["k_mas"])
            l_valid = int(mas["l_valid"])
            dur_mas = mas["dur_mas"][:l_valid].float()
            dur_pred_cmp = pred["dur_pred_mas_len"][:l_valid].float().to(dur_mas.device)
            dur_diff = dur_pred_cmp - dur_mas
            dur_l1 = float(dur_diff.abs().mean().detach().cpu().item())
            dur_rmse = float(torch.sqrt((dur_diff ** 2).mean()).detach().cpu().item())

            if args.duration_mode == "mas":
                zc_mean = mas["zc_mean_mas"]
                zc_logvar = mas["zc_logvar_mas"]
                used_frames = int(mas["k_mas"])
                used_dur = mas["dur_mas"][:l_valid].detach().cpu().tolist()
            else:
                zc_mean, zc_logvar = expand_by_duration(
                    mas["mu_tok"],
                    mas["logvar_tok"],
                    pred["dur_pred_gen"],
                    l_valid,
                )
                used_frames = int(zc_mean.shape[1])
                used_dur = pred["dur_pred_gen"][:l_valid].detach().cpu().tolist()

            spk_e, spk_info = maybe_get_speaker_condition(evaluator, row, speaker_encoder)
            out_wav = wav_dir / f"{idx:05d}_{utt_id}_{args.duration_mode}.wav"
            syn = synthesize_from_source(
                evaluator,
                h_enc=mas["h_enc"],
                maskL=mas["maskL"],
                zc_mean=zc_mean,
                zc_logvar=zc_logvar,
                spk_e=spk_e,
                out_wav=out_wav,
                solver=args.solver,
                ode_steps=args.ode_steps,
                cfg_scale=args.cfg_scale,
                prior_temp=args.prior_temp,
                style_temp=args.style_temp,
            )

            hyp = None
            wer = None
            if whisper_state is not None:
                hyp = transcribe_whisper(whisper_state, syn["wav_path"], args.device)
                wer = word_error_rate(text, hyp)

            utmos = score_utmos(utmos_model, syn["wav_path"], evaluator.sampling_rate, args.device) if utmos_model is not None else None
            spk_sim = None
            if (not args.skip_spk_sim) and speaker_encoder is not None:
                try:
                    spk_sim = speaker_encoder.similarity(wav, syn["wav_path"])
                except Exception as exc:
                    print(f"[WARN] speaker similarity failed for {utt_id}: {exc}")

            rec = {
                "idx": idx,
                "utt_id": utt_id,
                "speaker": spk,
                "ref_wav": wav,
                "text": text,
                "duration_mode": args.duration_mode,
                "dur_l1_vs_mas": dur_l1,
                "dur_rmse_vs_mas": dur_rmse,
                "mas_frames": int(mas["k_mas"]),
                "pred_frames": int(pred["k_pred"]),
                "used_frames": used_frames,
                "total_frame_abs_error": abs(int(pred["k_pred"]) - int(mas["k_mas"])),
                "mas_duration": mas["dur_mas"][:l_valid].detach().cpu().tolist(),
                "used_duration": used_dur,
                "whisper_hyp": hyp,
                "whisper_wer": wer,
                "utmos": utmos,
                "speaker_similarity": spk_sim,
                **spk_info,
                **syn,
            }
        except Exception as exc:
            rec = {
                "idx": idx,
                "utt_id": utt_id,
                "speaker": spk,
                "ref_wav": wav,
                "text": text,
                "duration_mode": args.duration_mode,
                "error": repr(exc),
            }
            print(f"[ERROR] {utt_id}: {exc}")
        records.append(rec)
        append_jsonl(results_path, rec)
        print(
            f"[{idx + 1}/{len(rows)}] {utt_id} mode={args.duration_mode} "
            f"dur_l1={rec.get('dur_l1_vs_mas')} wer={rec.get('whisper_wer')} "
            f"utmos={rec.get('utmos')} spk_sim={rec.get('speaker_similarity')}"
        )

    ok = [r for r in records if "error" not in r]
    summary = {
        "ckpt_dir": str(ckpt_dir.resolve()),
        "checkpoint": str(args.checkpoint),
        "ckpt_path": str(Path(ckpt_path).resolve()),
        "config": str(Path(cfg_path).resolve()),
        "manifest": str(Path(manifest).resolve()),
        "duration_mode": args.duration_mode,
        "solver": args.solver,
        "ode_steps": int(args.ode_steps),
        "cfg_scale": float(args.cfg_scale),
        "prior_temp": float(args.prior_temp),
        "style_temp": float(args.style_temp),
        "num_items": len(records),
        "num_success": len(ok),
        "duration_l1_mean": mean_or_none([r.get("dur_l1_vs_mas") for r in ok]),
        "duration_rmse_mean": mean_or_none([r.get("dur_rmse_vs_mas") for r in ok]),
        "total_frame_abs_error_mean": mean_or_none([r.get("total_frame_abs_error") for r in ok]),
        "tts_whisper_wer_mean": mean_or_none([r.get("whisper_wer") for r in ok]),
        "utmos_mean": mean_or_none([r.get("utmos") for r in ok]),
        "speaker_similarity_mean": mean_or_none([r.get("speaker_similarity") for r in ok]),
        "results_jsonl": str(results_path),
    }
    write_json(summary_path, summary)
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
