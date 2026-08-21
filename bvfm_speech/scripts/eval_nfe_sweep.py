#!/usr/bin/env python3
"""NFE sweep for TTS/ASR inference quality and speed.

Evaluates one checkpoint at several solver budgets and reports:
  - TTS WER from Whisper on generated speech
  - UTMOS on generated speech
  - ASR WER on real test speech through backward flow
  - TTS and ASR real-time factors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from biflow.alignment import downsample_time_bkd, durations_to_int_and_fixsum
from biflow.models import euler_integrate, heun_integrate
from biflow.tokenizer import ctc_greedy_decode
from biflow.utils import normalize_text_basic, save_wav

from eval_tts_test_topk import (  # noqa: E402
    TTSEvaluator,
    build_eval_pairs,
    load_json,
    load_utmos,
    load_whisper,
    mean_or_none,
    normalize_english_text_for_wer,
    pick_test_rows,
    row_speaker,
    row_text,
    safe_name,
    score_utmos,
    set_seed,
    transcribe_whisper,
    word_error_rate_notebook,
    write_json,
)


def parse_nfe_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).replace(",", " ").split():
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError(f"NFE must be positive, got {n}")
        out.append(n)
    if not out:
        raise ValueError("--nfe-list is empty")
    return out


def str2bool(value: str | bool | None):
    if value is None or isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def mean_finite(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def sync_if_cuda(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def euler_integrate_cfg(vf, z0, maskK, steps=10, direction=+1, cfg_scale=1.0, spk_e=None, style_e=None, text_cond=None):
    if float(cfg_scale) == 1.0:
        return euler_integrate(
            vf,
            z0,
            maskK,
            steps=int(steps),
            direction=direction,
            cfg_flag_value=1,
            spk_e=spk_e,
            style_e=style_e,
            text_cond=text_cond,
        )

    z = z0
    dt = direction * (1.0 / int(steps))
    t0 = 0.0 if direction == +1 else 1.0
    B = z.shape[0]
    device = z.device

    if spk_e is None:
        spk_e = torch.zeros(B, vf.E_spk, device=device, dtype=z.dtype)
    if getattr(vf, "style_dim", 0) > 0 and style_e is None:
        style_e = torch.zeros(B, vf.style_dim, device=device, dtype=z.dtype)

    cfg_cond = torch.ones(B, dtype=torch.long, device=device)
    cfg_un = torch.zeros(B, dtype=torch.long, device=device)
    spk_zero = torch.zeros_like(spk_e)
    style_zero = torch.zeros_like(style_e) if style_e is not None else None
    text_zero = torch.zeros_like(text_cond) if text_cond is not None else None

    def v_eval(z_now, t_now, cfg_flag, spk, style, text):
        t_tensor = torch.full((B,), float(t_now), device=device)
        return vf(z_now, t_tensor, maskK, cfg_flag=cfg_flag, spk_e=spk, style_e=style, text_cond=text)

    for i in range(int(steps)):
        t = t0 + i * dt
        v_c = v_eval(z, t, cfg_cond, spk_e, style_e, text_cond)
        v_u = v_eval(z, t, cfg_un, spk_zero, style_zero, text_zero)
        z = z + dt * (v_u + float(cfg_scale) * (v_c - v_u))
    return z


def asr_seconds_from_row(row: dict, frames: int | None = None, sampling_rate: int = 16000, hop_size: int = 400) -> float:
    for key in ("audio_sec", "duration", "duration_sec", "seconds"):
        if key in row and row[key] is not None:
            try:
                sec = float(row[key])
                if sec > 0:
                    return sec
            except Exception:
                pass
    if frames is not None and frames > 0:
        return float(frames) * float(hop_size) / float(sampling_rate)
    return 0.0


class ASRRuntime:
    """ASR inference path matching train.py ASR-FULL demo, including z_u."""

    def __init__(self, evaluator: TTSEvaluator, cfg: dict, device: str, *, asr_style_use_mean_override=None):
        self.evaluator = evaluator
        self.cfg = cfg
        self.device = device
        mc = cfg["model"]

        self.asr_use_spk_cond = bool(mc.get("asr_use_spk_cond", False))
        self.asr_spk_scale = float(mc.get("asr_spk_scale", 1.0))
        self.asr_spk_unknown = str(mc.get("asr_spk_unknown", "zero")).lower()
        self.vf_use_speaker_cond = bool(mc.get("vf_use_speaker_cond", True))
        asr_vf_use_speaker_cond_cfg = mc.get("asr_vf_use_speaker_cond", None)
        self.asr_vf_use_speaker_cond = (
            self.vf_use_speaker_cond
            if asr_vf_use_speaker_cond_cfg is None
            else bool(asr_vf_use_speaker_cond_cfg)
        )

        self.asr_use_style_cond = bool(mc.get("asr_use_style_cond", False))
        if asr_style_use_mean_override is None:
            self.asr_style_use_mean = bool(mc.get("asr_style_use_mean", True))
        else:
            self.asr_style_use_mean = bool(asr_style_use_mean_override)
        self.asr_style_temp = float(mc.get("asr_style_temp", 0.0))
        self.tts_style_post_mode = str(mc.get("tts_style_post_mode", "speech")).lower()

        factor = int(mc.get("ctc_subsample_factor", 1))
        apply_to = str(mc.get("ctc_subsample_apply_to", "hat")).lower()
        self.ctc_subsample_factor = max(1, factor if apply_to in {"hat", "both"} else 1)
        self.sampling_rate = int(evaluator.sampling_rate)
        self.hop_size = int(cfg.get("cache", {}).get("svae_hop_size", cfg.get("data", {}).get("hop_size", 400)))

    def _spk_cond_from_name(self, spk_name: str | None, dtype=None):
        if not self.asr_use_spk_cond:
            return None
        spk_name = str(spk_name)
        if spk_name not in self.evaluator.spk2id:
            if self.asr_spk_unknown == "error":
                raise RuntimeError(f"ASR speaker conditioning requested but speaker {spk_name!r} is unknown")
            return None
        spk_id = torch.tensor([self.evaluator.spk2id[spk_name]], device=self.device, dtype=torch.long)
        spk_e = self.evaluator.spk_table(spk_id)
        if dtype is not None:
            spk_e = spk_e.to(dtype=dtype)
        return spk_e * self.asr_spk_scale

    def _vf_spk_cond(self, spk_e):
        if spk_e is None:
            return None
        return spk_e if self.asr_vf_use_speaker_cond else torch.zeros_like(spk_e)

    def _cfg_flag_value(self, spk_e):
        if self.asr_use_style_cond:
            return 1
        return 1 if (self.asr_vf_use_speaker_cond and spk_e is not None) else 0

    def _style_from_source(self, z_s, mask, spk_e=None, dtype=None):
        post = getattr(self.evaluator, "tts_style_post", None)
        if (not self.asr_use_style_cond) or post is None:
            return None
        z_in = z_s.to(dtype=dtype) if dtype is not None else z_s
        if self.tts_style_post_mode == "path":
            if spk_e is None:
                spk_e = torch.zeros(z_s.shape[0], int(self.cfg["model"]["E_spk"]), device=z_s.device, dtype=z_in.dtype)
            t_asr = torch.ones(z_s.shape[0], device=z_s.device, dtype=z_in.dtype)
            mu, logvar = post(
                z_in,
                mask,
                z_t=z_in,
                t=t_asr,
                spk_e=spk_e.to(device=z_s.device, dtype=z_in.dtype),
            )
        else:
            mu, logvar = post(z_in, mask)
        if self.asr_style_use_mean or self.asr_style_temp <= 0:
            style = mu
        else:
            style = mu + self.asr_style_temp * torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return style.to(dtype=dtype) if dtype is not None else style

    def _prepare_ctc_input(self, z, mask):
        if self.ctc_subsample_factor > 1:
            z, mask, k_list = downsample_time_bkd(z, mask, self.ctc_subsample_factor)
            return z, mask, [int(k) for k in k_list]
        return z, mask, [int(k) for k in mask.long().sum(dim=1).tolist()]

    def _load_svae_latent_for_row(self, row: dict):
        latent_path = row.get("svae_latent_path")
        if latent_path and os.path.exists(str(latent_path)):
            arr = np.load(str(latent_path))
            z_s_log = torch.from_numpy(arr).float().contiguous()
            if z_s_log.ndim == 3 and z_s_log.shape[0] == 1:
                z_s_log = z_s_log[0].transpose(0, 1).contiguous()
            if z_s_log.ndim != 2:
                raise RuntimeError(f"Unsupported SVAE latent shape {tuple(z_s_log.shape)} from {latent_path}")
            return z_s_log
        return self.evaluator.load_svae_latent(str(row["wav"]))

    @torch.inference_mode()
    def decode_row(self, row: dict, *, nfe: int, solver: str = "euler"):
        wav_path = str(row["wav"])
        z_s_log = self._load_svae_latent_for_row(row)
        total_frames = int(z_s_log.shape[0])
        z_s = z_s_log.to(self.device, dtype=torch.float32).unsqueeze(0)
        z_s = (z_s - self.evaluator.mu_b) / self.evaluator.std_b
        mask = torch.ones(1, total_frames, device=self.device, dtype=torch.bool)

        spk_e_raw = self._spk_cond_from_name(row_speaker(row), dtype=z_s.dtype)
        spk_e = self._vf_spk_cond(spk_e_raw)
        style_e = self._style_from_source(z_s, mask, spk_e_raw, dtype=z_s.dtype)
        cfg_flag = self._cfg_flag_value(spk_e_raw)

        if solver == "euler":
            z_t = euler_integrate(
                self.evaluator.vf,
                z_s,
                mask,
                steps=int(nfe),
                direction=-1,
                cfg_flag_value=cfg_flag,
                spk_e=spk_e,
                style_e=style_e,
                text_cond=None,
            )
        elif solver == "heun":
            z_t = heun_integrate(
                self.evaluator.vf,
                z_s,
                mask,
                steps=max(1, int(math.ceil(float(nfe) / 2.0))),
                direction=-1,
                cfg_scale=1.0,
                spk_e=spk_e,
                style_e=style_e,
                text_cond=None,
            )
        else:
            raise ValueError(f"Unsupported ASR solver: {solver}")

        if self.evaluator.source_to_canonical is not None:
            z_c = self.evaluator.source_to_canonical(z_t, mask)
        else:
            z_c = z_t
        if self.evaluator.canonical_posterior is not None:
            z_c, _ = self.evaluator.canonical_posterior(z_c, mask)

        z_ctc, mask_ctc, k_list = self._prepare_ctc_input(z_c, mask)
        logits = self.evaluator.text_ctc_head(z_ctc, mask_ctc)
        decoded = ctc_greedy_decode(logits, [int(k_list[0])], self.evaluator.BLANK_ID)[0]
        hyp = self.evaluator.tok.decode(decoded)
        seconds = asr_seconds_from_row(row, total_frames, self.sampling_rate, self.hop_size)
        return {"hyp": hyp, "frames": total_frames, "seconds": seconds}


@torch.no_grad()
def synthesize_tts_with_solver(
    evaluator: TTSEvaluator,
    text,
    spk,
    wav_path,
    *,
    solver: str,
    cfg_scale: float,
    prior_temp: float,
    style_temp: float,
    ode_steps: int,
):
    if solver == "heun":
        return evaluator.synthesize(
            text,
            spk,
            wav_path,
            cfg_scale=cfg_scale,
            prior_temp=prior_temp,
            style_temp=style_temp,
            ode_steps=int(ode_steps),
        )
    if solver != "euler":
        raise ValueError(f"Unsupported TTS solver: {solver}")

    spk_id = torch.tensor([evaluator.spk2id[spk]], device=evaluator.device, dtype=torch.long)
    spk_e = evaluator.spk_table(spk_id)

    h_enc, maskL, mu_tok, logvar_tok, _, _ = evaluator.encode_text_batch([text])
    L_valid = int(maskL.sum().item())
    limit = int(getattr(evaluator.vf.rope, "max_seq_len", 4096))
    if evaluator.len_pred is not None:
        k_raw = int(max(16, round(float(evaluator.len_pred(h_enc, maskL).item()))))
        k_pred = min(max(k_raw, L_valid), limit)
    else:
        log_dur = evaluator.dur_pred(h_enc, maskL)
        dur = (torch.exp(log_dur) - 1.0) * maskL.float()
        k_pred = min(max(int(round(float(dur.sum().item()))), L_valid, 16), limit)
    log_dur = evaluator.dur_pred(h_enc, maskL)
    dur = (torch.exp(log_dur) - 1.0) * maskL.float()
    dur_int, _ = durations_to_int_and_fixsum(dur, maskL, k_pred)

    mu_feats = []
    lv_feats = []
    for i in range(L_valid):
        d = int(dur_int[i].item())
        mu_feats.append(mu_tok[0, i:i + 1].repeat(d, 1))
        lv_feats.append(logvar_tok[0, i:i + 1].repeat(d, 1))
    if not mu_feats:
        raise RuntimeError("No valid text tokens for TTS synthesis.")
    zT_mean = torch.cat(mu_feats, dim=0).unsqueeze(0)[:, :k_pred]
    zT_logvar = torch.cat(lv_feats, dim=0).unsqueeze(0)[:, :k_pred]
    if prior_temp > 0:
        zT0 = zT_mean + float(prior_temp) * torch.exp(0.5 * zT_logvar) * torch.randn_like(zT_mean)
    else:
        zT0 = zT_mean

    maskK = torch.ones(1, zT0.shape[1], device=evaluator.device, dtype=torch.bool)
    style_e_demo = None
    if evaluator.use_tts_style_latent:
        if evaluator.tts_style_prior is not None:
            if getattr(evaluator, "tts_style_prior_type", "") == "canonical_speaker":
                u_mu_p, u_logvar_p = evaluator.tts_style_prior(
                    zT0,
                    maskK,
                    spk_e.to(device=zT0.device, dtype=zT0.dtype),
                )
            else:
                u_mu_p, u_logvar_p = evaluator.tts_style_prior(spk_e.to(dtype=zT0.dtype))
        else:
            u_mu_p = torch.zeros(1, evaluator.tts_style_dim, device=evaluator.device, dtype=zT0.dtype)
            u_logvar_p = torch.zeros_like(u_mu_p)
        if style_temp > 0.0:
            style_e_demo = u_mu_p + float(style_temp) * torch.exp(0.5 * u_logvar_p) * torch.randn_like(u_mu_p)
        else:
            style_e_demo = u_mu_p

    text_cond_demo = None
    if evaluator.canonical_to_source is not None:
        zT0_source = evaluator.canonical_to_source(
            zT0,
            maskK,
            spk_e=spk_e.to(dtype=zT0.dtype),
            style_e=style_e_demo if evaluator.tts_style_into_source else None,
        )
        zT_mean_source = evaluator.canonical_to_source(
            zT_mean,
            maskK,
            spk_e=spk_e.to(dtype=zT_mean.dtype),
            style_e=(
                style_e_demo.to(dtype=zT_mean.dtype)
                if (evaluator.tts_style_into_source and style_e_demo is not None)
                else None
            ),
        )
        if evaluator.use_vf_canonical_text_cond:
            text_cond_demo = zT_mean
    else:
        zT0_source = zT0
        zT_mean_source = zT_mean
        if evaluator.tts_style_to_source is not None and style_e_demo is not None:
            style_bias = evaluator.tts_style_to_source(style_e_demo, spk_e.to(dtype=zT0.dtype)).to(dtype=zT0.dtype)
            zT0_source = zT0_source + evaluator.tts_style_source_scale * style_bias.unsqueeze(1)
            zT_mean_source = zT_mean_source + evaluator.tts_style_source_scale * style_bias.unsqueeze(1)
        if evaluator.use_vf_canonical_text_cond:
            text_cond_demo = zT_mean

    spk_e_demo = spk_e.to(dtype=zT0_source.dtype)
    if evaluator.tts_source_cond is not None and prior_temp > 0 and evaluator.canonical_to_source is None:
        source_delta_demo = zT0_source - zT_mean_source
        spk_e_demo = spk_e_demo + evaluator.tts_source_cond_scale * evaluator.tts_source_cond(
            source_delta_demo,
            maskK,
        ).to(dtype=spk_e_demo.dtype)

    zS_pred = euler_integrate_cfg(
        evaluator.vf,
        zT0_source,
        maskK,
        steps=int(ode_steps),
        direction=+1,
        cfg_scale=float(cfg_scale),
        spk_e=spk_e_demo,
        style_e=style_e_demo,
        text_cond=text_cond_demo,
    )
    zS_ref = evaluator.mel_refiner(zS_pred, cond=h_enc, cond_mask=maskL) if evaluator.mel_refiner is not None else zS_pred
    mel = (zS_ref * evaluator.std_b + evaluator.mu_b).float()
    wav = evaluator.svae_model.decode(mel).squeeze(0).float().detach().cpu()
    wav_path = str(wav_path)
    save_wav(wav_path, wav, sr=evaluator.sampling_rate)
    return {
        "wav_path": wav_path,
        "frames": int(mel.shape[1]),
        "seconds": float(wav.numel()) / float(evaluator.sampling_rate),
        "text_len": int(L_valid),
    }


def run_tts_sweep(
    *,
    args,
    cfg,
    evaluator: TTSEvaluator,
    device: str,
    nfe: int,
    tts_steps: int,
    tts_solver: str,
    pairs,
    out_dir: Path,
    utmos_model,
    whisper_model,
):
    wav_dir = out_dir / "wavs" / f"{tts_solver}_nfe{nfe:03d}"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    wers = []
    utmos_scores = []
    synth_wall = 0.0
    synth_audio = 0.0

    cfg_scale = float(args.demo_cfg_scale if args.demo_cfg_scale is not None else cfg["infer"].get("demo_cfg_scale", 1.0))
    prior_temp = float(args.demo_prior_temp if args.demo_prior_temp is not None else cfg["infer"].get("demo_prior_temp", 0.0))
    style_temp = float(args.demo_style_temp if args.demo_style_temp is not None else cfg["infer"].get("demo_style_temp", 0.0))

    for idx, (row_idx, row, text, spk) in enumerate(pairs):
        wav_path = wav_dir / f"{idx:06d}_test{row_idx:05d}_spk{safe_name(spk)}.wav"
        if wav_path.exists() and not args.force_resynthesize:
            synth_meta = {"wav_path": str(wav_path), "frames": None, "seconds": None}
            synth_elapsed = 0.0
        else:
            sync_if_cuda(device)
            t0 = time.perf_counter()
            synth_meta = synthesize_tts_with_solver(
                evaluator,
                text,
                spk,
                wav_path,
                solver=tts_solver,
                cfg_scale=cfg_scale,
                prior_temp=prior_temp,
                style_temp=style_temp,
                ode_steps=int(tts_steps),
            )
            sync_if_cuda(device)
            synth_elapsed = time.perf_counter() - t0

        seconds = synth_meta.get("seconds")
        if seconds is not None and seconds > 0:
            synth_wall += synth_elapsed
            synth_audio += float(seconds)

        utmos = score_utmos(utmos_model, str(wav_path), evaluator.sampling_rate, device)
        hyp = transcribe_whisper(whisper_model, str(wav_path), device)
        wer = word_error_rate_notebook(text, hyp) if hyp is not None else None
        if wer is not None:
            wers.append(float(wer))
        if utmos is not None:
            utmos_scores.append(float(utmos))

        rows.append(
            {
                "kind": "tts",
                "nfe": nfe,
                "tts_solver": tts_solver,
                "tts_steps": int(tts_steps),
                "idx": idx,
                "test_row_idx": row_idx,
                "speaker": spk,
                "text": text,
                "wav_path": str(wav_path),
                "seconds": seconds,
                "synth_wall_sec": synth_elapsed,
                "synth_rtf": (synth_elapsed / float(seconds)) if seconds and seconds > 0 else None,
                "utmos": utmos,
                "whisper_hyp": hyp,
                "tts_wer": wer,
                "ref_norm": normalize_english_text_for_wer(text),
                "hyp_norm": normalize_english_text_for_wer(hyp) if hyp is not None else None,
            }
        )

    return {
        "rows": rows,
        "summary": {
            "tts_pairs": len(rows),
            "tts_wer": mean_finite(wers),
            "utmos": mean_finite(utmos_scores),
            "tts_rtf": (synth_wall / synth_audio) if synth_audio > 0 else None,
            "tts_audio_sec": synth_audio,
            "tts_wall_sec": synth_wall,
        },
    }


def run_asr_sweep(*, args, asr_runtime: ASRRuntime, device: str, nfe: int, rows):
    out_rows = []
    wers = []
    wall_sum = 0.0
    audio_sum = 0.0
    for idx, row in enumerate(rows):
        ref = row_text(row, force_normalize=True)
        sync_if_cuda(device)
        t0 = time.perf_counter()
        error = None
        try:
            decoded = asr_runtime.decode_row(row, nfe=nfe, solver=args.asr_solver)
            hyp = decoded["hyp"]
            frames = decoded["frames"]
            seconds = decoded["seconds"]
        except Exception as exc:
            hyp = ""
            frames = None
            seconds = asr_seconds_from_row(row)
            error = repr(exc)
        sync_if_cuda(device)
        elapsed = time.perf_counter() - t0
        wer = None if error else word_error_rate_notebook(ref, hyp)
        if wer is not None:
            wers.append(float(wer))
        if seconds and seconds > 0:
            wall_sum += elapsed
            audio_sum += float(seconds)
        out_rows.append(
            {
                "kind": "asr",
                "nfe": nfe,
                "idx": idx,
                "wav": row.get("wav"),
                "speaker": row_speaker(row),
                "utt_id": row.get("utt_id"),
                "text": ref,
                "hyp": hyp,
                "frames": frames,
                "seconds": seconds,
                "asr_wall_sec": elapsed,
                "asr_rtf": (elapsed / float(seconds)) if seconds and seconds > 0 else None,
                "asr_wer": wer,
                "error": error,
            }
        )
    return {
        "rows": out_rows,
        "summary": {
            "asr_rows": len(out_rows),
            "asr_wer": mean_finite(wers),
            "asr_rtf": (wall_sum / audio_sum) if audio_sum > 0 else None,
            "asr_audio_sec": audio_sum,
            "asr_wall_sec": wall_sum,
            "asr_failures": sum(1 for r in out_rows if r["error"] is not None),
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Sweep NFE for TTS-WER/UTMOS/ASR-WER/RTF.")
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default="/work/dankker0900/bvfm/bvfm_speech/checkpoints/ckpt_joint_svae_zeroshot",
    )
    p.add_argument("--checkpoint", type=str, default="latest.pt")
    p.add_argument("--test-manifest", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--nfe-list", type=str, default="2,4,10,20")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--max-test-rows", type=int, default=50, help="Unique test texts for TTS.")
    p.add_argument("--max-asr-rows", type=int, default=50, help="Real test utterances for ASR.")
    p.add_argument("--num-speakers", type=int, default=1)
    p.add_argument("--pairing", choices=["round_robin", "cartesian"], default="round_robin")

    p.add_argument("--demo-cfg-scale", type=float, default=None)
    p.add_argument("--demo-prior-temp", type=float, default=None)
    p.add_argument("--demo-style-temp", type=float, default=None)
    p.add_argument(
        "--tts-heun-nfe-mode",
        choices=["exact", "steps"],
        default="exact",
        help="exact: Heun steps=ceil(NFE/2). steps: pass NFE as Heun steps.",
    )
    p.add_argument("--tts-solver", choices=["heun", "euler"], default="heun")
    p.add_argument("--asr-solver", choices=["euler", "heun"], default="euler")
    p.add_argument("--asr-style-use-mean", type=str2bool, default=None)

    p.add_argument("--skip-utmos", action="store_true")
    p.add_argument("--utmos-repo", type=str, default="tarepan/SpeechMOS:v1.2.0")
    p.add_argument("--utmos-model", type=str, default="utmos22_strong")
    p.add_argument("--skip-whisper", action="store_true")
    p.add_argument("--whisper-model", type=str, default="medium.en")
    p.add_argument("--force-resynthesize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = Path(args.ckpt_dir).resolve()
    cfg_path = ckpt_dir / "merged_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = load_json(cfg_path)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ckpt_dir / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_dir / "nfe_sweep_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_manifest = args.test_manifest or cfg["paths"].get("demo_aligned_manifest")
    if not test_manifest:
        raise RuntimeError("No --test-manifest and no paths.demo_aligned_manifest in config.")

    evaluator = TTSEvaluator(cfg, str(ckpt_path), str(out_dir), device)
    asr_runtime = ASRRuntime(evaluator, cfg, device, asr_style_use_mean_override=args.asr_style_use_mean)

    tts_text_rows = pick_test_rows(
        test_manifest,
        args.max_test_rows,
        force_normalize=bool(cfg["data"].get("force_text_normalize", True)),
    )
    asr_rows_all = []
    with open(test_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("include_asr", True):
                asr_rows_all.append(row)
            if args.max_asr_rows is not None and len(asr_rows_all) >= int(args.max_asr_rows):
                break
    if not asr_rows_all:
        raise RuntimeError(f"No ASR rows found in {test_manifest}")

    speakers = [spk for spk in list(evaluator.target_spks)[: int(args.num_speakers)] if spk in evaluator.spk2id]
    if not speakers:
        raise RuntimeError("No selected top-k speakers exist in speaker table.")
    pairs = build_eval_pairs(tts_text_rows, speakers, args.pairing)

    utmos_model = load_utmos(args, device)
    whisper_model = load_whisper(args, device)

    nfe_list = parse_nfe_list(args.nfe_list)
    summaries = []
    print(
        f"[NFE-SWEEP] ckpt={ckpt_path} device={device} manifest={test_manifest} "
        f"nfe={nfe_list} tts_pairs={len(pairs)} asr_rows={len(asr_rows_all)}"
    )
    print(
        f"[NFE-SWEEP] TTS solver={args.tts_solver} heun_mode={args.tts_heun_nfe_mode}; "
        f"ASR solver={args.asr_solver}; output={out_dir}"
    )

    for nfe in nfe_list:
        if args.tts_solver == "heun":
            tts_steps = int(nfe if args.tts_heun_nfe_mode == "steps" else max(1, math.ceil(float(nfe) / 2.0)))
        else:
            tts_steps = int(nfe)
        print(f"\n===== NFE {nfe} (tts_{args.tts_solver}_steps={tts_steps}, asr_{args.asr_solver}_steps={nfe}) =====")
        tts = run_tts_sweep(
            args=args,
            cfg=cfg,
            evaluator=evaluator,
            device=device,
            nfe=nfe,
            tts_steps=tts_steps,
            tts_solver=args.tts_solver,
            pairs=pairs,
            out_dir=out_dir,
            utmos_model=utmos_model,
            whisper_model=whisper_model,
        )
        asr = run_asr_sweep(args=args, asr_runtime=asr_runtime, device=device, nfe=nfe, rows=asr_rows_all)

        rows_path = out_dir / f"results_nfe{nfe:03d}.jsonl"
        with open(rows_path, "w", encoding="utf-8") as f:
            for row in tts["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in asr["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "nfe": int(nfe),
            "tts_solver": args.tts_solver,
            "tts_steps": int(tts_steps),
            "asr_solver": args.asr_solver,
            "asr_steps": int(nfe),
            **tts["summary"],
            **asr["summary"],
            "results_jsonl": str(rows_path),
        }
        summaries.append(summary)
        print(
            f"[NFE {nfe}] "
            f"tts-wer={summary['tts_wer']} utmos={summary['utmos']} "
            f"asr-wer={summary['asr_wer']} "
            f"tts-rtf={summary['tts_rtf']} asr-rtf={summary['asr_rtf']}"
        )

    summary_obj = {
        "checkpoint": str(ckpt_path),
        "config": str(cfg_path),
        "test_manifest": str(test_manifest),
        "device": device,
        "nfe_list": nfe_list,
        "tts_solver": args.tts_solver,
        "tts_heun_nfe_mode": args.tts_heun_nfe_mode,
        "asr_solver": args.asr_solver,
        "skip_utmos": bool(args.skip_utmos),
        "skip_whisper": bool(args.skip_whisper),
        "whisper_model": None if args.skip_whisper else args.whisper_model,
        "utmos_model": None if args.skip_utmos else f"{args.utmos_repo}/{args.utmos_model}",
        "summaries": summaries,
    }
    write_json(out_dir / "summary.json", summary_obj)
    print(f"\n[WRITE] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
