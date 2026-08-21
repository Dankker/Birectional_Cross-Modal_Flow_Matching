#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
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

from biflow.alignment import durations_to_int_and_fixsum
from biflow.models import euler_integrate, heun_integrate
from biflow.utils import save_wav, set_seed
from eval_tts_test_topk import TTSEvaluator


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_checkpoint(ckpt_dir, checkpoint):
    if os.path.isabs(str(checkpoint)):
        return str(checkpoint)
    return str(Path(ckpt_dir) / checkpoint)


def choose_arg(args, cfg, name, default=None):
    value = getattr(args, name)
    if value is not None:
        return value
    return cfg.get(name, default)


def resolve_seed(seed_value):
    if seed_value is None or str(seed_value).strip().lower() in {"", "none", "null", "random"}:
        return int.from_bytes(os.urandom(4), byteorder="little", signed=False)
    seed = int(seed_value)
    if seed < 0:
        return int.from_bytes(os.urandom(4), byteorder="little", signed=False)
    return seed


def tensor_sha1(tensor):
    arr = tensor.detach().cpu().float().contiguous().numpy()
    return hashlib.sha1(arr.tobytes()).hexdigest()


class ReferenceSpeakerEncoder:
    def __init__(
        self,
        *,
        model_name,
        savedir,
        cache_dir,
        device,
        sampling_rate,
        max_sec=12.0,
        l2_normalize=True,
    ):
        self.model_name = str(model_name)
        self.savedir = os.path.abspath(os.path.expanduser(str(savedir))) if savedir else None
        self.cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir))) if cache_dir else None
        self.device = str(device)
        self.sampling_rate = int(sampling_rate)
        self.max_sec = float(max_sec)
        self.l2_normalize = bool(l2_normalize)
        self.classifier = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _load_classifier(self):
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

    def _cache_path(self, wav_path):
        if not self.cache_dir:
            return None
        key = hashlib.sha1(os.path.abspath(str(wav_path)).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.pt")

    @torch.no_grad()
    def encode_one(self, wav_path, *, dtype=None):
        wav_path = os.path.abspath(str(wav_path))
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Reference wav not found: {wav_path}")
        cache_path = self._cache_path(wav_path)
        if cache_path and os.path.exists(cache_path):
            try:
                emb = torch.load(cache_path, map_location="cpu", weights_only=True).float().reshape(-1)
            except TypeError:
                emb = torch.load(cache_path, map_location="cpu").float().reshape(-1)
        else:
            import librosa

            wav_np, _ = librosa.load(wav_path, sr=self.sampling_rate, mono=True)
            if self.max_sec > 0:
                max_samples = int(round(self.max_sec * self.sampling_rate))
                wav_np = wav_np[:max_samples]
            wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(self.device)
            emb = self._load_classifier().encode_batch(wav).detach().float().cpu().reshape(-1)
            if self.l2_normalize:
                emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
            if cache_path:
                tmp_path = f"{cache_path}.tmp"
                torch.save(emb, tmp_path)
                os.replace(tmp_path, cache_path)
        emb = emb.unsqueeze(0).to(self.device)
        if dtype is not None:
            emb = emb.to(dtype=dtype)
        return emb


@torch.no_grad()
def synthesize_one(
    evaluator,
    *,
    text,
    ref_wav,
    solver,
    nfe,
    cfg_scale,
    prior_temp,
    style_temp,
    out_wav,
    out_mel=None,
    style_mode="prior",
):
    zero_cfg = evaluator.model_cfg.get("zero_shot", {})
    ref_encoder = ReferenceSpeakerEncoder(
        model_name=zero_cfg.get("ref_speaker_emb_model", "speechbrain/spkrec-ecapa-voxceleb"),
        savedir=zero_cfg.get("ref_speaker_emb_savedir"),
        cache_dir=zero_cfg.get("ref_emb_cache_dir"),
        device=evaluator.device,
        sampling_rate=evaluator.sampling_rate,
        max_sec=float(zero_cfg.get("ref_max_sec", 12.0)),
        l2_normalize=bool(zero_cfg.get("ref_l2_normalize", True)),
    )
    raw_spk = ref_encoder.encode_one(ref_wav)
    spk_e = evaluator.spk_table.from_pretrained_embedding(raw_spk)
    raw_spk_sha1 = tensor_sha1(raw_spk)
    spk_e_sha1 = tensor_sha1(spk_e)
    raw_spk_norm = float(raw_spk.detach().float().norm(dim=-1).mean().cpu().item())
    spk_e_norm = float(spk_e.detach().float().norm(dim=-1).mean().cpu().item())
    print(
        "[REF] "
        f"wav={os.path.abspath(str(ref_wav))} "
        f"raw_sha1={raw_spk_sha1[:12]} raw_norm={raw_spk_norm:.6f} "
        f"spk_cond_sha1={spk_e_sha1[:12]} spk_cond_norm={spk_e_norm:.6f}"
    )

    h_enc, maskL, mu_tok, logvar_tok, _, _ = evaluator.encode_text_batch([text])
    L_valid = int(maskL.sum().item())
    if L_valid <= 0:
        raise RuntimeError("No valid text tokens after text normalization.")

    limit = int(getattr(evaluator.vf.rope, "max_seq_len", 4096))
    if evaluator.len_pred is not None:
        k_raw = int(max(16, round(float(evaluator.len_pred(h_enc, maskL).item()))))
        k_pred = min(max(k_raw, L_valid), limit)
    else:
        log_dur_tmp = evaluator.dur_pred(h_enc, maskL)
        dur_tmp = (torch.exp(log_dur_tmp) - 1.0) * maskL.float()
        k_pred = min(max(int(round(float(dur_tmp.sum().item()))), L_valid, 16), limit)

    log_dur = evaluator.dur_pred(h_enc, maskL)
    dur = (torch.exp(log_dur) - 1.0) * maskL.float()
    dur_int, _ = durations_to_int_and_fixsum(dur, maskL, k_pred)

    mu_feats = []
    lv_feats = []
    for i in range(L_valid):
        d = int(dur_int[i].item())
        if d <= 0:
            continue
        mu_feats.append(mu_tok[0, i:i + 1].repeat(d, 1))
        lv_feats.append(logvar_tok[0, i:i + 1].repeat(d, 1))
    if not mu_feats:
        raise RuntimeError("Duration predictor produced no valid frames.")

    zT_mean = torch.cat(mu_feats, dim=0).unsqueeze(0)[:, :k_pred]
    zT_logvar = torch.cat(lv_feats, dim=0).unsqueeze(0)[:, :k_pred]
    if float(prior_temp) > 0.0:
        zT0 = zT_mean + float(prior_temp) * torch.exp(0.5 * zT_logvar) * torch.randn_like(zT_mean)
    else:
        zT0 = zT_mean

    maskK = torch.ones(1, zT0.shape[1], device=evaluator.device, dtype=torch.bool)

    style_mode = str(style_mode).strip().lower()
    if style_mode not in {"prior", "zero"}:
        raise ValueError(f"Unsupported style_mode={style_mode!r}; expected prior or zero")

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
        if float(style_temp) > 0.0:
            style_e = u_mu + float(style_temp) * torch.exp(0.5 * u_logvar) * torch.randn_like(u_mu)
        else:
            style_e = u_mu
        if style_mode == "zero":
            style_e = torch.zeros_like(style_e)

    text_cond = None
    if evaluator.canonical_to_source is not None:
        zT0_source = evaluator.canonical_to_source(
            zT0,
            maskK,
            spk_e=spk_e.to(dtype=zT0.dtype),
            style_e=style_e if evaluator.tts_style_into_source else None,
        )
        zT_mean_source = evaluator.canonical_to_source(
            zT_mean,
            maskK,
            spk_e=spk_e.to(dtype=zT_mean.dtype),
            style_e=(
                style_e.to(dtype=zT_mean.dtype)
                if (evaluator.tts_style_into_source and style_e is not None)
                else None
            ),
        )
        if evaluator.use_vf_canonical_text_cond:
            text_cond = zT_mean
    else:
        zT0_source = zT0
        zT_mean_source = zT_mean
        if evaluator.tts_style_to_source is not None and style_e is not None:
            style_bias = evaluator.tts_style_to_source(style_e, spk_e.to(dtype=zT0.dtype)).to(dtype=zT0.dtype)
            zT0_source = zT0_source + evaluator.tts_style_source_scale * style_bias.unsqueeze(1)
            zT_mean_source = zT_mean_source + evaluator.tts_style_source_scale * style_bias.unsqueeze(1)
        if evaluator.use_vf_canonical_text_cond:
            text_cond = zT_mean

    spk_e_vf = spk_e.to(dtype=zT0_source.dtype)
    if evaluator.tts_source_cond is not None and float(prior_temp) > 0.0 and evaluator.canonical_to_source is None:
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
            steps=int(nfe),
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
            steps=int(nfe),
            direction=+1,
            cfg_flag_value=1,
            spk_e=spk_e_vf,
            style_e=style_e,
            text_cond=text_cond,
        )
    else:
        raise ValueError(f"Unsupported solver={solver!r}; expected heun or euler")

    zS_ref = (
        evaluator.mel_refiner(zS_pred, cond=h_enc, cond_mask=maskL)
        if evaluator.mel_refiner is not None
        else zS_pred
    )
    mel = (zS_ref * evaluator.std_b + evaluator.mu_b).float()
    wav = evaluator.svae_model.decode(mel).squeeze(0).float().detach().cpu()
    infer_sec = time.perf_counter() - t0

    out_wav = str(out_wav)
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    save_wav(out_wav, wav, sr=evaluator.sampling_rate)
    if out_mel:
        np.save(str(out_mel), mel.squeeze(0).detach().cpu().numpy().astype(np.float32))

    audio_sec = float(wav.numel()) / float(evaluator.sampling_rate)
    return {
        "wav_path": out_wav,
        "mel_path": str(out_mel) if out_mel else None,
        "text": text,
        "ref_wav": os.path.abspath(str(ref_wav)),
        "ref_raw_emb_sha1": raw_spk_sha1,
        "ref_raw_emb_norm": raw_spk_norm,
        "ref_spk_cond_sha1": spk_e_sha1,
        "ref_spk_cond_norm": spk_e_norm,
        "solver": solver,
        "nfe": int(nfe),
        "cfg_scale": float(cfg_scale),
        "prior_temp": float(prior_temp),
        "style_temp": float(style_temp),
        "style_mode": style_mode,
        "style_norm": (
            float(style_e.detach().float().norm(dim=-1).mean().cpu().item())
            if style_e is not None
            else 0.0
        ),
        "text_tokens": int(L_valid),
        "frames": int(mel.shape[1]),
        "sampling_rate": int(evaluator.sampling_rate),
        "audio_seconds": audio_sec,
        "infer_seconds": float(infer_sec),
        "rtf": float(infer_sec / max(audio_sec, 1e-8)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Single zero-shot TTS inference for one text and one reference wav.")
    p.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "infer_tts_one_zeroshot.json"))
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--ref-wav", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--out-name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--solver", choices=["heun", "euler"], default=None)
    p.add_argument("--nfe", type=int, default=None)
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--prior-temp", type=float, default=None)
    p.add_argument("--style-temp", type=float, default=None)
    p.add_argument("--style-mode", choices=["prior", "zero"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-save-mel", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_json(args.config) if args.config and os.path.exists(args.config) else {}

    ckpt_dir = choose_arg(args, cfg, "ckpt_dir", str(REPO_ROOT / "ckpt_joint_svae_zeroshot_best"))
    checkpoint = choose_arg(args, cfg, "checkpoint", "latest.pt")
    weights_root = os.environ.get("BVFM_WEIGHTS_ROOT")
    if weights_root and args.ckpt_dir is None:
        ckpt_dir = os.path.join(weights_root, "speech")
    if weights_root and args.checkpoint is None:
        checkpoint = "bvfm_speech_step299999_inference.pt"
    text = choose_arg(args, cfg, "text", None)
    ref_wav = choose_arg(args, cfg, "ref_wav", None)
    out_dir = choose_arg(args, cfg, "out_dir", str(REPO_ROOT / "tts_infer_one_outputs"))
    out_name = choose_arg(args, cfg, "out_name", "tts_one")
    device = choose_arg(args, cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
    solver = choose_arg(args, cfg, "solver", "heun")
    nfe = int(choose_arg(args, cfg, "nfe", 20))
    cfg_scale = float(choose_arg(args, cfg, "cfg_scale", 1.0))
    prior_temp = float(choose_arg(args, cfg, "prior_temp", 0.0))
    style_temp = float(choose_arg(args, cfg, "style_temp", 0.0))
    style_mode = str(choose_arg(args, cfg, "style_mode", "prior"))
    seed = resolve_seed(choose_arg(args, cfg, "seed", None))
    save_mel = bool(cfg.get("save_mel", True)) and not args.no_save_mel

    if not text:
        raise ValueError("Missing text. Pass --text or set text in the JSON config.")
    if not ref_wav:
        raise ValueError("Missing reference wav. Pass --ref-wav or set ref_wav in the JSON config.")

    set_seed(seed)
    ckpt_path = resolve_checkpoint(ckpt_dir, checkpoint)
    model_cfg_path = os.path.join(ckpt_dir, "merged_config.json")
    if not os.path.exists(model_cfg_path):
        model_cfg_path = str(REPO_ROOT / "configs" / "cutmanifest_svae_latent.json")
    model_cfg = load_json(model_cfg_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"{out_name}.wav"
    out_mel = out_dir / f"{out_name}_mel.npy" if save_mel else None
    out_json = out_dir / f"{out_name}.json"

    print(f"[INFO] ckpt_path={ckpt_path}")
    print(f"[INFO] model_cfg={model_cfg_path}")
    print(f"[INFO] text={text}")
    print(f"[INFO] ref_wav={ref_wav}")
    print(f"[INFO] out_wav={out_wav}")
    print(f"[INFO] solver={solver} nfe={nfe} device={device}")

    evaluator = TTSEvaluator(model_cfg, ckpt_path, out_dir, device)
    result = synthesize_one(
        evaluator,
        text=text,
        ref_wav=ref_wav,
        solver=solver,
        nfe=nfe,
        cfg_scale=cfg_scale,
        prior_temp=prior_temp,
        style_temp=style_temp,
        style_mode=style_mode,
        out_wav=out_wav,
        out_mel=out_mel,
    )
    result.update({
        "ckpt_dir": os.path.abspath(str(ckpt_dir)),
        "checkpoint": str(checkpoint),
        "ckpt_path": os.path.abspath(str(ckpt_path)),
        "seed": int(seed),
    })
    write_json(out_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
