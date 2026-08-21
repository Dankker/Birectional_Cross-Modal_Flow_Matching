#!/usr/bin/env python3
"""Evaluate paired normalized latent transport error on LibriTTS/SVAE pairs."""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from biflow.utils import set_seed  # noqa: E402
from eval_tts_duration_modes import (  # noqa: E402
    SpeakerEncoder,
    load_json,
    maybe_get_speaker_condition,
    oracle_mas_durations,
    read_jsonl,
    resolve_checkpoint,
    row_speaker,
    row_text,
    row_wav,
    select_rows,
)
from eval_tts_test_topk import TTSEvaluator  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    weights_root = os.environ.get("BVFM_WEIGHTS_ROOT")
    parser.add_argument(
        "--ckpt-dir",
        default=(
            os.path.join(weights_root, "speech")
            if weights_root
            else str(REPO_ROOT / "checkpoints" / "ckpt_joint_svae_zeroshot_norm")
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "bvfm_speech_step299999_inference.pt" if weights_root else "latest.pt"
        ),
    )
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "runs" / "transport_error" / "speech_transport.npz"),
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--solver", choices=["euler", "heun"], default="heun")
    parser.add_argument("--tts-cfg", type=float, default=1.0)
    parser.add_argument("--zv-temperature", type=float, default=0.0)
    parser.add_argument("--reference-mode", choices=["other", "self"], default="other")
    parser.add_argument("--speaker-model")
    parser.add_argument("--speaker-savedir")
    parser.add_argument("--speaker-max-sec", type=float, default=12.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    return parser.parse_args()


def sample_gaussian(mu, logvar, temperature, generator):
    if float(temperature) <= 0.0:
        return mu
    noise = torch.randn(
        mu.shape,
        device=mu.device,
        dtype=mu.dtype,
        generator=generator,
    )
    return mu + float(temperature) * torch.exp(0.5 * logvar) * noise


@torch.no_grad()
def integrate_trace(
    vf,
    source,
    mask,
    *,
    steps,
    direction,
    solver,
    cfg_scale,
    spk_e,
    style_e,
    text_cond,
    style_enabled,
):
    """Integrate while preserving ``style_e=None`` as a true branch bypass."""
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    state = source.float()
    batch = state.shape[0]
    dt = float(direction) / float(steps)
    t0 = 0.0 if int(direction) == 1 else 1.0
    condition_flag = torch.ones(batch, dtype=torch.long, device=state.device)
    unconditional_flag = torch.zeros(batch, dtype=torch.long, device=state.device)
    if spk_e is None:
        spk_condition = torch.zeros(
            batch, vf.E_spk, device=state.device, dtype=state.dtype
        )
    else:
        spk_condition = spk_e.to(device=state.device, dtype=state.dtype)
    spk_zero = torch.zeros_like(spk_condition)
    style_condition = None
    if bool(style_enabled):
        if style_e is None:
            raise ValueError("style_enabled=True requires a style latent")
        style_condition = style_e.to(device=state.device, dtype=state.dtype)
    text_condition = (
        text_cond.to(device=state.device, dtype=state.dtype)
        if text_cond is not None
        else None
    )
    states = [state.detach().cpu()]

    def evaluate(z_now, t_now, cfg_flag, speaker, style, text):
        time = torch.full(
            (batch,), float(t_now), device=state.device, dtype=torch.float32
        )
        return vf(
            z_now,
            time,
            mask,
            cfg_flag=cfg_flag,
            spk_e=speaker,
            style_e=style,
            text_cond=text,
        )

    def velocity(z_now, t_now):
        conditional = evaluate(
            z_now,
            t_now,
            condition_flag,
            spk_condition,
            style_condition,
            text_condition,
        )
        if float(cfg_scale) == 1.0:
            return conditional
        unconditional = evaluate(
            z_now,
            t_now,
            unconditional_flag,
            spk_zero,
            None,
            None,
        )
        return unconditional + float(cfg_scale) * (conditional - unconditional)

    for index in range(int(steps)):
        time = t0 + index * dt
        k1 = velocity(state, time)
        if solver == "euler":
            state = state + dt * k1.float()
        else:
            predicted = state + dt * k1.float()
            k2 = velocity(predicted, time + dt)
            state = state + dt * 0.5 * (k1.float() + k2.float())
        states.append(state.detach().cpu())
    return torch.stack(states, dim=0)


def normalized_error(states, source, target, mask, epsilon):
    states = states.float()
    source = source.detach().float().cpu()
    target = target.detach().float().cpu()
    valid = mask.detach().cpu().bool().unsqueeze(-1)
    state_valid = valid.unsqueeze(0).to(dtype=states.dtype)
    numerator = ((states - target.unsqueeze(0)) * state_valid).flatten(2).norm(dim=-1)
    denominator = (((source - target) * valid).flatten(1).norm(dim=-1)).clamp_min(
        float(epsilon)
    )
    return (numerator / denominator.unsqueeze(0)).transpose(0, 1)


def build_style_latents(evaluator, z_s, z_c, mask, spk_e, temperature, generator):
    if not evaluator.use_tts_style_latent:
        raise RuntimeError("Checkpoint does not enable the z_v/style latent")
    if evaluator.tts_style_prior is None:
        tts_mu = torch.zeros(
            z_c.shape[0],
            evaluator.tts_style_dim,
            device=z_c.device,
            dtype=z_c.dtype,
        )
        tts_logvar = torch.zeros_like(tts_mu)
    elif getattr(evaluator, "tts_style_prior_type", "") == "canonical_speaker":
        tts_mu, tts_logvar = evaluator.tts_style_prior(
            z_c,
            mask,
            spk_e.to(device=z_c.device, dtype=z_c.dtype),
        )
    else:
        tts_mu, tts_logvar = evaluator.tts_style_prior(
            spk_e.to(device=z_c.device, dtype=z_c.dtype)
        )
    tts_style = sample_gaussian(
        tts_mu, tts_logvar, temperature, generator
    ).float()

    if evaluator.tts_style_post is None:
        raise RuntimeError("Checkpoint is missing the speech-side z_v posterior")
    post_mode = str(evaluator.model_cfg.get("tts_style_post_mode", "speech")).lower()
    if post_mode == "path":
        time = torch.ones(z_s.shape[0], device=z_s.device, dtype=z_s.dtype)
        spk_zero = torch.zeros(
            z_s.shape[0], evaluator.E_spk, device=z_s.device, dtype=z_s.dtype
        )
        asr_mu, asr_logvar = evaluator.tts_style_post(
            z_s,
            mask,
            z_t=z_s,
            t=time,
            spk_e=spk_zero,
        )
    else:
        asr_mu, asr_logvar = evaluator.tts_style_post(z_s, mask)
    asr_style = sample_gaussian(
        asr_mu, asr_logvar, temperature, generator
    ).float()
    return tts_style, asr_style


def standard_error(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def choose_reference(row, speaker_wavs, mode):
    target = row_wav(row)
    if mode == "self":
        return target
    for candidate in speaker_wavs.get(row_speaker(row), []):
        if os.path.abspath(candidate) != os.path.abspath(target):
            return candidate
    return target


def main():
    args = parse_args()
    if args.samples <= 0 or args.steps <= 0:
        raise ValueError("samples and steps must be positive")
    if args.zv_temperature < 0.0:
        raise ValueError("zv-temperature must be non-negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Speech transport evaluation requested CUDA, but CUDA is unavailable")
    set_seed(int(args.seed))
    np.random.seed(int(args.seed))

    ckpt_dir = Path(args.ckpt_dir).resolve()
    checkpoint_path = os.path.abspath(resolve_checkpoint(ckpt_dir, args.checkpoint))
    config_path = Path(args.config).resolve() if args.config else ckpt_dir / "merged_config.json"
    if not config_path.exists():
        config_path = REPO_ROOT / "configs" / "cutmanifest_svae_latent.json"
    config = load_json(config_path)
    bundled_svae_root = REPO_ROOT / "Semantic-VAE"
    bundled_svae_ckpt = bundled_svae_root / "ckpts" / "semantic_vae_1000k"
    cache_config = config.setdefault("cache", {})
    if not os.path.isdir(str(cache_config.get("semantic_vae_root", ""))):
        cache_config["semantic_vae_root"] = str(bundled_svae_root)
    if not os.path.isdir(str(cache_config.get("semantic_vae_ckpt", ""))):
        cache_config["semantic_vae_ckpt"] = str(bundled_svae_ckpt)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    evaluator = TTSEvaluator(config, checkpoint_path, os.path.dirname(output_path), args.device)
    if evaluator.use_true_canonical_latent:
        raise NotImplementedError(
            "This evaluator currently requires model.use_true_canonical_latent=false "
            "so both endpoints occupy the same vector-field state space"
        )
    if evaluator.tts_style_into_source or evaluator.tts_style_to_source is not None:
        raise NotImplementedError(
            "The current experiment requires z_v to condition only the vector field, "
            "so w/ and w/o z_v have an identical source endpoint"
        )
    if not bool(evaluator.model_cfg.get("asr_use_style_cond", False)):
        raise RuntimeError("Checkpoint has model.asr_use_style_cond=false")

    manifest = args.manifest or config.get("paths", {}).get("demo_aligned_manifest")
    if not manifest:
        processed = config.get("paths", {}).get("processed_unified_dir")
        manifest = os.path.join(processed, "full_manifest_clean.jsonl")
    manifest = os.path.abspath(manifest)
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest}")
    all_rows = read_jsonl(manifest)
    # TTSEvaluator initializes this lookup from the training manifests. Extend
    # it with the externally selected test manifest before MAS or latent loads.
    for row in all_rows:
        wav_path = row_wav(row)
        latent_path = (
            row.get("svae_latent_path")
            or row.get("speech_path")
            or row.get("latent_path")
        )
        if wav_path and latent_path:
            evaluator.speech_path_by_wav[os.path.abspath(wav_path)] = os.path.abspath(
                str(latent_path)
            )
    candidate_rows = select_rows(all_rows, evaluator, max_items=None)
    speaker_wavs = defaultdict(list)
    for row in candidate_rows:
        wav_path = row_wav(row)
        if wav_path and os.path.isfile(wav_path):
            speaker_wavs[row_speaker(row)].append(wav_path)

    zero_shot = evaluator.model_cfg.get("zero_shot", {})
    speaker_model = args.speaker_model or zero_shot.get(
        "ref_speaker_emb_model", "speechbrain/spkrec-ecapa-voxceleb"
    )
    speaker_savedir = args.speaker_savedir or zero_shot.get("ref_speaker_emb_savedir")
    speaker_encoder = None
    if hasattr(evaluator.spk_table, "from_pretrained_embedding"):
        speaker_encoder = SpeakerEncoder(
            speaker_model,
            speaker_savedir,
            args.device,
            evaluator.sampling_rate,
            max_sec=float(args.speaker_max_sec),
        )

    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(args.seed) + 104729)
    task_values = {
        "tts_without_zv": [],
        "tts_with_zv": [],
        "asr_without_zv": [],
        "asr_with_zv": [],
    }
    utterance_ids = []
    reference_wavs = []
    target_wavs = []
    texts = []
    failures = []

    for row_index, row in enumerate(candidate_rows):
        if len(utterance_ids) >= int(args.samples):
            break
        utterance_id = str(row.get("utt_id") or row.get("id") or f"row{row_index:06d}")
        text = evaluator.canonicalize_text(row_text(row))
        target_wav = row_wav(row)
        reference_wav = choose_reference(row, speaker_wavs, args.reference_mode)
        try:
            mas = oracle_mas_durations(evaluator, text, target_wav)
            z_s = evaluator.load_svae_latent(target_wav).to(args.device)
            z_s = ((z_s.unsqueeze(0) - evaluator.mu_b) / evaluator.std_b).float()
            z_c = mas["zc_mean_mas"].float()
            if z_s.shape != z_c.shape:
                raise RuntimeError(
                    f"paired endpoint shape mismatch: speech={tuple(z_s.shape)} "
                    f"text={tuple(z_c.shape)}"
                )
            mask = torch.ones(
                z_s.shape[0], z_s.shape[1], device=args.device, dtype=torch.bool
            )
            reference_row = dict(row)
            reference_row["wav"] = reference_wav
            reference_row["parent_wav"] = reference_wav
            spk_e, _ = maybe_get_speaker_condition(
                evaluator, reference_row, speaker_encoder
            )
            spk_e = spk_e.to(device=args.device, dtype=z_c.dtype)
            tts_style, asr_style = build_style_latents(
                evaluator,
                z_s,
                z_c,
                mask,
                spk_e,
                args.zv_temperature,
                generator,
            )
            text_cond = z_c if evaluator.use_vf_canonical_text_cond else None

            trajectories = {
                "tts_without_zv": integrate_trace(
                    evaluator.vf,
                    z_c,
                    mask,
                    steps=args.steps,
                    direction=1,
                    solver=args.solver,
                    cfg_scale=args.tts_cfg,
                    spk_e=spk_e,
                    style_e=None,
                    text_cond=text_cond,
                    style_enabled=False,
                ),
                "tts_with_zv": integrate_trace(
                    evaluator.vf,
                    z_c,
                    mask,
                    steps=args.steps,
                    direction=1,
                    solver=args.solver,
                    cfg_scale=args.tts_cfg,
                    spk_e=spk_e,
                    style_e=tts_style,
                    text_cond=text_cond,
                    style_enabled=True,
                ),
                "asr_without_zv": integrate_trace(
                    evaluator.vf,
                    z_s,
                    mask,
                    steps=args.steps,
                    direction=-1,
                    solver=args.solver,
                    cfg_scale=1.0,
                    spk_e=None,
                    style_e=None,
                    text_cond=None,
                    style_enabled=False,
                ),
                "asr_with_zv": integrate_trace(
                    evaluator.vf,
                    z_s,
                    mask,
                    steps=args.steps,
                    direction=-1,
                    solver=args.solver,
                    cfg_scale=1.0,
                    spk_e=None,
                    style_e=asr_style,
                    text_cond=None,
                    style_enabled=True,
                ),
            }
            for key in ("tts_without_zv", "tts_with_zv"):
                task_values[key].append(
                    normalized_error(
                        trajectories[key], z_c, z_s, mask, args.epsilon
                    )[0].numpy()
                )
            for key in ("asr_without_zv", "asr_with_zv"):
                task_values[key].append(
                    normalized_error(
                        trajectories[key], z_s, z_c, mask, args.epsilon
                    )[0].numpy()
                )
            utterance_ids.append(utterance_id)
            reference_wavs.append(reference_wav)
            target_wavs.append(target_wav)
            texts.append(text)
            print(
                f"[SAMPLE] {len(utterance_ids)}/{args.samples} utt={utterance_id} "
                f"frames={z_s.shape[1]} ref_mode="
                f"{'self' if reference_wav == target_wav else 'other'}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"utt_id": utterance_id, "error": repr(exc)})
            print(f"[WARN] skipping {utterance_id}: {exc}", flush=True)

    if len(utterance_ids) < int(args.samples):
        raise RuntimeError(
            f"Only {len(utterance_ids)} successful samples out of requested "
            f"{args.samples}; failures={len(failures)}"
        )

    arrays = {
        key: np.stack(values, axis=0).astype(np.float32)
        for key, values in task_values.items()
    }
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite normalized error in {key}")
        if not np.allclose(values[:, 0], 1.0, atol=2e-4, rtol=2e-4):
            raise RuntimeError(
                f"Normalization invariant failed for {key}: "
                f"D(0) range=({values[:, 0].min()}, {values[:, 0].max()})"
            )

    progress = np.linspace(0.0, 1.0, int(args.steps) + 1, dtype=np.float32)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(1, dtype=np.int64),
        domain=np.asarray("speech"),
        progress=progress,
        utterance_ids=np.asarray(utterance_ids, dtype=np.str_),
        texts=np.asarray(texts, dtype=np.str_),
        target_wavs=np.asarray(target_wavs, dtype=np.str_),
        reference_wavs=np.asarray(reference_wavs, dtype=np.str_),
        **arrays,
    )

    summary = {
        "schema_version": 1,
        "domain": "speech",
        "output": output_path,
        "checkpoint": checkpoint_path,
        "checkpoint_step": int(evaluator.step),
        "config": str(config_path),
        "manifest": manifest,
        "samples": len(utterance_ids),
        "failures": failures,
        "steps": int(args.steps),
        "solver": args.solver,
        "tts_cfg": float(args.tts_cfg),
        "zv_temperature": float(args.zv_temperature),
        "reference_mode": args.reference_mode,
        "seed": int(args.seed),
        "epsilon": float(args.epsilon),
        "paired_target": True,
        "tts_alignment": "oracle MAS to the paired target SVAE latent length",
        "without_zv": "true style-projector bypass (style_e=None at every VF call)",
        "normalization": "masked ||z_hat(s)-z_target||_F/(||z_source-z_target||_F+epsilon)",
        "tasks": {},
    }
    for task in ("tts", "asr"):
        without = arrays[f"{task}_without_zv"][:, -1]
        with_zv = arrays[f"{task}_with_zv"][:, -1]
        summary["tasks"][task] = {
            "without_zv_endpoint_mean": float(without.mean()),
            "without_zv_endpoint_se": standard_error(without),
            "with_zv_endpoint_mean": float(with_zv.mean()),
            "with_zv_endpoint_se": standard_error(with_zv),
            "paired_endpoint_delta_mean": float((without - with_zv).mean()),
        }
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
