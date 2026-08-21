#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
BIGVGAN_ROOT = os.path.join(REPO_ROOT, "BigVGAN")
if BIGVGAN_ROOT not in sys.path:
    sys.path.insert(0, BIGVGAN_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import librosa
import torch
import torch.nn.functional as F

from env import AttrDict
from meldataset import get_mel_spectrogram

from biflow.models import (
    BaselineCTCHead,
    CanonicalPosterior,
    DiTVectorField,
    FrameCTCConvHead,
    SourceToCanonical,
    euler_integrate,
    heun_integrate,
)
from biflow.alignment import downsample_time_bkd
from biflow.ctc_decode import KenLMCTCDecoderConfig, OptionalKenLMCTCDecoder
from biflow.tokenizer import CharTokenizer, build_tokenizer, ctc_greedy_decode
from biflow.utils import normalize_text_basic, read_jsonl_rows, word_error_rate_text


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Bi-Flow ASR WER on a manifest")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/work/dankker0900/biflow_repo_cutmanifest_notebook_masprior/ckpt_joint_notebook_fix/latest.pt",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
    )
    p.add_argument(
        "--manifest",
        type=str,
        default="/work/dankker0900/dataset/test_clean_manifest_aligned_FULL.jsonl",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use-ema", choices=["true", "false"], default="true")
    p.add_argument("--solver", choices=["auto", "euler", "heun"], default="auto")
    p.add_argument("--decode-mode", choices=["greedy", "beam", "kenlm"], default="greedy")
    p.add_argument("--beam-size", type=int, default=50)
    p.add_argument("--beam-size-token", type=int, default=8)
    p.add_argument("--kenlm-preset", type=str, default="librispeech-4-gram")
    p.add_argument("--kenlm-lexicon", type=str, default=None)
    p.add_argument("--kenlm-lm", type=str, default=None)
    p.add_argument("--kenlm-lexicon-corpus", type=str, default=None)
    p.add_argument("--kenlm-beam-threshold", type=float, default=100.0)
    p.add_argument("--kenlm-lm-weight", type=float, default=1.23)
    p.add_argument("--kenlm-word-score", type=float, default=-0.26)
    p.add_argument("--kenlm-no-fallback", action="store_true")
    p.add_argument("--euler-steps", type=int, default=None)
    p.add_argument("--heun-steps", type=int, default=None)
    p.add_argument(
        "--chunk-core",
        type=int,
        default=None,
        help="If unset, use the same full-ASR chunk_core as train/demo config.",
    )
    p.add_argument("--chunk-ctx", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--print-every", type=int, default=50)
    p.add_argument("--output-jsonl", type=str, default=None)
    p.add_argument("--summary-json", type=str, default=None)
    return p.parse_args()


def choose_device(requested: str | None):
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_kenlm_lexicon_corpus(cfg: dict) -> str | None:
    path_cfg = cfg.get("paths", {})
    processed_unified_dir = path_cfg.get("processed_unified_dir")
    use_processed_unified = bool(path_cfg.get("use_processed_unified", False) or processed_unified_dir)
    if processed_unified_dir:
        processed_unified_dir = os.path.abspath(str(processed_unified_dir))

    candidates = []
    if use_processed_unified and processed_unified_dir:
        candidates.append(path_cfg.get("full_manifest_clean"))
        candidates.append(os.path.join(processed_unified_dir, "full_manifest_clean.jsonl"))
    candidates.extend([
        path_cfg.get("aligned_manifest"),
        path_cfg.get("cut_manifest"),
    ])
    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(str(cand))
    return None


def normalize_asr_text(text: str) -> str:
    return normalize_text_basic(text)


def _edit_distance(ref, hyp):
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def word_error_rate_normalized(ref_text: str, hyp_text: str) -> float:
    ref = normalize_asr_text(ref_text).split()
    hyp = normalize_asr_text(hyp_text).split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return float(_edit_distance(ref, hyp)) / float(len(ref))


def char_error_rate_normalized(ref_text: str, hyp_text: str) -> float:
    ref = list(normalize_asr_text(ref_text).replace(" ", ""))
    hyp = list(normalize_asr_text(hyp_text).replace(" ", ""))
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return float(_edit_distance(ref, hyp)) / float(len(ref))


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_prefix_beam_decode(logits_bkv, blank_id: int, beam_size: int = 50, beam_size_token: int = 8):
    assert logits_bkv.dim() == 3 and logits_bkv.shape[0] == 1, "Expected logits shape [1, T, V]"
    log_probs = F.log_softmax(logits_bkv[0], dim=-1).detach().cpu()
    beam = {(): (0.0, -math.inf)}

    for t in range(int(log_probs.shape[0])):
        next_beam = {}
        step = log_probs[t]
        if beam_size_token and beam_size_token > 0 and beam_size_token < int(step.shape[0]):
            cand_scores, cand_ids = torch.topk(step, k=int(beam_size_token))
            candidates = [(int(i), float(s)) for i, s in zip(cand_ids.tolist(), cand_scores.tolist())]
        else:
            candidates = [(i, float(step[i].item())) for i in range(int(step.shape[0]))]

        for prefix, (p_b, p_nb) in beam.items():
            p_total = _logaddexp(p_b, p_nb)
            for token_id, token_lp in candidates:
                if token_id == blank_id:
                    nb_b, nb_nb = next_beam.get(prefix, (-math.inf, -math.inf))
                    next_beam[prefix] = (_logaddexp(nb_b, p_total + token_lp), nb_nb)
                    continue

                last = prefix[-1] if prefix else None
                if token_id == last:
                    nb_b, nb_nb = next_beam.get(prefix, (-math.inf, -math.inf))
                    next_beam[prefix] = (nb_b, _logaddexp(nb_nb, p_nb + token_lp))

                    prefix_plus = prefix + (token_id,)
                    nb_b2, nb_nb2 = next_beam.get(prefix_plus, (-math.inf, -math.inf))
                    next_beam[prefix_plus] = (nb_b2, _logaddexp(nb_nb2, p_b + token_lp))
                else:
                    prefix_plus = prefix + (token_id,)
                    nb_b, nb_nb = next_beam.get(prefix_plus, (-math.inf, -math.inf))
                    next_beam[prefix_plus] = (nb_b, _logaddexp(nb_nb, p_total + token_lp))

        beam_items = sorted(
            next_beam.items(),
            key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
            reverse=True,
        )
        beam = dict(beam_items[: max(1, int(beam_size))])

    best_prefix = max(beam.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]))[0]
    return list(best_prefix)


def build_vf(model_cfg, d_mel: int):
    use_true_canonical = bool(model_cfg.get("use_true_canonical_latent", False))
    canonical_dim = int(model_cfg.get("canonical_dim", 192))
    text_cond_dim = canonical_dim if (use_true_canonical and bool(model_cfg.get("use_vf_canonical_text_cond", True))) else 0
    vf = DiTVectorField(
        D=d_mel,
        E_spk=int(model_cfg["E_spk"]),
        style_dim=int(model_cfg.get("tts_style_dim", 64)) if bool(model_cfg.get("use_tts_style_latent", False)) else 0,
        text_cond_dim=text_cond_dim,
        hidden=int(model_cfg["vf_hidden"]),
        depth=int(model_cfg["vf_depth"]),
        n_heads=int(model_cfg["vf_heads"]),
        dropout=float(model_cfg["vf_dropout"]),
        max_len=int(model_cfg["vf_max_len"]),
    )
    asr_vf_use_speaker_cond_cfg = model_cfg.get("asr_vf_use_speaker_cond", None)
    vf_use_speaker_cond = bool(model_cfg.get("vf_use_speaker_cond", True))
    asr_vf_use_speaker_cond = (
        vf_use_speaker_cond
        if asr_vf_use_speaker_cond_cfg is None
        else bool(asr_vf_use_speaker_cond_cfg)
    )
    vf.direct_speaker_cond = bool(vf_use_speaker_cond or asr_vf_use_speaker_cond)
    return vf


def build_ctc_head(model_cfg, vocab_size: int, d_mel: int):
    ctc_head_type = str(model_cfg.get("ctc_head_type", "frame_conv"))
    if ctc_head_type == "baseline":
        return BaselineCTCHead(
            V=vocab_size,
            D=d_mel,
            hidden=int(model_cfg["ctc_hidden"]),
            conv_layers=int(model_cfg["ctc_layers"]),
            ksize=int(model_cfg["ctc_ksize"]),
            lstm_hidden=int(model_cfg["ctc_lstm_hidden"]),
            lstm_layers=int(model_cfg["ctc_lstm_layers"]),
            dropout=float(model_cfg["ctc_dropout"]),
        )
    return FrameCTCConvHead(
        V=vocab_size,
        D=d_mel,
        hidden=int(model_cfg["ctc_hidden"]),
        layers=int(model_cfg["ctc_layers"]),
        ksize=int(model_cfg["ctc_ksize"]),
    )


def prepare_ctc_input(z, mask, factor: int):
    factor = max(1, int(factor))
    if factor > 1:
        z, mask, k_list = downsample_time_bkd(z, mask, factor)
        k_list = [int(k) for k in k_list]
    else:
        k_list = [int(k) for k in mask.long().sum(dim=1).tolist()]
    return z, mask, k_list


def load_checkpoint_bundle(checkpoint_path: str, use_ema: bool):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    extra = ckpt["extra_state"]
    module_state = ckpt["inference_modules"] if (use_ema and ckpt.get("inference_modules") is not None) else ckpt["modules"]
    return ckpt, cfg, extra, module_state


def resolve_eval_config(checkpoint_path: str, config_path: str | None, ckpt_cfg):
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f), os.path.abspath(config_path)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    merged_config_path = os.path.join(ckpt_dir, "merged_config.json")
    if os.path.exists(merged_config_path):
        with open(merged_config_path, "r", encoding="utf-8") as f:
            return json.load(f), merged_config_path
    return ckpt_cfg, "<checkpoint-config>"


def load_bigvgan_h(bigvgan_name: str):
    model_name = str(bigvgan_name).split("/")[-1]
    config_candidates = [
        os.path.join(BIGVGAN_ROOT, "configs", f"{model_name}.json"),
    ]
    if "24khz_100band" in model_name:
        config_candidates.append(os.path.join(BIGVGAN_ROOT, "configs", "bigvgan_24khz_100band.json"))
    if "v2_24khz_100band_256x" in model_name:
        config_candidates.append(os.path.join(BIGVGAN_ROOT, "configs", "bigvgan_v2_24khz_100band_256x.json"))

    config_path = None
    for candidate in config_candidates:
        if os.path.exists(candidate):
            config_path = candidate
            break
    if config_path is None:
        raise FileNotFoundError(
            f"Could not resolve local BigVGAN config for {bigvgan_name}. "
            f"Tried: {config_candidates}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return AttrDict(json.load(f))


def wav_to_logmel(wav_path: str, h, device: str):
    wav_np, _ = librosa.load(wav_path, sr=int(h.sampling_rate), mono=True)
    hop_size = int(h.hop_size)
    rem = wav_np.shape[0] % hop_size
    if rem != 0:
        wav_np = wav_np[:-rem]
    wav_np = librosa.util.normalize(wav_np) * 0.95
    wav = torch.tensor(wav_np, dtype=torch.float32, device=device).unsqueeze(0)
    mel = get_mel_spectrogram(wav, h)
    return mel[0].transpose(0, 1).contiguous()


@torch.inference_mode()
def decode_one(
    wav_path: str,
    vf,
    text_ctc_head,
    source_to_canonical,
    canonical_posterior,
    mu_b,
    std_b,
    blank_id: int,
    tok,
    bigvgan_h,
    device: str,
    solver: str,
    decode_mode: str,
    beam_size: int,
    beam_size_token: int,
    kenlm_decoder,
    euler_steps: int,
    heun_steps: int,
    chunk_core: int,
    chunk_ctx: int,
    ctc_subsample_factor: int,
):
    zS_log = wav_to_logmel(wav_path, bigvgan_h, device)
    total_frames = int(zS_log.shape[0])
    zS_full = (zS_log.unsqueeze(0) - mu_b) / std_b

    if chunk_core <= 0:
        mask_full = torch.ones(1, total_frames, device=device, dtype=torch.bool)
        if solver == "euler":
            zT_hat = euler_integrate(
                vf,
                zS_full,
                mask_full,
                steps=int(euler_steps),
                direction=-1,
                cfg_flag_value=0,
                spk_e=None,
                style_e=None,
                text_cond=None,
            )
        else:
            zT_hat = heun_integrate(
                vf,
                zS_full,
                mask_full,
                steps=int(heun_steps),
                direction=-1,
                cfg_scale=1.0,
                spk_e=None,
                style_e=None,
                text_cond=None,
            )
        zC_hat = source_to_canonical(zT_hat, mask_full) if source_to_canonical is not None else zT_hat
        if canonical_posterior is not None:
            zC_hat, _ = canonical_posterior(zC_hat, mask_full)
        zC_hat, mask_ctc, k_ctc = prepare_ctc_input(zC_hat, mask_full, ctc_subsample_factor)
        logits_hat = text_ctc_head(zC_hat, mask_ctc)
        decode_frames = int(k_ctc[0])
    else:
        core = max(32, int(chunk_core))
        ctx = max(0, int(chunk_ctx))
        chunk_zc = []
        pos = 0
        while pos < total_frames:
            core_s = pos
            core_e = min(total_frames, pos + core)
            s = max(0, core_s - ctx)
            e = min(total_frames, core_e + ctx)

            zS_chunk = zS_full[:, s:e]
            mask_chunk = torch.ones(1, e - s, device=device, dtype=torch.bool)
            if solver == "euler":
                zT_chunk = euler_integrate(
                    vf,
                    zS_chunk,
                    mask_chunk,
                    steps=int(euler_steps),
                    direction=-1,
                    cfg_flag_value=0,
                    spk_e=None,
                    style_e=None,
                    text_cond=None,
                )
            else:
                zT_chunk = heun_integrate(
                    vf,
                    zS_chunk,
                    mask_chunk,
                    steps=int(heun_steps),
                    direction=-1,
                    cfg_scale=1.0,
                    spk_e=None,
                    style_e=None,
                    text_cond=None,
                )
            zC_chunk = source_to_canonical(zT_chunk, mask_chunk) if source_to_canonical is not None else zT_chunk
            if canonical_posterior is not None:
                zC_chunk, _ = canonical_posterior(zC_chunk, mask_chunk)
            keep_s = core_s - s
            keep_e = core_e - s
            chunk_zc.append(zC_chunk[:, keep_s:keep_e])
            pos = core_e
        zC_hat = torch.cat(chunk_zc, dim=1)
        mask_hat = torch.ones(1, zC_hat.shape[1], device=device, dtype=torch.bool)
        zC_hat, mask_ctc, k_ctc = prepare_ctc_input(zC_hat, mask_hat, ctc_subsample_factor)
        logits_hat = text_ctc_head(zC_hat, mask_ctc)
        decode_frames = int(k_ctc[0])

    if decode_mode == "kenlm" and kenlm_decoder is not None and kenlm_decoder.enabled:
        return kenlm_decoder.decode(logits_hat), total_frames
    if decode_mode == "beam":
        decoded = ctc_prefix_beam_decode(
            logits_hat,
            blank_id=blank_id,
            beam_size=beam_size,
            beam_size_token=beam_size_token,
        )
    else:
        decoded = ctc_greedy_decode(logits_hat, [decode_frames], blank_id=blank_id)[0]
    hyp_raw = tok.decode(decoded)
    return hyp_raw, total_frames


@torch.inference_mode()
def main():
    args = parse_args()
    device = choose_device(args.device)
    use_ema = args.use_ema == "true"

    _, ckpt_cfg, extra_state, module_state = load_checkpoint_bundle(args.checkpoint, use_ema=use_ema)
    cfg, cfg_source = resolve_eval_config(args.checkpoint, args.config, ckpt_cfg)

    model_cfg = cfg["model"]
    loss_cfg = cfg.get("loss", {})
    infer_cfg = cfg["infer"]

    d_mel = int(extra_state["D_mel"])
    tokenizer_cfg = cfg.get("tokenizer", {"type": "char"})
    if str(tokenizer_cfg.get("type", "char")).lower() in {"char", "character"}:
        tok = CharTokenizer()
        tok.stoi = dict(extra_state["tok_stoi"])
        tok.itos = list(extra_state["tok_itos"])
    else:
        tok = build_tokenizer(tokenizer_cfg)
        if list(tok.itos) != list(extra_state["tok_itos"]):
            raise RuntimeError("Configured BPE vocabulary does not match checkpoint tok_itos")
    blank_id = int(extra_state["BLANK_ID"])
    mu_b = extra_state["mu_g"].float().to(device)
    std_b = extra_state["std_g"].float().to(device)

    use_true_canonical = bool(model_cfg.get("use_true_canonical_latent", False))
    canonical_dim = int(model_cfg.get("canonical_dim", 192))
    ctc_input_dim = canonical_dim if use_true_canonical else d_mel
    vf = build_vf(model_cfg, d_mel).to(device).eval()
    text_ctc_head = build_ctc_head(model_cfg, len(tok.itos), ctc_input_dim).to(device).eval()
    source_to_canonical = None
    if use_true_canonical:
        source_to_canonical = SourceToCanonical(
            in_dim=d_mel,
            c_dim=canonical_dim,
            hidden=int(model_cfg.get("canonical_hidden", 256)),
            dropout=float(model_cfg.get("canonical_dropout", 0.1)),
        ).to(device).eval()
    canonical_match_mode = str(loss_cfg.get("canonical_match_mode", "nll")).lower()
    canonical_posterior = None
    if "canonical_posterior" in module_state and module_state["canonical_posterior"] is not None:
        canonical_posterior = CanonicalPosterior(
            dim=ctc_input_dim,
            hidden=int(model_cfg.get("canonical_post_hidden", model_cfg.get("canonical_hidden", 256))),
            dropout=float(model_cfg.get("canonical_post_dropout", model_cfg.get("canonical_dropout", 0.1))),
            logvar_bias=float(model_cfg.get("canonical_post_logvar_bias", -4.0)),
        ).to(device).eval()
    elif canonical_match_mode == "kl":
        print("[ASR-EVAL] canonical_match_mode=kl but checkpoint has no canonical_posterior; using raw backward endpoint.")

    vf_incompatible = vf.load_state_dict(module_state["vf"], strict=False)
    if vf_incompatible.missing_keys or vf_incompatible.unexpected_keys:
        print(
            "[ASR-EVAL][WARN] VF checkpoint key mismatch "
            f"missing={len(vf_incompatible.missing_keys)} "
            f"unexpected={len(vf_incompatible.unexpected_keys)}"
        )
    text_ctc_head.load_state_dict(module_state["text_ctc_head"], strict=True)
    if source_to_canonical is not None:
        source_to_canonical.load_state_dict(module_state["source_to_canonical"], strict=True)
    if canonical_posterior is not None and "canonical_posterior" in module_state:
        canonical_posterior.load_state_dict(module_state["canonical_posterior"], strict=True)

    bigvgan_h = load_bigvgan_h(str(model_cfg["bigvgan_name"]))

    if args.solver == "auto":
        solver = "euler" if bool(infer_cfg.get("full_asr_use_euler", True)) else "heun"
    else:
        solver = args.solver
    decode_mode = str(args.decode_mode)
    if decode_mode == "kenlm" and str(tokenizer_cfg.get("type", "char")).lower() not in {"char", "character"}:
        print("[ASR-EVAL][WARN] KenLM decoder currently expects character tokens; using greedy for BPE")
        decode_mode = "greedy"
    beam_size = int(args.beam_size)
    beam_size_token = int(args.beam_size_token)
    kenlm_decoder = None
    if decode_mode == "kenlm":
        kenlm_lexicon_corpus = args.kenlm_lexicon_corpus or default_kenlm_lexicon_corpus(cfg)
        kenlm_decoder = OptionalKenLMCTCDecoder(
            tok.itos,
            blank_id,
            KenLMCTCDecoderConfig(
                preset=str(args.kenlm_preset),
                lexicon=args.kenlm_lexicon,
                lm=args.kenlm_lm,
                lexicon_corpus_manifest=kenlm_lexicon_corpus,
                beam_size=int(args.beam_size),
                beam_threshold=float(args.kenlm_beam_threshold),
                beam_size_token=int(args.beam_size_token),
                lm_weight=float(args.kenlm_lm_weight),
                word_score=float(args.kenlm_word_score),
                allow_fallback=not bool(args.kenlm_no_fallback),
            ),
        )
        if kenlm_decoder.enabled:
            print(
                "[ASR-EVAL] KenLM enabled "
                f"preset={kenlm_decoder.cfg.preset} beam={kenlm_decoder.cfg.beam_size} "
                f"lm_weight={kenlm_decoder.cfg.lm_weight} word_score={kenlm_decoder.cfg.word_score} "
                f"lexicon={kenlm_decoder.lexicon_path}"
            )
        else:
            print(f"[ASR-EVAL][WARN] KenLM unavailable; fallback greedy. error={kenlm_decoder.error}")
            decode_mode = "greedy"
    euler_steps = int(args.euler_steps if args.euler_steps is not None else 5)
    heun_steps = int(args.heun_steps or infer_cfg.get("ode_steps_eval", 8))
    chunk_core = int(args.chunk_core if args.chunk_core is not None else infer_cfg.get("full_asr_chunk_core", 256))
    chunk_ctx = int(args.chunk_ctx if args.chunk_ctx is not None else infer_cfg.get("full_asr_chunk_ctx", 96))
    ctc_subsample_factor = int(model_cfg.get("ctc_subsample_factor", 1))
    ctc_subsample_apply_to = str(model_cfg.get("ctc_subsample_apply_to", "hat")).lower()
    if ctc_subsample_apply_to == "none":
        ctc_subsample_factor = 1
    ctc_subsample_factor = max(1, ctc_subsample_factor)
    print(f"[ASR-EVAL] ctc_subsample_factor={ctc_subsample_factor} apply_to={ctc_subsample_apply_to}")

    rows = read_jsonl_rows(args.manifest, max_rows=args.max_rows)
    assert len(rows) > 0, f"Empty manifest: {args.manifest}"

    pred_rows = []
    wers_basic = []
    wers_norm = []
    cers_norm = []

    print(
        f"[ASR-EVAL] rows={len(rows)} device={device} use_ema={use_ema} "
        f"solver={solver} decode={decode_mode} beam_size={beam_size} beam_size_token={beam_size_token} "
        f"euler_steps={euler_steps} heun_steps={heun_steps} "
        f"chunk_core={chunk_core} chunk_ctx={chunk_ctx}"
    )
    print(f"[CKPT] {os.path.abspath(args.checkpoint)}")
    print(f"[CONFIG] {cfg_source}")
    print(f"[MANIFEST] {os.path.abspath(args.manifest)}")

    for i, item in enumerate(rows, start=1):
        wav_path = item["wav"]
        gt_raw = str(item.get("text_norm", item.get("text_raw", ""))).strip()

        try:
            hyp_raw, frames = decode_one(
                wav_path=wav_path,
                vf=vf,
                text_ctc_head=text_ctc_head,
                source_to_canonical=source_to_canonical,
                canonical_posterior=canonical_posterior,
                mu_b=mu_b,
                std_b=std_b,
                blank_id=blank_id,
                tok=tok,
                bigvgan_h=bigvgan_h,
                device=device,
                solver=solver,
                decode_mode=decode_mode,
                beam_size=beam_size,
                beam_size_token=beam_size_token,
                kenlm_decoder=kenlm_decoder,
                euler_steps=euler_steps,
                heun_steps=heun_steps,
                chunk_core=chunk_core,
                chunk_ctx=chunk_ctx,
                ctc_subsample_factor=ctc_subsample_factor,
            )
            error = None
        except Exception as exc:
            hyp_raw = ""
            frames = -1
            error = repr(exc)

        hyp_basic = normalize_text_basic(hyp_raw)
        gt_basic = normalize_text_basic(gt_raw)
        hyp_norm = normalize_asr_text(hyp_raw)
        gt_norm = normalize_asr_text(gt_raw)
        wer_basic = word_error_rate_text(gt_raw, hyp_raw)
        wer_norm = word_error_rate_normalized(gt_raw, hyp_raw)
        cer_norm = char_error_rate_normalized(gt_raw, hyp_raw)

        wers_basic.append(wer_basic)
        wers_norm.append(wer_norm)
        cers_norm.append(cer_norm)

        pred_rows.append(
            {
                "wav": wav_path,
                "speaker": item.get("speaker"),
                "utt_id": item.get("utt_id"),
                "subset": item.get("subset"),
                "frames": frames,
                "gt_raw": gt_raw,
                "hyp_raw": hyp_raw,
                "gt_basic": gt_basic,
                "hyp_basic": hyp_basic,
                "gt_norm_asr": gt_norm,
                "hyp_norm_asr": hyp_norm,
                "wer_basic": wer_basic,
                "wer_norm_asr": wer_norm,
                "cer_norm_asr": cer_norm,
                "error": error,
            }
        )

        if args.print_every > 0 and (i % int(args.print_every) == 0 or i == len(rows)):
            mean_wer_basic = sum(wers_basic) / len(wers_basic)
            mean_wer_norm = sum(wers_norm) / len(wers_norm)
            mean_cer_norm = sum(cers_norm) / len(cers_norm)
            print(
                f"[ASR-EVAL] {i}/{len(rows)} "
                f"WER_basic={mean_wer_basic:.4f} "
                f"WER_norm_asr={mean_wer_norm:.4f} "
                f"CER_norm_asr={mean_cer_norm:.4f}"
            )
            print(f"GT : {gt_raw}")
            print(f"HYP: {hyp_raw}")

    summary = {
        "rows": len(pred_rows),
        "checkpoint": os.path.abspath(args.checkpoint),
        "manifest": os.path.abspath(args.manifest),
        "device": device,
        "use_ema": use_ema,
        "solver": solver,
        "decode_mode": decode_mode,
        "beam_size": beam_size,
        "beam_size_token": beam_size_token,
        "kenlm_preset": str(args.kenlm_preset),
        "kenlm_lm_weight": float(args.kenlm_lm_weight),
        "kenlm_word_score": float(args.kenlm_word_score),
        "euler_steps": euler_steps,
        "heun_steps": heun_steps,
        "chunk_core": chunk_core,
        "chunk_ctx": chunk_ctx,
        "wer_basic": (sum(wers_basic) / len(wers_basic)) if wers_basic else 0.0,
        "wer_norm_asr": (sum(wers_norm) / len(wers_norm)) if wers_norm else 0.0,
        "cer_norm_asr": (sum(cers_norm) / len(cers_norm)) if cers_norm else 0.0,
        "failures": int(sum(1 for row in pred_rows if row["error"] is not None)),
    }

    print("\n===== FINAL =====")
    print(f"N             = {summary['rows']}")
    print(f"WER_basic     = {summary['wer_basic']:.4f}")
    print(f"WER_norm_asr  = {summary['wer_norm_asr']:.4f}")
    print(f"CER_norm_asr  = {summary['cer_norm_asr']:.4f}")
    print(f"failures      = {summary['failures']}")

    if args.output_jsonl:
        out_path = os.path.abspath(args.output_jsonl)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WRITE] {out_path}")

    if args.summary_json:
        out_path = os.path.abspath(args.summary_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[WRITE] {out_path}")


if __name__ == "__main__":
    main()
