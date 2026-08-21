#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
BIGVGAN_ROOT = os.path.join(REPO_ROOT, "BigVGAN")
if BIGVGAN_ROOT not in sys.path:
    sys.path.insert(0, BIGVGAN_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from biflow.alignment import (
    downsample_time_bkd,
    durations_to_int_and_fixsum,
    gaussian_mas_score,
    monotonic_alignment_posterior,
    monotonic_alignment_search,
)
from biflow.checkpointing import (
    MultiModuleEMA,
    load_training_checkpoint,
    move_optimizer_state_to_device,
    resolve_resume_path,
    save_training_checkpoint,
)
from biflow.config import apply_overrides, load_config
from biflow.data import CutDataset, collate_cut_batch
from biflow.encoders import FrozenHubertSSLTeacher, FrozenSpeechT5TextEncoder
from biflow.losses import (
    amp_once,
    batch_pair_ratios,
    hidden_ssl_cosine_loss,
    masked_gaussian_nll,
    masked_l1,
    masked_mean_std,
    masked_mse,
    mel_range_penalty,
    mrstft_loss,
    summarize_ratios,
    time_delta_bkd,
    vf_lip_fd_ratio,
)
from biflow.models import (
    ADMASpeechAlignMLP,
    AttentionCTCDecoder,
    BaselineCTCHead,
    CanonicalPosterior,
    CanonicalToSource,
    CanonicalTextEncoder,
    DiTVectorField,
    FastSpeech2DurationPredictor,
    FrameCTCConvHead,
    LengthPredictor,
    ResidualAdapter,
    SpeakerConditioner,
    SpeakerTable,
    SourceStatsConditioner,
    SourceToCanonical,
    TextCondRefiner1xResidualPostNet,
    TextPriorHead,
    TrainableTokenTextEncoder,
    TTSStylePosterior,
    TTSStylePairPosterior,
    TTSStyleCanonicalPrior,
    TTSStylePrior,
    TTSStyleToSource,
    ZipformerCTCHead,
    euler_integrate,
    euler_integrate_grad,
    heun_integrate,
)
from biflow.samplers import LengthBucketBatchSampler
from biflow.ctc_decode import KenLMCTCDecoderConfig, OptionalKenLMCTCDecoder
from biflow.tokenizer import build_tokenizer, ctc_greedy_decode
from biflow.utils import (
    TeeIO,
    append_jsonl,
    extract_speaker_id_from_path,
    normalize_text_basic,
    read_jsonl_rows,
    save_wav,
    set_seed,
    word_error_rate_text,
)


def parse_args():
    p = argparse.ArgumentParser(description="Single-VF CUT-manifest training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--cut-manifest", type=str, default=None)
    p.add_argument("--aligned-manifest", type=str, default=None)
    p.add_argument("--max-cut-rows", type=int, default=None)
    p.add_argument("--target-spks", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--lr-all", type=float, default=None)
    p.add_argument("--lr-warmup-steps", type=int, default=None)
    p.add_argument("--ckpt-dir", type=str, default=None)
    p.add_argument("--demo-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-every-steps", type=int, default=None)
    p.add_argument("--keep-last-k", type=int, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--demo-every", type=int, default=None)
    p.add_argument("--load-bigvgan-model", choices=["true", "false"], default=None)
    p.add_argument("--use-ema", choices=["true", "false"], default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--compile-enable", choices=["true", "false"], default=None)
    p.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default=None)
    p.add_argument("--gpu-mel-cache", choices=["true", "false"], default=None)
    p.add_argument("--gpu-text-cache", choices=["true", "false"], default=None)
    p.add_argument("--gpu-mel-preload", choices=["true", "false"], default=None)
    p.add_argument("--gpu-text-preload", choices=["true", "false"], default=None)
    p.add_argument("--gpu-mel-cache-limit-gib", type=float, default=None)
    p.add_argument("--gpu-text-cache-limit-gib", type=float, default=None)
    p.add_argument("--gpu-cache-reserve-gib", type=float, default=None)
    p.add_argument("--speaker-emb-path", type=str, default=None)
    return p.parse_args()


def _load_tensor_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha1_key(text: str):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def resolve_cache_dtype(name, default=torch.float32):
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return default
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = str(name).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported cache dtype: {name}")
    return mapping[key]


def dtype_nbytes(dtype: torch.dtype):
    return torch.empty((), dtype=dtype).element_size()


def tensor_nbytes(tensor: torch.Tensor):
    return int(tensor.numel()) * int(tensor.element_size())


def gib_to_bytes(value):
    if value is None:
        return None
    return int(float(value) * (1024 ** 3))


def bytes_to_gib(value):
    return float(value) / float(1024 ** 3)


def set_runtime_flags(device: str, cfg):
    runtime_cfg = cfg["runtime"]
    if device != "cuda":
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(runtime_cfg["allow_tf32"])
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = bool(runtime_cfg["allow_tf32"])
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = bool(runtime_cfg["cudnn_benchmark"])
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision(str(runtime_cfg["matmul_precision"]))
    except Exception:
        pass


def maybe_compile(module, enable, mode, dynamic, name):
    if (module is None) or (not enable):
        return module
    if not hasattr(torch, "compile"):
        print(f"[COMPILE] torch.compile unavailable, skip {name}")
        return module
    try:
        compiled = torch.compile(module, mode=mode, dynamic=dynamic)
        print(f"[COMPILE] enabled for {name}")
        return compiled
    except Exception as exc:
        print(f"[COMPILE] failed for {name}: {exc}")
        return module


def select_target_speakers(rows, target_spks, top_k_spk):
    spk_count = {}
    for row in rows:
        spk = str(row.get("speaker", ""))
        if spk:
            spk_count[spk] = spk_count.get(spk, 0) + 1
    assert len(spk_count) > 0, "No speaker field found in cut manifest."
    if target_spks is not None:
        return list(target_spks)
    spk_sorted = sorted(spk_count.items(), key=lambda kv: kv[1], reverse=True)
    return [spk for spk, _ in spk_sorted[:top_k_spk]]


def load_speaker_embedding_bank(path, spk_list, normalize=True, missing="error"):
    if not path:
        raise ValueError("model.speaker_emb_path is required for pretrained speaker conditioning")
    path = os.path.abspath(str(path))
    obj = torch.load(path, map_location="cpu")
    speaker_ids = None
    emb_obj = obj
    if isinstance(obj, dict):
        speaker_ids = (
            obj.get("speaker_ids")
            or obj.get("spk_list")
            or obj.get("speakers")
        )
        for key in ("embeddings", "speaker_embeddings", "spk2emb", "spk_embeddings"):
            if key in obj:
                emb_obj = obj[key]
                break

    if torch.is_tensor(emb_obj):
        if speaker_ids is None:
            raise ValueError(
                f"{path} stores a tensor speaker bank but has no speaker_ids/spk_list"
            )
        if len(speaker_ids) != int(emb_obj.shape[0]):
            raise ValueError(
                f"{path} speaker_ids length={len(speaker_ids)} does not match "
                f"bank rows={int(emb_obj.shape[0])}"
            )
        emb_map = {
            str(spk): emb_obj[idx].detach().cpu().float()
            for idx, spk in enumerate(speaker_ids)
        }
    elif isinstance(emb_obj, dict):
        emb_map = {
            str(spk): torch.as_tensor(value).detach().cpu().float()
            for spk, value in emb_obj.items()
        }
    else:
        raise ValueError(
            f"Unsupported speaker embedding bank format at {path}: {type(emb_obj).__name__}"
        )

    missing = str(missing).lower()
    if missing not in {"error", "zero"}:
        raise ValueError("model.speaker_emb_missing must be 'error' or 'zero'")

    dims = sorted({int(value.numel()) for value in emb_map.values()})
    if not dims:
        raise ValueError(f"No speaker embeddings found in {path}")
    if len(dims) != 1:
        raise ValueError(f"Speaker embeddings in {path} have mixed dims: {dims}")
    dim = dims[0]

    rows = []
    missing_spks = []
    for spk in spk_list:
        value = emb_map.get(str(spk))
        if value is None:
            missing_spks.append(str(spk))
            value = torch.zeros(dim, dtype=torch.float32)
        rows.append(value.reshape(-1))

    if missing_spks and missing == "error":
        preview = ", ".join(missing_spks[:10])
        more = "" if len(missing_spks) <= 10 else f", ... (+{len(missing_spks) - 10})"
        raise ValueError(
            f"Missing pretrained speaker embeddings for {len(missing_spks)} selected speakers: "
            f"{preview}{more}. Build the bank with scripts/precompute_ecapa_speaker_bank.py "
            "or set model.speaker_emb_missing='zero'."
        )

    bank = torch.stack(rows, dim=0).float()
    if bool(normalize):
        bank = F.normalize(bank, p=2, dim=-1)
    return bank, missing_spks, path


class ReferenceSpeakerEncoder:
    """
    Lazily extracts speaker embeddings from reference audio for zero-shot TTS.

    The extracted raw embedding is projected by SpeakerConditioner, so this
    class deliberately does not own trainable parameters.
    """

    def __init__(
        self,
        *,
        model_name: str,
        savedir: str | None,
        cache_dir: str | None,
        device: str,
        sampling_rate: int,
        max_sec: float = 12.0,
        l2_normalize: bool = True,
    ):
        self.model_name = str(model_name)
        self.savedir = os.path.abspath(os.path.expanduser(str(savedir))) if savedir else None
        self.cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir))) if cache_dir else None
        self.device = str(device)
        self.sampling_rate = int(sampling_rate)
        self.max_sec = float(max_sec)
        self.l2_normalize = bool(l2_normalize)
        self.classifier = None
        self.mem_cache = {}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _load_classifier(self):
        if self.classifier is not None:
            return self.classifier
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            from speechbrain.pretrained import EncoderClassifier
        savedir = self.savedir
        if savedir is None:
            safe_name = self.model_name.replace("/", "_").replace(":", "_")
            savedir = os.path.join(REPO_ROOT, "ckpts", "speechbrain", safe_name)
        self.classifier = EncoderClassifier.from_hparams(
            source=self.model_name,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        self.classifier.eval()
        return self.classifier

    def _cache_path(self, wav_path: str):
        if not self.cache_dir:
            return None
        key = hashlib.sha1(os.path.abspath(str(wav_path)).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.pt")

    def _encode_one(self, wav_path: str):
        wav_path = os.path.abspath(str(wav_path))
        if wav_path in self.mem_cache:
            return self.mem_cache[wav_path]
        cache_path = self._cache_path(wav_path)
        if cache_path and os.path.exists(cache_path):
            try:
                emb = torch.load(cache_path, map_location="cpu", weights_only=True).float().reshape(-1)
            except TypeError:
                emb = torch.load(cache_path, map_location="cpu").float().reshape(-1)
            self.mem_cache[wav_path] = emb
            return emb

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Zero-shot reference wav not found: {wav_path}")
        import librosa

        wav_np, _ = librosa.load(wav_path, sr=self.sampling_rate, mono=True)
        if self.max_sec > 0:
            max_samples = int(round(self.max_sec * self.sampling_rate))
            if wav_np.shape[0] > max_samples:
                wav_np = wav_np[:max_samples]
        wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(self.device)
        classifier = self._load_classifier()
        with torch.no_grad():
            emb = classifier.encode_batch(wav).detach().float().cpu().reshape(-1)
        if self.l2_normalize:
            emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)
        if cache_path:
            tmp_path = f"{cache_path}.tmp"
            torch.save(emb, tmp_path)
            os.replace(tmp_path, cache_path)
        self.mem_cache[wav_path] = emb
        return emb

    def encode_paths(self, wav_paths, *, device=None, dtype=None):
        if len(wav_paths) == 0:
            raise ValueError("encode_paths received an empty path list")
        embs = [self._encode_one(path) for path in wav_paths]
        out = torch.stack(embs, dim=0)
        target_device = device if device is not None else self.device
        out = out.to(device=target_device)
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    set_seed(int(cfg["train"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_runtime_flags(device, cfg)
    use_amp = (device == "cuda")
    amp_device = "cuda" if use_amp else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    path_cfg = cfg["paths"]
    processed_unified_dir = path_cfg.get("processed_unified_dir")
    use_processed_unified = bool(path_cfg.get("use_processed_unified", False) or processed_unified_dir)
    if processed_unified_dir:
        processed_unified_dir = os.path.abspath(str(processed_unified_dir))

    def _processed_path(key, default_name):
        value = path_cfg.get(key)
        if value:
            return str(value)
        if processed_unified_dir:
            return os.path.join(processed_unified_dir, default_name)
        return None

    full_manifest_clean = _processed_path("full_manifest_clean", "full_manifest_clean.jsonl")
    fm_core_context_manifest = _processed_path("fm_core_context_manifest", "fm_core_context_cuts.jsonl")
    asr_full_chunks_manifest = _processed_path("asr_full_chunks_manifest", "asr_full_chunks.jsonl")
    tts_duration_manifest = _processed_path("tts_duration_manifest", "tts_duration_full.jsonl")

    cut_manifest = cfg["paths"]["cut_manifest"]
    aligned_manifest = cfg["paths"]["aligned_manifest"]
    demo_aligned_manifest = cfg["paths"].get("demo_aligned_manifest", aligned_manifest)
    max_cut_rows = cfg["paths"]["max_cut_rows"]

    cache_cfg = cfg["cache"]
    speech_backend = str(cache_cfg.get("speech_backend", "mel")).lower()
    if speech_backend not in {"mel", "svae"}:
        raise ValueError(f"Unsupported cache.speech_backend={speech_backend!r}; expected 'mel' or 'svae'")
    mel_cache_dir = cache_cfg.get("mel_dir")
    svae_latent_dir = cache_cfg.get("svae_latent_dir")
    svae_root = str(cache_cfg.get("semantic_vae_root", "/work/dankker0900/Semantic-VAE"))
    svae_ckpt = str(cache_cfg.get("semantic_vae_ckpt", "/work/dankker0900/Semantic-VAE/ckpts/semantic_vae_1000k"))
    svae_use_ema = bool(cache_cfg.get("semantic_vae_use_ema", True))
    svae_dim = int(cache_cfg.get("svae_dim", 64))
    svae_sample_rate = int(cache_cfg.get("svae_sample_rate", 16000))
    svae_hop_size = int(cache_cfg.get("svae_hop_size", 400))
    if speech_backend == "svae" and not mel_cache_dir:
        mel_cache_dir = svae_latent_dir or "svae_latents"
    speecht5_cache_dir = cache_cfg.get("speecht5_dir")
    wav_cache_max_items = int(cache_cfg["wav_cache_max_items"])
    mel_cache_max_items = int(cache_cfg["mel_cache_max_items"])
    text_hidden_cache_max_items = int(cache_cfg["text_hidden_cache_max_items"])
    stats_max_unique_wavs = cache_cfg["stats_max_unique_wavs"]
    mel_stats_mode = str(cache_cfg.get("mel_stats_mode", "per_bin")).lower()
    if mel_stats_mode not in {"per_bin", "scalar"}:
        raise ValueError(f"Unsupported cache.mel_stats_mode={mel_stats_mode!r}; expected 'per_bin' or 'scalar'")
    gpu_mel_cache = bool(cache_cfg.get("gpu_mel_cache", False))
    gpu_text_cache = bool(cache_cfg.get("gpu_text_cache", False))
    gpu_mel_preload = bool(cache_cfg.get("gpu_mel_preload", False))
    gpu_text_preload = bool(cache_cfg.get("gpu_text_preload", False))
    gpu_mel_cache = gpu_mel_cache or gpu_mel_preload
    gpu_text_cache = gpu_text_cache or gpu_text_preload
    gpu_mel_cache_dtype = resolve_cache_dtype(cache_cfg.get("gpu_mel_cache_dtype", "float32"), default=torch.float32)
    gpu_text_cache_dtype = resolve_cache_dtype(cache_cfg.get("gpu_text_cache_dtype", "bfloat16"), default=torch.bfloat16)
    gpu_mel_cache_limit_bytes = gib_to_bytes(cache_cfg.get("gpu_mel_cache_limit_gib"))
    gpu_text_cache_limit_bytes = gib_to_bytes(cache_cfg.get("gpu_text_cache_limit_gib"))
    gpu_cache_reserve_bytes = gib_to_bytes(cache_cfg.get("gpu_cache_reserve_gib", 64.0))
    if device != "cuda" and (gpu_mel_cache or gpu_text_cache):
        print("[CACHE] GPU resident cache requested without CUDA; disabling")
        gpu_mel_cache = False
        gpu_text_cache = False
        gpu_mel_preload = False
        gpu_text_preload = False

    data_cfg = cfg["data"]
    batch_size = int(data_cfg["batch_size"])
    batch_size_tts = int(data_cfg["batch_size_tts"] or batch_size)
    batch_size_asr = int(data_cfg["batch_size_asr"] or batch_size)

    def _optional_positive_int(value):
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    max_frames_per_batch = _optional_positive_int(data_cfg.get("max_frames_per_batch"))
    max_frames_per_batch_tts = _optional_positive_int(
        data_cfg.get("max_frames_per_batch_tts", max_frames_per_batch)
    )
    max_frames_per_batch_asr = _optional_positive_int(
        data_cfg.get("max_frames_per_batch_asr", max_frames_per_batch)
    )
    max_utts_per_batch = _optional_positive_int(data_cfg.get("max_utts_per_batch"))
    max_utts_per_batch_tts = _optional_positive_int(
        data_cfg.get("max_utts_per_batch_tts", max_utts_per_batch or batch_size_tts)
    )
    max_utts_per_batch_asr = _optional_positive_int(
        data_cfg.get("max_utts_per_batch_asr", max_utts_per_batch or batch_size_asr)
    )
    ds_factor = int(data_cfg["ds_factor"])
    ds_align = int(data_cfg["ds_align"])
    top_k_spk = int(data_cfg["top_k_spk"])
    target_spks_cfg = data_cfg["target_spks"]
    force_text_normalize = bool(data_cfg.get("force_text_normalize", True))
    use_dataloader = bool(data_cfg.get("use_dataloader", False))
    loader_num_workers = int(data_cfg.get("loader_num_workers", 4))
    loader_pin_memory = bool(data_cfg.get("loader_pin_memory", True))
    loader_persistent_workers = bool(data_cfg.get("loader_persistent_workers", True))
    loader_prefetch_factor = int(data_cfg.get("loader_prefetch_factor", 2))
    dataset_mel_worker_cache_max_items = int(data_cfg.get("dataset_mel_worker_cache_max_items", 128))
    enable_length_bucket = bool(data_cfg.get("enable_length_bucket", False))
    num_length_buckets = int(data_cfg.get("num_length_buckets", 12))
    core_loss_only = bool(data_cfg.get("core_loss_only", use_processed_unified))
    train_unit = str(data_cfg.get("train_unit", "cut")).lower()
    if train_unit not in {"cut", "utterance", "full", "full_utterance"}:
        raise ValueError(
            f"Unsupported data.train_unit={train_unit!r}; expected 'cut' or 'utterance'"
        )
    use_utterance_training = train_unit in {"utterance", "full", "full_utterance"}
    pad_to_multiple = int(data_cfg.get("pad_to_multiple", 1) or 1)
    pad_to_multiple = max(1, pad_to_multiple)

    model_cfg = cfg["model"]
    tokenizer_cfg = cfg.get("tokenizer", {"type": "char"})
    E_spk = int(model_cfg["E_spk"])
    spk_scale = float(model_cfg["spk_scale"])
    spk_drop_rate = float(model_cfg["spk_drop_rate"])
    speaker_cond_type = str(model_cfg.get("speaker_cond_type", "table")).lower()
    speaker_emb_path = model_cfg.get("speaker_emb_path")
    speaker_emb_l2_normalize = bool(model_cfg.get("speaker_emb_l2_normalize", True))
    speaker_emb_missing = str(model_cfg.get("speaker_emb_missing", "error")).lower()
    speaker_emb_trainable = bool(model_cfg.get("speaker_emb_trainable", False))
    speaker_delta_scale = float(model_cfg.get("speaker_delta_scale", 0.1))
    speaker_cond_layernorm = bool(model_cfg.get("speaker_cond_layernorm", True))
    zero_shot_cfg = model_cfg.get("zero_shot", {})
    zero_shot_enable = bool(zero_shot_cfg.get("enable", False))
    zero_shot_ref_model = str(
        zero_shot_cfg.get("ref_speaker_emb_model", "speechbrain/spkrec-ecapa-voxceleb")
    )
    zero_shot_ref_savedir = zero_shot_cfg.get("ref_speaker_emb_savedir")
    zero_shot_ref_cache_dir = zero_shot_cfg.get("ref_emb_cache_dir")
    zero_shot_ref_max_sec = float(zero_shot_cfg.get("ref_max_sec", 12.0))
    zero_shot_ref_l2_normalize = bool(zero_shot_cfg.get("ref_l2_normalize", speaker_emb_l2_normalize))
    zero_shot_train_ref_source = str(zero_shot_cfg.get("train_ref_source", "same_speaker")).lower()
    if zero_shot_train_ref_source not in {"same_speaker", "self"}:
        raise ValueError("model.zero_shot.train_ref_source must be 'same_speaker' or 'self'")
    zero_shot_asr_ref_source = str(zero_shot_cfg.get("asr_ref_source", "self")).lower()
    if zero_shot_asr_ref_source not in {"self", "same_speaker"}:
        raise ValueError("model.zero_shot.asr_ref_source must be 'self' or 'same_speaker'")
    vf_use_speaker_cond = bool(model_cfg.get("vf_use_speaker_cond", True))
    asr_vf_use_speaker_cond_cfg = model_cfg.get("asr_vf_use_speaker_cond", None)
    asr_vf_use_speaker_cond = (
        bool(vf_use_speaker_cond)
        if asr_vf_use_speaker_cond_cfg is None
        else bool(asr_vf_use_speaker_cond_cfg)
    )
    asr_use_spk_cond = bool(model_cfg.get("asr_use_spk_cond", False))
    asr_spk_scale = float(model_cfg.get("asr_spk_scale", 1.0))
    asr_spk_unknown = str(model_cfg.get("asr_spk_unknown", "zero")).lower()
    if asr_spk_unknown not in {"zero", "error"}:
        raise ValueError("model.asr_spk_unknown must be 'zero' or 'error'")
    asr_use_style_cond = bool(model_cfg.get("asr_use_style_cond", False))
    asr_style_use_mean = bool(model_cfg.get("asr_style_use_mean", True))
    asr_style_temp = float(model_cfg.get("asr_style_temp", 0.0))
    asr_style_detach = bool(model_cfg.get("asr_style_detach", True))
    use_tts_source_cond = bool(model_cfg.get("use_tts_source_cond", False))
    tts_source_cond_hidden = int(model_cfg.get("tts_source_cond_hidden", 128))
    tts_source_cond_scale = float(model_cfg.get("tts_source_cond_scale", 1.0))
    use_tts_style_latent = bool(model_cfg.get("use_tts_style_latent", False))
    tts_style_dim = int(model_cfg.get("tts_style_dim", 64))
    tts_style_hidden = int(model_cfg.get("tts_style_hidden", 256))
    tts_style_dropout = float(model_cfg.get("tts_style_dropout", 0.1))
    use_tts_style_pair_posterior = bool(model_cfg.get("use_tts_style_pair_posterior", False))
    tts_style_source_scale = float(model_cfg.get("tts_style_source_scale", 1.0))
    tts_style_post_mode = str(model_cfg.get("tts_style_post_mode", "speech")).lower()
    if tts_style_post_mode not in {"speech", "path"}:
        raise ValueError(
            f"Unsupported model.tts_style_post_mode={tts_style_post_mode!r}; "
            "expected 'speech' or 'path'"
        )
    if use_tts_style_pair_posterior and tts_style_post_mode != "speech":
        raise ValueError("model.use_tts_style_pair_posterior=true requires model.tts_style_post_mode='speech'")
    tts_style_prior_type = str(model_cfg.get("tts_style_prior_type", "standard_normal")).lower()
    if tts_style_prior_type not in {"standard_normal", "speaker", "canonical_speaker"}:
        raise ValueError(
            f"Unsupported model.tts_style_prior_type={tts_style_prior_type!r}; "
            "expected 'standard_normal', 'speaker', or 'canonical_speaker'"
        )
    tts_style_prior_canonical_detach = bool(model_cfg.get("tts_style_prior_canonical_detach", True))
    tts_style_prior_logvar_bias_cfg = model_cfg.get("tts_style_prior_logvar_bias", None)
    if tts_style_prior_logvar_bias_cfg is None:
        tts_style_prior_logvar_bias = 0.0 if tts_style_prior_type == "canonical_speaker" else -4.0
    else:
        tts_style_prior_logvar_bias = float(tts_style_prior_logvar_bias_cfg)
    tts_style_into_source = bool(model_cfg.get("tts_style_into_source", False))
    if use_tts_style_latent and tts_style_post_mode == "path" and tts_style_into_source:
        raise ValueError(
            "model.tts_style_post_mode='path' is incompatible with "
            "model.tts_style_into_source=true because z_t depends on the style source bias."
        )
    use_true_canonical_latent = bool(model_cfg.get("use_true_canonical_latent", False))
    canonical_dim = int(model_cfg.get("canonical_dim", 192))
    canonical_hidden = int(model_cfg.get("canonical_hidden", 256))
    canonical_dropout = float(model_cfg.get("canonical_dropout", 0.1))
    canonical_post_hidden = int(model_cfg.get("canonical_post_hidden", canonical_hidden))
    canonical_post_dropout = float(model_cfg.get("canonical_post_dropout", canonical_dropout))
    canonical_post_logvar_bias = float(model_cfg.get("canonical_post_logvar_bias", -4.0))
    use_vf_canonical_text_cond = bool(model_cfg.get("use_vf_canonical_text_cond", True))
    vf_condition_injection = str(model_cfg.get("vf_condition_injection", "legacy")).lower()
    if vf_condition_injection not in {"legacy", "separate_adaln"}:
        raise ValueError(
            f"Unsupported model.vf_condition_injection={vf_condition_injection!r}; "
            "expected 'legacy' or 'separate_adaln'."
        )
    st5_layer_idx = int(model_cfg["st5_layer_idx"])
    text_encoder_type = str(model_cfg.get("text_encoder_type", "speecht5")).lower()
    if text_encoder_type not in {"speecht5", "trainable", "trainable_token"}:
        raise ValueError(
            f"Unsupported model.text_encoder_type={text_encoder_type!r}; "
            "expected 'speecht5' or 'trainable'"
        )
    text_encoder_dim = int(model_cfg.get("text_encoder_dim", 384))
    text_encoder_layers = int(model_cfg.get("text_encoder_layers", 6))
    text_encoder_heads = int(model_cfg.get("text_encoder_heads", 6))
    text_encoder_ff_mult = int(model_cfg.get("text_encoder_ff_mult", 4))
    text_encoder_conv_ksize = int(model_cfg.get("text_encoder_conv_ksize", 5))
    text_encoder_dropout = float(model_cfg.get("text_encoder_dropout", 0.1))
    text_encoder_max_len = int(model_cfg.get("text_encoder_max_len", 1024))
    use_adapter = bool(model_cfg["use_adapter"])
    adapter_type = str(model_cfg.get("adapter_type", "residual")).lower()
    adapter_bottleneck = int(model_cfg["adapter_bottleneck"])
    adapter_dropout = float(model_cfg["adapter_dropout"])
    canonical_text_layers = int(model_cfg.get("canonical_text_layers", 4))
    canonical_text_heads = int(model_cfg.get("canonical_text_heads", 8))
    canonical_text_ff_mult = int(model_cfg.get("canonical_text_ff_mult", 4))
    canonical_text_conv_ksize = int(model_cfg.get("canonical_text_conv_ksize", 5))
    canonical_text_residual_scale = float(model_cfg.get("canonical_text_residual_scale", 1.0))
    ctc_blank_text_layers = int(model_cfg.get("ctc_blank_text_layers", canonical_text_layers))
    ctc_blank_text_heads = int(model_cfg.get("ctc_blank_text_heads", canonical_text_heads))
    ctc_blank_text_ff_mult = int(model_cfg.get("ctc_blank_text_ff_mult", canonical_text_ff_mult))
    ctc_blank_text_conv_ksize = int(model_cfg.get("ctc_blank_text_conv_ksize", canonical_text_conv_ksize))
    ctc_blank_text_dropout = float(model_cfg.get("ctc_blank_text_dropout", adapter_dropout))
    ctc_blank_text_residual_scale = float(model_cfg.get("ctc_blank_text_residual_scale", canonical_text_residual_scale))
    bigvgan_name = str(model_cfg.get("bigvgan_name", ""))
    ctc_head_type = str(model_cfg.get("ctc_head_type", "baseline")).lower()
    if ctc_head_type == "framectcconvhead":
        ctc_head_type = "conv"
    valid_ctc_head_types = {"baseline", "conv", "zipformer"}
    if ctc_head_type not in valid_ctc_head_types:
        raise ValueError(
            f"Unsupported model.ctc_head_type={ctc_head_type!r}; "
            f"expected one of {sorted(valid_ctc_head_types)}"
        )
    ctc_subsample_factor = int(model_cfg.get("ctc_subsample_factor", 1))
    ctc_subsample_factor = max(1, ctc_subsample_factor)
    ctc_subsample_apply_to = str(model_cfg.get("ctc_subsample_apply_to", "hat")).lower()
    if ctc_subsample_apply_to not in {"hat", "both", "none"}:
        raise ValueError(f"Unsupported model.ctc_subsample_apply_to={ctc_subsample_apply_to!r}; expected 'hat', 'both', or 'none'")
    att_decoder_hidden = int(model_cfg.get("att_decoder_hidden", model_cfg.get("ctc_hidden", 384)))
    att_decoder_layers = int(model_cfg.get("att_decoder_layers", 4))
    att_decoder_heads = int(model_cfg.get("att_decoder_heads", model_cfg.get("ctc_heads", 6)))
    att_decoder_ff_mult = float(model_cfg.get("att_decoder_ff_mult", 4.0))
    att_decoder_dropout = float(model_cfg.get("att_decoder_dropout", model_cfg.get("ctc_dropout", 0.1)))
    att_decoder_max_len = int(model_cfg.get("att_decoder_max_len", 512))
    use_refiner = bool(model_cfg.get("use_refiner", True))
    use_len_predictor = bool(model_cfg.get("use_len_predictor", False))
    len_hidden = int(model_cfg.get("len_hidden", 192))

    loss_cfg = cfg["loss"]
    mel_floor = float(loss_cfg["mel_floor"])
    mel_ceil = float(loss_cfg["mel_ceil"])
    w_fm = float(loss_cfg.get("w_fm", 1.0))
    w_end = float(loss_cfg.get("w_end", 0.0))
    enable_fwd_end_loss = bool(loss_cfg.get("enable_fwd_end_loss", True))
    w_end_fwd = float(loss_cfg.get("w_end_fwd", w_end))
    w_end_bwd = float(loss_cfg.get("w_end_bwd", w_end))
    fwd_ode_grad = bool(loss_cfg.get("fwd_ode_grad", True))
    bwd_ode_grad = bool(loss_cfg.get("bwd_ode_grad", True))
    w_ctc_hat = float(loss_cfg["w_ctc_hat"])
    w_ctc_T = float(loss_cfg["w_ctc_T"])
    w_ctc_start = int(loss_cfg["w_ctc_start"])
    enable_ctc_dur = bool(loss_cfg.get("enable_ctc_dur", False))
    w_ctc_dur = float(loss_cfg.get("w_ctc_dur", 0.1))
    ctc_dur_start = int(loss_cfg.get("ctc_dur_start", w_ctc_start))
    enable_source_ctc = bool(loss_cfg.get("enable_source_ctc", False))
    w_ctc_source = float(loss_cfg.get("w_ctc_source", 0.0))
    source_ctc_start = int(loss_cfg.get("source_ctc_start", w_ctc_start))
    source_ctc_lr_scale = float(loss_cfg.get("source_ctc_lr_scale", 1.0))
    enable_zc_sample_ctc = bool(loss_cfg.get("enable_zc_sample_ctc", False))
    w_ctc_sample = float(loss_cfg.get("w_ctc_sample", 0.0))
    ctc_sample_start = int(loss_cfg.get("ctc_sample_start", w_ctc_start))
    ctc_sample_temp = float(loss_cfg.get("ctc_sample_temp", 1.0))
    enable_dit_hidden_ctc = bool(loss_cfg.get("enable_dit_hidden_ctc", False))
    w_dit_hidden_ctc = float(loss_cfg.get("w_dit_hidden_ctc", 0.0))
    dit_hidden_ctc_start = int(loss_cfg.get("dit_hidden_ctc_start", w_ctc_start))
    dit_hidden_ctc_anneal_steps = int(loss_cfg.get("dit_hidden_ctc_anneal_steps", 0))
    dit_hidden_ctc_t_min = float(loss_cfg.get("dit_hidden_ctc_t_min", 0.0))
    dit_hidden_ctc_t_max = float(loss_cfg.get("dit_hidden_ctc_t_max", 0.3))
    dit_hidden_ctc_t_min = max(0.0, min(1.0, dit_hidden_ctc_t_min))
    dit_hidden_ctc_t_max = max(0.0, min(1.0, dit_hidden_ctc_t_max))
    if dit_hidden_ctc_t_max < dit_hidden_ctc_t_min:
        raise ValueError("loss.dit_hidden_ctc_t_max must be >= loss.dit_hidden_ctc_t_min")
    dit_hidden_ctc_tap_index = int(loss_cfg.get(
        "dit_hidden_ctc_tap_index",
        max(0, int(model_cfg.get("vf_depth", 1)) // 2),
    ))
    dit_hidden_ctc_anchor = str(loss_cfg.get("dit_hidden_ctc_anchor", "sample")).lower()
    dit_hidden_ctc_anchor_mix_alpha = float(loss_cfg.get("dit_hidden_ctc_anchor_mix_alpha", 0.5))
    dit_hidden_ctc_apply_subsample = bool(loss_cfg.get("dit_hidden_ctc_apply_subsample", False))
    if dit_hidden_ctc_anchor not in {"mean", "sample", "mix"}:
        raise ValueError("loss.dit_hidden_ctc_anchor must be one of: mean, sample, mix")
    enable_att_decoder = bool(loss_cfg.get("enable_att_decoder", False))
    w_att_decoder = float(loss_cfg.get("w_att_decoder", 0.0))
    att_decoder_start = int(loss_cfg.get("att_decoder_start", w_ctc_start))
    att_decoder_anneal_steps = int(loss_cfg.get("att_decoder_anneal_steps", 0))
    att_decoder_label_smoothing = float(loss_cfg.get("att_decoder_label_smoothing", 0.1))
    att_decoder_detach_input = bool(loss_cfg.get("att_decoder_detach_input", False))
    enable_bwd_fm = bool(loss_cfg.get("enable_bwd_fm", False))
    w_bwd_fm = float(loss_cfg.get("w_bwd_fm", 0.0))
    bwd_fm_start = int(loss_cfg.get("bwd_fm_start", w_ctc_start))
    bwd_fm_anneal_steps = int(loss_cfg.get("bwd_fm_anneal_steps", 10000))
    bwd_fm_t_min = float(loss_cfg.get("bwd_fm_t_min", 0.6))
    bwd_fm_t_max = float(loss_cfg.get("bwd_fm_t_max", 1.0))
    bwd_fm_anchor = str(loss_cfg.get("bwd_fm_anchor", "sample")).lower()
    bwd_fm_anchor_mix_alpha = float(loss_cfg.get("bwd_fm_anchor_mix_alpha", 0.5))
    bwd_fm_t_min = max(0.0, min(1.0, bwd_fm_t_min))
    bwd_fm_t_max = max(0.0, min(1.0, bwd_fm_t_max))
    if bwd_fm_t_max < bwd_fm_t_min:
        raise ValueError("loss.bwd_fm_t_max must be >= loss.bwd_fm_t_min")
    if bwd_fm_anchor not in {"mean", "sample", "mix"}:
        raise ValueError("loss.bwd_fm_anchor must be one of: mean, sample, mix")
    w_tts_mel = float(loss_cfg["w_tts_mel"])
    w_mel_high = float(loss_cfg.get("w_mel_high", 0.0))
    mel_high_start_bin = int(loss_cfg.get("mel_high_start_bin", 60))
    w_ref = float(loss_cfg.get("w_ref", 0.0))
    w_delta = float(loss_cfg["w_delta"])
    w_mel_range = float(loss_cfg.get("w_mel_range", 0.0))
    w_align_start = int(loss_cfg["w_align_start"])
    w_dur = float(loss_cfg["w_dur"])
    w_len = float(loss_cfg.get("w_len", 0.0))
    mas_temp = float(loss_cfg["mas_temp"])
    mas_mode = str(loss_cfg.get("mas_mode", "hard")).lower()
    mas_mix_alpha = float(loss_cfg.get("mas_mix_alpha", 0.3))
    duration_perturb_enable = bool(loss_cfg.get("duration_perturb_enable", False))
    duration_perturb_num = int(loss_cfg.get("duration_perturb_num", 1))
    duration_perturb_num = max(1, duration_perturb_num)
    duration_perturb_sigma = float(loss_cfg.get("duration_perturb_sigma", 0.0))
    duration_perturb_include_base = bool(loss_cfg.get("duration_perturb_include_base", True))
    fwd_prior_mode = str(loss_cfg.get("fwd_prior_mode", "mas")).lower()
    fwd_prior_mix_alpha = float(loss_cfg.get("fwd_prior_mix_alpha", 0.5))
    fwd_anchor_mode = str(loss_cfg.get("fwd_anchor_mode", "mean")).lower()
    fwd_anchor_mix_alpha = float(loss_cfg.get("fwd_anchor_mix_alpha", 0.5))
    use_full_tts_teacher = bool(loss_cfg.get("use_full_tts_teacher", False))
    full_tts_teacher_every = int(loss_cfg.get("full_tts_teacher_every", 1))
    full_tts_teacher_every = max(1, full_tts_teacher_every)
    alignment_prior_mode = str(loss_cfg.get("alignment_prior_mode", "mas")).lower()
    if alignment_prior_mode not in {"mas", "gt_word", "ctc_blank_repeat"}:
        raise ValueError(
            f"Unsupported loss.alignment_prior_mode={alignment_prior_mode!r}; "
            "expected 'mas', 'gt_word', or 'ctc_blank_repeat'"
        )
    use_gt_alignment_prior = bool(loss_cfg.get("use_gt_alignment_prior", use_processed_unified))
    use_gt_duration_teacher = bool(loss_cfg.get("use_gt_duration_teacher", use_processed_unified))
    use_ctc_blank_repeat_prior = alignment_prior_mode == "ctc_blank_repeat"
    if alignment_prior_mode == "gt_word":
        use_gt_alignment_prior = True
        use_gt_duration_teacher = True
    elif use_ctc_blank_repeat_prior:
        # This mode learns a char+blank CTC topology via MAS/duration
        # prediction. Mixing in the older word-span teachers would put two
        # incompatible token grids into the same prior.
        use_gt_alignment_prior = False
        use_gt_duration_teacher = False
    w_prior = float(loss_cfg["w_prior"])
    enable_acoustic_prior_nll = bool(loss_cfg.get("enable_acoustic_prior_nll", True))
    enable_canonical_nll = bool(loss_cfg.get("enable_canonical_nll", False))
    w_canonical_nll = float(loss_cfg.get("w_canonical_nll", 0.0))
    canonical_stopgrad_split = bool(loss_cfg.get("canonical_stopgrad_split", False))
    w_canonical_prior_nll = float(loss_cfg.get("w_canonical_prior_nll", w_canonical_nll))
    w_canonical_bwd_nll = float(loss_cfg.get("w_canonical_bwd_nll", w_canonical_nll))
    canonical_nll_start = int(loss_cfg.get("canonical_nll_start", 1000))
    canonical_nll_anneal_steps = int(loss_cfg.get("canonical_nll_anneal_steps", 10000))
    canonical_match_mode = str(loss_cfg.get("canonical_match_mode", "nll")).lower()
    canonical_align_candidates = int(loss_cfg.get("canonical_align_candidates", duration_perturb_num))
    canonical_align_candidates = max(1, canonical_align_candidates)
    canonical_softmin_tau = float(loss_cfg.get("canonical_softmin_tau", 0.1))
    canonical_kl_free_bits = float(loss_cfg.get("canonical_kl_free_bits", 0.0))
    if canonical_match_mode not in {"nll", "kl", "alignment_softmin_nll"}:
        raise ValueError(f"Unsupported canonical_match_mode: {canonical_match_mode}")
    prior_loss_mode = str(loss_cfg.get("prior_loss_mode", "gaussian_nll"))
    prior_mu_loss_type = str(loss_cfg.get("prior_mu_loss_type", "smooth_l1"))
    w_prior_mu = float(loss_cfg.get("w_prior_mu", 1.0))
    w_prior_var = float(loss_cfg.get("w_prior_var", 0.05))
    w_prior_nll = float(loss_cfg.get("w_prior_nll", 0.1))
    prior_fixed_logvar = float(loss_cfg.get("prior_fixed_logvar", -2.0))
    prior_var_reg_target = float(loss_cfg.get("prior_var_reg_target", prior_fixed_logvar))
    w_tts_style_kl = float(loss_cfg.get("w_tts_style_kl", 0.01))
    w_tts_style_asr_kl = float(loss_cfg.get("w_tts_style_asr_kl", w_tts_style_kl))
    tts_style_asr_kl_stopgrad_pair = bool(loss_cfg.get("tts_style_asr_kl_stopgrad_pair", True))
    tts_style_kl_start = int(loss_cfg.get("tts_style_kl_start", 1000))
    tts_style_kl_anneal_steps = int(loss_cfg.get("tts_style_kl_anneal_steps", 10000))
    tts_style_asr_kl_start = int(loss_cfg.get("tts_style_asr_kl_start", tts_style_kl_start))
    tts_style_asr_kl_anneal_steps = int(loss_cfg.get("tts_style_asr_kl_anneal_steps", tts_style_kl_anneal_steps))
    enable_full_asr_ctc_aux = bool(loss_cfg.get("enable_full_asr_ctc_aux", False))
    w_ctc_full = float(loss_cfg.get("w_ctc_full", 0.2))
    full_asr_ctc_aux_start = int(loss_cfg.get("full_asr_ctc_aux_start", 1000))
    full_asr_ctc_aux_every = int(loss_cfg.get("full_asr_ctc_aux_every", 4))
    full_asr_ctc_aux_batch_size = int(loss_cfg.get("full_asr_ctc_aux_batch_size", 1))
    full_asr_ctc_aux_ode_grad = bool(loss_cfg.get("full_asr_ctc_aux_ode_grad", True))
    full_asr_ctc_aux_steps = int(loss_cfg.get("full_asr_ctc_aux_steps", 5))
    full_asr_ctc_aux_steps = max(1, full_asr_ctc_aux_steps)
    full_asr_ctc_aux_whole_utterance = bool(loss_cfg.get(
        "full_asr_ctc_aux_whole_utterance",
        cfg.get("infer", {}).get("asr_demo_whole_utterance", False),
    ))
    use_stat_match = bool(loss_cfg["use_stat_match"])
    w_stat = float(loss_cfg["w_stat"])
    use_stft = bool(loss_cfg.get("use_stft", False))
    w_stft = float(loss_cfg.get("w_stft", 0.0))
    enable_vf_lip = bool(loss_cfg.get("enable_vf_lip", False))
    vf_lip_start = int(loss_cfg.get("vf_lip_start", 5000))
    w_vf_lip = float(loss_cfg.get("w_vf_lip", 0.0))
    vf_lip_L_hi = float(loss_cfg.get("vf_lip_L_hi", 1.0))
    vf_lip_sigma = float(loss_cfg.get("vf_lip_sigma", 0.01))
    vf_lip_every = int(loss_cfg.get("vf_lip_every", 1))
    vf_lip_print_every = int(loss_cfg.get("vf_lip_print_every", 200))
    enable_bilip_diag = bool(loss_cfg.get("enable_bilip_diag", False))
    diag_every = int(loss_cfg.get("diag_every", 500))
    amp_sigma = float(loss_cfg.get("amp_sigma", 0.01))

    ssl_hidden_cfg = cfg.get("ssl_hidden", {})
    ssl_hidden_enable = bool(ssl_hidden_cfg.get("enable", False))
    ssl_hidden_w = float(ssl_hidden_cfg.get("w", 0.0))
    ssl_hidden_start = int(ssl_hidden_cfg.get("start", 10000))
    ssl_hidden_tap_index = int(ssl_hidden_cfg.get("tap_index", 5))
    ssl_hidden_head_hidden = int(ssl_hidden_cfg.get("head_hidden", 512))
    ssl_hidden_pool_factors = tuple(ssl_hidden_cfg.get("pool_factors", [1, 1]))
    ssl_hidden_target = str(ssl_hidden_cfg.get("target", "hidden")).lower()
    if ssl_hidden_target not in {"hidden", "zc"}:
        raise ValueError(f"Unsupported ssl_hidden target: {ssl_hidden_target}")
    ssl_teacher_name = str(ssl_hidden_cfg.get("teacher_name", "facebook/hubert-base-ls960"))
    ssl_teacher_layer_idx = int(ssl_hidden_cfg.get("teacher_layer_idx", -1))

    train_cfg = cfg["train"]
    total_steps = int(train_cfg["total_steps"])
    lr_all = float(train_cfg["lr_all"])
    lr_schedule = str(train_cfg["lr_schedule"])
    lr_warmup_steps = int(train_cfg["lr_warmup_steps"])
    lr_min_scale = float(train_cfg["lr_min_scale"])
    save_every_steps = int(train_cfg["save_every_steps"])
    keep_last_k = int(train_cfg["keep_last_k"])
    resume_from = train_cfg["resume_from"]
    use_ema = bool(train_cfg["use_ema"])
    ema_decay = float(train_cfg["ema_decay"])
    log_every = int(train_cfg["log_every"])
    debug_every = int(train_cfg["debug_every"])
    grad_clip = float(train_cfg["grad_clip"])
    perf_log_every = int(train_cfg.get("perf_log_every", log_every))

    runtime_cfg = cfg["runtime"]

    infer_cfg = cfg["infer"]
    ode_steps_eval = int(infer_cfg["ode_steps_eval"])
    ode_steps_endloss = int(infer_cfg["ode_steps_endloss"])
    full_asr_chunk_core = int(infer_cfg.get("full_asr_chunk_core", 256))
    full_asr_chunk_ctx = int(infer_cfg.get("full_asr_chunk_ctx", 96))
    full_asr_use_euler = bool(infer_cfg.get("full_asr_use_euler", True))
    asr_demo_whole_utterance = bool(infer_cfg.get("asr_demo_whole_utterance", False))
    asr_demo_steps = int(infer_cfg.get("asr_demo_steps", 5))
    asr_demo_steps = max(1, asr_demo_steps)
    demo_cfg_scale = float(infer_cfg["demo_cfg_scale"])
    disable_demo_cfg_guidance = bool(infer_cfg.get("disable_demo_cfg_guidance", False))
    if disable_demo_cfg_guidance and abs(demo_cfg_scale - 1.0) > 1e-8:
        print(f"[CFG][WARN] disable_demo_cfg_guidance=true; override demo_cfg_scale {demo_cfg_scale} -> 1.0")
        demo_cfg_scale = 1.0
    demo_prior_temp = float(infer_cfg["demo_prior_temp"])
    demo_style_temp = float(infer_cfg.get("demo_style_temp", 1.0))
    demo_every = int(infer_cfg["demo_every"])
    demo_rtf = bool(infer_cfg.get("demo_rtf", True))
    demo_plot_trajectory = bool(infer_cfg.get("demo_plot_trajectory", False))
    demo_trajectory_export_csv = bool(infer_cfg.get("demo_trajectory_export_csv", True))
    demo_trajectory_projection = str(infer_cfg.get("demo_trajectory_projection", "pca")).lower()
    demo_trajectory_dims = infer_cfg.get("demo_trajectory_dims", [0, 1])
    if not isinstance(demo_trajectory_dims, (list, tuple)) or len(demo_trajectory_dims) != 2:
        demo_trajectory_dims = [0, 1]
    demo_trajectory_dims = [int(demo_trajectory_dims[0]), int(demo_trajectory_dims[1])]
    demo_trajectory_pool = str(infer_cfg.get("demo_trajectory_pool", "utterance_mean")).lower()
    demo_trajectory_frames = int(infer_cfg.get("demo_trajectory_frames", 12))
    demo_trajectory_speakers = int(infer_cfg.get("demo_trajectory_speakers", 3))
    demo_trajectory_samples = max(1, int(infer_cfg.get("demo_trajectory_samples", 1)))
    demo_trajectory_reverse = bool(infer_cfg.get("demo_trajectory_reverse", False))
    demo_trajectory_paper_style = bool(infer_cfg.get("demo_trajectory_paper_style", True))
    demo_trajectory_display_x_scale = float(infer_cfg.get("demo_trajectory_display_x_scale", 1.0))
    demo_trajectory_display_y_scale = float(infer_cfg.get("demo_trajectory_display_y_scale", 1.0))
    demo_trajectory_canonical_color = str(infer_cfg.get("demo_trajectory_canonical_color", "#1f77b4"))
    demo_trajectory_speaker_colors = infer_cfg.get("demo_trajectory_speaker_colors", ["#2ca02c", "#ff7f0e"])
    if not isinstance(demo_trajectory_speaker_colors, (list, tuple)) or len(demo_trajectory_speaker_colors) == 0:
        demo_trajectory_speaker_colors = ["#2ca02c", "#ff7f0e"]
    demo_trajectory_speaker_colors = [str(c) for c in demo_trajectory_speaker_colors]
    demo_trajectory_annotate_points = bool(infer_cfg.get("demo_trajectory_annotate_points", False))
    demo_trajectory_zu_fanout = bool(infer_cfg.get("demo_trajectory_zu_fanout", False))
    demo_trajectory_zu_zc_samples = max(1, int(infer_cfg.get("demo_trajectory_zu_zc_samples", 3)))
    demo_trajectory_zu_u_samples = max(1, int(infer_cfg.get("demo_trajectory_zu_u_samples", 6)))
    demo_trajectory_asr_many_to_one = bool(infer_cfg.get("demo_trajectory_asr_many_to_one", False))
    demo_trajectory_asr_realization_plot = bool(infer_cfg.get("demo_trajectory_asr_realization_plot", False))
    demo_trajectory_asr_realization_speakers = max(
        1,
        int(infer_cfg.get("demo_trajectory_asr_realization_speakers", demo_trajectory_speakers)),
    )
    demo_trajectory_asr_realization_styles = max(
        1,
        int(infer_cfg.get("demo_trajectory_asr_realization_styles", min(8, demo_trajectory_samples))),
    )
    tts_demo_text_source = str(infer_cfg.get("tts_demo_text_source", "train_same_spk")).lower()
    valid_tts_demo_text_sources = {"train_same_spk", "demo_manifest"}
    if tts_demo_text_source not in valid_tts_demo_text_sources:
        raise ValueError(
            f"Unsupported infer.tts_demo_text_source={tts_demo_text_source}; "
            f"expected one of {sorted(valid_tts_demo_text_sources)}"
        )
    spk_sim_enable = bool(infer_cfg.get("spk_sim_enable", False))
    spk_sim_model = str(infer_cfg.get("spk_sim_model", "speechbrain/spkrec-ecapa-voxceleb"))
    spk_sim_savedir = infer_cfg.get("spk_sim_savedir")
    if spk_sim_savedir is not None:
        spk_sim_savedir = os.path.abspath(os.path.expanduser(str(spk_sim_savedir)))
    demo_eval_utmos = bool(infer_cfg.get("demo_eval_utmos", False))
    demo_utmos_repo = str(infer_cfg.get("demo_utmos_repo", "tarepan/SpeechMOS:v1.2.0"))
    demo_utmos_model = str(infer_cfg.get("demo_utmos_model", "utmos22_strong"))
    demo_eval_whisper = bool(infer_cfg.get("demo_eval_whisper", False))
    demo_whisper_model = str(infer_cfg.get("demo_whisper_model", "medium.en"))
    demo_plot_generated_wav_mel = bool(infer_cfg.get("demo_plot_generated_wav_mel", True))
    demo_generated_mel_frontend = str(infer_cfg.get("demo_generated_mel_frontend", "bigvgan")).lower()
    if demo_generated_mel_frontend not in {"bigvgan", "librosa"}:
        raise ValueError("infer.demo_generated_mel_frontend must be 'bigvgan' or 'librosa'")
    demo_mel_plot_cmap = str(infer_cfg.get("demo_mel_plot_cmap", "viridis"))
    demo_mel_plot_top_percentile = float(infer_cfg.get("demo_mel_plot_top_percentile", 99.5))
    demo_mel_plot_dynamic_range_db = float(infer_cfg.get("demo_mel_plot_dynamic_range_db", 65.0))
    asr_demo_decode_mode = str(infer_cfg.get("asr_demo_decode_mode", "greedy")).lower()
    asr_demo_kenlm_cfg = infer_cfg.get("asr_demo_kenlm", {})
    asr_demo_kenlm_fallback = bool(asr_demo_kenlm_cfg.get("allow_fallback", False))
    load_bigvgan_model = bool(runtime_cfg.get("load_bigvgan_model", demo_every > 0))

    demo_dir = cfg["io"]["demo_dir"]
    ckpt_dir = cfg["io"]["ckpt_dir"]
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    if mel_cache_dir:
        os.makedirs(mel_cache_dir, exist_ok=True)
    if speecht5_cache_dir:
        os.makedirs(speecht5_cache_dir, exist_ok=True)

    log_path = os.path.join(ckpt_dir, "train.log")
    metrics_log_path = os.path.join(ckpt_dir, "train_metrics.jsonl")
    merged_cfg_path = os.path.join(ckpt_dir, "merged_config.json")
    with open(merged_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    tee_fp = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeIO(sys.__stdout__, tee_fp)
    sys.stderr = TeeIO(sys.__stderr__, tee_fp)
    print(f"[LOG] tee stdout/stderr -> {log_path}")
    print(f"[LOG] metrics -> {metrics_log_path}")
    print(f"[LOG] merged config -> {merged_cfg_path}")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))

    import librosa
    bigvgan_model = None
    svae_model = None
    bigvgan_h = None
    bigvgan_get_mel_spectrogram = None
    if speech_backend == "svae":
        sampling_rate = int(svae_sample_rate)
        D_mel = int(svae_dim)
        hop_size = int(svae_hop_size)
        if load_bigvgan_model:
            semantic_root = os.path.abspath(svae_root)
            if semantic_root not in sys.path:
                sys.path.insert(0, semantic_root)
            from dac.model.dac import DAC as SemanticVAEDAC
            from dac.model.utils import read_json_file as svae_read_json_file

            metainfo = svae_read_json_file(os.path.join(svae_ckpt, "metainfo.json"))
            bigvgan_conf = metainfo["DAC"].get("bigvgan_conf")
            if bigvgan_conf and not os.path.isabs(str(bigvgan_conf)):
                metainfo["DAC"]["bigvgan_conf"] = os.path.join(semantic_root, str(bigvgan_conf))
            ckpt_name = "ema_state_dict.pth" if svae_use_ema else "weights.pth"
            ckpt_obj = torch.load(os.path.join(svae_ckpt, "dac", ckpt_name), map_location="cpu")
            if svae_use_ema:
                ckpt_obj = {k.replace("ema_model.", ""): v for k, v in ckpt_obj.items()}
            else:
                ckpt_obj = ckpt_obj["state_dict"]
            ckpt_obj = {k: v for k, v in ckpt_obj.items() if not k.startswith("projectors")}
            svae_model = SemanticVAEDAC(**metainfo["DAC"])
            if hasattr(svae_model, "projectors"):
                del svae_model.projectors
            svae_model.load_state_dict(ckpt_obj, strict=False)
            svae_model = svae_model.eval().to(device)
            for p in svae_model.parameters():
                p.requires_grad_(False)
            print(f"Loaded Semantic-VAE decoder: {svae_ckpt} use_ema={svae_use_ema}")
        else:
            print("[SVAE] load_bigvgan_model=false; training without waveform demos")
    else:
        import bigvgan
        from meldataset import get_mel_spectrogram
        bigvgan_get_mel_spectrogram = get_mel_spectrogram

        if load_bigvgan_model:
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            bigvgan_model.remove_weight_norm()
            bigvgan_model = bigvgan_model.eval().to(device)
            for p in bigvgan_model.parameters():
                p.requires_grad_(False)
            bigvgan_h = bigvgan_model.h
            print("Loaded BigVGAN:", bigvgan_name)
        else:
            from huggingface_hub import hf_hub_download

            if os.path.isdir(bigvgan_name):
                bigvgan_config_file = os.path.join(bigvgan_name, "config.json")
            else:
                bigvgan_config_file = hf_hub_download(
                    repo_id=bigvgan_name,
                    filename="config.json",
                    local_files_only=True,
                )
            bigvgan_h = bigvgan.load_hparams_from_json(bigvgan_config_file)
            print(f"[BigVGAN] load_bigvgan_model=false; using config only: {bigvgan_config_file}")
        sampling_rate = int(bigvgan_h.sampling_rate)
        D_mel = int(bigvgan_h.num_mels)
        hop_size = int(bigvgan_h.hop_size)

    def canonicalize_text(text: str):
        text = str(text).strip()
        if force_text_normalize:
            text = normalize_text_basic(text)
        return text

    valid_prior_modes = {"gaussian_nll", "mu_var_reg", "mu_only_fixed_var"}
    if prior_loss_mode not in valid_prior_modes:
        raise ValueError(f"Unsupported prior_loss_mode={prior_loss_mode}; expected one of {sorted(valid_prior_modes)}")
    valid_mu_losses = {"l1", "mse", "smooth_l1"}
    if prior_mu_loss_type not in valid_mu_losses:
        raise ValueError(f"Unsupported prior_mu_loss_type={prior_mu_loss_type}; expected one of {sorted(valid_mu_losses)}")

    def masked_prior_mu_loss(mu_pred, z_target, maskK):
        if prior_mu_loss_type == "mse":
            return masked_mse(mu_pred, z_target, maskK)
        if prior_mu_loss_type == "l1":
            return masked_l1(mu_pred, z_target, maskK)
        mask = maskK.float().unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0) * mu_pred.shape[-1]
        diff = F.smooth_l1_loss(mu_pred, z_target, reduction="none")
        return (diff * mask).sum() / denom

    def diag_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
        # KL[N(mu_q, var_q) || N(mu_p, var_p)] averaged over batch and latent dims.
        logvar_q = logvar_q.clamp(-12.0, 8.0)
        logvar_p = logvar_p.clamp(-12.0, 8.0)
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p).clamp_min(1e-8)
        kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
        return kl.mean()

    def masked_diag_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p, maskK, free_bits=0.0):
        # KL[N(mu_q, var_q) || N(mu_p, var_p)] averaged over valid frames and dims.
        logvar_q = logvar_q.clamp(-12.0, 8.0)
        logvar_p = logvar_p.clamp(-12.0, 8.0)
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p).clamp_min(1e-8)
        kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
        if free_bits > 0.0:
            kl = torch.clamp(kl, min=float(free_bits))
        mask = maskK.float().unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0) * mu_q.shape[-1]
        return (kl * mask).sum() / denom

    def linear_anneal(step_now, start, steps):
        if step_now < start:
            return 0.0
        if steps <= 0:
            return 1.0
        return float(min(1.0, max(0.0, (step_now - start) / float(steps))))

    full_row_by_utt = {}
    duration_row_by_utt = {}
    asr_full_rows = []

    def _row_speaker(row):
        spk = row.get("speaker")
        if spk is not None:
            return str(spk)
        wav = row.get("wav", row.get("parent_wav", ""))
        return str(extract_speaker_id_from_path(wav))

    def _convert_unified_fm_row(row, full_by_utt):
        utt_id = str(row["utt_id"])
        full = full_by_utt.get(utt_id, {})
        out = dict(row)
        out["_schema"] = "processed_unified_fm"
        out["parent_wav"] = row["wav"]
        out["cut_start_mel"] = int(row["ctx_mel_start"])
        out["cut_end_mel"] = int(row["ctx_mel_end"])
        out["cut_mel_len"] = int(row["ctx_mel_len"])
        out["text_norm"] = row.get("text_norm_ctx", row.get("text_norm_full", ""))
        out["text_raw"] = row.get("text_raw_full", full.get("text_raw", out["text_norm"]))
        out["cut_type"] = "tts"
        out["npz_path"] = full.get("npz_path")
        out["_ctc_num_frames"] = int(full.get("ctc_num_frames", 0) or 0)
        out["_svae_len"] = int(full.get("svae_len", full.get("mel_len", row.get("ctx_mel_end", 0))) or 0)
        out["_full_words"] = full.get("words", [])
        out["_pause_after_word_mel_frames"] = full.get("pause_after_word_mel_frames", [])
        out["_mel_len"] = int(full.get("svae_len", full.get("mel_len", row.get("ctx_mel_end", 0))) or 0)
        return out

    def _convert_unified_full_row(row):
        out = dict(row)
        K = int(row.get("svae_len", row.get("mel_len", 0)) or 0)
        if K <= 0:
            K = int(row.get("ctc_num_frames", 1) or 1)
        out["_schema"] = "processed_unified_full_utterance"
        out["parent_wav"] = row["wav"]
        out["cut_start_mel"] = 0
        out["cut_end_mel"] = K
        out["cut_mel_len"] = K
        out["core_start_in_ctx"] = 0
        out["core_end_in_ctx"] = K
        out["core_mel_start"] = 0
        out["core_mel_end"] = K
        out["ctx_mel_start"] = 0
        out["ctx_mel_end"] = K
        out["ctx_mel_len"] = K
        out["text_norm"] = row.get("text_norm", "")
        out["text_raw"] = row.get("text_raw", out["text_norm"])
        out["text_norm_full"] = out["text_norm"]
        out["text_norm_ctx"] = out["text_norm"]
        out["text_norm_core"] = out["text_norm"]
        out["text_raw_full"] = out["text_raw"]
        out["cut_type"] = "full_utterance"
        out["is_full_utt"] = True
        out["_ctc_num_frames"] = int(row.get("ctc_num_frames", 0) or 0)
        out["_svae_len"] = K
        out["_mel_len"] = K
        out["_full_words"] = row.get("words", [])
        out["_pause_after_word_mel_frames"] = (
            row.get("pause_after_word_svae_frames")
            or row.get("pause_after_word_mel_frames", [])
        )
        return out

    if use_processed_unified:
        if not (full_manifest_clean and os.path.exists(full_manifest_clean)):
            raise FileNotFoundError(f"processed full_manifest_clean not found: {full_manifest_clean}")
        full_rows_all = read_jsonl_rows(full_manifest_clean, max_rows=None)
        assert len(full_rows_all) > 0, "processed full_manifest_clean is empty"
        full_row_by_utt = {str(row["utt_id"]): row for row in full_rows_all}
        if use_utterance_training:
            cut_rows_all = [_convert_unified_full_row(row) for row in full_rows_all]
        else:
            if not (fm_core_context_manifest and os.path.exists(fm_core_context_manifest)):
                raise FileNotFoundError(f"processed fm_core_context_manifest not found: {fm_core_context_manifest}")
            fm_rows_raw = read_jsonl_rows(fm_core_context_manifest, max_rows=max_cut_rows)
            assert len(fm_rows_raw) > 0, "processed fm_core_context_cuts is empty"
            cut_rows_all = [_convert_unified_fm_row(row, full_row_by_utt) for row in fm_rows_raw]
        if tts_duration_manifest and os.path.exists(tts_duration_manifest):
            duration_rows = read_jsonl_rows(tts_duration_manifest, max_rows=None)
            duration_row_by_utt = {str(row["utt_id"]): row for row in duration_rows}
        if asr_full_chunks_manifest and os.path.exists(asr_full_chunks_manifest):
            asr_full_rows = read_jsonl_rows(asr_full_chunks_manifest, max_rows=None)
        if use_utterance_training:
            asr_full_rows = list(full_rows_all)
        print(f"[DATA] processed_unified_dir={processed_unified_dir}")
        print(f"[DATA] train_unit={train_unit} utterance_training={use_utterance_training}")
        print(f"[DATA] full_manifest_clean={full_manifest_clean} rows={len(full_rows_all)}")
        print(f"[DATA] train_rows={len(cut_rows_all)}")
        print(f"[DATA] fm_core_context_manifest={fm_core_context_manifest} rows={'skipped' if use_utterance_training else len(cut_rows_all)}")
        print(f"[DATA] tts_duration_manifest={tts_duration_manifest} rows={len(duration_row_by_utt)}")
        print(f"[DATA] asr_full_chunks_manifest={asr_full_chunks_manifest} rows={len(asr_full_rows)}")
    else:
        cut_rows_all = read_jsonl_rows(cut_manifest, max_rows=max_cut_rows)
        assert len(cut_rows_all) > 0, "cut manifest is empty or missing"

    target_spks = select_target_speakers(cut_rows_all, target_spks_cfg, top_k_spk)
    target_spk_set = set(target_spks)
    cut_rows = [row for row in cut_rows_all if str(row.get("speaker", "")) in target_spk_set]
    assert len(cut_rows) > 0, "No cut rows left after speaker filtering"

    if use_processed_unified:
        tts_pool = list(cut_rows)
        asr_pool = list(cut_rows)
    else:
        tts_pool = [row for row in cut_rows if row.get("cut_type", "") == "tts"]
        asr_pool = [row for row in cut_rows if row.get("cut_type", "") == "asr"]
    assert len(tts_pool) > 0, "tts_pool is empty"
    assert len(asr_pool) > 0, "asr_pool is empty"

    spk_list = sorted(list({str(row["speaker"]) for row in cut_rows}))
    spk2id = {spk: idx for idx, spk in enumerate(spk_list)}
    n_spk = len(spk_list)

    speaker_bank = None
    if speaker_cond_type not in SpeakerConditioner.TABLE_MODES:
        speaker_bank, missing_spks, resolved_speaker_emb_path = load_speaker_embedding_bank(
            speaker_emb_path,
            spk_list,
            normalize=speaker_emb_l2_normalize,
            missing=speaker_emb_missing,
        )
        print(
            f"[SPEAKER-BANK] type={speaker_cond_type} path={resolved_speaker_emb_path} "
            f"shape={tuple(speaker_bank.shape)} l2_normalize={speaker_emb_l2_normalize} "
            f"missing={len(missing_spks)} trainable={speaker_emb_trainable}"
        )

    print("Selected speakers:", target_spks)
    print(f"Filtered cut rows: {len(cut_rows)} / {len(cut_rows_all)}")
    print("tts_pool =", len(tts_pool))
    print("asr_pool =", len(asr_pool))
    print("n_spk =", n_spk, "spk_list(head) =", spk_list[:10])

    if use_processed_unified:
        aligned_rows_all = full_rows_all
        aligned_rows = [row for row in aligned_rows_all if _row_speaker(row) in target_spk_set]
        if asr_full_rows:
            asr_full_rows = [row for row in asr_full_rows if _row_speaker(row) in target_spk_set]
        aligned_row_by_wav = {row["wav"]: row for row in aligned_rows}
    else:
        aligned_rows_all = read_jsonl_rows(aligned_manifest, max_rows=None)
        assert len(aligned_rows_all) > 0, "aligned manifest is empty or missing"
        aligned_rows = []
        for row in aligned_rows_all:
            spk = extract_speaker_id_from_path(row["wav"])
            if spk in target_spk_set:
                aligned_rows.append(row)
        aligned_row_by_wav = {row["wav"]: row for row in aligned_rows}
    assert len(aligned_rows) > 0, "No aligned rows left after speaker filtering"
    print("aligned_rows(train/full-teacher) =", len(aligned_rows))
    if os.path.abspath(str(demo_aligned_manifest)) == os.path.abspath(str(aligned_manifest)):
        demo_aligned_rows = aligned_rows
    else:
        demo_aligned_rows_all = read_jsonl_rows(demo_aligned_manifest, max_rows=None)
        assert len(demo_aligned_rows_all) > 0, "demo aligned manifest is empty or missing"
        demo_aligned_rows = []
        for row in demo_aligned_rows_all:
            spk = extract_speaker_id_from_path(row["wav"])
            if spk in target_spk_set:
                demo_aligned_rows.append(row)
        if len(demo_aligned_rows) == 0:
            print(
                "[WARN] No demo aligned rows left after speaker filtering; "
                "falling back to unfiltered demo aligned rows"
            )
            demo_aligned_rows = demo_aligned_rows_all
    print("demo_aligned_manifest =", demo_aligned_manifest)
    print("demo_aligned_rows(for ASR-FULL demo) =", len(demo_aligned_rows))

    def _row_text(row):
        return canonicalize_text(row.get("text_norm", row.get("text_raw", row.get("text", ""))))

    def _row_has_known_speaker(row):
        return _row_speaker(row) in spk2id

    asr_train_demo_rows = [
        row for row in aligned_rows
        if row.get("wav") and len(_row_text(row)) > 0
    ]
    if len(asr_train_demo_rows) == 0:
        raise RuntimeError("No train rows available for ASR-FULL-TRAIN demo")
    print("asr_train_demo_rows(for ASR-FULL-TRAIN demo) =", len(asr_train_demo_rows))

    train_ref_wav_by_spk = {}
    train_ref_wavs_by_spk = {}
    for row in aligned_rows:
        spk = _row_speaker(row)
        wav_path = row.get("wav", row.get("parent_wav"))
        if spk in spk2id and wav_path:
            train_ref_wavs_by_spk.setdefault(spk, []).append(wav_path)
            if spk not in train_ref_wav_by_spk:
                train_ref_wav_by_spk[spk] = wav_path
    print("train_ref_wav_by_spk(for TTS spk-sim) =", len(train_ref_wav_by_spk))

    if tts_demo_text_source == "train_same_spk":
        tts_demo_rows = [
            row for row in aligned_rows
            if _row_has_known_speaker(row) and len(_row_text(row)) > 0
        ]
    else:
        tts_demo_rows = [
            row for row in demo_aligned_rows
            if len(_row_text(row)) > 0
        ]
    if len(tts_demo_rows) == 0:
        raise RuntimeError(
            f"No TTS demo rows available for infer.tts_demo_text_source={tts_demo_text_source!r}"
        )
    print(f"tts_demo_rows(source={tts_demo_text_source}) =", len(tts_demo_rows))
    speech_path_by_wav = {}
    if speech_backend == "svae":
        for row in list(full_row_by_utt.values()) + list(cut_rows) + list(aligned_rows) + list(demo_aligned_rows):
            wav_key = row.get("parent_wav", row.get("wav"))
            latent_path = row.get("svae_latent_path") or row.get("speech_path") or row.get("latent_path")
            if wav_key and latent_path:
                speech_path_by_wav[os.path.abspath(str(wav_key))] = os.path.abspath(str(latent_path))
        print(f"[SVAE] speech_path_by_wav entries={len(speech_path_by_wav)}")
    mel_paths_all = list(OrderedDict.fromkeys(
        [row.get("parent_wav", row.get("wav")) for row in cut_rows] + [row["wav"] for row in aligned_rows]
    ).keys())

    texts_all = sorted(list({
        canonicalize_text(row.get("text_norm", row.get("text", "")))
        for row in cut_rows
        if len(canonicalize_text(row.get("text_norm", row.get("text", "")))) > 0
    }))
    text_preload_candidates = list(OrderedDict.fromkeys(
        texts_all + [
            canonicalize_text(row.get("text_norm", row.get("text_raw", "")))
            for row in aligned_rows
            if len(canonicalize_text(row.get("text_norm", row.get("text_raw", "")))) > 0
        ]
    ).keys())
    tok = build_tokenizer(tokenizer_cfg, texts=texts_all)
    Vt = len(tok.itos)
    PAD_ID = tok.stoi["<pad>"]
    UNK_ID = tok.stoi["<unk>"]
    BLANK_ID = tok.stoi["<blank>"]
    AED_SOS_ID = int(Vt)
    AED_EOS_ID = int(Vt + 1)
    AED_VOCAB_SIZE = int(Vt + 2)
    print(
        f"tokenizer={str(tokenizer_cfg.get('type', 'char')).lower()} "
        f"vocab_size={Vt} blank_id={BLANK_ID} aed_vocab={AED_VOCAB_SIZE}"
    )

    asr_demo_kenlm = None
    tokenizer_type = str(tokenizer_cfg.get("type", "char")).lower()
    if asr_demo_decode_mode == "kenlm" and tokenizer_type not in {"char", "character"}:
        print("[ASR-DECODE][WARN] KenLM decoder currently expects character tokens; using greedy for BPE")
        asr_demo_decode_mode = "greedy"
    if asr_demo_decode_mode == "kenlm":
        asr_demo_kenlm_corpus_manifest = asr_demo_kenlm_cfg.get("lexicon_corpus_manifest")
        if not asr_demo_kenlm_corpus_manifest:
            if use_processed_unified and full_manifest_clean and os.path.exists(full_manifest_clean):
                asr_demo_kenlm_corpus_manifest = full_manifest_clean
            elif aligned_manifest and os.path.exists(aligned_manifest):
                asr_demo_kenlm_corpus_manifest = aligned_manifest
        asr_demo_kenlm = OptionalKenLMCTCDecoder(
            tok.itos,
            BLANK_ID,
            KenLMCTCDecoderConfig(
                preset=str(asr_demo_kenlm_cfg.get("preset", "librispeech-4-gram")),
                lexicon=asr_demo_kenlm_cfg.get("lexicon"),
                lm=asr_demo_kenlm_cfg.get("lm"),
                lexicon_corpus_manifest=asr_demo_kenlm_corpus_manifest,
                lexicon_cache_dir=asr_demo_kenlm_cfg.get("lexicon_cache_dir"),
                beam_size=int(asr_demo_kenlm_cfg.get("beam_size", 100)),
                beam_threshold=float(asr_demo_kenlm_cfg.get("beam_threshold", 100.0)),
                beam_size_token=int(asr_demo_kenlm_cfg.get("beam_size_token", 30)),
                lm_weight=float(asr_demo_kenlm_cfg.get("lm_weight", 1.23)),
                word_score=float(asr_demo_kenlm_cfg.get("word_score", -0.26)),
                allow_fallback=asr_demo_kenlm_fallback,
            ),
        )
        if asr_demo_kenlm.enabled:
            print(
                "[ASR-DECODE] train/demo uses KenLM "
                f"preset={asr_demo_kenlm.cfg.preset} beam={asr_demo_kenlm.cfg.beam_size} "
                f"lm_weight={asr_demo_kenlm.cfg.lm_weight} word_score={asr_demo_kenlm.cfg.word_score} "
                f"lexicon={asr_demo_kenlm.lexicon_path}"
            )
        else:
            print(f"[ASR-DECODE][WARN] KenLM unavailable; fallback greedy. error={asr_demo_kenlm.error}")
    elif asr_demo_decode_mode != "greedy":
        print(f"[ASR-DECODE][WARN] unsupported asr_demo_decode_mode={asr_demo_decode_mode}; using greedy")
        asr_demo_decode_mode = "greedy"

    alpha_sample_rows = tts_pool if len(tts_pool) <= 20000 else random.sample(tts_pool, 20000)
    ratios = []
    for row in alpha_sample_rows:
        txt = canonicalize_text(row.get("text_norm", row.get("text", "")))
        ratios.append(float(row["cut_mel_len"]) / max(len(tok.encode(txt)), 1))
    alpha_K = float(np.mean(ratios)) if ratios else 1.0
    print("alpha_K ≈ mean(cut_mel_len/num_tokens) =", alpha_K)

    ctc_blank_skip_speecht5 = bool(model_cfg.get("ctc_blank_skip_speecht5", False))
    ctc_blank_use_speecht5 = (
        text_encoder_type == "speecht5"
        and use_ctc_blank_repeat_prior
        and not ctc_blank_skip_speecht5
    )

    st5 = None
    if text_encoder_type in {"trainable", "trainable_token"}:
        H_text = text_encoder_dim
    elif use_ctc_blank_repeat_prior and ctc_blank_skip_speecht5:
        H_text = int(model_cfg.get("ctc_blank_text_dim", 768))
    else:
        st5 = FrozenSpeechT5TextEncoder(
            model_name="microsoft/speecht5_tts",
            device=device,
            layer_idx=st5_layer_idx,
        )
        H_text = st5.hidden_size

    wav_cache = OrderedDict()
    mel_cache = OrderedDict()
    text_hidden_cache = OrderedDict()
    mel_gpu_cache = OrderedDict()
    text_hidden_gpu_cache = OrderedDict()
    gpu_cache_state = {
        "mel_bytes": 0,
        "text_bytes": 0,
        "mel_oom": False,
        "text_oom": False,
    }

    def _trim_cache(cache, max_items):
        while len(cache) > int(max_items):
            cache.popitem(last=False)

    def _cache_obj_nbytes(obj):
        total = 0
        for key in ("hidden", "mask"):
            value = obj.get(key)
            if torch.is_tensor(value):
                total += tensor_nbytes(value)
        return total

    def _gpu_cache_can_store(required_bytes, used_bytes, limit_bytes):
        if device != "cuda":
            return False
        if limit_bytes is not None:
            return (used_bytes + int(required_bytes)) <= int(limit_bytes)
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            return True
        return int(required_bytes) <= max(0, int(free_bytes) - int(gpu_cache_reserve_bytes))

    def _trim_tensor_gpu_cache(cache, limit_bytes):
        if limit_bytes is None:
            return
        while gpu_cache_state["mel_bytes"] > int(limit_bytes) and len(cache) > 1:
            _, old_value = cache.popitem(last=False)
            gpu_cache_state["mel_bytes"] -= tensor_nbytes(old_value)

    def _trim_obj_gpu_cache(cache, limit_bytes):
        if limit_bytes is None:
            return
        while gpu_cache_state["text_bytes"] > int(limit_bytes) and len(cache) > 1:
            _, old_value = cache.popitem(last=False)
            gpu_cache_state["text_bytes"] -= _cache_obj_nbytes(old_value)

    def _promote_mel_to_gpu(path: str, mel_src: torch.Tensor):
        if not gpu_mel_cache:
            return None
        if path in mel_gpu_cache:
            mel_gpu_cache.move_to_end(path)
            return mel_gpu_cache[path]
        required_bytes = int(mel_src.numel()) * dtype_nbytes(gpu_mel_cache_dtype)
        if not _gpu_cache_can_store(required_bytes, gpu_cache_state["mel_bytes"], gpu_mel_cache_limit_bytes):
            return None
        try:
            mel_gpu = mel_src.to(device=device, dtype=gpu_mel_cache_dtype).contiguous()
        except torch.OutOfMemoryError:
            if not gpu_cache_state["mel_oom"]:
                print("[CACHE] mel GPU cache promotion hit OOM; keep using CPU cache fallback")
                gpu_cache_state["mel_oom"] = True
            return None
        mel_gpu_cache[path] = mel_gpu
        mel_gpu_cache.move_to_end(path)
        gpu_cache_state["mel_bytes"] += tensor_nbytes(mel_gpu)
        _trim_tensor_gpu_cache(mel_gpu_cache, gpu_mel_cache_limit_bytes)
        return mel_gpu

    def _promote_text_obj_to_gpu(text: str, obj_src):
        if not gpu_text_cache:
            return None
        if text in text_hidden_gpu_cache:
            text_hidden_gpu_cache.move_to_end(text)
            return text_hidden_gpu_cache[text]
        hidden_cpu = obj_src["hidden"]
        mask_cpu = obj_src["mask"]
        required_bytes = int(hidden_cpu.numel()) * dtype_nbytes(gpu_text_cache_dtype)
        required_bytes += int(mask_cpu.numel()) * dtype_nbytes(torch.bool)
        if not _gpu_cache_can_store(required_bytes, gpu_cache_state["text_bytes"], gpu_text_cache_limit_bytes):
            return None
        try:
            obj_gpu = {
                "text": text,
                "hidden": hidden_cpu.to(device=device, dtype=gpu_text_cache_dtype).contiguous(),
                "mask": mask_cpu.to(device=device, dtype=torch.bool).contiguous(),
            }
        except torch.OutOfMemoryError:
            if not gpu_cache_state["text_oom"]:
                print("[CACHE] text GPU cache promotion hit OOM; keep using CPU cache fallback")
                gpu_cache_state["text_oom"] = True
            return None
        text_hidden_gpu_cache[text] = obj_gpu
        text_hidden_gpu_cache.move_to_end(text)
        gpu_cache_state["text_bytes"] += _cache_obj_nbytes(obj_gpu)
        _trim_obj_gpu_cache(text_hidden_gpu_cache, gpu_text_cache_limit_bytes)
        return obj_gpu

    def _mel_cache_path(wav_path: str):
        if not mel_cache_dir:
            return None
        if speech_backend == "svae":
            mapped = speech_path_by_wav.get(os.path.abspath(str(wav_path)))
            if mapped:
                return mapped
        return os.path.join(mel_cache_dir, f"{sha1_key(os.path.abspath(wav_path))}.pt")

    def _speecht5_cache_path(text: str):
        if not speecht5_cache_dir:
            return None
        return os.path.join(speecht5_cache_dir, f"{sha1_key(text)}.pt")

    def build_token_id_batch(texts, add_blank=False):
        seqs = []
        for text in texts:
            ids = tok.encode(canonicalize_text(text))
            if len(ids) == 0:
                ids = [UNK_ID]
            if add_blank:
                ext = [BLANK_ID]
                for token_id in ids:
                    ext.append(int(token_id))
                    ext.append(BLANK_ID)
                ids = ext
            seqs.append([int(token_id) for token_id in ids])

        Lmax = max(max(len(seq) for seq in seqs), 1)
        ids_pad = torch.full(
            (len(seqs), Lmax),
            fill_value=PAD_ID,
            device=device,
            dtype=torch.long,
        )
        mask = torch.zeros(len(seqs), Lmax, device=device, dtype=torch.bool)
        for b, seq in enumerate(seqs):
            n = len(seq)
            ids_pad[b, :n] = torch.tensor(seq, device=device, dtype=torch.long)
            mask[b, :n] = True
        return ids_pad, mask

    def build_ctc_blank_id_batch(texts):
        """
        Build the explicit CTC source topology used by 2501.09104-style
        addBlank/repeat priors:

            text -> <blank>, c_1, <blank>, c_2, ..., c_N, <blank>

        The label vocabulary is shared with the ASR CTC head,
        so this keeps p(z_c|x,a) and q(z_c|s) on the same collapse topology.
        """
        return build_token_id_batch(texts, add_blank=True)

    def resample_text_hidden_to_ctc_topology(hidden, src_mask, dst_mask):
        """Map SpeechT5 token hidden states onto addBlank token topology.

        The CTC-blank prior still uses the explicit
        <blank>, c1, <blank>, ..., cN, <blank> duration topology, but the
        semantic features come from SpeechT5 instead of a randomly initialized
        token encoder.
        """
        B, _, H = hidden.shape
        T_dst = int(dst_mask.shape[1])
        out = hidden.new_zeros(B, T_dst, H)
        for b in range(B):
            n_src = int(src_mask[b].long().sum().item())
            n_dst = int(dst_mask[b].long().sum().item())
            if n_src <= 0 or n_dst <= 0:
                continue
            if n_src == 1:
                out[b, :n_dst] = hidden[b, :1].expand(n_dst, H)
                continue
            pos = torch.linspace(0, n_src - 1, n_dst, device=hidden.device, dtype=torch.float32)
            left = pos.floor().long()
            right = pos.ceil().long().clamp(max=n_src - 1)
            w = (pos - left.float()).to(dtype=hidden.dtype).unsqueeze(-1)
            out[b, :n_dst] = hidden[b, left] * (1.0 - w) + hidden[b, right] * w
        return out, dst_mask

    def load_wav_full_cached(path: str):
        if path in wav_cache:
            wav_cache.move_to_end(path)
            return wav_cache[path]

        wav_np, _ = librosa.load(path, sr=sampling_rate, mono=True)
        rem = wav_np.shape[0] % hop_size
        if rem != 0:
            wav_np = wav_np[:-rem]
        wav_np = librosa.util.normalize(wav_np) * 0.95
        wav_np = wav_np.astype(np.float32)

        wav_cache[path] = wav_np
        wav_cache.move_to_end(path)
        _trim_cache(wav_cache, wav_cache_max_items)
        return wav_np

    def load_logmel_full_cached(path: str, prefer_gpu: bool = False):
        if prefer_gpu and path in mel_gpu_cache:
            mel_gpu_cache.move_to_end(path)
            return mel_gpu_cache[path]
        if path in mel_cache:
            mel_cache.move_to_end(path)
            mel_cpu = mel_cache[path]
            if prefer_gpu:
                mel_gpu = _promote_mel_to_gpu(path, mel_cpu)
                if mel_gpu is not None:
                    return mel_gpu
            return mel_cpu

        cache_path = _mel_cache_path(path)
        mel_cpu = None
        if cache_path and os.path.exists(cache_path):
            if speech_backend == "svae" or str(cache_path).endswith(".npy"):
                arr = np.load(cache_path)
                mel_cpu = torch.from_numpy(arr).float().contiguous()
                if mel_cpu.ndim == 3 and mel_cpu.shape[0] == 1:
                    mel_cpu = mel_cpu[0].transpose(0, 1).contiguous()
                if mel_cpu.ndim != 2:
                    raise RuntimeError(f"Expected 2D Semantic-VAE latent, got {tuple(mel_cpu.shape)} from {cache_path}")
            else:
                mel_cpu = _load_tensor_cache(cache_path).float().contiguous()
        else:
            if speech_backend == "svae":
                raise FileNotFoundError(f"Missing Semantic-VAE latent for wav={path}; expected {cache_path}")
            wav_np = load_wav_full_cached(path)
            wav = torch.tensor(wav_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mel = get_mel_spectrogram(wav, bigvgan_h)
            mel_cpu = mel[0].detach().cpu().float().transpose(0, 1).contiguous()

        mel_cache[path] = mel_cpu
        mel_cache.move_to_end(path)
        _trim_cache(mel_cache, mel_cache_max_items)
        if prefer_gpu:
            mel_gpu = _promote_mel_to_gpu(path, mel_cpu)
            if mel_gpu is not None:
                return mel_gpu
        return mel_cpu

    def build_hubert_online_targets(wav_paths, starts, ends, target_sr=16000):
        if ssl_teacher is None:
            raise RuntimeError("SSL teacher is not initialized but SSL targets were requested")
        wav_list = []
        for path, s, e in zip(wav_paths, starts, ends):
            wav_np = load_wav_full_cached(path)
            s_samp = int(max(0, s) * hop_size)
            e_samp = int(max(s + 1, e) * hop_size)
            seg = wav_np[s_samp:e_samp]
            if seg.size <= 16:
                seg = wav_np[:max(32, min(wav_np.shape[0], 512))]
            if target_sr != sampling_rate:
                seg = librosa.resample(seg, orig_sr=sampling_rate, target_sr=target_sr)
            wav_list.append(np.asarray(seg, dtype=np.float32))
        return ssl_teacher.forward_wave_list(wav_list, sampling_rate=target_sr)

    def load_text_hidden_obj_cached(text: str, prefer_gpu: bool = False):
        if prefer_gpu and text in text_hidden_gpu_cache:
            text_hidden_gpu_cache.move_to_end(text)
            return text_hidden_gpu_cache[text]

        obj = None
        if text in text_hidden_cache:
            obj = text_hidden_cache[text]
            text_hidden_cache.move_to_end(text)
        else:
            cache_path = _speecht5_cache_path(text)
            if cache_path and os.path.exists(cache_path):
                loaded = _load_tensor_cache(cache_path)
                obj = {
                    "text": text,
                    "hidden": loaded["hidden"].float().contiguous(),
                    "mask": loaded["mask"].to(dtype=torch.bool).contiguous(),
                }
                text_hidden_cache[text] = obj
                text_hidden_cache.move_to_end(text)
                _trim_cache(text_hidden_cache, text_hidden_cache_max_items)
        if obj is not None and prefer_gpu:
            obj_gpu = _promote_text_obj_to_gpu(text, obj)
            if obj_gpu is not None:
                return obj_gpu
        return obj

    def encode_text_batch(texts):
        texts = [canonicalize_text(text) for text in texts]

        if trainable_text_encoder is not None:
            ids_pad, mask = build_token_id_batch(
                texts,
                add_blank=use_ctc_blank_repeat_prior,
            )
            h_enc = trainable_text_encoder(ids_pad, mask)
            align_mu_tok, align_logvar_tok = text_prior(h_enc)
            if canonical_prior is not None:
                mu_tok, logvar_tok = canonical_prior(h_enc)
            else:
                mu_tok, logvar_tok = align_mu_tok, align_logvar_tok
            return h_enc, mask, mu_tok, logvar_tok, align_mu_tok, align_logvar_tok

        if use_ctc_blank_repeat_prior and not ctc_blank_use_speecht5:
            if ctc_blank_embed is None or ctc_blank_encoder is None:
                raise RuntimeError("ctc_blank_repeat prior requested but CTC-blank text modules are not initialized")
            ids_pad, mask = build_ctc_blank_id_batch(texts)
            hidden = ctc_blank_embed(ids_pad)
            h_enc = ctc_blank_encoder(hidden, mask)
            align_mu_tok, align_logvar_tok = text_prior(h_enc)
            if canonical_prior is not None:
                mu_tok, logvar_tok = canonical_prior(h_enc)
            else:
                mu_tok, logvar_tok = align_mu_tok, align_logvar_tok
            return h_enc, mask, mu_tok, logvar_tok, align_mu_tok, align_logvar_tok

        if st5 is None:
            raise RuntimeError("SpeechT5 text encoder is disabled; set loss.alignment_prior_mode='ctc_blank_repeat' or enable SpeechT5")

        cached_objs = [load_text_hidden_obj_cached(text, prefer_gpu=gpu_text_cache) for text in texts]
        missing_indices = [idx for idx, obj in enumerate(cached_objs) if obj is None]

        if missing_indices:
            missing_texts = [texts[idx] for idx in missing_indices]
            with torch.no_grad():
                hidden_missing, mask_missing = st5(missing_texts)
            for local_idx, batch_idx in enumerate(missing_indices):
                text = texts[batch_idx]
                L = int(mask_missing[local_idx].long().sum().item())
                obj_cpu = {
                    "text": text,
                    "hidden": hidden_missing[local_idx, :L].detach().cpu().float().contiguous(),
                    "mask": torch.ones(L, dtype=torch.bool),
                }
                text_hidden_cache[text] = obj_cpu
                text_hidden_cache.move_to_end(text)
                _trim_cache(text_hidden_cache, text_hidden_cache_max_items)
                obj_gpu = _promote_text_obj_to_gpu(text, obj_cpu)
                cached_objs[batch_idx] = obj_gpu if obj_gpu is not None else obj_cpu

        lengths = [int(obj["mask"].long().sum().item()) for obj in cached_objs]
        Lmax = max(lengths) if lengths else 1
        use_low_precision_hidden = gpu_text_cache and torch.is_autocast_enabled() and gpu_text_cache_dtype != torch.float32
        hidden_dtype = gpu_text_cache_dtype if use_low_precision_hidden else torch.float32
        hidden = torch.zeros(len(cached_objs), Lmax, H_text, device=device, dtype=hidden_dtype)
        mask = torch.zeros(len(cached_objs), Lmax, device=device, dtype=torch.bool)
        for b, obj in enumerate(cached_objs):
            L = lengths[b]
            hidden[b, :L] = obj["hidden"][:L].to(device=device, dtype=hidden_dtype)
            mask[b, :L] = obj["mask"][:L].to(device=device, dtype=torch.bool)

        if hidden.dtype != torch.float32 and not torch.is_autocast_enabled():
            hidden = hidden.float()

        h_enc = hidden
        if adapter is not None:
            h_enc = adapter(h_enc, mask)
        if use_ctc_blank_repeat_prior:
            _, ctc_blank_mask = build_ctc_blank_id_batch(texts)
            h_enc, mask = resample_text_hidden_to_ctc_topology(h_enc, mask, ctc_blank_mask)
        align_mu_tok, align_logvar_tok = text_prior(h_enc)
        if canonical_prior is not None:
            mu_tok, logvar_tok = canonical_prior(h_enc)
        else:
            mu_tok, logvar_tok = align_mu_tok, align_logvar_tok
        return h_enc, mask, mu_tok, logvar_tok, align_mu_tok, align_logvar_tok

    def preload_mel_gpu_cache(paths):
        if not gpu_mel_preload:
            return
        unique_paths = list(OrderedDict.fromkeys(paths).keys())
        print(f"[CACHE] preloading mel cache to GPU for up to {len(unique_paths)} wavs")
        for idx, wav_path in enumerate(unique_paths, start=1):
            mel_obj = load_logmel_full_cached(wav_path, prefer_gpu=True)
            if not (torch.is_tensor(mel_obj) and mel_obj.device.type == "cuda"):
                print(f"[CACHE] stop mel preload at {idx - 1} items; VRAM budget reached")
                break
            if idx % 1000 == 0 or idx == len(unique_paths):
                print(
                    f"[CACHE] mel preload {idx}/{len(unique_paths)} "
                    f"({bytes_to_gib(gpu_cache_state['mel_bytes']):.2f} GiB resident)"
                )
        print(
            f"[CACHE] mel GPU resident items={len(mel_gpu_cache)} "
            f"bytes={bytes_to_gib(gpu_cache_state['mel_bytes']):.2f} GiB"
        )

    def preload_text_gpu_cache(texts):
        if not gpu_text_preload or st5 is None:
            return
        unique_texts = list(OrderedDict.fromkeys(texts).keys())
        print(f"[CACHE] preloading text cache to GPU for up to {len(unique_texts)} texts")
        for idx, text in enumerate(unique_texts, start=1):
            obj = load_text_hidden_obj_cached(text, prefer_gpu=True)
            if obj is None:
                continue
            if not (torch.is_tensor(obj["hidden"]) and obj["hidden"].device.type == "cuda"):
                print(f"[CACHE] stop text preload at {idx - 1} items; VRAM budget reached")
                break
            if idx % 10000 == 0 or idx == len(unique_texts):
                print(
                    f"[CACHE] text preload {idx}/{len(unique_texts)} "
                    f"({bytes_to_gib(gpu_cache_state['text_bytes']):.2f} GiB resident)"
                )
        print(
            f"[CACHE] text GPU resident items={len(text_hidden_gpu_cache)} "
            f"bytes={bytes_to_gib(gpu_cache_state['text_bytes']):.2f} GiB"
        )

    print(f"Computing mel mean/std from unique parent wavs ... mode={mel_stats_mode}")
    unique_wavs = list(OrderedDict.fromkeys([row["parent_wav"] for row in cut_rows]).keys())
    if stats_max_unique_wavs is not None and len(unique_wavs) > int(stats_max_unique_wavs):
        random.shuffle(unique_wavs)
        unique_wavs = unique_wavs[:int(stats_max_unique_wavs)]

    count = 0
    mean = np.zeros((D_mel,), dtype=np.float64)
    M2 = np.zeros((D_mel,), dtype=np.float64)
    scalar_count = 0
    scalar_sum = 0.0
    scalar_sumsq = 0.0
    for wav_path in unique_wavs:
        x = load_logmel_full_cached(wav_path).numpy().astype(np.float64)
        if x.size == 0:
            continue
        scalar_count += int(x.size)
        scalar_sum += float(x.sum())
        scalar_sumsq += float(np.square(x).sum())
        b_n = x.shape[0]
        b_mean = x.mean(axis=0)
        b_var = x.var(axis=0, ddof=0)
        if count == 0:
            mean = b_mean
            M2 = b_var * b_n
            count = b_n
        else:
            delta = b_mean - mean
            tot = count + b_n
            mean = mean + delta * (b_n / tot)
            M2 = M2 + b_var * b_n + (delta * delta) * (count * b_n / tot)
            count = tot

    if mel_stats_mode == "scalar":
        scalar_mean = scalar_sum / max(scalar_count, 1)
        scalar_var = scalar_sumsq / max(scalar_count, 1) - scalar_mean * scalar_mean
        scalar_std = max(float(np.sqrt(max(scalar_var, 0.0) + 1e-8)), 0.2)
        mean = np.full((D_mel,), scalar_mean, dtype=np.float64)
        std = np.full((D_mel,), scalar_std, dtype=np.float64)
        print(f"mel scalar stats: mean={scalar_mean:.6f} std={scalar_std:.6f} frames={count} elems={scalar_count}")
    else:
        var = M2 / max(count, 1)
        std = np.sqrt(var + 1e-8)
        std = np.maximum(std, 0.2)
        print(
            "mel per-bin stats: "
            f"mean[min/med/max]={float(mean.min()):.6f}/{float(np.median(mean)):.6f}/{float(mean.max()):.6f} "
            f"std[min/med/max]={float(std.min()):.6f}/{float(np.median(std)):.6f}/{float(std.max()):.6f} "
            f"frames={count}"
        )
    mu_g = torch.tensor(mean, dtype=torch.float32).view(1, 1, D_mel)
    std_g = torch.tensor(std, dtype=torch.float32).view(1, 1, D_mel)
    mu_b = mu_g.to(device)
    std_b = std_g.to(device)
    print("global mu/std ready:", tuple(mu_b.shape), tuple(std_b.shape))

    sampler_num_buckets = num_length_buckets if enable_length_bucket else 1
    tts_sampler = LengthBucketBatchSampler(
        tts_pool,
        batch_size_tts,
        length_key="cut_mel_len",
        num_buckets=sampler_num_buckets,
        seed=int(cfg["train"]["seed"]) + 11,
        max_frames_per_batch=max_frames_per_batch_tts,
        max_utts_per_batch=max_utts_per_batch_tts,
    )
    asr_sampler = LengthBucketBatchSampler(
        asr_pool,
        batch_size_asr,
        length_key="cut_mel_len",
        num_buckets=sampler_num_buckets,
        seed=int(cfg["train"]["seed"]) + 29,
        max_frames_per_batch=max_frames_per_batch_asr,
        max_utts_per_batch=max_utts_per_batch_asr,
    )
    if enable_length_bucket:
        tts_batch_stats = tts_sampler.batch_stats()
        asr_batch_stats = asr_sampler.batch_stats()
        print(
            f"[BUCKET] built tts_buckets={len(tts_sampler.buckets)} "
            f"asr_buckets={len(asr_sampler.buckets)} "
            f"tts_max_frames={max_frames_per_batch_tts} asr_max_frames={max_frames_per_batch_asr} "
            f"tts_max_utts={max_utts_per_batch_tts} asr_max_utts={max_utts_per_batch_asr}"
        )
        print(
            "[BUCKET-STATS] "
            f"tts_batches={tts_batch_stats['batches']} "
            f"tts_B_mean={tts_batch_stats['utts_mean']:.2f} "
            f"tts_frames_mean={tts_batch_stats['frames_mean']:.1f} "
            f"tts_padded_mean={tts_batch_stats['padded_frames_mean']:.1f} "
            f"asr_batches={asr_batch_stats['batches']} "
            f"asr_B_mean={asr_batch_stats['utts_mean']:.2f} "
            f"asr_frames_mean={asr_batch_stats['frames_mean']:.1f} "
            f"asr_padded_mean={asr_batch_stats['padded_frames_mean']:.1f}"
        )

    def sample_cut_rows(pool, bs):
        if len(pool) >= bs:
            return random.sample(pool, bs)
        return random.choices(pool, k=bs)

    def round_up_to_multiple(x, multiple):
        x = int(x)
        multiple = int(multiple)
        if multiple <= 1:
            return x
        return int(((x + multiple - 1) // multiple) * multiple)

    def build_batch_from_cut_rows(batch_rows):
        z_list = []
        K_list = []
        texts_b = []
        spk_ids = []
        wav_paths = []
        ref_wav_paths = []
        starts = []
        ends = []
        loss_masks = []
        row_metas = []

        for row in batch_rows:
            wav_path = row.get("parent_wav", row.get("wav"))
            s = int(row.get("cut_start_mel", row.get("ctx_mel_start", 0)))
            e = int(row.get("cut_end_mel", row.get("ctx_mel_end", 0)))
            txt = canonicalize_text(row.get("text_norm", row.get("text_norm_ctx", row.get("text", ""))))
            spk = str(row["speaker"])

            mel_full = load_logmel_full_cached(wav_path, prefer_gpu=gpu_mel_cache)
            K0 = int(mel_full.shape[0])
            s = max(0, min(s, K0 - 1))
            if e <= 0:
                e = K0
            e = max(s + 1, min(e, K0))
            mel_seg = mel_full[s:e]
            K = int(mel_seg.shape[0])

            core_s = int(row.get("core_start_in_ctx", 0))
            core_e = int(row.get("core_end_in_ctx", K))
            core_s = max(0, min(core_s, K))
            core_e = max(core_s + 1, min(core_e, K)) if K > 0 else 0
            loss_mask = torch.zeros(K, dtype=torch.bool, device=mel_seg.device)
            if K > 0:
                loss_mask[core_s:core_e] = True

            z_list.append(mel_seg)
            K_list.append(K)
            texts_b.append(txt)
            spk_ids.append(int(spk2id[spk]))
            wav_paths.append(wav_path)
            ref_wav_path = wav_path
            if zero_shot_enable and zero_shot_train_ref_source == "same_speaker":
                candidates = train_ref_wavs_by_spk.get(spk) or [wav_path]
                other_candidates = [path for path in candidates if path != wav_path]
                ref_wav_path = random.choice(other_candidates or candidates)
            ref_wav_paths.append(ref_wav_path)
            starts.append(s)
            ends.append(e)
            loss_masks.append(loss_mask)
            row_metas.append(row)

        B = len(z_list)
        Kmax = round_up_to_multiple(max(K_list), pad_to_multiple)
        batch_device = device if gpu_mel_cache else "cpu"
        zS_log = torch.zeros(B, Kmax, D_mel, dtype=torch.float32, device=batch_device)
        maskK = torch.zeros(B, Kmax, dtype=torch.bool, device=batch_device)
        loss_maskK = torch.zeros(B, Kmax, dtype=torch.bool, device=batch_device)
        for b, mel_seg in enumerate(z_list):
            K = int(mel_seg.shape[0])
            zS_log[b, :K] = mel_seg.to(device=batch_device, dtype=torch.float32)
            maskK[b, :K] = True
            loss_maskK[b, :K] = loss_masks[b].to(device=batch_device, dtype=torch.bool)

        return (
            zS_log if batch_device == device else zS_log.to(device),
            maskK if batch_device == device else maskK.to(device),
            loss_maskK if batch_device == device else loss_maskK.to(device),
            K_list,
            texts_b,
            torch.tensor(spk_ids, dtype=torch.long, device=device),
            wav_paths,
            ref_wav_paths,
            starts,
            ends,
            row_metas,
        )

    def build_batch_from_aligned_rows(batch_rows):
        z_list = []
        K_list = []
        texts_b = []
        wav_paths = []
        spk_ids = []

        for row in batch_rows:
            wav_path = row["wav"]
            txt = canonicalize_text(row.get("text_norm", row.get("text_raw", "")))
            spk = _row_speaker(row)
            mel_full = load_logmel_full_cached(wav_path, prefer_gpu=gpu_mel_cache)
            K0 = int(mel_full.shape[0])
            if K0 <= 0:
                continue
            z_list.append(mel_full)
            K_list.append(K0)
            texts_b.append(txt)
            wav_paths.append(wav_path)
            if spk not in spk2id:
                if asr_use_spk_cond and asr_spk_unknown == "error":
                    raise RuntimeError(f"Full ASR aux speaker {spk!r} is not in spk2id")
                spk_ids.append(-1)
            else:
                spk_ids.append(int(spk2id[spk]))

        if len(z_list) == 0:
            raise RuntimeError("No valid aligned rows for full ASR auxiliary batch")

        B = len(z_list)
        Kmax = round_up_to_multiple(max(K_list), pad_to_multiple)
        batch_device = device if gpu_mel_cache else "cpu"
        zS_log = torch.zeros(B, Kmax, D_mel, dtype=torch.float32, device=batch_device)
        maskK = torch.zeros(B, Kmax, dtype=torch.bool, device=batch_device)
        for b, mel_full in enumerate(z_list):
            K = int(mel_full.shape[0])
            zS_log[b, :K] = mel_full[:K].to(device=batch_device, dtype=torch.float32)
            maskK[b, :K] = True

        return (
            zS_log if batch_device == device else zS_log.to(device),
            maskK if batch_device == device else maskK.to(device),
            K_list,
            texts_b,
            torch.tensor(spk_ids, dtype=torch.long, device=device),
            wav_paths,
        )

    use_dataloader_runtime = bool(use_dataloader and mel_cache_dir)
    if use_dataloader and not use_dataloader_runtime:
        print("[LOADER] disabled at runtime because mel_cache_dir is missing.")
    if use_dataloader_runtime and gpu_mel_preload:
        print("[LOADER] cut batches use worker-side mel loads; main-process mel GPU preload will still serve full-teacher/full-aux/demo paths.")

    def build_cut_loader(rows, sampler):
        dataset = CutDataset(
            rows,
            spk2id=spk2id,
            mel_cache_dir=mel_cache_dir,
            mel_cache_max_items=dataset_mel_worker_cache_max_items,
            sample_reference_wav=bool(zero_shot_enable and zero_shot_train_ref_source == "same_speaker"),
        )
        loader_kwargs = dict(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=max(0, int(loader_num_workers)),
            pin_memory=bool(loader_pin_memory),
            collate_fn=collate_cut_batch,
        )
        if int(loader_num_workers) > 0:
            loader_kwargs["persistent_workers"] = bool(loader_persistent_workers)
            loader_kwargs["prefetch_factor"] = int(loader_prefetch_factor)
        return DataLoader(**loader_kwargs)

    def next_loader_batch(loader, iterator_ref):
        try:
            batch = next(iterator_ref)
            return batch, iterator_ref
        except StopIteration:
            iterator_ref = iter(loader)
            batch = next(iterator_ref)
            return batch, iterator_ref

    def move_cut_batch_to_device(batch):
        return (
            batch["zS_log"].to(device, non_blocking=bool(loader_pin_memory)),
            batch["maskK"].to(device, non_blocking=bool(loader_pin_memory)),
            batch["loss_maskK"].to(device, non_blocking=bool(loader_pin_memory)),
            list(batch["K_list"]),
            list(batch["texts"]),
            batch["spk_ids"].to(device, non_blocking=bool(loader_pin_memory)),
            list(batch["wav_paths"]),
            list(batch.get("ref_wav_paths", batch["wav_paths"])),
            list(batch["starts"]),
            list(batch["ends"]),
            list(batch.get("row_metas", [{} for _ in batch["K_list"]])),
        )

    tts_loader = None
    asr_loader = None
    tts_iter = None
    asr_iter = None
    if use_dataloader_runtime:
        tts_loader = build_cut_loader(tts_pool, tts_sampler)
        asr_loader = build_cut_loader(asr_pool, asr_sampler)
        tts_iter = iter(tts_loader)
        asr_iter = iter(asr_loader)
        print(
            f"[LOADER] runtime_enabled=True workers={loader_num_workers} "
            f"pin_memory={loader_pin_memory} persistent_workers={loader_persistent_workers} "
            f"prefetch_factor={loader_prefetch_factor if loader_num_workers > 0 else 'n/a'} "
            f"tts_batches={len(tts_sampler)} asr_batches={len(asr_sampler)}"
        )

    trainable_text_encoder = None
    if text_encoder_type in {"trainable", "trainable_token"}:
        trainable_text_encoder = TrainableTokenTextEncoder(
            vocab_size=Vt,
            dim=text_encoder_dim,
            layers=text_encoder_layers,
            n_heads=text_encoder_heads,
            ff_mult=text_encoder_ff_mult,
            conv_ksize=text_encoder_conv_ksize,
            dropout=text_encoder_dropout,
            max_len=text_encoder_max_len,
            padding_idx=PAD_ID,
        ).to(device)
        print(
            "[TEXT] trainable token encoder "
            f"dim={text_encoder_dim} layers={text_encoder_layers} heads={text_encoder_heads} "
            f"vocab={Vt} max_len={text_encoder_max_len}"
        )

    adapter = None
    if (
        text_encoder_type == "speecht5"
        and use_adapter
        and (not use_ctc_blank_repeat_prior or ctc_blank_use_speecht5)
    ):
        if adapter_type in {"residual", "residual_adapter"}:
            adapter = ResidualAdapter(H_text, bottleneck=adapter_bottleneck, dropout=adapter_dropout).to(device)
        elif adapter_type in {"canonical_text_encoder", "canonical"}:
            adapter = CanonicalTextEncoder(
                H_text,
                layers=canonical_text_layers,
                n_heads=canonical_text_heads,
                ff_mult=canonical_text_ff_mult,
                conv_ksize=canonical_text_conv_ksize,
                dropout=adapter_dropout,
                residual_scale=canonical_text_residual_scale,
            ).to(device)
        else:
            raise ValueError(f"Unsupported adapter_type={adapter_type}")
    ctc_blank_embed = None
    ctc_blank_encoder = None
    if (
        text_encoder_type == "speecht5"
        and use_ctc_blank_repeat_prior
        and not ctc_blank_use_speecht5
    ):
        ctc_blank_embed = nn.Embedding(Vt, H_text, padding_idx=0).to(device)
        ctc_blank_encoder = CanonicalTextEncoder(
            H_text,
            layers=ctc_blank_text_layers,
            n_heads=ctc_blank_text_heads,
            ff_mult=ctc_blank_text_ff_mult,
            conv_ksize=ctc_blank_text_conv_ksize,
            dropout=ctc_blank_text_dropout,
            residual_scale=ctc_blank_text_residual_scale,
        ).to(device)
    text_prior = TextPriorHead(in_dim=H_text, hidden=256, out_dim=D_mel, logvar_bias=-2.0).to(device)
    canonical_prior = None
    canonical_to_source = None
    source_to_canonical = None
    canonical_posterior = None
    ctc_input_dim = D_mel
    vf_text_cond_dim = 0
    if use_true_canonical_latent:
        canonical_prior = TextPriorHead(
            in_dim=H_text,
            hidden=canonical_hidden,
            out_dim=canonical_dim,
            logvar_bias=-2.0,
        ).to(device)
        canonical_to_source = CanonicalToSource(
            c_dim=canonical_dim,
            spk_dim=E_spk,
            style_dim=tts_style_dim if (use_tts_style_latent and tts_style_into_source) else 0,
            out_dim=D_mel,
            hidden=canonical_hidden,
            dropout=canonical_dropout,
        ).to(device)
        source_to_canonical = SourceToCanonical(
            in_dim=D_mel,
            c_dim=canonical_dim,
            hidden=canonical_hidden,
            dropout=canonical_dropout,
        ).to(device)
        ctc_input_dim = canonical_dim
        vf_text_cond_dim = canonical_dim if use_vf_canonical_text_cond else 0
    if canonical_match_mode == "kl":
        canonical_posterior = CanonicalPosterior(
            dim=ctc_input_dim,
            hidden=canonical_post_hidden,
            dropout=canonical_post_dropout,
            logvar_bias=canonical_post_logvar_bias,
        ).to(device)
    dur_pred = FastSpeech2DurationPredictor(D=H_text, hidden=int(model_cfg["dur_hidden"]), ksize=3, dropout=float(model_cfg["dur_dropout"])).to(device)
    len_pred = LengthPredictor(D=H_text, hidden=len_hidden).to(device) if use_len_predictor else None
    spk_table = SpeakerConditioner(
        n_spk=n_spk,
        E=E_spk,
        scale=spk_scale,
        mode=speaker_cond_type,
        pretrained_emb=speaker_bank,
        pretrained_trainable=speaker_emb_trainable,
        delta_scale=speaker_delta_scale,
        use_layernorm=speaker_cond_layernorm,
    ).to(device)

    zero_shot_ref_encoder = None
    if zero_shot_enable:
        if speaker_cond_type in SpeakerConditioner.TABLE_MODES:
            raise ValueError("model.zero_shot.enable=true requires a pretrained/ecapa speaker_cond_type")
        zero_shot_ref_encoder = ReferenceSpeakerEncoder(
            model_name=zero_shot_ref_model,
            savedir=zero_shot_ref_savedir,
            cache_dir=zero_shot_ref_cache_dir,
            device=device,
            sampling_rate=sampling_rate,
            max_sec=zero_shot_ref_max_sec,
            l2_normalize=zero_shot_ref_l2_normalize,
        )
        print(
            f"[ZERO-SHOT] enabled ref_model={zero_shot_ref_model} "
            f"savedir={zero_shot_ref_savedir} cache_dir={zero_shot_ref_cache_dir} "
            f"train_ref_source={zero_shot_train_ref_source} asr_ref_source={zero_shot_asr_ref_source}"
        )

    def speaker_cond_from_ref_paths(ref_wav_paths, spk_ids: torch.LongTensor | None = None, *, dtype=None):
        if zero_shot_ref_encoder is None:
            if spk_ids is None:
                raise RuntimeError("speaker ids are required when zero-shot speaker conditioning is disabled")
            e = spk_table(spk_ids)
        else:
            raw = zero_shot_ref_encoder.encode_paths(ref_wav_paths, device=device)
            e = spk_table.from_pretrained_embedding(raw)
        if dtype is not None:
            e = e.to(dtype=dtype)
        return e

    def speaker_cond_from_name(spk_name: str, *, dtype=None):
        spk_name = str(spk_name)
        if zero_shot_ref_encoder is not None:
            ref_path = train_ref_wav_by_spk.get(spk_name)
            if not ref_path:
                raise RuntimeError(f"No zero-shot reference wav found for speaker {spk_name!r}")
            return speaker_cond_from_ref_paths([ref_path], dtype=dtype)
        spk_id = torch.tensor([spk2id[spk_name]], device=device, dtype=torch.long)
        return speaker_cond_from_ref_paths([train_ref_wav_by_spk.get(spk_name, "")], spk_id, dtype=dtype)

    def asr_spk_cond_from_ids(spk_ids: torch.LongTensor, *, dtype=None):
        if not asr_use_spk_cond:
            return None
        valid = spk_ids >= 0
        if asr_spk_unknown == "error" and bool((~valid).any().detach().cpu().item()):
            raise RuntimeError("ASR speaker conditioning requested with unknown speaker ids")
        safe_ids = torch.where(valid, spk_ids, torch.zeros_like(spk_ids))
        e = spk_table(safe_ids)
        e = torch.where(valid[:, None], e, torch.zeros_like(e))
        if dtype is not None:
            e = e.to(dtype=dtype)
        return e * float(asr_spk_scale)

    def asr_spk_cond_from_ref_paths(ref_wav_paths, spk_ids: torch.LongTensor | None = None, *, dtype=None):
        if not asr_use_spk_cond:
            return None
        e = speaker_cond_from_ref_paths(ref_wav_paths, spk_ids, dtype=dtype)
        return e * float(asr_spk_scale)

    def asr_spk_cond_from_name(spk_name: str, *, dtype=None):
        if not asr_use_spk_cond:
            return None
        spk_name = str(spk_name)
        if spk_name not in spk2id:
            if asr_spk_unknown == "error":
                raise RuntimeError(f"ASR speaker conditioning requested but speaker {spk_name!r} is not in spk2id")
            return None
        if zero_shot_ref_encoder is not None:
            ref_path = train_ref_wav_by_spk.get(spk_name)
            if not ref_path:
                if asr_spk_unknown == "error":
                    raise RuntimeError(f"No zero-shot ASR reference wav found for speaker {spk_name!r}")
                return None
            return asr_spk_cond_from_ref_paths([ref_path], dtype=dtype)
        spk_id = torch.tensor([spk2id[spk_name]], device=device, dtype=torch.long)
        return asr_spk_cond_from_ids(spk_id, dtype=dtype)

    def tts_vf_spk_cond(spk_e):
        if vf_use_speaker_cond or spk_e is None:
            return spk_e
        return torch.zeros_like(spk_e)

    def asr_vf_spk_cond(spk_e):
        if asr_vf_use_speaker_cond or spk_e is None:
            return spk_e
        return torch.zeros_like(spk_e)

    def asr_cfg_flag_value(spk_e):
        if asr_use_style_cond:
            return 1
        return 1 if (asr_vf_use_speaker_cond and spk_e is not None) else 0

    def asr_style_cond_from_source(zS, maskK, spk_e=None, *, dtype=None):
        if (not asr_use_style_cond) or (tts_style_post is None):
            return None
        B = zS.shape[0]
        zS_in = zS.to(dtype=dtype) if dtype is not None else zS
        spk_for_post = spk_e
        if spk_for_post is None:
            spk_for_post = torch.zeros(B, E_spk, device=zS.device, dtype=zS_in.dtype)
        else:
            spk_for_post = spk_for_post.to(device=zS.device, dtype=zS_in.dtype)

        if tts_style_post_mode == "path":
            t_asr = torch.ones(B, device=zS.device, dtype=zS_in.dtype)
            u_mu, u_logvar = tts_style_post(
                zS_in,
                maskK,
                z_t=zS_in,
                t=t_asr,
                spk_e=spk_for_post,
            )
        else:
            u_mu, u_logvar = tts_style_post(zS_in, maskK)

        if asr_style_use_mean or asr_style_temp <= 0.0:
            u = u_mu
        else:
            u = u_mu + float(asr_style_temp) * torch.exp(0.5 * u_logvar) * torch.randn_like(u_mu)
        if asr_style_detach:
            u = u.detach()
        if dtype is not None:
            u = u.to(dtype=dtype)
        return u

    tts_source_cond = SourceStatsConditioner(D=D_mel, hidden=tts_source_cond_hidden, out_dim=E_spk).to(device) if use_tts_source_cond else None
    tts_style_post = None
    tts_style_pair_post = None
    tts_style_prior = None
    tts_style_to_source = None
    if use_tts_style_latent:
        tts_style_post = TTSStylePosterior(
            D=D_mel,
            spk_dim=E_spk,
            latent_dim=tts_style_dim,
            hidden=tts_style_hidden,
            dropout=tts_style_dropout,
            mode=tts_style_post_mode,
        ).to(device)
        if use_tts_style_pair_posterior:
            tts_style_pair_post = TTSStylePairPosterior(
                s_dim=D_mel,
                c_dim=ctc_input_dim,
                spk_dim=E_spk,
                latent_dim=tts_style_dim,
                hidden=tts_style_hidden,
                dropout=tts_style_dropout,
            ).to(device)
        if tts_style_prior_type == "speaker":
            tts_style_prior = TTSStylePrior(
                spk_dim=E_spk,
                latent_dim=tts_style_dim,
                hidden=tts_style_hidden,
                dropout=tts_style_dropout,
            ).to(device)
        elif tts_style_prior_type == "canonical_speaker":
            tts_style_prior = TTSStyleCanonicalPrior(
                c_dim=ctc_input_dim,
                spk_dim=E_spk,
                latent_dim=tts_style_dim,
                hidden=tts_style_hidden,
                dropout=tts_style_dropout,
                logvar_bias=tts_style_prior_logvar_bias,
            ).to(device)
        if tts_style_into_source and abs(float(tts_style_source_scale)) > 0.0:
            tts_style_to_source = TTSStyleToSource(
                latent_dim=tts_style_dim,
                spk_dim=E_spk,
                out_dim=D_mel,
                hidden=tts_style_hidden,
            ).to(device)

    def tts_style_prior_dist(spk_e, zc=None, maskK=None, *, dtype=None):
        if not use_tts_style_latent:
            return None, None
        spk_base = spk_e
        if spk_base is None:
            if zc is None:
                raise RuntimeError("tts_style_prior_dist requires spk_e or zc to infer batch/device")
            spk_base = torch.zeros(zc.shape[0], E_spk, device=zc.device, dtype=zc.dtype)
        if dtype is not None:
            spk_base = spk_base.to(dtype=dtype)
        if tts_style_prior is None:
            u_mu_p = torch.zeros(
                spk_base.shape[0],
                tts_style_dim,
                device=spk_base.device,
                dtype=spk_base.dtype,
            )
            return u_mu_p, torch.zeros_like(u_mu_p)
        if tts_style_prior_type == "canonical_speaker":
            if zc is None or maskK is None:
                raise RuntimeError("model.tts_style_prior_type='canonical_speaker' requires zc and maskK")
            zc_prior = zc.detach() if tts_style_prior_canonical_detach else zc
            if dtype is not None:
                zc_prior = zc_prior.to(dtype=dtype)
            return tts_style_prior(zc_prior, maskK, spk_base.to(device=zc_prior.device, dtype=zc_prior.dtype))
        return tts_style_prior(spk_base)

    vf = DiTVectorField(
        D=D_mel,
        E_spk=E_spk,
        style_dim=tts_style_dim if use_tts_style_latent else 0,
        text_cond_dim=vf_text_cond_dim,
        hidden=int(model_cfg["vf_hidden"]),
        depth=int(model_cfg["vf_depth"]),
        n_heads=int(model_cfg["vf_heads"]),
        dropout=float(model_cfg["vf_dropout"]),
        max_len=int(model_cfg["vf_max_len"]),
        condition_injection=vf_condition_injection,
    ).to(device)
    # ZU-VF variant: speaker/reference information can shape z_u through
    # p(zu|zc,r) and q(zu|zs,zc,r), but the shared vector field itself can be
    # forced to ignore direct r/spk_e conditioning.
    vf.direct_speaker_cond = bool(vf_use_speaker_cond or asr_vf_use_speaker_cond)
    mel_refiner = None
    if use_refiner:
        mel_refiner = TextCondRefiner1xResidualPostNet(
            D=D_mel,
            hidden=int(model_cfg["ref_hidden"]),
            cond_dim=H_text,
            n_blocks=int(model_cfg["ref_blocks"]),
            ksize=int(model_cfg["ref_ksize"]),
            dropout=float(model_cfg["ref_dropout"]),
        ).to(device)

    def build_ctc_head(input_dim: int):
        if ctc_head_type == "baseline":
            return BaselineCTCHead(
                V=Vt,
                D=input_dim,
                hidden=int(model_cfg["ctc_hidden"]),
                conv_layers=int(model_cfg["ctc_layers"]),
                ksize=int(model_cfg["ctc_ksize"]),
                lstm_hidden=int(model_cfg.get("ctc_lstm_hidden", 384)),
                lstm_layers=int(model_cfg.get("ctc_lstm_layers", 2)),
                dropout=float(model_cfg["ctc_dropout"]),
            ).to(device)
        if ctc_head_type == "conv":
            return FrameCTCConvHead(
                V=Vt,
                D=input_dim,
                hidden=int(model_cfg["ctc_hidden"]),
                layers=int(model_cfg["ctc_layers"]),
                ksize=int(model_cfg["ctc_ksize"]),
            ).to(device)
        if ctc_head_type == "zipformer":
            return ZipformerCTCHead(
                V=Vt,
                D=input_dim,
                hidden=int(model_cfg["ctc_hidden"]),
                layers=int(model_cfg["ctc_layers"]),
                heads=int(model_cfg.get("ctc_heads", 8)),
                ff_mult=float(model_cfg.get("ctc_ff_mult", 4.0)),
                ksize=int(model_cfg["ctc_ksize"]),
                dropout=float(model_cfg["ctc_dropout"]),
                downsample_factor=int(model_cfg.get("ctc_zipformer_downsample", 2)),
            ).to(device)
        raise AssertionError(f"unreachable ctc_head_type={ctc_head_type}")

    text_ctc_head = build_ctc_head(ctc_input_dim)
    source_ctc_head = build_ctc_head(D_mel) if enable_source_ctc else None
    dit_hidden_ctc_head = build_ctc_head(int(model_cfg["vf_hidden"])) if enable_dit_hidden_ctc else None
    att_decoder = (
        AttentionCTCDecoder(
            vocab_size=AED_VOCAB_SIZE,
            encoder_dim=ctc_input_dim,
            d_model=att_decoder_hidden,
            layers=att_decoder_layers,
            heads=att_decoder_heads,
            ff_mult=att_decoder_ff_mult,
            dropout=att_decoder_dropout,
            pad_id=PAD_ID,
            max_len=att_decoder_max_len,
        ).to(device)
        if enable_att_decoder
        else None
    )

    compile_enable = bool(runtime_cfg["compile_enable"])
    compile_mode = str(runtime_cfg["compile_mode"])
    compile_dynamic = bool(runtime_cfg["compile_dynamic"])
    compile_ssl_hidden_head = bool(runtime_cfg.get("compile_ssl_hidden_head", False))
    vf = maybe_compile(vf, compile_enable and bool(runtime_cfg["compile_vf"]), compile_mode, compile_dynamic, "vf")
    mel_refiner = maybe_compile(mel_refiner, use_refiner and compile_enable and bool(runtime_cfg.get("compile_refiner", False)), compile_mode, compile_dynamic, "mel_refiner")
    text_ctc_head = maybe_compile(text_ctc_head, compile_enable and bool(runtime_cfg["compile_ctc_head"]), compile_mode, compile_dynamic, "text_ctc_head")
    source_ctc_head = maybe_compile(
        source_ctc_head,
        enable_source_ctc and compile_enable and bool(runtime_cfg["compile_ctc_head"]),
        compile_mode,
        compile_dynamic,
        "source_ctc_head",
    )
    dit_hidden_ctc_head = maybe_compile(
        dit_hidden_ctc_head,
        enable_dit_hidden_ctc and compile_enable and bool(runtime_cfg["compile_ctc_head"]),
        compile_mode,
        compile_dynamic,
        "dit_hidden_ctc_head",
    )
    att_decoder = maybe_compile(
        att_decoder,
        enable_att_decoder and compile_enable and bool(runtime_cfg.get("compile_att_decoder", False)),
        compile_mode,
        compile_dynamic,
        "att_decoder",
    )

    ssl_teacher = None
    ssl_hidden_head = None
    ssl_teacher_dim = None
    if ssl_hidden_enable:
        ssl_teacher = FrozenHubertSSLTeacher(
            model_name=ssl_teacher_name,
            device=device,
            layer_idx=ssl_teacher_layer_idx,
        )
        ssl_teacher_dim = int(ssl_teacher.hidden_size)
        ssl_head_in_dim = ctc_input_dim if ssl_hidden_target == "zc" else int(model_cfg["vf_hidden"])
        ssl_hidden_head = ADMASpeechAlignMLP(
            in_dim=ssl_head_in_dim,
            out_dim=ssl_teacher_dim,
            hidden=ssl_hidden_head_hidden,
            pool_factors=ssl_hidden_pool_factors,
            dropout=float(model_cfg.get("ctc_dropout", 0.1)),
        ).to(device)
        ssl_hidden_head = maybe_compile(
            ssl_hidden_head,
            compile_enable and compile_ssl_hidden_head,
            compile_mode,
            compile_dynamic,
            "ssl_hidden_head",
        )

    ctc_loss_fn = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    opt_params = list(text_prior.parameters()) + list(dur_pred.parameters()) + list(spk_table.parameters()) + list(vf.parameters()) + list(text_ctc_head.parameters())
    if canonical_prior is not None:
        opt_params += list(canonical_prior.parameters())
        opt_params += list(canonical_to_source.parameters())
        opt_params += list(source_to_canonical.parameters())
    if canonical_posterior is not None:
        opt_params += list(canonical_posterior.parameters())
    if len_pred is not None:
        opt_params += list(len_pred.parameters())
    if tts_source_cond is not None:
        opt_params += list(tts_source_cond.parameters())
    if tts_style_post is not None:
        opt_params += list(tts_style_post.parameters())
        if tts_style_pair_post is not None:
            opt_params += list(tts_style_pair_post.parameters())
        if tts_style_prior is not None:
            opt_params += list(tts_style_prior.parameters())
        if tts_style_to_source is not None:
            opt_params += list(tts_style_to_source.parameters())
    if mel_refiner is not None:
        opt_params += list(mel_refiner.parameters())
    if ssl_hidden_head is not None:
        opt_params += list(ssl_hidden_head.parameters())
    if dit_hidden_ctc_head is not None:
        opt_params += list(dit_hidden_ctc_head.parameters())
    if att_decoder is not None:
        opt_params += list(att_decoder.parameters())
    if adapter is not None:
        opt_params += list(adapter.parameters())
    if trainable_text_encoder is not None:
        opt_params += list(trainable_text_encoder.parameters())
    if ctc_blank_embed is not None:
        opt_params += list(ctc_blank_embed.parameters())
    if ctc_blank_encoder is not None:
        opt_params += list(ctc_blank_encoder.parameters())
    opt = torch.optim.AdamW(opt_params, lr=lr_all)
    source_ctc_opt = (
        torch.optim.AdamW(source_ctc_head.parameters(), lr=lr_all * source_ctc_lr_scale)
        if source_ctc_head is not None
        else None
    )

    module_map = OrderedDict(
        adapter=adapter,
        trainable_text_encoder=trainable_text_encoder,
        ctc_blank_embed=ctc_blank_embed,
        ctc_blank_encoder=ctc_blank_encoder,
        text_prior=text_prior,
        canonical_prior=canonical_prior,
        canonical_to_source=canonical_to_source,
        source_to_canonical=source_to_canonical,
        canonical_posterior=canonical_posterior,
        dur_pred=dur_pred,
        len_pred=len_pred,
        spk_table=spk_table,
        tts_source_cond=tts_source_cond,
        tts_style_post=tts_style_post,
        tts_style_pair_post=tts_style_pair_post,
        tts_style_prior=tts_style_prior,
        tts_style_to_source=tts_style_to_source,
        vf=vf,
        text_ctc_head=text_ctc_head,
        source_ctc_head=source_ctc_head,
        dit_hidden_ctc_head=dit_hidden_ctc_head,
        att_decoder=att_decoder,
        mel_refiner=mel_refiner,
        ssl_hidden_head=ssl_hidden_head,
    )
    ema = MultiModuleEMA(module_map, decay=ema_decay) if use_ema else None

    start_step = 0
    resume_path = resolve_resume_path(resume_from, ckpt_dir)
    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path,
            module_map=module_map,
            optimizer=opt,
            scaler=scaler,
            ema=ema,
            device=device,
            restore_rng=True,
        )
        if source_ctc_opt is not None:
            source_opt_state = checkpoint.get("extra_state", {}).get("source_ctc_optimizer")
            if source_opt_state is not None:
                try:
                    source_ctc_opt.load_state_dict(source_opt_state)
                    move_optimizer_state_to_device(source_ctc_opt, device)
                    print("[CKPT] restored source_ctc_optimizer")
                except Exception as exc:
                    print(f"[CKPT] skipped source_ctc_optimizer restore: {exc}")
        start_step = int(checkpoint.get("step", -1)) + 1
        print(f"[CKPT] resumed from {resume_path} at step={start_step}")
    elif resume_from:
        print(f"[CKPT] requested resume checkpoint not found: {resume_from}")

    preload_mel_gpu_cache(mel_paths_all)
    preload_text_gpu_cache(text_preload_candidates)

    def get_lr_scale(step_now: int):
        if lr_schedule == "constant":
            return 1.0
        if lr_schedule != "warmup_cosine":
            return 1.0
        if lr_warmup_steps > 0 and step_now < lr_warmup_steps:
            return float(step_now + 1) / float(max(1, lr_warmup_steps))
        if total_steps <= lr_warmup_steps:
            return 1.0
        progress = float(step_now - lr_warmup_steps) / float(max(1, total_steps - lr_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(lr_min_scale) + (1.0 - float(lr_min_scale)) * cosine

    def set_optimizer_lr(step_now: int):
        lr_now = float(lr_all) * get_lr_scale(step_now)
        for group in opt.param_groups:
            group["lr"] = lr_now
        if source_ctc_opt is not None:
            for group in source_ctc_opt.param_groups:
                group["lr"] = lr_now * source_ctc_lr_scale
        return lr_now

    print("\n=== Single-VF CUT-MANIFEST repo version ===")
    print(f"[RATE] ds_factor(model)={ds_factor} ds_align(MAS)={ds_align}")
    print(f"[CTC-DS] factor={ctc_subsample_factor} apply_to={ctc_subsample_apply_to}")
    print(
        f"[BATCH] batch_size={batch_size} tts={batch_size_tts} asr={batch_size_asr} "
        f"max_frames_tts={max_frames_per_batch_tts} max_frames_asr={max_frames_per_batch_asr} "
        f"max_utts_tts={max_utts_per_batch_tts} max_utts_asr={max_utts_per_batch_asr}"
    )
    print(
        f"[SPEAKER] n_spk={n_spk} E_spk={E_spk} mode={speaker_cond_type} "
        f"spk_scale={spk_scale} spk_drop_rate={spk_drop_rate} "
        f"delta_scale={speaker_delta_scale} layernorm={speaker_cond_layernorm}"
    )
    print(f"[VF-SPK-COND] tts_fm={vf_use_speaker_cond} asr={asr_vf_use_speaker_cond}")
    print(f"[ASR-SPK-COND] enable={asr_use_spk_cond} scale={asr_spk_scale} unknown={asr_spk_unknown}")
    print(
        f"[ASR-STYLE-U] enable={asr_use_style_cond} use_mean={asr_style_use_mean} "
        f"temp={asr_style_temp} detach={asr_style_detach}"
    )
    print(f"[TTS-SRC-COND] enable={use_tts_source_cond} hidden={tts_source_cond_hidden} scale={tts_source_cond_scale}")
    print(
        f"[TTS-STYLE-U] enable={use_tts_style_latent} mode={tts_style_post_mode} "
        f"dim={tts_style_dim} hidden={tts_style_hidden} "
        f"pair_post={use_tts_style_pair_posterior} prior={tts_style_prior_type} into_source={tts_style_into_source} "
        f"src_scale={tts_style_source_scale} demo_temp={demo_style_temp} "
        f"prior_detach={tts_style_prior_canonical_detach} "
        f"prior_logvar_bias={tts_style_prior_logvar_bias:g} "
        f"w_kl_tts={w_tts_style_kl} w_kl_asr={w_tts_style_asr_kl} "
        f"asr_kl_stopgrad_pair={tts_style_asr_kl_stopgrad_pair}"
    )
    print(
        f"[TRUE-CANONICAL] enable={use_true_canonical_latent} dim={canonical_dim} "
        f"hidden={canonical_hidden} vf_text_cond={use_vf_canonical_text_cond} "
        f"vf_cond={vf_condition_injection}"
    )
    print(
        f"[UNIFIED-DATA] enable={use_processed_unified} core_loss_only={core_loss_only} "
        f"train_unit={train_unit} utterance_training={use_utterance_training} "
        f"alignment_prior={alignment_prior_mode} gt_align={use_gt_alignment_prior} "
        f"gt_dur={use_gt_duration_teacher}"
    )
    print(
        f"[ALIGN-PERTURB] enable={duration_perturb_enable} "
        f"num={duration_perturb_num} sigma={duration_perturb_sigma} "
        f"include_base={duration_perturb_include_base} "
        f"canonical_candidates={canonical_align_candidates} "
        f"softmin_tau={canonical_softmin_tau}"
    )
    print(
        f"[LATENT-VI] acoustic_prior_nll={enable_acoustic_prior_nll} "
        f"canonical_nll={enable_canonical_nll} mode={canonical_match_mode} "
        f"split={canonical_stopgrad_split} w_can={w_canonical_nll} "
        f"w_can_prior={w_canonical_prior_nll} w_can_bwd={w_canonical_bwd_nll} "
        f"start={canonical_nll_start} "
        f"anneal={canonical_nll_anneal_steps} kl_free_bits={canonical_kl_free_bits}"
    )
    if st5 is None:
        print(f"[SpeechT5] disabled; text_encoder_type={text_encoder_type} text_dim={H_text}")
    else:
        print(
            f"[SpeechT5] layer_idx={st5_layer_idx} adapter={use_adapter} type={adapter_type} "
            f"layers={canonical_text_layers if adapter_type in {'canonical_text_encoder', 'canonical'} else 'n/a'}"
        )
    if use_ctc_blank_repeat_prior:
        if ctc_blank_use_speecht5:
            print(
                f"[CTC-BLANK-PRIOR] enabled with SpeechT5+adapter hidden; vocab={Vt} "
                f"blank_id={BLANK_ID} grid=<blank>,char,<blank>,..."
            )
        else:
            print(
                f"[CTC-BLANK-PRIOR] enabled with trainable token encoder; vocab={Vt} "
                f"blank_id={BLANK_ID} layers={ctc_blank_text_layers} heads={ctc_blank_text_heads} "
                "grid=<blank>,token,<blank>,..."
            )
    print(f"[MAS] mode={mas_mode} mix_alpha={mas_mix_alpha} Gaussian score temp={mas_temp}")
    print(f"[FWD-PRIOR] mode={fwd_prior_mode} mix_alpha={fwd_prior_mix_alpha}")
    print(f"[FWD-ANCHOR] mode={fwd_anchor_mode} mix_alpha={fwd_anchor_mix_alpha}")
    full_teacher_align = "gt-duration" if use_gt_alignment_prior else alignment_prior_mode
    print(f"[TTS-FULL-TEACHER] enable={use_full_tts_teacher} every={full_tts_teacher_every} align_source={full_teacher_align}")
    print(f"[FM] w_fm={w_fm}")
    print(
        f"[PRIOR] mode={prior_loss_mode} w_prior={w_prior} "
        f"mu_loss={prior_mu_loss_type} w_mu={w_prior_mu} w_var={w_prior_var} "
        f"w_nll={w_prior_nll} fixed_logvar={prior_fixed_logvar} var_target={prior_var_reg_target}"
    )
    print(
        f"[ASR-FULL-AUX] enable={enable_full_asr_ctc_aux} w={w_ctc_full} "
        f"start={full_asr_ctc_aux_start} every={full_asr_ctc_aux_every} "
        f"batch={full_asr_ctc_aux_batch_size} steps={full_asr_ctc_aux_steps} "
        f"ode_grad={full_asr_ctc_aux_ode_grad} whole={full_asr_ctc_aux_whole_utterance}"
    )
    print(
        f"[SOURCE-CTC] enable={enable_source_ctc} w={w_ctc_source} "
        f"start={source_ctc_start} lr_scale={source_ctc_lr_scale} "
        f"head_type={ctc_head_type} detach_input=True"
    )
    print(
        f"[ZC-SAMPLE-CTC] enable={enable_zc_sample_ctc} w={w_ctc_sample} "
        f"start={ctc_sample_start} temp={ctc_sample_temp} head_type={ctc_head_type}"
    )
    print(
        f"[BWD-FM] enable={enable_bwd_fm} w={w_bwd_fm} start={bwd_fm_start} "
        f"anneal={bwd_fm_anneal_steps} t=[{bwd_fm_t_min},{bwd_fm_t_max}] "
        f"anchor={bwd_fm_anchor} mix_alpha={bwd_fm_anchor_mix_alpha} "
        "condition=none detach_endpoints=True"
    )
    print(f"[ODE-GRAD] fwd_end={fwd_ode_grad} bwd_end={bwd_ode_grad} full_asr_aux={full_asr_ctc_aux_ode_grad}")
    print(
        f"[ASR-DEMO] whole_utterance={asr_demo_whole_utterance} "
        f"chunk_core={full_asr_chunk_core} chunk_ctx={full_asr_chunk_ctx} "
        f"use_euler={full_asr_use_euler} steps={asr_demo_steps} "
        f"decode={asr_demo_decode_mode} rtf={demo_rtf}"
    )
    print(
        f"[TRAJ-DEMO] enable={demo_plot_trajectory} projection={demo_trajectory_projection} "
        f"dims={demo_trajectory_dims} pool={demo_trajectory_pool} "
        f"frames={demo_trajectory_frames} speakers={demo_trajectory_speakers} "
        f"samples={demo_trajectory_samples} reverse={demo_trajectory_reverse} "
        f"export_csv={demo_trajectory_export_csv} "
        f"paper_style={demo_trajectory_paper_style} "
        f"display_scale=({demo_trajectory_display_x_scale},{demo_trajectory_display_y_scale}) "
        f"canonical_color={demo_trajectory_canonical_color} speaker_colors={demo_trajectory_speaker_colors} "
        f"annotate={demo_trajectory_annotate_points} zu_fanout={demo_trajectory_zu_fanout} "
        f"zu_zc={demo_trajectory_zu_zc_samples} zu_u={demo_trajectory_zu_u_samples} "
        f"asr_many_to_one={demo_trajectory_asr_many_to_one} "
        f"asr_realization={demo_trajectory_asr_realization_plot} "
        f"asr_real_spk={demo_trajectory_asr_realization_speakers} "
        f"asr_real_u={demo_trajectory_asr_realization_styles}"
    )
    print(
        f"[TTS-DEMO-EVAL] utmos={demo_eval_utmos} repo={demo_utmos_repo} model={demo_utmos_model} "
        f"whisper={demo_eval_whisper} whisper_model={demo_whisper_model} "
        f"generated_wav_mel={demo_plot_generated_wav_mel} frontend={demo_generated_mel_frontend} "
        f"mel_cmap={demo_mel_plot_cmap} mel_top_pct={demo_mel_plot_top_percentile} "
        f"mel_dyn_db={demo_mel_plot_dynamic_range_db}"
    )
    print(f"[CTC-DUR] enable={enable_ctc_dur} w={w_ctc_dur} start={ctc_dur_start}")
    print(f"[STAT] use={use_stat_match} w_stat={w_stat}")
    print(f"[CTC] head_type={ctc_head_type}")
    print(
        f"[DiT-HIDDEN-CTC] enable={enable_dit_hidden_ctc} w={w_dit_hidden_ctc} "
        f"start={dit_hidden_ctc_start} anneal={dit_hidden_ctc_anneal_steps} "
        f"tap={dit_hidden_ctc_tap_index} t=[{dit_hidden_ctc_t_min},{dit_hidden_ctc_t_max}] "
        f"anchor={dit_hidden_ctc_anchor} subsample={dit_hidden_ctc_apply_subsample}"
    )
    print(
        f"[ATT-DECODER] enable={enable_att_decoder} w={w_att_decoder} "
        f"start={att_decoder_start} anneal={att_decoder_anneal_steps} "
        f"hidden={att_decoder_hidden} layers={att_decoder_layers} heads={att_decoder_heads} "
        f"label_smoothing={att_decoder_label_smoothing} detach_input={att_decoder_detach_input}"
    )
    print(
        f"[SSL-HIDDEN] enable={ssl_hidden_enable} target={ssl_hidden_target} "
        f"w={ssl_hidden_w} start={ssl_hidden_start} tap={ssl_hidden_tap_index} "
        f"teacher={ssl_teacher_name} layer={ssl_teacher_layer_idx}"
    )
    print(f"[LEN] enable={use_len_predictor} w_len={w_len}")
    print(f"[REFINER] enable={use_refiner} w_ref={w_ref}")
    print(f"[STFT] use={use_stft} w_stft={w_stft} (refiner-only via detach)")
    print(f"[VF-LIP-FD] enable={enable_vf_lip} start={vf_lip_start} w={w_vf_lip} L_hi={vf_lip_L_hi} sigma={vf_lip_sigma}")
    print(
        f"[CACHE] mel_dir={mel_cache_dir} speecht5_dir={speecht5_cache_dir} "
        f"gpu_mel={gpu_mel_cache}/{gpu_mel_preload} dtype={gpu_mel_cache_dtype} "
        f"limit_gib={cache_cfg.get('gpu_mel_cache_limit_gib')} "
        f"gpu_text={gpu_text_cache}/{gpu_text_preload} dtype={gpu_text_cache_dtype} "
        f"limit_gib={cache_cfg.get('gpu_text_cache_limit_gib')} "
        f"reserve_gib={cache_cfg.get('gpu_cache_reserve_gib', 64.0)} "
        f"mel_resident={bytes_to_gib(gpu_cache_state['mel_bytes']):.2f}GiB "
        f"text_resident={bytes_to_gib(gpu_cache_state['text_bytes']):.2f}GiB"
    )
    print(
        f"[LOADER] enable={use_dataloader} runtime={use_dataloader_runtime} "
        f"workers={loader_num_workers} pin_memory={loader_pin_memory} "
        f"persistent_workers={loader_persistent_workers} "
        f"prefetch_factor={loader_prefetch_factor if loader_num_workers > 0 else 'n/a'} "
        f"length_bucket={enable_length_bucket} buckets={num_length_buckets}"
    )
    print(f"[PERF] log_every={perf_log_every}")
    print(f"[LR] schedule={lr_schedule} warmup={lr_warmup_steps} min_scale={lr_min_scale}")

    def compute_duration_losses(h_enc, maskL, attn, maskK, dur_teacher_full=None, dur_teacher_full_mask=None):
        loss_dur = torch.tensor(0.0, device=device)
        loss_len = torch.tensor(0.0, device=device)

        h_dp = h_enc.detach()
        log_dur_pred = dur_pred(h_dp, maskL)
        k_pred = len_pred(h_dp, maskL) if len_pred is not None else None

        dur_gt_align = attn.sum(dim=1).float()
        K_tar = maskK.sum(dim=1).float()
        # Use the aligned frame count, not token count. The teacher scaling
        # should map align-grid durations back to target-frame durations.
        K_align_cnt = dur_gt_align.sum(dim=1).clamp_min(1.0)
        K_est_full = (K_align_cnt * float(ds_align)).clamp_min(1.0)
        scale = (K_tar / K_est_full).unsqueeze(1)
        dur_gt_full = dur_gt_align * float(ds_align) * scale
        teacher_mask = maskL.float()
        if dur_teacher_full is not None and dur_teacher_full_mask is not None:
            full_teacher = dur_teacher_full[:, :maskL.shape[1]].to(device=device, dtype=dur_gt_full.dtype)
            full_teacher_mask = dur_teacher_full_mask[:, :maskL.shape[1]].to(device=device, dtype=dur_gt_full.dtype)
            use_full = (full_teacher_mask.sum(dim=1, keepdim=True) > 0).to(dtype=dur_gt_full.dtype)
            dur_gt_full = use_full * full_teacher + (1.0 - use_full) * dur_gt_full
            teacher_mask = use_full * full_teacher_mask + (1.0 - use_full) * teacher_mask

        log_dur_gt_full = torch.log(dur_gt_full + 1.0)
        denomL = teacher_mask.sum().clamp_min(1.0)
        loss_dur = (((log_dur_pred - log_dur_gt_full) ** 2) * teacher_mask).sum() / denomL

        dur_pred_lin = (torch.exp(log_dur_pred) - 1.0) * maskL.float()
        dur_sum = dur_pred_lin.sum(dim=1)
        loss_dur = loss_dur + 0.5 * ((dur_sum - K_tar) / (K_tar + 1e-6)).abs().mean()
        if k_pred is not None:
            loss_len = (((k_pred / (K_tar + 1e-6)) - 1.0) ** 2).mean()

        return loss_dur, loss_len

    def attn_grid_mask_from_full_mask(maskK_full, K_grid):
        if int(K_grid) == int(maskK_full.shape[1]):
            return maskK_full
        lengths = torch.ceil(maskK_full.sum(dim=1).float() / float(ds_align)).long()
        positions = torch.arange(int(K_grid), device=maskK_full.device).unsqueeze(0)
        return positions < lengths.unsqueeze(1)

    def perturb_monotonic_attn_once(attn_base, maskK_grid, maskL, sigma):
        cand = torch.zeros_like(attn_base)
        B, K_grid, Lmax = attn_base.shape
        with torch.no_grad():
            for b in range(B):
                K0 = int(maskK_grid[b].sum().item())
                L0 = int(maskL[b].sum().item())
                if K0 <= 0 or L0 <= 0:
                    continue
                dur = attn_base[b, :K0, :L0].sum(dim=0).float().clamp_min(0.0)
                if float(dur.sum().item()) <= 0.0:
                    dur = torch.ones(L0, device=device, dtype=torch.float32)
                if sigma > 0.0:
                    dur = dur * torch.exp(torch.randn_like(dur) * float(sigma))
                dur_int, _ = durations_to_int_and_fixsum(
                    dur.view(1, -1),
                    torch.ones(1, L0, device=device, dtype=torch.bool),
                    K0,
                )
                dur_int = dur_int.to(device=device)
                pos = 0
                for i in range(L0):
                    d = int(dur_int[i].item())
                    if d <= 0:
                        continue
                    end = min(K0, pos + d)
                    if end > pos:
                        cand[b, pos:end, i] = 1.0
                    pos = end
                if pos < K0:
                    cand[b, pos:K0, max(0, L0 - 1)] = 1.0
        cand = cand * maskK_grid[:, :, None].float() * maskL[:, None, :].float()
        denom = cand.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return cand / denom

    def make_alignment_candidates(attn_base, maskK_grid, maskL, n_candidates, sigma, include_base=True):
        candidates = []
        if include_base:
            candidates.append(attn_base)
        while len(candidates) < int(n_candidates):
            candidates.append(perturb_monotonic_attn_once(attn_base, maskK_grid, maskL, sigma))
        return candidates[: int(n_candidates)]

    def expand_attn_prior_to_full(
        attn_grid,
        maskK_grid,
        maskK_full,
        mu_tok,
        logvar_tok,
        align_mu_tok,
        align_logvar_tok,
        out_dtype,
    ):
        B, Kmax = maskK_full.shape
        D = mu_tok.shape[-1]
        D_align = align_mu_tok.shape[-1]

        mu_align = torch.bmm(attn_grid, mu_tok.float())
        logvar_align = torch.bmm(attn_grid, logvar_tok.float())
        score_mu_align = torch.bmm(attn_grid, align_mu_tok.float())
        score_logvar_align = torch.bmm(attn_grid, align_logvar_tok.float())

        mu_full = torch.zeros(B, Kmax, D, device=device, dtype=out_dtype)
        logvar_full = torch.zeros(B, Kmax, D, device=device, dtype=out_dtype)
        align_mu_full = torch.zeros(B, Kmax, D_align, device=device, dtype=out_dtype)
        align_logvar_full = torch.zeros(B, Kmax, D_align, device=device, dtype=out_dtype)
        for b in range(B):
            K0 = int(maskK_full[b].sum().item())
            Kg = int(maskK_grid[b].sum().item())
            if Kg <= 0 or K0 <= 0:
                continue
            if Kg == K0:
                mu_up = mu_align[b, :Kg]
                lv_up = logvar_align[b, :Kg]
                align_mu_up = score_mu_align[b, :Kg]
                align_lv_up = score_logvar_align[b, :Kg]
            else:
                mu_up = mu_align[b, :Kg].repeat_interleave(ds_align, dim=0)[:K0]
                lv_up = logvar_align[b, :Kg].repeat_interleave(ds_align, dim=0)[:K0]
                align_mu_up = score_mu_align[b, :Kg].repeat_interleave(ds_align, dim=0)[:K0]
                align_lv_up = score_logvar_align[b, :Kg].repeat_interleave(ds_align, dim=0)[:K0]
            mu_full[b, :K0] = mu_up.to(dtype=out_dtype)
            logvar_full[b, :K0] = lv_up.to(dtype=out_dtype)
            align_mu_full[b, :K0] = align_mu_up.to(dtype=out_dtype)
            align_logvar_full[b, :K0] = align_lv_up.to(dtype=out_dtype)
        return mu_full, logvar_full, align_mu_full, align_logvar_full

    def softmin_scalar_losses(losses, tau):
        stacked = torch.stack(losses)
        if float(tau) <= 0.0 or stacked.numel() <= 1:
            return stacked.min()
        return -float(tau) * torch.logsumexp(-stacked / float(tau), dim=0)

    def build_local_prior_batch(
        zS_log,
        maskK,
        h_enc,
        maskL,
        mu_tok,
        logvar_tok,
        align_mu_tok=None,
        align_logvar_tok=None,
        supervise_dur_len=False,
        dur_teacher_full=None,
        dur_teacher_full_mask=None,
        gt_token_durations=None,
        gt_token_duration_mask=None,
    ):
        zS = (zS_log - mu_b) / std_b
        score_mu_base = align_mu_tok if align_mu_tok is not None else mu_tok
        score_logvar_base = align_logvar_tok if align_logvar_tok is not None else logvar_tok
        fixed_logvar_tok = torch.full_like(score_logvar_base, prior_fixed_logvar)
        if prior_loss_mode == "gaussian_nll":
            score_logvar_tok = score_logvar_base
            sample_logvar_tok = logvar_tok
        elif prior_loss_mode == "mu_var_reg":
            score_logvar_tok = fixed_logvar_tok
            sample_logvar_tok = logvar_tok
        else:
            score_logvar_tok = fixed_logvar_tok
            sample_logvar_tok = fixed_logvar_tok

        B, Kmax = zS.shape[:2]
        D = mu_tok.shape[-1]
        D_align = score_mu_base.shape[-1]

        use_gt_attn = (
            gt_token_durations is not None
            and gt_token_duration_mask is not None
            and bool(gt_token_duration_mask.any().item())
        )
        attn_grid = None
        maskK_grid = None
        if use_gt_attn:
            if ds_align != 1:
                raise NotImplementedError("GT alignment prior currently requires ds_align == 1")
            attn = torch.zeros(B, Kmax, maskL.shape[1], device=device, dtype=torch.float32)
            with torch.no_grad():
                for b in range(B):
                    K0 = int(maskK[b].sum().item())
                    L0 = int(maskL[b].sum().item())
                    if K0 <= 0 or L0 <= 0:
                        continue
                    durs = gt_token_durations[b, :L0].detach().float().clamp_min(0.0)
                    if float(durs.sum().item()) <= 0.0:
                        durs = torch.ones_like(durs)
                    if use_ctc_blank_repeat_prior:
                        dur_int = torch.tensor(
                            _durations_to_int_allow_zero_and_fixsum(durs.tolist(), K0),
                            device=device,
                            dtype=torch.long,
                        )
                    else:
                        dur_int, _ = durations_to_int_and_fixsum(
                            durs.view(1, -1),
                            torch.ones(1, L0, device=device, dtype=torch.bool),
                            K0,
                        )
                    pos = 0
                    for i in range(L0):
                        d = int(dur_int[i].item())
                        if d <= 0:
                            continue
                        attn[b, pos:pos + d, i] = 1.0
                        pos += d
                    if pos < K0:
                        attn[b, pos:K0, max(0, L0 - 1)] = 1.0
            attn = attn * maskK[:, :, None].float() * maskL[:, None, :].float()
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1.0)
            attn = attn / denom

            mu_full = torch.bmm(attn, mu_tok.float()).to(dtype=zS.dtype)
            logvar_full = torch.bmm(attn, sample_logvar_tok.float()).to(dtype=zS.dtype)
            align_mu_full = torch.bmm(attn, score_mu_base.float()).to(dtype=zS.dtype)
            align_logvar_full = torch.bmm(attn, score_logvar_base.float()).to(dtype=zS.dtype)
            attn_grid = attn
            maskK_grid = maskK
        else:
            zS_align, maskK_align, _ = downsample_time_bkd(zS, maskK, ds_align)
            with torch.no_grad():
                score = gaussian_mas_score(
                    zS_align.float(),
                    score_mu_base.float(),
                    score_logvar_tok.float(),
                    maskK_align,
                    maskL,
                    score_temp=mas_temp,
                )
                attn_hard = monotonic_alignment_search(score, maskK_align, maskL, neg_inf=-1e4)
                if mas_mode == "hard":
                    attn = attn_hard
                else:
                    attn_soft = monotonic_alignment_posterior(score, maskK_align, maskL, neg_inf=-1e4)
                    if mas_mode == "soft":
                        attn = attn_soft
                    elif mas_mode == "mix":
                        mix_a = float(max(0.0, min(1.0, mas_mix_alpha)))
                        attn = (1.0 - mix_a) * attn_hard + mix_a * attn_soft
                    else:
                        raise ValueError(f"Unsupported mas_mode: {mas_mode}")
                    attn = attn * maskK_align[:, :, None].float() * maskL[:, None, :].float()
                    attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            mu_align = torch.bmm(attn, mu_tok.float())
            logvar_align = torch.bmm(attn, sample_logvar_tok.float())
            align_mu_align = torch.bmm(attn, score_mu_base.float())
            align_logvar_align = torch.bmm(attn, score_logvar_base.float())

            mu_full = torch.zeros(B, Kmax, D, device=device, dtype=zS.dtype)
            logvar_full = torch.zeros(B, Kmax, D, device=device, dtype=zS.dtype)
            align_mu_full = torch.zeros(B, Kmax, D_align, device=device, dtype=zS.dtype)
            align_logvar_full = torch.zeros(B, Kmax, D_align, device=device, dtype=zS.dtype)
            for b in range(B):
                K0 = int(maskK[b].sum().item())
                Ka = int(maskK_align[b].sum().item())
                if Ka <= 0 or K0 <= 0:
                    continue
                mu_up = mu_align[b, :Ka].repeat_interleave(ds_align, dim=0)[:K0]
                lv_up = logvar_align[b, :Ka].repeat_interleave(ds_align, dim=0)[:K0]
                align_mu_up = align_mu_align[b, :Ka].repeat_interleave(ds_align, dim=0)[:K0]
                align_lv_up = align_logvar_align[b, :Ka].repeat_interleave(ds_align, dim=0)[:K0]
                mu_full[b, :K0] = mu_up.to(dtype=zS.dtype)
                logvar_full[b, :K0] = lv_up.to(dtype=zS.dtype)
                align_mu_full[b, :K0] = align_mu_up.to(dtype=zS.dtype)
                align_logvar_full[b, :K0] = align_lv_up.to(dtype=zS.dtype)
            attn_grid = attn
            maskK_grid = maskK_align

        if (
            duration_perturb_enable
            and duration_perturb_num > 1
            and duration_perturb_sigma > 0.0
            and attn_grid is not None
            and maskK_grid is not None
        ):
            candidates = make_alignment_candidates(
                attn_grid,
                maskK_grid,
                maskL,
                duration_perturb_num,
                duration_perturb_sigma,
                include_base=duration_perturb_include_base,
            )
            attn_sample = candidates[random.randrange(len(candidates))]
            mu_full, logvar_full, align_mu_full, align_logvar_full = expand_attn_prior_to_full(
                attn_sample,
                maskK_grid,
                maskK,
                mu_tok,
                sample_logvar_tok,
                score_mu_base,
                score_logvar_tok,
                zS.dtype,
            )

        eps = torch.randn_like(mu_full)
        zT_sample = mu_full + torch.exp(0.5 * logvar_full.clamp(-6.0, 1.5)) * eps
        zT_sample = zT_sample * maskK.float().unsqueeze(-1)
        zT_mean = mu_full
        source_delta = (zT_sample - zT_mean) * maskK.float().unsqueeze(-1)

        loss_dur = torch.tensor(0.0, device=device)
        loss_len = torch.tensor(0.0, device=device)
        if supervise_dur_len:
            loss_dur, loss_len = compute_duration_losses(
                h_enc,
                maskL,
                attn,
                maskK,
                dur_teacher_full=dur_teacher_full,
                dur_teacher_full_mask=dur_teacher_full_mask,
            )

        return (
            zS,
            attn,
            mu_full,
            logvar_full,
            zT_sample,
            zT_mean,
            source_delta,
            loss_dur,
            loss_len,
            align_mu_full,
            align_logvar_full,
        )

    def build_ctc_targets_from_texts(texts):
        targets = []
        t_lens = []
        for txt in texts:
            ids = tok.encode(txt)
            if len(ids) == 0:
                ids = [UNK_ID]
            t = torch.tensor(ids, dtype=torch.long)
            targets.append(t)
            t_lens.append(len(ids))
        targets = torch.cat(targets, dim=0).to(device)
        target_lengths = torch.tensor(t_lens, dtype=torch.long, device=device)
        return targets, target_lengths

    def build_att_decoder_batch(texts):
        dec_in_seqs = []
        dec_tgt_seqs = []
        for txt in texts:
            ids = tok.encode(txt)
            if len(ids) == 0:
                ids = [UNK_ID]
            ids = [int(token_id) for token_id in ids]
            dec_in_seqs.append([AED_SOS_ID] + ids)
            dec_tgt_seqs.append(ids + [AED_EOS_ID])

        T_max = max(max(len(seq) for seq in dec_in_seqs), 1)
        dec_in = torch.full(
            (len(dec_in_seqs), T_max),
            fill_value=PAD_ID,
            device=device,
            dtype=torch.long,
        )
        dec_tgt = torch.full(
            (len(dec_tgt_seqs), T_max),
            fill_value=PAD_ID,
            device=device,
            dtype=torch.long,
        )
        for b, (inp, tgt) in enumerate(zip(dec_in_seqs, dec_tgt_seqs)):
            n = len(inp)
            dec_in[b, :n] = torch.tensor(inp, device=device, dtype=torch.long)
            dec_tgt[b, :n] = torch.tensor(tgt, device=device, dtype=torch.long)
        return dec_in, dec_tgt

    def prepare_ctc_input(z, mask, *, apply_subsample: bool):
        if apply_subsample and ctc_subsample_factor > 1:
            z, mask, k_list = downsample_time_bkd(z, mask, ctc_subsample_factor)
            k_list = [int(k) for k in k_list]
        else:
            k_list = [int(k) for k in mask.long().sum(dim=1).tolist()]
        return z, mask, k_list

    def build_duration_expanded_prior_batch(h_enc, maskL, mu_tok, K_targets=None):
        h_dp = h_enc.detach()
        log_dur = dur_pred(h_dp, maskL)
        dur = (torch.exp(log_dur) - 1.0) * maskL.float()
        k_pred = len_pred(h_dp, maskL) if len_pred is not None else None

        B, _, D = mu_tok.shape
        limit = int(getattr(vf.rope, "max_seq_len", 4096))
        k_list = []
        z_list = []
        for b in range(B):
            L_valid = int(maskL[b].sum().item())
            if L_valid <= 0:
                k_list.append(0)
                z_list.append(mu_tok.new_zeros(0, D))
                continue

            if K_targets is not None:
                k_target = min(max(int(K_targets[b]), L_valid, 16), limit)
            elif k_pred is not None:
                k_pred_full_raw = int(max(16, round(float(k_pred[b].item()))))
                k_target = min(max(k_pred_full_raw, L_valid), limit)
            else:
                dur_sum = float(dur[b].sum().item())
                k_target = min(max(int(round(dur_sum)), L_valid, 16), limit)

            dur_int, _ = durations_to_int_and_fixsum(dur[b:b + 1], maskL[b:b + 1], k_target)
            feats = []
            for i in range(L_valid):
                d = int(dur_int[i].item())
                feats.append(mu_tok[b, i:i + 1].repeat(d, 1))

            if feats:
                z_one = torch.cat(feats, dim=0)[:k_target]
            else:
                z_one = mu_tok.new_zeros(k_target, D)
            k_list.append(int(z_one.shape[0]))
            z_list.append(z_one)

        Kmax = max(max(k_list), 1)
        z_pad = mu_tok.new_zeros(B, Kmax, D)
        maskK = torch.zeros(B, Kmax, device=device, dtype=torch.bool)
        for b, z_one in enumerate(z_list):
            Kb = int(z_one.shape[0])
            if Kb <= 0:
                continue
            z_pad[b, :Kb] = z_one
            maskK[b, :Kb] = True
        return z_pad, maskK, k_list

    def _split_int_total(total, n):
        total = int(max(0, round(float(total))))
        n = int(max(0, n))
        if n <= 0:
            return []
        base = total // n
        rem = total - base * n
        return [base + (1 if i < rem else 0) for i in range(n)]

    def _fix_duration_sum(durs, target):
        target = int(max(0, target))
        durs = [int(max(0, round(float(x)))) for x in durs]
        if not durs:
            return durs
        diff = target - sum(durs)
        if diff > 0:
            durs[-1] += diff
        elif diff < 0:
            need = -diff
            for i in range(len(durs) - 1, -1, -1):
                take = min(need, durs[i])
                durs[i] -= take
                need -= take
                if need <= 0:
                    break
        return durs

    def _token_durations_from_word_alignment(text, L_valid, K_target, row_meta, seg_start, seg_end):
        """
        Convert processed_unified word-level mel spans into SpeechT5-token
        duration teachers. This bypasses online MAS while keeping z_c on the
        normalized frame grid. If metadata is missing, fall back to a uniform
        duration split instead of silently returning invalid zeros.
        """
        L_valid = int(max(0, L_valid))
        K_target = int(max(0, K_target))
        if L_valid <= 0:
            return []
        if K_target <= 0:
            return [0] * L_valid

        full_words = row_meta.get("_full_words") or row_meta.get("words") or []
        if not full_words:
            return _fix_duration_sum(_split_int_total(K_target, L_valid), K_target)

        seg_start = int(max(0, seg_start))
        seg_end = int(max(seg_start + 1, seg_end))
        pause_after = row_meta.get("_pause_after_word_mel_frames") or row_meta.get("pause_after_word_mel_frames") or []
        mel_len = int(row_meta.get("_mel_len", row_meta.get("mel_len", seg_end)))

        word_durs = []
        word_texts = []
        for idx, w in enumerate(full_words):
            try:
                ws = int(w.get("start_mel", 0))
                we = int(w.get("end_mel", ws))
            except Exception:
                continue
            if we <= seg_start or ws >= seg_end:
                continue

            pause = 0
            if idx < len(pause_after):
                try:
                    pause = int(max(0, pause_after[idx]))
                except Exception:
                    pause = 0
            unit_s = ws
            unit_e = min(max(we, ws + 1) + pause, mel_len)
            dur = max(0, min(unit_e, seg_end) - max(unit_s, seg_start))
            if dur <= 0:
                continue
            word_durs.append(dur)
            word_texts.append(str(w.get("word", "")).strip())

        if not word_durs:
            return _fix_duration_sum(_split_int_total(K_target, L_valid), K_target)

        missing = K_target - sum(word_durs)
        if missing > 0:
            # Local context may begin/end in silence. Attach boundary silence to
            # the nearest lexical unit so the expanded prior still covers K.
            if word_durs:
                word_durs[0] += missing // 2
                word_durs[-1] += missing - (missing // 2)
        elif missing < 0:
            word_durs = _fix_duration_sum(word_durs, K_target)

        weights = [max(1, len(normalize_text_basic(w).replace(" ", ""))) for w in word_texts]
        token_counts = [max(1, int(round(L_valid * (w / max(1, sum(weights)))))) for w in weights]
        token_counts = _fix_duration_sum(token_counts, L_valid)
        # _fix_duration_sum can zero out very small words if L_valid < n_words.
        # Merge zero-count words into the closest active bucket.
        if sum(1 for c in token_counts if c > 0) == 0:
            token_counts = _split_int_total(L_valid, len(word_durs))

        token_durs = []
        carry_dur = 0
        for dur, n_tok in zip(word_durs, token_counts):
            if n_tok <= 0:
                carry_dur += int(dur)
                continue
            dur = int(dur) + carry_dur
            carry_dur = 0
            token_durs.extend(_split_int_total(dur, n_tok))
        if carry_dur > 0:
            if token_durs:
                token_durs[-1] += carry_dur
            else:
                token_durs = [carry_dur]

        if len(token_durs) < L_valid:
            token_durs.extend([0] * (L_valid - len(token_durs)))
        elif len(token_durs) > L_valid:
            extra = sum(token_durs[L_valid - 1:])
            token_durs = token_durs[:L_valid]
            token_durs[-1] = extra
        return _fix_duration_sum(token_durs, K_target)

    ctc_repeat_cache = OrderedDict()
    ctc_repeat_warn_count = 0

    def _resolve_ctc_npz_path(row_meta):
        if not row_meta:
            return None
        npz_path = row_meta.get("npz_path")
        if npz_path:
            return str(npz_path)
        utt_id = row_meta.get("utt_id")
        if utt_id is not None:
            full = full_row_by_utt.get(str(utt_id))
            if full is not None and full.get("npz_path"):
                return str(full["npz_path"])
        wav = row_meta.get("parent_wav", row_meta.get("wav"))
        if wav:
            full = aligned_row_by_wav.get(wav)
            if full is not None and full.get("npz_path"):
                return str(full["npz_path"])
        return None

    def _load_ctc_ext_repeats(npz_path):
        nonlocal ctc_repeat_cache
        if not npz_path:
            return None
        npz_path = os.path.abspath(str(npz_path))
        if npz_path in ctc_repeat_cache:
            ctc_repeat_cache.move_to_end(npz_path)
            return ctc_repeat_cache[npz_path]
        if not os.path.exists(npz_path):
            return None
        try:
            z = np.load(npz_path, allow_pickle=True)
            reps = np.asarray(z["ext_repeats"], dtype=np.float32)
        except Exception:
            return None
        ctc_repeat_cache[npz_path] = reps
        ctc_repeat_cache.move_to_end(npz_path)
        while len(ctc_repeat_cache) > 4096:
            ctc_repeat_cache.popitem(last=False)
        return reps

    def _durations_to_int_allow_zero_and_fixsum(durs, target):
        target = int(max(0, target))
        durs = [float(max(0.0, float(x))) for x in durs]
        if not durs:
            return []
        if target <= 0:
            return [0] * len(durs)
        total = float(sum(durs))
        if total <= 0.0:
            durs = [1.0 for _ in durs]
            total = float(len(durs))

        scaled = [x * (float(target) / total) for x in durs]
        ints = [int(round(x)) for x in scaled]
        diff = target - sum(ints)
        if diff > 0:
            order = sorted(range(len(scaled)), key=lambda i: scaled[i] - math.floor(scaled[i]), reverse=True)
            if not order:
                order = [len(ints) - 1]
            for n in range(diff):
                ints[order[n % len(order)]] += 1
        elif diff < 0:
            need = -diff
            order = sorted(range(len(scaled)), key=lambda i: scaled[i] - math.floor(scaled[i]))
            for i in order:
                if need <= 0:
                    break
                take = min(need, ints[i])
                ints[i] -= take
                need -= take
        if sum(ints) != target:
            ints[-1] += target - sum(ints)
            ints[-1] = max(0, ints[-1])
        return [int(max(0, x)) for x in ints]

    def _ctc_blank_repeats_for_row(text, L_valid, K_target, row_meta, seg_start, seg_end):
        nonlocal ctc_repeat_warn_count
        L_valid = int(max(0, L_valid))
        K_target = int(max(0, K_target))
        if L_valid <= 0:
            return []
        if K_target <= 0:
            return [0] * L_valid

        npz_path = _resolve_ctc_npz_path(row_meta)
        reps = _load_ctc_ext_repeats(npz_path)
        state_offset = 0
        if reps is not None and int(reps.shape[0]) != L_valid:
            full_text = canonicalize_text(row_meta.get("text_norm_full", row_meta.get("text_norm", ""))) if row_meta else ""
            ctx_text = canonicalize_text(text)
            char_s = None
            char_e = None
            try:
                word_s = row_meta.get("ctx_word_start", None) if row_meta else None
                word_e = row_meta.get("ctx_word_end", None) if row_meta else None
                if word_s is not None and word_e is not None and full_text:
                    words = full_text.split()
                    word_s = int(word_s)
                    word_e = int(word_e)
                    if 0 <= word_s < word_e <= len(words):
                        spans = []
                        pos_find = 0
                        for w in words:
                            s_idx = full_text.find(w, pos_find)
                            if s_idx < 0:
                                s_idx = pos_find
                            e_idx = s_idx + len(w)
                            spans.append((s_idx, e_idx))
                            pos_find = e_idx + 1
                        char_s = spans[word_s][0]
                        char_e = spans[word_e - 1][1]
            except Exception:
                char_s = None
                char_e = None

            if (char_s is None or char_e is None) and full_text and ctx_text:
                idx = full_text.find(ctx_text)
                if idx >= 0:
                    char_s = idx
                    char_e = idx + len(ctx_text)

            if char_s is not None and char_e is not None:
                state_s = int(2 * char_s)
                state_e = int(2 * char_e + 1)
                if 0 <= state_s < state_e <= int(reps.shape[0]) and (state_e - state_s) == L_valid:
                    state_offset = int(np.asarray(reps[:state_s], dtype=np.float32).sum())
                    reps = reps[state_s:state_e]

        if reps is None or int(reps.shape[0]) != L_valid:
            if ctc_repeat_warn_count < 8:
                got = None if reps is None else int(reps.shape[0])
                print(
                    f"[CTC-BLANK-PRIOR][WARN] missing/mismatched ext_repeats "
                    f"utt={row_meta.get('utt_id') if row_meta else None} L={L_valid} reps={got}; fallback uniform"
                )
                ctc_repeat_warn_count += 1
            return _durations_to_int_allow_zero_and_fixsum([1.0] * L_valid, K_target)

        ctc_total = int(row_meta.get("_ctc_num_frames", row_meta.get("ctc_num_frames", 0)) or 0)
        if ctc_total <= 0:
            ctc_total = int(max(1, round(float(reps.sum()))))
        full_len = int(row_meta.get("_svae_len", row_meta.get("_mel_len", row_meta.get("svae_len", row_meta.get("mel_len", 0)))) or 0)
        if full_len <= 0:
            full_len = int(max(seg_end, K_target))
        seg_start = int(max(0, seg_start))
        seg_end = int(max(seg_start + 1, seg_end))
        ctc_s = int(math.floor(float(seg_start) * float(ctc_total) / float(max(1, full_len))))
        ctc_e = int(math.ceil(float(seg_end) * float(ctc_total) / float(max(1, full_len))))
        ctc_s = max(0, min(ctc_s, ctc_total))
        ctc_e = max(ctc_s + 1, min(ctc_e, ctc_total))

        overlaps = []
        pos = int(state_offset)
        for rep in reps.tolist():
            rep = int(max(0, round(float(rep))))
            next_pos = pos + rep
            ov = max(0, min(next_pos, ctc_e) - max(pos, ctc_s))
            overlaps.append(float(ov))
            pos = next_pos

        return _durations_to_int_allow_zero_and_fixsum(overlaps, K_target)

    def build_ctc_blank_duration_batch(texts, maskL, K_list, row_metas, starts, ends):
        B, Lmax = maskL.shape
        dur = torch.zeros(B, Lmax, device=device, dtype=torch.float32)
        dur_mask = torch.zeros(B, Lmax, device=device, dtype=torch.bool)
        any_valid = False
        for b in range(B):
            L_valid = int(maskL[b].long().sum().item())
            K_target = int(K_list[b])
            durs = _ctc_blank_repeats_for_row(
                texts[b],
                L_valid,
                K_target,
                row_metas[b] if b < len(row_metas) else {},
                starts[b],
                ends[b],
            )
            if len(durs) <= 0:
                continue
            n = min(L_valid, len(durs), Lmax)
            dur[b, :n] = torch.tensor(durs[:n], device=device, dtype=torch.float32)
            dur_mask[b, :n] = True
            any_valid = True
        if not any_valid:
            return None, None
        return dur, dur_mask

    def build_gt_token_duration_batch(texts, maskL, K_list, row_metas, starts, ends):
        if use_ctc_blank_repeat_prior:
            return build_ctc_blank_duration_batch(texts, maskL, K_list, row_metas, starts, ends)
        if not (use_gt_alignment_prior or use_gt_duration_teacher):
            return None, None
        B, Lmax = maskL.shape
        dur = torch.zeros(B, Lmax, device=device, dtype=torch.float32)
        dur_mask = torch.zeros(B, Lmax, device=device, dtype=torch.bool)
        any_valid = False
        for b in range(B):
            L_valid = int(maskL[b].long().sum().item())
            K_target = int(K_list[b])
            durs = _token_durations_from_word_alignment(
                texts[b],
                L_valid,
                K_target,
                row_metas[b] if b < len(row_metas) else {},
                starts[b],
                ends[b],
            )
            if len(durs) <= 0:
                continue
            n = min(L_valid, len(durs), Lmax)
            dur[b, :n] = torch.tensor(durs[:n], device=device, dtype=torch.float32)
            dur_mask[b, :n] = True
            any_valid = True
        if not any_valid:
            return None, None
        return dur, dur_mask

    text_piece_ids_cache = {}

    def get_st5_piece_ids(text: str):
        text = canonicalize_text(text)
        if text in text_piece_ids_cache:
            return text_piece_ids_cache[text]
        ids = list(st5.processor.tokenizer(text)["input_ids"])
        text_piece_ids_cache[text] = ids
        return ids

    def find_token_subsequence(full_ids, cut_ids):
        eos_id = getattr(st5.processor.tokenizer, "eos_token_id", None)
        if eos_id is not None and len(full_ids) > 0 and full_ids[-1] == eos_id:
            full_core = full_ids[:-1]
        else:
            full_core = full_ids
        if eos_id is not None and len(cut_ids) > 0 and cut_ids[-1] == eos_id:
            cut_core = cut_ids[:-1]
        else:
            cut_core = cut_ids
        if len(cut_core) == 0:
            return 0, 0, 0
        max_start = len(full_core) - len(cut_core)
        for s in range(max_start + 1):
            if full_core[s:s + len(cut_core)] == cut_core:
                return s, s + len(cut_core), len(cut_core)
        return None, None, len(cut_core)

    @torch.no_grad()
    def build_full_tts_duration_teacher_batch(batch_wav_paths):
        if not use_full_tts_teacher:
            return None
        if ds_align != 1:
            raise NotImplementedError("use_full_tts_teacher currently requires ds_align == 1")

        unique_wavs = []
        unique_rows = []
        wav_to_u = {}
        for wav_path in batch_wav_paths:
            if wav_path not in wav_to_u:
                full_row = aligned_row_by_wav.get(wav_path)
                if full_row is None:
                    continue
                wav_to_u[wav_path] = len(unique_wavs)
                unique_wavs.append(wav_path)
                unique_rows.append(full_row)
        if not unique_rows:
            return None

        zS_full_log, maskK_full, K_list_full, texts_full, _, _ = build_batch_from_aligned_rows(unique_rows)
        h_enc_full, maskL_full, mu_tok_full, logvar_tok_full, align_mu_tok_full, align_logvar_tok_full = encode_text_batch(texts_full)
        gt_dur_full, gt_dur_mask_full = (None, None)
        if use_gt_alignment_prior or use_ctc_blank_repeat_prior:
            gt_dur_full, gt_dur_mask_full = build_gt_token_duration_batch(
                texts_full,
                maskL_full,
                K_list_full,
                unique_rows,
                [0 for _ in unique_rows],
                K_list_full,
            )
        _, attn_full, _, _, _, _, _, _, _, _, _ = build_local_prior_batch(
            zS_full_log,
            maskK_full,
            h_enc_full,
            maskL_full,
            mu_tok_full,
            logvar_tok_full,
            align_mu_tok=align_mu_tok_full,
            align_logvar_tok=align_logvar_tok_full,
            supervise_dur_len=False,
            gt_token_durations=gt_dur_full if (use_gt_alignment_prior or use_ctc_blank_repeat_prior) else None,
            gt_token_duration_mask=gt_dur_mask_full if (use_gt_alignment_prior or use_ctc_blank_repeat_prior) else None,
        )

        return h_enc_full, maskL_full, maskK_full, attn_full

    def decode_ctc_demo_text(logits_bkv, input_len: int) -> str:
        if asr_demo_decode_mode == "kenlm" and asr_demo_kenlm is not None and asr_demo_kenlm.enabled:
            try:
                return canonicalize_text(asr_demo_kenlm.decode(logits_bkv[:, :int(input_len)]))
            except Exception as exc:
                if not asr_demo_kenlm_fallback:
                    raise
                print(f"[ASR-DECODE][WARN] KenLM decode failed; fallback greedy. error={repr(exc)}")
        decoded_ids = ctc_greedy_decode(logits_bkv, [int(input_len)], blank_id=BLANK_ID)[0]
        return canonicalize_text(tok.decode(decoded_ids))

    def demo_audio_sec_from_frames(n_frames: int) -> float:
        return max(float(max(1, int(n_frames))) * float(hop_size) / float(sampling_rate), 1e-6)

    def demo_audio_sec_from_wav(wav: torch.Tensor) -> float:
        return max(float(max(1, int(wav.numel()))) / float(sampling_rate), 1e-6)

    def demo_rtf_line(label: str, elapsed_sec: float, audio_sec: float) -> str:
        rtf = float(elapsed_sec) / max(float(audio_sec), 1e-6)
        return f"[RTF {label}] time={elapsed_sec:.4f}s audio={audio_sec:.4f}s rtf={rtf:.4f}"

    @torch.no_grad()
    def heun_integrate_trace(vf, z0, maskK, steps=30, direction=+1, cfg_scale=1.0, spk_e=None, style_e=None, text_cond=None):
        z = z0
        dt = direction * (1.0 / steps)
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
        states = [z.detach().float().cpu()]

        def v_eval(z_now, t_now, cfg_flag, spk, style, text):
            t_tensor = torch.full((B,), float(t_now), device=device)
            return vf(z_now, t_tensor, maskK, cfg_flag=cfg_flag, spk_e=spk, style_e=style, text_cond=text)

        def v_mix(z_now, t_now):
            is_un = (spk_e.abs().sum(dim=1) < 1e-8)
            if style_e is not None:
                is_un = is_un & (style_e.abs().sum(dim=1) < 1e-8)
            if text_cond is not None:
                is_un = is_un & (text_cond.abs().sum(dim=(1, 2)) < 1e-8)
            if cfg_scale == 1.0:
                cfg = torch.where(is_un, cfg_un, cfg_cond)
                spk = torch.where(is_un[:, None], spk_zero, spk_e)
                style = None
                if style_e is not None:
                    style = torch.where(is_un[:, None], style_zero, style_e)
                text = text_cond
                if text_cond is not None:
                    text = torch.where(is_un[:, None, None], text_zero, text_cond)
                return v_eval(z_now, t_now, cfg, spk, style, text)

            v_c = v_eval(z_now, t_now, cfg_cond, spk_e, style_e, text_cond)
            v_u = v_eval(z_now, t_now, cfg_un, spk_zero, style_zero, text_zero)
            return v_u + cfg_scale * (v_c - v_u)

        for i in range(steps):
            t = t0 + i * dt
            t_next = t + dt
            k1 = v_mix(z, t)
            z_pred = z + dt * k1
            k2 = v_mix(z_pred, t_next)
            z = z + dt * 0.5 * (k1 + k2)
            states.append(z.detach().float().cpu())
        return z, torch.stack(states, dim=0)

    def choose_demo_trajectory_speakers(primary_spk: str, count=None):
        count = max(1, int(demo_trajectory_speakers if count is None else count))
        chosen = [primary_spk]
        if count <= 1:
            return chosen
        candidates = [s for s in spk_list if s != primary_spk]
        if candidates:
            chosen.extend(random.sample(candidates, min(count - 1, len(candidates))))
        return chosen

    def save_demo_trajectory_plot(
        path,
        trajectory_items,
        *,
        frame_count: int,
        projection: str,
        dims,
        pool: str,
        title: str,
        plt,
        paper_style: bool = True,
        display_x_scale: float = 1.0,
        display_y_scale: float = 1.0,
        canonical_color: str = "#1f77b4",
        speaker_colors=None,
        annotate_points: bool = False,
        export_csv: bool = True,
    ):
        if not trajectory_items:
            return None
        trace_tensors = []
        for item in trajectory_items:
            trace_tensors.append(item["trace"])
            if item.get("reverse_trace") is not None:
                trace_tensors.append(item["reverse_trace"])
            if item.get("canonical_trace") is not None:
                trace_tensors.append(item["canonical_trace"])
        K = min(int(trace.shape[2]) for trace in trace_tensors)
        D = int(trace_tensors[0].shape[-1])
        if K <= 0 or D <= 0:
            return None

        pool = str(pool).lower()
        frame_level = pool in {"frame", "frames", "frame_level"}
        if frame_level:
            n_frames = max(1, min(int(frame_count), K))
            frame_idx = torch.linspace(0, K - 1, steps=n_frames).round().long().unique()
        else:
            frame_idx = None

        def flatten_trace_points(trace):
            if frame_level:
                return trace[:, :, frame_idx, :].reshape(-1, D).float()
            return trace[:, :, :K, :].mean(dim=2).reshape(-1, D).float()

        points = torch.cat([flatten_trace_points(trace) for trace in trace_tensors], dim=0).float()

        use_pca = projection == "pca" and points.shape[0] >= 2 and D >= 2
        center = points.mean(dim=0, keepdim=True)
        basis = None
        axis_label = None
        if use_pca:
            try:
                _, _, vh = torch.linalg.svd(points - center, full_matrices=False)
                basis = vh[:2].T.contiguous()
                axis_label = ("PC1", "PC2")
            except Exception:
                basis = None
        if basis is None:
            d0 = max(0, min(D - 1, int(dims[0])))
            d1 = max(0, min(D - 1, int(dims[1])))
            if d1 == d0 and D > 1:
                d1 = 1 if d0 == 0 else 0
            basis = torch.zeros(D, 2)
            basis[d0, 0] = 1.0
            basis[d1, 1] = 1.0
            center = torch.zeros_like(center)
            axis_label = (f"dim{d0}", f"dim{d1}")

        def project(trace):
            if frame_level:
                trace_sel = trace[:, :, frame_idx, :].float()
                shape = trace_sel.shape[:3]
                coords = (trace_sel.reshape(-1, D) - center) @ basis
                coords = coords.reshape(shape[0], shape[1], shape[2], 2)
                coords[..., 0] *= display_x_scale
                coords[..., 1] *= display_y_scale
                return coords.numpy()
            trace_sel = trace[:, :, :K, :].float().mean(dim=2)
            coords = (trace_sel - center) @ basis
            coords[..., 0] *= display_x_scale
            coords[..., 1] *= display_y_scale
            return coords.numpy()

        # Presentation convention: canonical/text samples on the left, speech/source
        # endpoints on the right. PCA signs are arbitrary, so flip PC1 when needed.
        try:
            ref_coords = project(trajectory_items[0]["trace"])
            if frame_level:
                start_x = float(ref_coords[0].reshape(-1, 2)[:, 0].mean())
                end_x = float(np.concatenate([project(item["trace"])[-1].reshape(-1, 2) for item in trajectory_items], axis=0)[:, 0].mean())
            else:
                start_x = float(ref_coords[0, :, 0].mean())
                end_x = float(np.concatenate([project(item["trace"])[-1] for item in trajectory_items], axis=0)[:, 0].mean())
            if end_x < start_x:
                basis[:, 0] *= -1.0
        except Exception:
            pass

        if not speaker_colors:
            speaker_colors = ["#2ca02c", "#ff7f0e"]

        def export_projected_csv(csv_path):
            def iter_projected_points(coords, *, direction, item_idx, item, point_kind):
                label = item.get("label") or f"spk {item['spk']}"
                spk = item.get("spk", "")
                color = item.get("color") or speaker_colors[item_idx % len(speaker_colors)]
                n_steps = int(coords.shape[0])
                denom = max(1, n_steps - 1)
                if frame_level:
                    for step_idx in range(n_steps):
                        t_value = step_idx / denom
                        for sample_idx in range(coords.shape[1]):
                            for local_frame_idx in range(coords.shape[2]):
                                frame_value = int(frame_idx[local_frame_idx].item())
                                xy = coords[step_idx, sample_idx, local_frame_idx]
                                yield {
                                    "point_kind": point_kind,
                                    "direction": direction,
                                    "speaker_index": item_idx,
                                    "speaker_id": spk,
                                    "speaker_label": label,
                                    "color": color,
                                    "series_id": f"{direction}_{item_idx}_{sample_idx}_{frame_value}",
                                    "sample_index": sample_idx,
                                    "frame_index": frame_value,
                                    "step_index": step_idx,
                                    "t": t_value,
                                    "x": float(xy[0]),
                                    "y": float(xy[1]),
                                }
                else:
                    for step_idx in range(n_steps):
                        t_value = step_idx / denom
                        for sample_idx in range(coords.shape[1]):
                            xy = coords[step_idx, sample_idx]
                            yield {
                                "point_kind": point_kind,
                                "direction": direction,
                                "speaker_index": item_idx,
                                "speaker_id": spk,
                                "speaker_label": label,
                                "color": color,
                                "series_id": f"{direction}_{item_idx}_{sample_idx}",
                                "sample_index": sample_idx,
                                "frame_index": "mean",
                                "step_index": step_idx,
                                "t": t_value,
                                "x": float(xy[0]),
                                "y": float(xy[1]),
                            }

            fieldnames = [
                "point_kind",
                "direction",
                "speaker_index",
                "speaker_id",
                "speaker_label",
                "color",
                "series_id",
                "sample_index",
                "frame_index",
                "step_index",
                "t",
                "x",
                "y",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item_idx, item in enumerate(trajectory_items):
                    coords = project(item["trace"])
                    for row in iter_projected_points(
                        coords,
                        direction="forward",
                        item_idx=item_idx,
                        item=item,
                        point_kind="sample",
                    ):
                        writer.writerow(row)
                    if frame_level:
                        mean_curve = coords.mean(axis=(1, 2))[:, None, None, :]
                    else:
                        mean_curve = coords.mean(axis=1)[:, None, :]
                    for row in iter_projected_points(
                        mean_curve,
                        direction="forward",
                        item_idx=item_idx,
                        item=item,
                        point_kind="mean",
                    ):
                        writer.writerow(row)

                    reverse_trace = item.get("reverse_trace")
                    if reverse_trace is None:
                        continue
                    reverse_coords = project(reverse_trace)
                    for row in iter_projected_points(
                        reverse_coords,
                        direction="backward",
                        item_idx=item_idx,
                        item=item,
                        point_kind="sample",
                    ):
                        writer.writerow(row)
                    if frame_level:
                        reverse_mean = reverse_coords.mean(axis=(1, 2))[:, None, None, :]
                    else:
                        reverse_mean = reverse_coords.mean(axis=1)[:, None, :]
                    for row in iter_projected_points(
                        reverse_mean,
                        direction="backward",
                        item_idx=item_idx,
                        item=item,
                        point_kind="mean",
                    ):
                        writer.writerow(row)
                    canonical_trace = item.get("canonical_trace")
                    if item_idx == 0 and canonical_trace is not None:
                        canonical_coords = project(canonical_trace)
                        for row in iter_projected_points(
                            canonical_coords,
                            direction="reference",
                            item_idx=item_idx,
                            item=item,
                            point_kind="canonical",
                        ):
                            writer.writerow(row)

        if export_csv:
            csv_path = os.path.splitext(path)[0] + ".csv"
            try:
                export_projected_csv(csv_path)
            except Exception as exc:
                print(f"[TRAJ-CSV][WARN] failed to write {csv_path}: {repr(exc)}")

        def draw_cov_ellipse(ax, xy, color, *, alpha=0.13, linewidth=1.4, linestyle="-"):
            xy = np.asarray(xy, dtype=np.float64)
            if xy.ndim != 2 or xy.shape[0] < 3 or xy.shape[1] != 2:
                return
            finite = np.isfinite(xy).all(axis=1)
            xy = xy[finite]
            if xy.shape[0] < 3:
                return
            cov = np.cov(xy[:, 0], xy[:, 1])
            if not np.isfinite(cov).all():
                return
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 1e-12)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            width, height = 4.0 * np.sqrt(vals)
            from matplotlib.patches import Ellipse

            ell = Ellipse(
                xy=xy.mean(axis=0),
                width=float(width),
                height=float(height),
                angle=float(angle),
                facecolor=color,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(ell)

        fig, ax = plt.subplots(figsize=(9.2, 6.0))
        for item_idx, item in enumerate(trajectory_items):
            coords = project(item["trace"])
            color = item.get("color") or speaker_colors[item_idx % len(speaker_colors)]
            item_label = item.get("label") or f"spk {item['spk']}"
            if item.get("plot_reverse_only") and item.get("reverse_trace") is not None:
                reverse_coords = project(item["reverse_trace"])
                mean_label = item.get("mean_label") or item_label
                source_label = item.get("source_label")
                endpoint_label = item.get("endpoint_label", "backward endpoints")
                canonical_label = item.get("canonical_label", "canonical samples")
                line_color = item.get("line_color", color)
                mean_color = item.get("mean_color", color)
                source_color = item.get("source_color", color)
                endpoint_color = item.get("endpoint_color", color)
                line_alpha = float(item.get("line_alpha", 0.055 if frame_level else 0.16))
                line_width = float(item.get("line_width", 0.65 if frame_level else 0.95))
                mean_width = float(item.get("mean_width", 3.0))
                source_alpha = float(item.get("source_alpha", 0.72))
                source_size = float(item.get("source_size", 34))
                endpoint_alpha = float(item.get("endpoint_alpha", 0.9))
                endpoint_size = float(item.get("endpoint_size", 42))
                source_ellipse_alpha = float(item.get("source_ellipse_alpha", 0.13))
                endpoint_ellipse_alpha = float(item.get("endpoint_ellipse_alpha", 0.06))
                show_mean_path = bool(item.get("show_mean_path", True))
                if frame_level:
                    reverse_mean_curve = reverse_coords.mean(axis=(1, 2))
                    source_cloud = reverse_coords[0].reshape(-1, 2)
                    target_cloud = reverse_coords[-1].reshape(-1, 2)
                    for b_idx in range(reverse_coords.shape[1]):
                        for f_idx in range(reverse_coords.shape[2]):
                            ax.plot(
                                reverse_coords[:, b_idx, f_idx, 0],
                                reverse_coords[:, b_idx, f_idx, 1],
                                color=line_color,
                                alpha=line_alpha,
                                linewidth=line_width,
                                solid_capstyle="round",
                                zorder=2,
                            )
                else:
                    reverse_mean_curve = reverse_coords.mean(axis=1)
                    source_cloud = reverse_coords[0]
                    target_cloud = reverse_coords[-1]
                    for b_idx in range(reverse_coords.shape[1]):
                        ax.plot(
                            reverse_coords[:, b_idx, 0],
                            reverse_coords[:, b_idx, 1],
                            color=line_color,
                            alpha=line_alpha,
                            linewidth=line_width,
                            solid_capstyle="round",
                            zorder=2,
                        )
                if show_mean_path:
                    ax.plot(
                        reverse_mean_curve[:, 0],
                        reverse_mean_curve[:, 1],
                        color=mean_color,
                        linewidth=mean_width,
                        label=mean_label,
                        solid_capstyle="round",
                        zorder=3,
                    )
                    if reverse_mean_curve.shape[0] >= 2:
                        ax.annotate(
                            "",
                            xy=(reverse_mean_curve[-1, 0], reverse_mean_curve[-1, 1]),
                            xytext=(reverse_mean_curve[-2, 0], reverse_mean_curve[-2, 1]),
                            arrowprops={"arrowstyle": "->", "color": mean_color, "lw": max(1.4, mean_width - 0.8)},
                        )
                ax.scatter(
                    source_cloud[:, 0],
                    source_cloud[:, 1],
                    color=source_color,
                    s=source_size,
                    alpha=source_alpha,
                    label=source_label,
                    zorder=4,
                )
                draw_cov_ellipse(ax, source_cloud, source_color, alpha=source_ellipse_alpha)
                if item.get("show_backward_endpoints"):
                    ax.scatter(
                        target_cloud[:, 0],
                        target_cloud[:, 1],
                        color=endpoint_color,
                        marker="x",
                        s=endpoint_size,
                        alpha=endpoint_alpha,
                        label=endpoint_label,
                        zorder=6,
                    )
                    draw_cov_ellipse(
                        ax,
                        target_cloud,
                        endpoint_color,
                        alpha=endpoint_ellipse_alpha,
                        linewidth=1.1,
                        linestyle="--",
                    )
                if item_idx == 0:
                    canonical_trace = item.get("canonical_trace")
                    if canonical_trace is not None:
                        canonical_coords = project(canonical_trace)
                        if frame_level:
                            target_cloud = canonical_coords[0].reshape(-1, 2)
                        else:
                            target_cloud = canonical_coords[0]
                    ax.scatter(
                        target_cloud[:, 0],
                        target_cloud[:, 1],
                        color=canonical_color,
                        edgecolors="white",
                        linewidths=0.6,
                        s=34 if target_cloud.shape[0] > 1 else 64,
                        label=canonical_label,
                        zorder=5,
                    )
                    draw_cov_ellipse(ax, target_cloud, canonical_color, alpha=0.10, linewidth=1.1)
                continue
            if frame_level:
                mean_curve = coords.mean(axis=(1, 2))
                for b_idx in range(coords.shape[1]):
                    for f_idx in range(coords.shape[2]):
                        ax.plot(
                            coords[:, b_idx, f_idx, 0],
                            coords[:, b_idx, f_idx, 1],
                            color=color,
                            alpha=0.055,
                            linewidth=0.65,
                            solid_capstyle="round",
                        )
                end_cloud = coords[-1].reshape(-1, 2)
            else:
                mean_curve = coords.mean(axis=1)
                end_cloud = coords[-1]
                for b_idx in range(coords.shape[1]):
                    ax.plot(
                        coords[:, b_idx, 0],
                        coords[:, b_idx, 1],
                        color=color,
                        alpha=0.18,
                        linewidth=1.0,
                        solid_capstyle="round",
                        zorder=2,
                    )

            ax.plot(
                mean_curve[:, 0],
                mean_curve[:, 1],
                color=color,
                linewidth=3.0,
                label=item_label,
                solid_capstyle="round",
                zorder=3,
            )
            if mean_curve.shape[0] >= 2:
                ax.annotate(
                    "",
                    xy=(mean_curve[-1, 0], mean_curve[-1, 1]),
                    xytext=(mean_curve[-2, 0], mean_curve[-2, 1]),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
                )
            ax.scatter(end_cloud[:, 0], end_cloud[:, 1], color=color, s=34, alpha=0.72, zorder=4)
            draw_cov_ellipse(ax, end_cloud, color)
            reverse_trace = item.get("reverse_trace")
            if reverse_trace is not None:
                reverse_coords = project(reverse_trace)
                if frame_level:
                    reverse_mean_curve = reverse_coords.mean(axis=(1, 2))
                    return_cloud = reverse_coords[-1].reshape(-1, 2)
                else:
                    reverse_mean_curve = reverse_coords.mean(axis=1)
                    return_cloud = reverse_coords[-1]
                    for b_idx in range(reverse_coords.shape[1]):
                        ax.plot(
                            reverse_coords[:, b_idx, 0],
                            reverse_coords[:, b_idx, 1],
                            color=color,
                            alpha=0.14,
                            linewidth=0.85,
                            linestyle="--",
                            dash_capstyle="round",
                            zorder=2,
                        )
                ax.plot(
                    reverse_mean_curve[:, 0],
                    reverse_mean_curve[:, 1],
                    color=color,
                    linewidth=2.2,
                    linestyle="--",
                    alpha=0.85,
                    label=item.get("reverse_label") or f"{item_label} bwd",
                    dash_capstyle="round",
                    zorder=3,
                )
                if reverse_mean_curve.shape[0] >= 2:
                    ax.annotate(
                        "",
                        xy=(reverse_mean_curve[-1, 0], reverse_mean_curve[-1, 1]),
                        xytext=(reverse_mean_curve[-2, 0], reverse_mean_curve[-2, 1]),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6, "linestyle": "--"},
                    )
                ax.scatter(
                    return_cloud[:, 0],
                    return_cloud[:, 1],
                    color=color,
                    marker="x",
                    s=36,
                    alpha=0.9,
                    zorder=6,
                )
                draw_cov_ellipse(ax, return_cloud, color, alpha=0.06, linewidth=1.1, linestyle="--")
            if item_idx == 0:
                if frame_level:
                    start_cloud = coords[0].reshape(-1, 2)
                else:
                    start_cloud = coords[0]
                start_label = "canonical samples" if start_cloud.shape[0] > 1 else "text distribution mean"
                ax.scatter(
                    start_cloud[:, 0],
                    start_cloud[:, 1],
                    color=canonical_color,
                    edgecolors="white",
                    linewidths=0.6,
                    s=34 if start_cloud.shape[0] > 1 else 64,
                    label=start_label,
                    zorder=5,
                )
                draw_cov_ellipse(ax, start_cloud, canonical_color, alpha=0.10, linewidth=1.1)

        ax.set_title(title)
        ax.set_xlabel(axis_label[0])
        ax.set_ylabel(axis_label[1])
        if paper_style:
            ax.grid(True, alpha=0.12, linewidth=0.8)
            ax.set_facecolor("#fbfbfb")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if annotate_points:
                try:
                    start_for_label = project(trajectory_items[0]["trace"])[0]
                    start_for_label = start_for_label.reshape(-1, 2) if frame_level else start_for_label
                    end_for_label = np.concatenate(
                        [
                            project(item["trace"])[-1].reshape(-1, 2) if frame_level else project(item["trace"])[-1]
                            for item in trajectory_items
                        ],
                        axis=0,
                    )
                    ax.text(
                        float(start_for_label[:, 0].mean()),
                        float(start_for_label[:, 1].mean()),
                        r"$z_c$ samples",
                        ha="right",
                        va="bottom",
                        fontsize=10,
                        color=canonical_color,
                    )
                    ax.text(
                        float(end_for_label[:, 0].mean()),
                        float(end_for_label[:, 1].mean()),
                        r"$z_s$ endpoints",
                        ha="left",
                        va="bottom",
                        fontsize=10,
                        color="#333333",
                    )
                except Exception:
                    pass
        else:
            ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    spk_sim_state = {"classifier": None, "failed": False}

    def get_spk_sim_classifier():
        if (not spk_sim_enable) or spk_sim_state["failed"]:
            return None
        if spk_sim_state["classifier"] is not None:
            return spk_sim_state["classifier"]
        try:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier

            savedir = spk_sim_savedir
            if savedir is None:
                safe_name = spk_sim_model.replace("/", "_").replace(":", "_")
                savedir = os.path.join(ckpt_dir, "speechbrain", safe_name)
            classifier = EncoderClassifier.from_hparams(
                source=spk_sim_model,
                savedir=savedir,
                run_opts={"device": device},
            )
            if hasattr(classifier, "eval"):
                classifier.eval()
            spk_sim_state["classifier"] = classifier
            print(f"[SPK-SIM] loaded model={spk_sim_model} savedir={savedir}")
            return classifier
        except Exception as exc:
            spk_sim_state["failed"] = True
            print(f"[SPK-SIM][WARN] disabled; failed to load model={spk_sim_model}. error={repr(exc)}")
            return None

    @torch.no_grad()
    def compute_spk_similarity(gen_wav: torch.Tensor, ref_wav_path: str):
        classifier = get_spk_sim_classifier()
        if classifier is None or not ref_wav_path:
            return None
        if not os.path.exists(ref_wav_path):
            print(f"[SPK-SIM][WARN] reference wav missing: {ref_wav_path}")
            return None
        try:
            gen = gen_wav.detach().float().view(1, -1).to(device)
            ref_np, _ = librosa.load(ref_wav_path, sr=sampling_rate, mono=True)
            ref = torch.from_numpy(np.asarray(ref_np, dtype=np.float32)).view(1, -1).to(device)
            emb_gen = classifier.encode_batch(gen).detach().float().reshape(1, -1)
            emb_ref = classifier.encode_batch(ref).detach().float().reshape(1, -1)
            return float(F.cosine_similarity(emb_gen, emb_ref, dim=-1).item())
        except Exception as exc:
            print(f"[SPK-SIM][WARN] failed for ref={ref_wav_path}. error={repr(exc)}")
            return None

    demo_utmos_state = {"model": None, "failed": False}

    def get_demo_utmos_model():
        if (not demo_eval_utmos) or demo_utmos_state["failed"]:
            return None
        if demo_utmos_state["model"] is not None:
            return demo_utmos_state["model"]
        try:
            print(f"[UTMOS] loading torch.hub {demo_utmos_repo} {demo_utmos_model}")
            try:
                model = torch.hub.load(demo_utmos_repo, demo_utmos_model, trust_repo=True)
            except TypeError:
                model = torch.hub.load(demo_utmos_repo, demo_utmos_model)
            if hasattr(model, "to"):
                model = model.to(device)
            if hasattr(model, "eval"):
                model.eval()
            demo_utmos_state["model"] = model
            return model
        except Exception as exc:
            demo_utmos_state["failed"] = True
            print(f"[UTMOS][WARN] disabled; failed to load {demo_utmos_repo} {demo_utmos_model}. error={repr(exc)}")
            return None

    @torch.no_grad()
    def score_demo_utmos(wav: torch.Tensor):
        model = get_demo_utmos_model()
        if model is None:
            return None
        try:
            wav_in = wav.detach().float().view(1, -1).to(device)
            try:
                score = model(wav_in, int(sampling_rate))
            except TypeError:
                score = model(wav_in)
            if torch.is_tensor(score):
                return float(score.detach().float().reshape(-1).mean().cpu().item())
            return float(np.asarray(score).reshape(-1).mean())
        except Exception as exc:
            print(f"[UTMOS][WARN] scoring failed: {repr(exc)}")
            return None

    demo_whisper_state = {"state": None, "failed": False}

    def get_demo_whisper_state():
        if (not demo_eval_whisper) or demo_whisper_state["failed"]:
            return None
        if demo_whisper_state["state"] is not None:
            return demo_whisper_state["state"]
        print(f"[WHISPER] loading {demo_whisper_model}")
        imported_whisper_path = None
        imported_whisper_error = None
        try:
            import whisper

            imported_whisper_path = getattr(whisper, "__file__", None)
            if hasattr(whisper, "load_model"):
                model = whisper.load_model(demo_whisper_model, device=device)
                demo_whisper_state["state"] = {"backend": "openai-whisper", "model": model}
                return demo_whisper_state["state"]
        except Exception as exc:
            imported_whisper_error = repr(exc)

        faster_error = None
        try:
            from faster_whisper import WhisperModel

            fw_device = "cuda" if str(device).startswith("cuda") else "cpu"
            compute_type = "float16" if fw_device == "cuda" else "int8"
            model = WhisperModel(demo_whisper_model, device=fw_device, compute_type=compute_type)
            print("[WHISPER] using faster-whisper backend")
            demo_whisper_state["state"] = {"backend": "faster-whisper", "model": model}
            return demo_whisper_state["state"]
        except Exception as exc:
            faster_error = repr(exc)

        transformers_error = None
        try:
            from transformers import pipeline

            model_id = demo_whisper_model if "/" in str(demo_whisper_model) else f"openai/whisper-{demo_whisper_model}"
            pipe_device = 0 if str(device).startswith("cuda") else -1
            model = pipeline("automatic-speech-recognition", model=model_id, device=pipe_device)
            print(f"[WHISPER] using transformers backend {model_id}")
            demo_whisper_state["state"] = {"backend": "transformers-whisper", "model": model}
            return demo_whisper_state["state"]
        except Exception as exc:
            transformers_error = repr(exc)

        demo_whisper_state["failed"] = True
        print(
            "[WHISPER][WARN] disabled; could not load ASR backend. "
            f"whisper_path={imported_whisper_path!r} whisper_error={imported_whisper_error!r} "
            f"faster_whisper_error={faster_error!r} transformers_error={transformers_error!r}"
        )
        return None

    def transcribe_demo_whisper(wav_path: str):
        state = get_demo_whisper_state()
        if state is None:
            return None, None
        backend = state.get("backend")
        model = state.get("model")
        try:
            if backend == "openai-whisper":
                audio, _ = librosa.load(wav_path, sr=16000, mono=True)
                result = model.transcribe(
                    audio,
                    language="en",
                    fp16=str(device).startswith("cuda"),
                    verbose=False,
                )
                return result.get("text", ""), backend
            if backend == "faster-whisper":
                segments, _ = model.transcribe(wav_path, language="en", beam_size=5)
                return " ".join(seg.text for seg in segments), backend
            if backend == "transformers-whisper":
                result = model(wav_path)
                if isinstance(result, dict):
                    return result.get("text", ""), backend
                return str(result), backend
            print(f"[WHISPER][WARN] unsupported backend={backend}")
            return None, backend
        except Exception as exc:
            print(f"[WHISPER][WARN] transcription failed for {wav_path}: {repr(exc)}")
            return None, backend

    demo_bigvgan_mel_state = {
        "h": bigvgan_h,
        "fn": bigvgan_get_mel_spectrogram,
        "failed": False,
    }

    def get_demo_bigvgan_mel_frontend():
        if demo_generated_mel_frontend != "bigvgan" or demo_bigvgan_mel_state["failed"]:
            return None, None
        if demo_bigvgan_mel_state["h"] is not None and demo_bigvgan_mel_state["fn"] is not None:
            return demo_bigvgan_mel_state["h"], demo_bigvgan_mel_state["fn"]
        try:
            import bigvgan
            from meldataset import get_mel_spectrogram as mel_fn

            from huggingface_hub import hf_hub_download

            if os.path.isdir(bigvgan_name):
                bigvgan_config_file = os.path.join(bigvgan_name, "config.json")
            else:
                bigvgan_config_file = hf_hub_download(
                    repo_id=bigvgan_name,
                    filename="config.json",
                    local_files_only=True,
                )
            mel_h = bigvgan.load_hparams_from_json(bigvgan_config_file)
            demo_bigvgan_mel_state["h"] = mel_h
            demo_bigvgan_mel_state["fn"] = mel_fn
            print(f"[DEMO-MEL] using BigVGAN mel frontend config={bigvgan_config_file}")
            return mel_h, mel_fn
        except Exception as exc:
            demo_bigvgan_mel_state["failed"] = True
            print(f"[DEMO-MEL][WARN] BigVGAN mel frontend unavailable; falling back to librosa. error={repr(exc)}")
            return None, None

    @torch.no_grad()
    def compute_generated_wav_mel_image(wav: torch.Tensor):
        wav_cpu = wav.detach().float().view(-1).cpu()
        if demo_generated_mel_frontend == "bigvgan":
            mel_h, mel_fn = get_demo_bigvgan_mel_frontend()
            if mel_h is not None and mel_fn is not None:
                try:
                    target_sr = int(getattr(mel_h, "sampling_rate", sampling_rate))
                    wav_np = wav_cpu.numpy()
                    if target_sr != int(sampling_rate):
                        wav_np = librosa.resample(wav_np, orig_sr=int(sampling_rate), target_sr=target_sr)
                    wav_t = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).view(1, -1).to(device)
                    mel = mel_fn(wav_t, mel_h)
                    if torch.is_tensor(mel):
                        mel_np = mel[0].detach().float().cpu().numpy()
                    else:
                        mel_np = np.asarray(mel)[0]
                    return mel_np, f"bigvgan_sr{target_sr}"
                except Exception as exc:
                    print(f"[DEMO-MEL][WARN] BigVGAN mel extraction failed; falling back to librosa. error={repr(exc)}")

        wav_np = wav_cpu.numpy()
        n_fft = 1024
        win_length = 1024
        hop_length = max(1, int(hop_size))
        mel_np = librosa.feature.melspectrogram(
            y=wav_np,
            sr=int(sampling_rate),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=80,
            fmin=0,
            fmax=float(sampling_rate) / 2.0,
            power=1.0,
        )
        mel_np = librosa.amplitude_to_db(np.maximum(mel_np, 1e-8), ref=np.max)
        return mel_np, f"librosa_sr{sampling_rate}"

    @torch.no_grad()
    def asr_eval_full_one_sample(
        step_tag: str,
        idx: int,
        rows=None,
        label: str = "ASR-FULL",
        source_name: str = "demo",
    ):
        rows = demo_aligned_rows if rows is None else rows
        item = rows[idx]
        wav_path = item["wav"]
        text = _row_text(item)

        zS_log_cpu = load_logmel_full_cached(wav_path, prefer_gpu=gpu_mel_cache)
        K0 = int(zS_log_cpu.shape[0])
        zS_raw_full = zS_log_cpu.to(device).unsqueeze(0)
        zS_full = (zS_raw_full - mu_b) / std_b
        asr_t0 = None
        if demo_rtf:
            maybe_cuda_sync()
            asr_t0 = time.perf_counter()
        if asr_demo_whole_utterance:
            mask_full = torch.ones(1, K0, device=device, dtype=torch.bool)
            spk_e_asr_demo = asr_spk_cond_from_name(_row_speaker(item), dtype=zS_full.dtype)
            style_e_asr_demo = asr_style_cond_from_source(zS_full, mask_full, spk_e_asr_demo, dtype=zS_full.dtype)
            cfg_asr_demo = asr_cfg_flag_value(spk_e_asr_demo)
            if full_asr_use_euler:
                zT_full = euler_integrate(
                    vf,
                    zS_full,
                    mask_full,
                    steps=asr_demo_steps,
                    direction=-1,
                    cfg_flag_value=cfg_asr_demo,
                    spk_e=spk_e_asr_demo,
                    style_e=style_e_asr_demo,
                )
            else:
                zT_full = heun_integrate(
                    vf,
                    zS_full,
                    mask_full,
                    steps=asr_demo_steps,
                    direction=-1,
                    cfg_scale=1.0,
                    spk_e=spk_e_asr_demo,
                    style_e=style_e_asr_demo,
                )
            zC_hat = source_to_canonical(zT_full, mask_full) if source_to_canonical is not None else zT_full
            if canonical_posterior is not None:
                zC_hat, _ = canonical_posterior(zC_hat, mask_full)
        else:
            core = int(max(32, full_asr_chunk_core))
            ctx = int(max(0, full_asr_chunk_ctx))
            chunk_logits = []
            pos = 0
            while pos < K0:
                core_s = pos
                core_e = min(K0, pos + core)
                s = max(0, core_s - ctx)
                e = min(K0, core_e + ctx)

                zS_chunk = zS_full[:, s:e]
                mask_chunk = torch.ones(1, e - s, device=device, dtype=torch.bool)
                spk_e_asr_demo = asr_spk_cond_from_name(_row_speaker(item), dtype=zS_chunk.dtype)
                style_e_asr_demo = asr_style_cond_from_source(zS_chunk, mask_chunk, spk_e_asr_demo, dtype=zS_chunk.dtype)
                cfg_asr_demo = asr_cfg_flag_value(spk_e_asr_demo)

                if full_asr_use_euler:
                    zT_chunk = euler_integrate(
                        vf,
                        zS_chunk,
                        mask_chunk,
                        steps=asr_demo_steps,
                        direction=-1,
                        cfg_flag_value=cfg_asr_demo,
                        spk_e=spk_e_asr_demo,
                        style_e=style_e_asr_demo,
                    )
                else:
                    zT_chunk = heun_integrate(
                        vf,
                        zS_chunk,
                        mask_chunk,
                        steps=asr_demo_steps,
                        direction=-1,
                        cfg_scale=1.0,
                        spk_e=spk_e_asr_demo,
                        style_e=style_e_asr_demo,
                    )

                zC_chunk = source_to_canonical(zT_chunk, mask_chunk) if source_to_canonical is not None else zT_chunk
                if canonical_posterior is not None:
                    zC_chunk, _ = canonical_posterior(zC_chunk, mask_chunk)
                keep_s = core_s - s
                keep_e = core_e - s
                chunk_logits.append(zC_chunk[:, keep_s:keep_e])
                pos = core_e

            zC_hat = torch.cat(chunk_logits, dim=1)
        mask_hat = torch.ones(1, zC_hat.shape[1], device=device, dtype=torch.bool)
        zC_hat_ctc, mask_hat_ctc, k_hat_ctc = prepare_ctc_input(
            zC_hat,
            mask_hat,
            apply_subsample=(ctc_subsample_apply_to in {"hat", "both"}),
        )
        logits_hat = text_ctc_head(zC_hat_ctc, mask_hat_ctc)
        hyp = decode_ctc_demo_text(logits_hat, k_hat_ctc[0])
        hyp_source = None
        source_wer = None
        if source_ctc_head is not None:
            mask_source = torch.ones(1, K0, device=device, dtype=torch.bool)
            zS_source_ctc, mask_source_ctc, k_source_ctc = prepare_ctc_input(
                zS_full,
                mask_source,
                apply_subsample=(ctc_subsample_apply_to in {"hat", "both"}),
            )
            logits_source = source_ctc_head(zS_source_ctc, mask_source_ctc)
            hyp_source = decode_ctc_demo_text(logits_source, k_source_ctc[0])
            source_wer = word_error_rate_text(text, hyp_source)
        asr_elapsed = None
        if demo_rtf and asr_t0 is not None:
            maybe_cuda_sync()
            asr_elapsed = time.perf_counter() - asr_t0
        wer = word_error_rate_text(text, hyp)

        spk_name = _row_speaker(item)
        print(f"\n[{label} @ {step_tag}] (source={source_name} spkGT={spk_name})\nGT : {text}\nHYP: {hyp}\nWER(norm): {wer:.4f}")
        if hyp_source is not None:
            print(f"HYP-zS: {hyp_source}\nWER-zS(norm): {source_wer:.4f}")
        if asr_elapsed is not None:
            print(demo_rtf_line(label, asr_elapsed, demo_audio_sec_from_frames(K0)))

    @torch.no_grad()
    def tts_demo_from_text(
        step_tag: str,
        text: str,
        spk_pick: str = None,
        file_prefix: str = "tts_demo",
        banner_prefix: str = "TTS-DEMO",
        ref_wav_path: str = None,
        ref_spk: str = None,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tts_t0 = None
        if demo_rtf:
            maybe_cuda_sync()
            tts_t0 = time.perf_counter()

        if spk_pick is None:
            spk_pick = random.choice(spk_list)
        spk_id = torch.tensor([spk2id[spk_pick]], device=device, dtype=torch.long)
        if ref_wav_path is None:
            ref_wav_path = train_ref_wav_by_spk.get(spk_pick)
        spk_e = speaker_cond_from_ref_paths([ref_wav_path], spk_id)

        h_enc, maskL, mu_tok, logvar_tok, align_mu_tok, align_logvar_tok = encode_text_batch([text])
        L_valid = int(maskL.sum().item())
        limit = getattr(vf.rope, "max_seq_len", 4096)
        if len_pred is not None:
            k_pred_full_raw = int(max(16, round(float(len_pred(h_enc, maskL).item()))))
            k_pred_full = min(max(k_pred_full_raw, L_valid), limit)
        else:
            log_dur = dur_pred(h_enc, maskL)
            dur = (torch.exp(log_dur) - 1.0) * maskL.float()
            dur_sum = float(dur.sum().item())
            k_pred_full = min(max(int(round(dur_sum)), L_valid, 16), limit)
        log_dur = dur_pred(h_enc, maskL)
        dur = (torch.exp(log_dur) - 1.0) * maskL.float()
        dur_int, _ = durations_to_int_and_fixsum(dur, maskL, k_pred_full)

        mu_feats = []
        lv_feats = []
        for i in range(L_valid):
            d = int(dur_int[i].item())
            mu_tok_i = mu_tok[0, i:i + 1]
            lv_tok_i = logvar_tok[0, i:i + 1]
            mu_feats.append(mu_tok_i.repeat(d, 1))
            lv_feats.append(lv_tok_i.repeat(d, 1))

        if len(mu_feats) == 0:
            raise RuntimeError("No valid text tokens for TTS demo")
        zT_mean = torch.cat(mu_feats, dim=0).unsqueeze(0)[:, :k_pred_full]
        zT_logvar = torch.cat(lv_feats, dim=0).unsqueeze(0)[:, :k_pred_full]

        if demo_prior_temp > 0:
            eps = torch.randn_like(zT_mean)
            zT0 = zT_mean + demo_prior_temp * torch.exp(0.5 * zT_logvar) * eps
        else:
            zT0 = zT_mean

        maskK = torch.ones(1, zT0.shape[1], device=device, dtype=torch.bool)
        style_e_demo = None
        if use_tts_style_latent:
            u_mu_p, u_logvar_p = tts_style_prior_dist(
                spk_e.to(dtype=zT0.dtype),
                zc=zT0,
                maskK=maskK,
                dtype=zT0.dtype,
            )
            if demo_style_temp > 0.0:
                style_e_demo = u_mu_p + float(demo_style_temp) * torch.exp(0.5 * u_logvar_p) * torch.randn_like(u_mu_p)
            else:
                style_e_demo = u_mu_p

        text_cond_demo = None
        if canonical_to_source is not None:
            zT0_source = canonical_to_source(
                zT0,
                maskK,
                spk_e=spk_e.to(dtype=zT0.dtype),
                style_e=style_e_demo if tts_style_into_source else None,
            )
            zT_mean_source = canonical_to_source(
                zT_mean,
                maskK,
                spk_e=spk_e.to(dtype=zT_mean.dtype),
                style_e=(
                    style_e_demo.to(dtype=zT_mean.dtype)
                    if (tts_style_into_source and style_e_demo is not None)
                    else None
                ),
            )
            if use_vf_canonical_text_cond:
                text_cond_demo = zT_mean
        else:
            zT0_source = zT0
            zT_mean_source = zT_mean
            if tts_style_to_source is not None and style_e_demo is not None:
                style_bias = tts_style_to_source(style_e_demo, spk_e.to(dtype=zT0.dtype)).to(dtype=zT0.dtype)
                zT0_source = zT0_source + float(tts_style_source_scale) * style_bias.unsqueeze(1)
                zT_mean_source = zT_mean_source + float(tts_style_source_scale) * style_bias.unsqueeze(1)
            if use_vf_canonical_text_cond:
                text_cond_demo = zT_mean

        spk_e_demo = spk_e.to(dtype=zT0_source.dtype)
        if tts_source_cond is not None and demo_prior_temp > 0 and canonical_to_source is None:
            source_delta_demo = zT0_source - zT_mean_source
            spk_e_demo = spk_e_demo + tts_source_cond_scale * tts_source_cond(source_delta_demo, maskK).to(dtype=spk_e_demo.dtype)
        zS_pred = heun_integrate(
            vf,
            zT0_source,
            maskK,
            steps=ode_steps_eval,
            direction=+1,
            cfg_scale=demo_cfg_scale,
            spk_e=spk_e_demo,
            style_e=style_e_demo,
            text_cond=text_cond_demo,
        )
        zS_ref = mel_refiner(zS_pred, cond=h_enc, cond_mask=maskL) if mel_refiner is not None else zS_pred

        mel_log = (zS_ref * std_b + mu_b).float()
        if speech_backend == "svae":
            mel_log_clamped = mel_log
            mel_for_vocoder = mel_log_clamped
        else:
            mel_log_clamped = mel_log.clamp(mel_floor, mel_ceil)
            mel_for_vocoder = mel_log_clamped.transpose(1, 2).contiguous()

        print(f"[{banner_prefix} {step_tag}] spk={spk_pick} L={L_valid} K={k_pred_full} cfg={demo_cfg_scale} prior_temp={demo_prior_temp}")

        mel_base = os.path.join(demo_dir, f"{file_prefix}_{step_tag}_spk{spk_pick}")
        if demo_plot_trajectory:
            try:
                trajectory_items = []
                n_traj = max(1, int(demo_trajectory_samples))
                zT_mean_traj = zT_mean.expand(n_traj, -1, -1).contiguous()
                zT_logvar_traj = zT_logvar.expand(n_traj, -1, -1).contiguous()
                if n_traj > 1 and demo_prior_temp > 0:
                    eps_traj = torch.randn_like(zT_mean_traj)
                    zT0_traj = zT_mean_traj + demo_prior_temp * torch.exp(0.5 * zT_logvar_traj) * eps_traj
                    zT0_traj[:1] = zT0
                elif n_traj > 1:
                    zT0_traj = zT_mean_traj.clone()
                else:
                    zT0_traj = zT0
                maskK_traj = maskK.expand(n_traj, -1).contiguous()
                def append_traj_for_speaker(traj_spk: str, traj_spk_e: torch.Tensor):
                    traj_spk_e = traj_spk_e.to(device=device, dtype=zT0_traj.dtype).expand(n_traj, -1).contiguous()
                    traj_style_e = None
                    if use_tts_style_latent:
                        traj_u_mu, traj_u_logvar = tts_style_prior_dist(
                            traj_spk_e,
                            zc=zT0_traj,
                            maskK=maskK_traj,
                            dtype=zT0_traj.dtype,
                        )
                        if demo_style_temp > 0.0:
                            traj_style_e = traj_u_mu + float(demo_style_temp) * torch.exp(
                                0.5 * traj_u_logvar
                            ) * torch.randn_like(traj_u_mu)
                        else:
                            traj_style_e = traj_u_mu
                    traj_text_cond = None
                    if canonical_to_source is not None:
                        traj_zT0_source = canonical_to_source(
                            zT0_traj,
                            maskK_traj,
                            spk_e=traj_spk_e,
                            style_e=(
                                traj_style_e.to(dtype=zT0_traj.dtype)
                                if (tts_style_into_source and traj_style_e is not None)
                                else None
                            ),
                        )
                        traj_zT_mean_source = canonical_to_source(
                            zT_mean_traj,
                            maskK_traj,
                            spk_e=traj_spk_e.to(dtype=zT_mean_traj.dtype),
                            style_e=(
                                traj_style_e.to(dtype=zT_mean_traj.dtype)
                                if (tts_style_into_source and traj_style_e is not None)
                                else None
                            ),
                        )
                        if use_vf_canonical_text_cond:
                            traj_text_cond = zT_mean_traj
                    else:
                        traj_zT0_source = zT0_traj
                        traj_zT_mean_source = zT_mean_traj
                        if tts_style_to_source is not None and traj_style_e is not None:
                            traj_style_bias = tts_style_to_source(
                                traj_style_e,
                                traj_spk_e.to(dtype=zT0_traj.dtype),
                            ).to(dtype=zT0_traj.dtype)
                            traj_zT0_source = traj_zT0_source + float(tts_style_source_scale) * traj_style_bias.unsqueeze(1)
                            traj_zT_mean_source = traj_zT_mean_source + float(tts_style_source_scale) * traj_style_bias.unsqueeze(1)
                        if use_vf_canonical_text_cond:
                            traj_text_cond = zT_mean_traj

                    traj_spk_e_demo = traj_spk_e.to(dtype=traj_zT0_source.dtype)
                    if tts_source_cond is not None and demo_prior_temp > 0 and canonical_to_source is None:
                        traj_source_delta = traj_zT0_source - traj_zT_mean_source
                        traj_spk_e_demo = traj_spk_e_demo + tts_source_cond_scale * tts_source_cond(
                            traj_source_delta,
                            maskK_traj,
                        ).to(dtype=traj_spk_e_demo.dtype)

                    _, trace = heun_integrate_trace(
                        vf,
                        traj_zT0_source,
                        maskK_traj,
                        steps=ode_steps_eval,
                        direction=+1,
                        cfg_scale=demo_cfg_scale,
                        spk_e=traj_spk_e_demo,
                        style_e=traj_style_e,
                        text_cond=traj_text_cond,
                    )
                    reverse_trace = None
                    if demo_trajectory_reverse:
                        # Backward check: generated zS -> zc/source start, using the ASR speaker condition when enabled.
                        reverse_start = trace[-1].to(device=device, dtype=traj_zT0_source.dtype)
                        traj_asr_spk_e = None
                        if asr_use_spk_cond:
                            traj_asr_spk_e = float(asr_spk_scale) * traj_spk_e.to(dtype=reverse_start.dtype)
                        traj_asr_style_e = asr_style_cond_from_source(
                            reverse_start,
                            maskK_traj,
                            traj_asr_spk_e,
                            dtype=reverse_start.dtype,
                        )
                        _, reverse_trace = heun_integrate_trace(
                            vf,
                            reverse_start,
                            maskK_traj,
                            steps=ode_steps_eval,
                            direction=-1,
                            cfg_scale=1.0,
                            spk_e=traj_asr_spk_e,
                            style_e=traj_asr_style_e,
                            text_cond=None,
                        )
                    speaker_ord = len(trajectory_items)
                    speaker_label = (
                        f"Speaker {chr(ord('A') + speaker_ord)}"
                        if speaker_ord < 26
                        else f"Speaker {speaker_ord + 1}"
                    )
                    trajectory_items.append(
                        {
                            "spk": traj_spk,
                            "label": speaker_label,
                            "reverse_label": f"{speaker_label} bwd",
                            "trace": trace,
                            "reverse_trace": reverse_trace,
                        }
                    )

                append_traj_for_speaker(spk_pick, spk_e)
                for extra_spk in choose_demo_trajectory_speakers(spk_pick)[1:]:
                    append_traj_for_speaker(extra_spk, speaker_cond_from_name(extra_spk))

                traj_pool_tag = demo_trajectory_pool.replace("/", "_").replace(" ", "_")
                reverse_tag = "_bwd" if demo_trajectory_reverse else ""
                traj_path = f"{mel_base}_traj_{demo_trajectory_projection}_{traj_pool_tag}_n{n_traj}{reverse_tag}.png"
                saved_traj = save_demo_trajectory_plot(
                    traj_path,
                    trajectory_items,
                    frame_count=demo_trajectory_frames,
                    projection=demo_trajectory_projection,
                    dims=demo_trajectory_dims,
                    pool=demo_trajectory_pool,
                    title=f"{banner_prefix} {step_tag}: VF trajectories",
                    plt=plt,
                    paper_style=demo_trajectory_paper_style,
                    display_x_scale=demo_trajectory_display_x_scale,
                    display_y_scale=demo_trajectory_display_y_scale,
                    canonical_color=demo_trajectory_canonical_color,
                    speaker_colors=demo_trajectory_speaker_colors,
                    annotate_points=demo_trajectory_annotate_points,
                    export_csv=demo_trajectory_export_csv,
                )
                if saved_traj:
                    print(f"[TRAJ {banner_prefix}] wrote {saved_traj}")
                if demo_trajectory_asr_many_to_one:
                    asr_items = []
                    for item_idx, item in enumerate(trajectory_items):
                        asr_start = item["trace"][-1].to(device=device, dtype=zT0_traj.dtype)
                        spk_e_asr_traj = None
                        if asr_use_spk_cond:
                            spk_e_asr_traj = asr_spk_cond_from_name(item.get("spk", ""), dtype=asr_start.dtype)
                            if spk_e_asr_traj is not None:
                                spk_e_asr_traj = spk_e_asr_traj.expand(asr_start.shape[0], -1).contiguous()
                        style_e_asr_traj = asr_style_cond_from_source(
                            asr_start,
                            maskK_traj,
                            spk_e_asr_traj,
                            dtype=asr_start.dtype,
                        )
                        _, asr_trace = heun_integrate_trace(
                            vf,
                            asr_start,
                            maskK_traj,
                            steps=ode_steps_eval,
                            direction=-1,
                            cfg_scale=1.0,
                            spk_e=spk_e_asr_traj,
                            style_e=style_e_asr_traj,
                            text_cond=None,
                        )
                        asr_items.append(
                            {
                                "spk": item.get("spk", ""),
                                "label": item.get("label") or f"Speaker {item_idx + 1}",
                                "trace": torch.flip(asr_trace, dims=(0,)),
                                "reverse_trace": asr_trace,
                                "canonical_trace": zT0_traj.detach().float().cpu().unsqueeze(0),
                                "plot_reverse_only": True,
                                "color": item.get("color")
                                or demo_trajectory_speaker_colors[item_idx % len(demo_trajectory_speaker_colors)],
                            }
                        )
                    asr_traj_path = (
                        f"{mel_base}_traj_asr_many_to_one_{demo_trajectory_projection}_{traj_pool_tag}"
                        f"_n{n_traj}.png"
                    )
                    saved_asr_traj = save_demo_trajectory_plot(
                        asr_traj_path,
                        asr_items,
                        frame_count=demo_trajectory_frames,
                        projection=demo_trajectory_projection,
                        dims=demo_trajectory_dims,
                        pool=demo_trajectory_pool,
                        title=f"{banner_prefix} {step_tag}: ASR many-to-one trajectories",
                        plt=plt,
                        paper_style=demo_trajectory_paper_style,
                        display_x_scale=demo_trajectory_display_x_scale,
                        display_y_scale=demo_trajectory_display_y_scale,
                        canonical_color=demo_trajectory_canonical_color,
                        speaker_colors=demo_trajectory_speaker_colors,
                        annotate_points=False,
                        export_csv=demo_trajectory_export_csv,
                    )
                    if saved_asr_traj:
                        print(f"[TRAJ-ASR {banner_prefix}] wrote {saved_asr_traj}")
                if demo_trajectory_asr_realization_plot:
                    real_spks = choose_demo_trajectory_speakers(
                        spk_pick,
                        demo_trajectory_asr_realization_speakers,
                    )
                    n_real_u = max(1, int(demo_trajectory_asr_realization_styles))
                    real_reverse_traces = []
                    real_canonical_refs = []
                    real_spk_labels = []
                    for real_spk in real_spks:
                        real_spk_e = speaker_cond_from_name(real_spk, dtype=zT_mean.dtype).to(device=device, dtype=zT_mean.dtype)
                        real_spk_e = real_spk_e.expand(n_real_u, -1).contiguous()
                        real_zT_mean = zT_mean.expand(n_real_u, -1, -1).contiguous()
                        real_zT0 = zT0.expand(n_real_u, -1, -1).contiguous()
                        real_maskK = maskK.expand(n_real_u, -1).contiguous()

                        real_style_e = None
                        if use_tts_style_latent:
                            real_u_mu, real_u_logvar = tts_style_prior_dist(
                                real_spk_e,
                                zc=real_zT0,
                                maskK=real_maskK,
                                dtype=real_zT0.dtype,
                            )
                            if demo_style_temp > 0.0:
                                real_style_e = real_u_mu + float(demo_style_temp) * torch.exp(
                                    0.5 * real_u_logvar
                                ) * torch.randn_like(real_u_mu)
                            else:
                                real_style_e = real_u_mu

                        real_text_cond = None
                        if canonical_to_source is not None:
                            real_zT0_source = canonical_to_source(
                                real_zT0,
                                real_maskK,
                                spk_e=real_spk_e,
                                style_e=(
                                    real_style_e.to(dtype=real_zT0.dtype)
                                    if (tts_style_into_source and real_style_e is not None)
                                    else None
                                ),
                            )
                            real_zT_mean_source = canonical_to_source(
                                real_zT_mean,
                                real_maskK,
                                spk_e=real_spk_e.to(dtype=real_zT_mean.dtype),
                                style_e=(
                                    real_style_e.to(dtype=real_zT_mean.dtype)
                                    if (tts_style_into_source and real_style_e is not None)
                                    else None
                                ),
                            )
                            if use_vf_canonical_text_cond:
                                real_text_cond = real_zT_mean
                        else:
                            real_zT0_source = real_zT0
                            real_zT_mean_source = real_zT_mean
                            if tts_style_to_source is not None and real_style_e is not None:
                                real_style_bias = tts_style_to_source(real_style_e, real_spk_e).to(dtype=real_zT0.dtype)
                                real_zT0_source = real_zT0_source + float(tts_style_source_scale) * real_style_bias.unsqueeze(1)
                                real_zT_mean_source = real_zT_mean_source + float(tts_style_source_scale) * real_style_bias.unsqueeze(1)
                            if use_vf_canonical_text_cond:
                                real_text_cond = real_zT_mean

                        real_spk_e_demo = real_spk_e.to(dtype=real_zT0_source.dtype)
                        if tts_source_cond is not None and demo_prior_temp > 0 and canonical_to_source is None:
                            real_source_delta = real_zT0_source - real_zT_mean_source
                            real_spk_e_demo = real_spk_e_demo + tts_source_cond_scale * tts_source_cond(
                                real_source_delta,
                                real_maskK,
                            ).to(dtype=real_spk_e_demo.dtype)

                        _, real_fwd_trace = heun_integrate_trace(
                            vf,
                            real_zT0_source,
                            real_maskK,
                            steps=ode_steps_eval,
                            direction=+1,
                            cfg_scale=demo_cfg_scale,
                            spk_e=real_spk_e_demo,
                            style_e=real_style_e,
                            text_cond=real_text_cond,
                        )
                        real_asr_start = real_fwd_trace[-1].to(device=device, dtype=real_zT0_source.dtype)
                        real_asr_spk_e = None
                        if asr_use_spk_cond:
                            real_asr_spk_e = asr_spk_cond_from_name(real_spk, dtype=real_asr_start.dtype)
                            if real_asr_spk_e is not None:
                                real_asr_spk_e = real_asr_spk_e.expand(n_real_u, -1).contiguous()
                        real_asr_style_e = asr_style_cond_from_source(
                            real_asr_start,
                            real_maskK,
                            real_asr_spk_e,
                            dtype=real_asr_start.dtype,
                        )
                        _, real_asr_trace = heun_integrate_trace(
                            vf,
                            real_asr_start,
                            real_maskK,
                            steps=ode_steps_eval,
                            direction=-1,
                            cfg_scale=1.0,
                            spk_e=real_asr_spk_e,
                            style_e=real_asr_style_e,
                            text_cond=None,
                        )
                        real_reverse_traces.append(real_asr_trace)
                        real_spk_labels.extend([real_spk] * n_real_u)

                        real_canon = real_zT_mean.detach().clone()
                        if demo_prior_temp > 0.0:
                            real_std = torch.exp(0.5 * zT_logvar.expand(n_real_u, -1, -1).contiguous())
                            real_canon = real_canon + float(demo_prior_temp) * real_std * torch.randn_like(real_canon)
                            real_canon[:1] = zT0
                        real_canonical_refs.append(real_canon.float().cpu())

                    if real_reverse_traces:
                        real_trace_all = torch.cat(real_reverse_traces, dim=1)
                        real_canonical_all = torch.cat(real_canonical_refs, dim=0).unsqueeze(0)
                        real_items = [
                            {
                                "spk": ",".join(real_spks),
                                "label": "speech realizations",
                                "mean_label": "mean backward path",
                                "source_label": "TTS-generated speech latents",
                                "endpoint_label": "backward endpoints",
                                "canonical_label": "canonical content",
                                "trace": torch.flip(real_trace_all, dims=(0,)),
                                "reverse_trace": real_trace_all,
                                "canonical_trace": real_canonical_all,
                                "plot_reverse_only": True,
                                "show_backward_endpoints": True,
                                "show_mean_path": False,
                                "color": "#6f6f6f",
                                "line_color": "#9a9a9a",
                                "line_alpha": 0.38,
                                "line_width": 1.05,
                                "mean_color": "#5f5f5f",
                                "mean_width": 2.8,
                                "source_color": "#9b9b9b",
                                "source_alpha": 0.58,
                                "source_size": 30,
                                "source_ellipse_alpha": 0.0,
                                "endpoint_color": "#5f5f5f",
                                "endpoint_alpha": 0.88,
                                "endpoint_size": 40,
                                "endpoint_ellipse_alpha": 0.05,
                            }
                        ]
                        real_traj_path = (
                            f"{mel_base}_traj_asr_realizations_{demo_trajectory_projection}_{traj_pool_tag}"
                            f"_spk{len(real_spks)}_u{n_real_u}.png"
                        )
                        saved_real_traj = save_demo_trajectory_plot(
                            real_traj_path,
                            real_items,
                            frame_count=demo_trajectory_frames,
                            projection=demo_trajectory_projection,
                            dims=demo_trajectory_dims,
                            pool=demo_trajectory_pool,
                            title=f"{banner_prefix} {step_tag}: ASR multi-speaker/style to canonical",
                            plt=plt,
                            paper_style=demo_trajectory_paper_style,
                            display_x_scale=demo_trajectory_display_x_scale,
                            display_y_scale=demo_trajectory_display_y_scale,
                            canonical_color=demo_trajectory_canonical_color,
                            speaker_colors=demo_trajectory_speaker_colors,
                            annotate_points=False,
                            export_csv=demo_trajectory_export_csv,
                        )
                        if saved_real_traj:
                            print(
                                f"[TRAJ-ASR-REAL {banner_prefix}] wrote {saved_real_traj} "
                                f"spks={real_spks} styles_per_spk={n_real_u}"
                            )
                if demo_trajectory_zu_fanout and use_tts_style_latent:
                    n_zc = max(1, int(demo_trajectory_zu_zc_samples))
                    n_u = max(1, int(demo_trajectory_zu_u_samples))
                    zT_mean_zu = zT_mean.expand(n_zc, -1, -1).contiguous()
                    zT_logvar_zu = zT_logvar.expand(n_zc, -1, -1).contiguous()
                    if n_zc > 1 and demo_prior_temp > 0:
                        eps_zu = torch.randn_like(zT_mean_zu)
                        zT0_zu = zT_mean_zu + demo_prior_temp * torch.exp(0.5 * zT_logvar_zu) * eps_zu
                        zT0_zu[:1] = zT0
                    elif n_zc > 1:
                        zT0_zu = zT_mean_zu.clone()
                    else:
                        zT0_zu = zT0
                    zT0_zu = zT0_zu.repeat_interleave(n_u, dim=0)
                    zT_mean_zu = zT_mean_zu.repeat_interleave(n_u, dim=0)
                    total_zu = zT0_zu.shape[0]
                    maskK_zu = maskK.expand(total_zu, -1).contiguous()
                    spk_e_zu = spk_e.to(device=device, dtype=zT0_zu.dtype).expand(total_zu, -1).contiguous()

                    u_mu_zu, u_logvar_zu = tts_style_prior_dist(
                        spk_e_zu,
                        zc=zT0_zu,
                        maskK=maskK_zu,
                        dtype=zT0_zu.dtype,
                    )
                    if demo_style_temp > 0.0:
                        style_e_zu = u_mu_zu + float(demo_style_temp) * torch.exp(0.5 * u_logvar_zu) * torch.randn_like(u_mu_zu)
                    else:
                        style_e_zu = u_mu_zu

                    text_cond_zu = None
                    if canonical_to_source is not None:
                        zT0_zu_source = canonical_to_source(
                            zT0_zu,
                            maskK_zu,
                            spk_e=spk_e_zu,
                            style_e=style_e_zu.to(dtype=zT0_zu.dtype) if tts_style_into_source else None,
                        )
                        zT_mean_zu_source = canonical_to_source(
                            zT_mean_zu,
                            maskK_zu,
                            spk_e=spk_e_zu.to(dtype=zT_mean_zu.dtype),
                            style_e=style_e_zu.to(dtype=zT_mean_zu.dtype) if tts_style_into_source else None,
                        )
                        if use_vf_canonical_text_cond:
                            text_cond_zu = zT_mean_zu
                    else:
                        zT0_zu_source = zT0_zu
                        zT_mean_zu_source = zT_mean_zu
                        if tts_style_to_source is not None:
                            style_bias_zu = tts_style_to_source(style_e_zu, spk_e_zu).to(dtype=zT0_zu.dtype)
                            zT0_zu_source = zT0_zu_source + float(tts_style_source_scale) * style_bias_zu.unsqueeze(1)
                            zT_mean_zu_source = zT_mean_zu_source + float(tts_style_source_scale) * style_bias_zu.unsqueeze(1)
                        if use_vf_canonical_text_cond:
                            text_cond_zu = zT_mean_zu

                    spk_e_zu_demo = spk_e_zu.to(dtype=zT0_zu_source.dtype)
                    if tts_source_cond is not None and demo_prior_temp > 0 and canonical_to_source is None:
                        source_delta_zu = zT0_zu_source - zT_mean_zu_source
                        spk_e_zu_demo = spk_e_zu_demo + tts_source_cond_scale * tts_source_cond(
                            source_delta_zu,
                            maskK_zu,
                        ).to(dtype=spk_e_zu_demo.dtype)

                    _, zu_trace = heun_integrate_trace(
                        vf,
                        zT0_zu_source,
                        maskK_zu,
                        steps=ode_steps_eval,
                        direction=+1,
                        cfg_scale=demo_cfg_scale,
                        spk_e=spk_e_zu_demo,
                        style_e=style_e_zu,
                        text_cond=text_cond_zu,
                    )
                    zu_color = demo_trajectory_speaker_colors[0]
                    zu_items = [
                        {
                            "spk": spk_pick,
                            "label": "Speaker A",
                            "trace": zu_trace,
                            "reverse_trace": None,
                            "color": zu_color,
                        }
                    ]
                    zu_traj_path = (
                        f"{mel_base}_traj_zu_fanout_{demo_trajectory_projection}_{traj_pool_tag}"
                        f"_zc{n_zc}_u{n_u}.png"
                    )
                    saved_zu_traj = save_demo_trajectory_plot(
                        zu_traj_path,
                        zu_items,
                        frame_count=demo_trajectory_frames,
                        projection=demo_trajectory_projection,
                        dims=demo_trajectory_dims,
                        pool=demo_trajectory_pool,
                        title=f"{banner_prefix} {step_tag}: z_u fan-out",
                        plt=plt,
                        paper_style=demo_trajectory_paper_style,
                        display_x_scale=demo_trajectory_display_x_scale,
                        display_y_scale=demo_trajectory_display_y_scale,
                        canonical_color=demo_trajectory_canonical_color,
                        speaker_colors=demo_trajectory_speaker_colors,
                        annotate_points=demo_trajectory_annotate_points,
                        export_csv=demo_trajectory_export_csv,
                    )
                    if saved_zu_traj:
                        print(f"[TRAJ-ZU {banner_prefix}] wrote {saved_zu_traj}")
            except Exception as exc:
                print(f"[TRAJ][WARN] failed to plot demo trajectory: {repr(exc)}")
        if speech_backend == "svae":
            if svae_model is None:
                raise RuntimeError("Semantic-VAE decoder is not loaded; set runtime.load_bigvgan_model=true for demos")
            wav = svae_model.decode(mel_for_vocoder).squeeze(0).float().detach().cpu()
        else:
            wav = bigvgan_model(mel_for_vocoder).squeeze(0).float().detach().cpu()
        tts_elapsed = None
        if demo_rtf and tts_t0 is not None:
            maybe_cuda_sync()
            tts_elapsed = time.perf_counter() - tts_t0
        wav_out_path = f"{mel_base}.wav"
        save_wav(wav_out_path, wav, sr=sampling_rate)
        if tts_elapsed is not None:
            print(demo_rtf_line(banner_prefix, tts_elapsed, demo_audio_sec_from_wav(wav)))
        spk_sim = compute_spk_similarity(wav, ref_wav_path)
        if spk_sim is not None:
            ref_label = ref_spk if ref_spk is not None else spk_pick
            print(f"[SPK-SIM {banner_prefix}] synth_spk={spk_pick} ref_spk={ref_label} cosine={spk_sim:.4f} ref={ref_wav_path}")

        tts_eval_info = {
            "step": step_tag,
            "speaker": spk_pick,
            "reference_speaker": ref_spk if ref_spk is not None else spk_pick,
            "text": text,
            "wav_path": wav_out_path,
            "sampling_rate": int(sampling_rate),
            "seconds": float(wav.numel()) / float(sampling_rate),
            "spk_similarity": spk_sim,
        }
        utmos_score = score_demo_utmos(wav)
        tts_eval_info["utmos"] = utmos_score
        whisper_hyp, whisper_backend = transcribe_demo_whisper(wav_out_path)
        whisper_wer = None
        if whisper_hyp is not None:
            whisper_wer = word_error_rate_text(canonicalize_text(text), canonicalize_text(whisper_hyp))
        tts_eval_info["whisper_model"] = demo_whisper_model if demo_eval_whisper else None
        tts_eval_info["whisper_backend"] = whisper_backend
        tts_eval_info["whisper_hyp"] = whisper_hyp
        tts_eval_info["whisper_wer"] = whisper_wer
        if utmos_score is not None or whisper_wer is not None:
            utmos_s = "NA" if utmos_score is None else f"{utmos_score:.4f}"
            wer_s = "NA" if whisper_wer is None else f"{whisper_wer:.4f}"
            print(
                f"[TTS-DEMO-EVAL {banner_prefix}] UTMOS={utmos_s} "
                f"WhisperWER={wer_s} backend={whisper_backend} hyp={whisper_hyp}"
            )
        with open(f"{mel_base}_eval.json", "w", encoding="utf-8") as f:
            json.dump(tts_eval_info, f, indent=2, ensure_ascii=False)

        if demo_plot_generated_wav_mel:
            mel_img, mel_frontend_used = compute_generated_wav_mel_image(wav)
            print(f"[DEMO-MEL {banner_prefix}] wrote generated-wav mel frontend={mel_frontend_used}")
        else:
            mel_img = mel_log_clamped[0].detach().cpu().numpy().T
            mel_frontend_used = "model_output"
        plt.figure(figsize=(12, 4))
        mel_plot = np.asarray(mel_img, dtype=np.float32)
        finite_mel = mel_plot[np.isfinite(mel_plot)]
        imshow_kwargs = {}
        if finite_mel.size > 0:
            top_pct = min(100.0, max(50.0, float(demo_mel_plot_top_percentile)))
            vmax = float(np.percentile(finite_mel, top_pct))
            dyn = max(1e-6, float(demo_mel_plot_dynamic_range_db))
            vmin = max(float(np.min(finite_mel)), vmax - dyn)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                imshow_kwargs.update(vmin=vmin, vmax=vmax)
        plt.imshow(mel_plot, origin="lower", aspect="auto", cmap=demo_mel_plot_cmap, **imshow_kwargs)
        plt.title(mel_frontend_used)
        plt.tight_layout()
        plt.savefig(f"{mel_base}_mel.png")
        plt.close()

        asr_t0 = None
        if demo_rtf:
            maybe_cuda_sync()
            asr_t0 = time.perf_counter()
        spk_e_asr_tts = asr_spk_cond_from_name(spk_pick, dtype=zS_ref.dtype)
        style_e_asr_tts = asr_style_cond_from_source(zS_ref, maskK, spk_e_asr_tts, dtype=zS_ref.dtype)
        zS_back_source = heun_integrate(
            vf,
            zS_ref,
            maskK,
            steps=ode_steps_eval,
            direction=-1,
            cfg_scale=1.0,
            spk_e=spk_e_asr_tts,
            style_e=style_e_asr_tts,
        )
        zS_back = source_to_canonical(zS_back_source, maskK) if source_to_canonical is not None else zS_back_source
        if canonical_posterior is not None:
            zS_back, _ = canonical_posterior(zS_back, maskK)
        zS_back_ctc, mask_back_ctc, k_back_ctc = prepare_ctc_input(
            zS_back,
            maskK,
            apply_subsample=(ctc_subsample_apply_to in {"hat", "both"}),
        )
        logits_back = text_ctc_head(zS_back_ctc, mask_back_ctc)
        hyp_back = decode_ctc_demo_text(logits_back, k_back_ctc[0])
        asr_elapsed = None
        if demo_rtf and asr_t0 is not None:
            maybe_cuda_sync()
            asr_elapsed = time.perf_counter() - asr_t0
        print(f"  ASR-on-TTS HYP: {hyp_back}")
        print(f"  ASR-on-TTS WER(norm): {word_error_rate_text(text, hyp_back):.4f}")
        if asr_elapsed is not None:
            print(demo_rtf_line("ASR-on-TTS", asr_elapsed, demo_audio_sec_from_wav(wav)))

    @torch.no_grad()
    def tts_demo_full_one_sample(step_tag: str, idx: int):
        item = tts_demo_rows[idx]
        wav_path = item["wav"]
        text = _row_text(item)
        text_spk = _row_speaker(item)
        spk_pick = text_spk
        if spk_pick not in spk2id:
            if tts_demo_text_source == "demo_manifest":
                spk_pick = random.choice(spk_list)
            else:
                raise RuntimeError(f"TTS demo row speaker not in closed-set speaker table: {spk_pick}")
        ref_wav_path = wav_path
        ref_spk = text_spk
        if ref_spk != spk_pick:
            ref_wav_path = train_ref_wav_by_spk.get(spk_pick)
            ref_spk = spk_pick
        print(
            f"\n[TTS-FULL-TEXT @ {step_tag}] "
            f"(source={tts_demo_text_source} txt_spk={text_spk} synth_spk={spk_pick})\nTXT: {text}"
        )
        tts_demo_from_text(
            step_tag,
            text,
            spk_pick=spk_pick,
            file_prefix="tts_full_demo",
            banner_prefix="TTS-FULL",
            ref_wav_path=ref_wav_path,
            ref_spk=ref_spk,
        )

    def maybe_cuda_sync():
        if device == "cuda":
            torch.cuda.synchronize()

    last_step = start_step - 1
    for step in range(start_step, total_steps):
        last_step = step
        lr_now = set_optimizer_lr(step)
        perf_due = perf_log_every and (step % perf_log_every == 0)
        perf_step = {
            "data": 0.0,
            "text": 0.0,
            "teacher": 0.0,
            "forward": 0.0,
            "vf_lip": 0.0,
            "stft": 0.0,
            "backward_opt": 0.0,
            "debug": 0.0,
            "log": 0.0,
            "diag": 0.0,
            "demo": 0.0,
            "save": 0.0,
            "total": 0.0,
        }
        if perf_due:
            maybe_cuda_sync()
            perf_step_t0 = time.perf_counter()
        if use_dataloader_runtime:
            tts_batch, tts_iter = next_loader_batch(tts_loader, tts_iter)
            asr_batch, asr_iter = next_loader_batch(asr_loader, asr_iter)
            (
                zS_tts_log, maskK_tts, loss_maskK_tts, K_list_tts,
                texts_tts, spk_ids_tts,
                wav_paths_tts, ref_wav_paths_tts, starts_tts, ends_tts,
                row_metas_tts,
            ) = move_cut_batch_to_device(tts_batch)
            (
                zS_asr_log, maskK_asr, loss_maskK_asr, K_list_asr,
                texts_asr, spk_ids_asr,
                wav_paths_asr, ref_wav_paths_asr, starts_asr, ends_asr,
                row_metas_asr,
            ) = move_cut_batch_to_device(asr_batch)
        else:
            tts_rows = tts_sampler.sample()
            asr_rows = asr_sampler.sample()
            (
                zS_tts_log, maskK_tts, loss_maskK_tts, K_list_tts,
                texts_tts, spk_ids_tts,
                wav_paths_tts, ref_wav_paths_tts, starts_tts, ends_tts,
                row_metas_tts,
            ) = build_batch_from_cut_rows(tts_rows)
            (
                zS_asr_log, maskK_asr, loss_maskK_asr, K_list_asr,
                texts_asr, spk_ids_asr,
                wav_paths_asr, ref_wav_paths_asr, starts_asr, ends_asr,
                row_metas_asr,
            ) = build_batch_from_cut_rows(asr_rows)
        if perf_due:
            maybe_cuda_sync()
            perf_step["data"] = time.perf_counter() - perf_step_t0
            perf_stage_t0 = time.perf_counter()

        full_asr_aux_active = (
            enable_full_asr_ctc_aux
            and full_asr_ctc_aux_every > 0
            and step >= full_asr_ctc_aux_start
            and (step % full_asr_ctc_aux_every == 0)
        )
        zS_full_aux_log = None
        maskK_full_aux = None
        K_list_full_aux = None
        texts_full_aux = None
        spk_ids_full_aux = None
        wav_paths_full_aux = None
        if full_asr_aux_active:
            full_source_rows = asr_full_rows if (use_processed_unified and len(asr_full_rows) > 0) else aligned_rows
            full_rows = sample_cut_rows(full_source_rows, max(1, full_asr_ctc_aux_batch_size))
            (
                zS_full_aux_log,
                maskK_full_aux,
                K_list_full_aux,
                texts_full_aux,
                spk_ids_full_aux,
                wav_paths_full_aux,
            ) = build_batch_from_aligned_rows(full_rows)
        if perf_due:
            maybe_cuda_sync()
            perf_step["data"] += time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        B_tts = zS_tts_log.shape[0]
        B_asr = zS_asr_log.shape[0]
        train_maskK_tts = loss_maskK_tts if core_loss_only else maskK_tts
        train_maskK_asr = loss_maskK_asr if core_loss_only else maskK_asr
        sumK_tts = int(maskK_tts.sum().item())
        sumK_asr = int(maskK_asr.sum().item())
        Kmax_tts = int(maskK_tts.shape[1])
        Kmax_asr = int(maskK_asr.shape[1])
        paddedK_tts = int(B_tts * Kmax_tts)
        paddedK_asr = int(B_asr * Kmax_asr)

        opt.zero_grad(set_to_none=True)
        if source_ctc_opt is not None:
            source_ctc_opt.zero_grad(set_to_none=True)

        zt_for_lip = None
        t_for_lip = None
        mask_for_lip = None
        cfg_for_lip = None
        spk_for_lip = None
        style_for_lip = None
        loss_vf_lip = torch.tensor(0.0, device=device)
        loss_stft = torch.tensor(0.0, device=device)
        loss_tts_style_kl = torch.tensor(0.0, device=device)
        loss_tts_style_kl_tts = torch.tensor(0.0, device=device)
        loss_tts_style_kl_asr = torch.tensor(0.0, device=device)
        loss_canonical_nll = torch.tensor(0.0, device=device)
        loss_canonical_prior_nll = torch.tensor(0.0, device=device)
        loss_canonical_bwd_nll = torch.tensor(0.0, device=device)
        loss_ssl_hidden = torch.tensor(0.0, device=device)
        loss_ctc_source = torch.tensor(0.0, device=device)
        loss_ctc_sample = torch.tensor(0.0, device=device)
        loss_bwd_fm = torch.tensor(0.0, device=device)
        loss_end_fwd = torch.tensor(0.0, device=device)
        loss_end_bwd = torch.tensor(0.0, device=device)
        loss_tts_mel = torch.tensor(0.0, device=device)
        loss_mel_high = torch.tensor(0.0, device=device)
        loss_delta = torch.tensor(0.0, device=device)
        loss_range = torch.tensor(0.0, device=device)
        loss_ref = torch.tensor(0.0, device=device)
        loss_prior_mu = torch.tensor(0.0, device=device)
        loss_prior_var = torch.tensor(0.0, device=device)
        loss_prior_nll = torch.tensor(0.0, device=device)
        loss_prior = torch.tensor(0.0, device=device)
        loss_ctc_T = torch.tensor(0.0, device=device)
        loss_ctc_hat = torch.tensor(0.0, device=device)
        loss_ctc_dur = torch.tensor(0.0, device=device)
        loss_ctc_full = torch.tensor(0.0, device=device)
        loss_dit_hidden_ctc = torch.tensor(0.0, device=device)
        loss_att_decoder = torch.tensor(0.0, device=device)
        canonical_nll_w_now = 0.0
        canonical_prior_nll_w_now = 0.0
        canonical_bwd_nll_w_now = 0.0
        ssl_hidden_w_now = 0.0
        source_ctc_w_now = 0.0
        sample_ctc_w_now = 0.0
        bwd_fm_w_now = 0.0
        dit_hidden_ctc_w_now = 0.0
        att_decoder_w_now = 0.0
        att_decoder_token_acc = 0.0
        source_ctc_active = False
        sample_ctc_active = False
        bwd_fm_active = False
        dit_hidden_ctc_active = False
        att_decoder_active = False
        ctc_hat_active = False
        bwd_rollout_active = False
        zT_hat_asr = None
        zT_hat_logvar_asr = None
        logits_hat = None
        logits_dit_hidden = None
        logits_att_decoder = None
        K_list_H_ctc = []
        K_list_dit_hidden_ctc = []

        with torch.amp.autocast(amp_device, enabled=use_amp):
            texts_joint = texts_tts + texts_asr
            h_enc_joint, maskL_joint, mu_tok_joint, logvar_tok_joint, align_mu_tok_joint, align_logvar_tok_joint = encode_text_batch(texts_joint)
            h_enc_tts, h_enc_asr = h_enc_joint[:B_tts], h_enc_joint[B_tts:]
            maskL_tts, maskL_asr = maskL_joint[:B_tts], maskL_joint[B_tts:]
            mu_tok_tts, mu_tok_asr = mu_tok_joint[:B_tts], mu_tok_joint[B_tts:]
            logvar_tok_tts, logvar_tok_asr = logvar_tok_joint[:B_tts], logvar_tok_joint[B_tts:]
            align_mu_tok_tts, align_mu_tok_asr = align_mu_tok_joint[:B_tts], align_mu_tok_joint[B_tts:]
            align_logvar_tok_tts, align_logvar_tok_asr = align_logvar_tok_joint[:B_tts], align_logvar_tok_joint[B_tts:]
            gt_dur_tts, gt_dur_mask_tts = build_gt_token_duration_batch(
                texts_tts,
                maskL_tts,
                K_list_tts,
                row_metas_tts,
                starts_tts,
                ends_tts,
            )
            gt_dur_asr, gt_dur_mask_asr = build_gt_token_duration_batch(
                texts_asr,
                maskL_asr,
                K_list_asr,
                row_metas_asr,
                starts_asr,
                ends_asr,
            )
            if perf_due:
                maybe_cuda_sync()
                perf_step["text"] = time.perf_counter() - perf_stage_t0
                perf_stage_t0 = time.perf_counter()
            loss_dur_tts = torch.tensor(0.0, device=device)
            loss_len_tts = torch.tensor(0.0, device=device)
            full_tts_aux = None
            full_tts_teacher_active = use_full_tts_teacher and (step % full_tts_teacher_every == 0)
            if full_tts_teacher_active:
                full_tts_aux = build_full_tts_duration_teacher_batch(wav_paths_tts)
            if perf_due:
                maybe_cuda_sync()
                perf_step["teacher"] = time.perf_counter() - perf_stage_t0
                perf_stage_t0 = time.perf_counter()

            (
                zS_tts, attn_tts, mu_tts, logvar_tts,
                zT_tts_sample, zT_tts_mean,
                source_delta_tts,
                _loss_dur_tts_cut,
                _loss_len_tts_cut,
                align_mu_tts,
                align_logvar_tts,
            ) = build_local_prior_batch(
                zS_tts_log,
                maskK_tts,
                h_enc_tts,
                maskL_tts,
                mu_tok_tts,
                logvar_tok_tts,
                align_mu_tok=align_mu_tok_tts,
                align_logvar_tok=align_logvar_tok_tts,
                supervise_dur_len=(not use_full_tts_teacher),
                dur_teacher_full=gt_dur_tts if (use_gt_duration_teacher or use_ctc_blank_repeat_prior) else None,
                dur_teacher_full_mask=gt_dur_mask_tts if (use_gt_duration_teacher or use_ctc_blank_repeat_prior) else None,
                gt_token_durations=gt_dur_tts if (use_gt_alignment_prior or use_ctc_blank_repeat_prior) else None,
                gt_token_duration_mask=gt_dur_mask_tts if (use_gt_alignment_prior or use_ctc_blank_repeat_prior) else None,
            )
            (
                zS_asr, attn_asr, mu_asr, logvar_asr,
                zT_asr_sample, zT_asr_mean,
                _,
                _,
                _,
                _,
                _,
            ) = build_local_prior_batch(
                zS_asr_log,
                maskK_asr,
                h_enc_asr,
                maskL_asr,
                mu_tok_asr,
                logvar_tok_asr,
                align_mu_tok=align_mu_tok_asr,
                align_logvar_tok=align_logvar_tok_asr,
                supervise_dur_len=False,
                gt_token_durations=gt_dur_asr if (use_gt_alignment_prior or use_ctc_blank_repeat_prior) else None,
                gt_token_duration_mask=gt_dur_mask_asr if (use_gt_alignment_prior or use_ctc_blank_repeat_prior) else None,
            )
            if use_full_tts_teacher and full_tts_aux is not None:
                h_enc_tts_full, maskL_tts_full, maskK_tts_full, attn_tts_full = full_tts_aux
                loss_dur_tts, loss_len_tts = compute_duration_losses(
                    h_enc_tts_full,
                    maskL_tts_full,
                    attn_tts_full,
                    maskK_tts_full,
                )
            else:
                loss_dur_tts = _loss_dur_tts_cut
                loss_len_tts = _loss_len_tts_cut

            spk_e_tts_match = speaker_cond_from_ref_paths(
                ref_wav_paths_tts,
                spk_ids_tts,
                dtype=zS_tts.dtype,
            )
            asr_ref_paths = wav_paths_asr if zero_shot_asr_ref_source == "self" else ref_wav_paths_asr
            spk_e_asr_match = asr_spk_cond_from_ref_paths(
                asr_ref_paths,
                spk_ids_asr,
                dtype=zS_asr.dtype,
            )
            asr_cfg_value = asr_cfg_flag_value(spk_e_asr_match)

            drop_mask = (torch.rand(B_tts, device=device) < spk_drop_rate)
            if B_tts < 2:
                drop_mask[:] = False
            if not vf_use_speaker_cond:
                # In the z_u-only VF variant, speaker/reference conditioning is
                # represented by z_u. Do not turn z_u into an unconditional CFG
                # drop just because direct speaker conditioning is disabled.
                drop_mask[:] = False

            cfg_flag_fm = torch.where(
                drop_mask,
                torch.zeros(B_tts, dtype=torch.long, device=device),
                torch.ones(B_tts, dtype=torch.long, device=device),
            )
            # Speaker-free CFG should remove only the speaker/style condition.
            # The speech endpoint must stay paired with the same text/source;
            # using a permuted target makes the unconditional branch learn an
            # averaged/random acoustic endpoint and pollutes the shared VF.
            zS_fm_tgt = zS_tts
            spk_e_fm = torch.where(drop_mask[:, None], torch.zeros_like(spk_e_tts_match), spk_e_tts_match)
            if tts_source_cond is not None:
                if canonical_to_source is not None:
                    src_cond_e = torch.zeros_like(spk_e_fm)
                else:
                    src_cond_e = tts_source_cond(source_delta_tts, maskK_tts).to(dtype=spk_e_fm.dtype)
                src_cond_e = torch.where(drop_mask[:, None], torch.zeros_like(src_cond_e), src_cond_e)
                spk_e_fm = spk_e_fm + tts_source_cond_scale * src_cond_e
            spk_e_fm = tts_vf_spk_cond(spk_e_fm)

            tts_style_kl_tts_w_now = 0.0
            tts_style_kl_asr_w_now = 0.0
            u_fm = None
            u_fwd = None
            if tts_style_pair_post is not None:
                u_mu_pair, u_logvar_pair = tts_style_pair_post(
                    zS_tts.detach(),
                    maskK_tts,
                    zT_tts_sample.detach(),
                    maskK_tts,
                    spk_e_tts_match.detach().to(dtype=zS_tts.dtype),
                )
                u_mu_p, u_logvar_p = tts_style_prior_dist(
                    spk_e_tts_match,
                    zc=zT_tts_sample,
                    maskK=maskK_tts,
                    dtype=u_mu_pair.dtype,
                )
                u_mu_asr, u_logvar_asr = tts_style_post(zS_tts, maskK_tts)
                eps_u = torch.randn_like(u_mu_pair)
                u = u_mu_pair + torch.exp(0.5 * u_logvar_pair) * eps_u
                u_fwd = u
                u_fm = torch.where(drop_mask[:, None], torch.zeros_like(u), u)
                loss_tts_style_kl_tts = diag_gaussian_kl(u_mu_pair, u_logvar_pair, u_mu_p, u_logvar_p)
                u_mu_pair_asr_kl = u_mu_pair.detach() if tts_style_asr_kl_stopgrad_pair else u_mu_pair
                u_logvar_pair_asr_kl = (
                    u_logvar_pair.detach() if tts_style_asr_kl_stopgrad_pair else u_logvar_pair
                )
                loss_tts_style_kl_asr = diag_gaussian_kl(
                    u_mu_pair_asr_kl,
                    u_logvar_pair_asr_kl,
                    u_mu_asr,
                    u_logvar_asr,
                )
                loss_tts_style_kl = loss_tts_style_kl_tts + loss_tts_style_kl_asr
                tts_style_kl_tts_w_now = float(w_tts_style_kl) * linear_anneal(
                    step,
                    tts_style_kl_start,
                    tts_style_kl_anneal_steps,
                )
                tts_style_kl_asr_w_now = float(w_tts_style_asr_kl) * linear_anneal(
                    step,
                    tts_style_asr_kl_start,
                    tts_style_asr_kl_anneal_steps,
                )
            elif tts_style_post is not None and tts_style_post_mode == "speech":
                u_mu_q, u_logvar_q = tts_style_post(zS_tts, maskK_tts)
                u_mu_p, u_logvar_p = tts_style_prior_dist(
                    spk_e_tts_match,
                    zc=zT_tts_sample,
                    maskK=maskK_tts,
                    dtype=u_mu_q.dtype,
                )
                eps_u = torch.randn_like(u_mu_q)
                u = u_mu_q + torch.exp(0.5 * u_logvar_q) * eps_u
                u_fwd = u
                u_fm = torch.where(drop_mask[:, None], torch.zeros_like(u), u)
                loss_tts_style_kl_tts = diag_gaussian_kl(u_mu_q, u_logvar_q, u_mu_p, u_logvar_p)
                loss_tts_style_kl = loss_tts_style_kl_tts
                tts_style_kl_tts_w_now = float(w_tts_style_kl) * linear_anneal(
                    step,
                    tts_style_kl_start,
                    tts_style_kl_anneal_steps,
                )

            style_e_fm = u_fm.to(dtype=zS_tts.dtype) if u_fm is not None else None
            vf_text_cond_fm = None
            if canonical_to_source is not None:
                zT_tts_mean_source = canonical_to_source(
                    zT_tts_mean,
                    maskK_tts,
                    spk_e=spk_e_fm.to(dtype=zT_tts_mean.dtype),
                    style_e=(
                        style_e_fm.to(dtype=zT_tts_mean.dtype)
                        if (tts_style_into_source and style_e_fm is not None)
                        else None
                    ),
                )
                zT_tts_sample_for_fm = canonical_to_source(
                    zT_tts_sample,
                    maskK_tts,
                    spk_e=spk_e_fm.to(dtype=zT_tts_sample.dtype),
                    style_e=(
                        style_e_fm.to(dtype=zT_tts_sample.dtype)
                        if (tts_style_into_source and style_e_fm is not None)
                        else None
                    ),
                )
                if use_vf_canonical_text_cond:
                    vf_text_cond_fm = torch.where(
                        drop_mask[:, None, None],
                        torch.zeros_like(zT_tts_mean),
                        zT_tts_mean,
                    )
            else:
                zT_tts_mean_source = zT_tts_mean
                zT_tts_sample_for_fm = zT_tts_sample
                if tts_style_to_source is not None and u_fm is not None:
                    style_bias = tts_style_to_source(u_fm, spk_e_tts_match).to(dtype=zT_tts_sample.dtype)
                    zT_tts_mean_source = zT_tts_mean + float(tts_style_source_scale) * style_bias.unsqueeze(1)
                    zT_tts_sample_for_fm = zT_tts_sample + float(tts_style_source_scale) * style_bias.unsqueeze(1)
                if use_vf_canonical_text_cond:
                    vf_text_cond_fm = zT_tts_mean

            t = torch.rand(B_tts, device=device)
            t_ = t.view(B_tts, 1, 1)
            zt = (1 - t_) * zT_tts_sample_for_fm + t_ * zS_fm_tgt
            U = zS_fm_tgt - zT_tts_sample_for_fm

            if tts_style_post is not None and tts_style_post_mode == "path":
                u_mu_q, u_logvar_q = tts_style_post(
                    zS_tts.detach(),
                    maskK_tts,
                    z_t=zt.detach(),
                    t=t.detach(),
                    spk_e=spk_e_tts_match.detach(),
                )
                u_mu_p, u_logvar_p = tts_style_prior_dist(
                    spk_e_tts_match,
                    zc=zT_tts_sample,
                    maskK=maskK_tts,
                    dtype=u_mu_q.dtype,
                )
                eps_u = torch.randn_like(u_mu_q)
                u = u_mu_q + torch.exp(0.5 * u_logvar_q) * eps_u
                u_fwd = u
                u_fm = torch.where(drop_mask[:, None], torch.zeros_like(u), u)
                loss_tts_style_kl_tts = diag_gaussian_kl(u_mu_q, u_logvar_q, u_mu_p, u_logvar_p)
                loss_tts_style_kl = loss_tts_style_kl_tts
                tts_style_kl_tts_w_now = float(w_tts_style_kl) * linear_anneal(
                    step,
                    tts_style_kl_start,
                    tts_style_kl_anneal_steps,
                )
                style_e_fm = u_fm.to(dtype=zS_tts.dtype)

            ssl_hidden_active = bool(ssl_hidden_enable) and ssl_hidden_head is not None and (step >= ssl_hidden_start)
            ssl_hidden_fm_active = ssl_hidden_active and ssl_hidden_target == "hidden"
            ssl_hidden_zc_active = ssl_hidden_active and ssl_hidden_target == "zc" and ssl_hidden_w > 0.0
            w_ctc = 0.0 if step < w_ctc_start else 1.0
            w_ctc_dur_on = 0.0 if step < ctc_dur_start else 1.0
            w_align = 0.0 if step < w_align_start else 1.0
            ctc_T_active = (w_ctc > 0.0) and (w_ctc_T > 0.0)
            ctc_hat_active = (w_ctc > 0.0) and (w_ctc_hat > 0.0)
            att_decoder_w_now = float(w_att_decoder) * linear_anneal(
                step,
                att_decoder_start,
                att_decoder_anneal_steps,
            )
            att_decoder_active = att_decoder is not None and att_decoder_w_now > 0.0
            canonical_anneal_now = 0.0
            canonical_prior_active = False
            canonical_bwd_active = False
            canonical_joint_active = False
            if enable_canonical_nll:
                canonical_anneal_now = linear_anneal(
                    step,
                    canonical_nll_start,
                    canonical_nll_anneal_steps,
                )
                if canonical_stopgrad_split:
                    canonical_prior_nll_w_now = float(w_canonical_prior_nll) * canonical_anneal_now
                    canonical_bwd_nll_w_now = float(w_canonical_bwd_nll) * canonical_anneal_now
                    canonical_prior_active = canonical_prior_nll_w_now > 0.0
                    canonical_bwd_active = canonical_bwd_nll_w_now > 0.0
                    canonical_nll_w_now = canonical_prior_nll_w_now + canonical_bwd_nll_w_now
                else:
                    canonical_nll_w_now = float(w_canonical_nll) * canonical_anneal_now
                    canonical_joint_active = canonical_nll_w_now > 0.0
            canonical_loss_enabled_now = canonical_prior_active or canonical_bwd_active or canonical_joint_active
            bwd_rollout_active = (
                (w_end_bwd > 0.0)
                or ctc_hat_active
                or att_decoder_active
                or canonical_loss_enabled_now
                or ssl_hidden_zc_active
            )
            h_fwd_ssl = None
            if ssl_hidden_fm_active:
                U_hat, h_fwd_ssl = vf(
                    zt,
                    t,
                    maskK_tts,
                    cfg_flag=cfg_flag_fm,
                    spk_e=spk_e_fm.to(dtype=zt.dtype),
                    style_e=style_e_fm.to(dtype=zt.dtype) if style_e_fm is not None else None,
                    text_cond=vf_text_cond_fm.to(dtype=zt.dtype) if vf_text_cond_fm is not None else None,
                    return_hidden=True,
                    hidden_tap_index=ssl_hidden_tap_index,
                )
            else:
                U_hat = vf(
                    zt,
                    t,
                    maskK_tts,
                    cfg_flag=cfg_flag_fm,
                    spk_e=spk_e_fm.to(dtype=zt.dtype),
                    style_e=style_e_fm.to(dtype=zt.dtype) if style_e_fm is not None else None,
                    text_cond=vf_text_cond_fm.to(dtype=zt.dtype) if vf_text_cond_fm is not None else None,
            )
            loss_fm = masked_mse(U_hat, U, train_maskK_tts)

            bwd_fm_active = enable_bwd_fm and (w_bwd_fm > 0.0) and (step >= bwd_fm_start)
            if bwd_fm_active:
                if bwd_fm_anchor == "mean":
                    zT_bwd_c = zT_asr_mean
                elif bwd_fm_anchor == "sample":
                    zT_bwd_c = zT_asr_sample
                else:
                    mix_a = float(max(0.0, min(1.0, bwd_fm_anchor_mix_alpha)))
                    zT_bwd_c = mix_a * zT_asr_mean + (1.0 - mix_a) * zT_asr_sample

                with torch.no_grad():
                    zT_bwd_c_det = zT_bwd_c.detach()
                    zS_bwd_det = zS_asr.detach()
                    spk_e_bwd_match = (
                        spk_e_asr_match.detach().to(dtype=zT_bwd_c_det.dtype)
                        if spk_e_asr_match is not None
                        else None
                    )
                    style_e_bwd_match = asr_style_cond_from_source(
                        zS_bwd_det,
                        maskK_asr,
                        spk_e_bwd_match,
                        dtype=zS_bwd_det.dtype,
                    )
                    spk_e_bwd_match = asr_vf_spk_cond(spk_e_bwd_match)
                    if canonical_to_source is not None:
                        zT_bwd_source = canonical_to_source(
                            zT_bwd_c_det,
                            maskK_asr,
                            spk_e=spk_e_bwd_match,
                            style_e=style_e_bwd_match if tts_style_into_source else None,
                        )
                    else:
                        zT_bwd_source = zT_bwd_c_det
                    U_bwd_fm = (zS_bwd_det - zT_bwd_source) * maskK_asr.float().unsqueeze(-1)

                t_bwd = bwd_fm_t_min + (bwd_fm_t_max - bwd_fm_t_min) * torch.rand(B_asr, device=device)
                t_bwd_ = t_bwd.view(B_asr, 1, 1)
                zt_bwd = (1.0 - t_bwd_) * zT_bwd_source + t_bwd_ * zS_bwd_det
                cfg_flag_bwd = torch.full((B_asr,), int(asr_cfg_value), dtype=torch.long, device=device)
                U_bwd_hat = vf(
                    zt_bwd,
                    t_bwd,
                    maskK_asr,
                    cfg_flag=cfg_flag_bwd,
                    spk_e=spk_e_bwd_match.to(dtype=zt_bwd.dtype) if spk_e_bwd_match is not None else None,
                    style_e=style_e_bwd_match.to(dtype=zt_bwd.dtype) if style_e_bwd_match is not None else None,
                    text_cond=None,
                )
                loss_bwd_fm = masked_mse(U_bwd_hat, U_bwd_fm, train_maskK_asr)
                bwd_fm_w_now = float(w_bwd_fm) * linear_anneal(
                    step,
                    bwd_fm_start,
                    bwd_fm_anneal_steps,
                )
            if ssl_hidden_fm_active:
                ssl_pred, ssl_pred_mask = ssl_hidden_head(h_fwd_ssl, maskK_tts)
                with torch.no_grad():
                    teacher_ssl, teacher_ssl_mask = build_hubert_online_targets(
                        wav_paths_tts,
                        starts_tts,
                        ends_tts,
                        target_sr=16000,
                    )
                loss_ssl_hidden = hidden_ssl_cosine_loss(ssl_pred, ssl_pred_mask, teacher_ssl, teacher_ssl_mask)
                ssl_hidden_w_now = float(ssl_hidden_w)

            zt_for_lip = zt.detach()
            t_for_lip = t.detach()
            mask_for_lip = maskK_tts
            cfg_for_lip = cfg_flag_fm
            spk_for_lip = spk_e_fm.detach()
            style_for_lip = style_e_fm.detach() if style_e_fm is not None else None

            loss_stat = torch.tensor(0.0, device=device)
            stat_match_active = use_stat_match and (w_stat > 0.0)
            tts_mel_active = w_tts_mel > 0.0
            mel_high_active = w_mel_high > 0.0
            delta_active = w_delta > 0.0
            mel_range_active = w_mel_range > 0.0
            ref_active = (mel_refiner is not None) and (w_ref > 0.0)
            prior_active = enable_acoustic_prior_nll and (w_prior > 0.0)
            fwd_rollout_active = enable_fwd_end_loss and (
                (w_end_fwd > 0.0)
                or tts_mel_active
                or mel_high_active
                or delta_active
                or mel_range_active
                or ref_active
            )
            if stat_match_active:
                muT, stdT = masked_mean_std(zT_tts_mean_source, train_maskK_tts)
                muS, stdS = masked_mean_std(zS_tts, train_maskK_tts)
                loss_stat = (muT - muS).abs().mean() + (stdT - stdS).abs().mean()

            zT_tts_fwd_c = zT_tts_mean
            maskK_tts_fwd = maskK_tts
            if fwd_rollout_active and fwd_prior_mode != "mas":
                zT_tts_dur_mean, maskK_tts_dur, _ = build_duration_expanded_prior_batch(
                    h_enc_tts,
                    maskL_tts,
                    mu_tok_tts,
                    K_targets=K_list_tts,
                )
                if fwd_prior_mode == "dur":
                    zT_tts_fwd_c = zT_tts_dur_mean
                    maskK_tts_fwd = maskK_tts_dur
                elif fwd_prior_mode == "mix":
                    mix_a = float(max(0.0, min(1.0, fwd_prior_mix_alpha)))
                    zT_tts_fwd_c = mix_a * zT_tts_mean + (1.0 - mix_a) * zT_tts_dur_mean
                    maskK_tts_fwd = maskK_tts_dur
                else:
                    raise ValueError(f"Unsupported fwd_prior_mode: {fwd_prior_mode}")
            if fwd_rollout_active and fwd_prior_mode == "mas":
                if fwd_anchor_mode == "mean":
                    zT_tts_fwd_c = zT_tts_mean
                elif fwd_anchor_mode == "sample":
                    zT_tts_fwd_c = zT_tts_sample
                elif fwd_anchor_mode == "mix":
                    mix_a = float(max(0.0, min(1.0, fwd_anchor_mix_alpha)))
                    zT_tts_fwd_c = mix_a * zT_tts_mean + (1.0 - mix_a) * zT_tts_sample
                else:
                    raise ValueError(f"Unsupported fwd_anchor_mode: {fwd_anchor_mode}")

            vf_text_cond_fwd = None
            if fwd_rollout_active and canonical_to_source is not None:
                style_e_fwd = u_fwd.to(dtype=zT_tts_fwd_c.dtype) if u_fwd is not None else None
                zT_tts_fwd = canonical_to_source(
                    zT_tts_fwd_c,
                    maskK_tts_fwd,
                    spk_e=tts_vf_spk_cond(spk_e_tts_match.to(dtype=zT_tts_fwd_c.dtype)),
                    style_e=style_e_fwd if tts_style_into_source else None,
                )
                if use_vf_canonical_text_cond:
                    vf_text_cond_fwd = zT_tts_fwd_c
            elif fwd_rollout_active:
                zT_tts_fwd = zT_tts_fwd_c
                if tts_style_to_source is not None and u_fwd is not None and fwd_anchor_mode != "mean":
                    style_bias_fwd = tts_style_to_source(u_fwd, spk_e_tts_match).to(dtype=zT_tts_fwd.dtype)
                    zT_tts_fwd = zT_tts_fwd + float(tts_style_source_scale) * style_bias_fwd.unsqueeze(1)
                if use_vf_canonical_text_cond:
                    vf_text_cond_fwd = zT_tts_fwd_c

            if fwd_rollout_active:
                fwd_integrate = euler_integrate_grad if fwd_ode_grad else euler_integrate
                if fwd_ode_grad:
                    zS_end_tts = fwd_integrate(
                        vf,
                        zT_tts_fwd,
                        maskK_tts_fwd,
                        steps=ode_steps_endloss,
                        direction=+1,
                        cfg_flag_value=1,
                        spk_e=tts_vf_spk_cond(spk_e_tts_match.to(dtype=zT_tts_fwd.dtype)),
                        style_e=u_fwd.to(dtype=zT_tts_fwd.dtype) if u_fwd is not None else None,
                        text_cond=vf_text_cond_fwd.to(dtype=zT_tts_fwd.dtype) if vf_text_cond_fwd is not None else None,
                    )
                else:
                    with torch.no_grad():
                        zS_end_tts = fwd_integrate(
                            vf,
                            zT_tts_fwd,
                            maskK_tts_fwd,
                            steps=ode_steps_endloss,
                            direction=+1,
                            cfg_flag_value=1,
                            spk_e=tts_vf_spk_cond(spk_e_tts_match.to(dtype=zT_tts_fwd.dtype)),
                            style_e=u_fwd.to(dtype=zT_tts_fwd.dtype) if u_fwd is not None else None,
                            text_cond=vf_text_cond_fwd.to(dtype=zT_tts_fwd.dtype) if vf_text_cond_fwd is not None else None,
                        )
                loss_end_fwd = masked_mse(zS_end_tts, zS_tts, train_maskK_tts)
                mel_end = (zS_end_tts * std_b + mu_b)
            else:
                mel_end = (zS_fm_tgt * std_b + mu_b)
            mel_gt = zS_tts_log
            if tts_mel_active:
                loss_tts_mel = masked_l1(mel_end, mel_gt, train_maskK_tts)
            if mel_high_active:
                high_start = int(max(0, min(D_mel - 1, mel_high_start_bin)))
                loss_mel_high = masked_l1(
                    mel_end[..., high_start:],
                    mel_gt[..., high_start:],
                    train_maskK_tts,
                )

            if delta_active:
                dt_pred = time_delta_bkd(mel_end)
                dt_gt = time_delta_bkd(mel_gt)
                mask_dt = train_maskK_tts[:, 1:] & train_maskK_tts[:, :-1]
                loss_delta = masked_l1(dt_pred, dt_gt, mask_dt)

            if mel_range_active:
                loss_range = mel_range_penalty(mel_end.float(), train_maskK_tts, mel_min=mel_floor, mel_max=mel_ceil)

            if ref_active and fwd_rollout_active:
                zS_ref = mel_refiner(zS_end_tts, cond=h_enc_tts, cond_mask=maskL_tts)
                mel_ref = (zS_ref * std_b + mu_b)
                loss_ref = masked_l1(mel_ref, mel_gt, maskK_tts)

            if not prior_active:
                pass
            elif canonical_to_source is not None:
                loss_prior_nll = masked_gaussian_nll(zS_tts, align_mu_tts, align_logvar_tts, train_maskK_tts)
                loss_prior = loss_prior_nll
            elif prior_loss_mode == "gaussian_nll":
                loss_prior_nll = masked_gaussian_nll(zS_tts, mu_tts, logvar_tts, train_maskK_tts)
                loss_prior = loss_prior_nll
            elif prior_loss_mode == "mu_var_reg":
                loss_prior_mu = masked_prior_mu_loss(mu_tts, zS_tts.detach(), train_maskK_tts)
                mask_prior = train_maskK_tts.float().unsqueeze(-1)
                denom_prior = mask_prior.sum().clamp_min(1.0) * logvar_tts.shape[-1]
                loss_prior_var = (((logvar_tts - prior_var_reg_target) ** 2) * mask_prior).sum() / denom_prior
                if w_prior_nll > 0.0:
                    loss_prior_nll = masked_gaussian_nll(zS_tts, mu_tts.detach(), logvar_tts, train_maskK_tts)
                loss_prior = (
                    w_prior_mu * loss_prior_mu
                    + w_prior_var * loss_prior_var
                    + w_prior_nll * loss_prior_nll
                )
            else:
                loss_prior_mu = masked_prior_mu_loss(mu_tts, zS_tts.detach(), train_maskK_tts)
                loss_prior = w_prior_mu * loss_prior_mu

            if bwd_rollout_active:
                bwd_integrate = euler_integrate_grad if bwd_ode_grad else euler_integrate
                style_e_asr_match = asr_style_cond_from_source(
                    zS_asr,
                    maskK_asr,
                    spk_e_asr_match,
                    dtype=zS_asr.dtype,
                )
                spk_e_asr_vf = asr_vf_spk_cond(
                    spk_e_asr_match.to(dtype=zS_asr.dtype) if spk_e_asr_match is not None else None
                )
                if bwd_ode_grad:
                    y0_end_asr = bwd_integrate(
                        vf,
                        zS_asr,
                        maskK_asr,
                        steps=ode_steps_endloss,
                        direction=-1,
                        cfg_flag_value=asr_cfg_value,
                        spk_e=spk_e_asr_vf,
                        style_e=style_e_asr_match,
                    )
                else:
                    with torch.no_grad():
                        y0_end_asr = bwd_integrate(
                            vf,
                            zS_asr,
                            maskK_asr,
                            steps=ode_steps_endloss,
                            direction=-1,
                            cfg_flag_value=asr_cfg_value,
                            spk_e=spk_e_asr_vf,
                            style_e=style_e_asr_match,
                        )
                zT_end_asr = source_to_canonical(y0_end_asr, maskK_asr) if source_to_canonical is not None else y0_end_asr
                if canonical_posterior is not None:
                    zT_hat_asr, zT_hat_logvar_asr = canonical_posterior(zT_end_asr, maskK_asr)
                else:
                    zT_hat_asr = zT_end_asr
                    zT_hat_logvar_asr = None
                if ssl_hidden_zc_active:
                    ssl_pred, ssl_pred_mask = ssl_hidden_head(zT_hat_asr, maskK_asr)
                    with torch.no_grad():
                        teacher_ssl, teacher_ssl_mask = build_hubert_online_targets(
                            wav_paths_asr,
                            starts_asr,
                            ends_asr,
                            target_sr=16000,
                        )
                    loss_ssl_hidden = hidden_ssl_cosine_loss(ssl_pred, ssl_pred_mask, teacher_ssl, teacher_ssl_mask)
                    ssl_hidden_w_now = float(ssl_hidden_w)
                if w_end_bwd > 0.0:
                    loss_end_bwd = masked_mse(zT_hat_asr, zT_asr_mean, train_maskK_asr)
                if canonical_loss_enabled_now:
                    if canonical_match_mode == "alignment_softmin_nll":
                        maskK_asr_grid = attn_grid_mask_from_full_mask(maskK_asr, attn_asr.shape[1])
                        align_candidates = make_alignment_candidates(
                            attn_asr.detach(),
                            maskK_asr_grid,
                            maskL_asr,
                            canonical_align_candidates,
                            duration_perturb_sigma,
                            include_base=True,
                        )

                        def _alignment_softmin_nll(z_hat, *, detach_z=False, detach_prior=False):
                            z_arg = z_hat.detach() if detach_z else z_hat
                            losses = []
                            for cand_attn in align_candidates:
                                mu_cand, logvar_cand, _, _ = expand_attn_prior_to_full(
                                    cand_attn,
                                    maskK_asr_grid,
                                    maskK_asr,
                                    mu_tok_asr,
                                    logvar_tok_asr,
                                    mu_tok_asr,
                                    logvar_tok_asr,
                                    z_hat.dtype,
                                )
                                if detach_prior:
                                    mu_cand = mu_cand.detach()
                                    logvar_cand = logvar_cand.detach()
                                losses.append(masked_gaussian_nll(z_arg, mu_cand, logvar_cand, train_maskK_asr))
                            return softmin_scalar_losses(losses, canonical_softmin_tau)

                        if canonical_stopgrad_split:
                            if canonical_prior_active:
                                loss_canonical_prior_nll = _alignment_softmin_nll(
                                    zT_hat_asr,
                                    detach_z=True,
                                    detach_prior=False,
                                )
                            if canonical_bwd_active:
                                loss_canonical_bwd_nll = _alignment_softmin_nll(
                                    zT_hat_asr,
                                    detach_z=False,
                                    detach_prior=True,
                                )
                        else:
                            loss_canonical_nll = _alignment_softmin_nll(
                                zT_hat_asr,
                                detach_z=False,
                                detach_prior=False,
                            )
                    elif canonical_match_mode == "kl":
                        if zT_hat_logvar_asr is None:
                            raise RuntimeError("canonical_match_mode='kl' requires canonical_posterior")
                        if canonical_stopgrad_split:
                            if canonical_prior_active:
                                loss_canonical_prior_nll = masked_diag_gaussian_kl(
                                    zT_hat_asr.detach(),
                                    zT_hat_logvar_asr.detach(),
                                    mu_asr,
                                    logvar_asr,
                                    train_maskK_asr,
                                    free_bits=canonical_kl_free_bits,
                                )
                            if canonical_bwd_active:
                                loss_canonical_bwd_nll = masked_diag_gaussian_kl(
                                    zT_hat_asr,
                                    zT_hat_logvar_asr,
                                    mu_asr.detach(),
                                    logvar_asr.detach(),
                                    train_maskK_asr,
                                    free_bits=canonical_kl_free_bits,
                                )
                        else:
                            loss_canonical_nll = masked_diag_gaussian_kl(
                                zT_hat_asr,
                                zT_hat_logvar_asr,
                                mu_asr,
                                logvar_asr,
                                train_maskK_asr,
                                free_bits=canonical_kl_free_bits,
                            )
                    else:
                        if canonical_stopgrad_split:
                            # Prior-side canonical fit:
                            # train p_phi(z_c|x,a) to cover the speech-side canonical
                            # point, while keeping the backward endpoint fixed.
                            if canonical_prior_active:
                                loss_canonical_prior_nll = masked_gaussian_nll(
                                    zT_hat_asr.detach(),
                                    mu_asr,
                                    logvar_asr,
                                    train_maskK_asr,
                                )
                            # Backward-side canonical fit:
                            # train q_theta(z_c|s) / the backward rollout endpoint to
                            # land in the current text prior family, while keeping the
                            # text prior fixed.
                            if canonical_bwd_active:
                                loss_canonical_bwd_nll = masked_gaussian_nll(
                                    zT_hat_asr,
                                    mu_asr.detach(),
                                    logvar_asr.detach(),
                                    train_maskK_asr,
                                )
                        else:
                            # Joint point-to-distribution canonical matching:
                            # q(z_c|s) is represented by zT_hat_asr and p_phi(z_c|x,a)
                            # by (mu_asr, logvar_asr).
                            loss_canonical_nll = masked_gaussian_nll(
                                zT_hat_asr,
                                mu_asr,
                                logvar_asr,
                                train_maskK_asr,
                            )
                    if canonical_stopgrad_split:
                        loss_canonical_nll = loss_canonical_prior_nll + loss_canonical_bwd_nll
            elif canonical_loss_enabled_now:
                raise RuntimeError(
                    "Canonical bwd/prior matching requires a backward rollout. "
                    "Set canonical weights to 0 to skip bwd rollout."
                )

            targets, target_lengths = build_ctc_targets_from_texts(texts_asr)
            dit_hidden_ctc_w_now = float(w_dit_hidden_ctc) * linear_anneal(
                step,
                dit_hidden_ctc_start,
                dit_hidden_ctc_anneal_steps,
            )
            dit_hidden_ctc_active = dit_hidden_ctc_head is not None and dit_hidden_ctc_w_now > 0.0
            if dit_hidden_ctc_active:
                with torch.no_grad():
                    if dit_hidden_ctc_anchor == "mean":
                        zT_hidden_c = zT_asr_mean.detach()
                    elif dit_hidden_ctc_anchor == "sample":
                        zT_hidden_c = zT_asr_sample.detach()
                    else:
                        mix_a = float(max(0.0, min(1.0, dit_hidden_ctc_anchor_mix_alpha)))
                        zT_hidden_c = (mix_a * zT_asr_mean + (1.0 - mix_a) * zT_asr_sample).detach()

                    zS_hidden = zS_asr.detach()
                    spk_e_hidden = (
                        spk_e_asr_match.detach().to(dtype=zS_hidden.dtype)
                        if spk_e_asr_match is not None
                        else None
                    )
                    style_e_hidden = asr_style_cond_from_source(
                        zS_hidden,
                        maskK_asr,
                        spk_e_hidden,
                        dtype=zS_hidden.dtype,
                    )
                    spk_e_hidden = asr_vf_spk_cond(spk_e_hidden)
                    if canonical_to_source is not None:
                        zT_hidden_source = canonical_to_source(
                            zT_hidden_c,
                            maskK_asr,
                            spk_e=spk_e_hidden,
                            style_e=style_e_hidden if tts_style_into_source else None,
                        )
                    else:
                        zT_hidden_source = zT_hidden_c
                    t_hidden = dit_hidden_ctc_t_min + (
                        dit_hidden_ctc_t_max - dit_hidden_ctc_t_min
                    ) * torch.rand(B_asr, device=device)
                    t_hidden_ = t_hidden.view(B_asr, 1, 1)
                    zt_hidden_ctc = (1.0 - t_hidden_) * zT_hidden_source + t_hidden_ * zS_hidden
                    cfg_flag_hidden = torch.full(
                        (B_asr,),
                        int(asr_cfg_value),
                        dtype=torch.long,
                        device=device,
                    )
                _, h_dit_hidden = vf(
                    zt_hidden_ctc,
                    t_hidden,
                    maskK_asr,
                    cfg_flag=cfg_flag_hidden,
                    spk_e=spk_e_hidden.to(dtype=zt_hidden_ctc.dtype) if spk_e_hidden is not None else None,
                    style_e=style_e_hidden.to(dtype=zt_hidden_ctc.dtype) if style_e_hidden is not None else None,
                    text_cond=None,
                    return_hidden=True,
                    hidden_tap_index=dit_hidden_ctc_tap_index,
                )
                hD_ctc, maskD_hidden_ctc, K_list_dit_hidden_ctc = prepare_ctc_input(
                    h_dit_hidden,
                    maskK_asr,
                    apply_subsample=dit_hidden_ctc_apply_subsample,
                )
                logits_dit_hidden = dit_hidden_ctc_head(hD_ctc, maskD_hidden_ctc)
                logp_dit_hidden = F.log_softmax(logits_dit_hidden, dim=-1).transpose(0, 1)
                input_lengths_dit_hidden = torch.tensor(
                    K_list_dit_hidden_ctc,
                    dtype=torch.long,
                    device=device,
                )
                loss_dit_hidden_ctc = ctc_loss_fn(
                    logp_dit_hidden,
                    targets,
                    input_lengths_dit_hidden,
                    target_lengths,
                )
            if ctc_T_active:
                zT_ctc, maskT_ctc, K_list_T_ctc = prepare_ctc_input(
                    zT_asr_mean,
                    maskK_asr,
                    apply_subsample=(ctc_subsample_apply_to == "both"),
                )
                logits_T = text_ctc_head(zT_ctc, maskT_ctc)
                logp_T = F.log_softmax(logits_T, dim=-1).transpose(0, 1)
                input_lengths_T = torch.tensor(K_list_T_ctc, dtype=torch.long, device=device)
                loss_ctc_T = ctc_loss_fn(logp_T, targets, input_lengths_T, target_lengths)
            if ctc_hat_active or att_decoder_active:
                if zT_hat_asr is None:
                    raise RuntimeError("ctcH/AED is active but backward rollout was skipped")
                zH_ctc, maskH_ctc, K_list_H_ctc = prepare_ctc_input(
                    zT_hat_asr,
                    maskK_asr,
                    apply_subsample=(ctc_subsample_apply_to in {"hat", "both"}),
                )
            if ctc_hat_active:
                logits_hat = text_ctc_head(zH_ctc, maskH_ctc)
                logp_hat = F.log_softmax(logits_hat, dim=-1).transpose(0, 1)
                input_lengths_H = torch.tensor(K_list_H_ctc, dtype=torch.long, device=device)
                loss_ctc_hat = ctc_loss_fn(logp_hat, targets, input_lengths_H, target_lengths)
            if att_decoder_active:
                dec_in, dec_tgt = build_att_decoder_batch(texts_asr)
                dec_memory = zH_ctc.detach() if att_decoder_detach_input else zH_ctc
                logits_att_decoder = att_decoder(dec_memory, maskH_ctc, dec_in)
                loss_att_decoder = F.cross_entropy(
                    logits_att_decoder.reshape(-1, AED_VOCAB_SIZE),
                    dec_tgt.reshape(-1),
                    ignore_index=PAD_ID,
                    label_smoothing=att_decoder_label_smoothing,
                )
                with torch.no_grad():
                    valid = dec_tgt.ne(PAD_ID)
                    if bool(valid.any().item()):
                        pred = logits_att_decoder.argmax(dim=-1)
                        att_decoder_token_acc = float(
                            (pred.eq(dec_tgt) & valid).float().sum().div(valid.float().sum()).item()
                        )
            sample_ctc_active = (
                enable_zc_sample_ctc
                and w_ctc_sample > 0.0
                and step >= ctc_sample_start
            )
            if sample_ctc_active:
                sample_temp = float(ctc_sample_temp)
                if sample_temp == 1.0:
                    zT_sample_for_ctc = zT_asr_sample
                else:
                    zT_sample_for_ctc = zT_asr_mean + sample_temp * (zT_asr_sample - zT_asr_mean)
                    zT_sample_for_ctc = zT_sample_for_ctc * train_maskK_asr.float().unsqueeze(-1)
                zZ_ctc, maskZ_ctc, K_list_Z_ctc = prepare_ctc_input(
                    zT_sample_for_ctc,
                    maskK_asr,
                    apply_subsample=(ctc_subsample_apply_to == "both"),
                )
                logits_sample = text_ctc_head(zZ_ctc, maskZ_ctc)
                logp_sample = F.log_softmax(logits_sample, dim=-1).transpose(0, 1)
                input_lengths_sample = torch.tensor(K_list_Z_ctc, dtype=torch.long, device=device)
                loss_ctc_sample = ctc_loss_fn(logp_sample, targets, input_lengths_sample, target_lengths)
                sample_ctc_w_now = float(w_ctc_sample)
            source_ctc_active = (
                source_ctc_head is not None
                and w_ctc_source > 0.0
                and step >= source_ctc_start
            )
            if source_ctc_active:
                zS_source_ctc, mask_source_ctc, K_list_source_ctc = prepare_ctc_input(
                    zS_asr.detach(),
                    train_maskK_asr,
                    apply_subsample=(ctc_subsample_apply_to in {"hat", "both"}),
                )
                logits_source = source_ctc_head(zS_source_ctc, mask_source_ctc)
                logp_source = F.log_softmax(logits_source, dim=-1).transpose(0, 1)
                input_lengths_source = torch.tensor(K_list_source_ctc, dtype=torch.long, device=device)
                loss_ctc_source = ctc_loss_fn(logp_source, targets, input_lengths_source, target_lengths)
                source_ctc_w_now = float(w_ctc_source)
            loss_ctc_dur = torch.tensor(0.0, device=device)
            if enable_ctc_dur:
                zT_dur_mean, maskK_dur, K_list_dur = build_duration_expanded_prior_batch(h_enc_asr, maskL_asr, mu_tok_asr)
                zD_ctc, maskD_ctc, K_list_D_ctc = prepare_ctc_input(
                    zT_dur_mean,
                    maskK_dur,
                    apply_subsample=(ctc_subsample_apply_to == "both"),
                )
                logits_dur = text_ctc_head(zD_ctc, maskD_ctc)
                logp_dur = F.log_softmax(logits_dur, dim=-1).transpose(0, 1)
                input_lengths_dur = torch.tensor(K_list_D_ctc, dtype=torch.long, device=device)
                loss_ctc_dur = ctc_loss_fn(logp_dur, targets, input_lengths_dur, target_lengths)
            loss_ctc_full = torch.tensor(0.0, device=device)
            if full_asr_aux_active and zS_full_aux_log is not None:
                zS_full_aux = (zS_full_aux_log - mu_b) / std_b
                spk_e_full_aux = None
                if spk_ids_full_aux is not None:
                    spk_e_full_aux = asr_spk_cond_from_ref_paths(
                        wav_paths_full_aux,
                        spk_ids_full_aux,
                        dtype=zS_full_aux.dtype,
                    )
                logits_full_chunks = []
                input_lengths_full = []
                core = int(max(32, full_asr_chunk_core))
                ctx = int(max(0, full_asr_chunk_ctx))
                for b, K_full in enumerate(K_list_full_aux):
                    zS_one = zS_full_aux[b:b + 1, :K_full]
                    spk_e_one_full = (
                        spk_e_full_aux[b:b + 1].to(dtype=zS_one.dtype)
                        if spk_e_full_aux is not None
                        else None
                    )
                    cfg_one_full = asr_cfg_flag_value(spk_e_one_full)
                    mask_one_full = torch.ones(1, K_full, device=device, dtype=torch.bool)
                    style_e_one_full = asr_style_cond_from_source(
                        zS_one,
                        mask_one_full,
                        spk_e_one_full,
                        dtype=zS_one.dtype,
                    )
                    spk_e_one_full_vf = asr_vf_spk_cond(spk_e_one_full)
                    if full_asr_ctc_aux_whole_utterance:
                        if full_asr_use_euler:
                            full_integrate = euler_integrate_grad if full_asr_ctc_aux_ode_grad else euler_integrate
                            if full_asr_ctc_aux_ode_grad:
                                zT_one = full_integrate(
                                    vf,
                                    zS_one,
                                    mask_one_full,
                                    steps=full_asr_ctc_aux_steps,
                                    direction=-1,
                                    cfg_flag_value=cfg_one_full,
                                    spk_e=spk_e_one_full_vf,
                                    style_e=style_e_one_full,
                                )
                            else:
                                with torch.no_grad():
                                    zT_one = full_integrate(
                                        vf,
                                        zS_one,
                                        mask_one_full,
                                        steps=full_asr_ctc_aux_steps,
                                        direction=-1,
                                        cfg_flag_value=cfg_one_full,
                                        spk_e=spk_e_one_full_vf,
                                        style_e=style_e_one_full,
                                    )
                        else:
                            if full_asr_ctc_aux_ode_grad:
                                zT_one = heun_integrate(
                                    vf,
                                    zS_one,
                                    mask_one_full,
                                    steps=full_asr_ctc_aux_steps,
                                    direction=-1,
                                    cfg_scale=1.0,
                                    spk_e=spk_e_one_full_vf,
                                    style_e=style_e_one_full,
                                )
                            else:
                                with torch.no_grad():
                                    zT_one = heun_integrate(
                                        vf,
                                        zS_one,
                                        mask_one_full,
                                        steps=full_asr_ctc_aux_steps,
                                        direction=-1,
                                        cfg_scale=1.0,
                                        spk_e=spk_e_one_full_vf,
                                        style_e=style_e_one_full,
                                    )
                        zC_one = source_to_canonical(zT_one, mask_one_full) if source_to_canonical is not None else zT_one
                        if canonical_posterior is not None:
                            zC_one, _ = canonical_posterior(zC_one, mask_one_full)
                    else:
                        chunk_zc = []
                        pos = 0
                        while pos < K_full:
                            core_s = pos
                            core_e = min(K_full, pos + core)
                            s = max(0, core_s - ctx)
                            e = min(K_full, core_e + ctx)
                            zS_chunk = zS_one[:, s:e]
                            mask_chunk = torch.ones(1, e - s, device=device, dtype=torch.bool)
                            if full_asr_use_euler:
                                full_integrate = euler_integrate_grad if full_asr_ctc_aux_ode_grad else euler_integrate
                                if full_asr_ctc_aux_ode_grad:
                                    zT_chunk = full_integrate(
                                        vf,
                                        zS_chunk,
                                        mask_chunk,
                                        steps=full_asr_ctc_aux_steps,
                                        direction=-1,
                                        cfg_flag_value=cfg_one_full,
                                        spk_e=spk_e_one_full_vf,
                                        style_e=style_e_one_full,
                                    )
                                else:
                                    with torch.no_grad():
                                        zT_chunk = full_integrate(
                                            vf,
                                            zS_chunk,
                                            mask_chunk,
                                            steps=full_asr_ctc_aux_steps,
                                            direction=-1,
                                            cfg_flag_value=cfg_one_full,
                                            spk_e=spk_e_one_full_vf,
                                            style_e=style_e_one_full,
                                        )
                            else:
                                if full_asr_ctc_aux_ode_grad:
                                    zT_chunk = heun_integrate(
                                        vf,
                                        zS_chunk,
                                        mask_chunk,
                                        steps=full_asr_ctc_aux_steps,
                                        direction=-1,
                                        cfg_scale=1.0,
                                        spk_e=spk_e_one_full_vf,
                                        style_e=style_e_one_full,
                                    )
                                else:
                                    with torch.no_grad():
                                        zT_chunk = heun_integrate(
                                            vf,
                                            zS_chunk,
                                            mask_chunk,
                                            steps=full_asr_ctc_aux_steps,
                                            direction=-1,
                                            cfg_scale=1.0,
                                            spk_e=spk_e_one_full_vf,
                                            style_e=style_e_one_full,
                                        )
                            zC_chunk = source_to_canonical(zT_chunk, mask_chunk) if source_to_canonical is not None else zT_chunk
                            if canonical_posterior is not None:
                                zC_chunk, _ = canonical_posterior(zC_chunk, mask_chunk)
                            keep_s = core_s - s
                            keep_e = core_e - s
                            chunk_zc.append(zC_chunk[:, keep_s:keep_e, :])
                            pos = core_e
                        zC_one = torch.cat(chunk_zc, dim=1)
                    mask_one = torch.ones(1, zC_one.shape[1], device=device, dtype=torch.bool)
                    zC_one_ctc, mask_one_ctc, K_one_ctc = prepare_ctc_input(
                        zC_one,
                        mask_one,
                        apply_subsample=(ctc_subsample_apply_to in {"hat", "both"}),
                    )
                    logits_one = text_ctc_head(zC_one_ctc, mask_one_ctc)
                    logits_full_chunks.append(logits_one)
                    input_lengths_full.append(int(K_one_ctc[0]))

                Tmax_full = max(input_lengths_full)
                logits_full = torch.zeros(len(logits_full_chunks), Tmax_full, Vt, device=device, dtype=logits_full_chunks[0].dtype)
                for b, logits_one in enumerate(logits_full_chunks):
                    logits_full[b, :logits_one.shape[1]] = logits_one
                logp_full = F.log_softmax(logits_full, dim=-1).transpose(0, 1)
                input_lengths_full = torch.tensor(input_lengths_full, dtype=torch.long, device=device)
                targets_full, target_lengths_full = build_ctc_targets_from_texts(texts_full_aux)
                loss_ctc_full = ctc_loss_fn(logp_full, targets_full, input_lengths_full, target_lengths_full)

            w_ctc = 0.0 if step < w_ctc_start else 1.0
            w_ctc_dur_on = 0.0 if step < ctc_dur_start else 1.0
            w_align = 0.0 if step < w_align_start else 1.0
            loss_end = w_end_fwd * loss_end_fwd + w_end_bwd * loss_end_bwd
            if canonical_stopgrad_split:
                loss_canonical_total = (
                    canonical_prior_nll_w_now * loss_canonical_prior_nll
                    + canonical_bwd_nll_w_now * loss_canonical_bwd_nll
                )
            else:
                loss_canonical_total = canonical_nll_w_now * loss_canonical_nll
            loss = (
                w_fm * loss_fm
                + bwd_fm_w_now * loss_bwd_fm
                + loss_end
                + w_tts_mel * loss_tts_mel
                + w_mel_high * loss_mel_high
                + w_delta * loss_delta
                + w_ref * loss_ref
                + w_prior * loss_prior
                + w_mel_range * loss_range
                + (w_stat * loss_stat if use_stat_match else 0.0)
                + w_ctc * (w_ctc_T * loss_ctc_T + w_ctc_hat * loss_ctc_hat)
                + sample_ctc_w_now * loss_ctc_sample
                + source_ctc_w_now * loss_ctc_source
                + dit_hidden_ctc_w_now * loss_dit_hidden_ctc
                + att_decoder_w_now * loss_att_decoder
                + w_ctc_dur_on * w_ctc_dur * loss_ctc_dur
                + w_ctc_full * loss_ctc_full
                + w_align * (w_dur * loss_dur_tts + w_len * loss_len_tts)
                + loss_canonical_total
                + tts_style_kl_tts_w_now * loss_tts_style_kl_tts
                + tts_style_kl_asr_w_now * loss_tts_style_kl_asr
                + ssl_hidden_w_now * loss_ssl_hidden
            )
        if perf_due:
            maybe_cuda_sync()
            perf_step["forward"] = time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        ratios_last = None
        if enable_vf_lip and (step >= vf_lip_start) and (step % vf_lip_every == 0) and (zt_for_lip is not None):
            with torch.amp.autocast(amp_device, enabled=False):
                ratios_now = vf_lip_fd_ratio(
                    vf,
                    zt_for_lip.float(),
                    t_for_lip.float(),
                    mask_for_lip,
                    cfg_for_lip,
                    spk_for_lip.float(),
                    style_e=style_for_lip.float() if style_for_lip is not None else None,
                    sigma=vf_lip_sigma,
                )
                loss_vf_lip = F.relu(ratios_now - float(vf_lip_L_hi)).mean()
                ratios_last = ratios_now.detach()
            loss = loss + float(w_vf_lip) * loss_vf_lip
        if perf_due:
            maybe_cuda_sync()
            perf_step["vf_lip"] = time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        if use_stft and (mel_refiner is not None) and enable_fwd_end_loss:
            K_stft = 256
            stft_one_sample = True
            with torch.amp.autocast(amp_device, enabled=False):
                zS_end_det = zS_end_tts.float().detach()
                h_enc_det = h_enc_tts.float().detach()
                zS_ref_full = mel_refiner(zS_end_det, cond=h_enc_det, cond_mask=maskL_tts)

            b_list = [0] if stft_one_sample else list(range(zS_ref_full.shape[0]))
            stft_acc = 0.0
            denom = 0
            for b in b_list:
                Kb = int(maskK_tts[b].sum().item())
                if Kb < 8:
                    continue
                Kseg2 = min(K_stft, Kb)
                s0 = 0 if Kb <= Kseg2 else int(np.random.randint(0, Kb - Kseg2 + 1))
                e0 = s0 + Kseg2

                with torch.amp.autocast(amp_device, enabled=False):
                    mel_hat_seg = (zS_ref_full[b:b + 1, s0:e0] * std_b.float() + mu_b.float())
                    mel_hat_seg = mel_hat_seg.clamp(mel_floor, mel_ceil).transpose(1, 2).contiguous()
                with torch.amp.autocast(amp_device, enabled=use_amp):
                    wav_hat = bigvgan_model(mel_hat_seg).squeeze()
                wav_hat = wav_hat.float()

                wav_np = load_wav_full_cached(wav_paths_tts[b])
                s_frame = int(starts_tts[b]) + int(s0)
                e_frame = int(starts_tts[b]) + int(e0)
                s_samp = s_frame * hop_size
                e_samp = e_frame * hop_size
                wav_gt_np = wav_np[s_samp:e_samp]
                if wav_gt_np.size <= 64 or wav_hat.numel() <= 64:
                    continue

                wav_gt = torch.tensor(wav_gt_np, device=device, dtype=torch.float32)
                Tm = min(wav_hat.numel(), wav_gt.numel())
                if Tm <= 64:
                    continue
                stft_acc = stft_acc + mrstft_loss(wav_hat[:Tm], wav_gt[:Tm])
                denom += 1

            loss_stft = stft_acc / max(denom, 1)
            loss = loss + float(w_stft) * loss_stft
        if perf_due:
            maybe_cuda_sync()
            perf_step["stft"] = time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        if torch.isnan(loss):
            print("[FATAL] loss is NaN at step", step)
            break

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        if source_ctc_active and source_ctc_opt is not None:
            scaler.unscale_(source_ctc_opt)

        torch.nn.utils.clip_grad_norm_(vf.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(text_prior.parameters(), grad_clip)
        if canonical_prior is not None:
            torch.nn.utils.clip_grad_norm_(canonical_prior.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(canonical_to_source.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(source_to_canonical.parameters(), grad_clip)
        if canonical_posterior is not None:
            torch.nn.utils.clip_grad_norm_(canonical_posterior.parameters(), grad_clip)
        if adapter is not None:
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip)
        if trainable_text_encoder is not None:
            torch.nn.utils.clip_grad_norm_(trainable_text_encoder.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(dur_pred.parameters(), grad_clip)
        if len_pred is not None:
            torch.nn.utils.clip_grad_norm_(len_pred.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(text_ctc_head.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(spk_table.parameters(), grad_clip)
        if tts_style_post is not None:
            torch.nn.utils.clip_grad_norm_(tts_style_post.parameters(), grad_clip)
            if tts_style_pair_post is not None:
                torch.nn.utils.clip_grad_norm_(tts_style_pair_post.parameters(), grad_clip)
            if tts_style_prior is not None:
                torch.nn.utils.clip_grad_norm_(tts_style_prior.parameters(), grad_clip)
            if tts_style_to_source is not None:
                torch.nn.utils.clip_grad_norm_(tts_style_to_source.parameters(), grad_clip)
        if ssl_hidden_head is not None:
            torch.nn.utils.clip_grad_norm_(ssl_hidden_head.parameters(), grad_clip)
        if dit_hidden_ctc_head is not None:
            torch.nn.utils.clip_grad_norm_(dit_hidden_ctc_head.parameters(), grad_clip)
        if att_decoder is not None:
            torch.nn.utils.clip_grad_norm_(att_decoder.parameters(), grad_clip)
        if source_ctc_active and source_ctc_head is not None:
            torch.nn.utils.clip_grad_norm_(source_ctc_head.parameters(), grad_clip)
        if mel_refiner is not None:
            torch.nn.utils.clip_grad_norm_(mel_refiner.parameters(), grad_clip)

        scaler.step(opt)
        if source_ctc_active and source_ctc_opt is not None:
            scaler.step(source_ctc_opt)
        scaler.update()
        if use_ema and ema is not None:
            ema.update(module_map)
        if perf_due:
            maybe_cuda_sync()
            perf_step["backward_opt"] = time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        if debug_every and (step % debug_every == 0):
            with torch.no_grad():
                kt0 = int(maskK_tts[0].sum().item())
                ka0 = int(maskK_asr[0].sum().item())
                zs_min = float(zS_tts[0, :kt0].min().item()) if kt0 > 0 else 0.0
                zs_max = float(zS_tts[0, :kt0].max().item()) if kt0 > 0 else 0.0
                zt_min = float(zT_tts_mean[0, :kt0].min().item()) if kt0 > 0 else 0.0
                zt_max = float(zT_tts_mean[0, :kt0].max().item()) if kt0 > 0 else 0.0
                mel_min_v = float(mel_end[0, :kt0].min().item()) if kt0 > 0 else 0.0
                mel_max_v = float(mel_end[0, :kt0].max().item()) if kt0 > 0 else 0.0
            print(
                f"[DBG step{step:05d}] "
                f"TTS_K0={kt0} ASR_K0={ka0} "
                f"zS_tts[min,max]=[{zs_min:.2f},{zs_max:.2f}] "
                f"zT_tts_mean[min,max]=[{zt_min:.2f},{zt_max:.2f}] "
                f"mel_end[min,max]=[{mel_min_v:.2f},{mel_max_v:.2f}] "
                f"stat={loss_stat.item() if use_stat_match else 0.0:.4f}"
            )
        if perf_due:
            maybe_cuda_sync()
            perf_step["debug"] = time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        if log_every and (step % log_every == 0):
            with torch.no_grad():
                fwd = loss_end_fwd.item() if fwd_rollout_active else 0.0
                bwd = masked_mse(zT_hat_asr, zT_asr_mean, maskK_asr).item() if zT_hat_asr is not None else 0.0
                if logits_hat is not None and K_list_H_ctc:
                    decoded_ids = ctc_greedy_decode(logits_hat.detach(), K_list_H_ctc, blank_id=BLANK_ID)
                else:
                    decoded_ids = []

                muT, stdT = masked_mean_std(zT_tts_mean_source, maskK_tts)
                muS, stdS = masked_mean_std(zS_tts, maskK_tts)
                print("[CHK-TTS] mean|muT-muS|", (muT - muS).abs().mean().item(), " mean|stdT-stdS|", (stdT - stdS).abs().mean().item())

                batch_wers = []
                for ref_txt, hyp_ids in zip(texts_asr, decoded_ids):
                    hyp_txt = canonicalize_text(tok.decode(hyp_ids))
                    batch_wers.append(word_error_rate_text(ref_txt, hyp_txt))
                batch_wer_avg = float(np.mean(batch_wers)) if batch_wers else 0.0
                decoded_sample_ids = []
                batch_sample_wer_avg = 0.0
                if sample_ctc_active:
                    decoded_sample_ids = ctc_greedy_decode(logits_sample.detach(), K_list_Z_ctc, blank_id=BLANK_ID)
                    sample_wers = []
                    for ref_txt, hyp_ids in zip(texts_asr, decoded_sample_ids):
                        hyp_txt = canonicalize_text(tok.decode(hyp_ids))
                        sample_wers.append(word_error_rate_text(ref_txt, hyp_txt))
                    batch_sample_wer_avg = float(np.mean(sample_wers)) if sample_wers else 0.0
                decoded_source_ids = []
                batch_source_wer_avg = 0.0
                if source_ctc_active:
                    decoded_source_ids = ctc_greedy_decode(logits_source.detach(), K_list_source_ctc, blank_id=BLANK_ID)
                    source_wers = []
                    for ref_txt, hyp_ids in zip(texts_asr, decoded_source_ids):
                        hyp_txt = canonicalize_text(tok.decode(hyp_ids))
                        source_wers.append(word_error_rate_text(ref_txt, hyp_txt))
                    batch_source_wer_avg = float(np.mean(source_wers)) if source_wers else 0.0
                decoded_dit_hidden_ids = []
                batch_dit_hidden_wer_avg = 0.0
                if dit_hidden_ctc_active and logits_dit_hidden is not None and K_list_dit_hidden_ctc:
                    decoded_dit_hidden_ids = ctc_greedy_decode(
                        logits_dit_hidden.detach(),
                        K_list_dit_hidden_ctc,
                        blank_id=BLANK_ID,
                    )
                    dit_hidden_wers = []
                    for ref_txt, hyp_ids in zip(texts_asr, decoded_dit_hidden_ids):
                        hyp_txt = canonicalize_text(tok.decode(hyp_ids))
                        dit_hidden_wers.append(word_error_rate_text(ref_txt, hyp_txt))
                    batch_dit_hidden_wer_avg = float(np.mean(dit_hidden_wers)) if dit_hidden_wers else 0.0

            lip_str = ""
            if enable_vf_lip and ratios_last is not None and (step >= vf_lip_start) and (step % vf_lip_every == 0):
                rr = ratios_last.float().cpu().numpy().tolist()
                rr_sorted = sorted(rr)
                p50 = rr_sorted[len(rr_sorted) // 2]
                p90 = rr_sorted[int(round(0.9 * (len(rr_sorted) - 1)))]
                lip_str = f" lip {loss_vf_lip.item():.4f} Lp50 {p50:.3f} Lp90 {p90:.3f}"
            elif enable_vf_lip and (step % max(1, vf_lip_print_every) == 0):
                lip_str = f" lip {loss_vf_lip.item():.4f}"

            stft_str = f" stft {loss_stft.item():.4f}" if use_stft else ""
            style_str = (
                f" ukl {loss_tts_style_kl.item():.4f}"
                f" uklT {loss_tts_style_kl_tts.item():.4f} ukwT {tts_style_kl_tts_w_now:.4g}"
                f" uklA {loss_tts_style_kl_asr.item():.4f} ukwA {tts_style_kl_asr_w_now:.4g}"
            ) if use_tts_style_latent else ""
            can_name = "cKL" if canonical_match_mode == "kl" else "can"
            if enable_canonical_nll and canonical_stopgrad_split:
                can_str = (
                    f" {can_name}P {loss_canonical_prior_nll.item():.4f} canPw {canonical_prior_nll_w_now:.4g}"
                    f" {can_name}B {loss_canonical_bwd_nll.item():.4f} canBw {canonical_bwd_nll_w_now:.4g}"
                )
            elif enable_canonical_nll:
                can_str = f" {can_name} {loss_canonical_nll.item():.4f} canw {canonical_nll_w_now:.4g}"
            else:
                can_str = ""
            ssl_name = "sslZ" if ssl_hidden_target == "zc" else "sslH"
            ssl_str = f" {ssl_name} {loss_ssl_hidden.item():.4f} sslw {ssl_hidden_w_now:.4g}" if ssl_hidden_enable else ""
            dit_hctc_str = (
                f" hCTC {loss_dit_hidden_ctc.item():.4f} hCTCw {dit_hidden_ctc_w_now:.4g}"
                if enable_dit_hidden_ctc
                else ""
            )
            att_decoder_str = (
                f" aed {loss_att_decoder.item():.4f} aedw {att_decoder_w_now:.4g} aedAcc {att_decoder_token_acc:.4f}"
                if enable_att_decoder
                else ""
            )
            bwd_fm_str = f" bwdFM {loss_bwd_fm.item():.4f} bwdFMw {bwd_fm_w_now:.4g}" if enable_bwd_fm else ""
            print(
                f"[JOINT-CUT] step {step:05d} loss {loss.item():.6f} lr {lr_now:.6g} "
                f"fm {loss_fm.item():.6f} fmw {w_fm:.4g}{bwd_fm_str} stat {loss_stat.item() if use_stat_match else 0.0:.4f} "
                f"end {loss_end.item():.4f} vf_lip {loss_vf_lip.item():.4f} "
                f"prior {loss_prior.item():.4f} pmu {loss_prior_mu.item():.4f} "
                f"pvar {loss_prior_var.item():.4f} pnll {loss_prior_nll.item():.4f} "
                f"tts {loss_tts_mel.item():.4f} hi {loss_mel_high.item():.4f} ref {loss_ref.item():.4f} d {loss_delta.item():.4f} "
                f"ctcT {loss_ctc_T.item():.4f} ctcH {loss_ctc_hat.item():.4f} ctcZ {loss_ctc_sample.item():.4f} ctcZw {sample_ctc_w_now:.4g} ctcS {loss_ctc_source.item():.4f} ctcSw {source_ctc_w_now:.4g} ctcD {loss_ctc_dur.item():.4f} ctcF {loss_ctc_full.item():.4f} "
                f"dur {loss_dur_tts.item():.4f} len {loss_len_tts.item():.4f} "
                f"fwd {fwd:.4f} bwd {bwd:.4f} batchWER {batch_wer_avg:.4f} batchWERZ {batch_sample_wer_avg:.4f} batchWERS {batch_source_wer_avg:.4f} batchWERh {batch_dit_hidden_wer_avg:.4f}"
                f" Bt {B_tts} Kt {sumK_tts}/{Kmax_tts} padt {paddedK_tts}"
                f" Ba {B_asr} Ka {sumK_asr}/{Kmax_asr} pada {paddedK_asr}"
                f"{style_str}{can_str}{ssl_str}{dit_hctc_str}{att_decoder_str}{lip_str}{stft_str}"
            )
            for i in range(min(2, B_tts)):
                print(f"  [TTS {i}] spk={spk_list[int(spk_ids_tts[i])]} TXT : {texts_tts[i]}")
            for i in range(min(2, B_asr)):
                print(f"  [ASR {i}] spk={spk_list[int(spk_ids_asr[i])]} GT  : {texts_asr[i]}")
                if i < len(decoded_ids):
                    print(f"           HYP : {tok.decode(decoded_ids[i])}")
                else:
                    print("           HYP : <ctcH disabled>")
                if sample_ctc_active and i < len(decoded_sample_ids):
                    print(f"    HYP-zCsample : {tok.decode(decoded_sample_ids[i])}")
                if source_ctc_active and i < len(decoded_source_ids):
                    print(f"        HYP-zS : {tok.decode(decoded_source_ids[i])}")
                if dit_hidden_ctc_active and i < len(decoded_dit_hidden_ids):
                    print(f"    HYP-DiTHid : {tok.decode(decoded_dit_hidden_ids[i])}")

            if perf_due:
                print(
                    f"[PERF] step {step:05d} "
                    f"data {perf_step['data']:.3f}s text {perf_step['text']:.3f}s "
                    f"teacher {perf_step['teacher']:.3f}s forward {perf_step['forward']:.3f}s "
                    f"vf_lip {perf_step['vf_lip']:.3f}s stft {perf_step['stft']:.3f}s "
                    f"bwd_opt {perf_step['backward_opt']:.3f}s debug {perf_step['debug']:.3f}s"
                )

            append_jsonl(
                metrics_log_path,
                dict(
                    step=int(step),
                    loss=float(loss.item()),
                    loss_fm=float(loss_fm.item()),
                    w_fm=float(w_fm),
                    loss_bwd_fm=float(loss_bwd_fm.item()),
                    w_bwd_fm_now=float(bwd_fm_w_now),
                    loss_end=float(loss_end.item()),
                    loss_end_fwd=float(loss_end_fwd.item()),
                    loss_end_bwd=float(loss_end_bwd.item()),
                    loss_prior=float(loss_prior.item()),
                    loss_prior_mu=float(loss_prior_mu.item()),
                    loss_prior_var=float(loss_prior_var.item()),
                    loss_prior_nll=float(loss_prior_nll.item()),
                    loss_tts_mel=float(loss_tts_mel.item()),
                    loss_mel_high=float(loss_mel_high.item()),
                    loss_ref=float(loss_ref.item()),
                    loss_delta=float(loss_delta.item()),
                    loss_ctc_T=float(loss_ctc_T.item()),
                    loss_ctc_hat=float(loss_ctc_hat.item()),
                    loss_ctc_sample=float(loss_ctc_sample.item()),
                    w_ctc_sample_now=float(sample_ctc_w_now),
                    loss_ctc_source=float(loss_ctc_source.item()),
                    w_ctc_source_now=float(source_ctc_w_now),
                    loss_ctc_dur=float(loss_ctc_dur.item()),
                    loss_ctc_full=float(loss_ctc_full.item()),
                    loss_dit_hidden_ctc=float(loss_dit_hidden_ctc.item()),
                    w_dit_hidden_ctc_now=float(dit_hidden_ctc_w_now),
                    loss_att_decoder=float(loss_att_decoder.item()),
                    w_att_decoder_now=float(att_decoder_w_now),
                    att_decoder_token_acc=float(att_decoder_token_acc),
                    loss_dur_tts=float(loss_dur_tts.item()),
                    loss_len_tts=float(loss_len_tts.item()),
                    loss_stat=float(loss_stat.item()) if use_stat_match else 0.0,
                    loss_vf_lip=float(loss_vf_lip.item()),
                    loss_canonical_nll=float(loss_canonical_nll.item()),
                    w_canonical_nll_now=float(canonical_nll_w_now),
                    loss_canonical_prior_nll=float(loss_canonical_prior_nll.item()),
                    loss_canonical_bwd_nll=float(loss_canonical_bwd_nll.item()),
                    w_canonical_prior_nll_now=float(canonical_prior_nll_w_now),
                    w_canonical_bwd_nll_now=float(canonical_bwd_nll_w_now),
                    loss_tts_style_kl=float(loss_tts_style_kl.item()),
                    loss_tts_style_kl_tts=float(loss_tts_style_kl_tts.item()),
                    loss_tts_style_kl_asr=float(loss_tts_style_kl_asr.item()),
                    w_tts_style_kl_now=float(tts_style_kl_tts_w_now),
                    w_tts_style_asr_kl_now=float(tts_style_kl_asr_w_now),
                    loss_ssl_hidden=float(loss_ssl_hidden.item()),
                    w_ssl_hidden_now=float(ssl_hidden_w_now),
                    loss_stft=float(loss_stft.item()),
                    fwd=float(fwd),
                    bwd=float(bwd),
                    batch_wer_avg=float(batch_wer_avg),
                    batch_wer_dit_hidden_avg=float(batch_dit_hidden_wer_avg),
                    batch_wer_source_avg=float(batch_source_wer_avg),
                    batch_tts_B=int(B_tts),
                    batch_asr_B=int(B_asr),
                    batch_tts_sumK=int(sumK_tts),
                    batch_asr_sumK=int(sumK_asr),
                    batch_tts_Kmax=int(Kmax_tts),
                    batch_asr_Kmax=int(Kmax_asr),
                    batch_tts_paddedK=int(paddedK_tts),
                    batch_asr_paddedK=int(paddedK_asr),
                    lr=float(lr_now),
                    perf_data=float(perf_step["data"]),
                    perf_text=float(perf_step["text"]),
                    perf_teacher=float(perf_step["teacher"]),
                    perf_forward=float(perf_step["forward"]),
                    perf_vf_lip=float(perf_step["vf_lip"]),
                    perf_stft=float(perf_step["stft"]),
                    perf_backward_opt=float(perf_step["backward_opt"]),
                    perf_debug=float(perf_step["debug"]),
                ),
            )
        if perf_due:
            maybe_cuda_sync()
            perf_step["log"] = time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        if enable_bilip_diag and diag_every and (step % diag_every == 0) and (batch_size_tts >= 2):
            with torch.no_grad():
                zT_diag_source = zT_tts_mean_source
                spk_e_cond = spk_e_tts_match.to(dtype=zT_diag_source.dtype)
                diag_asr_ref_paths = wav_paths_tts if zero_shot_asr_ref_source == "self" else ref_wav_paths_tts
                spk_e_bwd_diag = asr_spk_cond_from_ref_paths(
                    diag_asr_ref_paths,
                    spk_ids_tts,
                    dtype=zS_tts.dtype,
                )
                style_e_bwd_diag = asr_style_cond_from_source(
                    zS_tts,
                    maskK_tts,
                    spk_e_bwd_diag,
                    dtype=zS_tts.dtype,
                )
                cfg_bwd_diag = asr_cfg_flag_value(spk_e_bwd_diag)
                zS_map = euler_integrate(vf, zT_diag_source, maskK_tts, steps=ode_steps_endloss, direction=+1, cfg_flag_value=1, spk_e=spk_e_cond)
                zT_map = euler_integrate(
                    vf,
                    zS_tts,
                    maskK_tts,
                    steps=ode_steps_endloss,
                    direction=-1,
                    cfg_flag_value=cfg_bwd_diag,
                    spk_e=spk_e_bwd_diag,
                    style_e=style_e_bwd_diag,
                )

                r_fwd = batch_pair_ratios(zT_diag_source, zS_map, maskK_tts)
                r_bwd = batch_pair_ratios(zS_tts, zT_map, maskK_tts)
                if r_fwd is not None:
                    summarize_ratios("FWD (Tmean->S, CUT)", r_fwd)
                if r_bwd is not None:
                    summarize_ratios("BWD (S->Tmean, CUT)", r_bwd)

                f_amp = amp_once(
                    lambda xx: euler_integrate(vf, xx, maskK_tts, steps=ode_steps_endloss, direction=+1, cfg_flag_value=1, spk_e=spk_e_cond),
                    zT_diag_source,
                    maskK_tts,
                    sigma=amp_sigma,
                )
                b_amp = amp_once(
                    lambda xx: euler_integrate(
                        vf,
                        xx,
                        maskK_tts,
                        steps=ode_steps_endloss,
                        direction=-1,
                        cfg_flag_value=cfg_bwd_diag,
                        spk_e=spk_e_bwd_diag,
                        style_e=style_e_bwd_diag,
                    ),
                    zS_tts,
                    maskK_tts,
                    sigma=amp_sigma,
                )
                print(f"[AMP] fwd_amp={f_amp:.4g} bwd_amp={b_amp:.4g}")
        if perf_due:
            maybe_cuda_sync()
            perf_step["diag"] = time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        demo_decoder_ready = (svae_model is not None) if speech_backend == "svae" else (bigvgan_model is not None)
        if demo_every and demo_decoder_ready and (step % demo_every == 0):
            step_tag = f"step{step:05d}"
            try:
                idx_eval = int(np.random.randint(0, len(tts_demo_rows)))
                tts_demo_full_one_sample(step_tag, idx_eval)
            except Exception as exc:
                print(f"[WARN] TTS demo failed at {step_tag}: {repr(exc)}")
            try:
                idx_eval = int(np.random.randint(0, len(demo_aligned_rows)))
                asr_eval_full_one_sample(
                    step_tag,
                    idx_eval,
                    rows=demo_aligned_rows,
                    label="ASR-FULL-TEST",
                    source_name="demo",
                )
            except Exception as exc:
                print(f"[WARN] ASR full test eval failed at {step_tag}: {repr(exc)}")
            try:
                idx_eval = int(np.random.randint(0, len(asr_train_demo_rows)))
                asr_eval_full_one_sample(
                    step_tag,
                    idx_eval,
                    rows=asr_train_demo_rows,
                    label="ASR-FULL-TRAIN",
                    source_name="train",
                )
            except Exception as exc:
                print(f"[WARN] ASR full train eval failed at {step_tag}: {repr(exc)}")
        if perf_due:
            maybe_cuda_sync()
            perf_step["demo"] = time.perf_counter() - perf_stage_t0
            perf_stage_t0 = time.perf_counter()

        if save_every_steps and step > start_step and (step % save_every_steps == 0):
            config_snapshot = json.loads(json.dumps(cfg))
            extra_state = dict(
                mu_g=mu_g.detach().cpu().float(),
                std_g=std_g.detach().cpu().float(),
                tok_stoi=tok.stoi,
                tok_itos=tok.itos,
                tokenizer_type=tokenizer_type,
                tokenizer_model_path=getattr(tok, "model_path", None),
                spk_list=spk_list,
                spk2id=spk2id,
                n_spk=int(n_spk),
                alpha_K=float(alpha_K),
                D_mel=int(D_mel),
                Vt=int(Vt),
                UNK_ID=int(UNK_ID),
                BLANK_ID=int(BLANK_ID),
                PAD_ID=int(PAD_ID),
                AED_SOS_ID=int(AED_SOS_ID),
                AED_EOS_ID=int(AED_EOS_ID),
                AED_VOCAB_SIZE=int(AED_VOCAB_SIZE),
                source_ctc_optimizer=(source_ctc_opt.state_dict() if source_ctc_opt is not None else None),
            )
            ckpt_path, latest_path = save_training_checkpoint(
                ckpt_dir=ckpt_dir,
                tag=f"step{step:08d}",
                step=step,
                module_map=module_map,
                optimizer=opt,
                scaler=scaler,
                config_snapshot=config_snapshot,
                extra_state=extra_state,
                ema=ema,
                use_ema=use_ema,
                keep_last_k=keep_last_k,
                device=device,
            )
            print(f"[CKPT] saved {ckpt_path}")
            print(f"[CKPT] updated {latest_path}")
        if perf_due:
            maybe_cuda_sync()
            perf_step["save"] = time.perf_counter() - perf_stage_t0
            perf_step["total"] = time.perf_counter() - perf_step_t0
            print(
                f"[PERF-TOTAL] step {step:05d} "
                f"log {perf_step['log']:.3f}s diag {perf_step['diag']:.3f}s "
                f"demo {perf_step['demo']:.3f}s save {perf_step['save']:.3f}s "
                f"total {perf_step['total']:.3f}s"
            )

    final_step = max(last_step, start_step)
    config_snapshot = json.loads(json.dumps(cfg))
    extra_state = dict(
        mu_g=mu_g.detach().cpu().float(),
        std_g=std_g.detach().cpu().float(),
        tok_stoi=tok.stoi,
        tok_itos=tok.itos,
        tokenizer_type=tokenizer_type,
        tokenizer_model_path=getattr(tok, "model_path", None),
        spk_list=spk_list,
        spk2id=spk2id,
        n_spk=int(n_spk),
        alpha_K=float(alpha_K),
        D_mel=int(D_mel),
        Vt=int(Vt),
        UNK_ID=int(UNK_ID),
        BLANK_ID=int(BLANK_ID),
        PAD_ID=int(PAD_ID),
        AED_SOS_ID=int(AED_SOS_ID),
        AED_EOS_ID=int(AED_EOS_ID),
        AED_VOCAB_SIZE=int(AED_VOCAB_SIZE),
        source_ctc_optimizer=(source_ctc_opt.state_dict() if source_ctc_opt is not None else None),
    )
    ckpt_path, latest_path = save_training_checkpoint(
        ckpt_dir=ckpt_dir,
        tag="final",
        step=final_step,
        module_map=module_map,
        optimizer=opt,
        scaler=scaler,
        config_snapshot=config_snapshot,
        extra_state=extra_state,
        ema=ema,
        use_ema=use_ema,
        keep_last_k=keep_last_k,
        device=device,
    )
    print("\n[OK] Training done. Saved to:")
    print(" -", ckpt_path)
    print(" -", latest_path)


if __name__ == "__main__":
    main()
