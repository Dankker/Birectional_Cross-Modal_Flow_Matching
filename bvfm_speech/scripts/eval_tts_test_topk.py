#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
BIGVGAN_ROOT = REPO_ROOT / "BigVGAN"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BIGVGAN_ROOT) not in sys.path:
    sys.path.insert(0, str(BIGVGAN_ROOT))

from biflow.alignment import durations_to_int_and_fixsum
from biflow.checkpointing import load_module_map_state
from biflow.encoders import FrozenSpeechT5TextEncoder
from biflow.models import (
    BaselineCTCHead,
    CanonicalPosterior,
    CanonicalTextEncoder,
    CanonicalToSource,
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
    TTSStyleCanonicalPrior,
    TTSStylePosterior,
    TTSStylePrior,
    TTSStyleToSource,
    TrainableTokenTextEncoder,
    ZipformerCTCHead,
    heun_integrate,
)
from biflow.tokenizer import build_tokenizer
from biflow.utils import (
    extract_speaker_id_from_path,
    normalize_text_basic,
    read_jsonl_rows,
    save_wav,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate TTS from test text using train top-k speakers, then evaluate "
            "UTMOS and Whisper WER."
        )
    )
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default=str(REPO_ROOT / "ckpt_joint_svae_latent_ecapa"),
        help="Checkpoint directory containing merged_config.json and latest.pt/final.pt.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="latest.pt",
        help="Checkpoint file name under ckpt-dir, or an absolute path.",
    )
    p.add_argument("--test-manifest", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-test-rows", type=int, default=50)
    p.add_argument("--num-speakers", type=int, default=None)
    p.add_argument(
        "--pairing",
        choices=["round_robin", "cartesian"],
        default="round_robin",
        help=(
            "round_robin generates one speaker per test text from the train top-k list. "
            "cartesian generates every selected speaker for every selected test text."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--demo-cfg-scale", type=float, default=None)
    p.add_argument("--demo-prior-temp", type=float, default=None)
    p.add_argument("--demo-style-temp", type=float, default=None)
    p.add_argument("--ode-steps", type=int, default=None)
    p.add_argument("--force-resynthesize", action="store_true")
    p.add_argument("--skip-utmos", action="store_true")
    p.add_argument("--skip-whisper", action="store_true")
    p.add_argument("--whisper-model", type=str, default="medium.en")
    p.add_argument("--utmos-repo", type=str, default="tarepan/SpeechMOS:v1.2.0")
    p.add_argument("--utmos-model", type=str, default="utmos22_strong")
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print progress every N generated samples.",
    )
    return p.parse_args()


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


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "x"


def row_speaker(row):
    spk = row.get("speaker")
    if spk is not None:
        return str(spk)
    wav = row.get("wav", row.get("parent_wav", ""))
    return str(extract_speaker_id_from_path(wav))


def row_text(row, force_normalize=True):
    text = row.get("text_norm", row.get("text_raw", row.get("text", "")))
    text = str(text).strip()
    if force_normalize:
        text = normalize_text_basic(text)
    return text


def select_target_speakers(rows, target_spks, top_k_spk):
    spk_count = {}
    for row in rows:
        spk = str(row.get("speaker", ""))
        if spk:
            spk_count[spk] = spk_count.get(spk, 0) + 1
    if not spk_count:
        raise RuntimeError("No speaker field found in train rows.")
    if target_spks is not None:
        return [str(spk) for spk in target_spks]
    spk_sorted = sorted(spk_count.items(), key=lambda kv: kv[1], reverse=True)
    return [spk for spk, _ in spk_sorted[: int(top_k_spk)]]


def convert_unified_full_row(row):
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


def convert_unified_fm_row(row, full_by_utt):
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


def sha1_key(text):
    import hashlib

    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_english_text_for_wer(text):
    text = str(text).lower()
    text = re.sub(r"[-‐‑‒–—]+", " ", text)
    text = re.sub(r"[‘’‛`´ʼ]", "'", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_error_rate_notebook(ref_text, hyp_text):
    ref = normalize_english_text_for_wer(ref_text).split()
    hyp = normalize_english_text_for_wer(hyp_text).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dist[i][0] = i
    for j in range(len(hyp) + 1):
        dist[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )
    return float(dist[-1][-1]) / float(len(ref))


def tensor_cache_load(path):
    obj = torch_load_cpu(path)
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"]
    return obj


class LegacySpeakerTable(nn.Module):
    def __init__(self, n_spk: int, E: int, scale: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(n_spk, E)
        self.scale = float(scale)
        nn.init.zeros_(self.emb.weight)

    def forward(self, spk_id: torch.LongTensor):
        return self.emb(spk_id) * self.scale


def load_speaker_embedding_bank(path, spk_list, normalize=True, missing="error"):
    if not path:
        raise ValueError("model.speaker_emb_path is required for pretrained speaker conditioning")
    path = os.path.abspath(str(path))
    obj = torch_load_cpu(path)
    speaker_ids = None
    emb_obj = obj
    if isinstance(obj, dict):
        speaker_ids = obj.get("speaker_ids") or obj.get("spk_list") or obj.get("speakers")
        for key in ("embeddings", "speaker_embeddings", "spk2emb", "spk_embeddings"):
            if key in obj:
                emb_obj = obj[key]
                break

    if torch.is_tensor(emb_obj):
        if speaker_ids is None:
            raise ValueError(
                f"{path} stores a tensor speaker bank but has no speaker_ids/spk_list"
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

    if missing_spks and str(missing).lower() == "error":
        preview = ", ".join(missing_spks[:10])
        more = "" if len(missing_spks) <= 10 else f", ... (+{len(missing_spks) - 10})"
        raise ValueError(
            f"Missing pretrained speaker embeddings for {len(missing_spks)} selected speakers: "
            f"{preview}{more}"
        )

    bank = torch.stack(rows, dim=0).float()
    if bool(normalize):
        bank = F.normalize(bank, p=2, dim=-1)
    return bank, missing_spks, path


class TTSEvaluator:
    def __init__(self, cfg, ckpt_path, out_dir, device):
        self.cfg = cfg
        self.ckpt_path = ckpt_path
        self.out_dir = Path(out_dir)
        self.device = device
        self.paths_cfg = cfg["paths"]
        self.cache_cfg = cfg["cache"]
        self.data_cfg = cfg["data"]
        self.model_cfg = cfg["model"]
        self.tokenizer_cfg = cfg.get("tokenizer", {"type": "char"})
        self.loss_cfg = cfg["loss"]
        self.infer_cfg = cfg["infer"]
        self.runtime_cfg = cfg.get("runtime", {})

        self.force_text_normalize = bool(self.data_cfg.get("force_text_normalize", True))
        self.speech_backend = str(self.cache_cfg.get("speech_backend", "bigvgan")).lower()
        if self.speech_backend != "svae":
            raise NotImplementedError(
                "This evaluation script currently supports speech_backend='svae', "
                f"but config has {self.speech_backend!r}."
            )

        self.sampling_rate = int(self.cache_cfg.get("svae_sample_rate", 16000))
        self.hop_size = int(self.cache_cfg.get("svae_hop_size", 400))
        self.D_mel = int(self.cache_cfg.get("svae_dim", 64))
        self.use_ctc_blank_repeat_prior = (
            str(self.loss_cfg.get("alignment_prior_mode", "mas")).lower() == "ctc_blank_repeat"
        )
        self.ckpt_obj = None
        self.ckpt_state = None
        self.ctc_blank_skip_speecht5 = bool(self.model_cfg.get("ctc_blank_skip_speecht5", False))
        self.text_encoder_type = str(self.model_cfg.get("text_encoder_type", "speecht5")).lower()
        self.ctc_blank_use_speecht5 = (
            self.text_encoder_type == "speecht5"
            and self.use_ctc_blank_repeat_prior
            and not self.ctc_blank_skip_speecht5
        )
        self.gpu_text_cache = False
        self.text_hidden_cache = OrderedDict()
        self.speech_path_by_wav = {}
        self._read_checkpoint()
        self._load_data_and_vocab()
        self._load_svae_decoder()
        self._build_text_encoder()
        self._build_modules()
        self._load_checkpoint()
        self._compute_stats()
        self._set_eval()

    def canonicalize_text(self, text):
        text = str(text).strip()
        if self.force_text_normalize:
            text = normalize_text_basic(text)
        return text

    def _load_data_and_vocab(self):
        processed_unified_dir = self.paths_cfg.get("processed_unified_dir")
        full_manifest_clean = (
            os.path.join(processed_unified_dir, "full_manifest_clean.jsonl")
            if processed_unified_dir
            else None
        )
        fm_core_context_manifest = (
            os.path.join(processed_unified_dir, "fm_core_context_cuts.jsonl")
            if processed_unified_dir
            else None
        )
        use_processed_unified = bool(self.paths_cfg.get("use_processed_unified", False))
        if not (
            use_processed_unified
            and full_manifest_clean
            and os.path.isfile(full_manifest_clean)
        ):
            extra = self.ckpt_obj.get("extra_state", {})
            self.spk_list = [str(value) for value in extra.get("spk_list", [])]
            if not self.spk_list:
                raise FileNotFoundError(
                    "Training manifests are unavailable and checkpoint extra_state "
                    "does not contain spk_list metadata"
                )
            self.spk2id = {
                str(key): int(value)
                for key, value in extra.get("spk2id", {}).items()
            } or {spk: index for index, spk in enumerate(self.spk_list)}
            self.n_spk = int(extra.get("n_spk", len(self.spk_list)))
            self.target_spks = list(self.spk_list)
            self.cut_rows = []
            self.tok = build_tokenizer(self.tokenizer_cfg, texts=[])
            if extra.get("tok_stoi") and extra.get("tok_itos"):
                self.tok.stoi = dict(extra["tok_stoi"])
                self.tok.itos = list(extra["tok_itos"])
            self.Vt = len(self.tok.itos)
            self.PAD_ID = int(extra.get("PAD_ID", self.tok.stoi["<pad>"]))
            self.BLANK_ID = int(extra.get("BLANK_ID", self.tok.stoi["<blank>"]))
            self.UNK_ID = int(extra.get("UNK_ID", self.tok.stoi["<unk>"]))
            print(
                f"[DATA] using checkpoint metadata (no training manifest) "
                f"n_spk={self.n_spk} tokenizer={self.tokenizer_cfg.get('type', 'char')} "
                f"vocab={self.Vt}"
            )
            return
        full_rows_all = read_jsonl_rows(full_manifest_clean, max_rows=None)
        if not full_rows_all:
            raise RuntimeError(f"Empty train full manifest: {full_manifest_clean}")
        full_row_by_utt = {str(row["utt_id"]): row for row in full_rows_all}
        train_unit = str(self.data_cfg.get("train_unit", "cut")).lower()
        use_utterance_training = train_unit in {"utterance", "full", "full_utterance"}
        if use_utterance_training:
            cut_rows_all = [convert_unified_full_row(row) for row in full_rows_all]
        else:
            fm_rows = read_jsonl_rows(fm_core_context_manifest, max_rows=self.paths_cfg.get("max_cut_rows"))
            cut_rows_all = [convert_unified_fm_row(row, full_row_by_utt) for row in fm_rows]

        target_spks = select_target_speakers(
            cut_rows_all,
            self.data_cfg.get("target_spks"),
            int(self.data_cfg.get("top_k_spk", 1000)),
        )
        target_spk_set = set(target_spks)
        self.cut_rows = [
            row for row in cut_rows_all
            if str(row.get("speaker", "")) in target_spk_set
        ]
        if not self.cut_rows:
            raise RuntimeError("No train rows left after top-k speaker filtering.")
        self.target_spks = target_spks
        self.spk_list = sorted({str(row["speaker"]) for row in self.cut_rows})
        self.spk2id = {spk: idx for idx, spk in enumerate(self.spk_list)}
        self.n_spk = len(self.spk_list)

        aligned_rows = [row for row in full_rows_all if row_speaker(row) in target_spk_set]
        for row in list(full_rows_all) + self.cut_rows + aligned_rows:
            wav_key = row.get("parent_wav", row.get("wav"))
            latent_path = row.get("svae_latent_path") or row.get("speech_path") or row.get("latent_path")
            if wav_key and latent_path:
                self.speech_path_by_wav[os.path.abspath(str(wav_key))] = os.path.abspath(str(latent_path))

        texts_all = sorted({
            self.canonicalize_text(row.get("text_norm", row.get("text", "")))
            for row in self.cut_rows
            if self.canonicalize_text(row.get("text_norm", row.get("text", "")))
        })
        texts_all.extend(
            self.canonicalize_text(row.get("text_norm", row.get("text_raw", "")))
            for row in aligned_rows
            if self.canonicalize_text(row.get("text_norm", row.get("text_raw", "")))
        )
        texts_all = list(OrderedDict.fromkeys(texts_all))
        self.tok = build_tokenizer(self.tokenizer_cfg, texts=texts_all)
        self.Vt = len(self.tok.itos)
        self.PAD_ID = self.tok.stoi["<pad>"]
        self.BLANK_ID = self.tok.stoi["<blank>"]
        self.UNK_ID = self.tok.stoi["<unk>"]
        print(
            f"[DATA] train_rows={len(self.cut_rows)} train_topk={len(target_spks)} "
            f"n_spk={self.n_spk} tokenizer={self.tokenizer_cfg.get('type', 'char')} vocab={self.Vt}"
        )

    def _load_svae_decoder(self):
        configured_root = self.cache_cfg.get("semantic_vae_root")
        bundled_root = REPO_ROOT / "Semantic-VAE"
        semantic_root = os.environ.get("SEMANTIC_VAE_ROOT")
        if not semantic_root:
            semantic_root = (
                configured_root
                if configured_root and os.path.isdir(configured_root)
                else str(bundled_root)
            )
        semantic_root = os.path.abspath(semantic_root)

        weights_root = os.environ.get("BVFM_WEIGHTS_ROOT")
        checkpoint_candidates = [
            os.environ.get("SEMANTIC_VAE_CKPT"),
            (
                os.path.join(weights_root, "speech", "semantic_vae_1000k")
                if weights_root
                else None
            ),
            self.cache_cfg.get("semantic_vae_ckpt"),
            os.path.join(semantic_root, "ckpts", "semantic_vae_1000k"),
        ]
        svae_ckpt = next(
            (
                os.path.abspath(path)
                for path in checkpoint_candidates
                if path and os.path.isdir(path)
            ),
            None,
        )
        if svae_ckpt is None:
            raise FileNotFoundError(
                "Semantic-VAE checkpoint not found; set BVFM_WEIGHTS_ROOT or "
                "SEMANTIC_VAE_CKPT"
            )
        if semantic_root not in sys.path:
            sys.path.insert(0, semantic_root)
        from dac.model.dac import DAC as SemanticVAEDAC
        from dac.model.utils import read_json_file as svae_read_json_file

        metainfo = svae_read_json_file(os.path.join(svae_ckpt, "metainfo.json"))
        bigvgan_conf = metainfo["DAC"].get("bigvgan_conf")
        if bigvgan_conf and not os.path.isabs(str(bigvgan_conf)):
            metainfo["DAC"]["bigvgan_conf"] = os.path.join(semantic_root, str(bigvgan_conf))
        use_ema = bool(self.cache_cfg.get("semantic_vae_use_ema", True))
        ckpt_name = "ema_state_dict.pth" if use_ema else "weights.pth"
        ckpt_obj = torch_load_cpu(os.path.join(svae_ckpt, "dac", ckpt_name))
        if use_ema:
            ckpt_obj = {k.replace("ema_model.", ""): v for k, v in ckpt_obj.items()}
        else:
            ckpt_obj = ckpt_obj["state_dict"]
        ckpt_obj = {k: v for k, v in ckpt_obj.items() if not k.startswith("projectors")}
        self.svae_model = SemanticVAEDAC(**metainfo["DAC"])
        if hasattr(self.svae_model, "projectors"):
            del self.svae_model.projectors
        self.svae_model.load_state_dict(ckpt_obj, strict=False)
        self.svae_model = self.svae_model.eval().to(self.device)
        for p in self.svae_model.parameters():
            p.requires_grad_(False)
        print(f"[SVAE] loaded decoder {svae_ckpt} use_ema={use_ema}")

    def _read_checkpoint(self):
        self.ckpt_obj = torch_load_cpu(self.ckpt_path)
        self.ckpt_state = self.ckpt_obj.get("inference_modules") or self.ckpt_obj.get("modules")
        if self.ckpt_state is None:
            raise RuntimeError(f"Checkpoint has no modules: {self.ckpt_path}")

    def _checkpoint_spk_table_has_ln(self):
        if not isinstance(self.ckpt_state, dict):
            return True
        spk_state = self.ckpt_state.get("spk_table")
        if not isinstance(spk_state, dict):
            return True
        return any(str(key).startswith("ln.") for key in spk_state.keys())

    def _speaker_bank_from_checkpoint(self):
        if not isinstance(self.ckpt_state, dict):
            return None
        spk_state = self.ckpt_state.get("spk_table")
        if not isinstance(spk_state, dict):
            return None
        bank = spk_state.get("pretrained_emb")
        if bank is None or not torch.is_tensor(bank):
            return None
        if int(bank.shape[0]) != int(self.n_spk):
            return None
        return bank.detach().cpu().float()

    def _build_text_encoder(self):
        if self.text_encoder_type in {"trainable", "trainable_token"}:
            self.H_text = int(self.model_cfg.get("text_encoder_dim", 384))
            self.st5 = None
        elif self.use_ctc_blank_repeat_prior and self.ctc_blank_skip_speecht5:
            self.H_text = int(self.model_cfg.get("ctc_blank_text_dim", 768))
            self.st5 = None
        else:
            self.st5 = FrozenSpeechT5TextEncoder(
                model_name="microsoft/speecht5_tts",
                device=self.device,
                layer_idx=int(self.model_cfg["st5_layer_idx"]),
            )
            self.H_text = self.st5.hidden_size

    def _build_modules(self):
        mc = self.model_cfg
        lc = self.loss_cfg
        E_spk = int(mc["E_spk"])
        self.E_spk = E_spk
        use_adapter = bool(mc["use_adapter"])
        adapter_type = str(mc.get("adapter_type", "residual")).lower()
        adapter_dropout = float(mc.get("adapter_dropout", 0.1))
        adapter_bottleneck = int(mc.get("adapter_bottleneck", 192))
        canonical_text_layers = int(mc.get("canonical_text_layers", 4))
        canonical_text_heads = int(mc.get("canonical_text_heads", 8))
        canonical_text_ff_mult = int(mc.get("canonical_text_ff_mult", 4))
        canonical_text_conv_ksize = int(mc.get("canonical_text_conv_ksize", 5))
        canonical_text_residual_scale = float(mc.get("canonical_text_residual_scale", 1.0))

        self.trainable_text_encoder = None
        if self.text_encoder_type in {"trainable", "trainable_token"}:
            self.trainable_text_encoder = TrainableTokenTextEncoder(
                vocab_size=self.Vt,
                dim=int(mc.get("text_encoder_dim", 384)),
                layers=int(mc.get("text_encoder_layers", 6)),
                n_heads=int(mc.get("text_encoder_heads", 6)),
                ff_mult=int(mc.get("text_encoder_ff_mult", 4)),
                conv_ksize=int(mc.get("text_encoder_conv_ksize", 5)),
                dropout=float(mc.get("text_encoder_dropout", 0.1)),
                max_len=int(mc.get("text_encoder_max_len", 1024)),
                padding_idx=self.PAD_ID,
            ).to(self.device)

        self.adapter = None
        if (
            self.text_encoder_type == "speecht5"
            and use_adapter
            and (not self.use_ctc_blank_repeat_prior or self.ctc_blank_use_speecht5)
        ):
            if adapter_type in {"residual", "residual_adapter"}:
                self.adapter = ResidualAdapter(
                    self.H_text,
                    bottleneck=adapter_bottleneck,
                    dropout=adapter_dropout,
                ).to(self.device)
            elif adapter_type in {"canonical_text_encoder", "canonical"}:
                self.adapter = CanonicalTextEncoder(
                    self.H_text,
                    layers=canonical_text_layers,
                    n_heads=canonical_text_heads,
                    ff_mult=canonical_text_ff_mult,
                    conv_ksize=canonical_text_conv_ksize,
                    dropout=adapter_dropout,
                    residual_scale=canonical_text_residual_scale,
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported adapter_type={adapter_type!r}")

        self.ctc_blank_embed = None
        self.ctc_blank_encoder = None
        if (
            self.text_encoder_type == "speecht5"
            and self.use_ctc_blank_repeat_prior
            and not self.ctc_blank_use_speecht5
        ):
            self.ctc_blank_embed = nn.Embedding(self.Vt, self.H_text, padding_idx=0).to(self.device)
            self.ctc_blank_encoder = CanonicalTextEncoder(
                self.H_text,
                layers=int(mc.get("ctc_blank_text_layers", canonical_text_layers)),
                n_heads=int(mc.get("ctc_blank_text_heads", canonical_text_heads)),
                ff_mult=int(mc.get("ctc_blank_text_ff_mult", canonical_text_ff_mult)),
                conv_ksize=int(mc.get("ctc_blank_text_conv_ksize", canonical_text_conv_ksize)),
                dropout=float(mc.get("ctc_blank_text_dropout", adapter_dropout)),
                residual_scale=float(mc.get("ctc_blank_text_residual_scale", canonical_text_residual_scale)),
            ).to(self.device)

        self.text_prior = TextPriorHead(
            in_dim=self.H_text,
            hidden=256,
            out_dim=self.D_mel,
            logvar_bias=-2.0,
        ).to(self.device)

        self.use_true_canonical_latent = bool(mc.get("use_true_canonical_latent", False))
        self.use_vf_canonical_text_cond = bool(mc.get("use_vf_canonical_text_cond", True))
        self.canonical_prior = None
        self.canonical_to_source = None
        self.source_to_canonical = None
        self.canonical_posterior = None
        ctc_input_dim = self.D_mel
        vf_text_cond_dim = 0
        canonical_dim = int(mc.get("canonical_dim", 192))
        canonical_hidden = int(mc.get("canonical_hidden", 256))
        canonical_dropout = float(mc.get("canonical_dropout", 0.1))
        if self.use_true_canonical_latent:
            self.canonical_prior = TextPriorHead(
                in_dim=self.H_text,
                hidden=canonical_hidden,
                out_dim=canonical_dim,
                logvar_bias=-2.0,
            ).to(self.device)
            self.canonical_to_source = CanonicalToSource(
                c_dim=canonical_dim,
                spk_dim=E_spk,
                style_dim=0,
                out_dim=self.D_mel,
                hidden=canonical_hidden,
                dropout=canonical_dropout,
            ).to(self.device)
            self.source_to_canonical = SourceToCanonical(
                in_dim=self.D_mel,
                c_dim=canonical_dim,
                hidden=canonical_hidden,
                dropout=canonical_dropout,
            ).to(self.device)
            ctc_input_dim = canonical_dim
            vf_text_cond_dim = canonical_dim if self.use_vf_canonical_text_cond else 0

        if str(lc.get("canonical_match_mode", "nll")).lower() == "kl":
            self.canonical_posterior = CanonicalPosterior(
                dim=ctc_input_dim,
                hidden=int(mc.get("canonical_post_hidden", canonical_hidden)),
                dropout=float(mc.get("canonical_post_dropout", canonical_dropout)),
                logvar_bias=float(mc.get("canonical_post_logvar_bias", -4.0)),
            ).to(self.device)

        self.dur_pred = FastSpeech2DurationPredictor(
            D=self.H_text,
            hidden=int(mc["dur_hidden"]),
            ksize=3,
            dropout=float(mc["dur_dropout"]),
        ).to(self.device)
        self.len_pred = (
            LengthPredictor(D=self.H_text, hidden=int(mc.get("len_hidden", 192))).to(self.device)
            if bool(mc.get("use_len_predictor", False))
            else None
        )
        speaker_cond_type = str(mc.get("speaker_cond_type", "table")).lower()
        if speaker_cond_type not in SpeakerConditioner.TABLE_MODES:
            speaker_bank = self._speaker_bank_from_checkpoint()
            bank_source = "checkpoint"
            missing_spks = []
            if speaker_bank is None:
                speaker_bank, missing_spks, bank_source = load_speaker_embedding_bank(
                    mc.get("speaker_emb_path"),
                    self.spk_list,
                    normalize=bool(mc.get("speaker_emb_l2_normalize", True)),
                    missing=str(mc.get("speaker_emb_missing", "error")),
                )
            self.spk_table = SpeakerConditioner(
                n_spk=self.n_spk,
                E=E_spk,
                scale=float(mc["spk_scale"]),
                mode=speaker_cond_type,
                pretrained_emb=speaker_bank,
                pretrained_trainable=bool(mc.get("speaker_emb_trainable", False)),
                delta_scale=float(mc.get("speaker_delta_scale", 0.1)),
                use_layernorm=bool(mc.get("speaker_cond_layernorm", True)),
            ).to(self.device)
            print(
                f"[SPEAKER] mode={speaker_cond_type} bank={bank_source} "
                f"shape={tuple(speaker_bank.shape)} missing={len(missing_spks)}"
            )
        elif self._checkpoint_spk_table_has_ln():
            self.spk_table = SpeakerTable(
                n_spk=self.n_spk,
                E=E_spk,
                scale=float(mc["spk_scale"]),
            ).to(self.device)
            print("[SPEAKER] table=current_layernorm")
        else:
            self.spk_table = LegacySpeakerTable(
                n_spk=self.n_spk,
                E=E_spk,
                scale=float(mc["spk_scale"]),
            ).to(self.device)
            print("[SPEAKER] table=legacy_no_layernorm")
        self.tts_source_cond = (
            SourceStatsConditioner(
                D=self.D_mel,
                hidden=int(mc.get("tts_source_cond_hidden", 128)),
                out_dim=E_spk,
            ).to(self.device)
            if bool(mc.get("use_tts_source_cond", False))
            else None
        )
        self.tts_source_cond_scale = float(mc.get("tts_source_cond_scale", 1.0))

        self.use_tts_style_latent = bool(mc.get("use_tts_style_latent", False))
        self.tts_style_dim = int(mc.get("tts_style_dim", 64))
        self.tts_style_into_source = bool(mc.get("tts_style_into_source", False))
        self.tts_style_source_scale = float(mc.get("tts_style_source_scale", 1.0))
        self.vf_use_speaker_cond = bool(mc.get("vf_use_speaker_cond", True))
        asr_vf_use_speaker_cond_cfg = mc.get("asr_vf_use_speaker_cond", None)
        self.asr_vf_use_speaker_cond = (
            self.vf_use_speaker_cond
            if asr_vf_use_speaker_cond_cfg is None
            else bool(asr_vf_use_speaker_cond_cfg)
        )
        self.tts_style_prior = None
        self.tts_style_post = None
        self.tts_style_to_source = None
        if self.use_tts_style_latent:
            self.tts_style_post = TTSStylePosterior(
                D=self.D_mel,
                spk_dim=E_spk,
                latent_dim=self.tts_style_dim,
                hidden=int(mc.get("tts_style_hidden", 256)),
                dropout=float(mc.get("tts_style_dropout", 0.1)),
                mode=str(mc.get("tts_style_post_mode", "speech")).lower(),
            ).to(self.device)
            self.tts_style_prior_type = str(mc.get("tts_style_prior_type", "standard_normal")).lower()
            if self.tts_style_prior_type == "speaker":
                self.tts_style_prior = TTSStylePrior(
                    spk_dim=E_spk,
                    latent_dim=self.tts_style_dim,
                    hidden=int(mc.get("tts_style_hidden", 256)),
                    dropout=float(mc.get("tts_style_dropout", 0.1)),
                ).to(self.device)
            elif self.tts_style_prior_type == "canonical_speaker":
                self.tts_style_prior = TTSStyleCanonicalPrior(
                    c_dim=ctc_input_dim,
                    spk_dim=E_spk,
                    latent_dim=self.tts_style_dim,
                    hidden=int(mc.get("tts_style_hidden", 256)),
                    dropout=float(mc.get("tts_style_dropout", 0.1)),
                    logvar_bias=float(mc.get("tts_style_prior_logvar_bias", 0.0)),
                ).to(self.device)
            if self.tts_style_into_source and abs(self.tts_style_source_scale) > 0.0:
                self.tts_style_to_source = TTSStyleToSource(
                    latent_dim=self.tts_style_dim,
                    spk_dim=E_spk,
                    out_dim=self.D_mel,
                    hidden=int(mc.get("tts_style_hidden", 256)),
                ).to(self.device)

        self.vf = DiTVectorField(
            D=self.D_mel,
            E_spk=E_spk,
            style_dim=self.tts_style_dim if self.use_tts_style_latent else 0,
            text_cond_dim=vf_text_cond_dim,
            hidden=int(mc["vf_hidden"]),
            depth=int(mc["vf_depth"]),
            n_heads=int(mc["vf_heads"]),
            dropout=float(mc["vf_dropout"]),
            max_len=int(mc["vf_max_len"]),
            condition_injection=str(mc.get("vf_condition_injection", "legacy")).lower(),
        ).to(self.device)
        self.vf.direct_speaker_cond = bool(self.vf_use_speaker_cond or self.asr_vf_use_speaker_cond)

        self.mel_refiner = (
            TextCondRefiner1xResidualPostNet(
                D=self.D_mel,
                hidden=int(mc["ref_hidden"]),
                cond_dim=self.H_text,
                n_blocks=int(mc["ref_blocks"]),
                ksize=int(mc["ref_ksize"]),
                dropout=float(mc["ref_dropout"]),
            ).to(self.device)
            if bool(mc.get("use_refiner", False))
            else None
        )

        ctc_head_type = str(mc.get("ctc_head_type", "baseline")).lower()
        if ctc_head_type == "framectcconvhead":
            ctc_head_type = "conv"
        if ctc_head_type == "baseline":
            self.text_ctc_head = BaselineCTCHead(
                V=self.Vt,
                D=ctc_input_dim,
                hidden=int(mc["ctc_hidden"]),
                conv_layers=int(mc["ctc_layers"]),
                ksize=int(mc["ctc_ksize"]),
                lstm_hidden=int(mc.get("ctc_lstm_hidden", 384)),
                lstm_layers=int(mc.get("ctc_lstm_layers", 2)),
                dropout=float(mc["ctc_dropout"]),
            ).to(self.device)
        elif ctc_head_type == "conv":
            self.text_ctc_head = FrameCTCConvHead(
                V=self.Vt,
                D=ctc_input_dim,
                hidden=int(mc["ctc_hidden"]),
                layers=int(mc["ctc_layers"]),
                ksize=int(mc["ctc_ksize"]),
            ).to(self.device)
        elif ctc_head_type == "zipformer":
            self.text_ctc_head = ZipformerCTCHead(
                V=self.Vt,
                D=ctc_input_dim,
                hidden=int(mc["ctc_hidden"]),
                layers=int(mc["ctc_layers"]),
                heads=int(mc.get("ctc_heads", 8)),
                ff_mult=float(mc.get("ctc_ff_mult", 4.0)),
                ksize=int(mc["ctc_ksize"]),
                dropout=float(mc["ctc_dropout"]),
                downsample_factor=int(mc.get("ctc_zipformer_downsample", 2)),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported ctc_head_type={ctc_head_type!r}")

        self.module_map = OrderedDict(
            adapter=self.adapter,
            trainable_text_encoder=self.trainable_text_encoder,
            ctc_blank_embed=self.ctc_blank_embed,
            ctc_blank_encoder=self.ctc_blank_encoder,
            text_prior=self.text_prior,
            canonical_prior=self.canonical_prior,
            canonical_to_source=self.canonical_to_source,
            source_to_canonical=self.source_to_canonical,
            canonical_posterior=self.canonical_posterior,
            dur_pred=self.dur_pred,
            len_pred=self.len_pred,
            spk_table=self.spk_table,
            tts_source_cond=self.tts_source_cond,
            tts_style_post=self.tts_style_post,
            tts_style_prior=self.tts_style_prior,
            tts_style_to_source=self.tts_style_to_source,
            vf=self.vf,
            text_ctc_head=self.text_ctc_head,
            mel_refiner=self.mel_refiner,
            ssl_hidden_head=None,
        )

    def _load_checkpoint(self):
        if self.ckpt_obj is None or self.ckpt_state is None:
            self._read_checkpoint()
        load_module_map_state(self.module_map, self.ckpt_state)
        self.step = int(self.ckpt_obj.get("step", -1))
        print(f"[CKPT] loaded {self.ckpt_path} step={self.step}")

    def _set_eval(self):
        for module in self.module_map.values():
            if module is not None:
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)

    def _mel_cache_path(self, wav_path):
        mapped = self.speech_path_by_wav.get(os.path.abspath(str(wav_path)))
        if mapped:
            return mapped
        mel_dir = self.cache_cfg.get("svae_latent_dir") or self.cache_cfg.get("mel_dir")
        if not mel_dir:
            return None
        return os.path.join(mel_dir, f"{sha1_key(os.path.abspath(wav_path))}.npy")

    def load_svae_latent(self, wav_path):
        cache_path = self._mel_cache_path(wav_path)
        if not cache_path or not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing SVAE latent for wav={wav_path}; expected {cache_path}")
        arr = np.load(cache_path)
        mel = torch.from_numpy(arr).float().contiguous()
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel[0].transpose(0, 1).contiguous()
        if mel.ndim != 2:
            raise RuntimeError(f"Expected [K,D] latent, got {tuple(mel.shape)} from {cache_path}")
        return mel

    def _compute_stats(self):
        extra = self.ckpt_obj.get("extra_state", {})
        checkpoint_mean = extra.get("mu_g")
        checkpoint_std = extra.get("std_g")
        if checkpoint_mean is not None and checkpoint_std is not None:
            mean = torch.as_tensor(checkpoint_mean).detach().float().reshape(-1)
            std = torch.as_tensor(checkpoint_std).detach().float().reshape(-1)
            if mean.numel() == 1:
                mean = mean.expand(self.D_mel).clone()
            if std.numel() == 1:
                std = std.expand(self.D_mel).clone()
            if mean.numel() != self.D_mel or std.numel() != self.D_mel:
                raise RuntimeError(
                    f"Checkpoint stats shape mismatch: mean={tuple(mean.shape)} "
                    f"std={tuple(std.shape)} D={self.D_mel}"
                )
            self.mu_b = mean.to(self.device).view(1, 1, self.D_mel)
            self.std_b = std.clamp_min(1e-8).to(self.device).view(1, 1, self.D_mel)
            print(f"[STATS] loaded from checkpoint D={self.D_mel}")
            return
        mode = str(self.cache_cfg.get("mel_stats_mode", "per_bin")).lower()
        max_unique = self.cache_cfg.get("stats_max_unique_wavs")
        unique_wavs = list(OrderedDict.fromkeys([row["parent_wav"] for row in self.cut_rows]).keys())
        if max_unique is not None and len(unique_wavs) > int(max_unique):
            random.shuffle(unique_wavs)
            unique_wavs = unique_wavs[: int(max_unique)]
        count = 0
        mean = np.zeros((self.D_mel,), dtype=np.float64)
        M2 = np.zeros((self.D_mel,), dtype=np.float64)
        scalar_count = 0
        scalar_sum = 0.0
        scalar_sumsq = 0.0
        for idx, wav_path in enumerate(unique_wavs, start=1):
            x = self.load_svae_latent(wav_path).numpy().astype(np.float64)
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
                total = count + b_n
                mean = mean + delta * (b_n / total)
                M2 = M2 + b_var * b_n + (delta * delta) * (count * b_n / total)
                count = total
            if idx % 1000 == 0:
                print(f"[STATS] {idx}/{len(unique_wavs)} wavs")
        if mode == "scalar":
            scalar_mean = scalar_sum / max(scalar_count, 1)
            scalar_var = scalar_sumsq / max(scalar_count, 1) - scalar_mean * scalar_mean
            scalar_std = max(float(np.sqrt(max(scalar_var, 0.0) + 1e-8)), 0.2)
            mean = np.full((self.D_mel,), scalar_mean, dtype=np.float64)
            std = np.full((self.D_mel,), scalar_std, dtype=np.float64)
        else:
            var = M2 / max(count, 1)
            std = np.maximum(np.sqrt(var + 1e-8), 0.2)
        self.mu_b = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 1, self.D_mel)
        self.std_b = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 1, self.D_mel)
        print(f"[STATS] ready mode={mode} frames={count}")

    def build_token_id_batch(self, texts, add_blank=False):
        seqs = []
        for text in texts:
            ids = self.tok.encode(self.canonicalize_text(text))
            if not ids:
                ids = [self.UNK_ID]
            if add_blank:
                ext = [self.BLANK_ID]
                for token_id in ids:
                    ext.append(int(token_id))
                    ext.append(self.BLANK_ID)
                ids = ext
            seqs.append([int(token_id) for token_id in ids])
        Lmax = max(max(len(seq) for seq in seqs), 1)
        ids_pad = torch.full(
            (len(seqs), Lmax),
            fill_value=self.PAD_ID,
            device=self.device,
            dtype=torch.long,
        )
        mask = torch.zeros(len(seqs), Lmax, device=self.device, dtype=torch.bool)
        for b, seq in enumerate(seqs):
            ids_pad[b, : len(seq)] = torch.tensor(seq, device=self.device, dtype=torch.long)
            mask[b, : len(seq)] = True
        return ids_pad, mask

    def build_ctc_blank_id_batch(self, texts):
        return self.build_token_id_batch(texts, add_blank=True)

    def resample_text_hidden_to_ctc_topology(self, hidden, src_mask, dst_mask):
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

    def _speecht5_cache_path(self, text):
        cache_dir = self.cache_cfg.get("speecht5_dir")
        if not cache_dir:
            return None
        return os.path.join(cache_dir, f"{sha1_key(text)}.pt")

    def load_text_hidden_cached(self, text):
        if self.st5 is None:
            raise RuntimeError("SpeechT5 cache requested while using a trainable token text encoder")
        if text in self.text_hidden_cache:
            self.text_hidden_cache.move_to_end(text)
            return self.text_hidden_cache[text]
        path = self._speecht5_cache_path(text)
        obj = None
        if path and os.path.exists(path):
            loaded = tensor_cache_load(path)
            obj = {
                "text": text,
                "hidden": loaded["hidden"].float().contiguous(),
                "mask": loaded["mask"].to(dtype=torch.bool).contiguous(),
            }
        if obj is None:
            with torch.no_grad():
                hidden, mask = self.st5([text])
            L = int(mask[0].long().sum().item())
            obj = {
                "text": text,
                "hidden": hidden[0, :L].detach().cpu().float().contiguous(),
                "mask": torch.ones(L, dtype=torch.bool),
            }
        self.text_hidden_cache[text] = obj
        self.text_hidden_cache.move_to_end(text)
        max_items = int(self.cache_cfg.get("text_hidden_cache_max_items", 2048))
        while len(self.text_hidden_cache) > max_items:
            self.text_hidden_cache.popitem(last=False)
        return obj

    @torch.no_grad()
    def encode_text_batch(self, texts):
        texts = [self.canonicalize_text(text) for text in texts]
        if self.trainable_text_encoder is not None:
            ids_pad, mask = self.build_token_id_batch(
                texts,
                add_blank=self.use_ctc_blank_repeat_prior,
            )
            h_enc = self.trainable_text_encoder(ids_pad, mask)
            align_mu_tok, align_logvar_tok = self.text_prior(h_enc)
            if self.canonical_prior is not None:
                mu_tok, logvar_tok = self.canonical_prior(h_enc)
            else:
                mu_tok, logvar_tok = align_mu_tok, align_logvar_tok
            return h_enc, mask, mu_tok, logvar_tok, align_mu_tok, align_logvar_tok

        if self.use_ctc_blank_repeat_prior and not self.ctc_blank_use_speecht5:
            ids_pad, mask = self.build_ctc_blank_id_batch(texts)
            hidden = self.ctc_blank_embed(ids_pad)
            h_enc = self.ctc_blank_encoder(hidden, mask)
            align_mu_tok, align_logvar_tok = self.text_prior(h_enc)
            if self.canonical_prior is not None:
                mu_tok, logvar_tok = self.canonical_prior(h_enc)
            else:
                mu_tok, logvar_tok = align_mu_tok, align_logvar_tok
            return h_enc, mask, mu_tok, logvar_tok, align_mu_tok, align_logvar_tok

        objs = [self.load_text_hidden_cached(text) for text in texts]
        lengths = [int(obj["mask"].long().sum().item()) for obj in objs]
        Lmax = max(lengths) if lengths else 1
        hidden = torch.zeros(len(objs), Lmax, self.H_text, device=self.device, dtype=torch.float32)
        mask = torch.zeros(len(objs), Lmax, device=self.device, dtype=torch.bool)
        for b, obj in enumerate(objs):
            L = lengths[b]
            hidden[b, :L] = obj["hidden"][:L].to(device=self.device, dtype=torch.float32)
            mask[b, :L] = obj["mask"][:L].to(device=self.device, dtype=torch.bool)

        h_enc = hidden
        if self.adapter is not None:
            h_enc = self.adapter(h_enc, mask)
        if self.use_ctc_blank_repeat_prior:
            _, ctc_blank_mask = self.build_ctc_blank_id_batch(texts)
            h_enc, mask = self.resample_text_hidden_to_ctc_topology(h_enc, mask, ctc_blank_mask)
        align_mu_tok, align_logvar_tok = self.text_prior(h_enc)
        if self.canonical_prior is not None:
            mu_tok, logvar_tok = self.canonical_prior(h_enc)
        else:
            mu_tok, logvar_tok = align_mu_tok, align_logvar_tok
        return h_enc, mask, mu_tok, logvar_tok, align_mu_tok, align_logvar_tok

    @torch.no_grad()
    def synthesize(self, text, spk, wav_path, cfg_scale, prior_temp, style_temp, ode_steps):
        spk_id = torch.tensor([self.spk2id[spk]], device=self.device, dtype=torch.long)
        spk_e = self.spk_table(spk_id)

        h_enc, maskL, mu_tok, logvar_tok, _, _ = self.encode_text_batch([text])
        L_valid = int(maskL.sum().item())
        limit = int(getattr(self.vf.rope, "max_seq_len", 4096))
        if self.len_pred is not None:
            k_raw = int(max(16, round(float(self.len_pred(h_enc, maskL).item()))))
            k_pred = min(max(k_raw, L_valid), limit)
        else:
            log_dur = self.dur_pred(h_enc, maskL)
            dur = (torch.exp(log_dur) - 1.0) * maskL.float()
            k_pred = min(max(int(round(float(dur.sum().item()))), L_valid, 16), limit)
        log_dur = self.dur_pred(h_enc, maskL)
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

        maskK = torch.ones(1, zT0.shape[1], device=self.device, dtype=torch.bool)
        style_e_demo = None
        if self.use_tts_style_latent:
            if self.tts_style_prior is not None:
                if getattr(self, "tts_style_prior_type", "") == "canonical_speaker":
                    u_mu_p, u_logvar_p = self.tts_style_prior(
                        zT0,
                        maskK,
                        spk_e.to(device=zT0.device, dtype=zT0.dtype),
                    )
                else:
                    u_mu_p, u_logvar_p = self.tts_style_prior(spk_e.to(dtype=zT0.dtype))
            else:
                u_mu_p = torch.zeros(1, self.tts_style_dim, device=self.device, dtype=zT0.dtype)
                u_logvar_p = torch.zeros_like(u_mu_p)
            if style_temp > 0.0:
                style_e_demo = u_mu_p + float(style_temp) * torch.exp(0.5 * u_logvar_p) * torch.randn_like(u_mu_p)
            else:
                style_e_demo = u_mu_p

        text_cond_demo = None
        if self.canonical_to_source is not None:
            zT0_source = self.canonical_to_source(
                zT0,
                maskK,
                spk_e=spk_e.to(dtype=zT0.dtype),
                style_e=style_e_demo if self.tts_style_into_source else None,
            )
            zT_mean_source = self.canonical_to_source(
                zT_mean,
                maskK,
                spk_e=spk_e.to(dtype=zT_mean.dtype),
                style_e=(
                    style_e_demo.to(dtype=zT_mean.dtype)
                    if (self.tts_style_into_source and style_e_demo is not None)
                    else None
                ),
            )
            if self.use_vf_canonical_text_cond:
                text_cond_demo = zT_mean
        else:
            zT0_source = zT0
            zT_mean_source = zT_mean
            if self.tts_style_to_source is not None and style_e_demo is not None:
                style_bias = self.tts_style_to_source(style_e_demo, spk_e.to(dtype=zT0.dtype)).to(dtype=zT0.dtype)
                zT0_source = zT0_source + self.tts_style_source_scale * style_bias.unsqueeze(1)
                zT_mean_source = zT_mean_source + self.tts_style_source_scale * style_bias.unsqueeze(1)
            if self.use_vf_canonical_text_cond:
                text_cond_demo = zT_mean

        spk_e_demo = spk_e.to(dtype=zT0_source.dtype)
        if self.tts_source_cond is not None and prior_temp > 0 and self.canonical_to_source is None:
            source_delta_demo = zT0_source - zT_mean_source
            spk_e_demo = spk_e_demo + self.tts_source_cond_scale * self.tts_source_cond(
                source_delta_demo,
                maskK,
            ).to(dtype=spk_e_demo.dtype)

        zS_pred = heun_integrate(
            self.vf,
            zT0_source,
            maskK,
            steps=int(ode_steps),
            direction=+1,
            cfg_scale=float(cfg_scale),
            spk_e=spk_e_demo,
            style_e=style_e_demo,
            text_cond=text_cond_demo,
        )
        zS_ref = self.mel_refiner(zS_pred, cond=h_enc, cond_mask=maskL) if self.mel_refiner is not None else zS_pred
        mel = (zS_ref * self.std_b + self.mu_b).float()
        wav = self.svae_model.decode(mel).squeeze(0).float().detach().cpu()
        wav_path = str(wav_path)
        save_wav(wav_path, wav, sr=self.sampling_rate)
        return {
            "wav_path": wav_path,
            "frames": int(mel.shape[1]),
            "seconds": float(wav.numel()) / float(self.sampling_rate),
            "text_len": int(L_valid),
        }


def load_utmos(args, device):
    if args.skip_utmos:
        return None
    print(f"[UTMOS] loading torch.hub {args.utmos_repo} {args.utmos_model}")
    try:
        model = torch.hub.load(args.utmos_repo, args.utmos_model, trust_repo=True)
    except TypeError:
        model = torch.hub.load(args.utmos_repo, args.utmos_model)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


@torch.no_grad()
def score_utmos(model, wav_path, sr, device):
    if model is None:
        return None
    import soundfile as sf

    wav_np, file_sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    if int(file_sr) != int(sr):
        import librosa

        wav_np = librosa.resample(wav_np, orig_sr=int(file_sr), target_sr=int(sr))
    wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).unsqueeze(0).to(device)
    try:
        score = model(wav, int(sr))
    except TypeError:
        score = model(wav)
    if torch.is_tensor(score):
        return float(score.detach().float().reshape(-1).mean().cpu().item())
    return float(np.asarray(score).reshape(-1).mean())


def load_whisper(args, device):
    if args.skip_whisper:
        return None

    print(f"[WHISPER] loading {args.whisper_model}")
    imported_whisper_path = None
    imported_whisper_error = None
    try:
        import whisper

        imported_whisper_path = getattr(whisper, "__file__", None)
        if hasattr(whisper, "load_model"):
            model = whisper.load_model(args.whisper_model, device=device)
            return {"backend": "openai-whisper", "model": model}
    except Exception as exc:
        imported_whisper_error = repr(exc)

    try:
        from faster_whisper import WhisperModel

        fw_device = "cuda" if str(device).startswith("cuda") else "cpu"
        compute_type = "float16" if fw_device == "cuda" else "int8"
        model = WhisperModel(args.whisper_model, device=fw_device, compute_type=compute_type)
        print("[WHISPER] using faster-whisper backend")
        return {"backend": "faster-whisper", "model": model}
    except Exception as faster_exc:
        faster_error = repr(faster_exc)

    try:
        from transformers import pipeline

        model_id = args.whisper_model if "/" in str(args.whisper_model) else f"openai/whisper-{args.whisper_model}"
        pipe_device = 0 if str(device).startswith("cuda") else -1
        model = pipeline("automatic-speech-recognition", model=model_id, device=pipe_device)
        print(f"[WHISPER] using transformers backend {model_id}")
        return {"backend": "transformers-whisper", "model": model}
    except Exception as transformers_exc:
        transformers_error = repr(transformers_exc)

    raise RuntimeError(
        "Could not load a Whisper ASR backend. The imported `whisper` module is not OpenAI "
        f"Whisper or failed to import load_model. whisper_path={imported_whisper_path!r} "
        f"whisper_error={imported_whisper_error!r} faster_whisper_error={faster_error} "
        f"transformers_error={transformers_error}. In the biflow_fix2 env, install the correct "
        "package with `pip install -U openai-whisper`, or remove the conflicting `whisper` "
        "package and reinstall `openai-whisper`."
    )


def transcribe_whisper(whisper_state, wav_path, device):
    if whisper_state is None:
        return None
    backend = whisper_state.get("backend")
    model = whisper_state.get("model")
    if backend == "openai-whisper":
        import librosa

        audio, _ = librosa.load(wav_path, sr=16000, mono=True)
        result = model.transcribe(
            audio,
            language="en",
            fp16=(str(device).startswith("cuda")),
            verbose=False,
        )
        return result.get("text", "")
    if backend == "faster-whisper":
        segments, _ = model.transcribe(wav_path, language="en", beam_size=5)
        return " ".join(seg.text for seg in segments)
    if backend == "transformers-whisper":
        result = model(wav_path)
        if isinstance(result, dict):
            return result.get("text", "")
        return str(result)
    raise ValueError(f"Unsupported whisper backend: {backend}")


def pick_test_rows(path, max_rows, force_normalize):
    rows = read_jsonl_rows(path, max_rows=None)
    picked = []
    seen = set()
    for row in rows:
        text = row_text(row, force_normalize=force_normalize)
        if not text or text in seen:
            continue
        seen.add(text)
        picked.append((row, text))
        if max_rows is not None and len(picked) >= int(max_rows):
            break
    if not picked:
        raise RuntimeError(f"No usable test text rows found in {path}")
    return picked


def build_eval_pairs(test_rows, speakers, pairing):
    if pairing == "cartesian":
        return [
            (row_idx, row, text, spk)
            for row_idx, (row, text) in enumerate(test_rows)
            for spk in speakers
        ]
    return [
        (row_idx, row, text, speakers[row_idx % len(speakers)])
        for row_idx, (row, text) in enumerate(test_rows)
    ]


def mean_or_none(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return (sum(vals) / len(vals)) if vals else None


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_dir = Path(args.ckpt_dir).resolve()
    config_path = ckpt_dir / "merged_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = load_json(config_path)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ckpt_dir / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    out_dir = Path(args.output_dir) if args.output_dir else (ckpt_dir / "tts_eval_test_topk")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    if results_path.exists():
        results_path.unlink()

    evaluator = TTSEvaluator(cfg, str(ckpt_path), str(out_dir), device)

    test_manifest = args.test_manifest or cfg["paths"].get("demo_aligned_manifest")
    if not test_manifest:
        raise RuntimeError("No --test-manifest and no paths.demo_aligned_manifest in config.")
    test_rows = pick_test_rows(
        test_manifest,
        args.max_test_rows,
        force_normalize=bool(cfg["data"].get("force_text_normalize", True)),
    )

    speakers = list(evaluator.target_spks)
    if args.num_speakers is not None:
        speakers = speakers[: int(args.num_speakers)]
    speakers = [spk for spk in speakers if spk in evaluator.spk2id]
    if not speakers:
        raise RuntimeError("No selected top-k speakers exist in speaker table.")

    pairs = build_eval_pairs(test_rows, speakers, args.pairing)
    cfg_scale = float(args.demo_cfg_scale if args.demo_cfg_scale is not None else cfg["infer"]["demo_cfg_scale"])
    prior_temp = float(args.demo_prior_temp if args.demo_prior_temp is not None else cfg["infer"]["demo_prior_temp"])
    style_temp = float(args.demo_style_temp if args.demo_style_temp is not None else cfg["infer"].get("demo_style_temp", 0.0))
    ode_steps = int(args.ode_steps if args.ode_steps is not None else cfg["infer"]["ode_steps_eval"])

    utmos_model = load_utmos(args, device)
    whisper_model = load_whisper(args, device)

    print(
        f"[EVAL] test_manifest={test_manifest} rows={len(test_rows)} "
        f"speakers={len(speakers)} pairs={len(pairs)} pairing={args.pairing}"
    )
    print(f"[EVAL] cfg_scale={cfg_scale} prior_temp={prior_temp} style_temp={style_temp} ode_steps={ode_steps}")

    results = []
    for idx, (row_idx, row, text, spk) in enumerate(pairs):
        wav_name = f"{idx:06d}_test{row_idx:05d}_spk{safe_name(spk)}.wav"
        wav_path = wav_dir / wav_name
        synth_meta = None
        if wav_path.exists() and not args.force_resynthesize:
            synth_meta = {
                "wav_path": str(wav_path),
                "frames": None,
                "seconds": None,
                "text_len": None,
            }
        else:
            synth_meta = evaluator.synthesize(
                text,
                spk,
                wav_path,
                cfg_scale=cfg_scale,
                prior_temp=prior_temp,
                style_temp=style_temp,
                ode_steps=ode_steps,
            )

        utmos = score_utmos(utmos_model, str(wav_path), evaluator.sampling_rate, device)
        hyp = transcribe_whisper(whisper_model, str(wav_path), device)
        ref_norm = normalize_english_text_for_wer(text)
        hyp_norm = normalize_english_text_for_wer(hyp) if hyp is not None else None
        wer = word_error_rate_notebook(text, hyp) if hyp is not None else None
        item = {
            "idx": idx,
            "test_row_idx": row_idx,
            "test_utt_id": row.get("utt_id"),
            "test_wav": row.get("wav"),
            "speaker": spk,
            "text": text,
            "wav_path": str(wav_path),
            "frames": synth_meta.get("frames"),
            "seconds": synth_meta.get("seconds"),
            "utmos": utmos,
            "whisper_model": None if whisper_model is None else args.whisper_model,
            "whisper_backend": None if whisper_model is None else whisper_model.get("backend"),
            "hyp": hyp,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "wer": wer,
        }
        append_jsonl(results_path, item)
        results.append(item)
        if args.print_every > 0 and ((idx + 1) % args.print_every == 0 or idx + 1 == len(pairs)):
            wer_s = "NA" if wer is None else f"{wer:.4f}"
            utmos_s = "NA" if utmos is None else f"{utmos:.4f}"
            print(f"[{idx + 1}/{len(pairs)}] spk={spk} UTMOS={utmos_s} WER={wer_s} text={text[:80]}")

    summary = {
        "ckpt_dir": str(ckpt_dir),
        "checkpoint": str(ckpt_path),
        "step": evaluator.step,
        "test_manifest": str(test_manifest),
        "output_dir": str(out_dir),
        "pairing": args.pairing,
        "num_pairs": len(results),
        "num_test_rows": len(test_rows),
        "num_speakers": len(speakers),
        "speakers": speakers,
        "cfg_scale": cfg_scale,
        "prior_temp": prior_temp,
        "style_temp": style_temp,
        "ode_steps": ode_steps,
        "whisper_model": None if whisper_model is None else args.whisper_model,
        "whisper_backend": None if whisper_model is None else whisper_model.get("backend"),
        "wer_normalizer": "lowercase_asr_no_punctuation",
        "utmos_mean": mean_or_none([r["utmos"] for r in results]),
        "wer_mean": mean_or_none([r["wer"] for r in results]),
        "results_jsonl": str(results_path),
    }
    write_json(summary_path, summary)
    print(f"[DONE] wrote {results_path}")
    print(f"[DONE] wrote {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
