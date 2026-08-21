from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Sequence

from biflow.utils import normalize_text_basic, read_jsonl_rows


def torchaudio_tokens_from_itos(itos: Sequence[str], blank_id: int) -> list[str]:
    """Map the local char vocabulary to torchaudio/flashlight CTC token strings."""
    tokens: list[str] = []
    used: set[str] = set()
    for idx, token in enumerate(itos):
        if idx == int(blank_id):
            mapped = "-"
        elif token == " ":
            mapped = "|"
        elif token in {"<pad>", "<unk>", "<blank>"}:
            mapped = f"#{idx}"
        else:
            mapped = str(token).lower()
        if mapped in used:
            mapped = f"{mapped}#{idx}"
        used.add(mapped)
        tokens.append(mapped)
    return tokens


def build_normalized_lexicon_from_manifest(
    itos: Sequence[str],
    blank_id: int,
    manifest_path: str,
    cache_dir: Optional[str] = None,
) -> str:
    """Build a flashlight-compatible lexicon from normalized manifest text.

    The local tokenizer only contains normalized ASR characters (a-z, space),
    so the torchaudio LibriSpeech preset lexicon is not directly compatible.
    """
    manifest_path = os.path.abspath(str(manifest_path))
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"KenLM lexicon corpus manifest not found: {manifest_path}")

    tokens = torchaudio_tokens_from_itos(itos, blank_id)
    valid_spell_tokens = {
        token for token in tokens
        if token not in {"-", "|"} and not str(token).startswith("#")
    }
    if not valid_spell_tokens:
        raise RuntimeError("No valid spell tokens available to build a KenLM lexicon.")

    vocab_sig = hashlib.md5("\n".join(tokens).encode("utf-8")).hexdigest()[:12]
    manifest_sig = os.path.splitext(os.path.basename(manifest_path))[0]
    cache_root = os.path.abspath(cache_dir or "/tmp/biflow_kenlm_lexicons")
    os.makedirs(cache_root, exist_ok=True)
    lexicon_path = os.path.join(cache_root, f"{manifest_sig}.{vocab_sig}.lexicon.txt")
    if os.path.exists(lexicon_path) and os.path.getsize(lexicon_path) > 0:
        return lexicon_path

    rows = read_jsonl_rows(manifest_path, max_rows=None)
    words: set[str] = set()
    text_fields = (
        "text_norm",
        "text_norm_ctx",
        "text_norm_full",
        "text",
        "text_raw",
        "text_raw_full",
    )
    for row in rows:
        text = ""
        for field in text_fields:
            value = row.get(field)
            if value:
                text = normalize_text_basic(value)
                if text:
                    break
        if not text:
            continue
        words.update(text.split())

    kept = []
    for word in sorted(words):
        word_lc = str(word).lower()
        if all(ch in valid_spell_tokens for ch in word_lc):
            kept.append((word_lc, " ".join(word_lc)))
    if not kept:
        raise RuntimeError(
            f"Failed to build KenLM lexicon from {manifest_path}: "
            "no compatible normalized words matched the tokenizer."
        )

    with open(lexicon_path, "w", encoding="utf-8") as f:
        for word_lc, spelling in kept:
            f.write(f"{word_lc}\t{spelling}\n")
    return lexicon_path


def lexicon_spelling_compatible(lexicon_path: str, tokens: Sequence[str]) -> tuple[bool, Optional[str]]:
    spell_tokens = {
        str(token)
        for token in tokens
        if token not in {"-", "|"} and not str(token).startswith("#")
    }
    try:
        with open(lexicon_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                for token in parts[1:]:
                    if token not in spell_tokens:
                        return False, token
    except UnicodeDecodeError:
        with open(lexicon_path, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                for token in parts[1:]:
                    if token not in spell_tokens:
                        return False, token
    return True, None


@dataclass
class KenLMCTCDecoderConfig:
    preset: str = "librispeech-4-gram"
    lexicon: Optional[str] = None
    lm: Optional[str] = None
    lexicon_corpus_manifest: Optional[str] = None
    lexicon_cache_dir: Optional[str] = None
    beam_size: int = 100
    beam_threshold: float = 100.0
    beam_size_token: int = 30
    lm_weight: float = 1.23
    word_score: float = -0.26
    unk_score: float = float("-inf")
    allow_fallback: bool = True


class OptionalKenLMCTCDecoder:
    """Optional torchaudio CTC beam decoder with KenLM.

    This wrapper keeps train/eval robust on machines where torchaudio,
    flashlight-text, or the LM files are not installed/cached.
    """

    def __init__(self, itos: Sequence[str], blank_id: int, cfg: KenLMCTCDecoderConfig):
        self.enabled = False
        self.error: Optional[str] = None
        self.decoder = None
        self.cfg = cfg
        self.tokens = torchaudio_tokens_from_itos(itos, blank_id)
        self.lexicon_path: Optional[str] = None
        self.lm_path: Optional[str] = None

        try:
            from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

            lexicon = cfg.lexicon
            lm = cfg.lm
            if not lm:
                files = download_pretrained_files(cfg.preset)
                lm = files.lm
                if not lexicon and not cfg.lexicon_corpus_manifest:
                    lexicon = files.lexicon
            if not lexicon and cfg.lexicon_corpus_manifest:
                lexicon = build_normalized_lexicon_from_manifest(
                    itos,
                    blank_id,
                    cfg.lexicon_corpus_manifest,
                    cache_dir=cfg.lexicon_cache_dir,
                )
            if not lexicon:
                raise RuntimeError(
                    "KenLM lexicon is not available. Provide cfg.lexicon or "
                    "cfg.lexicon_corpus_manifest."
                )
            lexicon_path = os.path.abspath(str(lexicon))
            compatible, bad_token = lexicon_spelling_compatible(lexicon_path, self.tokens)
            if not compatible and cfg.lexicon_corpus_manifest:
                lexicon = build_normalized_lexicon_from_manifest(
                    itos,
                    blank_id,
                    cfg.lexicon_corpus_manifest,
                    cache_dir=cfg.lexicon_cache_dir,
                )
                lexicon_path = os.path.abspath(str(lexicon))
                compatible, bad_token = lexicon_spelling_compatible(lexicon_path, self.tokens)
            if not compatible:
                raise RuntimeError(
                    f"KenLM lexicon {lexicon_path} contains token {bad_token!r} "
                    "that is absent from the current CTC vocabulary. Provide "
                    "cfg.lexicon_corpus_manifest to build a normalized lexicon."
                )

            self.lexicon_path = lexicon_path
            self.lm_path = os.path.abspath(str(lm))

            self.decoder = ctc_decoder(
                lexicon=self.lexicon_path,
                tokens=self.tokens,
                lm=self.lm_path,
                nbest=1,
                beam_size=int(cfg.beam_size),
                beam_threshold=float(cfg.beam_threshold),
                beam_size_token=int(cfg.beam_size_token),
                lm_weight=float(cfg.lm_weight),
                word_score=float(cfg.word_score),
                unk_score=float(cfg.unk_score),
                blank_token="-",
                sil_token="|",
            )
            self.enabled = True
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps/files.
            self.error = repr(exc)
            if not cfg.allow_fallback:
                raise

    def decode(self, logits_btv) -> str:
        if not self.enabled or self.decoder is None:
            raise RuntimeError(f"KenLM decoder unavailable: {self.error}")
        import torch.nn.functional as F

        emissions = F.log_softmax(logits_btv.detach().float(), dim=-1).cpu()
        results = self.decoder(emissions)
        if not results or not results[0]:
            return ""
        best = results[0][0]
        words = getattr(best, "words", None)
        if words:
            return " ".join(str(w) for w in words).strip()
        tokens = getattr(best, "tokens", None)
        if tokens is None:
            return ""
        chars = []
        for token_id in tokens:
            token = self.tokens[int(token_id)]
            if token in {"-", "|"} or token.startswith("#"):
                if token == "|":
                    chars.append(" ")
                continue
            chars.append(token)
        return "".join(chars).strip()
