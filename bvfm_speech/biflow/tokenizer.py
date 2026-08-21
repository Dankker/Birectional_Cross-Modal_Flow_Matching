import os

import torch


class CharTokenizer:
    def __init__(self):
        self.stoi = {"<pad>": 0, "<unk>": 1, "<blank>": 2}
        self.itos = ["<pad>", "<unk>", "<blank>"]

    def build(self, texts):
        chars = set()
        for s in texts:
            for ch in s:
                chars.add(ch)
        for ch in sorted(list(chars)):
            if ch not in self.stoi:
                self.stoi[ch] = len(self.itos)
                self.itos.append(ch)

    def encode(self, s):
        return [self.stoi.get(ch, 1) for ch in s]

    def decode(self, ids):
        return ids_to_text(ids, self.itos)


class SentencePieceBPETokenizer:
    """Fixed SentencePiece BPE vocabulary shared by the text encoder and CTC."""

    def __init__(self, model_path):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "SentencePiece BPE requires the 'sentencepiece' package. "
                "Install it in the training environment before using tokenizer.type='sentencepiece_bpe'."
            ) from exc

        self.model_path = os.path.abspath(os.path.expanduser(str(model_path)))
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {self.model_path}")
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)
        self.itos = [self.processor.id_to_piece(i) for i in range(self.processor.vocab_size())]
        self.stoi = {piece: idx for idx, piece in enumerate(self.itos)}
        required = {"<pad>": 0, "<unk>": 1, "<blank>": 2}
        for piece, expected_id in required.items():
            actual_id = self.stoi.get(piece)
            if actual_id != expected_id:
                raise ValueError(
                    f"Invalid SentencePiece special ID for {piece}: expected {expected_id}, got {actual_id}. "
                    "Train the model with scripts/train_bpe_tokenizer.py."
                )

    def build(self, texts):
        # SentencePiece vocabularies are immutable after training.
        return None

    def encode(self, s):
        return list(self.processor.encode(str(s), out_type=int))

    def decode(self, ids):
        clean_ids = [
            int(idx)
            for idx in ids
            if 0 <= int(idx) < len(self.itos) and self.itos[int(idx)] not in {"<pad>", "<blank>"}
        ]
        return self.processor.decode(clean_ids)


def build_tokenizer(config=None, texts=None):
    config = config or {}
    tokenizer_type = str(config.get("type", "char")).lower()
    if tokenizer_type in {"char", "character"}:
        tokenizer = CharTokenizer()
        tokenizer.build(texts or [])
        return tokenizer
    if tokenizer_type in {"bpe", "sentencepiece", "sentencepiece_bpe", "spm_bpe"}:
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("tokenizer.model_path is required for SentencePiece BPE")
        return SentencePieceBPETokenizer(model_path)
    raise ValueError(f"Unsupported tokenizer.type={tokenizer_type!r}")

def ctc_greedy_decode(logits_bkv, input_len, blank_id=2):
    preds = logits_bkv.argmax(dim=-1)  # [B,K]
    outs = []
    for b in range(preds.shape[0]):
        T = int(input_len[b])
        seq = preds[b, :T].tolist()
        out = []
        prev = None
        for x in seq:
            if x == blank_id:
                prev = x
                continue
            if prev is not None and x == prev:
                prev = x
                continue
            out.append(x)
            prev = x
        outs.append(out)
    return outs

def ids_to_text(ids, itos):
    pieces = []
    for i in ids:
        if i < len(itos) and itos[i] not in ["<pad>", "<blank>"]:
            pieces.append("?" if itos[i] == "<unk>" else itos[i])
    text = "".join(pieces)
    if any("▁" in piece for piece in pieces):
        text = text.replace("▁", " ").strip()
    return text
