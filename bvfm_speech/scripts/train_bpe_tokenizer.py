#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sentencepiece as spm

from biflow.utils import normalize_text_basic


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fixed SentencePiece BPE vocabulary for text and CTC")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Training JSONL manifest. May be specified more than once.",
    )
    parser.add_argument("--output-prefix", required=True, help="Output path without .model/.vocab suffix")
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--text-key", action="append", default=None)
    parser.add_argument("--character-coverage", type=float, default=1.0)
    return parser.parse_args()


def iter_texts(paths, text_keys):
    seen = set()
    for manifest in paths:
        with open(manifest, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                raw = ""
                for key in text_keys:
                    value = row.get(key)
                    if value:
                        raw = value
                        break
                text = normalize_text_basic(str(raw).strip())
                if text and text not in seen:
                    seen.add(text)
                    yield text


def main():
    args = parse_args()
    text_keys = args.text_key or ["text_norm", "text_norm_ctx", "text", "text_raw"]
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    count = 0
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_prefix.parent) as corpus:
        corpus_path = corpus.name
        for text in iter_texts(args.manifest, text_keys):
            corpus.write(text + "\n")
            digest.update(text.encode("utf-8"))
            digest.update(b"\n")
            count += 1
    if count == 0:
        os.unlink(corpus_path)
        raise RuntimeError("No non-empty normalized training text found")

    try:
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=str(output_prefix),
            model_type="bpe",
            vocab_size=int(args.vocab_size),
            character_coverage=float(args.character_coverage),
            normalization_rule_name="identity",
            pad_id=0,
            pad_piece="<pad>",
            unk_id=1,
            unk_piece="<unk>",
            bos_id=-1,
            eos_id=-1,
            user_defined_symbols=["<blank>"],
            hard_vocab_limit=False,
            shuffle_input_sentence=True,
        )
    finally:
        os.unlink(corpus_path)

    processor = spm.SentencePieceProcessor(model_file=str(output_prefix) + ".model")
    expected = {"<pad>": 0, "<unk>": 1, "<blank>": 2}
    actual = {piece: int(processor.piece_to_id(piece)) for piece in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected special token IDs: {actual}")

    metadata = {
        "type": "sentencepiece_bpe",
        "model_path": str(output_prefix) + ".model",
        "vocab_path": str(output_prefix) + ".vocab",
        "requested_vocab_size": int(args.vocab_size),
        "actual_vocab_size": int(processor.vocab_size()),
        "num_unique_training_texts": count,
        "training_text_sha256": digest.hexdigest(),
        "manifests": [str(Path(path).expanduser().resolve()) for path in args.manifest],
        "special_ids": actual,
    }
    metadata_path = str(output_prefix) + ".json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
