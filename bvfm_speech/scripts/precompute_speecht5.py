#!/usr/bin/env python3

import argparse
import hashlib
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from biflow.config import load_config
from biflow.encoders import FrozenSpeechT5TextEncoder
from biflow.utils import normalize_text_basic, read_jsonl_rows


def sha1_key(text: str):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def select_target_speakers(rows, target_spks, top_k_spk):
    spk_count = {}
    for row in rows:
        spk = str(row.get("speaker", ""))
        if spk:
            spk_count[spk] = spk_count.get(spk, 0) + 1
    if target_spks is not None:
        return list(target_spks)
    spk_sorted = sorted(spk_count.items(), key=lambda kv: kv[1], reverse=True)
    return [s for (s, _) in spk_sorted[:top_k_spk]]


def batched(xs, batch_size):
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "cutmanifest_singlevf_stable.json"))
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg["cache"]["speecht5_dir"]
    assert output_dir, "cache.speecht5_dir or --output-dir is required"
    os.makedirs(output_dir, exist_ok=True)

    cut_rows_all = read_jsonl_rows(cfg["paths"]["cut_manifest"], max_rows=cfg["paths"]["max_cut_rows"])
    target_spks = select_target_speakers(
        cut_rows_all,
        cfg["data"]["target_spks"],
        int(cfg["data"]["top_k_spk"]),
    )
    target_spk_set = set(target_spks)
    texts = sorted(list({
        normalize_text_basic(str(r.get("text_norm", r.get("text", ""))).strip())
        for r in cut_rows_all
        if str(r.get("speaker", "")) in target_spk_set
        and len(normalize_text_basic(str(r.get("text_norm", r.get("text", ""))).strip())) > 0
    }))

    pending = [t for t in texts if not os.path.exists(os.path.join(output_dir, f"{sha1_key(t)}.pt"))]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st5 = FrozenSpeechT5TextEncoder(
        model_name="microsoft/speecht5_tts",
        device=device,
        layer_idx=int(cfg["model"]["st5_layer_idx"]),
    )

    print(f"[ST5-CACHE] device={device} texts={len(texts)} pending={len(pending)} out={output_dir}")
    done = 0
    for batch in batched(pending, args.batch_size):
        with torch.no_grad():
            hidden, mask = st5(batch)
        for i, text in enumerate(batch):
            L = int(mask[i].long().sum().item())
            out_path = os.path.join(output_dir, f"{sha1_key(text)}.pt")
            obj = {
                "text": text,
                "hidden": hidden[i, :L].detach().cpu().float().contiguous(),
                "mask": torch.ones(L, dtype=torch.bool),
            }
            torch.save(obj, out_path)
            done += 1
        print(f"[ST5-CACHE] progress {done}/{len(pending)}")


if __name__ == "__main__":
    main()
