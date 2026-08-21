import json
import random
import re

import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl_rows(path, max_rows=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] JSON parse failed at line {line_idx}: {e}")
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows

def save_wav(path, wav, sr=24000):
    import soundfile as sf
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu()
        if wav.dim() == 3:
            wav = wav[0]
        if wav.dim() == 2 and wav.shape[0] == 1:
            wav = wav[0]
        wav = wav.float().numpy()
    else:
        wav = np.asarray(wav, dtype=np.float32)
    sf.write(path, wav, sr)
    print("wrote", path)

def extract_speaker_id_from_path(path: str):
    """LibriTTS common: .../<spk>/<chapter>/<utt>.wav"""
    p = path.replace("\\", "/")
    parts = p.split("/")
    for seg in parts:
        if seg.isdigit() and len(seg) >= 3:
            return seg
    m = re.search(r"/(\d{3,})/", p)
    if m:
        return m.group(1)
    return None

class TeeIO:
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        try:
            return any(getattr(s, "isatty", lambda: False)() for s in self.streams)
        except Exception:
            return False

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def normalize_text_basic(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[-‐‑‒–—]+", " ", s)
    s = re.sub(r"[‘’‛`´ʼ]", "'", s)
    s = s.replace("'", "")
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _edit_distance_tokens(ref, hyp):
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

def word_error_rate_text(ref_text: str, hyp_text: str):
    ref = normalize_text_basic(ref_text).split()
    hyp = normalize_text_basic(hyp_text).split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    dist = _edit_distance_tokens(ref, hyp)
    return float(dist) / float(len(ref))
