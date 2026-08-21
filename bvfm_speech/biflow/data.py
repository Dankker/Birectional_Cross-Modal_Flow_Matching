import hashlib
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from biflow.utils import normalize_text_basic


def _load_tensor_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CutDataset(Dataset):
    def __init__(
        self,
        rows,
        spk2id,
        mel_cache_dir,
        mel_cache_max_items=128,
        sample_reference_wav=False,
    ):
        self.rows = list(rows)
        self.spk2id = dict(spk2id)
        self.mel_cache_dir = mel_cache_dir
        self.mel_cache_max_items = int(max(0, mel_cache_max_items))
        self._mel_cache = OrderedDict()
        self.sample_reference_wav = bool(sample_reference_wav)
        self._ref_wavs_by_spk = {}
        if self.sample_reference_wav:
            for row in self.rows:
                spk = str(row.get("speaker", ""))
                wav_path = row.get("parent_wav", row.get("wav"))
                if spk and wav_path:
                    self._ref_wavs_by_spk.setdefault(spk, []).append(wav_path)

    def __len__(self):
        return len(self.rows)

    def _sha1_key(self, text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _mel_cache_path(self, wav_path):
        if not self.mel_cache_dir:
            return None
        key = self._sha1_key(os.path.abspath(wav_path))
        return os.path.join(self.mel_cache_dir, f"{key}.pt")

    def _load_mel_full(self, wav_path):
        cache_path = self._mel_cache_path(wav_path)
        if cache_path is None or not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing mel cache for DataLoader path={wav_path} cache={cache_path}"
            )

        if cache_path in self._mel_cache:
            mel_full = self._mel_cache[cache_path]
            self._mel_cache.move_to_end(cache_path)
            return mel_full

        mel_full = _load_tensor_cache(cache_path).float()
        if self.mel_cache_max_items > 0:
            self._mel_cache[cache_path] = mel_full
            self._mel_cache.move_to_end(cache_path)
            while len(self._mel_cache) > self.mel_cache_max_items:
                self._mel_cache.popitem(last=False)
        return mel_full

    def _load_speech_full(self, row, wav_path):
        latent_path = row.get("svae_latent_path") or row.get("speech_path") or row.get("latent_path")
        if latent_path:
            latent_path = os.path.abspath(str(latent_path))
            if latent_path in self._mel_cache:
                speech_full = self._mel_cache[latent_path]
                self._mel_cache.move_to_end(latent_path)
                return speech_full
            if not os.path.exists(latent_path):
                raise FileNotFoundError(f"Missing speech latent path={latent_path}")
            arr = np.load(latent_path)
            speech_full = torch.from_numpy(arr).float().contiguous()
            if speech_full.ndim == 3 and speech_full.shape[0] == 1:
                # Official Semantic-VAE extractor stores [1, D, T]. Our custom
                # preprocessor stores [T, D], but support both.
                speech_full = speech_full[0].transpose(0, 1).contiguous()
            if speech_full.ndim != 2:
                raise RuntimeError(f"Expected 2D speech latent, got {tuple(speech_full.shape)} from {latent_path}")
            if self.mel_cache_max_items > 0:
                self._mel_cache[latent_path] = speech_full
                self._mel_cache.move_to_end(latent_path)
                while len(self._mel_cache) > self.mel_cache_max_items:
                    self._mel_cache.popitem(last=False)
            return speech_full
        return self._load_mel_full(wav_path)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav_path = row.get("parent_wav", row.get("wav"))
        spk = str(row["speaker"])
        ref_wav_path = wav_path
        if self.sample_reference_wav:
            candidates = self._ref_wavs_by_spk.get(spk) or [wav_path]
            other_candidates = [path for path in candidates if path != wav_path]
            ref_wav_path = random.choice(other_candidates or candidates)
        mel_full = self._load_speech_full(row, wav_path)
        K0 = int(mel_full.shape[0])
        start = int(row.get("cut_start_mel", row.get("ctx_mel_start", 0)))
        end = int(row.get("cut_end_mel", row.get("ctx_mel_end", K0)))
        start = max(0, min(start, K0 - 1))
        end = max(start + 1, min(end, K0))
        K = int(end - start)

        core_s = int(row.get("core_start_in_ctx", 0))
        core_e = int(row.get("core_end_in_ctx", K))
        core_s = max(0, min(core_s, K))
        core_e = max(core_s + 1, min(core_e, K)) if K > 0 else 0
        loss_mask = torch.zeros(K, dtype=torch.bool)
        if K > 0:
            loss_mask[core_s:core_e] = True

        return dict(
            zS_log=mel_full[start:end].contiguous(),
            K=K,
            text=normalize_text_basic(str(row.get("text_norm", row.get("text_norm_ctx", row.get("text", "")))).strip()),
            spk_id=int(self.spk2id[spk]),
            wav_path=wav_path,
            ref_wav_path=ref_wav_path,
            start=start,
            end=end,
            loss_mask=loss_mask,
            row_meta=row,
        )


def collate_cut_batch(samples):
    B = len(samples)
    K_list = [int(sample["K"]) for sample in samples]
    Kmax = max(K_list)
    D = int(samples[0]["zS_log"].shape[-1])

    zS_log = torch.zeros(B, Kmax, D, dtype=torch.float32)
    maskK = torch.zeros(B, Kmax, dtype=torch.bool)
    spk_ids = torch.zeros(B, dtype=torch.long)
    texts = []
    wav_paths = []
    ref_wav_paths = []
    starts = []
    ends = []
    loss_maskK = torch.zeros(B, Kmax, dtype=torch.bool)
    row_metas = []

    for b, sample in enumerate(samples):
        K = int(sample["K"])
        zS_log[b, :K] = sample["zS_log"]
        maskK[b, :K] = True
        loss_maskK[b, :K] = sample.get("loss_mask", torch.ones(K, dtype=torch.bool))
        spk_ids[b] = int(sample["spk_id"])
        texts.append(sample["text"])
        wav_paths.append(sample["wav_path"])
        ref_wav_paths.append(sample.get("ref_wav_path", sample["wav_path"]))
        starts.append(int(sample["start"]))
        ends.append(int(sample["end"]))
        row_metas.append(sample.get("row_meta", {}))

    return dict(
        zS_log=zS_log,
        maskK=maskK,
        loss_maskK=loss_maskK,
        K_list=K_list,
        texts=texts,
        spk_ids=spk_ids,
        wav_paths=wav_paths,
        ref_wav_paths=ref_wav_paths,
        starts=starts,
        ends=ends,
        row_metas=row_metas,
    )
