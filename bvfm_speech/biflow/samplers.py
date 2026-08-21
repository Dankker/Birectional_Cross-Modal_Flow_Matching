import random

import numpy as np


class LengthBucketBatchSampler:
    def __init__(
        self,
        rows,
        batch_size,
        length_key="cut_mel_len",
        num_buckets=12,
        seed=0,
        max_frames_per_batch=None,
        max_utts_per_batch=None,
    ):
        self.rows = list(rows)
        self.batch_size = int(batch_size)
        self.length_key = str(length_key)
        self.num_buckets = int(max(1, num_buckets))
        self.rng = random.Random(int(seed))
        self.max_frames_per_batch = (
            int(max_frames_per_batch) if max_frames_per_batch is not None and int(max_frames_per_batch) > 0 else None
        )
        if max_utts_per_batch is None:
            self.max_utts_per_batch = self.batch_size
        else:
            self.max_utts_per_batch = int(max(1, max_utts_per_batch))
        self.buckets = self._build_buckets()
        self._epoch = 0
        self._sample_batches = []
        self._sample_pos = 0
        self._last_epoch_batches = None

    def _row_length(self, row):
        return int(max(1, row.get(self.length_key, 1)))

    def _build_buckets(self):
        if not self.rows:
            return []
        lengths = np.asarray([self._row_length(row) for row in self.rows], dtype=np.int64)
        quantiles = np.linspace(0.0, 1.0, self.num_buckets + 1)
        edges = np.unique(np.quantile(lengths, quantiles).astype(np.int64))
        if edges.size <= 1:
            return [list(range(len(self.rows)))]

        buckets = []
        for start, end in zip(edges[:-1], edges[1:]):
            if end <= start:
                continue
            bucket = [
                idx for idx, row in enumerate(self.rows)
                if start <= self._row_length(row) < end
            ]
            if bucket:
                buckets.append(bucket)

        tail_bucket = [
            idx for idx, row in enumerate(self.rows)
            if self._row_length(row) >= int(edges[-1])
        ]
        if tail_bucket:
            buckets.append(tail_bucket)

        if not buckets:
            buckets = [list(range(len(self.rows)))]
        return buckets

    def sample_indices(self):
        if self._sample_pos >= len(self._sample_batches):
            self._sample_batches = self._build_epoch_batches()
            self._sample_pos = 0
        if not self._sample_batches:
            return []
        batch = self._sample_batches[self._sample_pos]
        self._sample_pos += 1
        return list(batch)

    def sample(self):
        return [self.rows[idx] for idx in self.sample_indices()]

    def __iter__(self):
        for batch in self._build_epoch_batches():
            yield batch

    def __len__(self):
        if self._last_epoch_batches is None:
            self._last_epoch_batches = self._build_epoch_batches(advance_epoch=False)
        return len(self._last_epoch_batches)

    def _build_epoch_batches(self, advance_epoch=True):
        if not self.rows:
            return []

        rng = random.Random(self.rng.randrange(2**31 - 1))
        if not advance_epoch:
            rng = random.Random(0)

        batches = []
        for bucket in self.buckets:
            indices = list(bucket)
            rng.shuffle(indices)
            indices.sort(key=lambda idx: self._row_length(self.rows[idx]), reverse=True)
            batches.extend(self._pack_indices(indices))

        rng.shuffle(batches)
        if advance_epoch:
            self._epoch += 1
            self._last_epoch_batches = batches
        return batches

    def _pack_indices(self, indices):
        if not indices:
            return []

        batches = []
        batch = []
        total_frames = 0
        max_items = self.max_utts_per_batch if self.max_frames_per_batch is not None else self.batch_size

        for idx in indices:
            length = self._row_length(self.rows[idx])
            over_frames = (
                self.max_frames_per_batch is not None
                and batch
                and (total_frames + length) > self.max_frames_per_batch
            )
            over_utts = batch and len(batch) >= max_items
            if over_frames or over_utts:
                batches.append(batch)
                batch = []
                total_frames = 0

            batch.append(idx)
            total_frames += length

        if batch:
            batches.append(batch)
        return batches

    def batch_stats(self, max_batches=None):
        batches = self._last_epoch_batches
        if batches is None:
            batches = self._build_epoch_batches(advance_epoch=False)
        if max_batches is not None:
            batches = batches[: int(max_batches)]
        if not batches:
            return {
                "batches": 0,
                "utts_mean": 0.0,
                "frames_mean": 0.0,
                "padded_frames_mean": 0.0,
            }
        utts = []
        frames = []
        padded = []
        for batch in batches:
            lengths = [self._row_length(self.rows[idx]) for idx in batch]
            utts.append(len(lengths))
            frames.append(sum(lengths))
            padded.append(max(lengths) * len(lengths))
        return {
            "batches": len(batches),
            "utts_mean": float(np.mean(utts)),
            "utts_p50": float(np.median(utts)),
            "utts_max": int(max(utts)),
            "frames_mean": float(np.mean(frames)),
            "frames_p50": float(np.median(frames)),
            "frames_max": int(max(frames)),
            "padded_frames_mean": float(np.mean(padded)),
            "padded_frames_p50": float(np.median(padded)),
            "padded_frames_max": int(max(padded)),
        }
