"""Locked corpus access: batches of aligned sequences in the order fixed by prepare_corpus.py (identical for M and G)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

import common

PAD = 1


class Corpus:
    def __init__(self, path: Path | None = None, split: str = "train"):
        path = path or (common.DATA_DIR / "corpus.npz")
        z = np.load(path)
        p = f"{split}_"
        self.s_ids, self.t_ids = z[p + "s_ids"], z[p + "t_ids"]
        self.s_span, self.t_span = z[p + "s_span"], z[p + "t_span"]
        self.s_off, self.t_off = z[p + "s_off"], z[p + "t_off"]
        self.n_units, self.doc = z[p + "n_units"], z[p + "doc"]
        self.n = len(self.n_units)

    def __len__(self) -> int:
        return self.n

    def sequence(self, i: int) -> dict:
        a, b = self.s_off[i], self.s_off[i + 1]
        c, d = self.t_off[i], self.t_off[i + 1]
        return {"s_ids": self.s_ids[a:b], "s_span": self.s_span[a:b], "t_ids": self.t_ids[c:d], "t_span": self.t_span[c:d], "n_units": int(self.n_units[i]), "doc": int(self.doc[i])}

    def collate(self, indices) -> dict[str, torch.Tensor]:
        rows = [self.sequence(int(i)) for i in indices]
        Ls = max(len(r["s_ids"]) for r in rows); Lt = max(len(r["t_ids"]) for r in rows)
        N = max(r["n_units"] for r in rows)
        B = len(rows)
        s_ids = torch.full((B, Ls), PAD, dtype=torch.long); s_am = torch.zeros((B, Ls), dtype=torch.long); s_sp = torch.full((B, Ls), -1, dtype=torch.long)
        t_ids = torch.full((B, Lt), PAD, dtype=torch.long); t_am = torch.zeros((B, Lt), dtype=torch.long); t_sp = torch.full((B, Lt), -1, dtype=torch.long)
        for k, r in enumerate(rows):
            n = len(r["s_ids"]); s_ids[k, :n] = torch.from_numpy(r["s_ids"].astype(np.int64)); s_am[k, :n] = 1; s_sp[k, :n] = torch.from_numpy(r["s_span"].astype(np.int64))
            m = len(r["t_ids"]); t_ids[k, :m] = torch.from_numpy(r["t_ids"].astype(np.int64)); t_am[k, :m] = 1; t_sp[k, :m] = torch.from_numpy(r["t_span"].astype(np.int64))
        return {"s_ids": s_ids, "s_am": s_am, "s_span": s_sp, "t_ids": t_ids, "t_am": t_am, "t_span": t_sp, "n_spans": N,
                "doc": torch.tensor([r["doc"] for r in rows]), "index": torch.as_tensor(np.asarray(indices, dtype=np.int64))}


def batch_hash(batch: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in ("s_ids", "s_am", "s_span", "t_ids", "t_am", "t_span", "index"):
        h.update(k.encode()); h.update(batch[k].numpy().tobytes())
    return h.hexdigest()


def global_batch_indices(step: int, global_batch: int = common.GLOBAL_BATCH) -> np.ndarray:
    """Sequences of optimizer step `step` (0-based) in the locked order."""
    return np.arange(step * global_batch, (step + 1) * global_batch)


def mask_generator(step: int, micro: int) -> torch.Generator:
    """One CPU generator per (step, micro-batch) so masks are identical across arms and independent of GPU."""
    return torch.Generator().manual_seed(common.SEED * 1_000_003 + step * 1_009 + micro)
