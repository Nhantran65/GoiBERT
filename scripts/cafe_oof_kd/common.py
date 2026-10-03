"""Shared constants and helpers for the CafeBERT -> GoiTurn-S out-of-fold KD pilot (validation only)."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from goibert import train as e1_train  # noqa: E402
from goibert.data import TaskSplit  # noqa: E402

ART = ROOT / "artifacts" / "cafe_oof_kd"
TASKS = ("absa_restaurant", "visfd", "vsfc_sentiment", "vsfc_topic", "absa_hotel")
N_FOLDS = 5
FOLD_SEED = 42
SEEDS = (42, 43, 44)
STUDENT_LR = {"absa_restaurant": 1e-4, "visfd": 1e-4, "absa_hotel": 1e-4, "vsfc_sentiment": 1e-5, "vsfc_topic": 1e-5}  # E7 task LRs
BAMI_LR = {"absa_restaurant": 5e-5, "visfd": 5e-5, "absa_hotel": 5e-5, "vsfc_sentiment": 2e-5, "vsfc_topic": 1e-5}  # E14 locked selections
KD_ALPHA = 0.5
KD_TEMPERATURE = 2.0


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def fold_assignment(n: int) -> np.ndarray:
    """Fold index per train position: rank in a seeded permutation, mod N_FOLDS."""
    order = np.random.default_rng(FOLD_SEED).permutation(n)
    fold = np.empty(n, dtype=np.int64)
    fold[order] = np.arange(n) % N_FOLDS
    return fold


def fold_bundle(bundle, fold: int):
    """Bundle whose train split excludes `fold`; the held-out fold is exposed as split 'oof'."""
    train = bundle.splits["train"]
    assign = fold_assignment(len(train))
    keep = tuple(ex for ex, f in zip(train.examples, assign) if f != fold)
    held = tuple(ex for ex, f in zip(train.examples, assign) if f == fold)
    splits = dict(bundle.splits)
    splits["train"] = TaskSplit("train", keep)
    splits["oof"] = TaskSplit("oof", held)
    return dataclasses.replace(bundle, splits=splits)


def label_widths(bundle) -> tuple[int, ...]:
    """Per-slot class widths; single-label tasks are one slot of num_classes."""
    counts = tuple(e1_train.task_aspect_class_counts(bundle))
    return counts if counts else (e1_train.task_num_classes(bundle),)


def slot_segments(counts) -> list[tuple[int, int]]:
    out, offset = [], 0
    for width in counts:
        out.append((offset, offset + width))
        offset += width
    return out


def flat_logits(logits, batch: int):
    """[B, A, C] or [B, sum(counts)] -> [B, sum(counts)] (slot-major)."""
    return logits.reshape(batch, -1)
