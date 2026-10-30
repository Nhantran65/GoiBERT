"""Loaders for ViSpamReviews, ViNLI and PhoNER_COVID19.

* ``vispam``: 4-class spam-type classification (``SpamLabel`` column).
* ``vinli``: 4-label natural language inference on sentence pairs.
* ``phoner``: syllable-level named-entity recognition with BIO tags.

Example IDs are ``"{task}:{file split}:{row}"``, where the file split is
``train``, ``dev`` or ``test``.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

TASK_IDS = ("vispam", "vinli", "phoner")
NLI_LABELS = ("entailment", "neutral", "contradiction", "other")
_FILE_SPLIT = {"train": "train", "validation": "dev", "test": "test"}


@dataclass
class Split:
    ids: list[str]
    texts: list[str] | None
    labels: list
    pairs: list[tuple[str, str]] | None = None
    words: list[list[str]] | None = None

    def subset(self, indices: list[int]) -> "Split":
        def pick(values):
            return [values[i] for i in indices] if values is not None else None

        return Split(pick(self.ids), pick(self.texts), pick(self.labels), pick(self.pairs), pick(self.words))


@dataclass
class Bundle:
    task_id: str
    train: Split
    validation: Split
    label_names: list[str]
    num_classes: int
    test: Split | None = None


def _ids(task: str, file_split: str, n: int) -> list[str]:
    return [f"{task}:{file_split}:{i}" for i in range(n)]


def _raw_dir(repo_root: Path | str | None) -> Path:
    if repo_root is None and os.environ.get("GOIBERT_DATA"):
        return Path(os.environ["GOIBERT_DATA"]).expanduser().resolve()
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    return root / "data" / "raw" if (root / "data" / "raw").is_dir() else root


def _load_vispam(raw: Path, split: str) -> Split:
    file_split = _FILE_SPLIT[split]
    with (raw / "vispamreviews" / f"{file_split}.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return Split(_ids("vispam", file_split, len(rows)), [r["Comment"] for r in rows], [int(r["SpamLabel"]) for r in rows])


def _load_vinli(raw: Path, split: str) -> Split:
    import pandas as pd

    file_split = _FILE_SPLIT[split]
    frame = pd.read_parquet(raw / "vinli" / f"{file_split}.parquet")
    pairs = list(zip(frame.sentence1.astype(str), frame.sentence2.astype(str)))
    return Split(_ids("vinli", file_split, len(frame)), None, [NLI_LABELS.index(x) for x in frame.gold_label], pairs=pairs)


def _load_phoner(raw: Path, split: str) -> Split:
    file_split = _FILE_SPLIT[split]
    with (raw / "phoner_covid19" / f"{file_split}_syllable.json").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(len(r["words"]) != len(r["tags"]) for r in rows):
        raise ValueError("word/tag length mismatch")
    return Split(_ids("phoner", file_split, len(rows)), None, [r["tags"] for r in rows], words=[r["words"] for r in rows])


_LOADERS = {"vispam": _load_vispam, "vinli": _load_vinli, "phoner": _load_phoner}


def load_task(task_id: str, repo_root: Path | str | None = None, *, include_test: bool = False) -> Bundle:
    if task_id not in TASK_IDS:
        raise ValueError(f"Unsupported task ID {task_id!r}; choose from {list(TASK_IDS)!r}")
    raw = _raw_dir(repo_root)
    load = _LOADERS[task_id]
    train, validation = load(raw, "train"), load(raw, "validation")
    test = load(raw, "test") if include_test else None
    if task_id == "vispam":
        labels = [str(i) for i in range(4)]
    elif task_id == "vinli":
        labels = list(NLI_LABELS)
    else:
        # "O" first, then every tag seen in train/validation, sorted (21 labels).
        labels = ["O"] + sorted({t for row in train.labels + validation.labels for t in row if t != "O"})
    return Bundle(task_id, train, validation, labels, len(labels), test)
