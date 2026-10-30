"""Shared training utilities: seeding, precision, data loaders, loss and validation.

Two data families use different loaders:

* the five UIT tasks (:mod:`goibert.data.uit`) are single-text tasks, encoded
  once and padded per batch by :func:`make_uit_loader`;
* ``vispam``, ``vinli`` (sentence pairs) and ``phoner`` (word-aligned tags)
  use :func:`make_extra_loader`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from . import metrics as task_metrics
from .data import uit
from .modeling import ABSA_NUM_CLASSES
from .ner_metrics import metrics as ner_metrics

MAX_TRAIN_TRUNCATION_RATE = 0.01


# --------------------------------------------------------------------------- I/O


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- reproducibility


def seed_everything(seed: int) -> None:
    """Seed every RNG and request deterministic kernels."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


def resolve_device(device_name: str | None) -> torch.device:
    device = torch.device(device_name) if device_name else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is unavailable")
    return device


def resolve_precision(precision: str, device: torch.device) -> str:
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if precision == "bf16" and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"bf16 was requested but is not supported on {device}")
    return precision


def make_autocast(precision: str, device: torch.device):
    if precision != "bf16":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


# ------------------------------------------------------------------- UIT tasks


@dataclass(frozen=True)
class LengthStats:
    train_count: int
    max_tokens: int
    p50_tokens: float
    p95_tokens: float
    p99_tokens: float
    truncation_rate_256: float
    truncation_rate_512: float
    max_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_train_lengths(tokenizer: Any, train_texts: Sequence[str]) -> LengthStats:
    """Use 256 tokens unless more than 1% of training texts would be truncated."""

    lengths = np.asarray(
        [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in train_texts],
        dtype=np.int64,
    )
    rate_256 = float(np.mean(lengths > 256))
    return LengthStats(
        train_count=len(lengths),
        max_tokens=int(lengths.max()),
        p50_tokens=float(np.percentile(lengths, 50)),
        p95_tokens=float(np.percentile(lengths, 95)),
        p99_tokens=float(np.percentile(lengths, 99)),
        truncation_rate_256=rate_256,
        truncation_rate_512=float(np.mean(lengths > 512)),
        max_length=256 if rate_256 <= MAX_TRAIN_TRUNCATION_RATE else 512,
    )


class _EncodedTexts(Dataset):
    def __init__(self, input_ids: list[list[int]], labels: Sequence[Any], ids: Sequence[Any], gold_labels: Sequence[Any]):
        if not (len(input_ids) == len(labels) == len(ids) == len(gold_labels)):
            raise ValueError("input_ids, labels, gold_labels and ids must have equal lengths")
        self.input_ids, self.labels, self.ids, self.gold_labels = input_ids, list(labels), list(ids), list(gold_labels)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"input_ids": self.input_ids[index], "labels": self.labels[index], "gold_labels": self.gold_labels[index], "id": self.ids[index]}


def make_uit_loader(
    tokenizer: Any,
    split: Any,
    *,
    labels: Sequence[Any],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Encode a UIT split once; pad per batch.

    ``labels`` are the model-space training labels; ``split.labels`` (the
    source labels) are kept as ``gold_labels`` for evaluation.
    """

    texts = list(split.texts)
    encoded = (
        [list(ids) for ids in tokenizer(texts, add_special_tokens=True, truncation=True, max_length=max_length, padding=False, return_attention_mask=False)["input_ids"]]
        if texts
        else []
    )
    dataset = _EncodedTexts(encoded, labels, split.ids, split.labels)

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = tokenizer.pad([{"input_ids": row["input_ids"]} for row in rows], padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        batch["gold_labels"] = torch.tensor([row["gold_labels"] for row in rows], dtype=torch.long)
        batch["ids"] = [row["id"] for row in rows]
        return batch

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )


def uit_task_shape(bundle: uit.TaskDataset) -> tuple[int, int, tuple[int, ...]]:
    """Return ``(num_labels, num_aspects, aspect_class_counts)``."""

    counts = tuple(bundle.aspect_class_counts)
    return (sum(counts) if counts else len(bundle.label_names)), bundle.num_aspects, counts


def evaluate_uit(task_id: str, bundle: uit.TaskDataset, gold: Sequence[Any], predictions: Sequence[Any], *, split: str) -> dict[str, Any]:
    return dict(
        task_metrics.evaluate_task(
            task_id=task_id,
            gold=gold,
            predictions=predictions,
            aspect_names=list(bundle.aspect_names),
            class_names=list(bundle.label_names),
            aspect_class_counts=list(bundle.aspect_class_counts),
            split=split,
        )
    )


def compute_loss(logits: Tensor, labels: Tensor, *, num_aspects: int, aspect_class_counts: Sequence[int] | None = None) -> Tensor:
    """Cross-entropy; for ABSA, the mean over aspect slots."""

    if num_aspects:
        counts = tuple(aspect_class_counts or (ABSA_NUM_CLASSES,) * num_aspects)
        if labels.ndim != 2 or labels.shape[1] != num_aspects:
            raise ValueError("ABSA labels must have shape [batch, aspects]")
        if len(set(counts)) == 1 and logits.ndim == 3:
            return nn.functional.cross_entropy(logits.reshape(-1, counts[0]), labels.reshape(-1))
        if logits.ndim != 2 or logits.shape[1] != sum(counts):
            raise ValueError("Variable-width ABSA logits must have shape [batch, sum(slot widths)]")
        losses, offset = [], 0
        for slot, width in enumerate(counts):
            losses.append(nn.functional.cross_entropy(logits[:, offset : offset + width], labels[:, slot]))
            offset += width
        return torch.stack(losses).mean()
    if labels.ndim != 1 or logits.ndim != 2:
        raise ValueError("Categorical labels/logits must be [batch] and [batch, classes]")
    return nn.functional.cross_entropy(logits, labels)


def _uit_predictions(logits: Tensor, aspect_class_counts: Sequence[int]) -> list[Any]:
    if not aspect_class_counts or logits.ndim == 3:
        return logits.argmax(dim=-1).detach().cpu().tolist()
    output = []
    for row in logits.detach().cpu():
        offset, labels = 0, []
        for width in aspect_class_counts:
            labels.append(int(row[offset : offset + width].argmax().item()))
            offset += width
        output.append(labels)
    return output


def validate_uit(model: nn.Module, loader: DataLoader, device: torch.device, precision: str, task_id: str, bundle: uit.TaskDataset, *, split: str = "validation"):
    model.eval()
    counts = tuple(bundle.aspect_class_counts)
    gold, predicted, ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            moved = move_batch(batch, device)
            with make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
            predicted.extend(_uit_predictions(logits, counts))
            gold.extend(batch["gold_labels"].tolist())
            ids.extend(batch["ids"])
    metrics = evaluate_uit(task_id, bundle, gold, predicted, split=split)
    return metrics, [{"id": i, "gold": g, "prediction": p} for i, g, p in zip(ids, gold, predicted)]


# ------------------------------------------------------------ ViSpam/ViNLI/NER


class _Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_extra_loader(split, task: str, tokenizer, label_names, batch_size: int, seed: int, shuffle: bool, max_length: int, limit: int | None = None):
    """Loader for ``vispam`` (text), ``vinli`` (pair) and ``phoner`` (words).

    For NER, labels are aligned to the first subword of each word; other
    subwords get ``-100``.  ``positions`` keeps ``(token index, word index)``
    pairs so predictions and soft labels can be mapped back to words.
    """

    rows, truncated, total_words, truncated_words = [], 0, 0, 0
    n = len(split.ids) if limit is None else min(limit, len(split.ids))
    for i in range(n):
        if task == "phoner":
            words, tags = split.words[i], split.labels[i]
            total_words += len(words)
            full = tokenizer(words, is_split_into_words=True, add_special_tokens=True, truncation=False)
            encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=True, truncation=True, max_length=max_length)
            seen, aligned, positions = set(), [], []
            for j, wid in enumerate(encoded.word_ids()):
                if wid is None or wid in seen:
                    aligned.append(-100)
                else:
                    seen.add(wid)
                    aligned.append(label_names.index(tags[wid]))
                    positions.append((j, wid))
            truncated_words += len(words) - len(seen)
            truncated += len(full["input_ids"]) > max_length
            rows.append({"id": split.ids[i], "enc": dict(encoded), "label": aligned, "gold": tags, "positions": positions, "word_count": len(words)})
        else:
            args = split.pairs[i] if task == "vinli" else (split.texts[i],)
            full = tokenizer(*args, add_special_tokens=True, truncation=False)
            encoded = tokenizer(*args, add_special_tokens=True, truncation=True, max_length=max_length)
            truncated += len(full["input_ids"]) > max_length
            rows.append({"id": split.ids[i], "enc": dict(encoded), "label": split.labels[i], "gold": split.labels[i]})

    def collate(items):
        batch = tokenizer.pad([r["enc"] for r in items], padding=True, return_tensors="pt")
        if task == "phoner":
            width = batch["input_ids"].shape[1]
            batch["labels"] = torch.tensor([r["label"] + [-100] * (width - len(r["label"])) for r in items])
            batch["positions"] = [r["positions"] for r in items]
            batch["word_count"] = [r["word_count"] for r in items]
        else:
            batch["labels"] = torch.tensor([r["label"] for r in items])
        batch["gold_labels"] = [r["gold"] for r in items]
        batch["ids"] = [r["id"] for r in items]
        return batch

    loader = DataLoader(
        _Rows(rows),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    stats = {"examples": n, "truncated_examples": truncated, "truncation_rate": truncated / n if n else 0.0, "words": total_words, "truncated_words": truncated_words}
    return loader, stats


def evaluate_extra(task: str, label_names: Sequence[str], gold: Sequence[Any], predictions: Sequence[Any], *, split: str = "validation") -> dict[str, Any]:
    if task == "phoner":
        result = ner_metrics(gold, predictions)
    else:
        result = task_metrics.classification_metrics(gold, predictions, labels=tuple(range(len(label_names))))
        result = {**result, "task_id": task, "selection_f1": result["macro_f1"], "selection_metric": "fixed_label_macro_f1"}
    result["split"] = split
    return result


def validate_extra(model: nn.Module, loader: DataLoader, task: str, label_names: Sequence[str], device: torch.device, precision: str, *, split: str = "validation"):
    model.eval()
    gold, predicted, ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            moved = move_batch(batch, device)
            with make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
            if task == "phoner":
                argmax = logits.argmax(-1).cpu().tolist()
                for k, positions in enumerate(batch["positions"]):
                    tags = ["O"] * batch["word_count"][k]
                    for pos, wid in positions:
                        tags[wid] = label_names[argmax[k][pos]]
                    predicted.append(tags)
            else:
                predicted.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(batch["gold_labels"])
            ids.extend(batch["ids"])
    metrics = evaluate_extra(task, label_names, gold, predicted, split=split)
    return metrics, [{"id": i, "gold": g, "prediction": p} for i, g, p in zip(ids, gold, predicted)]


def selection_score(metrics: Mapping[str, Any]) -> float:
    value = metrics.get("selection_f1")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise KeyError("Metrics must contain a finite selection_f1")
    return float(value)
