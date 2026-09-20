"""Supervised E1 training loop.

The module deliberately separates trial training from grid scheduling.  It
never looks at the test split unless the caller explicitly requests the final
``test`` stage.  Grid selection and the test gate live in :mod:`run_grid`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import transformers
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .modeling import ABSA_NUM_CLASSES, GoiBERTClassifier, build_classifier


TASK_IDS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
LEARNING_RATES = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4)
SCREENING_SEED = 42
SHORTLIST_SEEDS = (42, 43, 44)
FINAL_SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
MAX_LENGTH_CANDIDATES = (256, 512)
MAX_TRAIN_TRUNCATION_RATE = 0.01


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(repo_root: Path) -> str:
    """Hash runner source/config code when a usable git checkout is absent."""

    digest = hashlib.sha256()
    roots = (repo_root / "src" / "goibert", repo_root / "configs" / "e1")
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in {".py", ".json"}
            )
    for path in sorted(files):
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def directory_hashes(root: Path, *, suffixes: set[str] | None = None) -> dict[str, str]:
    """Hash model/tokenizer files for an auditable input snapshot."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    allowed = suffixes or {".json", ".safetensors", ".txt", ".model", ".vocab"}
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in allowed
    }


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the runner and request deterministic kernels."""

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
        # Older torch builds may not expose this switch.  Metadata records the
        # version; no silent claim of bitwise determinism is made.
        pass


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is unavailable")
    return device


def bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(checker()) if checker is not None else False


def resolve_precision(precision: str, device: torch.device) -> str:
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16; implicit auto is disallowed")
    if precision == "bf16" and not bf16_supported(device):
        raise RuntimeError(f"bf16 was requested but is not supported on {device}")
    return precision


def make_autocast(precision: str, device: torch.device):
    if precision != "bf16":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def _is_no_decay(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    lowered = name.lower()
    return leaf == "bias" or leaf == "weight" and (
        "layernorm" in lowered or "layer_norm" in lowered or ".norm." in lowered
    )


def make_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    layer_norm_weights = {
        f"{module_name}.weight" if module_name else "weight"
        for module_name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name in layer_norm_weights or _is_no_decay(name) else decay).append(parameter)
    if not decay or not no_decay:
        raise RuntimeError("Expected both decayed and non-decayed parameter groups")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


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
    warning_over_512: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_train_lengths(tokenizer: Any, train_texts: Sequence[str]) -> LengthStats:
    if not train_texts:
        raise ValueError("Cannot select max_length from an empty training split")
    lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in train_texts
    ]
    values = np.asarray(lengths, dtype=np.int64)
    rate_256 = float(np.mean(values > 256))
    rate_512 = float(np.mean(values > 512))
    max_length = 256 if rate_256 <= MAX_TRAIN_TRUNCATION_RATE else 512
    warning = rate_512 > MAX_TRAIN_TRUNCATION_RATE
    return LengthStats(
        train_count=len(lengths),
        max_tokens=int(values.max()),
        p50_tokens=float(np.percentile(values, 50)),
        p95_tokens=float(np.percentile(values, 95)),
        p99_tokens=float(np.percentile(values, 99)),
        truncation_rate_256=rate_256,
        truncation_rate_512=rate_512,
        max_length=max_length,
        warning_over_512=warning,
    )


class EncodedTextDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        input_ids: list[list[int]],
        labels: Sequence[Any],
        ids: Sequence[Any],
        gold_labels: Sequence[Any] | None = None,
    ):
        if not (len(input_ids) == len(labels) == len(ids)):
            raise ValueError("input_ids, labels and ids must have equal lengths")
        if gold_labels is not None and len(gold_labels) != len(input_ids):
            raise ValueError("gold_labels must have the same length as input_ids")
        self.input_ids = input_ids
        self.labels = list(labels)
        self.ids = list(ids)
        self.gold_labels = list(gold_labels) if gold_labels is not None else self.labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
            "gold_labels": self.gold_labels[index],
            "id": self.ids[index],
        }


def encode_texts(tokenizer: Any, texts: Sequence[str], max_length: int) -> list[list[int]]:
    if not texts:
        return []
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )
    return [list(ids) for ids in encoded["input_ids"]]


def make_collate_fn(tokenizer: Any):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer.pad(
            [{"input_ids": row["input_ids"]} for row in rows],
            padding=True,
            return_tensors="pt",
        )
        labels = [row["labels"] for row in rows]
        gold_labels = [row["gold_labels"] for row in rows]
        first = labels[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)
            encoded["gold_labels"] = torch.tensor(gold_labels, dtype=torch.long)
        else:
            encoded["labels"] = torch.tensor(labels, dtype=torch.long)
            encoded["gold_labels"] = torch.tensor(gold_labels, dtype=torch.long)
        encoded["ids"] = [row["id"] for row in rows]
        return encoded

    return collate


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def split_field(bundle: Any, split: str) -> Any:
    splits = _field(bundle, "splits")
    if splits is not None:
        if isinstance(splits, Mapping):
            return splits[split]
        return getattr(splits, split)
    return _field(bundle, split)


def split_texts(split: Any) -> Sequence[str]:
    return _field(split, "texts", _field(split, "text"))


def split_labels(split: Any) -> Sequence[Any]:
    return _field(split, "labels", _field(split, "label"))


def split_ids(split: Any) -> Sequence[Any]:
    values = _field(split, "ids", _field(split, "example_ids", None))
    if values is None:
        return list(range(len(split_texts(split))))
    return values


def task_num_aspects(bundle: Any) -> int:
    return int(_field(bundle, "num_aspects", len(_field(bundle, "aspect_names", [])) or 0))


def task_aspect_class_counts(bundle: Any) -> tuple[int, ...]:
    counts = _field(
        bundle,
        "aspect_class_counts",
        _field(bundle, "slot_class_counts", None),
    )
    if counts is None:
        slot_names = _field(bundle, "slot_label_names", None)
        if slot_names is not None:
            counts = [len(names) for names in slot_names]
    if counts is None:
        count = task_num_aspects(bundle)
        return (ABSA_NUM_CLASSES,) * count
    values = tuple(int(value) for value in counts)
    if len(values) != task_num_aspects(bundle):
        raise ValueError("aspect_class_counts must have one entry per aspect slot")
    if any(value <= 0 for value in values):
        raise ValueError("aspect_class_counts entries must be positive")
    return values


def task_num_classes(bundle: Any) -> int:
    if task_num_aspects(bundle):
        return sum(task_aspect_class_counts(bundle))
    return int(
        _field(
            bundle,
            "num_classes",
            len(_field(bundle, "class_names", _field(bundle, "label_names", []))) or 0,
        )
    )


def task_metadata(bundle: Any) -> dict[str, Any]:
    value = _field(bundle, "metadata", {})
    metadata = dict(value) if isinstance(value, Mapping) else {}
    for name in ("label_policy", "pair_scored_aspects", "slot_label_names"):
        field = _field(bundle, name, None)
        if field is not None:
            metadata[name] = field
    return metadata


def bundle_provenance(bundle: Any, repo_root: Path | None = None) -> dict[str, Any]:
    value = _field(bundle, "provenance", _field(bundle, "source", {}))
    result = dict(value) if isinstance(value, Mapping) else {}
    result.setdefault("source_files", list(_field(bundle, "source_files", ()) or ()))
    result.setdefault("label_policy", _field(bundle, "label_policy", None))
    hashes: dict[str, str] = {}
    for value in result["source_files"]:
        path = Path(value)
        if path.is_file():
            hashes[str(path)] = sha256_file(path)
    result["source_file_hashes"] = hashes
    if repo_root is not None:
        evaluator_path = repo_root / "src" / "goibert" / "metrics.py"
        if evaluator_path.is_file():
            result["evaluator_code_hash"] = sha256_file(evaluator_path)
    return result


def load_task_bundle(
    task_id: str,
    repo_root: Path,
    *,
    include_test: bool,
    smoke: bool = False,
) -> Any:
    """Load the audited data contract without silently widening its scope."""

    from . import data  # local import keeps model smoke independent of datasets

    loader = getattr(data, "load_task", None)
    if not callable(loader):
        raise ImportError("goibert.data must expose load_task(task_id, repo_root, include_test=...)")
    return loader(task_id, repo_root, include_test=include_test, smoke=smoke)


def load_metric_function() -> Any:
    from . import metrics

    evaluator = getattr(metrics, "evaluate_task", None)
    if not callable(evaluator):
        raise ImportError("goibert.metrics must expose evaluate_task(...)")
    return evaluator


def evaluate_predictions(
    task_id: str,
    bundle: Any,
    gold: Sequence[Any],
    predictions: Sequence[Any],
    *,
    split: str,
) -> dict[str, Any]:
    evaluator = load_metric_function()
    aspects = list(_field(bundle, "aspect_names", []) or [])
    classes = list(
        _field(bundle, "class_names", _field(bundle, "label_names", [])) or []
    )
    kwargs = {
        "task_id": task_id,
        "gold": gold,
        "predictions": predictions,
        "aspect_names": aspects,
        "class_names": classes,
        "aspect_class_counts": list(task_aspect_class_counts(bundle)),
        "split": split,
    }
    raw = evaluator(**kwargs)
    if not isinstance(raw, Mapping):
        raise TypeError("Task evaluator must return a mapping of named metrics")
    return dict(raw)


def selection_score(metrics: Mapping[str, Any]) -> float:
    """Use the metric audit's literal, task-specific selection field."""

    value = metrics.get("selection_f1")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise KeyError("Evaluator must return finite selection_f1 for E1 selection")
    return float(value)


def compute_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    num_aspects: int,
    aspect_class_counts: Sequence[int] | None = None,
) -> Tensor:
    if num_aspects:
        counts = tuple(aspect_class_counts or (ABSA_NUM_CLASSES,) * num_aspects)
        if labels.ndim != 2 or labels.shape[1] != num_aspects:
            raise ValueError("ABSA labels must have shape [batch, aspects]")
        if len(set(counts)) == 1 and logits.ndim == 3:
            if logits.shape[:2] != labels.shape or logits.shape[-1] != counts[0]:
                raise ValueError("ABSA logits and labels have incompatible shapes")
            return nn.functional.cross_entropy(logits.reshape(-1, counts[0]), labels.reshape(-1))
        if logits.ndim != 2 or logits.shape[1] != sum(counts):
            raise ValueError("Variable-width ABSA logits must have shape [batch, sum(slot_classes)]")
        losses: list[Tensor] = []
        offset = 0
        for slot, width in enumerate(counts):
            slot_logits = logits[:, offset : offset + width]
            losses.append(nn.functional.cross_entropy(slot_logits, labels[:, slot]))
            offset += width
        return torch.stack(losses).mean()
    if labels.ndim != 1 or logits.ndim != 2:
        raise ValueError("Categorical labels/logits must be [batch] and [batch, classes]")
    return nn.functional.cross_entropy(logits, labels)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _predictions(logits: Tensor, *, aspect_class_counts: Sequence[int] = ()) -> list[Any]:
    if not aspect_class_counts:
        return logits.argmax(dim=-1).detach().cpu().tolist()
    counts = tuple(int(value) for value in aspect_class_counts)
    if logits.ndim == 3:
        return logits.argmax(dim=-1).detach().cpu().tolist()
    if logits.ndim != 2 or logits.shape[1] != sum(counts):
        raise ValueError("Variable-width ABSA logits do not match slot schema")
    output: list[list[int]] = []
    for row in logits.detach().cpu():
        offset = 0
        labels: list[int] = []
        for width in counts:
            labels.append(int(row[offset : offset + width].argmax().item()))
            offset += width
        output.append(labels)
    return output


def model_labels(bundle: Any, split_name: str, split: Any) -> list[Any]:
    adapter = getattr(bundle, "labels_for_model", None)
    if callable(adapter):
        return list(adapter(split_name))
    return list(split_labels(split))


def run_validation(
    model: GoiBERTClassifier,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    precision: str,
    task_id: str,
    bundle: Any,
    *,
    split_name: str = "validation",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    counts = task_aspect_class_counts(bundle)
    gold: list[Any] = []
    predicted: list[Any] = []
    ids: list[Any] = []
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, device)
            with make_autocast(precision, device):
                logits = model(
                    moved["input_ids"],
                    moved["attention_mask"],
                    token_type_ids=moved.get("token_type_ids"),
                )
            predicted.extend(_predictions(logits, aspect_class_counts=counts))
            gold.extend(batch["gold_labels"].tolist())
            ids.extend(batch["ids"])
    metrics = evaluate_predictions(task_id, bundle, gold, predicted, split=split_name)
    rows = [
        {"id": example_id, "gold": target, "prediction": prediction}
        for example_id, target, prediction in zip(ids, gold, predicted)
    ]
    return metrics, rows


def run_test(
    model: GoiBERTClassifier,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    precision: str,
    task_id: str,
    bundle: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # Test is deliberately a separate function and is only called by the
    # explicit locked-checkpoint evaluator.
    return run_validation(model, loader, device, precision, task_id, bundle, split_name="test")


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state()}
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_trial_checkpoint(
    path: Path,
    model: GoiBERTClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    epoch: int,
    global_step: int,
    metadata: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "rng_state": capture_rng_state(),
        "metadata": dict(metadata),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_trial_checkpoint(
    path: Path,
    model: GoiBERTClassifier,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    """Load a fully resumable checkpoint and fail closed on missing state."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "epoch", "global_step", "rng_state"}
    missing = required.difference(state)
    if missing:
        raise ValueError(f"Checkpoint {path} is not safely resumable; missing {sorted(missing)}")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    return state


def _runtime_metadata(device: torch.device, precision: str) -> dict[str, Any]:
    cuda = torch.version.cuda
    hip = getattr(torch.version, "hip", None)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": cuda,
        "hip": hip,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": precision,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _make_loader(
    tokenizer: Any,
    split: Any,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    labels: Sequence[Any] | None = None,
    gold_labels: Sequence[Any] | None = None,
) -> DataLoader[dict[str, Any]]:
    texts = list(split_texts(split))
    ids = list(split_ids(split))
    resolved_labels = list(labels) if labels is not None else list(split_labels(split))
    dataset = EncodedTextDataset(
        encode_texts(tokenizer, texts, max_length),
        resolved_labels,
        ids,
        gold_labels=list(gold_labels) if gold_labels is not None else list(split_labels(split)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=make_collate_fn(tokenizer),
    )


def _unique_trial_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}__rerun{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def train_trial(
    *,
    task_id: str,
    learning_rate: float,
    seed: int,
    repo_root: Path,
    output_dir: Path,
    device_name: str | None = None,
    precision: str = "fp32",
    attention_implementation: str = "sdpa",
    epochs: int = EPOCHS,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    micro_batch_size: int | None = None,
    trial_stage: str = "screening",
    smoke: bool = False,
    expected_source_hash: str | None = None,
) -> dict[str, Any]:
    """Run one independent E1 trial and return its machine-readable summary."""

    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown task_id {task_id!r}; expected one of {TASK_IDS}")
    if learning_rate not in LEARNING_RATES and not smoke:
        raise ValueError(f"Learning rate {learning_rate} is outside the locked E1 grid")
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    micro = micro_batch_size or effective_batch_size
    if micro <= 0 or effective_batch_size % micro:
        raise ValueError("micro_batch_size must divide effective_batch_size exactly")
    accumulation_steps = effective_batch_size // micro
    if trial_stage not in {"screening", "shortlist", "final"}:
        raise ValueError("train_trial is validation-only; test evaluation uses the locked evaluator")

    seed_everything(seed)
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    current_source_hash = source_hash(repo_root)
    if expected_source_hash is not None and current_source_hash != expected_source_hash:
        raise RuntimeError(
            "Runner source changed after grid launch: "
            f"expected {expected_source_hash}, found {current_source_hash}"
        )
    trial_dir = _unique_trial_dir(output_dir)
    trial_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()

    # The include_test=False call is a hard boundary: data.py must not read
    # or materialise the hidden test split during screening/shortlist/final.
    bundle = load_task_bundle(task_id, repo_root, include_test=False, smoke=smoke)
    train_split = split_field(bundle, "train")
    valid_split = split_field(bundle, "validation")
    if valid_split is None:
        valid_split = split_field(bundle, "valid")
    if train_split is None or valid_split is None:
        raise ValueError("Task bundle must expose train and validation/valid splits")

    tokenizer = AutoTokenizer.from_pretrained(
        repo_root / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    lengths = inspect_train_lengths(tokenizer, list(split_texts(train_split)))
    if lengths.warning_over_512:
        raise RuntimeError(
            f"{task_id} train truncation remains >1% at max_length=512 "
            f"({lengths.truncation_rate_512:.4%}); task is blocked pending root decision"
        )
    max_length = lengths.max_length
    num_aspects = task_num_aspects(bundle)
    aspect_class_counts = task_aspect_class_counts(bundle)
    num_classes = task_num_classes(bundle)
    if num_classes <= 0:
        raise ValueError(f"Data bundle for {task_id} has no class/aspect count")
    model, model_spec = build_classifier(
        repo_root / "models" / "goiturn-s-gd",
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts if num_aspects else None,
        local_files_only=True,
        attention_implementation=attention_implementation,
    )
    backbone_dir = repo_root / "models" / "goiturn-s-gd"
    backbone_hashes = directory_hashes(backbone_dir)
    tokenizer_hashes = {
        name: digest
        for name, digest in backbone_hashes.items()
        if name in {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        }
    }
    model.to(device)
    optimizer = make_optimizer(model, learning_rate)
    train_loader = _make_loader(
        tokenizer,
        train_split,
        max_length=max_length,
        batch_size=micro,
        shuffle=True,
        seed=seed,
        labels=model_labels(bundle, "train", train_split),
        gold_labels=split_labels(train_split),
    )
    valid_loader = _make_loader(
        tokenizer,
        valid_split,
        max_length=max_length,
        batch_size=micro,
        shuffle=False,
        seed=seed,
        labels=model_labels(bundle, "validation", valid_split),
        gold_labels=split_labels(valid_split),
    )
    total_steps = math.ceil(len(train_loader) / accumulation_steps) * epochs
    warmup_steps = round(total_steps * WARMUP_FRACTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    metadata: dict[str, Any] = {
        "task_id": task_id,
        "stage": trial_stage,
        "seed": seed,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation_steps,
        "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999],
        "adamw_epsilon": 1e-8,
        "warmup_fraction": WARMUP_FRACTION,
        "warmup_steps": warmup_steps,
        "gradient_clip_norm": 1.0,
        "head": "CLS -> dropout0.1 -> dense384 -> tanh -> dropout0.1 -> logits",
        "loss": "cross_entropy",
        "max_length": max_length,
        "length_stats": lengths.to_dict(),
        "include_test": False,
        "source_hash": current_source_hash,
        "data_provenance": bundle_provenance(bundle, repo_root),
        "task_metadata": task_metadata(bundle),
        "model": model_spec.to_dict(),
        "runtime": _runtime_metadata(device, precision),
        "attention_implementation": attention_implementation,
        "backbone_hashes": backbone_hashes,
        "tokenizer_hashes": tokenizer_hashes,
        "resumable": False,
        "resume_support": "fresh_rerun_only; no resume CLI",
        "git_commit": None,
        "argv": list(sys.argv),
    }
    write_json(trial_dir / "resolved_config.json", metadata)

    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_predictions: list[dict[str, Any]] = []
    epoch_curves: list[dict[str, Any]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            moved = _move_batch(batch, device)
            with make_autocast(precision, device):
                logits = model(
                    moved["input_ids"],
                    moved["attention_mask"],
                    token_type_ids=moved.get("token_type_ids"),
                )
                loss = compute_loss(
                    logits,
                    moved["labels"],
                    num_aspects=num_aspects,
                    aspect_class_counts=aspect_class_counts,
                )
                window_start = (batch_index // accumulation_steps) * accumulation_steps
                window_end = min(window_start + accumulation_steps, len(train_loader))
                window_examples = sum(
                    min(micro, len(train_loader.dataset) - index * micro)
                    for index in range(window_start, window_end)
                )
                scaled_loss = loss * (len(batch["labels"]) / window_examples)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
            scaled_loss.backward()
            train_loss_sum += float(loss.detach().cpu()) * len(batch["labels"])
            train_examples += len(batch["labels"])
            is_boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if is_boundary:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        val_metrics, val_predictions = run_validation(
            model, valid_loader, device, precision, task_id, bundle
        )
        score = selection_score(val_metrics)
        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss_sum / max(train_examples, 1),
            "validation": val_metrics,
            "selection_f1": score,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
        }
        epoch_curves.append(epoch_row)
        write_json(trial_dir / "epoch_curves.json", {"epochs": epoch_curves})
        # Ties retain the earlier epoch.  LR tie-breaking belongs to run_grid,
        # where scores from independent LR trials are available.
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_predictions = val_predictions
            save_trial_checkpoint(
                trial_dir / "best_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                global_step=global_step,
                metadata=metadata,
            )
            write_json(trial_dir / "best_validation_metrics.json", best_metrics)
            write_json(trial_dir / "validation_predictions.json", {"rows": best_predictions})

    metadata.update(
        {
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "best_selection_f1": best_score,
            "elapsed_seconds": time.time() - started,
            "peak_gpu_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "peak_gpu_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            ),
            "status": "completed",
        }
    )
    write_json(trial_dir / "metadata.json", metadata)
    return {
        "task_id": task_id,
        "stage": trial_stage,
        "seed": seed,
        "learning_rate": learning_rate,
        "artifact_dir": str(trial_dir),
        "best_epoch": best_epoch,
        "best_selection_f1": best_score,
        "validation_metrics": best_metrics,
        "status": "completed",
    }


def evaluate_locked_checkpoint(
    *,
    selection_manifest: Path,
    artifact_dir: Path,
    repo_root: Path,
    device_name: str | None = None,
    precision: str = "fp32",
    attention_implementation: str = "sdpa",
) -> dict[str, Any]:
    """Evaluate one preselected checkpoint after the explicit test gate.

    This function has no training path and refuses an unlocked or mismatched
    manifest.  The grid launcher invokes it only from its separate ``test``
    command; validation stages never call it or request ``include_test=True``.
    """

    manifest = read_json(selection_manifest)
    if manifest.get("protocol") != "E1-test-gate-v1" or manifest.get("locked") is not True:
        raise RuntimeError("Test evaluation requires a locked E1-test-gate-v1 manifest")
    launch_hash = manifest.get("source_hash")
    if not isinstance(launch_hash, str):
        raise RuntimeError("Selection manifest must contain source_hash")
    current_hash = source_hash(repo_root)
    if current_hash != launch_hash:
        raise RuntimeError(
            f"Runner source changed after selection lock: expected {launch_hash}, found {current_hash}"
        )
    selections = manifest.get("selections")
    if not isinstance(selections, list):
        raise RuntimeError("Selection manifest must contain a selections list")
    artifact_value = str(artifact_dir.resolve())
    selection = next(
        (row for row in selections if isinstance(row, Mapping) and str(Path(row.get("artifact_dir", "")).resolve()) == artifact_value),
        None,
    )
    if selection is None:
        raise RuntimeError(f"Checkpoint is absent from locked selection manifest: {artifact_dir}")
    metadata_path = artifact_dir / "metadata.json"
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Locked artifact is incomplete: {artifact_dir}")
    preexisting_test_files = tuple(
        path.name
        for path in (
            artifact_dir / "test_metrics.json",
            artifact_dir / "test_predictions.json",
            artifact_dir / "test_evaluation_metadata.json",
        )
        if path.exists()
    )
    if preexisting_test_files:
        raise RuntimeError(
            "Refusing to overwrite an artifact that already contains test outputs: "
            + ", ".join(preexisting_test_files)
        )
    metadata = read_json(metadata_path)
    if metadata.get("source_hash") != launch_hash:
        raise RuntimeError("Checkpoint metadata source hash differs from selection lock")
    task_id = str(metadata.get("task_id"))
    if task_id not in TASK_IDS or selection.get("task_id") != task_id:
        raise RuntimeError("Selection manifest/task metadata mismatch")
    expected_values = {
        "seed": metadata.get("seed"),
        "learning_rate": metadata.get("learning_rate"),
        "best_epoch": metadata.get("best_epoch"),
        "precision": _field(metadata.get("runtime", {}), "precision", None),
    }
    for key, expected in expected_values.items():
        if key in selection and selection[key] != expected:
            raise RuntimeError(f"Selection manifest {key} does not match checkpoint metadata")
    checkpoint_hash = sha256_file(checkpoint_path)
    metadata_hash = sha256_file(metadata_path)
    if "checkpoint_sha256" in selection and selection["checkpoint_sha256"] != checkpoint_hash:
        raise RuntimeError("Selection manifest checkpoint hash mismatch")
    if "metadata_sha256" in selection and selection["metadata_sha256"] != metadata_hash:
        raise RuntimeError("Selection manifest metadata hash mismatch")
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    bundle = load_task_bundle(task_id, repo_root, include_test=True)
    train_split = split_field(bundle, "train")
    test_split = split_field(bundle, "test")
    if train_split is None or test_split is None:
        raise ValueError("Locked test evaluation requires train metadata and test split")
    tokenizer = AutoTokenizer.from_pretrained(
        repo_root / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    length_stats = inspect_train_lengths(tokenizer, list(split_texts(train_split)))
    if int(metadata.get("max_length", -1)) != length_stats.max_length:
        raise RuntimeError("Locked max_length no longer matches train-only length audit")
    num_aspects = task_num_aspects(bundle)
    aspect_class_counts = task_aspect_class_counts(bundle)
    current_provenance = bundle_provenance(bundle, repo_root)
    if "data_provenance" in selection and selection["data_provenance"] != current_provenance:
        raise RuntimeError("Selection manifest data provenance/hash mismatch")
    current_backbone_hashes = directory_hashes(repo_root / "models" / "goiturn-s-gd")
    if metadata.get("backbone_hashes") != current_backbone_hashes:
        raise RuntimeError("Backbone files changed after validation selection")
    expected_tokenizer_hashes = metadata.get("tokenizer_hashes")
    current_tokenizer_hashes = {
        name: digest
        for name, digest in current_backbone_hashes.items()
        if name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    }
    if expected_tokenizer_hashes != current_tokenizer_hashes:
        raise RuntimeError("Tokenizer files changed after validation selection")
    if "backbone_hashes" in selection and selection["backbone_hashes"] != current_backbone_hashes:
        raise RuntimeError("Selection manifest backbone hash mismatch")
    if "tokenizer_hashes" in selection and selection["tokenizer_hashes"] != current_tokenizer_hashes:
        raise RuntimeError("Selection manifest tokenizer hash mismatch")
    model, model_spec = build_classifier(
        repo_root / "models" / "goiturn-s-gd",
        num_classes=task_num_classes(bundle),
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts if num_aspects else None,
        local_files_only=True,
        attention_implementation=attention_implementation,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError("Locked checkpoint does not contain a model state")
    model.load_state_dict(state["model"])
    model.to(device)
    test_loader = _make_loader(
        tokenizer,
        test_split,
        max_length=length_stats.max_length,
        batch_size=int(metadata.get("micro_batch_size", EFFECTIVE_BATCH_SIZE)),
        shuffle=False,
        seed=int(metadata.get("seed", 42)),
        labels=model_labels(bundle, "test", test_split),
        gold_labels=split_labels(test_split),
    )
    metrics, predictions = run_test(model, test_loader, device, precision, task_id, bundle)
    write_json(artifact_dir / "test_metrics.json", metrics)
    write_json(artifact_dir / "test_predictions.json", {"rows": predictions})
    evaluation = {
        "protocol": "E1-test-gate-v1",
        "task_id": task_id,
        "seed": metadata.get("seed"),
        "learning_rate": metadata.get("learning_rate"),
        "artifact_dir": str(artifact_dir.resolve()),
        "source_hash": launch_hash,
        "metadata_sha256": metadata_hash,
        "checkpoint_sha256": checkpoint_hash,
        "evaluator_metadata": current_provenance,
        "backbone_hashes": current_backbone_hashes,
        "tokenizer_hashes": current_tokenizer_hashes,
        "model": model_spec.to_dict(),
        "runtime": _runtime_metadata(device, precision),
        "test_metrics": metrics,
        "status": "completed",
    }
    write_json(artifact_dir / "test_evaluation_metadata.json", evaluation)
    return evaluation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one GoiBERT supervised E1 trial")
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--micro-batch-size", type=int, default=EFFECTIVE_BATCH_SIZE)
    parser.add_argument("--stage", choices=("screening", "shortlist", "final"), default="screening")
    parser.add_argument("--expected-source-hash", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = train_trial(
        task_id=args.task_id,
        learning_rate=args.learning_rate,
        seed=args.seed,
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        device_name=args.device,
        precision=args.precision,
        attention_implementation=args.attention_implementation,
        epochs=2 if args.smoke else args.epochs,
        micro_batch_size=4 if args.smoke else args.micro_batch_size,
        trial_stage=args.stage,
        smoke=args.smoke,
        expected_source_hash=args.expected_source_hash,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
