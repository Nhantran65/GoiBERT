#!/usr/bin/env python3
"""Run the frozen D1 layerwise linear-probe diagnostic.

The encoder representation caches are produced by the separate D1 extraction
worker.  This script deliberately never loads an encoder and never reads the
test split.  It consumes only ``artifacts/d1/cache`` and fits one linear head
per cached state, using the same 200 full-batch AdamW updates for every state.

The output is a compact, auditable diagnostic rather than a model-selection
runner.  In particular, a low validation score is not interpreted as evidence
against a representation when the fixed probe is still under-converged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
MODELS = ("student", "teacher")
VIEWS = ("cls", "mean")
TRAIN_COUNT = 2000
SEED = 42
LABEL_SHUFFLE_SEED = 314159
STEPS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
STANDARDIZE_EPSILON = 1e-6

VISFD_OTHERS_INDEX = 5
VISFD_OTHERS_RAW_CODE = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_manifest(model_dir: Path) -> Path:
    candidates = (
        model_dir / "manifest.json",
        model_dir / "cache_manifest.json",
        model_dir / "metadata.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No completed cache manifest found under {model_dir}; tried "
        + ", ".join(str(path.name) for path in candidates)
    )


def _manifest_status(manifest: Mapping[str, Any]) -> str:
    for key in ("status", "state", "cache_status"):
        value = manifest.get(key)
        if isinstance(value, str):
            return value.lower()
    return ""


def validate_manifest(path: Path, *, task: str, model: str) -> dict[str, Any]:
    manifest = read_json(path)
    status = _manifest_status(manifest)
    if status != "completed":
        raise RuntimeError(f"Cache manifest is not completed: {path} (status={status!r})")
    manifest_task = manifest.get("task_id", manifest.get("task"))
    manifest_model = manifest.get("model")
    if manifest_task is not None and str(manifest_task) != task:
        raise RuntimeError(f"Cache manifest task mismatch in {path}: {manifest_task!r}")
    if manifest_model is not None and str(manifest_model) != model:
        raise RuntimeError(f"Cache manifest model mismatch in {path}: {manifest_model!r}")
    return manifest


def _split_metadata(dataset: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    splits = dataset.get("splits")
    if isinstance(splits, Mapping) and isinstance(splits.get(split), Mapping):
        return splits[split]  # type: ignore[return-value]
    value = dataset.get(split)
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"dataset.json has no {split!r} metadata")


def load_dataset_contract(task_dir: Path, *, task: str) -> dict[str, Any]:
    candidates = (task_dir / "dataset.json", task_dir.parent / "dataset.json")
    dataset_path = next((path for path in candidates if path.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError(
            f"No dataset.json for {task}; tried " + ", ".join(str(path) for path in candidates)
        )
    dataset = read_json(dataset_path)
    task_value = dataset.get("task_id", dataset.get("task"))
    if task_value is not None and str(task_value) != task:
        raise RuntimeError(f"dataset.json task mismatch: expected {task}, found {task_value!r}")
    for split in ("train", "validation"):
        metadata = _split_metadata(dataset, split)
        ids = metadata.get("ids", metadata.get("example_ids"))
        golds = metadata.get("golds", metadata.get("raw_gold_labels", metadata.get("labels", metadata.get("gold"))))
        if not isinstance(ids, list) or not isinstance(golds, list):
            raise ValueError(f"dataset.json {split} must contain list ids and golds/labels")
        if len(ids) != len(golds):
            raise ValueError(f"dataset.json {split} ids/golds length mismatch")
        metadata = dict(metadata)
        metadata["ids"] = ids
        metadata["golds"] = golds
        if split == "train" and len(ids) != TRAIN_COUNT:
            raise ValueError(
                f"D1 probe requires exactly {TRAIN_COUNT} cached train rows; found {len(ids)}"
            )
        dataset[split] = metadata
    dataset["dataset_path"] = str(dataset_path.resolve())
    dataset["dataset_sha256"] = sha256_file(dataset_path)
    return dataset


def _array_path(model_dir: Path, split: str, view: str) -> Path:
    return model_dir / f"{split}_{view}.npy"


def load_features(
    model_dir: Path,
    *,
    task: str,
    model: str,
    view: str,
    dataset: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path_train = _array_path(model_dir, "train", view)
    path_valid = _array_path(model_dir, "validation", view)
    if not path_train.is_file() or not path_valid.is_file():
        raise FileNotFoundError(f"Missing {view} feature arrays under {model_dir}")
    train = np.load(path_train, mmap_mode="r")
    valid = np.load(path_valid, mmap_mode="r")
    if train.ndim != 3 or valid.ndim != 3:
        raise ValueError(f"Expected [layers, rows, hidden] arrays for {task}/{model}/{view}")
    if train.shape[0] != valid.shape[0] or train.shape[2] != valid.shape[2]:
        raise ValueError(f"Train/validation feature shapes disagree for {task}/{model}/{view}")
    if train.shape[1] != len(dataset["train"]["ids"]):
        raise ValueError(f"Train feature rows disagree with dataset.json for {task}/{model}/{view}")
    if valid.shape[1] != len(dataset["validation"]["ids"]):
        raise ValueError(f"Validation feature rows disagree with dataset.json for {task}/{model}/{view}")
    if not np.isfinite(np.asarray(train[: min(1, train.shape[0]), : min(8, train.shape[1])])).all():
        raise ValueError(f"Non-finite probe features in {path_train}")
    if not np.isfinite(np.asarray(valid[: min(1, valid.shape[0]), : min(8, valid.shape[1])])).all():
        raise ValueError(f"Non-finite probe features in {path_valid}")
    return train, valid, {
        "train_path": str(path_train.resolve()),
        "validation_path": str(path_valid.resolve()),
        "train_sha256": sha256_file(path_train),
        "validation_sha256": sha256_file(path_valid),
        "train_shape": list(train.shape),
        "validation_shape": list(valid.shape),
    }


def _raw_label_to_model(task: str, labels: Sequence[Any]) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if task == "visfd":
        if values.ndim != 2 or values.shape[1] != 11:
            raise ValueError(f"ViSFD labels must be [N,11], got {values.shape}")
        values = values.copy()
        others = values[:, VISFD_OTHERS_INDEX]
        if not np.isin(others, (0, VISFD_OTHERS_RAW_CODE)).all():
            raise ValueError("ViSFD OTHERS raw labels must be 0 or 4")
        values[:, VISFD_OTHERS_INDEX] = (others == VISFD_OTHERS_RAW_CODE).astype(np.int64)
        return values
    return values


def _task_widths(task: str, labels: np.ndarray) -> tuple[int, ...]:
    if task == "vsfc_sentiment":
        return (3,)
    if task == "vsfc_topic":
        return (4,)
    if task == "visfd":
        return (4,) * 5 + (2,) + (4,) * 5
    if task == "absa_restaurant":
        return (4,) * 12
    if task == "absa_hotel":
        return (4,) * 34
    raise ValueError(task)


def _as_label_tensor(task: str, labels: Sequence[Any]) -> Tensor:
    values = _raw_label_to_model(task, labels)
    return torch.from_numpy(values.astype(np.int64, copy=False))


def _is_absa(task: str) -> bool:
    return task in {"visfd", "absa_restaurant", "absa_hotel"}


def majority_predictions(task: str, train_labels: np.ndarray, n_valid: int) -> list[Any]:
    if not _is_absa(task):
        value = int(np.bincount(train_labels.astype(np.int64)).argmax())
        return [value] * n_valid
    counts = []
    for column in range(train_labels.shape[1]):
        counts.append(int(np.bincount(train_labels[:, column]).argmax()))
    return [list(counts) for _ in range(n_valid)]


def _make_head(hidden: int, widths: Sequence[int]) -> nn.ModuleList:
    heads = nn.ModuleList([nn.Linear(hidden, int(width)) for width in widths])
    for head in heads:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return heads


def _loss_and_predictions(
    heads: nn.ModuleList,
    x: Tensor,
    labels: Tensor,
    widths: Sequence[int],
) -> tuple[Tensor, list[Any]]:
    logits = [head(x) for head in heads]
    losses = [
        nn.functional.cross_entropy(
            logit,
            labels if len(widths) == 1 and labels.ndim == 1 else labels[:, i],
        )
        for i, logit in enumerate(logits)
    ]
    loss = torch.stack(losses).mean()
    if len(widths) == 1:
        predictions = logits[0].argmax(dim=-1).detach().cpu().tolist()
    else:
        predictions = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1).detach().cpu().tolist()
    return loss, predictions


def _normalize_features(
    train: np.ndarray,
    valid: np.ndarray,
    *,
    device: torch.device,
    epsilon: float = STANDARDIZE_EPSILON,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    train_np = np.asarray(train, dtype=np.float32)
    valid_np = np.asarray(valid, dtype=np.float32)
    mean = train_np.mean(axis=0, dtype=np.float64).astype(np.float32)
    variance = ((train_np.astype(np.float64) - mean.astype(np.float64)) ** 2).mean(axis=0)
    std = np.sqrt(variance + epsilon).astype(np.float32)
    hidden = train_np.shape[1]
    train_norm = (train_np - mean) / std / math.sqrt(hidden)
    valid_norm = (valid_np - mean) / std / math.sqrt(hidden)
    stats_payload = {
        "epsilon": epsilon,
        "hidden_size": int(hidden),
        "mean_sha256": sha256_bytes(mean.tobytes()),
        "std_sha256": sha256_bytes(std.tobytes()),
        "train_mean_abs_mean": float(np.mean(np.abs(mean))),
        "train_std_min": float(np.min(std)),
        "train_std_max": float(np.max(std)),
        "divide_by_sqrt_hidden": True,
        "divide_by_sqrt_hidden_value": math.sqrt(hidden),
    }
    return (
        torch.from_numpy(train_norm).to(device=device),
        torch.from_numpy(valid_norm).to(device=device),
        stats_payload,
    )


def fit_probe(
    train: np.ndarray,
    valid: np.ndarray,
    train_labels: Tensor,
    valid_labels: Tensor,
    widths: Sequence[int],
    *,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    set_seed(seed)
    train_x, valid_x, norm_stats = _normalize_features(train, valid, device=device)
    heads = _make_head(train_x.shape[1], widths).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses: list[float] = []
    started = time.time()
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _loss_and_predictions(heads, train_x, train_labels.to(device), widths)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite linear-probe loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        train_loss, _ = _loss_and_predictions(heads, train_x, train_labels.to(device), widths)
        valid_loss, predictions = _loss_and_predictions(heads, valid_x, valid_labels.to(device), widths)
    last20_start = losses[-20]
    last20_end = losses[-1]
    checkpoint_step = max(1, STEPS - 20)
    checkpoint_loss = losses[checkpoint_step - 1]
    last20_relative_improvement = (last20_start - last20_end) / max(abs(last20_start), 1e-12)
    convergence = {
        "steps": STEPS,
        "initial_train_ce": losses[0],
        f"loss_at_step_{checkpoint_step}": checkpoint_loss,
        "final_step_train_ce": last20_end,
        "post_update_train_ce": float(train_loss.detach().cpu()),
        "last20_mean_train_ce": float(np.mean(losses[-20:])),
        "last20_step_improvement": float(last20_start - last20_end),
        "last20_relative_improvement": float(last20_relative_improvement),
        "last20_range": float(max(losses[-20:]) - min(losses[-20:])),
        "learning_failed_flag": bool(losses[0] - np.mean(losses[-20:]) < 1e-3),
        "underconverged_flag": bool(last20_relative_improvement > 0.01),
    }
    return (
        {
            "train_ce": float(train_loss.detach().cpu()),
            "validation_ce": float(valid_loss.detach().cpu()),
            "convergence": convergence,
            "normalization": norm_stats,
            "elapsed_seconds": time.time() - started,
            "probe_parameter_count": int(sum(parameter.numel() for parameter in heads.parameters())),
        },
        predictions,
        norm_stats,
    )


def evaluate(task: str, gold: Sequence[Any], predictions: Sequence[Any]) -> dict[str, Any]:
    # The official evaluator is imported only here.  It receives raw gold
    # labels and model-space predictions; its ViSFD adapter maps OTHERS back to
    # raw code 4 and excludes it from polarity-pair scoring.
    from goibert.metrics import evaluate_task

    if task in {"vsfc_sentiment", "vsfc_topic"}:
        aspect_names: list[str] = []
    elif task == "visfd":
        aspect_names = [
            "BATTERY",
            "CAMERA",
            "DESIGN",
            "FEATURES",
            "GENERAL",
            "OTHERS",
            "PERFORMANCE",
            "PRICE",
            "SCREEN",
            "SER&ACC",
            "STORAGE",
        ]
    elif task == "absa_restaurant":
        aspect_names = [
            "AMBIENCE#GENERAL",
            "DRINKS#PRICES",
            "DRINKS#QUALITY",
            "DRINKS#STYLE&OPTIONS",
            "FOOD#PRICES",
            "FOOD#QUALITY",
            "FOOD#STYLE&OPTIONS",
            "LOCATION#GENERAL",
            "RESTAURANT#GENERAL",
            "RESTAURANT#MISCELLANEOUS",
            "RESTAURANT#PRICES",
            "SERVICE#GENERAL",
        ]
    else:
        aspect_names = [
            "FACILITIES#CLEANLINESS",
            "FACILITIES#COMFORT",
            "FACILITIES#DESIGN&FEATURES",
            "FACILITIES#GENERAL",
            "FACILITIES#MISCELLANEOUS",
            "FACILITIES#PRICES",
            "FACILITIES#QUALITY",
            "FOOD&DRINKS#MISCELLANEOUS",
            "FOOD&DRINKS#PRICES",
            "FOOD&DRINKS#QUALITY",
            "FOOD&DRINKS#STYLE&OPTIONS",
            "HOTEL#CLEANLINESS",
            "HOTEL#COMFORT",
            "HOTEL#DESIGN&FEATURES",
            "HOTEL#GENERAL",
            "HOTEL#MISCELLANEOUS",
            "HOTEL#PRICES",
            "HOTEL#QUALITY",
            "LOCATION#GENERAL",
            "ROOMS#CLEANLINESS",
            "ROOMS#COMFORT",
            "ROOMS#DESIGN&FEATURES",
            "ROOMS#GENERAL",
            "ROOMS#MISCELLANEOUS",
            "ROOMS#PRICES",
            "ROOMS#QUALITY",
            "ROOM_AMENITIES#CLEANLINESS",
            "ROOM_AMENITIES#COMFORT",
            "ROOM_AMENITIES#DESIGN&FEATURES",
            "ROOM_AMENITIES#GENERAL",
            "ROOM_AMENITIES#MISCELLANEOUS",
            "ROOM_AMENITIES#PRICES",
            "ROOM_AMENITIES#QUALITY",
            "SERVICE#GENERAL",
        ]
    return dict(
        evaluate_task(
            task_id=task,
            gold=list(gold),
            predictions=list(predictions),
            aspect_names=aspect_names,
            class_names=[],
            aspect_class_counts=_task_widths(task, _raw_label_to_model(task, gold)),
            split="validation",
        )
    )


def prediction_hash(predictions: Sequence[Any]) -> str:
    payload = json.dumps(predictions, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def prediction_counts(task: str, predictions: Sequence[Any]) -> Any:
    if not _is_absa(task):
        values, counts = np.unique(np.asarray(predictions, dtype=np.int64), return_counts=True)
        return {str(int(value)): int(count) for value, count in zip(values, counts)}
    matrix = np.asarray(predictions, dtype=np.int64)
    return [
        {str(int(value)): int(count) for value, count in zip(*np.unique(matrix[:, i], return_counts=True))}
        for i in range(matrix.shape[1])
    ]


def run_task(task: str, cache_root: Path, output_root: Path, device: torch.device) -> dict[str, Any]:
    task_cache = cache_root / task
    task_out = output_root / task
    task_out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_contract(task_cache, task=task)
    manifests: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_dir = task_cache / model
        manifest_path = find_manifest(model_dir)
        manifests[model] = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "metadata": validate_manifest(manifest_path, task=task, model=model),
        }
        train_cls, valid_cls, hashes = load_features(
            model_dir, task=task, model=model, view="cls", dataset=dataset
        )
        # Mean caches are required by the D1 contract for CKA, but this probe
        # branch intentionally does not fit a second pooling hypothesis.
        for view in ("mean",):
            mean_train_path = _array_path(model_dir, "train", view)
            mean_valid_path = _array_path(model_dir, "validation", view)
            if not mean_train_path.is_file() or not mean_valid_path.is_file():
                raise FileNotFoundError(f"Missing required mean cache for {task}/{model}")
        loaded[model] = {"train": train_cls, "validation": valid_cls, "hashes": hashes}

    raw_train = np.asarray(dataset["train"]["golds"], dtype=np.int64)
    raw_valid = np.asarray(dataset["validation"]["golds"], dtype=np.int64)
    train_labels = _raw_label_to_model(task, raw_train)
    valid_labels = _raw_label_to_model(task, raw_valid)
    widths = _task_widths(task, train_labels)
    majority = majority_predictions(task, train_labels, len(valid_labels))
    majority_metrics = evaluate(task, dataset["validation"]["golds"], majority)
    permutation = np.random.default_rng(LABEL_SHUFFLE_SEED).permutation(len(train_labels))
    results: dict[str, Any] = {
        "task_id": task,
        "protocol": "D1-layerwise-linear-probe-v1",
        "train_count": int(len(train_labels)),
        "validation_count": int(len(valid_labels)),
        "train_ids_sha256": sha256_bytes(
            json.dumps(dataset["train"]["ids"], ensure_ascii=False, separators=(",", ":")).encode()
        ),
        "validation_ids_sha256": sha256_bytes(
            json.dumps(dataset["validation"]["ids"], ensure_ascii=False, separators=(",", ":")).encode()
        ),
        "label_schema": {"widths": list(widths), "task": task, "visfd_others_model_mapping": task == "visfd"},
        "probe": {
            "seed": SEED,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "full_batch": True,
            "standardization": "per-feature train mean/std with epsilon=1e-6, then divide by sqrt(hidden_size)",
            "selection": "final step only; no validation epoch/LR selection",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
        "dataset": {key: value for key, value in dataset.items() if key not in {"train", "validation"}},
        "cache_manifests": manifests,
        "majority_baseline": {
            "validation_metrics": majority_metrics,
            "prediction_counts": prediction_counts(task, majority),
            "prediction_sha256": prediction_hash(majority),
        },
        "models": {},
        "label_shuffle_control": {},
    }
    predictions_out: dict[str, Any] = {"protocol": results["protocol"], "task_id": task, "models": {}}
    for model in MODELS:
        train_cache = loaded[model]["train"]
        valid_cache = loaded[model]["validation"]
        layer_rows: list[dict[str, Any]] = []
        layer_predictions: list[Any] = []
        for layer in range(train_cache.shape[0]):
            fit_summary, predictions, _ = fit_probe(
                train_cache[layer],
                valid_cache[layer],
                torch.from_numpy(train_labels),
                torch.from_numpy(valid_labels),
                widths,
                device=device,
                seed=SEED,
            )
            metrics = evaluate(task, dataset["validation"]["golds"], predictions)
            row = {
                "layer_index": layer,
                "state_count": int(train_cache.shape[0]),
                "hidden_size": int(train_cache.shape[2]),
                "relative_depth": float(layer / max(train_cache.shape[0] - 1, 1)),
                "validation_metrics": metrics,
                "selection_f1": float(metrics["selection_f1"]),
                "prediction_counts": prediction_counts(task, predictions),
                "prediction_sha256": prediction_hash(predictions),
                "fit": fit_summary,
            }
            layer_rows.append(row)
            layer_predictions.append(predictions)
        results["models"][model] = {
            "representation": "CLS",
            "hidden_size": int(train_cache.shape[2]),
            "state_count": int(train_cache.shape[0]),
            "probe_parameter_count_per_layer": int(sum(width * (train_cache.shape[2] + 1) for width in widths)),
            "cache_hashes": loaded[model]["hashes"],
            "layers": layer_rows,
        }
        predictions_out["models"][model] = {
            "layer_predictions": layer_predictions,
            "layer_prediction_sha256": [prediction_hash(prediction) for prediction in layer_predictions],
        }
        # Label-shuffle control is intentionally restricted to the final state.
        final_layer = train_cache.shape[0] - 1
        shuffled_train = train_labels[permutation]
        control_fit, control_predictions, _ = fit_probe(
            train_cache[final_layer],
            valid_cache[final_layer],
            torch.from_numpy(shuffled_train),
            torch.from_numpy(valid_labels),
            widths,
            device=device,
            seed=SEED,
        )
        control_metrics = evaluate(task, dataset["validation"]["golds"], control_predictions)
        results["label_shuffle_control"][model] = {
            "layer_index": final_layer,
            "shuffle_seed": LABEL_SHUFFLE_SEED,
            "permutation_sha256": sha256_bytes(permutation.tobytes()),
            "validation_metrics": control_metrics,
            "selection_f1": float(control_metrics["selection_f1"]),
            "prediction_counts": prediction_counts(task, control_predictions),
            "prediction_sha256": prediction_hash(control_predictions),
            "fit": control_fit,
        }
    write_json(task_out / "probe_results.json", results)
    np.save(task_out / "validation_predictions.npy", np.asarray(
        [predictions_out["models"][model]["layer_predictions"] for model in MODELS], dtype=object
    ))
    predictions_out["array_path"] = str((task_out / "validation_predictions.npy").resolve())
    predictions_out["array_sha256"] = sha256_file(task_out / "validation_predictions.npy")
    write_json(task_out / "validation_predictions.json", predictions_out)
    return results


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _save_rgb_png(path: Path, pixels: np.ndarray) -> None:
    """Write an RGB PNG with only the Python standard library."""
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("PNG pixels must be uint8 [height,width,3]")
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _draw_line(pixels: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int], width: int = 2) -> None:
    height, width_px, _ = pixels.shape
    steps = max(int(abs(x1 - x0)), int(abs(y1 - y0)), 1)
    radius = max(width // 2, 0)
    for index in range(steps + 1):
        fraction = index / steps
        x = int(round(x0 + fraction * (x1 - x0)))
        y = int(round(y0 + fraction * (y1 - y0)))
        if x < 0 or x >= width_px or y < 0 or y >= height:
            continue
        pixels[max(0, y - radius) : min(height, y + radius + 1), max(0, x - radius) : min(width_px, x + radius + 1)] = color


def _draw_circle(pixels: np.ndarray, x: float, y: float, radius: int, color: tuple[int, int, int]) -> None:
    height, width, _ = pixels.shape
    xx0, xx1 = max(0, int(x - radius)), min(width, int(x + radius + 1))
    yy0, yy1 = max(0, int(y - radius)), min(height, int(y + radius + 1))
    for yy in range(yy0, yy1):
        for xx in range(xx0, xx1):
            if (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2:
                pixels[yy, xx] = color


def _svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_plots(all_results: Mapping[str, Any], output_root: Path) -> None:
    """Create a dependency-free SVG and a matching RGB PNG overview."""
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 820
    margin_x, margin_y = 80, 70
    panel_w, panel_h = 430, 300
    gap_x, gap_y = 45, 65
    colors = {"student": (31, 119, 180), "teacher": (214, 39, 40)}
    hex_colors = {key: "#%02x%02x%02x" % value for key, value in colors.items()}
    panels: list[str] = []
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    completed_tasks = [task for task in TASKS if task in all_results]
    for panel_index, task in enumerate(completed_tasks):
        row, column = divmod(panel_index, 3)
        x0 = margin_x + column * (panel_w + gap_x)
        y0 = margin_y + row * (panel_h + gap_y)
        plot_left, plot_top = x0 + 50, y0 + 30
        plot_right, plot_bottom = x0 + panel_w - 15, y0 + panel_h - 45
        panels.append(f'<text x="{x0}" y="{y0 + 15}" font-size="16" font-family="sans-serif" font-weight="bold">{_svg_escape(task)}</text>')
        panels.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#333"/>')
        panels.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#333"/>')
        for model in MODELS:
            rows = all_results[task]["models"][model]["layers"]
            xs = [float(item["layer_index"]) for item in rows]
            ys = [100.0 * float(item["selection_f1"]) for item in rows]
            ymin = min(0.0, min(ys) - 3.0)
            ymax = max(100.0, max(ys) + 3.0)
            xmax = max(xs) if xs else 1.0
            points: list[str] = []
            for x, y in zip(xs, ys):
                px = plot_left + (x / max(xmax, 1.0)) * (plot_right - plot_left)
                py = plot_bottom - ((y - ymin) / max(ymax - ymin, 1.0)) * (plot_bottom - plot_top)
                points.append(f"{px:.1f},{py:.1f}")
            panels.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{hex_colors[model]}" stroke-width="2"/>')
            for point in points:
                px, py = point.split(",")
                panels.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{hex_colors[model]}"/>')
            # Raster drawing uses the same coordinates.  The plot range is
            # intentionally local to each panel, like the SVG range above.
            for index in range(len(xs) - 1):
                pxa = plot_left + (xs[index] / max(xmax, 1.0)) * (plot_right - plot_left)
                pya = plot_bottom - ((ys[index] - ymin) / max(ymax - ymin, 1.0)) * (plot_bottom - plot_top)
                pxb = plot_left + (xs[index + 1] / max(xmax, 1.0)) * (plot_right - plot_left)
                pyb = plot_bottom - ((ys[index + 1] - ymin) / max(ymax - ymin, 1.0)) * (plot_bottom - plot_top)
                _draw_line(pixels, pxa, pya, pxb, pyb, colors[model], 3)
                _draw_circle(pixels, pxa, pya, 3, colors[model])
            if xs:
                pxf = plot_left + (xs[-1] / max(xmax, 1.0)) * (plot_right - plot_left)
                pyf = plot_bottom - ((ys[-1] - ymin) / max(ymax - ymin, 1.0)) * (plot_bottom - plot_top)
                _draw_circle(pixels, pxf, pyf, 3, colors[model])
        panels.append(f'<text x="{plot_left}" y="{plot_bottom + 25}" font-size="10" font-family="sans-serif">state 0</text>')
        panels.append(f'<text x="{plot_right - 35}" y="{plot_bottom + 25}" font-size="10" font-family="sans-serif">final</text>')
        panels.append(f'<text x="{plot_left}" y="{plot_top - 8}" font-size="10" font-family="sans-serif">Validation F1 (percent)</text>')
        # Simple raster axes; labels are carried by the SVG version.
        _draw_line(pixels, plot_left, plot_bottom, plot_right, plot_bottom, (60, 60, 60), 2)
        _draw_line(pixels, plot_left, plot_top, plot_left, plot_bottom, (60, 60, 60), 2)
    legend = '<text x="80" y="790" font-size="13" font-family="sans-serif">Blue=student (H=384, 11 states) · Red=teacher (H=768, 13 states). Compare relative depth; local state indices are not aligned.</text>'
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="80" y="35" font-size="20" font-family="sans-serif" font-weight="bold">D1 CLS linear probes: layerwise validation performance</text>{"".join(panels)}{legend}</svg>\n'
    (plot_dir / "layerwise_probe_f1.svg").write_text(svg, encoding="utf-8")
    _save_rgb_png(plot_dir / "layerwise_probe_f1.png", pixels)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    cache_root = (args.cache_root or repo_root / "artifacts" / "d1" / "cache").resolve()
    output_root = (args.output_root or repo_root / "artifacts" / "d1" / "probes").resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    started = time.time()
    all_results: dict[str, Any] = {}
    for task in args.tasks:
        all_results[task] = run_task(task, cache_root, output_root, device)
        write_json(output_root / "progress.json", {"status": "running", "completed_tasks": list(all_results)})
    make_plots(all_results, output_root)
    summary = {
        "protocol": "D1-layerwise-linear-probe-v1",
        "status": "completed",
        "tasks": list(all_results),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "results": {
            task: {
                model: {
                    "best_selection_f1": max(row["selection_f1"] for row in all_results[task]["models"][model]["layers"]),
                    "final_selection_f1": all_results[task]["models"][model]["layers"][-1]["selection_f1"],
                    "final_layer": len(all_results[task]["models"][model]["layers"]) - 1,
                }
                for model in MODELS
            }
            for task in all_results
        },
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "progress.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
