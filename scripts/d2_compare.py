#!/usr/bin/env python3
"""Compare D1 (before fine-tuning) and D2 (after fine-tuning) diagnostics.

This module only consumes representation caches and JSON probe artifacts.  It
never constructs an encoder, opens a dataset loader, or reads the test split.
The D2 after-cache is expected to have the D1 cache schema under
``artifacts/d2/cache/<task>/<student|teacher>``.  The layerwise 200-step
probes themselves are deliberately delegated to ``d1_linear_probes.py`` so
that the D1 fitting contract is reused byte-for-byte.

Subcommands:

``compare``
    Compare existing D1 and D2 200-step probe results and write compact JSON
    and CSV tables.
``sensitivity``
    Fit only the fixed final state of every D2 cache with the existing D1
    1,000-step probe helper.  There is no post-hoc layer/epoch selection.
``cka``
    Compute train-only BEFORE/AFTER same-model CKA and the AFTER
    student/teacher cross-depth matrix on the exact D1 sample (seed 123,
    1,024 rows).  CLS is primary; content-mean is retained as a secondary
    view.  No row-permutation control is rerun here; D1's passed control is
    referenced in the output metadata.
``plots``
    Render standard Matplotlib figures from JSON/CKA artifacts.
``boundary_sensitivity``
    Exploratory, post-hoc 1,000-step fits at the requested middle-layer
    boundary for ViSFD and Restaurant only.  This is not part of the fixed
    final-layer sensitivity sweep.
``weight_drift``
    Re-aggregate per-parameter checkpoint deltas from the extraction
    manifests with explicit attention/FFN/LayerNorm grouping.  This is a
    read-only summary and does not infer causal importance from norm changes.
``all``
    Run compare, final-state sensitivity, CKA, and plotting in sequence.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# The D1 scripts are intentionally the source of truth for labels, metrics,
# probe fitting, and CKA numerical conventions.  The script directory is on
# sys.path when this file is invoked directly, but keep this explicit for
# callers that import the module from the repository root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from d1_cka import (  # noqa: E402
    GD_MAPPING,
    _cross_cka,
    _linear_cka_matrix,
    _load_cache_view,
    _sample_indices,
)
from d1_linear_probes import (  # noqa: E402
    MODELS,
    TASKS,
    _raw_label_to_model,
    _task_widths,
    evaluate,
    load_dataset_contract,
    load_features,
    read_json,
    sha256_file,
    write_json,
)
from d1_probe_sensitivity import _fit_vectorized  # noqa: E402


EXPECTED_LAYERS = {"student": 11, "teacher": 13}
VIEWS = ("cls", "mean")
TRAIN_SAMPLE_SIZE = 1024
CKA_SAMPLE_SEED = 123
PROBE_STEPS = 200
SENSITIVITY_STEPS = 1000
PROBE_SEED = 42
DEVICE_DEFAULT = "cuda:3"
TASK_LABELS = {
    "vsfc_sentiment": "VSFC\nSentiment",
    "vsfc_topic": "VSFC\nTopic",
    "visfd": "ViSFD",
    "absa_restaurant": "UIT-ABSA\nRestaurant",
    "absa_hotel": "UIT-ABSA\nHotel",
}
MODEL_LABELS = {"student": "Student", "teacher": "Teacher"}


def _ensure_repo_src(repo_root: Path) -> None:
    src = str((repo_root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot serialise {type(value)!r}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = read_json(path)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    """Convert non-finite numeric leaves to JSON null for audit artifacts."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return value.item()
    return value


def _write_json_safe(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _json_safe(payload))


def _require_probe_contract(result: Mapping[str, Any], *, path: Path, task: str) -> None:
    if str(result.get("task_id")) != task:
        raise RuntimeError(f"Probe task mismatch in {path}: {result.get('task_id')!r}")
    probe = result.get("probe", {})
    expected = {
        "seed": PROBE_SEED,
        "steps": PROBE_STEPS,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
    }
    for key, value in expected.items():
        if probe.get(key) != value:
            raise RuntimeError(f"Probe contract mismatch in {path}: {key}={probe.get(key)!r}, expected {value!r}")
    dataset = result.get("dataset", {})
    if dataset.get("include_test") is not False:
        raise RuntimeError(f"Probe artifact does not certify test exclusion: {path}")
    for model in MODELS:
        rows = result.get("models", {}).get(model, {}).get("layers")
        if not isinstance(rows, list) or len(rows) != EXPECTED_LAYERS[model]:
            raise RuntimeError(f"Unexpected {model} layer count in {path}")
        for expected_layer, row in enumerate(rows):
            if int(row.get("layer_index", -1)) != expected_layer:
                raise RuntimeError(f"Non-contiguous layer index in {path}/{model}")


def _selection_f1(row: Mapping[str, Any]) -> float:
    value = row.get("selection_f1")
    if value is None:
        metrics = row.get("validation_metrics", {})
        value = metrics.get("selection_f1")
    if value is None:
        raise KeyError("Probe row has no selection_f1")
    return float(value)


def _control_selection_f1(row: Mapping[str, Any]) -> float | None:
    """Read the D1 label-shuffle score across both recorded schemas."""
    value = row.get("selection_f1")
    if value is None and isinstance(row.get("validation_metrics"), Mapping):
        value = row["validation_metrics"].get("selection_f1")
    return None if value is None else float(value)


def _dataset_rows(cache_root: Path, task: str, split: str) -> dict[str, Any]:
    path = cache_root / task / "dataset.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset contract: {path}")
    dataset = _read_json(path)
    split_value = dataset.get("splits", {}).get(split)
    if not isinstance(split_value, Mapping):
        raise RuntimeError(f"Missing {split} dataset metadata in {path}")
    ids = split_value.get("ids")
    source_indices = split_value.get("source_indices")
    labels = split_value.get("raw_gold_labels", split_value.get("golds"))
    input_sha = split_value.get("input_sha256")
    if not isinstance(ids, list) or not isinstance(source_indices, list):
        raise RuntimeError(f"Dataset contract lacks ids/source_indices: {path}/{split}")
    return {
        "path": str(path.resolve()),
        "path_sha256": sha256_file(path),
        "ids": ids,
        "source_indices": source_indices,
        "labels": labels,
        "input_sha256": input_sha,
        "count": len(ids),
    }


def _compare_dataset_contracts(before_root: Path, after_root: Path, task: str) -> dict[str, Any]:
    checks: dict[str, Any] = {"task": task, "splits": {}}
    for split in ("train", "validation"):
        before = _dataset_rows(before_root, task, split)
        after = _dataset_rows(after_root, task, split)
        ids_match = before["ids"] == after["ids"]
        indices_match = before["source_indices"] == after["source_indices"]
        labels_match = before["labels"] == after["labels"]
        inputs_match = before.get("input_sha256") == after.get("input_sha256")
        if not (ids_match and indices_match and labels_match):
            raise RuntimeError(f"D1/D2 cohort mismatch for {task}/{split}: ids={ids_match}, indices={indices_match}, labels={labels_match}")
        # Input digest is expected to match exactly for the same tokenizer and
        # max-length contract.  Older manifests may omit the digest, in which
        # case IDs/indices/labels still provide the auditable alignment.
        if before.get("input_sha256") and after.get("input_sha256") and not inputs_match:
            raise RuntimeError(f"D1/D2 tokenized-input fingerprint mismatch for {task}/{split}")
        checks["splits"][split] = {
            "count": before["count"],
            "ids_match": ids_match,
            "source_indices_match": indices_match,
            "labels_match": labels_match,
            "input_sha256_match": inputs_match if before.get("input_sha256") and after.get("input_sha256") else None,
            "before_dataset_sha256": before["path_sha256"],
            "after_dataset_sha256": after["path_sha256"],
        }
    return checks


def _probe_path(root: Path, task: str) -> Path:
    return root / task / "probe_results.json"


def compare_probes(
    repo_root: Path,
    *,
    before_probe_root: Path,
    after_probe_root: Path,
    output_root: Path,
    before_cache_root: Path | None = None,
    after_cache_root: Path | None = None,
    tasks: Sequence[str] = TASKS,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    before_cache_root = (before_cache_root or repo_root / "artifacts" / "d1" / "cache").resolve()
    after_cache_root = (after_cache_root or repo_root / "artifacts" / "d2" / "cache").resolve()
    rows: list[dict[str, Any]] = []
    task_payloads: dict[str, Any] = {}
    for task in tasks:
        before_path = _probe_path(before_probe_root, task)
        after_path = _probe_path(after_probe_root, task)
        before = _read_json(before_path)
        after = _read_json(after_path)
        _require_probe_contract(before, path=before_path, task=task)
        _require_probe_contract(after, path=after_path, task=task)
        cohort = _compare_dataset_contracts(before_cache_root, after_cache_root, task)
        task_payloads[task] = {"before": before, "after": after, "cohort": cohort}
        for model in MODELS:
            b_layers = before["models"][model]["layers"]
            a_layers = after["models"][model]["layers"]
            if len(b_layers) != len(a_layers):
                raise RuntimeError(f"Probe layer count changed for {task}/{model}")
            for b_row, a_row in zip(b_layers, a_layers):
                layer = int(b_row["layer_index"])
                rows.append({
                    "task": task,
                    "model": model,
                    "layer_index": layer,
                    "relative_depth": float(a_row.get("relative_depth", b_row.get("relative_depth", 0.0))),
                    "before_selection_f1": _selection_f1(b_row),
                    "after_selection_f1": _selection_f1(a_row),
                    "delta_selection_f1": _selection_f1(a_row) - _selection_f1(b_row),
                    "before_selection_f1_points": 100.0 * _selection_f1(b_row),
                    "after_selection_f1_points": 100.0 * _selection_f1(a_row),
                    "delta_selection_f1_points": 100.0 * (_selection_f1(a_row) - _selection_f1(b_row)),
                    "before_hidden_size": int(b_row.get("hidden_size", 0)),
                    "after_hidden_size": int(a_row.get("hidden_size", 0)),
                })
            b_control = before.get("label_shuffle_control", {}).get(model, {})
            a_control = after.get("label_shuffle_control", {}).get(model, {})
            b_final = b_layers[-1]
            a_final = a_layers[-1]
            b_control_f1 = _control_selection_f1(b_control) if b_control else None
            a_control_f1 = _control_selection_f1(a_control) if a_control else None
            rows.append({
                "task": task,
                "model": model,
                "layer_index": "label_shuffle_final",
                "relative_depth": 1.0,
                "before_selection_f1": b_control_f1,
                "after_selection_f1": a_control_f1,
                "delta_selection_f1": (a_control_f1 - b_control_f1) if b_control_f1 is not None and a_control_f1 is not None else None,
                "before_selection_f1_points": 100.0 * b_control_f1 if b_control_f1 is not None else None,
                "after_selection_f1_points": 100.0 * a_control_f1 if a_control_f1 is not None else None,
                "delta_selection_f1_points": 100.0 * (a_control_f1 - b_control_f1) if b_control_f1 is not None and a_control_f1 is not None else None,
                "before_hidden_size": int(b_final.get("hidden_size", 0)),
                "after_hidden_size": int(a_final.get("hidden_size", 0)),
            })

    comparison = {
        "schema": "goibert-d2-probe-comparison-v1",
        "status": "completed",
        "protocol": {
            "before": "D1 frozen backbone cache and 200-step probe",
            "after": "D2 fine-tuned backbone cache and same D1 200-step probe",
            "steps": PROBE_STEPS,
            "seed": PROBE_SEED,
            "metric": "validation selection_f1",
            "test_opened": False,
        },
        "tasks": list(tasks),
        "models": list(MODELS),
        "rows": rows,
        "cohort_checks": {task: payload["cohort"] for task, payload in task_payloads.items()},
        "sources": {
            "before_probe_root": str(before_probe_root.resolve()),
            "after_probe_root": str(after_probe_root.resolve()),
            "before_probe_sha256": {task: sha256_file(_probe_path(before_probe_root, task)) for task in tasks},
            "after_probe_sha256": {task: sha256_file(_probe_path(after_probe_root, task)) for task in tasks},
        },
        "interpretation_guard": "Probe deltas are descriptive; they do not prove causal forgetting or a bottleneck.",
    }
    _write_json_safe(output_root / "probe_comparison.json", comparison)
    _write_csv(
        output_root / "probe_comparison.csv",
        rows,
        ("task", "model", "layer_index", "relative_depth", "before_selection_f1", "after_selection_f1", "delta_selection_f1", "before_selection_f1_points", "after_selection_f1_points", "delta_selection_f1_points", "before_hidden_size", "after_hidden_size"),
    )

    final_rows: list[dict[str, Any]] = []
    for task in tasks:
        before = task_payloads[task]["before"]
        after = task_payloads[task]["after"]
        for model in MODELS:
            b = before["models"][model]["layers"][-1]
            a = after["models"][model]["layers"][-1]
            final_rows.append({
                "task": task,
                "model": model,
                "before_final_f1": _selection_f1(b),
                "after_final_f1": _selection_f1(a),
                "delta_final_f1": _selection_f1(a) - _selection_f1(b),
                "before_final_f1_points": 100.0 * _selection_f1(b),
                "after_final_f1_points": 100.0 * _selection_f1(a),
                "delta_final_f1_points": 100.0 * (_selection_f1(a) - _selection_f1(b)),
                "before_best_f1": max(_selection_f1(row) for row in before["models"][model]["layers"]),
                "after_best_f1": max(_selection_f1(row) for row in after["models"][model]["layers"]),
            })
    comparison["final_layer_rows"] = final_rows
    _write_json_safe(output_root / "probe_comparison.json", comparison)
    _write_csv(output_root / "probe_final_comparison.csv", final_rows, tuple(final_rows[0].keys()))
    # The D1 runner writes a progress/summary file on every invocation.  The
    # Hotel-only invocation used while the other caches were still extracting
    # would otherwise leave a misleading one-task summary in the canonical D2
    # probe directory.  Rebuild that small index from all completed per-task
    # JSON files without refitting any head or touching D1 artifacts.
    after_summary = {
        "protocol": "D1-layerwise-linear-probe-v1",
        "status": "completed",
        "tasks": list(tasks),
        "elapsed_seconds": None,
        "device": str(after.get("runtime", {}).get("device", "unknown")),
        "results": {
            task: {
                model: {
                    "best_selection_f1": max(_selection_f1(row) for row in task_payloads[task]["after"]["models"][model]["layers"]),
                    "final_selection_f1": _selection_f1(task_payloads[task]["after"]["models"][model]["layers"][-1]),
                    "final_layer": len(task_payloads[task]["after"]["models"][model]["layers"]) - 1,
                }
                for model in MODELS
            }
            for task in tasks
        },
        "source": "rebuilt by d2_compare.py from per-task D2 probe_results.json; no refit",
    }
    _write_json_safe(after_probe_root / "summary.json", after_summary)
    _write_json_safe(after_probe_root / "progress.json", after_summary)
    return comparison


def run_final_sensitivity(
    repo_root: Path,
    *,
    cache_root: Path,
    before_sensitivity_root: Path,
    output_root: Path,
    device_name: str,
    tasks: Sequence[str] = TASKS,
) -> dict[str, Any]:
    import torch

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(4)
    all_rows: list[dict[str, Any]] = []
    started = time.time()
    for task in tasks:
        dataset = load_dataset_contract(cache_root / task, task=task)
        train_labels_np = _raw_label_to_model(task, np.asarray(dataset["train"]["golds"], dtype=np.int64))
        valid_labels_np = _raw_label_to_model(task, np.asarray(dataset["validation"]["golds"], dtype=np.int64))
        train_labels = torch.from_numpy(train_labels_np)
        valid_labels = torch.from_numpy(valid_labels_np)
        widths = _task_widths(task, train_labels_np)
        for model in MODELS:
            train_cache, valid_cache, hashes = load_features(
                cache_root / task / model,
                task=task,
                model=model,
                view="cls",
                dataset=dataset,
            )
            final_layer = int(train_cache.shape[0] - 1)
            fit, predictions = _fit_vectorized(
                train_cache[final_layer],
                valid_cache[final_layer],
                train_labels,
                valid_labels,
                widths,
                device=device,
                steps=SENSITIVITY_STEPS,
            )
            metrics = evaluate(task, dataset["validation"]["golds"], predictions)
            before_path = before_sensitivity_root / task / "sensitivity_results.json"
            before = _read_json(before_path)
            before_model = before["models"][model]
            before_row = next(row for row in before_model["layers"] if int(row["layer_index"]) == final_layer)
            after_row = {
                "task": task,
                "model": model,
                "layer_index": final_layer,
                "scope": "final_layer_only",
                "before_selection_f1_1000": float(before_row["selection_f1_1000"]),
                "after_selection_f1_1000": float(metrics["selection_f1"]),
                "delta_selection_f1_1000": float(metrics["selection_f1"]) - float(before_row["selection_f1_1000"]),
                "before_selection_f1_1000_points": 100.0 * float(before_row["selection_f1_1000"]),
                "after_selection_f1_1000_points": 100.0 * float(metrics["selection_f1"]),
                "delta_selection_f1_1000_points": 100.0 * (float(metrics["selection_f1"]) - float(before_row["selection_f1_1000"])),
                "before_fit": before_row.get("fit_1000"),
                "after_fit": fit,
                "after_metrics": metrics,
                "after_cache_hashes": hashes,
            }
            all_rows.append(after_row)
            del train_cache, valid_cache
            gc.collect()
    result = {
        "schema": "goibert-d2-final-probe-sensitivity-v1",
        "status": "completed",
        "protocol": {
            "steps": SENSITIVITY_STEPS,
            "same_as_d1": True,
            "selection": "fixed final encoder state only; no post-hoc layer or epoch selection",
            "device": str(device),
            "test_opened": False,
        },
        "tasks": list(tasks),
        "models": list(MODELS),
        "rows": all_rows,
        "elapsed_seconds": time.time() - started,
        "interpretation_guard": "A 1,000-step probe still has convergence diagnostics; a score difference is not a causal capacity estimate.",
    }
    _write_json_safe(output_root / "final_layer_sensitivity.json", result)
    return result


def run_boundary_sensitivity(
    *,
    cache_root: Path,
    after_probe_root: Path,
    output_root: Path,
    device_name: str,
) -> dict[str, Any]:
    """Run the explicitly authorized four-head boundary follow-up.

    The layer choices (student state 7, teacher state 8) were made after the
    200-step curves were visible.  They are therefore reported as a post-hoc
    exploratory boundary check, never as an unbiased layer search.  We retain
    the fixed 1,000-step budget and compare only to the already-produced D2
    200-step value at the same state.
    """
    import torch

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(4)
    layer_by_model = {"student": 7, "teacher": 8}
    tasks = ("visfd", "absa_restaurant")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        dataset = load_dataset_contract(cache_root / task, task=task)
        train_labels_np = _raw_label_to_model(task, np.asarray(dataset["train"]["golds"], dtype=np.int64))
        valid_labels_np = _raw_label_to_model(task, np.asarray(dataset["validation"]["golds"], dtype=np.int64))
        train_labels = torch.from_numpy(train_labels_np)
        valid_labels = torch.from_numpy(valid_labels_np)
        widths = _task_widths(task, train_labels_np)
        after_probe = _read_json(after_probe_root / task / "probe_results.json")
        for model in MODELS:
            layer = layer_by_model[model]
            train_cache, valid_cache, hashes = load_features(
                cache_root / task / model,
                task=task,
                model=model,
                view="cls",
                dataset=dataset,
            )
            fit, predictions = _fit_vectorized(
                train_cache[layer],
                valid_cache[layer],
                train_labels,
                valid_labels,
                widths,
                device=device,
                steps=SENSITIVITY_STEPS,
            )
            metrics = evaluate(task, dataset["validation"]["golds"], predictions)
            row_200 = after_probe["models"][model]["layers"][layer]
            rows.append({
                "task": task,
                "model": model,
                "layer_index": layer,
                "selection": "posthoc_boundary_about_70_percent_depth",
                "after_selection_f1_200": float(row_200["selection_f1"]),
                "after_selection_f1_1000": float(metrics["selection_f1"]),
                "delta_1000_minus_200": float(metrics["selection_f1"]) - float(row_200["selection_f1"]),
                "after_selection_f1_200_points": 100.0 * float(row_200["selection_f1"]),
                "after_selection_f1_1000_points": 100.0 * float(metrics["selection_f1"]),
                "delta_1000_minus_200_points": 100.0 * (float(metrics["selection_f1"]) - float(row_200["selection_f1"])),
                "after_metrics_1000": metrics,
                "fit_1000": fit,
                "cache_hashes": hashes,
            })
            del train_cache, valid_cache
            gc.collect()
    result = {
        "schema": "goibert-d2-boundary-sensitivity-v1",
        "status": "completed",
        "protocol": {
            "steps": SENSITIVITY_STEPS,
            "same_probe_contract": True,
            "scope": "four extra AFTER heads: ViSFD and Restaurant, student layer 7 and teacher layer 8",
            "posthoc": True,
            "not_a_full_layer_1000_sweep": True,
            "device": str(device),
            "test_opened": False,
        },
        "rows": rows,
        "interpretation_guard": "This post-hoc boundary check cannot select a bottleneck or establish causality; it only checks whether the visible middle-layer curve changes under the longer probe budget.",
    }
    _write_json_safe(output_root / "boundary_sensitivity.json", result)
    return result


def _weight_group(parameter_name: str) -> str:
    """Assign a Transformer parameter to one unambiguous reporting group."""
    if parameter_name.startswith("embeddings."):
        return "embeddings"
    if not parameter_name.startswith("encoder.layer."):
        return "other"
    if "LayerNorm" in parameter_name or "layer_norm" in parameter_name:
        return "layernorm"
    if ".attention." in parameter_name:
        return "attention"
    # RoBERTa FFN has intermediate.dense and output.dense; the latter is
    # distinguished from attention.output.dense by the attention check above.
    if ".intermediate.dense." in parameter_name or ".output.dense." in parameter_name:
        return "ffn"
    return "other"


def _aggregate_parameter_rows(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate squared norms while preserving parameter-count denominators."""
    group_buckets: dict[str, dict[str, Any]] = {}
    layer_buckets: dict[str, dict[str, dict[str, Any]]] = {}

    def add(bucket: dict[str, Any], row: Mapping[str, Any]) -> None:
        bucket["parameter_count"] += int(row.get("parameter_count", 0))
        bucket["abs_l2_sq"] += float(row.get("abs_l2", 0.0)) ** 2
        bucket["before_l2_sq"] += float(row.get("before_l2", 0.0)) ** 2

    for name, raw_row in parameters.items():
        if not isinstance(raw_row, Mapping):
            continue
        name = str(name)
        group = _weight_group(name)
        group_buckets.setdefault(group, {"parameter_count": 0, "abs_l2_sq": 0.0, "before_l2_sq": 0.0})
        add(group_buckets[group], raw_row)
        if name.startswith("encoder.layer."):
            layer = "layer_" + name.split(".")[2]
            layer_buckets.setdefault(layer, {})
            layer_buckets[layer].setdefault(group, {"parameter_count": 0, "abs_l2_sq": 0.0, "before_l2_sq": 0.0})
            add(layer_buckets[layer][group], raw_row)

    def finish(bucket: Mapping[str, Any]) -> dict[str, Any]:
        abs_l2 = math.sqrt(float(bucket.get("abs_l2_sq", 0.0)))
        before_l2 = math.sqrt(float(bucket.get("before_l2_sq", 0.0)))
        return {
            "parameter_count": int(bucket.get("parameter_count", 0)),
            "abs_l2": abs_l2,
            "before_l2": before_l2,
            "relative_l2": abs_l2 / before_l2 if before_l2 > 0.0 else None,
        }

    all_bucket = {"parameter_count": 0, "abs_l2_sq": 0.0, "before_l2_sq": 0.0}
    for raw_row in parameters.values():
        if isinstance(raw_row, Mapping):
            add(all_bucket, raw_row)
    return {
        "groups": {key: finish(bucket) for key, bucket in sorted(group_buckets.items())},
        "layers": {
            layer: {group: finish(bucket) for group, bucket in sorted(group_values.items())}
            for layer, group_values in sorted(layer_buckets.items())
        },
        "all_encoder": finish(all_bucket),
    }


def summarize_weight_drift(*, cache_root: Path, output_root: Path, tasks: Sequence[str] = TASKS) -> dict[str, Any]:
    models_payload: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for task in tasks:
        models_payload[task] = {}
        for model in MODELS:
            path = cache_root / task / model / "weight_drift.json"
            source = _read_json(path)
            corrected = _aggregate_parameter_rows(source.get("parameters", {}))
            models_payload[task][model] = {
                "source": str(path.resolve()),
                "source_sha256": sha256_file(path),
                **corrected,
            }
            for group, row in corrected["groups"].items():
                flat_rows.append({"task": task, "model": model, "scope": "encoder_group", "layer": "all", "group": group, **row})
            for layer, layer_groups in corrected["layers"].items():
                for group, row in layer_groups.items():
                    flat_rows.append({"task": task, "model": model, "scope": "layer_group", "layer": layer, "group": group, **row})
    result = {
        "schema": "goibert-d2-weight-drift-summary-v1",
        "status": "completed",
        "tasks": list(tasks),
        "models": list(MODELS),
        "cache_root": str(cache_root.resolve()),
        "models_payload": models_payload,
        "rows": flat_rows,
        "group_rule": {
            "attention": "parameter name contains .attention. excluding LayerNorm",
            "ffn": ".intermediate.dense. or .output.dense. after attention exclusion",
            "layernorm": "parameter name contains LayerNorm/layer_norm",
            "embeddings": "parameter name starts embeddings.",
        },
        "interpretation_guard": "Relative parameter drift describes update magnitude; it is not a causal importance or bottleneck measure.",
    }
    _write_json_safe(output_root / "weight_drift_comparison.json", result)
    _write_csv(output_root / "weight_drift_comparison.csv", flat_rows, ("task", "model", "scope", "layer", "group", "parameter_count", "abs_l2", "before_l2", "relative_l2"))
    return result


def _load_views(cache_root: Path, task: str, model: str):
    return _load_cache_view(cache_root, task, model)


def run_cka(
    repo_root: Path,
    *,
    before_cache_root: Path,
    after_cache_root: Path,
    output_root: Path,
    views: Sequence[str] = VIEWS,
    tasks: Sequence[str] = TASKS,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    started = time.time()
    sample_info = {
        "train_sample_size": TRAIN_SAMPLE_SIZE,
        "seed": CKA_SAMPLE_SEED,
        "sampling_rule": "sorted(np.random.default_rng(123).choice(N_train, min(1024,N_train), replace=False))",
        "row_permutation_control_repeated": False,
        "row_permutation_reference": "D1 passed teacher-only row permutation control; D2 does not rerun it.",
    }
    for task in tasks:
        # Load each cache through the same strict schema reader used by D1.
        before_views = {model: _load_views(before_cache_root, task, model) for model in MODELS}
        after_views = {model: _load_views(after_cache_root, task, model) for model in MODELS}
        cohort = _compare_dataset_contracts(before_cache_root, after_cache_root, task)
        n_rows = before_views["student"].arrays["train_cls"].shape[1]
        indices = _sample_indices(n_rows, size=TRAIN_SAMPLE_SIZE, seed=CKA_SAMPLE_SEED)
        task_result: dict[str, Any] = {
            "task": task,
            "sample": {**sample_info, "indices": indices.tolist(), "rows_used": int(len(indices))},
            "cohort": cohort,
            "models": {},
            "after_student_teacher": {},
        }
        for model in MODELS:
            model_result: dict[str, Any] = {}
            for view in views:
                before_array = before_views[model].arrays[f"train_{view}"]
                after_array = after_views[model].arrays[f"train_{view}"]
                if before_array.shape != after_array.shape:
                    raise RuntimeError(f"BEFORE/AFTER shape mismatch for {task}/{model}/{view}: {before_array.shape} vs {after_array.shape}")
                matrix, info = _cross_cka(before_array, after_array, indices)
                diag = np.diag(matrix)
                finite_diag = diag[np.isfinite(diag)]
                # The D1 matrix is included for direct GD-pair comparison when
                # available.  Same-model drift is always the primary output.
                d1_path = repo_root / "artifacts" / "d1" / "cka" / task / f"{view}_cka.json"
                d1 = _read_json(d1_path) if d1_path.is_file() else None
                model_result[view] = {
                    "primary": view == "cls",
                    "before_after_cross": matrix.tolist(),
                    "before_after_info": info,
                    "same_layer_diagonal": diag.tolist(),
                    "same_layer_diagonal_summary": {
                        "finite_count": int(finite_diag.size),
                        "mean": float(np.mean(finite_diag)) if finite_diag.size else None,
                        "min": float(np.min(finite_diag)) if finite_diag.size else None,
                        "max": float(np.max(finite_diag)) if finite_diag.size else None,
                    },
                    "d1_before_student_teacher_cross": d1.get("train_cross") if d1 else None,
                    "d1_before_gd_pair_rows": d1.get("gd_pair_rows_train") if d1 else None,
                }
                del matrix, info, before_array, after_array
                gc.collect()
            task_result["models"][model] = model_result

        # AFTER student–teacher cross-depth matrix, a cheap reuse of the D1
        # Gram implementation.  This is descriptive and does not optimise GD
        # mapping.  The declared mapping is only used for a small summary.
        for view in views:
            student_array = after_views["student"].arrays[f"train_{view}"]
            teacher_array = after_views["teacher"].arrays[f"train_{view}"]
            after_cross, after_info = _cross_cka(student_array, teacher_array, indices)
            mapped = []
            for student_layer, teacher_layer in enumerate(GD_MAPPING):
                value = float(after_cross[student_layer, teacher_layer])
                mapped.append({
                    "student_layer": student_layer,
                    "teacher_layer": teacher_layer,
                    "value": value if math.isfinite(value) else None,
                })
            d1_path = repo_root / "artifacts" / "d1" / "cka" / task / f"{view}_cka.json"
            d1 = _read_json(d1_path) if d1_path.is_file() else None
            before_pairs = d1.get("gd_pair_rows_train") if d1 else None
            if before_pairs:
                for after_row, before_row in zip(mapped, before_pairs):
                    if after_row["value"] is not None and before_row.get("value") is not None:
                        after_row["before_d1_value"] = float(before_row["value"])
                        after_row["delta_vs_d1"] = float(after_row["value"]) - float(before_row["value"])
            task_result["after_student_teacher"][view] = {
                "primary": view == "cls",
                "cross": after_cross.tolist(),
                "info": after_info,
                "gd_mapping": list(GD_MAPPING),
                "gd_pair_rows": mapped,
                "d1_before_cross": d1.get("train_cross") if d1 else None,
            }
            del after_cross, after_info, student_array, teacher_array
            gc.collect()
        all_results[task] = task_result
        task_dir = output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json_safe(task_dir / "result.json", task_result)
        print(f"[D2 CKA] completed {task}", flush=True)

    result = {
        "schema": "goibert-d2-cka-v1",
        "status": "completed",
        "before_cache_root": str(before_cache_root.resolve()),
        "after_cache_root": str(after_cache_root.resolve()),
        "tasks": list(tasks),
        "views": list(views),
        "analysis_split": "train",
        "sample": sample_info,
        "tasks_payload": all_results,
        "elapsed_seconds": time.time() - started,
        "test_opened": False,
        "interpretation_guard": "CKA and weight drift describe geometry/change; neither alone identifies a causal bottleneck.",
    }
    _write_json_safe(output_root / "cka_comparison.json", result)
    return result


def _load_probe_comparison(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != "goibert-d2-probe-comparison-v1":
        raise RuntimeError(f"Unexpected D2 probe comparison schema in {path}")
    return value


def render_plots(*, comparison_path: Path, cka_path: Path, output_dir: Path) -> dict[str, Any]:
    """Render the two D2 overview figures using the isolated Matplotlib env."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Matplotlib is required for D2 figures; use the isolated D1 plotting environment") from exc

    comparison = _load_probe_comparison(comparison_path)
    cka = _read_json(cka_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "before_student": "#6baed6",
        "after_student": "#2171b5",
        "before_teacher": "#fdae6b",
        "after_teacher": "#d94801",
    }

    # First figure: one five-panel layerwise curve, four curves per task.
    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for row in comparison["rows"]:
        if isinstance(row["layer_index"], int):
            by_task[row["task"]].append(row)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(22, 4.9), constrained_layout=True)
    if len(TASKS) == 1:
        axes = [axes]
    for axis, task in zip(axes, TASKS):
        for model in MODELS:
            model_rows = [r for r in by_task[task] if r["model"] == model]
            model_rows.sort(key=lambda r: int(r["layer_index"]))
            x = np.asarray([int(r["layer_index"]) for r in model_rows])
            axis.plot(x, 100.0 * np.asarray([r["before_selection_f1"] for r in model_rows]), marker="o", ms=3, lw=1.4, color=colors[f"before_{model}"], label=f"{MODEL_LABELS[model]} before")
            axis.plot(x, 100.0 * np.asarray([r["after_selection_f1"] for r in model_rows]), marker="o", ms=3, lw=1.8, color=colors[f"after_{model}"], label=f"{MODEL_LABELS[model]} after")
        axis.set_title(TASK_LABELS[task], fontsize=10)
        axis.set_xlabel("Layer index", fontsize=8)
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.grid(axis="y", color="#dddddd", lw=0.7)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("Validation selection F1 (%)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=8)
    fig.suptitle("D2 layerwise linear probes before and after fine-tuning", fontsize=13, y=1.09)
    fig.savefig(output_dir / "probe_before_after.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "probe_before_after.svg", bbox_inches="tight")
    plt.close(fig)

    # Second figure: diagonal same-layer CKA before↔after.  This makes the
    # requested per-layer drift visible while avoiding an unreadable 10-panel
    # cross-depth heatmap.  Full matrices remain in JSON artifacts.
    fig, axes = plt.subplots(2, len(TASKS), figsize=(22, 7.2), sharey="row", constrained_layout=True)
    for row_index, model in enumerate(MODELS):
        for axis, task in zip(axes[row_index], TASKS):
            payload = cka["tasks_payload"][task]["models"][model]["cls"]
            diag = np.asarray(payload["same_layer_diagonal"], dtype=np.float64)
            x = np.arange(diag.size)
            axis.plot(x, diag, marker="o", ms=3.3, lw=1.6, color=colors[f"after_{model}"])
            axis.axhline(1.0, color="#bbbbbb", lw=0.8, ls="--")
            axis.set_title(f"{TASK_LABELS[task]} — {MODEL_LABELS[model]}", fontsize=9)
            axis.set_xlabel("Layer index", fontsize=8)
            axis.set_ylim(0, 1.02)
            axis.set_yticks(np.arange(0, 1.01, 0.2))
            axis.grid(axis="y", color="#dddddd", lw=0.7)
            axis.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("CKA(before, after)", fontsize=9)
    axes[1, 0].set_ylabel("CKA(before, after)", fontsize=9)
    fig.suptitle("D2 same-layer representation similarity after task fine-tuning", fontsize=13, y=1.02)
    fig.savefig(output_dir / "cka_before_after_diagonal.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "cka_before_after_diagonal.svg", bbox_inches="tight")
    plt.close(fig)
    return {
        "status": "completed",
        "comparison": str(comparison_path.resolve()),
        "cka": str(cka_path.resolve()),
        "outputs": [str((output_dir / name).resolve()) for name in ("probe_before_after.png", "probe_before_after.svg", "cka_before_after_diagonal.png", "cka_before_after_diagonal.svg")],
        "matplotlib": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare", "sensitivity", "boundary_sensitivity", "weight_drift", "cka", "plots", "all"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--before-cache-root", type=Path, default=None)
    parser.add_argument("--after-cache-root", type=Path, default=None)
    parser.add_argument("--before-probe-root", type=Path, default=None)
    parser.add_argument("--after-probe-root", type=Path, default=None)
    parser.add_argument("--before-sensitivity-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    _ensure_repo_src(repo_root)
    before_cache = (args.before_cache_root or repo_root / "artifacts" / "d1" / "cache").resolve()
    after_cache = (args.after_cache_root or repo_root / "artifacts" / "d2" / "cache").resolve()
    before_probes = (args.before_probe_root or repo_root / "artifacts" / "d1" / "probes").resolve()
    after_probes = (args.after_probe_root or repo_root / "artifacts" / "d2" / "probes").resolve()
    before_sensitivity = (args.before_sensitivity_root or repo_root / "artifacts" / "d1" / "probe_sensitivity").resolve()
    output_root = (args.output_root or repo_root / "artifacts" / "d2").resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if args.command in {"compare", "all"}:
        compare_probes(
            repo_root,
            before_probe_root=before_probes,
            after_probe_root=after_probes,
            output_root=output_root,
            before_cache_root=before_cache,
            after_cache_root=after_cache,
            tasks=args.tasks,
        )
    if args.command in {"sensitivity", "all"}:
        run_final_sensitivity(repo_root, cache_root=after_cache, before_sensitivity_root=before_sensitivity, output_root=output_root / "probe_sensitivity", device_name=args.device, tasks=args.tasks)
    if args.command == "boundary_sensitivity":
        run_boundary_sensitivity(cache_root=after_cache, after_probe_root=after_probes, output_root=output_root / "probe_sensitivity", device_name=args.device)
    if args.command == "weight_drift":
        summarize_weight_drift(cache_root=after_cache, output_root=output_root, tasks=args.tasks)
    if args.command in {"cka", "all"}:
        run_cka(repo_root, before_cache_root=before_cache, after_cache_root=after_cache, output_root=output_root / "cka", views=args.views, tasks=args.tasks)
    if args.command in {"plots", "all"}:
        render_plots(comparison_path=output_root / "probe_comparison.json", cka_path=output_root / "cka" / "cka_comparison.json", output_dir=output_root / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
