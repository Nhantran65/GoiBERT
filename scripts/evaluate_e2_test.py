"""Evaluate the locked E2 validation winners on the test splits.

This is an evaluation-only companion to the E2 training runner.  It reuses the
frozen E1 data loader, model constructor, batch builder, and evaluator, while
keeping the E2 test metadata and artifacts separate from the E1 artifacts.
The script deliberately has no training, teacher, cache, KD, or noise path.

The selection manifest is created before any E2 test inference and contains
the validation-locked checkpoint identities plus model, tokenizer, data, and
source hashes.  Every checkpoint is evaluated at most once.  A partial output
set is treated as an error; a complete output set is reused and is never
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]

# Keep direct script execution consistent with the project runners.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from goibert.modeling import build_classifier
from goibert.train import (
    _move_batch,
    _predictions,
    _make_loader,
    _runtime_metadata,
    bundle_provenance,
    directory_hashes,
    evaluate_predictions,
    inspect_train_lengths,
    load_task_bundle,
    model_labels,
    resolve_device,
    resolve_precision,
    run_test,
    run_validation,
    sha256_file,
    split_field,
    split_ids,
    split_labels,
    split_texts,
    task_aspect_class_counts,
    task_num_aspects,
    task_num_classes,
    write_json,
)
from transformers import AutoTokenizer


E2_OUTPUT = ROOT / "artifacts" / "e2"
MANIFEST_PATH = E2_OUTPUT / "test_selection_manifest.json"
READOUT_PATH = E2_OUTPUT / "test_readout.json"
ACCESS_LOG_PATH = E2_OUTPUT / "test_access_log.json"
SMOKE_PATH = E2_OUTPUT / "preflight" / "e2_test_smoke.json"

TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
SEEDS = (42, 43, 44)
EXPECTED_COUNTS = {
    "vsfc_sentiment": 3166,
    "vsfc_topic": 3166,
    "visfd": 2224,
    "absa_restaurant": 1938,
    "absa_hotel": 2030,
}
TASK_NAMES = {
    "vsfc_sentiment": "VSFC Sentiment",
    "vsfc_topic": "VSFC Topic",
    "visfd": "ViSFD",
    "absa_restaurant": "UIT-ABSA Restaurant",
    "absa_hotel": "UIT-ABSA Hotel",
}
LEARNING_RATES = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}
SELECTED = {
    "vsfc_sentiment": {"method": "nc", "coefficient": 0.1},
    "vsfc_topic": {"method": "kd", "coefficient": 0.75},
    "visfd": {"method": "nc", "coefficient": 0.5},
    "absa_restaurant": {"method": "nc", "coefficient": 0.1},
    "absa_hotel": {"method": "nc", "coefficient": 0.1},
}
E1_DEPENDENCY_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"
E2_SOURCE_HASHES = {
    "e2_code": "efb81ac232de2f0ae4287688ad0125a1d84fb1e689e2b1566ef26d8d6e2b60b5",
    "e2_config": "45e6c36196fe95ae0eb9193e9197224bdf2980515c15ddf01998267b668b61b8",
    "orchestrator": "f626112d163a4c9ed476d9811bb74447b337d23bd6e87d9e105ab8e0a66807b8",
}
PAPER_TARGETS_PATH = ROOT / "benchmarks" / "paper_targets.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def _write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _split_hash(split: Any) -> str:
    rows = []
    for example_id, text, label in zip(split_ids(split), split_texts(split), split_labels(split)):
        rows.append({"id": example_id, "text": text, "label": label})
    return _json_hash(rows)


def _artifact_for(task_id: str, method: str, coefficient: float, seed: int) -> Path:
    coef_slug = str(coefficient).replace(".", "p")
    stage = "stageA" if seed == 42 else "stageB"
    path = E2_OUTPUT / stage / method / task_id / f"coef_{coef_slug}" / f"seed_{seed}"
    # Sentiment seed 42 is a symlink to the verified rerun artifact.  Resolve
    # it so the locked identity is unambiguous and is counted once.
    return path.resolve(strict=True)


def _validate_e2_metadata(task_id: str, method: str, coefficient: float, seed: int, artifact: Path) -> dict[str, Any]:
    metadata_path = artifact / "metadata.json"
    checkpoint_path = artifact / "best_checkpoint.pt"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Incomplete E2 checkpoint artifact: {artifact}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "task_id": task_id,
        "method": method,
        "coefficient": coefficient,
        "seed": seed,
        "learning_rate": LEARNING_RATES[task_id],
        "max_length": 256,
        "effective_batch_size": 32,
        "micro_batch_size": 32,
        "attention_implementation": "sdpa",
        "include_test": False,
        "smoke": False,
        "status": "completed",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"E2 metadata mismatch for {artifact}: {key}={metadata.get(key)!r}, expected {value!r}")
    runtime = metadata.get("runtime", {})
    if runtime.get("precision") != "bf16":
        raise RuntimeError(f"E2 checkpoint is not BF16: {artifact}")
    source_hashes = metadata.get("source_hashes", {})
    if source_hashes.get("e1_dependency") != E1_DEPENDENCY_HASH:
        raise RuntimeError(f"E1 dependency hash mismatch in {metadata_path}")
    metadata_key_by_manifest_key = {
        "e2_code": "e2",
        "e2_config": "e2_config",
        "orchestrator": "orchestrator",
    }
    for manifest_key, value in E2_SOURCE_HASHES.items():
        metadata_key = metadata_key_by_manifest_key[manifest_key]
        if source_hashes.get(metadata_key) != value:
            raise RuntimeError(f"E2 {manifest_key} hash mismatch in {metadata_path}")
    return metadata


def _test_snapshot(task_id: str) -> dict[str, Any]:
    bundle = load_task_bundle(task_id, ROOT, include_test=True)
    train = split_field(bundle, "train")
    validation = split_field(bundle, "validation")
    test = split_field(bundle, "test")
    if train is None or validation is None or test is None:
        raise RuntimeError(f"Task {task_id} does not expose all frozen splits")
    provenance = bundle_provenance(bundle, ROOT)
    return {
        "data_provenance": provenance,
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_hashes": {
            "train": _split_hash(train),
            "validation": _split_hash(validation),
            "test": _split_hash(test),
        },
        "test_ids": list(split_ids(test)),
        "test_ids_hash": _json_hash(list(split_ids(test))),
        "test_row_count": len(test),
        "evaluator_code_hash": provenance.get("evaluator_code_hash"),
    }


def build_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("locked") is True and existing.get("checkpoint_count") == 15:
            return existing
        raise RuntimeError(f"Refusing to overwrite an existing malformed manifest: {MANIFEST_PATH}")

    rows: list[dict[str, Any]] = []
    task_snapshots: dict[str, Any] = {}
    for task_id in TASKS:
        choice = SELECTED[task_id]
        task_snapshots[task_id] = _test_snapshot(task_id)
        for seed in SEEDS:
            artifact = _artifact_for(task_id, choice["method"], choice["coefficient"], seed)
            metadata = _validate_e2_metadata(task_id, choice["method"], choice["coefficient"], seed, artifact)
            checkpoint = artifact / "best_checkpoint.pt"
            metadata_path = artifact / "metadata.json"
            backbone_hashes = directory_hashes(ROOT / "models" / "goiturn-s-gd")
            tokenizer_hashes = {
                key: value
                for key, value in backbone_hashes.items()
                if key in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
            }
            snapshot = task_snapshots[task_id]
            rows.append(
                {
                    "task_id": task_id,
                    "task": TASK_NAMES[task_id],
                    "stage": "stageA" if seed == 42 else "stageB",
                    "method": choice["method"],
                    "coefficient": choice["coefficient"],
                    "seed": seed,
                    "learning_rate": LEARNING_RATES[task_id],
                    "best_epoch": metadata["best_epoch"],
                    "artifact_dir": str(artifact),
                    "checkpoint_path": str(checkpoint),
                    "metadata_path": str(metadata_path),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "metadata_sha256": sha256_file(metadata_path),
                    "backbone_hashes": backbone_hashes,
                    "tokenizer_hashes": tokenizer_hashes,
                    "data_provenance": snapshot["data_provenance"],
                    "test_snapshot": {
                        "split_counts": snapshot["split_counts"],
                        "split_hashes": snapshot["split_hashes"],
                        "test_ids_hash": snapshot["test_ids_hash"],
                        "test_row_count": snapshot["test_row_count"],
                    },
                    "evaluator_code_hash": snapshot["evaluator_code_hash"],
                    "e1_dependency_source_hash": metadata["e1_dependency_source_hash"],
                    "e2_source_hash": metadata["e2_source_hash"],
                    "e2_code_hash": metadata["e2_code_hash"],
                    "e2_config_hash": metadata["e2_config_hash"],
                    "orchestrator_hash": metadata["orchestrator_hash"],
                    "model": metadata["model"],
                    "precision": metadata["runtime"]["precision"],
                    "attention_implementation": metadata["attention_implementation"],
                    "max_length": metadata["max_length"],
                    "effective_batch_size": metadata["effective_batch_size"],
                    "micro_batch_size": metadata["micro_batch_size"],
                    "selection_basis": "locked validation winner; seed42 coefficient selection and seeds43/44 confirmation already completed",
                }
            )

    manifest = {
        "protocol": "E2-test-gate-v1",
        "manifest_id": "E2-test-selection-20260909-v1",
        "created_at_utc": _now(),
        "locked": True,
        "checkpoint_count": len(rows),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "test_access_policy": "each locked checkpoint evaluated once; no test-based selection, tuning, or retraining",
        "evaluation_contract": {
            "mode": "inference_only",
            "precision": "bf16",
            "attention_implementation": "sdpa",
            "max_length": 256,
            "micro_batch_size": 32,
            "student_backbone": "models/goiturn-s-gd",
            "architecture": "E2 student constructor delegates to frozen E1 classifier; strict state_dict load",
            "teacher_inference": False,
            "noise_inference": False,
        },
        "e1_dependency_source_hash": E1_DEPENDENCY_HASH,
        "e2_source_hashes": E2_SOURCE_HASHES,
        "selection": {
            task: {"method": SELECTED[task]["method"], "coefficient": SELECTED[task]["coefficient"], "learning_rate": LEARNING_RATES[task]}
            for task in TASKS
        },
        "task_snapshots": task_snapshots,
        "selections": rows,
    }
    _write_atomic(MANIFEST_PATH, manifest)
    return manifest


def _preexisting_outputs(artifact: Path) -> tuple[Path, ...]:
    return tuple(
        artifact / name
        for name in ("e2_test_metrics.json", "e2_test_predictions.json", "e2_test_evaluation_metadata.json")
        if (artifact / name).exists()
    )


def _smoke(repo_root: Path) -> dict[str, Any]:
    """Strict-load a dummy checkpoint and run a tiny validation-only pass."""

    task_id = "vsfc_sentiment"
    bundle = load_task_bundle(task_id, repo_root, include_test=False, smoke=True)
    train = split_field(bundle, "train")
    validation = split_field(bundle, "validation")
    tokenizer = AutoTokenizer.from_pretrained(repo_root / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
    model, model_spec = build_classifier(
        repo_root / "models" / "goiturn-s-gd",
        num_classes=task_num_classes(bundle),
        num_aspects=task_num_aspects(bundle),
        aspect_class_counts=task_aspect_class_counts(bundle) if task_num_aspects(bundle) else None,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    smoke_dir = E2_OUTPUT / "preflight" / "e2_test_dummy_checkpoint"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    dummy_path = smoke_dir / "dummy_checkpoint.pt"
    torch.save({"model": model.state_dict()}, dummy_path)
    restored, restored_spec = build_classifier(
        repo_root / "models" / "goiturn-s-gd",
        num_classes=task_num_classes(bundle),
        num_aspects=task_num_aspects(bundle),
        aspect_class_counts=task_aspect_class_counts(bundle) if task_num_aspects(bundle) else None,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    state = torch.load(dummy_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(state["model"], strict=True)
    restored.eval()
    loader = _make_loader(
        tokenizer,
        validation,
        max_length=256,
        batch_size=8,
        shuffle=False,
        seed=42,
        labels=model_labels(bundle, "validation", validation),
        gold_labels=split_labels(validation),
    )
    metrics, predictions = run_validation(restored, loader, torch.device("cpu"), "fp32", task_id, bundle)
    result = {
        "protocol": "E2-test-wrapper-smoke-v1",
        "created_at_utc": _now(),
        "status": "passed",
        "test_split_loaded": False,
        "dummy_checkpoint": str(dummy_path),
        "model_spec": model_spec.to_dict(),
        "restored_model_spec": restored_spec.to_dict(),
        "strict_state_dict": True,
        "validation_rows": len(predictions),
        "validation_metrics": metrics,
    }
    _write_atomic(SMOKE_PATH, result)
    return result


def _run_cpu_test(model: Any, loader: Any, task_id: str, bundle: Any, precision: str):
    """Run the frozen E1 inference/evaluator contract on a CPU fallback.

    The normal path is ``goibert.train.run_test``.  That helper intentionally
    hard-codes CUDA BF16 because all recorded E1/E2 runs used L40S GPUs.  The
    execution host may temporarily lose its NVIDIA driver, so this local
    compatibility path keeps the same tokenizer, loader, model.eval/no-grad,
    prediction conversion, and task evaluator while selecting CPU autocast.
    """

    model.eval()
    counts = task_aspect_class_counts(bundle)
    gold: list[Any] = []
    predicted: list[Any] = []
    ids: list[Any] = []
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, torch.device("cpu"))
            if precision == "bf16":
                context = torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True)
            else:
                context = torch.autocast(device_type="cpu", enabled=False)
            with context:
                logits = model(
                    moved["input_ids"],
                    moved["attention_mask"],
                    token_type_ids=moved.get("token_type_ids"),
                )
            predicted.extend(_predictions(logits, aspect_class_counts=counts))
            gold.extend(batch["gold_labels"].tolist())
            ids.extend(batch["ids"])
    metrics = evaluate_predictions(task_id, bundle, gold, predicted, split="test")
    rows = [
        {"id": example_id, "gold": target, "prediction": prediction}
        for example_id, target, prediction in zip(ids, gold, predicted)
    ]
    return metrics, rows


def _evaluate_one(row: Mapping[str, Any], device_name: str, *, allow_cpu_diagnostic: bool = False) -> dict[str, Any]:
    task_id = str(row["task_id"])
    seed = int(row["seed"])
    artifact = Path(str(row["artifact_dir"]))
    outputs = {
        "metrics": artifact / "e2_test_metrics.json",
        "predictions": artifact / "e2_test_predictions.json",
        "evaluation": artifact / "e2_test_evaluation_metadata.json",
    }
    present = tuple(path for path in outputs.values() if path.exists())
    if present:
        if len(present) != len(outputs):
            raise RuntimeError(f"Partial E2 test output set; refusing overwrite: {artifact}")
        metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
        predictions = json.loads(outputs["predictions"].read_text(encoding="utf-8"))
        evaluation = json.loads(outputs["evaluation"].read_text(encoding="utf-8"))
        if evaluation.get("status") != "completed" or metrics.get("split") != "test":
            raise RuntimeError(f"Existing E2 test outputs are invalid: {artifact}")
        return {
            "task_id": task_id,
            "seed": seed,
            "artifact_dir": str(artifact),
            "status": "reused",
            "metrics": metrics,
            "prediction_count": len(predictions.get("rows", [])),
            "paths": {key: str(value) for key, value in outputs.items()},
            "device": evaluation.get("runtime", {}).get("device"),
        }
    if present:
        raise RuntimeError(f"Refusing to overwrite partial E2 test outputs: {artifact}")

    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    checkpoint = artifact / "best_checkpoint.pt"
    checkpoint_hash = sha256_file(checkpoint)
    metadata_hash = sha256_file(artifact / "metadata.json")
    if checkpoint_hash != row["checkpoint_sha256"] or metadata_hash != row["metadata_sha256"]:
        raise RuntimeError(f"Locked hash mismatch before test: {artifact}")
    bundle = load_task_bundle(task_id, ROOT, include_test=True)
    train = split_field(bundle, "train")
    test = split_field(bundle, "test")
    if train is None or test is None:
        raise RuntimeError(f"Missing train/test split for {task_id}")
    if len(test) != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Unexpected test row count for {task_id}: {len(test)}")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
    lengths = inspect_train_lengths(tokenizer, list(split_texts(train)))
    if lengths.max_length != 256:
        raise RuntimeError(f"E2 test max_length contract failed for {task_id}: {lengths.max_length}")
    device = resolve_device(device_name)
    if device.type == "cpu" and not allow_cpu_diagnostic:
        raise RuntimeError(
            "CUDA test protocol requested but this worker resolved to CPU; "
            "pass --allow-cpu-diagnostic only for a separately labelled diagnostic run"
        )
    if device.type == "cuda":
        precision = resolve_precision("bf16", device)
    else:
        # This fallback is only used when the host has no NVIDIA driver.  Try
        # CPU BF16 first so the numerical mode remains as close as possible to
        # the approved GPU contract; the caller records the actual mode.
        precision = "bf16"
    model, model_spec = build_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=task_num_classes(bundle),
        num_aspects=task_num_aspects(bundle),
        aspect_class_counts=task_aspect_class_counts(bundle) if task_num_aspects(bundle) else None,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    if _canonical(model_spec.to_dict()) != _canonical(metadata["model"]):
        raise RuntimeError(f"Model spec mismatch for {artifact}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Checkpoint does not contain model state: {checkpoint}")
    # Default strict=True is made explicit so missing/unexpected E2 keys fail.
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    loader = _make_loader(
        tokenizer,
        test,
        max_length=256,
        batch_size=32,
        shuffle=False,
        seed=seed,
        labels=model_labels(bundle, "test", test),
        gold_labels=split_labels(test),
    )
    started = time.time()
    if device.type == "cuda":
        metrics, prediction_rows = run_test(model, loader, device, precision, task_id, bundle)
    else:
        try:
            metrics, prediction_rows = _run_cpu_test(model, loader, task_id, bundle, "bf16")
            precision = "bf16"
        except (RuntimeError, TypeError):
            # CPU BF16 is unavailable on some hosts; retain an auditable FP32
            # fallback rather than claiming the GPU execution contract.
            metrics, prediction_rows = _run_cpu_test(model, loader, task_id, bundle, "fp32")
            precision = "fp32"
    if metrics.get("split") != "test" or metrics.get("n") != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Evaluator returned an unexpected test contract for {artifact}: {metrics}")
    if len(prediction_rows) != EXPECTED_COUNTS[task_id] or len({row["id"] for row in prediction_rows}) != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Prediction ID contract failed for {artifact}")
    eval_runtime = _runtime_metadata(device, precision)
    evaluation = {
        "protocol": "E2-test-gate-v1",
        "status": "completed",
        "evaluated_at_utc": _now(),
        "task_id": task_id,
        "seed": seed,
        "method": row["method"],
        "coefficient": row["coefficient"],
        "learning_rate": row["learning_rate"],
        "stage": row["stage"],
        "artifact_dir": str(artifact),
        "checkpoint_sha256": checkpoint_hash,
        "metadata_sha256": metadata_hash,
        "selection_manifest_sha256": sha256_file(MANIFEST_PATH),
        "evaluator_code_hash": row["evaluator_code_hash"],
        "data_provenance": row["data_provenance"],
        "backbone_hashes": row["backbone_hashes"],
        "tokenizer_hashes": row["tokenizer_hashes"],
        "e1_dependency_source_hash": row["e1_dependency_source_hash"],
        "e2_source_hash": row["e2_source_hash"],
        "e2_code_hash": row["e2_code_hash"],
        "e2_config_hash": row["e2_config_hash"],
        "orchestrator_hash": row["orchestrator_hash"],
        "model": model_spec.to_dict(),
        "runtime": eval_runtime,
        "contract": {
            "mode": "inference_only",
            "teacher_inference": False,
            "noise_inference": False,
            "dropout": "disabled via model.eval()",
            "max_length": 256,
            "micro_batch_size": 32,
            "attention_implementation": "sdpa",
            "precision": precision,
            "strict_state_dict": True,
        },
        "n_test": len(prediction_rows),
        "elapsed_seconds": time.time() - started,
        "test_metrics_path": str(outputs["metrics"]),
        "test_predictions_path": str(outputs["predictions"]),
    }
    _write_atomic(outputs["metrics"], metrics)
    _write_atomic(outputs["predictions"], {"rows": prediction_rows})
    _write_atomic(outputs["evaluation"], evaluation)
    return {
        "task_id": task_id,
        "seed": seed,
        "artifact_dir": str(artifact),
        "status": "evaluated",
        "metrics": metrics,
        "prediction_count": len(prediction_rows),
        "paths": {key: str(value) for key, value in outputs.items()},
        "device": eval_runtime.get("device"),
    }


def _load_e1_paired() -> dict[tuple[str, int], dict[str, Any]]:
    readout_path = ROOT / "artifacts" / "e1" / "test_readout.json"
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in readout.get("per_seed_results", []):
        if int(row["seed"]) not in SEEDS:
            continue
        metrics_path = Path(row["test_metrics_path"])
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("split") != "test" or metrics.get("n") != EXPECTED_COUNTS[row["task_id"]]:
            raise RuntimeError(f"Invalid paired E1 test metrics: {metrics_path}")
        result[(row["task_id"], int(row["seed"]))] = {
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "predictions_path": str(Path(row["artifact_dir"]) / "test_predictions.json"),
            "artifact_dir": row["artifact_dir"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    if set(result) != expected:
        raise RuntimeError("Paired E1 test readout does not contain exactly tasks × seeds 42,43,44")
    return result


def _paper_targets() -> dict[str, dict[str, float]]:
    mapping = {
        ("UIT-VSFC", "sentiment", "accuracy"): ("vsfc_sentiment", "accuracy"),
        ("UIT-VSFC", "sentiment", "F1"): ("vsfc_sentiment", "macro_f1"),
        ("UIT-VSFC", "topic", "accuracy"): ("vsfc_topic", "accuracy"),
        ("UIT-VSFC", "topic", "F1"): ("vsfc_topic", "macro_f1"),
        ("UIT-ViSFD", "aspect_detection", "F1"): ("visfd", "detection_micro_f1"),
        ("UIT-ViSFD", "aspect_sentiment_classification", "F1"): ("visfd", "pair_micro_f1"),
        ("UIT-ABSA-Restaurant", "aspect_detection", "F1"): ("absa_restaurant", "detection_micro_f1"),
        ("UIT-ABSA-Restaurant", "aspect_sentiment_classification", "F1"): ("absa_restaurant", "pair_micro_f1"),
        ("UIT-ABSA-Hotel", "aspect_detection", "F1"): ("absa_hotel", "detection_micro_f1"),
        ("UIT-ABSA-Hotel", "aspect_sentiment_classification", "F1"): ("absa_hotel", "pair_micro_f1"),
    }
    targets: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    with PAPER_TARGETS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["task"], row["metric"])
            if key in mapping:
                task, metric = mapping[key]
                targets[task][metric] = float(row["value"])
    if any(len(values) == 0 for values in targets.values()):
        raise RuntimeError("Paper target CSV is missing one or more mapped rows")
    return targets


def _build_readout(manifest: Mapping[str, Any], execution_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    e1 = _load_e1_paired()
    paper = _paper_targets()
    e2_by_key = {(str(row["task_id"]), int(row["seed"])): row for row in execution_results}
    metric_names = {
        "vsfc_sentiment": ("accuracy", "macro_f1"),
        "vsfc_topic": ("accuracy", "macro_f1"),
        "visfd": ("detection_micro_f1", "pair_micro_f1"),
        "absa_restaurant": ("detection_micro_f1", "pair_micro_f1"),
        "absa_hotel": ("detection_micro_f1", "pair_micro_f1"),
    }
    per_seed: list[dict[str, Any]] = []
    for task_id in TASKS:
        for seed in SEEDS:
            e2row = e2_by_key[(task_id, seed)]
            per_seed.append(
                {
                    "task_id": task_id,
                    "task": TASK_NAMES[task_id],
                    "seed": seed,
                    "stage": next(row["stage"] for row in manifest["selections"] if row["task_id"] == task_id and int(row["seed"]) == seed),
                    "method": next(row["method"] for row in manifest["selections"] if row["task_id"] == task_id and int(row["seed"]) == seed),
                    "coefficient": next(row["coefficient"] for row in manifest["selections"] if row["task_id"] == task_id and int(row["seed"]) == seed),
                    "artifact_dir": e2row["artifact_dir"],
                    "status": e2row["status"],
                    "n_test": e2row["prediction_count"],
                    "e2_metrics": e2row["metrics"],
                    "e1_metrics_path": e1[(task_id, seed)]["metrics_path"],
                    "e1_predictions_path": e1[(task_id, seed)]["predictions_path"],
                    "e1_metrics": e1[(task_id, seed)]["metrics"],
                    "e2_predictions_path": e2row["paths"]["predictions"],
                    "e2_evaluation_path": e2row["paths"]["evaluation"],
                }
            )
    aggregates: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        aggregates[task_id] = {}
        rows = [row for row in per_seed if row["task_id"] == task_id]
        for metric in metric_names[task_id]:
            e1_values = [float(row["e1_metrics"][metric]) * 100.0 for row in rows]
            e2_values = [float(row["e2_metrics"][metric]) * 100.0 for row in rows]
            e1_mean, e2_mean = statistics.mean(e1_values), statistics.mean(e2_values)
            e1_sd, e2_sd = statistics.stdev(e1_values), statistics.stdev(e2_values)
            paper_value = paper[task_id][metric]
            aggregate = {
                "e1_per_seed_percent": {str(seed): value for seed, value in zip(SEEDS, e1_values)},
                "e2_per_seed_percent": {str(seed): value for seed, value in zip(SEEDS, e2_values)},
                "e1_mean_percent": e1_mean,
                "e1_sample_std_percent": e1_sd,
                "e2_mean_percent": e2_mean,
                "e2_sample_std_percent": e2_sd,
                "delta_e2_minus_e1_percent": e2_mean - e1_mean,
                "bami_bert_paper_percent": paper_value,
                "delta_e2_minus_bami_percent": e2_mean - paper_value,
            }
            aggregates[task_id][metric] = aggregate
            comparison_rows.append(
                {
                    "task_id": task_id,
                    "task": TASK_NAMES[task_id],
                    "metric": metric,
                    "e1_mean_percent": e1_mean,
                    "e1_sample_std_percent": e1_sd,
                    "e2_mean_percent": e2_mean,
                    "e2_sample_std_percent": e2_sd,
                    "delta_e2_minus_e1_percent": e2_mean - e1_mean,
                    "bami_bert_paper_percent": paper_value,
                    "delta_e2_minus_bami_percent": e2_mean - paper_value,
                    "metric_equivalence": "comparable by definition" if metric == "accuracy" else "unresolved",
                }
            )
    return {
        "protocol": "E2-test-gate-v1",
        "readout_id": "E2-test-readout-20260909-v1",
        "generated_at_utc": _now(),
        "status": "complete",
        "selection_manifest": {"path": str(MANIFEST_PATH), "sha256": sha256_file(MANIFEST_PATH)},
        "smoke": {"path": str(SMOKE_PATH), "sha256": sha256_file(SMOKE_PATH)},
        "execution": {
            "mode": "inference_only",
            "checkpoint_count": len(per_seed),
            "evaluated_or_reused_count": len(per_seed),
            "unique_checkpoint_policy": "15/15 task-seed identities, each evaluated once; complete outputs reused without overwrite",
            "test_access_log": str(ACCESS_LOG_PATH),
        },
        "per_seed_results": per_seed,
        "aggregates": aggregates,
        "comparison_rows": comparison_rows,
        "paper_targets_source": str(PAPER_TARGETS_PATH),
        "paper_target_url": "https://arxiv.org/html/2607.02259",
        "caveats": {
            "f1_equivalence": "Paper F1 averaging/evaluator details are insufficient to prove exact equivalence with internal VSFC macro-F1 and ABSA detection/pair micro-F1.",
            "uit_absa_lineage": "UIT-ABSA files are the public candidate mirror used by this project; exact release lineage to the paper split is not independently proven.",
            "paired_baseline": "E1 comparison uses the same seeds 42,43,44; historical E1 five-seed means are intentionally not used in the main paired comparison.",
        },
    }


def _write_markdown(readout: Mapping[str, Any]) -> Path:
    lines = [
        "# E2 test evaluation and BamiBERT comparison",
        "",
        f"Generated `{readout['generated_at_utc']}`. The locked E2 selection evaluated **15/15 student checkpoints**: five tasks × seeds 42–44. Seed 42 comes from stageA and seeds 43–44 from stageB. The run is inference-only: BF16, SDPA, max length 256, batch size 32, dropout disabled by `model.eval()`, strict model-state loading, and no teacher/KD/noise computation at inference.",
        "",
        "The main baseline is E1 on the same seeds 42–44. E1 test predictions and metrics were reused from its locked artifacts; E1 was not rerun. Values are percentages and mean ± sample standard deviation across the three paired seeds.",
        "",
        "| Task | Metric | E1 test (42–44) | E2 test (42–44) | Δ E2−E1 | BamiBERT paper | Δ E2−paper |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "detection_micro_f1": "Detection micro F1",
        "pair_micro_f1": "Aspect-polarity pair micro F1",
    }
    for row in readout["comparison_rows"]:
        lines.append(
            f"| {row['task']} | {labels[row['metric']]} | {row['e1_mean_percent']:.2f} ± {row['e1_sample_std_percent']:.2f} | {row['e2_mean_percent']:.2f} ± {row['e2_sample_std_percent']:.2f} | {row['delta_e2_minus_e1_percent']:+.2f} | {row['bami_bert_paper_percent']:.2f} | {row['delta_e2_minus_bami_percent']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            "This is the requested direct test evaluation of the validation-selected E2 branches. The E2–E1 delta measures whether the new training objective transferred to held-out test data; it does not retroactively change the branch selection. Accuracy is directly comparable by definition. The F1 rows are reported for the project benchmark, but the paper does not publish enough evaluator and averaging detail to establish exact equivalence for every F1 row.",
            "",
            "UIT-ABSA Restaurant and Hotel use the public candidate mirror already recorded in the project. The paper target remains a useful reported reference, while exact split lineage and evaluator equivalence remain caveats.",
            "",
            "## Provenance",
            "",
            f"- Selection manifest: `{MANIFEST_PATH}` (SHA-256 `{readout['selection_manifest']['sha256']}`).",
            f"- Wrapper smoke: `{SMOKE_PATH}` (SHA-256 `{readout['smoke']['sha256']}`).",
            f"- Test access log: `{ACCESS_LOG_PATH}`. Each task-seed identity has its own evaluation metadata, metrics, and stable-ID predictions under the E2 checkpoint artifact directory.",
            f"- Paper targets: `{PAPER_TARGETS_PATH}`; source [BamiBERT paper](https://arxiv.org/html/2607.02259).",
            "- E1 paired metrics/predictions are referenced from `artifacts/e1/.orchestrator/attempts/` and were read without rerunning inference.",
        ]
    )
    path = ROOT / "docs" / "E2_TEST_EVALUATION.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_csv(readout: Mapping[str, Any]) -> Path:
    path = E2_OUTPUT / "test_comparison.csv"
    fieldnames = [
        "task_id", "task", "metric", "e1_mean_percent", "e1_sample_std_percent",
        "e2_mean_percent", "e2_sample_std_percent", "delta_e2_minus_e1_percent",
        "bami_bert_paper_percent", "delta_e2_minus_bami_percent", "metric_equivalence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(readout["comparison_rows"])
    return path


def _parse_devices(value: str) -> list[str]:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices:
        raise ValueError("At least one device is required")
    return devices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate locked E2 winners on test, without training")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--execute", action="store_true", help="run inference; without this flag only lock the manifest")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--allow-cpu-diagnostic", action="store_true", help="explicitly permit the non-final CPU fallback")
    args = parser.parse_args(argv)
    manifest = build_manifest()
    print(json.dumps({"manifest": str(MANIFEST_PATH), "manifest_sha256": sha256_file(MANIFEST_PATH), "checkpoint_count": manifest["checkpoint_count"]}, indent=2), flush=True)
    if not args.execute:
        return 0
    if not args.skip_smoke:
        smoke = _smoke(ROOT)
        if smoke.get("status") != "passed":
            raise RuntimeError("E2 test wrapper smoke failed")
        print(json.dumps({"smoke": str(SMOKE_PATH), "status": "passed"}, indent=2), flush=True)

    devices = _parse_devices(args.devices)
    rows = list(manifest["selections"])
    results: list[dict[str, Any]] = []
    started = _now()
    # Four independent inference workers match the available GPUs and keep
    # each task-seed checkpoint isolated.  Outputs are written to unique dirs.
    with ThreadPoolExecutor(max_workers=min(len(devices), len(rows))) as pool:
        futures = {
            pool.submit(
                _evaluate_one,
                row,
                devices[index % len(devices)],
                allow_cpu_diagnostic=args.allow_cpu_diagnostic,
            ): row
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"completed": len(results), "total": len(rows), "task_id": result["task_id"], "seed": result["seed"], "status": result["status"], "device": result.get("device")}, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: (row["task_id"], int(row["seed"])))
    if len(results) != 15 or any(row["prediction_count"] != EXPECTED_COUNTS[row["task_id"]] for row in results):
        raise RuntimeError("E2 test evaluation did not produce 15 complete task-seed outputs")
    access_log = {
        "protocol": "E2-test-gate-v1",
        "access_log_id": "E2-test-access-20260909-v1",
        "started_at_utc": started,
        "completed_at_utc": _now(),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "policy": "15 validation-locked E2 checkpoints, each evaluated once; no test-based selection, tuning, or retraining",
        "devices": devices,
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "mode": "inference_only",
        "rows": results,
    }
    _write_atomic(ACCESS_LOG_PATH, access_log)
    readout = _build_readout(manifest, results)
    _write_atomic(READOUT_PATH, readout)
    doc = _write_markdown(readout)
    csv_path = _write_csv(readout)
    print(json.dumps({"readout": str(READOUT_PATH), "doc": str(doc), "csv": str(csv_path), "access_log": str(ACCESS_LOG_PATH), "count": len(results)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
