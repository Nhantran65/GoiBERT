"""Evaluate the locked E8 NC+LLRD seed-42 checkpoints on test.

This is deliberately an evaluation-only runner.  It uses the original E1
classifier constructor and loader/evaluator helpers, and does not import the
E8 training/noise path.  The five validation-selected checkpoints are locked
in a manifest before any test prediction is produced.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Set this before torch can initialize a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoTokenizer, __version__ as transformers_version

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from goibert.modeling import build_classifier
from goibert.train import (
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
    seed_everything,
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


TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
EXPECTED_COUNTS = {
    "vsfc_sentiment": 3166,
    "vsfc_topic": 3166,
    "visfd": 2224,
    "absa_restaurant": 1938,
    "absa_hotel": 2030,
}
EXPECTED_EPOCHS = {
    "vsfc_sentiment": 21,
    "vsfc_topic": 25,
    "visfd": 28,
    "absa_restaurant": 25,
    "absa_hotel": 26,
}
EXPECTED_COEFFICIENTS = {
    "vsfc_sentiment": 0.1,
    "vsfc_topic": 0.5,
    "visfd": 0.5,
    "absa_restaurant": 0.1,
    "absa_hotel": 0.1,
}
TASK_NAMES = {
    "vsfc_sentiment": "VSFC Sentiment",
    "vsfc_topic": "VSFC Topic",
    "visfd": "ViSFD",
    "absa_restaurant": "UIT-ABSA Restaurant",
    "absa_hotel": "UIT-ABSA Hotel",
}
PAPER_TARGETS = {
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

E8_ROOT = ROOT / "artifacts" / "e8"
MANIFEST_PATH = E8_ROOT / "test_selection_manifest.json"
READOUT_PATH = E8_ROOT / "test_readout.json"
COMPARISON_PATH = E8_ROOT / "test_comparison.csv"
ACCESS_LOG_PATH = E8_ROOT / "test_access_log.json"
RESULT_DOC = ROOT / "docs" / "E8_TEST_RESULT.md"
BACKBONE = ROOT / "models" / "goiturn-s-gd"
PAPER_TARGET_PATH = ROOT / "benchmarks" / "paper_targets.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _split_snapshot(split: Any) -> dict[str, Any]:
    ids = list(split_ids(split))
    texts = list(split_texts(split))
    labels = list(split_labels(split))
    if not (len(ids) == len(texts) == len(labels)):
        raise RuntimeError("Split ID/text/label lengths do not match")
    if len(ids) != len(set(ids)):
        raise RuntimeError("Split IDs are not unique")
    rows = [{"id": i, "text": t, "label": y} for i, t, y in zip(ids, texts, labels)]
    return {
        "count": len(ids),
        "ids_sha256": _json_hash(ids),
        "labels_sha256": _json_hash(labels),
        "rows_sha256": _json_hash(rows),
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "src" / "goibert" / "data.py",
        ROOT / "src" / "goibert" / "metrics.py",
        ROOT / "src" / "goibert" / "modeling.py",
        ROOT / "src" / "goibert" / "train.py",
        ROOT / "scripts" / "run_e8_nc_llrd.py",
        Path(__file__).resolve(),
    )
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths if path.is_file()}


def _runtime(device: torch.device) -> dict[str, Any]:
    result = _runtime_metadata(device, "bf16")
    result["torch_version"] = torch.__version__
    result["transformers_version"] = transformers_version
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    result["backends"] = {
        "attention_implementation": "sdpa",
        "cuda_flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "cuda_mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "cuda_math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    return result


def _validate_locked_metadata(task_id: str, metadata: Mapping[str, Any]) -> None:
    expected = {
        "task_id": task_id,
        "seed": 42,
        "status": "completed",
        "method": "nc_plus_llrd",
        "mode": "nc",
        "best_epoch": EXPECTED_EPOCHS[task_id],
        "coefficient": EXPECTED_COEFFICIENTS[task_id],
        "max_length": 256,
        "micro_batch_size": 32,
        "effective_batch_size": 32,
        "attention_implementation": "sdpa",
        "include_test": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"E8 locked metadata mismatch for {task_id}: {key}={metadata.get(key)!r}, expected {value!r}")
    if metadata.get("runtime", {}).get("precision") != "bf16":
        raise RuntimeError(f"E8 checkpoint is not BF16: {task_id}")


def _manifest_row(task_id: str) -> dict[str, Any]:
    artifact = E8_ROOT / task_id / "seed_42"
    checkpoint = artifact / "best_checkpoint.pt"
    metadata_path = artifact / "metadata.json"
    resolved_path = artifact / "resolved_config.json"
    for path in (checkpoint, metadata_path, resolved_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_locked_metadata(task_id, metadata)
    bundle = load_task_bundle(task_id, ROOT, include_test=True)
    train = split_field(bundle, "train")
    validation = split_field(bundle, "validation")
    test = split_field(bundle, "test")
    if train is None or validation is None or test is None:
        raise RuntimeError(f"{task_id} does not expose train/validation/test")
    if len(test) != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Unexpected test row count for {task_id}: {len(test)}")
    provenance = bundle_provenance(bundle, ROOT)
    backbone_hashes = directory_hashes(BACKBONE)
    tokenizer_hashes = {
        name: digest for name, digest in backbone_hashes.items()
        if name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    }
    return {
        "task_id": task_id,
        "task": TASK_NAMES[task_id],
        "seed": 42,
        "method": metadata["method"],
        "mode": metadata["mode"],
        "coefficient": metadata["coefficient"],
        "best_epoch": metadata["best_epoch"],
        "artifact_dir": str(artifact.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "resolved_config_path": str(resolved_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metadata_sha256": sha256_file(metadata_path),
        "resolved_config_sha256": sha256_file(resolved_path),
        "backbone_hashes": backbone_hashes,
        "tokenizer_hashes": tokenizer_hashes,
        "source_hashes": _source_hashes(),
        "evaluator_code_hash": sha256_file(ROOT / "src" / "goibert" / "metrics.py"),
        "data_provenance": provenance,
        "split_snapshots": {
            "train": _split_snapshot(train),
            "validation": _split_snapshot(validation),
            "test": _split_snapshot(test),
        },
        "test_row_count": len(test),
        "model": metadata["model"],
        "precision": metadata["runtime"]["precision"],
        "attention_implementation": metadata["attention_implementation"],
        "max_length": metadata["max_length"],
        "micro_batch_size": metadata["micro_batch_size"],
        "selection_basis": "locked E8 NC+LLRD validation winner; seed 42; no test reselection",
    }


def build_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("locked") is not True or manifest.get("checkpoint_count") != 5:
            raise RuntimeError(f"Malformed existing E8 test manifest: {MANIFEST_PATH}")
        return manifest
    rows = [_manifest_row(task_id) for task_id in TASKS]
    manifest = {
        "protocol": "E8-test-gate-v1",
        "manifest_id": "E8-test-selection-20260910-seed42-v1",
        "created_at_utc": _now(),
        "locked": True,
        "checkpoint_count": len(rows),
        "tasks": list(TASKS),
        "seed": 42,
        "selection_recipe": "NC+LLRD for all tasks; one validation-locked checkpoint per task",
        "test_access_policy": "each locked checkpoint evaluated once; no test-based selection, tuning, or retraining",
        "evaluation_contract": {
            "mode": "inference_only",
            "precision": "bf16",
            "attention_implementation": "sdpa",
            "max_length": 256,
            "micro_batch_size": 32,
            "student_backbone": "models/goiturn-s-gd",
            "classifier": "original src/goibert/modeling.py CLS head; strict checkpoint['model'] load",
            "teacher_inference": False,
            "noise_inference": False,
            "dropout": "disabled via model.eval()",
        },
        "selections": rows,
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def _assert_finite(value: Any, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _complete_outputs(artifact: Path) -> tuple[Path, Path, Path] | None:
    paths = tuple(artifact / name for name in (
        "e8_test_metrics.json", "e8_test_predictions.json", "e8_test_evaluation_metadata.json"
    ))
    present = tuple(path.exists() for path in paths)
    if any(present) and not all(present):
        raise RuntimeError(f"Partial E8 test output set; refusing overwrite: {artifact}")
    return paths if all(present) else None


def _verify_saved(task_id: str, row: Mapping[str, Any], bundle: Any, paths: Sequence[Path]) -> dict[str, Any]:
    metrics = json.loads(paths[0].read_text(encoding="utf-8"))
    prediction_payload = json.loads(paths[1].read_text(encoding="utf-8"))
    evaluation = json.loads(paths[2].read_text(encoding="utf-8"))
    prediction_rows = prediction_payload.get("rows")
    if not isinstance(prediction_rows, list) or len(prediction_rows) != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Saved prediction count mismatch for {task_id}")
    test = split_field(bundle, "test")
    expected_ids = list(split_ids(test))
    expected_gold = list(split_labels(test))
    actual_ids = [item.get("id") for item in prediction_rows]
    actual_gold = [item.get("gold") for item in prediction_rows]
    if actual_ids != expected_ids or _json_hash(actual_gold) != _json_hash(expected_gold):
        raise RuntimeError(f"Saved test IDs/golds do not match frozen split for {task_id}")
    if len(set(actual_ids)) != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Saved test IDs are not unique for {task_id}")
    predicted = [item.get("prediction") for item in prediction_rows]
    recomputed = evaluate_predictions(task_id, bundle, actual_gold, predicted, split="test")
    if json.dumps(recomputed, sort_keys=True, separators=(",", ":")) != json.dumps(metrics, sort_keys=True, separators=(",", ":")):
        raise RuntimeError(f"Saved metrics do not equal one-pass recomputation for {task_id}")
    _assert_finite(metrics, f"{task_id}.metrics")
    if evaluation.get("status") != "completed" or evaluation.get("n_test") != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Invalid evaluation metadata for {task_id}")
    current_checkpoint_hash = sha256_file(Path(row["checkpoint_path"]))
    current_metadata_hash = sha256_file(Path(row["metadata_path"]))
    if current_checkpoint_hash != row["checkpoint_sha256"] or current_metadata_hash != row["metadata_sha256"]:
        raise RuntimeError(f"Locked E8 input hash changed after inference: {task_id}")
    return {"metrics": metrics, "prediction_count": len(prediction_rows), "evaluation": evaluation}


def _choose_device() -> str:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("E8 test requires the host CUDA runtime with at least four visible GPUs")
    threshold = 5 * 1024**3
    for index in (2, 1, 3):
        free, _ = torch.cuda.mem_get_info(index)
        if free >= threshold:
            return f"cuda:{index}"
    raise RuntimeError("No authorized E8 test GPU has at least 5 GiB free")


def evaluate_one(row: Mapping[str, Any], device_name: str) -> dict[str, Any]:
    task_id = str(row["task_id"])
    artifact = Path(row["artifact_dir"])
    existing = _complete_outputs(artifact)
    bundle = load_task_bundle(task_id, ROOT, include_test=True)
    if existing is not None:
        verified = _verify_saved(task_id, row, bundle, existing)
        return {"task_id": task_id, "status": "reused", "prediction_count": verified["prediction_count"], "metrics": verified["metrics"], "paths": [str(path) for path in existing], "runtime": verified["evaluation"].get("runtime", {})}

    checkpoint_hash = sha256_file(Path(row["checkpoint_path"]))
    metadata_hash = sha256_file(Path(row["metadata_path"]))
    if checkpoint_hash != row["checkpoint_sha256"] or metadata_hash != row["metadata_sha256"]:
        raise RuntimeError(f"Locked E8 input hash mismatch before inference: {task_id}")
    train = split_field(bundle, "train")
    test = split_field(bundle, "test")
    if train is None or test is None or len(test) != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Frozen E8 test contract failed for {task_id}")
    device = resolve_device(device_name)
    if device.type != "cuda":
        raise RuntimeError("CPU fallback is forbidden by the E8 test contract")
    precision = resolve_precision("bf16", device)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True, use_fast=True)
    lengths = inspect_train_lengths(tokenizer, list(split_texts(train)))
    if lengths.max_length != 256:
        raise RuntimeError(f"Train-only length contract failed for {task_id}: {lengths.max_length}")
    counts = task_aspect_class_counts(bundle)
    model, model_spec = build_classifier(
        BACKBONE,
        num_classes=task_num_classes(bundle),
        num_aspects=task_num_aspects(bundle),
        aspect_class_counts=counts if task_num_aspects(bundle) else None,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    if json.dumps(model_spec.to_dict(), sort_keys=True) != json.dumps(row["model"], sort_keys=True):
        raise RuntimeError(f"Model specification mismatch for {task_id}")
    state = torch.load(Path(row["checkpoint_path"]), map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Checkpoint missing model state: {task_id}")
    model.load_state_dict(state["model"], strict=True)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Non-finite checkpoint parameter {task_id}:{name}")
    model.to(device)
    model.eval()
    loader = _make_loader(
        tokenizer,
        test,
        max_length=256,
        batch_size=32,
        shuffle=False,
        seed=42,
        labels=model_labels(bundle, "test", test),
        gold_labels=split_labels(test),
    )
    started = time.time()
    metrics, prediction_rows = run_test(model, loader, device, precision, task_id, bundle)
    if len(prediction_rows) != EXPECTED_COUNTS[task_id]:
        raise RuntimeError(f"Evaluator returned wrong row count for {task_id}")
    # Verify IDs/golds and metrics before writing the output set.
    expected_ids = list(split_ids(test))
    expected_gold = list(split_labels(test))
    if [item["id"] for item in prediction_rows] != expected_ids or _json_hash([item["gold"] for item in prediction_rows]) != _json_hash(expected_gold):
        raise RuntimeError(f"Evaluator ID/gold order mismatch for {task_id}")
    recomputed = evaluate_predictions(task_id, bundle, expected_gold, [item["prediction"] for item in prediction_rows], split="test")
    if json.dumps(recomputed, sort_keys=True, separators=(",", ":")) != json.dumps(metrics, sort_keys=True, separators=(",", ":")):
        raise RuntimeError(f"Metric recomputation mismatch for {task_id}")
    _assert_finite(metrics, f"{task_id}.metrics")
    paths = (
        artifact / "e8_test_metrics.json",
        artifact / "e8_test_predictions.json",
        artifact / "e8_test_evaluation_metadata.json",
    )
    runtime = _runtime(device)
    evaluation = {
        "protocol": "E8-test-gate-v1",
        "status": "completed",
        "evaluated_at_utc": _now(),
        "task_id": task_id,
        "seed": 42,
        "method": row["method"],
        "mode": row["mode"],
        "coefficient": row["coefficient"],
        "best_epoch": row["best_epoch"],
        "artifact_dir": str(artifact),
        "checkpoint_sha256": checkpoint_hash,
        "metadata_sha256": metadata_hash,
        "selection_manifest_sha256": sha256_file(MANIFEST_PATH),
        "source_hashes": row["source_hashes"],
        "evaluator_code_hash": row["evaluator_code_hash"],
        "data_provenance": row["data_provenance"],
        "backbone_hashes": row["backbone_hashes"],
        "tokenizer_hashes": row["tokenizer_hashes"],
        "model": model_spec.to_dict(),
        "runtime": runtime,
        "contract": {
            "mode": "inference_only",
            "teacher_inference": False,
            "noise_inference": False,
            "dropout": "disabled via model.eval()",
            "strict_state_dict": True,
            "max_length": 256,
            "micro_batch_size": 32,
            "precision": "bf16",
            "attention_implementation": "sdpa",
        },
        "n_test": len(prediction_rows),
        "elapsed_seconds": time.time() - started,
    }
    write_json(paths[0], metrics)
    write_json(paths[1], {"rows": prediction_rows})
    write_json(paths[2], evaluation)
    del model
    torch.cuda.empty_cache()
    return {"task_id": task_id, "status": "evaluated", "prediction_count": len(prediction_rows), "metrics": metrics, "paths": [str(path) for path in paths], "runtime": runtime}


def _paper_values() -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {task: {} for task in TASKS}
    with PAPER_TARGET_PATH.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            key = (item["dataset"], item["task"], item["metric"])
            if key in PAPER_TARGETS:
                task, metric = PAPER_TARGETS[key]
                values[task][metric] = float(item["value"])
    if any(len(values[task]) != 2 for task in TASKS):
        raise RuntimeError("Paper target CSV is missing one or more of the ten target metrics")
    return values


def build_readout(manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]], started: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paper = _paper_values()
    metrics_by_task = {str(item["task_id"]): item["metrics"] for item in results}
    metric_names = {
        "vsfc_sentiment": ("accuracy", "macro_f1"),
        "vsfc_topic": ("accuracy", "macro_f1"),
        "visfd": ("detection_micro_f1", "pair_micro_f1"),
        "absa_restaurant": ("detection_micro_f1", "pair_micro_f1"),
        "absa_hotel": ("detection_micro_f1", "pair_micro_f1"),
    }
    rows: list[dict[str, Any]] = []
    comparison: list[dict[str, Any]] = []
    for task_id in TASKS:
        result = next(item for item in results if item["task_id"] == task_id)
        for metric in metric_names[task_id]:
            score = float(metrics_by_task[task_id][metric]) * 100.0
            target = paper[task_id][metric]
            comparison.append({
                "task_id": task_id,
                "task": TASK_NAMES[task_id],
                "metric": metric,
                "e8_test_percent": score,
                "bami_bert_paper_percent": target,
                "delta_e8_minus_bami_percent": score - target,
                "metric_equivalence": "comparable by definition" if metric == "accuracy" else "unresolved",
            })
        rows.append({
            "task_id": task_id,
            "task": TASK_NAMES[task_id],
            "seed": 42,
            "best_epoch": next(item["best_epoch"] for item in manifest["selections"] if item["task_id"] == task_id),
            "method": result.get("method", next(item["method"] for item in manifest["selections"] if item["task_id"] == task_id)),
            "status": result["status"],
            "n_test": result["prediction_count"],
            "metrics": result["metrics"],
            "paths": result["paths"],
        })
    readout = {
        "protocol": "E8-test-gate-v1",
        "readout_id": "E8-test-readout-20260910-seed42-v1",
        "generated_at_utc": _now(),
        "status": "complete",
        "selection_manifest": {"path": str(MANIFEST_PATH), "sha256": sha256_file(MANIFEST_PATH)},
        "execution": {"mode": "inference_only", "seed": 42, "started_at_utc": started, "completed_at_utc": _now(), "checkpoint_count": len(results), "test_row_counts": {task: EXPECTED_COUNTS[task] for task in TASKS}},
        "results": rows,
        "comparison_rows": comparison,
        "paper_targets_source": str(PAPER_TARGET_PATH),
        "paper_target_url": "https://arxiv.org/html/2607.02259",
        "caveats": {
            "single_seed": "These are seed-42 point comparisons; they do not establish statistical significance.",
            "f1_equivalence": "Local VSFC macro-F1 and ABSA pair-micro-F1 evaluator/averaging equivalence to the paper remains unresolved.",
            "visfd_others": "ViSFD OTHERS (raw code 4) is included in detection and excluded from polarity pairs by the official local evaluator.",
        },
    }
    return readout, comparison


def write_outputs(readout: Mapping[str, Any], comparison: Sequence[Mapping[str, Any]]) -> None:
    write_json(READOUT_PATH, readout)
    fields = ["task_id", "task", "metric", "e8_test_percent", "bami_bert_paper_percent", "delta_e8_minus_bami_percent", "metric_equivalence"]
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)
    lines = [
        "# E8 test result",
        "",
        "The locked E8 NC+LLRD seed-42 checkpoint was evaluated once on each of the five test splits with the original CLS classifier, strict model-state loading, CUDA BF16, SDPA, max length 256, batch size 32, and dropout disabled by `model.eval()`. No teacher, noise, retraining, or test-based reselection was used.",
        "",
        "| Task | Metric | E8 test | BamiBERT paper | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {"accuracy": "Accuracy", "macro_f1": "Macro F1", "detection_micro_f1": "Detection micro F1", "pair_micro_f1": "Aspect-polarity pair micro F1"}
    for row in comparison:
        lines.append(f"| {row['task']} | {labels[row['metric']]} | {row['e8_test_percent']:.2f} | {row['bami_bert_paper_percent']:.2f} | {row['delta_e8_minus_bami_percent']:+.2f} |")
    lines.extend([
        "",
        "The single-seed differences are point comparisons and do not establish statistical significance. Local VSFC macro-F1 and ABSA pair-micro-F1 evaluator/averaging equivalence with the paper remains unresolved; the reported rows are retained as the project’s explicit local metrics. ViSFD OTHERS is included in detection and excluded from polarity pairs.",
        "",
        f"Manifest: `{MANIFEST_PATH}` (SHA-256 `{readout['selection_manifest']['sha256']}`). Paper targets: `{PAPER_TARGET_PATH}`; [BamiBERT paper](https://arxiv.org/html/2607.02259).",
    ])
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--device" in argv:
        index = argv.index("--device")
        device_name = argv[index + 1]
    else:
        device_name = _choose_device()
    manifest = build_manifest()
    print(json.dumps({"manifest": str(MANIFEST_PATH), "manifest_sha256": sha256_file(MANIFEST_PATH), "checkpoint_count": manifest["checkpoint_count"], "device": device_name}), flush=True)
    if "--manifest-only" in argv:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA disappeared after manifest creation; refusing CPU fallback")
    started = _now()
    results: list[dict[str, Any]] = []
    for row in manifest["selections"]:
        result = evaluate_one(row, device_name)
        results.append(result)
        print(json.dumps({"completed": len(results), "total": 5, "task_id": result["task_id"], "status": result["status"], "n": result["prediction_count"], "device": result["runtime"].get("device")}), flush=True)
    results.sort(key=lambda item: TASKS.index(item["task_id"]))
    if len(results) != 5 or any(item["prediction_count"] != EXPECTED_COUNTS[item["task_id"]] for item in results):
        raise RuntimeError("E8 test did not produce five complete outputs")
    access_log = {
        "protocol": "E8-test-gate-v1",
        "access_log_id": "E8-test-access-20260910-seed42-v1",
        "started_at_utc": started,
        "completed_at_utc": _now(),
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "policy": "five locked E8 NC+LLRD seed-42 checkpoints, each evaluated once; no test selection or tuning",
        "device": device_name,
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "mode": "inference_only",
        "results": results,
    }
    write_json(ACCESS_LOG_PATH, access_log)
    readout, comparison = build_readout(manifest, results, started)
    write_outputs(readout, comparison)
    print(json.dumps({"readout": str(READOUT_PATH), "comparison": str(COMPARISON_PATH), "access_log": str(ACCESS_LOG_PATH), "doc": str(RESULT_DOC), "count": len(results)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
