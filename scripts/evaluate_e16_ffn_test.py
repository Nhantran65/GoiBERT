"""Authorized E16 test inference for baseline and ffn_input_gate only.

This file is isolated from the E16 training runner.  It reads the locked
validation selections, writes a new test authorization manifest before any
``include_test=True`` call, strictly restores the ten selected checkpoints,
and records raw IDs/gold labels, logits, predictions and metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e16"
MODEL_DIR = ROOT / "models" / "bamibert-base-original"
AUTH_PATH = ARTIFACT_ROOT / "test_authorization_ffn_baseline.json"
VALIDATION_LOCK = ARTIFACT_ROOT / "locked_validation_manifest.json"
TEST_SUMMARY = ARTIFACT_ROOT / "test_summary_ffn_baseline.json"
TEST_ACCESS_LOG = ARTIFACT_ROOT / "test_access_log_ffn_baseline.json"
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
ARMS = ("baseline", "ffn_input_gate")
SEED = 42
MAX_LENGTH = 256
MICRO_BATCH_SIZE = 16
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
MIN_FREE_MIB = 12_000
WEIGHT_SHA256 = "c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627"
PROTOCOL = "E16-authorized-test-baseline-ffn-input-gate-v1"
REVISION = "e16-test-20260912-r1"

if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e16.modeling import build_e16_classifier  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=lambda item: item.tolist() if isinstance(item, torch.Tensor) else str(item))
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def gpu_snapshot() -> list[dict[str, int]]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT, timeout=30)
    rows = []
    for line in output.strip().splitlines():
        index, free, used = [int(float(value.strip())) for value in line.split(",")[:3]]
        rows.append({"index": index, "free_mib": free, "used_mib": used})
    if not rows:
        raise RuntimeError("No GPUs available for authorized E16 test")
    return rows


def choose_device(snapshot: list[dict[str, int]]) -> str:
    eligible = [row for row in snapshot if row["free_mib"] >= MIN_FREE_MIB]
    if not eligible:
        raise RuntimeError(f"No GPU has the required {MIN_FREE_MIB} MiB free for E16 test: {snapshot}")
    return f"cuda:{max(eligible, key=lambda row: row["free_mib"])["index"]}"


def task_contract(task_id: str, *, include_test: bool) -> dict[str, Any]:
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=include_test)
    return {
        "bundle": bundle,
        "train": e1_train.split_field(bundle, "train"),
        "validation": e1_train.split_field(bundle, "validation"),
        "test": e1_train.split_field(bundle, "test") if include_test else None,
        "num_classes": e1_train.task_num_classes(bundle),
        "num_aspects": e1_train.task_num_aspects(bundle),
        "aspect_class_counts": e1_train.task_aspect_class_counts(bundle),
    }


def verify_calibration_state(state: Mapping[str, Any], metadata: Mapping[str, Any], sc: float, sx: float) -> dict[str, Any]:
    model_state = state.get("model", {})
    scalar_keys = {"model_shared_sc": "shared_sc", "model_shared_sx": "shared_sx", "layer_sc": "encoder.encoder.layer.7.calibration_sc", "layer_sx": "encoder.encoder.layer.7.calibration_sx"}
    values: dict[str, float] = {}
    for label, key in scalar_keys.items():
        if key not in model_state:
            raise RuntimeError(f"Checkpoint missing calibration buffer {key}")
        value = float(model_state[key].detach().cpu().item())
        values[label] = value
        expected = sc if "sc" in label else sx
        if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError(f"Calibration buffer mismatch for {key}: {value} != {expected}")
    metadata_calibration = metadata.get("calibration", {})
    if not math.isclose(float(metadata_calibration.get("sc", -1.0)), sc, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(float(metadata_calibration.get("sx", -1.0)), sx, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Checkpoint metadata calibration differs from authorized scalars")
    return values


def selections_from_lock() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock = read_json(VALIDATION_LOCK)
    if not lock.get("locked") or lock.get("test_accessed_before_lock"):
        raise RuntimeError("Validation lock is missing or was accessed before lock")
    rows = [row for row in lock.get("selections", []) if row.get("arm") in ARMS]
    if len(rows) != 10 or {row["arm"] for row in rows} != set(ARMS) or {row["task_id"] for row in rows} != set(TASKS):
        raise RuntimeError("Authorized test scope must contain exactly baseline and ffn_input_gate across all five tasks")
    for row in rows:
        checkpoint = Path(row["artifact_dir"]) / "best_checkpoint.pt"
        if not checkpoint.is_file() or sha256_file(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Locked checkpoint hash mismatch: {row['arm']}/{row['task_id']}")
        metadata = read_json(checkpoint.parent / "metadata.json")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        verify_calibration_state(state, metadata, float(row["calibration_sc"]), float(row["calibration_sx"]))
        if int(metadata["best_epoch"]) != int(row["best_epoch"]):
            raise RuntimeError(f"Best epoch mismatch: {row['arm']}/{row['task_id']}")
    return lock, rows


def write_authorization(lock: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if AUTH_PATH.is_file():
        existing = read_json(AUTH_PATH)
        if existing.get("test_data_load_started") or existing.get("test_data_loaded"):
            raise RuntimeError("E16 authorized test was already started; refusing duplicate test access")
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "authorization_timestamp": now(),
        "status": "authorized_before_test_data_load",
        "approved_scope": {"arms": list(ARMS), "tasks": list(TASKS), "seed": SEED, "checkpoint_inferences": 10},
        "excluded_scope": {"arms": ["attention_update_gate"], "new_training": False, "new_seeds": False},
        "validation_manifest": str(VALIDATION_LOCK.relative_to(ROOT)),
        "validation_manifest_sha256": sha256_file(VALIDATION_LOCK),
        "preserve_previous_validation_flag": True,
        "test_data_load_started": False,
        "test_data_loaded": False,
        "selections": [{"arm": row["arm"], "task_id": row["task_id"], "best_epoch": row["best_epoch"], "checkpoint_sha256": row["checkpoint_sha256"], "artifact_dir": row["artifact_dir"], "calibration_sc": row["calibration_sc"], "calibration_sx": row["calibration_sx"]} for row in rows],
    }
    write_json(AUTH_PATH, payload)
    return payload


def collect_predictions(model: Any, loader: Any, device: torch.device, task_id: str, bundle: Any) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    model.eval()
    counts = e1_train.task_aspect_class_counts(bundle)
    gold: list[Any] = []
    predictions: list[Any] = []
    rows: list[dict[str, Any]] = []
    logits_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            moved = e1_train._move_batch(batch, device)
            with e1_train.make_autocast(PRECISION, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
            prediction = e1_train._predictions(logits, aspect_class_counts=counts)
            batch_gold = batch["gold_labels"].tolist()
            batch_ids = list(batch["ids"])
            gold.extend(batch_gold)
            predictions.extend(prediction)
            logits_parts.append(logits.detach().float().cpu())
            rows.extend({"id": example_id, "gold_raw": target, "prediction_model": pred} for example_id, target, pred in zip(batch_ids, batch_gold, prediction))
    metrics = e1_train.evaluate_predictions(task_id, bundle, gold, predictions, split="test")
    recomputed = e1_train.evaluate_predictions(task_id, bundle, [row["gold_raw"] for row in rows], [row["prediction_model"] for row in rows], split="test")
    if abs(float(metrics["selection_f1"]) - float(recomputed["selection_f1"])) > 1e-12:
        raise RuntimeError(f"Test metric recompute mismatch for {task_id}")
    return metrics, rows, torch.cat(logits_parts, dim=0)


def test_one(row: Mapping[str, Any], bundle_contract: Mapping[str, Any], device: torch.device, tokenizer: Any) -> dict[str, Any]:
    task_id = str(row["task_id"]); arm = str(row["arm"]); artifact = Path(str(row["artifact_dir"]))
    metadata = read_json(artifact / "metadata.json")
    state = torch.load(artifact / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model, spec = build_e16_classifier(MODEL_DIR, num_classes=int(bundle_contract["num_classes"]), num_aspects=int(bundle_contract["num_aspects"]), aspect_class_counts=bundle_contract["aspect_class_counts"] if bundle_contract["num_aspects"] else None, arm=arm, calibration_sc=float(row["calibration_sc"]), calibration_sx=float(row["calibration_sx"]), attention_implementation=ATTENTION_IMPLEMENTATION)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    test_split = bundle_contract["test"]
    loader = e1_train._make_loader(tokenizer, test_split, max_length=MAX_LENGTH, batch_size=MICRO_BATCH_SIZE, shuffle=False, seed=SEED, labels=e1_train.model_labels(bundle_contract["bundle"], "test", test_split), gold_labels=e1_train.split_labels(test_split))
    metrics, prediction_rows, logits = collect_predictions(model, loader, device, task_id, bundle_contract["bundle"])
    prediction_path = artifact / "test_predictions_ffn_baseline.json"
    logits_path = artifact / "test_logits_ffn_baseline.pt"
    write_json(prediction_path, {"arm": arm, "task_id": task_id, "split": "test", "rows": prediction_rows, "ids_sha256": sha256_json([row["id"] for row in prediction_rows]), "gold_sha256": sha256_json([row["gold_raw"] for row in prediction_rows])})
    torch.save(logits, logits_path)
    return {"arm": arm, "task_id": task_id, "best_epoch": int(row["best_epoch"]), "selection_f1": float(metrics["selection_f1"]), "selection_metric": metrics["selection_metric"], "accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1"), "detection_micro_f1": metrics.get("detection_micro_f1"), "pair_micro_f1": metrics.get("pair_micro_f1"), "test_count": len(prediction_rows), "ids_sha256": sha256_json([item["id"] for item in prediction_rows]), "gold_sha256": sha256_json([item["gold_raw"] for item in prediction_rows]), "artifact_dir": str(artifact), "checkpoint_sha256": sha256_file(artifact / "best_checkpoint.pt"), "prediction_path": str(prediction_path.relative_to(ROOT)), "logits_path": str(logits_path.relative_to(ROOT)), "model_spec": spec.to_dict(), "metadata_calibration": metadata["calibration"]}


def write_report(payload: Mapping[str, Any]) -> None:
    rows = payload["results"]
    baseline = {row["task_id"]: row for row in rows if row["arm"] == "baseline"}
    ffn = {row["task_id"]: row for row in rows if row["arm"] == "ffn_input_gate"}
    lines = ["# E16 authorized test results", "", "Scope: baseline and ffn_input_gate, seed 42, all five tasks. attention_update_gate was excluded. Validation lock and source history were preserved.", "", "| Task | Baseline test selection F1 | FFN input gate test selection F1 | FFN − baseline (pp) |", "|---|---:|---:|---:|"]
    for task in TASKS:
        lines.append(f"| {task} | {baseline[task]['selection_f1']:.8f} | {ffn[task]['selection_f1']:.8f} | {100.0 * (ffn[task]['selection_f1'] - baseline[task]['selection_f1']):+.4f} |")
    lines += ["", "All values are evaluator fractions in [0,1]; displayed deltas are percentage points.", "", "## Checkpoint and data facts", "", f"- Authorized checkpoint inferences: {len(rows)}.", f"- Test counts: {', '.join(f'{row['task_id']}={row['test_count']}' for row in rows if row['arm']=='baseline') }.", "- ABSA metadata overlay: 0 absent, 1 positive, 2 negative, 3 neutral; integer gold labels were preserved.", "- ViSFD OTHERS: raw code 4/model code 1 conversion for detection; pair metric masked.", "- Test access was authorized in the separate manifest before the first test bundle load.", ""]
    (ARTIFACT_ROOT / "E16_FFN_TEST_RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    lock, rows = selections_from_lock()
    authorization = write_authorization(lock, rows)
    # This status transition is durable and occurs before the first
    # include_test=True call below.
    authorization["status"] = "test_data_load_started"
    authorization["test_data_load_started"] = True
    authorization["test_data_load_started_at"] = now()
    write_json(AUTH_PATH, authorization)
    snapshot = gpu_snapshot()
    device = torch.device(choose_device(snapshot))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    by_task: dict[str, dict[str, Any]] = {}
    access_entries: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    access_started = now()
    for task_id in TASKS:
        contract = task_contract(task_id, include_test=True)
        by_task[task_id] = contract
        for row in [item for item in rows if item["task_id"] == task_id]:
            result = test_one(row, contract, device, tokenizer)
            result["test_loaded_at"] = access_started
            results.append(result)
            access_entries.append({"arm": row["arm"], "task_id": task_id, "split": "test", "count": result["test_count"], "ids_sha256": result["ids_sha256"], "gold_sha256": result["gold_sha256"], "accessed_at": now(), "artifact_dir": result["artifact_dir"]})
        write_json(TEST_ACCESS_LOG, {"protocol": PROTOCOL, "authorization_manifest": str(AUTH_PATH.relative_to(ROOT)), "entries": access_entries, "test_accessed": True})
    results.sort(key=lambda item: (ARMS.index(item["arm"]), TASKS.index(item["task_id"])))
    payload = {"protocol": PROTOCOL, "revision": REVISION, "authorization_manifest": str(AUTH_PATH.relative_to(ROOT)), "validation_manifest": str(VALIDATION_LOCK.relative_to(ROOT)), "source_history_preserved": True, "test_split": "test", "seed": SEED, "precision": PRECISION, "attention_implementation": ATTENTION_IMPLEMENTATION, "micro_batch_size": MICRO_BATCH_SIZE, "max_length": MAX_LENGTH, "fractions_0_to_1": True, "results": results, "gpu_snapshot": snapshot, "device": str(device), "test_accessed": True}
    write_json(TEST_SUMMARY, payload)
    with (ARTIFACT_ROOT / "test_summary_ffn_baseline.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["arm", "task_id", "best_epoch", "selection_f1", "selection_metric", "accuracy", "macro_f1", "detection_micro_f1", "pair_micro_f1", "test_count", "ids_sha256", "gold_sha256", "artifact_dir", "checkpoint_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({field: row.get(field) for field in fields} for row in results)
    write_json(TEST_ACCESS_LOG, {"protocol": PROTOCOL, "authorization_manifest": str(AUTH_PATH.relative_to(ROOT)), "entries": access_entries, "test_accessed": True, "completed_at": now()})
    authorization["status"] = "completed"
    authorization["test_data_loaded"] = True
    authorization["test_data_loaded_at"] = now()
    authorization["test_access_log"] = str(TEST_ACCESS_LOG.relative_to(ROOT))
    authorization["test_summary"] = str(TEST_SUMMARY.relative_to(ROOT))
    write_json(AUTH_PATH, authorization)
    write_report(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
