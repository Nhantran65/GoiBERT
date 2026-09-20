"""Read-only E16 artifact audit after the validation controller completes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from goibert import train as e1_train
from goibert_e16.modeling import build_e16_classifier


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "e16"
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
ARMS = ("baseline", "attention_update_gate", "ffn_input_gate")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    summary = json.loads((ARTIFACT_ROOT / "validation_summary.json").read_text())
    preflight = json.loads((ARTIFACT_ROOT / "preflight.json").read_text())
    calibration = json.loads((ARTIFACT_ROOT / "calibration_manifest.json").read_text())
    rows = summary["rows"]
    checks: dict[str, Any] = {
        "row_count_15": len(rows) == 15,
        "all_arm_task_pairs_unique": len({(row["arm"], row["task_id"]) for row in rows}) == 15,
        "all_required_artifacts": True,
        "all_checkpoint_required_keys": True,
        "all_gate_shapes": True,
        "all_30_epoch_curves": True,
        "all_gradient_probe_steps": True,
        "all_30_epoch_train_probes": True,
        "recipe_fixed": True,
        "calibration_shared": True,
        "test_access_false": bool(summary.get("test_accessed") is False and summary.get("test_authorized") is False),
        "preflight_passed": all(bool(value) for value in preflight["checks"].values()),
        "strict_constructor_roundtrip": True,
    }
    trial_rows: list[dict[str, Any]] = []
    strict_roundtrip: list[dict[str, Any]] = []
    for row in rows:
        artifact = Path(row["artifact_dir"])
        required = ["metadata.json", "resolved_config.json", "best_checkpoint.pt", "best_validation_metrics.json", "validation_predictions.json", "epoch_curves.json", "gradient_logs.jsonl", "probe_logs.jsonl", "status.json"]
        if not all((artifact / name).is_file() for name in required):
            checks["all_required_artifacts"] = False
            continue
        metadata = json.loads((artifact / "metadata.json").read_text())
        curves = json.loads((artifact / "epoch_curves.json").read_text()).get("epochs", [])
        gradients = [json.loads(line) for line in (artifact / "gradient_logs.jsonl").read_text().splitlines() if line.strip()]
        probes = [json.loads(line) for line in (artifact / "probe_logs.jsonl").read_text().splitlines() if line.strip()]
        state = torch.load(artifact / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        required_state = {"model", "optimizer", "scheduler", "epoch", "global_step", "rng_state"}
        if not required_state.issubset(state):
            checks["all_checkpoint_required_keys"] = False
        gate_keys = [key for key in state.get("model", {}) if key.endswith(".gate.weight")]
        expected_gate = 0 if row["arm"] == "baseline" else 1
        if len(gate_keys) != expected_gate or any(tuple(state["model"][key].shape) != (3072, 768) for key in gate_keys):
            checks["all_gate_shapes"] = False
        if len(curves) != 30:
            checks["all_30_epoch_curves"] = False
        if {int(item.get("optimizer_step", -1)) for item in gradients} != {0, 1, 10, 100}:
            checks["all_gradient_probe_steps"] = False
        if len(probes) != 30:
            checks["all_30_epoch_train_probes"] = False
        recipe_ok = metadata.get("micro_batch_size") == 16 and metadata.get("gradient_accumulation_steps") == 2 and metadata.get("effective_batch_size") == 32 and metadata.get("epochs") == 30 and metadata.get("precision") == "bf16" and metadata.get("max_length") == 256 and metadata.get("include_test") is False
        if not recipe_ok:
            checks["recipe_fixed"] = False
        if float(metadata.get("calibration", {}).get("sc", 0.0)) != float(calibration["sc"]) or float(metadata.get("calibration", {}).get("sx", 0.0)) != float(calibration["sx"]):
            checks["calibration_shared"] = False
        try:
            bundle = e1_train.load_task_bundle(row["task_id"], ROOT, include_test=False)
            model, spec = build_e16_classifier(ROOT / "models" / "bamibert-base-original", num_classes=e1_train.task_num_classes(bundle), num_aspects=e1_train.task_num_aspects(bundle), aspect_class_counts=e1_train.task_aspect_class_counts(bundle) if e1_train.task_num_aspects(bundle) else None, arm=row["arm"], calibration_sc=float(calibration["sc"]), calibration_sx=float(calibration["sx"]))
            model.load_state_dict(state["model"], strict=True)
            strict_roundtrip.append({"arm": row["arm"], "task_id": row["task_id"], "added_parameter_count": spec.added_parameter_count})
            del model
        except Exception as error:
            checks["strict_constructor_roundtrip"] = False
            strict_roundtrip.append({"arm": row["arm"], "task_id": row["task_id"], "error": repr(error)})
        trial_rows.append({"arm": row["arm"], "task_id": row["task_id"], "artifact_dir": str(artifact), "checkpoint_sha256": sha256(artifact / "best_checkpoint.pt"), "gate_keys": gate_keys, "curves": len(curves), "gradient_steps": sorted({int(item.get("optimizer_step", -1)) for item in gradients}), "probe_rows": len(probes), "source_hash": metadata.get("source_hash"), "test_accessed": metadata.get("test_accessed")})
    result = {"protocol": summary["protocol"], "audited_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()), "checks": checks, "source_hashes_observed": sorted({row.get("source_hash") for row in trial_rows}), "calibration": {"sc": calibration["sc"], "sx": calibration["sx"], "actual_total": calibration["actual_total"], "domain_counts": calibration["domain_counts"]}, "strict_constructor_roundtrip": strict_roundtrip, "trials": trial_rows}
    write_json(ARTIFACT_ROOT / "postrun_validation.json", result)
    if not all(bool(value) for value in checks.values()):
        raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
