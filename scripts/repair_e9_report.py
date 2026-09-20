"""Repair E9's post-training report units from immutable validation artifacts.

The training artifacts store F1 as a fraction in [0, 1].  This report tool
loads the actual E8 seed-42 ``best_validation_metrics.json`` files, recomputes
E9 validation metrics from saved predictions, and writes deltas in percentage
points.  It does not load or run a test split and does not alter checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from goibert import train as e1_train


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "e9"
TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
PROTOCOL = "E9-label-aware-attention-screening-v1"
SEED = 42
EPOCHS = 30
EXPECTED_E8_FROM_PLAN_PERCENT = {
    "vsfc_sentiment": 85.1604428,
    "vsfc_topic": 80.3677890,
    "visfd": 82.4420542,
    "absa_restaurant": 71.2765957,
    "absa_hotel": 73.0490748,
}


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _archive(path: Path) -> str | None:
    if not path.is_file():
        return None
    candidate = path.with_name(f"{path.stem}_buggy_units{path.suffix}")
    if candidate.exists():
        index = 1
        while True:
            candidate = path.with_name(f"{path.stem}_buggy_units__rerun{index}{path.suffix}")
            if not candidate.exists():
                break
            index += 1
    shutil.copy2(path, candidate)
    return str(candidate)


def _assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{name}: {actual} != {expected}")


def _task_row(task_id: str) -> dict[str, Any]:
    artifact = ARTIFACT_ROOT / task_id / f"seed_{SEED}"
    status = _read(artifact / "status.json")
    metadata = _read(artifact / "metadata.json")
    if status.get("status") != "completed":
        raise AssertionError(f"{task_id}: status is not completed")
    if metadata.get("protocol") != PROTOCOL:
        raise AssertionError(f"{task_id}: wrong protocol")
    if int(metadata.get("epochs", -1)) != EPOCHS:
        raise AssertionError(f"{task_id}: metadata epochs is not {EPOCHS}")
    curves = _read(artifact / "epoch_curves.json").get("epochs")
    if not isinstance(curves, list) or len(curves) != EPOCHS:
        raise AssertionError(f"{task_id}: expected exactly {EPOCHS} epoch rows")

    predictions = _read(artifact / "validation_predictions.json").get("rows")
    if not isinstance(predictions, list) or not predictions:
        raise AssertionError(f"{task_id}: missing validation predictions")
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
    valid_split = e1_train.split_field(bundle, "validation")
    expected_ids = [str(value) for value in e1_train.split_ids(valid_split)]
    actual_ids = [str(row["id"]) for row in predictions]
    if actual_ids != expected_ids:
        raise AssertionError(f"{task_id}: validation IDs/order mismatch")
    e9_metrics = e1_train.evaluate_predictions(
        task_id,
        bundle,
        [row["gold"] for row in predictions],
        [row["prediction"] for row in predictions],
        split="validation",
    )
    e9_fraction = float(e9_metrics["selection_f1"])
    _assert_close(
        f"{task_id} saved/recomputed E9 selection_f1",
        float(metadata["best_selection_f1"]),
        e9_fraction,
    )

    e8_metrics_path = ROOT / "artifacts" / "e8" / task_id / f"seed_{SEED}" / "best_validation_metrics.json"
    e8_metrics = _read(e8_metrics_path)
    e8_fraction = float(e8_metrics["selection_f1"])
    expected_plan_fraction = EXPECTED_E8_FROM_PLAN_PERCENT[task_id] / 100.0
    _assert_close(f"{task_id} actual E8 vs plan baseline", e8_fraction, expected_plan_fraction, 1e-8)
    delta_fraction = e9_fraction - e8_fraction
    delta_pp = delta_fraction * 100.0
    return {
        "task_id": task_id,
        "artifact_dir": str(artifact),
        "seed": SEED,
        "epochs": EPOCHS,
        "best_epoch": metadata.get("best_epoch"),
        "selection_f1_e8_fraction": e8_fraction,
        "selection_f1_e8_percent": e8_fraction * 100.0,
        "selection_f1_e9_fraction": e9_fraction,
        "selection_f1_e9_percent": e9_fraction * 100.0,
        "delta_e9_minus_e8_fraction": delta_fraction,
        "delta_e9_minus_e8_pp": delta_pp,
        "improved": delta_fraction > 0.0,
        "improved_at_least_0p3pp": delta_pp >= 0.3,
        "decreased_over_0p3pp": delta_pp < -0.3,
        "elapsed_seconds": metadata.get("elapsed_seconds"),
        "peak_gpu_memory_allocated_bytes": metadata.get("peak_gpu_memory_allocated_bytes"),
        "peak_gpu_memory_reserved_bytes": metadata.get("peak_gpu_memory_reserved_bytes"),
        "prediction_count": len(predictions),
        "e8_metrics_source": str(e8_metrics_path),
        "metrics_e9": e9_metrics,
    }


def main() -> int:
    archived_summary = _archive(ARTIFACT_ROOT / "summary.json")
    archived_csv = _archive(ARTIFACT_ROOT / "summary.csv")
    archived_readout = _archive(ROOT / "docs" / "E9_RESULT.md")
    rows = [_task_row(task_id) for task_id in TASKS]
    increased = sum(bool(row["improved"]) for row in rows)
    strong = sum(bool(row["improved_at_least_0p3pp"]) for row in rows)
    largest_decrease = min(float(row["delta_e9_minus_e8_pp"]) for row in rows)
    screening_pass = increased >= 4 and strong >= 2 and largest_decrease >= -0.3
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "revision": "e9-20260910-r1",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "tasks": list(TASKS),
        "units": {
            "selection_f1_storage": "fraction in [0,1]",
            "selection_f1_display": "percent",
            "delta_e9_minus_e8_pp": "(e9_fraction - e8_fraction) * 100",
            "thresholds": "positive increase; strong increase >= +0.3 percentage points; allowed decrease >= -0.3 percentage points",
        },
        "metric_recomputation": "saved validation_predictions.json, validation split only; test excluded",
        "baseline_source": "artifacts/e8/<task>/seed_42/best_validation_metrics.json selection_f1",
        "archived_buggy_report": {
            "summary_json": archived_summary,
            "summary_csv": archived_csv,
            "readout": archived_readout,
        },
        "verification": {
            "all_tasks_completed": True,
            "all_tasks_exactly_30_epochs": True,
            "all_validation_ids_and_order_verified": True,
            "all_saved_e9_metrics_recomputed_equal": True,
            "all_e8_baselines_match_plan_within_1e-8": True,
            "test_loaded": False,
        },
        "screening_gate": {
            "definition": "at least 4/5 increases, no task decreases over 0.3pp, at least 2 increases by 0.3pp",
            "tasks_increased": increased,
            "tasks_increased_at_least_0p3pp": strong,
            "largest_decrease_pp": largest_decrease,
            "passed": screening_pass,
        },
        "rows": rows,
    }
    _write(ARTIFACT_ROOT / "summary.json", payload)
    fields = [
        "task_id", "seed", "epochs", "best_epoch", "selection_f1_e8_fraction",
        "selection_f1_e8_percent", "selection_f1_e9_fraction", "selection_f1_e9_percent",
        "delta_e9_minus_e8_fraction", "delta_e9_minus_e8_pp", "improved",
        "improved_at_least_0p3pp", "decreased_over_0p3pp", "elapsed_seconds",
        "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes", "prediction_count",
    ]
    with (ARTIFACT_ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    lines = [
        "# E9 Result: Label-aware Attention",
        "",
        "Five original GoiTurn-S-GD students use one learned query per VSFC class or ABSA aspect, with the fixed E8 NC plus E7 LLRD recipe. Validation only; seed 42; no test split.",
        "",
        "F1 is stored as a fraction in artifacts and displayed as percent below. Delta is `(E9 fraction - E8 fraction) * 100` percentage points.",
        "",
        "| Task | E8 baseline (%) | E9 (%) | Delta (pp) | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['selection_f1_e8_percent']:.4f} | {row['selection_f1_e9_percent']:.4f} | {row['delta_e9_minus_e8_pp']:+.3f} | {row['best_epoch']} |"
        )
    gate = payload["screening_gate"]
    lines.extend(
        [
            "",
            f"Screening gate: {'PASS' if gate['passed'] else 'FAIL'} ({gate['tasks_increased']}/5 increased; {gate['tasks_increased_at_least_0p3pp']} increased by at least 0.3pp; largest decrease {gate['largest_decrease_pp']:+.3f}pp).",
            "",
            "Every task has completed 30 epochs. Validation IDs/order and saved-vs-recomputed E9 F1 were verified; E8 baselines came from the actual E8 best-validation metric files.",
            "",
        ]
    )
    (ROOT / "docs" / "E9_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
