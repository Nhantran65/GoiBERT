"""One-pass CPU audit of completed E7 validation predictions.

The audit checks validation order/IDs and gold labels, then recomputes the
task metrics from the saved predictions.  It deliberately calls the audited
loader with ``include_test=False``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goibert import train  # noqa: E402


TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
E7_ROOT = ROOT / "artifacts" / "e7"
OUTPUT = E7_ROOT / "prediction_audit.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=train._json_default)
        handle.write("\n")
    temporary.replace(path)


def _canonical(value: Any) -> Any:
    """Make tuple/list labels comparable after JSON serialization."""

    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def main() -> int:
    rows: list[dict[str, Any]] = []
    checks: list[bool] = []
    for task_id in TASKS:
        artifact_dir = E7_ROOT / task_id / "seed_42"
        metadata = _read_json(artifact_dir / "metadata.json")
        prediction_path = artifact_dir / "validation_predictions.json"
        prediction_payload = _read_json(prediction_path)
        saved_rows = prediction_payload.get("rows")
        if not isinstance(saved_rows, list) or not saved_rows:
            raise ValueError(f"Missing saved predictions for {task_id}")
        bundle = train.load_task_bundle(task_id, ROOT, include_test=False)
        valid_split = train.split_field(bundle, "validation")
        expected_ids = list(train.split_ids(valid_split))
        expected_gold = [_canonical(value) for value in train.split_labels(valid_split)]
        actual_ids = [row["id"] for row in saved_rows]
        actual_gold = [_canonical(row["gold"]) for row in saved_rows]
        predicted = [row["prediction"] for row in saved_rows]
        ids_match = [str(value) for value in actual_ids] == [str(value) for value in expected_ids]
        gold_match = actual_gold == expected_gold
        recomputed = train.evaluate_predictions(task_id, bundle, actual_gold, predicted, split="validation")
        saved_metrics = _read_json(artifact_dir / "best_validation_metrics.json")
        metric_match = all(
            isinstance(saved_metrics.get(key), (int, float))
            and isinstance(value, (int, float))
            and abs(float(saved_metrics[key]) - float(value)) <= 1e-12
            for key, value in recomputed.items()
            if isinstance(value, (int, float))
        )
        row_checks = {
            "status_completed": metadata.get("status") == "completed",
            "test_excluded": metadata.get("include_test") is False,
            "prediction_count_matches_validation": len(saved_rows) == len(expected_ids),
            "ids_match_validation_order": ids_match,
            "gold_labels_match_validation": gold_match,
            "metrics_recomputed_match_saved": metric_match,
        }
        checks.extend(row_checks.values())
        rows.append(
            {
                "task_id": task_id,
                "artifact_dir": str(artifact_dir),
                "prediction_sha256": _sha256(prediction_path),
                "prediction_count": len(saved_rows),
                "best_epoch": metadata.get("best_epoch"),
                "saved_selection_f1": saved_metrics.get("selection_f1"),
                "recomputed_selection_f1": recomputed.get("selection_f1"),
                "checks": row_checks,
            }
        )
    payload = {
        "protocol": "E7-validation-prediction-audit-v1",
        "status": "passed" if all(checks) else "failed",
        "test_accessed": False,
        "metric_recomputation": "goibert.train.evaluate_predictions on saved validation rows",
        "rows": rows,
    }
    _write_json(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=train._json_default))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
