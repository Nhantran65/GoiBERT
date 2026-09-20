#!/usr/bin/env python3
"""Read-only audit of E17 validation IDs, gold labels, and selection F1.

The audit intentionally consumes saved predictions and the frozen validation
split.  It does not load a model, run inference, or alter any trial artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "e17"
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
ARMS = ("baseline", "full_context", "cavity")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    return value


def f1_for_single(gold: list[int], prediction: list[int], classes: list[int]) -> float:
    scores = []
    for label in classes:
        tp = sum(g == label and p == label for g, p in zip(gold, prediction))
        fp = sum(g != label and p == label for g, p in zip(gold, prediction))
        fn = sum(g == label and p != label for g, p in zip(gold, prediction))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / len(scores)


def f1_for_pairs(gold: list[list[int]], prediction: list[list[int]], pair_scored: list[bool]) -> float:
    tp = fp = fn = 0
    for gold_row, prediction_row in zip(gold, prediction):
        for g, p, scored in zip(gold_row, prediction_row, pair_scored):
            if not scored:
                continue
            if g > 0 and p == g:
                tp += 1
            elif p > 0:
                fp += 1
                if g > 0:
                    fn += 1
            elif g > 0:
                fn += 1
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def task_contract(task_id: str) -> tuple[list[str], list[Any], int]:
    import sys

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from goibert_e2 import train as e2_train  # type: ignore

    bundle = e2_train.load_task_bundle(task_id, ROOT, include_test=False, smoke=False)
    split = e2_train.split_field(bundle, "validation")
    ids = [str(value) for value in e2_train.split_ids(split)]
    labels = plain(e2_train.split_labels(split))
    return ids, labels, int(e2_train.task_num_classes(bundle))


def audit_trial(task_id: str, arm: str, expected_ids: list[str], expected_gold: list[Any], num_classes: int) -> dict[str, Any]:
    trial = ARTIFACT_ROOT / arm / task_id / "seed_42"
    status = read_json(trial / "status.json")
    pred_payload = read_json(trial / "validation_predictions.json")
    rows = pred_payload.get("rows", [])
    saved_ids = [str(row["id"]) for row in rows]
    saved_gold = plain([row["gold"] for row in rows])
    saved_prediction = plain([row["prediction"] for row in rows])
    ids_match = saved_ids == expected_ids
    gold_match = saved_gold == expected_gold
    if task_id in {"vsfc_sentiment", "vsfc_topic"}:
        selection_f1 = f1_for_single(saved_gold, saved_prediction, list(range(num_classes)))
    else:
        metrics = read_json(trial / "best_validation_metrics.json")
        pair_scored = [bool(value) for value in metrics["pair_scored_aspects"]]
        selection_f1 = f1_for_pairs(saved_gold, saved_prediction, pair_scored)
    recorded = read_json(trial / "best_validation_metrics.json").get("selection_f1")
    return {
        "task": task_id,
        "arm": arm,
        "status": status.get("status"),
        "n_saved": len(rows),
        "n_expected": len(expected_ids),
        "ids_exact": ids_match,
        "gold_exact": gold_match,
        "selection_f1_recomputed": selection_f1,
        "selection_f1_recorded": recorded,
        "selection_f1_abs_error": None if recorded is None else abs(selection_f1 - float(recorded)),
        "prediction_path": str(trial / "validation_predictions.json"),
    }


def run() -> dict[str, Any]:
    rows = []
    task_checks = []
    for task in TASKS:
        expected_ids, expected_gold, num_classes = task_contract(task)
        task_rows = []
        for arm in ARMS:
            trial = ARTIFACT_ROOT / arm / task / "seed_42"
            status_path = trial / "status.json"
            if not (trial / "validation_predictions.json").is_file() or not status_path.is_file():
                continue
            if read_json(status_path).get("status") != "completed":
                continue
            result = audit_trial(task, arm, expected_ids, expected_gold, num_classes)
            rows.append(result)
            task_rows.append(result)
        task_checks.append({
            "task": task,
            "expected_n": len(expected_ids),
            "arms_audited": len(task_rows),
            "ids_exact_all": all(row["ids_exact"] for row in task_rows),
            "gold_exact_all": all(row["gold_exact"] for row in task_rows),
        })
    return {
        "protocol": "E17-recipient-excluded-keys-screening-v1",
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "validation_only": True,
        "rows": rows,
        "task_checks": task_checks,
        "all_ids_exact": bool(rows) and all(row["ids_exact"] for row in rows),
        "all_gold_exact": bool(rows) and all(row["gold_exact"] for row in rows),
        "all_f1_recomputed_within_1e-12": bool(rows) and all((row["selection_f1_abs_error"] or 0.0) <= 1e-12 for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "recovery_audit.json")
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
