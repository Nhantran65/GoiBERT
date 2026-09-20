#!/usr/bin/env python3
"""Recompute test metrics from the saved prediction files on disk."""

from __future__ import annotations

import json
from pathlib import Path

from train_intent import BENCHMARK, load_locked_config, metric_report


ROOT = Path(__file__).resolve().parents[2]
RESULTS = BENCHMARK / "results"


def main() -> None:
    config = load_locked_config()
    audit = []
    for model_name in ("bamibert", "goi_dapt", "goi_student"):
        prediction_path = RESULTS / f"test_predictions_{model_name}.jsonl"
        metric_path = RESULTS / f"test_metrics_{model_name}.json"
        rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        stored = json.loads(metric_path.read_text(encoding="utf-8"))["test_metrics"]
        ids = [row["sample_id"] for row in rows]
        gold = [row["gold_id"] for row in rows]
        predictions = [row["pred_id"] for row in rows]
        if len(rows) != 77 or len(ids) != len(set(ids)):
            raise AssertionError(f"saved prediction identity/count mismatch: {model_name}")
        recomputed = metric_report(gold, predictions, config["labels"])
        # Loss is not recoverable from class IDs, so compare all prediction-derived
        # metrics and confusion/per-intent values exactly.
        comparable = {key: recomputed[key] for key in ("accuracy", "macro_f1", "per_intent", "confusion_matrix", "support")}
        stored_comparable = {key: stored[key] for key in comparable}
        if comparable != stored_comparable:
            raise AssertionError(f"saved prediction metric mismatch: {model_name}")
        audit.append({"model_name": model_name, "rows": len(rows), "metrics_match": True, "macro_f1": recomputed["macro_f1"], "accuracy": recomputed["accuracy"]})
    payload = {"status": "pass", "models": audit, "source": "saved test prediction JSONL files"}
    (RESULTS / "saved_prediction_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
