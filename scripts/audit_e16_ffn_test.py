"""Postrun integrity audit for the authorized E16 baseline/FFN test pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from goibert import train as e1_train


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "e16"
AUTH_PATH = ARTIFACT_ROOT / "test_authorization_ffn_baseline.json"
SUMMARY_PATH = ARTIFACT_ROOT / "test_summary_ffn_baseline.json"
LOCK_PATH = ARTIFACT_ROOT / "locked_validation_manifest.json"
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
ARMS = ("baseline", "ffn_input_gate")
KNOWN_COUNTS = {"vsfc_sentiment": 3166, "vsfc_topic": 3166, "visfd": 2224, "absa_restaurant": 1938, "absa_hotel": 2030}


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


def canonical(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(canonical(item) for item in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return int(value)
    return value


def main() -> int:
    auth = json.loads(AUTH_PATH.read_text())
    summary = json.loads(SUMMARY_PATH.read_text())
    lock = json.loads(LOCK_PATH.read_text())
    rows = summary["results"]
    checks: dict[str, bool] = {
        "authorization_completed": auth.get("status") == "completed" and auth.get("test_data_loaded") is True,
        "previous_validation_manifest_preserved": lock.get("test_accessed_before_lock") is False,
        "exact_unique_10_pairs": len(rows) == 10 and len({(row["arm"], row["task_id"]) for row in rows}) == 10 and {(row["arm"], row["task_id"]) for row in rows} == {(arm, task) for arm in ARMS for task in TASKS},
        "attention_gate_absent": all(row["arm"] != "attention_update_gate" for row in rows),
        "checkpoint_hashes_locked": True,
        "known_test_counts": True,
        "ordered_ids_match_test_split": True,
        "ordered_raw_gold_match_test_split": True,
        "metric_recompute_match": True,
        "numeric_gold_preserved": True,
    }
    locked = {(row["arm"], row["task_id"]): row for row in lock["selections"]}
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        key = (row["arm"], row["task_id"])
        if key not in locked:
            checks["checkpoint_hashes_locked"] = False
        artifact = Path(row["artifact_dir"])
        checkpoint = artifact / "best_checkpoint.pt"
        if not checkpoint.is_file() or sha256(checkpoint) != locked[key]["checkpoint_sha256"] or sha256(checkpoint) != row["checkpoint_sha256"]:
            checks["checkpoint_hashes_locked"] = False
        prediction_path = ROOT / row["prediction_path"]
        prediction = json.loads(prediction_path.read_text())
        prediction_rows = prediction["rows"]
        task_id = row["task_id"]
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=True)
        test_split = e1_train.split_field(bundle, "test")
        expected_ids = list(e1_train.split_ids(test_split))
        expected_gold = list(e1_train.split_labels(test_split))
        actual_ids = [item["id"] for item in prediction_rows]
        actual_gold = [item["gold_raw"] for item in prediction_rows]
        if len(prediction_rows) != KNOWN_COUNTS[task_id] or row["test_count"] != KNOWN_COUNTS[task_id]:
            checks["known_test_counts"] = False
        if actual_ids != expected_ids:
            checks["ordered_ids_match_test_split"] = False
        if canonical(actual_gold) != canonical(expected_gold):
            checks["ordered_raw_gold_match_test_split"] = False
        if task_id in {"visfd", "absa_restaurant", "absa_hotel"}:
            if any(any(int(value) not in ({0, 1, 2, 3, 4}) for value in values) for values in actual_gold):
                checks["numeric_gold_preserved"] = False
        recomputed = e1_train.evaluate_predictions(task_id, bundle, actual_gold, [item["prediction_model"] for item in prediction_rows], split="test")
        if abs(float(recomputed["selection_f1"]) - float(row["selection_f1"])) > 1e-12:
            checks["metric_recompute_match"] = False
        audit_rows.append({"arm": row["arm"], "task_id": task_id, "count": len(prediction_rows), "ids_exact": actual_ids == expected_ids, "gold_exact": canonical(actual_gold) == canonical(expected_gold), "selection_f1": row["selection_f1"], "checkpoint_sha256": row["checkpoint_sha256"], "prediction_path": row["prediction_path"], "logits_path": row["logits_path"]})
    result = {"protocol": "E16-authorized-test-postrun-audit-v1", "audited_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()), "authorization_manifest": str(AUTH_PATH.relative_to(ROOT)), "checks": checks, "known_test_counts": KNOWN_COUNTS, "rows": audit_rows}
    write_json(ARTIFACT_ROOT / "test_postrun_validation_ffn_baseline.json", result)
    if not all(checks.values()):
        raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
