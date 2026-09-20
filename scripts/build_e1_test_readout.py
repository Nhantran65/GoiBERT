"""Aggregate the locked E1 test evaluations into an auditable readout."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "artifacts" / "e1" / "test_selection_manifest.json"
OUTPUT_DIR = ROOT / "artifacts" / "e1"
EXPECTED_N = {
    "vsfc_sentiment": 3166,
    "vsfc_topic": 3166,
    "visfd": 2224,
    "absa_restaurant": 1938,
    "absa_hotel": 2030,
}
TASK_NAMES = {
    "vsfc_sentiment": "VSFC sentiment",
    "vsfc_topic": "VSFC topic",
    "visfd": "ViSFD",
    "absa_restaurant": "UIT-ABSA Restaurant",
    "absa_hotel": "UIT-ABSA Hotel",
}
PAPER_TARGETS = {
    "vsfc_sentiment": {"accuracy": 93.86, "macro_f1": 83.41},
    "vsfc_topic": {"accuracy": 89.34, "macro_f1": 79.90},
    "visfd": {"detection_micro_f1": 89.14, "pair_micro_f1": 84.24},
    "absa_restaurant": {"detection_micro_f1": 88.01, "pair_micro_f1": 74.89},
    "absa_hotel": {"detection_micro_f1": 79.99, "pair_micro_f1": 72.65},
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "detection_micro_f1": "Aspect detection micro F1",
    "pair_micro_f1": "End-to-end aspect-polarity pair micro F1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percent(value: float) -> float:
    return float(value) * 100.0


def _utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _find_index() -> Path:
    candidates = []
    for path in sorted(OUTPUT_DIR.glob("test_index_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("stage") == "test" and payload.get("executed") is True:
            candidates.append(path)
    if not candidates:
        raise RuntimeError("No executed test index found")
    return candidates[-1]


def _find_plan_index(executed_index: Path) -> Path | None:
    candidates = []
    for path in sorted(OUTPUT_DIR.glob("test_index_*.json")):
        if path == executed_index:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            payload.get("stage") == "test"
            and payload.get("trial_count") == 25
            and payload.get("executed") is not True
        ):
            candidates.append(path)
    return candidates[-1] if candidates else None


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("locked") is not True or manifest.get("checkpoint_count") != 25:
        raise RuntimeError("Expected a locked 25-checkpoint test manifest")
    selections = manifest["selections"]
    if len(selections) != 25:
        raise RuntimeError("Manifest selection count is not 25")

    selection_by_artifact = {str(Path(row["artifact_dir"]).resolve()): row for row in selections}
    if len(selection_by_artifact) != 25:
        raise RuntimeError("Manifest contains duplicate artifact selections")
    result_rows = []
    seen = set()
    for row in selections:
        artifact_dir = Path(row["artifact_dir"])
        evaluation_path = artifact_dir / "test_evaluation_metadata.json"
        metrics_path = artifact_dir / "test_metrics.json"
        predictions_path = artifact_dir / "test_predictions.json"
        for path in (evaluation_path, metrics_path, predictions_path):
            if not path.is_file():
                raise RuntimeError(f"Missing test output: {path}")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        identity = (row["task_id"], int(row["seed"]))
        if identity in seen:
            raise RuntimeError(f"Duplicate evaluated identity: {identity}")
        seen.add(identity)
        if evaluation.get("status") != "completed" or evaluation.get("protocol") != "E1-test-gate-v1":
            raise RuntimeError(f"Incomplete evaluation metadata: {evaluation_path}")
        if evaluation.get("source_hash") != manifest["source_hash"]:
            raise RuntimeError(f"Source hash mismatch in {evaluation_path}")
        if evaluation.get("checkpoint_sha256") != row["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch in {evaluation_path}")
        if evaluation.get("metadata_sha256") != row["metadata_sha256"]:
            raise RuntimeError(f"Metadata hash mismatch in {evaluation_path}")
        if evaluation.get("task_id") != row["task_id"] or int(evaluation.get("seed")) != int(row["seed"]):
            raise RuntimeError(f"Selection identity mismatch in {evaluation_path}")
        if metrics.get("split") != "test" or metrics.get("n") != EXPECTED_N[row["task_id"]]:
            raise RuntimeError(f"Unexpected test split/row count in {metrics_path}")
        prediction_rows = predictions.get("rows")
        if not isinstance(prediction_rows, list) or len(prediction_rows) != EXPECTED_N[row["task_id"]]:
            raise RuntimeError(f"Prediction row count mismatch in {predictions_path}")
        if len({item.get("id") for item in prediction_rows}) != len(prediction_rows):
            raise RuntimeError(f"Prediction IDs are not unique in {predictions_path}")
        seen.add(identity)
        result_rows.append(
            {
                "task_id": row["task_id"],
                "task": TASK_NAMES[row["task_id"]],
                "seed": int(row["seed"]),
                "learning_rate": row["learning_rate"],
                "stage": row["stage"],
                "best_epoch": row["best_epoch"],
                "n_test": metrics["n"],
                "artifact_dir": str(artifact_dir),
                "test_evaluation_metadata": str(evaluation_path),
                "test_metrics_path": str(metrics_path),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "metadata_sha256": row["metadata_sha256"],
                "source_hash": evaluation["source_hash"],
                "evaluator_code_hash": evaluation["evaluator_metadata"].get("evaluator_code_hash"),
                "runtime": evaluation["runtime"],
                "metrics": {key: metrics[key] for key in METRIC_LABELS if key in metrics},
            }
        )

    if len(result_rows) != 25 or len(seen) != 25:
        raise RuntimeError(f"Expected 25 unique test results, got {len(result_rows)}")

    by_task = {task: sorted((row for row in result_rows if row["task_id"] == task), key=lambda item: item["seed"]) for task in TASK_NAMES}
    if any(len(rows) != 5 or [row["seed"] for row in rows] != [42, 43, 44, 45, 46] for rows in by_task.values()):
        raise RuntimeError("Every task must have exactly seeds 42 through 46")

    aggregates = {}
    comparison_rows = []
    for task, rows in by_task.items():
        metrics = {}
        for metric in PAPER_TARGETS[task]:
            values = [_percent(row["metrics"][metric]) for row in rows]
            mean = statistics.mean(values)
            sd = statistics.stdev(values)
            target = PAPER_TARGETS[task][metric]
            metrics[metric] = {
                "label": METRIC_LABELS[metric],
                "per_seed_percent": {str(row["seed"]): value for row, value in zip(rows, values)},
                "mean_percent": mean,
                "sample_std_percent": sd,
                "bami_bert_paper_percent": target,
                "arithmetic_delta_percent": mean - target,
                "metric_equivalence": "resolved for Accuracy" if metric == "accuracy" else "unresolved",
            }
            comparison_rows.append(
                {
                    "task_id": task,
                    "task": TASK_NAMES[task],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "bami_bert_paper_percent": target,
                    "goibert_mean_percent": mean,
                    "goibert_sample_std_percent": sd,
                    "arithmetic_delta_percent": mean - target,
                    "metric_equivalence": metrics[metric]["metric_equivalence"],
                }
            )
        aggregates[task] = metrics

    index_path = _find_index()
    plan_index_path = _find_plan_index(index_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    access_log_path = OUTPUT_DIR / "test_access_log.json"
    access_log = {
        "protocol": "E1-test-gate-v1",
        "access_log_id": "E1-test-access-20260908-v1",
        "generated_at_utc": generated_at,
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "plan_index_path": str(plan_index_path.resolve()) if plan_index_path else None,
        "plan_index_sha256": sha256_file(plan_index_path) if plan_index_path else None,
        "executed_index_path": str(index_path.resolve()),
        "executed_index_sha256": sha256_file(index_path),
        "devices": ["cuda:0", "cuda:1", "cuda:3"],
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "mode": "inference_only",
        "policy": "25 unique locked checkpoints evaluated once; no test-based selection, tuning, or retraining",
        "rows": [
            {
                "task_id": row["task_id"],
                "seed": row["seed"],
                "artifact_dir": row["artifact_dir"],
                "test_rows": row["n_test"],
                "test_evaluation_metadata": row["test_evaluation_metadata"],
                "test_metrics": row["test_metrics_path"],
            }
            for row in sorted(result_rows, key=lambda item: (item["task_id"], item["seed"]))
        ],
    }
    access_log_path.write_text(json.dumps(access_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readout = {
        "protocol": "E1-test-gate-v1",
        "readout_id": "E1-test-readout-20260908-v1",
        "generated_at_utc": generated_at,
        "status": "complete",
        "manifest": {
            "path": str(MANIFEST_PATH.resolve()),
            "sha256": sha256_file(MANIFEST_PATH),
            "filesystem_mtime_utc": _utc_mtime(MANIFEST_PATH),
            "declared_source_hash": manifest["source_hash"],
            "selection_count": len(selections),
        },
        "execution": {
            "index_path": str(index_path.resolve()),
            "index_sha256": sha256_file(index_path),
            "index_filesystem_mtime_utc": _utc_mtime(index_path),
            "plan_index_path": str(plan_index_path.resolve()) if plan_index_path else None,
            "plan_index_sha256": sha256_file(plan_index_path) if plan_index_path else None,
            "access_log_path": str(access_log_path.resolve()),
            "access_log_sha256": sha256_file(access_log_path),
            "devices": ["cuda:0", "cuda:1", "cuda:3"],
            "precision": "bf16",
            "attention_implementation": "sdpa",
            "mode": "inference_only",
            "test_access_log": "25/25 unique locked checkpoints evaluated once; no retraining or test-based selection",
        },
        "test_row_counts": EXPECTED_N,
        "paper_metric_equivalence": {
            "vsfc_accuracy": "comparable by definition",
            "vsfc_macro_f1": "unresolved; paper does not specify averaging sufficiently",
            "absa_detection_micro_f1": "unresolved; paper evaluator/averaging is not published sufficiently",
            "absa_pair_micro_f1": "unresolved; paper evaluator/OTHERS and conditional-vs-end-to-end scope are not published sufficiently",
            "uit_absa_lineage": "public candidate mirror; release lineage to the original paper split is not independently proven",
        },
        "per_seed_results": sorted(result_rows, key=lambda row: (row["task_id"], row["seed"])),
        "aggregates": aggregates,
        "comparison_rows": comparison_rows,
        "sources": {
            "bami_bert_targets": "benchmarks/paper_targets.csv",
            "paper_target_source": "https://arxiv.org/html/2607.02259",
        },
    }
    out_path = OUTPUT_DIR / "test_readout.json"
    out_path.write_text(json.dumps(readout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# E1 test evaluation and BamiBERT comparison",
        "",
        f"Generated: `{readout['generated_at_utc']}`. The locked E1 test gate evaluated **25/25 checkpoints** exactly once: five tasks × seeds 42–46. All runs used inference only with BF16/SDPA; no checkpoint, learning rate, or preprocessing choice was changed after test access.",
        "",
        "The table reports GoiBERT mean ± sample standard deviation over five selected seeds, in percentage points. BamiBERT values are the paper targets recorded in `benchmarks/paper_targets.csv`. Arithmetic deltas are shown for orientation; F1 rows are not treated as proof of a win because the paper does not publish enough evaluator detail to establish metric equivalence.",
        "",
        "| Task | Metric | BamiBERT paper | GoiBERT test mean ± SD | Arithmetic Δ | Comparability |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['task']} | {row['metric_label']} | {row['bami_bert_paper_percent']:.2f} | {row['goibert_mean_percent']:.2f} ± {row['goibert_sample_std_percent']:.2f} | {row['arithmetic_delta_percent']:+.2f} | {row['metric_equivalence']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "On the explicitly comparable VSFC Accuracy rows, the supervised GoiBERT baseline is below the paper values: 92.21 ± 0.29 versus 93.86 for sentiment, and 88.49 ± 0.25 versus 89.34 for topic. The internal F1 values and all ABSA values are useful measurements of this pipeline, but they remain provisional for a direct BamiBERT ranking until the paper evaluator and the UIT-ABSA mirror lineage are resolved.",
            "",
            "The selected checkpoints are the best-validation checkpoints from the frozen supervised E1 runs. Test rows were loaded only in the locked evaluator. The backbone inherits the prior corpus exposure caveat recorded in the E0 audit; the current test run itself did not train on test rows.",
            "",
            "## Provenance",
            "",
            f"- Selection manifest: `{MANIFEST_PATH}` (SHA-256 `{readout['manifest']['sha256']}`; filesystem mtime `{readout['manifest']['filesystem_mtime_utc']}`).",
            f"- The manifest's declared creation field is retained as provenance; the filesystem mtime above is the observed seal time used for this audit.",
            f"- Test plan index (dry run, no inference): `{plan_index_path}` (SHA-256 `{readout['execution']['plan_index_sha256']}`) if present.",
            f"- Executed test index: `{index_path}` (SHA-256 `{readout['execution']['index_sha256']}`).",
            f"- Access log: `{access_log_path}` (SHA-256 `{readout['execution']['access_log_sha256']}`).",
            "- Source/evaluator/backbone/tokenizer hashes are retained per seed in `test_readout.json` and each checkpoint's `test_evaluation_metadata.json`.",
            "- Per-checkpoint predictions and metrics remain beside the locked checkpoint under `artifacts/e1/.orchestrator/attempts/`.",
        ]
    )
    doc_path = ROOT / "docs" / "E1_TEST_EVALUATION.md"
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"readout": str(out_path), "doc": str(doc_path), "count": len(result_rows)}, indent=2))


if __name__ == "__main__":
    main()
