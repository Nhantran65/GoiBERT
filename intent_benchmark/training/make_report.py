#!/usr/bin/env python3
"""Render the final benchmark report from locked trial/results artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "intent_benchmark"
RUNS = BENCHMARK / "runs"
RESULTS = BENCHMARK / "results"


def f(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    test = json.loads((RESULTS / "test_results.json").read_text(encoding="utf-8"))
    timing = json.loads((RESULTS / "timing.json").read_text(encoding="utf-8"))
    selected_by_model = {row["model_name"]: row for row in selection["selected"]}
    test_by_model = {row["model_name"]: row for row in test["models"]}
    rows = []
    for run_dir in sorted(RUNS.iterdir()):
        path = run_dir / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            rows.append(summary)
    lines = [
        "# GoiBERT intent benchmark report",
        "",
        "The six locked trials used seed 42, the frozen 353/77/77 grouped split, batch 16, max 15 epochs, patience 3, unweighted eight-class cross entropy, AdamW with uniform learning rate, BF16 autocast/SDPA where supported, and no test-based selection. All 507 reviewed rows were used normally, including the 82 uncertainty-marked rows. This is a single-seed pilot on 77 test examples, not a universal SOTA claim.",
        "",
        "## Dev trial results",
        "",
        "| Model | LR | Best epoch | Best dev macro-F1 | Dev accuracy | Peak allocated GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in sorted(rows, key=lambda x: (x["model_name"], x["learning_rate"])):
        lines.append(f"| {summary['model_name']} | {summary['learning_rate']:.0e} | {summary['best_epoch']} | {f(summary['best_dev_macro_f1'])} | {f(summary['best_dev_metrics']['accuracy'])} | {summary['peak_memory_allocated_bytes'] / 1024**3:.3f} |")
    lines += [
        "",
        "The locked winner per backbone maximized dev macro-F1. Within a trial, strict improvement retained the earlier epoch on ties; across equal learning-rate scores the 2e-5 trial wins.",
        "",
        "| Selected model | LR | Epoch | Dev macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for winner in selection["selected"]:
        lines.append(f"| {winner['model_name']} | {winner['learning_rate']:.0e} | {winner['best_epoch']} | {f(winner['best_dev_macro_f1'])} |")
    lines += ["", "## Frozen test results", "", "Exactly the three dev-selected winners were evaluated on the 77-row test split. Saved prediction files were independently re-read and checked by `saved_prediction_audit.json`.", "", "| Model | Accuracy | Macro-F1 | Prediction file |", "|---|---:|---:|---|"]
    for result in test["models"]:
        lines.append(f"| {result['model_name']} | {f(result['test_metrics']['accuracy'])} | {f(result['test_metrics']['macro_f1'])} | `{Path(result['prediction_path']).name}` |")
    lines += ["", "### Test per-intent metrics", "", "| Model | Intent | Precision | Recall | F1 | Support |", "|---|---|---:|---:|---:|---:|"]
    for result in test["models"]:
        for label, metrics in result["test_metrics"]["per_intent"].items():
            lines.append(f"| {result['model_name']} | {label} | {f(metrics['precision'])} | {f(metrics['recall'])} | {f(metrics['f1'])} | {metrics['support']} |")
    lines += ["", "## Timing and resource results", "", "Timing is model-forward only: tokenizer, padding, and host-to-device transfer are excluded. Each selected model ran 10 warmup forwards and two passes over all 77 test inputs at batch size 1 on the same NVIDIA L40S/GPU0.", "", "| Model | Params | Mean ms | Median ms | P95 ms | Peak allocated GiB |", "|---|---:|---:|---:|---:|---:|"]
    for result in timing["models"]:
        lines.append(f"| {result['model_name']} | {result['trainable_parameters']:,} | {f(result['timing_ms']['mean'])} | {f(result['timing_ms']['median'])} | {f(result['timing_ms']['p95'])} | {result['peak_memory_allocated_bytes'] / 1024**3:.3f} |")
    lines += [
        "",
        "## Artifacts and caveats",
        "",
        "- `selection.json` records the dev-only lock and checkpoint hashes; no test score influenced selection.",
        "- `test_results.json`, `test_metrics_*.json`, and `test_predictions_*.jsonl` contain the three test evaluations.",
        "- `timing.json` and `benchmark.csv` contain the shared-GPU latency/resource measurements.",
        "- `saved_prediction_audit.json` recomputes metrics from saved prediction files.",
        "- The PHONE and ADDRESS classes contain anonymized placeholders and small supports; the report should not be read as real-world generalization evidence.",
        "- The input snapshot contains customer-only history and no assistant replies; buffered-burst source labels are preserved as provided.",
        "",
    ]
    timing_by_model = {row["model_name"]: row for row in timing["models"]}
    with (RESULTS / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model_name", "learning_rate", "best_epoch", "dev_macro_f1",
            "test_macro_f1", "test_accuracy", "trainable_parameters",
            "peak_memory_allocated_bytes", "peak_memory_reserved_bytes",
            "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms",
            "sequence_min", "sequence_median", "sequence_max", "gpu",
            "gpu_name", "passes", "warmup_forwards",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in sorted(timing_by_model):
            timing_row = timing_by_model[model_name]
            selected_row = selected_by_model[model_name]
            test_row = test_by_model[model_name]
            writer.writerow({
                "model_name": model_name,
                "learning_rate": selected_row["learning_rate"],
                "best_epoch": selected_row["best_epoch"],
                "dev_macro_f1": selected_row["best_dev_macro_f1"],
                "test_macro_f1": test_row["test_metrics"]["macro_f1"],
                "test_accuracy": test_row["test_metrics"]["accuracy"],
                "trainable_parameters": timing_row["trainable_parameters"],
                "peak_memory_allocated_bytes": timing_row["peak_memory_allocated_bytes"],
                "peak_memory_reserved_bytes": timing_row["peak_memory_reserved_bytes"],
                "mean_ms": timing_row["timing_ms"]["mean"],
                "median_ms": timing_row["timing_ms"]["median"],
                "p95_ms": timing_row["timing_ms"]["p95"],
                "min_ms": timing_row["timing_ms"]["min"],
                "max_ms": timing_row["timing_ms"]["max"],
                "sequence_min": timing_row["sequence_lengths"]["min"],
                "sequence_median": timing_row["sequence_lengths"]["median"],
                "sequence_max": timing_row["sequence_lengths"]["max"],
                "gpu": timing_row["gpu"],
                "gpu_name": timing_row["gpu_name"],
                "passes": timing_row["passes"],
                "warmup_forwards": timing_row["warmup_forwards"],
            })
    (RESULTS / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(RESULTS / "REPORT.md"))


if __name__ == "__main__":
    main()
