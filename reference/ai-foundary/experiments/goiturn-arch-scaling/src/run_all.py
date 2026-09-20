"""Run the architecture-scaling matrix in dependency order and summarize it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("vsfc", "vlsp_restaurant", "visfd")
RUNS = (
    ("baseline", "finetune"),
    ("variant_a", "scratch"),
    ("variant_a", "distill"),
    ("variant_b", "scratch"),
    ("variant_b", "distill"),
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_cell(result: dict[str, Any]) -> str:
    metrics = result["test_metrics"]
    if result["dataset"] == "vsfc":
        return f"macro {metrics['macro_f1']:.4f}"
    return (
        f"det {metrics['detection_micro_f1']:.4f} / "
        f"cls {metrics['classification_micro_f1']:.4f}"
    )


def write_summary(datasets: tuple[str, ...]) -> Path:
    results_dir = EXPERIMENT_ROOT / "results"
    lines = [
        "# GoiTurn Architecture Scaling Results",
        "",
        "Generated from the run JSON files. F1 values are test-set metrics from the best validation epoch.",
        "",
        "| Dataset | Model | Mode | Layers | Hidden | Heads | FFN | Params | F1 | Accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for dataset in datasets:
        for model, mode in RUNS:
            path = results_dir / f"{dataset}__{model}__{mode}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Cannot summarize missing run result: {path}")
            result = _read_json(path)
            arch = result["config"]
            accuracy = (
                f"{result['test_metrics']['accuracy']:.4f}" if dataset == "vsfc" else "—"
            )
            lines.append(
                f"| {dataset} | {model} | {mode} | {arch['num_hidden_layers']} | "
                f"{arch['hidden_size']} | {arch['num_attention_heads']} | "
                f"{arch['intermediate_size']} | "
                f"{result['param_counts']['encoder_and_embeddings']:,} | "
                f"{_metric_cell(result)} | {accuracy} |"
            )
    latency_path = results_dir / "latency.json"
    if not latency_path.is_file():
        raise FileNotFoundError(f"Cannot summarize missing latency result: {latency_path}")
    latency = _read_json(latency_path)
    lines.extend(
        [
            "",
            "## CPU latency",
            "",
            f"Batch 1, fp32, {latency['warmup_iterations']} warmup and "
            f"{latency['timed_iterations']} timed iterations.",
            "",
            "| Model | Seq | Threads | p50 ms | p95 ms | fp32 state dict MiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for measurement in latency["measurements"]:
        model = measurement["architecture"]
        size_mib = latency["state_dict_sizes"][model]["fp32_state_dict_bytes"] / 1_048_576
        lines.append(
            f"| {model} | {measurement['sequence_length']} | {measurement['threads']} | "
            f"{measurement['p50_ms']:.3f} | {measurement['p95_ms']:.3f} | {size_mib:.2f} |"
        )
    output_path = results_dir / "summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    return output_path


def run(args: argparse.Namespace) -> None:
    datasets = (args.only,) if args.only else DATASETS
    for dataset in datasets:
        for model, mode in RUNS:
            command = [
                sys.executable,
                str(EXPERIMENT_ROOT / "src" / "train.py"),
                "--dataset",
                dataset,
                "--model",
                model,
                "--mode",
                mode,
                "--seed",
                "42",
                "--repo-root",
                str(args.repo_root.resolve()),
            ]
            if args.smoke:
                command.append("--smoke")
            print(f"Running {dataset}: {model} {mode}", flush=True)
            subprocess.run(command, cwd=EXPERIMENT_ROOT, check=True)
    latency_command = [sys.executable, str(EXPERIMENT_ROOT / "src" / "latency.py")]
    if args.smoke:
        latency_command.append("--smoke")
    subprocess.run(latency_command, cwd=EXPERIMENT_ROOT, check=True)
    write_summary(datasets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", choices=DATASETS)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
