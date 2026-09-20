#!/usr/bin/env python3
"""Measure batch-1 model-forward latency for the three locked winners."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from train_intent import (
    BENCHMARK,
    TEST_PATH,
    IntentDataset,
    build_dataset_rows,
    load_locked_config,
    load_tokenizer_and_model,
    resolve_device,
    seed_everything,
)


ROOT = Path(__file__).resolve().parents[2]
SELECTION = BENCHMARK / "results/selection.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=BENCHMARK / "results")
    args = parser.parse_args()
    if args.passes < 2 or args.warmup < 10:
        raise ValueError("timing requires at least two passes and ten warmup forwards")
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("selection_status") != "locked":
        raise RuntimeError("selection.json is not locked")
    selected = selection["selected"]
    if len(selected) != 3:
        raise RuntimeError("timing must cover exactly three selected winners")
    config = load_locked_config()
    seed_everything(int(config["seed"]))
    device = resolve_device(args.gpu)
    if device.type != "cuda":
        raise RuntimeError("timing protocol requires CUDA")
    test_rows = build_dataset_rows(TEST_PATH)
    if len(test_rows) != 77:
        raise AssertionError("frozen test split must contain 77 rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for winner in selected:
        model_name = winner["model_name"]
        run_dir = ROOT / winner["run_dir"]
        checkpoint = run_dir / "best.pt"
        tokenizer, model, loading = load_tokenizer_and_model(model_name, device, config)
        dataset = IntentDataset(test_rows, tokenizer, config["labels"], int(config["max_length"]))
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint_data["model"], strict=True)
        model.eval()
        examples = []
        for example in dataset.rows:
            # Tokenization, padding, and host-to-device transfer are outside timing.
            ids = torch.tensor(example.input_ids, dtype=torch.long, device=device).unsqueeze(0)
            mask = torch.ones_like(ids)
            examples.append((ids, mask, len(example.input_ids), example.sample_id))
        with torch.inference_mode():
            for index in range(args.warmup):
                ids, mask, _, _ = examples[index % len(examples)]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(ids, mask)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            timings_ms = []
            lengths = []
            for _ in range(args.passes):
                for ids, mask, length, _ in examples:
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model(ids, mask)
                    torch.cuda.synchronize(device)
                    timings_ms.append((time.perf_counter() - started) * 1000.0)
                    lengths.append(length)
        record = {
            "model_name": model_name,
            "run_dir": winner["run_dir"],
            "learning_rate": winner["learning_rate"],
            "best_epoch": winner["best_epoch"],
            "gpu": args.gpu,
            "gpu_name": torch.cuda.get_device_name(device),
            "passes": args.passes,
            "warmup_forwards": args.warmup,
            "forward_count": len(timings_ms),
            "batch_size": 1,
            "timing_scope": "model-forward-only; tokenizer, padding, and host-to-device transfer excluded",
            "precision": "bf16_autocast_fp32_master_weights",
            "sequence_lengths": {
                "min": min(lengths),
                "median": statistics.median(lengths),
                "max": max(lengths),
            },
            "timing_ms": {
                "mean": statistics.fmean(timings_ms),
                "median": statistics.median(timings_ms),
                "p95": percentile(timings_ms, 0.95),
                "min": min(timings_ms),
                "max": max(timings_ms),
            },
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "trainable_parameters": loading["trainable_parameters"],
            "total_parameters": loading["total_parameters"],
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "test_sha256": sha256(TEST_PATH),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "transformers_version": __import__("transformers").__version__,
        }
        records.append(record)
        del model, tokenizer
        torch.cuda.empty_cache()
    payload = {
        "selection_path": str(SELECTION.relative_to(ROOT)),
        "selection_sha256": sha256(SELECTION),
        "device": str(device),
        "models": records,
    }
    (output_dir / "timing.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "learning_rate", "best_epoch", "trainable_parameters", "peak_memory_allocated_bytes", "peak_memory_reserved_bytes", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "sequence_min", "sequence_median", "sequence_max", "gpu", "gpu_name", "passes", "warmup_forwards"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "model_name": record["model_name"],
                "learning_rate": record["learning_rate"],
                "best_epoch": record["best_epoch"],
                "trainable_parameters": record["trainable_parameters"],
                "peak_memory_allocated_bytes": record["peak_memory_allocated_bytes"],
                "peak_memory_reserved_bytes": record["peak_memory_reserved_bytes"],
                "mean_ms": record["timing_ms"]["mean"],
                "median_ms": record["timing_ms"]["median"],
                "p95_ms": record["timing_ms"]["p95"],
                "min_ms": record["timing_ms"]["min"],
                "max_ms": record["timing_ms"]["max"],
                "sequence_min": record["sequence_lengths"]["min"],
                "sequence_median": record["sequence_lengths"]["median"],
                "sequence_max": record["sequence_lengths"]["max"],
                "gpu": record["gpu"],
                "gpu_name": record["gpu_name"],
                "passes": record["passes"],
                "warmup_forwards": record["warmup_forwards"],
            })
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
