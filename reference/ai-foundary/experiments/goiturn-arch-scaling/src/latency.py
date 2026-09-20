"""Benchmark eager CPU fp32 latency for each architecture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import RobertaModel

from modeling import RobertaClassifier, load_architectures, make_config, parameter_counts


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_LENGTHS = (32, 128)
THREAD_COUNTS = (1, 4)


def _state_dict_size(model: RobertaClassifier, architecture: str, results_dir: Path) -> int:
    path = results_dir / f".latency_{architecture}.tmp.pt"
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size
    finally:
        path.unlink(missing_ok=True)


def run(smoke: bool = False) -> Path:
    torch.manual_seed(42)
    results_dir = EXPERIMENT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    architectures = load_architectures(EXPERIMENT_ROOT)
    warmup_iterations = 2 if smoke else 30
    timed_iterations = 5 if smoke else 300
    measurements: list[dict[str, Any]] = []
    state_sizes: dict[str, dict[str, int]] = {}
    for architecture, arch in architectures.items():
        config = make_config(arch)
        encoder = RobertaModel(config, add_pooling_layer=False)
        model = RobertaClassifier(encoder, output_size=3, n_aspects=0).cpu().float().eval()
        encoder_count, head_count = parameter_counts(model)
        state_sizes[architecture] = {
            "fp32_state_dict_bytes": _state_dict_size(model, architecture, results_dir),
            "encoder_and_embeddings_params": encoder_count,
            "head_params": head_count,
        }
        for sequence_length in SEQUENCE_LENGTHS:
            input_ids = torch.randint(3, config.vocab_size, (1, sequence_length), dtype=torch.long)
            input_ids[:, 0] = config.bos_token_id
            attention_mask = torch.ones_like(input_ids)
            for threads in THREAD_COUNTS:
                torch.set_num_threads(threads)
                with torch.inference_mode():
                    for _ in range(warmup_iterations):
                        model(input_ids, attention_mask)
                    samples_ms: list[float] = []
                    for _ in range(timed_iterations):
                        started = time.perf_counter_ns()
                        model(input_ids, attention_mask)
                        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                measurement = {
                    "architecture": architecture,
                    "sequence_length": sequence_length,
                    "threads": threads,
                    "batch_size": 1,
                    "p50_ms": float(np.percentile(samples_ms, 50)),
                    "p95_ms": float(np.percentile(samples_ms, 95)),
                }
                measurements.append(measurement)
                print(
                    f"{architecture} seq={sequence_length} threads={threads} "
                    f"p50={measurement['p50_ms']:.3f}ms p95={measurement['p95_ms']:.3f}ms",
                    flush=True,
                )
        del model
    output = {
        "device": "CPU",
        "dtype": "float32",
        "batch_size": 1,
        "smoke": smoke,
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
        "state_dict_sizes": state_sizes,
        "measurements": measurements,
        "torch_version": torch.__version__,
    }
    output_path = results_dir / "latency.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.smoke)
