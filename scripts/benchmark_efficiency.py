#!/usr/bin/env python3
"""Inference latency, throughput and peak memory of an encoder with a 3-class head.

Protocol of the paper (Table 3): GPU in bf16 with SDPA attention, CPU in fp32
with eight threads; 20 warm-up and 200 timed GPU iterations, 5 warm-up and 50
timed CPU iterations; the median is reported.  Tokenization and the task head
are not included in the speed-up claims.

    python scripts/benchmark_efficiency.py TwoheartsTechnologies/GoiBERT-M Qualcomm-AI-Research/BamiBERT
"""

import argparse
import json
import statistics
import time

import torch

from goibert.modeling import build_model


def gpu_timing(model, batch: int, length: int) -> dict:
    device = torch.device("cuda:0")
    model = model.to(device)
    ids = torch.randint(4, int(model.encoder.config.vocab_size), (batch, length), device=device)
    mask = torch.ones_like(ids)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    def once():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids, mask)

    for _ in range(20):
        once()
    torch.cuda.synchronize()
    samples = []
    for _ in range(200):
        start = time.perf_counter()
        once()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    median = statistics.median(samples)
    model.to("cpu")
    return {"median_ms": median, "sentences_per_second": batch * 1000 / median, "peak_vram_bytes": torch.cuda.max_memory_allocated(device)}


def cpu_timing(model, length: int) -> dict:
    model = model.cpu().eval()
    ids = torch.randint(4, int(model.encoder.config.vocab_size), (1, length))
    mask = torch.ones_like(ids)

    def once():
        with torch.inference_mode():
            model(ids, mask)

    for _ in range(5):
        once()
    samples = []
    for _ in range(50):
        start = time.perf_counter()
        once()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="Hub IDs or local directories")
    parser.add_argument("--out", help="write results as JSON")
    args = parser.parse_args()
    torch.set_num_threads(8)
    results = {}
    for name in args.models:
        kind = "bamibert" if "bamibert" in name.lower() else "student"
        model = build_model(name, kind=kind, task_type="sequence", num_labels=3).eval()
        entry = {"parameters": sum(p.numel() for p in model.parameters())}
        if torch.cuda.is_available():
            entry["gpu"] = {f"b{b}_l{l}": gpu_timing(model, b, l) for b in (1, 32) for l in (128, 256)}
        entry["cpu"] = {"b1_l128": cpu_timing(model, 128)}
        results[name] = entry
        print(name, json.dumps(entry), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
