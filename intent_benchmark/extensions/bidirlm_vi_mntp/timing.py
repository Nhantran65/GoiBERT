#!/usr/bin/env python3
"""Model-forward-only timing on the frozen 77 test inputs."""

import json
import statistics
import sys
import time
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run


def main():
    cfg = run.config()
    device = run.old.resolve_device(1)
    tok, model, loading = run.load_model(run.MODEL_NAME, device, cfg)
    choice = json.loads((run.OUT / "selection.json").read_text())
    state = torch.load(run.OUT / f"lr_{choice['winner']}" / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    rows = run.old.build_dataset_rows(run.old.TEST_PATH)
    data = run.old.IntentDataset(rows, tok, cfg["labels"], int(cfg["max_length"]))
    examples = [run.old.collate_examples([item], tok.pad_token_id) for item in data]
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    with torch.inference_mode():
        for i in range(10 + 2 * len(examples)):
            batch = examples[i % len(examples)]
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with run.old.autocast_context(device, "bf16"):
                model(ids, mask)
            torch.cuda.synchronize(device)
            if i >= 10:
                elapsed.append((time.perf_counter() - start) * 1000)
    elapsed.sort()
    output = {
        "scope": "model-forward-only, batch 1, same 77 test inputs, 10 warmup + 154 measured, BF16 autocast, CUDA synchronize",
        "device": torch.cuda.get_device_name(device),
        "mean_ms": statistics.mean(elapsed),
        "median_ms": statistics.median(elapsed),
        "p95_ms": elapsed[int(0.95 * (len(elapsed) - 1))],
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "parameters": loading["total_parameters"],
    }
    (run.OUT / "timing.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
