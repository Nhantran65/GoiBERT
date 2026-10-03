#!/usr/bin/env python3
"""Small model-only GPU latency probe for P_vi base handoff variants."""

from __future__ import annotations

import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "artifacts" / "p_vi_handoff"
TEXTS = [
    "Dịch vụ nhanh, nhân viên nhiệt tình và đồ ăn ngon.",
    "Giao hàng trễ ba ngày nhưng sản phẩm vẫn hoạt động tốt.",
    "Tôi chưa có ý kiến về chất lượng phòng và giá tiền.",
    "Bài giảng dễ hiểu, tuy nhiên cơ sở vật chất cần cải thiện.",
    "Một câu ngắn.",
    "Trải nghiệm này khá bình thường, không có điểm nổi bật.",
]


def run_variant(name: str, tokenizer) -> dict:
    path = HANDOFF / name
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    kwargs = {"trust_remote_code": True, "local_files_only": True, "attn_implementation": "sdpa"}
    if name == "bf16":
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["device_map"] = {"": 0}
    model = AutoModel.from_pretrained(str(path), **kwargs).eval()
    if name == "bf16":
        model.to("cuda:0")
    load_seconds = time.perf_counter() - started
    loaded_mib = torch.cuda.memory_allocated() / 2**20
    rows = {}
    for size in (1, 8):
        batch_texts = (TEXTS * 2)[:size]
        batch = tokenizer(batch_texts, padding=True, truncation=True,
                          max_length=256, return_tensors="pt").to("cuda:0")
        samples = []
        for repeat in range(13):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                model(**batch, return_dict=True)
            torch.cuda.synchronize()
            if repeat >= 3:
                samples.append(time.perf_counter() - t0)
        rows[str(size)] = {"shape": list(batch["input_ids"].shape),
                           "median_ms": 1000 * statistics.median(samples),
                           "min_ms": 1000 * min(samples)}
    result = {"load_seconds": load_seconds, "loaded_gpu_mib": loaded_mib,
              "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20, "batches": rows}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(HANDOFF / "fp32"),
                                               local_files_only=True, use_fast=True)
    report = {"protocol": "p-vi-base-handoff-gpu-latency-probe-v1", "device": torch.cuda.get_device_name(0),
              "scope": "model forward only; warm; shared GPU; no tokenization or API",
              "variants": {name: run_variant(name, tokenizer) for name in ("bf16", "gpu_int8")}}
    (HANDOFF / "gpu_latency_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
