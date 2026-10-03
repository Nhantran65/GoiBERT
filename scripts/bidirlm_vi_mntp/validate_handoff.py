#!/usr/bin/env python3
"""Reload and compare P_vi base handoff variants without touching test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_HANDOFF = ROOT / "artifacts" / "p_vi_handoff"

SENTENCES = [
    "Dịch vụ nhanh, nhân viên nhiệt tình và đồ ăn ngon.",
    "Giao hàng trễ ba ngày nhưng sản phẩm vẫn hoạt động tốt.",
    "Tôi chưa có ý kiến về chất lượng phòng và giá tiền.",
    "Bài giảng dễ hiểu, tuy nhiên cơ sở vật chất cần cải thiện.",
    "Một câu ngắn.",
    "Trải nghiệm này khá bình thường, không có điểm nổi bật.",
]


def load(path: Path, *, dtype: torch.dtype, device: str):
    model, info = AutoModel.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa", dtype=dtype, output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"{path}: incomplete load: {info}")
    model.eval().to(device)
    return model, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    handoff = args.handoff.resolve()
    fp32_dir, bf16_dir = handoff / "fp32", handoff / "bf16"
    tokenizer = AutoTokenizer.from_pretrained(str(fp32_dir), local_files_only=True, use_fast=True)
    tokenizer_bf16 = AutoTokenizer.from_pretrained(str(bf16_dir), local_files_only=True, use_fast=True)
    encoded = tokenizer(SENTENCES, return_tensors="pt", padding=True, truncation=True, max_length=256)
    if encoded["input_ids"].tolist() != tokenizer_bf16(SENTENCES, return_tensors="pt", padding=True,
                                                           truncation=True, max_length=256)["input_ids"].tolist():
        raise RuntimeError("Tokenizer mismatch between variants")
    cpu_fp32, cpu_info = load(fp32_dir, dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        cpu_hidden = cpu_fp32(**encoded, return_dict=True).last_hidden_state
    cpu_ok = bool(torch.isfinite(cpu_hidden).all())
    del cpu_fp32

    gpu_fp32, fp32_info = load(fp32_dir, dtype=torch.float32, device=args.device)
    gpu_bf16, bf16_info = load(bf16_dir, dtype=torch.bfloat16, device=args.device)
    ids = encoded["input_ids"].to(args.device)
    mask = encoded["attention_mask"].to(args.device)
    with torch.inference_mode():
        reference = gpu_fp32(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state.float()
        candidate = gpu_bf16(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state.float()
        perturbed = ids.clone()
        last = int(mask[0].sum()) - 1
        perturbed[0, last] = (int(perturbed[0, last]) + 1) % tokenizer.vocab_size
        changed = gpu_bf16(input_ids=perturbed, attention_mask=mask, return_dict=True).last_hidden_state.float()
    valid = mask.bool().unsqueeze(-1).expand_as(reference)
    max_abs = float((reference[valid] - candidate[valid]).abs().max().item())
    ref_pool = (reference * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    bf16_pool = (candidate * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    cosine = torch.nn.functional.cosine_similarity(ref_pool, bf16_pool, dim=-1)
    direction_delta = float((candidate[0, 0] - changed[0, 0]).abs().max().item())
    params = sum(p.numel() for p in gpu_bf16.parameters())
    result = {
        "protocol": "p-vi-base-handoff-validation-v1", "samples": len(SENTENCES),
        "shape": list(ids.shape), "parameters": params,
        "fp32_weight_dtype": str(next(gpu_fp32.parameters()).dtype),
        "bf16_weight_dtype": str(next(gpu_bf16.parameters()).dtype),
        "tokenizer_identical": True, "cpu_fp32_finite": cpu_ok,
        "bf16_vs_fp32_hidden_max_abs": max_abs,
        "bf16_vs_fp32_pooled_cosine_min": float(cosine.min().item()),
        "bf16_vs_fp32_pooled_cosine_mean": float(cosine.mean().item()),
        "bf16_bidirectional_delta": direction_delta,
        "load_unexpected_keys": {"cpu_fp32": cpu_info.get("unexpected_keys", []),
                                 "gpu_fp32": fp32_info.get("unexpected_keys", []),
                                 "gpu_bf16": bf16_info.get("unexpected_keys", [])},
    }
    result["passed"] = bool(cpu_ok and params == 596049920 and direction_delta > 0
                            and result["bf16_vs_fp32_pooled_cosine_min"] >= 0.995)
    (handoff / "validation_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
