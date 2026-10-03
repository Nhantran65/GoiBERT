#!/usr/bin/env python3
"""Create and verify an INT8 GPU inference variant of the P_vi base encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_HANDOFF = ROOT / "artifacts" / "p_vi_handoff"
COPY_FILES = ("configuration_bidirlm.py", "modeling_bidirlm.py", "tokenizer.json",
              "tokenizer_config.json", "vocab.json", "merges.txt")
SENTENCES = ["Dịch vụ nhanh, nhân viên nhiệt tình và đồ ăn ngon.",
             "Giao hàng trễ ba ngày nhưng sản phẩm vẫn hoạt động tốt.",
             "Tôi chưa có ý kiến về chất lượng phòng và giá tiền.",
             "Bài giảng dễ hiểu, tuy nhiên cơ sở vật chất cần cải thiện.",
             "Một câu ngắn.", "Trải nghiệm này khá bình thường, không có điểm nổi bật."]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pooled(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (hidden.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


def finalize_saved(handoff: Path, temporary: Path, destination: Path, source: Path,
                   original_hidden: torch.Tensor | None = None):
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    batch = tokenizer(SENTENCES, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda:0")
    first = AutoModel.from_pretrained(
        str(temporary), trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa", device_map={"": 0},
    ).eval()
    linear8 = sum(type(module).__name__ == "Linear8bitLt" for module in first.modules())
    with torch.inference_mode():
        first_hidden = first(**batch, return_dict=True).last_hidden_state
    del first
    torch.cuda.empty_cache()
    second = AutoModel.from_pretrained(
        str(temporary), trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa", device_map={"": 0},
    ).eval()
    with torch.inference_mode():
        second_hidden = second(**batch, return_dict=True).last_hidden_state
    del second
    torch.cuda.empty_cache()
    reference = AutoModel.from_pretrained(
        str(handoff / "bf16"), trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa", dtype=torch.bfloat16,
    ).eval().to("cuda:0")
    with torch.inference_mode():
        ref_hidden = reference(**batch, return_dict=True).last_hidden_state
    cosine = torch.nn.functional.cosine_similarity(
        pooled(first_hidden, batch["attention_mask"]), pooled(ref_hidden, batch["attention_mask"]), dim=-1)
    reload_max_abs = float((first_hidden.float() - second_hidden.float()).abs().max().item())
    original_max_abs = None if original_hidden is None else float((original_hidden.float() - first_hidden.float()).abs().max().item())
    result = {
        "protocol": "p-vi-base-gpu-int8-v2", "samples": len(SENTENCES),
        "linear8bit_modules": linear8, "dtype_non_quantized": "float16",
        "pooled_cosine_vs_bf16_min": float(cosine.min().item()),
        "pooled_cosine_vs_bf16_mean": float(cosine.mean().item()),
        "independent_reload_hidden_max_abs": reload_max_abs,
        "before_save_vs_reload_hidden_max_abs_diagnostic": original_max_abs,
        "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(temporary.iterdir()) if p.is_file()},
    }
    result["passed"] = bool(torch.isfinite(first_hidden).all() and linear8 >= 100
                            and result["pooled_cosine_vs_bf16_min"] >= 0.995
                            and reload_max_abs <= 1e-3)
    (handoff / "gpu_int8_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
    temporary.rename(destination)
    manifest_path = handoff / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["variants"]["gpu_int8"] = result["files"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--finalize-existing", action="store_true", help="verify an existing gpu_int8.tmp without requantizing")
    args = parser.parse_args()
    handoff = args.handoff.resolve()
    source = handoff / "fp32"
    destination = handoff / "gpu_int8"
    temporary = handoff / "gpu_int8.tmp"
    if destination.exists() or (temporary.exists() and not args.finalize_existing):
        raise FileExistsError(f"Refusing to overwrite {destination} or {temporary}")
    if args.finalize_existing:
        if not temporary.is_dir():
            raise FileNotFoundError(temporary)
        finalize_saved(handoff, temporary, destination, source)
        return
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, use_fast=True)
    batch = tokenizer(SENTENCES, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    quantization = BitsAndBytesConfig(load_in_8bit=True)
    quantized = AutoModel.from_pretrained(
        str(source), trust_remote_code=True, local_files_only=True, attn_implementation="sdpa",
        dtype=torch.float16, quantization_config=quantization, device_map={"": 0},
    ).eval()
    linear8 = sum(type(module).__name__ == "Linear8bitLt" for module in quantized.modules())
    if linear8 < 100:
        raise RuntimeError(f"Only {linear8} Linear8bitLt modules; expected most transformer projections")
    with torch.inference_mode():
        hidden = quantized(**batch, return_dict=True).last_hidden_state
    quantized.save_pretrained(str(temporary), safe_serialization=True, max_shard_size="5GB")
    for name in COPY_FILES:
        shutil.copy2(source / name, temporary / name)
    del quantized
    torch.cuda.empty_cache()
    finalize_saved(handoff, temporary, destination, source, original_hidden=hidden)


if __name__ == "__main__":
    main()
