#!/usr/bin/env python3
"""Package the *pre-fine-tune* P_vi encoder for downstream fine-tuning.

Keep the exact FP32 export, and derive a separate BF16 artifact. This does not
package any of the five benchmark-specific fine-tuned checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "artifacts" / "bidirlm_vi_mntp" / "mntp" / "hf_bidirlm"
DEFAULT_OUT = ROOT / "artifacts" / "p_vi_handoff"
SUPPORT_FILES = ("config.json", "configuration_bidirlm.py", "modeling_bidirlm.py",
                 "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_support(destination: Path, *, dtype: str):
    destination.mkdir(parents=True, exist_ok=False)
    for name in SUPPORT_FILES:
        shutil.copy2(SOURCE / name, destination / name)
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dtype"] = dtype
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_bf16(source: Path, destination: Path):
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            tensors[name] = tensor.to(torch.bfloat16).contiguous() if tensor.is_floating_point() else tensor.contiguous()
    save_file(tensors, str(destination), metadata={"format": "pt", "variant": "p_vi_base_bf16"})


def file_manifest(directory: Path) -> dict:
    return {item.name: {"bytes": item.stat().st_size, "sha256": sha256(item)}
            for item in sorted(directory.iterdir()) if item.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing handoff directory: {out}")
    if not (SOURCE / "model.safetensors").is_file():
        raise FileNotFoundError(SOURCE / "model.safetensors")

    out.mkdir(parents=True)
    fp32 = out / "fp32"
    bf16 = out / "bf16"
    copy_support(fp32, dtype="float32")
    copy_support(bf16, dtype="bfloat16")
    shutil.copy2(SOURCE / "model.safetensors", fp32 / "model.safetensors")
    make_bf16(SOURCE / "model.safetensors", bf16 / "model.safetensors")
    manifest = {
        "artifact": "P_vi base encoder before downstream fine-tuning",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_weights_sha256": sha256(SOURCE / "model.safetensors"),
        "upstream": "BidirLM/BidirLM-0.6B-Base @ 29794508bbb4e378b4789eebff42ae4c2d47d524",
        "vi_mntp_tokens": 249987072,
        "parameters": 596049920,
        "variants": {"fp32": file_manifest(fp32), "bf16": file_manifest(bf16)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "fp32_bytes": manifest["variants"]["fp32"]["model.safetensors"]["bytes"],
                      "bf16_bytes": manifest["variants"]["bf16"]["model.safetensors"]["bytes"]}, indent=2))


if __name__ == "__main__":
    main()
