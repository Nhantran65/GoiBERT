#!/usr/bin/env python3
"""Collect a reproducible E0 environment and model inventory.

This command is intentionally read-only with respect to model weights and
running processes.  It records an unavailable NVIDIA driver as evidence
instead of attempting to reset, reserve, or terminate any GPU process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_NAMES = (
    "torch",
    "transformers",
    "safetensors",
    "tokenizers",
    "accelerate",
    "numpy",
    "scikit-learn",
    "pandas",
)


def run_command(command: list[str]) -> dict[str, Any]:
    """Run a bounded inventory command without raising on unavailable hardware."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "available": False,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "available": completed.returncode == 0,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count_from_safetensors(path: Path) -> int | None:
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    count = 0
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                count += math.prod(shape)
    except Exception:
        return None
    return int(count)


def expected_roberta_encoder_parameters(config: dict[str, Any]) -> int | None:
    """Count the encoder tensors described by a pooler-free RoBERTa config."""
    required = (
        "vocab_size",
        "hidden_size",
        "max_position_embeddings",
        "type_vocab_size",
        "num_hidden_layers",
        "intermediate_size",
    )
    if any(key not in config for key in required):
        return None
    hidden = int(config["hidden_size"])
    embeddings = (
        int(config["vocab_size"]) * hidden
        + int(config["max_position_embeddings"]) * hidden
        + int(config["type_vocab_size"]) * hidden
        + 2 * hidden
    )
    per_layer = 4 * hidden * hidden + 2 * hidden * int(config["intermediate_size"]) + 9 * hidden + int(
        config["intermediate_size"]
    )
    return embeddings + int(config["num_hidden_layers"]) * per_layer


def model_inventory(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    files: dict[str, Any] = {}
    for path in sorted(model_dir.iterdir()) if model_dir.is_dir() else []:
        if path.is_file():
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "path": str(model_dir),
        "config": {
            key: config.get(key)
            for key in (
                "architectures",
                "model_type",
                "transformers_version",
                "vocab_size",
                "max_position_embeddings",
                "num_hidden_layers",
                "hidden_size",
                "num_attention_heads",
                "intermediate_size",
                "torch_dtype",
            )
            if key in config
        },
        "parameter_count_from_safetensors": parameter_count_from_safetensors(weights_path)
        if weights_path.is_file()
        else None,
        "expected_pooler_free_encoder_parameter_count": expected_roberta_encoder_parameters(config),
        "files": files,
    }


def package_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import torch

        versions["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "hip_version": torch.version.hip,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:
        versions["torch_runtime"] = {"error": repr(exc)}
    return versions


def parse_gpu_query(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not result["available"]:
        return []
    rows: list[dict[str, Any]] = []
    fields = (
        "index",
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "memory.free",
        "temperature.gpu",
        "utilization.gpu",
    )
    for line in result["stdout"].splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(fields):
            continue
        rows.append(dict(zip(fields, values)))
    return rows


def collect_report(project_root: Path) -> dict[str, Any]:
    gpu_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_query = run_command(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    )
    process_query_with_uuid = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,gpu_name,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = parse_gpu_query(gpu_query)
    package_info = package_versions()
    models = {
        name: model_inventory(project_root / "models" / name)
        for name in ("goiturn-s-gd", "goibert-b-dapt")
    }
    disk = shutil.disk_usage(project_root)
    expected_l40s = len(gpus) == 4 and all("L40S" in gpu.get("name", "") for gpu in gpus)
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "packages": package_info,
        "gpu": {
            "nvidia_smi": gpu_query,
            "compute_processes": process_query,
            "compute_processes_with_gpu_uuid": process_query_with_uuid,
            "devices": gpus,
            "device_count": len(gpus),
            "four_l40s_verified": expected_l40s,
            "allocation_status": "verified_4x_l40s"
            if expected_l40s
            else "not_verified",
        },
        "resources": {
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "cpu_count": os.cpu_count(),
        },
        "models": models,
        "checks": {
            "gpu_available": bool(gpus),
            "gpu_four_l40s": expected_l40s,
            "student_present": (project_root / "models" / "goiturn-s-gd" / "model.safetensors").is_file(),
            "teacher_present": (project_root / "models" / "goibert-b-dapt" / "model.safetensors").is_file(),
            "source_weights_modified": False,
        },
        "notes": [
            "GPU/process inventory is read-only; no process was terminated or reserved.",
            "A failed nvidia-smi means GPU availability is unverified, not that four L40S exist.",
            "Model hashes are evidence for this checkout and do not reproduce historical training.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output or (root / "artifacts" / "e0" / "environment.json")
    report = collect_report(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gpu": report["gpu"], "python": report["python"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
