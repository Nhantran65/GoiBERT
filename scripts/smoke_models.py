#!/usr/bin/env python3
"""E0 checkpoint smoke, optimizer update, round-trip, and synthetic profile.

The script loads only local checkpoint files.  It never writes into
``models/``; round-trip files are placed under a temporary directory and the
small classifier head is saved under the requested artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TinyClassificationHead(nn.Module):
    """The E1 head shape, kept local so this smoke does not import a runner."""

    def __init__(self, hidden_size: int, num_labels: int = 3) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states[:, 0]
        return self.output(self.dropout(torch.tanh(self.dense(pooled))))


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().item())


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{requested} requested but CUDA is unavailable")
        return torch.device(requested)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unsupported device {requested!r}")


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_synthetic(
    model: nn.Module,
    device: torch.device,
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    model.eval()
    input_ids = torch.randint(
        low=4,
        high=vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = 0
    attention_mask = torch.ones_like(input_ids)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(**batch)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(steps):
            output = model(**batch).last_hidden_state
            if not finite(output):
                raise RuntimeError(f"Non-finite output during profile at length {sequence_length}")
        sync(device)
        elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "steps": steps,
        "warmup_steps": warmup,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed if elapsed else None,
        "sequences_per_second": batch_size * steps / elapsed if elapsed else None,
        "tokens_per_second": batch_size * sequence_length * steps / elapsed if elapsed else None,
        "device": str(device),
        "precision": "fp32",
    }
    if device.type == "cuda":
        result["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    else:
        result["peak_memory_allocated_bytes"] = None
        result["peak_memory_reserved_bytes"] = None
    return result


def profile_bf16_train_step(
    model: nn.Module,
    device: torch.device,
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    hidden_size: int,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    """Measure a real BF16 forward/backward/AdamW step at E1 batch shapes."""
    if device.type != "cuda":
        raise RuntimeError("BF16 training profile requires a CUDA device")
    model.train()
    head = TinyClassificationHead(hidden_size, num_labels=3).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=1e-4, weight_decay=0.01
    )
    input_ids = torch.randint(
        low=4,
        high=vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, 0] = 0
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 3, (batch_size,), dtype=torch.long, device=device)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = model(**batch).last_hidden_state
            logits = head(hidden)
            loss = functional.cross_entropy(logits.float(), labels)
        if not finite(loss):
            raise RuntimeError(f"Non-finite BF16 loss at length {sequence_length}")
        loss.backward()
        gradients = [parameter.grad for parameter in list(model.parameters()) + list(head.parameters())]
        if not all(gradient is not None and finite(gradient) for gradient in gradients):
            raise RuntimeError(f"Non-finite BF16 gradient at length {sequence_length}")
        optimizer.step()
        return float(loss.detach().item())

    for _ in range(warmup):
        step()
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = [step() for _ in range(steps)]
    sync(device)
    elapsed = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "steps": steps,
        "warmup_steps": warmup,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps / elapsed if elapsed else None,
        "sequences_per_second": batch_size * steps / elapsed if elapsed else None,
        "tokens_per_second": batch_size * sequence_length * steps / elapsed if elapsed else None,
        "device": str(device),
        "precision": "bf16_autocast",
        "backward_and_optimizer": True,
        "losses_finite": all(math.isfinite(loss) for loss in losses),
        "losses": losses,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--profile", action="store_true", help="run batch32 length256/512 profile")
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument(
        "--bf16-profile",
        action="store_true",
        help="run BF16 batch32 forward/backward/optimizer profile at length256/512",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output or (root / "artifacts" / "e0" / "smoke_models.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    student_dir = root / "models" / "goiturn-s-gd"
    teacher_dir = root / "models" / "goibert-b-dapt"
    source_files = [student_dir / "model.safetensors", teacher_dir / "model.safetensors"]
    hashes_before = {str(path): sha256_file(path) for path in source_files}
    started_at = datetime.now(timezone.utc).isoformat()

    tokenizer_student = AutoTokenizer.from_pretrained(student_dir, local_files_only=True)
    tokenizer_teacher = AutoTokenizer.from_pretrained(teacher_dir, local_files_only=True)
    tokenizer_compatible = tokenizer_student.get_vocab() == tokenizer_teacher.get_vocab()
    sample_texts = [
        "Sản phẩm giao nhanh và đóng gói cẩn thận.",
        "Chất lượng chưa tốt nhưng nhân viên hỗ trợ nhiệt tình.",
    ]
    batch_cpu = tokenizer_student(
        sample_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    if not tokenizer_compatible:
        raise RuntimeError("Student and teacher tokenizers have different vocabularies")

    student = AutoModel.from_pretrained(
        student_dir,
        local_files_only=True,
        add_pooling_layer=False,
        use_safetensors=True,
    ).to(device)
    teacher = AutoModel.from_pretrained(
        teacher_dir,
        local_files_only=True,
        add_pooling_layer=False,
        use_safetensors=True,
    ).to(device)
    student.eval()
    teacher.eval()
    batch = move_batch(batch_cpu, device)
    with torch.inference_mode():
        student_output = student(**batch).last_hidden_state
        teacher_output = teacher(**batch).last_hidden_state
    forward = {
        "student_shape": list(student_output.shape),
        "teacher_shape": list(teacher_output.shape),
        "student_finite": finite(student_output),
        "teacher_finite": finite(teacher_output),
        "student_hidden_size": student.config.hidden_size,
        "teacher_hidden_size": teacher.config.hidden_size,
        "tokenizer_vocab_size": len(tokenizer_student),
        "tokenized_lengths": batch_cpu["attention_mask"].sum(dim=1).tolist(),
    }

    # Add a fresh task head and prove that a real optimizer step changes both
    # the new head and an encoder parameter.  This is a tiny supervised smoke,
    # not an E1 experiment.
    student.train()
    head = TinyClassificationHead(student.config.hidden_size, num_labels=3).to(device)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=0.01
    )
    labels = torch.tensor([0, 2], dtype=torch.long, device=device)
    encoder_probe = next(parameter for parameter in student.parameters() if parameter.requires_grad)
    head_probe = next(parameter for parameter in head.parameters() if parameter.requires_grad)
    encoder_before = encoder_probe.detach().clone()
    head_before = head_probe.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    train_output = student(**batch).last_hidden_state
    logits = head(train_output)
    loss = functional.cross_entropy(logits, labels)
    loss.backward()
    gradient_values = [parameter.grad for parameter in list(student.parameters()) + list(head.parameters())]
    finite_gradients = all(gradient is not None and finite(gradient) for gradient in gradient_values)
    gradient_norm = float(
        torch.sqrt(
            sum(
                gradient.detach().float().pow(2).sum()
                for gradient in gradient_values
                if gradient is not None
            )
        ).item()
    )
    optimizer.step()
    encoder_delta = float((encoder_probe.detach() - encoder_before).abs().max().item())
    head_delta = float((head_probe.detach() - head_before).abs().max().item())
    update = {
        "loss": float(loss.detach().item()),
        "loss_finite": finite(loss),
        "gradients_finite": finite_gradients,
        "gradient_l2_norm": gradient_norm,
        "encoder_parameter_max_delta": encoder_delta,
        "head_parameter_max_delta": head_delta,
        "optimizer_updated_encoder": encoder_delta > 0,
        "optimizer_updated_head": head_delta > 0,
    }

    # Round-trip the updated encoder and fresh head in an ephemeral directory.
    student.eval()
    head.eval()
    with torch.inference_mode():
        reference_logits = head(student(**batch).last_hidden_state).detach().cpu()
        reference_hidden = student(**batch).last_hidden_state.detach().cpu()
    with tempfile.TemporaryDirectory(prefix="goibert-e0-roundtrip-") as temporary:
        temporary_path = Path(temporary)
        student.to("cpu")
        student.save_pretrained(temporary_path, safe_serialization=True)
        head_path = output.parent / "student_head_roundtrip.safetensors"
        save_file({key: value.detach().cpu() for key, value in head.state_dict().items()}, str(head_path))
        reloaded_student = AutoModel.from_pretrained(
            temporary_path,
            local_files_only=True,
            add_pooling_layer=False,
            use_safetensors=True,
        ).eval()
        reloaded_head = TinyClassificationHead(reloaded_student.config.hidden_size, 3).eval()
        reloaded_head.load_state_dict(load_file(str(head_path), device="cpu"))
        with torch.inference_mode():
            roundtrip_hidden = reloaded_student(**batch_cpu).last_hidden_state
            roundtrip_logits = reloaded_head(roundtrip_hidden)
        hidden_max_abs_diff = float((reference_hidden - roundtrip_hidden).abs().max().item())
        logits_max_abs_diff = float((reference_logits - roundtrip_logits).abs().max().item())
        roundtrip = {
            "encoder_save_load_ok": bool(torch.isfinite(roundtrip_hidden).all().item()),
            "head_save_load_ok": bool(torch.isfinite(roundtrip_logits).all().item()),
            "hidden_max_abs_diff": hidden_max_abs_diff,
            "logits_max_abs_diff": logits_max_abs_diff,
            "head_artifact": str(head_path),
            "ephemeral_encoder_artifact": "temporary_directory_removed_after_check",
        }
    student.to(device)
    teacher.to(device)

    profile: list[dict[str, Any]] = []
    if args.profile:
        student.eval()
        for sequence_length in (256, 512):
            profile.append(
                profile_synthetic(
                    student,
                    device,
                    batch_size=32,
                    sequence_length=sequence_length,
                    vocab_size=student.config.vocab_size,
                    warmup=args.profile_warmup,
                    steps=args.profile_steps,
                )
            )

    bf16_profile: list[dict[str, Any]] = []
    if args.bf16_profile:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Requested BF16 profile but this CUDA device does not support BF16")
        for sequence_length in (256, 512):
            bf16_profile.append(
                profile_bf16_train_step(
                    student,
                    device,
                    batch_size=32,
                    sequence_length=sequence_length,
                    vocab_size=student.config.vocab_size,
                    hidden_size=student.config.hidden_size,
                    warmup=args.profile_warmup,
                    steps=args.profile_steps,
                )
            )

    hashes_after = {str(path): sha256_file(path) for path in source_files}
    report = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "device": {
            "requested": args.device,
            "resolved": str(device),
            "torch_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "precision": "fp32",
            "bf16_supported": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
        },
        "student": {
            "path": str(student_dir),
            "parameter_count": count_parameters(student),
            "config": student.config.to_dict(),
        },
        "teacher": {
            "path": str(teacher_dir),
            "parameter_count": count_parameters(teacher),
            "config": teacher.config.to_dict(),
        },
        "tokenizer": {
            "student_teacher_vocab_equal": tokenizer_compatible,
            "student_vocab_size": len(tokenizer_student),
            "teacher_vocab_size": len(tokenizer_teacher),
        },
        "forward": forward,
        "backward_optimizer_update": update,
        "roundtrip": roundtrip,
        "profile": profile,
        "bf16_profile": bf16_profile,
        "source_weight_hashes_before": hashes_before,
        "source_weight_hashes_after": hashes_after,
        "source_weights_unchanged": hashes_before == hashes_after,
        "pass": bool(
            tokenizer_compatible
            and forward["student_finite"]
            and forward["teacher_finite"]
            and update["loss_finite"]
            and update["gradients_finite"]
            and update["optimizer_updated_encoder"]
            and update["optimizer_updated_head"]
            and roundtrip["encoder_save_load_ok"]
            and roundtrip["head_save_load_ok"]
            and roundtrip["hidden_max_abs_diff"] < 1e-5
            and roundtrip["logits_max_abs_diff"] < 1e-5
            and hashes_before == hashes_after
            and all(item["losses_finite"] for item in bf16_profile)
        ),
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "pass": report["pass"],
                "device": report["device"],
                "profile": profile,
                "bf16_profile": bf16_profile,
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
