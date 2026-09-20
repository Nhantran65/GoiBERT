"""Run the fixed E9 label-aware-attention validation-only screening.

The frozen E2 training loop supplies data loading, loss composition,
validation, checkpoint writing and scheduler behavior.  This runner patches
only the E2 student constructor and optimizer factory in each worker, then
adds E9 provenance and durable scheduling around that call.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

# This must happen before importing torch in the controller and every worker.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e9"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
PROTOCOL_PATH = ROOT / "configs" / "e9" / "protocol.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402
from goibert_e9.modeling import (  # noqa: E402
    E9GoiBERTClassifier,
    build_e9_student,
    content_attention_mask,
)


PROTOCOL = "E9-label-aware-attention-screening-v1"
REVISION = "e9-20260910-r1"
TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
TASK_LRS = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}
NC_LAMBDAS = {
    "vsfc_sentiment": 0.1,
    "vsfc_topic": 0.5,
    "visfd": 0.5,
    "absa_restaurant": 0.1,
    "absa_hotel": 0.1,
}
E8_BASELINES = {
    "vsfc_sentiment": 85.1604428,
    "vsfc_topic": 80.3677890,
    "visfd": 82.4420542,
    "absa_restaurant": 71.2765957,
    "absa_hotel": 73.0490748,
}
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 32
MAX_LENGTH = 256
GAMMA = 0.8
NUM_LAYERS = 10
WARMUP_FRACTION = 0.1
GPU_GUARD_MIB = 5 * 1024
# E9 is deliberately serialized on GPU2 for this approved round so it cannot
# claim a second GPU used by another experiment.
PREFERRED_GPU_INDICES = ("2",)
E1_SOURCE_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"
E2_CODE_HASH = "efb81ac232de2f0ae4287688ad0125a1d84fb1e689e2b1566ef26d8d6e2b60b5"
E2_CONFIG_HASH = "45e6c36196fe95ae0eb9193e9197224bdf2980515c15ddf01998267b668b61b8"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    return e1_train._json_default(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(root: Path, suffixes: set[str]) -> str:
    digest = hashlib.sha256()
    files = [
        path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes
    ] if root.is_dir() else []
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def source_hashes() -> dict[str, Any]:
    """Resolve E9's own code/config and frozen E1/E2/E7 dependencies."""

    e2_values = e2_train.source_hashes(ROOT)
    if e2_values["e1_dependency"] != E1_SOURCE_HASH:
        raise RuntimeError(
            f"Frozen E1 dependency drift: expected {E1_SOURCE_HASH}, "
            f"found {e2_values['e1_dependency']}"
        )
    if e2_values["e2"] != E2_CODE_HASH:
        raise RuntimeError(f"Frozen E2 code drift: expected {E2_CODE_HASH}, found {e2_values['e2']}")
    if e2_values["e2_config"] != E2_CONFIG_HASH:
        raise RuntimeError(
            f"Frozen E2 config drift: expected {E2_CONFIG_HASH}, found {e2_values['e2_config']}"
        )
    e9_code = _tree_hash(ROOT / "src" / "goibert_e9", {".py"})
    e9_config = _tree_hash(ROOT / "configs" / "e9", {".json"})
    e9_source = hashlib.sha256((e9_code + e9_config).encode("ascii")).hexdigest()
    e7_paths = [
        *sorted((ROOT / "src" / "goibert_e7").rglob("*.py")),
        ROOT / "configs" / "e7" / "protocol.json",
    ]
    e7_digest = hashlib.sha256()
    for path in e7_paths:
        e7_digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        e7_digest.update(_sha256(path).encode("ascii"))
    return {
        "e1_dependency": e2_values["e1_dependency"],
        "e2": e2_values["e2"],
        "e2_source": e2_values["e2_source"],
        "e2_config": e2_values["e2_config"],
        "e2_orchestrator": e2_values["orchestrator"],
        "e7_dependency": e7_digest.hexdigest(),
        "e9": e9_code,
        "e9_config": e9_config,
        "e9_source": e9_source,
        "e9_runner": _sha256(Path(__file__).resolve()),
    }


def _gpu_snapshot() -> dict[str, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as exc:
        raise RuntimeError(f"Unable to query host GPUs with nvidia-smi: {exc}") from exc
    snapshot: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 3:
            continue
        index, free, used = fields[:3]
        snapshot[index] = {"free_mib": int(float(free)), "used_mib": int(float(used))}
    if not snapshot:
        raise RuntimeError(f"nvidia-smi returned no GPU rows: {output!r}")
    return snapshot


def _eligible_devices() -> tuple[list[str], dict[str, dict[str, int]]]:
    snapshot = _gpu_snapshot()
    devices = [
        f"cuda:{index}"
        for index in PREFERRED_GPU_INDICES
        if index in snapshot and snapshot[index]["free_mib"] >= GPU_GUARD_MIB
    ]
    if not devices:
        raise RuntimeError(
            f"No preferred GPU has the {GPU_GUARD_MIB} MiB free-memory guard; snapshot={snapshot}"
        )
    return devices, snapshot


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _fresh_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}__rerun{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Unable to allocate fresh path for {path}")


def _preflight_model_checks(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    encoded = tokenizer(
        ["món ăn ngon và phục vụ nhanh", "dịch vụ tốt"],
        add_special_tokens=True,
        truncation=True,
        max_length=16,
        padding=True,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
    shape_rows: list[dict[str, Any]] = []
    gradient_check: dict[str, Any] | None = None
    init_equal = True
    optimizer_coverage: dict[str, Any] | None = None
    roundtrip: dict[str, Any] | None = None

    for task_index, task_id in enumerate(TASKS):
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False, smoke=False)
        num_aspects = e1_train.task_num_aspects(bundle)
        counts = e1_train.task_aspect_class_counts(bundle)
        num_labels = e1_train.task_num_classes(bundle)
        model, spec = build_e9_student(
            ROOT / "models" / "goiturn-s-gd",
            num_classes=num_labels,
            num_aspects=num_aspects,
            aspect_class_counts=counts if num_aspects else None,
            attention_implementation="sdpa",
        )
        model.to(device)
        model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(batch["input_ids"], batch["attention_mask"])
        expected = (
            (len(encoded["input_ids"]), num_aspects, counts[0])
            if num_aspects and len(set(counts)) == 1
            else (len(encoded["input_ids"]), num_labels)
        )
        if tuple(logits.shape) != tuple(expected):
            raise AssertionError(f"{task_id} shape {tuple(logits.shape)} != expected {expected}")
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError(f"{task_id} produced non-finite logits")
        shape_rows.append(
            {
                "task_id": task_id,
                "num_aspects": num_aspects,
                "aspect_class_counts": list(counts),
                "num_labels": num_labels,
                "logits_shape": list(logits.shape),
                "model_spec": spec.to_dict(),
            }
        )

        if task_index == 0:
            model.train()
            labels = torch.tensor([0, 1], dtype=torch.long, device=device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                train_logits = model(batch["input_ids"], batch["attention_mask"])
                loss = e1_train.compute_loss(
                    train_logits,
                    labels,
                    num_aspects=0,
                    aspect_class_counts=(),
                )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("E9 preflight loss is non-finite")
            loss.backward()
            query_grad = model.head.queries.grad
            encoder_grad = model.encoder.embeddings.word_embeddings.weight.grad
            gradient_check = {
                "loss_finite": bool(torch.isfinite(loss).item()),
                "query_gradient_present": query_grad is not None,
                "query_gradient_finite": bool(query_grad is not None and torch.isfinite(query_grad).all().item()),
                "encoder_gradient_present": encoder_grad is not None,
                "encoder_gradient_finite": bool(encoder_grad is not None and torch.isfinite(encoder_grad).all().item()),
            }
            if not all(gradient_check.values()):
                raise AssertionError(f"E9 gradient preflight failed: {gradient_check}")
            optimizer = e7_optim.build_layerwise_optimizer(model, TASK_LRS[task_id], gamma=GAMMA, num_layers=NUM_LAYERS)
            optimizer_coverage = e7_optim.validate_optimizer_groups(
                model,
                optimizer,
                base_lr=TASK_LRS[task_id],
                gamma=GAMMA,
                num_layers=NUM_LAYERS,
            )
            if not optimizer_coverage["all_trainable"] or optimizer_coverage["duplicate_parameter_count"] != 0:
                raise AssertionError("E9 optimizer does not cover each trainable parameter exactly once")
            for group in optimizer.param_groups:
                expected_lr = TASK_LRS[task_id] * float(group["e7_lr_scale"])
                if not math.isclose(float(group["lr"]), expected_lr, rel_tol=1e-12, abs_tol=1e-15):
                    raise AssertionError("E9 optimizer LR scale mismatch")

            # The explicit FP32 score/softmax block is checked under the same
            # BF16 autocast context used by trials.
            allowed, content = content_attention_mask(
                batch["input_ids"], batch["attention_mask"], tokenizer.all_special_ids
            )
            if not bool(content.any(dim=1).all().item()):
                raise AssertionError("real preflight text unexpectedly has no content token")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                hidden_for_attention = model.encoder(
                    batch["input_ids"], batch["attention_mask"], return_dict=True
                ).last_hidden_state
                weights = model.head._attention_weights(
                    hidden_for_attention,
                    batch["input_ids"],
                    batch["attention_mask"],
                )
            if weights.dtype != torch.float32:
                raise AssertionError(f"E9 attention weights dtype is {weights.dtype}, expected float32")
            if not bool(torch.isfinite(weights).all().item()):
                raise FloatingPointError("E9 attention weights are non-finite")

            # Strict checkpoint round-trip against a fresh E9 constructor,
            # including canonical tuple/list metadata equality.
            state = {"model": model.state_dict(), "metadata": spec.to_dict()}
            buffer = io.BytesIO()
            torch.save(state, buffer)
            buffer.seek(0)
            loaded = torch.load(buffer, map_location="cpu", weights_only=False)
            fresh, fresh_spec = build_e9_student(
                ROOT / "models" / "goiturn-s-gd",
                num_classes=num_labels,
                num_aspects=num_aspects,
                aspect_class_counts=counts if num_aspects else None,
                attention_implementation="sdpa",
            )
            fresh.load_state_dict(loaded["model"], strict=True)
            if _canonical(loaded["metadata"]) != _canonical(fresh_spec.to_dict()):
                raise AssertionError("E9 checkpoint metadata changed across constructor round-trip")
            roundtrip = {"strict_load": True, "metadata_canonical_equal": True}
            del fresh, loaded, buffer, state, optimizer

            # Compare encoder tensors against a direct original GD load.  The
            # E9 constructor must not substitute E6/E8/fine-tuned weights.
            from transformers import RobertaModel
            original_encoder = RobertaModel.from_pretrained(
                ROOT / "models" / "goiturn-s-gd",
                local_files_only=True,
                add_pooling_layer=False,
                attn_implementation="sdpa",
            )
            current_encoder = {
                key: value.detach().cpu()
                for key, value in model.encoder.state_dict().items()
            }
            init_equal = all(
                torch.equal(current_encoder[key], value.detach().cpu())
                for key, value in original_encoder.state_dict().items()
            )
            if not init_equal:
                raise AssertionError("E9 encoder initialization differs from original GD")
            del original_encoder, current_encoder
        del model, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "task_shapes": shape_rows,
        "mask_and_empty_fallback": _mask_fallback_check(tokenizer),
        "gradient_check": gradient_check,
        "optimizer_coverage": optimizer_coverage,
        "strict_checkpoint_roundtrip": roundtrip,
        "encoder_init_equal_original_gd": init_equal,
    }


def _canonical(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    return value


def _mask_fallback_check(tokenizer: Any) -> dict[str, Any]:
    special = set(int(value) for value in tokenizer.all_special_ids)
    content_id = next(value for value in range(int(tokenizer.vocab_size)) if value not in special)
    bos = int(tokenizer.bos_token_id)
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)
    ids = torch.tensor([[bos, eos, pad, pad], [bos, content_id, eos, pad]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)
    allowed, content = content_attention_mask(ids, mask, tokenizer.all_special_ids)
    expected_allowed = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)
    expected_content = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool)
    result = {
        "passed": bool(torch.equal(allowed, expected_allowed) and torch.equal(content, expected_content)),
        "all_special_empty_allowed_cls_only": allowed[0].tolist(),
        "content_row_allowed": allowed[1].tolist(),
        "content_mask": content.tolist(),
        "special_token_ids": sorted(special),
    }
    if not result["passed"]:
        raise AssertionError(f"E9 special/padding mask fallback failed: {result}")
    return result


def run_preflight(device_name: str) -> dict[str, Any]:
    started = time.time()
    values = source_hashes()
    snapshot = _gpu_snapshot()
    physical_index = device_name.split(":", 1)[1] if ":" in device_name else device_name
    free_mib = snapshot.get(physical_index, {}).get("free_mib", 0)
    checks: dict[str, Any] = {
        "frozen_e1_dependency": values["e1_dependency"] == E1_SOURCE_HASH,
        "frozen_e2_code": values["e2"] == E2_CODE_HASH,
        "frozen_e2_config": values["e2_config"] == E2_CONFIG_HASH,
        "memory_guard": free_mib >= GPU_GUARD_MIB,
    }
    model_checks: dict[str, Any] = {"skipped": True, "reason": "memory guard failed"}
    if checks["memory_guard"]:
        model_checks = _preflight_model_checks(device_name)
        checks.update(
            {
                "all_task_shapes": len(model_checks["task_shapes"]) == len(TASKS),
                "mask_and_empty_fallback": bool(model_checks["mask_and_empty_fallback"]["passed"]),
                "finite_loss_and_gradients": all(model_checks["gradient_check"].values()),
                "optimizer_unique_and_lr_correct": bool(
                    model_checks["optimizer_coverage"]["all_trainable"]
                    and model_checks["optimizer_coverage"]["duplicate_parameter_count"] == 0
                ),
                "strict_checkpoint_roundtrip": bool(
                    model_checks["strict_checkpoint_roundtrip"]["strict_load"]
                    and model_checks["strict_checkpoint_roundtrip"]["metadata_canonical_equal"]
                ),
                "encoder_init_equal_original_gd": bool(model_checks["encoder_init_equal_original_gd"]),
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "source_hashes": values,
        "device": device_name,
        "gpu_snapshot": snapshot,
        "memory_profile": {"free_mib_before": free_mib, "guard_mib": GPU_GUARD_MIB},
        "checks": checks,
        "model_checks": model_checks,
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "precision": "bf16",
            "attention_implementation": "sdpa",
            "include_test": False,
        },
    }
    path = _fresh_path(ORCH_ROOT / "preflight.json")
    _write_json(path, payload)
    payload["artifact_path"] = str(path)
    return payload


def _common_schedule_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return float(step / max(1, warmup_steps))
    return float(max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)))


def _annotate_artifact(
    artifact_dir: Path,
    *,
    task_id: str,
    device_name: str,
    values: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    coverage: Mapping[str, Any],
    started_at: str,
) -> dict[str, Any]:
    group_rows = e7_optim.describe_optimizer_groups(optimizer)
    for row in group_rows:
        row["parameter_tensor_count"] = row.pop("parameter_count")
        row["current_lr_after_training"] = row["lr"]
    epoch_path = artifact_dir / "epoch_curves.json"
    epoch_payload = _read_json(epoch_path)
    epoch_rows = epoch_payload.get("epochs")
    if not isinstance(epoch_rows, list) or not epoch_rows:
        raise ValueError(f"E9 epoch curves missing from {artifact_dir}")
    total_steps = int(epoch_rows[-1].get("global_step", 0))
    warmup_steps = round(total_steps * WARMUP_FRACTION)
    for row in epoch_rows:
        factor = _common_schedule_factor(
            int(row.get("global_step", 0)),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        row["scheduler_factor"] = factor
        row["learning_rates_by_group"] = [
            {
                "name": group["name"],
                "lr": float(group["initial_lr"] * factor),
                "initial_lr": float(group["initial_lr"]),
                "lr_scale": float(group["lr_scale"]),
            }
            for group in group_rows
        ]
    epoch_payload["e9_learning_rate_logging"] = {
        "source": "reconstructed from frozen E2 scheduler and captured E7 initial groups",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "group_count": len(group_rows),
    }
    _write_json(epoch_path, epoch_payload)

    fields: dict[str, Any] = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "base_protocol": "E2-training-ablation-v1",
        "e9_condition": "label_aware_attention_plus_nc_llrd",
        "e9_source_hash": values["e9_source"],
        "e9_code_hash": values["e9"],
        "e9_config_hash": values["e9_config"],
        "e9_runner_hash": values["e9_runner"],
        "e1_dependency_source_hash": values["e1_dependency"],
        "e2_dependency_source_hash": values["e2_source"],
        "e2_code_hash": values["e2"],
        "e2_config_hash": values["e2_config"],
        "e7_dependency_source_hash": values["e7_dependency"],
        "architecture": {
            "name": "label_aware_attention",
            "query_size": 384,
            "query_initializer": "normal(std=0.02)",
            "attention_softmax_dtype": "float32",
            "readout": "(CLS + weighted_context) / 2",
            "new_parameters_prefix": "head.queries",
        },
        "nc": {
            "mode": "nc",
            "consistency_lambda": NC_LAMBDAS[task_id],
            "temperature": 1.0,
        },
        "optimizer_variant": "layerwise_lr_decay",
        "layerwise_lr_decay": {
            "gamma": GAMMA,
            "num_layers": NUM_LAYERS,
            "embedding_exponent": NUM_LAYERS,
            "block_exponent": "9-layer_index",
            "head_exponent": 0,
        },
        "optimizer_group_contract": {
            "name": "AdamW",
            "base_learning_rate": TASK_LRS[task_id],
            "gamma": GAMMA,
            "num_layers": NUM_LAYERS,
            "scheduler": "E2 linear warmup/decay; common multiplier across groups",
            "groups": group_rows,
            "coverage": dict(coverage),
        },
        "worker": {"task_id": task_id, "device": device_name, "started_at": started_at},
    }
    for name in ("metadata.json", "resolved_config.json"):
        path = artifact_dir / name
        content = _read_json(path)
        content.update(fields)
        _write_json(path, content)
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = dict(checkpoint.get("metadata", {}))
    checkpoint_metadata.update(fields)
    checkpoint["metadata"] = checkpoint_metadata
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    return {"group_contract": group_rows, "total_steps": total_steps, "warmup_steps": warmup_steps}


def run_worker(task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task {task_id!r}")
    values = source_hashes()
    started_at = _now()
    status_started = {
        "protocol": PROTOCOL,
        "task_id": task_id,
        "status": "running",
        "pid": os.getpid(),
        "started_at": started_at,
        "device": device_name,
        "source_hashes": values,
        "output_dir": str(output_dir),
    }
    _write_json(WORKER_ROOT / f"{task_id}.json", status_started)
    captured: dict[str, Any] = {}
    original_builder = e2_train.build_e2_student
    original_optimizer = e2_train.make_optimizer

    def e9_make_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
        optimizer = e7_optim.build_layerwise_optimizer(
            model, learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS
        )
        captured["optimizer"] = optimizer
        captured["coverage"] = e7_optim.validate_optimizer_groups(
            model,
            optimizer,
            base_lr=learning_rate,
            gamma=GAMMA,
            num_layers=NUM_LAYERS,
        )
        return optimizer

    e2_train.build_e2_student = build_e9_student  # type: ignore[assignment]
    e2_train.make_optimizer = e9_make_optimizer  # type: ignore[assignment]
    try:
        result = e2_train.train_student_trial(
            task_id=task_id,
            mode="nc",
            repo_root=ROOT,
            output_dir=output_dir,
            seed=SEED,
            consistency_lambda=NC_LAMBDAS[task_id],
            learning_rate=TASK_LRS[task_id],
            device_name=device_name,
            precision="bf16",
            attention_implementation="sdpa",
            epochs=EPOCHS,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            stage="stageE9_label_attention",
            method="label_attention_plus_nc_llrd",
            coefficient=NC_LAMBDAS[task_id],
            revision=REVISION,
            expected_e2_code_hash=values["e2"],
            expected_e2_config_hash=values["e2_config"],
            expected_orchestrator_hash=values["e2_orchestrator"],
        )
    except Exception as exc:
        _write_json(
            WORKER_ROOT / f"{task_id}.json",
            {
                **status_started,
                "status": "failed",
                "finished_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        e2_train.build_e2_student = original_builder  # type: ignore[assignment]
        e2_train.make_optimizer = original_optimizer  # type: ignore[assignment]

    optimizer = captured.get("optimizer")
    coverage = captured.get("coverage")
    if optimizer is None or coverage is None:
        raise RuntimeError("E9 optimizer adapter did not capture optimizer/coverage")
    artifact_dir = Path(str(result["artifact_dir"]))
    annotation = _annotate_artifact(
        artifact_dir,
        task_id=task_id,
        device_name=device_name,
        values=values,
        optimizer=optimizer,
        coverage=coverage,
        started_at=started_at,
    )
    completed = {
        **status_started,
        "status": "completed",
        "finished_at": _now(),
        "artifact_dir": str(artifact_dir),
        "best_epoch": result.get("best_epoch"),
        "best_selection_f1": result.get("best_selection_f1"),
        "validation_metrics": result.get("validation_metrics"),
        "optimizer_group_count": len(annotation["group_contract"]),
    }
    _write_json(WORKER_ROOT / f"{task_id}.json", completed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return result


def _artifact_candidates(task_id: str) -> list[Path]:
    parent = ARTIFACT_ROOT / task_id
    return sorted(parent.glob(f"seed_{SEED}*")) if parent.is_dir() else []


def _completed_artifact(task_id: str) -> Path | None:
    for path in _artifact_candidates(task_id):
        try:
            status = _read_json(path / "status.json")
            metadata = _read_json(path / "metadata.json")
        except Exception:
            continue
        if (
            status.get("status") == "completed"
            and metadata.get("protocol") == PROTOCOL
            and (path / "best_checkpoint.pt").is_file()
            and (path / "validation_predictions.json").is_file()
        ):
            return path
    return None


def _assert_no_duplicate_tasks() -> dict[str, Path]:
    completed: dict[str, Path] = {}
    for task_id in TASKS:
        worker_path = WORKER_ROOT / f"{task_id}.json"
        if worker_path.is_file():
            worker = _read_json(worker_path)
            if worker.get("status") == "running" and _pid_alive(worker.get("pid")):
                raise RuntimeError(f"E9 task {task_id} already has active worker PID {worker.get('pid')}")
        artifact = _completed_artifact(task_id)
        if artifact is not None:
            completed[task_id] = artifact
    return completed


def _summary_from_artifacts(artifacts: Mapping[str, Path], values: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        artifact_dir = Path(artifacts[task_id])
        predictions = _read_json(artifact_dir / "validation_predictions.json").get("rows")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError(f"Missing validation predictions for {task_id}")
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
        valid_split = e1_train.split_field(bundle, "validation")
        expected_ids = [str(value) for value in e1_train.split_ids(valid_split)]
        actual_ids = [str(row["id"]) for row in predictions]
        if actual_ids != expected_ids:
            raise ValueError(f"E9 prediction IDs do not match validation order for {task_id}")
        recomputed = e1_train.evaluate_predictions(
            task_id,
            bundle,
            [row["gold"] for row in predictions],
            [row["prediction"] for row in predictions],
            split="validation",
        )
        metadata = _read_json(artifact_dir / "metadata.json")
        e9_f1 = float(recomputed["selection_f1"])
        delta = e9_f1 - E8_BASELINES[task_id]
        rows.append(
            {
                "task_id": task_id,
                "artifact_dir": str(artifact_dir),
                "seed": SEED,
                "best_epoch": metadata.get("best_epoch"),
                "selection_f1_e8": E8_BASELINES[task_id],
                "selection_f1_e9_saved": float(metadata["best_selection_f1"]),
                "selection_f1_e9_recomputed": e9_f1,
                "delta_e9_minus_e8": delta,
                "delta_e9_minus_e8_pp": delta * 100.0,
                "improved": delta > 0,
                "improved_at_least_0p3pp": delta >= 0.003,
                "decreased_over_0p3pp": delta < -0.003,
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "peak_gpu_memory_allocated_bytes": metadata.get("peak_gpu_memory_allocated_bytes"),
                "peak_gpu_memory_reserved_bytes": metadata.get("peak_gpu_memory_reserved_bytes"),
                "prediction_count": len(predictions),
                "metrics_e9": recomputed,
            }
        )
    increased = sum(row["improved"] for row in rows)
    strong = sum(row["improved_at_least_0p3pp"] for row in rows)
    largest_decrease = min(row["delta_e9_minus_e8_pp"] for row in rows)
    screening_pass = increased >= 4 and largest_decrease >= -0.3 and strong >= 2
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "completed",
        "created_at": _now(),
        "seed": SEED,
        "tasks": list(TASKS),
        "source_hashes": dict(values),
        "metric_recomputation": "one pass from saved validation_predictions.json; test excluded",
        "baselines": {"e8": "docs/E9_PLAN.md fixed validation selection F1 values"},
        "screening_gate": {
            "definition": "at least 4/5 increases, no task decreases over 0.3pp, at least 2 increases by 0.3pp",
            "tasks_increased": increased,
            "tasks_increased_at_least_0p3pp": strong,
            "largest_decrease_pp": largest_decrease,
            "passed": screening_pass,
        },
        "rows": rows,
    }
    _write_json(ARTIFACT_ROOT / "summary.json", payload)
    fieldnames = [
        "task_id", "seed", "best_epoch", "selection_f1_e8", "selection_f1_e9_recomputed",
        "delta_e9_minus_e8", "delta_e9_minus_e8_pp", "improved", "improved_at_least_0p3pp",
        "decreased_over_0p3pp", "elapsed_seconds", "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes", "prediction_count",
    ]
    with (ARTIFACT_ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    _write_readout(payload)
    return payload


def _write_readout(summary: Mapping[str, Any]) -> None:
    lines = [
        "# E9 Result: Label-aware Attention",
        "",
        "Five original GoiTurn-S-GD students use one learned query per VSFC class or ABSA aspect, with the fixed E8 NC plus E7 LLRD recipe. Validation only; seed 42; no test split.",
        "",
        "| Task | E8 baseline | E9 | Delta (pp) | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['task_id']} | {row['selection_f1_e8']:.4f} | {row['selection_f1_e9_recomputed']:.4f} | {row['delta_e9_minus_e8_pp']:+.3f} | {row['best_epoch']} |"
        )
    gate = summary["screening_gate"]
    lines.extend(
        [
            "",
            f"Screening gate: {'PASS' if gate['passed'] else 'FAIL'} ({gate['tasks_increased']}/5 increased; {gate['tasks_increased_at_least_0p3pp']} increased by at least 0.3pp; largest decrease {gate['largest_decrease_pp']:+.3f}pp).",
            "",
            f"Source hashes: E9 `{summary['source_hashes']['e9_source']}`, E7 dependency `{summary['source_hashes']['e7_dependency']}`, E2 `{summary['source_hashes']['e2_source']}`, E1 `{summary['source_hashes']['e1_dependency']}`.",
            "",
        ]
    )
    (ROOT / "docs" / "E9_RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def _existing_state_pid(state_path: Path) -> int | None:
    if not state_path.is_file():
        return None
    try:
        state = _read_json(state_path)
    except Exception:
        return None
    pid = state.get("controller_pid")
    return int(pid) if isinstance(pid, int) else None


def run_all(*, poll_seconds: float = 15.0) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    values = source_hashes()
    completed = _assert_no_duplicate_tasks()
    if len(completed) == len(TASKS):
        return _summary_from_artifacts(completed, values)
    state_path = ORCH_ROOT / "state.json"
    old_pid = _existing_state_pid(state_path)
    if old_pid is not None and old_pid != os.getpid() and _pid_alive(old_pid):
        raise RuntimeError(f"E9 controller already active with PID {old_pid}")
    devices, snapshot = _eligible_devices()
    preflight = run_preflight(devices[0])
    if preflight.get("status") != "passed":
        raise RuntimeError("E9 preflight did not pass; refusing to launch full trials")
    queue = [task_id for task_id in TASKS if task_id not in completed]
    free_devices = list(devices)
    running: dict[str, tuple[subprocess.Popen[str], str]] = {}
    jobs: dict[str, dict[str, Any]] = {
        task_id: {
            "task_id": task_id,
            "status": "completed",
            "artifact_dir": str(path),
            "reused_existing": True,
        }
        for task_id, path in completed.items()
    }
    controller_state: dict[str, Any] = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "running",
        "controller_pid": os.getpid(),
        "started_at": _now(),
        "source_hashes": values,
        "preferred_gpu_indices": list(PREFERRED_GPU_INDICES),
        "eligible_devices": devices,
        "gpu_snapshot_at_launch": snapshot,
        "tasks": list(TASKS),
        "jobs": jobs,
        "recovered_existing_completed_tasks": sorted(completed),
        "preflight": preflight.get("artifact_path"),
    }
    _write_json(state_path, controller_state)
    controller_log = (LOG_ROOT / "controller.log").open("a", encoding="utf-8", buffering=1)
    controller_log.write(json.dumps({"event": "controller_start", "time": _now(), "devices": devices}) + "\n")

    def launch(task_id: str, device_name: str) -> None:
        output_dir = ARTIFACT_ROOT / task_id / f"seed_{SEED}"
        log_path = LOG_ROOT / f"{task_id}__seed{SEED}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--task-id",
            task_id,
            "--device",
            device_name,
            "--output-dir",
            str(output_dir),
        ]
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(json.dumps({"event": "launch", "time": _now(), "command": command}) + "\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            },
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        log_handle.close()
        running[task_id] = (process, device_name)
        jobs[task_id] = {
            "task_id": task_id,
            "device": device_name,
            "pid": process.pid,
            "artifact_dir": str(output_dir),
            "log_path": str(log_path),
            "status": "running",
            "launched_at": _now(),
            "command": command,
        }
        _write_json(state_path, controller_state)

    try:
        while queue or running:
            while queue and free_devices:
                launch(queue.pop(0), free_devices.pop(0))
            finished: list[str] = []
            for task_id, (process, device_name) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                job = jobs[task_id]
                job.update({"status": "completed" if return_code == 0 else "failed", "returncode": return_code, "finished_at": _now()})
                if return_code == 0:
                    worker = _read_json(WORKER_ROOT / f"{task_id}.json")
                    if worker.get("status") != "completed":
                        raise RuntimeError(f"E9 worker {task_id} did not report completed")
                    job["artifact_dir"] = worker.get("artifact_dir")
                    completed[task_id] = Path(str(worker["artifact_dir"]))
                finished.append(task_id)
                free_devices.append(device_name)
            for task_id in finished:
                running.pop(task_id)
            if finished:
                _write_json(state_path, controller_state)
            failed = [task_id for task_id, job in jobs.items() if job.get("status") == "failed"]
            if failed:
                controller_state.update({"status": "failed", "finished_at": _now(), "failed_tasks": failed})
                _write_json(state_path, controller_state)
                raise RuntimeError(f"E9 workers failed: {failed}")
            if running:
                time.sleep(poll_seconds)
    finally:
        controller_log.close()
    summary = _summary_from_artifacts(completed, values)
    controller_state.update({"status": "completed", "finished_at": _now(), "summary": str(ARTIFACT_ROOT / "summary.json")})
    _write_json(state_path, controller_state)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed E9 label-aware-attention trials")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight:
            device = args.preflight_device or _eligible_devices()[0][0]
            payload = run_preflight(device)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0 if payload.get("status") == "passed" else 1
        if args.task_id:
            if args.output_dir is None:
                raise ValueError("--task-id requires --output-dir")
            run_worker(args.task_id, args.device, args.output_dir.resolve())
            return 0
        if args.run_all:
            payload = run_all()
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0
        raise ValueError("Choose one of --preflight, --run-all, or --task-id")
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
