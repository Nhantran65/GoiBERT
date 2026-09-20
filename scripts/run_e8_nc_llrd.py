"""Run the fixed E8 noise-consistency plus layer-wise LR-decay experiment.

E2 remains the training implementation and E7 remains the optimizer
implementation.  This runner only composes them in a worker process by
overriding ``goibert_e2.train.make_optimizer`` for the duration of one call,
then annotates the resulting artifact with complete E8 provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

# Set before importing torch, including in every durable child process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e8"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
PROTOCOL_PATH = ROOT / "configs" / "e8" / "protocol.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e2.losses import symmetric_kl_consistency_loss  # noqa: E402
from goibert_e2.modeling import build_e2_student  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402


PROTOCOL = "E8-noise-consistency-layerwise-learning-rate-decay-v1"
REVISION = "e8-20260910-r1"
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
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 32
MAX_LENGTH = 256
GAMMA = 0.8
NUM_LAYERS = 10
WARMUP_FRACTION = 0.1
GPU_GUARD_MIB = 5 * 1024
PREFERRED_GPU_INDICES = ("2", "1", "3")
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


def _hash_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.resolve().relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def e8_source_hash() -> str:
    """Hash the frozen E8 runner and protocol before launching workers."""

    return _hash_files((Path(__file__).resolve(), PROTOCOL_PATH))


def e7_dependency_hash() -> str:
    """Hash the imported E7 optimizer/config dependency, excluding this runner."""

    paths = [*sorted((ROOT / "src" / "goibert_e7").rglob("*.py")), ROOT / "configs" / "e7" / "protocol.json"]
    return _hash_files(paths)


def _source_hashes() -> dict[str, str]:
    values = e2_train.source_hashes(ROOT)
    if values["e1_dependency"] != E1_SOURCE_HASH:
        raise RuntimeError(
            f"Frozen E1 dependency drift: expected {E1_SOURCE_HASH}, found {values['e1_dependency']}"
        )
    if values["e2"] != E2_CODE_HASH:
        raise RuntimeError(f"Frozen E2 code drift: expected {E2_CODE_HASH}, found {values['e2']}")
    if values["e2_config"] != E2_CONFIG_HASH:
        raise RuntimeError(
            f"Frozen E2 config drift: expected {E2_CONFIG_HASH}, found {values['e2_config']}"
        )
    return values


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


def _worker_status(task_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_ROOT / f"{task_id}.json", dict(payload))


def _common_schedule_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return float(step / max(1, warmup_steps))
    return float(max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)))


def _preflight_optimizer(model: torch.nn.Module) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    optimizer = e7_optim.build_layerwise_optimizer(
        model, 1e-5, gamma=GAMMA, num_layers=NUM_LAYERS
    )
    coverage = e7_optim.validate_optimizer_groups(
        model,
        optimizer,
        base_lr=1e-5,
        gamma=GAMMA,
        num_layers=NUM_LAYERS,
        expected_group_names=[
            f"{region}_{kind}"
            for region in ("embeddings", *[f"block_{i:02d}" for i in range(NUM_LAYERS)], "head")
            for kind in ("decay", "no_decay")
        ],
    )
    if not coverage["all_trainable"] or coverage["duplicate_parameter_count"] != 0:
        raise AssertionError("E8 optimizer does not cover every trainable parameter exactly once")
    for group in optimizer.param_groups:
        expected = 0.01 if group.get("e7_decay") else 0.0
        if float(group["weight_decay"]) != expected:
            raise AssertionError("E8 optimizer no-decay contract failed")
    return optimizer, coverage


def _preflight_combined_loss(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    torch.manual_seed(777)
    model, model_spec = build_e2_student(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        attention_implementation="sdpa",
    )
    model.to(device)
    model.train()
    optimizer, coverage = _preflight_optimizer(model)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    base = "món ăn ngon và phục vụ nhanh "
    clean_texts = [base * 90 for _ in range(MICRO_BATCH_SIZE)]
    noisy_texts = ["món ăn tuyệt vời và phục vụ nhanh " * 90 for _ in range(MICRO_BATCH_SIZE)]
    clean = tokenizer(
        clean_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    noisy = tokenizer(
        noisy_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    clean = {key: value.to(device) for key, value in clean.items()}
    noisy = {key: value.to(device) for key, value in noisy.items()}
    labels = torch.arange(MICRO_BATCH_SIZE, device=device) % 3
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        clean_logits = model(clean["input_ids"], clean["attention_mask"])
        noisy_logits = model(noisy["input_ids"], noisy["attention_mask"])
        ce = torch.nn.functional.cross_entropy(clean_logits, labels)
        consistency = symmetric_kl_consistency_loss(
            clean_logits, noisy_logits, temperature=1.0
        )
        total = ce + 0.1 * consistency
    if not torch.isfinite(total):
        raise FloatingPointError("E8 preflight combined loss is non-finite")
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    finite_grads = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    result = {
        "passed": bool(torch.isfinite(clean_logits).all().item() and torch.isfinite(noisy_logits).all().item() and torch.isfinite(total).item() and finite_grads),
        "batch_size": MICRO_BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "clean_sequence_length": int(clean["input_ids"].shape[1]),
        "noisy_sequence_length": int(noisy["input_ids"].shape[1]),
        "clean_logits_finite": bool(torch.isfinite(clean_logits).all().item()),
        "noisy_logits_finite": bool(torch.isfinite(noisy_logits).all().item()),
        "loss_finite": bool(torch.isfinite(total).item()),
        "gradient_finite": bool(finite_grads),
        "loss": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "optimizer_coverage": coverage,
        "model_spec": model_spec.to_dict(),
    }
    del clean_logits, noisy_logits, total, ce, consistency, clean, noisy, optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_preflight(device_name: str) -> dict[str, Any]:
    started = time.time()
    source_values = _source_hashes()
    snapshot = _gpu_snapshot()
    physical_index = device_name.split(":", 1)[1] if ":" in device_name else device_name
    free_mib = snapshot.get(physical_index, {}).get("free_mib", 0)
    memory_guard = free_mib >= GPU_GUARD_MIB
    combined: dict[str, Any] = {"passed": False, "skipped": True, "reason": "memory guard failed"}
    coverage: dict[str, Any] = {}
    if memory_guard:
        combined = _preflight_combined_loss(device_name)
        coverage = dict(combined.get("optimizer_coverage", {}))
    checks = {
        "frozen_e1_dependency": source_values["e1_dependency"] == E1_SOURCE_HASH,
        "frozen_e2_code": source_values["e2"] == E2_CODE_HASH,
        "frozen_e2_config": source_values["e2_config"] == E2_CONFIG_HASH,
        "memory_guard": memory_guard,
        "all_parameters_trainable_once": bool(coverage.get("all_trainable", False)),
        "no_duplicate_parameters": coverage.get("duplicate_parameter_count") == 0,
        "loss_finite_one_combined_forward_backward": bool(combined.get("passed", False)),
    }
    payload = {
        "protocol": PROTOCOL,
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "e8_source_hash": e8_source_hash(),
        "e7_dependency_source_hash": e7_dependency_hash(),
        "e2_source_hashes": source_values,
        "device": device_name,
        "gpu_snapshot": snapshot,
        "memory_profile": {"free_mib_before": free_mib, "guard_mib": GPU_GUARD_MIB},
        "checks": checks,
        "combined_loss_check": combined,
    }
    _write_json(ORCH_ROOT / "preflight.json", payload)
    return payload


def _annotate_artifact(
    artifact_dir: Path,
    *,
    task_id: str,
    device_name: str,
    source_values: Mapping[str, str],
    captured_optimizer: torch.optim.Optimizer,
    coverage: Mapping[str, Any],
    started_at: str,
) -> dict[str, Any]:
    """Add E8 identity and truthful all-group optimizer traces."""

    group_rows = e7_optim.describe_optimizer_groups(captured_optimizer)
    for row in group_rows:
        row["parameter_tensor_count"] = row.pop("parameter_count")
        row["current_lr_after_training"] = row["lr"]
    epoch_path = artifact_dir / "epoch_curves.json"
    epoch_payload = _read_json(epoch_path)
    epoch_rows = epoch_payload.get("epochs")
    if not isinstance(epoch_rows, list) or not epoch_rows:
        raise ValueError(f"E2 epoch curves missing from {artifact_dir}")
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
    epoch_payload["e8_learning_rate_logging"] = {
        "source": "reconstructed from frozen E2 scheduler step and captured E7 initial group LRs",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "group_count": len(group_rows),
    }
    _write_json(epoch_path, epoch_payload)

    optimizer_contract = {
        "name": "AdamW",
        "base_learning_rate": TASK_LRS[task_id],
        "gamma": GAMMA,
        "num_layers": NUM_LAYERS,
        "scheduler": "E2 linear warmup/decay; common multiplier across groups",
        "groups": group_rows,
        "coverage": dict(coverage),
    }
    fields: dict[str, Any] = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "base_protocol": "E2-training-ablation-v1",
        "e8_source_hash": e8_source_hash(),
        "e7_dependency_source_hash": e7_dependency_hash(),
        "e2_dependency_source_hash": source_values["e2_source"],
        "e2_code_hash": source_values["e2"],
        "e2_config_hash": source_values["e2_config"],
        "e1_dependency_source_hash": source_values["e1_dependency"],
        "e8_condition": "original_student_nc_plus_e7_llrd",
        "e2_mode": "nc",
        "consistency_lambda": NC_LAMBDAS[task_id],
        "consistency_temperature": 1.0,
        "noise_contract": {
            "source": "frozen goibert_e2.noise",
            "seed": SEED,
            "epoch_dependent": True,
            "stable_id_dependent": True,
            "independent_dropout": True,
            "teacher_inference": False,
        },
        "optimizer_variant": "layerwise_lr_decay",
        "layerwise_lr_decay": {
            "gamma": GAMMA,
            "num_layers": NUM_LAYERS,
            "embedding_exponent": NUM_LAYERS,
            "block_exponent": "9-layer_index",
            "head_exponent": 0,
        },
        "optimizer_group_contract": optimizer_contract,
        "learning_rate_logging": {
            "epoch_curve_field": "learning_rates_by_group",
            "group_count": len(group_rows),
            "scalar_learning_rate_field": "learning_rate",
            "scalar_field_note": "retained frozen E2 first-group field; use learning_rates_by_group for all E8 groups",
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
    source_values = _source_hashes()
    e8_hash = e8_source_hash()
    e7_hash = e7_dependency_hash()
    started_at = _now()
    status_started = {
        "protocol": PROTOCOL,
        "task_id": task_id,
        "status": "running",
        "pid": os.getpid(),
        "started_at": started_at,
        "device": device_name,
        "e8_source_hash": e8_hash,
        "e7_dependency_source_hash": e7_hash,
        "e2_source_hashes": source_values,
        "output_dir": str(output_dir),
    }
    _worker_status(task_id, status_started)
    captured: dict[str, Any] = {}
    original_optimizer_factory = e2_train.make_optimizer

    def e8_make_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
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

    e2_train.make_optimizer = e8_make_optimizer  # type: ignore[assignment]
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
            stage="stageE8_nc_llrd",
            method="nc_plus_llrd",
            coefficient=NC_LAMBDAS[task_id],
            revision=REVISION,
            expected_e2_code_hash=E2_CODE_HASH,
            expected_e2_config_hash=E2_CONFIG_HASH,
            expected_orchestrator_hash=source_values["orchestrator"],
        )
    except Exception as exc:
        _worker_status(
            task_id,
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
        e2_train.make_optimizer = original_optimizer_factory  # type: ignore[assignment]

    optimizer = captured.get("optimizer")
    coverage = captured.get("coverage")
    if optimizer is None or coverage is None:
        raise RuntimeError("E8 optimizer adapter did not capture its optimizer/coverage contract")
    artifact_dir = Path(str(result["artifact_dir"]))
    annotation = _annotate_artifact(
        artifact_dir,
        task_id=task_id,
        device_name=device_name,
        source_values=source_values,
        captured_optimizer=optimizer,
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
    _worker_status(task_id, completed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return result


def _artifact_candidates(task_id: str) -> list[Path]:
    parent = ARTIFACT_ROOT / task_id
    if not parent.is_dir():
        return []
    return sorted(path for path in parent.glob(f"seed_{SEED}*") if path.is_dir())


def _completed_artifact(task_id: str) -> Path | None:
    for path in _artifact_candidates(task_id):
        status_path = path / "status.json"
        metadata_path = path / "metadata.json"
        checkpoint_path = path / "best_checkpoint.pt"
        if not (status_path.is_file() and metadata_path.is_file() and checkpoint_path.is_file()):
            continue
        try:
            status = _read_json(status_path)
            metadata = _read_json(metadata_path)
        except Exception:
            continue
        if status.get("status") == "completed" and metadata.get("protocol") == PROTOCOL:
            return path
    return None


def _assert_no_duplicate_tasks() -> dict[str, Path]:
    completed: dict[str, Path] = {}
    for task_id in TASKS:
        worker_path = WORKER_ROOT / f"{task_id}.json"
        if worker_path.is_file():
            worker = _read_json(worker_path)
            if worker.get("status") == "running" and _pid_alive(worker.get("pid")):
                raise RuntimeError(
                    f"E8 task {task_id} already has an active worker PID {worker.get('pid')}"
                )
        artifact = _completed_artifact(task_id)
        if artifact is not None:
            completed[task_id] = artifact
    return completed


def _summary_from_artifacts(artifacts: Mapping[str, Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    diagnostics = _read_json(ROOT / "artifacts" / "diagnostics" / "teacher_gap.json")
    diagnostic_results = diagnostics["results"]
    for task_id in TASKS:
        artifact_dir = Path(artifacts[task_id])
        predictions = _read_json(artifact_dir / "validation_predictions.json").get("rows")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError(f"Missing validation prediction rows for {task_id}")
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
        valid_split = e1_train.split_field(bundle, "validation")
        expected_ids = [str(value) for value in e1_train.split_ids(valid_split)]
        actual_ids = [str(row["id"]) for row in predictions]
        if actual_ids != expected_ids:
            raise ValueError(f"E8 prediction IDs do not match validation order for {task_id}")
        recomputed = e1_train.evaluate_predictions(
            task_id,
            bundle,
            [row["gold"] for row in predictions],
            [row["prediction"] for row in predictions],
            split="validation",
        )
        metadata = _read_json(artifact_dir / "metadata.json")
        best_metrics = _read_json(artifact_dir / "best_validation_metrics.json")
        e1_metrics = diagnostic_results[task_id]["e1"]["seed_metrics"]["42"]
        e2_metrics = diagnostic_results[task_id]["e2"]["nc"]["seed_metrics"]["42"]
        e7_metadata = _read_json(ROOT / "artifacts" / "e7" / task_id / "seed_42" / "metadata.json")
        e7_f1 = float(e7_metadata["best_selection_f1"])
        e8_f1 = float(recomputed["selection_f1"])
        better_component = max(float(e2_metrics["selection_f1"]), e7_f1)
        rows.append(
            {
                "task_id": task_id,
                "artifact_dir": str(artifact_dir),
                "seed": SEED,
                "learning_rate": TASK_LRS[task_id],
                "consistency_lambda": NC_LAMBDAS[task_id],
                "best_epoch": metadata.get("best_epoch"),
                "selection_f1_e1": float(e1_metrics["selection_f1"]),
                "selection_f1_e2_nc": float(e2_metrics["selection_f1"]),
                "selection_f1_e7_llrd": e7_f1,
                "selection_f1_e8_saved": float(metadata["best_selection_f1"]),
                "selection_f1_e8_recomputed": e8_f1,
                "delta_e8_minus_e1": e8_f1 - float(e1_metrics["selection_f1"]),
                "delta_e8_minus_e2_nc": e8_f1 - float(e2_metrics["selection_f1"]),
                "delta_e8_minus_e7_llrd": e8_f1 - e7_f1,
                "delta_e8_minus_better_component": e8_f1 - better_component,
                "beats_better_component": e8_f1 > better_component,
                "weak_signal_under_0p3pp": (e8_f1 - better_component) < 0.003,
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "peak_gpu_memory_allocated_bytes": metadata.get("peak_gpu_memory_allocated_bytes"),
                "peak_gpu_memory_reserved_bytes": metadata.get("peak_gpu_memory_reserved_bytes"),
                "prediction_count": len(predictions),
                "metrics_e8": recomputed,
                "best_metrics_file": best_metrics,
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "status": "completed",
        "created_at": _now(),
        "seed": SEED,
        "tasks": list(TASKS),
        "e8_source_hash": e8_source_hash(),
        "e7_dependency_source_hash": e7_dependency_hash(),
        "e2_dependency_source_hash": e2_train.source_hashes(ROOT)["e2_source"],
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "metric_recomputation": "one pass from saved validation_predictions.json; test excluded",
        "baselines": {
            "e1": "artifacts/diagnostics/teacher_gap.json results[task].e1.seed_metrics['42']",
            "e2_nc": "artifacts/diagnostics/teacher_gap.json results[task].e2.nc.seed_metrics['42']",
            "e7_llrd": "artifacts/e7/<task>/seed_42/metadata.json best_selection_f1",
        },
        "rows": rows,
    }
    _write_json(ARTIFACT_ROOT / "summary.json", payload)
    fieldnames = [
        "task_id", "seed", "learning_rate", "consistency_lambda", "best_epoch",
        "selection_f1_e1", "selection_f1_e2_nc", "selection_f1_e7_llrd",
        "selection_f1_e8_recomputed", "delta_e8_minus_e1", "delta_e8_minus_e2_nc",
        "delta_e8_minus_e7_llrd", "delta_e8_minus_better_component",
        "beats_better_component", "weak_signal_under_0p3pp", "elapsed_seconds",
        "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes", "prediction_count",
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
        "# E8 Result: Noise Consistency + Layer-wise Learning-rate Decay",
        "",
        "Five original GoiTurn-S-GD students were trained once each with seed 42, 30 epochs, effective and micro batch size 32, max length 256, BF16, and SDPA. E2 noise consistency used the fixed task lambdas; E7 AdamW layer-wise decay used gamma 0.8.",
        "",
        "Validation selection F1 is recomputed from the saved prediction rows. The test split was never loaded.",
        "",
        "| Task | E1 | E2 NC | E7 LLRD | E8 NC+LLRD | Best epoch | Δ E8 vs better component |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['task_id']} | {row['selection_f1_e1']:.4f} | {row['selection_f1_e2_nc']:.4f} | {row['selection_f1_e7_llrd']:.4f} | {row['selection_f1_e8_recomputed']:.4f} | {row['best_epoch']} | {row['delta_e8_minus_better_component']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "The combination adds value where its E8 score exceeds both fixed components. Differences below 0.3 percentage points are marked as weak signals; a negative delta indicates over-regularization relative to the better component for that task.",
            "",
            f"Source hashes: E8 `{summary['e8_source_hash']}`, E7 dependency `{summary['e7_dependency_source_hash']}`, E2 dependency `{summary['e2_dependency_source_hash']}`, E1 `{summary['e1_dependency_source_hash']}`.",
            "",
            "No coefficient retuning, multi-seed confirmation, extra tests, architecture changes, or encoder probes were run.",
            "",
        ]
    )
    path = ROOT / "docs" / "E8_RESULT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
    source_values = _source_hashes()
    e8_hash = e8_source_hash()
    e7_hash = e7_dependency_hash()
    completed = _assert_no_duplicate_tasks()
    if len(completed) == len(TASKS):
        return _summary_from_artifacts(completed)
    state_path = ORCH_ROOT / "state.json"
    old_pid = _existing_state_pid(state_path)
    if old_pid is not None and old_pid != os.getpid() and _pid_alive(old_pid):
        raise RuntimeError(f"E8 controller already active with PID {old_pid}")
    devices, snapshot = _eligible_devices()
    preflight = run_preflight(devices[0])
    if preflight.get("status") != "passed":
        raise RuntimeError("E8 preflight did not pass; refusing to launch full trials")
    queue = [task_id for task_id in TASKS if task_id not in completed]
    free_devices = list(devices)
    running: dict[str, tuple[subprocess.Popen[str], str, Any]] = {}
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
        "status": "running",
        "controller_pid": os.getpid(),
        "started_at": _now(),
        "e8_source_hash": e8_hash,
        "e7_dependency_source_hash": e7_hash,
        "e2_source_hashes": source_values,
        "preferred_gpu_indices": list(PREFERRED_GPU_INDICES),
        "eligible_devices": devices,
        "gpu_snapshot_at_launch": snapshot,
        "tasks": list(TASKS),
        "jobs": jobs,
        "recovered_existing_completed_tasks": sorted(completed),
        "preflight": str(ORCH_ROOT / "preflight.json"),
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
            "--task-id", task_id,
            "--device", device_name,
            "--output-dir", str(output_dir),
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
        running[task_id] = (process, device_name, None)
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
        controller_log.write(json.dumps({"event": "launch", "task_id": task_id, "pid": process.pid, "device": device_name}) + "\n")
        _write_json(state_path, controller_state)

    try:
        while queue or running:
            while queue and free_devices:
                launch(queue.pop(0), free_devices.pop(0))
            finished: list[str] = []
            for task_id, (process, device_name, _) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                job = jobs[task_id]
                job.update({"status": "completed" if return_code == 0 else "failed", "returncode": return_code, "finished_at": _now()})
                if return_code == 0:
                    worker_path = WORKER_ROOT / f"{task_id}.json"
                    if not worker_path.is_file():
                        raise RuntimeError(f"Worker {task_id} exited without status artifact")
                    worker = _read_json(worker_path)
                    if worker.get("status") != "completed":
                        raise RuntimeError(f"Worker {task_id} did not report completed status")
                    job["artifact_dir"] = worker.get("artifact_dir")
                    completed[task_id] = Path(str(worker["artifact_dir"]))
                finished.append(task_id)
                free_devices.append(device_name)
                controller_log.write(json.dumps({"event": "finish", "task_id": task_id, "returncode": return_code}) + "\n")
            for task_id in finished:
                running.pop(task_id)
            if finished:
                _write_json(state_path, controller_state)
            failed = [task_id for task_id, job in jobs.items() if job.get("status") == "failed"]
            if failed:
                controller_state.update({"status": "failed", "finished_at": _now(), "failed_tasks": failed})
                _write_json(state_path, controller_state)
                raise RuntimeError(f"E8 workers failed: {failed}")
            if running:
                time.sleep(poll_seconds)
    finally:
        controller_log.close()

    summary = _summary_from_artifacts(completed)
    controller_state.update({"status": "completed", "finished_at": _now(), "summary": str(ARTIFACT_ROOT / "summary.json")})
    _write_json(state_path, controller_state)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed E8 NC plus E7 layer-wise LR-decay trials")
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
