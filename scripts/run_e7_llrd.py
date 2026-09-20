"""Run the minimal E7 layer-wise-learning-rate-decay screening.

E1's model, data path, loss, and training loop are intentionally reused.  A
process-local optimizer adapter changes only the AdamW parameter groups.  The
validation runner never materialises the test split.  The controller assigns
one worker to each actually eligible preferred GPU and records durable state
and per-worker logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

# These must be set before importing torch through goibert.train, including in
# every child process launched by the controller.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e7"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
PROTOCOL_PATH = ROOT / "configs" / "e7" / "protocol.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from goibert import modeling as e1_modeling  # noqa: E402
from goibert import train as e1_train  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402


PROTOCOL = "E7-layerwise-learning-rate-decay-screening-v1"
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
SEED = 42
EPOCHS = 30
MICRO_BATCH_SIZE = 32
GPU_GUARD_MIB = 5 * 1024
PREFERRED_GPU_INDICES = ("2", "1", "3")
E1_SOURCE_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"
GAMMA = 0.8
NUM_LAYERS = 10
WARMUP_FRACTION = 0.1
PRIOR_METHODS: dict[str, dict[str, Any]] = {
    "vsfc_sentiment": {"method": "E2_NC", "selection_f1": 0.847722},
    "vsfc_topic": {"method": "E2_KD", "selection_f1": 0.807842},
    "visfd": {"method": "E2_NC", "selection_f1": 0.810884},
    "absa_restaurant": {"method": "E3_polarity_loss", "selection_f1": 0.701299},
    "absa_hotel": {"method": "E2_NC", "selection_f1": 0.735776},
}
# E5's tiny Sentiment signal is retained as a contextual comparison only.
E5_SENTIMENT_CANDIDATE = 0.848285


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


def e7_source_hash() -> str:
    """Hash only E7 code/config; the E1 dependency hash is recorded separately."""

    paths = [
        *sorted((ROOT / "src" / "goibert_e7").rglob("*.py")),
        PROTOCOL_PATH,
        Path(__file__).resolve(),
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.resolve().relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


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


def _assert_frozen_dependency() -> None:
    actual = e1_train.source_hash(ROOT)
    if actual != E1_SOURCE_HASH:
        raise RuntimeError(f"Frozen E1 source drift: expected {E1_SOURCE_HASH}, found {actual}")


def _common_schedule_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return float(step / max(1, warmup_steps))
    return float(max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)))


def _scheduler_ratio_check(base_lr: float) -> dict[str, Any]:
    """Check the common E1 scheduler keeps all E7 LR ratios intact."""

    torch.manual_seed(777)
    model, _ = e1_modeling.build_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    optimizer = e7_optim.build_layerwise_optimizer(model, base_lr, gamma=GAMMA)
    scheduler = get_linear_schedule_with_warmup(optimizer, 1, 4)
    # Step once after the one-step warmup.  At this point every group has a
    # positive LR and ratios can be compared without a zero denominator.
    optimizer.step()
    scheduler.step()
    current = [float(group["lr"]) for group in optimizer.param_groups]
    initial = [float(group["initial_lr"]) for group in optimizer.param_groups]
    initial_ratios = [value / initial[-1] for value in initial]
    current_ratios = [value / current[-1] for value in current]
    ratio_ok = all(
        math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-14)
        for left, right in zip(initial_ratios, current_ratios)
    )
    return {
        "passed": ratio_ok,
        "step": 1,
        "initial_ratios_to_head": initial_ratios,
        "current_ratios_to_head": current_ratios,
        "scheduler_last_lr": [float(value) for value in scheduler.get_last_lr()],
    }


def _gamma_one_equivalence(base_lr: float) -> dict[str, Any]:
    """Compare gamma=1 parameterwise settings with frozen E1 AdamW."""

    torch.manual_seed(778)
    model, _ = e1_modeling.build_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    e1_optimizer = e1_train.make_optimizer(model, base_lr)
    e7_optimizer = e7_optim.build_layerwise_optimizer(model, base_lr, gamma=1.0)
    e1_by_id: dict[int, tuple[float, float]] = {}
    for group in e1_optimizer.param_groups:
        for parameter in group["params"]:
            e1_by_id[id(parameter)] = (float(group["lr"]), float(group["weight_decay"]))
    e7_by_id: dict[int, tuple[float, float]] = {}
    for group in e7_optimizer.param_groups:
        for parameter in group["params"]:
            e7_by_id[id(parameter)] = (float(group["lr"]), float(group["weight_decay"]))
    same_parameter_ids = set(e1_by_id) == set(e7_by_id)
    same_settings = same_parameter_ids and all(e1_by_id[key] == e7_by_id[key] for key in e1_by_id)
    return {
        "passed": same_settings,
        "same_parameter_ids": same_parameter_ids,
        "same_parameterwise_lr_and_weight_decay": same_settings,
        "e1_group_count": len(e1_optimizer.param_groups),
        "e7_gamma1_group_count": len(e7_optimizer.param_groups),
    }


def run_preflight(device_name: str) -> dict[str, Any]:
    """Run bounded optimizer coverage, schedule, and finite GPU checks."""

    started = time.time()
    _assert_frozen_dependency()
    torch.manual_seed(777)
    model, model_spec = e1_modeling.build_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    optimizer = e7_optim.build_layerwise_optimizer(model, 1e-5, gamma=GAMMA)
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
    schedule = _scheduler_ratio_check(1e-5)
    gamma_one = _gamma_one_equivalence(1e-5)

    snapshot = _gpu_snapshot()
    physical_index = device_name.split(":", 1)[1] if ":" in device_name else device_name
    free_mib = snapshot.get(physical_index, {}).get("free_mib", 0)
    memory_guard = free_mib >= GPU_GUARD_MIB
    finite_forward_backward = False
    peak_allocated = None
    peak_reserved = None
    if memory_guard:
        device = torch.device(device_name)
        model.to(device)
        model.train()
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
        )
        encoded = tokenizer(
            ["món ăn ngon", "phục vụ nhanh"],
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        optimizer = e7_optim.build_layerwise_optimizer(model, 1e-5, gamma=GAMMA)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(encoded["input_ids"], encoded["attention_mask"])
            loss = logits.float().square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        finite_forward_backward = bool(torch.isfinite(logits).all().item() and torch.isfinite(loss).item())
        finite_forward_backward = finite_forward_backward and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        del logits, loss, encoded, optimizer, model
        torch.cuda.empty_cache()

    checks = {
        "frozen_e1_dependency": e1_train.source_hash(ROOT) == E1_SOURCE_HASH,
        "all_parameters_trainable": coverage["all_trainable"],
        "no_duplicate_parameters": coverage["duplicate_parameter_count"] == 0,
        "expected_group_endpoints": coverage["endpoint_scales"]["embeddings"] == GAMMA**NUM_LAYERS
        and coverage["endpoint_scales"]["head"] == 1.0,
        "scheduler_common_ratio": schedule["passed"],
        "gamma_one_e1_equivalence": gamma_one["passed"],
        "memory_guard": memory_guard,
        "finite_forward_backward": finite_forward_backward,
    }
    payload = {
        "protocol": PROTOCOL,
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "e7_source_hash": e7_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "device": device_name,
        "model_spec": model_spec.to_dict(),
        "coverage": coverage,
        "scheduler_ratio_check": schedule,
        "gamma_one_equivalence": gamma_one,
        "memory_profile": {
            "free_mib_before": free_mib,
            "guard_mib": GPU_GUARD_MIB,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "batch_size": 2,
            "max_length": 32,
            "precision": "bf16",
        },
        "checks": checks,
    }
    _write_json(ORCH_ROOT / "preflight.json", payload)
    return payload


def _worker_status(task_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_ROOT / f"{task_id}.json", payload)


def _schedule_factor_for_epoch(epoch_row: Mapping[str, Any], *, warmup_steps: int, total_steps: int) -> float:
    step = int(epoch_row.get("global_step", 0))
    return _common_schedule_factor(step, warmup_steps=warmup_steps, total_steps=total_steps)


def _annotate_artifact(
    artifact_dir: Path,
    *,
    e7_hash: str,
    e1_dependency_hash: str,
    base_lr: float,
    captured_optimizer: torch.optim.Optimizer,
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Add truthful E7 metadata and all-group LR traces to E1 artifacts."""

    group_rows = e7_optim.describe_optimizer_groups(captured_optimizer)
    # ``parameter_count`` in this metadata counts tensors, while the model
    # spec contains scalar parameter counts.  Keep the distinction explicit.
    for row in group_rows:
        row["parameter_tensor_count"] = row.pop("parameter_count")
        row["current_lr_after_training"] = row["lr"]
    metadata_path = artifact_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    total_steps = 0
    epoch_path = artifact_dir / "epoch_curves.json"
    epoch_payload = _read_json(epoch_path)
    epoch_rows = epoch_payload.get("epochs")
    if not isinstance(epoch_rows, list) or not epoch_rows:
        raise ValueError(f"E1 epoch curves missing from {artifact_dir}")
    total_steps = int(epoch_rows[-1].get("global_step", 0))
    warmup_steps = int(metadata.get("warmup_steps", round(total_steps * WARMUP_FRACTION)))

    epoch_group_logs: list[dict[str, Any]] = []
    for row in epoch_rows:
        factor = _schedule_factor_for_epoch(row, warmup_steps=warmup_steps, total_steps=total_steps)
        all_rates = [
            {
                "name": group["name"],
                "lr": float(group["initial_lr"] * factor),
                "initial_lr": float(group["initial_lr"]),
                "lr_scale": float(group["lr_scale"]),
            }
            for group in group_rows
        ]
        row["scheduler_factor"] = factor
        row["learning_rates_by_group"] = all_rates
        epoch_group_logs.append({"epoch": row.get("epoch"), "global_step": row.get("global_step"), "groups": all_rates})
    epoch_payload["epochs"] = epoch_rows
    epoch_payload["e7_learning_rate_logging"] = {
        "source": "reconstructed from frozen E1 scheduler step and each group's initial_lr",
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "group_count": len(group_rows),
    }
    _write_json(epoch_path, epoch_payload)

    optimizer_contract = {
        "name": "AdamW",
        "base_learning_rate": base_lr,
        "gamma": GAMMA,
        "num_layers": NUM_LAYERS,
        "scheduler": "E1 linear warmup/decay; common multiplier across groups",
        "groups": group_rows,
        "coverage": dict(coverage),
    }
    fields: dict[str, Any] = {
        "protocol": PROTOCOL,
        "architecture_variant": "goiturn-s-gd-e7-layerwise-learning-rate-decay",
        "e7_source_hash": e7_hash,
        "e1_dependency_source_hash": e1_dependency_hash,
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
            "scalar_field_note": "retained frozen E1 first-group field; use learning_rates_by_group for all E7 groups",
        },
    }
    metadata.update(fields)
    _write_json(metadata_path, metadata)
    resolved_path = artifact_dir / "resolved_config.json"
    resolved = _read_json(resolved_path)
    resolved.update(fields)
    _write_json(resolved_path, resolved)

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
    _assert_frozen_dependency()
    e7_hash = e7_source_hash()
    dependency_hash = e1_train.source_hash(ROOT)
    started = _now()
    status_started = {
        "protocol": PROTOCOL,
        "task_id": task_id,
        "status": "running",
        "started_at": started,
        "device": device_name,
        "e7_source_hash": e7_hash,
        "e1_dependency_source_hash": dependency_hash,
        "output_dir": str(output_dir),
    }
    _worker_status(task_id, status_started)
    captured: dict[str, Any] = {}
    original_optimizer_factory = e1_train.make_optimizer
    original_hash_function = e1_train.source_hash

    def e7_make_optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
        optimizer = e7_optim.build_layerwise_optimizer(model, learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS)
        captured["optimizer"] = optimizer
        captured["coverage"] = e7_optim.validate_optimizer_groups(
            model,
            optimizer,
            base_lr=learning_rate,
            gamma=GAMMA,
            num_layers=NUM_LAYERS,
        )
        return optimizer

    # E1 train_trial resolves these module attributes at call time.  The
    # narrow adapter restores both attributes even when a worker fails.
    e1_train.make_optimizer = e7_make_optimizer  # type: ignore[assignment]
    e1_train.source_hash = lambda _repo_root: e7_hash  # type: ignore[assignment]
    try:
        result = e1_train.train_trial(
            task_id=task_id,
            learning_rate=TASK_LRS[task_id],
            seed=SEED,
            repo_root=ROOT,
            output_dir=output_dir,
            device_name=device_name,
            precision="bf16",
            attention_implementation="sdpa",
            epochs=EPOCHS,
            effective_batch_size=MICRO_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            trial_stage="screening",
            smoke=False,
            expected_source_hash=e7_hash,
        )
    except Exception as exc:
        _worker_status(
            task_id,
            {
                **status_started,
                "status": "failed",
                "finished_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        e1_train.make_optimizer = original_optimizer_factory  # type: ignore[assignment]
        e1_train.source_hash = original_hash_function  # type: ignore[assignment]

    optimizer = captured.get("optimizer")
    coverage = captured.get("coverage")
    if optimizer is None or coverage is None:
        raise RuntimeError("E7 optimizer adapter did not capture its optimizer/coverage contract")
    artifact_dir = Path(str(result["artifact_dir"]))
    annotation = _annotate_artifact(
        artifact_dir,
        e7_hash=e7_hash,
        e1_dependency_hash=dependency_hash,
        base_lr=TASK_LRS[task_id],
        captured_optimizer=optimizer,
        coverage=coverage,
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


def _e1_seed42_artifact(task_id: str) -> Path:
    lr_name = "1e-05" if TASK_LRS[task_id] == 1e-5 else "1e-04"
    matches = sorted(
        (ROOT / "artifacts" / "e1" / ".orchestrator" / "attempts").glob(
            f"{task_id}__screening__{lr_name}__seed42__attempt_*/metadata.json"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one canonical E1 seed42 artifact for {task_id}; found {matches}")
    return matches[0].parent


def _independent_summary(job_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        result = job_results[task_id]
        artifact_dir = Path(str(result["artifact_dir"]))
        prediction_payload = _read_json(artifact_dir / "validation_predictions.json")
        prediction_rows = prediction_payload.get("rows")
        if not isinstance(prediction_rows, list) or not prediction_rows:
            raise ValueError(f"Missing E7 validation rows for {task_id}")
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
        valid_split = e1_train.split_field(bundle, "validation")
        expected_ids = list(e1_train.split_ids(valid_split))
        actual_ids = [row["id"] for row in prediction_rows]
        if [str(value) for value in actual_ids] != [str(value) for value in expected_ids]:
            raise ValueError(f"E7 prediction IDs do not match validation order for {task_id}")
        gold = [row["gold"] for row in prediction_rows]
        predicted = [row["prediction"] for row in prediction_rows]
        recomputed = e1_train.evaluate_predictions(task_id, bundle, gold, predicted, split="validation")
        e1_dir = _e1_seed42_artifact(task_id)
        e1_metrics = _read_json(e1_dir / "best_validation_metrics.json")
        e7_f1 = float(recomputed["selection_f1"])
        e1_f1 = float(e1_metrics["selection_f1"])
        prior = PRIOR_METHODS[task_id]
        rows.append(
            {
                "task_id": task_id,
                "artifact_dir": str(artifact_dir),
                "e1_artifact_dir": str(e1_dir),
                "seed": SEED,
                "learning_rate": TASK_LRS[task_id],
                "best_epoch": result.get("best_epoch"),
                "selection_f1_e1": e1_f1,
                "selection_f1_e7_saved": result.get("best_selection_f1"),
                "selection_f1_e7_recomputed": e7_f1,
                "delta_e7_minus_e1": e7_f1 - e1_f1,
                "prior_method": prior["method"],
                "selection_f1_prior_method": prior["selection_f1"],
                "delta_e7_minus_prior_method": e7_f1 - float(prior["selection_f1"]),
                "metrics_e1": e1_metrics,
                "metrics_e7": recomputed,
                "prediction_count": len(prediction_rows),
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "status": "completed",
        "created_at": _now(),
        "e7_source_hash": e7_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "metric_recomputation": "one pass from saved validation_predictions.json; test excluded",
        "prior_method_values_are_contextual_single_screening_mocks": True,
        "e5_sentiment_candidate_selection_f1": E5_SENTIMENT_CANDIDATE,
        "rows": rows,
    }
    _write_json(ARTIFACT_ROOT / "summary.json", payload)
    with (ARTIFACT_ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "task_id",
            "seed",
            "learning_rate",
            "best_epoch",
            "selection_f1_e1",
            "selection_f1_e7_recomputed",
            "delta_e7_minus_e1",
            "prior_method",
            "selection_f1_prior_method",
            "delta_e7_minus_prior_method",
            "prediction_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    return payload


def run_all(*, poll_seconds: float = 15.0) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    _assert_frozen_dependency()
    e7_hash = e7_source_hash()
    devices, snapshot = _eligible_devices()
    preflight = run_preflight(devices[0])
    if preflight.get("status") != "passed":
        raise RuntimeError("E7 preflight did not pass; refusing to launch full trials")

    state_path = ORCH_ROOT / "state.json"
    if state_path.is_file():
        old = _read_json(state_path)
        if old.get("status") in {"running", "completed"}:
            raise RuntimeError("E7 controller already launched; inspect its state before rerunning")
    for task_id in TASKS:
        output_dir = ARTIFACT_ROOT / task_id / f"seed_{SEED}"
        if output_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing E7 output directory: {output_dir}")

    queue = list(TASKS)
    free_devices = list(devices)
    running: dict[str, tuple[subprocess.Popen[str], str]] = {}
    jobs: dict[str, dict[str, Any]] = {}
    completed: dict[str, Mapping[str, Any]] = {}
    controller_state: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "running",
        "started_at": _now(),
        "e7_source_hash": e7_hash,
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "preferred_gpu_indices": list(PREFERRED_GPU_INDICES),
        "eligible_devices": devices,
        "gpu_snapshot_at_launch": snapshot,
        "tasks": list(TASKS),
        "jobs": jobs,
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
        controller_log.write(json.dumps({"event": "launch", "task_id": task_id, "pid": process.pid, "device": device_name}) + "\n")
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
                job.update(
                    {
                        "status": "completed" if return_code == 0 else "failed",
                        "returncode": return_code,
                        "finished_at": _now(),
                    }
                )
                if return_code == 0:
                    status_path = WORKER_ROOT / f"{task_id}.json"
                    if not status_path.is_file():
                        raise RuntimeError(f"Worker {task_id} exited without status artifact")
                    worker_status = _read_json(status_path)
                    if worker_status.get("status") != "completed":
                        raise RuntimeError(f"Worker {task_id} did not report completed status")
                    completed[task_id] = worker_status
                finished.append(task_id)
                free_devices.append(device_name)
                controller_log.write(json.dumps({"event": "finish", "task_id": task_id, "returncode": return_code}) + "\n")
            for task_id in finished:
                running.pop(task_id)
            if finished:
                _write_json(state_path, controller_state)
            failed = [task for task, job in jobs.items() if job.get("status") == "failed"]
            if failed:
                controller_state.update({"status": "failed", "finished_at": _now(), "failed_tasks": failed})
                _write_json(state_path, controller_state)
                raise RuntimeError(f"E7 workers failed: {failed}")
            if running:
                time.sleep(poll_seconds)
    finally:
        controller_log.close()

    summary = _independent_summary(completed)
    controller_state.update(
        {"status": "completed", "finished_at": _now(), "summary": str(ARTIFACT_ROOT / "summary.json")}
    )
    _write_json(state_path, controller_state)
    return {"state": controller_state, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E7 layer-wise learning-rate-decay screening")
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
