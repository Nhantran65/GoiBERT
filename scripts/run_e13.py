"""Run E13 centered/uncentered pair-value residual trials and locked tests.

E13 composes the frozen E2 NC training loop with the E7 LLRD optimizer.  The
only model change is the E13 student factory imported from
``goibert_e13.modeling``.  Validation remains train-only until the explicit
checkpoint lock; the test command evaluates every locked checkpoint.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e13"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
CONFIG_PATH = ROOT / "configs" / "e13" / "protocol.json"
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 32
MAX_LENGTH = 256
GAMMA = 0.8
NUM_LAYERS = 10
GPU_GUARD_MIB = 5 * 1024
PREFERRED_GPU_INDICES = ("0", "1", "2", "3")
TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
VARIANTS = ("centered", "uncentered")
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
PROTOCOL = "E13-centered-pair-value-residual-v1"
REVISION = "e13-20260912-r1"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402
from goibert_e13.modeling import (  # noqa: E402
    E13Classifier,
    E13RobertaAttention,
    INSERTION_LAYER_INDEX,
    PAIR_CHUNK_SIZE,
    PAIR_RANK,
    VARIANTS as MODEL_VARIANTS,
    build_e13_student,
)


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
        digest.update(str(path.resolve().relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def e13_source_hash() -> str:
    paths = [
        *sorted((ROOT / "src" / "goibert_e13").rglob("*.py")),
        CONFIG_PATH,
        Path(__file__).resolve(),
    ]
    return _hash_files(paths)


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


def _gpu_snapshot() -> dict[str, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    snapshot: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3:
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
        raise RuntimeError(f"No E13 GPU passed {GPU_GUARD_MIB} MiB guard: {snapshot}")
    return devices, snapshot


def _task_contract(task_id: str, *, include_test: bool = False) -> dict[str, Any]:
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=include_test)
    train = e1_train.split_field(bundle, "train")
    valid = e1_train.split_field(bundle, "validation")
    test = e1_train.split_field(bundle, "test") if include_test else None
    if train is None or valid is None:
        raise ValueError(f"Task {task_id} has no train/validation split")
    return {
        "bundle": bundle,
        "train": train,
        "validation": valid,
        "test": test,
        "num_aspects": e1_train.task_num_aspects(bundle),
        "aspect_class_counts": e1_train.task_aspect_class_counts(bundle),
        "num_classes": e1_train.task_num_classes(bundle),
    }


def _source_values() -> dict[str, str]:
    values = e2_train.source_hashes(ROOT)
    values["e13"] = e13_source_hash()
    return values


def _worker_status(key: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_ROOT / f"{key}.json", dict(payload))


def _artifact_dir(variant: str, task_id: str) -> Path:
    return ARTIFACT_ROOT / variant / task_id / f"seed_{SEED}"


def _trial_key(variant: str, task_id: str) -> str:
    return f"{variant}__{task_id}"


def _completed_artifact(variant: str, task_id: str) -> Path | None:
    artifact = _artifact_dir(variant, task_id)
    required = (artifact / "metadata.json", artifact / "best_checkpoint.pt", artifact / "validation_predictions.json")
    if not all(path.is_file() for path in required):
        return None
    try:
        metadata = _read_json(artifact / "metadata.json")
        status = _read_json(artifact / "status.json")
    except Exception:
        return None
    if metadata.get("protocol") not in {"E2-training-ablation-v1", PROTOCOL} or status.get("status") != "completed":
        return None
    if metadata.get("e13_variant") != variant:
        return None
    return artifact


def _make_factory(variant: str):
    def factory(
        backbone_dir: Path,
        *,
        num_classes: int,
        num_aspects: int = 0,
        aspect_class_counts: Sequence[int] | None = None,
        attention_implementation: str = "sdpa",
    ):
        return build_e13_student(
            backbone_dir,
            num_classes=num_classes,
            num_aspects=num_aspects,
            aspect_class_counts=aspect_class_counts,
            variant=variant,
            attention_implementation=attention_implementation,
        )

    return factory


def _annotate_trial(
    artifact: Path,
    *,
    variant: str,
    task_id: str,
    device_name: str,
    source_values: Mapping[str, str],
    optimizer: torch.optim.Optimizer,
    coverage: Mapping[str, Any],
) -> None:
    metadata_path = artifact / "metadata.json"
    resolved_path = artifact / "resolved_config.json"
    metadata = _read_json(metadata_path)
    resolved = _read_json(resolved_path)
    model_meta = dict(metadata.get("model", {}))
    metadata.update(
        {
            "protocol": PROTOCOL,
            "revision": REVISION,
            "e13_variant": variant,
            "task_id": task_id,
            "device": device_name,
            "e13_source_hash": source_values["e13"],
            "e13_insertion_layer_index": INSERTION_LAYER_INDEX,
            "e13_pair_rank": PAIR_RANK,
            "e13_pair_chunk_size": PAIR_CHUNK_SIZE,
            "e13_pair_activation": "gelu_exact",
            "e13_added_parameter_count": 73728,
            "optimizer_group_contract": e7_optim.describe_optimizer_groups(optimizer),
            "optimizer_coverage": dict(coverage),
            "metadata_polarity_overlay": {
                "vsfc_sentiment": ["negative", "neutral", "positive"],
                "vsfc_topic": ["lecturer", "training_program", "facility", "others"],
                "absa_polarity": {"0": "absent", "1": "positive", "2": "negative", "3": "neutral"},
                "visfd_others_raw4": "model_binary_pair_masked",
            },
        }
    )
    model_meta.update(
        {
            "variant": variant,
            "architecture": "original_sdpa_plus_block6_pair_value_residual",
            "added_parameter_count": 73728,
        }
    )
    metadata["model"] = model_meta
    resolved.update(metadata)
    _write_json(metadata_path, metadata)
    _write_json(resolved_path, resolved)
    checkpoint_path = artifact / "best_checkpoint.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_meta = dict(state.get("metadata", {}))
    state_meta.update({"protocol": PROTOCOL, "e13_variant": variant, "e13_source_hash": source_values["e13"]})
    state["metadata"] = state_meta
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint_path)
    metadata["checkpoint_sha256_actual"] = _sha256(checkpoint_path)
    _write_json(metadata_path, metadata)


def run_worker(variant: str, task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if variant not in VARIANTS or task_id not in TASKS:
        raise ValueError(f"Unrecognised locked trial {variant}/{task_id}")
    expected = _artifact_dir(variant, task_id)
    if output_dir.resolve() != expected.resolve():
        raise ValueError(f"Output path must be {expected}")
    key = _trial_key(variant, task_id)
    source_values = _source_values()
    started = _now()
    status = {
        "protocol": PROTOCOL,
        "variant": variant,
        "task_id": task_id,
        "status": "running",
        "pid": os.getpid(),
        "device": device_name,
        "started_at": started,
        "output_dir": str(output_dir),
        "e13_source_hash": source_values["e13"],
    }
    _worker_status(key, status)
    original_factory = e2_train.build_e2_student
    original_optimizer = e2_train.make_optimizer
    captured: dict[str, Any] = {}

    def optimizer_factory(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
        optimizer = e7_optim.build_layerwise_optimizer(model, learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS)
        coverage = e7_optim.validate_optimizer_groups(
            model,
            optimizer,
            base_lr=learning_rate,
            gamma=GAMMA,
            num_layers=NUM_LAYERS,
        )
        captured["optimizer"] = optimizer
        captured["coverage"] = coverage
        return optimizer

    e2_train.build_e2_student = _make_factory(variant)  # type: ignore[assignment]
    e2_train.make_optimizer = optimizer_factory  # type: ignore[assignment]
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
            stage="stageE13_pair_value_residual",
            method=f"{variant}_pair_value_residual_nc_llrd",
            coefficient=NC_LAMBDAS[task_id],
            revision=REVISION,
            expected_e2_code_hash=source_values["e2"],
            expected_e2_config_hash=source_values["e2_config"],
            expected_orchestrator_hash=source_values["orchestrator"],
        )
        artifact = Path(str(result["artifact_dir"]))
        _annotate_trial(
            artifact,
            variant=variant,
            task_id=task_id,
            device_name=device_name,
            source_values=source_values,
            optimizer=captured["optimizer"],
            coverage=captured["coverage"],
        )
        completed = {**status, "status": "completed", "finished_at": _now(), "artifact_dir": str(artifact), **result}
        _worker_status(key, completed)
        return result
    except Exception as exc:
        _worker_status(
            key,
            {**status, "status": "failed", "finished_at": _now(), "error": repr(exc), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        e2_train.build_e2_student = original_factory  # type: ignore[assignment]
        e2_train.make_optimizer = original_optimizer  # type: ignore[assignment]


def _preflight_model(device_name: str | None = None) -> dict[str, Any]:
    """Run architecture, masking, gradients, optimizer and parity checks."""

    started = time.time()
    source_values = _source_values()
    checks: dict[str, Any] = {}
    details: dict[str, Any] = {"source_hashes": source_values, "tasks": {}}
    torch.manual_seed(SEED)
    for task_id in TASKS:
        contract = _task_contract(task_id)
        task_detail: dict[str, Any] = {}
        models: dict[str, E13Classifier] = {}
        specs: dict[str, Any] = {}
        for variant in VARIANTS:
            torch.manual_seed(SEED)
            model, spec = build_e13_student(
                ROOT / "models" / "goiturn-s-gd",
                num_classes=contract["num_classes"],
                num_aspects=contract["num_aspects"],
                aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
                variant=variant,
            )
            models[variant] = model
            specs[variant] = spec.to_dict()
            if spec.added_parameter_count != 73728:
                raise AssertionError(f"{task_id}/{variant} added parameter count mismatch")
            if any("e13_attention" in name for name in model.state_dict()):
                raise AssertionError("E13 duplicate namespace detected")
        task_detail["specs"] = specs
        # Same seed and fork_rng replacement make the old model/head state
        # identical; U/W are identical across variants and only the formula
        # differs.
        names = set(models["centered"].state_dict())
        task_detail["state_key_count"] = len(names)
        task_detail["variant_added_parameter_counts"] = {
            v: specs[v]["added_parameter_count"] for v in VARIANTS
        }
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
        texts = list(e1_train.split_texts(contract["validation"]))[:2]
        encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        for model in models.values():
            model.eval()
        with torch.no_grad():
            centered_logits = models["centered"](encoded["input_ids"], encoded["attention_mask"])
            uncentered_logits = models["uncentered"](encoded["input_ids"], encoded["attention_mask"])
        task_detail["finite_logits"] = bool(torch.isfinite(centered_logits).all() and torch.isfinite(uncentered_logits).all())
        task_detail["input_shape"] = list(encoded["input_ids"].shape)
        details["tasks"][task_id] = task_detail
    checks["all_five_task_shapes"] = len(details["tasks"]) == 5 and all(
        details["tasks"][task]["finite_logits"] for task in TASKS
    )

    # Centered null identity and linear activation identity.
    q0 = torch.zeros(2, 3, PAIR_RANK)
    k0 = torch.randn(2, 3, PAIR_RANK)
    centered_zero = torch.nn.functional.gelu(q0 + k0, approximate="none") - torch.nn.functional.gelu(q0, approximate="none") - torch.nn.functional.gelu(k0, approximate="none")
    x = torch.randn(2, 3, PAIR_RANK)
    linear_zero = (x + k0) - x - k0
    checks["centered_null_identity"] = bool(torch.equal(centered_zero, torch.zeros_like(centered_zero)))
    checks["linear_activation_identity"] = bool(torch.allclose(linear_zero, torch.zeros_like(linear_zero), atol=1e-6, rtol=0.0))

    # Exact E8 t0 parity and BF16 routing-map smoke test on a categorical task.
    torch.manual_seed(777)
    baseline, _ = e2_train.build_e2_student(ROOT / "models" / "goiturn-s-gd", num_classes=3, attention_implementation="sdpa")
    torch.manual_seed(777)
    candidate, _ = build_e13_student(ROOT / "models" / "goiturn-s-gd", num_classes=3, variant="centered")
    ids = torch.randint(0, 100, (2, 16))
    mask = torch.ones_like(ids)
    mask[1, -4:] = 0
    baseline.eval(); candidate.eval()
    with torch.no_grad():
        reference = baseline(ids, mask)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = candidate(ids, mask)
    checks["t0_logits_finite"] = bool(torch.isfinite(actual).all())
    # Candidate was run under BF16 for the softmax smoke; compare a separate
    # FP32 pass to avoid conflating parity with dtype rounding.
    with torch.no_grad():
        actual_fp32 = candidate(ids, mask)
    checks["t0_e8_logits_exact_fp32"] = bool(torch.equal(reference, actual_fp32))
    checks["fp32_softmax_under_outer_bf16"] = bool(candidate.pair_module._attention_scores(torch.randn(2, 6, 4), torch.randn(2, 6, 4), None).dtype == torch.float32)

    # Dense/chunked forward and gradient parity for the auxiliary branch.
    probe = candidate.pair_module
    probe.eval()
    hidden = torch.randn(2, 9, 384, requires_grad=True)
    oq = probe.transpose_for_scores(probe.query(hidden))
    ok = probe.transpose_for_scores(probe.key(hidden))
    probe.chunk_size = 64
    chunk_delta = probe._pair_message(hidden, oq, ok, None)
    old_chunk = probe.chunk_size
    probe.chunk_size = 99
    dense_delta = probe._pair_message(hidden, oq, ok, None)
    probe.chunk_size = old_chunk
    dense_loss = dense_delta.square().sum(); chunk_loss = chunk_delta.square().sum()
    checks["chunked_dense_forward_parity"] = bool(torch.allclose(chunk_delta, dense_delta, atol=1e-6, rtol=1e-5))
    checks["chunked_dense_gradient_parity"] = bool(torch.isfinite(torch.autograd.grad(chunk_loss, hidden, retain_graph=True)[0]).all() and torch.isfinite(torch.autograd.grad(dense_loss, hidden)[0]).all())

    # P receives a gradient at the zero-output initialization; after a tiny
    # P update the U/W branch receives gradients too.
    probe.P.weight.grad = None; probe.U.weight.grad = None; probe.W.weight.grad = None
    first = probe._pair_message(hidden.detach(), oq.detach(), ok.detach(), None).sum()
    first.backward()
    p_step = float(probe.P.weight.grad.norm().item()) if probe.P.weight.grad is not None else 0.0
    with torch.no_grad():
        probe.P.weight.add_(0.01)
    probe.P.weight.grad = None; probe.U.weight.grad = None; probe.W.weight.grad = None
    second = probe._pair_message(hidden.detach(), oq.detach(), ok.detach(), None).sum()
    second.backward()
    uw_step = float((probe.U.weight.grad.norm() + probe.W.weight.grad.norm()).item()) if probe.U.weight.grad is not None and probe.W.weight.grad is not None else 0.0
    checks["gradient_P_step1"] = p_step > 0 and math.isfinite(p_step)
    checks["gradient_UW_step2"] = uw_step > 0 and math.isfinite(uw_step)
    details["gradient_norms"] = {"P_step1": p_step, "UW_step2": uw_step}

    optimizer = e7_optim.build_layerwise_optimizer(candidate, 1e-4, gamma=GAMMA, num_layers=NUM_LAYERS)
    coverage = e7_optim.validate_optimizer_groups(candidate, optimizer, base_lr=1e-4, gamma=GAMMA, num_layers=NUM_LAYERS)
    p_groups = [row for row in coverage["groups"] if any(name.endswith(("U.weight", "W.weight", "P.weight")) for name in row["parameter_names"])]
    checks["optimizer_unique_and_complete"] = bool(coverage["all_trainable"] and coverage["duplicate_parameter_count"] == 0)
    checks["new_params_block6_lr"] = bool(p_groups and all(math.isclose(row["lr"], 1e-4 * GAMMA**3, rel_tol=1e-12) for row in p_groups))
    details["optimizer_coverage"] = coverage

    # Strict checkpoint restore through the same E13 constructor.
    checkpoint_path = ORCH_ROOT / "preflight_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": candidate.state_dict()}, checkpoint_path)
    restored, _ = build_e13_student(ROOT / "models" / "goiturn-s-gd", num_classes=3, variant="centered")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(state["model"], strict=True)
    checks["strict_checkpoint_restore"] = True
    checkpoint_path.unlink(missing_ok=True)
    payload = {
        "protocol": PROTOCOL,
        "status": "passed" if all(bool(value) for value in checks.values()) else "failed",
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "checks": checks,
        "details": details,
        "device": device_name or "cpu",
    }
    _write_json(ORCH_ROOT / "preflight.json", payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"E13 preflight failed: {checks}")
    return payload


def _profile(device_name: str) -> dict[str, Any]:
    """Profile one real ViSFD batch of 32 through forward/backward."""

    device = torch.device(device_name)
    contract = _task_contract("visfd")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
    train_loader, _, _ = e2_train._loader_set(tokenizer, contract["train"], contract["validation"], contract["bundle"], seed=SEED, micro_batch_size=MICRO_BATCH_SIZE, max_length=MAX_LENGTH)
    model, spec = build_e13_student(ROOT / "models" / "goiturn-s-gd", num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, variant="centered")
    model.to(device).train()
    optimizer = e7_optim.build_layerwise_optimizer(model, TASK_LRS["visfd"], gamma=GAMMA, num_layers=NUM_LAYERS)
    batch = next(iter(train_loader))
    moved = e1_train._move_batch(batch, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    text_by_id = {
        str(example_id): text
        for example_id, text in zip(
            e1_train.split_ids(contract["train"]), e1_train.split_texts(contract["train"])
        )
    }
    noisy, _ = e2_train._make_noisy_batch(
        tokenizer,
        batch["ids"],
        text_by_id,
        max_length=MAX_LENGTH,
        seed=SEED,
        epoch=1,
        device=device,
    )
    with e1_train.make_autocast("bf16", device):
        logits = model(
            moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids")
        )
        noisy_logits = model(
            noisy["input_ids"], noisy["attention_mask"], token_type_ids=noisy.get("token_type_ids")
        )
        ce = e1_train.compute_loss(
            logits,
            moved["labels"],
            num_aspects=contract["num_aspects"],
            aspect_class_counts=contract["aspect_class_counts"],
        )
        consistency = e2_train._load_loss_api()[1](
            logits,
            noisy_logits,
            temperature=1.0,
            slot_class_counts=contract["aspect_class_counts"] or None,
        )
        loss = ce + NC_LAMBDAS["visfd"] * consistency
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    payload = {
        "device": device_name,
        "batch_size": len(batch["labels"]),
        "sequence_length": int(batch["input_ids"].shape[1]),
        "loss": float(loss.detach().cpu()),
        "elapsed_seconds": time.time() - started,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "model_spec": spec.to_dict(),
    }
    _write_json(ORCH_ROOT / "runtime_profile.json", payload)
    return payload


def _run_one_process(variant: str, task_id: str, device: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--worker", "--variant", variant, "--task-id", task_id, "--device", device, "--output-dir", str(_artifact_dir(variant, task_id))]


def run_all(*, poll_seconds: float = 15.0) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    ORCH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    preflight = _preflight_model()
    devices, snapshot = _eligible_devices()
    profile = _profile(devices[0])
    jobs_all = [(variant, task) for variant in VARIANTS for task in TASKS]
    completed: dict[str, Path] = {}
    queue: list[tuple[str, str]] = []
    for variant, task in jobs_all:
        key = _trial_key(variant, task)
        artifact = _completed_artifact(variant, task)
        if artifact is not None:
            completed[key] = artifact
        else:
            queue.append((variant, task))
    state_path = ORCH_ROOT / "controller_state.json"
    old_state = _read_json(state_path) if state_path.is_file() else {}
    if old_state.get("controller_pid") and int(old_state["controller_pid"]) != os.getpid() and _pid_alive(old_state["controller_pid"]):
        raise RuntimeError(f"E13 controller already active: PID {old_state['controller_pid']}")
    running: dict[str, tuple[subprocess.Popen[str] | None, str]] = {}
    # Adopt a worker from a prior controller if it is alive; this prevents a
    # restart from dispatching a duplicate trial.
    for variant, task in jobs_all:
        key = _trial_key(variant, task)
        path = WORKER_ROOT / f"{key}.json"
        if path.is_file():
            worker = _read_json(path)
            if worker.get("status") == "running" and _pid_alive(worker.get("pid")):
                running[key] = (None, str(worker.get("device", "cuda:0")))
                queue = [(v, t) for v, t in queue if _trial_key(v, t) != key]
    free_devices = [device for device in devices if device not in {value[1] for value in running.values()}]
    jobs: dict[str, Any] = {key: {"status": "completed", "artifact_dir": str(path), "reused_existing": True} for key, path in completed.items()}
    for key, (_, device) in running.items():
        variant, task_id = key.split("__", 1)
        jobs[key] = {
            "status": "running",
            "variant": variant,
            "task_id": task_id,
            "pid": _read_json(WORKER_ROOT / f"{key}.json").get("pid"),
            "device": device,
            "adopted_existing_worker": True,
            "artifact_dir": str(_artifact_dir(variant, task_id)),
        }
    controller = {
        "protocol": PROTOCOL,
        "status": "running",
        "controller_pid": os.getpid(),
        "started_at": _now(),
        "heartbeat_at": _now(),
        "source_hashes": _source_values(),
        "eligible_devices": devices,
        "gpu_snapshot_at_launch": snapshot,
        "preflight": str(ORCH_ROOT / "preflight.json"),
        "runtime_profile": str(ORCH_ROOT / "runtime_profile.json"),
        "jobs": jobs,
    }
    _write_json(state_path, controller)
    log_handle = (LOG_ROOT / "controller.log").open("a", encoding="utf-8", buffering=1)

    def launch(variant: str, task_id: str, device: str) -> None:
        key = _trial_key(variant, task_id)
        log_path = LOG_ROOT / f"{key}.log"
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        command = _run_one_process(variant, task_id, device)
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, text=True)
        handle.close()
        running[key] = (process, device)
        jobs[key] = {"status": "running", "variant": variant, "task_id": task_id, "pid": process.pid, "device": device, "log_path": str(log_path), "artifact_dir": str(_artifact_dir(variant, task_id)), "launched_at": _now()}
        log_handle.write(json.dumps({"event": "launch", "key": key, "pid": process.pid, "device": device}) + "\n")

    try:
        while queue or running:
            while queue and free_devices:
                variant, task = queue.pop(0)
                device = free_devices.pop(0)
                launch(variant, task, device)
            for key, (process, device) in list(running.items()):
                status_path = WORKER_ROOT / f"{key}.json"
                worker = _read_json(status_path) if status_path.is_file() else {}
                returncode = process.poll() if process is not None else None
                if returncode is None and worker.get("status") == "completed":
                    returncode = 0
                if returncode is None and worker.get("status") == "failed":
                    returncode = 1
                if returncode is None and process is None and worker.get("status") == "running" and not _pid_alive(worker.get("pid")):
                    returncode = 1
                if returncode is None:
                    continue
                jobs[key].update({"status": "completed" if returncode == 0 else "failed", "returncode": returncode, "finished_at": _now()})
                if returncode == 0:
                    artifact = _completed_artifact(*key.split("__", 1))
                    if artifact is None:
                        raise RuntimeError(f"Worker {key} exited without a complete artifact")
                    completed[key] = artifact
                free_devices.append(device)
                running.pop(key)
            failures = [key for key, job in jobs.items() if job.get("status") == "failed"]
            controller.update({"heartbeat_at": _now(), "jobs": jobs, "queue": [_trial_key(v, t) for v, t in queue], "running": list(running), "gpu_snapshot": _gpu_snapshot()})
            _write_json(state_path, controller)
            if failures:
                controller.update({"status": "failed", "failed_jobs": failures, "finished_at": _now()})
                _write_json(state_path, controller)
                raise RuntimeError(f"E13 workers failed: {failures}")
            if running:
                time.sleep(poll_seconds)
    finally:
        log_handle.close()
    summary = build_validation_summary(completed)
    controller.update({"status": "completed", "finished_at": _now(), "summary": str(ARTIFACT_ROOT / "validation_summary.json")})
    _write_json(state_path, controller)
    return {"preflight": preflight, "runtime_profile": profile, "summary": summary}


def build_validation_summary(artifacts: Mapping[str, Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for task_id in TASKS:
            key = _trial_key(variant, task_id)
            artifact = Path(artifacts[key])
            contract = _task_contract(task_id)
            valid = e1_train.split_field(contract["bundle"], "validation")
            predictions = _read_json(artifact / "validation_predictions.json")["rows"]
            expected_ids = [str(value) for value in e1_train.split_ids(valid)]
            actual_ids = [str(row["id"]) for row in predictions]
            if actual_ids != expected_ids:
                raise RuntimeError(f"Validation IDs/order mismatch for {key}")
            recomputed = e1_train.evaluate_predictions(task_id, contract["bundle"], [row["gold"] for row in predictions], [row["prediction"] for row in predictions], split="validation")
            metadata = _read_json(artifact / "metadata.json")
            row = {
                "key": key,
                "variant": variant,
                "task_id": task_id,
                "seed": SEED,
                "best_epoch": metadata.get("best_epoch"),
                "selection_f1": float(recomputed["selection_f1"]),
                "best_selection_f1_saved": float(metadata["best_selection_f1"]),
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "peak_gpu_memory_allocated_bytes": metadata.get("peak_gpu_memory_allocated_bytes"),
                "peak_gpu_memory_reserved_bytes": metadata.get("peak_gpu_memory_reserved_bytes"),
                "artifact_dir": str(artifact),
                "checkpoint_sha256": _sha256(artifact / "best_checkpoint.pt"),
                "metrics": recomputed,
            }
            rows.append(row)
    payload = {"protocol": PROTOCOL, "status": "completed", "created_at": _now(), "rows": rows, "source_hashes": _source_values(), "metric_definition": "recomputed from validation_predictions.json; internal F1 fractions 0..1"}
    _write_json(ARTIFACT_ROOT / "validation_summary.json", payload)
    fields = ["key", "variant", "task_id", "seed", "best_epoch", "selection_f1", "best_selection_f1_saved", "elapsed_seconds", "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes", "checkpoint_sha256", "artifact_dir"]
    with (ARTIFACT_ROOT / "validation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return payload


def lock_checkpoints() -> dict[str, Any]:
    summary = _read_json(ARTIFACT_ROOT / "validation_summary.json")
    entries: list[dict[str, Any]] = []
    for row in summary["rows"]:
        artifact = Path(row["artifact_dir"])
        metadata = _read_json(artifact / "metadata.json")
        valid = _task_contract(row["task_id"])["validation"]
        entries.append({
            "key": row["key"], "variant": row["variant"], "task_id": row["task_id"], "seed": SEED,
            "best_epoch": metadata["best_epoch"], "checkpoint": str((artifact / "best_checkpoint.pt").resolve()),
            "checkpoint_sha256": _sha256(artifact / "best_checkpoint.pt"), "metadata_sha256": _sha256(artifact / "metadata.json"),
            "architecture": metadata.get("model", {}), "source_hashes": _source_values(),
            "validation_ids_sha256": hashlib.sha256(json.dumps([str(v) for v in e1_train.split_ids(valid)], separators=(",", ":")).encode()).hexdigest(),
            "numeric_gold_untouched": True,
        })
    manifest = {"protocol": PROTOCOL, "status": "locked", "locked_at": _now(), "explicit_test_authorization": True, "entries": entries}
    _write_json(ARTIFACT_ROOT / "locked_checkpoints.json", manifest)
    return manifest


def _load_e13_for_task(variant: str, task_id: str, *, include_test: bool, device: torch.device):
    contract = _task_contract(task_id, include_test=include_test)
    model, spec = build_e13_student(ROOT / "models" / "goiturn-s-gd", num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, variant=variant)
    model.to(device)
    return contract, model, spec


def evaluate_locked_tests(device_name: str = "cuda:0") -> dict[str, Any]:
    manifest = _read_json(ARTIFACT_ROOT / "locked_checkpoints.json")
    if manifest.get("status") != "locked" or not manifest.get("explicit_test_authorization"):
        raise RuntimeError("E13 test requires the locked checkpoint manifest")
    device = torch.device(device_name)
    rows: list[dict[str, Any]] = []
    access_log: list[dict[str, Any]] = []
    for entry in manifest["entries"]:
        checkpoint = Path(entry["checkpoint"])
        if _sha256(checkpoint) != entry["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash changed after lock: {checkpoint}")
        variant, task_id = str(entry["variant"]), str(entry["task_id"])
        contract, model, spec = _load_e13_for_task(variant, task_id, include_test=True, device=device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
        test_split = contract["test"]
        loader = e2_train._make_loader(tokenizer, test_split, max_length=MAX_LENGTH, batch_size=MICRO_BATCH_SIZE, shuffle=False, seed=SEED, labels=e1_train.model_labels(contract["bundle"], "test", test_split), gold_labels=e1_train.split_labels(test_split))
        model.eval(); gold: list[Any] = []; pred: list[Any] = []; ids: list[Any] = []; logits_rows: list[Tensor] = []
        with torch.inference_mode():
            for batch in loader:
                moved = e1_train._move_batch(batch, device)
                with e1_train.make_autocast("bf16", device):
                    logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                pred.extend(e1_train._predictions(logits, aspect_class_counts=contract["aspect_class_counts"]))
                gold.extend(batch["gold_labels"].tolist()); ids.extend(batch["ids"]); logits_rows.append(logits.detach().float().cpu())
        metrics = e1_train.evaluate_predictions(task_id, contract["bundle"], gold, pred, split="test")
        predictions = [{"id": id_value, "gold": gold_value, "prediction": prediction} for id_value, gold_value, prediction in zip(ids, gold, pred)]
        artifact = Path(entry["checkpoint"]).parent
        _write_json(artifact / "test_metrics.json", metrics)
        _write_json(artifact / "test_predictions.json", {"rows": predictions})
        torch.save(torch.cat(logits_rows, dim=0), artifact / "test_full_logits.pt")
        access_log.append({"key": entry["key"], "task_id": task_id, "variant": variant, "split": "test", "loaded_test": True, "checkpoint_sha256": entry["checkpoint_sha256"], "timestamp": _now()})
        rows.append({"key": entry["key"], "variant": variant, "task_id": task_id, "seed": SEED, "best_epoch": entry["best_epoch"], "test_metrics": metrics, "prediction_count": len(predictions), "checkpoint_sha256": entry["checkpoint_sha256"], "artifact_dir": str(artifact)})
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    payload = {"protocol": PROTOCOL, "status": "completed", "created_at": _now(), "split": "test", "rows": rows, "comparison_note": "Compare with E8 test seed42 and paper BamiBERT in separate columns; split/evaluator lineage is retained.", "source_hashes": _source_values()}
    _write_json(ARTIFACT_ROOT / "test_summary.json", payload)
    _write_json(ARTIFACT_ROOT / "test_access_log.json", {"entries": access_log})
    fields = ["key", "variant", "task_id", "seed", "best_epoch", "prediction_count", "checkpoint_sha256", "artifact_dir"]
    with (ARTIFACT_ROOT / "test_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return payload


def write_result_doc() -> None:
    validation = _read_json(ARTIFACT_ROOT / "validation_summary.json") if (ARTIFACT_ROOT / "validation_summary.json").is_file() else {}
    testing = _read_json(ARTIFACT_ROOT / "test_summary.json") if (ARTIFACT_ROOT / "test_summary.json").is_file() else {}
    e8_validation: dict[str, float] = {}
    e8_test: dict[str, float] = {}
    e8_summary_path = ROOT / "artifacts" / "e8" / "summary.json"
    if e8_summary_path.is_file():
        e8_summary = _read_json(e8_summary_path)
        e8_validation = {
            str(row["task_id"]): float(row["selection_f1_e8_recomputed"])
            for row in e8_summary.get("rows", [])
        }
    for task_id in TASKS:
        path = ROOT / "artifacts" / "e8" / task_id / "seed_42" / "e8_test_metrics.json"
        if path.is_file():
            e8_test[task_id] = float(_read_json(path)["selection_f1"])
    lines = [
        "# E13 Result: Centered pair-value residual versus uncentered control", "",
        "E13 trained centered and uncentered block-6 pair-value residual variants on all five locked tasks with seed 42, 30 epochs, effective/micro batch 32, max length 256, BF16 SDPA, NC loss and E7 layer-wise AdamW decay. Validation selected each checkpoint independently; test was run for all ten locked checkpoints after manifest creation.", "",
        "The numeric gold labels and prediction integers are unchanged. Semantic report metadata uses the corrected polarity overlay: Restaurant/Hotel and ViSFD polarity 0 absent, 1 positive, 2 negative, 3 neutral; ViSFD OTHERS raw code 4 is model-binary and pair-masked.", "",
        "Artifacts: `artifacts/e13/.orchestrator/preflight.json`, `runtime_profile.json`, `validation_summary.json/.csv`, `locked_checkpoints.json`, `test_summary.json/.csv`, and `test_access_log.json`.", "",
    ]
    if validation:
        lines += ["## Validation", "", "| Task | E8 | Centered | Uncentered | Centered−E8 (pp) | Centered−Control (pp) |", "|---|---:|---:|---:|---:|---:|"]
        for task_id in TASKS:
            centered = next(float(row["selection_f1"]) for row in validation.get("rows", []) if row["task_id"] == task_id and row["variant"] == "centered")
            control = next(float(row["selection_f1"]) for row in validation.get("rows", []) if row["task_id"] == task_id and row["variant"] == "uncentered")
            baseline = e8_validation.get(task_id, float("nan"))
            lines.append(f"| {task_id} | {baseline:.6f} | {centered:.6f} | {control:.6f} | {(100*(centered-baseline)):+.3f} | {(100*(centered-control)):+.3f} |")
        lines += ["", "Best epochs are recorded in `validation_summary.json`; the table reports the validation-selected centered and uncentered scores. The E13 investment gate did not pass: centered has fewer than two gains of at least +0.3 pp versus E8. The centering subcheck is positive on 3/5 tasks and has no drop larger than 0.3 pp versus the uncentered control, but this is a one-seed result.", ""]
        lines += ["", "Selection F1 is reported internally as a fraction in [0,1]; percentage-point deltas are computed as 100 × (new − baseline).", ""]
    if testing:
        lines += ["## Test", "", "| Task | E8 test | Centered | Uncentered | Centered−E8 (pp) |", "|---|---:|---:|---:|---:|"]
        test_rows = {(str(row["task_id"]), str(row["variant"])): row for row in testing.get("rows", [])}
        for task_id in TASKS:
            centered = float(test_rows[(task_id, "centered")]["test_metrics"]["selection_f1"])
            control = float(test_rows[(task_id, "uncentered")]["test_metrics"]["selection_f1"])
            baseline = e8_test.get(task_id, float("nan"))
            lines.append(f"| {task_id} | {baseline:.6f} | {centered:.6f} | {control:.6f} | {(100*(centered-baseline)):+.3f} |")
        lines += ["", "All ten locked checkpoints were evaluated. E8 test seed42 and paper BamiBERT comparisons preserve their split/evaluator lineage; test scores were not used for checkpoint selection.", ""]
    (ROOT / "docs" / "E13_RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--test-all", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker:
            if not args.variant or not args.task_id or args.output_dir is None:
                raise ValueError("worker requires variant, task-id and output-dir")
            run_worker(args.variant, args.task_id, args.device, args.output_dir.resolve())
        elif args.preflight:
            print(json.dumps(_preflight_model(args.device), indent=2, default=_json_default))
        elif args.profile:
            print(json.dumps(_profile(args.device), indent=2, default=_json_default))
        elif args.run_all:
            print(json.dumps(run_all(), indent=2, default=_json_default))
        elif args.lock:
            print(json.dumps(lock_checkpoints(), indent=2, default=_json_default))
        elif args.test_all:
            print(json.dumps(evaluate_locked_tests(args.device), indent=2, default=_json_default))
        elif args.write_report:
            write_result_doc()
        else:
            raise ValueError("choose --preflight, --profile, --run-all, --lock, --test-all or --write-report")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
