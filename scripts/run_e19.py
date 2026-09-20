#!/usr/bin/env python3
"""E19 preflight, worker, and run-all controller.

Screening whether the recipe that won on the student (E7 layer-wise-LR-decay
optimizer + E2 noise-consistency loss, "nc" mode) also helps the 110M DAPT
encoder ``models/goibert-b-dapt`` fine-tuned with the E2 teacher head (768
hidden, ``goibert_e2.modeling.build_e2_teacher``).  See ``docs/E19_PLAN.md``
(locked spec).

The E2 data/loss/validation loop (``goibert_e2.train._train_any`` /
``_train_loop``) and the E7 optimizer (``goibert_e7.optim``) stay frozen and
unmodified.  This runner only, for the duration of one worker process:

* monkeypatches ``goibert_e2.train.build_e2_student`` so the frozen
  ``_train_any`` (mode="nc" branch) builds a fresh ``build_e2_teacher`` model
  over ``models/goibert-b-dapt`` instead of the E1 student backbone
  (``models/goiturn-s-gd``, which ``_train_any`` still passes as the ignored
  positional ``backbone_dir`` argument -- only used by ``_train_any`` itself
  for an unused ``directory_hashes`` metadata call);
* monkeypatches ``goibert_e2.train.make_optimizer`` to build the E7
  layer-wise-LR-decay AdamW (12 encoder blocks + embeddings + head) instead
  of the E1 flat-LR AdamW;
* calls ``goibert_e2.train._train_any`` directly (not ``train_student_trial``,
  which locks the LR to the E1/E2 *student* winners and would reject the
  teacher LRs used here) with ``mode="nc"``.

No new parameters, no new src/ package.  Never edit ``src/goibert*``, other
``scripts/run_e*.py``, ``models/``, or ``artifacts/e1``..``artifacts/e18``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e19"
CONFIG_PATH = ROOT / "configs" / "e19" / "protocol.json"
ENCODER_DIR = ROOT / "models" / "goibert-b-dapt"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e2.modeling import build_e2_teacher  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402


PROTOCOL = "E19-dapt-nc-llrd-screening-v1"
REVISION = "e19-20260914-r1"
STAGE = "stageE19_dapt_nc_llrd"
METHOD = "dapt_nc_llrd"
ARM = "dapt_nc_llrd"

# Dispatch order per docs/E19_PLAN.md.
TASKS = ("absa_restaurant", "absa_hotel", "visfd", "vsfc_topic", "vsfc_sentiment")
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
MAX_LENGTH = 256
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
GAMMA = 0.8
NUM_LAYERS = 12
WEIGHT_DECAY = 0.01
TASK_LRS: dict[str, float] = {
    "vsfc_sentiment": 2e-5,
    "vsfc_topic": 1e-5,
    "visfd": 5e-5,
    "absa_restaurant": 2e-5,
    "absa_hotel": 5e-5,
}
NC_LAMBDAS: dict[str, float] = {
    "vsfc_sentiment": 0.1,
    "vsfc_topic": 0.5,
    "visfd": 0.5,
    "absa_restaurant": 0.1,
    "absa_hotel": 0.1,
}
GUARD_USED_MIB = 5 * 1024
GUARD_UTIL_PCT = 5.0
# GPU1 was confirmed idle (20 MiB used, 0% util) at plan time; GPU0/2/3
# belonged to other users' jobs.  Preference order only; eligibility is
# re-checked against the live snapshot every time.
PREFERRED_GPU_INDICES = ("1", "0", "2", "3")


# ---------------------------------------------------------------------------
# Generic helpers (ported from scripts/run_e18.py)
# ---------------------------------------------------------------------------
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
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".safetensors", ".txt", ".model", ".vocab"}
    }


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as exc:  # noqa: BLE001 - report, do not crash preflight
        return {"available": False, "error": repr(exc), "rows": {}}
    rows: dict[str, dict[str, float]] = {}
    for line in output.strip().splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 4:
            rows[fields[0]] = {
                "free_mib": int(float(fields[1])),
                "used_mib": int(float(fields[2])),
                "utilization_pct": float(fields[3]),
            }
    return {"available": bool(rows), "rows": rows}


def _eligible_devices(snapshot: Mapping[str, Any]) -> list[str]:
    rows = snapshot.get("rows", {})
    eligible = [
        index
        for index, row in rows.items()
        if float(row["used_mib"]) < GUARD_USED_MIB and float(row["utilization_pct"]) <= GUARD_UTIL_PCT
    ]
    ordered = [index for index in PREFERRED_GPU_INDICES if index in eligible]
    ordered += [index for index in eligible if index not in ordered]
    return [f"cuda:{index}" for index in ordered]


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


def _source_manifest(seed: int = SEED) -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        CONFIG_PATH,
        ROOT / "src" / "goibert_e2" / "train.py",
        ROOT / "src" / "goibert_e2" / "modeling.py",
        ROOT / "src" / "goibert_e2" / "losses.py",
        ROOT / "src" / "goibert_e2" / "noise.py",
        ROOT / "src" / "goibert_e2" / "cache.py",
        ROOT / "src" / "goibert_e7" / "optim.py",
        ROOT / "src" / "goibert" / "train.py",
        ROOT / "src" / "goibert" / "modeling.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in source_files if path.is_file()}
    return {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "created_at": _now(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "encoder": {
            "path": str(ENCODER_DIR.resolve()),
            "files": _directory_hashes(ENCODER_DIR),
        },
        "frozen_e2_source_hashes": e2_train.source_hashes(ROOT),
        "recipe": {
            "arm": ARM,
            "tasks": list(TASKS),
            "seed": seed,
            "epochs": EPOCHS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "max_length": MAX_LENGTH,
            "precision": PRECISION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "gamma": GAMMA,
            "num_layers": NUM_LAYERS,
            "weight_decay": WEIGHT_DECAY,
            "task_lrs": dict(TASK_LRS),
            "nc_lambdas": dict(NC_LAMBDAS),
        },
    }


# ---------------------------------------------------------------------------
# Model / data construction helpers
# ---------------------------------------------------------------------------
def _task_contract(task_id: str) -> dict[str, Any]:
    return e2_train._task_contract(task_id, ROOT, smoke=False)


def _tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)


def _build_teacher(contract: Mapping[str, Any]) -> tuple[Any, Any]:
    return build_e2_teacher(
        ENCODER_DIR,
        num_classes=int(contract["num_classes"]),
        num_aspects=int(contract["num_aspects"]),
        aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )


def _fixed_batch(task_id: str) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Real micro-batch (clean + noisy) for ``task_id``, tensors on CPU."""

    contract = _task_contract(task_id)
    tokenizer, _lengths = e2_train._tokenizer_and_lengths(ROOT, contract["train"])
    train_loader, _, _ = e2_train._loader_set(
        tokenizer,
        contract["train"],
        contract["validation"],
        contract["bundle"],
        seed=SEED,
        micro_batch_size=MICRO_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    batch = next(iter(train_loader))
    text_by_id = {
        str(example_id): text
        for example_id, text in zip(e2_train.split_ids(contract["train"]), e2_train.split_texts(contract["train"]))
    }
    noisy, noise_stats = e2_train._make_noisy_batch(
        tokenizer,
        batch["ids"],
        text_by_id,
        max_length=MAX_LENGTH,
        seed=SEED,
        epoch=1,
        device=torch.device("cpu"),
    )
    return contract, batch, noisy, noise_stats


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def _shape_check(device: torch.device) -> list[dict[str, Any]]:
    """(1) Model builds for all 5 tasks with the right output shapes, hidden 768."""

    tokenizer = _tokenizer()
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        contract = _task_contract(task_id)
        texts = list(e2_train.split_texts(contract["validation"]))[:4]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        model, spec = _build_teacher(contract)
        model.to(device).eval()
        with torch.no_grad():
            logits = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
        rows.append(
            {
                "task_id": task_id,
                "hidden_size": int(model.encoder.config.hidden_size),
                "num_classes": int(contract["num_classes"]),
                "num_aspects": int(contract["num_aspects"]),
                "logits_shape": list(logits.shape),
                "logits_numel": int(logits.numel()),
                "expected_numel": len(texts) * int(contract["num_classes"]),
                "finite": bool(torch.isfinite(logits).all().item()),
                "model_spec": spec.to_dict(),
                "checks": {
                    "hidden_768": int(model.encoder.config.hidden_size) == 768,
                    "numel_matches_num_classes": logits.numel() == len(texts) * int(contract["num_classes"]),
                    "finite": bool(torch.isfinite(logits).all().item()),
                },
            }
        )
        del model
    return rows


def _optimizer_coverage_check(device: torch.device) -> dict[str, Any]:
    """(2) Optimizer coverage: every parameter exactly once, 12 blocks +
    embeddings + head groups, endpoint LR scales -- for every task's LR."""

    expected_group_names = [
        f"{region}_{kind}"
        for region in ("embeddings", *[f"block_{i:02d}" for i in range(NUM_LAYERS)], "head")
        for kind in ("decay", "no_decay")
    ]
    rows: dict[str, Any] = {}
    for task_id in TASKS:
        contract = _task_contract(task_id)
        model, _ = _build_teacher(contract)
        model.to(device)
        lr = TASK_LRS[task_id]
        optimizer = e7_optim.build_layerwise_optimizer(
            model, lr, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
        )
        coverage = e7_optim.validate_optimizer_groups(
            model,
            optimizer,
            base_lr=lr,
            gamma=GAMMA,
            num_layers=NUM_LAYERS,
            weight_decay=WEIGHT_DECAY,
            expected_group_names=expected_group_names,
        )
        rows[task_id] = {
            "base_lr": lr,
            "all_trainable": coverage["all_trainable"],
            "duplicate_parameter_count": coverage["duplicate_parameter_count"],
            "parameter_count": coverage["parameter_count"],
            "group_count": coverage["group_count"],
            "group_names": coverage["group_names"],
            "endpoint_scales": coverage["endpoint_scales"],
            "endpoint_lrs": {
                region: lr * scale for region, scale in coverage["endpoint_scales"].items()
            },
        }
        del model, optimizer
    return rows


def _nc_step_and_vram_check(device: torch.device) -> dict[str, Any]:
    """(3) One NC training step on a real micro-batch is finite, and
    (4) peak VRAM for one step at micro 16 x 256 tokens -- for every task."""

    rows: dict[str, Any] = {}
    for task_id in TASKS:
        contract, clean_cpu, noisy_cpu, noise_stats = _fixed_batch(task_id)
        model, spec = _build_teacher(contract)
        model.to(device).train()
        lr = TASK_LRS[task_id]
        optimizer = e7_optim.build_layerwise_optimizer(
            model, lr, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
        )
        clean = {key: value.to(device) for key, value in clean_cpu.items() if isinstance(value, torch.Tensor)}
        noisy = {key: value.to(device) for key, value in noisy_cpu.items()}
        counts = contract["aspect_class_counts"]
        num_aspects = contract["num_aspects"]
        nc_fn = e2_train._load_loss_api()[1]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        last_total = None
        for _ in range(ACCUMULATION_STEPS):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(clean["input_ids"], clean["attention_mask"])
                ce = e2_train.compute_loss(logits, clean["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                noisy_logits = model(noisy["input_ids"], noisy["attention_mask"])
                consistency = nc_fn(logits, noisy_logits, temperature=1.0, slot_class_counts=counts or None)
                total = ce + NC_LAMBDAS[task_id] * consistency
            (total / ACCUMULATION_STEPS).backward()
            last_total = total
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        rows[task_id] = {
            "base_lr": lr,
            "nc_lambda": NC_LAMBDAS[task_id],
            "input_shape": list(clean["input_ids"].shape),
            "ce_loss": float(ce.detach().cpu().item()),
            "consistency_loss": float(consistency.detach().cpu().item()),
            "total_loss": float(last_total.detach().cpu().item()),
            "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "noise_change_rate": noise_stats["changed_count"] / max(noise_stats["count"], 1),
            "checks": {
                "loss_finite": math.isfinite(float(last_total.detach().cpu().item())),
                "grad_norm_finite": math.isfinite(float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)),
            },
        }
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _strict_reload_check(device: torch.device) -> dict[str, Any]:
    """(5) Strict state_dict round-trip into a fresh build_e2_teacher model."""

    contract = _task_contract("absa_restaurant")
    model, _ = _build_teacher(contract)
    model.to(device)
    path = ARTIFACT_ROOT / "preflight" / "strict_reload.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)

    fresh, _ = _build_teacher(contract)
    fresh.to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    result = fresh.load_state_dict(state["model"], strict=True)
    return {
        "task_id": "absa_restaurant",
        "path": str(path),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "strict_pass": not result.missing_keys and not result.unexpected_keys,
    }


def run_preflight(device_name: str | None = None) -> dict[str, Any]:
    started = time.time()
    snapshot = _gpu_snapshot()
    if device_name is None:
        eligible = _eligible_devices(snapshot)
        device_name = eligible[0] if eligible else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable in this process")

    manifest = _source_manifest()
    before_encoder = manifest["encoder"]["files"]

    functional = {
        "shape_check": _shape_check(device),
        "optimizer_coverage": _optimizer_coverage_check(device),
        "nc_step_and_vram": _nc_step_and_vram_check(device),
        "strict_reload": _strict_reload_check(device),
    }

    after_manifest = _source_manifest()
    after_encoder = after_manifest["encoder"]["files"]

    checks = {
        "shape_check_all_pass": all(row["checks"]["hidden_768"] and row["checks"]["numel_matches_num_classes"] and row["checks"]["finite"] for row in functional["shape_check"]),
        "optimizer_coverage_all_tasks": all(
            row["all_trainable"] and row["duplicate_parameter_count"] == 0 and row["group_count"] == 2 * (NUM_LAYERS + 2)
            for row in functional["optimizer_coverage"].values()
        ),
        "nc_step_all_finite": all(row["checks"]["loss_finite"] and row["checks"]["grad_norm_finite"] for row in functional["nc_step_and_vram"].values()),
        "strict_reload_pass": functional["strict_reload"]["strict_pass"],
        "encoder_no_mutation": before_encoder == after_encoder,
    }
    gpu_ready = device.type == "cuda" and snapshot.get("available", False)
    peak_reserved_by_task = {
        task_id: row["peak_reserved_bytes"] for task_id, row in functional["nc_step_and_vram"].items()
    }
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "passed" if all(checks.values()) and gpu_ready else ("blocked_no_cuda" if not gpu_ready else "failed"),
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "gpu_snapshot": snapshot,
        "eligible_devices": _eligible_devices(snapshot),
        "gpu_preflight_ready": gpu_ready,
        "checks": checks,
        "peak_reserved_bytes_by_task": peak_reserved_by_task,
        "peak_reserved_max_bytes": max(peak_reserved_by_task.values()) if gpu_ready and all(v is not None for v in peak_reserved_by_task.values()) else None,
        "functional": functional,
        "encoder_hash_unchanged": before_encoder == after_encoder,
    }
    _write_json(ARTIFACT_ROOT / "preflight.json", payload)
    _write_json(ARTIFACT_ROOT / "source_manifest.json", manifest)
    return payload


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_worker(task_id: str, device_name: str, output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"E19 task not locked: {task_id}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing duplicate E19 output: {output_dir}")

    source_manifest = _source_manifest(seed)
    holder: dict[str, Any] = {}

    original_builder = e2_train.build_e2_student
    original_optimizer = e2_train.make_optimizer

    def e19_builder(
        backbone_dir: Path,  # ignored: always models/goibert-b-dapt (see module docstring)
        *,
        num_classes: int,
        num_aspects: int = 0,
        aspect_class_counts: Any = None,
        attention_implementation: str = "sdpa",
    ):
        model, spec = build_e2_teacher(
            ENCODER_DIR,
            num_classes=num_classes,
            num_aspects=num_aspects,
            aspect_class_counts=aspect_class_counts,
            attention_implementation=attention_implementation,
        )
        return model, spec

    def e19_optimizer(model: torch.nn.Module, learning_rate: float):
        optimizer = e7_optim.build_layerwise_optimizer(
            model, learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
        )
        coverage = e7_optim.validate_optimizer_groups(
            model, optimizer, base_lr=learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
        )
        holder["optimizer_coverage"] = {
            "all_trainable": coverage["all_trainable"],
            "duplicate_parameter_count": coverage["duplicate_parameter_count"],
            "parameter_count": coverage["parameter_count"],
            "group_count": coverage["group_count"],
            "group_names": coverage["group_names"],
            "endpoint_scales": coverage["endpoint_scales"],
        }
        return optimizer

    e2_train.build_e2_student = e19_builder  # type: ignore[assignment]
    e2_train.make_optimizer = e19_optimizer  # type: ignore[assignment]
    try:
        result = e2_train._train_any(
            task_id=task_id,
            mode="nc",
            learning_rate=TASK_LRS[task_id],
            seed=seed,
            repo_root=ROOT,
            output_dir=output_dir,
            device_name=device_name,
            precision=PRECISION,
            attention_implementation=ATTENTION_IMPLEMENTATION,
            epochs=EPOCHS,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            smoke=False,
            alpha=None,
            consistency_lambda=NC_LAMBDAS[task_id],
            cache_path=None,
            stage=STAGE,
            method=METHOD,
            coefficient=NC_LAMBDAS[task_id],
            revision=REVISION,
            expected_e2_code_hash=None,
            expected_e2_config_hash=None,
            expected_orchestrator_hash=None,
        )
    finally:
        e2_train.build_e2_student = original_builder  # type: ignore[assignment]
        e2_train.make_optimizer = original_optimizer  # type: ignore[assignment]

    artifact = Path(result["artifact_dir"])
    metadata = _read_json(artifact / "metadata.json")
    metadata.update(
        {
            "e19_protocol": PROTOCOL,
            "e19_revision": REVISION,
            "e19_arm": ARM,
            "e19_seed": seed,
            "e19_encoder_dir": str(ENCODER_DIR.resolve()),
            "e19_encoder_hashes": _directory_hashes(ENCODER_DIR),
            "e19_optimizer_coverage": holder.get("optimizer_coverage"),
            "e19_source_manifest": source_manifest,
            "e19_note": (
                "backbone_hashes above is directory_hashes(models/goiturn-s-gd) as "
                "recorded internally by goibert_e2.train._train_any's non-teacher "
                "branch; it is a metadata artifact of calling _train_any with "
                "mode='nc' and is NOT the encoder actually trained. The real "
                "encoder is models/goibert-b-dapt (E2TeacherClassifier, 768 "
                "hidden, 12 layers); see e19_encoder_dir/e19_encoder_hashes above "
                "and model.backbone='goibert-b-dapt' in the model spec below."
            ),
        }
    )
    _write_json(artifact / "metadata.json", metadata)
    checkpoint = artifact / "best_checkpoint.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state.setdefault("metadata", {}).update(
            {
                "e19_protocol": PROTOCOL,
                "e19_arm": ARM,
                "e19_encoder_dir": str(ENCODER_DIR.resolve()),
            }
        )
        torch.save(state, checkpoint)
    return result


# ---------------------------------------------------------------------------
# run-all controller
# ---------------------------------------------------------------------------
def run_all(
    device_names: list[str], *, dispatch_approved: bool, tasks: Sequence[str] = TASKS, seed: int = SEED
) -> dict[str, Any]:
    if not dispatch_approved:
        raise RuntimeError("Full E19 dispatch requires explicit --dispatch-approved")
    if not device_names:
        raise RuntimeError("No GPU devices available for E19 dispatch")
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown E19 task requested for dispatch: {task}")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = ARTIFACT_ROOT / "controller_state.json"
    source_manifest = _source_manifest(seed)
    source_manifest_hash = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode("utf-8")).hexdigest()

    previous: dict[str, Any] = {}
    if state_path.is_file():
        previous = _read_json(state_path)
        if previous.get("status") == "running":
            active = [job for job in previous.get("jobs", []) if _pid_alive(job.get("pid")) and job.get("status") == "running"]
            if active:
                raise RuntimeError(f"E19 controller already has active jobs; adopt/refuse duplicate: {active}")

    queue = list(tasks)
    completed_keys = {job.get("task_id") for job in previous.get("jobs", []) if job.get("status") == "completed"}
    for task in list(queue):
        artifact = ARTIFACT_ROOT / task / f"seed_{seed}"
        status_path = artifact / "status.json"
        if status_path.is_file():
            status = _read_json(status_path)
            if status.get("status") == "completed":
                completed_keys.add(task)
    queue = [task for task in queue if task not in completed_keys]

    controller = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "running",
        "created_at": _now(),
        "source_manifest_sha256": source_manifest_hash,
        "dispatch_approved": True,
        "tasks": list(tasks),
        "jobs": [job for job in previous.get("jobs", []) if job.get("task_id") in completed_keys],
        "queue_remaining": list(queue),
    }
    _write_json(state_path, controller)

    active: list[tuple[subprocess.Popen[str], dict[str, Any], Any]] = []
    completed: list[dict[str, Any]] = list(controller["jobs"])

    def wait_job(process: subprocess.Popen[str], job: dict[str, Any], log_handle: Any) -> int:
        while True:
            code = process.poll()
            if code is not None:
                log_handle.close()
                return int(code)
            controller["heartbeat_at"] = _now()
            controller["active_pids"] = [item[1].get("pid") for item in active]
            _write_json(state_path, controller)
            time.sleep(10)

    for index, task in enumerate(queue):
        device = device_names[index % len(device_names)]
        snapshot_now = _gpu_snapshot()
        eligible_now = _eligible_devices(snapshot_now)
        if device not in eligible_now:
            raise RuntimeError(f"E19 dispatch gate failed: {device} no longer eligible ({snapshot_now})")
        output = ARTIFACT_ROOT / task / f"seed_{seed}"
        log_path = ARTIFACT_ROOT / "logs" / f"{task}__seed{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--task-id",
            task,
            "--device",
            device,
            "--output-dir",
            str(output),
            "--seed",
            str(seed),
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"})
        job = {"task_id": task, "device": device, "pid": process.pid, "output_dir": str(output), "log": str(log_path), "command": command, "status": "running", "started_at": _now()}
        controller["jobs"].append(job)
        controller["queue_remaining"] = list(queue[index + 1 :])
        _write_json(state_path, controller)
        active.append((process, job, log))
        if len(active) >= len(device_names):
            process0, meta0, log0 = active[0]
            code = wait_job(process0, meta0, log0)
            active.pop(0)
            meta0["returncode"] = code
            meta0["status"] = "completed" if code == 0 else "failed"
            meta0["finished_at"] = _now()
            completed.append(meta0)
            _write_json(state_path, controller)
            if code != 0:
                raise RuntimeError(f"E19 worker failed: {meta0}")
    for process, meta, log in active:
        code = wait_job(process, meta, log)
        meta["returncode"] = code
        meta["status"] = "completed" if code == 0 else "failed"
        meta["finished_at"] = _now()
        completed.append(meta)
        _write_json(state_path, controller)
        if code != 0:
            raise RuntimeError(f"E19 worker failed: {meta}")
    controller.update({"status": "completed", "finished_at": _now(), "jobs": completed, "queue_remaining": []})
    _write_json(state_path, controller)
    return controller


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E19 preflight/worker/run-all runner")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--dispatch-approved", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Run seed (default 42). Threaded into _train_any(seed=...), the artifact dir (seed_<seed>), and metadata.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=(
            "Override the automatic eligible-device scan for --run-all (e.g. "
            "cuda:1). Still subject to the per-launch eligibility recheck "
            "inside run_all. Use this to pin dispatch to a device confirmed "
            "idle out-of-band instead of opportunistically picking up every "
            "GPU that happens to read <5GiB/<=5%% at the single scan moment."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not any((args.preflight, args.run_all, args.worker)):
        raise SystemExit("Choose --preflight, --run-all, or --worker")
    if args.worker:
        if not args.task_id or not args.device or args.output_dir is None:
            raise SystemExit("--worker requires --task-id, --device, and --output-dir")
        result = run_worker(args.task_id, args.device, args.output_dir, seed=args.seed)
    elif args.run_all:
        if args.devices:
            devices = list(args.devices)
        else:
            snapshot = _gpu_snapshot()
            devices = _eligible_devices(snapshot)
        devices = [d for d in devices for _ in range(max(1, int(args.workers_per_gpu)))]
        result = run_all(devices, dispatch_approved=args.dispatch_approved, tasks=args.tasks, seed=args.seed)
    else:
        result = run_preflight(args.device)
    print(json.dumps(result, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
