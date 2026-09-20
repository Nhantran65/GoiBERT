"""E11 signed versus convex two-map attention screening.

This runner composes the frozen E2 NC training loop with an isolated E11
student constructor and E7 optimizer.  It never edits E1--E10 sources and
keeps all five tasks validation-only.
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
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e11"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
PROTOCOL_PATH = ROOT / "configs" / "e11" / "protocol.json"
MANIFEST_PATH = ARTIFACT_ROOT / "source_manifest.json"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e2.noise import noisy_batch  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402
from goibert_e11.modeling import (  # noqa: E402
    INSERTION_LAYER_INDEX,
    VARIANTS,
    E11Classifier,
    E11TwoMapSdpaSelfAttention,
    build_e11_student,
    iter_e11_attentions,
)


PROTOCOL = "E11-signed-versus-convex-two-map-attention-v1"
REVISION = "e11-20260911-r1"
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
TRIALS = tuple((variant, task_id) for variant in VARIANTS for task_id in TASKS)
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
PREFERRED_GPU_INDICES = ("0", "1", "2", "3")
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
    with path.open(encoding="utf-8") as handle:
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


def e11_source_hash() -> str:
    return _hash_files((Path(__file__).resolve(), PROTOCOL_PATH, SRC / "goibert_e11" / "modeling.py"))


def _source_hashes() -> dict[str, str]:
    values = e2_train.source_hashes(ROOT)
    if values["e1_dependency"] != E1_SOURCE_HASH or values["e2"] != E2_CODE_HASH or values["e2_config"] != E2_CONFIG_HASH:
        raise RuntimeError(f"Frozen E1/E2 dependency drifted: {values}")
    return values


def _gpu_snapshot() -> dict[str, dict[str, int]]:
    command = ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    snapshot: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3:
            snapshot[fields[0]] = {
                "free_mib": int(float(fields[1])),
                "used_mib": int(float(fields[2])),
                "utilization_gpu": int(float(fields[3])) if len(fields) >= 4 else -1,
            }
    if not snapshot:
        raise RuntimeError(f"nvidia-smi returned no rows: {output!r}")
    return snapshot


def _eligible_devices() -> tuple[list[str], dict[str, dict[str, int]]]:
    snapshot = _gpu_snapshot()
    devices = [
        f"cuda:{index}"
        for index in PREFERRED_GPU_INDICES
        if index in snapshot and snapshot[index]["free_mib"] >= GPU_GUARD_MIB
    ]
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


def _trial_key(variant: str, task_id: str) -> str:
    return f"{variant}__{task_id}__seed{SEED}"


def _trial_dir(variant: str, task_id: str) -> Path:
    return ARTIFACT_ROOT / variant / task_id / f"seed_{SEED}"


def _worker_path(variant: str, task_id: str) -> Path:
    return WORKER_ROOT / f"{_trial_key(variant, task_id)}.json"


def _set_worker_status(variant: str, task_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(_worker_path(variant, task_id), dict(payload))


def _completed_artifact(variant: str, task_id: str) -> Path | None:
    artifact = _trial_dir(variant, task_id)
    if not all((artifact / name).is_file() for name in ("status.json", "metadata.json", "best_checkpoint.pt")):
        return None
    try:
        status = _read_json(artifact / "status.json")
        metadata = _read_json(artifact / "metadata.json")
    except Exception:
        return None
    if status.get("status") != "completed" or metadata.get("protocol") != "E2-training-ablation-v1":
        return None
    return artifact if metadata.get("variant") == variant else None


def _make_e11_optimizer(model: nn.Module, base_lr: float) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    optimizer = e7_optim.build_layerwise_optimizer(model, base_lr, gamma=GAMMA, num_layers=NUM_LAYERS)
    gate_ids = {id(parameter) for name, parameter in model.named_parameters() if name.endswith("gate_logits")}
    if len(gate_ids) != 1:
        raise AssertionError(f"Expected one gate_logits parameter, got {len(gate_ids)}")
    moved = []
    decay_group = next(group for group in optimizer.param_groups if group.get("e7_group_name") == "block_08_decay")
    no_decay_group = next(group for group in optimizer.param_groups if group.get("e7_group_name") == "block_08_no_decay")
    for parameter in list(decay_group["params"]):
        if id(parameter) in gate_ids:
            decay_group["params"].remove(parameter)
            no_decay_group["params"].append(parameter)
            moved.append(parameter)
    decay_group["e7_param_names"] = [name for name in decay_group["e7_param_names"] if not name.endswith("gate_logits")]
    no_decay_group["e7_param_names"].extend(name for name, parameter in model.named_parameters() if id(parameter) in gate_ids)
    if len(moved) != 1:
        raise AssertionError("gate_logits was not moved to the block_08 no-decay group")
    coverage = e7_optim.validate_optimizer_groups(model, optimizer, base_lr=base_lr, gamma=GAMMA, num_layers=NUM_LAYERS)
    gate_rows = [row for row in coverage["groups"] if row["region"] == "block_08"]
    if any("gate_logits" in name for row in gate_rows if row["decay"] for name in row["parameter_names"]):
        raise AssertionError("gate_logits still appears in decay parameter names")
    if any(float(row["weight_decay"]) != 0.0 and any("gate_logits" in name for name in row["parameter_names"]) for row in gate_rows):
        raise AssertionError("gate_logits weight decay is nonzero")
    return optimizer, coverage


def _attention(model: nn.Module) -> E11TwoMapSdpaSelfAttention:
    values = list(iter_e11_attentions(model))
    if len(values) != 1:
        raise AssertionError(f"Expected one E11 attention, got {len(values)}")
    return values[0]


def _preflight_model(variant: str, num_classes: int = 3) -> tuple[E11Classifier, dict[str, Any]]:
    e2_train.seed_everything(SEED)
    model, spec = build_e11_student(ROOT / "models" / "goiturn-s-gd", variant=variant, num_classes=num_classes, attention_implementation="sdpa", warmup_steps=7)
    if _attention(model).warmup_steps.item() != 7:
        raise AssertionError("warmup buffer was not initialized")
    return model, spec.to_dict()


def run_preflight(device_name: str | None = None) -> dict[str, Any]:
    source_values = _source_hashes()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
    text = ["giảng viên rất nhiệt tình", "giảng viên không nhiệt tình"]
    encoded = tokenizer(text, padding=True, truncation=True, max_length=32, return_tensors="pt")

    shape_rows: list[dict[str, Any]] = []
    parameter_counts: dict[str, int] = {}
    for variant in VARIANTS:
        model, _ = _preflight_model(variant)
        model.eval()
        with torch.inference_mode():
            e2_train.seed_everything(SEED)
            base, _ = e2_train.build_e2_student(ROOT / "models" / "goiturn-s-gd", num_classes=3, attention_implementation="sdpa")
            base.eval()
            base_logits = base(encoded["input_ids"], encoded["attention_mask"])
            e11_logits = model(encoded["input_ids"], encoded["attention_mask"])
        if not torch.equal(base_logits, e11_logits):
            raise AssertionError(f"t0 eval parity failed for {variant}: max diff={float((base_logits-e11_logits).abs().max())}")
        model.set_warmup_steps(1)
        model.set_optimizer_step(1)
        model.train()
        out = model(encoded["input_ids"], encoded["attention_mask"])
        if not torch.isfinite(out).all():
            raise FloatingPointError(f"non-finite preflight output for {variant}")
        parameter_counts[variant] = sum(parameter.numel() for parameter in model.parameters())
        shape_rows.append({"variant": variant, "output_shape": list(out.shape), "model_spec": model.spec().to_dict()})
        del model, base
    if len(set(parameter_counts.values())) != 1:
        raise AssertionError(f"signed/convex parameter counts differ: {parameter_counts}")

    # Check Q2/K2/gate gradients at lambda>0 and signed/convex effective weights.
    gradient_rows = []
    for variant in VARIANTS:
        model, _ = _preflight_model(variant)
        model.set_optimizer_step(1)
        model.train()
        logits = model(encoded["input_ids"], encoded["attention_mask"])
        loss = logits.float().square().mean()
        loss.backward()
        module = _attention(model)
        gradient_rows.append({
            "variant": variant,
            "q2_grad_finite": bool(module.q2.weight.grad is not None and torch.isfinite(module.q2.weight.grad).all()),
            "k2_grad_finite": bool(module.k2.weight.grad is not None and torch.isfinite(module.k2.weight.grad).all()),
            "gate_grad_finite": bool(module.gate_logits.grad is not None and torch.isfinite(module.gate_logits.grad).all()),
            "effective_weights": module.effective_weights().tolist(),
        })
        if not all(gradient_rows[-1][key] for key in ("q2_grad_finite", "k2_grad_finite", "gate_grad_finite")):
            raise AssertionError(f"E11 gradient preflight failed: {gradient_rows[-1]}")
        weights = module.effective_weights()
        if variant == "signed" and not bool((weights[:, 1] < 0).all()):
            raise AssertionError("signed effective O2 weights are not negative")
        if variant == "convex" and not bool((weights >= 0).all()):
            raise AssertionError("convex effective weights are negative")
        del model

    model, _ = _preflight_model("signed")
    _, coverage = _make_e11_optimizer(model, 1e-5)
    gate_group = [row for row in coverage["groups"] if row["region"] == "block_08" and not row["decay"]][0]
    if not any("gate_logits" in name for name in gate_group["parameter_names"]):
        raise AssertionError("gate no-decay group missing gate_logits")
    if not math.isclose(float(gate_group["lr"]), 1e-5 * GAMMA, rel_tol=1e-12):
        raise AssertionError("block_08 gate LR is not base_lr * gamma")

    # Exercise the frozen task schemas and the actual additive padding mask
    # for every task before any worker can launch.
    task_shape_rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        contract = e2_train._task_contract(task_id, ROOT, smoke=False)
        model, _ = build_e11_student(
            ROOT / "models" / "goiturn-s-gd",
            variant="signed",
            num_classes=contract["num_classes"],
            num_aspects=contract["num_aspects"],
            aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
            attention_implementation="sdpa",
            warmup_steps=1,
        )
        valid = contract["validation"]
        texts = list(e2_train.split_texts(valid)[:2])
        padded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        single = tokenizer([texts[0]], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        model.eval()
        with torch.inference_mode():
            batch_logits = model(padded["input_ids"], padded["attention_mask"])
            single_logits = model(single["input_ids"], single["attention_mask"])
        if not torch.isfinite(batch_logits).all() or not torch.isfinite(single_logits).all():
            raise FloatingPointError(f"non-finite padded output for {task_id}")
        if not torch.allclose(batch_logits[0], single_logits[0], atol=1e-6, rtol=1e-6):
            raise AssertionError(f"padding/mask changed the first example for {task_id}")
        task_shape_rows.append({"task_id": task_id, "num_labels": contract["num_classes"], "num_aspects": contract["num_aspects"], "aspect_class_counts": list(contract["aspect_class_counts"]), "output_shape": list(batch_logits.shape)})
        del model

    roundtrip = {}
    for variant in VARIANTS:
        model, spec = _preflight_model(variant)
        model.set_warmup_steps(123)
        model.set_optimizer_step(17)
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        restored, restored_spec = _preflight_model(variant)
        missing, unexpected = restored.load_state_dict(state, strict=True)
        if missing or unexpected or restored_spec != spec:
            raise AssertionError(f"strict checkpoint roundtrip failed for {variant}: {missing}, {unexpected}")
        if int(_attention(restored).optimizer_step.item()) != 17 or int(_attention(restored).warmup_steps.item()) != 123:
            raise AssertionError("checkpoint did not restore gate schedule buffers")
        roundtrip[variant] = {"strict_load": True, "step": 17, "warmup_steps": 123, "state_key_count": len(state)}
        del model, restored

    devices: list[str] = []
    snapshot: dict[str, Any] = {}
    gpu_error = None
    try:
        devices, snapshot = _eligible_devices()
    except Exception as exc:
        gpu_error = f"{type(exc).__name__}: {exc}"
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "passed",
        "created_at": _now(),
        "e11_source_hash": e11_source_hash(),
        "e2_source_hashes": source_values,
        "recipe": {"seed": SEED, "epochs": EPOCHS, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "micro_batch_size": MICRO_BATCH_SIZE, "max_length": MAX_LENGTH, "precision": "bf16", "attention_implementation": "sdpa", "gamma": GAMMA, "nc_lambdas": NC_LAMBDAS},
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "insertion_layer_index": INSERTION_LAYER_INDEX,
        "shape_rows": shape_rows,
        "task_shape_rows": task_shape_rows,
        "parameter_counts": parameter_counts,
        "gradient_rows": gradient_rows,
        "optimizer_coverage": coverage,
        "strict_roundtrip": roundtrip,
        "eligible_devices": devices,
        "gpu_snapshot": snapshot,
        "gpu_query_error": gpu_error,
        "full_training_launched": False,
    }
    _write_json(ORCH_ROOT / "preflight.json", payload)
    return payload


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _make_noisy_batch(tokenizer: Any, ids: Sequence[Any], text_by_id: Mapping[str, str], *, max_length: int, seed: int, epoch: int, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Any]]:
    texts = [text_by_id[str(example_id)] for example_id in ids]
    noise = noisy_batch(texts, ids, seed=seed, epoch=epoch)
    encoded = tokenizer(list(noise.texts), add_special_tokens=True, truncation=True, max_length=max_length, padding=False, return_attention_mask=False)
    batch = tokenizer.pad([{"input_ids": row} for row in encoded["input_ids"]], padding=True, return_tensors="pt")
    return ({key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)}, noise.to_dict())


def _run_validation(model: E11Classifier, loader: Any, device: torch.device, precision: str, task_id: str, bundle: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    counts = e1_train.task_aspect_class_counts(bundle)
    gold: list[Any] = []
    predicted: list[Any] = []
    ids: list[Any] = []
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
            predicted.extend(e1_train._predictions(logits, aspect_class_counts=counts))
            gold.extend(batch["gold_labels"].tolist())
            ids.extend(batch["ids"])
    metrics = e1_train.evaluate_predictions(task_id, bundle, gold, predicted, split="validation")
    rows = [{"id": example_id, "gold": target, "prediction": prediction} for example_id, target, prediction in zip(ids, gold, predicted)]
    return metrics, rows


def _train_loop_e11(*, model: E11Classifier, train_loader: Any, valid_loader: Any, bundle: Any, task_id: str, device: torch.device, precision: str, optimizer: torch.optim.Optimizer, scheduler: Any, epochs: int, effective_batch_size: int, micro_batch_size: int, mode: str, alpha: float | None, consistency_lambda: float | None, cache: Any, tokenizer: Any, train_split: Any, max_length: int, seed: int, artifact_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if mode != "nc" or cache is not None or alpha is not None or consistency_lambda is None:
        raise ValueError("E11 only permits frozen E8 NC objective")
    accumulation_steps = effective_batch_size // micro_batch_size
    total_steps = math.ceil(len(train_loader) / accumulation_steps) * epochs
    warmup_steps = round(total_steps * WARMUP_FRACTION)
    model.set_warmup_steps(max(1, warmup_steps))
    model.set_optimizer_step(0)
    metadata.update(model.architecture_metadata())
    metadata["gate_schedule"] = {"warmup_steps": max(1, warmup_steps), "step_unit": "optimizer_step", "clean_noisy_same_step": True}
    counts = e2_train.task_aspect_class_counts(bundle)
    num_aspects = e2_train.task_num_aspects(bundle)
    text_by_id = {str(example_id): text for example_id, text in zip(e2_train.split_ids(train_split), e2_train.split_texts(train_split))}
    _, nc_fn = e2_train._load_loss_api()
    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_predictions: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_examples = 0
        loss_totals = {"total": 0.0, "ce": 0.0, "consistency": 0.0}
        noise_stats: dict[str, Any] = {"count": 0, "changed_count": 0, "operation_counts": {}}
        started = time.time()
        for batch_index, batch in enumerate(train_loader):
            moved = _move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                clean_logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                ce = e2_train.compute_loss(clean_logits, moved["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                noisy, noise_info = _make_noisy_batch(tokenizer, batch["ids"], text_by_id, max_length=max_length, seed=seed, epoch=epoch, device=device)
                noisy_logits = model(noisy["input_ids"], noisy["attention_mask"], token_type_ids=noisy.get("token_type_ids"))
                consistency = nc_fn(clean_logits, noisy_logits, temperature=1.0, slot_class_counts=counts or None)
                total = ce + float(consistency_lambda) * consistency
                window_start = (batch_index // accumulation_steps) * accumulation_steps
                window_end = min(window_start + accumulation_steps, len(train_loader))
                window_examples = sum(min(micro_batch_size, len(train_loader.dataset) - index * micro_batch_size) for index in range(window_start, window_end))
                scaled = total * (len(batch["labels"]) / max(window_examples, 1))
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite E11 loss at epoch={epoch}, batch={batch_index}")
            scaled.backward()
            count = len(batch["labels"])
            total_examples += count
            loss_totals["total"] += float(total.detach().cpu()) * count
            loss_totals["ce"] += float(ce.detach().cpu()) * count
            loss_totals["consistency"] += float(consistency.detach().cpu()) * count
            noise_stats["count"] += int(noise_info["count"])
            noise_stats["changed_count"] += int(noise_info["changed_count"])
            for operation, operation_count in noise_info["operation_counts"].items():
                noise_stats["operation_counts"][operation] = noise_stats["operation_counts"].get(operation, 0) + int(operation_count)
            boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if boundary:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                model.set_optimizer_step(global_step)
        val_metrics, val_predictions = _run_validation(model, valid_loader, device, precision, task_id, bundle)
        score = e1_train.selection_score(val_metrics)
        noise_stats["change_rate"] = noise_stats["changed_count"] / max(noise_stats["count"], 1)
        row = {"epoch": epoch, "global_step": global_step, "train_losses": {key: value / max(total_examples, 1) for key, value in loss_totals.items()}, "validation": val_metrics, "selection_f1": score, "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - started, "noise": noise_stats, "gate_values": _attention(model).gate_values().detach().cpu().tolist(), "relative_correction_norm": float(_attention(model).last_correction_norm.item())}
        curves.append(row)
        _write_json(artifact_dir / "epoch_curves.json", {"epochs": curves})
        if score > best_score:
            best_score, best_epoch, best_metrics, best_predictions = score, epoch, val_metrics, val_predictions
            e2_train._save_model_checkpoint(artifact_dir / "best_checkpoint.pt", model, optimizer, scheduler, epoch=epoch, global_step=global_step, metadata=metadata)
            _write_json(artifact_dir / "best_validation_metrics.json", best_metrics)
            _write_json(artifact_dir / "validation_predictions.json", {"rows": best_predictions})
    metadata.update({"best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics, "elapsed_seconds": time.time() - metadata["started_at_unix"], "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None, "status": "completed", "noise_statistics": [row["noise"] for row in curves], "final_optimizer_step": global_step, "final_gate_values": _attention(model).gate_values().detach().cpu().tolist()})
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    if checkpoint_path.is_file():
        metadata["checkpoint_sha256"] = e1_train.sha256_file(checkpoint_path)
    _write_json(artifact_dir / "metadata.json", metadata)
    e2_train._set_status(artifact_dir, "completed", best_epoch=best_epoch, best_selection_f1=best_score)
    return {"task_id": task_id, "mode": mode, "seed": metadata["seed"], "learning_rate": metadata["learning_rate"], "artifact_dir": str(artifact_dir), "best_epoch": best_epoch, "best_selection_f1": best_score, "validation_metrics": best_metrics, "status": "completed"}


def _install_runtime_adapters(variant: str) -> dict[str, Any]:
    originals = {"builder": e2_train.build_e2_student, "train_loop": e2_train._train_loop, "optimizer": e2_train.make_optimizer}

    def builder(backbone_dir: Path, *, num_classes: int, num_aspects: int = 0, aspect_class_counts: Sequence[int] | None = None, attention_implementation: str = "sdpa"):
        return build_e11_student(backbone_dir, variant=variant, num_classes=num_classes, num_aspects=num_aspects, aspect_class_counts=aspect_class_counts if num_aspects else None, attention_implementation=attention_implementation, warmup_steps=1)

    def optimizer(model: nn.Module, learning_rate: float):
        value, coverage = _make_e11_optimizer(model, learning_rate)
        captured["optimizer"] = value
        captured["coverage"] = coverage
        return value

    e2_train.build_e2_student = builder  # type: ignore[assignment]
    e2_train._train_loop = _train_loop_e11  # type: ignore[assignment]
    captured: dict[str, Any] = {}
    e2_train.make_optimizer = optimizer  # type: ignore[assignment]
    return {**originals, "captured": captured}


def _restore_runtime_adapters(originals: Mapping[str, Any]) -> None:
    e2_train.build_e2_student = originals["builder"]  # type: ignore[assignment]
    e2_train._train_loop = originals["train_loop"]  # type: ignore[assignment]
    e2_train.make_optimizer = originals["optimizer"]  # type: ignore[assignment]


def run_worker(variant: str, task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if variant not in VARIANTS or task_id not in TASKS:
        raise ValueError(f"Unknown E11 trial {variant}/{task_id}")
    source_values = _source_hashes()
    source_hash = e11_source_hash()
    if MANIFEST_PATH.is_file() and _read_json(MANIFEST_PATH).get("e11_source_hash") != source_hash:
        raise RuntimeError("E11 source changed after manifest freeze")
    started_at = _now()
    base_status = {"protocol": PROTOCOL, "variant": variant, "task_id": task_id, "seed": SEED, "status": "running", "pid": os.getpid(), "device": device_name, "e11_source_hash": source_hash, "e2_source_hashes": source_values, "output_dir": str(output_dir), "started_at": started_at}
    _set_worker_status(variant, task_id, base_status)
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(30.0):
            _set_worker_status(variant, task_id, {**base_status, "heartbeat_at": _now()})

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    originals = _install_runtime_adapters(variant)
    try:
        result = e2_train.train_student_trial(task_id=task_id, mode="nc", repo_root=ROOT, output_dir=output_dir, seed=SEED, consistency_lambda=NC_LAMBDAS[task_id], learning_rate=TASK_LRS[task_id], device_name=device_name, precision="bf16", attention_implementation="sdpa", epochs=EPOCHS, effective_batch_size=EFFECTIVE_BATCH_SIZE, micro_batch_size=MICRO_BATCH_SIZE, stage=f"stageE11_{variant}", method=f"e11_{variant}_plus_e8_nc_llrd", coefficient=NC_LAMBDAS[task_id], revision=REVISION, expected_e2_code_hash=E2_CODE_HASH, expected_e2_config_hash=E2_CONFIG_HASH, expected_orchestrator_hash=source_values["orchestrator"])
    except Exception as exc:
        _set_worker_status(variant, task_id, {**base_status, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "finished_at": _now()})
        raise
    finally:
        stop_heartbeat.set()
        _restore_runtime_adapters(originals)
    captured = originals["captured"]
    artifact_dir = Path(str(result["artifact_dir"]))
    fields = {"e11_protocol": PROTOCOL, "e11_revision": REVISION, "e11_source_hash": source_hash, "e2_source_hashes": source_values, "variant": variant, "device": device_name, "worker_started_at": started_at, "optimizer_group_contract": captured["coverage"]["groups"]}
    for name in ("metadata.json", "resolved_config.json"):
        content = _read_json(artifact_dir / name)
        content.update(fields)
        _write_json(artifact_dir / name, content)
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = dict(checkpoint.get("metadata", {}))
    checkpoint_metadata.update(fields)
    checkpoint["metadata"] = checkpoint_metadata
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    _set_worker_status(variant, task_id, {**base_status, "status": "completed", "artifact_dir": str(artifact_dir), "best_epoch": result.get("best_epoch"), "best_selection_f1": result.get("best_selection_f1"), "finished_at": _now()})
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return result


def _recompute_validation(variant: str, task_id: str, artifact: Path) -> dict[str, Any]:
    predictions = _read_json(artifact / "validation_predictions.json").get("rows")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"Missing predictions for {variant}/{task_id}")
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
    valid = e1_train.split_field(bundle, "validation")
    expected_ids = [str(value) for value in e1_train.split_ids(valid)]
    if [str(row["id"]) for row in predictions] != expected_ids:
        raise ValueError(f"Validation IDs mismatch for {variant}/{task_id}")
    return e1_train.evaluate_predictions(task_id, bundle, [row["gold"] for row in predictions], [row["prediction"] for row in predictions], split="validation")


def _baseline_e8(task_id: str) -> dict[str, Any]:
    return _read_json(ROOT / "artifacts" / "e8" / task_id / "seed_42" / "best_validation_metrics.json")


def _summary_from_artifacts(artifacts: Mapping[tuple[str, str], Path]) -> dict[str, Any]:
    rows = []
    for variant, task_id in TRIALS:
        artifact = artifacts[(variant, task_id)]
        metrics = _recompute_validation(variant, task_id, artifact)
        baseline = _baseline_e8(task_id)
        metadata = _read_json(artifact / "metadata.json")
        candidate = float(metrics["selection_f1"])
        base = float(baseline["selection_f1"])
        rows.append({"variant": variant, "task_id": task_id, "seed": SEED, "artifact_dir": str(artifact), "best_epoch": metadata.get("best_epoch"), "selection_f1": candidate, "candidate_f1_percent": 100.0 * candidate, "e8_selection_f1": base, "candidate_minus_e8_pp": 100.0 * (candidate - base), "elapsed_seconds": metadata.get("elapsed_seconds"), "peak_gpu_memory_allocated_bytes": metadata.get("peak_gpu_memory_allocated_bytes"), "peak_gpu_memory_reserved_bytes": metadata.get("peak_gpu_memory_reserved_bytes"), "parameter_count": metadata.get("model", {}).get("total_parameters"), "gate_values": metadata.get("final_gate_values"), "relative_correction_norm": (metadata.get("epoch_curves", [{}])[-1].get("relative_correction_norm") if isinstance(metadata.get("epoch_curves"), list) else None), "metrics": metrics})
    by = {(row["variant"], row["task_id"]): row for row in rows}
    gate_rows = []
    for task_id in TASKS:
        signed = by[("signed", task_id)]
        convex = by[("convex", task_id)]
        gate_rows.append({"task_id": task_id, "signed_minus_e8_pp": signed["candidate_minus_e8_pp"], "convex_minus_e8_pp": convex["candidate_minus_e8_pp"], "signed_minus_convex_pp": signed["candidate_f1_percent"] - convex["candidate_f1_percent"]})
    signed_e8 = [row["signed_minus_e8_pp"] for row in gate_rows]
    signed_control = [row["signed_minus_convex_pp"] for row in gate_rows]
    gate = {"signed_vs_e8": {"tasks_gain": sum(value > 0 for value in signed_e8), "tasks_gain_at_least_0p3pp": sum(value >= 0.3 for value in signed_e8), "no_drop_worse_than_0p3pp": all(value >= -0.3 for value in signed_e8)}, "signed_vs_convex": {"tasks_win": sum(value > 0 for value in signed_control), "no_loss_worse_than_0p3pp": all(value >= -0.3 for value in signed_control)}}
    gate["passes_screening_heuristic"] = bool(gate["signed_vs_e8"]["tasks_gain"] >= 4 and gate["signed_vs_e8"]["tasks_gain_at_least_0p3pp"] >= 2 and gate["signed_vs_e8"]["no_drop_worse_than_0p3pp"] and gate["signed_vs_convex"]["tasks_win"] >= 3 and gate["signed_vs_convex"]["no_loss_worse_than_0p3pp"])
    payload = {"protocol": PROTOCOL, "revision": REVISION, "status": "completed", "created_at": _now(), "e11_source_hash": e11_source_hash(), "e2_source_hashes": _source_hashes(), "baseline_units": "E8 selection_f1 is fraction 0..1; display and pp deltas use *100", "rows": rows, "gate_rows": gate_rows, "gate": gate}
    _write_json(ARTIFACT_ROOT / "summary.json", payload)
    fields = ["variant", "task_id", "seed", "best_epoch", "selection_f1", "candidate_f1_percent", "e8_selection_f1", "candidate_minus_e8_pp", "elapsed_seconds", "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes", "parameter_count"]
    with (ARTIFACT_ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    return payload


def _adopt_or_requeue() -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], dict[str, Any]]]:
    completed: dict[tuple[str, str], Path] = {}
    active: dict[tuple[str, str], dict[str, Any]] = {}
    for variant, task_id in TRIALS:
        artifact = _completed_artifact(variant, task_id)
        if artifact is not None:
            completed[(variant, task_id)] = artifact
            continue
        worker_path = _worker_path(variant, task_id)
        if worker_path.is_file():
            worker = _read_json(worker_path)
            if worker.get("status") == "running" and worker.get("e11_source_hash") == e11_source_hash() and _pid_alive(worker.get("pid")):
                active[(variant, task_id)] = worker
    return completed, active


def run_all(*, poll_seconds: float = 15.0) -> dict[str, Any]:
    for directory in (ARTIFACT_ROOT, ORCH_ROOT, LOG_ROOT, WORKER_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    source_values = _source_hashes()
    devices, snapshot = _eligible_devices()
    if not devices:
        raise RuntimeError(f"No GPU meets {GPU_GUARD_MIB} MiB guard: {snapshot}")
    preflight_path = ORCH_ROOT / "preflight.json"
    if not preflight_path.is_file() or _read_json(preflight_path).get("status") != "passed":
        raise RuntimeError("Run --preflight before --run-all")
    if _read_json(preflight_path).get("e11_source_hash") != e11_source_hash():
        raise RuntimeError("E11 source changed after preflight")
    manifest = {"protocol": PROTOCOL, "revision": REVISION, "frozen_at": _now(), "e11_source_hash": e11_source_hash(), "e2_source_hashes": source_values, "gpu_snapshot_at_freeze": snapshot, "tasks": list(TASKS), "variants": list(VARIANTS), "seed": SEED, "epochs": EPOCHS}
    if MANIFEST_PATH.is_file() and _read_json(MANIFEST_PATH).get("e11_source_hash") != manifest["e11_source_hash"]:
        raise RuntimeError("Existing source manifest does not match code")
    _write_json(MANIFEST_PATH, manifest)
    completed, adopted = _adopt_or_requeue()
    state_path = ORCH_ROOT / "state.json"
    if state_path.is_file():
        old = _read_json(state_path)
        if old.get("status") == "running" and old.get("controller_pid") != os.getpid() and _pid_alive(old.get("controller_pid")):
            raise RuntimeError(f"E11 controller already active: PID {old.get('controller_pid')}")
    if len(completed) == len(TRIALS):
        return _summary_from_artifacts(completed)
    jobs = {_trial_key(*key): {"variant": key[0], "task_id": key[1], "status": "completed", "artifact_dir": str(path)} for key, path in completed.items()}
    running_adopted = dict(adopted)
    assigned_devices = {str(worker.get("device")) for worker in adopted.values()}
    queue = [key for key in TRIALS if key not in completed and key not in adopted]
    state = {"protocol": PROTOCOL, "status": "running", "controller_pid": os.getpid(), "started_at": _now(), "e11_source_hash": e11_source_hash(), "e2_source_hashes": source_values, "gpu_snapshot_at_launch": snapshot, "eligible_devices": devices, "trials": [list(key) for key in TRIALS], "jobs": jobs, "adopted_active": [list(key) for key in adopted]}
    _write_json(state_path, state)
    controller_log = (LOG_ROOT / "controller.log").open("a", encoding="utf-8", buffering=1)
    running: dict[tuple[str, str], tuple[subprocess.Popen[str], str]] = {}

    def launch(key: tuple[str, str], device_name: str) -> None:
        variant, task_id = key
        output_dir = _trial_dir(variant, task_id)
        log_path = LOG_ROOT / f"{_trial_key(*key)}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--variant", variant, "--task-id", task_id, "--device", device_name, "--output-dir", str(output_dir)]
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(json.dumps({"event": "launch", "time": _now(), "command": command}) + "\n")
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, text=True)
        handle.close()
        running[key] = (process, device_name)
        jobs[_trial_key(*key)] = {"variant": variant, "task_id": task_id, "device": device_name, "pid": process.pid, "artifact_dir": str(output_dir), "log_path": str(log_path), "status": "running", "launched_at": _now(), "command": command}
        assigned_devices.add(device_name)
        controller_log.write(json.dumps({"event": "launch", "trial": list(key), "pid": process.pid, "device": device_name}) + "\n")
        state["jobs"] = jobs
        _write_json(state_path, state)

    try:
        while queue or running or running_adopted:
            current = _gpu_snapshot()
            free = [f"cuda:{index}" for index in PREFERRED_GPU_INDICES if index in current and current[index]["free_mib"] >= GPU_GUARD_MIB and f"cuda:{index}" not in assigned_devices]
            while queue and free:
                launch(queue.pop(0), free.pop(0))
            finished = []
            for key, (process, device_name) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                trial = _trial_key(*key)
                assigned_devices.discard(device_name)
                jobs[trial].update({"status": "completed" if code == 0 else "failed", "returncode": code, "finished_at": _now()})
                if code != 0:
                    state.update({"status": "failed", "failed_trial": list(key), "finished_at": _now()})
                    _write_json(state_path, state)
                    raise RuntimeError(f"E11 worker failed: {trial}; inspect {jobs[trial].get('log_path')}")
                artifact = _completed_artifact(*key)
                if artifact is None:
                    raise RuntimeError(f"E11 worker exited without artifact: {trial}")
                completed[key] = artifact
                finished.append(key)
                controller_log.write(json.dumps({"event": "finish", "trial": list(key), "returncode": code}) + "\n")
            for key in finished:
                running.pop(key)
            for key in list(running_adopted):
                worker_path = _worker_path(*key)
                worker = _read_json(worker_path) if worker_path.is_file() else {}
                if worker.get("status") == "completed":
                    artifact = _completed_artifact(*key)
                    if artifact is None:
                        raise RuntimeError(f"Adopted worker completed without artifact: {key}")
                    completed[key] = artifact
                    assigned_devices.discard(str(worker.get("device")))
                    running_adopted.pop(key)
                elif worker.get("status") == "failed" or not _pid_alive(worker.get("pid")):
                    assigned_devices.discard(str(worker.get("device")))
                    running_adopted.pop(key)
                    queue.insert(0, key)
            state.update({"jobs": jobs, "completed": [list(key) for key in completed], "queue": [list(key) for key in queue], "adopted_active": [list(key) for key in running_adopted], "heartbeat_at": _now()})
            _write_json(state_path, state)
            if queue or running or running_adopted:
                time.sleep(poll_seconds)
    finally:
        controller_log.close()
    summary = _summary_from_artifacts(completed)
    state.update({"status": "completed", "finished_at": _now(), "summary": str(ARTIFACT_ROOT / "summary.json")})
    _write_json(state_path, state)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E11 signed versus convex two-map attention screening")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight:
            payload = run_preflight()
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0 if payload.get("status") == "passed" else 1
        if args.variant or args.task_id:
            if not args.variant or not args.task_id or args.output_dir is None:
                raise ValueError("worker mode requires --variant, --task-id and --output-dir")
            run_worker(args.variant, args.task_id, args.device, args.output_dir.resolve())
            return 0
        if args.run_all:
            payload = run_all(poll_seconds=args.poll_seconds)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0
        raise ValueError("Choose one of --preflight, --run-all, or worker arguments")
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
