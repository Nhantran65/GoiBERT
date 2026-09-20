#!/usr/bin/env python3
"""E17 preflight, profile, and trial runner.

The E2 data/loss loop and E7 schedule remain imported dependencies.  This
runner only patches the student builder and optimizer inside one worker
process, writes E17-isolated artifacts, and refuses full dispatch until the
caller explicitly uses ``--run-all``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e17"
CONFIG_PATH = ROOT / "configs" / "e17" / "protocol.json"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e17.modeling import ARMS, E17Classifier, build_e17_student  # noqa: E402
from goibert_e17.optim import build_e17_optimizer  # noqa: E402


PROTOCOL = "E17-recipient-excluded-keys-screening-v1"
REVISION = "e17-20260913-r1"
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
ARMS_ORDER = ("baseline", "full_context", "cavity")
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
MAX_LENGTH = 256
HIDDEN_SIZE = 384
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
GAMMA = 0.8
WEIGHT_DECAY = 0.01
GUARD_MIB = 5 * 1024
RESERVE_HEADROOM_MIB = 4 * 1024
PREFERRED_GPU_INDICES = ("2", "1", "0")
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
    command = ["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as exc:
        return {"available": False, "error": repr(exc), "rows": {}}
    rows: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 3:
            rows[fields[0]] = {"free_mib": int(float(fields[1])), "used_mib": int(float(fields[2]))}
    return {"available": bool(rows), "rows": rows}


def _eligible_devices(snapshot: Mapping[str, Any]) -> list[str]:
    rows = snapshot.get("rows", {})
    return [
        f"cuda:{index}"
        for index in PREFERRED_GPU_INDICES
        if index in rows and int(rows[index]["free_mib"]) >= GUARD_MIB
    ]


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


def _source_manifest() -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        ROOT / "src" / "goibert_e17" / "modeling.py",
        ROOT / "src" / "goibert_e17" / "optim.py",
        ROOT / "src" / "goibert_e17" / "__init__.py",
        CONFIG_PATH,
        ROOT / "src" / "goibert_e2" / "train.py",
        ROOT / "src" / "goibert_e2" / "modeling.py",
        ROOT / "src" / "goibert_e7" / "optim.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in source_files}
    return {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "created_at": _now(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "backbone": {
            "path": str((ROOT / "models" / "goiturn-s-gd").resolve()),
            "files": _directory_hashes(ROOT / "models" / "goiturn-s-gd"),
        },
        "frozen_e2_source_hashes": e2_train.source_hashes(ROOT),
        "frozen_e7_files": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in sorted((ROOT / "src" / "goibert_e7").rglob("*.py"))
        },
        "recipe": {
            "tasks": list(TASKS),
            "arms": list(ARMS_ORDER),
            "seed": SEED,
            "epochs": EPOCHS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "max_length": MAX_LENGTH,
            "precision": PRECISION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "gamma": GAMMA,
            "weight_decay": WEIGHT_DECAY,
        },
    }


def _stable_source_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": manifest["protocol"],
        "revision": manifest["revision"],
        "runner_sha256": manifest["runner_sha256"],
        "source_hashes": manifest["source_hashes"],
        "backbone": manifest["backbone"],
        "frozen_e2_source_hashes": manifest["frozen_e2_source_hashes"],
        "frozen_e7_files": manifest["frozen_e7_files"],
        "recipe": manifest["recipe"],
    }


def _read_root_approval(manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = ARTIFACT_ROOT / "root_approval.json"
    if not path.is_file():
        raise RuntimeError(f"Missing root approval file: {path}")
    approval = _read_json(path)
    if approval.get("status") != "approved":
        raise RuntimeError(f"Root approval status is not approved: {approval.get('status')!r}")
    expected = _stable_source_binding(manifest)
    if approval.get("source_binding") != expected:
        raise RuntimeError("Root approval source_binding does not match current E17 source")
    return approval


def _dispatch_vram_gate(device: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    profile_path = ARTIFACT_ROOT / "profile.json"
    if not profile_path.is_file():
        raise RuntimeError("E17 profile.json is required before dispatch")
    profile = _read_json(profile_path)
    if profile.get("status") != "passed":
        raise RuntimeError(f"E17 profile gate is not passed: {profile.get('status')!r}")
    max_reserved = max(int(row["peak_reserved_bytes"]) for row in profile.get("rows", []))
    index = device.split(":", 1)[1]
    free_mib = int(snapshot.get("rows", {}).get(index, {}).get("free_mib", 0))
    required_mib = math.ceil(max_reserved / (1024 * 1024)) + RESERVE_HEADROOM_MIB
    gate = {"device": device, "free_mib": free_mib, "profile_peak_reserved_bytes": max_reserved, "required_free_mib": required_mib, "pass": free_mib >= required_mib}
    if not gate["pass"]:
        raise RuntimeError(f"E17 per-launch VRAM gate failed: {gate}")
    return gate


def _task_contract(task_id: str) -> dict[str, Any]:
    bundle = e2_train.load_task_bundle(task_id, ROOT, include_test=False, smoke=False)
    train_split = e2_train.split_field(bundle, "train")
    valid_split = e2_train.split_field(bundle, "validation")
    return {
        "bundle": bundle,
        "train": train_split,
        "validation": valid_split,
        "num_aspects": e2_train.task_num_aspects(bundle),
        "aspect_class_counts": e2_train.task_aspect_class_counts(bundle),
        "num_classes": e2_train.task_num_classes(bundle),
    }


def _build(arm: str, contract: Mapping[str, Any]) -> tuple[E17Classifier, Any]:
    return build_e17_student(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=int(contract["num_classes"]),
        num_aspects=int(contract["num_aspects"]),
        aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
        arm=arm,
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def _small_batch(device: torch.device, *, batch_size: int = 2, seq_len: int = 8) -> tuple[Tensor, Tensor]:
    return (
        torch.randint(0, 20481, (batch_size, seq_len), device=device),
        torch.tensor([[1] * (seq_len - 3) + [0] * 3, [1] * seq_len][:batch_size], dtype=torch.long, device=device),
    )


def _literal_scores_from_preview_fixed(model: E17Classifier) -> Tensor:
    preview = model.target_self.last_preview
    z, u, a, query = preview["z"], preview["u"], preview["a"], preview["query"].float()
    cavity = z[:, None, :, :] - torch.einsum("bijh,bihd->bijd", a, u)
    ln = F.layer_norm(cavity, (model.target_self.all_head_size,), model.target_attention.output.LayerNorm.weight.float(), model.target_attention.output.LayerNorm.bias.float(), 1.0e-5)
    kw = model.target_self.key.weight.float().reshape(model.target_self.num_attention_heads, model.target_self.attention_head_size, model.target_self.all_head_size)
    kb = model.target_self.key.bias.float().reshape(model.target_self.num_attention_heads, model.target_self.attention_head_size)
    keys = torch.einsum("hpd,bijd->bijhp", kw, ln) + kb.view(1, 1, 1, model.target_self.num_attention_heads, model.target_self.attention_head_size)
    q_bihp = query.permute(0, 2, 1, 3)
    return torch.einsum("bihp,bijhp->bhij", q_bihp, keys) / math.sqrt(float(model.target_self.attention_head_size))


def _max_abs(left: Tensor, right: Tensor) -> float:
    return float((left.float() - right.float()).abs().max().detach().cpu().item())


def _literal_factorized_check(device: torch.device) -> dict[str, Any]:
    contract = {"num_classes": 3, "num_aspects": 0, "aspect_class_counts": ()}
    model, _ = _build("cavity", contract)
    model.to(device).train()
    model.set_diagnostics(True)
    input_ids, mask = _small_batch(device, batch_size=1, seq_len=6)
    model.target_attention.set_query_mask(mask)
    hidden = torch.randn(1, 6, HIDDEN_SIZE, device=device, requires_grad=True)
    additive_mask = (1.0 - mask[:, None, None, :].float()) * -1.0e4
    context, _ = model.target_self.route(hidden, additive_mask, output_dense=model.target_attention.output.dense, target_ln=model.target_attention.output.LayerNorm)
    preview = model.target_self.last_preview
    factorized = preview["selected_scores"]
    literal = _literal_scores_from_preview_fixed(model)
    pair_mask = mask[:, :, None].bool() & mask[:, None, :].bool()
    pair_mask = pair_mask[:, None, :, :].expand_as(factorized)
    forward_error = _max_abs(factorized[pair_mask], literal[pair_mask])
    cotangent = torch.randn_like(factorized) * pair_mask
    factor_loss = (factorized * cotangent).sum()
    literal_loss = (literal * cotangent).sum()
    target_params = [hidden, model.target_self.query.weight, model.target_self.key.weight, model.target_self.value.weight]
    factor_grads = torch.autograd.grad(factor_loss, target_params, retain_graph=True, allow_unused=False)
    literal_grads = torch.autograd.grad(literal_loss, target_params, retain_graph=False, allow_unused=False)
    backward_errors = [float((a - b).abs().max().detach().cpu().item()) for a, b in zip(factor_grads, literal_grads)]
    model.target_attention.set_query_mask(None)
    return {
        "device": str(device),
        "input_shape": list(input_ids.shape),
        "num_attention_heads": model.target_self.num_attention_heads,
        "attention_head_dim": model.target_self.attention_head_size,
        "qkv_weight_shapes": {
            "query": list(model.target_self.query.weight.shape),
            "key": list(model.target_self.key.weight.shape),
            "value": list(model.target_self.value.weight.shape),
        },
        "forward_max_abs": forward_error,
        "backward_max_abs_by_group": backward_errors,
        "finite": bool(torch.isfinite(factorized).all().item() and all(torch.isfinite(g).all().item() for g in factor_grads + literal_grads)),
        "chunk_size": model.chunk_size,
    }


def _bias_pad_diagonal_check(device: torch.device) -> dict[str, Any]:
    contract = {"num_classes": 3, "num_aspects": 0, "aspect_class_counts": ()}
    model, _ = _build("cavity", contract)
    model.to(device).eval(); model.set_diagnostics(True)
    input_ids, mask = _small_batch(device, batch_size=1, seq_len=6)
    with torch.no_grad():
        _ = model(input_ids, mask)
    preview = model.target_self.last_preview
    probs = preview["probabilities"]
    a = preview["a"]
    diag = torch.diagonal(a, dim1=1, dim2=2).abs().max()
    pad_prob = probs[:, :, :, ~mask[0].bool()].abs().max()
    hidden = preview["hidden"]
    content = torch.einsum("bnhp,hdp->bnhd", preview["content_value"].permute(0, 2, 1, 3).float(), model.target_attention.output.dense.weight.float().reshape(HIDDEN_SIZE, 6, 64).permute(1, 0, 2))
    bias_v = model.target_self.value.bias.float().reshape(6, 64)
    wo = model.target_attention.output.dense.weight.float().reshape(HIDDEN_SIZE, 6, 64).permute(1, 0, 2)
    direct_u = content + torch.einsum("hdp,hp->hd", wo, bias_v).view(1, 1, 6, HIDDEN_SIZE)
    direct_z = preview["z"] * 0.0 + hidden.float() + model.target_attention.output.dense.bias.float().view(1, 1, HIDDEN_SIZE) + torch.einsum("bhjk,bkhd->bjd", probs, direct_u)
    bias_error = _max_abs(direct_z, preview["z"])
    return {
        "target_layer_norm_reused": model.target_attention.output.LayerNorm is model.encoder.encoder.layer[6].attention.output.LayerNorm,
        "layer_norm_eps": float(model.target_attention.output.LayerNorm.eps),
        "bias_relocation_max_abs": bias_error,
        "max_cavity_a_diagonal": float(diag.detach().cpu().item()),
        "max_padded_key_probability": float(pad_prob.detach().cpu().item()),
        "finite": bool(torch.isfinite(direct_z).all().item()),
    }


def _eta0_and_backward_checks(device: torch.device) -> dict[str, Any]:
    contract = {"num_classes": 3, "num_aspects": 0, "aspect_class_counts": ()}
    models: dict[str, E17Classifier] = {}
    for arm in ARMS_ORDER:
        model, _ = _build(arm, contract)
        model.to(device).eval()
        models[arm] = model
    base_state = models["baseline"].state_dict()
    for arm in ("full_context", "cavity"):
        missing = models[arm].load_state_dict(base_state, strict=False)
        if missing.unexpected_keys or set(missing.missing_keys) != {"encoder.encoder.layer.6.attention.self.eta"}:
            raise AssertionError(f"Unexpected eta0 state mismatch for {arm}: {missing}")
        assert models[arm].eta is not None
        models[arm].eta.data.zero_()
    input_ids, mask = _small_batch(device, batch_size=2, seq_len=8)
    with torch.no_grad():
        base_logits = models["baseline"](input_ids, mask)
        full_logits = models["full_context"](input_ids, mask)
        cavity_logits = models["cavity"](input_ids, mask)
    eta0_errors = {arm: _max_abs(base_logits, logits) for arm, logits in (("full_context", full_logits), ("cavity", cavity_logits))}

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            with torch.no_grad():
                bf16_base_logits = models["baseline"](input_ids, mask)
                bf16_full_logits = models["full_context"](input_ids, mask)
                bf16_cavity_logits = models["cavity"](input_ids, mask)
        bf16_eta0_errors = {
            "full_context": _max_abs(bf16_base_logits, bf16_full_logits),
            "cavity": _max_abs(bf16_base_logits, bf16_cavity_logits),
        }
    else:
        bf16_eta0_errors = None

    train_model, _ = _build("cavity", contract)
    train_model.to(device).train(); train_model.set_diagnostics(True)
    optimizer, coverage = build_e17_optimizer(train_model, 1e-5, gamma=GAMMA, num_layers=10, weight_decay=WEIGHT_DECAY)
    labels = torch.tensor([0, 1], device=device)
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = train_model(input_ids, mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
    else:
        logits = train_model(input_ids, mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    eta_grad = train_model.eta.grad
    old_grad = train_model.target_self.query.weight.grad
    return {
        "eta0_errors": eta0_errors,
        "eta0_max_abs": max(eta0_errors.values()),
        "eta0_pass": max(eta0_errors.values()) <= 1.0e-5,
        "bf16_eta0_errors": bf16_eta0_errors,
        "bf16_eta0_pass": None if bf16_eta0_errors is None else max(bf16_eta0_errors.values()) <= 1.0e-5,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "eta_gradient_finite": bool(eta_grad is not None and torch.isfinite(eta_grad).all().item()),
        "eta_gradient_nonzero": bool(eta_grad is not None and eta_grad.norm().item() > 0),
        "old_weight_gradient_finite": bool(old_grad is not None and torch.isfinite(old_grad).all().item()),
        "optimizer_coverage": coverage,
        "eta_param_shape": list(train_model.eta.shape),
        "eta_param_name": "encoder.encoder.layer.6.attention.self.eta",
        "eta_zero_weight_decay": coverage.get("eta_weight_decay") == 0.0,
        "eta_block6_lr": coverage.get("eta_learning_rate"),
    }


def _all_task_output_shape_check(device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        contract = _task_contract(task_id)
        model, spec = _build("cavity", contract)
        model.to(device).eval()
        input_ids, mask = _small_batch(device, batch_size=1, seq_len=6)
        with torch.no_grad():
            logits = model(input_ids, mask)
        rows.append(
            {
                "task_id": task_id,
                "num_classes": int(contract["num_classes"]),
                "num_aspects": int(contract["num_aspects"]),
                "aspect_class_counts": list(contract["aspect_class_counts"]),
                "logits_shape": list(logits.shape),
                "finite": bool(torch.isfinite(logits).all().item()),
                "model_spec": spec.to_dict(),
            }
        )
        del model, logits
    return {"task_count": len(rows), "rows": rows, "all_finite": all(row["finite"] for row in rows)}


def _strict_reload_check(device: torch.device) -> dict[str, Any]:
    contract = {"num_classes": 3, "num_aspects": 0, "aspect_class_counts": ()}
    model, spec = _build("cavity", contract)
    model.to(device)
    path = ARTIFACT_ROOT / "preflight" / "strict_reload.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "spec": spec.to_dict()}, path)
    reloaded, _ = _build("cavity", contract)
    state = torch.load(path, map_location=device, weights_only=False)
    result = reloaded.load_state_dict(state["model"], strict=True)
    return {"path": str(path), "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys), "strict_pass": not result.missing_keys and not result.unexpected_keys}


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
    before_backbone = manifest["backbone"]["files"]
    functional = {
        "literal_factorized": _literal_factorized_check(device),
        "bias_pad_diagonal": _bias_pad_diagonal_check(device),
        "eta0_backward_optimizer": _eta0_and_backward_checks(device),
        "all_task_output_shapes": _all_task_output_shape_check(device),
        "strict_reload": _strict_reload_check(device),
    }
    after_backbone = _directory_hashes(ROOT / "models" / "goiturn-s-gd")
    checks = {
        "functional_literal_factorized": functional["literal_factorized"]["finite"] and functional["literal_factorized"]["forward_max_abs"] <= 1e-5 and max(functional["literal_factorized"]["backward_max_abs_by_group"]) <= 1e-5,
        "target_attention_heads": functional["literal_factorized"]["num_attention_heads"] == 6 and functional["literal_factorized"]["attention_head_dim"] == 64 and all(tuple(shape) == (384, 384) for shape in functional["literal_factorized"]["qkv_weight_shapes"].values()),
        "bias_relocation": functional["bias_pad_diagonal"]["bias_relocation_max_abs"] <= 1e-5,
        "target_layer_norm_location": functional["bias_pad_diagonal"]["target_layer_norm_reused"] and abs(functional["bias_pad_diagonal"]["layer_norm_eps"] - 1e-5) <= 1e-12,
        "pad_and_diagonal": functional["bias_pad_diagonal"]["max_cavity_a_diagonal"] == 0.0 and functional["bias_pad_diagonal"]["max_padded_key_probability"] == 0.0,
        "eta0_parity": functional["eta0_backward_optimizer"]["eta0_pass"],
        "first_backward_eta": functional["eta0_backward_optimizer"]["eta_gradient_finite"] and functional["eta0_backward_optimizer"]["eta_gradient_nonzero"],
        "bf16_eta0_and_backward": (
            functional["eta0_backward_optimizer"]["bf16_eta0_pass"] is not False
            and functional["eta0_backward_optimizer"]["eta_gradient_finite"]
            and functional["eta0_backward_optimizer"]["eta_gradient_nonzero"]
        ),
        "old_weight_gradient": functional["eta0_backward_optimizer"]["old_weight_gradient_finite"],
        "all_task_output_shapes": functional["all_task_output_shapes"]["task_count"] == len(TASKS) and functional["all_task_output_shapes"]["all_finite"],
        "optimizer_unique_lr": functional["eta0_backward_optimizer"]["optimizer_coverage"]["all_trainable"] and functional["eta0_backward_optimizer"]["optimizer_coverage"]["duplicate_parameter_count"] == 0 and functional["eta0_backward_optimizer"]["eta_zero_weight_decay"],
        "strict_checkpoint_reload": functional["strict_reload"]["strict_pass"],
        "backbone_no_mutation": before_backbone == after_backbone,
    }
    gpu_ready = device.type == "cuda" and snapshot.get("available", False)
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
        "functional": functional,
        "source_manifest": manifest,
        "backbone_hash_unchanged": before_backbone == after_backbone,
        "full_training_dispatch": "not launched; root review required",
    }
    _write_json(ARTIFACT_ROOT / "preflight.json", payload)
    _write_json(ARTIFACT_ROOT / "source_manifest.json", manifest)
    return payload


def run_plumbing_check() -> dict[str, Any]:
    manifest = _source_manifest()
    _write_json(ARTIFACT_ROOT / "source_manifest.json", manifest)
    binding = _stable_source_binding(manifest)
    altered_manifest = dict(manifest)
    altered_manifest["created_at"] = "different-created-at"
    before = torch.random.get_rng_state()
    with torch.random.fork_rng(devices=[]):
        _ = torch.rand(17)
    after = torch.random.get_rng_state()
    queue = [(task, arm) for task in TASKS for arm in ARMS_ORDER]
    selected_steps: list[int] = []
    for step in range(1, 101):
        if step in {1, 10, 100}:
            selected_steps.append(step)
    result = {
        "protocol": PROTOCOL,
        "status": "passed" if binding == _stable_source_binding(altered_manifest) and torch.equal(before, after) and queue[:3] == [(TASKS[0], arm) for arm in ARMS_ORDER] else "failed",
        "stable_binding_ignores_created_at": binding == _stable_source_binding(altered_manifest),
        "fork_rng_restores_cpu_state": bool(torch.equal(before, after)),
        "queue_task_major": queue[:3] == [(TASKS[0], arm) for arm in ARMS_ORDER],
        "queue_length": len(queue),
        "dispatch_requires_approval_file_and_flag": "reviewed_not_exercised",
        "worker_startup_source_binding_gate": "reviewed_not_exercised",
        "epoch_telemetry_path": "artifact/epoch_telemetry.json via run_validation wrapper (runtime reviewed_not_exercised)",
        "optimizer_post_hook_steps": {"expected": [1, 10, 100], "countercheck_selected": selected_steps, "runtime": "reviewed_not_exercised"},
        "source_binding": binding,
    }
    _write_json(ARTIFACT_ROOT / "plumbing_check.json", result)
    return result


def _fixed_profile_batch(task_id: str) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    contract = _task_contract(task_id)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
    train_split = contract["train"]
    valid_split = contract["validation"]
    train_loader, _, _ = e2_train._loader_set(
        tokenizer,
        train_split,
        valid_split,
        contract["bundle"],
        seed=SEED,
        micro_batch_size=MICRO_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    batch = next(iter(train_loader))
    text_by_id = {str(example_id): text for example_id, text in zip(e2_train.split_ids(train_split), e2_train.split_texts(train_split))}
    noisy, noise_stats = e2_train._make_noisy_batch(
        tokenizer,
        batch["ids"],
        text_by_id,
        max_length=MAX_LENGTH,
        seed=SEED,
        epoch=1,
        device=torch.device("cpu"),
    )
    return tokenizer, batch, contract, {"noisy": noisy, "noise_stats": noise_stats}


def _profile_one(arm: str, device: torch.device, task_id: str, shape_name: str, clean: dict[str, Tensor], noisy: dict[str, Tensor], contract: Mapping[str, Any]) -> dict[str, Any]:
    model, spec = _build(arm, contract)
    model.to(device).train(); model.set_diagnostics(False)
    optimizer, coverage = build_e17_optimizer(model, TASK_LRS[task_id], gamma=GAMMA, num_layers=10, weight_decay=WEIGHT_DECAY)
    clean = {key: value.to(device) for key, value in clean.items()}
    noisy = {key: value.to(device) for key, value in noisy.items()}
    counts = e2_train.task_aspect_class_counts(contract["bundle"])
    num_aspects = e2_train.task_num_aspects(contract["bundle"])
    nc_fn = e2_train._load_loss_api()[1]

    def one_step(current_model: E17Classifier, current_optimizer: torch.optim.Optimizer) -> tuple[float, float]:
        current_optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        last_total = None
        for _ in range(ACCUMULATION_STEPS):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                clean_logits = current_model(clean["input_ids"], clean["attention_mask"])
                noisy_logits = current_model(noisy["input_ids"], noisy["attention_mask"])
                ce = e2_train.compute_loss(clean_logits, clean["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                consistency = nc_fn(clean_logits, noisy_logits, temperature=1.0, slot_class_counts=counts or None)
                total = ce + NC_LAMBDAS[task_id] * consistency
            if not torch.isfinite(total):
                raise FloatingPointError(f"E17 profile loss non-finite: {arm}/{shape_name}")
            (total / ACCUMULATION_STEPS).backward()
            last_total = total
        torch.nn.utils.clip_grad_norm_(current_model.parameters(), 1.0, error_if_nonfinite=True)
        current_optimizer.step()
        torch.cuda.synchronize(device)
        assert last_total is not None
        return time.perf_counter() - started, float(last_total.detach().cpu().item())

    for _ in range(2):
        one_step(model, optimizer)
    measured_model, measured_spec = model, spec
    measured_optimizer, measured_coverage = optimizer, coverage
    torch.cuda.reset_peak_memory_stats(device)
    measured_steps: list[float] = []
    loss_value = 0.0
    for _ in range(3):
        step_seconds, loss_value = one_step(measured_model, measured_optimizer)
        measured_steps.append(step_seconds)
    step_seconds = float(torch.tensor(measured_steps).median().item())
    measured_model.set_diagnostics(True)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            measured_model(clean["input_ids"], clean["attention_mask"])
            measured_model(noisy["input_ids"], noisy["attention_mask"])
    stats = measured_model.diagnostics()
    total = torch.tensor(loss_value, device=device)
    result = {
        "arm": arm,
        "task_id": task_id,
        "shape_name": shape_name,
        "input_shape": list(clean["input_ids"].shape),
        "step_seconds": step_seconds,
        "measured_step_seconds": measured_steps,
        "loss": float(total.detach().cpu().item()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "fallback_stats": stats,
        "model_spec": measured_spec.to_dict(),
        "optimizer_coverage": measured_coverage,
        "finite": bool(torch.isfinite(total).item()),
    }
    del measured_model, measured_optimizer, total
    torch.cuda.empty_cache()
    return result


def run_profile(device_name: str | None = None, task_id: str = "vsfc_sentiment") -> dict[str, Any]:
    snapshot = _gpu_snapshot()
    manifest = _source_manifest()
    if device_name is None:
        eligible = _eligible_devices(snapshot)
        device_name = eligible[0] if eligible else "cuda:0"
    if not torch.cuda.is_available():
        payload = {"protocol": PROTOCOL, "status": "blocked_no_cuda", "gpu_snapshot": snapshot, "requested_device": device_name, "source_manifest": manifest, "reason": "CUDA unavailable in this process; no profile launched"}
        _write_json(ARTIFACT_ROOT / "profile.json", payload)
        return payload
    device = torch.device(device_name)
    tokenizer, actual_batch, contract, noisy_payload = _fixed_profile_batch(task_id)
    clean_actual = {key: value for key, value in actual_batch.items() if isinstance(value, torch.Tensor)}
    noisy_actual = noisy_payload["noisy"]
    synthetic_ids = torch.randint(0, 20481, (MICRO_BATCH_SIZE, MAX_LENGTH), dtype=torch.long)
    synthetic_mask = torch.ones_like(synthetic_ids)
    synthetic_labels = torch.arange(MICRO_BATCH_SIZE, dtype=torch.long) % int(contract["num_classes"])
    synthetic = {"input_ids": synthetic_ids, "attention_mask": synthetic_mask, "labels": synthetic_labels}
    profile_rows = []
    for shape_name, clean, noisy in (
        ("fixed_train_batch", clean_actual, noisy_actual),
        ("synthetic_16x256", synthetic, {"input_ids": synthetic_ids.clone(), "attention_mask": synthetic_mask.clone(), "labels": synthetic_labels.clone()}),
    ):
        for arm in ARMS_ORDER:
            profile_rows.append(_profile_one(arm, device, task_id, shape_name, clean, noisy, contract))
    by_shape = {shape: {row["arm"]: row for row in profile_rows if row["shape_name"] == shape} for shape in ("fixed_train_batch", "synthetic_16x256")}
    gates = {}
    for shape, rows in by_shape.items():
        baseline = rows["baseline"]["step_seconds"]
        cavity = rows["cavity"]["step_seconds"]
        free_mib = snapshot.get("rows", {}).get(device_name.split(":")[-1], {}).get("free_mib")
        gates[shape] = {
            "cavity_to_baseline_step_ratio": cavity / max(baseline, 1e-12),
            "under_3x_baseline": cavity <= 3.0 * baseline,
            "reserve_under_free_minus4GiB": free_mib is None or max(row["peak_reserved_bytes"] for row in rows.values()) <= (free_mib - RESERVE_HEADROOM_MIB) * 1024 * 1024,
            "free_mib_before": free_mib,
        }
    payload = {"protocol": PROTOCOL, "revision": REVISION, "status": "passed" if all(g["under_3x_baseline"] and g["reserve_under_free_minus4GiB"] for g in gates.values()) else "failed", "device": device_name, "gpu_snapshot": snapshot, "task_id": task_id, "rows": profile_rows, "budget_gates": gates, "source_manifest": manifest, "telemetry_probe_precision": "bf16 autocast; timing also bf16 autocast; diagnostics probe is no_grad/eval-equivalent model probe", "note": "Profile only; full 15-trial dispatch remains root-review gated."}
    _write_json(ARTIFACT_ROOT / "profile.json", payload)
    return payload


def run_worker(arm: str, task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if arm not in ARMS_ORDER or task_id not in TASKS:
        raise ValueError(f"E17 arm/task not locked: {arm}, {task_id}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing duplicate E17 output: {output_dir}")
    source_manifest = _source_manifest()
    root_approval = _read_root_approval(source_manifest)
    source_binding = _stable_source_binding(source_manifest)
    # Build the fixed probe before E2 calls seed_everything. Only the batch
    # tensors/IDs are retained; probe execution later is RNG-forked.
    _, probe_batch_raw, _, _ = _fixed_profile_batch(task_id)
    probe_batch = {
        key: value.clone()
        for key, value in probe_batch_raw.items()
        if key in {"input_ids", "attention_mask", "labels"} and isinstance(value, torch.Tensor)
    }
    probe_ids = [str(value) for value in probe_batch_raw.get("ids", [])]
    captured: dict[str, Any] = {}
    epoch_telemetry: list[dict[str, Any]] = []
    optimizer_step_telemetry: list[dict[str, Any]] = []
    optimizer_step_counter = {"value": 0}
    original_builder = e2_train.build_e2_student
    original_optimizer = e2_train.make_optimizer
    original_run_validation = e2_train.run_validation

    def e17_builder(backbone_dir: Path, *, num_classes: int, num_aspects: int = 0, aspect_class_counts: Any = None, attention_implementation: str = "sdpa"):
        return build_e17_student(backbone_dir, num_classes=num_classes, num_aspects=num_aspects, aspect_class_counts=aspect_class_counts, arm=arm, attention_implementation=attention_implementation)

    def e17_optimizer(model: torch.nn.Module, learning_rate: float):
        optimizer, coverage = build_e17_optimizer(model, learning_rate, gamma=GAMMA, num_layers=10, weight_decay=WEIGHT_DECAY)
        captured["optimizer"] = optimizer; captured["coverage"] = coverage
        if hasattr(optimizer, "register_step_post_hook"):
            def step_post_hook(opt: torch.optim.Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                optimizer_step_counter["value"] += 1
                step_index = optimizer_step_counter["value"]
                if step_index not in {1, 10, 100}:
                    return
                eta = getattr(model, "eta", None)
                grad = None if eta is None else eta.grad
                optimizer_step_telemetry.append(
                    {
                        "optimizer_step": step_index,
                        "eta_grad_norm_postclip": None if grad is None else float(grad.detach().float().norm().item()),
                        "eta_grad_finite_postclip": None if grad is None else bool(torch.isfinite(grad).all().item()),
                    }
                )
                _write_json(
                    output_dir / "optimizer_step_telemetry.json",
                    {"protocol": PROTOCOL, "arm": arm, "task_id": task_id, "rows": optimizer_step_telemetry, "timing": "post-clip optimizer step hook"},
                )
            optimizer.register_step_post_hook(step_post_hook)
            captured["optimizer_step_post_hook"] = True
        else:
            captured["optimizer_step_post_hook"] = False
        return optimizer

    def e17_run_validation(model: Any, loader: Any, device: torch.device, precision: str, validation_task_id: str, bundle: Any, *, split_name: str = "validation"):
        metrics, predictions = original_run_validation(model, loader, device, precision, validation_task_id, bundle, split_name=split_name)
        was_training = bool(model.training)
        diagnostics_was_enabled = bool(model.target_self.collect_diagnostics)
        model.eval()
        model.set_diagnostics(True)
        gpu_index = int(device.index) if device.type == "cuda" and device.index is not None else None
        fork_devices = [gpu_index] if gpu_index is not None else []
        probe = {key: value.to(device) for key, value in probe_batch.items()}
        with torch.random.fork_rng(devices=fork_devices):
            with torch.no_grad():
                with e2_train.make_autocast(precision, device):
                    _ = model(probe["input_ids"], probe["attention_mask"])
        stats = model.diagnostics()
        fallback_count = int(stats.get("flagged_count", 0))
        valid_pairs = int(stats.get("total_valid_pairs", 0))
        epoch_telemetry.append(
            {
                "epoch": len(epoch_telemetry) + 1,
                "task_id": validation_task_id,
                "arm": arm,
                "probe_ids": probe_ids,
                "probe_input_shape": list(probe["input_ids"].shape),
                "eta": stats.get("eta", [0.0] * 6),
                "eta_delta_norm": stats.get("eta_delta_norm", 0.0),
                "correction_norm": stats.get("eta_delta_norm", 0.0),
                "attention_probability_delta_norm": stats.get("attention_probability_delta_norm", 0.0),
                "fallback_count": fallback_count,
                "valid_pair_count": valid_pairs,
                "fallback_rate": fallback_count / max(valid_pairs, 1),
                "orig_logits_norm": stats.get("orig_logits_norm", 0.0),
                "variant_delta_norm": stats.get("variant_delta_norm", 0.0),
                "finite": all(math.isfinite(float(value)) for value in (stats.get("eta_delta_norm", 0.0), stats.get("attention_probability_delta_norm", 0.0))),
            }
        )
        _write_json(
            output_dir / "epoch_telemetry.json",
            {"protocol": PROTOCOL, "arm": arm, "task_id": task_id, "probe_ids": probe_ids, "rows": epoch_telemetry, "gradient_eta_postclip": "see optimizer_step_telemetry.json"},
        )
        model.set_diagnostics(diagnostics_was_enabled)
        model.train(was_training)
        return metrics, predictions

    e2_train.build_e2_student = e17_builder  # type: ignore[assignment]
    e2_train.make_optimizer = e17_optimizer  # type: ignore[assignment]
    e2_train.run_validation = e17_run_validation  # type: ignore[assignment]
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
            precision=PRECISION,
            attention_implementation=ATTENTION_IMPLEMENTATION,
            epochs=EPOCHS,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            stage="stageE17_recipient_excluded_keys",
            method=arm,
            coefficient=NC_LAMBDAS[task_id],
            revision=REVISION,
        )
    finally:
        e2_train.build_e2_student = original_builder  # type: ignore[assignment]
        e2_train.make_optimizer = original_optimizer  # type: ignore[assignment]
        e2_train.run_validation = original_run_validation  # type: ignore[assignment]
    artifact = Path(result["artifact_dir"])
    metadata = _read_json(artifact / "metadata.json")
    telemetry_path = artifact / "epoch_telemetry.json"
    _write_json(telemetry_path, {"protocol": PROTOCOL, "arm": arm, "task_id": task_id, "probe_ids": probe_ids, "rows": epoch_telemetry, "gradient_eta_postclip": "not captured"})
    optimizer_telemetry_path = artifact / "optimizer_step_telemetry.json"
    _write_json(optimizer_telemetry_path, {"protocol": PROTOCOL, "arm": arm, "task_id": task_id, "rows": optimizer_step_telemetry, "timing": "post-clip optimizer step hook"})
    metadata.update({"e17_protocol": PROTOCOL, "e17_revision": REVISION, "e17_arm": arm, "e17_optimizer_coverage": captured.get("coverage"), "e17_optimizer_step_post_hook": captured.get("optimizer_step_post_hook", False), "e17_source_binding": source_binding, "e17_root_approval": root_approval, "e17_epoch_telemetry_path": str(telemetry_path), "e17_optimizer_step_telemetry_path": str(optimizer_telemetry_path), "e17_telemetry_policy": "validation-wrapper fixed probe; eta/correction/probability/fallback telemetry; post-clip eta gradients at selected steps; no full tensors"})
    _write_json(artifact / "metadata.json", metadata)
    checkpoint = artifact / "best_checkpoint.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state.setdefault("metadata", {}).update({"e17_protocol": PROTOCOL, "e17_arm": arm, "e17_optimizer_coverage": captured.get("coverage"), "e17_optimizer_step_post_hook": captured.get("optimizer_step_post_hook", False), "e17_source_binding": source_binding, "e17_epoch_telemetry_path": str(artifact / "epoch_telemetry.json"), "e17_optimizer_step_telemetry_path": str(artifact / "optimizer_step_telemetry.json")})
        torch.save(state, checkpoint)
    return result


def run_all(device_names: list[str], *, dispatch_approved: bool) -> dict[str, Any]:
    if not dispatch_approved:
        raise RuntimeError("Full E17 dispatch requires explicit --dispatch-approved after root review")
    if not device_names:
        raise RuntimeError("No GPU devices available for E17 dispatch")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = ARTIFACT_ROOT / "controller_state.json"
    source_manifest = _source_manifest()
    source_binding = _stable_source_binding(source_manifest)
    source_manifest_hash = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode("utf-8")).hexdigest()
    previous: dict[str, Any] = {}
    if state_path.is_file():
        previous = _read_json(state_path)
        if previous.get("source_binding") is not None and previous.get("source_binding") != source_binding:
            raise RuntimeError("Existing E17 controller source binding differs; refusing resume")
        if previous.get("status") == "running":
            active = [job for job in previous.get("jobs", []) if _pid_alive(job.get("pid")) and job.get("status") == "running"]
            if active:
                raise RuntimeError(f"E17 controller already has active jobs; adopt/refuse duplicate: {active}")
    _read_root_approval(source_manifest)
    _write_json(ARTIFACT_ROOT / "source_manifest_dispatch.json", {"manifest": source_manifest, "manifest_sha256": source_manifest_hash, "source_binding": source_binding, "dispatch_approved": True, "approved_at": _now()})
    queue = [(task, arm) for task in TASKS for arm in ARMS_ORDER]
    completed_keys = {
        (job.get("task_id"), job.get("arm"))
        for job in previous.get("jobs", [])
        if job.get("status") == "completed"
    }
    if previous and previous.get("source_binding") == source_binding:
        for task, arm in queue:
            artifact = ARTIFACT_ROOT / arm / task / "seed_42"
            status_path = artifact / "status.json"
            if status_path.is_file():
                status = _read_json(status_path)
                if status.get("status") == "completed":
                    completed_keys.add((task, arm))
    queue = [(task, arm) for task, arm in queue if (task, arm) not in completed_keys]
    controller = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "running",
        "created_at": _now(),
        "source_manifest_sha256": source_manifest_hash,
        "source_binding": source_binding,
        "dispatch_approved": True,
        "jobs": [job for job in previous.get("jobs", []) if (job.get("task_id"), job.get("arm")) in completed_keys],
        "queue_remaining": [{"task_id": task, "arm": arm} for task, arm in queue],
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

    for index, (task, arm) in enumerate(queue):
        device = device_names[index % len(device_names)]
        snapshot_now = _gpu_snapshot()
        vram_gate = _dispatch_vram_gate(device, snapshot_now)
        output = ARTIFACT_ROOT / arm / task / "seed_42"
        log_path = ARTIFACT_ROOT / "logs" / f"{arm}__{task}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm, "--task-id", task, "--device", device, "--output-dir", str(output)]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"})
        job = {"arm": arm, "task_id": task, "device": device, "pid": process.pid, "output_dir": str(output), "log": str(log_path), "command": command, "status": "running", "started_at": _now(), "vram_gate": vram_gate}
        controller["jobs"].append(job)
        controller["queue_remaining"] = [{"task_id": t, "arm": a} for t, a in queue[index + 1 :]]
        _write_json(state_path, controller)
        active.append((process, job, log))
        if len(active) >= len(device_names):
            process0, meta0, log0 = active[0]
            code = wait_job(process0, meta0, log0)
            active.pop(0)
            meta0["returncode"] = code; meta0["status"] = "completed" if code == 0 else "failed"; meta0["finished_at"] = _now(); completed.append(meta0); _write_json(state_path, controller)
            if code != 0:
                raise RuntimeError(f"E17 worker failed: {meta0}")
    for process, meta, log in active:
        code = wait_job(process, meta, log); meta["returncode"] = code; meta["status"] = "completed" if code == 0 else "failed"; meta["finished_at"] = _now(); completed.append(meta); _write_json(state_path, controller)
        if code != 0:
            raise RuntimeError(f"E17 worker failed: {meta}")
    controller.update({"status": "completed", "finished_at": _now(), "jobs": completed, "queue_remaining": []})
    _write_json(state_path, controller)
    return controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E17 preflight/profile/worker runner")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--plumbing-check", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--dispatch-approved", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--arm", choices=ARMS_ORDER)
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not any((args.preflight, args.profile, args.run_all, args.worker, args.plumbing_check)):
        raise SystemExit("Choose --preflight, --profile, --plumbing-check, --run-all, or --worker")
    if args.worker:
        if not args.arm or not args.task_id or not args.device or args.output_dir is None:
            raise SystemExit("--worker requires --arm, --task-id, --device, and --output-dir")
        result = run_worker(args.arm, args.task_id, args.device, args.output_dir)
    elif args.run_all:
        snapshot = _gpu_snapshot()
        result = run_all(_eligible_devices(snapshot), dispatch_approved=args.dispatch_approved)
    elif args.profile:
        result = run_profile(args.device)
    elif args.plumbing_check:
        result = run_plumbing_check()
    else:
        result = run_preflight(args.device)
    print(json.dumps(result, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
