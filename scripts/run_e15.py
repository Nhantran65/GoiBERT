"""Run E15: centered/uncentered pair-value residual on original BamiBERT.

Validation runs exactly two variants at the fixed E14-selected learning rate
for each of five tasks.  Test evaluation is a separate, locked command and is
refused until the validation screen passes and root explicitly approves it.
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e15"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
MODEL_DIR = ROOT / "models" / "bamibert-base-original"
E14_GRID = ROOT / "artifacts" / "e14" / "validation_grid.json"
E14_SOURCE = ROOT / "artifacts" / "e14" / "source_checkpoint_manifest.json"
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
PAIR_CHUNK_SIZE = 64
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
VARIANTS = ("centered", "uncentered")
TASK_LRS = {
    "vsfc_sentiment": 2e-5,
    "vsfc_topic": 1e-5,
    "visfd": 5e-5,
    "absa_restaurant": 5e-5,
    "absa_hotel": 5e-5,
}
GPU_INDICES = ("0", "1", "2", "3")
MIN_FREE_MIB = 12_000
HF_REPO_ID = "Qualcomm-AI-Research/BamiBERT"
HF_COMMIT = "57bc1340debbe4e348ec549047a763caebe4a977"
WEIGHT_SHA256 = "c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627"
PROTOCOL = "E15-BamiBERT-original-pair-value-residual-v1"
REVISION = "e15-20260912-r1"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e14.modeling import build_bamibert_classifier  # noqa: E402
from goibert_e15.modeling import (  # noqa: E402
    HIDDEN_SIZE,
    INSERTION_LAYER_INDEX,
    NUM_HEADS,
    PAIR_RANK,
    E15ModelSpec,
    build_e15_classifier,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return e1_train._json_default(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
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
        if path.is_file()
    }


def _hash_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.resolve().relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def source_hash() -> str:
    paths = [
        Path(__file__).resolve(),
        ROOT / "configs" / "e15" / "protocol.json",
        *sorted((SRC / "goibert_e15").rglob("*.py")),
        *[SRC / "goibert" / name for name in ("train.py", "data.py", "metrics.py")],
        SRC / "goibert_e14" / "modeling.py",
    ]
    return _hash_paths([path for path in paths if path.is_file()])


def _runtime(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": PRECISION,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _gpu_snapshot() -> dict[str, dict[str, int]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    result: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 3:
            result[fields[0]] = {"free_mib": int(float(fields[1])), "used_mib": int(float(fields[2]))}
    if not result:
        raise RuntimeError(f"nvidia-smi returned no rows: {output!r}")
    return result


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except (ProcessLookupError, OSError):
        return False
    except PermissionError:
        return True
    return True


def _task_contract(task_id: str, *, include_test: bool = False) -> dict[str, Any]:
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=include_test)
    train = e1_train.split_field(bundle, "train")
    validation = e1_train.split_field(bundle, "validation")
    test = e1_train.split_field(bundle, "test") if include_test else None
    return {
        "bundle": bundle,
        "train": train,
        "validation": validation,
        "test": test,
        "num_aspects": e1_train.task_num_aspects(bundle),
        "aspect_class_counts": e1_train.task_aspect_class_counts(bundle),
        "num_classes": e1_train.task_num_classes(bundle),
    }


def _tokenizer_hashes(model_dir: Path) -> dict[str, str]:
    hashes = _directory_hashes(model_dir)
    return {name: value for name, value in hashes.items() if Path(name).name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}}


def _label_overlay(bundle: Any) -> dict[str, Any]:
    return {
        "vsfc_sentiment": {"0": "negative", "1": "neutral", "2": "positive"},
        "vsfc_topic": {str(index): name for index, name in enumerate(getattr(bundle, "label_names", ()))},
        "absa_polarity": {"0": "absent", "1": "positive", "2": "negative", "3": "neutral"},
        "visfd_others": {"raw_0": "absent", "raw_4": "present_detection_only", "model_0": "absent", "model_1": "present_detection_only"},
        "numeric_gold_unchanged": True,
    }


def _model_manifest() -> dict[str, Any]:
    hashes = _directory_hashes(MODEL_DIR)
    config = _read_json(MODEL_DIR / "config.json")
    weight_hash = hashes.get("model.safetensors")
    if weight_hash != WEIGHT_SHA256:
        raise RuntimeError(f"Pinned BamiBERT weight hash mismatch: {weight_hash} != {WEIGHT_SHA256}")
    return {
        "source": "huggingface",
        "repository": HF_REPO_ID,
        "hf_commit": HF_COMMIT,
        "local_path": str(MODEL_DIR.relative_to(ROOT)),
        "files": hashes,
        "config": config,
        "architecture_check": {"num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072},
        "weight_file_sha256": weight_hash,
        "not_teacher_dapt": True,
    }


def _trial_key(variant: str, task_id: str) -> str:
    return f"{variant}__{task_id}"


def _artifact_dir(variant: str, task_id: str) -> Path:
    return ARTIFACT_ROOT / "trials" / variant / task_id / f"seed_{SEED}"


def _worker_status(key: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_ROOT / f"{key}.json", dict(payload))


def _make_optimizer_coverage(model: nn.Module, optimizer: torch.optim.Optimizer, lr: float) -> dict[str, Any]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    seen = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len({id(parameter) for parameter in seen}) != len(seen) or set(map(id, seen)) != set(map(id, params)):
        raise RuntimeError("E15 optimizer does not cover each trainable parameter exactly once")
    if any(float(group["lr"]) != float(lr) for group in optimizer.param_groups):
        raise RuntimeError("E15 optimizer groups do not use a uniform learning rate")
    return {
        "trainable_parameter_count": sum(parameter.numel() for parameter in params),
        "optimizer_parameter_count": sum(parameter.numel() for parameter in seen),
        "group_count": len(optimizer.param_groups),
        "uniform_lr": True,
        "unique_coverage": True,
        "group_weight_decays": [float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups],
    }


def _preflight_model(device_name: str, micro_batch_size: int) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"E15 requires CUDA preflight, got {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    encoded = tokenizer(["đây là kiểm tra BamiBERT"] * micro_batch_size, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
    model, spec = build_e15_classifier(MODEL_DIR, num_classes=3, variant="centered", pair_chunk_size=PAIR_CHUNK_SIZE)
    model.to(device).train()
    torch.cuda.reset_peak_memory_stats(device)
    batch = {key: value.to(device) for key, value in encoded.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = logits.float().square().mean()
    loss.backward()
    finite = bool(torch.isfinite(loss).item() and all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()))
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    result = {"device": device_name, "micro_batch_size": micro_batch_size, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "dtype": "bfloat16", "finite_loss_and_gradients": finite, "logits_shape": list(logits.shape), "peak_memory_allocated_bytes": peak_alloc, "peak_memory_reserved_bytes": peak_reserved, "runtime": _runtime(device), "model_spec": spec.to_dict()}
    del model, logits, loss, batch
    torch.cuda.empty_cache()
    return result


def _architecture_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    details: dict[str, Any] = {"tasks": {}}
    for task_id in TASKS:
        contract = _task_contract(task_id)
        models: dict[str, nn.Module] = {}
        specs: dict[str, dict[str, Any]] = {}
        for variant in VARIANTS:
            e1_train.seed_everything(SEED)
            model, spec = build_e15_classifier(MODEL_DIR, num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, variant=variant)
            models[variant] = model
            specs[variant] = spec.to_dict()
        if any(spec["added_parameter_count"] != 147456 for spec in specs.values()):
            raise AssertionError("E15 added parameter count mismatch")
        state_a, state_b = models["centered"].state_dict(), models["uncentered"].state_dict()
        if set(state_a) != set(state_b) or any(not torch.equal(state_a[key], state_b[key]) for key in state_a):
            raise AssertionError(f"E15 variants are not same-initialized at {task_id}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
        texts = list(e1_train.split_texts(contract["validation"]))[:2]
        encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        for model in models.values():
            model.eval()
        with torch.no_grad():
            centered_logits = models["centered"](encoded["input_ids"], encoded["attention_mask"])
            uncentered_logits = models["uncentered"](encoded["input_ids"], encoded["attention_mask"])
        details["tasks"][task_id] = {"specs": specs, "input_shape": list(encoded["input_ids"].shape), "finite_logits": bool(torch.isfinite(centered_logits).all() and torch.isfinite(uncentered_logits).all()), "same_initial_state": True}
        del models
    checks["all_five_task_shapes"] = len(details["tasks"]) == 5 and all(item["finite_logits"] for item in details["tasks"].values())
    checks["same_parameterization"] = True

    # Exact E14 initialization/logit parity with P=0, comparing all common keys.
    e1_train.seed_everything(777)
    baseline, _ = build_bamibert_classifier(MODEL_DIR, num_classes=3, attention_implementation=ATTENTION_IMPLEMENTATION)
    e1_train.seed_everything(777)
    candidate, _ = build_e15_classifier(MODEL_DIR, num_classes=3, variant="centered")
    baseline.eval(); candidate.eval()
    ids = torch.randint(0, 20_000, (2, 16))
    mask = torch.ones_like(ids); mask[1, -4:] = 0
    common = set(baseline.state_dict()).intersection(candidate.state_dict())
    checks["e14_common_state_exact"] = bool(all(torch.equal(baseline.state_dict()[key], candidate.state_dict()[key]) for key in common))
    with torch.no_grad():
        reference = baseline(ids, mask)
        actual = candidate(ids, mask)
    checks["e14_t0_logits_exact"] = bool(torch.equal(reference, actual))
    checks["e14_t0_logits_finite"] = bool(torch.isfinite(actual).all())
    pair = candidate.pair_module
    checks["fp32_routing_scores"] = pair._attention_scores(torch.randn(2, 12, 4), torch.randn(2, 12, 4), None).dtype == torch.float32
    q0 = torch.zeros(2, 3, PAIR_RANK); k0 = torch.randn(2, 3, PAIR_RANK)
    centered_zero = torch.nn.functional.gelu(q0 + k0, approximate="none") - torch.nn.functional.gelu(q0, approximate="none") - torch.nn.functional.gelu(k0, approximate="none")
    checks["centered_null_identity"] = bool(torch.equal(centered_zero, torch.zeros_like(centered_zero)))
    x = torch.randn(2, 3, PAIR_RANK)
    checks["linear_activation_identity"] = bool(torch.allclose((x + k0) - x - k0, torch.zeros_like(x), atol=1e-6, rtol=0.0))

    # Chunked versus dense forward and gradient parity with non-zero P.
    pair.eval()
    with torch.no_grad():
        pair.P.weight.normal_(0.0, 0.01)
    hidden = torch.randn(2, 9, HIDDEN_SIZE, requires_grad=True)
    oq = pair.transpose_for_scores(pair.query(hidden)); ok = pair.transpose_for_scores(pair.key(hidden))
    pair.chunk_size = PAIR_CHUNK_SIZE
    chunk = pair._pair_message(hidden, oq, ok, None)
    pair.chunk_size = 99
    dense = pair._pair_message(hidden, oq, ok, None)
    pair.chunk_size = PAIR_CHUNK_SIZE
    checks["chunked_dense_forward_parity"] = bool(torch.allclose(chunk, dense, atol=1e-6, rtol=1e-5))
    grad_chunk = torch.autograd.grad(chunk.square().sum(), hidden, retain_graph=True)[0]
    grad_dense = torch.autograd.grad(dense.square().sum(), hidden)[0]
    checks["chunked_dense_gradient_parity"] = bool(torch.allclose(grad_chunk, grad_dense, atol=1e-5, rtol=1e-4))

    # P receives gradient at P=0; after a tiny P step U/W receive gradient.
    probe = candidate.pair_module
    probe.P.weight.data.zero_(); probe.U.weight.grad = None; probe.W.weight.grad = None; probe.P.weight.grad = None
    hidden_probe = torch.randn(2, 9, HIDDEN_SIZE)
    q_probe = probe.transpose_for_scores(probe.query(hidden_probe)); k_probe = probe.transpose_for_scores(probe.key(hidden_probe))
    first = probe._pair_message(hidden_probe, q_probe, k_probe, None).sum(); first.backward()
    p_norm = float(probe.P.weight.grad.norm()) if probe.P.weight.grad is not None else 0.0
    with torch.no_grad(): probe.P.weight.add_(0.01)
    probe.P.weight.grad = None; probe.U.weight.grad = None; probe.W.weight.grad = None
    q_probe = probe.transpose_for_scores(probe.query(hidden_probe)); k_probe = probe.transpose_for_scores(probe.key(hidden_probe))
    second = probe._pair_message(hidden_probe, q_probe, k_probe, None).sum(); second.backward()
    uw_norm = float((probe.U.weight.grad.norm() + probe.W.weight.grad.norm())) if probe.U.weight.grad is not None and probe.W.weight.grad is not None else 0.0
    checks["gradient_P_step1"] = p_norm > 0 and math.isfinite(p_norm)
    checks["gradient_UW_step2"] = uw_norm > 0 and math.isfinite(uw_norm)

    optimizer = e1_train.make_optimizer(candidate, 2e-5)
    coverage = _make_optimizer_coverage(candidate, optimizer, 2e-5)
    checks["uniform_optimizer_coverage"] = bool(coverage["unique_coverage"] and coverage["uniform_lr"])
    details["optimizer_coverage"] = coverage
    return {"checks": checks, "details": details}


def preflight() -> dict[str, Any]:
    snapshot = _gpu_snapshot()
    manifest = _model_manifest()
    architecture = _architecture_checks()
    contracts: dict[str, Any] = {}
    for task_id in TASKS:
        contract = _task_contract(task_id)
        contracts[task_id] = {"train_count": len(contract["train"]), "validation_count": len(contract["validation"]), "num_classes": contract["num_classes"], "num_aspects": contract["num_aspects"], "aspect_class_counts": list(contract["aspect_class_counts"]), "label_overlay": _label_overlay(contract["bundle"])}
    profiles: dict[str, Any] = {}
    for index in GPU_INDICES:
        if index not in snapshot or snapshot[index]["free_mib"] < MIN_FREE_MIB:
            profiles[index] = {"skipped": True, "reason": "free_memory_guard", "snapshot": snapshot.get(index)}
            continue
        preferred_micro = 32 if snapshot[index]["free_mib"] >= 28_000 else 16
        try:
            profiles[index] = _preflight_model(f"cuda:{index}", preferred_micro)
        except RuntimeError as error:
            if preferred_micro == 32 and "out of memory" in str(error).lower():
                profiles[index] = _preflight_model(f"cuda:{index}", 16)
            else:
                raise
    checks = {**architecture["checks"], "official_model_dir": MODEL_DIR.is_dir(), "weights_sha256_pinned": manifest["weight_file_sha256"] == WEIGHT_SHA256, "all_tasks_loaded_without_test": len(contracts) == 5, "cuda_profile_available": any(not value.get("skipped", False) and value.get("finite_loss_and_gradients", False) for value in profiles.values()), "no_test_access": True}
    if not all(bool(value) for value in checks.values()):
        raise RuntimeError(f"E15 preflight failed: {checks}")
    return {"protocol": PROTOCOL, "revision": REVISION, "created_at": _now(), "source_hash": source_hash(), "model_manifest": manifest, "gpu_snapshot": snapshot, "gpu_profiles": profiles, "task_contracts": contracts, "checks": checks, "architecture_details": architecture["details"], "recipe": {"seed": SEED, "epochs": EPOCHS, "task_learning_rates": TASK_LRS, "max_length": MAX_LENGTH, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "precision": PRECISION, "attention_implementation": ATTENTION_IMPLEMENTATION, "pair_chunk_size": PAIR_CHUNK_SIZE, "added_parameter_count": 147456}}


def _profile_micro(profiles: Mapping[str, Any], gpu_index: str) -> int:
    value = profiles.get(gpu_index, {}).get("micro_batch_size")
    return int(value) if value in {8, 16, 32} else 16


def _validate_checkpoint_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "epoch", "global_step", "rng_state"}
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(f"Checkpoint missing required keys: {missing}")
    dense = state["model"].get("head.dense.weight")
    if dense is None or tuple(dense.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise RuntimeError("E15 checkpoint head is not dense768x768")
    branch = [key for key in state["model"] if key.endswith(".U.weight") or key.endswith(".W.weight") or key.endswith(".P.weight")]
    if len(branch) != 3:
        raise RuntimeError(f"E15 checkpoint branch keys mismatch: {branch}")
    return {"required_state_keys": sorted(required), "head_dense_shape": list(dense.shape), "branch_keys": sorted(branch), "epoch": state["epoch"], "global_step": state["global_step"]}


def _run_trial(variant: str, task_id: str, device_name: str, micro_batch_size: int, output_dir: Path, expected_source_hash: str) -> dict[str, Any]:
    if source_hash() != expected_source_hash:
        raise RuntimeError("E15 source changed after controller freeze")
    e1_train.seed_everything(SEED)
    device = e1_train.resolve_device(device_name)
    precision = e1_train.resolve_precision(PRECISION, device)
    if micro_batch_size <= 0 or EFFECTIVE_BATCH_SIZE % micro_batch_size:
        raise ValueError("micro batch must divide effective batch size")
    accumulation_steps = EFFECTIVE_BATCH_SIZE // micro_batch_size
    contract = _task_contract(task_id)
    bundle, train_split, valid_split = contract["bundle"], contract["train"], contract["validation"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    lengths = e1_train.inspect_train_lengths(tokenizer, list(e1_train.split_texts(train_split)))
    model, model_spec = build_e15_classifier(MODEL_DIR, num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, variant=variant, pair_chunk_size=PAIR_CHUNK_SIZE)
    model.to(device)
    optimizer = e1_train.make_optimizer(model, TASK_LRS[task_id])
    coverage = _make_optimizer_coverage(model, optimizer, TASK_LRS[task_id])
    train_loader = e1_train._make_loader(tokenizer, train_split, max_length=MAX_LENGTH, batch_size=micro_batch_size, shuffle=True, seed=SEED, labels=e1_train.model_labels(bundle, "train", train_split), gold_labels=e1_train.split_labels(train_split))
    valid_loader = e1_train._make_loader(tokenizer, valid_split, max_length=MAX_LENGTH, batch_size=micro_batch_size, shuffle=False, seed=SEED, labels=e1_train.model_labels(bundle, "validation", valid_split), gold_labels=e1_train.split_labels(valid_split))
    total_steps = math.ceil(len(train_loader) / accumulation_steps) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, round(total_steps * 0.1), total_steps)
    metadata: dict[str, Any] = {
        "protocol": PROTOCOL, "revision": REVISION, "variant": variant, "task_id": task_id, "seed": SEED, "stage": "validation", "learning_rate": TASK_LRS[task_id], "epochs": EPOCHS, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "micro_batch_size": micro_batch_size, "gradient_accumulation_steps": accumulation_steps, "max_length": MAX_LENGTH, "precision": precision, "attention_implementation": ATTENTION_IMPLEMENTATION, "weight_decay": 0.01, "betas": [0.9, 0.999], "adamw_epsilon": 1e-8, "warmup_fraction": 0.1, "warmup_steps": round(total_steps * 0.1), "gradient_clip_norm": 1.0, "loss": "plain_cross_entropy", "head": "CLS -> dropout0.1 -> dense768 -> tanh -> dropout0.1 -> logits", "no_noise_consistency": True, "no_knowledge_distillation": True, "no_layerwise_learning_rate_decay": True, "include_test": False, "source_hash": expected_source_hash, "model_source": HF_REPO_ID, "hf_commit": HF_COMMIT, "checkpoint_source": "models/bamibert-base-original", "weight_sha256": WEIGHT_SHA256, "model_hashes": _directory_hashes(MODEL_DIR), "tokenizer_hashes": _tokenizer_hashes(MODEL_DIR), "model": model_spec.to_dict(), "pair_module": {"variant": variant, "insertion_layer_index": INSERTION_LAYER_INDEX, "pair_rank": PAIR_RANK, "pair_chunk_size": PAIR_CHUNK_SIZE, "pair_activation": "gelu_exact", "added_parameter_count": 147456, "routing": "mean_12_original_attention_maps_fp32_softmax_no_dropout", "query_padding_delta_zero": True}, "optimizer_coverage": coverage, "data_provenance": e1_train.bundle_provenance(bundle, ROOT), "task_metadata": e1_train.task_metadata(bundle), "label_name_overlay": _label_overlay(bundle), "e14_baseline_grid": str(E14_GRID.relative_to(ROOT)), "runtime": _runtime(device), "length_stats": lengths.to_dict(), "started_at": _now(), "started_at_unix": time.time(), "argv": list(sys.argv),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "resolved_config.json", metadata)
    best_score = -math.inf; best_epoch = 0; best_metrics: dict[str, Any] = {}; best_predictions: list[dict[str, Any]] = []; curves: list[dict[str, Any]] = []; global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); total_loss = 0.0; total_examples = 0; epoch_started = time.time()
        for batch_index, batch in enumerate(train_loader):
            moved = e1_train._move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                loss = e1_train.compute_loss(logits, moved["labels"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"])
                window_start = (batch_index // accumulation_steps) * accumulation_steps; window_end = min(window_start + accumulation_steps, len(train_loader))
                window_examples = sum(min(micro_batch_size, len(train_loader.dataset) - index * micro_batch_size) for index in range(window_start, window_end))
                scaled_loss = loss * (len(batch["labels"]) / max(window_examples, 1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite E15 loss at epoch={epoch}, batch={batch_index}")
            scaled_loss.backward(); total_loss += float(loss.detach().cpu()) * len(batch["labels"]); total_examples += len(batch["labels"])
            if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
        val_metrics, val_predictions = e1_train.run_validation(model, valid_loader, device, precision, task_id, bundle)
        score = e1_train.selection_score(val_metrics)
        row = {"epoch": epoch, "global_step": global_step, "train_loss": total_loss / max(total_examples, 1), "validation": val_metrics, "selection_f1": score, "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - epoch_started}
        curves.append(row); _write_json(output_dir / "epoch_curves.json", {"epochs": curves})
        if score > best_score:
            best_score, best_epoch, best_metrics, best_predictions = score, epoch, val_metrics, val_predictions
            e1_train.save_trial_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, scheduler, epoch=epoch, global_step=global_step, metadata=metadata)
            _write_json(output_dir / "best_validation_metrics.json", best_metrics); _write_json(output_dir / "validation_predictions.json", {"rows": best_predictions})
    metadata.update({"best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics, "elapsed_seconds": time.time() - metadata["started_at_unix"], "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "status": "completed"})
    checkpoint = output_dir / "best_checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False); state["metadata"] = dict(metadata); torch.save(state, checkpoint)
    metadata["checkpoint_sha256"] = _sha256(checkpoint); metadata["checkpoint_roundtrip"] = _validate_checkpoint_state(checkpoint)
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "status.json", {"status": "completed", "protocol": PROTOCOL, "variant": variant, "task_id": task_id, "best_epoch": best_epoch, "best_selection_f1": best_score, "finished_at": _now(), "pid": os.getpid(), "device": device_name})
    # Correction magnitude is sampled outside the training hot loop from the
    # locked best checkpoint and a real validation batch.
    model.load_state_dict(state["model"], strict=True); model.eval(); model.pair_module.set_diagnostics(True)
    with torch.inference_mode():
        first_batch = next(iter(valid_loader)); moved = e1_train._move_batch(first_batch, device)
        with e1_train.make_autocast(precision, device): model(moved["input_ids"], moved["attention_mask"])
    metadata["correction_norm_probe"] = float(model.pair_module.last_delta_norm.item()); metadata["correction_relative_norm_probe"] = float(model.pair_module.last_delta_relative_norm.item()); _write_json(output_dir / "metadata.json", metadata)
    return {"variant": variant, "task_id": task_id, "learning_rate": TASK_LRS[task_id], "artifact_dir": str(output_dir), "best_epoch": best_epoch, "best_selection_f1": best_score, "validation_metrics": best_metrics, "checkpoint_sha256": metadata["checkpoint_sha256"], "status": "completed"}


def _load_worker(key: str) -> dict[str, Any] | None:
    path = WORKER_ROOT / f"{key}.json"
    return _read_json(path) if path.is_file() else None


def _controller_state(payload: Mapping[str, Any]) -> None:
    _write_json(ORCH_ROOT / "controller_state.json", dict(payload))


def _spawn_worker(spec: Mapping[str, Any], gpu_index: str, micro_batch_size: int, expected_source_hash: str) -> subprocess.Popen[Any]:
    key = str(spec["key"]); log_path = LOG_ROOT / f"{key}__gpu{gpu_index}.log"; log_path.parent.mkdir(parents=True, exist_ok=True); handle = log_path.open("a", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--variant", str(spec["variant"]), "--task-id", str(spec["task_id"]), "--device", "cuda:0", "--micro-batch-size", str(micro_batch_size), "--expected-source-hash", expected_source_hash]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = gpu_index; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"; env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env, start_new_session=True); handle.close(); return process


def worker_main(args: argparse.Namespace) -> int:
    key = _trial_key(args.variant, args.task_id); output = _artifact_dir(args.variant, args.task_id); payload = {"protocol": PROTOCOL, "revision": REVISION, "status": "running", "pid": os.getpid(), "variant": args.variant, "task_id": args.task_id, "device": args.device, "micro_batch_size": args.micro_batch_size, "started_at": _now(), "artifact_dir": str(output)}; _worker_status(key, payload)
    try:
        result = _run_trial(args.variant, args.task_id, args.device, args.micro_batch_size, output, args.expected_source_hash); _worker_status(key, {**payload, **result, "status": "completed", "finished_at": _now()}); return 0
    except Exception as error:
        _worker_status(key, {**payload, "status": "failed", "finished_at": _now(), "error": repr(error), "traceback": traceback.format_exc()}); traceback.print_exc(); return 1


def run_controller() -> dict[str, Any]:
    for directory in (ARTIFACT_ROOT, ORCH_ROOT, WORKER_ROOT, LOG_ROOT): directory.mkdir(parents=True, exist_ok=True)
    preflight_path = ARTIFACT_ROOT / "preflight.json"
    if preflight_path.is_file():
        preflight_value = _read_json(preflight_path)
        if preflight_value.get("source_hash") != source_hash(): raise RuntimeError("Existing E15 preflight source hash differs; refuse to continue")
    else:
        preflight_value = preflight(); _write_json(preflight_path, preflight_value)
    expected = str(preflight_value["source_hash"]); profiles = preflight_value["gpu_profiles"]
    specs = [{"key": _trial_key(variant, task), "variant": variant, "task_id": task, "learning_rate": TASK_LRS[task]} for variant in VARIANTS for task in TASKS]
    completed: set[str] = set(); active: set[str] = set(); running: dict[str, tuple[subprocess.Popen[Any] | None, str, int]] = {}; queue: list[dict[str, Any]] = []
    for spec in specs:
        status = _load_worker(str(spec["key"]))
        artifact = _artifact_dir(str(spec["variant"]), str(spec["task_id"]))
        if status and status.get("status") == "completed" and all((artifact / name).is_file() for name in ("metadata.json", "best_checkpoint.pt", "validation_predictions.json")):
            completed.add(str(spec["key"]))
        elif status and status.get("status") == "running" and _pid_alive(status.get("pid")):
            active.add(str(spec["key"])); running[str(spec["key"])] = (None, str(status.get("gpu", "0")), int(status.get("micro_batch_size", _profile_micro(profiles, "0"))))
        else:
            queue.append(spec)
    while queue or running:
        snapshot = _gpu_snapshot(); used = {value[1] for value in running.values()}
        for gpu_index in GPU_INDICES:
            if gpu_index in used or not queue or gpu_index not in snapshot or snapshot[gpu_index]["free_mib"] < MIN_FREE_MIB: continue
            spec = queue.pop(0); micro = _profile_micro(profiles, gpu_index); process = _spawn_worker(spec, gpu_index, micro, expected); running[str(spec["key"])] = (process, gpu_index, micro); used.add(gpu_index)
        _controller_state({"protocol": PROTOCOL, "revision": REVISION, "status": "validation_running", "updated_at": _now(), "pid": os.getpid(), "expected_source_hash": expected, "gpu_snapshot": snapshot, "completed_keys": sorted(completed), "queued_keys": [str(spec["key"]) for spec in queue], "running": {key: {"gpu": gpu, "micro_batch_size": micro, "pid": proc.pid if proc is not None else (_load_worker(key) or {}).get("pid")} for key, (proc, gpu, micro) in running.items()}})
        for key, (process, gpu, micro) in list(running.items()):
            status = _load_worker(key) or {}
            if status.get("status") == "completed": completed.add(key); del running[key]; continue
            if status.get("status") == "failed" or (process is not None and process.poll() is not None and process.returncode != 0):
                error = str(status.get("error", ""))
                if "out of memory" in error.lower() and micro > 8:
                    spec = next(spec for spec in specs if spec["key"] == key); queue.insert(0, spec); del running[key]; continue
                raise RuntimeError(f"E15 worker failed for {key}: {status.get('error', process.returncode if process else 'unknown')}")
        time.sleep(10)
    _controller_state({"protocol": PROTOCOL, "revision": REVISION, "status": "validation_trials_completed", "updated_at": _now(), "completed_keys": sorted(completed), "source_hash": expected})
    return {"preflight": preflight_value, "specs": specs, "source_hash": expected}


def _read_e14_baseline() -> dict[str, float]:
    grid = _read_json(E14_GRID)["grid"]; result: dict[str, float] = {}
    for task in TASKS:
        rows = [row for row in grid if row["task_id"] == task]
        result[task] = max(float(row["selection_f1"]) for row in rows)
    return result


def _write_validation_outputs(specs: Sequence[Mapping[str, Any]], source_hash_value: str) -> dict[str, Any]:
    baseline = _read_e14_baseline(); rows: list[dict[str, Any]] = []
    for spec in specs:
        artifact = _artifact_dir(str(spec["variant"]), str(spec["task_id"])); metadata = _read_json(artifact / "metadata.json"); metrics = _read_json(artifact / "best_validation_metrics.json"); predictions = _read_json(artifact / "validation_predictions.json")["rows"]
        recomputed = e1_train.evaluate_predictions(str(spec["task_id"]), _task_contract(str(spec["task_id"]))["bundle"], [row["gold"] for row in predictions], [row["prediction"] for row in predictions], split="validation")
        if abs(float(recomputed["selection_f1"]) - float(metrics["selection_f1"])) > 1e-12: raise RuntimeError(f"validation metric recompute mismatch: {spec}")
        rows.append({"variant": spec["variant"], "task_id": spec["task_id"], "learning_rate": float(spec["learning_rate"]), "best_epoch": int(metadata["best_epoch"]), "selection_f1": float(metrics["selection_f1"]), "selection_metric": metrics["selection_metric"], "e14_selection_f1": baseline[str(spec["task_id"])], "delta_vs_e14_pp": 100.0 * (float(metrics["selection_f1"]) - baseline[str(spec["task_id"])]), "artifact_dir": str(artifact), "checkpoint_sha256": _sha256(artifact / "best_checkpoint.pt"), "source_hash": metadata.get("source_hash"), "test_accessed": False, "correction_norm": metadata.get("correction_norm_probe"), "correction_relative_norm": metadata.get("correction_relative_norm_probe")})
    centered = {row["task_id"]: row for row in rows if row["variant"] == "centered"}; control = {row["task_id"]: row for row in rows if row["variant"] == "uncentered"}
    gate_rows = [{"task_id": task, "centered_selection_f1": centered[task]["selection_f1"], "uncentered_selection_f1": control[task]["selection_f1"], "e14_selection_f1": baseline[task], "centered_delta_vs_e14_pp": centered[task]["delta_vs_e14_pp"], "centered_delta_vs_control_pp": 100.0 * (centered[task]["selection_f1"] - control[task]["selection_f1"])} for task in TASKS]
    gain = [row["centered_delta_vs_e14_pp"] for row in gate_rows]; control_gain = [row["centered_delta_vs_control_pp"] for row in gate_rows]
    gate = {"centered_gain_tasks": sum(value >= 0 for value in gain), "centered_gain_ge_0_3pp_tasks": sum(value >= 0.3 for value in gain), "centered_drop_gt_0_3pp_tasks": sum(value < -0.3 for value in gain), "centered_better_than_control_tasks": sum(value >= 0 for value in control_gain), "passes_centered_vs_e14_broad": sum(value >= 0 for value in gain) >= 4 and sum(value >= 0.3 for value in gain) >= 2 and sum(value < -0.3 for value in gain) == 0, "passes_centered_vs_control": sum(value >= 0 for value in control_gain) >= 3}
    gate["passes_broad_validation_screen"] = bool(gate["passes_centered_vs_e14_broad"] and gate["passes_centered_vs_control"])
    payload = {"protocol": PROTOCOL, "revision": REVISION, "source_hash": source_hash_value, "fractions_0_to_1": True, "display_delta_unit": "percentage_points", "baseline_source": str(E14_GRID.relative_to(ROOT)), "rows": rows, "task_comparison": gate_rows, "gate": gate, "test_authorized": False}
    _write_json(ARTIFACT_ROOT / "validation_summary.json", payload)
    with (ARTIFACT_ROOT / "validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    _write_json(ARTIFACT_ROOT / "gate.json", payload)
    return payload


def _lock_validation(specs: Sequence[Mapping[str, Any]], source_hash_value: str) -> dict[str, Any]:
    summary = _read_json(ARTIFACT_ROOT / "validation_summary.json")
    selections: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for task in TASKS:
            row = next(row for row in summary["rows"] if row["variant"] == variant and row["task_id"] == task)
            selections.append({**row, "locked": True, "locked_at": _now(), "test_accessed": False})
    manifest = {"protocol": "E15-locked-validation-v1", "experiment_protocol": PROTOCOL, "revision": REVISION, "locked": True, "locked_at": _now(), "seed": SEED, "source_hash": source_hash_value, "model_source": HF_REPO_ID, "hf_commit": HF_COMMIT, "validation_only_selection": True, "test_accessed_before_lock": False, "gate": summary["gate"], "selections": selections}
    _write_json(ARTIFACT_ROOT / "locked_checkpoints.json", manifest)
    return manifest


def _evaluate_test(selection: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not manifest.get("gate", {}).get("passes_broad_validation_screen") or manifest.get("test_accessed_before_lock"):
        raise RuntimeError("E15 test requires a passing validation gate")
    task_id, variant = str(selection["task_id"]), str(selection["variant"]); artifact = Path(str(selection["artifact_dir"])); metadata = _read_json(artifact / "metadata.json")
    if _sha256(artifact / "best_checkpoint.pt") != selection["checkpoint_sha256"]: raise RuntimeError(f"checkpoint changed after validation lock: {variant}/{task_id}")
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=True); tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    model, spec = build_e15_classifier(MODEL_DIR, num_classes=e1_train.task_num_classes(bundle), num_aspects=e1_train.task_num_aspects(bundle), aspect_class_counts=e1_train.task_aspect_class_counts(bundle) if e1_train.task_num_aspects(bundle) else None, variant=variant)
    device = e1_train.resolve_device("cuda:0"); model.to(device); state = torch.load(artifact / "best_checkpoint.pt", map_location="cpu", weights_only=False); model.load_state_dict(state["model"], strict=True)
    test_split = e1_train.split_field(bundle, "test"); loader = e1_train._make_loader(tokenizer, test_split, max_length=MAX_LENGTH, batch_size=int(metadata["micro_batch_size"]), shuffle=False, seed=SEED, labels=e1_train.model_labels(bundle, "test", test_split), gold_labels=e1_train.split_labels(test_split))
    metrics, rows = e1_train.run_test(model, loader, device, PRECISION, task_id, bundle); recomputed = e1_train.evaluate_predictions(task_id, bundle, [row["gold"] for row in rows], [row["prediction"] for row in rows], split="test")
    if abs(float(recomputed["selection_f1"]) - float(metrics["selection_f1"])) > 1e-12: raise RuntimeError(f"test metric recompute mismatch: {variant}/{task_id}")
    _write_json(artifact / "test_metrics.json", metrics); _write_json(artifact / "test_predictions.json", {"rows": rows})
    return {"variant": variant, "task_id": task_id, "learning_rate": selection["learning_rate"], "best_epoch": selection["best_epoch"], "test_metrics": metrics, "test_count": len(rows), "artifact_dir": str(artifact), "checkpoint_sha256": _sha256(artifact / "best_checkpoint.pt"), "model_spec": spec.to_dict()}


def run_tests() -> int:
    summary = _read_json(ARTIFACT_ROOT / "validation_summary.json"); manifest = _read_json(ARTIFACT_ROOT / "locked_checkpoints.json")
    if not summary["gate"]["passes_broad_validation_screen"]: raise RuntimeError("E15 validation gate failed; test remains closed")
    if not os.environ.get("E15_ROOT_TEST_APPROVED") == "1": raise RuntimeError("E15 test requires root technical approval via E15_ROOT_TEST_APPROVED=1")
    results = [_evaluate_test(row, manifest) for row in manifest["selections"]]
    rows = [{"variant": result["variant"], "task_id": result["task_id"], "learning_rate": result["learning_rate"], "best_epoch": result["best_epoch"], "selection_f1": result["test_metrics"]["selection_f1"], "selection_metric": result["test_metrics"]["selection_metric"], "accuracy": result["test_metrics"].get("accuracy"), "macro_f1": result["test_metrics"].get("macro_f1"), "detection_micro_f1": result["test_metrics"].get("detection_micro_f1"), "pair_micro_f1": result["test_metrics"].get("pair_micro_f1"), "artifact_dir": result["artifact_dir"], "checkpoint_sha256": result["checkpoint_sha256"]} for result in results]
    _write_json(ARTIFACT_ROOT / "test_summary.json", {"protocol": "E15-test-gate-v1", "fractions_0_to_1": True, "results": rows, "test_access_log": [{"variant": row["variant"], "task_id": row["task_id"], "split": "test", "count": result["test_count"], "artifact_dir": row["artifact_dir"]} for row, result in zip(rows, results)]})
    with (ARTIFACT_ROOT / "test_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _write_json(ARTIFACT_ROOT / "test_access_log.json", {"protocol": "E15-test-gate-v1", "locked_selection": True, "entries": [{"variant": row["variant"], "task_id": row["task_id"], "split": "test", "count": result["test_count"], "artifact_dir": row["artifact_dir"], "accessed_at": _now()} for row, result in zip(rows, results)]})
    return 0


def controller_main() -> int:
    context = run_controller(); summary = _write_validation_outputs(context["specs"], context["source_hash"]); _lock_validation(context["specs"], context["source_hash"])
    _controller_state({"protocol": PROTOCOL, "revision": REVISION, "status": "validation_completed", "updated_at": _now(), "source_hash": context["source_hash"], "gate": summary["gate"], "test_authorized": False})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--controller", action="store_true"); mode.add_argument("--worker", action="store_true"); mode.add_argument("--test", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS); parser.add_argument("--task-id", choices=TASKS); parser.add_argument("--device"); parser.add_argument("--micro-batch-size", type=int); parser.add_argument("--expected-source-hash")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if not all(value is not None for value in (args.variant, args.task_id, args.device, args.micro_batch_size, args.expected_source_hash)): raise SystemExit("worker requires variant, task, device, micro-batch-size and expected-source-hash")
        return worker_main(args)
    if args.test: return run_tests()
    return controller_main()


if __name__ == "__main__":
    raise SystemExit(main())
