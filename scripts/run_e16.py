"""Run E16 identity-gated FFN validation on the original BamiBERT encoder.

The controller performs train-only calibration, runs the locked preflight, and
then schedules exactly 15 validation trials (three arms by five tasks). Test
access is intentionally absent from this runner and remains closed for root
review after validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
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
ARTIFACT_ROOT = ROOT / "artifacts" / "e16"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
MODEL_DIR = ROOT / "models" / "bamibert-base-original"
CONFIG_PATH = ROOT / "configs" / "e16" / "protocol.json"
SEED = 42
CALIBRATION_SEED = 16042
CALIBRATION_PER_DOMAIN = 512
EPOCHS = 30
MICRO_BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH_SIZE = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
ARMS = ("baseline", "attention_update_gate", "ffn_input_gate")
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
PROTOCOL = "E16-BamiBERT-original-identity-gated-ffn-v1"
REVISION = "e16-20260912-r1"
LOG_STEPS = {0, 1, 10, 100}

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e14.modeling import build_bamibert_classifier  # noqa: E402
from goibert_e16.modeling import (  # noqa: E402
    GATE_PARAMETER_COUNT,
    HIDDEN_SIZE,
    INSERTION_LAYER_INDEX,
    INTERMEDIATE_SIZE,
    build_e16_classifier,
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


def source_hash() -> str:
    paths = [
        Path(__file__).resolve(),
        CONFIG_PATH,
        *sorted((SRC / "goibert_e16").rglob("*.py")),
        *[SRC / "goibert" / name for name in ("train.py", "data.py", "metrics.py")],
        SRC / "goibert_e14" / "modeling.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(path for path in paths if path.is_file()):
        digest.update(str(path.resolve().relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _runtime(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": PRECISION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
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
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
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
        "not_finetuned_checkpoint": True,
    }


def _tokenizer_hashes(model_dir: Path) -> dict[str, str]:
    hashes = _directory_hashes(model_dir)
    return {name: value for name, value in hashes.items() if Path(name).name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}}


def _trial_key(arm: str, task_id: str) -> str:
    return f"{arm}__{task_id}"


def _artifact_dir(arm: str, task_id: str) -> Path:
    return ARTIFACT_ROOT / "trials" / arm / task_id / f"seed_{SEED}"


def _worker_status(key: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_ROOT / f"{key}.json", dict(payload))


def _optimizer_coverage(model: nn.Module, optimizer: torch.optim.Optimizer, lr: float) -> dict[str, Any]:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    seen = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len({id(parameter) for parameter in seen}) != len(seen) or set(map(id, seen)) != set(map(id, params)):
        raise RuntimeError("E16 optimizer does not cover each trainable parameter exactly once")
    if any(float(group["lr"]) != float(lr) for group in optimizer.param_groups):
        raise RuntimeError("E16 optimizer groups do not use a uniform learning rate")
    return {
        "trainable_parameter_count": sum(parameter.numel() for parameter in params),
        "optimizer_parameter_count": sum(parameter.numel() for parameter in seen),
        "group_count": len(optimizer.param_groups),
        "uniform_lr": True,
        "unique_coverage": True,
        "group_weight_decays": [float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups],
        "gate_in_optimizer": bool(model.gate is None or any(id(model.gate.weight) == id(parameter) for parameter in seen)),
    }


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, default=_json_default) + "\n")


def _calibration_domains() -> tuple[dict[str, Any], ...]:
    # VSFC sentiment/topic share one text pool and are deliberately loaded once.
    return (
        {"domain": "VSFC", "task_id": "vsfc_sentiment"},
        {"domain": "ViSFD", "task_id": "visfd"},
        {"domain": "Restaurant", "task_id": "absa_restaurant"},
        {"domain": "Hotel", "task_id": "absa_hotel"},
    )


def _select_calibration_texts() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(CALIBRATION_SEED)
    global_seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    domain_candidates: dict[str, int] = {}
    for entry in _calibration_domains():
        task_id = str(entry["task_id"])
        domain = str(entry["domain"])
        contract = _task_contract(task_id, include_test=False)
        train = contract["train"]
        texts = list(e1_train.split_texts(train))
        ids = list(e1_train.split_ids(train))
        pairs = list(zip(texts, ids, range(len(texts))))
        rng.shuffle(pairs)
        unique: list[tuple[str, Any, int]] = []
        local_seen: set[str] = set()
        for text, example_id, index in pairs:
            text_hash = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            if text_hash in local_seen or text_hash in global_seen:
                continue
            local_seen.add(text_hash)
            unique.append((str(text), example_id, index))
        domain_candidates[domain] = len(unique)
        take = unique[:CALIBRATION_PER_DOMAIN]
        for text, example_id, index in take:
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            global_seen.add(text_hash)
            selected.append({"domain": domain, "task_id": task_id, "split": "train", "index": int(index), "id": example_id, "text": text, "text_sha256": text_hash})
        domain_counts[domain] = len(take)
    manifest = {
        "protocol": PROTOCOL,
        "seed": CALIBRATION_SEED,
        "requested_per_domain": CALIBRATION_PER_DOMAIN,
        "domains": list(domain_counts),
        "domain_counts": domain_counts,
        "domain_candidates_after_exact_dedup": domain_candidates,
        "actual_total": len(selected),
        "train_only": True,
        "test_loaded": False,
        "selection_rule": "seeded_shuffle_then_exact_text_dedup_across_domains",
        "rows": [{key: value for key, value in row.items() if key != "text"} for row in selected],
        "text_sha256_ordered": [row["text_sha256"] for row in selected],
    }
    return selected, manifest


def _calibrate(device_name: str, *, expected_source_hash: str | None = None) -> dict[str, Any]:
    path = ARTIFACT_ROOT / "calibration_manifest.json"
    if path.is_file():
        value = _read_json(path)
        if expected_source_hash is not None and value.get("source_hash") != expected_source_hash:
            raise RuntimeError("Existing E16 calibration source hash differs; refuse to reuse")
        if float(value.get("sc", 0.0)) < 1e-6 or float(value.get("sx", 0.0)) < 1e-6:
            raise RuntimeError("Existing E16 calibration scalar is invalid")
        return value
    device = e1_train.resolve_device(device_name)
    precision = e1_train.resolve_precision(PRECISION, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    rows, manifest = _select_calibration_texts()
    if len(rows) == 0:
        raise RuntimeError("E16 calibration selected no train texts")
    encoder = RobertaModel.from_pretrained(MODEL_DIR, local_files_only=True, add_pooling_layer=False, attn_implementation=ATTENTION_IMPLEMENTATION).to(device)
    encoder.eval()
    target = encoder.encoder.layer[INSERTION_LAYER_INDEX]
    captured: dict[str, Tensor] = {}

    def capture_x(_module: nn.Module, args: tuple[Any, ...]) -> None:
        captured["x"] = args[0]

    def capture_c(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> None:
        captured["c"] = output

    hook_x = target.intermediate.register_forward_pre_hook(capture_x)
    hook_c = target.attention.output.dense.register_forward_hook(capture_c)
    sum_x = 0.0
    sum_c = 0.0
    count_x = 0
    count_c = 0
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    try:
        for start in range(0, len(rows), MICRO_BATCH_SIZE):
            batch_rows = rows[start : start + MICRO_BATCH_SIZE]
            encoded = tokenizer([row["text"] for row in batch_rows], add_special_tokens=True, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            moved = {key: value.to(device) for key, value in encoded.items() if isinstance(value, Tensor)}
            captured.clear()
            with torch.no_grad():
                with e1_train.make_autocast(precision, device):
                    encoder(**moved, return_dict=True)
            if "x" not in captured or "c" not in captured:
                raise RuntimeError("E16 calibration failed to capture target x/c")
            valid = moved["attention_mask"].bool()
            for special_id in special_ids:
                valid &= moved["input_ids"] != special_id
            for key in ("x", "c"):
                value = captured[key].float()
                selected = value.masked_select(valid.unsqueeze(-1))
                if key == "x":
                    sum_x += float(selected.square().sum().item())
                    count_x += int(selected.numel())
                else:
                    sum_c += float(selected.square().sum().item())
                    count_c += int(selected.numel())
    finally:
        hook_x.remove()
        hook_c.remove()
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    sx = math.sqrt(sum_x / max(count_x, 1))
    sc = math.sqrt(sum_c / max(count_c, 1))
    if not math.isfinite(sc) or not math.isfinite(sx) or sc < 1e-6 or sx < 1e-6:
        raise RuntimeError(f"Invalid E16 calibration scalars sc={sc}, sx={sx}")
    value = {
        **manifest,
        "protocol": PROTOCOL,
        "revision": REVISION,
        "source_hash": expected_source_hash or source_hash(),
        "model_source": HF_REPO_ID,
        "hf_commit": HF_COMMIT,
        "weight_sha256": WEIGHT_SHA256,
        "sc": sc,
        "sx": sx,
        "sc_sum_sq": sum_c,
        "sx_sum_sq": sum_x,
        "sc_element_count": count_c,
        "sx_element_count": count_x,
        "max_length": MAX_LENGTH,
        "mask_policy": "attention_mask and not tokenizer.all_special_ids",
        "runtime": _runtime(device),
        "created_at": _now(),
    }
    _write_json(path, value)
    _write_json(ARTIFACT_ROOT / "calibration_values.json", {key: value[key] for key in ("sc", "sx", "seed", "domain_counts", "actual_total", "source_hash", "created_at")})
    return value


def _profile_model(device_name: str, task_id: str, calibration: Mapping[str, Any]) -> dict[str, Any]:
    device = e1_train.resolve_device(device_name)
    precision = e1_train.resolve_precision(PRECISION, device)
    contract = _task_contract(task_id)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    model, spec = build_e16_classifier(MODEL_DIR, num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, arm="attention_update_gate", calibration_sc=float(calibration["sc"]), calibration_sx=float(calibration["sx"]), attention_implementation=ATTENTION_IMPLEMENTATION)
    model.to(device).train()
    train_loader = e1_train._make_loader(tokenizer, contract["train"], max_length=MAX_LENGTH, batch_size=MICRO_BATCH_SIZE, shuffle=True, seed=SEED, labels=e1_train.model_labels(contract["bundle"], "train", contract["train"]), gold_labels=e1_train.split_labels(contract["train"]))
    batch = e1_train._move_batch(next(iter(train_loader)), device)
    optimizer = e1_train.make_optimizer(model, TASK_LRS[task_id])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with e1_train.make_autocast(precision, device):
        logits = model(batch["input_ids"], batch["attention_mask"], token_type_ids=batch.get("token_type_ids"))
        loss = e1_train.compute_loss(logits, batch["labels"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"])
    loss.backward()
    finite = bool(torch.isfinite(loss).item() and all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()))
    result = {"device": device_name, "micro_batch_size": MICRO_BATCH_SIZE, "accumulation_steps": ACCUMULATION_STEPS, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "finite_loss_and_gradients": finite, "logits_shape": list(logits.shape), "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0, "runtime": _runtime(device), "model_spec": spec.to_dict()}
    del optimizer, model, logits, loss, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _architecture_checks(calibration: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    details: dict[str, Any] = {"tasks": {}, "gradient_probe": None}
    for task_id in TASKS:
        contract = _task_contract(task_id)
        models: dict[str, nn.Module] = {}
        specs: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            e1_train.seed_everything(SEED)
            model, spec = build_e16_classifier(MODEL_DIR, num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, arm=arm, calibration_sc=float(calibration["sc"]), calibration_sx=float(calibration["sx"]))
            models[arm] = model
            specs[arm] = spec.to_dict()
        gate_counts = [specs[arm]["added_parameter_count"] for arm in ARMS]
        if gate_counts != [0, GATE_PARAMETER_COUNT, GATE_PARAMETER_COUNT]:
            raise AssertionError(f"E16 added parameter count mismatch: {gate_counts}")
        state_base = models["baseline"].state_dict()
        state_att = models["attention_update_gate"].state_dict()
        state_ffn = models["ffn_input_gate"].state_dict()
        common = set(state_base) & set(state_att) & set(state_ffn)
        if any(not torch.equal(state_base[key], state_att[key]) or not torch.equal(state_base[key], state_ffn[key]) for key in common):
            raise AssertionError(f"E16 old weights are not shared at {task_id}")
        if not torch.equal(models["attention_update_gate"].gate.weight, models["ffn_input_gate"].gate.weight) or models["attention_update_gate"].gate.weight.norm().item() != 0.0:
            raise AssertionError("E16 gate initialization mismatch")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
        texts = list(e1_train.split_texts(contract["validation"]))[:2]
        encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        for model in models.values():
            model.eval()
        with torch.no_grad():
            outputs = {arm: model(encoded["input_ids"], encoded["attention_mask"]) for arm, model in models.items()}
        if any(not torch.equal(outputs["baseline"], outputs[arm]) for arm in ("attention_update_gate", "ffn_input_gate")):
            raise AssertionError(f"E16 zero-gate baseline parity failed at {task_id}")
        details["tasks"][task_id] = {"specs": specs, "input_shape": list(encoded["input_ids"].shape), "finite_logits": all(bool(torch.isfinite(value).all()) for value in outputs.values()), "same_old_weights": True, "zero_gate_parity": True, "capture_c_cleared": models["attention_update_gate"].target_layer.attention.last_c is None}
        for model in models.values():
            del model
    e1_train.seed_everything(777)
    e14, _ = build_bamibert_classifier(MODEL_DIR, num_classes=3, attention_implementation=ATTENTION_IMPLEMENTATION)
    e1_train.seed_everything(777)
    base, _ = build_e16_classifier(MODEL_DIR, num_classes=3, arm="baseline", calibration_sc=float(calibration["sc"]), calibration_sx=float(calibration["sx"]))
    common = set(e14.state_dict()) & set(base.state_dict())
    checks["e14_common_state_exact"] = bool(all(torch.equal(e14.state_dict()[key], base.state_dict()[key]) for key in common))
    ids = torch.randint(0, 20_000, (2, 16)); mask = torch.ones_like(ids); mask[1, -4:] = 0
    e14.eval(); base.eval()
    with torch.no_grad():
        checks["e14_t0_logits_exact"] = bool(torch.equal(e14(ids, mask), base(ids, mask)))
    gate, _ = build_e16_classifier(MODEL_DIR, num_classes=3, arm="attention_update_gate", calibration_sc=float(calibration["sc"]), calibration_sx=float(calibration["sx"]))
    gate.train(); gate.zero_grad(set_to_none=True)
    logits = gate(ids, mask); loss = logits.float().square().mean(); loss.backward()
    grad_norm = float(gate.gate.weight.grad.norm().item()) if gate.gate is not None and gate.gate.weight.grad is not None else 0.0
    details["gradient_probe"] = {"loss_finite": bool(torch.isfinite(loss).item()), "gate_grad_norm": grad_norm, "gate_grad_finite": bool(gate.gate is not None and gate.gate.weight.grad is not None and torch.isfinite(gate.gate.weight.grad).all().item())}
    checks["gradient_gate_first_backward_nonzero"] = grad_norm > 0.0 and math.isfinite(grad_norm)
    checks["all_five_task_shapes"] = len(details["tasks"]) == 5 and all(value["finite_logits"] for value in details["tasks"].values())
    checks["zero_gate_parity_all_tasks"] = all(value["zero_gate_parity"] for value in details["tasks"].values())
    checks["capture_c_explicitly_cleared"] = all(value["capture_c_cleared"] for value in details["tasks"].values())
    checks["same_gate_parameterization"] = True
    checks["scalar_valid"] = all(math.isfinite(float(calibration[key])) and float(calibration[key]) >= 1e-6 for key in ("sc", "sx"))
    return {"checks": checks, "details": details}


def preflight() -> dict[str, Any]:
    snapshot = _gpu_snapshot()
    manifest = _model_manifest()
    calibration = dict(_calibrate(f"cuda:{next((index for index in GPU_INDICES if index in snapshot and snapshot[index]["free_mib"] >= MIN_FREE_MIB), GPU_INDICES[0])}"))
    # Calibration values remain valid when only the downstream runner changes;
    # refresh the provenance pointer before freezing the new source hash.
    calibration["source_hash"] = source_hash()
    _write_json(ARTIFACT_ROOT / "calibration_manifest.json", calibration)
    architecture = _architecture_checks(calibration)
    contracts: dict[str, Any] = {}
    for task_id in TASKS:
        contract = _task_contract(task_id)
        contracts[task_id] = {"train_count": len(contract["train"]), "validation_count": len(contract["validation"]), "num_classes": contract["num_classes"], "num_aspects": contract["num_aspects"], "aspect_class_counts": list(contract["aspect_class_counts"]), "label_overlay": _label_overlay(contract["bundle"]), "test_loaded": False}
    profiles: dict[str, Any] = {}
    for index in GPU_INDICES:
        if index not in snapshot or snapshot[index]["free_mib"] < MIN_FREE_MIB:
            profiles[index] = {"skipped": True, "reason": "free_memory_guard", "snapshot": snapshot.get(index)}
            continue
        profiles[index] = _profile_model(f"cuda:{index}", "vsfc_sentiment", calibration)
    checks = {**architecture["checks"], "official_model_dir": MODEL_DIR.is_dir(), "weights_sha256_pinned": manifest["weight_file_sha256"] == WEIGHT_SHA256, "all_tasks_loaded_without_test": len(contracts) == 5 and all(not value["test_loaded"] for value in contracts.values()), "cuda_profile_available": any(not value.get("skipped", False) and value.get("finite_loss_and_gradients", False) for value in profiles.values()), "micro_batch_fixed_16": True, "accumulation_fixed_2": True, "no_test_access": True}
    if not all(bool(value) for value in checks.values()):
        raise RuntimeError(f"E16 preflight failed: {checks}")
    value = {"protocol": PROTOCOL, "revision": REVISION, "created_at": _now(), "source_hash": source_hash(), "model_manifest": manifest, "calibration": calibration, "gpu_snapshot": snapshot, "gpu_profiles": profiles, "task_contracts": contracts, "checks": checks, "architecture_details": architecture["details"], "recipe": {"seed": SEED, "epochs": EPOCHS, "task_learning_rates": TASK_LRS, "max_length": MAX_LENGTH, "micro_batch_size": MICRO_BATCH_SIZE, "gradient_accumulation_steps": ACCUMULATION_STEPS, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "precision": PRECISION, "attention_implementation": ATTENTION_IMPLEMENTATION, "arms": ARMS, "gate_parameter_count": GATE_PARAMETER_COUNT}}
    _write_json(ARTIFACT_ROOT / "preflight.json", value)
    _write_json(ARTIFACT_ROOT / "source_provenance.json", {"protocol": PROTOCOL, "revision": REVISION, "source_hash": value["source_hash"], "script": str(Path(__file__).relative_to(ROOT)), "config": str(CONFIG_PATH.relative_to(ROOT)), "model_hashes": manifest["files"], "e14_commit": HF_COMMIT, "frozen_at": _now()})
    return value


def _validate_checkpoint_state(path: Path, arm: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "optimizer", "scheduler", "epoch", "global_step", "rng_state"}
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(f"Checkpoint missing required keys: {missing}")
    gate_keys = [key for key in state["model"] if key.endswith(".gate.weight")]
    expected = 0 if arm == "baseline" else 1
    if len(gate_keys) != expected:
        raise RuntimeError(f"E16 checkpoint gate keys mismatch for {arm}: {gate_keys}")
    if gate_keys and tuple(state["model"][gate_keys[0]].shape) != (INTERMEDIATE_SIZE, HIDDEN_SIZE):
        raise RuntimeError("E16 gate checkpoint shape mismatch")
    return {"required_state_keys": sorted(required), "gate_keys": gate_keys, "epoch": state["epoch"], "global_step": state["global_step"]}


def _append_epoch_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    _write_json(path / "metadata.json", dict(metadata))


def _gate_norm(model: nn.Module) -> float | None:
    return None if model.gate is None else float(model.gate.weight.detach().float().norm().item())


def _grad_norm(model: nn.Module) -> float | None:
    values = [parameter.grad.detach().float().norm() for parameter in model.parameters() if parameter.grad is not None]
    return float(torch.stack(values).norm().item()) if values else 0.0


def _run_probe(model: Any, batch: Mapping[str, Any], device: torch.device, precision: str) -> dict[str, Any]:
    model.eval(); model.set_diagnostics(True)
    moved = e1_train._move_batch(batch, device)
    with torch.inference_mode():
        with e1_train.make_autocast(precision, device):
            model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
    diagnostics = model.diagnostics()
    model.set_diagnostics(False); model.train()
    return {"gate_s_rms": diagnostics.get("s_rms", 0.0), "gate_s_abs_p95": diagnostics.get("s_abs_p95", 0.0), "correction_norm": diagnostics.get("correction_norm", 0.0), "correction_relative_norm": diagnostics.get("delta_relative_norm", 0.0), "f0_norm": diagnostics.get("f0_norm", 0.0), "dtypes": model.target_layer.last_dtypes}


def _run_trial(arm: str, task_id: str, device_name: str, output_dir: Path, expected_source_hash: str, calibration: Mapping[str, Any]) -> dict[str, Any]:
    if source_hash() != expected_source_hash:
        raise RuntimeError("E16 source changed after controller freeze")
    e1_train.seed_everything(SEED)
    device = e1_train.resolve_device(device_name)
    precision = e1_train.resolve_precision(PRECISION, device)
    contract = _task_contract(task_id)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    lengths = e1_train.inspect_train_lengths(tokenizer, list(e1_train.split_texts(contract["train"])))
    model, model_spec = build_e16_classifier(MODEL_DIR, num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, arm=arm, calibration_sc=float(calibration["sc"]), calibration_sx=float(calibration["sx"]), attention_implementation=ATTENTION_IMPLEMENTATION)
    model.to(device)
    optimizer = e1_train.make_optimizer(model, TASK_LRS[task_id])
    coverage = _optimizer_coverage(model, optimizer, TASK_LRS[task_id])
    train_loader = e1_train._make_loader(tokenizer, contract["train"], max_length=MAX_LENGTH, batch_size=MICRO_BATCH_SIZE, shuffle=True, seed=SEED, labels=e1_train.model_labels(contract["bundle"], "train", contract["train"]), gold_labels=e1_train.split_labels(contract["train"]))
    valid_loader = e1_train._make_loader(tokenizer, contract["validation"], max_length=MAX_LENGTH, batch_size=MICRO_BATCH_SIZE, shuffle=False, seed=SEED, labels=e1_train.model_labels(contract["bundle"], "validation", contract["validation"]), gold_labels=e1_train.split_labels(contract["validation"]))
    probe_loader = e1_train._make_loader(tokenizer, contract["train"], max_length=MAX_LENGTH, batch_size=MICRO_BATCH_SIZE, shuffle=False, seed=SEED, labels=e1_train.model_labels(contract["bundle"], "train", contract["train"]), gold_labels=e1_train.split_labels(contract["train"]))
    probe_batch = next(iter(probe_loader))
    total_steps = math.ceil(len(train_loader) / ACCUMULATION_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, round(total_steps * 0.1), total_steps)
    metadata: dict[str, Any] = {
        "protocol": PROTOCOL, "revision": REVISION, "arm": arm, "task_id": task_id, "seed": SEED, "stage": "validation", "learning_rate": TASK_LRS[task_id], "epochs": EPOCHS, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "micro_batch_size": MICRO_BATCH_SIZE, "gradient_accumulation_steps": ACCUMULATION_STEPS, "max_length": MAX_LENGTH, "precision": precision, "attention_implementation": ATTENTION_IMPLEMENTATION, "weight_decay": 0.01, "betas": [0.9, 0.999], "adamw_epsilon": 1e-8, "warmup_fraction": 0.1, "warmup_steps": round(total_steps * 0.1), "gradient_clip_norm": 1.0, "loss": "plain_cross_entropy", "no_noise_consistency": True, "no_knowledge_distillation": True, "no_layerwise_learning_rate_decay": True, "include_test": False, "test_accessed": False, "source_hash": expected_source_hash, "model_source": HF_REPO_ID, "hf_commit": HF_COMMIT, "checkpoint_source": "models/bamibert-base-original", "weight_sha256": WEIGHT_SHA256, "model_hashes": _directory_hashes(MODEL_DIR), "tokenizer_hashes": _tokenizer_hashes(MODEL_DIR), "model": model_spec.to_dict(), "gate": {"shape": [INTERMEDIATE_SIZE, HIDDEN_SIZE] if model.gate is not None else None, "bias": False, "init": "zero" if model.gate is not None else "none", "target_layer_index": INSERTION_LAYER_INDEX, "formula": "fnew=f0+W2(z*tanh(Gv))"}, "calibration": {"manifest": str((ARTIFACT_ROOT / "calibration_manifest.json").relative_to(ROOT)), "sc": float(calibration["sc"]), "sx": float(calibration["sx"]), "seed": CALIBRATION_SEED, "train_only": True}, "optimizer_coverage": coverage, "data_provenance": e1_train.bundle_provenance(contract["bundle"], ROOT), "task_metadata": e1_train.task_metadata(contract["bundle"]), "label_name_overlay": _label_overlay(contract["bundle"]), "runtime": _runtime(device), "length_stats": lengths.to_dict(), "started_at": _now(), "started_at_unix": time.time(), "argv": list(sys.argv), "fixed_probe": {"split": "train", "count": len(probe_batch["ids"]), "ids": list(probe_batch["ids"])}}
    if output_dir.exists():
        required_complete = all((output_dir / name).is_file() for name in ("metadata.json", "best_checkpoint.pt", "validation_predictions.json"))
        if required_complete:
            raise FileExistsError(f"E16 trial artifact already completed: {output_dir}")
        stale = output_dir.with_name(f"{output_dir.name}__stale_{int(time.time())}_{os.getpid()}")
        output_dir.rename(stale)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "resolved_config.json", metadata)
    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_predictions: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    global_step = 0
    gradient_log = output_dir / "gradient_logs.jsonl"
    probe_log = output_dir / "probe_logs.jsonl"
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0; total_examples = 0; epoch_started = time.time(); last_epoch_step: dict[str, Any] | None = None
        for batch_index, batch in enumerate(train_loader):
            moved = e1_train._move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                loss = e1_train.compute_loss(logits, moved["labels"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"])
                window_start = (batch_index // ACCUMULATION_STEPS) * ACCUMULATION_STEPS
                window_end = min(window_start + ACCUMULATION_STEPS, len(train_loader))
                window_examples = sum(min(MICRO_BATCH_SIZE, len(train_loader.dataset) - index * MICRO_BATCH_SIZE) for index in range(window_start, window_end))
                scaled_loss = loss * (len(batch["labels"]) / max(window_examples, 1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite E16 loss at epoch={epoch}, batch={batch_index}")
            scaled_loss.backward()
            total_loss += float(loss.detach().cpu()) * len(batch["labels"]); total_examples += len(batch["labels"])
            if (batch_index + 1) % ACCUMULATION_STEPS == 0 or batch_index + 1 == len(train_loader):
                step_label = global_step
                pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item())
                post_clip = _grad_norm(model)
                gate_before = model.gate.weight.detach().float().clone() if model.gate is not None else None
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                update_norm = float((model.gate.weight.detach().float() - gate_before).norm().item()) if gate_before is not None else None
                row = {"epoch": epoch, "optimizer_step": step_label, "logged_at": _now(), "arm": arm, "grad_norm_pre_clip": pre_clip, "grad_norm_post_clip": post_clip, "gate_weight_norm": _gate_norm(model), "gate_update_norm": update_norm, "learning_rate": scheduler.get_last_lr()[0]}
                if step_label in LOG_STEPS:
                    _write_jsonl(gradient_log, row)
                last_epoch_step = row
                global_step += 1
        probe = _run_probe(model, probe_batch, device, precision)
        _write_jsonl(probe_log, {"epoch": epoch, "optimizer_step": global_step, "logged_at": _now(), **probe})
        val_metrics, val_predictions = e1_train.run_validation(model, valid_loader, device, precision, task_id, contract["bundle"])
        score = e1_train.selection_score(val_metrics)
        row = {"epoch": epoch, "global_step": global_step, "train_loss": total_loss / max(total_examples, 1), "validation": val_metrics, "selection_f1": score, "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - epoch_started, "gradient_epoch_end": last_epoch_step, "train_probe": probe}
        curves.append(row); _write_json(output_dir / "epoch_curves.json", {"epochs": curves})
        if score > best_score:
            best_score, best_epoch, best_metrics, best_predictions = score, epoch, val_metrics, val_predictions
            e1_train.save_trial_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, scheduler, epoch=epoch, global_step=global_step, metadata=metadata)
            _write_json(output_dir / "best_validation_metrics.json", best_metrics); _write_json(output_dir / "validation_predictions.json", {"rows": best_predictions})
    metadata.update({"best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics, "elapsed_seconds": time.time() - metadata["started_at_unix"], "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "status": "completed", "final_optimizer_step": global_step, "gradient_log": str(gradient_log.relative_to(ROOT)), "probe_log": str(probe_log.relative_to(ROOT))})
    checkpoint = output_dir / "best_checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False); state["metadata"] = dict(metadata); torch.save(state, checkpoint)
    metadata["checkpoint_sha256"] = _sha256(checkpoint); metadata["checkpoint_roundtrip"] = _validate_checkpoint_state(checkpoint, arm)
    model.load_state_dict(state["model"], strict=True)
    final_probe = _run_probe(model, probe_batch, device, precision)
    metadata["final_train_probe"] = final_probe
    _append_epoch_metadata(output_dir, metadata)
    _write_json(output_dir / "status.json", {"status": "completed", "protocol": PROTOCOL, "arm": arm, "task_id": task_id, "best_epoch": best_epoch, "best_selection_f1": best_score, "finished_at": _now(), "pid": os.getpid(), "device": device_name, "artifact_dir": str(output_dir)})
    return {"arm": arm, "task_id": task_id, "learning_rate": TASK_LRS[task_id], "artifact_dir": str(output_dir), "best_epoch": best_epoch, "best_selection_f1": best_score, "validation_metrics": best_metrics, "checkpoint_sha256": metadata["checkpoint_sha256"], "status": "completed"}


def _repair_gate_artifacts() -> int:
    """Finalize gate trials that trained before the checkpoint-key validator fix."""
    expected = source_hash()
    calibration = _read_json(ARTIFACT_ROOT / "calibration_manifest.json")
    device = e1_train.resolve_device("cuda:0")
    precision = e1_train.resolve_precision(PRECISION, device)
    for task_id in TASKS:
        arm = "attention_update_gate"
        key = _trial_key(arm, task_id)
        artifact = _artifact_dir(arm, task_id)
        checkpoint = artifact / "best_checkpoint.pt"
        if not checkpoint.is_file():
            continue
        if (artifact / "metadata.json").is_file() and (_load_worker(key) or {}).get("status") == "completed":
            continue
        contract = _task_contract(task_id)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        curves = _read_json(artifact / "epoch_curves.json").get("epochs", [])
        if len(curves) < EPOCHS:
            continue
        _validate_checkpoint_state(checkpoint, arm)
        model, _spec = build_e16_classifier(MODEL_DIR, num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, arm=arm, calibration_sc=float(calibration["sc"]), calibration_sx=float(calibration["sx"]))
        model.to(device)
        model.load_state_dict(state["model"], strict=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
        probe_loader = e1_train._make_loader(tokenizer, contract["train"], max_length=MAX_LENGTH, batch_size=MICRO_BATCH_SIZE, shuffle=False, seed=SEED, labels=e1_train.model_labels(contract["bundle"], "train", contract["train"]), gold_labels=e1_train.split_labels(contract["train"]))
        probe = _run_probe(model, next(iter(probe_loader)), device, precision)
        metadata = _read_json(artifact / "resolved_config.json")
        metrics = _read_json(artifact / "best_validation_metrics.json")
        best = max(curves, key=lambda row: (float(row.get("selection_f1", -math.inf)), -int(row.get("epoch", 0)))) if curves else {}
        metadata.update({"source_hash": expected, "best_epoch": int(best.get("epoch", state.get("epoch", 0))), "best_selection_f1": float(metrics["selection_f1"]), "best_validation_metrics": metrics, "status": "completed", "test_accessed": False, "final_optimizer_step": int(state.get("global_step", 0)), "final_train_probe": probe, "checkpoint_roundtrip": _validate_checkpoint_state(checkpoint, arm)})
        state["metadata"] = dict(metadata)
        torch.save(state, checkpoint)
        metadata["checkpoint_sha256"] = _sha256(checkpoint)
        _write_json(artifact / "metadata.json", metadata)
        payload = _load_worker(key) or {"protocol": PROTOCOL, "revision": REVISION, "arm": arm, "task_id": task_id, "artifact_dir": str(artifact), "pid": os.getpid()}
        _worker_status(key, {**payload, "status": "completed", "finished_at": _now(), "artifact_dir": str(artifact), "best_epoch": metadata["best_epoch"], "best_selection_f1": metadata["best_selection_f1"], "checkpoint_sha256": metadata["checkpoint_sha256"]})
        del model
        torch.cuda.empty_cache()
    return 0


def _load_worker(key: str) -> dict[str, Any] | None:
    path = WORKER_ROOT / f"{key}.json"
    return _read_json(path) if path.is_file() else None


def _controller_state(payload: Mapping[str, Any]) -> None:
    _write_json(ORCH_ROOT / "controller_state.json", dict(payload))


def _spawn_worker(spec: Mapping[str, Any], gpu_index: str, expected_source_hash: str) -> subprocess.Popen[Any]:
    key = str(spec["key"])
    log_path = LOG_ROOT / f"{key}__gpu{gpu_index}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", str(spec["arm"]), "--task-id", str(spec["task_id"]), "--device", "cuda:0", "--expected-source-hash", expected_source_hash]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = gpu_index; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"; env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env, start_new_session=True)
    handle.close()
    _write_json(ORCH_ROOT / f"launch_{key}.json", {"key": key, "pid": process.pid, "gpu": gpu_index, "device": "cuda:0", "micro_batch_size": MICRO_BATCH_SIZE, "launched_at": _now(), "log": str(log_path.relative_to(ROOT))})
    return process


def _worker_main(args: argparse.Namespace) -> int:
    key = _trial_key(args.arm, args.task_id)
    output = _artifact_dir(args.arm, args.task_id)
    calibration = _read_json(ARTIFACT_ROOT / "calibration_manifest.json")
    payload = {"protocol": PROTOCOL, "revision": REVISION, "status": "running", "pid": os.getpid(), "arm": args.arm, "task_id": args.task_id, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", args.device), "device": args.device, "micro_batch_size": MICRO_BATCH_SIZE, "gradient_accumulation_steps": ACCUMULATION_STEPS, "started_at": _now(), "artifact_dir": str(output)}
    _worker_status(key, payload)
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(30.0):
            _worker_status(key, {**payload, "status": "running", "heartbeat_at": _now()})

    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        result = _run_trial(args.arm, args.task_id, args.device, output, args.expected_source_hash, calibration)
        _worker_status(key, {**payload, **result, "status": "completed", "finished_at": _now()})
        return 0
    except Exception as error:
        _worker_status(key, {**payload, "status": "failed", "finished_at": _now(), "error": repr(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    finally:
        stop_heartbeat.set(); thread.join(timeout=1.0)


def _run_controller() -> dict[str, Any]:
    for directory in (ARTIFACT_ROOT, ORCH_ROOT, WORKER_ROOT, LOG_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    preflight_path = ARTIFACT_ROOT / "preflight.json"
    if preflight_path.is_file():
        preflight_value = _read_json(preflight_path)
        if preflight_value.get("source_hash") != source_hash():
            raise RuntimeError("Existing E16 preflight source hash differs; refuse to continue")
        calibration = _read_json(ARTIFACT_ROOT / "calibration_manifest.json")
    else:
        preflight_value = preflight()
        calibration = preflight_value["calibration"]
    expected = str(preflight_value["source_hash"])
    specs = [{"key": _trial_key(arm, task), "arm": arm, "task_id": task, "learning_rate": TASK_LRS[task]} for arm in ARMS for task in TASKS]
    completed: set[str] = set(); running: dict[str, tuple[subprocess.Popen[Any] | None, str, int]] = {}; queue: list[dict[str, Any]] = []
    for spec in specs:
        key = str(spec["key"]); status = _load_worker(key)
        artifact = _artifact_dir(str(spec["arm"]), str(spec["task_id"]))
        if status and status.get("status") == "completed" and Path(str(status.get("artifact_dir", artifact))).is_dir() and all((Path(str(status.get("artifact_dir", artifact))) / name).is_file() for name in ("metadata.json", "best_checkpoint.pt", "validation_predictions.json")):
            completed.add(key)
        elif status and status.get("status") == "running" and _pid_alive(status.get("pid")):
            running[key] = (None, str(status.get("gpu", "0")), MICRO_BATCH_SIZE)
        else:
            queue.append(spec)
    while queue or running:
        snapshot = _gpu_snapshot()
        used = {value[1] for value in running.values()}
        for gpu_index in GPU_INDICES:
            if gpu_index in used or not queue or gpu_index not in snapshot or snapshot[gpu_index]["free_mib"] < MIN_FREE_MIB:
                continue
            spec = queue.pop(0)
            process = _spawn_worker(spec, gpu_index, expected)
            running[str(spec["key"])] = (process, gpu_index, MICRO_BATCH_SIZE)
            used.add(gpu_index)
        _controller_state({"protocol": PROTOCOL, "revision": REVISION, "status": "validation_running", "updated_at": _now(), "pid": os.getpid(), "expected_source_hash": expected, "gpu_snapshot": snapshot, "completed_keys": sorted(completed), "queued_keys": [str(spec["key"]) for spec in queue], "running": {key: {"gpu": gpu, "micro_batch_size": micro, "pid": process.pid if process is not None else (_load_worker(key) or {}).get("pid"), "arm": next(item["arm"] for item in specs if item["key"] == key), "task_id": next(item["task_id"] for item in specs if item["key"] == key)} for key, (process, gpu, micro) in running.items()}})
        for key, (process, gpu, micro) in list(running.items()):
            status = _load_worker(key) or {}
            if status.get("status") == "completed":
                completed.add(key); del running[key]; continue
            if status.get("status") == "failed":
                # A just-launched process can still be replacing a stale
                # worker JSON. Trust failure only after that exact process has
                # exited; this prevents the controller startup race where a
                # previous failed status killed a healthy new worker.
                if process is not None and process.poll() is None:
                    continue
                raise RuntimeError(f"E16 worker failed for {key}: {status.get('error')}")
            if process is not None and process.poll() is not None and process.returncode != 0:
                raise RuntimeError(f"E16 worker process failed for {key}: returncode={process.returncode}")
            if process is None and not _pid_alive(status.get("pid")):
                raise RuntimeError(f"E16 adopted worker disappeared for {key}: {status}")
        time.sleep(10)
    _controller_state({"protocol": PROTOCOL, "revision": REVISION, "status": "validation_trials_completed", "updated_at": _now(), "completed_keys": sorted(completed), "source_hash": expected})
    return {"preflight": preflight_value, "specs": specs, "source_hash": expected}


def _read_e14_baseline() -> dict[str, float]:
    path = ROOT / "artifacts" / "e14" / "validation_grid.json"
    if not path.is_file():
        return {task: float("nan") for task in TASKS}
    grid = _read_json(path).get("grid", [])
    return {task: max((float(row["selection_f1"]) for row in grid if row["task_id"] == task), default=float("nan")) for task in TASKS}


def _write_validation_outputs(specs: Sequence[Mapping[str, Any]], source_hash_value: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    e14 = _read_e14_baseline()
    for spec in specs:
        status = _load_worker(str(spec["key"])) or {}
        artifact = Path(str(status.get("artifact_dir", _artifact_dir(str(spec["arm"]), str(spec["task_id"]))))); metadata = _read_json(artifact / "metadata.json"); metrics = _read_json(artifact / "best_validation_metrics.json"); predictions = _read_json(artifact / "validation_predictions.json")["rows"]
        bundle = _task_contract(str(spec["task_id"]))["bundle"]
        recomputed = e1_train.evaluate_predictions(str(spec["task_id"]), bundle, [row["gold"] for row in predictions], [row["prediction"] for row in predictions], split="validation")
        if abs(float(recomputed["selection_f1"]) - float(metrics["selection_f1"])) > 1e-12:
            raise RuntimeError(f"E16 validation metric recompute mismatch: {spec}")
        rows.append({"arm": spec["arm"], "task_id": spec["task_id"], "learning_rate": float(spec["learning_rate"]), "best_epoch": int(metadata["best_epoch"]), "selection_f1": float(metrics["selection_f1"]), "selection_metric": metrics["selection_metric"], "e16_baseline_selection_f1": None, "delta_vs_e16_baseline_pp": None, "e14_historical_selection_f1": e14[str(spec["task_id"])], "delta_vs_e14_historical_pp": 100.0 * (float(metrics["selection_f1"]) - e14[str(spec["task_id"])]), "artifact_dir": str(artifact), "checkpoint_sha256": _sha256(artifact / "best_checkpoint.pt"), "source_hash": metadata.get("source_hash"), "test_accessed": False, "calibration_sc": metadata["calibration"]["sc"], "calibration_sx": metadata["calibration"]["sx"]})
    baseline = {row["task_id"]: row for row in rows if row["arm"] == "baseline"}
    for row in rows:
        row["e16_baseline_selection_f1"] = baseline[row["task_id"]]["selection_f1"]
        row["delta_vs_e16_baseline_pp"] = 100.0 * (row["selection_f1"] - baseline[row["task_id"]]["selection_f1"])
    attention = {row["task_id"]: row for row in rows if row["arm"] == "attention_update_gate"}
    ffn = {row["task_id"]: row for row in rows if row["arm"] == "ffn_input_gate"}
    gate_rows = [{"task_id": task, "baseline_selection_f1": baseline[task]["selection_f1"], "attention_update_gate_selection_f1": attention[task]["selection_f1"], "ffn_input_gate_selection_f1": ffn[task]["selection_f1"], "attention_delta_vs_baseline_pp": attention[task]["delta_vs_e16_baseline_pp"], "ffn_delta_vs_baseline_pp": ffn[task]["delta_vs_e16_baseline_pp"], "attention_delta_vs_ffn_pp": 100.0 * (attention[task]["selection_f1"] - ffn[task]["selection_f1"])} for task in TASKS]
    attention_gain = [float(row["attention_delta_vs_baseline_pp"]) for row in gate_rows]
    attention_vs_ffn = [float(row["attention_delta_vs_ffn_pp"]) for row in gate_rows]
    gate = {"predeclared_attention_vs_baseline": {"gain_tasks": sum(value >= 0 for value in attention_gain), "gain_ge_0_3pp_tasks": sum(value >= 0.3 for value in attention_gain), "drop_worse_than_0_3pp_tasks": sum(value < -0.3 for value in attention_gain), "passes": sum(value >= 0 for value in attention_gain) >= 4 and sum(value >= 0.3 for value in attention_gain) >= 2 and all(value >= -0.3 for value in attention_gain)}, "predeclared_attention_vs_ffn": {"win_tasks": sum(value >= 0 for value in attention_vs_ffn), "loss_worse_than_0_3pp_tasks": sum(value < -0.3 for value in attention_vs_ffn), "passes": sum(value >= 0 for value in attention_vs_ffn) >= 3 and all(value >= -0.3 for value in attention_vs_ffn)}}
    gate["test_review_eligible"] = bool(gate["predeclared_attention_vs_baseline"]["passes"] and gate["predeclared_attention_vs_ffn"]["passes"])
    payload = {"protocol": PROTOCOL, "revision": REVISION, "source_hash": source_hash_value, "fractions_0_to_1": True, "display_delta_unit": "percentage_points", "primary_baseline": "E16 baseline arm", "historical_reference": "E14 validation grid only", "rows": rows, "task_comparison": gate_rows, "gate": gate, "test_accessed": False, "test_authorized": False}
    _write_json(ARTIFACT_ROOT / "validation_summary.json", payload)
    with (ARTIFACT_ROOT / "validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    _write_json(ARTIFACT_ROOT / "gate.json", payload)
    _write_json(ARTIFACT_ROOT / "locked_validation_manifest.json", {"protocol": PROTOCOL, "locked": True, "locked_at": _now(), "source_hash": source_hash_value, "test_accessed_before_lock": False, "selections": rows, "gate": gate})
    _write_result_markdown(payload)
    return payload


def _write_result_markdown(payload: Mapping[str, Any]) -> None:
    lines = [f"# E16 validation results", "", f"Protocol: `{payload['protocol']}`", "", "All 15 validation trials completed with fractions in [0, 1]. Display deltas use percentage points. E14 values are historical context; E16 baseline is the primary reference.", "", "| Arm | Task | Best epoch | Validation selection F1 | Delta vs E16 baseline (pp) | Delta vs E14 historical (pp) |", "|---|---|---:|---:|---:|---:|"]
    for row in payload["rows"]:
        delta = "" if row["delta_vs_e16_baseline_pp"] is None else f"{row['delta_vs_e16_baseline_pp']:.4f}"
        hist = "" if not math.isfinite(float(row["delta_vs_e14_historical_pp"])) else f"{row['delta_vs_e14_historical_pp']:.4f}"
        lines.append(f"| {row['arm']} | {row['task_id']} | {row['best_epoch']} | {row['selection_f1']:.8f} | {delta} | {hist} |")
    lines += ["", "## Predeclared gate facts", "", "```json", json.dumps(payload["gate"], indent=2), "```", "", "Test split access: false.", ""]
    (ARTIFACT_ROOT / "E16_RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def _controller_main() -> int:
    context = _run_controller()
    _write_validation_outputs(context["specs"], context["source_hash"])
    _controller_state({"protocol": PROTOCOL, "revision": REVISION, "status": "validation_completed_test_closed", "updated_at": _now(), "source_hash": context["source_hash"], "test_authorized": False})
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--controller", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--repair", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--device")
    parser.add_argument("--expected-source-hash")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.worker:
        if not all(value is not None for value in (args.arm, args.task_id, args.device, args.expected_source_hash)):
            raise SystemExit("worker requires --arm, --task-id, --device and --expected-source-hash")
        return _worker_main(args)
    if args.preflight:
        preflight()
        return 0
    if args.repair:
        return _repair_gate_artifacts()
    return _controller_main()


if __name__ == "__main__":
    raise SystemExit(main())
