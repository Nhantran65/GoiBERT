"""Execute the E14 original BamiBERT local baseline.

The runner is self-contained around the frozen E1 loader, loss, and evaluator.
It has two modes: ``--controller`` schedules the locked 15 validation trials
and then the five locked test evaluations; ``--worker`` executes one trial.
No test split is loaded during trial selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
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
ARTIFACT_ROOT = ROOT / "artifacts" / "e14"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
MODEL_DIR = ROOT / "models" / "bamibert-base-original"
GD_MODEL_DIR = ROOT / "models" / "goiturn-s-gd"
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MAX_LENGTH = 256
ATTENTION_IMPLEMENTATION = "sdpa"
PRECISION = "bf16"
LEARNING_RATES = (1e-5, 2e-5, 5e-5)
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
HF_REPO_ID = "Qualcomm-AI-Research/BamiBERT"
HF_COMMIT = "57bc1340debbe4e348ec549047a763caebe4a977"
PROTOCOL = "E14-BamiBERT-original-local-baseline-v1"
REVISION = "e14-20260912-r1"
GPU_INDICES = ("0", "1", "2", "3")
MIN_FREE_MIB = 12_000

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e14.modeling import build_bamibert_classifier  # noqa: E402


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
        *sorted((SRC / "goibert_e14").rglob("*.py")),
        SRC / "goibert" / "train.py",
        SRC / "goibert" / "data.py",
        SRC / "goibert" / "metrics.py",
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


def _tokenizer_comparison(tokenizer: Any, task_ids: Sequence[str] = TASKS) -> dict[str, Any]:
    gd_tokenizer = AutoTokenizer.from_pretrained(GD_MODEL_DIR, local_files_only=True, use_fast=True)
    original_hashes = _tokenizer_hashes(MODEL_DIR)
    gd_hashes = _tokenizer_hashes(GD_MODEL_DIR)
    details: dict[str, Any] = {"original_hashes": original_hashes, "gd_hashes": gd_hashes, "hashes_equal": original_hashes == gd_hashes, "tasks": {}}
    for task_id in task_ids:
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
        texts = list(e1_train.split_texts(e1_train.split_field(bundle, "train")))[:200]
        valid_texts = list(e1_train.split_texts(e1_train.split_field(bundle, "validation")))[:200]
        sample = texts + valid_texts
        original = tokenizer(sample, add_special_tokens=True, truncation=False)["input_ids"]
        gd = gd_tokenizer(sample, add_special_tokens=True, truncation=False)["input_ids"]
        differences = sum(a != b for a, b in zip(original, gd))
        details["tasks"][task_id] = {"sample_count": len(sample), "different_token_sequences": differences, "same": differences == 0}
    return details


def _label_overlay(bundle: Any) -> dict[str, Any]:
    return {
        "vsfc_sentiment": {"0": "negative", "1": "neutral", "2": "positive"},
        "vsfc_topic": {str(index): name for index, name in enumerate(getattr(bundle, "label_names", ())) } if getattr(bundle, "task_id", "") == "vsfc_topic" else {"0": "lecturer", "1": "training_program", "2": "facility", "3": "others"},
        "absa": {"0": "absent", "1": "positive", "2": "negative", "3": "neutral"},
        "visfd_others": {"raw_0": "absent", "raw_4": "present_detection_only", "model_0": "absent", "model_1": "present_detection_only"},
        "numeric_gold_unchanged": True,
    }


def _model_manifest() -> dict[str, Any]:
    hashes = _directory_hashes(MODEL_DIR)
    config = _read_json(MODEL_DIR / "config.json")
    return {
        "source": "huggingface",
        "repository": HF_REPO_ID,
        "hf_commit": HF_COMMIT,
        "local_path": str(MODEL_DIR.relative_to(ROOT)),
        "files": hashes,
        "config": config,
        "architecture_check": {"num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 3072},
        "weight_file_sha256": hashes.get("model.safetensors"),
        "not_teacher_dapt": True,
    }


def _validate_checkpoint_state(path: Path, expected_head_hidden: int = 768) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Checkpoint missing model state: {path}")
    keys = set(state["model"])
    dense_weight = state["model"].get("head.dense.weight")
    if dense_weight is None or tuple(dense_weight.shape) != (expected_head_hidden, expected_head_hidden):
        raise RuntimeError(f"Checkpoint head.dense.weight does not round-trip as [{expected_head_hidden},{expected_head_hidden}]")
    return {"required_state_keys": sorted({"model", "optimizer", "scheduler", "epoch", "global_step", "rng_state"}.intersection(state)), "model_key_count": len(keys), "head_dense_shape": list(dense_weight.shape), "epoch": state.get("epoch"), "global_step": state.get("global_step")}


def _preflight_model(device_name: str, micro_batch_size: int) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"E14 requires CUDA preflight, got {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    encoded = tokenizer(["đây là kiểm tra BamiBERT"] * micro_batch_size, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
    model, spec = build_bamibert_classifier(MODEL_DIR, num_classes=3, attention_implementation=ATTENTION_IMPLEMENTATION)
    model.to(device)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    batch = {key: value.to(device) for key, value in encoded.items()}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = logits.float().square().mean()
    loss.backward()
    finite = bool(torch.isfinite(loss).item() and all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()))
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    del model, logits, loss, batch
    torch.cuda.empty_cache()
    return {"device": device_name, "micro_batch_size": micro_batch_size, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "dtype": "bfloat16", "finite_loss_and_gradients": finite, "logits_shape": list(logits.shape) if "logits" in locals() else [micro_batch_size, 3], "peak_memory_allocated_bytes": peak_alloc, "peak_memory_reserved_bytes": peak_reserved, "runtime": _runtime(device), "model_spec": spec.to_dict()}


def preflight() -> dict[str, Any]:
    snapshot = _gpu_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    tokenizer_comparison = _tokenizer_comparison(tokenizer)
    contracts = {}
    for task_id in TASKS:
        contract = _task_contract(task_id)
        bundle = contract["bundle"]
        contracts[task_id] = {"train_count": len(contract["train"]), "validation_count": len(contract["validation"]), "num_classes": contract["num_classes"], "num_aspects": contract["num_aspects"], "aspect_class_counts": list(contract["aspect_class_counts"]), "label_overlay": _label_overlay(bundle)}
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
    return {"protocol": PROTOCOL, "revision": REVISION, "created_at": _now(), "source_hash": source_hash(), "model_manifest": _model_manifest(), "gpu_snapshot": snapshot, "gpu_profiles": profiles, "task_contracts": contracts, "tokenizer_comparison": tokenizer_comparison, "checks": {"official_model_dir": MODEL_DIR.is_dir(), "weights_present": (MODEL_DIR / "model.safetensors").is_file(), "all_tasks_loaded_without_test": len(contracts) == 5, "encoder_shape": {"layers": 12, "hidden": 768, "heads": 12, "ffn": 3072}, "head_hidden": 768, "max_length": MAX_LENGTH, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "no_test_access": True}}


def _worker_status(key: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_ROOT / f"{key}.json", dict(payload))


def _trial_key(task_id: str, learning_rate: float) -> str:
    return f"{task_id}__lr_{learning_rate:.0e}".replace("+", "")


def _artifact_dir(task_id: str, learning_rate: float) -> Path:
    return ARTIFACT_ROOT / "trials" / task_id / f"lr_{learning_rate:.0e}" / f"seed_{SEED}"


def _run_trial(task_id: str, learning_rate: float, device_name: str, micro_batch_size: int, output_dir: Path, expected_source_hash: str) -> dict[str, Any]:
    key = _trial_key(task_id, learning_rate)
    if source_hash() != expected_source_hash:
        raise RuntimeError("E14 source changed after controller freeze")
    e1_train.seed_everything(SEED)
    device = e1_train.resolve_device(device_name)
    precision = e1_train.resolve_precision(PRECISION, device)
    if micro_batch_size <= 0 or EFFECTIVE_BATCH_SIZE % micro_batch_size:
        raise ValueError("micro batch must divide effective batch size")
    accumulation_steps = EFFECTIVE_BATCH_SIZE // micro_batch_size
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
    train_split = e1_train.split_field(bundle, "train")
    valid_split = e1_train.split_field(bundle, "validation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    lengths = e1_train.inspect_train_lengths(tokenizer, list(e1_train.split_texts(train_split)))
    model, model_spec = build_bamibert_classifier(MODEL_DIR, num_classes=e1_train.task_num_classes(bundle), num_aspects=e1_train.task_num_aspects(bundle), aspect_class_counts=e1_train.task_aspect_class_counts(bundle) if e1_train.task_num_aspects(bundle) else None, attention_implementation=ATTENTION_IMPLEMENTATION)
    model.to(device)
    optimizer = e1_train.make_optimizer(model, learning_rate)
    train_loader = e1_train._make_loader(tokenizer, train_split, max_length=MAX_LENGTH, batch_size=micro_batch_size, shuffle=True, seed=SEED, labels=e1_train.model_labels(bundle, "train", train_split), gold_labels=e1_train.split_labels(train_split))
    valid_loader = e1_train._make_loader(tokenizer, valid_split, max_length=MAX_LENGTH, batch_size=micro_batch_size, shuffle=False, seed=SEED, labels=e1_train.model_labels(bundle, "validation", valid_split), gold_labels=e1_train.split_labels(valid_split))
    total_steps = math.ceil(len(train_loader) / accumulation_steps) * EPOCHS
    warmup_steps = round(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    model_hashes = _directory_hashes(MODEL_DIR)
    metadata: dict[str, Any] = {
        "protocol": PROTOCOL, "revision": REVISION, "task_id": task_id, "seed": SEED, "stage": "validation_grid", "learning_rate": learning_rate, "epochs": EPOCHS, "effective_batch_size": EFFECTIVE_BATCH_SIZE, "micro_batch_size": micro_batch_size, "gradient_accumulation_steps": accumulation_steps, "max_length": MAX_LENGTH, "precision": precision, "attention_implementation": ATTENTION_IMPLEMENTATION, "weight_decay": 0.01, "betas": [0.9, 0.999], "adamw_epsilon": 1e-8, "warmup_fraction": 0.1, "warmup_steps": warmup_steps, "gradient_clip_norm": 1.0, "loss": "plain_cross_entropy", "head": "CLS -> dropout0.1 -> dense768 -> tanh -> dropout0.1 -> logits", "no_noise_consistency": True, "no_knowledge_distillation": True, "no_layerwise_learning_rate_decay": True, "include_test": False, "source_hash": expected_source_hash, "model_source": HF_REPO_ID, "hf_commit": HF_COMMIT, "checkpoint_source": "models/bamibert-base-original", "model_hashes": model_hashes, "tokenizer_hashes": _tokenizer_hashes(MODEL_DIR), "model": model_spec.to_dict(), "data_provenance": e1_train.bundle_provenance(bundle, ROOT), "task_metadata": e1_train.task_metadata(bundle), "label_name_overlay": _label_overlay(bundle), "tokenizer_comparison": {"comparison": "see artifacts/e14/preflight.json", "source_is_original_bamibert": True}, "runtime": _runtime(device), "length_stats": lengths.to_dict(), "started_at": _now(), "started_at_unix": time.time(), "argv": list(sys.argv),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "resolved_config.json", metadata)
    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_predictions: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_examples = 0
        epoch_started = time.time()
        for batch_index, batch in enumerate(train_loader):
            moved = e1_train._move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                loss = e1_train.compute_loss(logits, moved["labels"], num_aspects=e1_train.task_num_aspects(bundle), aspect_class_counts=e1_train.task_aspect_class_counts(bundle))
                window_start = (batch_index // accumulation_steps) * accumulation_steps
                window_end = min(window_start + accumulation_steps, len(train_loader))
                window_examples = sum(min(micro_batch_size, len(train_loader.dataset) - index * micro_batch_size) for index in range(window_start, window_end))
                scaled_loss = loss * (len(batch["labels"]) / max(window_examples, 1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite E14 loss at epoch={epoch}, batch={batch_index}")
            scaled_loss.backward()
            total_loss += float(loss.detach().cpu()) * len(batch["labels"])
            total_examples += len(batch["labels"])
            boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if boundary:
                nn = torch.nn
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
        val_metrics, val_predictions = e1_train.run_validation(model, valid_loader, device, precision, task_id, bundle)
        score = e1_train.selection_score(val_metrics)
        row = {"epoch": epoch, "global_step": global_step, "train_loss": total_loss / max(total_examples, 1), "validation": val_metrics, "selection_f1": score, "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - epoch_started}
        curves.append(row); _write_json(output_dir / "epoch_curves.json", {"epochs": curves})
        if score > best_score:
            best_score, best_epoch, best_metrics, best_predictions = score, epoch, val_metrics, val_predictions
            e1_train.save_trial_checkpoint(output_dir / "best_checkpoint.pt", model, optimizer, scheduler, epoch=epoch, global_step=global_step, metadata=metadata)
            _write_json(output_dir / "best_validation_metrics.json", best_metrics)
            _write_json(output_dir / "validation_predictions.json", {"rows": best_predictions})
    metadata.update({"best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics, "elapsed_seconds": time.time() - metadata["started_at_unix"], "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None, "status": "completed"})
    checkpoint = output_dir / "best_checkpoint.pt"
    metadata["checkpoint_sha256"] = _sha256(checkpoint)
    metadata["checkpoint_roundtrip"] = _validate_checkpoint_state(checkpoint)
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "status.json", {"status": "completed", "task_id": task_id, "learning_rate": learning_rate, "best_epoch": best_epoch, "best_selection_f1": best_score, "finished_at": _now(), "pid": os.getpid(), "device": device_name})
    return {"task_id": task_id, "learning_rate": learning_rate, "artifact_dir": str(output_dir), "best_epoch": best_epoch, "best_selection_f1": best_score, "validation_metrics": best_metrics, "checkpoint_sha256": metadata["checkpoint_sha256"], "status": "completed"}


def worker_main(args: argparse.Namespace) -> int:
    task_id, learning_rate = args.task_id, float(args.learning_rate)
    key = _trial_key(task_id, learning_rate)
    payload = {"protocol": PROTOCOL, "revision": REVISION, "status": "running", "pid": os.getpid(), "task_id": task_id, "learning_rate": learning_rate, "device": args.device, "micro_batch_size": args.micro_batch_size, "started_at": _now(), "artifact_dir": str(_artifact_dir(task_id, learning_rate))}
    _worker_status(key, payload)
    try:
        result = _run_trial(task_id, learning_rate, args.device, args.micro_batch_size, _artifact_dir(task_id, learning_rate), args.expected_source_hash)
        _worker_status(key, {**payload, **result, "status": "completed", "finished_at": _now()})
        return 0
    except Exception as error:
        _worker_status(key, {**payload, "status": "failed", "finished_at": _now(), "error": repr(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1


def _profile_micro(profiles: Mapping[str, Any], gpu_index: str) -> int:
    profile = profiles.get(gpu_index, {})
    if profile.get("micro_batch_size") in {8, 16, 32}:
        return int(profile["micro_batch_size"])
    return 16 if gpu_index == "3" else 32


def _load_worker(key: str) -> dict[str, Any] | None:
    path = WORKER_ROOT / f"{key}.json"
    return _read_json(path) if path.is_file() else None


def _controller_state(payload: Mapping[str, Any]) -> None:
    _write_json(ORCH_ROOT / "controller_state.json", dict(payload))


def _spawn_worker(spec: Mapping[str, Any], gpu_index: str, micro_batch_size: int, expected_source_hash: str) -> subprocess.Popen[Any]:
    key = str(spec["key"])
    log_path = LOG_ROOT / f"{key}__gpu{gpu_index}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--task-id", str(spec["task_id"]), "--learning-rate", str(spec["learning_rate"]), "--device", f"cuda:0", "--micro-batch-size", str(micro_batch_size), "--expected-source-hash", expected_source_hash]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = gpu_index; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"; env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env, start_new_session=True)
    handle.close()
    return process


def _collect_completed_specs(specs: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    completed: set[str] = set(); active: set[str] = set()
    for spec in specs:
        status = _load_worker(str(spec["key"]))
        if not status:
            continue
        if status.get("status") == "completed" and Path(str(status.get("artifact_dir", ""))).is_dir():
            completed.add(str(spec["key"]))
        elif status.get("status") == "running" and _pid_alive(status.get("pid")):
            active.add(str(spec["key"]))
    return completed, active


def run_controller() -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True); ORCH_ROOT.mkdir(parents=True, exist_ok=True); WORKER_ROOT.mkdir(parents=True, exist_ok=True); LOG_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_path = ARTIFACT_ROOT / "preflight.json"
    if preflight_path.is_file():
        preflight_value = _read_json(preflight_path)
        if preflight_value.get("source_hash") != source_hash():
            raise RuntimeError("Existing E14 preflight source hash differs; refuse to continue")
    else:
        preflight_value = preflight(); _write_json(preflight_path, preflight_value)
    expected_source_hash = str(preflight_value["source_hash"])
    profiles = preflight_value["gpu_profiles"]
    specs = [{"key": _trial_key(task, lr), "task_id": task, "learning_rate": lr} for task in TASKS for lr in LEARNING_RATES]
    completed, active = _collect_completed_specs(specs)
    queue = [spec for spec in specs if spec["key"] not in completed and spec["key"] not in active]
    running: dict[str, tuple[subprocess.Popen[Any] | None, str, int]] = {}
    for key in active:
        status = _load_worker(key) or {}
        running[key] = (None, str(status.get("device", "cuda:0")).split(":")[-1], int(status.get("micro_batch_size", 16)))
    free_gpus = [index for index in GPU_INDICES if index in preflight_value["gpu_snapshot"] and preflight_value["gpu_snapshot"][index]["free_mib"] >= MIN_FREE_MIB]
    if not free_gpus:
        raise RuntimeError("No GPU passed E14 free-memory guard")
    while queue or running:
        used = {value[1] for value in running.values()}
        for gpu_index in free_gpus:
            if gpu_index in used or not queue:
                continue
            spec = queue.pop(0); micro = _profile_micro(profiles, gpu_index); process = _spawn_worker(spec, gpu_index, micro, expected_source_hash)
            running[str(spec["key"])] = (process, gpu_index, micro); used.add(gpu_index)
        state = {"protocol": PROTOCOL, "revision": REVISION, "status": "running", "updated_at": _now(), "pid": os.getpid(), "expected_source_hash": expected_source_hash, "completed_keys": sorted(completed), "queued_keys": [str(s["key"]) for s in queue], "running": {key: {"gpu": gpu, "micro_batch_size": micro, "pid": proc.pid if proc is not None else (_load_worker(key) or {}).get("pid")} for key, (proc, gpu, micro) in running.items()}}
        _controller_state(state)
        for key, (process, gpu_index, micro) in list(running.items()):
            status = _load_worker(key) or {}
            if status.get("status") == "completed":
                completed.add(key); del running[key]; continue
            if status.get("status") == "failed" or (process is not None and process.poll() is not None and process.returncode != 0):
                error_text = str(status.get("error", ""))
                if "out of memory" in error_text.lower() and micro > 8:
                    trial = next(spec for spec in specs if spec["key"] == key)
                    queue.insert(0, trial); del running[key]; continue
                raise RuntimeError(f"E14 worker failed for {key}: {status.get('error', process.returncode if process else 'unknown')}")
        time.sleep(10)
    _controller_state({"protocol": PROTOCOL, "revision": REVISION, "status": "validation_trials_completed", "updated_at": _now(), "completed_keys": sorted(completed), "source_hash": expected_source_hash})
    return {"preflight": preflight_value, "specs": specs, "source_hash": expected_source_hash}


def _select_and_lock(specs: Sequence[Mapping[str, Any]], source_hash_value: str) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    grid: list[dict[str, Any]] = []
    for spec in specs:
        artifact = _artifact_dir(str(spec["task_id"]), float(spec["learning_rate"]))
        metadata = _read_json(artifact / "metadata.json"); metrics = _read_json(artifact / "best_validation_metrics.json")
        score = float(metadata["best_selection_f1"])
        grid.append({"task_id": spec["task_id"], "learning_rate": float(spec["learning_rate"]), "best_epoch": int(metadata["best_epoch"]), "selection_f1": score, "selection_metric": metrics.get("selection_metric"), "artifact_dir": str(artifact), "checkpoint_sha256": _sha256(artifact / "best_checkpoint.pt"), "source_hash": metadata.get("source_hash"), "model_source": metadata.get("model_source"), "hf_commit": metadata.get("hf_commit"), "test_accessed": False})
    for task_id in TASKS:
        candidates = [row for row in grid if row["task_id"] == task_id]
        chosen = sorted(candidates, key=lambda row: (-row["selection_f1"], row["learning_rate"]))[0]
        selections.append({**chosen, "locked": True, "lock_time": _now()})
    manifest = {"protocol": "E14-locked-selection-v1", "experiment_protocol": PROTOCOL, "revision": REVISION, "locked": True, "locked_at": _now(), "seed": SEED, "source_hash": source_hash_value, "model_source": HF_REPO_ID, "hf_commit": HF_COMMIT, "validation_only_selection": True, "test_accessed_before_lock": False, "grid": grid, "selections": selections}
    _write_json(ARTIFACT_ROOT / "validation_grid.json", {"protocol": PROTOCOL, "grid": grid})
    with (ARTIFACT_ROOT / "validation_grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "learning_rate", "best_epoch", "selection_f1", "selection_metric", "artifact_dir", "checkpoint_sha256", "source_hash", "model_source", "hf_commit", "test_accessed"]); writer.writeheader(); writer.writerows(grid)
    _write_json(ARTIFACT_ROOT / "locked_selection_manifest.json", manifest)
    return manifest


def _evaluate_test(selection: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(selection["task_id"]); artifact = Path(str(selection["artifact_dir"])); metadata = _read_json(artifact / "metadata.json")
    if selection.get("test_accessed_before_lock") or selection.get("locked") is not True:
        raise RuntimeError("Test gate requires a locked E14 selection")
    if metadata.get("model_source") != HF_REPO_ID or metadata.get("hf_commit") != HF_COMMIT:
        raise RuntimeError("Locked artifact is not the official BamiBERT source")
    if _sha256(artifact / "best_checkpoint.pt") != selection["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint hash changed after lock for {task_id}")
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    contract = {"bundle": bundle, "train": e1_train.split_field(bundle, "train"), "test": e1_train.split_field(bundle, "test")}
    model, spec = build_bamibert_classifier(MODEL_DIR, num_classes=e1_train.task_num_classes(bundle), num_aspects=e1_train.task_num_aspects(bundle), aspect_class_counts=e1_train.task_aspect_class_counts(bundle) if e1_train.task_num_aspects(bundle) else None, attention_implementation=ATTENTION_IMPLEMENTATION)
    device_name = "cuda:0"
    device = e1_train.resolve_device(device_name)
    model.to(device)
    state = torch.load(artifact / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    test_split = contract["test"]
    test_loader = e1_train._make_loader(tokenizer, test_split, max_length=MAX_LENGTH, batch_size=int(metadata["micro_batch_size"]), shuffle=False, seed=SEED, labels=e1_train.model_labels(bundle, "test", test_split), gold_labels=e1_train.split_labels(test_split))
    metrics, rows = e1_train.run_test(model, test_loader, device, PRECISION, task_id, bundle)
    recomputed = e1_train.evaluate_predictions(task_id, bundle, [row["gold"] for row in rows], [row["prediction"] for row in rows], split="test")
    if abs(float(recomputed["selection_f1"]) - float(metrics["selection_f1"])) > 1e-12:
        raise RuntimeError(f"Test metric recompute mismatch for {task_id}")
    _write_json(artifact / "test_metrics.json", metrics); _write_json(artifact / "test_predictions.json", {"rows": rows})
    evaluation = {"protocol": "E14-test-gate-v1", "task_id": task_id, "split": "test", "locked_selection_manifest": str((ARTIFACT_ROOT / "locked_selection_manifest.json").relative_to(ROOT)), "checkpoint_sha256": _sha256(artifact / "best_checkpoint.pt"), "model_source": HF_REPO_ID, "hf_commit": HF_COMMIT, "model_spec": spec.to_dict(), "test_count": len(rows), "id_count": len({row["id"] for row in rows}), "gold_count": len(rows), "metrics": metrics, "metric_recomputed_from_saved_predictions": True, "accessed_at": _now(), "tokenizer_hashes": _tokenizer_hashes(MODEL_DIR)}
    _write_json(artifact / "test_evaluation_metadata.json", evaluation)
    return {"task_id": task_id, "learning_rate": selection["learning_rate"], "best_epoch": selection["best_epoch"], "test_metrics": metrics, "test_count": len(rows), "artifact_dir": str(artifact), "checkpoint_sha256": evaluation["checkpoint_sha256"]}


def _write_test_outputs(results: Sequence[Mapping[str, Any]]) -> None:
    summary_rows = []
    for result in results:
        metrics = result["test_metrics"]
        summary_rows.append({"task_id": result["task_id"], "learning_rate": result["learning_rate"], "best_epoch": result["best_epoch"], "selection_f1": metrics["selection_f1"], "metric": metrics["selection_metric"], "accuracy": metrics.get("accuracy"), "macro_f1": metrics.get("macro_f1"), "detection_micro_f1": metrics.get("detection_micro_f1"), "pair_micro_f1": metrics.get("pair_micro_f1"), "artifact_dir": result["artifact_dir"], "checkpoint_sha256": result["checkpoint_sha256"]})
    _write_json(ARTIFACT_ROOT / "test_summary.json", {"protocol": "E14-test-gate-v1", "results": summary_rows, "fractions_0_to_1": True, "test_access_log": [{"task_id": row["task_id"], "split": "test", "count": row["test_count"], "artifact_dir": row["artifact_dir"]} for row in results]})
    with (ARTIFACT_ROOT / "test_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary_rows[0]); writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(summary_rows)
    _write_json(ARTIFACT_ROOT / "test_access_log.json", {"protocol": "E14-test-gate-v1", "locked_selection": True, "entries": [{"task_id": row["task_id"], "split": "test", "count": row["test_count"], "artifact_dir": row["artifact_dir"], "accessed_at": _now()} for row in results]})


def _write_report(preflight_value: Mapping[str, Any], manifest: Mapping[str, Any], results: Sequence[Mapping[str, Any]]) -> None:
    e8 = _read_json(ROOT / "artifacts" / "e8" / "summary.json") if (ROOT / "artifacts" / "e8" / "summary.json").is_file() else {}
    lines = ["# E14 — BamiBERT gốc trên pipeline GoiBERT", "", f"Protocol: `{PROTOCOL}`; seed: `{SEED}`; HF commit: `{HF_COMMIT}`.", "", "Đây là local baseline dùng đúng checkpoint BamiBERT gốc và evaluator/split local. Nó không phải exact paper reproduction và không cô lập pure-size effect: BamiBERT khác pretraining, tokenizer, recipe và head 768 so với student head 384.", "", "## Preflight và nguồn", "", f"- Encoder: RobertaModel 12 layers × 768 hidden, 12 heads, FFN 3072; pooling/MLM head bỏ khi downstream.", f"- Head: CLS → dropout 0.1 → dense 768 → tanh → dropout 0.1 → task logits.", f"- Recipe: AdamW, CE plain, weight decay 0.01, warmup 10% linear, max length 256, BF16 SDPA, effective batch 32; LR grid `{LEARNING_RATES}`; 30 epochs; no NC/KD/LLRD.", f"- Weights SHA256: `{preflight_value['model_manifest']['weight_file_sha256']}`.", f"- Tokenizer comparison with GD is recorded in `artifacts/e14/preflight.json`; original BamiBERT tokenizer is used regardless of mismatch.", "", "## Validation grid", "", "All scores are fractions in [0, 1]. LR and epoch were selected using validation only; ties prefer the smaller LR and the earliest best epoch.", "", "| Task | LR | Best epoch | Validation selection F1 | Metric |", "|---|---:|---:|---:|---|"]
    for row in manifest["grid"]:
        lines.append(f"| {row['task_id']} | {row['learning_rate']:.0e} | {row['best_epoch']} | {row['selection_f1']:.6f} | {row['selection_metric']} |" )
    lines += ["", "## Locked test results", "", "| Task | Selected LR | Epoch | Test selection F1 | Metric | Accuracy | Macro F1 | Detection micro F1 | Pair micro F1 |", "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for result in results:
        m = result["test_metrics"]; lines.append(f"| {result['task_id']} | {result['learning_rate']:.0e} | {result['best_epoch']} | {m['selection_f1']:.6f} | {m['selection_metric']} | {m.get('accuracy', '')} | {m.get('macro_f1', '')} | {m.get('detection_micro_f1', '')} | {m.get('pair_micro_f1', '')} |" )
    lines += ["", "## Scope and interpretation", "", "ABSA labels in this E14 readout use the requested name overlay 0=absent, 1=positive, 2=negative, 3=neutral while preserving the frozen integer gold codes. ViSFD OTHERS raw code 4 remains detection-only and pair-excluded.", "", "Paper values are retained only as contextual references in `benchmarks/paper_targets.csv`; this report does not relabel local metrics as paper reproduction. A single seed has no stable confidence interval and is not a SOTA claim.", "", f"E8 summary artifact available for clearly labeled student reference: `{str((ROOT / 'artifacts/e8/summary.json').relative_to(ROOT))}`; E14 does not rewrite or depend on E8 checkpoints. E8 summary keys present: `{sorted(e8) if isinstance(e8, dict) else []}`.", ""]
    (ROOT / "docs" / "E14_BAMIBERT_BASELINE_RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def controller_main() -> int:
    context = run_controller(); manifest = _select_and_lock(context["specs"], context["source_hash"])
    results = [_evaluate_test(selection) for selection in manifest["selections"]]
    _write_test_outputs(results); _write_report(context["preflight"], manifest, results)
    _controller_state({"protocol": PROTOCOL, "revision": REVISION, "status": "completed", "updated_at": _now(), "source_hash": context["source_hash"], "test_tasks": [row["task_id"] for row in results]})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--controller", action="store_true"); mode.add_argument("--worker", action="store_true")
    parser.add_argument("--task-id", choices=TASKS); parser.add_argument("--learning-rate", type=float); parser.add_argument("--device"); parser.add_argument("--micro-batch-size", type=int); parser.add_argument("--expected-source-hash")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if not all(value is not None for value in (args.task_id, args.learning_rate, args.device, args.micro_batch_size, args.expected_source_hash)):
            raise SystemExit("worker requires task, learning-rate, device, micro-batch-size, expected-source-hash")
        return worker_main(args)
    return controller_main()


if __name__ == "__main__":
    raise SystemExit(main())
