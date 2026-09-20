"""E2 validation-only training runner.

E2 is an ablation of *how* the frozen E1 student is trained.  It deliberately
imports the E1 data loader, evaluator, head constructor, optimiser, loss,
loader and validation loop instead of reimplementing any of those contracts.
The only new pieces are teacher construction, immutable train-logit lookup,
KD/consistency loss composition and provenance/status handling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from goibert.train import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    _field,
    _make_loader,
    _move_batch,
    bundle_provenance,
    capture_rng_state,
    compute_loss,
    directory_hashes,
    inspect_train_lengths,
    load_task_bundle,
    make_autocast,
    make_optimizer,
    model_labels,
    resolve_device,
    resolve_precision,
    run_validation,
    save_trial_checkpoint,
    selection_score,
    sha256_file,
    source_hash as e1_source_hash,
    split_field,
    split_ids,
    split_labels,
    split_texts,
    task_aspect_class_counts,
    task_num_aspects,
    task_num_classes,
    task_metadata,
    seed_everything,
    _runtime_metadata,
    _json_default,
)

from .cache import (
    TeacherLogitCache,
    _restore_rng_state,
    build_teacher_logit_cache,
    load_teacher_logit_cache,
    text_hash,
)
from .modeling import E2TeacherClassifier, build_e2_student, build_e2_teacher


TASK_IDS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
E2_MODES = ("teacher", "ce", "kd", "nc", "kd_nc")
TEACHER_LEARNING_RATES = (1e-5, 2e-5, 5e-5)
STUDENT_LEARNING_RATES = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}
KD_ALPHAS = (0.25, 0.5, 0.75)
NC_LAMBDAS = (0.1, 0.5)
TEMPERATURE = 2.0
MAX_LENGTH = 256
EXPECTED_E1_SOURCE_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def e2_source_hash(repo_root: Path) -> str:
    """Hash E2 source/config files; caches and pyc files are excluded."""

    digest = hashlib.sha256()
    roots = (repo_root / "src" / "goibert_e2", repo_root / "configs" / "e2")
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in {".py", ".json"}
            )
    if not any(path.is_file() for path in (repo_root / "src" / "goibert_e2").rglob("*")):
        raise FileNotFoundError(repo_root / "src" / "goibert_e2")
    for path in sorted(files):
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _tree_hash(repo_root: Path, root: Path, suffixes: set[str]) -> str:
    digest = hashlib.sha256()
    if root.is_dir():
        files = [path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes]
    else:
        files = []
    for path in sorted(files):
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def source_hashes(repo_root: Path) -> dict[str, str]:
    dependency = e1_source_hash(repo_root)
    if dependency != EXPECTED_E1_SOURCE_HASH:
        raise RuntimeError(
            "E1 source dependency is not frozen at the E2 gate: "
            f"expected {EXPECTED_E1_SOURCE_HASH}, found {dependency}"
        )
    package_hash = _tree_hash(repo_root, repo_root / "src" / "goibert_e2", {".py", ".json"})
    config_hash = _tree_hash(repo_root, repo_root / "configs" / "e2", {".py", ".json"})
    # ``e2`` remains the runner-code hash used by the durable scheduler;
    # ``e2_source`` binds code and config together for artifact provenance.
    combined = hashlib.sha256((package_hash + config_hash).encode("ascii")).hexdigest()
    orchestrator = repo_root / "scripts" / "execute_e2_validation.py"
    return {
        "e1_dependency": dependency,
        "e2": package_hash,
        "e2_source": combined,
        "e2_config": config_hash,
        "orchestrator": sha256_file(orchestrator) if orchestrator.is_file() else "",
    }


def _validated_source_hashes(
    repo_root: Path,
    *,
    expected_e2_code_hash: str | None = None,
    expected_e2_config_hash: str | None = None,
    expected_orchestrator_hash: str | None = None,
) -> dict[str, str]:
    """Resolve and fail closed on every launch hash before doing work."""

    values = source_hashes(repo_root)
    for name, expected, actual in (
        ("e2 code", expected_e2_code_hash, values["e2"]),
        ("e2 config", expected_e2_config_hash, values["e2_config"]),
        ("orchestrator", expected_orchestrator_hash, values["orchestrator"]),
    ):
        if expected is not None and expected != actual:
            raise RuntimeError(
                f"{name} hash changed before E2 launch: expected {expected}, found {actual}"
            )
    return values


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}__rerun{index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Unable to allocate a fresh artifact directory for {path}")


def _set_status(path: Path, status: str, **details: Any) -> None:
    payload = {"status": status, "updated_at_unix": time.time(), **details}
    _write_json(path / "status.json", payload)


def _task_contract(task_id: str, repo_root: Path, *, smoke: bool = False) -> dict[str, Any]:
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown task_id {task_id!r}; expected one of {TASK_IDS}")
    bundle = load_task_bundle(task_id, repo_root, include_test=False, smoke=smoke)
    train_split = split_field(bundle, "train")
    valid_split = split_field(bundle, "validation")
    if train_split is None or valid_split is None:
        raise ValueError(f"Task {task_id} must expose train and validation splits")
    return {
        "bundle": bundle,
        "train": train_split,
        "validation": valid_split,
        "num_aspects": task_num_aspects(bundle),
        "aspect_class_counts": task_aspect_class_counts(bundle),
        "num_classes": task_num_classes(bundle),
    }


def _tokenizer_and_lengths(repo_root: Path, train_split: Any):
    tokenizer = AutoTokenizer.from_pretrained(
        repo_root / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    lengths = inspect_train_lengths(tokenizer, list(split_texts(train_split)))
    if lengths.warning_over_512:
        raise RuntimeError("E1 length audit blocks this task above max_length=512")
    if lengths.max_length != MAX_LENGTH:
        raise RuntimeError(
            f"E2 requires the E1 max_length={MAX_LENGTH}; audit selected {lengths.max_length}"
        )
    return tokenizer, lengths


def _tokenizer_hashes(repo_root: Path) -> dict[str, str]:
    return _tokenizer_hashes_for_dir(repo_root / "models" / "goiturn-s-gd")


def _tokenizer_hashes_for_dir(model_dir: Path) -> dict[str, str]:
    hashes = directory_hashes(model_dir)
    return {
        key: value
        for key, value in hashes.items()
        if key in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    }


def _assert_teacher_student_tokenizer_identity(repo_root: Path) -> None:
    student = _tokenizer_hashes_for_dir(repo_root / "models" / "goiturn-s-gd")
    teacher = _tokenizer_hashes_for_dir(repo_root / "models" / "goibert-b-dapt")
    if student != teacher:
        raise ValueError(
            "Teacher and student tokenizer files differ; E2 cache/token IDs cannot be shared"
        )


def _loader_set(
    tokenizer: Any,
    train_split: Any,
    valid_split: Any,
    bundle: Any,
    *,
    seed: int,
    micro_batch_size: int,
    max_length: int,
):
    train_loader = _make_loader(
        tokenizer,
        train_split,
        max_length=max_length,
        batch_size=micro_batch_size,
        shuffle=True,
        seed=seed,
        labels=model_labels(bundle, "train", train_split),
        gold_labels=split_labels(train_split),
    )
    ordered_train_loader = _make_loader(
        tokenizer,
        train_split,
        max_length=max_length,
        batch_size=micro_batch_size,
        shuffle=False,
        seed=seed,
        labels=model_labels(bundle, "train", train_split),
        gold_labels=split_labels(train_split),
    )
    valid_loader = _make_loader(
        tokenizer,
        valid_split,
        max_length=max_length,
        batch_size=micro_batch_size,
        shuffle=False,
        seed=seed,
        labels=model_labels(bundle, "validation", valid_split),
        gold_labels=split_labels(valid_split),
    )
    return train_loader, ordered_train_loader, valid_loader


def _load_loss_api() -> tuple[Any, Any]:
    """Load the exact sibling-owned E2 loss functions lazily."""

    from .losses import logits_kd_loss, symmetric_kl_consistency_loss

    return logits_kd_loss, symmetric_kl_consistency_loss


def _make_noisy_batch(
    tokenizer: Any,
    ids: Sequence[Any],
    text_by_id: Mapping[str, str],
    *,
    max_length: int,
    seed: int,
    epoch: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    from .noise import noisy_batch

    clean_texts = []
    for example_id in ids:
        try:
            text = text_by_id[str(example_id)]
        except KeyError as error:
            raise ValueError(f"Cannot build noisy view: missing train ID {example_id!r}") from error
        clean_texts.append(text)
    noisy_result = noisy_batch(clean_texts, ids, seed=seed, epoch=epoch)
    encoded = tokenizer(
        list(noisy_result.texts),
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )
    batch = tokenizer.pad(
        [{"input_ids": ids_value} for ids_value in encoded["input_ids"]],
        padding=True,
        return_tensors="pt",
    )
    return (
        {key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)},
        noisy_result.to_dict(),
    )


def _validate_cache_for_task(
    cache: TeacherLogitCache,
    *,
    task_id: str,
    train_split: Any,
    tokenizer_hashes: Mapping[str, str],
    max_length: int,
    counts: Sequence[int],
    num_aspects: int,
    num_labels: int,
    bundle: Any,
    repo_root: Path,
    precision: str,
    e1_dependency_source_hash: str,
    source_hashes_value: Mapping[str, str],
) -> None:
    metadata = cache.metadata
    expected_ids = list(split_ids(train_split))
    if metadata.get("task_id") != task_id or metadata.get("split") != "train":
        raise ValueError("Teacher-logit cache task/split does not match the student task")
    if list(metadata.get("source_ids", ())) != expected_ids:
        raise ValueError("Teacher-logit cache source IDs do not match the current train split")
    if metadata.get("source_text_hash") != text_hash(list(split_texts(train_split))):
        raise ValueError("Teacher-logit cache train text hash does not match the current split")
    if metadata.get("tokenizer_hashes") != dict(tokenizer_hashes):
        raise ValueError("Teacher-logit cache tokenizer hashes do not match E1 tokenizer")
    if int(metadata.get("max_length", -1)) != int(max_length):
        raise ValueError("Teacher-logit cache max_length does not match E1")
    if list(metadata.get("aspect_class_counts", ())) != [int(value) for value in counts]:
        raise ValueError("Teacher-logit cache slot schema does not match task")
    if int(metadata.get("num_aspects", -1)) != int(num_aspects):
        raise ValueError("Teacher-logit cache aspect count does not match task")
    if int(metadata.get("num_labels", -1)) != int(num_labels):
        raise ValueError("Teacher-logit cache label width does not match task")
    current_provenance = bundle_provenance(bundle, repo_root)
    if metadata.get("data_provenance") != current_provenance:
        raise ValueError("Teacher-logit cache data/evaluator provenance does not match current task")
    if metadata.get("inference_precision") != precision:
        raise ValueError("Teacher-logit cache inference precision does not match student run")
    if metadata.get("e1_dependency_source_hash") != e1_dependency_source_hash:
        raise ValueError("Teacher-logit cache E1 dependency hash does not match the frozen runner")
    expected_cache_hashes = dict(source_hashes_value)
    if metadata.get("source_hashes") != expected_cache_hashes:
        raise ValueError("Teacher-logit cache source/code hashes do not match current E2 inputs")


def _load_checkpoint_state(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise ValueError(f"E2 checkpoint {path} has no model state")
    return state


def _save_model_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    epoch: int,
    global_step: int,
    metadata: Mapping[str, Any],
) -> None:
    # The frozen E1 writer already defines the checkpoint state contract.
    save_trial_checkpoint(
        path,
        model,  # type: ignore[arg-type]
        optimizer,
        scheduler,
        epoch=epoch,
        global_step=global_step,
        metadata=metadata,
    )


def _base_metadata(
    *,
    repo_root: Path,
    task_id: str,
    mode: str,
    seed: int,
    learning_rate: float,
    epochs: int,
    effective_batch_size: int,
    micro_batch_size: int,
    precision: str,
    device: torch.device,
    attention_implementation: str,
    lengths: Any,
    bundle: Any,
    model_spec: Any,
    model_hashes: Mapping[str, str],
    tokenizer_hashes: Mapping[str, str],
    hyperparameters: Mapping[str, Any],
    source_hashes_value: Mapping[str, str],
    stage: str | None = None,
    method: str | None = None,
    coefficient: float | None = None,
    revision: str = "e2-20260908-r1",
    e2_config_hash_value: str | None = None,
    orchestrator_hash_value: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    default_stage = {
        "teacher": "stage0_teacher",
        "ce": "stagea_ce_replay",
        "kd": "stagea_kd",
        "nc": "stagea_nc",
        "kd_nc": "stageb",
    }[mode]
    resolved_stage = stage or default_stage
    resolved_method = method or ("ce_replay" if mode == "ce" else mode)
    return {
        "protocol": "E2-training-ablation-v1",
        "revision": revision,
        "stage": resolved_stage,
        "method": resolved_method,
        "coefficient": coefficient if coefficient is not None else None,
        "task_id": task_id,
        "mode": mode,
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "epochs": int(epochs),
        "effective_batch_size": int(effective_batch_size),
        "micro_batch_size": int(micro_batch_size),
        "gradient_accumulation_steps": int(effective_batch_size // micro_batch_size),
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "gradient_clip_norm": 1.0,
        "max_length": int(lengths.max_length),
        "length_stats": lengths.to_dict(),
        "include_test": False,
        "smoke": bool(smoke),
        "source_hashes": dict(source_hashes_value),
        "e1_dependency_source_hash": source_hashes_value["e1_dependency"],
        "e2_source_hash": source_hashes_value.get("e2_source", source_hashes_value["e2"]),
        "e2_code_config_hash": source_hashes_value.get("e2_source"),
        "e2_code_hash": source_hashes_value["e2"],
        "e2_config_hash": e2_config_hash_value,
        "orchestrator_hash": orchestrator_hash_value,
        "data_provenance": bundle_provenance(bundle, repo_root),
        "task_metadata": task_metadata(bundle),
        "model": model_spec.to_dict(),
        "runtime": _runtime_metadata(device, precision),
        "attention_implementation": attention_implementation,
        "backbone_hashes": dict(model_hashes),
        "tokenizer_hashes": dict(tokenizer_hashes),
        "hyperparameters": dict(hyperparameters),
        "resumable": False,
        "resume_support": "fresh_rerun_only; no resume CLI",
        "argv": list(sys.argv),
    }


def _train_loop(
    *,
    model: nn.Module,
    train_loader: Any,
    valid_loader: Any,
    bundle: Any,
    task_id: str,
    device: torch.device,
    precision: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epochs: int,
    effective_batch_size: int,
    micro_batch_size: int,
    mode: str,
    alpha: float | None,
    consistency_lambda: float | None,
    cache: TeacherLogitCache | None,
    tokenizer: Any,
    train_split: Any,
    max_length: int,
    seed: int,
    artifact_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    counts = task_aspect_class_counts(bundle)
    num_aspects = task_num_aspects(bundle)
    accumulation_steps = effective_batch_size // micro_batch_size
    text_by_id = {str(example_id): text for example_id, text in zip(split_ids(train_split), split_texts(train_split))}
    kd_fn = nc_fn = None
    if mode in {"kd", "nc", "kd_nc"}:
        kd_fn, nc_fn = _load_loss_api()
    cache_position = None
    if cache is not None:
        cache_position = cache
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
        loss_totals: dict[str, float] = {"total": 0.0, "ce": 0.0}
        noise_stats: dict[str, Any] = {"count": 0, "changed_count": 0, "operation_counts": {}}
        if mode in {"kd", "kd_nc"}:
            loss_totals["kd"] = 0.0
        if mode in {"nc", "kd_nc"}:
            loss_totals["consistency"] = 0.0
        epoch_started = time.time()
        for batch_index, batch in enumerate(train_loader):
            moved = _move_batch(batch, device)
            with make_autocast(precision, device):
                logits = model(
                    moved["input_ids"],
                    moved["attention_mask"],
                    token_type_ids=moved.get("token_type_ids"),
                )
                ce = compute_loss(
                    logits,
                    moved["labels"],
                    num_aspects=num_aspects,
                    aspect_class_counts=counts,
                )
                total = ce
                kd_value = None
                consistency_value = None
                if mode in {"kd", "kd_nc"}:
                    assert cache_position is not None and kd_fn is not None
                    teacher_logits = cache_position.lookup(batch["ids"], device=device)
                    kd_value = kd_fn(
                        logits,
                        teacher_logits,
                        temperature=TEMPERATURE,
                        slot_class_counts=counts or None,
                    )
                    # logits_kd_loss already includes T**2.  Multiplying it
                    # again would silently turn the declared T**2 objective
                    # into T**4.
                    total = (1.0 - float(alpha)) * ce + float(alpha) * kd_value
                if mode in {"nc", "kd_nc"}:
                    assert nc_fn is not None
                    noisy, batch_noise_stats = _make_noisy_batch(
                        tokenizer,
                        batch["ids"],
                        text_by_id,
                        max_length=max_length,
                        seed=seed,
                        epoch=epoch,
                        device=device,
                    )
                    noisy_logits = model(
                        noisy["input_ids"],
                        noisy["attention_mask"],
                        token_type_ids=noisy.get("token_type_ids"),
                    )
                    consistency_value = nc_fn(
                        logits,
                        noisy_logits,
                        temperature=1.0,
                        slot_class_counts=counts or None,
                    )
                    total = total + float(consistency_lambda) * consistency_value
                    noise_stats["count"] += int(batch_noise_stats["count"])
                    noise_stats["changed_count"] += int(batch_noise_stats["changed_count"])
                    for operation, count in batch_noise_stats["operation_counts"].items():
                        noise_stats["operation_counts"][operation] = (
                            noise_stats["operation_counts"].get(operation, 0) + int(count)
                        )
                window_start = (batch_index // accumulation_steps) * accumulation_steps
                window_end = min(window_start + accumulation_steps, len(train_loader))
                window_examples = sum(
                    min(micro_batch_size, len(train_loader.dataset) - index * micro_batch_size)
                    for index in range(window_start, window_end)
                )
                scaled = total * (len(batch["labels"]) / max(window_examples, 1))
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite E2 loss at epoch={epoch}, batch={batch_index}")
            scaled.backward()
            count = len(batch["labels"])
            total_examples += count
            loss_totals["total"] += float(total.detach().cpu()) * count
            loss_totals["ce"] += float(ce.detach().cpu()) * count
            if kd_value is not None:
                loss_totals["kd"] += float(kd_value.detach().cpu()) * count
            if consistency_value is not None:
                loss_totals["consistency"] += float(consistency_value.detach().cpu()) * count
            is_boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if is_boundary:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        val_metrics, val_predictions = run_validation(
            model, valid_loader, device, precision, task_id, bundle
        )
        score = selection_score(val_metrics)
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite validation selection_f1 at epoch {epoch}")
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_losses": {key: value / max(total_examples, 1) for key, value in loss_totals.items()},
            "validation": val_metrics,
            "selection_f1": score,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
        }
        if mode in {"nc", "kd_nc"}:
            noise_stats["change_rate"] = noise_stats["changed_count"] / max(noise_stats["count"], 1)
            row["noise"] = noise_stats
            metadata.setdefault("noise_statistics", []).append(noise_stats)
        curves.append(row)
        _write_json(artifact_dir / "epoch_curves.json", {"epochs": curves})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_predictions = val_predictions
            _save_model_checkpoint(
                artifact_dir / "best_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                global_step=global_step,
                metadata=metadata,
            )
            _write_json(artifact_dir / "best_validation_metrics.json", best_metrics)
            _write_json(artifact_dir / "validation_predictions.json", {"rows": best_predictions})
    metadata.update(
        {
            "best_epoch": best_epoch,
            "best_selection_f1": best_score,
            "best_validation_metrics": best_metrics,
            "elapsed_seconds": time.time() - metadata["started_at_unix"],
            "peak_gpu_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "peak_gpu_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            ),
            "status": "completed",
        }
    )
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint_hash = sha256_file(checkpoint_path)
        metadata["checkpoint_sha256"] = checkpoint_hash
        if mode == "teacher":
            metadata["teacher_checkpoint_sha256"] = checkpoint_hash
    _write_json(artifact_dir / "metadata.json", metadata)
    _set_status(artifact_dir, "completed", best_epoch=best_epoch, best_selection_f1=best_score)
    return {
        "task_id": task_id,
        "mode": mode,
        "seed": metadata["seed"],
        "learning_rate": metadata["learning_rate"],
        "artifact_dir": str(artifact_dir),
        "best_epoch": best_epoch,
        "best_selection_f1": best_score,
        "validation_metrics": best_metrics,
        "status": "completed",
    }


def train_teacher_trial(
    *,
    task_id: str,
    learning_rate: float,
    repo_root: Path,
    output_dir: Path,
    seed: int = 42,
    device_name: str | None = None,
    precision: str = "bf16",
    attention_implementation: str = "sdpa",
    epochs: int = EPOCHS,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    micro_batch_size: int | None = None,
    smoke: bool = False,
    stage: str | None = None,
    method: str | None = None,
    coefficient: float | None = None,
    revision: str = "e2-20260908-r1",
    expected_e2_code_hash: str | None = None,
    expected_e2_config_hash: str | None = None,
    expected_orchestrator_hash: str | None = None,
) -> dict[str, Any]:
    """Train one validation-only task teacher trial."""

    if learning_rate not in TEACHER_LEARNING_RATES and not smoke:
        raise ValueError(f"Teacher LR must be one of {TEACHER_LEARNING_RATES}")
    return _train_any(
        task_id=task_id,
        mode="teacher",
        learning_rate=learning_rate,
        seed=seed,
        repo_root=repo_root,
        output_dir=output_dir,
        device_name=device_name,
        precision=precision,
        attention_implementation=attention_implementation,
        epochs=epochs,
        effective_batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
        smoke=smoke,
        alpha=None,
        consistency_lambda=None,
        cache_path=None,
        stage=stage,
        method=method,
        coefficient=coefficient,
        revision=revision,
        expected_e2_code_hash=expected_e2_code_hash,
        expected_e2_config_hash=expected_e2_config_hash,
        expected_orchestrator_hash=expected_orchestrator_hash,
    )


def train_student_trial(
    *,
    task_id: str,
    mode: str,
    repo_root: Path,
    output_dir: Path,
    seed: int,
    alpha: float | None = None,
    consistency_lambda: float | None = None,
    cache_path: Path | None = None,
    learning_rate: float | None = None,
    device_name: str | None = None,
    precision: str = "bf16",
    attention_implementation: str = "sdpa",
    epochs: int = EPOCHS,
    effective_batch_size: int = EFFECTIVE_BATCH_SIZE,
    micro_batch_size: int | None = None,
    smoke: bool = False,
    stage: str | None = None,
    method: str | None = None,
    coefficient: float | None = None,
    revision: str = "e2-20260908-r1",
    expected_e2_code_hash: str | None = None,
    expected_e2_config_hash: str | None = None,
    expected_orchestrator_hash: str | None = None,
) -> dict[str, Any]:
    """Train a fresh student for CE, KD, NC or KD+NC on clean validation."""

    if mode not in {"ce", "kd", "nc", "kd_nc"}:
        raise ValueError("train_student_trial mode must be ce, kd, nc or kd_nc")
    if task_id not in STUDENT_LEARNING_RATES:
        raise ValueError(f"Unknown task_id {task_id!r}")
    expected_lr = STUDENT_LEARNING_RATES[task_id]
    if learning_rate is None:
        learning_rate = expected_lr
    if float(learning_rate) != float(expected_lr):
        raise ValueError(f"E2 student LR is locked to E1 winner {expected_lr:g} for {task_id}")
    if mode in {"kd", "kd_nc"} and alpha not in KD_ALPHAS and not smoke:
        raise ValueError(f"KD alpha must be one of {KD_ALPHAS}")
    if mode in {"nc", "kd_nc"} and consistency_lambda not in NC_LAMBDAS and not smoke:
        raise ValueError(f"NC lambda must be one of {NC_LAMBDAS}")
    if mode in {"kd", "kd_nc"} and cache_path is None:
        raise ValueError("KD modes require an immutable train teacher-logit cache")
    return _train_any(
        task_id=task_id,
        mode=mode,
        learning_rate=float(learning_rate),
        seed=seed,
        repo_root=repo_root,
        output_dir=output_dir,
        device_name=device_name,
        precision=precision,
        attention_implementation=attention_implementation,
        epochs=epochs,
        effective_batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
        smoke=smoke,
        alpha=alpha,
        consistency_lambda=consistency_lambda,
        cache_path=cache_path,
        stage=stage,
        method=method,
        coefficient=coefficient,
        revision=revision,
        expected_e2_code_hash=expected_e2_code_hash,
        expected_e2_config_hash=expected_e2_config_hash,
        expected_orchestrator_hash=expected_orchestrator_hash,
    )


def _train_any(
    *,
    task_id: str,
    mode: str,
    learning_rate: float,
    seed: int,
    repo_root: Path,
    output_dir: Path,
    device_name: str | None,
    precision: str,
    attention_implementation: str,
    epochs: int,
    effective_batch_size: int,
    micro_batch_size: int | None,
    smoke: bool,
    alpha: float | None,
    consistency_lambda: float | None,
    cache_path: Path | None,
    stage: str | None = None,
    method: str | None = None,
    coefficient: float | None = None,
    revision: str = "e2-20260908-r1",
    expected_e2_code_hash: str | None = None,
    expected_e2_config_hash: str | None = None,
    expected_orchestrator_hash: str | None = None,
) -> dict[str, Any]:
    if mode not in E2_MODES:
        raise ValueError(f"Unknown E2 mode {mode!r}")
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    micro = int(micro_batch_size or effective_batch_size)
    if micro <= 0 or effective_batch_size % micro:
        raise ValueError("micro_batch_size must divide effective_batch_size")
    source_hash_values = _validated_source_hashes(
        repo_root,
        expected_e2_code_hash=expected_e2_code_hash,
        expected_e2_config_hash=expected_e2_config_hash,
        expected_orchestrator_hash=expected_orchestrator_hash,
    )
    seed_everything(seed)
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    contract = _task_contract(task_id, repo_root, smoke=smoke)
    bundle = contract["bundle"]
    train_split = contract["train"]
    valid_split = contract["validation"]
    tokenizer, lengths = _tokenizer_and_lengths(repo_root, train_split)
    _assert_teacher_student_tokenizer_identity(repo_root)
    tokenizer_hashes = _tokenizer_hashes(repo_root)
    student_dir = repo_root / "models" / "goiturn-s-gd"
    teacher_dir = repo_root / "models" / "goibert-b-dapt"
    if mode == "teacher":
        model, model_spec = build_e2_teacher(
            teacher_dir,
            num_classes=contract["num_classes"],
            num_aspects=contract["num_aspects"],
            aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
            attention_implementation=attention_implementation,
        )
        model_hashes = directory_hashes(teacher_dir)
    else:
        # This is the only student construction path: a fresh E1 backbone and
        # fresh E1 head per run, never an E1 fine-tuned checkpoint.
        model, model_spec = build_e2_student(
            student_dir,
            num_classes=contract["num_classes"],
            num_aspects=contract["num_aspects"],
            aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
            attention_implementation=attention_implementation,
        )
        model_hashes = directory_hashes(student_dir)
    model.to(device)
    train_loader, ordered_loader, valid_loader = _loader_set(
        tokenizer,
        train_split,
        valid_split,
        bundle,
        seed=seed,
        micro_batch_size=micro,
        max_length=MAX_LENGTH,
    )
    cache = None
    if mode in {"kd", "kd_nc"}:
        cache = load_teacher_logit_cache(Path(cache_path))  # type: ignore[arg-type]
        _validate_cache_for_task(
            cache,
            task_id=task_id,
            train_split=train_split,
            tokenizer_hashes=tokenizer_hashes,
            max_length=MAX_LENGTH,
            counts=contract["aspect_class_counts"],
            num_aspects=contract["num_aspects"],
            num_labels=contract["num_classes"],
            bundle=bundle,
            repo_root=repo_root,
            precision=precision,
            e1_dependency_source_hash=source_hash_values["e1_dependency"],
            source_hashes_value=source_hash_values,
        )
    trial_dir = _unique_dir(Path(output_dir))
    trial_dir.mkdir(parents=True, exist_ok=False)
    _set_status(trial_dir, "running", mode=mode, task_id=task_id, seed=seed)
    metadata = _base_metadata(
        repo_root=repo_root,
        task_id=task_id,
        mode=mode,
        seed=seed,
        learning_rate=learning_rate,
        epochs=1 if smoke else epochs,
        effective_batch_size=effective_batch_size,
        micro_batch_size=micro,
        precision=precision,
        device=device,
        attention_implementation=attention_implementation,
        lengths=lengths,
        bundle=bundle,
        model_spec=model_spec,
        model_hashes=model_hashes,
        tokenizer_hashes=tokenizer_hashes,
        hyperparameters={
            "temperature": TEMPERATURE,
            "alpha": alpha,
            "consistency_lambda": consistency_lambda,
            "teacher_cache": str(cache_path) if cache_path is not None else None,
            "teacher_cache_hash": sha256_file(Path(cache_path)) if cache_path is not None else None,
        },
        source_hashes_value=source_hash_values,
        stage=stage,
        method=method,
        coefficient=(
            coefficient
            if coefficient is not None
            else learning_rate
            if mode == "teacher"
            else alpha
            if mode in {"kd", "kd_nc"}
            else consistency_lambda
            if mode == "nc"
            else None
        ),
        revision=revision,
        e2_config_hash_value=source_hash_values["e2_config"],
        orchestrator_hash_value=source_hash_values["orchestrator"],
        smoke=smoke,
    )
    metadata["started_at_unix"] = time.time()
    _write_json(trial_dir / "resolved_config.json", metadata)
    try:
        optimizer = make_optimizer(model, learning_rate)
        accumulation_steps = effective_batch_size // micro
        total_steps = math.ceil(len(train_loader) / accumulation_steps) * (1 if smoke else epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, round(total_steps * WARMUP_FRACTION), total_steps
        )
        result = _train_loop(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            bundle=bundle,
            task_id=task_id,
            device=device,
            precision=precision,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=1 if smoke else epochs,
            effective_batch_size=effective_batch_size,
            micro_batch_size=micro,
            mode=mode,
            alpha=alpha,
            consistency_lambda=consistency_lambda,
            cache=cache,
            tokenizer=tokenizer,
            train_split=train_split,
            max_length=MAX_LENGTH,
            seed=seed,
            artifact_dir=trial_dir,
            metadata=metadata,
        )
        return result
    except Exception as error:
        metadata.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        _write_json(trial_dir / "metadata.json", metadata)
        _set_status(trial_dir, "failed", error=repr(error))
        raise


def _cache_teacher_checkpoint_impl(
    *,
    task_id: str,
    teacher_checkpoint: Path,
    repo_root: Path,
    output_path: Path,
    device_name: str | None = None,
    precision: str = "bf16",
    attention_implementation: str = "sdpa",
    micro_batch_size: int = EFFECTIVE_BATCH_SIZE,
    smoke: bool = False,
) -> TeacherLogitCache:
    """Create the train-only cache from a validation-selected teacher."""

    source_hash_values = source_hashes(repo_root)
    contract = _task_contract(task_id, repo_root, smoke=smoke)
    bundle = contract["bundle"]
    train_split = contract["train"]
    tokenizer, lengths = _tokenizer_and_lengths(repo_root, train_split)
    _assert_teacher_student_tokenizer_identity(repo_root)
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    teacher, _ = build_e2_teacher(
        repo_root / "models" / "goibert-b-dapt",
        num_classes=contract["num_classes"],
        num_aspects=contract["num_aspects"],
        aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
        attention_implementation=attention_implementation,
    )
    state = _load_checkpoint_state(Path(teacher_checkpoint))
    teacher.load_state_dict(state["model"])
    teacher.to(device)
    _, ordered_loader, _ = _loader_set(
        tokenizer,
        train_split,
        contract["validation"],
        bundle,
        seed=42,
        micro_batch_size=micro_batch_size,
        max_length=MAX_LENGTH,
    )
    return build_teacher_logit_cache(
        Path(output_path),
        teacher=teacher,
        loader=ordered_loader,
        device=device,
        precision=precision,
        task_id=task_id,
        source_ids=split_ids(train_split),
        source_texts=split_texts(train_split),
        tokenizer_hashes=_tokenizer_hashes(repo_root),
        max_length=lengths.max_length,
        teacher_hash=sha256_file(Path(teacher_checkpoint)),
        aspect_class_counts=contract["aspect_class_counts"],
        num_aspects=contract["num_aspects"],
        num_labels=contract["num_classes"],
        e1_dependency_source_hash=source_hash_values["e1_dependency"],
        data_provenance=bundle_provenance(bundle, repo_root),
        inference_precision=precision,
        source_hashes=source_hash_values,
    )


def cache_teacher_checkpoint(
    *,
    task_id: str,
    teacher_checkpoint: Path,
    repo_root: Path,
    output_path: Path,
    device_name: str | None = None,
    precision: str = "bf16",
    attention_implementation: str = "sdpa",
    micro_batch_size: int = EFFECTIVE_BATCH_SIZE,
    smoke: bool = False,
) -> TeacherLogitCache:
    """Create a cache without consuming caller RNG state.

    The teacher head is constructed before inference, so preserving only the
    forward pass would still perturb the next student's initialisation.  The
    outer guard covers model construction, tokenisation and cache writing.
    """

    before = capture_rng_state()
    try:
        return _cache_teacher_checkpoint_impl(
            task_id=task_id,
            teacher_checkpoint=teacher_checkpoint,
            repo_root=repo_root,
            output_path=output_path,
            device_name=device_name,
            precision=precision,
            attention_implementation=attention_implementation,
            micro_batch_size=micro_batch_size,
            smoke=smoke,
        )
    finally:
        _restore_rng_state(before)


def train_trial(**kwargs: Any) -> dict[str, Any]:
    """Dispatch one operator request to teacher or student training."""

    mode = str(kwargs.pop("mode"))
    if mode == "teacher":
        # The CLI shares the alpha/cache flags across modes; teacher training
        # must not accidentally record or consume student-only parameters.
        for key in ("alpha", "consistency_lambda", "cache_path"):
            kwargs.pop(key, None)
        return train_teacher_trial(**kwargs)
    return train_student_trial(mode=mode, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one validation-only GoiBERT E2 trial")
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--mode", choices=(*E2_MODES, "cache"), required=True)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--consistency-lambda", type=float, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--effective-batch-size", type=int, default=EFFECTIVE_BATCH_SIZE)
    parser.add_argument("--micro-batch-size", type=int, default=EFFECTIVE_BATCH_SIZE)
    # Durable-orchestrator identity fields.  They are optional for the direct
    # canonical CLI but are recorded and checked when supplied.
    parser.add_argument("--stage", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--coefficient", type=float, default=None)
    parser.add_argument("--protocol", default="E2-training-ablation-v1")
    parser.add_argument("--revision", default="e2-20260908-r1")
    parser.add_argument("--expected-e2-code-hash", default=None)
    parser.add_argument("--expected-e2-config-hash", default=None)
    parser.add_argument("--expected-orchestrator-hash", default=None)
    parser.add_argument("--include-test", default="false")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.protocol != "E2-training-ablation-v1":
        raise SystemExit(f"Unsupported E2 protocol: {args.protocol!r}")
    if str(args.include_test).lower() not in {"false", "0", "no"}:
        raise SystemExit("E2 is validation-only; --include-test must be false")
    if args.mode == "cache":
        if args.teacher_checkpoint is None or args.cache_path is None:
            raise SystemExit("cache mode requires --teacher-checkpoint and --cache-path")
        # Validate the complete launch identity before constructing the
        # teacher or touching the cache path.  A stale operator snapshot must
        # never produce a cache that looks current.
        _validated_source_hashes(
            args.repo_root.resolve(),
            expected_e2_code_hash=args.expected_e2_code_hash,
            expected_e2_config_hash=args.expected_e2_config_hash,
            expected_orchestrator_hash=args.expected_orchestrator_hash,
        )
        cache = cache_teacher_checkpoint(
            task_id=args.task_id,
            teacher_checkpoint=args.teacher_checkpoint,
            repo_root=args.repo_root.resolve(),
            output_path=args.cache_path.resolve(),
            device_name=args.device,
            precision=args.precision,
            attention_implementation=args.attention_implementation,
            micro_batch_size=args.micro_batch_size,
            smoke=args.smoke,
        )
        print(json.dumps({"status": "completed", "cache_path": str(cache.path), "metadata": cache.metadata}, default=_json_default, indent=2), flush=True)
        return 0
    mode = args.mode
    coefficient = args.coefficient
    alpha = args.alpha if args.alpha is not None else coefficient if mode in {"kd", "kd_nc"} else None
    consistency_lambda = args.consistency_lambda if args.consistency_lambda is not None else coefficient if mode in {"nc", "kd_nc"} else None
    result = train_trial(
        task_id=args.task_id,
        mode=args.mode,
        learning_rate=args.learning_rate,
        seed=args.seed,
        alpha=alpha,
        consistency_lambda=consistency_lambda,
        cache_path=args.cache_path,
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        device_name=args.device,
        precision=args.precision,
        attention_implementation=args.attention_implementation,
        epochs=args.epochs,
        effective_batch_size=args.effective_batch_size,
        micro_batch_size=args.micro_batch_size,
        stage=args.stage,
        method=args.method,
        coefficient=coefficient,
        revision=args.revision,
        expected_e2_code_hash=args.expected_e2_code_hash,
        expected_e2_config_hash=args.expected_e2_config_hash,
        expected_orchestrator_hash=args.expected_orchestrator_hash,
        smoke=args.smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
