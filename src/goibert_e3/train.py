"""Validation-only E3 runner.

This module reuses the frozen E1 data, model, optimiser, scheduler, and
evaluator helpers.  Its only training change is the conditional polarity loss
from :mod:`goibert_e3.losses`.  The runner never loads the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from goibert.modeling import build_classifier
from goibert.train import (
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    _make_loader,
    _move_batch,
    _runtime_metadata,
    _json_default,
    bundle_provenance,
    capture_rng_state,
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
    seed_everything,
    sha256_file,
    split_field,
    split_labels,
    split_texts,
    task_aspect_class_counts,
    task_metadata,
    task_num_aspects,
    task_num_classes,
)

from .losses import POLARITY_WEIGHT, combined_polarity_loss


TASK_IDS = ("visfd", "absa_restaurant")
LEARNING_RATE = 1e-4
SCREENING_SEED = 42
MAX_LENGTH = 256
MICRO_BATCH_SIZE = 32
PROTOCOL = "E3-conditional-polarity-v1"
REVISION = "e3-20260909-r1"
EXPECTED_E1_SOURCE_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _set_status(path: Path, status: str, **details: Any) -> None:
    _write_json(
        path / "status.json",
        {"status": status, "updated_at": _now(), "updated_at_unix": time.time(), **details},
    )


def _set_heartbeat(path: Path, **details: Any) -> None:
    _write_json(
        path / "heartbeat.json",
        {"updated_at": _now(), "updated_at_unix": time.time(), **details},
    )


def _tree_hash(repo_root: Path, root: Path, suffixes: set[str]) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes]
    for path in sorted(files):
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def source_hashes(repo_root: Path) -> dict[str, str]:
    """Return the complete source/config dependency snapshot for E3."""

    from goibert.train import source_hash as e1_source_hash

    e1_hash = e1_source_hash(repo_root)
    if e1_hash != EXPECTED_E1_SOURCE_HASH:
        raise RuntimeError(
            "E1 dependency is not frozen at the E3 gate: "
            f"expected {EXPECTED_E1_SOURCE_HASH}, found {e1_hash}"
        )
    e3_code = _tree_hash(repo_root, repo_root / "src" / "goibert_e3", {".py"})
    e3_config = _tree_hash(repo_root, repo_root / "configs" / "e3", {".json"})
    orchestrator_path = repo_root / "scripts" / "execute_e3_validation.py"
    orchestrator = sha256_file(orchestrator_path) if orchestrator_path.is_file() else ""
    combined = hashlib.sha256((e3_code + e3_config + orchestrator).encode("ascii")).hexdigest()
    return {
        "e1_dependency": e1_hash,
        "e3_code": e3_code,
        "e3_config": e3_config,
        "e3_orchestrator": orchestrator,
        "e3_source": combined,
    }


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}__rerun{index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Unable to allocate a fresh artifact directory for {path}")


def _task_contract(task_id: str, repo_root: Path) -> dict[str, Any]:
    if task_id not in TASK_IDS:
        raise ValueError(f"E3 supports only {TASK_IDS}, got {task_id!r}")
    bundle = load_task_bundle(task_id, repo_root, include_test=False, smoke=False)
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


def _resolved_polarity_slots(counts: Sequence[int]) -> tuple[int, ...]:
    """Resolve the source-specific auxiliary slots and make them auditable."""

    values = tuple(int(value) for value in counts)
    if values == (4,) * 10 + (2, 4):
        # Defensive support for no known alternate order is intentionally not
        # inferred here.  The audited ViSFD order is 5=OTHERS.
        raise ValueError("Unexpected ViSFD slot order; OTHERS must be slot index 5")
    return tuple(index for index, width in enumerate(values) if width == 4)


def _polarity_slots(task_id: str, counts: Sequence[int]) -> tuple[int, ...]:
    if task_id == "visfd":
        expected = (4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4)
        if tuple(int(value) for value in counts) != expected:
            raise ValueError(f"ViSFD slot schema drifted: {tuple(counts)!r}")
        return tuple(index for index in range(len(expected)) if index != 5)
    if task_id == "absa_restaurant":
        if tuple(int(value) for value in counts) != (4,) * 12:
            raise ValueError(f"Restaurant slot schema drifted: {tuple(counts)!r}")
        return tuple(range(12))
    raise ValueError(f"No E3 polarity mapping for {task_id!r}")


def _count_present(labels: Tensor, slots: Sequence[int]) -> int:
    if labels.ndim != 2:
        raise ValueError("ABSA labels must be a matrix")
    return int((labels[:, list(slots)] > 0).sum().item()) if slots else 0


def _base_metadata(
    *,
    repo_root: Path,
    task_id: str,
    seed: int,
    device: torch.device,
    precision: str,
    attention_implementation: str,
    lengths: Any,
    bundle: Any,
    model_spec: Any,
    model_hashes: Mapping[str, str],
    tokenizer_hashes: Mapping[str, str],
    counts: Sequence[int],
    polarity_slots: Sequence[int],
    source_hash_values: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "stage": "screening",
        "method": "conditional_polarity_ce",
        "coefficient": POLARITY_WEIGHT,
        "task_id": task_id,
        "seed": int(seed),
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRACTION,
        "gradient_clip_norm": 1.0,
        "max_length": int(lengths.max_length),
        "length_stats": lengths.to_dict(),
        "include_test": False,
        "test_access": "not_loaded",
        "smoke": False,
        "source_hashes": dict(source_hash_values),
        "e1_dependency_source_hash": source_hash_values["e1_dependency"],
        "e3_source_hash": source_hash_values["e3_source"],
        "data_provenance": bundle_provenance(bundle, repo_root),
        "task_metadata": task_metadata(bundle),
        "model": model_spec.to_dict(),
        "runtime": _runtime_metadata(device, precision),
        "attention_implementation": attention_implementation,
        "backbone_hashes": dict(model_hashes),
        "tokenizer_hashes": dict(tokenizer_hashes),
        "slot_class_counts": [int(value) for value in counts],
        "polarity_slots": [int(value) for value in polarity_slots],
        "loss_contract": {
            "base": "E1 compute_loss unchanged",
            "auxiliary": "pooled CE over gold-present slots, logits[:,1:4], target=gold-1",
            "polarity_weight": POLARITY_WEIGHT,
            "visfd_others_excluded": task_id == "visfd",
            "all_absent_auxiliary": "differentiable_zero",
        },
        "resumable": False,
        "resume_support": "fresh_rerun_only; no resume CLI",
        "git_commit": None,
        "argv": list(sys.argv),
    }


def train_trial(
    *,
    task_id: str,
    repo_root: Path,
    output_dir: Path,
    seed: int = SCREENING_SEED,
    device_name: str | None = None,
    precision: str = "bf16",
    attention_implementation: str = "sdpa",
    epochs: int = EPOCHS,
    expected_source_hash: str | None = None,
) -> dict[str, Any]:
    """Run one full validation-only E3 trial."""

    if task_id not in TASK_IDS:
        raise ValueError(f"E3 supports only {TASK_IDS}, got {task_id!r}")
    if seed != SCREENING_SEED:
        raise ValueError("E3 screening is locked to seed 42")
    if epochs != EPOCHS:
        raise ValueError("Full E3 trials are locked to 30 epochs")

    # Preserve E1's seed/device/source/model/data-loader construction order.
    seed_everything(seed)
    device = resolve_device(device_name)
    precision = resolve_precision(precision, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_hash_values = source_hashes(repo_root)
    if expected_source_hash is not None and source_hash_values["e3_source"] != expected_source_hash:
        raise RuntimeError(
            "E3 source changed after launch: "
            f"expected {expected_source_hash}, found {source_hash_values['e3_source']}"
        )
    trial_dir = _unique_dir(output_dir)
    trial_dir.mkdir(parents=True, exist_ok=False)
    _set_status(trial_dir, "starting", task_id=task_id, seed=seed)
    started = time.time()
    metadata: dict[str, Any] | None = None
    try:
        contract = _task_contract(task_id, repo_root)
        bundle = contract["bundle"]
        train_split = contract["train"]
        valid_split = contract["validation"]
        tokenizer = AutoTokenizer.from_pretrained(
            repo_root / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
        )
        lengths = inspect_train_lengths(tokenizer, list(split_texts(train_split)))
        if lengths.warning_over_512:
            raise RuntimeError(
                f"{task_id} train truncation remains above 1% at max_length=512"
            )
        if lengths.max_length != MAX_LENGTH:
            raise RuntimeError(
                f"E3 requires the locked E1 max_length={MAX_LENGTH}; audit selected {lengths.max_length}"
            )
        num_aspects = contract["num_aspects"]
        counts = contract["aspect_class_counts"]
        num_classes = contract["num_classes"]
        polarity_slots = _polarity_slots(task_id, counts)
        model, model_spec = build_classifier(
            repo_root / "models" / "goiturn-s-gd",
            num_classes=num_classes,
            num_aspects=num_aspects,
            aspect_class_counts=counts,
            local_files_only=True,
            attention_implementation=attention_implementation,
        )
        backbone_dir = repo_root / "models" / "goiturn-s-gd"
        backbone_hashes = directory_hashes(backbone_dir)
        tokenizer_hashes = {
            name: digest
            for name, digest in backbone_hashes.items()
            if name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
        }
        model.to(device)
        optimizer = make_optimizer(model, LEARNING_RATE)
        train_loader = _make_loader(
            tokenizer,
            train_split,
            max_length=MAX_LENGTH,
            batch_size=MICRO_BATCH_SIZE,
            shuffle=True,
            seed=seed,
            labels=model_labels(bundle, "train", train_split),
            gold_labels=split_labels(train_split),
        )
        valid_loader = _make_loader(
            tokenizer,
            valid_split,
            max_length=MAX_LENGTH,
            batch_size=MICRO_BATCH_SIZE,
            shuffle=False,
            seed=seed,
            labels=model_labels(bundle, "validation", valid_split),
            gold_labels=split_labels(valid_split),
        )
        total_steps = len(train_loader) * EPOCHS
        warmup_steps = round(total_steps * WARMUP_FRACTION)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        metadata = _base_metadata(
            repo_root=repo_root,
            task_id=task_id,
            seed=seed,
            device=device,
            precision=precision,
            attention_implementation=attention_implementation,
            lengths=lengths,
            bundle=bundle,
            model_spec=model_spec,
            model_hashes=backbone_hashes,
            tokenizer_hashes=tokenizer_hashes,
            counts=counts,
            polarity_slots=polarity_slots,
            source_hash_values=source_hash_values,
        )
        metadata["started_at_unix"] = time.time()
        _write_json(trial_dir / "resolved_config.json", metadata)
        _set_status(trial_dir, "running", task_id=task_id, seed=seed, epoch=0, global_step=0)

        best_score = -math.inf
        best_epoch = 0
        best_metrics: dict[str, Any] = {}
        best_predictions: list[dict[str, Any]] = []
        curves: list[dict[str, Any]] = []
        global_step = 0
        for epoch in range(1, EPOCHS + 1):
            epoch_started = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_examples = 0
            loss_totals = {"total": 0.0, "base_ce": 0.0, "polarity_ce": 0.0}
            active_slots = 0
            for batch_index, batch in enumerate(train_loader):
                moved = _move_batch(batch, device)
                with make_autocast(precision, device):
                    logits = model(
                        moved["input_ids"],
                        moved["attention_mask"],
                        token_type_ids=moved.get("token_type_ids"),
                    )
                    components = combined_polarity_loss(
                        logits,
                        moved["labels"],
                        num_aspects=num_aspects,
                        aspect_class_counts=counts,
                        polarity_weight=POLARITY_WEIGHT,
                        polarity_slots=polarity_slots,
                    )
                    loss = components.total
                    # E1 uses batch-size weighting.  E3 has no accumulation,
                    # so this is exactly loss itself for every full/partial batch.
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite E3 loss at epoch={epoch}, batch={batch_index}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                count = len(batch["labels"])
                total_examples += count
                loss_totals["total"] += float(components.total.detach().cpu()) * count
                loss_totals["base_ce"] += float(components.base_ce.detach().cpu()) * count
                loss_totals["polarity_ce"] += float(components.polarity_ce.detach().cpu()) * count
                active_slots += components.active_slot_count
            val_metrics, val_predictions = run_validation(
                model, valid_loader, device, precision, task_id, bundle
            )
            score = selection_score(val_metrics)
            if not math.isfinite(score):
                raise RuntimeError(f"Non-finite validation selection_f1 at epoch {epoch}")
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_losses": {
                    key: value / max(total_examples, 1) for key, value in loss_totals.items()
                },
                "train_active_polarity_slots": active_slots,
                "validation": val_metrics,
                "selection_f1": score,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_seconds": time.time() - epoch_started,
            }
            curves.append(row)
            _write_json(trial_dir / "epoch_curves.json", {"epochs": curves})
            _set_heartbeat(
                trial_dir,
                task_id=task_id,
                seed=seed,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_selection_f1=best_score if math.isfinite(best_score) else None,
            )
            _set_status(
                trial_dir,
                "running",
                task_id=task_id,
                seed=seed,
                epoch=epoch,
                global_step=global_step,
                best_epoch=best_epoch,
                best_selection_f1=best_score if math.isfinite(best_score) else None,
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = val_metrics
                best_predictions = val_predictions
                save_trial_checkpoint(
                    trial_dir / "best_checkpoint.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    metadata=metadata,
                )
                _write_json(trial_dir / "best_validation_metrics.json", best_metrics)
                _write_json(trial_dir / "validation_predictions.json", {"rows": best_predictions})

        metadata.update(
            {
                "best_epoch": best_epoch,
                "best_selection_f1": best_score,
                "best_validation_metrics": best_metrics,
                "elapsed_seconds": time.time() - started,
                "peak_gpu_memory_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
                "peak_gpu_memory_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                ),
                "checkpoint_sha256": sha256_file(trial_dir / "best_checkpoint.pt"),
                "status": "completed",
            }
        )
        _write_json(trial_dir / "metadata.json", metadata)
        _set_status(
            trial_dir,
            "completed",
            task_id=task_id,
            seed=seed,
            epoch=EPOCHS,
            global_step=global_step,
            best_epoch=best_epoch,
            best_selection_f1=best_score,
        )
        _set_heartbeat(
            trial_dir,
            task_id=task_id,
            seed=seed,
            epoch=EPOCHS,
            global_step=global_step,
            status="completed",
        )
        return {
            "task_id": task_id,
            "method": "conditional_polarity_ce",
            "seed": seed,
            "learning_rate": LEARNING_RATE,
            "artifact_dir": str(trial_dir),
            "best_epoch": best_epoch,
            "best_selection_f1": best_score,
            "validation_metrics": best_metrics,
            "status": "completed",
        }
    except Exception as error:
        failure_metadata = dict(metadata or {})
        failure_metadata.update(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
            }
        )
        _write_json(trial_dir / "metadata.json", failure_metadata)
        _set_status(trial_dir, "failed", task_id=task_id, seed=seed, error=repr(error))
        _set_heartbeat(trial_dir, task_id=task_id, seed=seed, status="failed")
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one validation-only GoiBERT E3 trial")
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SCREENING_SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--expected-source-hash", default=None)
    parser.add_argument("--smoke", action="store_true", help="reserved; full trial contract refuses smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.smoke:
        raise SystemExit("E3 runner has no smoke mode; use the orchestrator preflight")
    result = train_trial(
        task_id=args.task_id,
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        device_name=args.device,
        precision=args.precision,
        attention_implementation=args.attention_implementation,
        epochs=args.epochs,
        expected_source_hash=args.expected_source_hash,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
