"""Fine-tune one pilot arm (T / B / K) on absa_restaurant, validation-only, best checkpoint by pair micro-F1.

The loop is the E14 trial loop (scripts/run_e14.py::_run_trial) with an optional logit-KD term for arm K:
    total = (1 - alpha) * CE + alpha * T**2 * KL(teacher || student)      (goibert_e2.losses.logits_kd_loss)
Teacher logits come from an immutable train-only cache joined by stable example ID (goibert_e2.cache).

Outputs (artifacts/cdbk_pilot/runs/<arm>_absa_restaurant_seed42/):
    resolved_config.json, epoch_curves.json, best_checkpoint.pt (model state only), best_validation_metrics.json,
    validation_predictions.json, metadata.json (incl. post-run reload / prediction-file checks), status.json.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (  # noqa: E402
    ARMS, ATTENTION_IMPLEMENTATION, EFFECTIVE_BATCH_SIZE, EPOCHS, GRADIENT_CLIP_NORM, KD_ALPHA, KD_TEMPERATURE,
    MAX_LENGTH, PRECISION, PROTOCOL, ROOT, SEED, TASK_ID, WARMUP_FRACTION, WEIGHT_DECAY, e1_train,
)

import torch  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from goibert_e2.cache import load_teacher_logit_cache  # noqa: E402
from goibert_e2.losses import logits_kd_loss  # noqa: E402


def _save_model_checkpoint(path: Path, model: torch.nn.Module, *, epoch: int, global_step: int, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "cdbk-pilot-model-only-v1",
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metadata": {k: metadata[k] for k in ("protocol", "arm", "task_id", "seed", "learning_rate", "model", "source_hash") if k in metadata},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    arm = args.arm
    spec = ARMS[arm]
    output_dir = Path(args.out_dir) if args.out_dir else common.run_dir(arm)
    existing = [n for n in ("epoch_curves.json", "best_checkpoint.pt", "metadata.json") if (output_dir / n).exists()]
    if existing and not args.smoke:
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir} has {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_unix = time.time()
    common.write_json(output_dir / "status.json", {"status": "running", "arm": arm, "pid": os.getpid(), "device": args.device, "started_at": common.now()})

    epochs = 1 if args.smoke else EPOCHS
    micro = int(args.micro_batch)
    if micro <= 0 or EFFECTIVE_BATCH_SIZE % micro:
        raise ValueError("micro batch must divide effective batch size")
    accumulation_steps = EFFECTIVE_BATCH_SIZE // micro

    expected_source_hash = common.source_hash()
    e1_train.seed_everything(SEED)
    device = e1_train.resolve_device(args.device)
    precision = e1_train.resolve_precision(PRECISION, device)
    contract = common.task_contract()
    bundle = contract["bundle"]
    num_aspects, counts = contract["num_aspects"], contract["aspect_class_counts"]
    tokenizer, tokenizer_manifest = common.load_tokenizer(spec["init"])
    lengths = e1_train.inspect_train_lengths(tokenizer, list(e1_train.split_texts(contract["train"])))
    model, model_spec = common.build_model(spec["init"], contract)
    model.to(device)
    learning_rate = float(spec["learning_rate"])
    optimizer = e1_train.make_optimizer(model, learning_rate)
    train_loader, valid_loader = common.make_loaders(tokenizer, contract, micro)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = round(total_steps * WARMUP_FRACTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    teacher_cache = None
    kd_meta: dict[str, Any] = {"enabled": bool(spec["kd"])}
    if spec["kd"]:
        if not args.teacher_cache:
            raise ValueError("arm K requires --teacher-cache")
        cache_path = Path(args.teacher_cache)
        teacher_cache = load_teacher_logit_cache(cache_path)
        train_ids = [str(v) for v in e1_train.split_ids(contract["train"])]
        if [str(v) for v in teacher_cache.ids] != train_ids:
            raise RuntimeError("teacher cache IDs are not the ordered train IDs of this task")
        if teacher_cache.task_id != TASK_ID or tuple(teacher_cache.logits.shape[1:]) != (num_aspects, counts[0]):
            raise RuntimeError(f"teacher cache schema mismatch: task={teacher_cache.task_id}, shape={tuple(teacher_cache.logits.shape)}")
        from goibert_e2.cache import text_hash
        if teacher_cache.metadata["source_text_hash"] != text_hash(list(e1_train.split_texts(contract["train"]))):
            raise RuntimeError("teacher cache text hash differs from the loaded train split")
        kd_meta.update({
            "teacher_cache": str(cache_path), "teacher_cache_sha256": common.sha256_file(cache_path),
            "teacher_hash": teacher_cache.metadata["teacher_hash"], "teacher_run": teacher_cache.metadata.get("source_hashes", {}).get("teacher_run"),
            "temperature": KD_TEMPERATURE, "alpha": KD_ALPHA, "loss": "(1-alpha)*CE + alpha*T^2*KL(teacher||student); logits_kd_loss already multiplies by T^2",
            "join": "stable example id (absa_restaurant:train:NNNNNN), not token ids", "cache_logits_shape": list(teacher_cache.logits.shape),
        })

    metadata: dict[str, Any] = {
        "protocol": PROTOCOL, "arm": arm, "task_id": TASK_ID, "seed": SEED, "stage": "validation_only",
        "init": spec["init"], "learning_rate": learning_rate, "epochs": epochs, "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "micro_batch_size": micro, "gradient_accumulation_steps": accumulation_steps, "max_length": MAX_LENGTH,
        "precision": precision, "attention_implementation": ATTENTION_IMPLEMENTATION, "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999], "adamw_epsilon": 1e-8, "warmup_fraction": WARMUP_FRACTION, "warmup_steps": warmup_steps,
        "total_steps": total_steps, "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "loss": "plain_cross_entropy (absent class 0 included)" if not spec["kd"] else "0.5*CE + 0.5*KD",
        "kd": kd_meta,
        "head": f"CLS -> dropout0.1 -> dense{spec['head_hidden']} -> tanh -> dropout0.1 -> 48 logits (12 slots x 4)",
        "selection": "validation pair_micro_f1 (selection_f1), best epoch", "include_test": False,
        "smoke": int(args.smoke) if args.smoke else None,
        "source_hash": expected_source_hash, "model": model_spec, "tokenizer": tokenizer_manifest,
        "data_provenance": e1_train.bundle_provenance(bundle, ROOT), "task_metadata": e1_train.task_metadata(bundle),
        "train_count": len(contract["train"]), "validation_count": len(contract["validation"]),
        "num_aspects": num_aspects, "aspect_class_counts": list(counts),
        "runtime": common.runtime(device), "gpu_snapshot_at_start": common.gpu_snapshot(), "length_stats": lengths.to_dict(),
        "started_at": common.now(), "started_at_unix": started_unix, "argv": list(sys.argv), "pid": os.getpid(),
    }
    common.write_json(output_dir / "resolved_config.json", metadata)
    print(f"[{arm}] start lr={learning_rate} epochs={epochs} micro={micro} accum={accumulation_steps} steps={total_steps} warmup={warmup_steps} device={device}", flush=True)

    best_score, best_epoch, best_metrics, best_predictions = -math.inf, 0, {}, []
    curves: list[dict[str, Any]] = []
    global_step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"total": 0.0, "ce": 0.0, "kd": 0.0}
        total_examples = 0
        epoch_started = time.time()
        n_batches = len(train_loader) if not args.smoke else min(len(train_loader), int(args.smoke))
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= n_batches:
                break
            moved = e1_train._move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                ce = e1_train.compute_loss(logits, moved["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                total = ce
                kd_value = None
                if teacher_cache is not None:
                    teacher_logits = teacher_cache.lookup(batch["ids"], device=device)
                    kd_value = logits_kd_loss(logits, teacher_logits, temperature=KD_TEMPERATURE, slot_class_counts=counts)
                    total = (1.0 - KD_ALPHA) * ce + KD_ALPHA * kd_value
                window_start = (batch_index // accumulation_steps) * accumulation_steps
                window_end = min(window_start + accumulation_steps, n_batches)
                window_examples = sum(min(micro, len(train_loader.dataset) - index * micro) for index in range(window_start, window_end))
                scaled_loss = total * (len(batch["labels"]) / max(window_examples, 1))
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, batch={batch_index}")
            scaled_loss.backward()
            count = len(batch["labels"])
            totals["total"] += float(total.detach().cpu()) * count
            totals["ce"] += float(ce.detach().cpu()) * count
            if kd_value is not None:
                totals["kd"] += float(kd_value.detach().cpu()) * count
            total_examples += count
            boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == n_batches
            if boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
        if args.smoke:
            # evaluate on a prefix of validation only; metrics are not meaningful, just exercises the path
            from torch.utils.data import DataLoader, Subset
            n_val = min(len(valid_loader.dataset), int(args.smoke) * micro)
            valid_loader = DataLoader(Subset(valid_loader.dataset, range(n_val)), batch_size=micro, shuffle=False, collate_fn=valid_loader.collate_fn)
        val_metrics, val_predictions = e1_train.run_validation(model, valid_loader, device, precision, TASK_ID, bundle)
        score = e1_train.selection_score(val_metrics)
        row = {
            "epoch": epoch, "global_step": global_step, "train_loss": totals["total"] / max(total_examples, 1),
            "train_ce": totals["ce"] / max(total_examples, 1), "train_kd": (totals["kd"] / max(total_examples, 1)) if teacher_cache is not None else None,
            "selection_f1": score, "validation": val_metrics, "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - epoch_started,
        }
        curves.append(row)
        common.write_json(output_dir / "epoch_curves.json", {"epochs": curves})
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_metrics, best_predictions = score, epoch, val_metrics, val_predictions
            _save_model_checkpoint(output_dir / "best_checkpoint.pt", model, epoch=epoch, global_step=global_step, metadata=metadata)
            common.write_json(output_dir / "best_validation_metrics.json", {"epoch": epoch, **best_metrics})
            common.write_json(output_dir / "validation_predictions.json", {"epoch": epoch, "rows": best_predictions})
        print(f"[{arm}] epoch {epoch:2d} loss {row['train_loss']:.4f} ce {row['train_ce']:.4f} kd {row['train_kd'] if row['train_kd'] is not None else '-'} val_f1 {100*score:.2f} best {100*best_score:.2f}@{best_epoch} ({row['epoch_seconds']:.0f}s){' *' if improved else ''}", flush=True)

    elapsed = time.time() - started_unix
    checkpoint = output_dir / "best_checkpoint.pt"
    metadata.update({
        "best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics,
        "elapsed_seconds": elapsed, "finished_at": common.now(),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "checkpoint_sha256": common.sha256_file(checkpoint), "checkpoint_format": "model state_dict only (not resumable)",
    })

    # ---- post-run checks: reload best checkpoint -> validation F1 must match; predictions file -> same F1
    del optimizer, scheduler
    reload_model, _ = common.build_model(spec["init"], contract)
    reload_state = common.load_model_checkpoint(checkpoint, reload_model)
    reload_model.to(device)
    reload_metrics, reload_rows = e1_train.run_validation(reload_model, valid_loader, device, precision, TASK_ID, bundle)
    reload_f1 = e1_train.selection_score(reload_metrics)
    saved = common.read_json(output_dir / "validation_predictions.json")["rows"]
    pred_file_f1 = e1_train.selection_score(e1_train.evaluate_predictions(TASK_ID, bundle, [r["gold"] for r in saved], [r["prediction"] for r in saved], split="validation"))
    same_rows = [r["prediction"] for r in reload_rows] == [r["prediction"] for r in saved]
    metadata["postrun_checks"] = {
        "checkpoint_epoch": reload_state.get("epoch"), "best_epoch": best_epoch,
        "reload_selection_f1": reload_f1, "best_selection_f1": best_score, "reload_matches_best": abs(reload_f1 - best_score) < 1e-9,
        "prediction_file_selection_f1": pred_file_f1, "prediction_file_matches_best": abs(pred_file_f1 - best_score) < 1e-9,
        "reload_predictions_identical": same_rows,
    }
    ok = metadata["postrun_checks"]["reload_matches_best"] and metadata["postrun_checks"]["prediction_file_matches_best"] and same_rows
    metadata["status"] = "completed" if ok else "completed_with_check_failure"
    common.write_json(output_dir / "metadata.json", metadata)
    common.write_json(output_dir / "status.json", {"status": metadata["status"], "arm": arm, "best_epoch": best_epoch, "best_selection_f1": best_score, "elapsed_seconds": elapsed, "finished_at": common.now(), "pid": os.getpid(), "device": args.device, "postrun_checks": metadata["postrun_checks"]})
    print(f"[{arm}] done best {100*best_score:.2f}@{best_epoch} elapsed {elapsed/60:.1f} min; postrun_checks={metadata['postrun_checks']}", flush=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--teacher-cache", default=None, help="arm K: path to the T train-logit cache")
    parser.add_argument("--out-dir", default=None, help="override output directory (used for smoke runs)")
    parser.add_argument("--smoke", type=int, default=0, help="debug: 1 epoch, N train batches, N validation batches")
    args = parser.parse_args()
    output_dir = Path(args.out_dir) if args.out_dir else common.run_dir(args.arm)
    try:
        run(args)
        return 0
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        common.write_json(output_dir / "status.json", {"status": "failed", "arm": args.arm, "pid": os.getpid(), "finished_at": common.now(), "error": repr(error), "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
