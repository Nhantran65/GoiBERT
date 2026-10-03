"""Run one validation-only CafeBERT cross-head pilot cell."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cafe_cross_head import common  # noqa: E402
from scripts.cafe_cross_head.modeling import ARMS  # noqa: E402

import torch  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402


def _save_model_checkpoint(path: Path, model: torch.nn.Module, *, epoch: int, global_step: int, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "cafe-cross-head-model-only-v1", "model": model.state_dict(), "epoch": epoch, "global_step": global_step,
        "metadata": {k: metadata[k] for k in ("protocol", "model_kind", "task_id", "seed", "learning_rate", "model", "source_hash") if k in metadata},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train_one(model_kind: str, task_id: str, learning_rate: float, *, device_name: str, micro_batch_size: int, smoke: int = 0, out_dir: Path | None = None) -> dict[str, Any]:
    if model_kind not in ARMS:
        raise ValueError(model_kind)
    if task_id not in common.TASKS:
        raise ValueError(task_id)
    if learning_rate not in common.LEARNING_RATES:
        raise ValueError(learning_rate)
    output_dir = out_dir or common.run_dir(model_kind, task_id, learning_rate)
    existing = [n for n in ("epoch_curves.json", "best_checkpoint.pt", "metadata.json") if (output_dir / n).exists()]
    if existing and not smoke:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; existing {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    common.write_json(output_dir / "status.json", {"status": "running", "model_kind": model_kind, "task_id": task_id, "learning_rate": learning_rate, "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": device_name, "started_at": common.now()})
    common.write_json(output_dir / "heartbeat.json", {"status": "starting", "pid": os.getpid(), "updated_at": common.now()})

    epochs = 1 if smoke else common.EPOCHS
    if common.EFFECTIVE_BATCH_SIZE % micro_batch_size:
        raise ValueError("micro batch must divide effective batch size")
    accumulation_steps = common.EFFECTIVE_BATCH_SIZE // micro_batch_size
    common.e1_train.seed_everything(common.SEED)
    device = common.e1_train.resolve_device(device_name)
    precision = common.e1_train.resolve_precision(common.PRECISION, device)
    contract = common.task_contract(task_id)
    tokenizer = common.load_cafe_tokenizer()
    tokenizer_obj, tokenizer_manifest = tokenizer
    lengths = common.e1_train.inspect_train_lengths(tokenizer_obj, list(common.e1_train.split_texts(contract["train"])))
    model, model_spec = common.build_model(model_kind, contract)
    model.to(device)
    optimizer = common.e1_train.make_optimizer(model, learning_rate)
    train_loader, valid_loader = common.make_loaders(tokenizer_obj, contract, micro_batch_size)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs
    warmup_steps = round(total_steps * common.WARMUP_FRACTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    source_digest = common.source_hash()
    metadata: dict[str, Any] = {
        "protocol": common.PROTOCOL, "model_kind": model_kind, "task_id": task_id, "seed": common.SEED,
        "stage": "validation_only", "learning_rate": learning_rate, "epochs": epochs,
        "effective_batch_size": common.EFFECTIVE_BATCH_SIZE, "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps, "max_length": common.MAX_LENGTH,
        "precision": precision, "attention_implementation": common.ATTENTION_IMPLEMENTATION,
        "weight_decay": common.WEIGHT_DECAY, "betas": [0.9, 0.999], "adamw_epsilon": 1e-8,
        "warmup_fraction": common.WARMUP_FRACTION, "warmup_steps": warmup_steps, "total_steps": total_steps,
        "gradient_clip_norm": common.GRADIENT_CLIP_NORM, "loss": "plain_cross_entropy (absent class 0 included)",
        "head": f"first token -> dropout0.1 -> dense{model_spec.get('head_hidden_size', common.HEAD_HIDDEN_SIZE)} -> tanh -> dropout0.1 -> task logits",
        "pooling": model_spec.get("pooling", "first token"),
        "selection": "validation task-specific selection_f1, best epoch", "include_test": False,
        "smoke": int(smoke) if smoke else None, "source_hash": source_digest, "model": model_spec,
        "tokenizer": tokenizer_manifest, "data_provenance": common.e1_train.bundle_provenance(contract["bundle"], common.ROOT),
        "task_metadata": common.e1_train.task_metadata(contract["bundle"]), "train_count": len(contract["train"]),
        "validation_count": len(contract["validation"]), "num_aspects": contract["num_aspects"],
        "aspect_class_counts": list(contract["aspect_class_counts"]), "runtime": common.runtime(device),
        "gpu_snapshot_at_start": common.gpu_snapshot(), "length_stats": lengths.to_dict(),
        "started_at": common.now(), "started_at_unix": started, "argv": list(sys.argv), "pid": os.getpid(),
    }
    common.write_json(output_dir / "resolved_config.json", metadata)
    print(f"[{model_kind}/{task_id}/lr={learning_rate:.0e}] start epochs={epochs} micro={micro_batch_size} accum={accumulation_steps} steps={total_steps} warmup={warmup_steps} device={device}", flush=True)
    best_score, best_epoch, best_metrics, best_predictions = -math.inf, 0, {}, []
    curves: list[dict[str, Any]] = []
    global_step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_loss = total_ce = 0.0; total_examples = 0; epoch_started = time.time()
        n_batches = len(train_loader) if not smoke else min(len(train_loader), int(smoke))
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= n_batches:
                break
            moved = common.e1_train._move_batch(batch, device)
            with common.e1_train.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                ce = common.e1_train.compute_loss(logits, moved["labels"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"])
                # Scale each accumulation window by the number of examples it
                # contains.  This keeps a short final window unbiased when a
                # micro batch does not divide the split size.
                window_start = (batch_index // accumulation_steps) * accumulation_steps
                window_end = min(window_start + accumulation_steps, n_batches)
                window_examples = sum(min(micro_batch_size, len(train_loader.dataset) - index * micro_batch_size) for index in range(window_start, window_end))
                loss = ce * (len(batch["labels"]) / max(window_examples, 1))
            if not torch.isfinite(ce.detach().float()):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, batch={batch_index}")
            loss.backward()
            bs = int(moved["input_ids"].shape[0]); total_examples += bs; total_loss += float(ce.detach().float()) * bs; total_ce += float(ce.detach().float()) * bs
            boundary = ((batch_index + 1) % accumulation_steps == 0) or (batch_index + 1 == n_batches)
            if boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), common.GRADIENT_CLIP_NORM, error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            if batch_index % 25 == 0:
                common.write_json(output_dir / "heartbeat.json", {"status": "running", "epoch": epoch, "batch": batch_index, "global_step": global_step, "updated_at": common.now(), "pid": os.getpid()})
        val_metrics, val_predictions = common.e1_train.run_validation(model, valid_loader, device, precision, task_id, contract["bundle"])
        score = common.e1_train.selection_score(val_metrics)
        row = {"epoch": epoch, "global_step": global_step, "train_loss": total_loss / max(total_examples, 1), "train_ce": total_ce / max(total_examples, 1), "selection_f1": score, "validation": val_metrics, "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - epoch_started}
        curves.append(row); common.write_json(output_dir / "epoch_curves.json", {"epochs": curves})
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_metrics, best_predictions = score, epoch, val_metrics, val_predictions
            _save_model_checkpoint(output_dir / "best_checkpoint.pt", model, epoch=epoch, global_step=global_step, metadata=metadata)
            common.write_json(output_dir / "best_validation_metrics.json", {"epoch": epoch, **best_metrics})
            common.write_json(output_dir / "validation_predictions.json", {"epoch": epoch, "rows": best_predictions})
        common.write_json(output_dir / "heartbeat.json", {"status": "running", "epoch": epoch, "global_step": global_step, "best_epoch": best_epoch, "best_selection_f1": best_score, "updated_at": common.now(), "pid": os.getpid()})
        print(f"[{model_kind}/{task_id}/lr={learning_rate:.0e}] epoch {epoch:2d} loss {row['train_loss']:.4f} val_f1 {100*score:.2f} best {100*best_score:.2f}@{best_epoch} ({row['epoch_seconds']:.0f}s){' *' if improved else ''}", flush=True)
    elapsed = time.time() - started
    checkpoint = output_dir / "best_checkpoint.pt"
    metadata.update({"best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics, "elapsed_seconds": elapsed, "finished_at": common.now(), "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None, "checkpoint_sha256": common.sha256_file(checkpoint), "checkpoint_format": "model state_dict only (not resumable)"})
    del optimizer, scheduler
    reload_model, _ = common.build_model(model_kind, contract)
    reload_state = common.load_model_checkpoint(checkpoint, reload_model); reload_model.to(device)
    reload_metrics, reload_rows = common.e1_train.run_validation(reload_model, valid_loader, device, precision, task_id, contract["bundle"])
    reload_f1 = common.e1_train.selection_score(reload_metrics)
    saved = common.read_json(output_dir / "validation_predictions.json")["rows"]
    pred_file_f1 = common.e1_train.selection_score(common.e1_train.evaluate_predictions(task_id, contract["bundle"], [r["gold"] for r in saved], [r["prediction"] for r in saved], split="validation"))
    metadata["postrun_checks"] = {"checkpoint_epoch": reload_state.get("epoch"), "best_epoch": best_epoch, "reload_selection_f1": reload_f1, "best_selection_f1": best_score, "reload_matches_best": abs(reload_f1 - best_score) < 1e-9, "prediction_file_selection_f1": pred_file_f1, "prediction_file_matches_best": abs(pred_file_f1 - best_score) < 1e-9, "reload_ids_identical": [r["id"] for r in reload_rows] == [r["id"] for r in saved], "reload_predictions_identical": [r["prediction"] for r in reload_rows] == [r["prediction"] for r in saved]}
    ok = all((metadata["postrun_checks"][k] for k in ("reload_matches_best", "prediction_file_matches_best", "reload_ids_identical", "reload_predictions_identical")))
    metadata["status"] = "completed" if ok else "completed_with_check_failure"
    common.write_json(output_dir / "metadata.json", metadata)
    common.write_json(output_dir / "status.json", {"status": metadata["status"], "model_kind": model_kind, "task_id": task_id, "learning_rate": learning_rate, "best_epoch": best_epoch, "best_selection_f1": best_score, "elapsed_seconds": elapsed, "finished_at": common.now(), "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": device_name, "postrun_checks": metadata["postrun_checks"]})
    common.write_json(output_dir / "heartbeat.json", {"status": metadata["status"], "best_epoch": best_epoch, "best_selection_f1": best_score, "updated_at": common.now(), "pid": os.getpid()})
    print(f"[{model_kind}/{task_id}/lr={learning_rate:.0e}] done best {100*best_score:.2f}@{best_epoch} elapsed {elapsed/60:.1f} min; checks={metadata['postrun_checks']}", flush=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=ARMS, required=True)
    parser.add_argument("--task", choices=common.TASKS, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    output_dir = Path(args.out_dir) if args.out_dir else common.run_dir(args.model_kind, args.task, args.learning_rate)
    try:
        train_one(args.model_kind, args.task, args.learning_rate, device_name=args.device, micro_batch_size=args.micro_batch, smoke=args.smoke, out_dir=output_dir)
        return 0
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        common.write_json(output_dir / "status.json", {"status": "failed", "model_kind": args.model_kind, "task_id": args.task, "learning_rate": args.learning_rate, "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "finished_at": common.now(), "error": repr(error), "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
