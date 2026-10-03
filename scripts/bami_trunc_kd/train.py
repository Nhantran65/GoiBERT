#!/usr/bin/env python3
"""Validation-only BamiBERT lower-layer truncation and Cafe OOF KD pilot."""
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "cafe_oof_kd")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

import common as C  # noqa: E402
from goibert import train as T  # noqa: E402
from goibert_e14.modeling import build_bamibert_classifier  # noqa: E402

ART = ROOT / "artifacts" / "bami_trunc_kd"
MODEL = ROOT / "models" / "bamibert-base-original"
TASKS = ("vsfc_sentiment", "absa_restaurant")
LR = {"vsfc_sentiment": 2e-5, "absa_restaurant": 5e-5}
MICRO = 32
EPOCHS = 30


def write_json(path: Path, obj: object) -> None:
    C.write_json(path, obj)


def truncate(model, layers: int) -> dict[str, int]:
    actual = len(model.encoder.encoder.layer)
    if actual != 12:
        raise RuntimeError(f"Expected 12 pretrained layers, got {actual}")
    model.encoder.encoder.layer = torch.nn.ModuleList(list(model.encoder.encoder.layer[:layers]))
    model.encoder.config.num_hidden_layers = layers
    if len(model.encoder.encoder.layer) != layers or model.encoder.config.num_hidden_layers != layers:
        raise RuntimeError("Truncation failed")
    return {"encoder": sum(p.numel() for p in model.encoder.parameters()),
            "head": sum(p.numel() for p in model.head.parameters()),
            "total": sum(p.numel() for p in model.parameters())}


def teacher_logits(task: str, bundle, split) -> tuple[dict[str, torch.Tensor], dict]:
    path = C.ART / "soft" / f"{task}.npz"
    with np.load(path, allow_pickle=False) as data:
        ids = [str(i) for i in data["ids"]]
        logits = data["logits"]
        counts = tuple(int(n) for n in data["aspect_class_counts"])
    widths = C.label_widths(bundle)
    if counts and counts != tuple(widths):
        raise ValueError(f"Teacher aspect widths disagree for {task}: {counts} != {widths}")
    if logits.ndim != 2 or logits.shape != (len(ids), sum(widths)):
        raise ValueError(f"Teacher logits shape {logits.shape}, expected {(len(ids), sum(widths))}")
    if len(set(ids)) != len(ids) or not np.isfinite(logits).all():
        raise ValueError("Duplicate teacher IDs or non-finite logits")
    lookup = {i: torch.tensor(row, dtype=torch.float32) for i, row in zip(ids, logits)}
    train_ids = [str(i) for i in T.split_ids(split)]
    missing = set(train_ids) - set(lookup)
    if missing:
        raise ValueError(f"{len(missing)} train IDs lack teacher logits")
    return lookup, {"path": str(path.relative_to(ROOT)), "train_ids": len(train_ids),
                    "covered_train_ids": len(train_ids), "widths": list(widths),
                    "logit_width": int(logits.shape[1])}


def kd_term(student_flat, teacher_flat, segments, temperature: float):
    return torch.stack([
        F.kl_div(F.log_softmax(student_flat[:, a:b] / temperature, -1),
                 F.softmax(teacher_flat[:, a:b] / temperature, -1), reduction="batchmean") * temperature**2
        for a, b in segments
    ]).mean()


def build_model(bundle, layers: int):
    aspects = T.task_num_aspects(bundle)
    model, _ = build_bamibert_classifier(
        MODEL, num_classes=T.task_num_classes(bundle), num_aspects=aspects,
        aspect_class_counts=T.task_aspect_class_counts(bundle) if aspects else None,
        attention_implementation="sdpa")
    params = truncate(model, layers)
    return model, params


def run(args) -> int:
    out = args.out or ART / "runs" / f"L{args.layers}_{args.arm}" / args.task / f"seed_{args.seed}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    write_json(out / "status.json", {"status": "running", "pid": os.getpid(), "gpu": gpu, "started_at_unix": started})
    try:
        # Mirrors E14 _run_trial call order for L12-CE seed 42.
        T.seed_everything(args.seed)
        device = T.resolve_device("cuda:0")
        precision = T.resolve_precision("bf16", device)
        bundle = T.load_task_bundle(args.task, ROOT, include_test=False)
        train_split = T.split_field(bundle, "train")
        valid_split = T.split_field(bundle, "validation")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
        lengths = T.inspect_train_lengths(tokenizer, list(T.split_texts(train_split)))
        model, params = build_model(bundle, args.layers)
        model.to(device)
        optimizer = T.make_optimizer(model, LR[args.task])
        train_loader = T._make_loader(tokenizer, train_split, max_length=256, batch_size=MICRO,
                                      shuffle=True, seed=args.seed, labels=T.model_labels(bundle, "train", train_split),
                                      gold_labels=T.split_labels(train_split))
        valid_loader = T._make_loader(tokenizer, valid_split, max_length=256, batch_size=MICRO,
                                      shuffle=False, seed=args.seed, labels=T.model_labels(bundle, "validation", valid_split),
                                      gold_labels=T.split_labels(valid_split))
        batches = min(len(train_loader), args.max_train_batches or len(train_loader))
        total_steps = batches * args.epochs
        warmup_steps = round(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        teacher, coverage = (teacher_logits(args.task, bundle, train_split) if args.arm == "kd" else (None, None))
        segments = C.slot_segments(C.label_widths(bundle))
        config = {"task": args.task, "layers": args.layers, "arm": args.arm, "seed": args.seed,
                  "learning_rate": LR[args.task], "epochs": args.epochs, "micro_batch_size": MICRO,
                  "effective_batch_size": 32, "gradient_accumulation_steps": 1,
                  "max_train_batches": args.max_train_batches, "max_length": 256, "precision": precision,
                  "attention_implementation": "sdpa",
                  "warmup_steps": warmup_steps, "total_steps": total_steps, "weight_decay": 0.01,
                  "gradient_clip_norm": 1.0, "kd_alpha": 0.5 if teacher is not None else None,
                  "kd_temperature": 2.0 if teacher is not None else None,
                  "include_test": False, "model_spec": model.spec().to_dict(),
                  "parameter_counts": params, "teacher_coverage": coverage,
                  "length_stats": lengths.to_dict(), "started_at_unix": started, "gpu": gpu}
        write_json(out / "resolved_config.json", config)
        best, best_epoch, curves = -math.inf, 0, []
        for epoch in range(1, args.epochs + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss_sum = kd_sum = 0.0
            n = 0
            epoch_started = time.time()
            for batch_index, batch in enumerate(train_loader):
                if batch_index >= batches:
                    break
                moved = T._move_batch(batch, device)
                with T.make_autocast(precision, device):
                    logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                    ce = T.compute_loss(logits, moved["labels"], num_aspects=T.task_num_aspects(bundle),
                                        aspect_class_counts=T.task_aspect_class_counts(bundle))
                loss = ce
                if teacher is not None:
                    tl = torch.stack([teacher[str(i)] for i in batch["ids"]]).to(device)
                    kd = kd_term(C.flat_logits(logits.float(), len(batch["ids"])), tl, segments, 2.0)
                    loss = 0.5 * ce + 0.5 * kd
                    kd_sum += float(kd.detach().cpu()) * len(batch["ids"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                loss_sum += float(loss.detach().cpu()) * len(batch["ids"])
                n += len(batch["ids"])
            metrics, rows = T.run_validation(model, valid_loader, device, precision, args.task, bundle)
            score = T.selection_score(metrics)
            curves.append({"epoch": epoch, "global_step": epoch * batches, "train_loss": loss_sum / n,
                           "train_kd": kd_sum / n if teacher is not None else None, "validation": metrics,
                           "selection_f1": score, "learning_rate": scheduler.get_last_lr()[0],
                           "epoch_seconds": time.time() - epoch_started})
            write_json(out / "epoch_curves.json", {"epochs": curves})
            if score > best:
                best, best_epoch = score, epoch
                T.save_trial_checkpoint(out / "best_checkpoint.pt", model, optimizer, scheduler,
                                        epoch=epoch, global_step=epoch * batches, metadata=config)
                write_json(out / "validation_predictions.json", {"rows": rows})
                write_json(out / "best_validation_metrics.json", metrics)
            print(f"L{args.layers}-{args.arm} {args.task} seed={args.seed} epoch={epoch} loss={loss_sum/n:.8f} val={score:.10f} best={best:.10f}@{best_epoch}", flush=True)
        reload_model, _ = build_model(bundle, args.layers)
        checkpoint = out / "best_checkpoint.pt"
        reload_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
        reload_model.to(device)
        reload_metrics, reload_rows = T.run_validation(reload_model, valid_loader, device, precision, args.task, bundle)
        saved = json.loads((out / "validation_predictions.json").read_text())["rows"]
        file_f1 = T.selection_score(T.evaluate_predictions(args.task, bundle,
                            [r["gold"] for r in saved], [r["prediction"] for r in saved], split="validation"))
        checks = {"reload_matches_best": abs(T.selection_score(reload_metrics) - best) < 1e-9,
                  "prediction_file_matches_best": abs(file_f1 - best) < 1e-9,
                  "reload_predictions_identical": [r["prediction"] for r in reload_rows] == [r["prediction"] for r in saved],
                  "ids_stable_and_complete": [r["id"] for r in reload_rows] == [r["id"] for r in saved] == list(T.split_ids(valid_split)),
                  "finite_train_loss": all(math.isfinite(r["train_loss"]) for r in curves),
                  "teacher_train_coverage": coverage is None or coverage["covered_train_ids"] == coverage["train_ids"]}
        status = "completed" if all(checks.values()) else "completed_with_check_failure"
        meta = {**config, "status": status, "best_epoch": best_epoch, "best_selection_f1": best,
                "best_validation_metrics": json.loads((out / "best_validation_metrics.json").read_text()),
                "postrun_checks": checks, "elapsed_seconds": time.time() - started,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)}
        write_json(out / "metadata.json", meta)
        write_json(out / "status.json", {"status": status, "pid": os.getpid(), "gpu": gpu,
                                         "best_epoch": best_epoch, "best_selection_f1": best})
        return 0 if status == "completed" else 1
    except Exception as error:
        write_json(out / "status.json", {"status": "failed", "pid": os.getpid(), "gpu": gpu,
                                         "error": repr(error), "traceback": traceback.format_exc()})
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--layers", type=int, choices=(12, 10, 8), required=True)
    ap.add_argument("--arm", choices=("ce", "kd"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.epochs <= 0 or (args.max_train_batches is not None and args.max_train_batches <= 0):
        ap.error("epochs and max-train-batches must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
