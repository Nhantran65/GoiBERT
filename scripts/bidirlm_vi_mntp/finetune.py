#!/usr/bin/env python3
"""Stage 1 §4 screen cell: --arm {O,P_vi} --task <task> --lr <lr> --seed <seed> (copy of the stage-0 runner).

E14 recipe (30 epochs, effective batch 32, maxlen 256, bf16 autocast + master fp32, AdamW wd 0.01,
warmup 10%, clip 1.0, CE, best epoch on validation), validation-only.
Outputs artifacts/bidirlm_vi_mntp/ft/<arm>/<task>/lr_<lr>/seed_<seed>/{resolved_config,epoch_curves,
best_validation_metrics,validation_predictions,metadata,status}.json + best_checkpoint.pt
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import torch  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    e1_train.write_json(path, payload)


def arm_runtime_config(arm: str) -> dict[str, Any]:
    return {"attention": common.ARMS[arm]["attention"], "mask_mode": common.ARMS[arm]["mask"],
            "pooling": common.ARMS[arm]["pooling"], "eos_appended": common.ARMS[arm]["mask"] != "native",
            "micro_batch_default": common.ARMS[arm]["micro_batch"]}


def _hash_params(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    arm, task_id, lr, seed = args.arm, args.task, float(args.lr), int(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else common.arm_dir(arm, task_id, lr, seed)
    if any((out_dir / n).exists() for n in ("metadata.json", "best_checkpoint.pt")) and not args.smoke:
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(out_dir / "status.json", {"status": "running", "arm": arm, "task_id": task_id, "learning_rate": lr,
                                         "seed": seed, "pid": os.getpid(), "device": args.device, "started_at": common.now()})
    epochs = 1 if args.smoke else common.EPOCHS
    micro = int(args.micro_batch)
    accum = common.EFFECTIVE_BATCH // micro
    if common.EFFECTIVE_BATCH % micro:
        raise ValueError("micro batch must divide 32")
    e1_train.seed_everything(seed)
    device = e1_train.resolve_device(args.device)
    precision = e1_train.resolve_precision(common.PRECISION, device)
    bundle = e1_train.load_task_bundle(task_id, common.ROOT, include_test=False)
    train_split, valid_split = e1_train.split_field(bundle, "train"), e1_train.split_field(bundle, "validation")
    num_aspects, counts = e1_train.task_num_aspects(bundle), e1_train.task_aspect_class_counts(bundle)
    tokenizer = common.load_tokenizer(arm)
    lengths = e1_train.inspect_train_lengths(tokenizer, list(e1_train.split_texts(train_split)))
    model, spec = common.build_model(arm, num_classes=e1_train.task_num_classes(bundle), num_aspects=num_aspects,
                                     aspect_class_counts=counts)
    head_init_hash = _hash_params(model.head)
    model.to(device)
    optimizer = e1_train.make_optimizer(model, lr)
    train_loader = e1_train._make_loader(tokenizer, train_split, max_length=common.MAX_LENGTH, batch_size=micro, shuffle=True,
                                         seed=seed, labels=e1_train.model_labels(bundle, "train", train_split),
                                         gold_labels=e1_train.split_labels(train_split))
    valid_loader = e1_train._make_loader(tokenizer, valid_split, max_length=common.MAX_LENGTH, batch_size=micro, shuffle=False,
                                         seed=seed, labels=e1_train.model_labels(bundle, "validation", valid_split),
                                         gold_labels=e1_train.split_labels(valid_split))
    total_steps = math.ceil(len(train_loader) / accum) * epochs
    warmup = round(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    meta: dict[str, Any] = {
        "protocol": common.PROTOCOL, "arm": arm, "task_id": task_id, "seed": seed, "learning_rate": lr, "epochs": epochs,
        "effective_batch_size": common.EFFECTIVE_BATCH, "micro_batch_size": micro, "gradient_accumulation_steps": accum,
        "max_length": common.MAX_LENGTH, "precision": precision, "attention_implementation": common.ATTN,
        "attention_dropout": common.ATTENTION_DROPOUT, "weight_decay": 0.01, "betas": [0.9, 0.999], "adamw_epsilon": 1e-8,
        "warmup_fraction": 0.1, "warmup_steps": warmup, "total_steps": total_steps, "gradient_clip_norm": 1.0,
        "loss": "plain_cross_entropy",
        "head": f"CLS -> dropout0.1 -> dense{spec['hidden_size']} -> tanh -> dropout0.1 -> logits (E14)",
        "include_test": False, "smoke": int(args.smoke) if args.smoke else None, "model_spec": spec,
        "model_manifest": common.model_manifest(common.ARMS[arm]["model"]),
        "arm_runtime_config": arm_runtime_config(arm), "head_init_sha256": head_init_hash,
        "tokenizer_class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
        "data_provenance": e1_train.bundle_provenance(bundle, common.ROOT), "task_metadata": e1_train.task_metadata(bundle),
        "train_count": len(train_split), "validation_count": len(valid_split), "length_stats": lengths.to_dict(),
        "runtime": e1_train._runtime_metadata(device, precision) if hasattr(e1_train, "_runtime_metadata") else None,
        "started_at": common.now(), "argv": sys.argv, "pid": os.getpid(),
    }
    write_json(out_dir / "resolved_config.json", meta)
    print(f"[{arm}/{task_id}/lr{lr:g}/seed{seed}] steps {total_steps} warmup {warmup} micro {micro} "
          f"non-emb {spec['encoder_non_embedding_parameters']/1e6:.1f}M trunc>{common.MAX_LENGTH}: {lengths.to_dict()}", flush=True)
    best_score, best_epoch, best_metrics, best_preds = -math.inf, 0, {}, []
    curves: list[dict] = []
    global_step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_ex = 0
        t_ep = time.time()
        n_batches = len(train_loader) if not args.smoke else min(len(train_loader), int(args.smoke))
        for bi, batch in enumerate(train_loader):
            if bi >= n_batches:
                break
            moved = e1_train._move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"])
                loss = e1_train.compute_loss(logits, moved["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                ws = (bi // accum) * accum
                we = min(ws + accum, n_batches)
                wex = sum(min(micro, len(train_loader.dataset) - i * micro) for i in range(ws, we))
                scaled = loss * (len(batch["labels"]) / max(wex, 1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {bi}")
            scaled.backward()
            total_loss += float(loss.detach().cpu()) * len(batch["labels"])
            total_ex += len(batch["labels"])
            if (bi + 1) % accum == 0 or bi + 1 == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        if args.smoke:
            from torch.utils.data import DataLoader, Subset
            n_val = min(len(valid_loader.dataset), int(args.smoke) * micro)
            valid_loader = DataLoader(Subset(valid_loader.dataset, range(n_val)), batch_size=micro, shuffle=False,
                                      collate_fn=valid_loader.collate_fn)
        val_metrics, val_preds = e1_train.run_validation(model, valid_loader, device, precision, task_id, bundle)
        score = e1_train.selection_score(val_metrics)
        row = {"epoch": epoch, "global_step": global_step, "train_loss": total_loss / max(total_ex, 1), "selection_f1": score,
               "validation": val_metrics, "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - t_ep}
        curves.append(row)
        write_json(out_dir / "epoch_curves.json", {"epochs": curves})
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_metrics, best_preds = score, epoch, val_metrics, val_preds
            tmp = out_dir / "best_checkpoint.pt.tmp"
            torch.save({"format": "bidirlm-vi-mntp-stage1-model-only-v1", "model": model.state_dict(), "epoch": epoch,
                        "global_step": global_step,
                        "metadata": {k: meta[k] for k in ("protocol", "arm", "task_id", "seed", "learning_rate")}}, tmp)
            tmp.replace(out_dir / "best_checkpoint.pt")
            write_json(out_dir / "best_validation_metrics.json", {"epoch": epoch, **best_metrics})
            write_json(out_dir / "validation_predictions.json", {"epoch": epoch, "rows": best_preds})
        print(f"[{arm}/{task_id}/lr{lr:g}/seed{seed}] epoch {epoch:2d} loss {row['train_loss']:.4f} "
              f"val {100*score:.2f} best {100*best_score:.2f}@{best_epoch} ({row['epoch_seconds']:.0f}s){' *' if improved else ''}",
              flush=True)
    elapsed = time.time() - started
    ckpt = out_dir / "best_checkpoint.pt"
    meta.update({"best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics,
                 "elapsed_seconds": elapsed, "finished_at": common.now(), "checkpoint_sha256": common.sha256_file(ckpt),
                 "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None})
    del optimizer, scheduler
    re_model, _ = common.build_model(arm, num_classes=e1_train.task_num_classes(bundle), num_aspects=num_aspects,
                                     aspect_class_counts=counts)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = re_model.load_state_dict(state["model"], strict=True)
    re_model.to(device)
    re_metrics, re_rows = e1_train.run_validation(re_model, valid_loader, device, precision, task_id, bundle)
    saved = e1_train.read_json(out_dir / "validation_predictions.json")["rows"]
    file_f1 = e1_train.selection_score(e1_train.evaluate_predictions(task_id, bundle, [r["gold"] for r in saved],
                                                                    [r["prediction"] for r in saved], split="validation"))
    meta["postrun_checks"] = {
        "checkpoint_epoch": state.get("epoch"), "reload_selection_f1": e1_train.selection_score(re_metrics),
        "reload_matches_best": abs(e1_train.selection_score(re_metrics) - best_score) < 1e-9,
        "prediction_file_selection_f1": file_f1, "prediction_file_matches_best": abs(file_f1 - best_score) < 1e-9,
        "reload_predictions_identical": [r["prediction"] for r in re_rows] == [r["prediction"] for r in saved],
        "ids_stable_and_complete": [r["id"] for r in saved] == [r["id"] for r in re_rows] and len(saved) == len(valid_loader.dataset),
        "strict_load_missing": list(missing), "strict_load_unexpected": list(unexpected),
    }
    ok = all(meta["postrun_checks"][k] for k in ("reload_matches_best", "prediction_file_matches_best",
                                                 "reload_predictions_identical", "ids_stable_and_complete"))
    meta["status"] = "completed" if ok else "completed_with_check_failure"
    write_json(out_dir / "metadata.json", meta)
    write_json(out_dir / "status.json", {"status": meta["status"], "arm": arm, "task_id": task_id, "learning_rate": lr,
                                         "seed": seed, "best_epoch": best_epoch, "best_selection_f1": best_score,
                                         "elapsed_seconds": elapsed, "finished_at": common.now(), "pid": os.getpid(),
                                         "postrun_checks": meta["postrun_checks"]})
    print(f"[{arm}/{task_id}/lr{lr:g}/seed{seed}] done best {100*best_score:.2f}@{best_epoch} in {elapsed/60:.1f} min; checks ok={ok}",
          flush=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(common.ARMS))
    ap.add_argument("--task", required=True, choices=common.TASKS)
    ap.add_argument("--lr", required=True, type=float)
    ap.add_argument("--seed", type=int, default=common.SEED)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    if args.micro_batch <= 0:
        args.micro_batch = common.ARMS[args.arm]["micro_batch"]
    out_dir = Path(args.out_dir) if args.out_dir else common.arm_dir(args.arm, args.task, float(args.lr), int(args.seed))
    try:
        run(args)
        return 0
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        write_json(out_dir / "status.json", {"status": "failed", "arm": args.arm, "task_id": args.task,
                                             "learning_rate": args.lr, "seed": args.seed, "pid": os.getpid(),
                                             "finished_at": common.now(), "error": repr(error),
                                             "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
