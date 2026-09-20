#!/usr/bin/env python3
"""Fine-tune an exported encoder (arm M or G, or the original BamiBERT for smoke tests) with the frozen E14 recipe (plan §6).

Restaurant / ViSFD lr 5e-5, Topic lr 1e-5; seed 42; 30 epochs; effective batch 32; max_len 256; AdamW wd 0.01 betas
(0.9, 0.999) eps 1e-8; 10 % warmup + linear decay; clip 1.0; bf16 autocast + SDPA; CE; CLS -> dropout .1 -> dense 768 ->
tanh -> dropout .1 -> logits (goibert_e14.modeling.ClassificationHead); best epoch by the E14 selection metric on
validation; test never loaded.  M and G share head init (same RNG consumption after seed_everything) and data order
(E1 loader, shuffle seed 42); both are hashed into metadata so equality can be verified across arms.

Outputs artifacts/cafe_bami_repdistill_pilot/ft/<arm>/<task>/{resolved_config.json, epoch_curves.json, best_checkpoint.pt,
best_validation_metrics.json, validation_predictions.json, metadata.json, status.json}.
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
from goibert_e14.modeling import build_bamibert_classifier  # noqa: E402

TASK_LR = {"absa_restaurant": 5e-5, "vsfc_topic": 1e-5, "visfd": 5e-5}
EPOCHS = 30
EFFECTIVE_BATCH = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTN = "sdpa"


def encoder_dir(arm: str) -> Path:
    if arm == "O_smoke":
        return common.STUDENT_DIR
    return common.ART_DIR / "cpt" / arm / "hf_encoder"


def _hash_params(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _save(path: Path, model: torch.nn.Module, *, epoch: int, global_step: int, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"format": "repdistill-pilot-model-only-v1", "model": model.state_dict(), "epoch": epoch, "global_step": global_step,
                "metadata": {k: meta[k] for k in ("protocol", "arm", "task_id", "seed", "learning_rate", "source_hash") if k in meta}}, tmp)
    tmp.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    arm, task_id = args.arm, args.task
    out_dir = Path(args.out_dir) if args.out_dir else common.ART_DIR / "ft" / arm / task_id
    if any((out_dir / n).exists() for n in ("metadata.json", "best_checkpoint.pt")) and not args.smoke:
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    common.write_json(out_dir / "status.json", {"status": "running", "arm": arm, "task_id": task_id, "pid": os.getpid(), "device": args.device, "started_at": common.now()})
    epochs = 1 if args.smoke else EPOCHS
    micro = int(args.micro_batch)
    if EFFECTIVE_BATCH % micro:
        raise ValueError("micro batch must divide 32")
    accum = EFFECTIVE_BATCH // micro
    lr = TASK_LR[task_id]
    enc_dir = encoder_dir(arm)
    enc_hash = common.sha256_file(enc_dir / "model.safetensors")
    if arm in ("M", "G"):
        summ = common.read_json(common.ART_DIR / "cpt" / arm / "summary.json")
        if summ.get("status") != "completed" or summ["export"]["hf_encoder_sha256"] != enc_hash:
            raise RuntimeError(f"CPT export for arm {arm} is not completed/consistent")

    e1_train.seed_everything(common.SEED)
    device = e1_train.resolve_device(args.device)
    precision = e1_train.resolve_precision(PRECISION, device)
    bundle = e1_train.load_task_bundle(task_id, common.ROOT, include_test=False)
    train_split, valid_split = e1_train.split_field(bundle, "train"), e1_train.split_field(bundle, "validation")
    num_aspects = e1_train.task_num_aspects(bundle); counts = e1_train.task_aspect_class_counts(bundle)
    tokenizer = common.load_student_tokenizer()
    lengths = e1_train.inspect_train_lengths(tokenizer, list(e1_train.split_texts(train_split)))
    model, spec = build_bamibert_classifier(enc_dir, num_classes=e1_train.task_num_classes(bundle), num_aspects=num_aspects, aspect_class_counts=counts if num_aspects else None, attention_implementation=ATTN)
    head_init_hash = _hash_params(model.head)
    model.to(device)
    optimizer = e1_train.make_optimizer(model, lr)
    train_loader = e1_train._make_loader(tokenizer, train_split, max_length=MAX_LENGTH, batch_size=micro, shuffle=True, seed=common.SEED, labels=e1_train.model_labels(bundle, "train", train_split), gold_labels=e1_train.split_labels(train_split))
    valid_loader = e1_train._make_loader(tokenizer, valid_split, max_length=MAX_LENGTH, batch_size=micro, shuffle=False, seed=common.SEED, labels=e1_train.model_labels(bundle, "validation", valid_split), gold_labels=e1_train.split_labels(valid_split))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * epochs
    warmup = round(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    meta: dict[str, Any] = {
        "protocol": common.PROTOCOL, "stage": "finetune_validation_only", "arm": arm, "task_id": task_id, "seed": common.SEED, "learning_rate": lr, "epochs": epochs, "effective_batch_size": EFFECTIVE_BATCH,
        "micro_batch_size": micro, "gradient_accumulation_steps": accum, "max_length": MAX_LENGTH, "precision": precision, "attention_implementation": ATTN, "weight_decay": 0.01, "betas": [0.9, 0.999], "adamw_epsilon": 1e-8,
        "warmup_fraction": 0.1, "warmup_steps": warmup, "total_steps": total_steps, "gradient_clip_norm": 1.0, "loss": "plain_cross_entropy", "head": "CLS -> dropout0.1 -> dense768 -> tanh -> dropout0.1 -> logits (E14)",
        "no_nc_kd_llrd_composite": True, "include_test": False, "smoke": int(args.smoke) if args.smoke else None, "encoder_dir": str(enc_dir.relative_to(common.ROOT)), "encoder_sha256": enc_hash,
        "model": spec.to_dict(), "head_init_sha256": head_init_hash, "data_provenance": e1_train.bundle_provenance(bundle, common.ROOT), "task_metadata": e1_train.task_metadata(bundle),
        "train_count": len(train_split), "validation_count": len(valid_split), "num_aspects": num_aspects, "aspect_class_counts": list(counts), "source_hash": common.source_hash(),
        "runtime": common.runtime(device), "gpu_snapshot_at_start": common.gpu_snapshot(), "length_stats": lengths.to_dict(), "started_at": common.now(), "argv": sys.argv, "pid": os.getpid(),
    }
    common.write_json(out_dir / "resolved_config.json", meta)
    print(f"[{arm}/{task_id}] lr={lr} epochs={epochs} micro={micro} accum={accum} steps={total_steps} warmup={warmup} head_init={head_init_hash[:12]}", flush=True)

    best_score, best_epoch, best_metrics, best_preds = -math.inf, 0, {}, []
    curves: list[dict] = []; global_step = 0; order_hash = hashlib.sha256()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0; total_ex = 0; t_ep = time.time()
        n_batches = len(train_loader) if not args.smoke else min(len(train_loader), int(args.smoke))
        for bi, batch in enumerate(train_loader):
            if bi >= n_batches:
                break
            if epoch == 1:
                order_hash.update(",".join(str(i) for i in batch["ids"]).encode())
            moved = e1_train._move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                loss = e1_train.compute_loss(logits, moved["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                ws = (bi // accum) * accum; we = min(ws + accum, n_batches)
                wex = sum(min(micro, len(train_loader.dataset) - i * micro) for i in range(ws, we))
                scaled = loss * (len(batch["labels"]) / max(wex, 1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {bi}")
            scaled.backward()
            total_loss += float(loss.detach().cpu()) * len(batch["labels"]); total_ex += len(batch["labels"])
            if (bi + 1) % accum == 0 or bi + 1 == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
        if args.smoke:
            from torch.utils.data import DataLoader, Subset
            n_val = min(len(valid_loader.dataset), int(args.smoke) * micro)
            valid_loader = DataLoader(Subset(valid_loader.dataset, range(n_val)), batch_size=micro, shuffle=False, collate_fn=valid_loader.collate_fn)
        val_metrics, val_preds = e1_train.run_validation(model, valid_loader, device, precision, task_id, bundle)
        score = e1_train.selection_score(val_metrics)
        row = {"epoch": epoch, "global_step": global_step, "train_loss": total_loss / max(total_ex, 1), "selection_f1": score, "validation": val_metrics, "learning_rate": scheduler.get_last_lr()[0], "epoch_seconds": time.time() - t_ep}
        curves.append(row); common.write_json(out_dir / "epoch_curves.json", {"epochs": curves})
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_metrics, best_preds = score, epoch, val_metrics, val_preds
            _save(out_dir / "best_checkpoint.pt", model, epoch=epoch, global_step=global_step, meta=meta)
            common.write_json(out_dir / "best_validation_metrics.json", {"epoch": epoch, **best_metrics})
            common.write_json(out_dir / "validation_predictions.json", {"epoch": epoch, "rows": best_preds})
        print(f"[{arm}/{task_id}] epoch {epoch:2d} loss {row['train_loss']:.4f} val {100*score:.2f} best {100*best_score:.2f}@{best_epoch} ({row['epoch_seconds']:.0f}s){' *' if improved else ''}", flush=True)

    elapsed = time.time() - started
    ckpt = out_dir / "best_checkpoint.pt"
    meta.update({"best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics, "elapsed_seconds": elapsed, "finished_at": common.now(),
                 "train_order_epoch1_sha256": order_hash.hexdigest(), "checkpoint_sha256": common.sha256_file(ckpt),
                 "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None})
    # post-run: reload best checkpoint -> identical predictions / F1; F1 recomputed from the saved prediction file
    del optimizer, scheduler
    re_model, _ = build_bamibert_classifier(enc_dir, num_classes=e1_train.task_num_classes(bundle), num_aspects=num_aspects, aspect_class_counts=counts if num_aspects else None, attention_implementation=ATTN)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = re_model.load_state_dict(state["model"], strict=True)
    re_model.to(device)
    re_metrics, re_rows = e1_train.run_validation(re_model, valid_loader, device, precision, task_id, bundle)
    saved = common.read_json(out_dir / "validation_predictions.json")["rows"]
    file_f1 = e1_train.selection_score(e1_train.evaluate_predictions(task_id, bundle, [r["gold"] for r in saved], [r["prediction"] for r in saved], split="validation"))
    ids_ok = [r["id"] for r in saved] == [r["id"] for r in re_rows] and len(saved) == len(valid_loader.dataset)
    meta["postrun_checks"] = {"checkpoint_epoch": state.get("epoch"), "reload_selection_f1": e1_train.selection_score(re_metrics), "reload_matches_best": abs(e1_train.selection_score(re_metrics) - best_score) < 1e-9,
                              "prediction_file_selection_f1": file_f1, "prediction_file_matches_best": abs(file_f1 - best_score) < 1e-9, "reload_predictions_identical": [r["prediction"] for r in re_rows] == [r["prediction"] for r in saved],
                              "ids_stable_and_complete": ids_ok, "strict_load_missing": list(missing), "strict_load_unexpected": list(unexpected)}
    ok = all(meta["postrun_checks"][k] for k in ("reload_matches_best", "prediction_file_matches_best", "reload_predictions_identical", "ids_stable_and_complete"))
    meta["status"] = "completed" if ok else "completed_with_check_failure"
    common.write_json(out_dir / "metadata.json", meta)
    common.write_json(out_dir / "status.json", {"status": meta["status"], "arm": arm, "task_id": task_id, "best_epoch": best_epoch, "best_selection_f1": best_score, "elapsed_seconds": elapsed, "finished_at": common.now(), "postrun_checks": meta["postrun_checks"]})
    print(f"[{arm}/{task_id}] done best {100*best_score:.2f}@{best_epoch} in {elapsed/60:.1f} min; checks {meta['postrun_checks']}", flush=True)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("M", "G", "O_smoke"))
    ap.add_argument("--task", required=True, choices=sorted(TASK_LR))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", type=int, default=0, help="debug: 1 epoch, N train batches, N validation batches")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else common.ART_DIR / "ft" / args.arm / args.task
    try:
        run(args); return 0
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        common.write_json(out_dir / "status.json", {"status": "failed", "arm": args.arm, "task_id": args.task, "pid": os.getpid(), "finished_at": common.now(), "error": repr(error), "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
