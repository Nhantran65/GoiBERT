#!/usr/bin/env python3
"""GoiTurn-S fine-tune with the E7 recipe (E1 loop + LLRD gamma 0.8); arm `ce` = gold only, arm `kd` adds
out-of-fold CafeBERT soft labels: loss = a*CE + (1-a)*T^2*KL(teacher_T || student_T) per slot."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from goibert.modeling import build_classifier  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402

T = C.e1_train
BACKBONE = C.ROOT / "models" / "goiturn-s-gd"
EPOCHS, MICRO = 30, 32


def kd_term(student_flat, teacher_flat, segments, temperature):
    losses = []
    for a, b in segments:
        s = F.log_softmax(student_flat[:, a:b] / temperature, -1)
        t = F.softmax(teacher_flat[:, a:b] / temperature, -1)
        losses.append(F.kl_div(s, t, reduction="batchmean") * temperature ** 2)
    return torch.stack(losses).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=C.TASKS, required=True)
    ap.add_argument("--arm", choices=("ce", "kd"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="smoke tests only")
    ap.add_argument("--soft", type=Path, default=None, help="smoke tests only; default artifacts soft labels")
    ap.add_argument("--out", type=Path, default=None, help="smoke tests only")
    args = ap.parse_args()
    out = args.out or C.ART / "student" / args.arm / args.task / f"seed_{args.seed}"
    if (out / "metadata.json").exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    C.write_json(out / "status.json", {"status": "running", "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})

    # Same call order as goibert.train.train_trial so the CE arm mirrors E7.
    T.seed_everything(args.seed)
    device = T.resolve_device("cuda:0")
    precision = T.resolve_precision("bf16", device)
    bundle = T.load_task_bundle(args.task, C.ROOT, include_test=False)
    train_split, valid_split = T.split_field(bundle, "train"), T.split_field(bundle, "validation")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True, use_fast=True)
    lengths = T.inspect_train_lengths(tokenizer, list(T.split_texts(train_split)))
    num_aspects, counts = T.task_num_aspects(bundle), T.task_aspect_class_counts(bundle)
    model, spec = build_classifier(BACKBONE, num_classes=T.task_num_classes(bundle), num_aspects=num_aspects,
                                   aspect_class_counts=counts if num_aspects else None, local_files_only=True,
                                   attention_implementation="sdpa")
    model.to(device)
    lr = C.STUDENT_LR[args.task]
    optimizer = e7_optim.build_layerwise_optimizer(model, lr, gamma=0.8)
    train_loader = T._make_loader(tokenizer, train_split, max_length=lengths.max_length, batch_size=MICRO, shuffle=True,
                                  seed=args.seed, labels=T.model_labels(bundle, "train", train_split),
                                  gold_labels=T.split_labels(train_split))
    valid_loader = T._make_loader(tokenizer, valid_split, max_length=lengths.max_length, batch_size=MICRO, shuffle=False,
                                  seed=args.seed, labels=T.model_labels(bundle, "validation", valid_split),
                                  gold_labels=T.split_labels(valid_split))
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    segments = C.slot_segments(C.label_widths(bundle))

    teacher = None
    if args.arm == "kd":
        z = np.load(args.soft or C.ART / "soft" / f"{args.task}.npz", allow_pickle=False)
        teacher = {str(i): torch.tensor(row, dtype=torch.float32) for i, row in zip(z["ids"], z["logits"])}
        missing = set(train_split.ids) - set(teacher)
        if missing:
            raise RuntimeError(f"{len(missing)} train ids without teacher logits")

    config = {"task": args.task, "arm": args.arm, "seed": args.seed, "learning_rate": lr, "llrd_gamma": 0.8,
              "epochs": args.epochs, "micro_batch": MICRO, "max_length": lengths.max_length, "precision": precision,
              "warmup_steps": warmup_steps, "total_steps": total_steps, "backbone": str(BACKBONE.relative_to(C.ROOT)),
              "kd_alpha": C.KD_ALPHA if teacher else None, "kd_temperature": C.KD_TEMPERATURE if teacher else None,
              "include_test": False, "model_spec": getattr(spec, "to_dict", lambda: str(spec))()}
    C.write_json(out / "resolved_config.json", config)

    best, best_epoch, curves = -math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        t0, loss_sum, kd_sum, n = time.time(), 0.0, 0.0, 0
        for batch in train_loader:
            moved = T._move_batch(batch, device)
            with T.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                ce = T.compute_loss(logits, moved["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
            loss = ce
            if teacher is not None:
                tl = torch.stack([teacher[str(i)] for i in batch["ids"]]).to(device)
                kd = kd_term(C.flat_logits(logits.float(), len(batch["ids"])), tl, segments, C.KD_TEMPERATURE)
                loss = C.KD_ALPHA * ce + (1 - C.KD_ALPHA) * kd
                kd_sum += float(kd.detach()) * len(batch["ids"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(batch["ids"]); n += len(batch["ids"])
        metrics, rows = T.run_validation(model, valid_loader, device, precision, args.task, bundle)
        score = T.selection_score(metrics)
        curves.append({"epoch": epoch, "train_loss": loss_sum / n, "train_kd": kd_sum / n if teacher else None,
                       "selection_f1": score, "seconds": time.time() - t0})
        C.write_json(out / "epoch_curves.json", {"epochs": curves})
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, out / "best_checkpoint.pt")
            C.write_json(out / "validation_predictions.json", {"epoch": epoch, "rows": rows})
            C.write_json(out / "best_validation_metrics.json", {"epoch": epoch, **metrics})
        print(f"[{args.arm}/{args.task}/seed{args.seed}] epoch {epoch:2d} loss {loss_sum/n:.4f} val {100*score:.2f} "
              f"best {100*best:.2f}@{best_epoch} ({time.time()-t0:.0f}s)", flush=True)

    reload_model, _ = build_classifier(BACKBONE, num_classes=T.task_num_classes(bundle), num_aspects=num_aspects,
                                       aspect_class_counts=counts if num_aspects else None, local_files_only=True,
                                       attention_implementation="sdpa")
    reload_model.load_state_dict(torch.load(out / "best_checkpoint.pt", map_location="cpu")["model"], strict=True)
    reload_model.to(device)
    r_metrics, r_rows = T.run_validation(reload_model, valid_loader, device, precision, args.task, bundle)
    saved = C.json.loads((out / "validation_predictions.json").read_text())["rows"]
    file_f1 = T.selection_score(T.evaluate_predictions(args.task, bundle, [r["gold"] for r in saved],
                                                       [r["prediction"] for r in saved], split="validation"))
    checks = {"reload_matches_best": abs(T.selection_score(r_metrics) - best) < 1e-9,
              "prediction_file_matches_best": abs(file_f1 - best) < 1e-9,
              "reload_predictions_identical": [r["prediction"] for r in r_rows] == [r["prediction"] for r in saved],
              "ids_stable_and_complete": [r["id"] for r in r_rows] == [r["id"] for r in saved] == list(T.split_ids(valid_split))}
    status = "completed" if all(checks.values()) else "completed_with_check_failure"
    meta = {**config, "status": status, "best_epoch": best_epoch, "best_selection_f1": best, "postrun_checks": checks,
            "elapsed_seconds": time.time() - started}
    C.write_json(out / "metadata.json", meta)
    C.write_json(out / "status.json", {"status": status, "best_selection_f1": best, "best_epoch": best_epoch})
    print(f"[{args.arm}/{args.task}/seed{args.seed}] done best {100*best:.2f}@{best_epoch} checks {checks}", flush=True)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
