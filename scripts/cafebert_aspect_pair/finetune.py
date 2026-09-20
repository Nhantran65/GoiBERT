"""Fine-tune one aspect-pair arm (A: aspect id, B: aspect description) on absa_restaurant, validation-only.

Optimisation unit is a *review group*: <= 32 reviews drawn by the same seeded review sampler as the reference
run, expanded to all 12 pairs each (<= 384 pairs), split into micro-batches of --micro pairs.  Every micro-batch
contributes CE_mean * n_pairs_micro / n_pairs_group, so the accumulated gradient equals the mean CE over all
pairs of the group (the reference's mean over 32 x 12 slot terms).  One clip / optimizer step / scheduler
step per group: 220 steps per epoch, 6600 in total, warmup 660.

Outputs (artifacts/cafebert_aspect_pair/runs/<arm>_absa_restaurant_seed42/):
    resolved_config.json (incl. locked aspect mapping), epoch_curves.json, best_checkpoint.pt (model state only),
    best_validation_metrics.json, validation_predictions.json (regrouped 12-slot rows), metadata.json, status.json.
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
from common import (  # noqa: E402
    ARMS, ATTENTION_IMPLEMENTATION, EFFECTIVE_BATCH_REVIEWS, EPOCHS, GRADIENT_CLIP_NORM, LEARNING_RATE, MAX_LENGTH,
    NUM_ASPECTS, NUM_CLASSES, PRECISION, PROTOCOL, ROOT, SEED, TASK_ID, TRUNCATION, WARMUP_FRACTION, WEIGHT_DECAY, e1_train,
)

import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402


def run(args: argparse.Namespace) -> dict[str, Any]:
    arm = args.arm
    output_dir = Path(args.out_dir) if args.out_dir else common.run_dir(arm)
    existing = [n for n in ("epoch_curves.json", "best_checkpoint.pt", "metadata.json") if (output_dir / n).exists()]
    if existing and not args.smoke:
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir} has {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_unix = time.time()
    common.write_json(output_dir / "status.json", {"status": "running", "arm": arm, "pid": os.getpid(), "device": args.device, "started_at": common.now()})

    epochs = 1 if args.smoke else EPOCHS
    micro = int(args.micro)
    if micro <= 0:
        raise ValueError("--micro must be positive")

    expected_source_hash = common.source_hash()
    e1_train.seed_everything(SEED)
    device = e1_train.resolve_device(args.device)
    precision = e1_train.resolve_precision(PRECISION, device)
    contract = common.task_contract()
    bundle = contract["bundle"]
    tokenizer, tokenizer_manifest = common.load_tokenizer()
    train_reviews = common.reviews_of(contract["train"])
    valid_reviews = common.reviews_of(contract["validation"])
    train_pairs = common.encode_pairs(tokenizer, train_reviews, arm)
    valid_pairs = common.encode_pairs(tokenizer, valid_reviews, arm)
    truncated = {"train": sum(p.truncated for g in train_pairs for p in g), "validation": sum(p.truncated for g in valid_pairs for p in g)}
    if truncated["train"] or truncated["validation"]:
        raise RuntimeError(f"pair truncation would alter review text: {truncated}; plan requires verbatim reviews")
    lengths = [len(p.input_ids) for g in train_pairs for p in g]
    length_stats = {"pairs": len(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths),
                    "p50": sorted(lengths)[len(lengths) // 2], "p95": sorted(lengths)[int(0.95 * len(lengths))]}

    model, model_spec = common.build_model()
    model.to(device)
    optimizer = e1_train.make_optimizer(model, LEARNING_RATE)
    loader = common.review_loader(train_pairs, reviews_per_group=EFFECTIVE_BATCH_REVIEWS, shuffle=True, seed=SEED)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = round(total_steps * WARMUP_FRACTION)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if not args.smoke and (steps_per_epoch != 220 or total_steps != 6600 or warmup_steps != 660):
        raise RuntimeError(f"update budget mismatch: {steps_per_epoch}/epoch, {total_steps} total, warmup {warmup_steps}")

    metadata: dict[str, Any] = {
        "protocol": PROTOCOL, "arm": arm, "task_id": TASK_ID, "seed": SEED, "stage": "validation_only",
        "init": "cafebert_pretrained", "learning_rate": LEARNING_RATE, "epochs": epochs,
        "effective_batch_reviews": EFFECTIVE_BATCH_REVIEWS, "effective_batch_pairs": EFFECTIVE_BATCH_REVIEWS * NUM_ASPECTS,
        "micro_batch_pairs": micro, "validation_micro_batch_pairs": int(args.eval_micro), "max_length": MAX_LENGTH, "truncation": TRUNCATION,
        "precision": precision, "attention_implementation": ATTENTION_IMPLEMENTATION, "weight_decay": WEIGHT_DECAY,
        "betas": [0.9, 0.999], "adamw_epsilon": 1e-8, "warmup_fraction": WARMUP_FRACTION, "warmup_steps": warmup_steps,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "loss": "cross_entropy over 4 classes per pair (absent 0 included); micro-batch weight n_pairs_micro / n_pairs_group; one optimizer step per review group",
        "pair_input": "tokenizer(review, aspect_text) -> <s> review </s></s> aspect </s>",
        "aspect_mapping": common.mapping_table(arm),
        "label_codes": "targets are the loader's integer codes gold[slot] in {0,1,2,3}; 0 = absent; never recoded",
        "selection": "validation pair_micro_f1 (selection_f1) after regrouping pair argmax by (id, slot); best epoch, ties -> earlier",
        "include_test": False, "smoke": int(args.smoke) if args.smoke else None,
        "source_hash": expected_source_hash, "model": model_spec, "tokenizer": tokenizer_manifest,
        "data_provenance": e1_train.bundle_provenance(bundle, ROOT), "task_metadata": e1_train.task_metadata(bundle),
        "train_reviews": len(train_reviews), "train_pairs": sum(len(g) for g in train_pairs),
        "validation_reviews": len(valid_reviews), "validation_pairs": sum(len(g) for g in valid_pairs),
        "truncated_pairs": truncated, "train_pair_length_stats": length_stats,
        "runtime": common.runtime(device), "gpu_snapshot_at_start": common.gpu_snapshot(),
        "started_at": common.now(), "started_at_unix": started_unix, "argv": list(sys.argv), "pid": os.getpid(),
    }
    common.write_json(output_dir / "resolved_config.json", metadata)
    print(f"[{arm}] start lr={LEARNING_RATE} epochs={epochs} groups/epoch={steps_per_epoch} micro={micro} steps={total_steps} warmup={warmup_steps} "
          f"train_pairs={metadata['train_pairs']} val_pairs={metadata['validation_pairs']} device={device} head_init={model_spec['head_init_sha256'][:12]}", flush=True)

    best_score, best_epoch, best_metrics, best_predictions = -math.inf, 0, {}, []
    curves: list[dict[str, Any]] = []
    global_step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, pair_count, group_count = 0.0, 0, 0
        order_digest = hashlib.sha256()
        epoch_started = time.time()
        n_groups = steps_per_epoch if not args.smoke else min(steps_per_epoch, int(args.smoke))
        for group_index, group in enumerate(loader):
            if group_index >= n_groups:
                break
            n_group = len(group)
            seen_ids = [p.id for p in group[::NUM_ASPECTS]]
            order_digest.update(("\n".join(seen_ids) + "\n").encode("utf-8"))
            if n_group % NUM_ASPECTS or n_group > EFFECTIVE_BATCH_REVIEWS * NUM_ASPECTS:
                raise RuntimeError(f"group of {n_group} pairs is not a whole number of <= 32 reviews")
            for chunk in common.iter_microbatches(group, micro):
                moved = common.pad_microbatch(tokenizer, chunk, device)
                with e1_train.make_autocast(precision, device):
                    logits = model(moved["input_ids"], moved["attention_mask"])
                    if logits.shape != (len(chunk), NUM_CLASSES):
                        raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)}")
                    ce = nn.functional.cross_entropy(logits.float(), moved["labels"])
                    scaled = ce * (len(chunk) / n_group)
                if not torch.isfinite(ce):
                    raise FloatingPointError(f"non-finite loss at epoch={epoch}, group={group_index}")
                scaled.backward()
                loss_sum += float(ce.detach()) * len(chunk)
                pair_count += len(chunk)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM, error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            group_count += 1
            if args.log_every and group_count % int(args.log_every) == 0:
                elapsed = time.time() - epoch_started
                print(f"[{arm}] epoch {epoch} group {group_count}/{n_groups} loss {loss_sum / max(pair_count, 1):.4f} "
                      f"{elapsed / group_count:.2f}s/group eta_epoch {elapsed / group_count * (n_groups - group_count) / 60:.1f} min", flush=True)
        train_seconds = time.time() - epoch_started
        if args.smoke:
            n_val = min(len(valid_reviews), int(args.smoke) * 4)
            val_reviews_used, val_pairs_used = valid_reviews[:n_val], valid_pairs[:n_val]
        else:
            val_reviews_used, val_pairs_used = valid_reviews, valid_pairs
        val_started = time.time()
        val_metrics, val_predictions = common.run_validation(model, tokenizer, val_reviews_used, val_pairs_used, bundle, device, precision, micro=int(args.eval_micro))
        score = e1_train.selection_score(val_metrics)
        row = {
            "epoch": epoch, "global_step": global_step, "train_loss": loss_sum / max(pair_count, 1), "train_pairs": pair_count,
            "groups": group_count, "review_order_sha256": order_digest.hexdigest(), "selection_f1": score, "validation": val_metrics,
            "learning_rate": scheduler.get_last_lr()[0], "train_seconds": train_seconds, "validation_seconds": time.time() - val_started,
            "epoch_seconds": time.time() - epoch_started,
        }
        curves.append(row)
        common.write_json(output_dir / "epoch_curves.json", {"epochs": curves})
        improved = score > best_score
        if improved:
            best_score, best_epoch, best_metrics, best_predictions = score, epoch, val_metrics, val_predictions
            common.save_model_checkpoint(output_dir / "best_checkpoint.pt", model, epoch=epoch, global_step=global_step, metadata=metadata)
            common.write_json(output_dir / "best_validation_metrics.json", {"epoch": epoch, **best_metrics})
            common.write_json(output_dir / "validation_predictions.json", {"epoch": epoch, "regrouped_from": "per-pair argmax by (id, slot)", "rows": best_predictions})
        print(f"[{arm}] epoch {epoch:2d} loss {row['train_loss']:.4f} val_f1 {100*score:.2f} best {100*best_score:.2f}@{best_epoch} "
              f"(train {train_seconds:.0f}s, val {row['validation_seconds']:.0f}s, order {row['review_order_sha256'][:8]}){' *' if improved else ''}", flush=True)

    elapsed = time.time() - started_unix
    checkpoint = output_dir / "best_checkpoint.pt"
    metadata.update({
        "best_epoch": best_epoch, "best_selection_f1": best_score, "best_validation_metrics": best_metrics,
        "elapsed_seconds": elapsed, "finished_at": common.now(),
        "review_order_sha256_all_epochs": hashlib.sha256("".join(r["review_order_sha256"] for r in curves).encode("ascii")).hexdigest(),
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "checkpoint_sha256": common.sha256_file(checkpoint), "checkpoint_format": "model state_dict only (not resumable)",
    })

    # ---- post-run checks: reload best checkpoint -> validation F1 must match; predictions file -> same F1
    del optimizer, scheduler
    model.cpu(); del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reload_model, _ = common.build_model()
    reload_state = common.load_model_checkpoint(checkpoint, reload_model)
    reload_model.to(device)
    reload_metrics, reload_rows = common.run_validation(reload_model, tokenizer, val_reviews_used, val_pairs_used, bundle, device, precision, micro=int(args.eval_micro))
    reload_f1 = e1_train.selection_score(reload_metrics)
    saved = common.read_json(output_dir / "validation_predictions.json")["rows"]
    pred_file_f1 = common.rows_f1(bundle, saved)
    same_rows = [r["prediction"] for r in reload_rows] == [r["prediction"] for r in saved]
    gold_ok = [r["gold"] for r in saved] == [list(r.gold) for r in val_reviews_used] and [r["id"] for r in saved] == [r.id for r in val_reviews_used]
    metadata["postrun_checks"] = {
        "checkpoint_epoch": reload_state.get("epoch"), "best_epoch": best_epoch,
        "reload_selection_f1": reload_f1, "best_selection_f1": best_score, "reload_matches_best": abs(reload_f1 - best_score) < 1e-9,
        "prediction_file_selection_f1": pred_file_f1, "prediction_file_matches_best": abs(pred_file_f1 - best_score) < 1e-9,
        "reload_predictions_identical": same_rows, "prediction_file_ids_and_gold_match_loader": gold_ok,
    }
    ok = all(metadata["postrun_checks"][k] for k in ("reload_matches_best", "prediction_file_matches_best", "reload_predictions_identical", "prediction_file_ids_and_gold_match_loader"))
    metadata["status"] = "completed" if ok else "completed_with_check_failure"
    common.write_json(output_dir / "metadata.json", metadata)
    common.write_json(output_dir / "status.json", {"status": metadata["status"], "arm": arm, "best_epoch": best_epoch, "best_selection_f1": best_score,
                                                   "elapsed_seconds": elapsed, "finished_at": common.now(), "pid": os.getpid(), "device": args.device,
                                                   "postrun_checks": metadata["postrun_checks"]})
    print(f"[{arm}] done best {100*best_score:.2f}@{best_epoch} elapsed {elapsed/60:.1f} min; postrun_checks={metadata['postrun_checks']}", flush=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro", type=int, required=True, help="training micro-batch in pairs (same for A and B; fixed from the preflight profile)")
    parser.add_argument("--eval-micro", type=int, default=128, help="validation micro-batch in pairs (inference only)")
    parser.add_argument("--out-dir", default=None, help="override output directory (used for smoke runs)")
    parser.add_argument("--smoke", type=int, default=0, help="debug: 1 epoch, N review groups, 4N validation reviews")
    parser.add_argument("--log-every", type=int, default=20, help="print progress every N groups")
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
