#!/usr/bin/env python3
"""Train CafeBERT on 4/5 of a task's train split (frozen sealion recipe) and write logits for the held-out fold."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from scripts.sealion_benchmark import common as SB  # noqa: E402
from scripts.sealion_benchmark import train as SBT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=C.TASKS, required=True)
    ap.add_argument("--fold", type=int, choices=range(C.N_FOLDS), required=True)
    ap.add_argument("--smoke", type=int, default=0, help="smoke tests only: batches in a single epoch")
    ap.add_argument("--out", type=Path, default=None, help="smoke tests only")
    args = ap.parse_args()
    out = args.out or C.ART / "teacher" / args.task / f"fold_{args.fold}"

    base = SB.task_contract(args.task)
    bundle = C.fold_bundle(base["bundle"], args.fold)
    contract = dict(base, bundle=bundle, train=bundle.splits["train"])
    SB.task_contract = lambda task_id, include_test=False: contract  # consumed inside train_one
    meta = SBT.train_one("cafebert", args.task, 2e-5, device_name="cuda:0", micro_batch_size=32, smoke=args.smoke, out_dir=out)
    if meta["status"] != "completed":
        raise RuntimeError(f"teacher fold failed checks: {meta['postrun_checks']}")

    tokenizer, _ = SB.load_cafe_tokenizer()
    model, _ = SB.build_model("cafebert", contract)
    SB.load_model_checkpoint(out / "best_checkpoint.pt", model)
    device = torch.device("cuda:0")
    model.to(device).eval()
    held = bundle.splits["oof"]
    loader = C.e1_train._make_loader(tokenizer, held, max_length=SB.MAX_LENGTH, batch_size=64, shuffle=False, seed=42,
                                     labels=bundle.labels_for_model("oof"), gold_labels=held.labels)
    ids, logits = [], []
    with torch.inference_mode():
        for batch in loader:
            moved = C.e1_train._move_batch(batch, device)
            with C.e1_train.make_autocast("bf16", device):
                out_logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
            logits.append(C.flat_logits(out_logits.float(), len(batch["ids"])).cpu().numpy())
            ids.extend(batch["ids"])
    logits = np.concatenate(logits)
    counts = list(C.label_widths(bundle))
    np.savez(out / "oof_logits.npz", ids=np.array(ids), logits=logits, aspect_class_counts=np.array(counts))
    C.write_json(out / "oof_summary.json", {"task": args.task, "fold": args.fold, "train_count": len(contract["train"]),
                                            "oof_count": len(ids), "logit_width": int(logits.shape[1]),
                                            "best_epoch": meta["best_epoch"], "best_selection_f1": meta["best_selection_f1"],
                                            "elapsed_seconds": meta["elapsed_seconds"]})
    print(f"[teacher {args.task} fold {args.fold}] oof {len(ids)} x {logits.shape[1]} written", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
