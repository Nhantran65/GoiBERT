#!/usr/bin/env python3
"""Stage 2a: out-of-fold CafeBERT soft labels for one task.

The training split is divided into five folds (seeded permutation).  For each
fold, CafeBERT is fine-tuned on the other four folds (30 epochs, AdamW,
learning rate 2e-5, seed 42, best epoch on validation) and writes its logits
for the held-out fold.  The five folds are then merged into one soft-label
file in training order, so no example is labelled by a teacher that saw it.

    python scripts/train_teacher.py --task absa_restaurant --out runs/teacher/absa_restaurant

Use ``--fold k`` to run one fold (for example, one per GPU) and
``--merge-only`` to merge folds that are already finished.
"""

import argparse
from pathlib import Path

from goibert.data import TASKS
from goibert.finetune import N_FOLDS, RunConfig, merge_soft_labels, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(N_FOLDS), help="run only this fold")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))
    if not args.merge_only:
        for fold in folds:
            meta = run(RunConfig(task=args.task, kind="cafebert", out=args.out / f"fold_{fold}", learning_rate=2e-5,
                                 seed=42, optimizer="adamw", fold=fold, epochs=args.epochs, device=args.device))
            if meta["status"] != "completed":
                raise RuntimeError(f"Fold {fold} failed its checks: {meta['checks']}")
    if args.fold is None or args.merge_only:
        result = merge_soft_labels(args.task, [args.out / f"fold_{k}" for k in range(N_FOLDS)], args.out / "soft_labels.npz")
        print(result)


if __name__ == "__main__":
    main()
