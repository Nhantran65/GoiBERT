#!/usr/bin/env python3
"""Stage 2b: fine-tune a model on one task (validation only).

GoiBERT student with CafeBERT knowledge distillation and LLRD (the paper's
recipe; learning rates are in ``configs/paper.json``):

    python scripts/finetune.py --task absa_restaurant --backbone TwoheartsTechnologies/GoiBERT-M \\
        --lr 1e-4 --seed 42 --soft-labels runs/teacher/absa_restaurant/soft_labels.npz \\
        --out runs/finetune/large/absa_restaurant/seed_42

BamiBERT baseline (plain AdamW, cross-entropy):

    python scripts/finetune.py --task absa_restaurant --kind bamibert --backbone Qualcomm-AI-Research/BamiBERT \\
        --optimizer adamw --micro-batch-size 16 --lr 5e-5 --seed 42 --out runs/finetune/bamibert/absa_restaurant/seed_42

Ablations: omit ``--soft-labels`` (no task KD) and/or use ``--optimizer adamw``
(no LLRD); a ``--steps 0`` backbone from ``scripts/distill.py`` gives the
"no general distillation" setting.
"""

import argparse
from pathlib import Path

from goibert.data import TASKS
from goibert.finetune import RunConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--backbone", required=True, help="Hub ID or local directory")
    parser.add_argument("--kind", choices=("student", "bamibert"), default="student")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=("llrd", "adamw"), default="llrd")
    parser.add_argument("--soft-labels", type=Path, help="teacher soft labels from scripts/train_teacher.py")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--micro-batch-size", type=int, default=32, help="gradient accumulation up to a batch of 32")
    parser.add_argument("--limit", type=int, help="smoke test: use the first N training examples")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = run(RunConfig(task=args.task, kind=args.kind, backbone=args.backbone, out=args.out, learning_rate=args.lr,
                         seed=args.seed, optimizer=args.optimizer, soft_labels=args.soft_labels, epochs=args.epochs,
                         micro_batch_size=args.micro_batch_size, limit=args.limit, device=args.device,
                         precision=args.precision))
    print({k: meta[k] for k in ("status", "best_epoch", "best_selection_f1")})


if __name__ == "__main__":
    main()
