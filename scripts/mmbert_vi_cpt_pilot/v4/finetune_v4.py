#!/usr/bin/env python3
"""v4 §4 one fine-tune cell: E14 recipe on top of O (pinned mmBERT) or one CPT arm's encoder export.

Wraps `scripts/mmbert_vi_cpt_pilot/finetune.py` (the v2 driver) with v4 paths and the `_NC_research_only` directory
suffix for the two arms that contain ViSoBERT data.  Recipe is unchanged: E14 CLS head, CE, 30 epochs, effective batch 32,
max length 256, bf16/SDPA, best epoch on validation, locked LRs (Restaurant 5e-5, ViSFD 5e-5, Topic 2e-5), no test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

V4_DIR = Path(__file__).resolve().parent
PILOT_DIR = V4_DIR.parent
sys.path.insert(0, str(V4_DIR))
import common_v4 as C4  # noqa: E402

import common as V1  # noqa: E402
V1.ART_DIR = C4.ART_V4                      # the v2 fine-tune driver reads paths from the v1 shared contract
sys.path.insert(0, str(PILOT_DIR))
import finetune as F  # noqa: E402


def arm_dir(arm: str) -> str:
    return arm + (C4.NC_SUFFIX if arm in C4.ARMS_NC else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("O",) + C4.ARMS)
    ap.add_argument("--task", required=True, choices=C4.common.FT_TASKS)
    ap.add_argument("--seed", type=int, default=C4.SEED)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    d = arm_dir(args.arm)
    encoder = "O" if args.arm == "O" else str(C4.ART_V4 / "cpt" / d / "hf_encoder")
    out_dir = C4.ART_V4 / "ft" / d / args.task / f"seed_{args.seed}"
    ns = argparse.Namespace(arm=args.arm, encoder=encoder, task=args.task, seed=args.seed, lr=args.lr,
                            device=args.device, micro_batch=args.micro_batch, out_dir=str(out_dir))
    meta = F.run(ns)
    print(f"[v4-ft] {args.arm}/{args.task}/seed{args.seed} -> {meta['status']} f1={meta.get('best_selection_f1')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
