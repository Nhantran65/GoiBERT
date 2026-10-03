#!/usr/bin/env python3
"""Extra BamiBERT seeds with the frozen E14 recipe at its locked LR; writes only under artifacts/cafe_oof_kd/bami."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "scripts"))
import run_e14 as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=C.TASKS, required=True)
    ap.add_argument("--seed", type=int, choices=(43, 44), required=True)
    args = ap.parse_args()
    out = C.ART / "bami" / args.task / f"seed_{args.seed}"
    if (out / "metadata.json").exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)  # _run_trial creates `out` itself and refuses an existing one
    E.SEED = args.seed  # read by seed_everything and both loaders inside _run_trial
    result = E._run_trial(args.task, C.BAMI_LR[args.task], "cuda:0", 32, out, E.source_hash())
    print(f"[bami/{args.task}/seed{args.seed}] best {100*result['best_selection_f1']:.2f}@{result['best_epoch']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
