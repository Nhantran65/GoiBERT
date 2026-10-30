#!/usr/bin/env python3
"""Stage 1: general distillation of a GoiBERT backbone from BamiBERT.

Example (GoiBERT-large, as in the paper):

    python scripts/distill.py --variant large --corpus data/corpus/mlm_mix.txt --out runs/distill/large

``--steps 0`` writes the PCA-initialised student without distillation (the
"no general distillation" ablation).  ``--resume`` continues from the latest
checkpoint in ``--out`` with the same data order and RNG state.
"""

import argparse
from pathlib import Path

from goibert.distill import VARIANTS, distill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="one paragraph per line, blank line between documents")
    parser.add_argument("--teacher", default="Qualcomm-AI-Research/BamiBERT", help="Hub ID or local directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    final = distill(args.variant, args.corpus, args.teacher, args.out, steps=args.steps, resume=args.resume,
                    checkpoint_every=args.checkpoint_every, device_name=args.device)
    print(f"backbone written to {final}")


if __name__ == "__main__":
    main()
