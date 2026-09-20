"""Run the segment-and-read ceiling (contiguous + random-mask control) on E18 best checkpoints.
Usage: python segment_read_ceiling_e18.py --arm b1_fork --task absa_restaurant --device cuda:1
Writes artifacts/e18/<arm>/<task>/seed_42/segment_read_ceiling.json with both modes."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research_checks")); sys.path.insert(0, str(ROOT / "scripts" / "independent_review")); sys.path.insert(0, str(ROOT / "src"))
import mechanism_battery as mb
import segment_read_ceiling as src

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--task", required=True); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--max-splits", type=int, default=8)
    args = ap.parse_args()
    ckpt = ROOT / "artifacts" / "e18" / args.arm / args.task / "seed_42" / "best_checkpoint.pt"
    mb.MODELS["student_e8"]["ckpt"] = lambda t, _c=ckpt: _c
    out = {}
    for random_masks in (False, True):
        sys.argv = ["x", "--device", args.device, "--tasks", args.task, "--max-splits", str(args.max_splits)] + (["--random-masks"] if random_masks else [])
        # reuse main() but capture its json output path
        src.OUT = ROOT / "artifacts" / "e18" / args.arm / args.task / "seed_42"
        src.main()
        name = "segment_read_ceiling_randomctrl.json" if random_masks else "segment_read_ceiling.json"
        out["random_mask" if random_masks else "contiguous"] = json.load(open(src.OUT / name))[args.task]
    res = {"arm": args.arm, "task": args.task, "baseline_f1": out["contiguous"]["baseline"], "oracle_contiguous": out["contiguous"]["oracle_split"], "oracle_random_mask": out["random_mask"]["oracle_split"],
           "segment_specific_gap": round(out["contiguous"]["oracle_split"] - out["random_mask"]["oracle_split"], 3), "raw": out}
    json.dump(res, open(src.OUT / "segment_read_ceiling_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "raw"}))

if __name__ == "__main__":
    main()
