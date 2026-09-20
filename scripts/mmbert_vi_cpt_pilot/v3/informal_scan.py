#!/usr/bin/env python3
"""v3 §3.2 support: per-document informal features for the whole v1 pool (and, later, for bins 5-6).

For every pool document the scan records teencode count, emoji count, first-person pronoun count and the plan's
"informal cue" boolean, plus the v1/v2 metadata the bucket builder needs (bin, host, token count, v2 S1/S2 score and
v2 drop decision, ID register score).  One JSONL per part so later stages can stream them.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v3 as C3  # noqa: E402

OUT = C3.DATA_V3 / "informal_scan"


def _scan_part(job):
    part, rows = job
    teen_rx = C3.teencode_pattern(C3.load_teencode())
    need = {r["kh64"]: r for r in rows}
    out = []
    for kh64, text in C3.C2.part_texts(part, set(need)).items():
        r = need[kh64]
        n_teen, n_emoji, n_pron, cue = C3.informal_features(text, teen_rx)
        out.append({"pool_index": r["pool_index"], "bin": r["bin"], "group": r["group"], "host": r["host"],
                    "n_tokens": r["n_tokens"], "n_teencode": n_teen, "n_emoji": n_emoji, "n_pronoun": n_pron,
                    "informal_cue": bool(cue)})
    return part, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = [r for r in C3.C2.pool_table() if r["split"] == "pool"]
    by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        by_part[r["part"]].append(r)
    jobs = sorted(by_part.items())
    stats = collections.Counter()
    with mp.Pool(args.workers) as pool_exec:
        for i, (part, rows) in enumerate(pool_exec.imap_unordered(_scan_part, jobs), 1):
            with (out_dir / f"{part}.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                    stats["docs"] += 1
                    stats["informal_cue"] += int(r["informal_cue"])
                    stats["with_teencode"] += int(r["n_teencode"] > 0)
                    stats["with_emoji"] += int(r["n_emoji"] > 0)
            if i % 48 == 0:
                print(f"[scan] {i}/{len(jobs)} parts ({time.time()-t0:.0f}s)", flush=True)
    C3.write_json(out_dir / "scan_manifest.json", {"created_at": C3.common.now(), "parts": len(jobs), "workers": args.workers,
                                                   "counters": dict(stats), "elapsed_s": round(time.time() - t0, 1),
                                                   "definition": "informal cue = >=3 teencode OR >=1 emoji OR (>=3 first-person pronouns AND >=1 teencode)"})
    print(json.dumps(dict(stats), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
