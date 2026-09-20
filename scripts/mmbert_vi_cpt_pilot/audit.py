#!/usr/bin/env python3
"""§3 manual data audit: 100 documents per cell (seed 42), labels filed by the reader, then scored against the gate.

`--sample` draws the cells from the stage-1 excerpt files (1/64 of kept documents per part, so the sample spans every
shard/part), the MT flag cell from the MT-dropped documents kept for auditing, and the Wikipedia cell from the prepared
wiki slice.  `--score` reads the label file and applies the plan's gate:
  * opinion_review precision >= 70 % (cells opinion_bin910 and opinion_bin78, label `opinion_voice`)
  * general cells >= 90 % real Vietnamese and <= 10 % boilerplate-majority
  * the MT flag is correct (machine translation / generated text) for >= 70 % of the MT cell
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CELLS = ("bin10_general", "bin9_general", "bin8_general", "opinion_bin910", "opinion_bin78", "discussion", "mt_flag", "wikipedia")


def cell_of(rec: dict) -> str | None:
    group, b = rec["group"], str(rec["bin"])
    if group == "register_opinion":
        return "opinion_bin910" if b in ("9", "10") else ("opinion_bin78" if b in ("7", "8") else None)
    if group == "register_discussion":
        return "discussion"
    if group == "general":
        return {"10": "bin10_general", "9": "bin9_general", "8": "bin8_general"}.get(b)
    if group == "wiki":
        return "wikipedia"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir", default=str(common.DATA_DIR / "scan"))
    ap.add_argument("--wiki-dir", default=str(common.DATA_DIR / "wiki"))
    ap.add_argument("--out-dir", default=str(common.ART_DIR / "data_audit"))
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--mode", choices=("sample", "score"), default="sample")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    labels_path = out_dir / "labels.jsonl"

    if args.mode == "sample":
        buckets: dict[str, list[dict]] = {c: [] for c in CELLS}
        for f in sorted(Path(args.scan_dir).glob("*.excerpts.jsonl")):
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    c = cell_of(r)
                    if c:
                        buckets[c].append(r)
        for f in sorted(Path(args.scan_dir).glob("*.mt_audit.jsonl")):
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    buckets["mt_flag"].append({"id": r["id"], "bin": r.get("shard", "")[:1], "group": "mt_dropped",
                                               "url": r.get("url"), "op": r.get("op"), "re": r.get("re"), "id_reg": r.get("id_reg"),
                                               "mt": r.get("mt"), "chars": len(r.get("text", "")), "excerpt": r.get("text", "")[:1200]})
        wiki_docs = Path(args.wiki_dir) / "wiki_audit_excerpts.jsonl"
        if wiki_docs.is_file():
            with wiki_docs.open(encoding="utf-8") as fh:
                for line in fh:
                    buckets["wikipedia"].append(json.loads(line))
        rng = random.Random(common.SEED)
        rows = []
        for c in CELLS:
            pool = buckets[c]
            rng.shuffle(pool)
            picked = pool[: args.per_cell]
            print(f"[audit] cell {c}: {len(pool)} available -> {len(picked)} sampled", flush=True)
            for i, r in enumerate(picked):
                rows.append({"cell": c, "index": i, "id": r["id"], "bin": r.get("bin"), "group": r.get("group"), "url": r.get("url"),
                             "op": r.get("op"), "re": r.get("re"), "id_reg": r.get("id_reg"), "mt": r.get("mt"), "excerpt": r.get("excerpt", "")})
        with samples_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with labels_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"cell": r["cell"], "index": r["index"], "id": r["id"], "real_vietnamese": None,
                                     "boilerplate_majority": None, "machine_translated_or_generated": None, "adult_or_spam": None,
                                     "opinion_voice": None}, ensure_ascii=False) + "\n")
        print(f"[audit] wrote {len(rows)} samples to {samples_path} and a label template to {labels_path}")
        return 0

    samples = {}
    with samples_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            samples[(r["cell"], r["index"])] = r
    labels = []
    with labels_path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("real_vietnamese") is None:
                raise RuntimeError(f"unlabelled row {r['cell']}#{r['index']} -- the audit must be read by a human")
            labels.append(r)
    by_cell: dict[str, list[dict]] = {}
    for r in labels:
        by_cell.setdefault(r["cell"], []).append(r)
    stats = {}
    for c, rows in by_cell.items():
        n = len(rows)
        stats[c] = {"n": n,
                    "real_vietnamese": sum(bool(r["real_vietnamese"]) for r in rows) / n,
                    "boilerplate_majority": sum(bool(r["boilerplate_majority"]) for r in rows) / n,
                    "machine_translated_or_generated": sum(bool(r["machine_translated_or_generated"]) for r in rows) / n,
                    "adult_or_spam": sum(bool(r["adult_or_spam"]) for r in rows) / n,
                    "opinion_voice": (sum(bool(r["opinion_voice"]) for r in rows) / n) if any(r["opinion_voice"] is not None for r in rows) else None}
    opinion_precision = [stats[c]["opinion_voice"] for c in ("opinion_bin910", "opinion_bin78") if c in stats and stats[c]["opinion_voice"] is not None]
    general_cells = [c for c in stats if c.endswith("_general") or c == "wikipedia"]
    gate = {"opinion_precision_min": min(opinion_precision) if opinion_precision else None,
            "opinion_precision_ge_70": bool(opinion_precision) and min(opinion_precision) >= 0.70,
            "general_real_vietnamese_ge_90": all(stats[c]["real_vietnamese"] >= 0.90 for c in general_cells),
            "general_boilerplate_le_10": all(stats[c]["boilerplate_majority"] <= 0.10 for c in general_cells),
            "mt_flag_correct_ge_70": bool("mt_flag" in stats) and stats["mt_flag"]["machine_translated_or_generated"] >= 0.70}
    gate["all_ok"] = all(v for k, v in gate.items() if isinstance(v, bool))
    report = {"created_at": common.now(), "cells": stats, "gate": gate, "seed": common.SEED,
              "reader": "single reader (root agent); excerpts of the first 1200 characters", "source_hash": common.source_hash()}
    common.write_json(out_dir / "audit_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
