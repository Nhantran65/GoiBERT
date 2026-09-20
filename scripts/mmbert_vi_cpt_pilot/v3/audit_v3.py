#!/usr/bin/env python3
"""v3 §3.3: general pool for the v3 filter, audit sampling (informal bucket + general bin 9), teencode coverage
and the data-gate verdict.

  sample : builds the v3 general pool (bins 7-9 after the v3 S1+S2 filter, excluding informal-bucket documents,
           host cap 2 %), samples 100 documents per audit cell and computes the teencode coverage table of the planned
           P_inf arm
  print  : prints one cell's samples for the single reader
  gate   : applies the §3.3 thresholds to the hand labels and writes DATA_GATE_VERDICT_V3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v3 as C3  # noqa: E402

import numpy as np  # noqa: E402

OUT = C3.ART_V3 / "data_audit_v3"
EXCERPT_CHARS = 1200
TARGET_RECALL = 0.90


def load_v3_scores() -> dict[int, dict]:
    out = {}
    for f in sorted((C3.DATA_V3 / "spun_scores_v3").glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[int(r["pool_index"])] = r
    return out


def load_features() -> dict[int, dict]:
    out = {}
    for f in sorted((C3.DATA_V3 / "informal_scan").glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[int(r["pool_index"])] = r
    return out


def build_general_v3(exclude: set[int]) -> dict:
    scores = load_v3_scores()
    pool = [r for r in C3.C2.pool_table() if r["split"] == "pool"]
    c = collections.Counter()
    docs = []
    for r in pool:
        if r["bin"] not in ("7", "8", "9"):
            c[f"drop_bin{r['bin']}"] += 1
            continue
        s = scores.get(r["pool_index"])
        if s is None:
            c["no_score"] += 1
            continue
        if s["drop"]:
            c["drop_v3_filter"] += 1
            continue
        if r["pool_index"] in exclude:
            c["drop_informal_bucket"] += 1
            continue
        docs.append(r)
    total_before = sum(r["n_tokens"] for r in docs)
    cap = 0.02 * total_before
    per_host = collections.Counter()
    kept = []
    for r in sorted(docs, key=lambda r: (-r["n_tokens"], r["pool_index"])):
        if per_host[r["host"]] + r["n_tokens"] > cap:
            c["drop_host_cap"] += 1
            continue
        per_host[r["host"]] += r["n_tokens"]
        kept.append(r)
    kept.sort(key=lambda r: r["pool_index"])
    total = sum(r["n_tokens"] for r in kept)
    by_bin = collections.Counter()
    for r in kept:
        by_bin[r["bin"]] += r["n_tokens"]
    return {"documents": len(kept), "tokens": total, "counters": dict(c), "tokens_by_bin": dict(by_bin),
            "docs": kept, "volume_gate_ge_1B": total >= 1_000_000_000}


def texts_for(rows: list[dict], chars: int | None = None) -> dict[int, str]:
    by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_part[r["part"]].append(r)
    out: dict[int, str] = {}
    for part, rs in by_part.items():
        need = {r["kh64"]: r for r in rs}
        for kh64, text in C3.C2.part_texts(part, set(need)).items():
            out[need[kh64]["pool_index"]] = text if chars is None else text[:chars]
    return out


def cmd_sample(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bucket = [json.loads(l) for l in (C3.DATA_V3 / "informal_bucket" / "informal_docs.jsonl").open(encoding="utf-8")]
    general = build_general_v3({r["pool_index"] for r in bucket})
    C3.write_json(C3.ART_V3 / "general_pool_v3.json",
                  {k: v for k, v in general.items() if k != "docs"} | {"created_at": C3.common.now()})
    rng = random.Random(C3.SEED)
    cells: dict[str, list[dict]] = {}
    train_bucket = [r for r in bucket if not r["held_out"]]
    rng.shuffle(train_bucket)
    cells["informal_bucket"] = train_bucket[: args.per_cell]
    bin9 = [r for r in general["docs"] if r["bin"] == "9"]
    rng.shuffle(bin9)
    cells["general_bin9"] = bin9[: args.per_cell]
    rows = []
    for name, sel in cells.items():
        texts = texts_for(sel, EXCERPT_CHARS)
        for i, r in enumerate(sel):
            rows.append({"cell": name, "index": i, "pool_index": r["pool_index"], "host": r.get("host"), "url": r.get("url"),
                         "bin": r.get("bin"), "n_tokens": r["n_tokens"], "source": r.get("source"),
                         "n_teencode": r.get("n_teencode"), "n_emoji": r.get("n_emoji"), "text": texts.get(r["pool_index"], "")})
    with (OUT / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"cell": r["cell"], "index": r["index"], "pool_index": r["pool_index"],
                                 "real_vietnamese": None, "informal_voice": None, "machine_translated_or_generated": None,
                                 "adult_or_spam": None, "boilerplate_majority": None}, ensure_ascii=False) + "\n")
    print(json.dumps({"cells": {k: len(v) for k, v in cells.items()},
                      "general_v3": {k: general[k] for k in ("documents", "tokens", "tokens_by_bin", "counters", "volume_gate_ge_1B")}},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_print(args) -> int:
    rows = [json.loads(l) for l in (OUT / "samples.jsonl").open(encoding="utf-8")]
    sel = [r for r in rows if r["cell"] == args.cell][args.start:args.end]
    for r in sel:
        t = " ".join(r["text"].split())[: args.chars]
        print(f"{r['index']:03d}|{r.get('source') or ''}|{(r.get('host') or '')[:22]}|teen={r.get('n_teencode')}|{t}")
    return 0


def cmd_teencode(args) -> int:
    """Coverage of the top-100 teencode forms inside the planned P_inf arm (bucket + general + wiki shares)."""
    teen = C3.load_teencode()
    rx = C3.teencode_pattern(teen)
    bucket = [json.loads(l) for l in (C3.DATA_V3 / "informal_bucket" / "informal_docs.jsonl").open(encoding="utf-8")]
    counts: collections.Counter = collections.Counter()
    total_texts = 0
    for r in bucket[: args.max_docs]:
        text = C3.C2.part_texts(r["part"], {r["kh64"]}).get(r["kh64"], "")
        if not text:
            continue
        total_texts += 1
        counts.update(m.group(0).lower() for m in rx.finditer(text))
    top = counts.most_common(100)
    report = {"created_at": C3.common.now(), "documents_scanned": total_texts, "bucket_documents": len(bucket),
              "top100": [{"form": f, "count": n} for f, n in top],
              "forms_ge_100": sum(1 for _, n in top if n >= 100), "forms_ge_5": sum(1 for _, n in top if n >= 5)}
    C3.write_json(OUT / "teencode_coverage.json", report)
    print(json.dumps({k: report[k] for k in ("documents_scanned", "forms_ge_100", "forms_ge_5")}, indent=1))
    print(json.dumps(report["top100"][:25], ensure_ascii=False))
    return 0


def cmd_gate(args) -> int:
    labels = [json.loads(l) for l in (OUT / "labels.jsonl").open(encoding="utf-8")]
    if any(r.get("real_vietnamese") is None for r in labels):
        raise RuntimeError("unlabelled rows")

    def rate(rows, key):
        vals = [bool(r[key]) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    by_cell = collections.defaultdict(list)
    for r in labels:
        by_cell[r["cell"]].append(r)
    bucket = by_cell["informal_bucket"]
    general9 = by_cell["general_bin9"]
    side = json.loads((OUT / "side_effect_report.json").read_text(encoding="utf-8"))
    gen = json.loads((C3.ART_V3 / "general_pool_v3.json").read_text(encoding="utf-8"))
    bucket_manifest = json.loads((C3.DATA_V3 / "informal_bucket" / "manifest.json").read_text(encoding="utf-8"))
    teen = json.loads((OUT / "teencode_coverage.json").read_text(encoding="utf-8"))
    checks = {
        "bucket_vietnamese_ge_90": rate(bucket, "real_vietnamese") >= 0.90,
        "bucket_informal_ge_70": rate(bucket, "informal_voice") >= 0.70,
        "bucket_spam_le_5": rate(bucket, "adult_or_spam") <= 0.05,
        "bucket_boilerplate_le_15": rate(bucket, "boilerplate_majority") <= 0.15,
        "general_bin9_spam_le_5": rate(general9, "adult_or_spam") <= 0.05,
        "side_effect_le_15": bool(side["gate_le_15pct"]),
        "bucket_tokens_ge_75M": bucket_manifest["volume_gate_ge_75M"],
        "bucket_tokens_ge_37_5M": bucket_manifest["volume_gate_ge_37_5M"],
        "general_tokens_ge_1B": gen["volume_gate_ge_1B"],
        "teencode_top100_each_ge_100_in_arm": int(teen.get("forms_ge_100_in_arm", teen.get("forms_ge_100", 0))) >= 100,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    report = {"created_at": C3.common.now(), "verdict": verdict, "checks": checks,
              "cells": {c: {"n": len(v), "real_vietnamese": rate(v, "real_vietnamese"), "informal_voice": rate(v, "informal_voice"),
                            "spun": rate(v, "machine_translated_or_generated"), "spam": rate(v, "adult_or_spam"),
                            "boilerplate": rate(v, "boilerplate_majority")} for c, v in by_cell.items()},
              "volumes": {"bucket_tokens": bucket_manifest["tokens"], "bucket_sources": bucket_manifest["tokens_by_source"],
                          "general_tokens": gen["tokens"], "general_bins": gen["tokens_by_bin"]},
              "side_effect": side, "teencode": {"forms_ge_100_in_arm": teen.get("forms_ge_100_in_arm"),
                                                "forms_ge_5_in_arm": teen.get("forms_ge_5_in_arm"),
                                                "method": teen.get("method")},
              "thresholds": {"bucket_informal": 0.70, "bucket_vietnamese": 0.90, "bucket_spam_max": 0.05,
                             "bucket_boilerplate_max": 0.15, "general_bin9_spam_max": 0.05, "side_effect_max": 0.15,
                             "bucket_tokens_min": 75_000_000, "bucket_tokens_min_with_repeat": 37_500_000}}
    C3.write_json(C3.ART_V3 / "DATA_GATE_VERDICT_V3.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:3500])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--per-cell", type=int, default=100); s.set_defaults(func=cmd_sample)
    p = sub.add_parser("print"); p.add_argument("--cell", required=True); p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100); p.add_argument("--chars", type=int, default=120); p.set_defaults(func=cmd_print)
    t = sub.add_parser("teencode"); t.add_argument("--max-docs", type=int, default=20000); t.set_defaults(func=cmd_teencode)
    g = sub.add_parser("gate"); g.set_defaults(func=cmd_gate)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
