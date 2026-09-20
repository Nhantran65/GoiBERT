#!/usr/bin/env python3
"""v4 §2 gate: sample 100 ViSoBERT comments for the hand audit, teencode coverage table and the data-gate verdict.

  sample : 100 comments (seed 42) from the prepared training split + the teencode coverage table (top-100 forms after
           dropping the ambiguous entries flagged in v3: single/double-character forms and the multi-meaning list)
  print  : prints the samples for the single reader
  gate   : applies the §2 thresholds and writes DATA_GATE_VERDICT_V4.json
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v4 as C4  # noqa: E402

OUT = C4.ART_V4 / "data_audit_v4"
AMBIGUOUS = {"vs", "a", "s", "e", "t", "d", "i", "ô", "dừa", "bia", "seo", "âu", "mí", "kon", "k", "v", "m", "b", "p", "l",
             "r", "z", "h", "j", "u", "c", "o", "x", "g", "n", "y"}


def load_comments(split: str) -> list[dict]:
    rows = []
    with (C4.DATA_V4 / "visobert" / "comments.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == split:
                rows.append(r)
    return rows


def cmd_sample(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_comments("train")
    rng = random.Random(C4.SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    picked = [rows[i] for i in idx[: args.n]]
    with (OUT / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(picked):
            fh.write(json.dumps({"index": i, "source_index": r["i"], "text": r["text"], "n_tokens": r["n_tokens"]}, ensure_ascii=False) + "\n")
    with (OUT / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(len(picked)):
            fh.write(json.dumps({"index": i, "real_vietnamese": None, "informal_voice": None, "adult_or_spam": None}) + "\n")
    # teencode coverage table (description only; ambiguous forms removed)
    teen = C4.C3.load_teencode()
    rx = C4.C3.teencode_pattern(teen)
    counts: collections.Counter = collections.Counter()
    scans = [r for r in rows[: args.scan]]
    for r in scans:
        counts.update(m.group(0).lower() for m in rx.finditer(r["text"]))
    filtered = [(f, n) for f, n in counts.most_common() if f not in AMBIGUOUS and len(f) > 2]
    total_tokens = sum(r["n_tokens"] for r in scans)
    scale = (75_000_000 / max(1, total_tokens))          # informal share of the 250M arm = 75M tokens
    table = [{"form": f, "count_scanned": n, "count_arm_estimate": int(round(n * scale))} for f, n in filtered[:100]]
    report = {"created_at": C4.common.now(), "scanned_comments": len(scans), "scanned_tokens": total_tokens,
              "excluded_ambiguous_forms": sorted(AMBIGUOUS), "scale_to_arm": round(scale, 4),
              "top100": table, "forms_ge_100_in_arm": sum(1 for r in table if r["count_arm_estimate"] >= 100),
              "note": "descriptive only (no stopping condition); counts are over a scan subset and scaled to the 75M informal tokens of P_inf"}
    C4.write_json(OUT / "teencode_coverage.json", report)
    print(json.dumps({"samples": len(picked), "scanned": len(scans), "forms_ge_100_in_arm": report["forms_ge_100_in_arm"]}, indent=1))
    return 0


def cmd_print(args) -> int:
    rows = [json.loads(l) for l in (OUT / "samples.jsonl").open(encoding="utf-8")]
    for r in rows[args.start:args.end]:
        t = " ".join(r["text"].split())[: args.chars]
        print(f"{r['index']:03d}|tok={r['n_tokens']}|{t}")
    return 0


def cmd_gate(args) -> int:
    labels = [json.loads(l) for l in (OUT / "labels.jsonl").open(encoding="utf-8")]
    if any(r.get("real_vietnamese") is None for r in labels):
        raise RuntimeError("unlabelled rows")
    n = len(labels)
    rates = {k: sum(bool(r[k]) for r in labels) / n for k in ("real_vietnamese", "informal_voice", "adult_or_spam")}
    prep = json.loads((C4.DATA_V4 / "visobert" / "manifest.json").read_text(encoding="utf-8"))
    teen = json.loads((OUT / "teencode_coverage.json").read_text(encoding="utf-8"))
    checks = {"vietnamese_ge_90": rates["real_vietnamese"] >= 0.90,
              "informal_ge_70": rates["informal_voice"] >= 0.70,
              "spam_le_10": rates["adult_or_spam"] <= 0.10,
              "tokens_ge_75M": prep["volume_gate_ge_75M"]}
    verdict = "PASS" if all(checks.values()) else "FAIL"
    report = {"created_at": C4.common.now(), "verdict": verdict, "checks": checks, "audit": {"n": n, **rates},
              "volume": {"train_token_positions": prep["train_token_positions"], "train_sequences": prep["train_sequences"],
                         "holdout_comments": prep["counters"]["holdout_comments"], "holdout_sequences": prep["holdout_sequences"]},
              "decontam_by_split": prep["decontam_by_split"], "prep_counters": prep["counters"],
              "teencode": {"forms_ge_100_in_arm": teen["forms_ge_100_in_arm"], "note": teen["note"]},
              "license": C4.LICENSE_NOTE, "thresholds": {"vietnamese": 0.90, "informal": 0.70, "spam_max": 0.10, "tokens_min": 75_000_000}}
    C4.write_json(C4.ART_V4 / "DATA_GATE_VERDICT_V4.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2500])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--n", type=int, default=100); s.add_argument("--scan", type=int, default=20000)
    s.set_defaults(func=cmd_sample)
    p = sub.add_parser("print"); p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int, default=100)
    p.add_argument("--chars", type=int, default=160); p.set_defaults(func=cmd_print)
    g = sub.add_parser("gate"); g.set_defaults(func=cmd_gate)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
