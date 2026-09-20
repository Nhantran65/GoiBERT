#!/usr/bin/env python3
"""Step 0(a): (1) Cafe-Bami gap after removing units where both models make the SAME wrong prediction;
(2) 'right sentiment, wrong aspect' events per model.  Read-only over events.jsonl / prediction files."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts" / "cafe_bami_paired_error_audit"
from goibert import data as gdata  # noqa: E402

ev = [json.loads(l) for l in (ART / "events.jsonl").open(encoding="utf-8")]
NAMES = {0: "absent", 1: "positive", 2: "negative", 3: "neutral"}
out: dict = {}


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def pair_counts_units(units, key):
    tp = fp = fn = 0
    for u in units:
        g, p = u["gold"], u[key]
        if g > 0 and p == g:
            tp += 1
        if p > 0 and not (g > 0 and p == g):
            fp += 1
        if g > 0 and not (p > 0 and p == g):
            fn += 1
    return tp, fp, fn


for task in ("absa_restaurant", "visfd"):
    units = [e for e in ev if e["task"] == task and not e.get("detection_only")]
    res = {}
    for name, keep in (("all", lambda u: True), ("drop_both_wrong_same", lambda u: u["paired"] != "both_wrong_same"), ("drop_all_both_wrong", lambda u: not u["paired"].startswith("both_wrong"))):
        sub = [u for u in units if keep(u)]
        row = {"units": len(sub)}
        for m in ("bami", "cafe"):
            tp, fp, fn = pair_counts_units(sub, m)
            row[m] = {"tp": tp, "fp": fp, "fn": fn, "f1": f1(tp, fp, fn)}
        row["gap_pp"] = 100 * (row["cafe"]["f1"] - row["bami"]["f1"])
        res[name] = row
    # (2) right sentiment, wrong aspect: same review, model misses gold aspect A (gold pol p) and is spurious at B with predicted pol p
    by_rev = defaultdict(list)
    for u in units:
        by_rev[u["id"]].append(u)
    swap = {m: Counter() for m in ("bami", "cafe")}; swap_strict = {m: Counter() for m in ("bami", "cafe")}; swap_any_pol = {m: Counter() for m in ("bami", "cafe")}
    n_reviews_with_swap = {m: 0 for m in ("bami", "cafe")}
    for rid, us in by_rev.items():
        for m in ("bami", "cafe"):
            misses = [u for u in us if u["gold"] > 0 and u[m] == 0]
            spur = [u for u in us if u["gold"] == 0 and u[m] > 0]
            found = False
            for a in misses:
                for b in spur:
                    swap_any_pol[m][f"{a['aspect']}->{b['aspect']}"] += 1
                    if a["gold"] == b[m]:
                        swap[m][f"{a['aspect']}->{b['aspect']}"] += 1; found = True
            if found:
                n_reviews_with_swap[m] += 1
            # strict: review has exactly one gold-present scored aspect, model predicts exactly one scored aspect, different, same polarity
            gp = [u for u in us if u["gold"] > 0]; pp = [u for u in us if u[m] > 0]
            if len(gp) == 1 and len(pp) == 1 and gp[0]["aspect"] != pp[0]["aspect"] and gp[0]["gold"] == pp[0][m]:
                swap_strict[m][f"{gp[0]['aspect']}->{pp[0]['aspect']}"] += 1
    res["sentiment_right_aspect_wrong"] = {m: {"events_same_polarity": sum(swap[m].values()), "reviews_with_event": n_reviews_with_swap[m], "top_pairs": swap[m].most_common(8),
                                              "strict_single_aspect_swaps": sum(swap_strict[m].values()), "strict_top": swap_strict[m].most_common(6),
                                              "miss_plus_spurious_any_polarity": sum(swap_any_pol[m].values()), "total_miss": sum(1 for u in units if u["gold"] > 0 and u[m] == 0), "total_spurious": sum(1 for u in units if u["gold"] == 0 and u[m] > 0)} for m in ("bami", "cafe")}
    # gold-present miss ONLY when another aspect predicted with same polarity (fraction of misses explainable as aspect swap)
    out[task] = res

# Topic
units = [e for e in ev if e["task"] == "vsfc_topic"]
res = {}
for name, keep in (("all", lambda u: True), ("drop_both_wrong_same", lambda u: u["paired"] != "both_wrong_same"), ("drop_all_both_wrong", lambda u: not u["paired"].startswith("both_wrong"))):
    sub = [u for u in units if keep(u)]
    row = {"units": len(sub)}
    for m in ("bami", "cafe"):
        f1s = []
        for c in range(4):
            tp = sum(1 for u in sub if u["gold"] == c and u[m] == c); fp = sum(1 for u in sub if u["gold"] != c and u[m] == c); fn = sum(1 for u in sub if u["gold"] == c and u[m] != c)
            f1s.append(f1(tp, fp, fn))
        row[m] = {"macro_f1": sum(f1s) / 4, "per_class_f1": f1s, "acc": sum(u[m] == u["gold"] for u in sub) / len(sub)}
    row["gap_pp"] = 100 * (row["cafe"]["macro_f1"] - row["bami"]["macro_f1"])
    res[name] = row
out["vsfc_topic"] = res
(ART / "step0_counts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
for task, r in out.items():
    print("==", task)
    for k in ("all", "drop_both_wrong_same", "drop_all_both_wrong"):
        row = r[k]
        if task == "vsfc_topic":
            print(f"  {k:24s} units {row['units']:5d}  bami {100*row['bami']['macro_f1']:.2f}  cafe {100*row['cafe']['macro_f1']:.2f}  gap {row['gap_pp']:+.2f}")
        else:
            print(f"  {k:24s} units {row['units']:5d}  bami {100*row['bami']['f1']:.2f}  cafe {100*row['cafe']['f1']:.2f}  gap {row['gap_pp']:+.2f}")
    if "sentiment_right_aspect_wrong" in r:
        for m, v in r["sentiment_right_aspect_wrong"].items():
            print(f"  swap[{m}]: same-pol miss+spurious events {v['events_same_polarity']} in {v['reviews_with_event']} reviews (misses {v['total_miss']}, spurious {v['total_spurious']}); strict single-aspect swaps {v['strict_single_aspect_swaps']}; top {v['top_pairs'][:5]}; strict top {v['strict_top'][:4]}")
