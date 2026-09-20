#!/usr/bin/env python3
"""v2 §3 data-gate verdict: apply the plan's per-cell thresholds to the hand labels and the volume manifests.

Reads `artifacts/mmbert_vi_cpt_pilot/v2/data_audit_v2/labels.jsonl` (100 hand-labelled samples per cell) plus the volume
artifacts (`general_pool.json`, `register/manifest.json`, `wiki_filter.json`) and writes
`artifacts/mmbert_vi_cpt_pilot/v2/DATA_GATE_VERDICT_V2.json`.  Nothing is tuned here: the thresholds are the plan's.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C  # noqa: E402

GENERAL_CELLS = ("general_bin9", "general_bin8", "general_bin7")
REGISTER_CELLS = ("register_forum_qa", "register_review_commerce")
THRESHOLDS = {"general_vietnamese": 0.90, "general_boilerplate_max": 0.10, "general_spun_max": 0.05,
              "general_spam_max": 0.03, "register_opinion": 0.70, "register_spun_max": 0.05,
              "discussion_opinion": 0.70, "wiki_real_content": 0.70, "side_effect_max": 0.15,
              "general_min_tokens": 1_000_000_000, "register_min_tokens": 50_000_000}


def rate(rows, key) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(bool(v) for v in vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.ART_V2 / "DATA_GATE_VERDICT_V2.json"))
    args = ap.parse_args()
    labels = [json.loads(l) for l in (C.ART_V2 / "data_audit_v2" / "labels.jsonl").open(encoding="utf-8")]
    missing = [r for r in labels if r.get("real_vietnamese") is None]
    if missing:
        raise RuntimeError(f"{len(missing)} unlabelled rows")
    by_cell: dict[str, list[dict]] = collections.defaultdict(list)
    for r in labels:
        by_cell[r["cell"]].append(r)
    cells = {}
    for cell, rows in sorted(by_cell.items()):
        n = len(rows)
        cells[cell] = {"n": n,
                       "real_vietnamese": rate(rows, "real_vietnamese"), "boilerplate": rate(rows, "boilerplate_majority"),
                       "spun": rate(rows, "machine_translated_or_generated"), "spam": rate(rows, "adult_or_spam"),
                       "opinion_voice": rate(rows, "opinion_voice"), "real_content": rate(rows, "real_content")}
    general = json.loads((C.ART_V2 / "general_pool.json").read_text(encoding="utf-8"))
    register = json.loads((C.ART_V2 / "register" / "manifest.json").read_text(encoding="utf-8"))
    wiki = json.loads((C.ART_V2 / "wiki_filter.json").read_text(encoding="utf-8"))
    checks: dict[str, bool | None] = {}
    for cell in GENERAL_CELLS:
        s = cells.get(cell)
        if not s:
            checks[f"{cell}_present"] = False
            continue
        checks[f"{cell}_vietnamese_ge_90"] = s["real_vietnamese"] >= THRESHOLDS["general_vietnamese"]
        checks[f"{cell}_boilerplate_le_10"] = s["boilerplate"] <= THRESHOLDS["general_boilerplate_max"]
        checks[f"{cell}_spun_le_5"] = s["spun"] <= THRESHOLDS["general_spun_max"]
        checks[f"{cell}_spam_le_3"] = s["spam"] <= THRESHOLDS["general_spam_max"]
    for cell in REGISTER_CELLS:
        s = cells.get(cell)
        if not s:
            checks[f"{cell}_present"] = False
            continue
        checks[f"{cell}_opinion_ge_70"] = s["opinion_voice"] >= THRESHOLDS["register_opinion"]
        checks[f"{cell}_spun_le_5"] = s["spun"] <= THRESHOLDS["register_spun_max"]
    if "discussion" in cells:
        checks["discussion_opinion_ge_70"] = cells["discussion"]["opinion_voice"] >= THRESHOLDS["discussion_opinion"]
    if "wikipedia" in cells:
        checks["wikipedia_real_content_ge_70"] = cells["wikipedia"]["real_content"] >= THRESHOLDS["wiki_real_content"]
    if "ugc_side_effect" in cells:
        rows = [r for r in by_cell["ugc_side_effect"] if r.get("opinion_voice")]
        dropped = sum(bool(r.get("dropped_by_va1")) for r in rows)
        cells["ugc_side_effect"]["genuine_opinion_rows"] = len(rows)
        cells["ugc_side_effect"]["genuine_dropped_by_va1"] = dropped
        cells["ugc_side_effect"]["genuine_drop_rate"] = (dropped / len(rows)) if rows else None
        thr = json.loads((C.ART_V2 / "spun_filter" / "thresholds.json").read_text(encoding="utf-8"))

        def drop_p2(r) -> bool:
            s1, s2 = r.get("s1"), r.get("s2")
            return s2 is not None and (s2 >= thr["threshold_s2"] or (s1 is not None and s1 < thr["threshold_s1_p2"]))

        dropped_p2 = sum(drop_p2(r) for r in rows)
        cells["ugc_side_effect"]["genuine_drop_rate_p2_s1"] = (dropped_p2 / len(rows)) if rows else None
        cells["ugc_side_effect"]["note_p2"] = "the plan's remedy (S1 at the 2nd percentile) does not change the side-effect rate: the genuine documents are removed by the S2 term"
        checks["side_effect_le_15"] = bool(rows) and (dropped / len(rows)) <= THRESHOLDS["side_effect_max"]
    checks["general_tokens_ge_1_0B"] = general["tokens"] >= THRESHOLDS["general_min_tokens"]
    checks["register_tokens_ge_50M"] = register["tokens"] >= THRESHOLDS["register_min_tokens"]
    checks["register_tokens_ge_240M"] = register["tokens"] >= 240_000_000
    checks["wikipedia_kept_or_dropped_per_plan"] = wiki["verdict"] in ("keep", "drop_and_move_5pct_to_general")
    all_ok = all(v for k, v in checks.items() if v is not None and k != "register_tokens_ge_240M")
    verdict = "PASS" if all_ok else "FAIL"
    report = {"created_at": C.common.now(), "verdict": verdict, "checks": checks, "cells": cells,
              "thresholds": THRESHOLDS, "volumes": {"general_tokens": general["tokens"], "register_tokens": register["tokens"],
                                                    "register_tokens_by_group": register["tokens_by_group"], "wiki_tokens": wiki["tokens"]},
              "note": "register_tokens_ge_240M is reported but not required for the 250M pilot (plan: >= 50M, report 240M)",
              "reader": "single reader (root agent), 1 200-character excerpts / single paragraphs"}
    C.write_json(Path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
