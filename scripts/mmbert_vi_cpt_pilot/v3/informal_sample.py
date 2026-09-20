#!/usr/bin/env python3
"""v3 §3.1 sampling: 400 informal documents for the hand labels (300 train + 100 side-effect gate).

Four strata (100 each, seed 42), all requiring the plan's informal cue:
  `discussion`    v1 discussion documents with ID >= 0.3
  `ugc_host`      documents from the v2 UGC whitelist hosts
  `general_teen`  general documents with >= 3 teencode forms
  `v2_dropped`    documents the v2 S2 filter dropped, with a strong informal signal (emoji or >= 5 teencode forms)

Subcommands: sample | print | check
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

OUT = C3.ART_V3 / "data_audit_v3"
EXCERPT_CHARS = 1200


def load_features() -> list[dict]:
    rows = []
    for f in sorted((C3.DATA_V3 / "informal_scan").glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
    return rows


def load_part_index() -> dict[int, dict]:
    return {r["pool_index"]: r for r in C3.C2.pool_table() if r["split"] == "pool"}


def load_v2_scores() -> dict[int, dict]:
    out = {}
    for f in sorted((C3.C2.DATA_V2 / "spun_scores").glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[int(r["pool_index"])] = r
    return out


def strata(features: list[dict], parts: dict[int, dict], scores: dict[int, dict], id_reg: dict[int, float]):
    ugc_hosts = set(C3.C2.UGC_WHITELIST)
    out = collections.defaultdict(list)
    for r in features:
        if not r["informal_cue"]:
            continue
        idx = r["pool_index"]
        p = parts.get(idx)
        if p is None:
            continue
        if r["group"] == "register_discussion" and id_reg.get(idx, 0.0) >= 0.3:
            out["discussion"].append(r)
        if C3.C2.host_matches(r["host"], ugc_hosts):
            out["ugc_host"].append(r)
        if r["bin"] in ("7", "8", "9") and r["n_teencode"] >= 3:
            out["general_teen"].append(r)
        s = scores.get(idx)
        if s is not None and s["drop"] and (r["n_emoji"] >= 1 or r["n_teencode"] >= 5):
            out["v2_dropped"].append(r)
    return out


def cmd_sample(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    features = load_features()
    parts = load_part_index()
    scores = load_v2_scores()
    id_reg = C3.id_register_by_pool_index()
    strat = strata(features, parts, scores, id_reg)
    rng = random.Random(C3.SEED)
    picked = []
    for name in ("discussion", "ugc_host", "general_teen", "v2_dropped"):
        rows = strat.get(name, [])
        rng.shuffle(rows)
        chosen = rows[args.start: args.start + args.per_stratum]
        print(f"[sample] stratum {name}: {len(rows)} candidates -> {len(chosen)}", flush=True)
        for r in chosen:
            p = parts[r["pool_index"]]
            picked.append({"stratum": name, "pool_index": r["pool_index"], "part": p["part"], "local_index": p["local_index"],
                           "kh64": p["kh64"], "bin": r["bin"], "group": r["group"], "host": r["host"], "url": p["url"],
                           "n_tokens": r["n_tokens"], "n_teencode": r["n_teencode"], "n_emoji": r["n_emoji"],
                           "n_pronoun": r["n_pronoun"], "id_reg": id_reg.get(r["pool_index"], 0.0),
                           "v2_drop": bool(scores.get(r["pool_index"], {}).get("drop", False))})
        strat[name] = chosen
    path = OUT / (f"informal_sample_index{args.tag}.json")
    C3.write_json(path, {"per_stratum": args.per_stratum, "start": args.start, "rows": picked})
    print(f"[sample] wrote {len(picked)} rows to {path}")
    return 0


def cmd_print(args) -> int:
    rows = C3.read_json(OUT / f"informal_sample_index{args.tag}.json")["rows"]
    rows = [r for r in rows if r["stratum"] == args.stratum][args.start:args.end]
    by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_part[r["part"]].append(r)
    texts: dict[int, str] = {}
    for part, rs in by_part.items():
        by_kh = {r["kh64"]: r for r in rs}
        for kh64, text in C3.C2.part_texts(part, set(by_kh)).items():
            texts[by_kh[kh64]["pool_index"]] = text
    for r in rows:
        t = " ".join(texts.get(r["pool_index"], "").split())[: args.chars]
        print(f"{r['pool_index']}|{r['stratum']}|{r['host'][:22]}|teen={r['n_teencode']}|emoji={r['n_emoji']}|pron={r['n_pronoun']}|v2drop={int(r['v2_drop'])}|{t}")
    return 0


def cmd_check(args) -> int:
    """Side-effect gate (§3.1): drop rate of the new filter on hand-labelled real-informal documents."""
    labels = [r for r in C3.iter_jsonl(OUT / "informal_labels.jsonl") if r.get("informal_real")]
    thr = C3.read_json(C3.ART_V3 / "spun_filter_v3" / "thresholds.json")
    scores = {}
    for f in sorted((C3.DATA_V3 / "spun_scores_v3").glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                scores[int(r["pool_index"])] = r
    dropped = 0
    total = 0
    detail = []
    for r in labels:
        s = scores.get(int(r["pool_index"]))
        if s is None:
            continue
        d = bool(s["drop"])
        dropped += int(d)
        total += 1
        detail.append({"pool_index": r["pool_index"], "stratum": r.get("stratum"), "drop": d, "s1": s.get("s1"), "s2": s.get("s2")})
    rate = dropped / total if total else None
    report = {"thresholds": thr, "real_informal_labelled": len(labels), "scored": total, "dropped": dropped, "drop_rate": rate,
              "gate_le_15pct": bool(rate is not None and rate <= 0.15), "detail": detail,
              "created_at": C3.common.now()}
    C3.write_json(OUT / "side_effect_report.json", report)
    print(json.dumps({k: report[k] for k in ("real_informal_labelled", "scored", "dropped", "drop_rate", "gate_le_15pct")}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--per-stratum", type=int, default=100); s.add_argument("--start", type=int, default=0)
    s.add_argument("--tag", default=""); s.set_defaults(func=cmd_sample)
    p = sub.add_parser("print"); p.add_argument("--stratum", required=True); p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=100); p.add_argument("--chars", type=int, default=150); p.add_argument("--tag", default="")
    p.set_defaults(func=cmd_print)
    c = sub.add_parser("check"); c.set_defaults(func=cmd_check)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
