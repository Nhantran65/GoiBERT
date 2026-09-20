#!/usr/bin/env python3
"""v2 §2 "vá 3" + §3 volume accounting: the v2 general pool.

general = pool documents in bins 7-9 that survive vá 1 (S1+S2), excluding the documents that feed the register bucket
(UGC whitelist hosts and the `discussion` bucket), with a per-host cap of 2 % of the general tokens.  Bin 10 is dropped
entirely (41 % spun, ~1 % of tokens).  Writes the kept document list, the host/bin statistics and the volume gate numbers.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C  # noqa: E402


def load_scores() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in sorted((C.DATA_V2 / "spun_scores").glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[int(r["pool_index"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(C.ART_V2))
    ap.add_argument("--host-cap", type=float, default=0.02)
    args = ap.parse_args()
    pool = C.pool_table()
    scores = load_scores()
    ugc_hosts = set(C.UGC_WHITELIST) | {"thuvienphapluat.vn", "danluat.thuvienphapluat.vn", "hoidap.thuvienphapluat.vn", "luatminhkhue.vn"}
    c = collections.Counter()
    drop_by_bin = collections.Counter()
    tot_by_bin = collections.Counter()
    drop_by_host = collections.Counter()
    docs = []
    for r in pool:
        if r["split"] != "pool":
            continue
        tot_by_bin[r["bin"]] += r["n_tokens"]
        s = scores.get(r["pool_index"])
        is_ugc = C.host_matches(r["host"], ugc_hosts)
        is_discussion = r["group"] == "register_discussion"
        if r["bin"] == "10":
            c["drop_bin10"] += 1
            continue
        if r["bin"] not in ("7", "8", "9"):
            c["drop_other_bin"] += 1
            continue
        if s is None:
            c["no_score"] += 1
            continue
        if s["drop"]:
            c["drop_spun"] += 1
            drop_by_bin[r["bin"]] += r["n_tokens"]
            drop_by_host[r["host"]] += r["n_tokens"]
            continue
        if is_ugc or is_discussion:
            c["drop_reserved_for_register"] += 1
            continue
        docs.append(r)
    total_before_cap = sum(r["n_tokens"] for r in docs)
    cap_tokens = args.host_cap * total_before_cap
    per_host = collections.Counter()
    kept, capped = [], collections.Counter()
    for r in sorted(docs, key=lambda r: (-r["n_tokens"], r["pool_index"])):
        if per_host[r["host"]] + r["n_tokens"] > cap_tokens:
            capped[r["host"]] += r["n_tokens"]
            c["drop_host_cap"] += 1
            continue
        per_host[r["host"]] += r["n_tokens"]
        kept.append(r)
    kept.sort(key=lambda r: r["pool_index"])
    total = sum(r["n_tokens"] for r in kept)
    by_bin = collections.Counter()
    by_host = collections.Counter()
    by_group = collections.Counter()
    for r in kept:
        by_bin[r["bin"]] += r["n_tokens"]
        by_host[r["host"]] += r["n_tokens"]
        by_group[r["group"]] += r["n_tokens"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "general_docs.jsonl").open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps({k: r[k] for k in ("pool_index", "part", "local_index", "bin", "group", "host", "url", "n_tokens", "kh64")}, ensure_ascii=False) + "\n")
    report = {"created_at": C.common.now(), "counters": dict(c), "documents": len(kept), "tokens": total,
              "tokens_before_host_cap": total_before_cap, "host_cap_fraction": args.host_cap, "host_cap_tokens": cap_tokens,
              "tokens_by_bin": dict(by_bin), "tokens_by_group": dict(by_group),
              "share_by_bin": {b: v / max(1, total) for b, v in by_bin.items()},
              "top_hosts_by_tokens": {h: {"tokens": t, "share": t / max(1, total)} for h, t in by_host.most_common(25)},
              "dropped_tokens_by_bin": dict(drop_by_bin), "dropped_tokens_by_host_top": dict(drop_by_host.most_common(30)),
              "host_cap_dropped_tokens": dict(capped.most_common(20)),
              "volume_gate_ge_1_0B": total >= 1_000_000_000}
    C.write_json(out_dir / "general_pool.json", report)
    print(json.dumps({"tokens": total, "documents": len(kept), "by_bin": dict(by_bin), "counters": dict(c),
                      "ge_1_0B": report["volume_gate_ge_1_0B"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
