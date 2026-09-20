#!/usr/bin/env python3
"""v3 §3.2: build the informal-register bucket (HPLT only) + the 2 000-document informal MLM holdout.

Candidates (all from the v1 pool, i.e. already filtered, deduplicated and benchmark-decontaminated):
  `discussion`  documents with web-register.ID >= 0.3 and an informal cue                 (§3.2b)
  `ugc_host`    documents from the v2 UGC whitelist hosts with an informal cue            (§3.2a/c)
  `v2_dropped`  documents the v2 S2 filter dropped, with a strong informal signal         (§3.2a)
  `bin56`       documents from HPLT bins 5-6 with an informal cue (optional, shards 5_1/6_1)

Because the v3 §3.1 side-effect gate failed (the retrained S2 still drops >15 % of hand-labelled real-informal
documents), the plan's remedy applies: **the informal bucket uses the S1-only rule** (`S1 >= reference p5`), while the
general pool keeps the S1+S2 rule.  A small explicit spam-host guard (gambling/porn hosts seen in the v1/v2 audits) is
applied on top and recorded.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v3 as C3  # noqa: E402
import spun_filter as S  # noqa: E402

import numpy as np  # noqa: E402

OUT = C3.DATA_V3 / "informal_bucket"
HOLDOUT_DOCS = 2000
MAX_SEQUENCES = 4000
TOKEN_TARGET = 160_000_000

SPAM_HOSTS = {  # explicit guard: gambling/porn hosts observed in the v1/v2 manual audits
    "911win.in", "mtot88.com", "kasynoonlinemy.com", "rikvip2.club", "789bet.in", "w388hanoi.com", "1gom.net",
    "bamboo21.win", "clipnong.tv", "phimxec.pro", "javhayz.com", "xxphim.pro", "thongtinnhacai.org",
    "nhacaibongda.com", "caulototheothu.com", "tipkeomoingay.club", "muabanvietnam.com.vn", "s666.diy",
    "e5e.microspaze.com", "outerstats.com", "xoicunho.vn", "linkhay.com",
}


def load_features() -> list[dict]:
    rows = []
    for f in sorted((C3.DATA_V3 / "informal_scan").glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
    return rows


def load_v3_scores() -> dict[int, dict]:
    out = {}
    for f in sorted((C3.DATA_V3 / "spun_scores_v3").glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[int(r["pool_index"])] = r
    return out


def pack(arrays: list[np.ndarray], seq_len: int, max_sequences: int) -> np.ndarray:
    out = np.zeros((max_sequences, seq_len), dtype=np.uint32)
    buf = np.empty(seq_len * 2, dtype=np.uint32)
    filled = emitted = 0
    for toks in arrays:
        pos = 0
        while pos < toks.size and emitted < max_sequences:
            take = min(seq_len - filled, toks.size - pos)
            buf[filled:filled + take] = toks[pos:pos + take]
            filled += take
            pos += take
            if filled == seq_len:
                out[emitted] = buf[:seq_len]
                emitted += 1
                filled = 0
        if emitted >= max_sequences:
            break
        if filled == seq_len:
            out[emitted] = buf[:seq_len]
            emitted += 1
            filled = 0
        buf[filled] = 1
        filled += 1
    return out[:emitted]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-target", type=int, default=TOKEN_TARGET)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    thr = C3.read_json(C3.ART_V3 / "spun_filter_v3" / "thresholds.json")
    s1_thr = float(thr["threshold_s1_p5"])
    features = load_features()
    scores = load_v3_scores()
    id_reg = C3.id_register_by_pool_index()
    parts = {r["pool_index"]: r for r in C3.C2.pool_table() if r["split"] == "pool"}
    ugc_hosts = set(C3.C2.UGC_WHITELIST)
    c = collections.Counter()
    cand: dict[str, list[dict]] = collections.defaultdict(list)
    for f in features:
        pi = f["pool_index"]
        p = parts.get(pi)
        s = scores.get(pi)
        if p is None or s is None:
            c["no_meta"] += 1
            continue
        if s["s1"] is None or s["s1"] < s1_thr:
            c["drop_s1"] += 1                     # S1-only rule for the informal bucket (plan §3.1 remedy)
            continue
        if C3.C2.host_matches(f["host"], SPAM_HOSTS):
            c["drop_spam_host"] += 1
            continue
        if not f["informal_cue"]:
            c["drop_no_cue"] += 1
            continue
        if f["group"] == "register_discussion" and id_reg.get(pi, 0.0) >= 0.3:
            src = "discussion"
        elif C3.C2.host_matches(f["host"], ugc_hosts):
            src = "ugc_host"
        elif s["drop"]:
            src = "v2_dropped"
        else:
            src = "general_cue"
        # informal-signal density decides the ranking inside every source (documented selection rule)
        density = (f["n_teencode"] + 3 * f["n_emoji"] + 0.2 * min(f["n_pronoun"], 40)) / max(1.0, f["n_tokens"] / 1000)
        cand[src].append({**f, "s1": s["s1"], "s2": s["s2"], "id_reg": id_reg.get(pi, 0.0), "density": density, "source": src,
                          "part": p["part"], "local_index": p["local_index"], "kh64": p["kh64"], "url": p["url"]})
    # rank the union: source priority (discussion > UGC hosts > v2-dropped > general cue), then informal-signal density
    priority = {"discussion": 0, "ugc_host": 1, "v2_dropped": 2, "general_cue": 3}
    merged = [r for name in ("discussion", "ugc_host", "v2_dropped", "general_cue") for r in cand.get(name, [])]
    merged.sort(key=lambda r: (priority[r["source"]], -r["density"], r["pool_index"]))
    kept: list[dict] = []
    total = 0
    per_source = collections.Counter()
    for r in merged:
        if total >= args.token_target:
            break
        kept.append(r)
        total += r["n_tokens"]
        per_source[r["source"]] += r["n_tokens"]
    kept.sort(key=lambda r: (r["kh64"], r["pool_index"]))
    hold_idx = set(range(min(HOLDOUT_DOCS, len(kept))))
    hold_rows = [kept[i] for i in sorted(hold_idx)]
    hold_arrays = []
    for r in hold_rows:
        offsets = np.load(C3.C2.DATA_V1 / "tokens" / f"{r['part']}.offsets.npy")
        toks = np.load(C3.C2.DATA_V1 / "tokens" / f"{r['part']}.tokens.npy", mmap_mode="r")
        hold_arrays.append(np.asarray(toks[int(offsets[r["local_index"]]):int(offsets[r["local_index"] + 1])], dtype=np.uint32))
    hold_dir = C3.ART_V3 / "holdouts"
    hold_dir.mkdir(parents=True, exist_ok=True)
    hold = pack(hold_arrays, C3.common.MAX_LEN, MAX_SEQUENCES)
    np.save(hold_dir / "c_informal.npy", hold)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "informal_docs.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(kept):
            fh.write(json.dumps({k: r[k] for k in ("pool_index", "part", "local_index", "kh64", "bin", "group", "host", "url",
                                                   "n_tokens", "n_teencode", "n_emoji", "n_pronoun", "id_reg", "s1", "s2")}
                                | {"i": i, "held_out": i in hold_idx}, ensure_ascii=False) + "\n")
    by_host = collections.Counter()
    by_group = collections.Counter()
    for r in kept:
        by_host[r["host"]] += r["n_tokens"]
        by_group[r["group"]] += r["n_tokens"]
    manifest = {"created_at": C3.common.now(), "documents": len(kept), "tokens": int(sum(r["n_tokens"] for r in kept)),
                "tokens_by_source": dict(per_source), "tokens_by_group": dict(by_group),
                "tokens_by_host_top": dict(by_host.most_common(30)), "counters": dict(c),
                "candidates_available": {k: {"documents": len(v), "tokens": int(sum(x["n_tokens"] for x in v))} for k, v in cand.items()},
                "rule": "informal bucket = S1-only (S1 >= reference p5) + informal cue + spam-host guard; general pool keeps S1+S2",
                "s1_threshold_p5": s1_thr, "spam_host_guard": sorted(SPAM_HOSTS),
                "holdout": {"documents": len(hold_rows), "sequences": int(hold.shape[0]), "tokens": int(hold.size),
                            "file": str((hold_dir / "c_informal.npy").relative_to(C3.ROOT))},
                "volume_gate_ge_75M": int(sum(r["n_tokens"] for r in kept)) >= 75_000_000,
                "volume_gate_ge_37_5M": int(sum(r["n_tokens"] for r in kept)) >= 37_500_000}
    C3.write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("documents", "tokens", "tokens_by_source", "counters", "volume_gate_ge_75M",
                                               "candidates_available")}, ensure_ascii=False, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
