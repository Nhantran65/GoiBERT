#!/usr/bin/env python3
"""v2 §2 "vá 2": assemble the register bucket = UGC paragraphs (whitelist platforms) ∪ `discussion` (ID >= 0.5).

  * UGC documents come from `ugc_extract.py` (all whitelist-platform documents in bins 7-9, paragraph-filtered, already
    through vá 1); 2 000 of them (deterministic hash order, seed 42) are held out as the fourth MLM holdout set.
  * `discussion` documents are the v1 pool documents with `ID >= 0.5` that survived vá 1; the plan caps them at 30 % of the
    register bucket, which allows at most 0.4286 x the UGC token count — the cap is applied by deterministic hash order.

Outputs: artifacts/mmbert_vi_cpt_pilot/v2/register/{register.docs.jsonl, tokens.npy, offsets.npy, manifest.json} and the
holdout `artifacts/mmbert_vi_cpt_pilot/v2/holdouts/d_ugc.npy`.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C  # noqa: E402

import numpy as np  # noqa: E402

UGC_HOLDOUT = 2000
DISCUSSION_MAX_SHARE = 0.30
MAX_SEQUENCES = 4000


def load_scores() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for f in sorted((C.DATA_V2 / "spun_scores").glob("*.jsonl")):
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
    ap.add_argument("--ugc-dir", default=str(C.DATA_V2 / "ugc"))
    ap.add_argument("--out-dir", default=str(C.ART_V2 / "register"))
    ap.add_argument("--holdout-dir", default=str(C.ART_V2 / "holdouts"))
    args = ap.parse_args()
    ugc_dir = Path(args.ugc_dir)
    ugc_tokens = np.load(ugc_dir / "ugc.tokens.npy", mmap_mode="r")
    ugc_offsets = np.load(ugc_dir / "ugc.offsets.npy")
    ugc_docs = [json.loads(l) for l in (ugc_dir / "ugc.docs.jsonl").open(encoding="utf-8")]
    by_host = {h: g for g, hosts in C.UGC_GROUP.items() for h in hosts}

    def ugc_group(host: str) -> str:
        for g, hosts in C.UGC_GROUP.items():
            if C.host_matches(host, hosts):
                return g
        return "forum_qa" if ("thuvienphapluat" in host or "luatminhkhue" in host) else "review_commerce"

    order = sorted(range(len(ugc_docs)), key=lambda i: (ugc_docs[i]["kh64"], ugc_docs[i]["id"]))
    hold_idx = set(order[:UGC_HOLDOUT])
    hold_arrays = [np.asarray(ugc_tokens[int(ugc_offsets[i]):int(ugc_offsets[i + 1])], dtype=np.uint32) for i in sorted(hold_idx)]
    hold = pack(hold_arrays, C.common.MAX_LEN, MAX_SEQUENCES)
    hold_dir = Path(args.holdout_dir)
    hold_dir.mkdir(parents=True, exist_ok=True)
    np.save(hold_dir / "d_ugc.npy", hold)

    register_arrays: list[np.ndarray] = []
    register_rows: list[dict] = []
    for i in sorted(set(range(len(ugc_docs))) - hold_idx):
        d = ugc_docs[i]
        a, b = int(ugc_offsets[i]), int(ugc_offsets[i + 1])
        register_arrays.append(np.asarray(ugc_tokens[a:b], dtype=np.uint32))
        register_rows.append({"source": "ugc", "group": ugc_group(d["host"]), "host": d["host"], "bin": d["bin"],
                              "url": d.get("url"), "id": d.get("id"), "n_tokens": int(b - a)})
    ugc_tokens_total = int(sum(r["n_tokens"] for r in register_rows))

    # discussion (ID >= 0.5) from the v1 pool that survived vá 1, capped at 30 % of the bucket
    scores = load_scores()
    pool = C.pool_table()
    disc = [r for r in pool if r["split"] == "pool" and r["group"] == "register_discussion" and r["bin"] in ("7", "8", "9")
            and not (scores.get(r["pool_index"], {}).get("drop", True))]
    disc.sort(key=lambda r: (r["kh64"], r["pool_index"]))
    allow = int(DISCUSSION_MAX_SHARE / (1 - DISCUSSION_MAX_SHARE) * ugc_tokens_total)
    disc_kept, acc = [], 0
    for r in disc:
        if acc + r["n_tokens"] > allow:
            break
        disc_kept.append(r)
        acc += r["n_tokens"]
    disc_by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for r in disc_kept:
        disc_by_part[r["part"]].append(r)
    disc_arrays: list[tuple[dict, np.ndarray]] = []
    for part, rows in disc_by_part.items():
        offsets = np.load(C.DATA_V1 / "tokens" / f"{part}.offsets.npy")
        tokens = np.load(C.DATA_V1 / "tokens" / f"{part}.tokens.npy", mmap_mode="r")
        for r in rows:
            a, b = int(offsets[r["local_index"]]), int(offsets[r["local_index"] + 1])
            disc_arrays.append((r, np.asarray(tokens[a:b], dtype=np.uint32)))
    for r, arr in disc_arrays:
        register_arrays.append(arr)
        register_rows.append({"source": "discussion", "group": "discussion", "host": r["host"], "bin": r["bin"], "url": r["url"],
                              "id": None, "pool_index": r["pool_index"], "part": r["part"], "local_index": r["local_index"],
                              "n_tokens": int(arr.size)})
    flat = np.concatenate(register_arrays) if register_arrays else np.zeros(0, dtype=np.uint32)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tokens.npy", flat)
    np.save(out_dir / "offsets.npy", np.concatenate([[0], np.cumsum([r["n_tokens"] for r in register_rows])]).astype(np.int64))
    with (out_dir / "register.docs.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(register_rows):
            fh.write(json.dumps({**r, "i": i}, ensure_ascii=False) + "\n")
    tok_by_host = collections.Counter()
    tok_by_group = collections.Counter()
    for r in register_rows:
        tok_by_host[r["host"]] += r["n_tokens"]
        tok_by_group[r["group"]] += r["n_tokens"]
    total = int(flat.size)
    manifest = {"created_at": C.common.now(), "documents": len(register_rows), "tokens": total,
                "tokens_by_group": dict(tok_by_group), "tokens_by_host_top": dict(tok_by_host.most_common(30)),
                "ugc_tokens": ugc_tokens_total, "discussion_tokens": total - ugc_tokens_total,
                "discussion_share": (total - ugc_tokens_total) / max(1, total), "discussion_cap_share": DISCUSSION_MAX_SHARE,
                "ugc_documents_holdout": len(hold_idx), "ugc_holdout_sequences": int(hold.shape[0]),
                "volume_gate_ge_50M": total >= 50_000_000, "volume_gate_ge_240M": total >= 240_000_000,
                "whitelist_reasons": C.UGC_WHITELIST, "excluded_hosts": sorted(C.UGC_EXCLUDED_HOSTS)}
    C.write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("documents", "tokens", "tokens_by_group", "ugc_tokens", "discussion_tokens",
                                               "discussion_share", "volume_gate_ge_50M", "volume_gate_ge_240M")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
