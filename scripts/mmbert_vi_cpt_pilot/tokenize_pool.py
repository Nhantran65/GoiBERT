#!/usr/bin/env python3
"""Data gate, stage 3 (§3): tokenise the deduplicated pool with the mmBERT (Gemma 256K) tokenizer.

Reads the stage-1 `<shard>_<part>.docs.jsonl` files and keeps the documents listed in `pool/pool.jsonl` (plus the two
HPLT holdout files), in exactly the locked stage-2 order.  Documents whose exact mmBERT token length is outside
[50, 20 000] are dropped here (the stage-1 char prefilter is only approximate) and counted.  Tokens are written per part
as a flat uint32 array with per-document offsets, so later stages can memory-map them; a token count is produced for
every document, which is what the §3 gate (>= 1.2 B unique tokens, register >= 50 M / 240 M) is measured on.
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
import common  # noqa: E402

import numpy as np  # noqa: E402

MIN_TOKENS, MAX_TOKENS = 50, 20_000
_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        _TOK = common.load_tokenizer()
        _TOK.model_max_length = int(1e9)
    return _TOK


def work(job) -> dict:
    docs_path, out_prefix, keep = job
    tok = _tok()
    ids: list[np.ndarray] = []
    rows: list[dict] = []
    stats = collections.Counter()
    batch_texts: list[str] = []
    batch_meta: list[dict] = []

    def flush() -> None:
        if not batch_texts:
            return
        enc = tok(batch_texts, add_special_tokens=False)["input_ids"]
        for meta, seq in zip(batch_meta, enc):
            stats["docs_tokenized"] += 1
            if len(seq) < MIN_TOKENS:
                stats["drop_too_short"] += 1
                continue
            if len(seq) > MAX_TOKENS:
                stats["drop_too_long"] += 1
                continue
            arr = np.asarray(seq, dtype=np.uint32)
            rows.append({**meta, "n_tokens": int(arr.size)})
            ids.append(arr)
            stats["docs_kept"] += 1
        batch_texts.clear()
        batch_meta.clear()

    with Path(docs_path).open(encoding="utf-8") as fh:
        for line in fh:
            stats["docs_read"] += 1
            o = json.loads(line)
            meta = keep.get(o["id"])
            if meta is None:
                stats["skipped_not_in_pool"] += 1
                continue
            batch_texts.append(o["text"])
            batch_meta.append({**meta, "chars": len(o["text"])})
            if len(batch_texts) >= 64:
                flush()
    flush()

    out_prefix = Path(out_prefix)
    flat = np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint32)
    np.save(out_prefix.with_suffix(".tokens.npy"), flat)
    offsets = np.concatenate([[0], np.cumsum([r["n_tokens"] for r in rows])]).astype(np.int64) if rows else np.zeros(1, np.int64)
    np.save(out_prefix.with_suffix(".offsets.npy"), offsets)
    with out_prefix.with_suffix(".docs.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"part": Path(docs_path).name, "out": str(out_prefix), "tokens": int(flat.size), "docs": len(rows), "stats": dict(stats)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir", default=str(common.DATA_DIR / "scan"))
    ap.add_argument("--pool-dir", default=str(common.DATA_DIR / "pool"))
    ap.add_argument("--out-dir", default=str(common.DATA_DIR / "tokens"))
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    t0 = time.time()
    scan, pool_dir, out_dir = Path(args.scan_dir), Path(args.pool_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep: dict[str, dict] = {}
    for name in ("pool.jsonl", "holdout_hlpt.jsonl"):
        with (pool_dir / name).open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                if r["id"] in keep:
                    raise RuntimeError(f"duplicate pool id {r['id']}")
                keep[r["id"]] = {"pool_index": i, "split": r.get("split", "pool"), "group": r["group"], "bin": r["bin"], "shard": r["shard"],
                                 "op": r.get("op"), "re": r.get("re"), "id_reg": r.get("id_reg"), "kh64": r["kh64"], "url": r.get("url")}
    print(f"[tok] pool documents to tokenise: {len(keep)}", flush=True)

    jobs, parts_index = [], []
    for shard in args.shards:
        for docs_path in sorted(scan.glob(f"{shard}_*.docs.jsonl")):
            prefix = out_dir / docs_path.name.replace(".docs.jsonl", "")
            jobs.append((str(docs_path), str(prefix), keep))
            parts_index.append({"shard": shard, "part": docs_path.name.replace(".docs.jsonl", "")})
    results = []
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(work, jobs), 1):
            results.append(res)
            if i % 16 == 0:
                print(f"[tok] {i}/{len(jobs)} parts ({time.time()-t0:.0f}s)", flush=True)

    stats = collections.Counter()
    for r in results:
        stats.update(r["stats"])
        stats["tokens_kept"] += r["tokens"]
    manifest = {"stage": "tokenize_pool", "created_at": common.now(), "tokenizer": common.tokenizer_manifest(),
                "bounds": {"min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS}, "counters": dict(stats),
                "pool_documents": len(keep), "workers": args.workers, "parts": parts_index,
                "part_tokens": {Path(r["out"]).name: r["tokens"] for r in results},
                "source_hash": common.source_hash(), "elapsed_s": round(time.time() - t0, 1)}
    common.write_json(pool_dir / "tokenize_manifest.json", manifest)
    print(json.dumps({"tokens_kept": stats["tokens_kept"], "docs_kept": stats["docs_kept"],
                      "docs_read": stats["docs_read"], "drop_too_short": stats["drop_too_short"], "drop_too_long": stats["drop_too_long"],
                      "elapsed_s": manifest["elapsed_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
