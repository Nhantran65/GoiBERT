#!/usr/bin/env python3
"""§3 source preparation: Vietnamese Wikipedia slice used as the 5 % component of every arm.

Pinned: `wikimedia/wikipedia`, config `20231101.vi`, dataset revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa, parquet
`20231101.vi/train/0000.parquet` (the first of four auto-converted shards), licence CC BY-SA 4.0.  The file is filtered
like the HPLT pool (content Vietnamese, length, exact/near dedup inside Wikipedia, exact dedup against every scanned
HPLT document, benchmark decontamination), tokenised with the mmBERT tokenizer and written in the same
`<prefix>.tokens.npy` / `<prefix>.offsets.npy` / `<prefix>.docs.jsonl` layout as the pool parts, under
`data/mmbert_vi_cpt_pilot/wiki/`.  Only `pool_index < 0` documents are produced, so the wiki slice can never enter the
HPLT holdout split.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import scan_hplt  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

WIKI_DATASET_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
TARGET_TOKENS = 120_000_000       # ~10x the 12.5 M needed per arm
MIN_CHARS, MAX_CHARS = 200, 120_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=str(common.WIKI_DIR / "wiki_vi_20231101_train_0000.parquet"))
    ap.add_argument("--scan-dir", default=str(common.DATA_DIR / "scan"))
    ap.add_argument("--out-dir", default=str(common.DATA_DIR / "wiki"))
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = common.load_tokenizer()
    bench_keys, _ = common.benchmark_key_index()
    bench_shingles = common.benchmark_shingle_array()

    # exact keys of every scanned HPLT document (cross-source exact dedup, plan §3)
    hlpt_keys: set[str] = set()
    for f in sorted(Path(args.scan_dir).glob("*.meta.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                hlpt_keys.add(json.loads(line)["key_h"])
    print(f"[wiki] HPLT keys {len(hlpt_keys)} ({time.time()-t0:.0f}s)", flush=True)

    pf = pq.ParquetFile(args.parquet)
    c = collections.Counter()
    seen_keys: set[str] = set()
    seen_sh: set[int] = set()
    docs, ids_arr = [], []
    tokens_total = 0
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=["id", "url", "text"])
        cols = {k: tb.column(k).to_pylist() for k in ("id", "url", "text")}
        for doc_id, url, text in zip(cols["id"], cols["url"], cols["text"]):
            c["docs_in"] += 1
            full = common.normalize(text)
            if not full:
                c["drop_empty"] += 1
                continue
            if not (MIN_CHARS <= len(full) <= MAX_CHARS):
                c["drop_length"] += 1
                continue
            if not common.is_vietnamese(full):
                c["drop_not_vi"] += 1
                continue
            key = common.key_of(full)
            kh = hashlib.sha1(key.encode()).hexdigest()
            if kh in bench_keys:
                c["drop_decontam_exact"] += 1
                continue
            if kh in hlpt_keys:
                c["drop_dup_hlpt"] += 1
                continue
            if kh in seen_keys:
                c["drop_dup_wiki"] += 1
                continue
            sh = common.sampled_shingle_hashes(key)
            if len(sh) >= 3 and sum(h in seen_sh for h in sh) / len(sh) >= 0.5:
                c["drop_near_dup"] += 1
                continue
            kept_segments, n_segments, n_hit = scan_hplt.hit_segments(full, bench_shingles)
            if n_segments and n_hit / n_segments > 0.20:
                c["drop_decontam_doc"] += 1
                continue
            if n_hit:
                c["decontam_segments_removed"] += n_hit
                full = common.normalize(" ".join(kept_segments))
                if len(full) < MIN_CHARS:
                    c["drop_decontam_short"] += 1
                    continue
                key = common.key_of(full)
                kh = hashlib.sha1(key.encode()).hexdigest()
                sh = common.sampled_shingle_hashes(key)
            seq = tok(full, add_special_tokens=False)["input_ids"]
            if len(seq) < 50:
                c["drop_too_short"] += 1
                continue
            if len(seq) > 20_000:
                c["drop_too_long"] += 1
                continue
            seen_keys.add(kh)
            seen_sh.update(sh)
            idx = len(docs)
            docs.append({"id": str(doc_id), "url": url, "title": None, "chars": len(full), "n_tokens": len(seq),
                         "pool_index": -1, "split": "pool", "group": "wiki", "bin": "wiki", "source": "wiki",
                         "kh64": int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")})
            ids_arr.append(np.asarray(seq, dtype=np.uint32))
            tokens_total += len(seq)
            c["kept"] += 1
            if tokens_total >= args.target_tokens:
                break
        if tokens_total >= args.target_tokens:
            break
    flat = np.concatenate(ids_arr) if ids_arr else np.zeros(0, dtype=np.uint32)
    np.save(out_dir / "wiki.tokens.npy", flat)
    np.save(out_dir / "wiki.offsets.npy", np.concatenate([[0], np.cumsum([r["n_tokens"] for r in docs])]).astype(np.int64) if docs else np.zeros(1, np.int64))
    with (out_dir / "wiki.docs.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(docs):
            fh.write(json.dumps({**r, "part": "wiki", "local_index": i}, ensure_ascii=False) + "\n")
    with (out_dir / "wiki_audit_excerpts.jsonl").open("w", encoding="utf-8") as fh:
        for r, tokens_arr in zip(docs, ids_arr):
            excerpt = tok.decode([int(x) for x in tokens_arr[:400]])
            fh.write(json.dumps({k: r[k] for k in ("id", "url", "chars", "n_tokens")} | {"bin": "wiki", "group": "wiki", "excerpt": excerpt}, ensure_ascii=False) + "\n")
    manifest = {"stage": "prepare_wiki", "created_at": common.now(),
                "source": {"repo": common.WIKI_REPO, "config": common.WIKI_CONFIG, "dataset_revision": WIKI_DATASET_REVISION,
                           "file": "20231101.vi/train/0000.parquet (auto-converted parquet, 4 shards total; only shard 0000 used)",
                           "sha256": common.sha256_file(Path(args.parquet)), "license": common.WIKI_LICENSE},
                "filters": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS, "token_bounds": [50, 20_000],
                            "dedup": "exact NFC-casefold inside Wikipedia and against every scanned HPLT document; sampled 8-word shingles (hash%16==0) near-dup inside Wikipedia",
                            "decontam": "same benchmark rule as the HPLT pool (exact key or shared 8-word shingle; >20 % hit segments drops the document)"},
                "counters": dict(c), "documents": len(docs), "tokens": int(flat.size),
                "hlpt_keys_compared": len(hlpt_keys), "source_hash": common.source_hash(), "elapsed_s": round(time.time() - t0, 1)}
    common.write_json(out_dir / "wiki_manifest.json", manifest)
    print(json.dumps({"docs": len(docs), "tokens": int(flat.size), "counters": dict(c), "elapsed_s": manifest["elapsed_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
