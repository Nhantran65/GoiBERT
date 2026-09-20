#!/usr/bin/env python3
"""§3 holdouts: the three diagnostic MLM validation sets (never trained on, decontaminated like the pool).

  (a) HPLT held-out by document: the `holdout_hlpt` documents of dedup_pool.py (0.5 % of the pool, cap 5 000)
  (b) 5 000 documents from the local FineWeb-2 vie_Latn shard `data/raw/fineweb2_vie/.../000_00000.parquet`
      (sha 337df3c3...), chosen by document-id hash, decontaminated and deduplicated against the HPLT pool
  (c) register held-out: the `holdout_register` documents of dedup_pool.py (cap 2 000)

Each set is packed into 512-token sequences (documents separated by `</s>`), truncated to MAX_SEQUENCES sequences so the
MLM diagnostic is cheap (mask seed 1234, same for every checkpoint and arm).
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

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

MAX_SEQUENCES = 4000
FW2_KEEP_MOD = 16


def pack(docs_tokens: list[np.ndarray], seq_len: int, max_sequences: int) -> np.ndarray:
    out = np.zeros((max_sequences, seq_len), dtype=np.uint32)
    buf = np.empty(seq_len * 2, dtype=np.uint32)
    filled = emitted = 0
    for toks in docs_tokens:
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


def load_pool_splits(token_dir: Path, split: str) -> tuple[list[np.ndarray], list[dict]]:
    """Return the token arrays and records of the pool parts for one split name."""
    parts, offsets = {}, {}
    for f in sorted(token_dir.glob("*.tokens.npy")):
        key = f.name.replace(".tokens.npy", "")
        parts[key] = np.load(f, mmap_mode="r")
        offsets[key] = np.load(f.with_name(key + ".offsets.npy"))
    arrays, recs = [], []
    for f in sorted(token_dir.glob("*.docs.jsonl")):
        key = f.name.replace(".docs.jsonl", "")
        if key not in parts:
            continue
        with f.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                if r.get("split") != split:
                    continue
                a, b = int(offsets[key][i]), int(offsets[key][i + 1])
                arrays.append(np.asarray(parts[key][a:b], dtype=np.uint32))
                recs.append(r)
    order = sorted(range(len(recs)), key=lambda i: recs[i]["kh64"])
    return [arrays[i] for i in order], [recs[i] for i in order]


def fineweb2_docs(parquet: Path, n_docs: int, hlpt_keys: set[str]) -> tuple[list[np.ndarray], list[dict], dict]:
    tok = common.load_tokenizer()
    bench_keys, _ = common.benchmark_key_index()
    pf = pq.ParquetFile(str(parquet))
    c = collections.Counter()
    arrays, recs = [], []
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=["id", "text", "url"])
        cols = {k: tb.column(k).to_pylist() for k in ("id", "text", "url")}
        for doc_id, text, url in zip(cols["id"], cols["text"], cols["url"]):
            c["docs_in"] += 1
            if int.from_bytes(hashlib.blake2b(str(doc_id).encode(), digest_size=8).digest(), "big") % FW2_KEEP_MOD:
                continue
            c["candidates"] += 1
            full = common.normalize(text)
            if not full or not common.is_vietnamese(full):
                c["drop_language"] += 1
                continue
            key = common.key_of(full)
            kh = hashlib.sha1(key.encode()).hexdigest()
            if kh in bench_keys or kh in hlpt_keys:
                c["drop_duplicate_or_decontam"] += 1
                continue
            seq = tok(full, add_special_tokens=False)["input_ids"]
            if not (50 <= len(seq) <= 20_000):
                c["drop_length"] += 1
                continue
            arrays.append(np.asarray(seq, dtype=np.uint32))
            recs.append({"id": str(doc_id), "url": url, "n_tokens": len(seq), "chars": len(full)})
            c["kept"] += 1
            if len(arrays) >= n_docs:
                break
        if len(arrays) >= n_docs:
            break
    return arrays, recs, dict(c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", default=str(common.DATA_DIR / "pool"))
    ap.add_argument("--token-dir", default=str(common.DATA_DIR / "tokens"))
    ap.add_argument("--out-dir", default=str(common.ART_DIR / "holdouts"))
    ap.add_argument("--fw2", default=str(common.FW2_LOCAL))
    ap.add_argument("--fw2-docs", type=int, default=5000)
    ap.add_argument("--max-sequences", type=int, default=MAX_SEQUENCES)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"stage": "prepare_holdouts", "created_at": common.now(), "max_sequences": args.max_sequences,
                      "masking": "fixed mask seed 1234 (common.VAL_SEED), 15 % masked, 80/10/10", "source_hash": common.source_hash()}

    for name, split in (("a_hlpt", "holdout_hlpt"), ("c_register", "holdout_register")):
        arrays, recs = load_pool_splits(Path(args.token_dir), split)
        arr = pack(arrays, common.MAX_LEN, args.max_sequences)
        np.save(out_dir / f"{name}.npy", arr)
        manifest[name] = {"documents": len(recs), "doc_tokens": int(sum(a.size for a in arrays)), "sequences": int(arr.shape[0]),
                          "tokens": int(arr.size), "sha256": common.sha256_array(arr)}
        print(f"[holdout] {name}: {len(recs)} docs -> {arr.shape[0]} sequences ({time.time()-t0:.0f}s)", flush=True)

    hlpt_keys: set[str] = set()
    for f in sorted(Path(common.DATA_DIR / "scan").glob("*.meta.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                hlpt_keys.add(json.loads(line)["key_h"])
    if common.sha256_file(Path(args.fw2)) != common.FW2_SHARD_SHA256:
        raise RuntimeError("FineWeb-2 shard sha256 mismatch")
    arrays, recs, counters = fineweb2_docs(Path(args.fw2), args.fw2_docs, hlpt_keys)
    arr = pack(arrays, common.MAX_LEN, args.max_sequences)
    np.save(out_dir / "b_fineweb2.npy", arr)
    manifest["b_fineweb2"] = {"documents": len(recs), "doc_tokens": int(sum(a.size for a in arrays)), "sequences": int(arr.shape[0]),
                              "tokens": int(arr.size), "sha256": common.sha256_array(arr), "counters": counters,
                              "source": {"repo": common.FW2_REPO, "commit": common.FW2_COMMIT, "file": common.FW2_SHARD_REL,
                                         "sha256": common.FW2_SHARD_SHA256, "license": common.FW2_LICENSE,
                                         "sampling": f"document id hash % {FW2_KEEP_MOD} == 0, first {args.fw2_docs} kept"}}
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    common.write_json(out_dir / "holdout_manifest.json", manifest)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("documents", "sequences", "tokens")} for k, v in manifest.items() if isinstance(v, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
