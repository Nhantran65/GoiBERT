#!/usr/bin/env python3
"""v4 §2: prepare the ViSoBERT comment corpus (research-only) for mmBERT continued pretraining.

Same D1 logic as `scripts/cdbk_dapt_pilot/prepare_corpus.py` — read the 3 pinned parquet shards, HTML-unescape/NFC/
whitespace normalisation, drop empty or letter-less rows, exact casefold dedup, text-only decontamination against the
train/dev/test text of all five benchmarks (exact key OR shared 8-word shingle, labels never read) — but tokenised with
the **mmBERT (Gemma 256K) tokenizer** (the old XLM-R token arrays are not reused), with 2 000 comments held out first and
comments packed into 512-token sequences separated by EOS (one source per sequence).

LICENCE: `phucdev/ViSoBERT` is research-only / CC BY-NC 4.0.  Everything produced here is `_NC_research_only`: not for
release, not for commercial use.  P_web/O of this pilot do not contain any of it.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import multiprocessing as mp
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v4 as C4  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

RAW = C4.ROOT / "data" / "cdbk_dapt_pilot" / "raw" / "data"
DOWNLOAD_JSON = C4.ROOT / "data" / "cdbk_dapt_pilot" / "raw" / "download.json"
OUT = C4.DATA_V4 / "visobert"
SEQ = 512
EOS = 1                     # mmBERT `</s>`
HOLDOUT_COMMENTS = 2000
SHINGLE = 8
SEED = 42
_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[^\W_]", re.U)
_TOK = None


def normalize(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", html.unescape(text or ""))).strip()


def _init_worker():
    global _TOK
    _TOK = C4.common.load_tokenizer()
    _TOK.model_max_length = int(1e30)


def _tok_batch(texts: list[str]) -> list[list[int]]:
    return _TOK(texts, add_special_tokens=False)["input_ids"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = collections.Counter()
    pinned = json.loads(DOWNLOAD_JSON.read_text(encoding="utf-8"))["files"]
    shards = sorted(RAW.glob("train-*.parquet"))
    files = {}
    for p in shards:
        meta = pinned.get(f"data/{p.name}", {})
        sha = C4.common.sha256_file(p)
        if meta.get("sha256") and meta["sha256"] != sha:
            raise RuntimeError(f"sha256 mismatch for {p.name}")
        files[p.name] = {"bytes": p.stat().st_size, "sha256": sha, "pinned_sha256": meta.get("sha256")}
    manifest: dict = {"stage": "prep_visobert", "created_at": C4.common.now(),
                      "license": C4.LICENSE_NOTE, "research_only": True, "nc_suffix_required": "_NC_research_only",
                      "corpus": {"repo": C4.VISOBERT_REPO, "revision": C4.VISOBERT_REVISION, "files": files},
                      "tokenizer": "mmBERT Gemma 256K (pinned jhu-clsp/mmBERT-base @ c5955035)", "seq_len": SEQ, "eos_id": EOS}
    # 1. read + normalise
    texts: list[str] = []
    for p in shards:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            texts.extend(pf.read_row_group(rg, columns=["text"]).column("text").to_pylist())
    c["raw_rows"] = len(texts)
    seen: set[str] = set()
    kept: list[str] = []
    keys: list[str] = []
    for t in texts:
        t = normalize(t)
        if not t:
            c["drop_empty"] += 1
            continue
        if not _ALNUM.search(t):
            c["drop_no_alnum"] += 1
            continue
        k = t.casefold()
        if k in seen:
            c["drop_exact_dup"] += 1
            continue
        seen.add(k)
        kept.append(t)
        keys.append(k)
    del texts, seen
    print(f"[prep] raw {c['raw_rows']} kept {len(kept)} ({time.time()-t0:.0f}s)", flush=True)
    # 2. text-only decontamination against the five benchmarks
    bench_keys, bench_stats = C4.common.benchmark_key_index()
    bench_shingles = set()
    for name, rows in C4.common.benchmark_texts().items():
        for r in rows:
            k = C4.common.key_of(r)
            if k:
                for s in C4.common.shingles(k):
                    bench_shingles.add(hash(s) if False else s)
    decontam_by_split = collections.Counter()
    clean_texts, clean_keys = [], []
    for t, k in zip(kept, keys):
        hit = bench_keys.get(hashlib.sha1(k.encode()).hexdigest())
        if hit is not None:
            c["drop_decontam_exact"] += 1
            decontam_by_split[hit] += 1
            continue
        if len(k.split()) >= SHINGLE:
            shared = None
            for s in C4.common.shingles(k):
                if s in bench_shingles:
                    shared = s
                    break
            if shared is not None:
                c["drop_decontam_shingle"] += 1
                decontam_by_split["shingle"] += 1
                continue
        clean_texts.append(t)
        clean_keys.append(k)
    del kept, keys
    print(f"[prep] decontam -> {len(clean_texts)} ({time.time()-t0:.0f}s)", flush=True)
    # 3. hold out 2 000 comments first (seeded shuffle), then tokenise in parallel
    order = list(range(len(clean_texts)))
    random.Random(SEED).shuffle(order)
    hold_idx = set(order[:HOLDOUT_COMMENTS])
    train_texts = [clean_texts[i] for i in range(len(clean_texts)) if i not in hold_idx]
    hold_texts = [clean_texts[i] for i in sorted(hold_idx)]
    batches = [train_texts[i:i + 512] for i in range(0, len(train_texts), 512)]
    hold_batches = [hold_texts[i:i + 512] for i in range(0, len(hold_texts), 512)]
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        train_ids = []
        for i, enc in enumerate(pool.imap(_tok_batch, batches, chunksize=4), 1):
            train_ids.extend(enc)
            if i % 2000 == 0:
                print(f"[prep] tokenised {i}/{len(batches)} batches ({time.time()-t0:.0f}s)", flush=True)
        hold_ids = []
        for enc in pool.imap(_tok_batch, hold_batches, chunksize=4):
            hold_ids.extend(enc)
    n_tokens = sum(len(x) for x in train_ids)
    n_hold_tokens = sum(len(x) for x in hold_ids)
    c["train_comments"] = len(train_ids)
    c["holdout_comments"] = len(hold_ids)
    c["train_tokens_before_pack"] = n_tokens
    c["holdout_tokens"] = n_hold_tokens

    def pack(streams: list[list[int]], path: Path, drop_tail: bool) -> int:
        total = sum(len(s) + 1 for s in streams)
        n_seq = total // SEQ
        arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint32, shape=(n_seq, SEQ))
        pos = 0
        row = 0
        buf = np.empty(SEQ, dtype=np.uint32)
        for s in streams:
            toks = np.asarray(s, dtype=np.uint32)
            start = 0
            while start < len(toks) and row < n_seq:
                take = min(SEQ - pos, len(toks) - start)
                buf[pos:pos + take] = toks[start:start + take]
                pos += take
                start += take
                if pos == SEQ:
                    arr[row] = buf
                    row += 1
                    pos = 0
            if row >= n_seq:
                break
            buf[pos] = EOS
            pos += 1
            if pos == SEQ:
                arr[row] = buf
                row += 1
                pos = 0
        arr.flush()
        del arr
        return row

    n_train_seq = pack(train_ids, out_dir / "visobert_train.npy", drop_tail=True)
    n_hold_seq = pack(hold_ids, out_dir / "visobert_holdout.npy", drop_tail=True)
    with (out_dir / "comments.jsonl").open("w", encoding="utf-8") as fh:
        for i, (t, ids) in enumerate(zip(train_texts, train_ids)):
            fh.write(json.dumps({"i": i, "split": "train", "text": t, "n_tokens": len(ids)}, ensure_ascii=False) + "\n")
        for i, (t, ids) in enumerate(zip(hold_texts, hold_ids)):
            fh.write(json.dumps({"i": i, "split": "holdout", "text": t, "n_tokens": len(ids)}, ensure_ascii=False) + "\n")
    nonpad = int(n_train_seq * SEQ)
    manifest.update({"counters": dict(c), "decontam_by_split": dict(decontam_by_split),
                     "benchmark_index": {"unique_keys": len(bench_keys), "shingles": len(bench_shingles), **{k: v for k, v in bench_stats.items() if k == "rows_per_split"}},
                     "train_sequences": n_train_seq, "train_token_positions": nonpad,
                     "holdout_sequences": n_hold_seq, "holdout_token_positions": int(n_hold_seq * SEQ),
                     "volume_gate_ge_75M": nonpad >= 75_000_000,
                     "outputs": {"train": str((out_dir / "visobert_train.npy").relative_to(C4.ROOT)),
                                 "holdout": str((out_dir / "visobert_holdout.npy").relative_to(C4.ROOT)),
                                 "comments": str((out_dir / "comments.jsonl").relative_to(C4.ROOT))},
                     "elapsed_s": round(time.time() - t0, 1)})
    C4.write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("counters", "decontam_by_split", "train_sequences", "train_token_positions",
                                               "holdout_comments", "volume_gate_ge_75M")}, ensure_ascii=False, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
