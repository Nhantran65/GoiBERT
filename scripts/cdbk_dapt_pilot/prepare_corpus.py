#!/usr/bin/env python3
"""D1 corpus prep: phucdev/ViSoBERT (Vietnamese social-media comments) -> packed 512-token XLM-R sequences.

Steps (all counts go to data/cdbk_dapt_pilot/manifest.json):
  1. read the 3 pinned parquet shards; html-unescape, NFC, strip, collapse whitespace; drop empty / no letter-or-digit.
  2. exact dedup on a casefolded key (obvious duplicates only; no MinHash).
  3. text-only decontamination against the 5 downstream benchmarks (train/dev/test text columns read directly from the
     raw files, no label column is read): drop a doc whose key equals a benchmark key, or that shares an 8-word shingle
     with any benchmark text. Only counts and set hashes are logged.
  4. shuffle docs (seed 42), tokenize (XLM-R-large @ pinned revision, no special tokens, EOS appended per doc),
     split *by document* into holdout (first docs up to HOLDOUT_SEQ*(SEQ-1) tokens, i.e. ~1% capped at ~1M tokens) and
     train, then pack each into [n, 512] uint32 rows (BOS + stream of docs separated by EOS; docs may straddle rows).
Packing follows scripts/pilot_b/prepare_data.py.  Nothing here reads FineWeb-2.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "cdbk_pilot"))
from common import TEACHER_TOKENIZER_COMMIT, TEACHER_TOKENIZER_REPO, now, sha256_file  # noqa: E402

RAW_DIR = ROOT / "data" / "cdbk_dapt_pilot" / "raw"
OUT_DIR = ROOT / "data" / "cdbk_dapt_pilot"
CORPUS_REPO = "phucdev/ViSoBERT"
CORPUS_REVISION = "74dbc64e92c627c70670dd91344b776c8500a104"
SEQ = 512
BOS, EOS = 0, 2
HOLDOUT_SEQ = 2000          # ~1.02M token positions, i.e. the 1 %-capped-at-1M-tokens holdout of the plan
SHINGLE = 8
SEED = 42
N_WORKERS = 32
TOKENS_TARGET = 300_023_808  # 2289 global batches of 256 x 512, as in Pilot B / the plan (<= 1 batch over 300M)
REVIEW_KW = re.compile(r"\b(quán|món|ngon|nhà hàng|phục vụ|đồ ăn|khách sạn|shop|sản phẩm|giao hàng|đặt hàng|review|đánh giá|ăn uống|giá cả|chất lượng)\b", re.I)
_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[^\W_]", re.U)


def normalize(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    return _WS.sub(" ", text).strip()


def key_of(text: str) -> str:
    return text.casefold()


def shingles(key: str) -> list[str]:
    words = key.split()
    if len(words) < SHINGLE:
        return []
    return [" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)]


def benchmark_texts() -> dict[str, list[str]]:
    """Text columns only; label columns are never read."""
    out: dict[str, list[str]] = {}
    for split in ("train", "dev", "test"):
        out[f"vsfc/{split}"] = [l.rstrip("\n") for l in (ROOT / "data/raw/uit-vsfc" / split / "sents.txt").open(encoding="utf-8")]
    for split in ("Train", "Dev", "Test"):
        with (ROOT / "data/raw/uit-visfd" / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
            out[f"visfd/{split}"] = [row["comment"] for row in csv.DictReader(fh)]
        for dom in ("Restaurant", "Hotel"):
            with (ROOT / "data/raw/uit-absa-public-candidate/dataset" / f"UIT-ABSA-{dom}" / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
                out[f"absa_{dom.lower()}/{split}"] = [row["Review"] for row in csv.DictReader(fh)]
    return out


def _init_worker():
    global _TOK
    _TOK = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER_REPO, revision=TEACHER_TOKENIZER_COMMIT, local_files_only=True, use_fast=True)
    _TOK.model_max_length = int(1e30)


def _tok_batch(texts: list[str]) -> list[list[int]]:
    return _TOK(texts, add_special_tokens=False)["input_ids"]


def pack(stream: np.ndarray, path: Path) -> int:
    n_seq = len(stream) // (SEQ - 1)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint32, shape=(n_seq, SEQ))
    arr[:, 0] = BOS
    arr[:, 1:] = stream[: n_seq * (SEQ - 1)].reshape(n_seq, SEQ - 1)
    arr.flush()
    del arr
    return n_seq


def main() -> int:
    t0 = time.time()
    manifest: dict = {"corpus": {"repo": CORPUS_REPO, "revision": CORPUS_REVISION, "description": "ViSoBERT pre-training corpus: Vietnamese social-media texts (Facebook posts, TikTok/YouTube comments, 2020-2022); terms: research purposes only (uitnlp/visobert card), CC-BY-NC-4.0 (SEACrowd/visobert)", "domain_note": "broad social media, NOT restaurant/food reviews; domain mismatch vs UIT-ABSA Restaurant is expected and recorded"}, "tokenizer": {"repo": TEACHER_TOKENIZER_REPO, "revision": TEACHER_TOKENIZER_COMMIT}, "seq_len": SEQ, "seed": SEED}
    shards = sorted((RAW_DIR / "data").glob("train-*.parquet"))
    manifest["corpus"]["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in shards}

    # 1. read + normalize
    texts: list[str] = []
    for p in shards:
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            texts.extend(pf.read_row_group(rg, columns=["text"]).column("text").to_pylist())
    n_raw = len(texts)
    print(f"[prep] raw rows {n_raw} ({time.time()-t0:.0f}s)", flush=True)
    n_empty = n_noalnum = 0
    seen: set[str] = set()
    n_dup = 0
    kept: list[str] = []
    for t in texts:
        t = normalize(t)
        if not t:
            n_empty += 1; continue
        if not _ALNUM.search(t):
            n_noalnum += 1; continue
        k = key_of(t)
        if k in seen:
            n_dup += 1; continue
        seen.add(k)
        kept.append(t)
    del texts
    print(f"[prep] after normalize/dedup {len(kept)} (empty {n_empty}, no_alnum {n_noalnum}, exact_dup {n_dup}) ({time.time()-t0:.0f}s)", flush=True)

    # 3. text-only decontamination
    bench = benchmark_texts()
    bench_keys: set[str] = set()
    bench_shingles: set[str] = set()
    bench_counts = {}
    for name, rows in bench.items():
        keys = [key_of(normalize(r)) for r in rows]
        bench_counts[name] = len(keys)
        bench_keys.update(k for k in keys if k)
        for k in keys:
            bench_shingles.update(shingles(k))
    n_decontam_exact = n_decontam_shingle = 0
    clean: list[str] = []
    for t in kept:
        k = key_of(t)
        if k in bench_keys:
            n_decontam_exact += 1; continue
        if len(k) >= 20 and any(s in bench_shingles for s in shingles(k)):
            n_decontam_shingle += 1; continue
        clean.append(t)
    del kept, seen
    manifest["decontamination"] = {
        "method": "exact casefolded-NFC key match OR any shared 8-word shingle; text columns only (sents.txt / comment / Review), labels never read; test text accessed solely for this overlap removal",
        "benchmark_rows": bench_counts, "benchmark_unique_keys": len(bench_keys), "benchmark_shingles": len(bench_shingles),
        "benchmark_key_set_sha256": hashlib.sha256("\n".join(sorted(bench_keys)).encode()).hexdigest(),
        "dropped_exact": n_decontam_exact, "dropped_shingle": n_decontam_shingle,
    }
    print(f"[prep] decontam dropped exact {n_decontam_exact} shingle {n_decontam_shingle} -> {len(clean)} docs ({time.time()-t0:.0f}s)", flush=True)

    # descriptive stats + inspection sample
    rng = random.Random(SEED)
    sample = rng.sample(clean, 50)
    words = np.fromiter((len(t.split()) for t in clean), dtype=np.int32, count=len(clean))
    manifest["docs"] = {"raw": n_raw, "dropped_empty": n_empty, "dropped_no_alnum": n_noalnum, "dropped_exact_dup": n_dup,
                        "after_dedup": n_raw - n_empty - n_noalnum - n_dup, "after_decontam": len(clean),
                        "words_mean": float(words.mean()), "words_median": float(np.median(words)), "words_p95": float(np.percentile(words, 95)), "words_max": int(words.max()),
                        "review_keyword_fraction": float(np.mean([bool(REVIEW_KW.search(t)) for t in rng.sample(clean, 200000)])),
                        "review_keyword_regex": REVIEW_KW.pattern}
    manifest["inspection_sample_50"] = sample

    # 4. shuffle by document, tokenize, split by document, pack
    rng.shuffle(clean)
    tok_docs: list[list[int]] = []
    batches = [clean[i:i + 2000] for i in range(0, len(clean), 2000)]
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for bi, ids_list in enumerate(pool.imap(_tok_batch, batches, chunksize=4)):
            tok_docs.extend(ids_list)
            if bi % 500 == 0:
                print(f"[prep] tokenized {len(tok_docs)}/{len(clean)} ({time.time()-t0:.0f}s)", flush=True)
    del clean
    lengths = np.fromiter((len(d) for d in tok_docs), dtype=np.int64, count=len(tok_docs))
    n_tokens_docs = int(lengths.sum())
    holdout_target = HOLDOUT_SEQ * (SEQ - 1)
    cum = np.cumsum(lengths + 1)  # +1 EOS per doc
    n_holdout_docs = int(np.searchsorted(cum, holdout_target, side="right")) + 1
    n_holdout_docs = min(n_holdout_docs, max(1, int(0.01 * len(tok_docs))))
    def stream(docs):
        out = np.empty(int(sum(len(d) + 1 for d in docs)), dtype=np.uint32)
        c = 0
        for d in docs:
            out[c:c + len(d)] = d; out[c + len(d)] = EOS; c += len(d) + 1
        return out
    hold = stream(tok_docs[:n_holdout_docs]); train = stream(tok_docs[n_holdout_docs:])
    del tok_docs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_hold_seq = pack(hold, OUT_DIR / "visobert_xlmr_512_holdout.npy")
    n_train_seq = pack(train, OUT_DIR / "visobert_xlmr_512_train.npy")
    n_train_tokens = n_train_seq * SEQ
    manifest["tokens"] = {
        "doc_tokens_excluding_eos": n_tokens_docs, "tokens_per_doc_mean": float(lengths.mean()), "tokens_per_doc_median": float(np.median(lengths)),
        "holdout_docs": n_holdout_docs, "holdout_sequences": n_hold_seq, "holdout_token_positions": n_hold_seq * SEQ,
        "train_docs": len(lengths) - n_holdout_docs, "train_sequences": n_train_seq, "train_token_positions": n_train_tokens,
        "dapt_tokens_target": TOKENS_TARGET, "passes_over_train": TOKENS_TARGET / n_train_tokens,
        "floor_100M_train_positions_met": n_train_tokens >= 100_000_000, "passes_le_3": TOKENS_TARGET / n_train_tokens <= 3.0,
    }
    manifest["outputs"] = {"train": "data/cdbk_dapt_pilot/visobert_xlmr_512_train.npy", "holdout": "data/cdbk_dapt_pilot/visobert_xlmr_512_holdout.npy"}
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    manifest["created_at"] = now()
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("inspection_sample_50",)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
