#!/usr/bin/env python3
"""Step 1.1 — count XLM-R sub-word frequencies on Vietnamese text and build the keep set K.

Sources (plan §1.1): general pool v3 (shuffled parts, seed 42, >= 300M XLM-R tokens), Wikipedia v2 filtered
(counting sources); train+dev of the five benchmarks and of the social datasets (mandatory tokens).  Test text
is never used here.  Writes artifacts/vocab_trim_student/{vocab_counts.json,keep_set.json,counts.npz}.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np

import sources as S

LATIN_RANGES = ((0x0000, 0x007F), (0x0080, 0x00FF), (0x0100, 0x017F), (0x0180, 0x024F), (0x0300, 0x036F),
                (0x1E00, 0x1EFF), (0x1EA0, 0x1EF9), (0x2000, 0x206F), (0x2C60, 0x2C7F), (0xA720, 0xA7FF),
                (0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF), (0xFE0F, 0xFE0F))


def allowed_single_char(ch: str) -> bool:
    if ch == " ":
        return True
    code = ord(ch)
    if any(lo <= code <= hi for lo, hi in LATIN_RANGES):
        return True
    category = unicodedata.category(ch)
    return category[0] in "PSN"  # punctuation, symbols (incl. emoji), numbers


def is_single_char_piece(token: str) -> bool:
    stripped = token.replace("▁", "")
    return len(stripped) <= 1 and (stripped == "" or allowed_single_char(stripped))


def count_streams(tokenizer, texts, counts: np.ndarray, *, batch: int = 512, limit_tokens: int = 0,
                   seen_ids: set[int] | None = None, progress=None) -> int:
    buf: list[str] = []
    total = 0
    for text in texts:
        buf.append(text)
        if len(buf) >= batch:
            encoded = tokenizer(buf, add_special_tokens=False)["input_ids"]
            flat = np.concatenate([np.asarray(ids, dtype=np.int64) for ids in encoded]) if encoded else np.zeros(0, np.int64)
            counts += np.bincount(flat, minlength=S.VOCAB_SIZE)
            if seen_ids is not None:
                seen_ids.update(flat.tolist())
            total += int(flat.size)
            buf = []
            if progress and total % (20 * 1024 * 1024) < batch * 40:
                progress(total)
            if limit_tokens and total >= limit_tokens:
                break
    if buf and not (limit_tokens and total >= limit_tokens):
        encoded = tokenizer(buf, add_special_tokens=False)["input_ids"]
        flat = np.concatenate([np.asarray(ids, dtype=np.int64) for ids in encoded]) if encoded else np.zeros(0, np.int64)
        counts += np.bincount(flat, minlength=S.VOCAB_SIZE)
        if seen_ids is not None:
            seen_ids.update(flat.tolist())
        total += int(flat.size)
    return total


def pool_stream(tokenizer, docs: list[dict], *, seed: int, target_tokens: int, log_every: int = 20):
    """Parts in shuffled order; documents shuffled inside each part; stop once the token target is reached."""
    import random

    by_part: dict[str, list[dict]] = {}
    for row in docs:
        by_part.setdefault(row["part"], []).append(row)
    rng = random.Random(seed)
    parts = sorted(by_part)
    rng.shuffle(parts)
    counted = 0
    for index, part in enumerate(parts, 1):
        rows = by_part[part]
        rng.shuffle(rows)
        texts = S.part_texts(part, {row["kh64"] for row in rows})
        for row in rows:
            text = texts.get(row["kh64"])
            if text:
                yield text
                counted += 1
        if index % log_every == 0:
            print(f"[count] pool part {index}/{len(parts)} docs yielded {counted}", flush=True)
        del texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-tokens", type=int, default=300_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap", type=int, default=45_000)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--out", default=str(S.ART / "vocab_counts.json"))
    ap.add_argument("--keep-out", default=str(S.ART / "keep_set.json"))
    args = ap.parse_args()
    started = time.time()
    S.ART.mkdir(parents=True, exist_ok=True)
    S.DATA.mkdir(parents=True, exist_ok=True)
    tokenizer = S.xlmr_tokenizer()
    report: dict = {"protocol": "vocab-trim-count-v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seed": args.seed, "pool_token_target": args.pool_tokens, "min_count": args.min_count, "cap": args.cap,
                    "tokenizer": {"repo": S.XLM_R_LARGE[0], "commit": S.XLM_R_LARGE[1], "vocab_size": len(tokenizer)}}

    # ---- counting sources -------------------------------------------------------------------
    counts_pool = np.zeros(S.VOCAB_SIZE, dtype=np.int64)
    docs = S.pool_docs()
    t0 = time.time()
    tokens_pool = count_streams(tokenizer, pool_stream(tokenizer, docs, seed=args.seed, target_tokens=args.pool_tokens),
                                counts_pool, limit_tokens=args.pool_tokens)
    print(f"[count] pool: {tokens_pool/1e6:.1f}M XLM-R tokens in {time.time()-t0:.0f}s", flush=True)

    counts_wiki = np.zeros(S.VOCAB_SIZE, dtype=np.int64)
    t0 = time.time()
    tokens_wiki = count_streams(tokenizer, S.wiki_docs(), counts_wiki)
    print(f"[count] wiki: {tokens_wiki/1e6:.1f}M tokens in {time.time()-t0:.0f}s", flush=True)

    counts_bench = np.zeros(S.VOCAB_SIZE, dtype=np.int64)
    mandatory_ids: set[int] = set()
    tokens_bench = count_streams(tokenizer, S.benchmark_texts(), counts_bench, seen_ids=mandatory_ids)
    counts_social = np.zeros(S.VOCAB_SIZE, dtype=np.int64)
    tokens_social = count_streams(tokenizer, S.social_texts(), counts_social, seen_ids=mandatory_ids)
    print(f"[count] benchmark train+dev {tokens_bench/1e6:.2f}M tokens, social train+dev {tokens_social/1e6:.2f}M tokens",
          flush=True)

    # ---- keep set ---------------------------------------------------------------------------
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    single_char_ids = {i for i, tok in id_to_token.items() if is_single_char_piece(tok)}
    counts_total = counts_pool + counts_wiki
    frequent_ids = set(np.nonzero(counts_total >= args.min_count)[0].tolist())
    core = {S.MASK_ID} | set(S.SPECIAL_IDS) | mandatory_ids | single_char_ids
    keep = core | frequent_ids
    dropped_for_cap: list[int] = []
    if len(keep) > args.cap:
        # never drop specials / mandatory / single characters: rank the rest by pool frequency
        flexible = sorted(keep - core, key=lambda i: (-int(counts_total[i]), i))
        allowed = max(0, args.cap - len(core))
        keep = core | set(flexible[:allowed])
        dropped_for_cap = flexible[allowed:]
    keep_ordered = list(S.SPECIAL_IDS) + sorted(i for i in keep if i not in S.SPECIAL_IDS and i != S.MASK_ID) + [S.MASK_ID]
    assert len(set(keep_ordered)) == len(keep_ordered)
    assert set(keep_ordered) == keep | set(S.SPECIAL_IDS), "keep set assembly mismatch"

    def coverage(counts: np.ndarray) -> float:
        total = int(counts.sum())
        return float(counts[keep_ordered].sum() / total) if total else 1.0

    report.update({
        "tokens_counted": {"pool": tokens_pool, "wiki": tokens_wiki, "benchmark_train_dev": tokens_bench, "social_train_dev": tokens_social},
        "keep_set": {"size": len(keep_ordered), "cap_hit": len(keep) >= args.cap, "dropped_for_cap": len(dropped_for_cap),
                     "mandatory_tokens": len(mandatory_ids), "single_char_pieces": len(single_char_ids),
                     "frequent_tokens": len(frequent_ids)},
        "coverage": {"pool": coverage(counts_pool), "wiki": coverage(counts_wiki), "benchmark_train_dev": coverage(counts_bench),
                     "social_train_dev": coverage(counts_social)},
        "unk_proxy_on_counting_sources": {"pool": 1 - coverage(counts_pool), "benchmark_train_dev": 1 - coverage(counts_bench)},
    })
    top = sorted(keep_ordered, key=lambda i: -int(counts_total[i]))[:50]
    report["top_tokens"] = [{"old_id": int(i), "token": id_to_token.get(i), "count": int(counts_total[i])} for i in top]
    np.savez_compressed(S.ART / "counts.npz", pool=counts_pool, wiki=counts_wiki, benchmark=counts_bench, social=counts_social,
                        keep=np.asarray(keep_ordered, dtype=np.int64))
    S.write_json(Path(args.out), report)
    keep_payload = {
        "protocol": "vocab-trim-keep-v1", "created_at": report["created_at"], "vocab_size": len(keep_ordered),
        "special_ids": list(S.SPECIAL_IDS), "mask_id": S.MASK_ID,
        "old_ids": [int(i) for i in keep_ordered],
        "tokens": [id_to_token.get(int(i)) for i in keep_ordered],
        "flags": {"mandatory": [int(i) in mandatory_ids for i in keep_ordered],
                  "single_char": [int(i) in single_char_ids for i in keep_ordered],
                  "frequent": [int(i) in frequent_ids for i in keep_ordered]},
    }
    S.write_json(Path(args.keep_out), keep_payload)
    print(f"[count] |K| = {len(keep_ordered)} (cap {args.cap}, mandatory {len(mandatory_ids)}, single-char {len(single_char_ids)}) "
          f"pool coverage {100*report['coverage']['pool']:.4f}% bench coverage {100*report['coverage']['benchmark_train_dev']:.4f}%",
          flush=True)
    print(f"[count] done in {(time.time()-started)/60:.1f} min -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
