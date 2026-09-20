#!/usr/bin/env python3
"""Data gate (plan §3): FineWeb-2 vie_Latn shard -> hash-sampled documents -> aligned BamiBERT/XLM-R sequences.

Steps
  1. sample documents from the WHOLE shard by sha256(f"{seed}:{id}") < fraction (seed 42); normalise; keep Vietnamese
     documents (content check) with >= 20 letter/digit units.
  2. document-level exact dedup (casefolded NFC key) and near-dup (>= 50 % of 8-word shingles already seen).
  3. chunk each document into 8..96-unit sequences that fit 256 tokens under BOTH tokenizers (textproc.document_chunks);
     per token, the unit-span index from character offsets (no truncation, no re-segmentation, diacritics kept).
  4. text-only decontamination of every sequence against train/dev/test text of the 5 benchmarks (exact key OR any shared
     8-word shingle; labels never read).
  5. holdout by document (1 % of documents, cap 1,000) before train; train sequence order shuffled once with seed 42 and
     locked; batch schedule of 128 sequences -> steps needed for the 30M non-pad token budget (<= 1 pass).
Gate: train non-pad BamiBERT positions >= 31M.  Outputs data/cafe_bami_repdistill_pilot/{corpus.npz, texts.jsonl, manifest.json}.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

N_WORKERS = 48
HOLDOUT_FRACTION = 0.01
HOLDOUT_MAX_DOCS = 1000
NEAR_DUP_FRACTION = 0.5
MIN_DOC_UNITS = 20


def _keep(doc_id: str, fraction: float) -> bool:
    h = int(hashlib.sha256(f"{common.SEED}:{doc_id}".encode()).hexdigest()[:16], 16)
    return h / float(1 << 64) < fraction


def _sample_rg(args) -> tuple[int, list[tuple[str, str, str]], dict]:
    rg, fraction = args
    pf = pq.ParquetFile(str(common.FW2_LOCAL))
    tb = pf.read_row_group(rg, columns=["id", "text", "url"])
    out, st = [], collections.Counter()
    for doc_id, text, url in zip(tb.column("id").to_pylist(), tb.column("text").to_pylist(), tb.column("url").to_pylist()):
        st["docs"] += 1
        if not _keep(doc_id, fraction):
            continue
        st["sampled"] += 1
        full = common.normalize(text)
        if not full:
            st["empty"] += 1; continue
        if len(common.UNIT.findall(full)) < MIN_DOC_UNITS:
            st["too_short"] += 1; continue
        if not common.is_vietnamese(full):
            st["not_vi"] += 1; continue
        out.append((doc_id, url or "", text))
        st["kept"] += 1
    return rg, out, dict(st)


def _chunk_doc(args):
    import textproc
    idx, text = args
    st: dict = {}
    chunks = textproc.document_chunks(text, st)
    return idx, [(c.text, c.n_units, c.s_ids, c.t_ids, c.s_span, c.t_span, c.s_valid, c.t_valid) for c in chunks], st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.016)
    ap.add_argument("--out", default=str(common.DATA_DIR))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if common.sha256_file(common.FW2_LOCAL) != common.FW2_SHARD_SHA256:
        raise RuntimeError("FineWeb-2 shard sha256 mismatch")
    pf = pq.ParquetFile(str(common.FW2_LOCAL))
    manifest: dict = {"protocol": common.PROTOCOL, "seed": common.SEED, "created_at": common.now(),
                      "source": {"repo": common.FW2_REPO, "commit": common.FW2_COMMIT, "file": common.FW2_SHARD, "sha256": common.FW2_SHARD_SHA256, "bytes": common.FW2_LOCAL.stat().st_size,
                                 "rows": pf.metadata.num_rows, "row_groups": pf.num_row_groups, "license": common.FW2_LICENSE,
                                 "sampling": f"document kept iff sha256('{common.SEED}:'+id)[:16]/2^64 < {args.fraction} over all {pf.num_row_groups} row groups"},
                      "tokenizers": common.tokenizer_manifest(), "limits": {"max_tokens_both_tokenizers": common.MAX_TOKENS, "units_per_sequence": [common.MIN_UNITS, common.MAX_UNITS]}}

    # 1. sample
    docs: list[tuple[str, str, str]] = []
    st1 = collections.Counter()
    with mp.Pool(N_WORKERS) as pool:
        for rg, out, st in pool.imap_unordered(_sample_rg, [(rg, args.fraction) for rg in range(pf.num_row_groups)], chunksize=8):
            docs.extend(out); st1.update(st)
    docs.sort(key=lambda d: d[0])
    print(f"[prep] sampled {st1['sampled']} / {st1['docs']} docs, kept {len(docs)} after empty/short/language ({time.time()-t0:.0f}s)", flush=True)

    # 2. document dedup
    seen_keys: set[str] = set(); seen_sh: set[str] = set()
    kept: list[tuple[str, str, str]] = []
    st2 = collections.Counter()
    for doc_id, url, text in docs:
        k = common.key_of(text)
        if k in seen_keys:
            st2["exact_dup"] += 1; continue
        sh = common.shingles(k)
        if len(sh) >= 3 and sum(s in seen_sh for s in sh) / len(sh) >= NEAR_DUP_FRACTION:
            st2["near_dup"] += 1; continue
        seen_keys.add(k); seen_sh.update(sh); kept.append((doc_id, url, text)); st2["kept"] += 1
    del seen_keys, seen_sh
    print(f"[prep] dedup: {dict(st2)} ({time.time()-t0:.0f}s)", flush=True)

    # 3. chunk + align (parallel)
    per_doc: dict[int, list] = {}
    st3 = collections.Counter()
    with mp.Pool(N_WORKERS) as pool:
        for idx, chunks, st in pool.imap_unordered(_chunk_doc, [(i, d[2]) for i, d in enumerate(kept)], chunksize=16):
            per_doc[idx] = chunks; st3.update(st)
    print(f"[prep] chunked: {dict(st3)} ({time.time()-t0:.0f}s)", flush=True)

    # 4. decontamination (sequence level)
    bench = common.benchmark_texts()
    bench_keys: dict[str, str] = {}; bench_sh: dict[str, str] = {}
    for name, rows in bench.items():
        for r in rows:
            k = common.key_of(r)
            if k:
                bench_keys.setdefault(k, name)
                for s in common.shingles(k):
                    bench_sh.setdefault(s, name)
    st4 = collections.Counter(); by_split = collections.Counter()
    seqs: list[dict] = []
    for idx in range(len(kept)):
        for (text, n_units, s_ids, t_ids, s_span, t_span, s_valid, t_valid) in per_doc.get(idx, []):
            k = common.key_of(text)
            hit = bench_keys.get(k)
            if hit is None:
                for s in common.shingles(k):
                    if s in bench_sh:
                        hit = bench_sh[s]; break
            if hit is not None:
                st4["decontam"] += 1; by_split[hit] += 1; continue
            st4["kept"] += 1
            seqs.append({"doc": idx, "text": text, "n_units": n_units, "s_ids": s_ids, "t_ids": t_ids, "s_span": s_span, "t_span": t_span, "s_valid": s_valid, "t_valid": t_valid})
    del per_doc
    print(f"[prep] decontam: {dict(st4)} by split {dict(by_split)} ({time.time()-t0:.0f}s)", flush=True)

    # 5. holdout by document, lock train order, batch schedule
    doc_ids = list(range(len(kept)))
    rng = random.Random(common.SEED); rng.shuffle(doc_ids)
    n_hold = min(HOLDOUT_MAX_DOCS, max(1, int(round(HOLDOUT_FRACTION * len(doc_ids)))))
    holdout_docs = set(doc_ids[:n_hold])
    train = [s for s in seqs if s["doc"] not in holdout_docs]
    hold = [s for s in seqs if s["doc"] in holdout_docs]
    order = list(range(len(train))); random.Random(common.SEED + 1).shuffle(order)
    train = [train[i] for i in order]
    hold.sort(key=lambda s: (s["doc"], s["text"]))

    def pack(rows: list[dict]) -> dict[str, np.ndarray]:
        s_len = np.array([len(r["s_ids"]) for r in rows], dtype=np.int32)
        t_len = np.array([len(r["t_ids"]) for r in rows], dtype=np.int32)
        return {
            "s_ids": np.concatenate([np.asarray(r["s_ids"], dtype=np.int32) for r in rows]) if rows else np.zeros(0, np.int32),
            "t_ids": np.concatenate([np.asarray(r["t_ids"], dtype=np.int32) for r in rows]) if rows else np.zeros(0, np.int32),
            "s_span": np.concatenate([np.asarray(r["s_span"], dtype=np.int16) for r in rows]) if rows else np.zeros(0, np.int16),
            "t_span": np.concatenate([np.asarray(r["t_span"], dtype=np.int16) for r in rows]) if rows else np.zeros(0, np.int16),
            "s_off": np.concatenate([[0], np.cumsum(s_len)]).astype(np.int64), "t_off": np.concatenate([[0], np.cumsum(t_len)]).astype(np.int64),
            "n_units": np.array([r["n_units"] for r in rows], dtype=np.int16), "doc": np.array([r["doc"] for r in rows], dtype=np.int32),
            "s_valid": np.array([r["s_valid"] for r in rows], dtype=np.int16), "t_valid": np.array([r["t_valid"] for r in rows], dtype=np.int16),
        }
    arrays = {f"train_{k}": v for k, v in pack(train).items()}
    arrays.update({f"holdout_{k}": v for k, v in pack(hold).items()})
    s_len_train = np.diff(arrays["train_s_off"])
    n_train_tokens = int(s_len_train.sum())
    n_train_tokens_nospecial = int(n_train_tokens - 2 * len(train))
    cum = np.cumsum(np.add.reduceat(s_len_train, np.arange(0, len(train), common.GLOBAL_BATCH))) if len(train) else np.zeros(0)
    steps_target = int(np.searchsorted(cum, common.TOKEN_BUDGET, side="left")) + 1 if len(cum) and cum[-1] >= common.TOKEN_BUDGET else None
    tokens_at_target = int(cum[steps_target - 1]) if steps_target else None
    np.savez(out_dir / "corpus.npz", **arrays)
    with (out_dir / "texts.jsonl").open("w", encoding="utf-8") as fh:
        for split, rows in (("train", train), ("holdout", hold)):
            for i, r in enumerate(rows):
                fh.write(json.dumps({"split": split, "i": i, "doc": r["doc"], "doc_id": kept[r["doc"]][0], "text": r["text"]}, ensure_ascii=False) + "\n")
    with (out_dir / "documents.jsonl").open("w", encoding="utf-8") as fh:
        for i, (doc_id, url, _) in enumerate(kept):
            fh.write(json.dumps({"doc": i, "doc_id": doc_id, "url": url, "holdout": i in holdout_docs}, ensure_ascii=False) + "\n")

    units = np.array([r["n_units"] for r in train]) if train else np.zeros(1)
    t_len_train = np.diff(arrays["train_t_off"]) if train else np.zeros(1)
    rng_a = random.Random(common.SEED)
    audit = [train[i]["text"] for i in rng_a.sample(range(len(train)), min(20, len(train)))]
    manifest.update({
        "sampling_counters": dict(st1), "dedup_counters": dict(st2), "chunk_counters": dict(st3), "decontam": {"counters": dict(st4), "dropped_by_benchmark_split": dict(by_split),
                     "benchmark_rows": {k: len(v) for k, v in bench.items()}, "benchmark_unique_keys": len(bench_keys), "benchmark_shingles": len(bench_sh), "benchmark_key_set_sha256": common.sha256_lines(sorted(bench_keys)),
                     "method": "exact casefolded-NFC key OR any shared 8-word shingle with train/dev/test text of vsfc, visfd, absa_restaurant, absa_hotel; text columns only, labels never read"},
        "documents": {"kept": len(kept), "holdout": n_hold, "train": len(kept) - n_hold, "holdout_doc_ids_sha256": common.sha256_lines(sorted(kept[i][0] for i in holdout_docs)), "train_doc_ids_sha256": common.sha256_lines(sorted(kept[i][0] for i in range(len(kept)) if i not in holdout_docs))},
        "sequences": {"train": len(train), "holdout": len(hold), "train_order": "shuffled once with seed 43 (SEED+1) and locked in corpus.npz", "train_s_ids_sha256": common.sha256_array(arrays["train_s_ids"]), "train_t_ids_sha256": common.sha256_array(arrays["train_t_ids"]), "holdout_s_ids_sha256": common.sha256_array(arrays["holdout_s_ids"])},
        "tokens": {"train_nonpad_bami_incl_specials": n_train_tokens, "train_nonpad_bami_excl_specials": n_train_tokens_nospecial, "train_xlmr_incl_specials": int(t_len_train.sum()),
                   "holdout_nonpad_bami_incl_specials": int(np.diff(arrays["holdout_s_off"]).sum()),
                   "bami_len_mean": float(s_len_train.mean()), "bami_len_median": float(np.median(s_len_train)), "bami_len_p95": float(np.percentile(s_len_train, 95)), "bami_len_max": int(s_len_train.max()),
                   "xlmr_len_mean": float(t_len_train.mean()), "xlmr_len_max": int(t_len_train.max()), "units_mean": float(units.mean()), "units_min": int(units.min()), "units_max": int(units.max())},
        "alignment_coverage": {"train_spans_total": int(units.sum()), "train_spans_valid_student": int(arrays["train_s_valid"].astype(np.int64).sum()), "train_spans_valid_teacher": int(arrays["train_t_valid"].astype(np.int64).sum())},
        "budget": {"global_batch": common.GLOBAL_BATCH, "token_budget": common.TOKEN_BUDGET, "steps_target": steps_target, "tokens_at_steps_target": tokens_at_target,
                   "sequences_used": steps_target * common.GLOBAL_BATCH if steps_target else None, "passes": (steps_target * common.GLOBAL_BATCH / len(train)) if steps_target and train else None,
                   "budget_error_batches": ((tokens_at_target - common.TOKEN_BUDGET) / (tokens_at_target / steps_target)) if steps_target else None},
        "gate": {"train_tokens_ge_31M": n_train_tokens >= common.MIN_TRAIN_TOKENS, "train_tokens": n_train_tokens, "min_required": common.MIN_TRAIN_TOKENS, "one_pass": bool(steps_target) and steps_target * common.GLOBAL_BATCH <= len(train)},
        "quality_audit_examples": audit, "outputs": {"corpus": str((out_dir / "corpus.npz").relative_to(common.ROOT)), "corpus_sha256": common.sha256_file(out_dir / "corpus.npz"), "texts": str((out_dir / "texts.jsonl").relative_to(common.ROOT)), "texts_sha256": common.sha256_file(out_dir / "texts.jsonl")},
        "source_hash": common.source_hash(), "elapsed_s": round(time.time() - t0, 1),
    })
    manifest["gate"]["all_ok"] = bool(manifest["gate"]["train_tokens_ge_31M"] and manifest["gate"]["one_pass"])
    common.write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("documents", "sequences", "tokens", "alignment_coverage", "budget", "gate")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
