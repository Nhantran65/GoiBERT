#!/usr/bin/env python3
"""Stage 1 §2 — build the 250M-token Vietnamese MNTP stream from the general pool v3 (CPU only).

Reads the v3 general pool read-only (`audit_v3.build_general_v3`), re-tokenises the raw Vietnamese
documents with the pinned BidirLM tokenizer, appends EOS 151643 to every document, concatenates and cuts
fixed 512-token sequences (no BOS, no padding), shuffles with seed 42 and reserves 2 000 held-out
sequences before training.  Writes data/bidirlm_vi_mntp/{tokens.npy,order.npy} and the data manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V3 = ROOT / "scripts" / "mmbert_vi_cpt_pilot" / "v3"
V2 = ROOT / "scripts" / "mmbert_vi_cpt_pilot" / "v2"
PILOT = ROOT / "scripts" / "mmbert_vi_cpt_pilot"
for path in (V3, V2, PILOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_v3 as A3  # noqa: E402
import common_v2 as C2  # noqa: E402
import common_v3 as C3  # noqa: E402
import numpy as np  # noqa: E402

SEQ = 512
TRAIN_TOKENS = 250_000_000
HELDOUT_SEQS = 2_000
SHUFFLE_SEED = 42
EOS_ID = 151643
MODEL_REPO = "BidirLM/BidirLM-0.6B-Base"
MODEL_COMMIT = "29794508bbb4e378b4789eebff42ae4c2d47d524"
DATA_DIR = ROOT / "data" / "bidirlm_vi_mntp"
ART_DIR = ROOT / "artifacts" / "bidirlm_vi_mntp"


def pinned_snapshot(repo: str, commit: str) -> Path:
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DATA_DIR))
    ap.add_argument("--manifest", default=str(ART_DIR / "data_manifest.json"))
    ap.add_argument("--parts-limit", type=int, default=0, help="debug: only the first N parts")
    ap.add_argument("--docs-limit", type=int, default=0, help="debug: only the first N documents")
    ap.add_argument("--pool-cache", default="/tmp/bidirlm_vi_mntp_general_docs.json", help="cache of the pool metadata (outside the repo)")
    args = ap.parse_args()
    out_dir, started = Path(args.out_dir), time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    snapshot = pinned_snapshot(MODEL_REPO, MODEL_COMMIT)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    checks = {"tokenizer_length": len(tokenizer), "pad_token_id": tokenizer.pad_token_id,
              "mask_token_id": tokenizer.mask_token_id, "eos_token_id": tokenizer.eos_token_id,
              "mask_token": tokenizer.mask_token, "padding_side": tokenizer.padding_side}
    if not (len(tokenizer) == 151669 and tokenizer.pad_token_id == 151643 and tokenizer.mask_token_id == 151663):
        raise RuntimeError(f"tokenizer contract mismatch: {checks}")
    print(f"[data] tokenizer ok {checks}", flush=True)

    t0 = time.time()
    cache_path = Path(args.pool_cache)
    if cache_path.exists():
        general = json.loads(cache_path.read_text())
        print(f"[data] general pool metadata from cache {cache_path}", flush=True)
    else:
        bucket = [json.loads(line) for line in (C3.DATA_V3 / "informal_bucket" / "informal_docs.jsonl").open(encoding="utf-8")]
        general = A3.build_general_v3({row["pool_index"] for row in bucket})
        try:
            cache_path.write_text(json.dumps(general))
            print(f"[data] cached general pool metadata -> {cache_path}", flush=True)
        except OSError as error:  # cache is a convenience only
            print(f"[data] pool cache not written: {error!r}", flush=True)
    docs = general["docs"]
    print(f"[data] pool: {general['documents']} docs / {general['tokens']} pool tokens (bins {general['tokens_by_bin']}) "
          f"in {time.time()-t0:.0f}s", flush=True)
    if args.docs_limit:
        docs = docs[: args.docs_limit]

    by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for row in docs:
        by_part[row["part"]].append(row)
    parts = sorted(by_part)
    if args.parts_limit:
        parts = parts[: args.parts_limit]
    print(f"[data] {len(docs)} docs in {len(parts)} parts", flush=True)

    chunks: list[np.ndarray] = []
    n_docs = n_words = n_tokens = 0
    t0 = time.time()
    for pi, part in enumerate(parts, 1):
        rows = sorted(by_part[part], key=lambda r: r["pool_index"])
        texts = C2.part_texts(part, {r["kh64"] for r in rows})
        pending: list[str] = []
        for row in rows:
            text = texts.get(row["kh64"])
            if text is None:
                continue
            pending.append(text)
            if len(pending) >= 512:
                encoded = tokenizer(pending, add_special_tokens=False)["input_ids"]
                for ids in encoded:
                    chunks.append(np.asarray(ids + [EOS_ID], dtype=np.uint32))
                    n_tokens += len(ids) + 1
                n_words += sum(len(t.split()) for t in pending)
                n_docs += len(pending)
                pending = []
        if pending:
            encoded = tokenizer(pending, add_special_tokens=False)["input_ids"]
            for ids in encoded:
                chunks.append(np.asarray(ids + [EOS_ID], dtype=np.uint32))
                n_tokens += len(ids) + 1
            n_words += sum(len(t.split()) for t in pending)
            n_docs += len(pending)
        del texts
        if pi % 25 == 0 or pi == len(parts):
            print(f"[data] part {pi}/{len(parts)} docs {n_docs} qwen_tokens {n_tokens/1e6:.1f}M "
                  f"({time.time()-t0:.0f}s)", flush=True)

    flat = np.concatenate(chunks)
    del chunks
    n_seq = int(flat.size // SEQ)
    usable = int(n_seq * SEQ)
    tokens = np.ascontiguousarray(flat[:usable].reshape(n_seq, SEQ))
    del flat
    print(f"[data] {n_docs} docs -> {n_tokens} qwen tokens (flat {usable}) -> {n_seq} sequences of {SEQ}", flush=True)

    rng = np.random.default_rng(SHUFFLE_SEED)
    order = rng.permutation(n_seq).astype(np.int64)
    n_train = int(TRAIN_TOKENS // SEQ)
    if n_train + HELDOUT_SEQS > n_seq:
        raise RuntimeError(f"pool too small: {n_seq} sequences < {n_train + HELDOUT_SEQS}")
    heldout = order[n_train : n_train + HELDOUT_SEQS]

    tokens_path = out_dir / "tokens.npy"
    order_path = out_dir / "order.npy"
    np.save(tokens_path, tokens)
    np.save(order_path, order)
    heldout_path = out_dir / "heldout_indices.npy"
    np.save(heldout_path, heldout)
    print(f"[data] wrote {tokens_path} ({tokens.nbytes/1e9:.2f} GB), {order_path}, {heldout_path}", flush=True)

    manifest = {
        "protocol": "bidirlm-vi-mntp-stage1-v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "pool": "general pool v3 (bins 7-9, v3 S1+S2 filter, host cap 2 %, informal bucket excluded)",
            "builder": "scripts/mmbert_vi_cpt_pilot/v3/audit_v3.build_general_v3 (read-only)",
            "license": "CC0", "nc": False, "wikipedia": False, "benchmark_text": False,
            "pool_documents": general["documents"], "pool_tokens_reported": general["tokens"],
            "tokens_by_bin": general["tokens_by_bin"], "pool_counters": general["counters"],
        },
        "tokenizer": {"repo": MODEL_REPO, "commit": MODEL_COMMIT, **checks, "eos_appended": True, "bos_in_data": False, "padding": False},
        "dataset": {
            "documents_tokenized": n_docs, "words": n_words, "tokens": n_tokens, "tokens_per_word": n_tokens / max(n_words, 1),
            "tokens_used_by_sequences": usable, "sequences": n_seq, "seq_len": SEQ,
            "shuffle_seed": SHUFFLE_SEED, "train_sequences": n_train, "train_tokens": n_train * SEQ,
            "heldout_sequences": HELDOUT_SEQS, "heldout_indices_sha256": sha256_file(heldout_path),
            "heldout_rule": "order[n_train : n_train + 2000] (reserved before training; never trained on)",
            "parts": len(parts), "docs_limit": args.docs_limit or None,
        },
        "files": {
            "tokens.npy": {"path": str(tokens_path), "shape": list(tokens.shape), "dtype": str(tokens.dtype), "sha256": sha256_file(tokens_path)},
            "order.npy": {"path": str(order_path), "shape": list(order.shape), "dtype": str(order.dtype), "sha256": sha256_file(order_path)},
            "heldout_indices.npy": {"path": str(heldout_path), "shape": list(heldout.shape), "sha256": sha256_file(heldout_path)},
        },
        "elapsed_seconds": time.time() - started,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"[data] manifest -> {manifest_path}", flush=True)
    print(f"[data] done in {(time.time()-started)/60:.1f} min; tokens/word {manifest['dataset']['tokens_per_word']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
