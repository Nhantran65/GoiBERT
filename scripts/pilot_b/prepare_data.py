#!/usr/bin/env python3
"""Pilot B data prep: one FineWeb-2 vie_Latn train shard -> packed 512-token XLM-R sequences, plus the
XLM-R diacritic map (skeleton / pattern / ascii-sibling per vocab token). See
docs/PILOT_B_DIACRITIC_FACTORIZED.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import shutil
import time
import unicodedata
from pathlib import Path

# Each worker below tokenizes serially; disable the Rust tokenizer's own internal thread pool so
# multiprocessing.Pool(16) doesn't oversubscribe cores 16x (must be set before any tokenizer import/use).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = "FacebookAI/xlm-roberta-large"
DATASET_REPO = "HuggingFaceFW/fineweb-2"
RAW_DIR = ROOT / "data" / "raw" / "fineweb2_vie"
OUT_DIR = ROOT / "data" / "cpt"
SEQ = 512
BOS, EOS = 0, 2
N_VAL = 2000
N_WORKERS = 16
LOG_EVERY_BATCHES = 20
MAX_BATCHES = 900  # safety cap (~3x the calibrated ~280 batches needed for 320M tokens), see bounded_batches below

TONE_MARKS = "̣̀́̃̉"
VOWEL_MARKS = "̛̂̆"
MARK_SET = set(TONE_MARKS + VOWEL_MARKS)
SPIECE = "▁"


def decompose(s: str) -> tuple[str, str]:
    """ASCII skeleton + sorted diacritic-mark pattern ("" if none) for a display string."""
    n_dj = s.count("đ") + s.count("Đ")
    pre = s.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", pre)
    marks = [ch for ch in nfd if ch in MARK_SET] + ["đ"] * n_dj
    skel = unicodedata.normalize("NFC", "".join(ch for ch in nfd if ch not in MARK_SET))
    return skel, "".join(sorted(marks))


def build_diacritic_map(tok) -> dict:
    V = len(tok)
    specials = set(tok.all_special_ids)
    raw_tokens = tok.convert_ids_to_tokens(list(range(V)))

    skel_key: list[str] = [""] * V
    pattern: list[str] = [""] * V
    ascii_cand: list[str | None] = [None] * V
    for i, t in enumerate(raw_tokens):
        if i in specials:
            skel_key[i] = f"<SPECIAL:{t}>"
            continue
        body = tok.convert_tokens_to_string([t])
        skel, pat = decompose(body)
        # convert_tokens_to_string drops the SPIECE word-boundary marker entirely, so a word-initial piece
        # ("▁an") and an unrelated mid-word piece ("an") would otherwise collide into one skeleton group;
        # re-attach the marker to the grouping key (ascii_cand, used for the vocab lookup, already does this).
        ascii_form = (SPIECE + skel) if t.startswith(SPIECE) else skel
        skel_key[i] = ascii_form
        pattern[i] = pat
        if pat:
            ascii_cand[i] = ascii_form

    group_of: dict[str, int] = {}
    skel_id = np.empty(V, dtype=np.int64)
    for i, key in enumerate(skel_key):
        skel_id[i] = group_of.setdefault(key, len(group_of))
    n_groups = len(group_of)

    pat_names = [""] + sorted({p for p in pattern if p})
    pat_of_id = {p: k for k, p in enumerate(pat_names)}
    pat_id = np.array([pat_of_id[p] for p in pattern], dtype=np.int64)
    n_patterns = len(pat_names)

    vocab = tok.get_vocab()
    ascii_tok = np.full(V, -1, dtype=np.int64)
    for i, cand in enumerate(ascii_cand):
        if cand is None:
            continue
        j = vocab.get(cand)
        if j is not None and j != i:
            ascii_tok[i] = j

    diacritized = int((pat_id != 0).sum())
    groups_ge2 = int((np.bincount(skel_id, minlength=n_groups) >= 2).sum())
    rescued = int(((pat_id != 0) & (ascii_tok >= 0)).sum())
    print(f"[diacritic_map] V={V} diacritized_tokens={diacritized} patterns={n_patterns - 1} "
          f"n_groups={n_groups} groups>=2members={groups_ge2} "
          f"ascii_tok_coverage={100 * rescued / max(1, diacritized):.1f}%", flush=True)

    return dict(skel_id=skel_id, pat_id=pat_id, ascii_tok=ascii_tok,
                pat_names=np.array(pat_names, dtype=object), n_groups=n_groups, n_patterns=n_patterns)


def _init_worker():
    global _TOK
    _TOK = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    _TOK.model_max_length = int(1e30)  # docs are packed manually; never truncate here


def _tok_batch(texts: list[str]) -> list[list[int]]:
    return _TOK(texts, add_special_tokens=False)["input_ids"]


def row_group_batches(parquet_path: Path):
    pf = pq.ParquetFile(parquet_path)
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
        yield [t or "" for t in texts]


def pick_shard() -> str:
    api = HfApi()
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith("data/vie_Latn/train/"))
    if not shards:
        raise RuntimeError("no data/vie_Latn/train/ shards found in HuggingFaceFW/fineweb-2")
    return shards[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=320_000_000)
    args = ap.parse_args()
    tokens_target = int(args.tokens)

    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    print(f"[prepare_data] free disk: {free_gb:.1f} GB", flush=True)
    if free_gb < 15:
        raise RuntimeError(f"only {free_gb:.1f} GB free; need >= 15 GB headroom (expected footprint < 10 GB)")

    shard = pick_shard()
    print(f"[prepare_data] downloading shard {shard}", flush=True)
    local_path = Path(hf_hub_download(repo_id=DATASET_REPO, filename=shard, repo_type="dataset",
                                       local_dir=str(RAW_DIR)))
    print(f"[prepare_data] shard at {local_path} ({local_path.stat().st_size / 1024**3:.2f} GB)", flush=True)

    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    tok.model_max_length = int(1e30)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dmap = build_diacritic_map(tok)
    np.savez(OUT_DIR / "xlmr_diacritic_map.npz", **dmap)
    print(f"[prepare_data] wrote {OUT_DIR / 'xlmr_diacritic_map.npz'}", flush=True)

    n_seq_target = tokens_target // SEQ
    raw_target = n_seq_target * (SEQ - 1)
    raw = np.empty(raw_target, dtype=np.uint32)
    cursor = 0
    n_docs = 0
    t0 = time.time()
    # Bound the batches fed to the pool (~3x the calibrated need) so Pool.imap's task-feeder thread never
    # queues the whole shard behind our backs; a plain, finite iterable also makes pool shutdown clean
    # (no reliance on terminate() racing an unbounded feeder thread).
    bounded_batches = itertools.islice(row_group_batches(local_path), MAX_BATCHES)
    with mp.Pool(N_WORKERS, initializer=_init_worker) as pool:
        for batch_idx, ids_list in enumerate(pool.imap(_tok_batch, bounded_batches, chunksize=1)):
            for ids in ids_list:
                if cursor >= raw_target:
                    break
                n_docs += 1
                ids.append(EOS)
                take = min(len(ids), raw_target - cursor)
                raw[cursor:cursor + take] = np.asarray(ids[:take], dtype=np.uint32)
                cursor += take
            if batch_idx % LOG_EVERY_BATCHES == 0:
                print(f"[prepare_data] batch {batch_idx} docs={n_docs} raw={cursor}/{raw_target} "
                      f"({100 * cursor / raw_target:.1f}%) {time.time() - t0:.0f}s", flush=True)
            if cursor >= raw_target:
                break
        pool.terminate()

    if cursor < raw_target:
        print(f"[prepare_data] WARNING: shard exhausted at {cursor} raw tokens (< target {raw_target}); shrinking",
              flush=True)
        n_seq_target = cursor // (SEQ - 1)
        raw_target = n_seq_target * (SEQ - 1)
        raw = raw[:raw_target]

    packed_path = OUT_DIR / "fineweb2_vie_xlmr_512.npy"
    packed = np.lib.format.open_memmap(packed_path, mode="w+", dtype=np.uint32, shape=(n_seq_target, SEQ))
    packed[:, 0] = BOS
    packed[:, 1:] = raw.reshape(n_seq_target, SEQ - 1)
    packed.flush()
    del packed

    n_tokens = n_seq_target * SEQ
    meta = dict(n_seq=n_seq_target, n_tokens=n_tokens, n_docs=n_docs, n_val=N_VAL, shard=shard,
                tokenizer=TOKENIZER, seq_len=SEQ, elapsed_s=round(time.time() - t0, 1))
    (OUT_DIR / "fineweb2_vie_xlmr_512.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepare_data] packed {n_seq_target} sequences, {n_tokens} tokens, {n_docs} docs -> {packed_path}",
          flush=True)
    print(f"[prepare_data] meta: {meta}", flush=True)


if __name__ == "__main__":
    main()
