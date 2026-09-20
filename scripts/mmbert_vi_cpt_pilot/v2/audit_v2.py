#!/usr/bin/env python3
"""v2 §2/§3 sampling and audit helpers.

  sample-s2 : 400 documents drawn uniformly (seed 42) from pool bins 7-9 for the hand-labelled S2 training set
  print-s2  : print a slice of the S2 samples for the single reader (same 1 200-character window as v1)
  sample    : the v2 audit cells (100 samples each, seed 42) over the vetted v2 corpus
  score     : apply the v2 audit gate to the hand labels
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C  # noqa: E402

OUT_DIR = C.ART_V2 / "data_audit_v2"
EXCERPT_CHARS = 1200
_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = C.common.load_tokenizer()
        _TOK.model_max_length = int(1e9)
    return _TOK


def texts_for(rows: list[dict], chars: int | None = None) -> dict[int, str]:
    """Texts (optionally truncated) for pool rows, keyed by pool_index."""
    by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_part[r["part"]].append(r)
    out: dict[int, str] = {}
    for part, rs in by_part.items():
        need = {r["kh64"]: r for r in rs}
        for kh64, text in C.part_texts(part, set(need)).items():
            r = need[kh64]
            out[r["pool_index"]] = text if chars is None else text[:chars]
    return out


def cmd_sample_s2(args) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = [r for r in C.pool_table() if r["split"] == "pool" and r["bin"] in ("7", "8", "9")]
    rng = random.Random(C.SEED)
    rng.shuffle(pool)
    picked = pool[: args.n]
    print(f"[s2-sample] pool bins 7-9 documents: {len(pool)}; sampling {len(picked)}")
    C.write_json(OUT_DIR / "s2_sample_index.json",
                 {"n": len(picked), "seed": C.SEED, "pool_rows_bins_7_9": len(pool),
                  "rows": [{k: r[k] for k in ("pool_index", "part", "local_index", "bin", "group", "host", "url", "n_tokens", "kh64")} for r in picked]})
    print(f"[s2-sample] wrote {OUT_DIR / 's2_sample_index.json'}")
    return 0


def cmd_print_s2(args) -> int:
    rows = C.read_json(OUT_DIR / "s2_sample_index.json")["rows"][args.start:args.end]
    texts = texts_for(rows)
    for i, r in enumerate(rows):
        t = " ".join(texts.get(r["pool_index"], "").split())[:args.chars]
        print(f"{args.start + i:03d}|{r['bin']}|{r['host'][:28]}|{t}")
    return 0


# ------------------------------------------------------------------ v2 audit cells
def _decode_tokens(tokens, offsets, i: int, chars: int) -> str:
    a, b = int(offsets[i]), int(offsets[i + 1])
    tok = tokenizer()
    return tok.decode([int(x) for x in tokens[a:b]])[:chars]


def cmd_sample_cells(args) -> int:
    import numpy as np
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(C.SEED)
    rows: list[dict] = []
    art = C.ART_V2

    def add(cell: str, texts: list[str], meta: list[dict]) -> None:
        idx = list(range(len(texts)))
        rng.shuffle(idx)
        for i in idx[: args.per_cell]:
            rows.append({"cell": cell, "index": len([r for r in rows if r["cell"] == cell]), "text": texts[i], **meta[i]})

    # general cells: decode the first 1 200 characters of the sampled documents
    general = [json.loads(l) for l in (art / "general_docs.jsonl").open(encoding="utf-8")]
    for b in ("9", "8", "7"):
        idx = [i for i, r in enumerate(general) if r["bin"] == b]
        rng.shuffle(idx)
        picked = [general[i] for i in idx[: args.per_cell]]
        by_part: dict[str, list[dict]] = collections.defaultdict(list)
        for r in picked:
            by_part[r["part"]].append(r)
        texts, metas = [], []
        for part, rs in by_part.items():
            offsets = np.load(C.DATA_V1 / "tokens" / f"{part}.offsets.npy")
            tokens = np.load(C.DATA_V1 / "tokens" / f"{part}.tokens.npy", mmap_mode="r")
            for r in rs:
                texts.append(_decode_tokens(tokens, offsets, r["local_index"], EXCERPT_CHARS))
                metas.append({"host": r["host"], "url": r["url"], "tokens": r["n_tokens"]})
        add(f"general_bin{b}", texts, metas)

    # register cells: one paragraph per sampled UGC document
    reg_dir = art / "register"
    reg_docs = [json.loads(l) for l in (reg_dir / "register.docs.jsonl").open(encoding="utf-8")]
    tokens = np.load(reg_dir / "tokens.npy", mmap_mode="r")
    offsets = np.load(reg_dir / "offsets.npy")
    tok = tokenizer()
    for group in ("forum_qa", "review_commerce"):
        sel = [r for r in reg_docs if r["source"] == "ugc" and r["group"] == group]
        picks = []
        rng2 = random.Random(C.SEED + 1)
        idx = list(range(len(sel)))
        rng2.shuffle(idx)
        for i in idx[: args.per_cell]:
            r = sel[i]
            text = _decode_tokens(tokens, offsets, r["i"], 20000)
            paras = [p.strip() for p in text.split("\n") if p.strip()]
            pick = rng2.choice(paras) if paras else text[:800]
            picks.append((pick[:EXCERPT_CHARS], {"host": r["host"], "url": r["url"], "tokens": r["n_tokens"]}))
        add(f"register_{group}", [p for p, _ in picks], [m for _, m in picks])

    # discussion cell
    sel = [r for r in reg_docs if r["source"] == "discussion"]
    idx = list(range(len(sel)))
    rng.shuffle(idx)
    disc = []
    for i in idx[: args.per_cell]:
        r = sel[i]
        disc.append((_decode_tokens(tokens, offsets, r["i"], EXCERPT_CHARS), {"host": r["host"], "url": r["url"], "tokens": r["n_tokens"]}))
    add("discussion", [p for p, _ in disc], [m for _, m in disc])

    # wikipedia cell
    wiki = [json.loads(l) for l in (art / "wiki_filtered.jsonl").open(encoding="utf-8")]
    wtokens = np.load(C.DATA_V1 / "wiki" / "wiki.tokens.npy", mmap_mode="r")
    woffsets = np.load(C.DATA_V1 / "wiki" / "wiki.offsets.npy")
    idx = list(range(len(wiki)))
    rng.shuffle(idx)
    wk = []
    for i in idx[: args.per_cell]:
        r = wiki[i]
        wk.append((_decode_tokens(wtokens, woffsets, r["i"], EXCERPT_CHARS), {"host": "vi.wikipedia.org", "url": r.get("url"), "tokens": r["n_tokens"]}))
    add("wikipedia", [p for p, _ in wk], [m for _, m in wk])

    # side-effect sample: UGC documents judged by vá 1 *before* it was applied (produced by ugc_extract --side-effect-sample)
    se_path = C.DATA_V2 / "ugc" / "side_effect_sample.jsonl"
    if se_path.is_file():
        se = [json.loads(l) for l in se_path.open(encoding="utf-8")]
        idx = list(range(len(se)))
        rng.shuffle(idx)
        sel = [se[i] for i in idx[: args.per_cell]]
        add("ugc_side_effect", [r["text"][:EXCERPT_CHARS] for r in sel],
            [{"host": r["host"], "url": r.get("url"), "dropped_by_va1": r["drop"], "s1": r.get("s1"), "s2": r.get("s2")} for r in sel])

    # loose-extraction register comparison cell (paragraphs before the §2 refinement)
    loose = Path("/tmp/register_loose")
    if (loose / "register.docs.jsonl").is_file():
        ldocs = [json.loads(l) for l in (loose / "register.docs.jsonl").open(encoding="utf-8")]
        ltokens = np.load(loose / "tokens.npy", mmap_mode="r")
        loffsets = np.load(loose / "offsets.npy")
        sel = [r for r in ldocs if r["source"] == "ugc"]
        rng3 = random.Random(C.SEED + 2)
        idx = list(range(len(sel)))
        rng3.shuffle(idx)
        picks = []
        for i in idx[: args.per_cell]:
            r = sel[i]
            text = _decode_tokens(ltokens, loffsets, r["i"], 20000)
            paras = [p.strip() for p in text.split("\n") if p.strip()]
            picks.append((rng3.choice(paras)[:EXCERPT_CHARS] if paras else text[:800], {"host": r["host"], "url": r["url"], "tokens": r["n_tokens"]}))
        add("register_loose", [p for p, _ in picks], [m for _, m in picks])
    with (OUT_DIR / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT_DIR / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"cell": r["cell"], "index": r["index"], "real_vietnamese": None, "boilerplate_majority": None,
                                 "machine_translated_or_generated": None, "adult_or_spam": None, "opinion_voice": None}, ensure_ascii=False) + "\n")
    counts = collections.Counter(r["cell"] for r in rows)
    print(json.dumps(dict(counts), ensure_ascii=False))
    return 0


def cmd_print_cell(args) -> int:
    rows = [json.loads(l) for l in (OUT_DIR / "samples.jsonl").open(encoding="utf-8")]
    sel = [r for r in rows if r["cell"] == args.cell][args.start:args.end]
    for r in sel:
        t = " ".join(r["text"].split())[:args.chars]
        extra = ""
        if "dropped_by_va1" in r:
            extra = f"|drop={r['dropped_by_va1']}|s1={r['s1']:.2f}|s2={r['s2']:.2f}"
        print(f"{r['index']:03d}|{(r.get('host') or '')[:24]}{extra}|{t}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample-s2"); s.add_argument("--n", type=int, default=400); s.set_defaults(func=cmd_sample_s2)
    p = sub.add_parser("print-s2"); p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int, default=100)
    p.add_argument("--chars", type=int, default=150); p.set_defaults(func=cmd_print_s2)
    sc = sub.add_parser("sample-cells"); sc.add_argument("--per-cell", type=int, default=100); sc.set_defaults(func=cmd_sample_cells)
    pc = sub.add_parser("print-cell"); pc.add_argument("--cell", required=True); pc.add_argument("--start", type=int, default=0)
    pc.add_argument("--end", type=int, default=100); pc.add_argument("--chars", type=int, default=150); pc.set_defaults(func=cmd_print_cell)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
