#!/usr/bin/env python3
"""v2 §2 "vá 2": UGC register bucket at paragraph level from whitelist platforms.

Streams the pinned raw HPLT shards, keeps documents whose host is on the UGC whitelist (bins 7-9), applies the v1
document filters plus the benchmark decontamination, extracts paragraphs that (a) have >= 12 letter/digit units,
(b) carry a first-person / experience / evaluation / colloquial cue from `scripts/bami_as_cpt_pilot/fw2_filter.py`
(no domain marker is used), and (c) are not listings/prices/URLs/phones; kept paragraphs of one document are joined
again by document (never across documents).  A paragraph is dropped when its parent document fails vá 1 (S1+S2) or when
the paragraph's own S1 falls below the locked reference percentile.  Kept register documents are tokenised with the
mmBERT tokenizer and written with the same per-part layout as the pool, so `build_arms.py`-style code can consume them.

Outputs: data/mmbert_vi_cpt_pilot/v2/ugc/{ugc.docs.jsonl, ugc.tokens.npy, ugc.offsets.npy, manifest.json}
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C  # noqa: E402
import spun_filter as S  # noqa: E402

PILOT_DIR = C.PILOT_DIR
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))
import scan_hplt  # noqa: E402  (v1 segment-level benchmark decontamination)

sys.path.insert(0, str(C.ROOT / "scripts" / "bami_as_cpt_pilot"))
from fw2_filter import FIRST_PERSON, EXPERIENCE, EVALUATION, COLLOQUIAL, PRICE_LIST, URL as URL_RX, PHONE, LIST_ITEM  # noqa: E402

OUT_DIR = C.DATA_V2 / "ugc"
URL_RE = re.compile(rb'"u"\s*:\s*"([^"\\]{1,512})"')
MIN_UNITS = 12
MIN_CHARS, MAX_CHARS = 200, 120_000
UGC_HOSTS = set(C.UGC_WHITELIST) | {"thuvienphapluat.vn", "danluat.thuvienphapluat.vn", "hoidap.thuvienphapluat.vn", "luatminhkhue.vn"}
UGC_HOST_REASONS = dict(C.UGC_WHITELIST) | {
    "thuvienphapluat.vn": "census expansion: legal Q&A pages carry first-person user questions (court-judgment pages are rejected by the cue filter)",
    "danluat.thuvienphapluat.vn": "census expansion: subdomain of the same legal Q&A board",
    "hoidap.thuvienphapluat.vn": "census expansion: subdomain of the same legal Q&A board",
    "luatminhkhue.vn": "census expansion: law-firm Q&A pages open with the user's own question",
}


def paragraph_cues(p: str) -> bool:
    if len(C.common.UNIT.findall(p)) < MIN_UNITS:
        return False
    if not (FIRST_PERSON.search(p) or EXPERIENCE.search(p) or EVALUATION.search(p) or COLLOQUIAL.search(p)):
        return False
    if PRICE_LIST.search(p) or URL_RX.search(p) or PHONE.search(p):
        return False
    if LIST_ITEM.match(p) and len(p) < 200:
        return False
    return True


def extract_paragraphs(text: str) -> list[str]:
    out = []
    for block in re.split(r"\n{1,}", text):
        block = block.strip()
        if not block:
            continue
        if paragraph_cues(block):
            out.append(block)
            continue
        # long blocks without newlines: fall back to sentences
        if len(block) > 400:
            for sent in scan_hplt.SEG_SPLIT.split(block):
                sent = sent.strip()
                if sent and paragraph_cues(sent):
                    out.append(sent)
    return out


def scan_shard(job) -> dict:
    shard, path, se_n = job
    t0 = time.time()
    tok = C.common.load_tokenizer()
    tok.model_max_length = int(1e9)
    bench_keys, _ = C.common.benchmark_key_index()
    bench_shingles = C.common.benchmark_shingle_array()
    c = collections.Counter()
    docs: list[dict] = []
    ids: list = []
    import numpy as np
    proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE, bufsize=1 << 22)
    try:
        for line in proc.stdout:
            c["docs_in"] += 1
            m = URL_RE.search(line[:1024])
            if m is None:
                continue
            host = C.host_of(m.group(1).decode("utf-8", "ignore"))
            if not C.host_matches(host, UGC_HOSTS):
                continue
            c["docs_whitelist"] += 1
            o = json.loads(line)
            if o.get("filter") != "keep":
                c["drop_filter"] += 1
                continue
            lang, prob = o.get("lang") or [], o.get("prob") or []
            if not lang or lang[0] != "vie_Latn" or not prob or float(prob[0]) < 0.9:
                c["drop_lang"] += 1
                continue
            seg = o.get("seg_langs") or []
            if seg and sum(1 for s in seg if s == "vie_Latn") / len(seg) < 0.9:
                c["drop_seg_langs"] += 1
                continue
            wr = o.get("web-register") or {}
            if float(wr.get("MT", 0.0)) >= 0.5:
                c["drop_mt"] += 1
                continue
            text = C.common.normalize(o.get("text", ""))
            if not text:
                c["drop_empty"] += 1
                continue
            pii = o.get("pii") or []
            pii_chars = sum(max(0, int(b) - int(a)) for a, b in pii if isinstance(a, (int, float)) and isinstance(b, (int, float)))
            if len(pii) > 5 or (len(text) and pii_chars / len(text) > 0.01):
                c["drop_pii"] += 1
                continue
            if not C.common.is_vietnamese(text) or not (MIN_CHARS <= len(text) <= MAX_CHARS):
                c["drop_language_or_length"] += 1
                continue
            key = C.common.key_of(text)
            if hashlib.sha1(key.encode()).hexdigest() in bench_keys:
                c["drop_decontam_exact"] += 1
                continue
            paragraphs = extract_paragraphs(text)
            if not paragraphs:
                c["drop_no_paragraph"] += 1
                continue
            # benchmark decontamination at paragraph level (drop hitting paragraphs)
            kept, hit = [], 0
            for p in paragraphs:
                segs, n_seg, n_hit = scan_hplt.hit_segments(p, bench_shingles)
                if n_seg and n_hit / n_seg > 0.2:
                    hit += 1
                    continue
                kept.append(" ".join(segs) if n_hit else p)
            if not kept or hit / max(1, len(paragraphs)) > 0.2:
                c["drop_decontam_doc"] += 1
                continue
            joined = C.common.normalize("\n".join(kept))
            if len(joined) < MIN_CHARS:
                c["drop_short_after_extract"] += 1
                continue
            c["docs_extracted"] += 1
            docs.append({"shard": shard, "bin": shard.split("_")[0], "host": host, "url": o.get("u"), "id": str(o.get("id")),
                         "kh64": int.from_bytes(hashlib.blake2b(C.common.key_of(joined).encode(), digest_size=8).digest(), "big"),
                         "chars": len(joined), "n_paragraphs": len(kept), "text": joined})
    finally:
        proc.stdout.close()
        proc.wait()
    # vá 1 (document level) + paragraph-level S1, then tokenise
    texts = [d["text"] for d in docs]
    if texts:
        res = S.score_texts(texts)
        import numpy as np
        s1_p5 = float(C.read_json(S.MODEL_DIR / "thresholds.json")["threshold_s1_p5"])
        for d, s1, prob, drop in zip(docs, res["s1_mean"], res["s2_prob"], res["drop"]):
            d["s1_doc"] = None if s1 != s1 else float(s1)
            d["s2_doc"] = float(prob)
            d["spun_drop"] = bool(drop)
        kept_docs = [d for d in docs if not d["spun_drop"]]
        c["docs_after_spun"] += len(kept_docs)
        for d in kept_docs:
            paras = [p for p in d["text"].split("\n") if p.strip()]
            scores = S.s1_score(paras)
            keep = [p for p, (s1, _) in zip(paras, scores) if (s1 == s1 and s1 >= s1_p5)]
            if keep:
                d["text"] = "\n".join(keep)
                d["n_paragraphs"] = len(keep)
                d["chars"] = len(d["text"])
            else:
                c["drop_all_paragraphs_spun"] += 1
                d["text"] = ""
        kept_docs = [d for d in kept_docs if d["text"]]
        kept_with_tokens = []
        for d in kept_docs:
            seq = tok(d["text"], add_special_tokens=False)["input_ids"]
            if len(seq) < 50:
                c["drop_too_short_tokens"] += 1
                continue
            d["n_tokens"] = len(seq)
            ids.append(np.asarray(seq, dtype=np.uint32))
            kept_with_tokens.append(d)
        kept_docs = kept_with_tokens
        flat = np.concatenate(ids) if ids else np.zeros(0, dtype=np.uint32)
    else:
        kept_docs, flat = [], np.zeros(0, dtype=np.uint32)
        c["docs_after_spun"] = 0
    side_effect = []
    if se_n:
        before = sorted(docs, key=lambda d: (d["kh64"], d["id"]))[:se_n]
        for d in before:
            side_effect.append({"shard": d["shard"], "host": d["host"], "url": d.get("url"), "id": d.get("id"),
                                "drop": bool(d.get("spun_drop")), "s1": d.get("s1_doc"), "s2": d.get("s2_doc"),
                                "text": d["text"][:4000]})
    return {"shard": shard, "counters": dict(c), "docs": kept_docs, "tokens": flat, "side_effect": side_effect,
            "elapsed_s": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--side-effect-sample", type=int, default=0, help="record N pre-vá-1 UGC documents per shard")
    ap.add_argument("--only-shard", default=None, help="run a single shard (used for the side-effect sample)")
    args = ap.parse_args()
    import numpy as np
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(C.common.HPLT_DIR.glob("*.jsonl.zst"))
    if args.only_shard:
        shards = [p for p in shards if p.name.split(".")[0] == args.only_shard]
    jobs = [(p.name.split(".")[0], p, args.side_effect_sample) for p in shards]
    counters = collections.Counter()
    all_docs: list[dict] = []
    token_arrays = []
    side_effect: list[dict] = []
    with mp.Pool(args.workers) as pool:
        for res in pool.imap_unordered(scan_shard, jobs):
            counters.update(res["counters"])
            all_docs.extend(res["docs"])
            token_arrays.append(res["tokens"])
            side_effect.extend(res.get("side_effect", []))
            print(f"[ugc] {res['shard']}: {len(res['docs'])} docs, {int(res['tokens'].size)} tokens, {res['elapsed_s']}s", flush=True)
    flat = np.concatenate(token_arrays) if token_arrays else np.zeros(0, dtype=np.uint32)
    all_docs.sort(key=lambda d: (d["shard"], d["id"]))
    np.save(out_dir / "ugc.tokens.npy", flat)
    np.save(out_dir / "ugc.offsets.npy", np.concatenate([[0], np.cumsum([d["n_tokens"] for d in all_docs])]).astype(np.int64) if all_docs else np.zeros(1, np.int64))
    with (out_dir / "ugc.docs.jsonl").open("w", encoding="utf-8") as fh:
        for i, d in enumerate(all_docs):
            rec = {k: v for k, v in d.items() if k != "text"}
            rec["i"] = i
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    by_host = collections.Counter()
    tok_by_host = collections.Counter()
    by_bin = collections.Counter()
    for d in all_docs:
        by_host[d["host"]] += 1
        tok_by_host[d["host"]] += d["n_tokens"]
        by_bin[d["bin"]] += 1
    manifest = {"created_at": C.common.now(), "counters": dict(counters), "documents": len(all_docs), "tokens": int(flat.size),
                "by_host_docs": dict(by_host.most_common()), "by_host_tokens": dict(tok_by_host.most_common()), "by_bin_docs": dict(by_bin),
                "whitelist": UGC_HOST_REASONS, "excluded_hosts": sorted(C.UGC_EXCLUDED_HOSTS),
                "rules": {"min_units": MIN_UNITS, "cues": "FIRST_PERSON|EXPERIENCE|EVALUATION|COLLOQUIAL (fw2_filter lexicon)",
                          "dropped": "PRICE_LIST|URL|PHONE|short list items", "spun": "document-level S1+S2 then paragraph-level S1 >= reference p5"},
                "elapsed_s": round(time.time() - t0, 1)}
    C.write_json(out_dir / "manifest.json", manifest)
    if side_effect:
        with (out_dir / "side_effect_sample.jsonl").open("w", encoding="utf-8") as fh:
            for r in side_effect:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ugc] side-effect sample: {len(side_effect)} pre-vá-1 documents written", flush=True)
    print(json.dumps({"documents": len(all_docs), "tokens": int(flat.size), "counters": dict(counters)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
