#!/usr/bin/env python3
"""Data gate, stage 1 (§3): one raw HPLT 3.0 `vie_Latn` jsonl part -> filtered documents.

Fixed filters (never tuned on downstream F1):
  * `filter == "keep"`; `lang[0] == "vie_Latn"` and `prob[0] >= 0.9`; >= 90 % of `seg_langs` vie_Latn
  * drop `web-register.MT >= 0.5` (a deterministic sample of those is written for the manual audit cell)
  * drop documents with more than 5 PII spans or more than 1 % PII characters
  * content Vietnamese check (`common.is_vietnamese`)
  * normalise (HTML-unescape, NFC, whitespace collapse); keep diacritics/spelling
  * decontamination against train/dev/test text of all five benchmarks: exact key OR any shared 8-word shingle;
    hit segments are removed from the text and a document with > 20 % hit segments is dropped entirely
  * register buckets from `web-register`: opinion_review = max(OP, re) >= 0.5; discussion = ID >= 0.5 (and not
    opinion_review); general = the rest
  * deterministic uniform document subsample (all register buckets included): keep when blake2b(id) % sample_mod == 0;
    the rule is id-based, so identical texts keep identical decisions and the sample is unbiased within a shard

Writes `<out>.docs.jsonl` (text + metadata), `<out>.meta.jsonl` (dedup/decontam fields, no text) and
`<out>.counters.json`.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402

SEG_SPLIT = re.compile(r"(?<=[.!?…])[\s\"'’”)\]]+|[\n\r]+")
MIN_CHARS = 200            # ~50 mmBERT tokens; exact token bounds enforced after tokenisation
MAX_CHARS = 120_000        # ~29k mmBERT tokens; exact token bounds enforced after tokenisation
MAX_PII_SPANS = 5
MAX_PII_CHAR_FRACTION = 0.01
MT_AUDIT_MOD = 512         # keep 1/512 of MT-dropped documents for the audit cell
EXCERPT_MOD = 64           # keep a short excerpt of 1/64 of the kept documents for the manual audit cells


def id_sample_keep(doc_id: str, mod: int) -> bool:
    return int.from_bytes(hashlib.blake2b(doc_id.encode(), digest_size=8).digest(), "big") % mod == 0


def hit_segments(text: str, bench_shingles: np.ndarray) -> tuple[list[str], int, int]:
    """Split into sentence segments; return (segments without benchmark overlap, n_segments, n_hit)."""
    segs = [s.strip() for s in SEG_SPLIT.split(text)]
    segs = [s for s in segs if s]
    keep, hit = [], 0
    for s in segs:
        h = common.shingle_hashes(common.key_of(s))
        if h.size:
            pos = np.searchsorted(bench_shingles, h)
            pos[pos >= bench_shingles.size] = 0
            overlap = bool((bench_shingles[pos] == h).any())
        else:
            overlap = False
        if overlap:
            hit += 1
        else:
            keep.append(s)
    return keep, len(segs), hit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, help="raw .jsonl part of one HPLT shard")
    ap.add_argument("--shard", required=True, help="shard name, e.g. 9_1 (bin = leading component)")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--sample-mod", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bench_keys, _bench_stats = common.benchmark_key_index()
    bench_shingles = common.benchmark_shingle_array()
    c = collections.Counter()
    by_group = collections.Counter()
    reg_hist = collections.Counter()
    docs_f = (prefix.parent / (prefix.name + ".docs.jsonl")).open("w", encoding="utf-8")
    meta_f = (prefix.parent / (prefix.name + ".meta.jsonl")).open("w", encoding="utf-8")
    mt_f = (prefix.parent / (prefix.name + ".mt_audit.jsonl")).open("w", encoding="utf-8")
    excerpt_f = (prefix.parent / (prefix.name + ".excerpts.jsonl")).open("w", encoding="utf-8")
    n = 0
    with Path(args.part).open("r", encoding="utf-8") as fh:
        for line in fh:
            if args.limit and n >= args.limit:
                break
            n += 1
            c["docs_in"] += 1
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                c["bad_json"] += 1
                continue
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
            mt = float(wr.get("MT", 0.0))
            if mt >= 0.5:
                c["drop_mt"] += 1
                if id_sample_keep(str(o.get("id", "")), MT_AUDIT_MOD):
                    mt_f.write(json.dumps({"shard": args.shard, "id": o.get("id"), "url": o.get("u"), "mt": mt,
                                           "op": float(wr.get("OP", 0.0)), "re": float(wr.get("re", 0.0)), "id_reg": float(wr.get("ID", 0.0)),
                                           "text": common.normalize(o.get("text", ""))[:4000]}, ensure_ascii=False) + "\n")
                continue
            text = common.normalize(o.get("text", ""))
            if not text:
                c["drop_empty"] += 1
                continue
            pii = o.get("pii") or []
            pii_chars = sum(max(0, int(b) - int(a)) for a, b in pii if isinstance(a, (int, float)) and isinstance(b, (int, float)))
            if len(pii) > MAX_PII_SPANS or (len(text) and pii_chars / len(text) > MAX_PII_CHAR_FRACTION):
                c["drop_pii"] += 1
                continue
            if not common.is_vietnamese(text):
                c["drop_not_vi"] += 1
                continue
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                c["drop_length_prefilter"] += 1
                continue
            key = common.key_of(text)
            if hashlib.sha1(key.encode()).hexdigest() in bench_keys:
                c["drop_decontam_exact"] += 1
                continue
            kept_segments, n_segments, n_hit = hit_segments(text, bench_shingles)
            if n_segments and n_hit / n_segments > 0.20:
                c["drop_decontam_doc"] += 1
                continue
            if n_hit:
                c["decontam_segments_removed"] += n_hit
                text = common.normalize(" ".join(kept_segments))
                if len(text) < MIN_CHARS:
                    c["drop_decontam_short"] += 1
                    continue
                key = common.key_of(text)
            op, re_, idr = float(wr.get("OP", 0.0)), float(wr.get("re", 0.0)), float(wr.get("ID", 0.0))
            group = "register_opinion" if max(op, re_) >= 0.5 else ("register_discussion" if idr >= 0.5 else "general")
            doc_id = str(o.get("id") or common.doc_id_of(text, args.shard))
            if not id_sample_keep(doc_id, args.sample_mod):
                c["drop_subsample"] += 1
                continue
            sh = common.sampled_shingle_hashes(key)
            rec = {"id": doc_id, "shard": args.shard, "bin": args.shard.split("_")[0], "group": group,
                   "chars": len(text), "n_units": common.n_units(text), "key_h": hashlib.sha1(key.encode()).hexdigest(),
                   "kh64": int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big"),
                   "sh": sh, "op": op, "re": re_, "id_reg": idr, "mt": mt, "pii_n": len(pii), "pii_chars": pii_chars,
                   "cluster_size": int(o.get("cluster_size") or 0), "url": o.get("u"), "crawl_id": o.get("crawl_id"), "ts": o.get("ts")}
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            docs_f.write(json.dumps({**rec, "text": text}, ensure_ascii=False) + "\n")
            c["kept"] += 1
            by_group[group] += 1
            if id_sample_keep(doc_id, EXCERPT_MOD):
                excerpt_f.write(json.dumps({k: rec[k] for k in ("id", "bin", "group", "url", "op", "re", "id_reg", "mt", "chars")}
                                           | {"excerpt": text[:1200]}, ensure_ascii=False) + "\n")
                c["excerpts"] += 1
            for key_name, val in (("OP", op), ("re", re_), ("ID", idr)):
                if val >= 0.5:
                    reg_hist[key_name] += 1
    docs_f.close()
    meta_f.close()
    mt_f.close()
    excerpt_f.close()
    counters = {"part": str(args.part), "shard": args.shard, "sample_mod": args.sample_mod, "docs_read": n,
                "counters": dict(c), "kept_by_group": dict(by_group), "register_ge_0.5": dict(reg_hist),
                "source": {"repo": common.HPLT_REPO, "hf_revision": common.HPLT_HF_REVISION, "shard": args.shard,
                           "url": f"{common.HPLT_BASE_URL}/{args.shard}.jsonl.zst", "license": common.HPLT_LICENSE},
                "benchmark_index": {"unique_keys": len(bench_keys), "shingle_set_size": int(bench_shingles.size)},
                "elapsed_s": round(time.time() - t0, 1)}
    common.write_json(prefix.parent / (prefix.name + ".counters.json"), counters)
    print(json.dumps({"part": prefix.name, "kept": c["kept"], "docs": n, "group": dict(by_group), "elapsed_s": counters["elapsed_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
