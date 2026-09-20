#!/usr/bin/env python3
"""Data gate for the BamiBERT aspect-sentiment CPT pilot (plan §3).

Candidates:
  * v3  = phamluan/vn_msa_sar_v3 train split @ pinned revision, Vietnamese-by-content rows only, sentences re-joined per
          review_id (first-appearance order; original sentence order cannot be verified from the release).
  * fw2 = first-person restaurant/hotel review passages extracted from the pinned FineWeb-2 shard (fw2_extract.py),
          tier 'strict' (primary).  The 'relaxed' tier is reported as a diagnostic upper bound only.

Pipeline (fixed before any downstream number is looked at):
  normalize -> exact dedup (casefolded NFC key) -> near-dup/boilerplate (>=50 % of 8-word shingles already seen)
  -> text-only decontamination vs train/dev/test text of all five benchmarks (exact key OR any shared 8-word shingle;
  labels never read) -> manual-audit sample (100 per source x domain, seed 42; labels in manual_audit_labels.json)
  -> BamiBERT tokenisation, chunking to <=254 content tokens at sentence boundaries -> holdout split by document
  -> gate: audited cells >=70 % firsthand, train >=4M unique non-special tokens, >=0.5M per domain, budget <=5 passes.
Writes data/bami_as_cpt_pilot/corpus_manifest.json plus corpus files.  Nothing here is tuned on downstream F1.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fw2_filter  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

AUDIT_PER_CELL = 100
NEAR_DUP_FRACTION = 0.5
HOLDOUT_DOC_FRACTION = 0.02
HOLDOUT_MAX_DOCS = 600
CONTENT_TOKENS = common.MAX_TOKENS - 2
SENT_SPLIT = fw2_filter.SENT_SPLIT


def load_v3() -> tuple[list[dict], dict]:
    path = common.V3_LOCAL
    sha = common.sha256_file(path)
    if sha != common.V3_TRAIN_SHA256:
        raise RuntimeError(f"v3 train parquet sha256 {sha} != pinned {common.V3_TRAIN_SHA256}")
    tb = pq.read_table(path).to_pandas()
    n_rows = len(tb)
    tb["text_n"] = tb["sentence"].map(common.normalize)
    tb["vi_text"] = tb["text_n"].map(fw2_filter.is_vietnamese)
    audit = {
        "rows": n_rows, "reviews": int(tb.review_id.nunique()), "unique_sentences": int(tb.text_n.nunique()),
        "language_tag_counts": {k: int(v) for k, v in tb.language.value_counts().items()},
        "source_counts": {k: int(v) for k, v in tb.source.value_counts().items()},
        "place_type_counts": {k: int(v) for k, v in tb.place_type.value_counts().items()},
        "vi_tag_rows": int((tb.language == "vi").sum()), "vi_tag_but_not_vi_text_rows": int(((tb.language == "vi") & ~tb.vi_text).sum()),
        "non_vi_tag_but_vi_text_rows": int(((tb.language != "vi") & tb.vi_text).sum()),
        "vi_by_content_rows": int(tb.vi_text.sum()),
        "language_check": "content-based: CJK absent AND (>=2 % Vietnamese-specific letters OR >=15 % common Vietnamese function words); the release's `language` tag is not trusted (259 vi-tagged rows are non-Vietnamese, 4,221 non-vi-tagged rows are Vietnamese)",
    }
    vi = tb[tb.vi_text]
    docs: list[dict] = []
    for rid, g in vi.groupby("review_id", sort=False):
        sents = list(dict.fromkeys(s for s in g.text_n if s))
        if not sents:
            continue
        domain = "hotel" if g.place_type.iloc[0] == "hotel" else "restaurant"
        docs.append({"source": "v3", "doc_id": f"v3:{rid}", "domain": domain, "platform": str(g.source.iloc[0]), "text": " ".join(sents), "n_sentences": len(sents)})
    audit.update({"vi_reviews": len(docs), "vi_unique_sentences": int(vi.text_n.nunique()),
                  "vi_reviews_by_domain": dict(collections.Counter(d["domain"] for d in docs)),
                  "vi_reviews_by_platform": dict(collections.Counter(d["platform"] for d in docs)),
                  "review_reconstruction": "unique sentences of a review_id joined in first-appearance row order; the original within-review order is not recoverable from the release, so reconstructed reviews are approximate"})
    return docs, audit


def load_fw2(tier: str) -> list[dict]:
    path = common.DATA_DIR / f"fw2_passages_{tier}.jsonl"
    docs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            docs.append({"source": "fw2", "doc_id": f"fw2:{r['doc_id']}", "unit_id": f"fw2:{r['doc_id']}:{r['passage_index']}", "domain": r["domain"], "platform": r["host"], "text": r["text"], "n_sentences": r["kept_sentences"]})
    return docs


def dedup_and_decontam(units: list[dict], bench: dict[str, list[str]]) -> tuple[list[dict], dict]:
    bench_keys: dict[str, str] = {}
    bench_shingles: dict[str, str] = {}
    for name, rows in bench.items():
        for r in rows:
            k = common.key_of(r)
            if k:
                bench_keys.setdefault(k, name)
                for s in common.shingles(k):
                    bench_shingles.setdefault(s, name)
    seen_keys: set[str] = set()
    seen_shingles: set[str] = set()
    kept: list[dict] = []
    c = collections.Counter()
    decontam_by_split = collections.Counter()
    decontam_by_source = collections.Counter()
    for u in units:
        k = common.key_of(u["text"])
        src = u["source"]
        c[f"{src}/candidates"] += 1
        if not k or not fw2_filter.WORD.search(k):
            c[f"{src}/empty"] += 1; continue
        if k in seen_keys:
            c[f"{src}/exact_dup"] += 1; continue
        sh = common.shingles(k)
        if len(sh) >= 3 and sum(s in seen_shingles for s in sh) / len(sh) >= NEAR_DUP_FRACTION:
            c[f"{src}/near_dup"] += 1; continue
        hit = bench_keys.get(k)
        if hit is None:
            for s in sh:
                if s in bench_shingles:
                    hit = bench_shingles[s]; break
        if hit is not None:
            c[f"{src}/decontam"] += 1; decontam_by_split[hit] += 1; decontam_by_source[src] += 1; continue
        seen_keys.add(k); seen_shingles.update(sh)
        c[f"{src}/kept"] += 1
        kept.append(u)
    report = {"counters": dict(c), "decontam_dropped_by_benchmark_split": dict(decontam_by_split), "decontam_dropped_by_source": dict(decontam_by_source),
              "benchmark_rows": {k: len(v) for k, v in bench.items()}, "benchmark_unique_keys": len(bench_keys), "benchmark_shingles": len(bench_shingles),
              "benchmark_key_set_sha256": common.sha256_lines(sorted(bench_keys)),
              "method": "exact casefolded-NFC key OR any shared 8-word shingle with train/dev/test text of vsfc, visfd, absa_restaurant, absa_hotel (text columns only, labels never read); near-dup = >=50 % of a unit's 8-word shingles already seen in kept units"}
    return kept, report


def audit_sample(units: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    cells = collections.defaultdict(list)
    for u in units:
        cells[(u["source"], u["domain"])].append(u)
    sample = []
    for cell in sorted(cells):
        pool = cells[cell]
        pick = rng.sample(pool, min(AUDIT_PER_CELL, len(pool)))
        for u in pick:
            sample.append({"source": u["source"], "domain": u["domain"], "unit_id": u.get("unit_id", u["doc_id"]), "platform": u["platform"], "text": u["text"]})
    return sample


def chunk_tokens(text: str, tok) -> list[list[int]]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= CONTENT_TOKENS:
        return [ids]
    sents = [s for s in SENT_SPLIT.split(text) if s and s.strip()]
    chunks: list[list[int]] = []
    cur: list[int] = []
    for s in sents:
        sid = tok(s, add_special_tokens=False)["input_ids"]
        while len(sid) > CONTENT_TOKENS:      # a single over-long sentence: hard split
            if cur:
                chunks.append(cur); cur = []
            chunks.append(sid[:CONTENT_TOKENS]); sid = sid[CONTENT_TOKENS:]
        if len(cur) + len(sid) > CONTENT_TOKENS:
            chunks.append(cur); cur = []
        cur.extend(sid)
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="strict", choices=("strict", "relaxed"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out) if args.out else (common.DATA_DIR if args.tier == "strict" else common.DATA_DIR / f"tier_{args.tier}")
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_stats = common.read_json(common.DATA_DIR / "fw2_extract_stats.json")
    manifest: dict = {"protocol": common.PROTOCOL, "tier": args.tier, "seed": common.SEED, "created_at": common.now(),
                      "sources": {"v3": {"repo": common.V3_REPO, "revision": common.V3_REVISION, "file": common.V3_TRAIN_FILE, "sha256": common.V3_TRAIN_SHA256, "license": common.V3_LICENSE, "split": "train only"},
                                  "fw2": {**extract_stats["shard"], "filter_source_sha256": extract_stats["filter_source_sha256"], "tier": args.tier, "extract_counters": extract_stats["tiers"][args.tier]["counters"], "passages_extracted": extract_stats["tiers"][args.tier]["passages"]}},
                      "excluded_sources": ["ViSoBERT social corpus (research-only / CC BY-NC)", "ViHoRec / CAMEO (CC BY-NC)", "pqthinh232/vietnamese-restaurant-review-sentiment-dataset (rights unclear)", "benchmark train text (never used for CPT)", "generic FineWeb-2 web text (only user-voice review passages are extracted)", "tripadvisor.* pages (machine-translated reviews)"]}

    v3_docs, v3_audit = load_v3()
    manifest["v3_audit"] = v3_audit
    fw2_units = load_fw2(args.tier)
    for d in v3_docs:
        d["unit_id"] = d["doc_id"]
    units = v3_docs + fw2_units
    print(f"[prep] candidates v3 {len(v3_docs)} reviews, fw2({args.tier}) {len(fw2_units)} passages ({time.time()-t0:.0f}s)", flush=True)

    bench = common.benchmark_texts()
    kept, dd = dedup_and_decontam(units, bench)
    manifest["dedup_decontam"] = dd
    print(f"[prep] kept {len(kept)} units after dedup/decontam: {dd['counters']} ({time.time()-t0:.0f}s)", flush=True)

    sample = audit_sample(kept, common.SEED)
    sample_path = out_dir / "manual_audit_sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels_path = out_dir / "manual_audit_labels.json"
    audit_report: dict = {"sample_file": str(sample_path.relative_to(common.ROOT)), "sample_sha256": common.sha256_file(sample_path), "per_cell": {}, "labels_present": labels_path.exists()}
    cell_sizes = collections.Counter((r["source"], r["domain"]) for r in sample)
    if labels_path.exists():
        labels = common.read_json(labels_path)
        lab = labels["labels"]
        for cell, n in sorted(cell_sizes.items()):
            ids = [r["unit_id"] for r in sample if (r["source"], r["domain"]) == cell]
            missing = [i for i in ids if i not in lab]
            firsthand = sum(lab.get(i) == "firsthand" for i in ids)
            audit_report["per_cell"][f"{cell[0]}/{cell[1]}"] = {"sampled": n, "labelled": n - len(missing), "firsthand": firsthand, "other": n - len(missing) - firsthand,
                                                             "firsthand_fraction": firsthand / max(1, n - len(missing)), "pass_70": (n - len(missing)) > 0 and firsthand / (n - len(missing)) >= common.MANUAL_AUDIT_MIN_FRACTION, "missing_labels": len(missing)}
        audit_report["labels_sha256"] = common.sha256_file(labels_path)
        audit_report["labelling_note"] = labels.get("note")
    else:
        for cell, n in sorted(cell_sizes.items()):
            audit_report["per_cell"][f"{cell[0]}/{cell[1]}"] = {"sampled": n, "labelled": 0}
    manifest["manual_audit"] = audit_report

    tok = common.load_bami_tokenizer()
    # tokenise + chunk; group units by document for the holdout split
    by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for u in kept:
        by_doc[u["doc_id"]].append(u)
    doc_ids = sorted(by_doc)
    rng = random.Random(common.SEED)
    rng.shuffle(doc_ids)
    per_source_docs = collections.defaultdict(list)
    for d in doc_ids:
        per_source_docs[by_doc[d][0]["source"]].append(d)
    holdout_docs: set[str] = set()
    for src, ids in per_source_docs.items():
        n_h = min(HOLDOUT_MAX_DOCS, max(1, int(round(HOLDOUT_DOC_FRACTION * len(ids)))))
        holdout_docs.update(ids[:n_h])
    rows = {"train": [], "holdout": []}
    tok_stats = collections.Counter()
    doc_stats = collections.Counter()
    for d in doc_ids:
        split = "holdout" if d in holdout_docs else "train"
        src = by_doc[d][0]["source"]
        doc_stats[f"{split}/{src}/docs"] += 1
        doc_stats[f"{split}/{src}/{by_doc[d][0]['domain']}/docs"] += 1
        for u in by_doc[d]:
            for ci, ids in enumerate(chunk_tokens(u["text"], tok)):
                rows[split].append({"unit_id": u["unit_id"], "doc_id": d, "chunk": ci, "source": src, "domain": u["domain"], "platform": u["platform"], "ids": ids, "text": u["text"] if ci == 0 else None})
                tok_stats[f"{split}/tokens"] += len(ids)
                tok_stats[f"{split}/{src}/tokens"] += len(ids)
                tok_stats[f"{split}/{u['domain']}/tokens"] += len(ids)
                tok_stats[f"{split}/{src}/{u['domain']}/tokens"] += len(ids)
                tok_stats[f"{split}/sequences"] += 1
    # fixed order for CPT: shuffle train sequences once with the seed (R and S read the same order)
    order = list(range(len(rows["train"])))
    random.Random(common.SEED + 1).shuffle(order)
    rows["train"] = [rows["train"][i] for i in order]
    for split in ("train", "holdout"):
        path = out_dir / f"corpus_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows[split]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[f"{split}_file"] = {"path": str(path.relative_to(common.ROOT)), "sha256": common.sha256_file(path), "sequences": len(rows[split])}
    manifest["holdout_doc_ids_sha256"] = common.sha256_lines(sorted(holdout_docs))
    manifest["holdout_docs"] = len(holdout_docs)
    manifest["tokens"] = {k: int(v) for k, v in sorted(tok_stats.items())}
    manifest["documents"] = {k: int(v) for k, v in sorted(doc_stats.items())}
    lengths = np.array([len(r["ids"]) for r in rows["train"]] or [0])
    manifest["train_sequence_length"] = {"mean": float(lengths.mean()), "median": float(np.median(lengths)), "p95": float(np.percentile(lengths, 95)), "max": int(lengths.max()), "chunked_units": int(sum(r["chunk"] > 0 for r in rows["train"]))}

    train_tokens = int(tok_stats["train/tokens"])
    per_domain = {d: int(tok_stats[f"train/{d}/tokens"]) for d in ("restaurant", "hotel")}
    passes = common.CPT_TOKEN_BUDGET / max(1, train_tokens)
    audit_ok = bool(audit_report["per_cell"]) and all(v.get("pass_70", False) for v in audit_report["per_cell"].values()) and audit_report["labels_present"]
    gate = {
        "train_unique_nonspecial_tokens": train_tokens, "min_required": common.MIN_TRAIN_TOKENS, "tokens_ok": train_tokens >= common.MIN_TRAIN_TOKENS,
        "per_domain_tokens": per_domain, "min_per_domain": common.MIN_DOMAIN_TOKENS, "domains_ok": all(v >= common.MIN_DOMAIN_TOKENS for v in per_domain.values()),
        "budget_tokens": common.CPT_TOKEN_BUDGET, "passes_at_budget": passes, "max_passes": common.MAX_PASSES, "passes_ok": passes <= common.MAX_PASSES,
        "manual_audit_ok": audit_ok, "manual_audit_cells": audit_report["per_cell"],
        "license_ok": True, "license_note": "v3 CC BY 4.0; FineWeb-2 ODC-By 1.0 (page-level rights unresolved; no redistribution of raw text in this pilot)",
    }
    gate["all_ok"] = bool(gate["tokens_ok"] and gate["domains_ok"] and gate["passes_ok"] and gate["manual_audit_ok"] and gate["license_ok"])
    manifest["gate"] = gate
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    manifest["source_hash"] = common.source_hash()
    common.write_json(out_dir / "corpus_manifest.json", manifest)
    print(json.dumps({"tokens": manifest["tokens"], "documents": manifest["documents"], "gate": gate}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
