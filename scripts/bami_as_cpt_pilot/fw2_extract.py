#!/usr/bin/env python3
"""Run fw2_filter over the whole pinned FineWeb-2 vie_Latn shard (2,319 row groups x 1000 docs) with multiprocessing.

Outputs (data/bami_as_cpt_pilot/):
  fw2_passages_strict.jsonl   one JSON object per kept passage (doc id, url host, row group, paragraph index, domain, scores, text)
  fw2_passages_relaxed.jsonl  same for the diagnostic relaxed tier (superset in spirit; computed independently per paragraph)
  fw2_extract_stats.json      per-tier counters, shard provenance/hash, filter source hash, elapsed
No benchmark text is read here; decontamination happens in prepare_corpus.py.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fw2_filter  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

TIERS = ("strict", "relaxed")
N_WORKERS = 48


def _work(rg: int) -> tuple[int, dict, dict]:
    pf = pq.ParquetFile(str(common.FW2_LOCAL))
    tb = pf.read_row_group(rg, columns=["id", "url", "text"])
    ids, urls, texts = tb.column("id").to_pylist(), tb.column("url").to_pylist(), tb.column("text").to_pylist()
    out = {t: [] for t in TIERS}
    stats = {t: {} for t in TIERS}
    for i, (doc_id, url, text) in enumerate(zip(ids, urls, texts)):
        for tier in TIERS:
            passages, st = fw2_filter.extract_passages(text, url, normalize=common.normalize, tier=tier)
            for k, v in st.items():
                stats[tier][k] = stats[tier].get(k, 0) + v
            host = urlparse(url or "").netloc.lower().replace("www.", "")
            for j, p in enumerate(passages):
                out[tier].append({"doc_id": doc_id, "row_group": rg, "doc_index": i, "passage_index": j, "host": host, "url": url,
                                  "domain": p.domain, "n_words": p.n_words, "first_person": p.first_person, "evaluation": p.evaluation,
                                  "review_nouns": p.review_nouns, "colloquial": p.colloquial, "experience": p.experience,
                                  "kept_sentences": p.kept_sentences, "total_sentences": p.total_sentences, "text": p.text})
    return rg, out, stats


def main() -> int:
    t0 = time.time()
    if common.sha256_file(common.FW2_LOCAL) != common.FW2_SHARD_SHA256:
        raise RuntimeError("FineWeb-2 shard sha256 does not match the pinned value")
    pf = pq.ParquetFile(str(common.FW2_LOCAL))
    n_rg = pf.num_row_groups
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = {t: (common.DATA_DIR / f"fw2_passages_{t}.jsonl").open("w", encoding="utf-8") for t in TIERS}
    totals = {t: {} for t in TIERS}
    counts = {t: 0 for t in TIERS}
    with mp.Pool(N_WORKERS) as pool:
        for done, (rg, out, stats) in enumerate(pool.imap_unordered(_work, range(n_rg), chunksize=4), 1):
            for tier in TIERS:
                for rec in sorted(out[tier], key=lambda r: (r["doc_index"], r["passage_index"])):
                    files[tier].write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[tier] += len(out[tier])
                for k, v in stats[tier].items():
                    totals[tier][k] = totals[tier].get(k, 0) + v
            if done % 100 == 0 or done == n_rg:
                print(f"[fw2] {done}/{n_rg} row groups, strict {counts['strict']} relaxed {counts['relaxed']} passages, {time.time()-t0:.0f}s", flush=True)
    for fh in files.values():
        fh.close()
    summary = {
        "shard": {"repo": common.FW2_REPO, "commit": common.FW2_COMMIT, "file": common.FW2_SHARD, "sha256": common.FW2_SHARD_SHA256,
                  "bytes": common.FW2_LOCAL.stat().st_size, "rows": pf.metadata.num_rows, "row_groups": n_rg, "license": common.FW2_LICENSE},
        "filter_source_sha256": common.sha256_file(common.SCRIPT_DIR / "fw2_filter.py"),
        "excluded_hosts": list(fw2_filter.EXCLUDED_HOSTS), "story_host_regex": fw2_filter.STORY_HOST.pattern, "mt_page_regex": fw2_filter.MT_PAGE.pattern,
        "tiers": {t: {"passages": counts[t], "counters": totals[t], "output": f"data/bami_as_cpt_pilot/fw2_passages_{t}.jsonl",
                      "output_sha256": common.sha256_file(common.DATA_DIR / f"fw2_passages_{t}.jsonl")} for t in TIERS},
        "elapsed_s": round(time.time() - t0, 1), "created_at": common.now(), "workers": N_WORKERS,
    }
    common.write_json(common.DATA_DIR / "fw2_extract_stats.json", summary)
    print(json.dumps({t: {"passages": counts[t], "counters": totals[t]} for t in TIERS}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
