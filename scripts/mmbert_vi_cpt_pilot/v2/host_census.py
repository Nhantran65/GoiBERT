#!/usr/bin/env python3
"""v2 §2 "vá 2" step 1: host census over the five pinned HPLT 3.0 shards (no tokenisation).

Each raw `.jsonl.zst` line is a document whose fourth field is the source URL; the URL is taken from the first bytes of
the line with a cheap regex (no JSON parse), so the whole 77 GB of compressed shards can be censused.  The census reports
documents per host overall and per bin (a file = a bin), plus — joined with the v1 tokenised pool — tokens per host and
the hosts that match the mainstream-news reference list and the UGC whitelist.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C  # noqa: E402

URL_RE = re.compile(rb'"u"\s*:\s*"([^"\\]{1,512})"')


def scan_shard(job) -> dict:
    shard, path = job
    counts: collections.Counter = collections.Counter()
    n = 0
    t0 = time.time()
    proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE, bufsize=1 << 22)
    try:
        for line in proc.stdout:
            n += 1
            m = URL_RE.search(line[:1024])
            if m is None:
                continue
            url = m.group(1).decode("utf-8", "ignore")
            host = C.host_of(url)
            if host:
                counts[host] += 1
    finally:
        proc.stdout.close()
        proc.wait()
    return {"shard": shard, "docs": n, "hosts": counts, "elapsed_s": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.ART_V2 / "host_census.json"))
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    shards = sorted(C.common.HPLT_DIR.glob("*.jsonl.zst"))
    if not shards:
        raise FileNotFoundError("no raw HPLT shards")
    jobs = [(p.name.split(".")[0], p) for p in shards]
    by_host: collections.Counter = collections.Counter()
    by_host_bin: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per_shard = {}
    with mp.Pool(args.workers) as pool:
        for res in pool.imap_unordered(scan_shard, jobs):
            bin_name = res["shard"].split("_")[0]
            for host, c in res["hosts"].items():
                by_host[host] += c
                by_host_bin[host][bin_name] += c
            per_shard[res["shard"]] = {"docs": res["docs"], "hosts": len(res["hosts"]), "elapsed_s": res["elapsed_s"]}
            print(f"[census] {res['shard']}: {res['docs']} docs, {len(res['hosts'])} hosts, {res['elapsed_s']}s", flush=True)
    total_docs = sum(v["docs"] for v in per_shard.values())

    # join with the v1 tokenised pool for token shares per host
    pool_tokens: collections.Counter = collections.Counter()
    pool_docs: collections.Counter = collections.Counter()
    for r in C.pool_table():
        if r["split"] != "pool":
            continue
        pool_tokens[r["host"]] += r["n_tokens"]
        pool_docs[r["host"]] += 1
    pool_total_tokens = sum(pool_tokens.values())

    top = by_host.most_common(args.top)
    rows = []
    for host, c in top:
        rows.append({"host": host, "docs_raw": c, "docs_raw_share": c / max(1, total_docs), "pool_docs": pool_docs.get(host, 0),
                     "pool_tokens": pool_tokens.get(host, 0), "pool_token_share": pool_tokens.get(host, 0) / max(1, pool_total_tokens),
                     "by_bin": dict(by_host_bin[host]),
                     "mainstream_news": C.host_matches(host, C.MAINSTREAM_NEWS_HOSTS),
                     "ugc_whitelist": C.host_matches(host, set(C.UGC_WHITELIST)),
                     "ugc_excluded": C.host_matches(host, C.UGC_EXCLUDED_HOSTS)})
    seed_rows = [{"host": h, "docs_raw": by_host.get(h, 0), "pool_tokens": pool_tokens.get(h, 0), "by_bin": dict(by_host_bin.get(h, {}))}
                 for h in sorted(set(C.MAINSTREAM_NEWS_HOSTS) | set(C.UGC_WHITELIST) | C.UGC_EXCLUDED_HOSTS)]
    report = {"created_at": C.common.now(), "source": {"shards": [j[0] for j in jobs], "sha256_file": str(C.common.HPLT_DIR / "sha256.txt"),
                                                       "license": C.common.HPLT_LICENSE},
              "method": "regex extraction of the `u` field from the first 1 KB of every raw jsonl line; no JSON parse, no tokenisation",
              "totals": {"docs_raw": total_docs, "unique_hosts": len(by_host), "pool_docs": sum(pool_docs.values()), "pool_tokens": pool_total_tokens},
              "top_hosts": rows, "seed_hosts": seed_rows, "per_shard": per_shard,
              "elapsed_s": round(time.time() - t0, 1)}
    C.write_json(Path(args.out), report)
    print(f"[census] {total_docs} docs, {len(by_host)} hosts in {report['elapsed_s']}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
