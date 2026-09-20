#!/usr/bin/env python3
"""Range-parallel downloader for the pinned HPLT 3.0 `vie_Latn` shards (plan §3).

The server serves ~3-4 MB/s per connection, so each shard is fetched as byte ranges by a thread pool, reassembled by
concatenation (valid for byte ranges of one file), integrity-checked with zstandard and sha256-summed.  Chunks that are
already present with the expected length are reused, so the tool resumes after an interruption.

Usage: python fetch_hplt.py 9_1 9_2 8_4 7_1 [--jobs 10] [--chunk-mb 256]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BASE = common.HPLT_BASE_URL
OUT = common.HPLT_DIR


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, dest: Path, log: list) -> bool:
    want = end - start + 1
    if dest.exists() and dest.stat().st_size == want:
        return True
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, 7):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=3600) as resp, tmp.open("wb") as fh:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
            if tmp.stat().st_size == want:
                tmp.replace(dest)
                return True
            log.append(f"chunk {start}: got {tmp.stat().st_size} want {want} (attempt {attempt})")
        except Exception as error:  # noqa: BLE001
            log.append(f"chunk {start}: {type(error).__name__} {error} (attempt {attempt})")
        tmp.unlink(missing_ok=True)
        time.sleep(2 * attempt)
    return False


def check_zstd(path: Path) -> bool:
    import zstandard as zstd
    d = zstd.ZstdDecompressor()
    try:
        with path.open("rb") as fh, d.stream_reader(fh) as reader:
            while reader.read(1 << 22):
                pass
        return True
    except Exception as error:  # noqa: BLE001
        print(f"[fetch] zstd integrity error: {error!r}", flush=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--verify-all", action="store_true", help="re-verify existing complete files instead of skipping")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_mb * 1024 * 1024
    for shard in args.shards:
        final = OUT / f"{shard}.jsonl.zst"
        url = f"{BASE}/{shard}.jsonl.zst"
        size = remote_size(url)
        print(f"[fetch] {shard} size={size} bytes", flush=True)
        if final.exists() and final.stat().st_size == size and check_zstd(final):
            print(f"[fetch] {shard} already complete and valid", flush=True)
            continue
        final.unlink(missing_ok=True)
        parts = OUT / ".parts" / shard
        parts.mkdir(parents=True, exist_ok=True)
        todo = []
        for start in range(0, size, chunk):
            end = min(start + chunk, size) - 1
            dest = parts / str(start)
            if not (dest.exists() and dest.stat().st_size == end - start + 1):
                todo.append((start, end, dest))
        print(f"[fetch] {shard}: {len(todo)} chunks to fetch of {len(range(0, size, chunk))}", flush=True)
        done_bytes = sum(d.stat().st_size for d in parts.iterdir() if d.name.isdigit())
        lock = threading.Lock()
        log: list[str] = []
        t0 = time.time()

        def worker(item):
            nonlocal done_bytes
            start, end, dest = item
            ok = fetch_range(url, start, end, dest, log)
            with lock:
                if ok:
                    done_bytes += end - start + 1
            return ok

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(worker, item) for item in todo]
            while any(not f.done() for f in futures):
                time.sleep(15)
                print(f"[fetch] {shard}: {done_bytes/size*100:.1f}% ({done_bytes/1e9:.1f}/{size/1e9:.1f} GB) {time.time()-t0:.0f}s", flush=True)
            results = [f.result() for f in futures]
        for line in log[-20:]:
            print(f"[fetch] {shard} {line}", flush=True)
        if not all(results):
            print(f"[fetch] {shard}: {results.count(False)} chunks FAILED", flush=True)
            return 1
        chunk_files = sorted((d for d in parts.iterdir() if d.name.isdigit()), key=lambda p: int(p.name))
        with final.open("wb") as out:
            for d in chunk_files:
                with d.open("rb") as fh:
                    while True:
                        block = fh.read(1 << 22)
                        if not block:
                            break
                        out.write(block)
        if final.stat().st_size != size:
            print(f"[fetch] {shard} INCOMPLETE {final.stat().st_size} != {size}", flush=True)
            return 1
        if not check_zstd(final):
            print(f"[fetch] {shard} integrity check FAILED", flush=True)
            return 1
        sha = hashlib.sha256()
        with final.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                sha.update(block)
        line = f"{shard} bytes={size} sha256={sha.hexdigest()}"
        print(f"[fetch] {line}", flush=True)
        with (OUT / "sha256.txt").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        for d in chunk_files:
            d.unlink()
        shutil.rmtree(parts, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
