#!/usr/bin/env python3
"""Data gate, stage 2 (§3): global exact/near-duplicate removal and holdout split over the scanned HPLT pool.

Streams the `<shard>_<part>.meta.jsonl` files of stage 1 in a locked order (shard order given on the command line, then
part name) and applies, in that order:
  * exact duplicate removal on the NFC-casefold key (sha1);
  * near-duplicate removal on the sampled 8-word shingles (hash % 16 == 0): a document is dropped when at least 50 % of
    its sampled shingles have already been seen and it has >= 3 of them.  Membership uses a 2**32-bit bloom filter
    (single hash, false-positive rate ~9 % at pool scale), which is documented in the manifest;
  * holdout split by document id (blake2b hash), before any arm sampling:
      (a) HPLT holdout: 1/200 of the pool, capped at 5 000 documents;
      (c) register holdout: 1/512 of the register documents, capped at 2 000 documents;
  * the remaining documents form the arm pool, written in the locked order to `pool/pool.jsonl` together with the
    stage-1 counters aggregated over all parts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402

BLOOM_BITS = 1 << 32
NEAR_DUP_FRACTION = 0.5
HOLDOUT_A_MOD = 200           # 0.5 % of the pool
HOLDOUT_A_CAP = 5000
HOLDOUT_C_MOD = 512
HOLDOUT_C_CAP = 2000


class Bloom:
    def __init__(self, bits: int = BLOOM_BITS):
        self.bits = bits
        self.words = np.zeros(bits // 64, dtype=np.uint64)

    def seen_fraction(self, hashes: np.ndarray) -> float:
        if hashes.size == 0:
            return 0.0
        idx = (hashes & np.uint64(self.bits - 1)).astype(np.uint64)
        w, b = idx >> np.uint64(6), idx & np.uint64(63)
        return float(((self.words[w] >> b) & np.uint64(1)).mean())

    def add(self, hashes: np.ndarray) -> None:
        if hashes.size == 0:
            return
        idx = (hashes & np.uint64(self.bits - 1)).astype(np.uint64)
        w, b = (idx >> np.uint64(6)), (idx & np.uint64(63))
        uniq, inv = np.unique(w, return_inverse=True)
        vals = np.zeros(uniq.size, dtype=np.uint64)
        np.bitwise_or.at(vals, inv, np.left_shift(np.uint64(1), b))
        self.words[uniq] |= vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-dir", default=str(common.DATA_DIR / "scan"))
    ap.add_argument("--out-dir", default=str(common.DATA_DIR / "pool"))
    ap.add_argument("--shards", nargs="+", required=True, help="shard names in locked order, e.g. 10_1 9_2 9_1 8_4 7_1")
    args = ap.parse_args()
    t0 = time.time()
    scan = Path(args.scan_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bloom = Bloom()
    seen_keys: set[str] = set()
    counters = collections.Counter()
    by_group = collections.Counter()
    by_bin = collections.Counter()
    holdout_c_docs: list[str] = []
    c = collections.Counter()
    pool_f = (out_dir / "pool.jsonl").open("w", encoding="utf-8")
    holdout_f = (out_dir / "holdout_hlpt.jsonl").open("w", encoding="utf-8")

    for shard in args.shards:
        parts = sorted(p for p in scan.glob(f"{shard}_*.meta.jsonl"))
        if not parts:
            raise FileNotFoundError(f"no stage-1 meta files for shard {shard} in {scan}")
        for meta_path in parts:
            ctr_path = Path(str(meta_path).replace(".meta.jsonl", ".counters.json"))
            if ctr_path.is_file():
                counters.update(common.read_json(ctr_path)["counters"])
            with meta_path.open(encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    c["docs_in"] += 1
                    if r["key_h"] in seen_keys:
                        c["dup_exact"] += 1
                        continue
                    sh = np.asarray(r["sh"], dtype=np.uint64)
                    if sh.size >= 3 and bloom.seen_fraction(sh) >= NEAR_DUP_FRACTION:
                        c["dup_near"] += 1
                        continue
                    seen_keys.add(r["key_h"])
                    bloom.add(sh)
                    kh64 = int(r["kh64"])
                    is_register = r["group"].startswith("register")
                    if is_register and kh64 % HOLDOUT_C_MOD == 0 and len(holdout_c_docs) < HOLDOUT_C_CAP:
                        rec = {k: v for k, v in r.items() if k != "sh"}
                        rec["split"] = "holdout_register"
                        holdout_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        holdout_c_docs.append(r["id"])
                        c["holdout_register"] += 1
                        continue
                    if kh64 % HOLDOUT_A_MOD == 0 and c["holdout_hlpt"] < HOLDOUT_A_CAP:
                        rec = {k: v for k, v in r.items() if k != "sh"}
                        rec["split"] = "holdout_hlpt"
                        holdout_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        c["holdout_hlpt"] += 1
                        continue
                    rec = {k: v for k, v in r.items() if k != "sh"}
                    pool_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    c["pool"] += 1
                    by_group[r["group"]] += 1
                    by_bin[r["bin"]] += 1
    pool_f.close()
    holdout_f.close()
    manifest = {"stage": "dedup_pool", "created_at": common.now(), "shards_locked_order": args.shards,
                "horizon": "stage-1 filtered pool; decontamination (benchmark text) already applied in stage 1",
                "dedup": {"exact_key": "sha1(NFC-casefold key)", "near_dup": f"sampled 8-word shingles (hash % {common.SHINGLE_MOD} == 0), drop when seen fraction >= {NEAR_DUP_FRACTION} and >= 3 sampled shingles",
                          "near_dup_membership": f"bloom filter, {BLOOM_BITS} bits, 1 hash per shingle", "hash": "polynomial rolling hash of blake2b(word) hashes, base 0x100000001B3 mod 2**64"},
                "holdout": {"hlpt": {"rule": f"blake2b(id) % {HOLDOUT_A_MOD} == 0", "cap": HOLDOUT_A_CAP, "docs": c["holdout_hlpt"]},
                            "register": {"rule": f"blake2b(id) % {HOLDOUT_C_MOD} == 0", "cap": HOLDOUT_C_CAP, "docs": c["holdout_register"]}},
                "counters": dict(c), "stage1_counters_aggregate": dict(counters), "pool_by_group": dict(by_group), "pool_by_bin": dict(by_bin),
                "outputs": {"pool": str((out_dir / "pool.jsonl").relative_to(common.ROOT)), "holdout": str((out_dir / "holdout_hlpt.jsonl").relative_to(common.ROOT))},
                "source_hash": common.source_hash(), "elapsed_s": round(time.time() - t0, 1)}
    common.write_json(out_dir / "dedup_manifest.json", manifest)
    print(json.dumps({"pool": c["pool"], "dup_exact": c["dup_exact"], "dup_near": c["dup_near"], "holdout_hlpt": c["holdout_hlpt"],
                      "holdout_register": c["holdout_register"], "by_group": dict(by_group), "elapsed_s": manifest["elapsed_s"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
