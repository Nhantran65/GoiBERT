#!/usr/bin/env python3
"""Data gate, stage 4 (§2/§3): materialise the three arm corpora with locked mixture and locked batch order.

Groups (source slices of the filtered, deduplicated, decontaminated pool):
  general  : HPLT bins 9-10, non-register documents           (P_web 95 %, P_mix/C_mix 65 %)
  register : HPLT bins 7-10, max(OP, re) >= 0.5 or ID >= 0.5  (P_mix/C_mix 20 %)
  bin8     : HPLT bin 8, non-register documents               (P_mix/C_mix 10 %)
  wiki     : Vietnamese Wikipedia (CC BY-SA 4.0)              (every arm 5 %)

Every sequence is exactly 512 mmBERT tokens (documents concatenated with the `</s>` id 1) and contains no padding, so one
sequence = 512 non-pad tokens.  Each arm is 3815 steps x 128 sequences = 250 019 840 tokens (+0.008 % vs the 250 M
budget, far below one batch).

  * P_web : mixture general+wiki, one shuffled order (seed 42).
  * P_mix : mixture general/register/bin8/wiki = 65/20/10/5, one shuffled order.
  * C_mix : **exactly the same packed sequences (same documents, same multiplicity) as P_mix**; only the batch
            assignment differs: stage 1 (0-40 % of tokens) general+wiki only, stage 2 (40-70 %) the remaining
            non-register slices, stage 3 (70-100 %, the LR decay stage) 60 % register / 40 % general.
            Documented deviation: because P_mix fixes the register share at 20 % overall, stage 2 can only hold
            ~6.7 % register (0.3*r2 + 0.3*0.6 = 0.20).  The plan's "50/50" stage-2 label is arithmetically
            incompatible with the same-document/same-count invariant, and the invariant wins.

Outputs (artifacts/mmbert_vi_cpt_pilot/arms/<arm>/): tokens.npy (uint32 [steps,128,512]), manifest.json.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402

SEP = 1                      # </s>
GROUP_ORDER = ("general", "register", "bin8", "wiki")


def stable_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def group_for(rec: dict) -> str:
    if rec.get("source") == "wiki":
        return "wiki"
    if str(rec["group"]).startswith("register"):
        return "register"
    if str(rec["bin"]) == "8":
        return "bin8"
    if str(rec["bin"]) == "7":
        return "lowbin"          # bin-7 general documents: read only to reach the register bucket; not used by any arm
    return "general"


class Pool:
    """Random access over the per-part token arrays written by tokenize_pool.py (and the Wikipedia slice)."""

    def __init__(self, dirs: list[Path]):
        self.parts: dict[str, np.ndarray] = {}
        self.offsets: dict[str, np.ndarray] = {}
        self.docs: list[dict] = []
        for d in dirs:
            for f in sorted(Path(d).glob("*.tokens.npy")):
                key = f.name.replace(".tokens.npy", "")
                if key in self.parts:
                    raise RuntimeError(f"duplicate token part {key}")
                self.parts[key] = np.load(f, mmap_mode="r")
                self.offsets[key] = np.load(f.with_name(key + ".offsets.npy"))
            for f in sorted(Path(d).glob("*.docs.jsonl")):
                key = f.name.replace(".docs.jsonl", "")
                if key not in self.parts:
                    continue
                with f.open(encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        rec = json.loads(line)
                        rec["part"] = key
                        rec["local_index"] = i
                        rec["n_tokens"] = int(self.offsets[key][i + 1] - self.offsets[key][i])
                        self.docs.append(rec)
        self.docs.sort(key=lambda r: (int(r.get("pool_index", -1)), r["part"], r["local_index"]))

    def tokens_of(self, rec: dict) -> np.ndarray:
        part = rec["part"]
        a = int(self.offsets[part][rec["local_index"]])
        b = int(self.offsets[part][rec["local_index"] + 1])
        return np.asarray(self.parts[part][a:b], dtype=np.uint32)


def target_counts(targets: dict[str, float], total_tokens: int, seq_len: int) -> dict[str, int]:
    total_seqs = total_tokens // seq_len
    counts = {g: n for g, n in ((g, int(np.floor(targets.get(g, 0.0) * total_seqs))) for g in GROUP_ORDER) if n > 0}
    remainder = total_seqs - sum(counts.values())
    frac = sorted(counts, key=lambda g: (-(targets.get(g, 0.0) * total_seqs - counts[g]), g))
    for i in range(remainder):
        counts[frac[i % len(frac)]] += 1
    return counts


def pack_group(docs: list[dict], pool: Pool, n_sequences: int, seq_len: int) -> tuple[np.ndarray, dict]:
    """Pack documents (with a `</s>` after each) into exactly `n_sequences` sequences of `seq_len` tokens."""
    out = np.zeros((n_sequences, seq_len), dtype=np.uint32)
    buf = np.empty(seq_len * 2, dtype=np.uint32)
    filled = 0                      # tokens currently in buf
    emitted = 0
    docs_used, doc_tokens = 0, 0
    for rec in docs:
        toks = pool.tokens_of(rec)
        if toks.size == 0:
            continue
        pos = 0
        while pos < toks.size and emitted < n_sequences:
            take = min(seq_len - filled, toks.size - pos)
            buf[filled:filled + take] = toks[pos:pos + take]
            filled += take
            pos += take
            if filled == seq_len:
                out[emitted] = buf[:seq_len]
                emitted += 1
                filled = 0
        if emitted >= n_sequences:
            doc_tokens += pos
            docs_used += 1
            break
        # document separator (may be split across sequences, which is fine: it always ends a document)
        if filled == seq_len:
            out[emitted] = buf[:seq_len]
            emitted += 1
            filled = 0
        buf[filled] = SEP
        filled += 1
        docs_used += 1
        doc_tokens += int(toks.size)
    return out, {"documents": docs_used, "doc_tokens": doc_tokens, "sequences_emitted": emitted, "tail_tokens_dropped": int(filled)}


def per_batch_allocation(counts: dict[str, int], batches: int) -> dict[str, np.ndarray]:
    out = {}
    for g, n in counts.items():
        x = np.rint(np.linspace(0.0, float(n), batches + 1)).astype(np.int64)
        x[-1] = n
        out[g] = np.diff(x)
    return out


def build_uniform_order(counts: dict[str, int], steps: int, batch: int, seed: int) -> tuple[list[int], dict[str, tuple[int, int]]]:
    ranges, cursor = {}, 0
    for g in GROUP_ORDER:
        if g in counts:
            ranges[g] = (cursor, cursor + counts[g])
            cursor += counts[g]
    alloc = per_batch_allocation(counts, steps)
    queues = {g: list(range(a, b)) for g, (a, b) in ranges.items()}
    rng = random.Random(seed)
    order: list[int] = []
    for b in range(steps):
        slots: list[int] = []
        for g in queues:
            n = int(alloc[g][b])
            slots.extend(queues[g][:n])
            del queues[g][:n]
        rng.shuffle(slots)
        order.extend(slots)
    # sequence ids in `order` are dense 0..total-1 in the order groups were laid out
    dense = {g: list(range(*r)) for g, r in ranges.items()}
    return order, dense


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", default=str(common.DATA_DIR / "tokens"))
    ap.add_argument("--wiki-dir", default=str(common.DATA_DIR / "wiki"))
    ap.add_argument("--out-dir", default=str(common.ART_DIR / "arms"))
    ap.add_argument("--seed", type=int, default=common.SEED)
    ap.add_argument("--rebuild-check", action="store_true", help="only rebuild hashes of existing arms to verify determinism")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len, batch, steps = common.MAX_LEN, common.GLOBAL_BATCH, common.STEPS
    total_tokens = steps * batch * seq_len

    pool = Pool([Path(args.pool_dir), Path(args.wiki_dir)])
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    avail = collections.Counter()
    for rec in pool.docs:
        if rec.get("split", "pool") != "pool":
            continue
        g = group_for(rec)
        by_group[g].append(rec)
        avail[g] += rec["n_tokens"]
    print(f"[arms] pool documents {len(pool.docs)} available tokens {dict(avail)}", flush=True)

    for g in GROUP_ORDER:
        if g in common.MIX_TARGET and avail[g] < int(common.MIX_TARGET[g] * total_tokens):
            print(f"[arms] WARNING group {g}: {avail[g]} tokens < target {int(common.MIX_TARGET[g]*total_tokens)}", flush=True)

    def sample(g: str, budget: int) -> list[dict]:
        docs = list(by_group[g])
        random.Random(args.seed + stable_seed(g)).shuffle(docs)
        return docs

    arm_data: dict[str, dict] = {}
    packed: dict[str, dict[str, np.ndarray]] = {}
    for arm, targets in (("P_web", common.WEB_TARGET), ("P_mix", common.MIX_TARGET)):
        counts = target_counts(targets, total_tokens, seq_len)
        packed[arm] = {}
        stats = {}
        for g, n in counts.items():
            docs = sample(g, (n + 1) * seq_len)
            seqs, st = pack_group(docs, pool, n, seq_len)
            packed[arm][g] = seqs
            stats[g] = {**st, "sequences": n, "tokens": n * seq_len, "target_fraction": targets.get(g, 0.0),
                        "available_tokens": int(avail[g])}
        arm_data[arm] = {"counts": counts, "stats": stats}
        print(f"[arms] {arm} packed {counts} ({time.time()-t0:.0f}s)", flush=True)

    # ---- P_mix multiset is the shared one; C_mix reuses the identical packed sequences
    shared_seqs: list[np.ndarray] = []
    shared_meta: list[dict] = []
    for g in GROUP_ORDER:
        if g not in packed["P_mix"]:
            continue
        for i, s in enumerate(packed["P_mix"][g]):
            shared_meta.append({"sequence_id": len(shared_seqs), "group": g, "group_index": i})
            shared_seqs.append(s)
    n_shared = len(shared_seqs)
    if n_shared != steps * batch:
        raise RuntimeError(f"P_mix has {n_shared} sequences != {steps * batch}")

    counts_mix = arm_data["P_mix"]["counts"]
    ranges, cursor = {}, 0
    for g in GROUP_ORDER:
        if g in counts_mix:
            ranges[g] = (cursor, cursor + counts_mix[g])
            cursor += counts_mix[g]

    # P_mix order: exact per-batch allocation of each group, shuffled inside each batch
    alloc = per_batch_allocation(counts_mix, steps)
    queues = {g: list(range(a, b)) for g, (a, b) in ranges.items()}
    rng = random.Random(args.seed)
    order_p_mix: list[int] = []
    for b in range(steps):
        slots: list[int] = []
        for g in queues:
            n = int(alloc[g][b])
            slots.extend(queues[g][:n])
            del queues[g][:n]
        rng.shuffle(slots)
        order_p_mix.extend(slots)
    assert all(not v for v in queues.values())

    # C_mix order: stage 1 general+wiki, stage 2 remainder, stage 3 60 % register / 40 % general
    s1 = int(round(0.40 * steps))
    s3 = int(round(0.30 * steps))
    s2 = steps - s1 - s3
    reg_total = counts_mix.get("register", 0)
    gen_total, wiki_total, bin8_total = counts_mix.get("general", 0), counts_mix.get("wiki", 0), counts_mix.get("bin8", 0)
    stage3_reg = min(int(round(0.60 * s3 * batch)), reg_total)
    stage3_gen = min(s3 * batch - stage3_reg, gen_total)
    stage1_slots = s1 * batch
    stage1_wiki = min(wiki_total, int(round(stage1_slots * wiki_total / max(1, wiki_total + gen_total))))
    stage1_gen = stage1_slots - stage1_wiki
    stage2_comp = {"register": reg_total - stage3_reg, "general": gen_total - stage1_gen - stage3_gen, "bin8": bin8_total,
                   "wiki": wiki_total - stage1_wiki}
    stage2_comp = {g: v for g, v in stage2_comp.items() if v > 0}
    if sum(stage2_comp.values()) != s2 * batch:
        raise RuntimeError(f"C_mix stage arithmetic mismatch: {stage2_comp} vs {s2 * batch}")
    c_queues = {g: list(range(a, b)) for g, (a, b) in ranges.items()}
    rng2 = random.Random(args.seed + 1)
    order_c_mix: list[int] = []
    stage_plan = {}
    for name, comp, n_batches in (("stage1", {"general": stage1_gen, "wiki": stage1_wiki}, s1), ("stage2", stage2_comp, s2), ("stage3", {"register": stage3_reg, "general": stage3_gen}, s3)):
        comp = {g: int(v) for g, v in comp.items() if v > 0}
        alloc_stage = per_batch_allocation(comp, n_batches)
        for b in range(n_batches):
            slots: list[int] = []
            for g in alloc_stage:
                n = int(alloc_stage[g][b])
                slots.extend(c_queues[g][:n])
                del c_queues[g][:n]
            rng2.shuffle(slots)
            order_c_mix.extend(slots)
        stage_plan[name] = {"batches": n_batches, "composition_sequences": comp}
    if len(order_c_mix) != n_shared or any(c_queues[g] for g in c_queues):
        raise RuntimeError("C_mix did not consume exactly the P_mix multiset")

    multiset_hash = common.sha256_lines(common.sha256_array(s) for s in shared_seqs)

    # ---- materialise
    manifests: dict[str, dict] = {}
    for arm in ("P_web", "P_mix", "C_mix"):
        if arm == "P_web":
            counts = arm_data["P_web"]["counts"]
            seqs_by_group = packed["P_web"]
            order, _ = build_uniform_order(counts, steps, batch, args.seed)
            seqs_by_id: list[np.ndarray] = [None] * (steps * batch)      # type: ignore[list-item]
            cursor = 0
            for g in GROUP_ORDER:
                if g in counts:
                    seqs_by_id[cursor:cursor + counts[g]] = list(seqs_by_group[g])
                    cursor += counts[g]
        else:
            seqs_by_id = shared_seqs
            order = order_p_mix if arm == "P_mix" else order_c_mix
            counts = counts_mix
        arr = np.zeros((steps, batch, seq_len), dtype=np.uint32)
        for slot, sid in enumerate(order):
            b, k = divmod(slot, batch)
            arr[b, k] = seqs_by_id[sid]
        (out_dir / arm).mkdir(parents=True, exist_ok=True)
        np.save(out_dir / arm / "tokens.npy", arr)
        stats = arm_data["P_mix" if arm != "P_web" else "P_web"]["stats"]
        manifest = {"arm": arm, "created_at": common.now(), "steps": steps, "global_batch": batch, "seq_len": seq_len,
                    "tokens_non_pad": int(arr.size), "budget_error_vs_250M": int(arr.size) - common.TOKEN_BUDGET,
                    "budget_error_batches": (int(arr.size) - common.TOKEN_BUDGET) / (batch * seq_len),
                    "targets": dict(common.WEB_TARGET if arm == "P_web" else common.MIX_TARGET),
                    "sequence_counts_by_group": counts, "group_stats": stats,
                    "achieved_fractions": {g: counts[g] * seq_len / int(arr.size) for g in counts},
                    "group_config": {"general": "HPLT bins 9-10 non-register", "register": "HPLT bins 7-10 max(OP,re)>=0.5 or ID>=0.5", "bin8": "HPLT bin 8 non-register", "wiki": "wikimedia/wikipedia 20231101.vi (CC BY-SA 4.0)"},
                    "order": "single shuffle seed 42" if arm != "C_mix" else stage_plan,
                    "tokens_hash": common.sha256_array(arr), "order_hash": common.sha256_lines(json.dumps(order[i:i + batch]) for i in range(0, len(order), batch)),
                    "batch_hashes_step0_3": [common.sha256_array(arr[i]) for i in range(4)],
                    "multiset_hash": multiset_hash if arm != "P_web" else None,
                    "source_hash": common.source_hash()}
        common.write_json(out_dir / arm / "manifest.json", manifest)
        manifests[arm] = manifest
        print(f"[arms] wrote {arm} fractions { {g: round(v,4) for g, v in manifest['achieved_fractions'].items()} } hash {manifest['tokens_hash'][:12]} ({time.time()-t0:.0f}s)", flush=True)
        del arr

    if manifests["P_mix"]["multiset_hash"] != manifests["C_mix"]["multiset_hash"]:
        raise RuntimeError("P_mix and C_mix multisets differ")
    common.write_json(out_dir / "arms_manifest.json", {"created_at": common.now(), "available_tokens": {g: int(avail[g]) for g in avail},
                                                       "arms": manifests, "budget": {"token_budget": common.TOKEN_BUDGET, "steps": steps, "scheduled_tokens": total_tokens},
                                                       "deviation_note": "C_mix stage 2 holds the remainder (~6.7 % register): the register share is fixed at 20 % overall by P_mix",
                                                       "source_hash": common.source_hash(), "elapsed_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
