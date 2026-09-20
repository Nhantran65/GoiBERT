#!/usr/bin/env python3
"""v4 §3: materialise the three arms (250 M non-pad tokens each, 3815 x 128 x 512).

  P_web : general v3 pool (bins 7-9, v3 filter, cap host 2 %) 95 % + Wikipedia v2 5 %             (no NC data)
  P_inf : ViSoBERT comments (_NC_research_only) 30 % + general 65 % + Wikipedia 5 %, uniform order
  C_inf : **exactly the same multiset of packed sequences as P_inf**, only the batch order differs:
          stage 1 (0-40 % of tokens) 10 % informal, stage 2 (40-70 %) 20 %, stage 3 (70-100 %, the LR decay stage) 66.7 %.

Manifests record the mixture, the locked order/multiset hashes and the measured informal-token share; P_inf/C_inf carry
the NC licence note and the `_NC_research_only` suffix on their output directories.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v4 as C4  # noqa: E402
import audit_v3 as A3  # noqa: E402  (v3 general-pool builder, reused read-only)

import numpy as np  # noqa: E402

SEQ = 512
BATCH = 128
STEPS = 3815
TOTAL_SEQS = STEPS * BATCH


def pack_docs(docs: list[dict], n_seq: int, part_tokens: dict[str, np.ndarray], part_offsets: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((n_seq, SEQ), dtype=np.uint32)
    buf = np.empty(SEQ * 2, dtype=np.uint32)
    filled = emitted = 0
    for r in docs:
        part = r["part"]
        a, b = int(part_offsets[part][r["local_index"]]), int(part_offsets[part][r["local_index"] + 1])
        toks = np.asarray(part_tokens[part][a:b], dtype=np.uint32)
        if toks.size == 0:
            continue
        pos = 0
        while pos < toks.size and emitted < n_seq:
            take = min(SEQ - filled, toks.size - pos)
            buf[filled:filled + take] = toks[pos:pos + take]
            filled += take
            pos += take
            if filled == SEQ:
                out[emitted] = buf[:SEQ]
                emitted += 1
                filled = 0
        if emitted >= n_seq:
            break
        if filled == SEQ:
            out[emitted] = buf[:SEQ]
            emitted += 1
            filled = 0
        buf[filled] = C4.common.sep_token_id if hasattr(C4.common, "sep_token_id") else 1
        filled += 1
    return out[:emitted]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(C4.ART_V4 / "arms"))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(C4.SEED)

    # ---- sources
    bucket = [json.loads(l) for l in (C4.C3.DATA_V3 / "informal_bucket" / "informal_docs.jsonl").open(encoding="utf-8")]
    general = A3.build_general_v3({r["pool_index"] for r in bucket})["docs"]
    rng.shuffle(general)
    wiki_rows = [json.loads(l) for l in (C4.C2.ART_V2 / "wiki_filtered.jsonl").open(encoding="utf-8")]
    wiki_docs = [{"part": "wiki", "local_index": r["i"], "n_tokens": r["n_tokens"]} for r in wiki_rows]
    rng.shuffle(wiki_docs)
    visobert = np.load(C4.DATA_V4 / "visobert" / "visobert_train.npy", mmap_mode="r")
    counts = {"P_web": {"general": 463904, "wiki": 24416},
              "P_inf": {"visobert": 146496, "general": 317408, "wiki": 24416}}
    # general and wiki packed streams (shared between arms where the share is equal)
    part_names = sorted({r["part"] for r in general})
    part_tokens = {p: np.load(C4.C2.DATA_V1 / "tokens" / f"{p}.tokens.npy", mmap_mode="r") for p in part_names}
    part_offsets = {p: np.load(C4.C2.DATA_V1 / "tokens" / f"{p}.offsets.npy") for p in part_names}
    general_packed = pack_docs(general, max(counts["P_web"]["general"], counts["P_inf"]["general"]), part_tokens, part_offsets)
    w_tokens = np.load(C4.C2.DATA_V1 / "wiki" / "wiki.tokens.npy", mmap_mode="r")
    w_offsets = np.load(C4.C2.DATA_V1 / "wiki" / "wiki.offsets.npy")
    wiki_packed = pack_docs(wiki_docs, counts["P_web"]["wiki"], {"wiki": w_tokens}, {"wiki": w_offsets})
    visobert_packed = np.asarray(visobert[: counts["P_inf"]["visobert"]], dtype=np.uint32)
    sources = {"general": general_packed, "wiki": wiki_packed, "visobert": visobert_packed}
    print(f"[arms] packed general {general_packed.shape}, wiki {wiki_packed.shape}, visobert {visobert_packed.shape} ({time.time()-t0:.0f}s)", flush=True)

    def allocate(counts_row: dict[str, int], batches: int) -> dict[str, np.ndarray]:
        out = {}
        for g, n in counts_row.items():
            x = np.rint(np.linspace(0.0, float(n), batches + 1)).astype(np.int64)
            x[-1] = n
            out[g] = np.diff(x)
        return out

    def assemble(order_plan: list[dict]) -> list[int]:
        queues = {g: list(range(arr.shape[0])) for g, arr in sources.items()}
        order: list[int] = []
        base = {"general": 0, "wiki": general_packed.shape[0], "visobert": general_packed.shape[0] + wiki_packed.shape[0]}
        rng2 = random.Random(C4.SEED)
        for row in order_plan:
            slots = []
            for g, n in row.items():
                for _ in range(int(n)):
                    local = queues[g].pop(0)
                    slots.append(local if g == "general" else (base[g] + local if g != "general" else local))
            rng2.shuffle(slots)
            order.extend(slots)
        return order

    arrays: dict[str, np.ndarray] = {}
    orders: dict[str, list[int]] = {}
    for arm in ("P_web", "P_inf"):
        c = counts[arm]
        alloc = allocate(c, STEPS)
        order = assemble([{g: int(alloc[g][b]) for g in alloc} for b in range(STEPS)])
        arr = np.zeros((STEPS, BATCH, SEQ), dtype=np.uint32)
        for slot, sid in enumerate(order):
            b, k = divmod(slot, BATCH)
            if sid < general_packed.shape[0]:
                arr[b, k] = general_packed[sid]
            elif sid < general_packed.shape[0] + wiki_packed.shape[0]:
                arr[b, k] = wiki_packed[sid - general_packed.shape[0]]
            else:
                arr[b, k] = visobert_packed[sid - general_packed.shape[0] - wiki_packed.shape[0]]
        arrays[arm] = arr
        orders[arm] = order
    order_p_inf = orders["P_inf"]
    # C_inf: identical multiset, staged order
    s1, s2 = int(round(0.40 * STEPS)), int(round(0.30 * STEPS))
    s3 = STEPS - s1 - s2
    stage_defs = [("stage1", s1, 0.10), ("stage2", s2, 0.20), ("stage3", s3, 0.667)]
    inf_total = counts["P_inf"]["visobert"]
    stage_inf = [int(round(frac * n * BATCH)) for _, n, frac in stage_defs]
    stage_inf[-1] = inf_total - sum(stage_inf[:-1])
    gen_total = counts["P_inf"]["general"]
    wiki_total = counts["P_inf"]["wiki"]
    noninf_total = gen_total + wiki_total
    stage_noninf = [n * BATCH - i for (_, n, _), i in zip(stage_defs, stage_inf)]
    if sum(stage_noninf) != noninf_total:
        raise RuntimeError(f"stage arithmetic mismatch {stage_noninf} vs {noninf_total}")
    queues = {g: list(range(counts["P_inf"][g])) for g in sources}      # exactly the P_inf multiset
    base = {"general": 0, "wiki": general_packed.shape[0], "visobert": general_packed.shape[0] + wiki_packed.shape[0]}
    rng3 = random.Random(C4.SEED + 1)
    order_c: list[int] = []
    stage_plan = {}
    cum = np.cumsum([0] + stage_noninf) / noninf_total * wiki_total
    stage_wiki = [int(round(cum[i + 1])) - int(round(cum[i])) for i in range(len(stage_noninf))]
    stage_wiki[-1] += wiki_total - sum(stage_wiki)          # exact split of the wiki share
    order_c_plan = list(zip(stage_defs, stage_inf, stage_noninf, stage_wiki))
    for (name, n_batches, frac), n_inf, n_noninf, n_wiki in order_c_plan:
        n_gen = n_noninf - n_wiki
        alloc = allocate({"visobert": n_inf, "general": n_gen, "wiki": n_wiki}, n_batches)
        for b in range(n_batches):
            slots = []
            for g, n in ((g, int(alloc[g][b])) for g in alloc):
                for _ in range(n):
                    local = queues[g].pop(0)
                    slots.append(base[g] + local)
            rng3.shuffle(slots)
            order_c.extend(slots)
        stage_plan[name] = {"batches": n_batches, "target_informal_fraction": frac,
                            "sequences": {"visobert": n_inf, "general": n_gen, "wiki": n_wiki}}
    if any(queues[g] for g in queues):
        raise RuntimeError("C_inf did not consume exactly the P_inf multiset")
    c_arr = np.zeros((STEPS, BATCH, SEQ), dtype=np.uint32)
    for slot, sid in enumerate(order_c):
        b, k = divmod(slot, BATCH)
        if sid < general_packed.shape[0]:
            c_arr[b, k] = general_packed[sid]
        elif sid < general_packed.shape[0] + wiki_packed.shape[0]:
            c_arr[b, k] = wiki_packed[sid - general_packed.shape[0]]
        else:
            c_arr[b, k] = visobert_packed[sid - general_packed.shape[0] - wiki_packed.shape[0]]
    arrays["C_inf"] = c_arr
    orders["C_inf"] = order_c
    counts["C_inf"] = dict(counts["P_inf"])

    manifests = {}
    for arm in C4.ARMS:
        arr = arrays[arm]
        dir_name = arm + (C4.NC_SUFFIX if arm in C4.ARMS_NC else "")
        d = out_dir / dir_name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "tokens.npy", arr)
        c = counts[arm]
        order = orders[arm]
        inf_tokens = c.get("visobert", 0) * SEQ
        try:
            dir_label = str(d.relative_to(C4.ROOT))
        except ValueError:
            dir_label = str(d)
        manifest = {"arm": arm, "dir": dir_label, "created_at": C4.common.now(),
                    "steps": STEPS, "global_batch": BATCH, "seq_len": SEQ, "tokens_non_pad": int(arr.size),
                    "budget_error_vs_250M": int(arr.size) - 250_000_000,
                    "targets": C4.ARMS_WEIGHTS["P_inf" if arm == "C_inf" else arm],
                    "sequence_counts_by_group": c, "informal_tokens": inf_tokens,
                    "informal_token_share": inf_tokens / int(arr.size),
                    "sources": {"general": "v3 general pool (bins 7-9, v3 filter, cap host 2%)", "wiki": "v2 wiki_filtered (36 551 articles)",
                                "visobert": f"{C4.VISOBERT_REPO}@{C4.VISOBERT_REVISION} (research-only, CC BY-NC 4.0)"},
                    "order": "uniform shuffle seed 42" if arm != "C_inf" else stage_plan,
                    "tokens_hash": C4.common.sha256_array(arr), "order_hash": C4.common.sha256_lines(json.dumps(order[i:i + BATCH]) for i in range(0, len(order), BATCH)),
                    "batch_hashes_step0_3": [C4.common.sha256_array(arr[i]) for i in range(4)],
                    "multiset_hash": None if arm == "P_web" else C4.common.sha256_lines([str(x) for x in sorted(order_p_inf)]),
                    "license": C4.LICENSE_NOTE if arm in C4.ARMS_NC else "no NC data: general v3 pool + Wikipedia v2 (CC BY-SA 4.0)",
                    "research_only": arm in C4.ARMS_NC, "source_hash": C4.common.source_hash()}
        C4.write_json(d / "manifest.json", manifest)
        manifests[arm] = manifest
        print(f"[arms] {arm} -> {d.name}: {c} informal_share={manifest['informal_token_share']:.3f} hash {manifest['tokens_hash'][:12]} ({time.time()-t0:.0f}s)", flush=True)
    if manifests["P_inf"]["multiset_hash"] != manifests["C_inf"]["multiset_hash"]:
        raise RuntimeError("P_inf/C_inf multisets differ")
    C4.write_json(out_dir / "arms_manifest.json", {"created_at": C4.common.now(), "arms": manifests,
                                                   "license_note": C4.LICENSE_NOTE, "elapsed_s": round(time.time() - t0, 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
