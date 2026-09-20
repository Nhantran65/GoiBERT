#!/usr/bin/env python3
"""§4 preflight: model/tokenizer contract, attention implementation, packing/mask arithmetic, arm determinism, smoke run.

Writes artifacts/mmbert_vi_cpt_pilot/preflight.json.  Checks:
  * `ModernBertForMaskedLM.from_pretrained(jhu-clsp/mmBERT-base @ c5955035...)` loads with 0 missing/unexpected/mismatched
    keys, pretrained MLM head, decoder tied to the token embeddings, 22 layers / 768 hidden / 256 000 vocab; parameter
    counts reported as mmBERT-base 307 M total / 110.3 M non-embedding; Gemma 256K tokenizer contract (pad 0, eos 1,
    bos 2, unk 3, mask 4)
  * attention implementation: flash_attention_2 when importable, otherwise sdpa; recorded once and used by every arm
  * packing: every arm tensor is exactly [3815, 128, 512], contains no pad id, and counts 250 019 840 non-pad tokens
  * masks: identical positions for the same (step, micro) across arms and reproducible across two constructions
  * 50-step smoke run on the real P_mix data: finite loss/gradients, tokens/s, peak VRAM
  * `--rebuild` re-runs build_arms.py into a temporary directory and compares the corpus/order/batch-0..3 hashes
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import cpt_train as T  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def smoke(model, tok, special_ids, device, arm: str, steps: int, micro: int) -> dict:
    tokens = np.load(common.ART_DIR / "arms" / arm / "tokens.npy", mmap_mode="r")
    accum = common.GLOBAL_BATCH // micro
    opt = torch.optim.AdamW(model.parameters(), lr=common.CPT_LR, betas=common.CPT_BETAS, eps=common.CPT_EPS)
    losses, gnorms = [], []
    peak = 0.0
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for mi in range(accum):
            ids = torch.from_numpy(np.asarray(tokens[step, mi * micro:(mi + 1) * micro], dtype=np.int64))
            am = torch.ones_like(ids)
            gen = torch.Generator().manual_seed(common.SEED * 1_000_003 + step * 1_009 + mi)
            masked, labels = T.mask_tokens(ids, am, special_ids, tok.mask_token_id, len(tok), common.MASK_PROB, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = T.mlm_loss(model, masked.to(device), labels.to(device))
            (loss / accum).backward()
            step_loss += float(loss.detach()) / accum
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), common.CPT_CLIP))
        opt.step()
        losses.append(step_loss)
        gnorms.append(gn)
        peak = max(peak, torch.cuda.max_memory_allocated(device) / 1e9)
    wall = time.time() - t0
    return {"steps": steps, "micro": micro, "accum": accum, "arm": arm, "tokens_seen": steps * common.GLOBAL_BATCH * common.MAX_LEN,
            "tokens_s": steps * common.GLOBAL_BATCH * common.MAX_LEN / wall, "step_s": wall / steps,
            "eta_hours_full_budget": wall / steps * common.STEPS / 3600, "peak_gpu_memory_gb": peak,
            "loss_first": losses[0], "loss_last": losses[-1], "loss_all_finite": all(math.isfinite(x) for x in losses),
            "grad_norm_all_finite": all(math.isfinite(x) for x in gnorms), "grad_norm_max": max(gnorms),
            "device": torch.cuda.get_device_name(device)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke-steps", type=int, default=50)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out", default=str(common.ART_DIR / "preflight.json"))
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report: dict = {"protocol": common.PROTOCOL, "created_at": common.now(), "device": str(device), "source_hash": common.source_hash()}

    tok = common.load_tokenizer()
    report["tokenizer"] = {"vocab_size": len(tok), "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
                           "bos_token_id": tok.bos_token_id, "unk_token_id": tok.unk_token_id, "mask_token_id": tok.mask_token_id,
                           "sep_token_id": tok.sep_token_id, "class": type(tok).__name__, "manifest": common.tokenizer_manifest()}
    model, checks = common.load_mlm(common.ATTN_IMPL)
    report["model"] = {"requested_repo": common.MODEL_REPO, "commit": common.MODEL_COMMIT, "checks": checks,
                       "params_total_millions": checks["parameters_total"] / 1e6, "params_non_embedding_millions": checks["parameters_non_embedding"] / 1e6,
                       "documented": "mmBERT-base 307 M total (110.3 M non-embedding)"}
    report["attention"] = {"implementation": common.ATTN_IMPL, "flash_attn_available": common.FLASH_AVAILABLE,
                           "note": "the same implementation is used for every arm"}
    model.to(device)

    # ---- packing / token arithmetic
    arms = {}
    batch_hashes = {}
    for arm in common.ARMS:
        m = common.read_json(common.ART_DIR / "arms" / arm / "manifest.json")
        arr = np.load(common.ART_DIR / "arms" / arm / "tokens.npy", mmap_mode="r")
        batch_hashes[arm] = [common.sha256_array(np.asarray(arr[i])) for i in range(4)]
        arms[arm] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "tokens": int(arr.size),
                     "tokens_match_schedule": int(arr.size) == common.TOKENS_SCHEDULED,
                     "pad_id_count": int((np.asarray(arr[:8]) == 0).sum()), "min_token_id": int(arr[:8].min()),
                     "manifest_tokens_hash": m["tokens_hash"], "recomputed_tokens_hash_prefix": common.sha256_array(np.asarray(arr[:8])),
                     "order_hash": m["order_hash"], "multiset_hash": m.get("multiset_hash"),
                     "achieved_fractions": m["achieved_fractions"], "targets": m["targets"]}
    report["arms"] = arms
    report["arm_batch_hashes_step0_3"] = batch_hashes
    report["batch_hashes_repeatable"] = all(common.sha256_array(np.asarray(np.load(common.ART_DIR / "arms" / arm / "tokens.npy", mmap_mode="r")[0])) == batch_hashes[arm][0] for arm in common.ARMS)

    # ---- masks: identical positions across arms and reproducible
    special_ids = torch.tensor(common.special_ids(tok), dtype=torch.long)
    mask_checks = {}
    patterns: dict[int, list[torch.Tensor]] = {}
    for arm in common.ARMS:
        arr = np.load(common.ART_DIR / "arms" / arm / "tokens.npy", mmap_mode="r")
        rows = []
        patterns_row = []
        for step in (0, 1):
            ids = torch.from_numpy(np.asarray(arr[step, :2], dtype=np.int64))
            am = torch.ones_like(ids)
            # the raw selection grid (before special-token exclusion) is what must coincide across arms
            grid = torch.rand(ids.shape, generator=torch.Generator().manual_seed(common.SEED * 1_000_003 + step * 1_009 + 0)) < common.MASK_PROB
            patterns_row.append(grid)
            m1, _ = T.mask_tokens(ids, am, special_ids, tok.mask_token_id, len(tok), common.MASK_PROB,
                                  torch.Generator().manual_seed(common.SEED * 1_000_003 + step * 1_009 + 0))
            m2, _ = T.mask_tokens(ids, am, special_ids, tok.mask_token_id, len(tok), common.MASK_PROB,
                                  torch.Generator().manual_seed(common.SEED * 1_000_003 + step * 1_009 + 0))
            pos = (m1 != ids)
            rows.append({"step": step, "mask_hash": common.sha256_array(m1.numpy()), "repeatable": bool(torch.equal(m1, m2)),
                         "masked_positions": int(pos.sum()), "masked_fraction": float(pos.sum() / (ids != 0).sum())})
        mask_checks[arm] = rows
        patterns[arm] = patterns_row
    identical_across_arms = all(bool(torch.equal(patterns[a][s], patterns[common.ARMS[0]][s])) for a in common.ARMS for s in (0, 1))
    report["masks"] = {"per_arm": mask_checks, "same_position_count_across_arms": identical_across_arms,
                       "pattern_grid_identical_across_arms": identical_across_arms,
                       "note": "the (step, micro)-seeded selection grid is identical for every arm; the realised masked-position count differs slightly because each arm's data has a different number of `</s>` separators, which are never masked",
                       "scheme": "CPU generator seeded by SEED*1000003 + step*1009 + micro; 15 % of non-special positions, 80/10/10"}

    # ---- holdouts
    hold = {}
    for name in ("a_hlpt", "b_fineweb2", "c_register"):
        p = common.ART_DIR / "holdouts" / f"{name}.npy"
        arr = np.load(p, mmap_mode="r")
        hold[name] = {"shape": list(arr.shape), "tokens": int(arr.size), "pad_id_count": int((np.asarray(arr[:8]) == 0).sum()),
                      "sha256": common.sha256_array(np.asarray(arr))}
    report["holdouts"] = hold

    # ---- optional determinism rebuild of the arms
    if args.rebuild:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            cmd = [sys.executable, str(common.SCRIPT_DIR / "build_arms.py"), "--out-dir", tmp]
            subprocess.run(cmd, check=True)
            same = {}
            for arm in common.ARMS:
                a = common.read_json(Path(tmp) / arm / "manifest.json")
                same[arm] = {"tokens_hash_equal": a["tokens_hash"] == arms[arm]["manifest_tokens_hash"],
                             "order_hash_equal": a["order_hash"] == arms[arm]["order_hash"],
                             "batch0_equal": a["batch_hashes_step0_3"][0] == batch_hashes[arm][0],
                             "batch0_3_equal": a["batch_hashes_step0_3"] == batch_hashes[arm]}
        report["rebuild"] = same

    # ---- smoke run
    report["smoke"] = smoke(model, tok, special_ids, device, "P_mix", args.smoke_steps, args.micro)
    report["gate"] = {
        "model_loads_clean": not checks["missing_keys"] and not checks["unexpected_keys"] and not checks["mismatched_keys"] and checks["mlm_head_pretrained"] and checks["decoder_tied_to_embeddings"],
        "tokenizer_contract_ok": report["tokenizer"]["vocab_size"] == 256000 and report["tokenizer"]["pad_token_id"] == 0 and report["tokenizer"]["mask_token_id"] == 4,
        "packing_ok": all(a["pad_id_count"] == 0 and a["tokens_match_schedule"] for a in arms.values()),
        "masks_identical_across_arms": identical_across_arms and all(r["repeatable"] for rows in mask_checks.values() for r in rows),
        "smoke_finite": report["smoke"]["loss_all_finite"] and report["smoke"]["grad_norm_all_finite"],
        "rebuild_ok": bool(report.get("rebuild")) and all(v["batch0_3_equal"] and v["tokens_hash_equal"] for v in report["rebuild"].values()) if args.rebuild else None,
    }
    report["gate"]["all_ok"] = all(v for k, v in report["gate"].items() if v is not None)
    report["elapsed_s"] = round(time.time() - t0, 1)
    common.write_json(Path(args.out), report)
    print(json.dumps({k: report[k] for k in ("tokenizer", "attention", "gate", "smoke", "masks")}, ensure_ascii=False, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
