#!/usr/bin/env python3
"""v4 §3 preflight: model/tokenizer contract, packing, cross-arm mask alignment, rebuild determinism, 50-step smoke.

Writes artifacts/mmbert_vi_cpt_pilot/v4/preflight.json.
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
import common_v4 as C4  # noqa: E402
import cpt_train_v4 as T  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke-steps", type=int, default=50)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--rebuild", action="store_true", default=True)
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report: dict = {"protocol": "mmbert-vi-cpt-pilot-v4", "created_at": C4.common.now(), "device": str(device), "source_hash": C4.common.source_hash()}
    tok = C4.common.load_tokenizer()
    report["tokenizer"] = {"vocab_size": len(tok), "pad": tok.pad_token_id, "eos": tok.eos_token_id, "bos": tok.bos_token_id,
                           "unk": tok.unk_token_id, "mask": tok.mask_token_id}
    model, checks = C4.common.load_mlm("sdpa")
    report["model"] = {"repo": C4.common.MODEL_REPO, "commit": C4.common.MODEL_COMMIT, "checks": checks,
                       "params_total_millions": checks["parameters_total"] / 1e6,
                       "params_non_embedding_millions": checks["parameters_non_embedding"] / 1e6,
                       "documented": "mmBERT-base 307 M total (110 M non-embedding)"}
    report["attention"] = {"implementation": "sdpa", "flash_attn_available": C4.common.FLASH_AVAILABLE}
    model.to(device)
    arms = {}
    for arm in C4.ARMS:
        d = C4.ART_V4 / "arms" / T.arm_dir_name(arm)
        m = C4.read_json(d / "manifest.json")
        arr = np.load(d / "tokens.npy", mmap_mode="r")
        arms[arm] = {"dir": d.name, "shape": list(arr.shape), "tokens": int(arr.size), "pad_id_count_head": int((np.asarray(arr[:8]) == 0).sum()),
                     "tokens_hash": m["tokens_hash"], "order_hash": m["order_hash"], "multiset_hash": m.get("multiset_hash"),
                     "informal_token_share": m["informal_token_share"], "research_only": m["research_only"],
                     "license": m["license"], "batch_hashes_step0_3": m["batch_hashes_step0_3"]}
    report["arms"] = arms
    # masks: the (step, micro)-seeded selection grid must be identical for every arm
    special = torch.tensor(C4.common.special_ids(tok), dtype=torch.long)
    grids = {}
    for arm in C4.ARMS:
        arr = np.load(C4.ART_V4 / "arms" / T.arm_dir_name(arm) / "tokens.npy", mmap_mode="r")
        rows = []
        for step in (0, 1):
            ids = torch.from_numpy(np.asarray(arr[step, :2], dtype=np.int64))
            g1 = torch.rand(ids.shape, generator=torch.Generator().manual_seed(C4.SEED * 1_000_003 + step * 1_009 + 0)) < 0.15
            m1, _ = T.mask_tokens(ids, torch.ones_like(ids), special, tok.mask_token_id, len(tok), 0.15,
                                  torch.Generator().manual_seed(C4.SEED * 1_000_003 + step * 1_009 + 0))
            m2, _ = T.mask_tokens(ids, torch.ones_like(ids), special, tok.mask_token_id, len(tok), 0.15,
                                  torch.Generator().manual_seed(C4.SEED * 1_000_003 + step * 1_009 + 0))
            rows.append({"step": step, "grid_hash": C4.common.sha256_array(g1.numpy()), "repeatable": bool(torch.equal(m1, m2)),
                         "masked_positions": int((m1 != ids).sum())})
        grids[arm] = rows
    identical = all(grids[a][s]["grid_hash"] == grids[C4.ARMS[0]][s]["grid_hash"] for a in C4.ARMS for s in (0, 1))
    report["masks"] = {"per_arm": grids, "grid_identical_across_arms": identical}

    def smoke() -> dict:
        arr = np.load(C4.ART_V4 / "arms" / T.arm_dir_name("P_inf") / "tokens.npy", mmap_mode="r")
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6)
        losses, gnorms, peak = [], [], 0.0
        t = time.time()
        for step in range(args.smoke_steps):
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for mi in range(128 // args.micro):
                ids = torch.from_numpy(np.asarray(arr[step, mi * args.micro:(mi + 1) * args.micro], dtype=np.int64))
                gen = torch.Generator().manual_seed(C4.SEED * 1_000_003 + step * 1_009 + mi)
                masked, labels = T.mask_tokens(ids, torch.ones_like(ids), special, tok.mask_token_id, len(tok), 0.15, gen)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = T.mlm_loss(model, masked.to(device), labels.to(device))
                (loss / (128 // args.micro)).backward()
                step_loss += float(loss.detach()) / (128 // args.micro)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            losses.append(step_loss); gnorms.append(gn)
            peak = max(peak, torch.cuda.max_memory_allocated(device) / 1e9)
        wall = time.time() - t
        return {"steps": args.smoke_steps, "micro": args.micro, "tokens_seen": args.smoke_steps * 65536,
                "tokens_s": args.smoke_steps * 65536 / wall, "step_s": wall / args.smoke_steps,
                "eta_hours_full_budget": wall / args.smoke_steps * 3815 / 3600, "peak_gpu_memory_gb": peak,
                "loss_first": losses[0], "loss_last": losses[-1], "loss_all_finite": all(math.isfinite(x) for x in losses),
                "grad_norm_all_finite": all(math.isfinite(x) for x in gnorms), "device": torch.cuda.get_device_name(device)}

    report["smoke"] = smoke()
    if args.rebuild:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            subprocess.run([sys.executable, str(C4.V4_DIR / "build_arms_v4.py"), "--out-dir", tmp], check=True, capture_output=True)
            same = {}
            for arm in C4.ARMS:
                m = C4.read_json(Path(tmp) / T.arm_dir_name(arm) / "manifest.json")
                same[arm] = {"tokens_hash_equal": m["tokens_hash"] == arms[arm]["tokens_hash"],
                             "order_hash_equal": m["order_hash"] == arms[arm]["order_hash"],
                             "batch0_3_equal": m["batch_hashes_step0_3"] == arms[arm]["batch_hashes_step0_3"]}
        report["rebuild"] = same
    report["gate"] = {"model_loads_clean": not checks["missing_keys"] and not checks["unexpected_keys"] and checks["mlm_head_pretrained"] and checks["decoder_tied_to_embeddings"],
                      "packing_ok": all(a["pad_id_count_head"] == 0 and a["tokens"] == 250_019_840 for a in arms.values()),
                      "masks_identical_across_arms": identical,
                      "multiset_p_inf_c_inf_equal": arms["P_inf"]["multiset_hash"] == arms["C_inf"]["multiset_hash"],
                      "smoke_finite": report["smoke"]["loss_all_finite"] and report["smoke"]["grad_norm_all_finite"],
                      "rebuild_ok": all(all(v.values()) for v in report.get("rebuild", {}).values()) if args.rebuild else None}
    report["gate"]["all_ok"] = all(v for v in report["gate"].values() if v is not None)
    report["elapsed_s"] = round(time.time() - t0, 1)
    C4.write_json(C4.ART_V4 / "preflight.json", report)
    print(json.dumps({k: report[k] for k in ("model", "attention", "gate", "smoke")}, ensure_ascii=False, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
