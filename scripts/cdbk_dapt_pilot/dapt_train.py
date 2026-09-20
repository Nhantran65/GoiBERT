#!/usr/bin/env python3
"""D1 DAPT: plain continued MLM of CafeBERT on the ViSoBERT social corpus (adapter over scripts/pilot_b/cpt_train.py).

Recipe (plan §3): init uitnlp/CafeBERT@af76fcf2 (XLMRobertaForMaskedLM, pretrained MLM head, tied embeddings), tokenizer
XLM-R-large@c23d21b0, standard MLM 15 % (80/10/10, no special/pad), seq 512, global batch 256, bf16, lr 5e-5,
AdamW(.9,.98) eps 1e-6 wd .01 (bias/norm no decay), warmup 6 % + linear decay, clip 1, seed 42, 300 M processed tokens
(2289 global batches), checkpoint = end of budget.  No diacritic dropout / teencode augmentation.

Launch (DDP): CUDA_VISIBLE_DEVICES=2,3 .venv-e0-cuda/bin/python -m torch.distributed.run --nproc_per_node 2 \
    --master_port 29631 scripts/cdbk_dapt_pilot/dapt_train.py --micro 16
Outputs: artifacts/cdbk_dapt_pilot/dapt/{hf/, log.jsonl, summary.json, status.json}
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from transformers import AutoTokenizer, XLMRobertaForMaskedLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "cdbk_pilot"))
sys.path.insert(0, str(ROOT / "scripts" / "pilot_b"))
import common  # noqa: E402  (cdbk_pilot)
from common import TEACHER_HF_COMMIT, TEACHER_HF_REPO, TEACHER_TOKENIZER_COMMIT, TEACHER_TOKENIZER_REPO  # noqa: E402
import cpt_train  # noqa: E402  (pilot_b helpers: mask_batch, SparseMLM, ShardSampler, param_groups, lr_lambda, evaluate)

DATA_DIR = ROOT / "data" / "cdbk_dapt_pilot"
OUT_DEFAULT = ROOT / "artifacts" / "cdbk_dapt_pilot" / "dapt"
VAL_EVERY = 500
LOG_EVERY = 20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=float, default=300e6)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--mask", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bench", type=int, default=0, help="profile N steps and exit (no outputs)")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    rank, world, local_rank = cpt_train.setup_dist()
    device = torch.device("cuda", local_rank)
    cpt_train.seed_all(args.seed + rank)
    out_dir = Path(args.out)
    if rank == 0 and not args.bench:
        if (out_dir / "summary.json").exists():
            raise FileExistsError(f"Refusing to overwrite a completed DAPT run: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        common.write_json(out_dir / "status.json", {"status": "running", "pid": os.getpid(), "world": world, "started_at": common.now()})

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    train_mm = np.load(DATA_DIR / "visobert_xlmr_512_train.npy", mmap_mode="r")
    hold_mm = np.load(DATA_DIR / "visobert_xlmr_512_holdout.npy", mmap_mode="r")
    assert train_mm.shape[1] == args.seq == hold_mm.shape[1]
    n_train = train_mm.shape[0]
    hold_idx = np.arange(hold_mm.shape[0])

    tok = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER_REPO, revision=TEACHER_TOKENIZER_COMMIT, local_files_only=True, use_fast=True)
    special_ids = torch.tensor(sorted(set(tok.all_special_ids)), device=device, dtype=torch.long)
    vocab_size = len(tok)
    assert tok.mask_token_id == cpt_train.MASK_ID, (tok.mask_token_id, cpt_train.MASK_ID)

    model, loading_info = XLMRobertaForMaskedLM.from_pretrained(
        TEACHER_HF_REPO, revision=TEACHER_HF_COMMIT, local_files_only=True, attn_implementation="sdpa", output_loading_info=True)
    emb_w = model.roberta.embeddings.word_embeddings.weight
    dec_w = model.lm_head.decoder.weight
    source_checks = {
        "backbone": TEACHER_HF_REPO, "hf_commit": TEACHER_HF_COMMIT, "architecture": model.config.architectures,
        "missing_keys": list(loading_info.get("missing_keys", [])), "unexpected_keys": list(loading_info.get("unexpected_keys", [])),
        "mismatched_keys": [str(k) for k in loading_info.get("mismatched_keys", [])],
        "mlm_head_pretrained": not any(k.startswith("lm_head") for k in loading_info.get("missing_keys", [])),
        "decoder_tied_to_embeddings": bool(dec_w.data_ptr() == emb_w.data_ptr()), "vocab_size": int(emb_w.shape[0]), "hidden": int(emb_w.shape[1]),
        "sparse_mlm_uses_embedding_matrix_and_lm_head_bias": True,
        "tokenizer": {"repo": TEACHER_TOKENIZER_REPO, "revision": TEACHER_TOKENIZER_COMMIT, "vocab": vocab_size, "mask_id": tok.mask_token_id, "special_ids": special_ids.tolist()},
    }
    if source_checks["missing_keys"] or source_checks["unexpected_keys"] or source_checks["mismatched_keys"] or not source_checks["decoder_tied_to_embeddings"]:
        raise RuntimeError(f"CafeBERT MLM source does not load cleanly: {source_checks}")
    if rank == 0:
        print(f"[dapt] source ok: {json.dumps({k: v for k, v in source_checks.items() if k != 'tokenizer'})}", flush=True)

    wrapper = cpt_train.SparseMLM(model).to(device)
    ddp_model = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else wrapper
    global_batch = args.global_batch
    accum = global_batch // (args.micro * world)
    assert accum >= 1 and accum * args.micro * world == global_batch, f"global batch {global_batch} != micro {args.micro} x world {world} x accum"
    tokens_target = int(args.tokens)
    steps_target = math.ceil(tokens_target / (global_batch * args.seq))
    warmup_steps = max(1, round(args.warmup * steps_target))
    n_steps_run = args.bench if args.bench else steps_target

    opt = torch.optim.AdamW(cpt_train.param_groups(wrapper, args.wd, args.lr), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cpt_train.lr_lambda(s, warmup_steps, steps_target))
    sampler = cpt_train.ShardSampler(n_train, world, rank, args.micro, args.seed)

    config = {
        "protocol": "CDBK-DAPT-pilot-D1-v1", "stage": "dapt", "corpus": manifest["corpus"], "corpus_tokens": manifest["tokens"], "seq": args.seq,
        "micro": args.micro, "world": world, "accum": accum, "global_batch": global_batch, "tokens_target": tokens_target, "steps_target": steps_target,
        "tokens_processed_planned": steps_target * global_batch * args.seq, "lr": args.lr, "warmup_frac": args.warmup, "warmup_steps": warmup_steps, "wd": args.wd,
        "betas": [0.9, 0.98], "eps": 1e-6, "clip": 1.0, "mask_prob": args.mask, "mask_scheme": "80/10/10, no special/pad", "precision": "bf16 autocast", "attn": "sdpa",
        "seed": args.seed, "seed_per_rank": "seed + rank", "checkpoint_policy": "end of token budget (no downstream selection)",
        "source_checks": source_checks, "source_hash": common.source_hash(), "adapter_source_hash": common.sha256_file(Path(__file__)),
        "pilot_b_cpt_train_sha256": common.sha256_file(ROOT / "scripts" / "pilot_b" / "cpt_train.py"), "runtime": common.runtime(device), "argv": sys.argv, "started_at": common.now(),
    }
    if rank == 0 and not args.bench:
        common.write_json(out_dir / "resolved_config.json", config)
        print(f"[dapt] world={world} micro={args.micro} accum={accum} global={global_batch} steps={n_steps_run}/{steps_target} warmup={warmup_steps} n_train_seq={n_train} holdout_seq={len(hold_idx)}", flush=True)

    val_losses: list[dict] = []
    if rank == 0 and not args.bench:
        v0 = cpt_train.evaluate(wrapper, hold_mm, hold_idx, args.micro, args.mask, vocab_size, special_ids, device)
        val_losses.append({"step": 0, "tokens_seen": 0, "holdout_loss": v0})
        print(f"[dapt] step 0 holdout_loss {v0:.4f}", flush=True)
    if world > 1:
        dist.barrier()

    log_f = (out_dir / "log.jsonl").open("w") if (rank == 0 and not args.bench) else None
    torch.cuda.reset_peak_memory_stats(device)
    tokens_seen = 0
    masked_local = 0
    loss_hist: list[float] = []
    t0 = time.time()
    for step in range(1, n_steps_run + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for mi in range(accum):
            idx = sampler.next_batch()
            input_ids = torch.from_numpy(train_mm[idx].astype(np.int64)).to(device, non_blocking=True)
            inp, labels = cpt_train.mask_batch(input_ids, args.mask, cpt_train.MASK_ID, vocab_size, special_ids)
            sync_ctx = ddp_model.no_sync() if (world > 1 and mi < accum - 1) else contextlib.nullcontext()
            with sync_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, n_masked = ddp_model(inp, labels)
            (loss / accum).backward()
            step_loss += loss.item() / accum
            masked_local += int(n_masked)
        if not math.isfinite(step_loss):
            raise FloatingPointError(f"non-finite MLM loss at step {step}")
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        opt.step(); sched.step()
        tokens_seen += global_batch * args.seq
        loss_hist.append(step_loss)
        if rank == 0 and step % LOG_EVERY == 0:
            dt = time.time() - t0
            rec = {"step": step, "loss": round(step_loss, 4), "lr": sched.get_last_lr()[0], "tokens_s": round(tokens_seen / dt), "elapsed": round(dt, 1), "tokens_seen": tokens_seen, "sampler_epoch": sampler.epoch}
            print(f"[dapt] step {step}/{n_steps_run} loss {step_loss:.4f} lr {rec['lr']:.2e} tok/s {rec['tokens_s']} elapsed {dt:.0f}s pass {sampler.epoch}", flush=True)
            if log_f:
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if not args.bench and (step % VAL_EVERY == 0 or step == n_steps_run):
            if rank == 0:
                v = cpt_train.evaluate(wrapper, hold_mm, hold_idx, args.micro, args.mask, vocab_size, special_ids, device)
                val_losses.append({"step": step, "tokens_seen": tokens_seen, "holdout_loss": v})
                print(f"[dapt] step {step} holdout_loss {v:.4f}", flush=True)
                common.write_json(out_dir / "status.json", {"status": "running", "pid": os.getpid(), "step": step, "tokens_seen": tokens_seen, "holdout_loss": v, "updated_at": common.now()})
            if world > 1:
                dist.barrier()
        if not args.bench and tokens_seen >= tokens_target:
            break

    wall = time.time() - t0
    masked_t = torch.tensor([masked_local], device=device, dtype=torch.long)
    if world > 1:
        dist.all_reduce(masked_t, op=dist.ReduceOp.SUM)
    masked_total = int(masked_t.item())
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    if args.bench:
        if rank == 0:
            print(f"[dapt] BENCH {n_steps_run} steps, {tokens_seen} tokens, tok/s {tokens_seen / wall:.0f}, peak_mem {peak_gb:.2f} GB, first_loss {loss_hist[0]:.4f}, est_hours_300M {tokens_target / (tokens_seen / wall) / 3600:.2f}", flush=True)
    elif rank == 0:
        hf_dir = out_dir / "hf"
        model.save_pretrained(hf_dir, safe_serialization=True)
        tok.save_pretrained(hf_dir)
        # sanity: reload and compare a parameter to the in-memory model
        reloaded = XLMRobertaForMaskedLM.from_pretrained(hf_dir, local_files_only=True)
        same = torch.equal(reloaded.roberta.encoder.layer[0].attention.self.query.weight, model.roberta.encoder.layer[0].attention.self.query.weight.detach().cpu())
        tied_reload = reloaded.lm_head.decoder.weight.data_ptr() == reloaded.roberta.embeddings.word_embeddings.weight.data_ptr()
        summary = {
            **config, "finished_at": common.now(), "wall_time_s": round(wall, 1), "steps_run": len(loss_hist), "tokens_processed": tokens_seen,
            "token_budget_error_batches": (tokens_seen - tokens_target) / (global_batch * args.seq),
            "masked_positions_total": masked_total, "masked_fraction_of_processed": masked_total / max(tokens_seen, 1),
            "passes_over_train": tokens_seen / (n_train * args.seq), "sampler_epochs_started": sampler.epoch + 1,
            "tokens_per_s": round(tokens_seen / wall), "peak_gpu_memory_gb_rank0": round(peak_gb, 2),
            "train_loss_first_50_mean": float(np.mean(loss_hist[:50])), "train_loss_last_50_mean": float(np.mean(loss_hist[-50:])),
            "holdout": {"masking_seed": cpt_train.VAL_SEED, "sequences": int(len(hold_idx)), "losses": val_losses, "initial": val_losses[0]["holdout_loss"], "final": val_losses[-1]["holdout_loss"]},
            "export": {"hf_dir": str(hf_dir.relative_to(ROOT)), "model_safetensors_sha256": common.sha256_file(hf_dir / "model.safetensors"), "reload_layer0_query_equal": bool(same), "reload_decoder_tied": bool(tied_reload)},
            "status": "completed",
        }
        common.write_json(out_dir / "summary.json", summary)
        common.write_json(out_dir / "status.json", {"status": "completed", "pid": os.getpid(), "tokens_processed": tokens_seen, "holdout_final": val_losses[-1]["holdout_loss"], "finished_at": common.now()})
        print(f"[dapt] done: {tokens_seen} tokens in {wall/3600:.2f} h, holdout {val_losses[0]['holdout_loss']:.4f} -> {val_losses[-1]['holdout_loss']:.4f}, saved {hf_dir}", flush=True)
    if world > 1:
        dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if os.environ.get("RANK", "0") == "0":
            out = Path([a for a in sys.argv if a.startswith("--out=")][0][6:]) if any(a.startswith("--out=") for a in sys.argv) else OUT_DEFAULT
            try:
                common.write_json(out / "status.json", {"status": "failed", "error": repr(error), "finished_at": common.now()})
            except Exception:
                pass
        raise
