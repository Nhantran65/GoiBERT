#!/usr/bin/env python3
"""v4 §3 CPT run for one arm (P_web | P_inf | C_inf) of the mmBERT Vietnamese pilot.

Recipe (v1 §5 / v3 §4): 250 019 840 non-pad tokens = 3815 steps x 128 x 512, AdamW lr 5e-5, betas (0.9, 0.98), eps 1e-6,
weight decay 0.01, grad clip 1.0, BF16 autocast + FP32 master, seed 42, WSD (6 % warmup, constant to 70 %, linear decay),
15 % masking with the BERT 80/10/10 split generated on CPU from a generator seeded by (step, micro) — identical mask
positions for every arm.  Holdout MLM (four sets) at 0 %, 50 % and 100 % of the budget; the checkpoint is the end of the
budget (no downstream selection); the encoder-only export is reload-checked.

LICENCE: P_inf and C_inf contain ViSoBERT data (research-only / CC BY-NC 4.0) and are exported under a directory name with
the `_NC_research_only` suffix; they must not be released or used commercially.  P_web has no NC data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v4 as C4  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

LOG_EVERY = 20
HASH_STEPS = (0, 1, 2, 3)
HOLDOUT_SETS = ("a_hlpt", "b_fineweb2", "c_visobert", "d_v3review")
VAL_MICRO = 64
DECODER_CHUNK = 2048


def arm_dir_name(arm: str) -> str:
    return arm + (C4.NC_SUFFIX if arm in C4.ARMS_NC else "")


def lr_at(step: int, steps: int, peak: float, warmup_fraction: float = 0.06, decay_start: float = 0.70) -> float:
    warmup = max(1, round(warmup_fraction * steps))
    decay_at = round(decay_start * steps)
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < decay_at:
        return peak
    return peak * max(0.0, (steps - step) / max(1, steps - decay_at))


def param_groups(model, lr: float, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (n.endswith("bias") or "norm" in n.lower()) else decay).append(p)
    return [{"params": decay, "weight_decay": wd, "lr": lr}, {"params": no_decay, "weight_decay": 0.0, "lr": lr}]


def mask_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: torch.Tensor, mask_id: int,
                vocab_size: int, prob: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    special = torch.isin(input_ids, special_ids)
    maskable = attention_mask.bool() & ~special
    r = torch.rand(input_ids.shape, generator=generator)
    selected = maskable & (r < prob)
    p = torch.rand(input_ids.shape, generator=generator)
    do_mask = selected & (p < 0.8)
    do_random = selected & ~do_mask & (p < 0.9)
    labels = input_ids.clone()
    labels[~selected] = -100
    masked = input_ids.clone()
    masked[do_mask] = mask_id
    rand_ids = torch.randint(0, vocab_size, input_ids.shape, generator=generator)
    masked[do_random] = rand_ids[do_random]
    return masked, labels


def mlm_loss(model, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = labels.reshape(-1)
    selected = flat != -100
    n = int(selected.sum())
    if n == 0:
        return torch.zeros((), device=input_ids.device, dtype=torch.float32), 0
    hidden = model.model(input_ids=input_ids).last_hidden_state
    hidden = model.head(hidden).reshape(-1, hidden.size(-1))[selected]
    target = flat[selected]
    total = torch.zeros((), device=input_ids.device, dtype=torch.float32)
    for start in range(0, n, DECODER_CHUNK):
        logits = model.decoder(hidden[start:start + DECODER_CHUNK].float())
        total = total + F.cross_entropy(logits, target[start:start + DECODER_CHUNK], reduction="sum")
    return total / n, n


@torch.no_grad()
def evaluate_holdouts(model, holdouts: dict[str, np.ndarray], special_ids: torch.Tensor, tok, device, autocast) -> dict:
    model.eval()
    out = {}
    for name, arr in holdouts.items():
        total, count = 0.0, 0
        for mi, start in enumerate(range(0, arr.shape[0], VAL_MICRO)):
            chunk = torch.from_numpy(np.asarray(arr[start:start + VAL_MICRO], dtype=np.int64))
            am = torch.ones_like(chunk)
            gen = torch.Generator().manual_seed(1234 + mi)
            masked, labels = mask_tokens(chunk, am, special_ids, tok.mask_token_id, len(tok), 0.15, gen)
            with autocast():
                loss, n = mlm_loss(model, masked.to(device), labels.to(device))
            total += float(loss) * n
            count += n
        out[name] = total / max(1, count)
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=C4.ARMS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=500)
    args = ap.parse_args()
    arm = args.arm
    arm_dir = C4.ART_V4 / "arms" / arm_dir_name(arm)
    out_dir = Path(args.out) if args.out else C4.ART_V4 / "cpt" / arm_dir_name(arm)
    arm_manifest = C4.read_json(arm_dir / "manifest.json")
    if not args.profile:
        if (out_dir / "summary.json").exists():
            raise FileExistsError(f"refusing to overwrite {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        C4.write_json(out_dir / "status.json", {"status": "running", "arm": arm, "pid": os.getpid(), "device": args.device,
                                                "started_at": C4.common.now(), "research_only": arm in C4.ARMS_NC,
                                                "license": C4.LICENSE_NOTE if arm in C4.ARMS_NC else "no NC data"})
    torch.manual_seed(C4.SEED)
    np.random.seed(C4.SEED)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16)

    tokens = np.load(arm_dir / "tokens.npy", mmap_mode="r")
    steps = int(arm_manifest["steps"]); batch = int(arm_manifest["global_batch"]); seq_len = int(arm_manifest["seq_len"])
    if tokens.shape != (steps, batch, seq_len):
        raise RuntimeError(f"arm tensor shape {tokens.shape} != {(steps, batch, seq_len)}")
    micro = args.micro
    if batch % micro:
        raise ValueError("micro must divide the global batch")
    accum = batch // micro
    tok = C4.common.load_tokenizer()
    special_ids = torch.tensor(C4.common.special_ids(tok), dtype=torch.long)
    model, model_checks = C4.common.load_mlm(args.attn)
    model.to(device).train()
    opt = torch.optim.AdamW(param_groups(model, 5e-5, 0.01), lr=5e-5, betas=(0.9, 0.98), eps=1e-6)
    holdouts = {name: np.load(C4.ART_V4 / "holdouts" / f"{name}.npy", mmap_mode="r") for name in HOLDOUT_SETS}
    config = {"protocol": "mmbert-vi-cpt-pilot-v4", "arm": arm, "arm_dir": arm_dir_name(arm), "steps": steps, "global_batch": batch,
              "micro": micro, "accum": accum, "seq_len": seq_len, "tokens_scheduled": int(tokens.size), "token_budget": 250_000_000,
              "budget_error_batches": (int(tokens.size) - 250_000_000) / (batch * seq_len), "lr": 5e-5, "betas": [0.9, 0.98], "eps": 1e-6,
              "weight_decay": 0.01, "clip": 1.0, "schedule": "WSD: warmup 6 %, constant to 70 %, linear decay to 0 (C_inf stage 3 = decay stage)",
              "mask": "15 % module masking, 80/10/10, no special/pad, CPU generator seeded by (step, micro)", "seed": C4.SEED,
              "attn": args.attn, "model_checks": model_checks,
              "arm_manifest": {k: arm_manifest.get(k) for k in ("tokens_hash", "order_hash", "multiset_hash", "sequence_counts_by_group",
                                                               "informal_token_share", "license", "research_only")},
              "holdouts": {k: {"shape": list(v.shape), "sha256": C4.common.sha256_array(np.asarray(v[:64]))} for k, v in holdouts.items()},
              "license": C4.LICENSE_NOTE if arm in C4.ARMS_NC else "no NC data (general v3 pool + Wikipedia v2)",
              "research_only": arm in C4.ARMS_NC, "source_hash": C4.common.source_hash(), "runtime": C4.common.runtime(device),
              "argv": sys.argv, "started_at": C4.common.now()}
    if not args.profile:
        C4.write_json(out_dir / "resolved_config.json", config)
    print(f"[{arm}] steps {steps} micro {micro} accum {accum} tokens {int(tokens.size)} device {device} NC={arm in C4.ARMS_NC}", flush=True)

    history = {"loss": [], "lr": [], "grad_norm": []}
    val_rows: list[dict] = []
    batch_hashes: dict[str, str] = {}
    n_steps = args.profile if args.profile else steps
    val_points = {0, round(0.5 * steps), steps}
    start_step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if not args.profile and args.resume and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        start_step = int(state["step"]); history = state.get("history", history); val_rows = state.get("val_rows", val_rows)
        print(f"[{arm}] resuming from step {start_step}", flush=True)
    if not args.profile and start_step == 0:
        v = evaluate_holdouts(model, holdouts, special_ids, tok, device, autocast)
        val_rows.append({"step": 0, "fraction": 0.0, **v})
        print(f"[{arm}] step 0 holdout {json.dumps({k: round(x, 4) for k, x in v.items()})}", flush=True)
    log_f = None if args.profile else (out_dir / "log.jsonl").open("w")
    torch.cuda.reset_peak_memory_stats(device)
    tokens_seen = masked_seen = 0
    t0 = time.time()
    for step in range(start_step, n_steps):
        lr = lr_at(step, steps, 5e-5)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for mi in range(accum):
            ids = torch.from_numpy(np.asarray(tokens[step, mi * micro:(mi + 1) * micro], dtype=np.int64))
            am = torch.ones_like(ids)
            if step in HASH_STEPS and mi == 0:
                batch_hashes[str(step)] = C4.common.sha256_array(ids.numpy())
            gen = torch.Generator().manual_seed(C4.SEED * 1_000_003 + step * 1_009 + mi)
            masked, labels = mask_tokens(ids, am, special_ids, tok.mask_token_id, len(tok), 0.15, gen)
            if step in HASH_STEPS and mi == 0:
                batch_hashes[f"{step}_mask"] = C4.common.sha256_array(masked.numpy())
            with autocast():
                loss, n_masked = mlm_loss(model, masked.to(device), labels.to(device))
            (loss / accum).backward()
            step_loss += float(loss.detach()) / accum
            tokens_seen += int(ids.numel()); masked_seen += n_masked
        if not math.isfinite(step_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        history["loss"].append(step_loss); history["lr"].append(lr); history["grad_norm"].append(gnorm)
        if (step + 1) % LOG_EVERY == 0 or step + 1 == n_steps:
            dt = time.time() - t0
            rec = {"step": step + 1, "loss": round(step_loss, 4), "lr": lr, "grad_norm": round(gnorm, 3),
                   "tokens_seen": tokens_seen, "tokens_s": round(tokens_seen / max(dt, 1e-9)), "elapsed": round(dt, 1)}
            print(f"[{arm}] step {step+1}/{n_steps} loss {step_loss:.4f} lr {lr:.3e} gn {gnorm:.2f} tok/s {rec['tokens_s']} {dt:.0f}s", flush=True)
            if log_f:
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if not args.profile and (step + 1) in val_points:
            v = evaluate_holdouts(model, holdouts, special_ids, tok, device, autocast)
            val_rows.append({"step": step + 1, "fraction": (step + 1) / steps, **v})
            print(f"[{arm}] step {step+1} holdout {json.dumps({k: round(x, 4) for k, x in v.items()})}", flush=True)
            C4.write_json(out_dir / "status.json", {"status": "running", "arm": arm, "pid": os.getpid(), "step": step + 1,
                                                    "tokens_seen": tokens_seen, "holdout": v, "updated_at": C4.common.now()})
        if not args.profile and args.ckpt_every and (step + 1) % args.ckpt_every == 0 and (step + 1) < n_steps:
            tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step + 1, "history": history, "val_rows": val_rows}, tmp)
            tmp.replace(ckpt_path)
    if log_f:
        log_f.close()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    if args.profile:
        prof = {"arm": arm, "micro": micro, "accum": accum, "profile_steps": n_steps, "tokens_seen": tokens_seen,
                "tokens_s": tokens_seen / wall, "step_s": wall / n_steps, "peak_gb": peak, "eta_hours_full": wall / n_steps * steps / 3600,
                "first_loss": history["loss"][0], "last_loss": history["loss"][-1], "device": torch.cuda.get_device_name(device)}
        C4.write_json(C4.ART_V4 / "cpt" / f"profile_{arm}_micro{micro}.json", prof)
        print(f"[{arm}] PROFILE {json.dumps(prof)}", flush=True)
        return 0

    hf_mlm, hf_enc = out_dir / "hf_mlm", out_dir / "hf_encoder"
    model.save_pretrained(hf_mlm, safe_serialization=True); tok.save_pretrained(hf_mlm)
    model.model.save_pretrained(hf_enc, safe_serialization=True); tok.save_pretrained(hf_enc)
    from transformers import ModernBertForMaskedLM, ModernBertModel
    re_mlm = ModernBertForMaskedLM.from_pretrained(hf_mlm, local_files_only=True)
    re_enc, info = ModernBertModel.from_pretrained(hf_enc, local_files_only=True, output_loading_info=True)
    q_now = model.model.layers[0].attn.Wqkv.weight.detach().cpu().numpy()
    export = {"hf_mlm": str(hf_mlm.relative_to(C4.ROOT)), "hf_mlm_sha256": C4.common.sha256_file(hf_mlm / "model.safetensors"),
              "hf_encoder": str(hf_enc.relative_to(C4.ROOT)), "hf_encoder_sha256": C4.common.sha256_file(hf_enc / "model.safetensors"),
              "reload_mlm_layer0_query_equal": bool(np.array_equal(re_mlm.model.layers[0].attn.Wqkv.weight.detach().cpu().numpy(), q_now)),
              "reload_mlm_decoder_tied": bool(re_mlm.decoder.weight.data_ptr() == re_mlm.model.embeddings.tok_embeddings.weight.data_ptr()),
              "reload_encoder_layer0_query_equal": bool(np.array_equal(re_enc.layers[0].attn.Wqkv.weight.detach().cpu().numpy(), q_now)),
              "reload_encoder_missing_keys": list(info.get("missing_keys", [])), "reload_encoder_unexpected_keys": list(info.get("unexpected_keys", [])),
              "encoder_parameters": sum(p.numel() for p in re_enc.parameters()),
              "encoder_keys_contain_head_or_projection": [k for k in re_enc.state_dict() if k.startswith("head") or "decoder" in k or "proj" in k],
              "encoder_non_embedding_parameters": sum(p.numel() for p in re_enc.parameters()) - int(re_enc.embeddings.tok_embeddings.weight.numel()),
              "research_only": arm in C4.ARMS_NC, "license": C4.LICENSE_NOTE if arm in C4.ARMS_NC else "no NC data"}
    with torch.no_grad():
        re_enc.to(device).eval()
        enc_in = tok(["Quán ăn ngon, nhân viên nhiệt tình, giá cả hợp lý."], return_tensors="pt").to(device)
        smoke = re_enc(**enc_in).last_hidden_state
        export["inference_smoke"] = {"shape": list(smoke.shape), "finite": bool(torch.isfinite(smoke).all())}
    summary = {**config, "finished_at": C4.common.now(), "wall_time_s": round(wall, 1), "gpu_hours": round(wall / 3600, 3),
               "steps_run": n_steps, "tokens_processed": tokens_seen, "tokens_per_s": round(tokens_seen / wall),
               "tokens_processed_matches_schedule": tokens_seen == int(tokens.size), "peak_gpu_memory_gb": round(peak, 2),
               "masked_fraction": masked_seen / max(1, tokens_seen),
               "train_loss_first_100_mean": float(np.mean(history["loss"][:100])), "train_loss_last_100_mean": float(np.mean(history["loss"][-100:])),
               "holdout": {"sets": list(HOLDOUT_SETS), "mask_seed": 1234, "rows": val_rows, "initial": val_rows[0], "final": val_rows[-1]},
               "batch_hashes": batch_hashes, "export": export, "status": "completed"}
    C4.write_json(out_dir / "summary.json", summary)
    C4.write_json(out_dir / "status.json", {"status": "completed", "arm": arm, "pid": os.getpid(), "tokens_processed": tokens_seen,
                                            "holdout_final": val_rows[-1], "finished_at": C4.common.now()})
    print(f"[{arm}] done {tokens_seen} tokens in {wall/3600:.2f} h; holdout {val_rows[0]} -> {val_rows[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
