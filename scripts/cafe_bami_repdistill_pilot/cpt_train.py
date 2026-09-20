#!/usr/bin/env python3
"""Intermediate training of BamiBERT, arm M (plain MLM) or G (same MLM + frozen-CafeBERT relational loss), plan §5.

Both arms: same locked corpus order (data.global_batch_indices), same per-(step, micro) mask generator, same initial
checkpoint, global batch 128 sequences, AdamW lr 2e-5 betas (0.9, 0.98) eps 1e-6 wd 0.01, warmup 6 % + linear decay,
clip 1.0, BF16 autocast + SDPA, seed 42, exactly `steps_target` optimizer steps from the manifest (30M non-pad
BamiBERT tokens, <= 1 pass).  G adds a clean student forward and a no-grad CafeBERT forward per micro-batch:
L_G = L_MLM + 1.0 * mean_{(T12,S6),(T24,S12)} KL(P_T || P_S).  Checkpoint = end of budget (no downstream selection).

  --profile N   run N steps with the real settings, print throughput / peak memory / ETA, write nothing else, exit
                (the full run then starts in a fresh process from the original checkpoint).
Outputs: artifacts/cafe_bami_repdistill_pilot/cpt/<arm>/{hf_mlm/, hf_encoder/, log.jsonl, summary.json, status.json}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import data as D  # noqa: E402
import objective as O  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

LOG_EVERY = 20
GRAD_RATIO_EVERY = int(os.environ.get("REPDISTILL_GRAD_RATIO_EVERY", "100"))
HASH_STEPS = {0, 1, 2, 3, 500, 1000, 2000, 3000}
VAL_SEED = 1234
VAL_MICRO = 64


def param_groups(model, wd: float, lr: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if ("bias" in n or "LayerNorm" in n or "layer_norm" in n) else decay).append(p)
    return [{"params": decay, "weight_decay": wd, "lr": lr}, {"params": no_decay, "weight_decay": 0.0, "lr": lr}]


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))


def encoder_grad_norm(model) -> float:
    total = 0.0
    for n, p in model.roberta.named_parameters():
        if p.grad is not None:
            total += float(p.grad.float().norm() ** 2)
    return math.sqrt(total)


def relational_loss(student, teacher, batch: dict, device, autocast) -> tuple[torch.Tensor, dict]:
    s_ids, s_am, s_sp = batch["s_ids"].to(device), batch["s_am"].to(device), batch["s_span"].to(device)
    t_ids, t_am, t_sp = batch["t_ids"].to(device), batch["t_am"].to(device), batch["t_span"].to(device)
    N = int(batch["n_spans"])
    with autocast():
        s_out = student.roberta(input_ids=s_ids, attention_mask=s_am, output_hidden_states=True)
    with torch.no_grad(), autocast():
        t_out = teacher(input_ids=t_ids, attention_mask=t_am, output_hidden_states=True)
    losses, rows = [], []
    for t_layer, s_layer in common.LAYER_PAIRS:
        tp, tv = O.span_pool(t_out.hidden_states[t_layer].detach(), t_sp, N)
        sp, sv = O.span_pool(s_out.hidden_states[s_layer], s_sp, N)
        kl, n_rows = O.relational_kl(tp, tv, sp, sv, common.REL_TEMPERATURE)
        losses.append(kl); rows.append(n_rows)
    return torch.stack(losses).mean(), {"rel_by_layer": [float(l.detach()) for l in losses], "rel_rows": rows}


@torch.no_grad()
def holdout_mlm(student, corpus: D.Corpus, special_ids, device, autocast, micro: int = VAL_MICRO) -> float:
    student.eval()
    total, count = 0.0, 0
    for mi, start in enumerate(range(0, len(corpus), micro)):
        batch = corpus.collate(np.arange(start, min(start + micro, len(corpus))))
        gen = torch.Generator().manual_seed(VAL_SEED + mi)
        masked, labels = O.mask_tokens(batch["s_ids"], batch["s_am"], special_ids, 20480, 20481, common.MASK_PROB, gen)
        with autocast():
            loss, n = O.mlm_loss(student, masked.to(device), batch["s_am"].to(device), labels.to(device))
        total += float(loss) * n; count += n
    student.train()
    return total / max(1, count)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("M", "G"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t_start = time.time()
    arm = args.arm
    out_dir = Path(args.out) if args.out else common.ART_DIR / "cpt" / arm
    if not args.profile:
        if (out_dir / "summary.json").exists():
            raise FileExistsError(f"refusing to overwrite a completed run: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        common.write_json(out_dir / "status.json", {"status": "running", "arm": arm, "pid": os.getpid(), "device": args.device, "started_at": common.now()})

    torch.manual_seed(common.SEED); np.random.seed(common.SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731

    manifest = common.read_json(common.DATA_DIR / "manifest.json")
    if not manifest["gate"]["all_ok"]:
        raise RuntimeError("data gate not passed")
    steps_target = int(manifest["budget"]["steps_target"])
    global_batch = common.GLOBAL_BATCH
    micro = args.micro
    if global_batch % micro:
        raise ValueError("micro must divide the global batch")
    accum = global_batch // micro
    warmup = max(1, round(0.06 * steps_target))
    corpus = D.Corpus(split="train"); holdout = D.Corpus(split="holdout")
    if steps_target * global_batch > len(corpus):
        raise RuntimeError("budget exceeds one pass")
    tok = common.load_student_tokenizer()
    special_ids = torch.tensor(sorted(set(tok.all_special_ids)), dtype=torch.long)

    student, student_checks = common.load_student_mlm()
    student.to(device).train()
    teacher, teacher_checks = (None, None)
    if arm == "G":
        teacher, teacher_checks = common.load_teacher()
        teacher.to(device)
    init_hash = common.sha256_array(student.roberta.encoder.layer[0].attention.self.query.weight.detach().cpu().numpy())
    opt = torch.optim.AdamW(param_groups(student, 0.01, 2e-5), lr=2e-5, betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup, steps_target))

    n_steps = args.profile if args.profile else steps_target
    config = {
        "protocol": common.PROTOCOL, "arm": arm, "stage": "intermediate_training", "objective": "MLM" if arm == "M" else "MLM + 1.0 * relational KL (CafeBERT frozen; layer pairs T12->S6, T24->S12; cosine/0.2; KL(P_T||P_S))",
        "layer_pairs": common.LAYER_PAIRS, "rel_temperature": common.REL_TEMPERATURE, "rel_lambda": common.REL_LAMBDA if arm == "G" else 0.0, "mask_prob": common.MASK_PROB, "mask_scheme": "80/10/10, no special/pad, CPU generator seeded per (step, micro)",
        "seed": common.SEED, "global_batch": global_batch, "micro": micro, "accum": accum, "steps_target": steps_target, "warmup_steps": warmup, "token_budget": common.TOKEN_BUDGET,
        "tokens_at_steps_target": manifest["budget"]["tokens_at_steps_target"], "lr": 2e-5, "betas": [0.9, 0.98], "eps": 1e-6, "weight_decay": 0.01, "clip": 1.0, "precision": "fp32 master + bf16 autocast", "attn": "sdpa",
        "corpus": {"path": manifest["outputs"]["corpus"], "sha256": manifest["outputs"]["corpus_sha256"], "train_sequences": len(corpus), "holdout_sequences": len(holdout), "train_s_ids_sha256": manifest["sequences"]["train_s_ids_sha256"]},
        "student_checks": student_checks, "teacher_checks": teacher_checks, "initial_layer0_query_sha256": init_hash, "checkpoint_policy": "end of fixed budget; no downstream selection",
        "source_hash": common.source_hash(), "runtime": common.runtime(device), "gpu_snapshot_at_start": common.gpu_snapshot(), "argv": sys.argv, "started_at": common.now(), "profile_steps": args.profile or None,
    }
    if not args.profile:
        common.write_json(out_dir / "resolved_config.json", config)
    print(f"[{arm}] steps {n_steps}/{steps_target} micro {micro} accum {accum} warmup {warmup} train_seq {len(corpus)} holdout_seq {len(holdout)} device {device}", flush=True)

    val = []
    if not args.profile:
        v0 = holdout_mlm(student, holdout, special_ids, device, autocast)
        val.append({"step": 0, "tokens_seen": 0, "holdout_mlm": v0}); print(f"[{arm}] step 0 holdout_mlm {v0:.4f}", flush=True)
    log_f = (out_dir / "log.jsonl").open("w") if not args.profile else None
    batch_hashes: dict[str, str] = {}
    torch.cuda.reset_peak_memory_stats(device)
    tokens_seen = 0; masked_total = 0
    hist = {"loss": [], "mlm": [], "rel": []}
    ratios = []
    t0 = time.time()
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        idx = D.global_batch_indices(step, global_batch)
        step_loss = step_mlm = step_rel = 0.0
        for mi in range(accum):
            batch = corpus.collate(idx[mi * micro:(mi + 1) * micro])
            if step in HASH_STEPS and mi == 0:
                batch_hashes[str(step)] = D.batch_hash(batch)
            gen = D.mask_generator(step, mi)
            masked, labels = O.mask_tokens(batch["s_ids"], batch["s_am"], special_ids, tok.mask_token_id, len(tok), common.MASK_PROB, gen)
            if step in HASH_STEPS and mi == 0:
                batch_hashes[f"{step}_mask"] = common.sha256_array(masked.numpy()) 
            s_am = batch["s_am"].to(device)
            with autocast():
                l_mlm, n_masked = O.mlm_loss(student, masked.to(device), s_am, labels.to(device))
            l_rel = None; rel_info = {}
            if arm == "G":
                l_rel, rel_info = relational_loss(student, teacher, batch, device, autocast)
                loss = l_mlm + common.REL_LAMBDA * l_rel
            else:
                loss = l_mlm
            measure = (step % GRAD_RATIO_EVERY == 0) and mi == 0 and arm == "G"
            if measure:
                l_mlm.backward(retain_graph=True); g_mlm = encoder_grad_norm(student); opt.zero_grad(set_to_none=True)
                l_rel.backward(retain_graph=True); g_rel = encoder_grad_norm(student); opt.zero_grad(set_to_none=True)
                ratios.append({"step": step, "grad_norm_mlm": g_mlm, "grad_norm_rel": g_rel, "ratio": g_rel / max(g_mlm, 1e-12)})
            (loss / accum).backward()
            step_loss += float(loss.detach()) / accum; step_mlm += float(l_mlm.detach()) / accum
            if l_rel is not None:
                step_rel += float(l_rel.detach()) / accum
            tokens_seen += int(s_am.sum()); masked_total += n_masked
        if not math.isfinite(step_loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        gnorm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
        opt.step(); sched.step()
        hist["loss"].append(step_loss); hist["mlm"].append(step_mlm); hist["rel"].append(step_rel)
        if (step + 1) % LOG_EVERY == 0 or step + 1 == n_steps:
            dt = time.time() - t0
            rec = {"step": step + 1, "loss": round(step_loss, 4), "mlm": round(step_mlm, 4), "rel": round(step_rel, 4) if arm == "G" else None, "lr": sched.get_last_lr()[0], "grad_norm": round(gnorm, 3),
                   "tokens_seen": tokens_seen, "tokens_s": round(tokens_seen / dt), "elapsed": round(dt, 1), "grad_ratio": ratios[-1]["ratio"] if ratios and ratios[-1]["step"] == step else None}
            print(f"[{arm}] step {step+1}/{n_steps} loss {step_loss:.4f} mlm {step_mlm:.4f} rel {step_rel:.4f} lr {rec['lr']:.2e} gn {gnorm:.2f} tok/s {rec['tokens_s']} {dt:.0f}s", flush=True)
            if log_f:
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if not args.profile and (step + 1) % 500 == 0:
            common.write_json(out_dir / "status.json", {"status": "running", "arm": arm, "pid": os.getpid(), "step": step + 1, "tokens_seen": tokens_seen, "updated_at": common.now()})
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    if args.profile:
        eta_h = wall / n_steps * steps_target / 3600
        prof = {"arm": arm, "micro": micro, "accum": accum, "profile_steps": n_steps, "tokens_seen": tokens_seen, "tokens_s": tokens_seen / wall, "step_s": wall / n_steps, "peak_gb": peak, "eta_hours_full": eta_h,
                "first_loss": hist["loss"][0], "last_loss": hist["loss"][-1], "grad_ratios": ratios, "device": str(device), "gpu_name": torch.cuda.get_device_name(device)}
        common.write_json(common.ART_DIR / "cpt" / f"profile_{arm}_micro{micro}.json", prof)
        print(f"[{arm}] PROFILE {json.dumps({k: v for k, v in prof.items() if k != 'grad_ratios'})}", flush=True)
        return 0

    v1 = holdout_mlm(student, holdout, special_ids, device, autocast)
    val.append({"step": n_steps, "tokens_seen": tokens_seen, "holdout_mlm": v1})
    print(f"[{arm}] final holdout_mlm {v1:.4f} (start {val[0]['holdout_mlm']:.4f})", flush=True)

    # export: full MLM model (for reload/holdout checks) + encoder-only (deployment / fine-tune input)
    hf_mlm, hf_enc = out_dir / "hf_mlm", out_dir / "hf_encoder"
    student.save_pretrained(hf_mlm, safe_serialization=True); tok.save_pretrained(hf_mlm)
    student.roberta.save_pretrained(hf_enc, safe_serialization=True); tok.save_pretrained(hf_enc)
    from transformers import RobertaForMaskedLM, RobertaModel
    re_mlm = RobertaForMaskedLM.from_pretrained(hf_mlm, local_files_only=True)
    re_enc, info = RobertaModel.from_pretrained(hf_enc, local_files_only=True, add_pooling_layer=False, output_loading_info=True)
    q_train = student.roberta.encoder.layer[0].attention.self.query.weight.detach().cpu()
    export = {
        "hf_mlm": str(hf_mlm.relative_to(common.ROOT)), "hf_mlm_sha256": common.sha256_file(hf_mlm / "model.safetensors"),
        "hf_encoder": str(hf_enc.relative_to(common.ROOT)), "hf_encoder_sha256": common.sha256_file(hf_enc / "model.safetensors"),
        "reload_mlm_layer0_query_equal": bool(torch.equal(re_mlm.roberta.encoder.layer[0].attention.self.query.weight, q_train)),
        "reload_mlm_decoder_tied": bool(re_mlm.lm_head.decoder.weight.data_ptr() == re_mlm.roberta.embeddings.word_embeddings.weight.data_ptr()),
        "reload_encoder_layer0_query_equal": bool(torch.equal(re_enc.encoder.layer[0].attention.self.query.weight, q_train)),
        "reload_encoder_missing_keys": list(info.get("missing_keys", [])), "reload_encoder_unexpected_keys": list(info.get("unexpected_keys", [])),
        "encoder_parameters": sum(p.numel() for p in re_enc.parameters()), "encoder_keys_contain_teacher_or_projection": any(("teacher" in k or "proj" in k or "lm_head" in k) for k in re_enc.state_dict()),
        "final_layer0_query_sha256": common.sha256_array(q_train.numpy()), "changed_from_init": common.sha256_array(q_train.numpy()) != init_hash,
    }
    with torch.no_grad():
        re_enc.to(device).eval()
        enc = tok(["Quán ăn ngon, nhân viên nhiệt tình."], return_tensors="pt").to(device)
        smoke = re_enc(**enc).last_hidden_state
        export["inference_smoke_one_pass"] = {"shape": list(smoke.shape), "finite": bool(torch.isfinite(smoke).all())}
    summary = {
        **config, "finished_at": common.now(), "wall_time_s": round(wall, 1), "gpu_hours": round(wall / 3600, 3), "steps_run": n_steps, "tokens_processed": tokens_seen,
        "tokens_processed_matches_schedule": tokens_seen == manifest["budget"]["tokens_at_steps_target"], "budget_error_batches": (tokens_seen - common.TOKEN_BUDGET) / (tokens_seen / n_steps),
        "passes_over_train": n_steps * global_batch / len(corpus), "masked_positions_total": masked_total, "masked_fraction": masked_total / max(1, tokens_seen),
        "tokens_per_s": round(tokens_seen / wall), "peak_gpu_memory_gb": round(peak, 2),
        "train_loss_first_50_mean": float(np.mean(hist["loss"][:50])), "train_loss_last_50_mean": float(np.mean(hist["loss"][-50:])),
        "mlm_first_50_mean": float(np.mean(hist["mlm"][:50])), "mlm_last_50_mean": float(np.mean(hist["mlm"][-50:])),
        "rel_first_50_mean": float(np.mean(hist["rel"][:50])) if arm == "G" else None, "rel_last_50_mean": float(np.mean(hist["rel"][-50:])) if arm == "G" else None,
        "grad_ratios": ratios, "batch_hashes": batch_hashes, "holdout": {"masking_seed": VAL_SEED, "sequences": len(holdout), "losses": val, "initial": val[0]["holdout_mlm"], "final": val[-1]["holdout_mlm"]},
        "export": export, "status": "completed",
    }
    common.write_json(out_dir / "summary.json", summary)
    common.write_json(out_dir / "status.json", {"status": "completed", "arm": arm, "pid": os.getpid(), "tokens_processed": tokens_seen, "holdout_final": v1, "finished_at": common.now()})
    print(f"[{arm}] done {tokens_seen} tokens in {wall/3600:.2f} h; holdout {val[0]['holdout_mlm']:.4f} -> {v1:.4f}; export {export['hf_encoder']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            arm = sys.argv[sys.argv.index("--arm") + 1]
            common.write_json(common.ART_DIR / "cpt" / arm / "status.json", {"status": "failed", "error": repr(error), "finished_at": common.now()})
        except Exception:
            pass
        raise
