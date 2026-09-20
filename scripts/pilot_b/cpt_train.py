#!/usr/bin/env python3
"""Pilot B: DDP continued MLM pretraining of CafeBERT.

Arms: a = original embedding, no diacritic dropout (control). c = original embedding + diacritic dropout.
b = diacritic-factorized embedding (rank r) + diacritic dropout. b2 = additive skeleton/pattern channels (exact init) + diacritic dropout. See docs/PILOT_B_DIACRITIC_FACTORIZED.md.

Launch: CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-e0-cuda/bin/python -m torch.distributed.run \
    --nproc_per_node 4 scripts/pilot_b/cpt_train.py --arm a ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from goibert_pilot_b.factorized_embedding import AdditiveDiacriticEmbedding, DiacriticFactorizedEmbedding, apply_diacritic_dropout  # noqa: E402
from transformers import AutoTokenizer, XLMRobertaForMaskedLM  # noqa: E402

TOKENIZER = "FacebookAI/xlm-roberta-large"
BACKBONE = "uitnlp/CafeBERT"
DATA = ROOT / "data" / "cpt" / "fineweb2_vie_xlmr_512.npy"
META = ROOT / "data" / "cpt" / "fineweb2_vie_xlmr_512.meta.json"
DMAP = ROOT / "data" / "cpt" / "xlmr_diacritic_map.npz"
MASK_ID = 250001
VAL_SEED = 1234
VAL_EVERY = 500
LOG_EVERY = 20


def parse_tokens(s: str) -> int:
    return int(float(s))


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def mask_batch(input_ids, mask_p, mask_id, vocab_size, special_ids, generator=None):
    """Standard BERT masking: 15% of non-special tokens, 80/10/10 mask/random/unchanged. labels=-100 elsewhere."""
    special = torch.isin(input_ids, special_ids)
    prob = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    selected = (prob < mask_p) & ~special
    labels = input_ids.clone()
    labels[~selected] = -100
    choice = torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
    out = input_ids.clone()
    out[selected & (choice < 0.8)] = mask_id
    rand_pos = selected & (choice >= 0.8) & (choice < 0.9)
    if rand_pos.any():
        rand_tok = torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device, generator=generator)
        out[rand_pos] = rand_tok[rand_pos]
    return out, labels


class SparseMLM(nn.Module):
    """Wraps an XLMRobertaForMaskedLM and computes the MLM loss only at masked positions (sparse head)."""

    def __init__(self, base: XLMRobertaForMaskedLM):
        super().__init__()
        self.base = base

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        hidden = self.base.roberta(input_ids=input_ids, attention_mask=None)[0]
        sel = labels.ne(-100)
        n_masked = int(sel.sum().item())
        if n_masked == 0:
            return hidden.sum() * 0.0, 0
        h = hidden[sel]
        targets = labels[sel]
        h = self.base.lm_head.dense(h)
        h = F.gelu(h)
        h = self.base.lm_head.layer_norm(h)
        emb = self.base.roberta.embeddings.word_embeddings
        E_mat = emb.materialize() if hasattr(emb, "materialize") else emb.weight
        logits = h.float() @ E_mat.float().t() + self.base.lm_head.bias.float()
        loss = F.cross_entropy(logits, targets)
        return loss, n_masked


def rank_epoch_indices(n_train: int, world: int, rank: int, epoch: int, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed + epoch).permutation(n_train)
    return perm[rank::world]


class ShardSampler:
    """Each rank walks its own rank-strided slice of a (per-epoch) global shuffle; reshuffles on wraparound."""

    def __init__(self, n_train: int, world: int, rank: int, micro: int, seed: int):
        self.n_train, self.world, self.rank, self.micro, self.seed = n_train, world, rank, micro, seed
        self.epoch = 0
        self.idx = rank_epoch_indices(n_train, world, rank, 0, seed)
        self.pos = 0

    def next_batch(self) -> np.ndarray:
        if self.pos + self.micro > len(self.idx):
            self.epoch += 1
            self.idx = rank_epoch_indices(self.n_train, self.world, self.rank, self.epoch, self.seed)
            self.pos = 0
        batch = self.idx[self.pos:self.pos + self.micro]
        self.pos += self.micro
        return batch


def param_groups(model: nn.Module, wd: float, lr: float, shared_lr_mult: float = 1.0):
    """Shared diacritic channels (E_skel / E_pat) are freshly initialised; they can get a larger lr than the backbone."""
    decay, no_decay, shared = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("E_skel") or n.endswith("E_pat"):
            shared.append(p)
        else:
            (no_decay if ("bias" in n or "norm" in n.lower()) else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd, "lr": lr}, {"params": no_decay, "weight_decay": 0.0, "lr": lr}]
    if shared:
        groups.append({"params": shared, "weight_decay": wd, "lr": lr * shared_lr_mult})
    return groups


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if total <= warmup:
        return 0.0
    return max(0.0, 1.0 - (step - warmup) / (total - warmup))


@torch.no_grad()
def evaluate(module: nn.Module, packed_mm, val_idx: np.ndarray, micro: int, mask_p: float, vocab_size: int,
             special_ids: torch.Tensor, device) -> float:
    module.eval()
    gen = torch.Generator(device=device).manual_seed(VAL_SEED)
    total_loss, total_masked = 0.0, 0
    for i in range(0, len(val_idx), micro):
        idx = val_idx[i:i + micro]
        rows = packed_mm[idx].astype(np.int64)
        input_ids = torch.from_numpy(rows).to(device)
        inp, labels = mask_batch(input_ids, mask_p, MASK_ID, vocab_size, special_ids, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, n_masked = module(inp, labels)
        total_loss += loss.item() * n_masked
        total_masked += n_masked
    module.train()
    return total_loss / max(1, total_masked)


def save_model(wrapper: SparseMLM, factorized, dmap, args, out_dir: Path, loss_hist, val_losses, tokens_seen,
               wall_s: float):
    model = wrapper.base
    if factorized is not None:
        with torch.no_grad():
            mat = factorized.materialize().detach().float().cpu()
        V, H = mat.shape
        device = next(model.parameters()).device
        new_emb = nn.Embedding(V, H, padding_idx=model.config.pad_token_id)
        new_emb.weight.data.copy_(mat)
        model.roberta.embeddings.word_embeddings = new_emb.to(device)
        decoder = nn.Linear(H, V, bias=True).to(device)
        decoder.weight = new_emb.weight  # tie
        decoder.bias = model.lm_head.bias
        model.lm_head.decoder = decoder
        torch.save(dict(arm=args.arm, state_dict=factorized.state_dict(), n_groups=int(dmap["n_groups"]),
                         n_patterns=int(dmap["n_patterns"]), ascii_tok=dmap["ascii_tok"],
                         pat_names=dmap["pat_names"], rank=args.rank),
                   out_dir / "factorized_state.pt")

    hf_dir = out_dir / "hf"
    model.save_pretrained(hf_dir)
    AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(hf_dir)

    summary = dict(arm=args.arm, args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    final_train_loss=float(np.mean(loss_hist[-50:])) if loss_hist else None,
                    val_losses=val_losses, tokens_seen=tokens_seen, wall_time_s=round(wall_s, 1),
                    tokens_per_s=round(tokens_seen / max(wall_s, 1e-9)))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[cpt_train:{args.arm}] saved hf/ + summary.json to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b", "b2", "c"], required=True)
    ap.add_argument("--tokens", type=parse_tokens, default=int(300e6))
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--global-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--mask", type=float, default=0.15)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--dropout-p", type=float, default=0.1)
    ap.add_argument("--dmap", default=None, help="skeleton/pattern map npz (default data/cpt/xlmr_diacritic_map.npz; use xlmr_diacritic_teencode_map.npz for the teencode-extended block)")
    ap.add_argument("--variant-p", type=float, default=0.0, help="prob. of replacing a canonical word token by its teencode variant (needs variant_tok in --dmap)")
    ap.add_argument("--shared-lr-mult", type=float, default=1.0, help="lr multiplier for E_skel/E_pat (arms b, b2)")
    ap.add_argument("--tag", default="", help="suffix for the default output dir (artifacts/pilot_b/cpt/<arm><tag>)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rank, world, local_rank = setup_dist()
    device = torch.device("cuda", local_rank)
    seed_all(args.seed + rank)

    out_dir = Path(args.out) if args.out else ROOT / "artifacts" / "pilot_b" / "cpt" / (args.arm + args.tag)
    if rank == 0 and not args.bench:
        out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(META.read_text())
    packed_mm = np.load(DATA, mmap_mode="r")
    n_total, seq_len = packed_mm.shape
    assert seq_len == args.seq, f"packed seq_len {seq_len} != --seq {args.seq}"
    n_val = meta["n_val"]
    n_train = n_total - n_val
    val_idx = np.arange(n_train, n_total)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    special_ids = torch.tensor(sorted(set(tok.all_special_ids)), device=device, dtype=torch.long)
    vocab_size = len(tok)

    dmap = np.load(args.dmap or DMAP, allow_pickle=True) if args.arm in ("b", "b2", "c") else None
    ascii_tok_t = torch.from_numpy(dmap["ascii_tok"]).to(device).long() if dmap is not None else None
    variant_tok_t = torch.from_numpy(dmap["variant_tok"]).to(device).long() if (dmap is not None and args.variant_p > 0) else None
    if args.variant_p > 0: assert variant_tok_t is not None, "--variant-p needs a --dmap with variant_tok"

    model = XLMRobertaForMaskedLM.from_pretrained(BACKBONE, attn_implementation="sdpa")
    factorized = None
    if args.arm == "b":
        emb = model.roberta.embeddings.word_embeddings
        E_orig = emb.weight.data.clone().to(device)
        skel_id = torch.from_numpy(dmap["skel_id"]).to(device)
        pat_id = torch.from_numpy(dmap["pat_id"]).to(device)
        n_groups, n_patterns = int(dmap["n_groups"]), int(dmap["n_patterns"])
        factorized, fstats = DiacriticFactorizedEmbedding.from_pretrained_matrix(
            E_orig, skel_id, pat_id, n_groups, n_patterns, rank=args.rank)
        model.roberta.embeddings.word_embeddings = factorized
        model.lm_head.decoder = None  # no separate 250k x 1024 decoder weight; bias stays on lm_head.bias
    elif args.arm == "b2":
        emb = model.roberta.embeddings.word_embeddings
        factorized = AdditiveDiacriticEmbedding(emb.weight.data, int(dmap["n_groups"]), int(dmap["n_patterns"]),
                                                torch.from_numpy(dmap["skel_id"]), torch.from_numpy(dmap["pat_id"]))
        model.roberta.embeddings.word_embeddings = factorized
        model.lm_head.decoder = None

    wrapper = SparseMLM(model).to(device)
    ddp_model = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else wrapper

    global_batch = args.global_batch
    accum = global_batch // (args.micro * world)
    assert accum >= 1 and accum * args.micro * world == global_batch, \
        f"--global-batch {global_batch} must be divisible by micro({args.micro}) * world({world})"

    steps_target = math.ceil(args.tokens / (global_batch * args.seq))
    warmup_steps = max(1, round(args.warmup * steps_target))
    n_steps_run = args.bench if args.bench else steps_target

    opt = torch.optim.AdamW(param_groups(wrapper, args.wd, args.lr, args.shared_lr_mult), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, warmup_steps, steps_target))

    sampler = ShardSampler(n_train, world, rank, args.micro, args.seed)

    log_f = (out_dir / "log.jsonl").open("w") if (rank == 0 and not args.bench) else None
    if rank == 0:
        print(f"[cpt_train:{args.arm}] world={world} micro={args.micro} accum={accum} global_batch={global_batch} "
              f"steps={n_steps_run} (target {steps_target}) warmup={warmup_steps} tokens_target={args.tokens}",
              flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    tokens_seen = 0
    loss_hist: list[float] = []
    val_losses: list[dict] = []
    t0 = time.time()
    for step in range(1, n_steps_run + 1):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for mi in range(accum):
            idx = sampler.next_batch()
            rows = packed_mm[idx].astype(np.int64)
            input_ids = torch.from_numpy(rows).to(device, non_blocking=True)
            inp, labels = mask_batch(input_ids, args.mask, MASK_ID, vocab_size, special_ids)
            if args.arm in ("b", "b2", "c"):
                inp = apply_diacritic_dropout(inp, labels, ascii_tok_t, args.dropout_p)
                if variant_tok_t is not None:
                    inp = apply_diacritic_dropout(inp, labels, variant_tok_t, args.variant_p)
            sync_ctx = ddp_model.no_sync() if (world > 1 and mi < accum - 1) else contextlib.nullcontext()
            with sync_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                loss, n_masked = ddp_model(inp, labels)
            (loss / accum).backward()
            step_loss += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
        opt.step()
        sched.step()
        tokens_seen += global_batch * args.seq
        loss_hist.append(step_loss)

        if rank == 0 and step % LOG_EVERY == 0:
            dt = time.time() - t0
            tps = tokens_seen / dt
            lr_now = sched.get_last_lr()[0]
            rec = dict(step=step, loss=round(step_loss, 4), lr=lr_now, tokens_s=round(tps), elapsed=round(dt, 1),
                       tokens_seen=tokens_seen)
            print(f"[cpt_train:{args.arm}] step {step}/{n_steps_run} loss {step_loss:.4f} lr {lr_now:.2e} "
                  f"tok/s {tps:.0f} elapsed {dt:.0f}s", flush=True)
            if log_f:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

        if not args.bench and (step % VAL_EVERY == 0 or step == n_steps_run):
            if rank == 0:
                vloss = evaluate(wrapper, packed_mm, val_idx, args.micro, args.mask, vocab_size, special_ids, device)
                val_losses.append(dict(step=step, val_loss=vloss))
                print(f"[cpt_train:{args.arm}] step {step} val_loss {vloss:.4f}", flush=True)
            if world > 1:
                dist.barrier()

        if args.bench == 0 and tokens_seen >= args.tokens:
            break

    if args.bench:
        if rank == 0:
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"[cpt_train:{args.arm}] BENCH done: {n_steps_run} steps, {tokens_seen} tokens, "
                  f"tok/s {tokens_seen / dt:.0f}, peak_mem {peak:.2f} GB, first_step_loss {loss_hist[0]:.4f}",
                  flush=True)
        if world > 1:
            dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    if rank == 0:
        save_model(wrapper, factorized, dmap, args, out_dir, loss_hist, val_losses, tokens_seen, time.time() - t0)
    if world > 1:
        dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
