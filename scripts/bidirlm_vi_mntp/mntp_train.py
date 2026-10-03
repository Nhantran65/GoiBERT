#!/usr/bin/env python3
"""Stage 1 §3 — 250M-token Vietnamese MNTP on top of BidirLM-0.6B-Base.

Fast path: BidirLM weights are loaded into the stock ``Qwen3ForCausalLM`` (keys prefixed with ``model.``,
tied ``lm_head``, ``strict=True``), every attention module gets ``is_causal=False`` and batches are
unpadded, so the read is bidirectional.  MNTP loss = CE of the lm_head at hidden position j−1 predicting
the original token at position j, gathered only at the 20 % selected positions (no full 513×152K logits).
``preflight.py`` checks this path against the custom ``BidirLMForMaskedLM`` before training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (ROOT / "src", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402  (stage-1 contract: pinned paths, arms, artifacts)

SEQ = 512
BOS_ID = C.IM_START_TOKEN_ID
MASK_ID = C.MASK_TOKEN_ID
MASK_PROB = 0.20
HELDOUT_MASK_SEED = 1234
DATA_DIR = ROOT / "data" / "bidirlm_vi_mntp"
MNTP_DIR = C.MNTP_ART


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str))


def disable_causal(model: Any) -> int:
    touched = 0
    for module in model.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
            touched += 1
    return touched


def build_fast_model(load_custom: bool = False, attn_implementation: str = "sdpa", dtype: Any = torch.float32):
    """Pinned BidirLM weights inside the stock Qwen3ForCausalLM; optionally also return the custom model."""
    from transformers import AutoModelForMaskedLM, Qwen3Config, Qwen3ForCausalLM

    snapshot = str(C.pinned_snapshot("bidirlm_base"))
    custom = AutoModelForMaskedLM.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True,
                                                  attn_implementation=attn_implementation, dtype=dtype)
    source_config = custom.config
    config = Qwen3Config(
        vocab_size=source_config.vocab_size, hidden_size=source_config.hidden_size,
        intermediate_size=source_config.intermediate_size, num_hidden_layers=source_config.num_hidden_layers,
        num_attention_heads=source_config.num_attention_heads, num_key_value_heads=source_config.num_key_value_heads,
        head_dim=source_config.head_dim, hidden_act=source_config.hidden_act,
        max_position_embeddings=source_config.max_position_embeddings, rms_norm_eps=source_config.rms_norm_eps,
        rope_theta=source_config.rope_theta, rope_scaling=source_config.rope_scaling, tie_word_embeddings=True,
        attention_bias=False, attention_dropout=0.0, pad_token_id=151643, bos_token_id=151644, eos_token_id=151645,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config)
    missing, unexpected = model.load_state_dict(custom.state_dict(), strict=True)
    model.tie_weights()
    is_causal_modules = disable_causal(model)
    info = {"missing_keys": list(missing), "unexpected_keys": list(unexpected),
            "is_causal_modules_set_false": is_causal_modules,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "tied_lm_head_is_input_embedding": model.lm_head.weight is model.model.embed_tokens.weight,
            "config": {"vocab_size": config.vocab_size, "hidden_size": config.hidden_size,
                       "num_hidden_layers": config.num_hidden_layers, "head_dim": config.head_dim,
                       "rope_theta": float(config.rope_theta), "max_position_embeddings": config.max_position_embeddings}}
    return (model, custom, info) if load_custom else (model, info)


def make_masked_batch(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bert-style 20 % MNTP corruption: 80 % <|mask|>, 10 % random, 10 % keep."""
    values = x.astype(np.int64)  # uint32 token ids -> int64 so -100 labels are representable
    selected = rng.random(values.shape) < MASK_PROB
    empty_rows = ~selected.any(axis=1)
    if empty_rows.any():  # guarantee at least one prediction per sequence
        rows = np.nonzero(empty_rows)[0]
        selected[rows, rng.integers(0, values.shape[1], size=rows.size)] = True
    draw = rng.random(values.shape)
    masked = values.copy()
    mask_positions = selected & (draw < 0.8)
    random_positions = selected & (draw >= 0.8) & (draw < 0.9)
    masked[mask_positions] = MASK_ID
    if random_positions.any():
        masked[random_positions] = rng.integers(0, 151936, size=int(random_positions.sum()), dtype=np.int64)
    labels = np.where(selected, values, -100).astype(np.int64)
    return masked.astype(np.int64), labels, selected


def batch_loss(model: Any, masked_x: torch.Tensor, labels: torch.Tensor, *, autocast: bool) -> tuple[torch.Tensor, int, int]:
    """CE at hidden j−1 predicting token j for the selected positions; returns (loss, correct, count)."""
    batch, length = masked_x.shape
    input_ids = torch.cat([torch.full((batch, 1), BOS_ID, dtype=masked_x.dtype, device=masked_x.device), masked_x], dim=1)
    # An all-ones 2D mask (not None) is required: with attention_mask=None and position_ids set,
    # transformers 4.57 detects "packed sequences" and forces a *causal* materialised mask even when
    # every module has is_causal=False (measured: backward influence 0.0).  All-ones + is_causal=False
    # gives the bidirectional read (measured: 8.5) and matches the custom code's input contract.
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        head_hidden = hidden[:, :length]
        selected = labels != -100
        logits = model.lm_head(head_hidden[selected])
    logits = logits.float()
    targets = labels[selected]
    loss = F.cross_entropy(logits, targets)
    correct = int((logits.argmax(dim=-1) == targets).sum().item())
    return loss, correct, int(targets.numel())


def lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.05, decay_start_frac: float = 0.70) -> float:
    warmup = max(1, int(round(total * warmup_frac)))
    decay_start = int(round(total * decay_start_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    if step < decay_start:
        return peak
    return peak * max(0.0, (total - step) / max(1, total - decay_start))


def make_optimizer(model: Any, lr: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        leaf = name.rsplit(".", 1)[-1]
        lowered = name.lower()
        if leaf == "bias" or (leaf == "weight" and ("norm" in lowered)):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.98), eps=1e-8)


@torch.no_grad()
def evaluate(model: Any, tokens: np.ndarray, indices: np.ndarray, *, device: torch.device, micro: int,
             autocast: bool, mask_seed: int = HELDOUT_MASK_SEED) -> dict[str, float]:
    model.eval()
    total_loss = total_correct = total_count = 0
    for start in range(0, len(indices), micro):
        chunk = indices[start : start + micro]
        x = tokens[chunk]
        rng = np.random.default_rng([mask_seed, int(start)])
        masked, labels, _ = make_masked_batch(x, rng)
        loss, correct, count = batch_loss(model, torch.from_numpy(masked.astype(np.int64)).to(device),
                                          torch.from_numpy(labels).to(device), autocast=autocast)
        total_loss += float(loss.item()) * count
        total_correct += correct
        total_count += count
    model.train()
    return {"masked_loss": total_loss / max(total_count, 1), "masked_accuracy": total_correct / max(total_count, 1),
            "masked_tokens": total_count}


def run(args: argparse.Namespace) -> int:
    started = time.time()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    out_dir = Path(args.out_dir) if args.out_dir else MNTP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    micro = int(args.micro_batch)
    global_batch = micro * int(args.accum)
    if global_batch != args.global_batch:
        raise ValueError(f"micro {micro} x accum {args.accum} != global batch {args.global_batch}")
    total_steps = int(args.total_tokens // (global_batch * SEQ))
    print(f"[mntp] device {torch.cuda.get_device_name(device)} micro {micro} accum {args.accum} "
          f"global {global_batch} steps {total_steps} ({total_steps*global_batch*SEQ/1e6:.1f}M tokens)", flush=True)

    tokens = np.load(args.tokens, mmap_mode="r")
    order = np.load(args.order)
    heldout = np.load(args.heldout)
    n_train = total_steps * global_batch
    train_indices = order[:n_train]
    print(f"[mntp] pool {tokens.shape} usable train sequences {n_train} heldout {len(heldout)}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, build_info = build_fast_model(attn_implementation="sdpa", dtype=torch.float32)
    model.to(device)
    model.train()
    optimizer = make_optimizer(model, args.lr)
    print(f"[mntp] fast path built: {build_info['parameters']} params, "
          f"is_causal modules {build_info['is_causal_modules_set_false']}, tied {build_info['tied_lm_head_is_input_embedding']}", flush=True)

    def step_batches(step: int):
        base = step * global_batch
        for m in range(int(args.accum)):
            chunk = train_indices[base + m * micro : base + (m + 1) * micro]
            x = np.asarray(tokens[chunk])
            rng = np.random.default_rng([args.seed, step, m])
            masked, labels, _ = make_masked_batch(x, rng)
            yield (torch.from_numpy(masked.astype(np.int64)).to(device), torch.from_numpy(labels).to(device))

    def run_steps(first_step: int, last_step: int, *, measure: bool) -> dict[str, Any]:
        losses, step_seconds = [], []
        if measure:
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(first_step, last_step):
            lr = lr_at(step, total_steps, args.lr, args.warmup_frac, args.decay_start_frac)
            for group in optimizer.param_groups:
                group["lr"] = lr
            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)
            step_loss = step_count = 0
            for masked, labels in step_batches(step):
                loss, _, count = batch_loss(model, masked, labels, autocast=args.precision == "bf16")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                (loss / int(args.accum)).backward()  # mean over the global batch = mean of the micro-batch means
                step_loss += float(loss.item()) * count
                step_count += count
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(step_loss / max(step_count, 1))
            if measure:
                step_seconds.append(time.time() - t0)
            if (step + 1) % args.log_every == 0 or step == first_step:
                recent = losses[-args.log_every:]
                print(f"[mntp] step {step+1}/{total_steps} lr {lr:.2e} loss {sum(recent)/len(recent):.4f} "
                      f"tok/s {global_batch*SEQ/max(step_seconds[-1] if measure else (time.time()-t0), 1e-9):.0f} "
                      f"({(time.time()-started)/60:.1f} min)", flush=True)
        out: dict[str, Any] = {"steps": last_step - first_step, "loss_first": losses[0] if losses else None,
                               "loss_last": sum(losses[-50:]) / max(len(losses[-50:]), 1) if losses else None}
        if measure:
            out.update({"seconds": sum(step_seconds), "seconds_per_step": float(np.mean(step_seconds)),
                        "tokens_per_second": global_batch * SEQ / float(np.mean(step_seconds)),
                        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "step_seconds": step_seconds[:5]})
        return out

    start_step = 0
    if args.resume:
        checkpoints = sorted(out_dir.glob("ckpt_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        checkpoints = [p for p in checkpoints if int(p.stem.split("_")[-1]) < total_steps]
        if not checkpoints:
            raise FileNotFoundError(f"--resume given but no checkpoint below step {total_steps} in {out_dir}")
        latest = checkpoints[-1]
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.to(device)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        print(f"[mntp] resumed from {latest.name} at step {start_step}", flush=True)

    profile: dict[str, Any] | None = None
    if args.profile_steps and not args.resume:
        print(f"[mntp] profiling {args.profile_steps} steps (model is re-initialised afterwards)", flush=True)
        profile = run_steps(0, args.profile_steps, measure=True)
        print(f"[mntp] profile: {profile['seconds_per_step']:.3f} s/step, {profile['tokens_per_second']:.0f} tok/s, "
              f"peak {profile['peak_gpu_memory_allocated_bytes']/1e9:.1f} GB", flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model, build_info = build_fast_model(attn_implementation="sdpa", dtype=torch.float32)
        model.to(device)
        optimizer = make_optimizer(model, args.lr)
        model.train()

    heldout_log: list[dict[str, Any]] = []

    def collect(step: int) -> None:
        record = {"step": step, "fraction": step / total_steps, **evaluate(model, tokens, heldout, device=device, micro=micro,
                                                                          autocast=args.precision == "bf16")}
        heldout_log.append(record)
        print(f"[mntp] heldout step {step} ({record['fraction']*100:.0f}%): loss {record['masked_loss']:.4f} "
              f"acc {record['masked_accuracy']:.4f}", flush=True)
        write_json(out_dir / "heldout.json", {"records": heldout_log, "mask_seed": HELDOUT_MASK_SEED})

    if start_step == 0:
        collect(0)
    history: list[dict[str, Any]] = []
    t_train = time.time()
    for block_start in range(start_step, total_steps, args.ckpt_every):
        block_end = min(block_start + args.ckpt_every, total_steps)
        summary = run_steps(block_start, block_end, measure=False)
        history.append({"first_step": block_start, "last_step": block_end, **summary})
        if block_end < total_steps:
            torch.save({"step": block_end, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                       out_dir / f"ckpt_step_{block_end}.pt")
        if block_end >= total_steps // 2 and all(r["step"] != total_steps // 2 for r in heldout_log):
            collect(total_steps // 2)
    train_seconds = time.time() - t_train
    collect(total_steps)

    final_state = {k: v.detach().to(torch.float32).cpu() for k, v in model.state_dict().items()}
    torch.save({"step": total_steps, "model": final_state, "format": "bidirlm-vi-mntp-final-v1"}, out_dir / "final_state.pt")
    report = {
        "protocol": "bidirlm-vi-mntp-stage1-mntp-v1", "created_at": C.now(),
        "objective": "MNTP: 20 % positions, 80 % <|mask|> 151663 / 10 % random / 10 % keep; loss = CE(hidden j-1 -> token j); "
                     "attention bidirectional (is_causal=False, unpadded batch, attention_mask=None); lm_head tied",
        "data": {"tokens_file": str(args.tokens), "order_file": str(args.order), "heldout_file": str(args.heldout),
                 "train_sequences": n_train, "train_tokens": n_train * SEQ, "seq_len": SEQ, "heldout_sequences": int(len(heldout))},
        "hyperparameters": {"optimizer": "AdamW", "peak_lr": args.lr, "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.01,
                            "gradient_clip_norm": 1.0, "global_batch": global_batch, "micro_batch": micro,
                            "gradient_accumulation_steps": int(args.accum), "steps": total_steps, "epochs": 1,
                            "schedule": f"WSD warmup {args.warmup_frac:.0%} linear -> constant -> linear decay to 0 over "
                                        f"{args.decay_start_frac:.0%}-100 %",
                            "precision": "bf16 autocast + float32 master weights", "seed": args.seed,
                            "mask_rng": "numpy default_rng([seed, step, micro]) per (step, micro-batch)"},
        "model": {"backbone": "BidirLM/BidirLM-0.6B-Base", "commit": C.MODELS["bidirlm_base"]["commit"],
                  "fast_path": "transformers Qwen3ForCausalLM (stock) with BidirLM weights, strict=True", **build_info},
        "profile": profile, "training": {"seconds": train_seconds, "gpu_hours": train_seconds / 3600, "history": history},
        "heldout": {"mask_seed": HELDOUT_MASK_SEED, "records": heldout_log},
        "elapsed_seconds": time.time() - started,
    }
    write_json(out_dir / "mntp_report.json", report)
    print(f"[mntp] done in {(time.time()-started)/60:.1f} min ({train_seconds/3600:.2f} GPU-h training); "
          f"final_state -> {out_dir/'final_state.pt'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", default=str(DATA_DIR / "tokens.npy"))
    ap.add_argument("--order", default=str(DATA_DIR / "order.npy"))
    ap.add_argument("--heldout", default=str(DATA_DIR / "heldout_indices.npy"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(MNTP_DIR))
    ap.add_argument("--total-tokens", type=int, default=250_000_000)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--decay-start-frac", type=float, default=0.70)
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile-steps", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="continue from the latest ckpt_step_*.pt if the run was interrupted")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
