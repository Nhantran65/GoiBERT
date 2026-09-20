"""Slot-reader probe v3: routing targets from LATENT SPLITS of natural multi-aspect TRAIN sentences (no composites).

For each natural train sentence with >=2 gold aspects, enumerate candidate binary split points (after punctuation /
connector tokens, capped), run two hard-masked fork rows per split in the same forward pass (fork_test operator), and
select the split whose fork pair best explains the gold (aspect, polarity) set:

    score(s) = sum_{a in gold} max_i log p_{fork_i(s)}(y_a | a)

A split is accepted only if it beats the unsplit CLS score.  Gold aspects are assigned to the fork with the higher
log-probability; slots are matched to forks by Hungarian assignment and trained with KL(fork || slot) on the assigned
aspects (L_route), on top of the ordinary union/CE loss on all natural rows (L_union).  Encoder and head stay frozen;
trainable parameters are the slot offsets and per-layer query biases only.  Validation-only evaluation; no test split.
"""
from __future__ import annotations

import argparse, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "independent_review")); sys.path.insert(0, str(ROOT / "scripts" / "research_checks"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from mechanism_battery import load_model, build_pairs, scored_aspects, MAX_LEN, sha256  # noqa: E402
from goibert import train as tr  # noqa: E402
from competitive_slots_probe_v2 import (SlotEncoder, encode, eval_logits, composite_report, multi_missed,  # noqa: E402
                                        error_decomposition, partition_stats, margin_envelope, OUT)

CONNECTOR_IDS = {15, 17, 30, 4, 34, 258, 643, 617, 572, 2238, 659, 656}   # , . ; ! ?  và nhưng còn mà tuy nên thì


def candidate_splits(ids, max_splits):
    """Split points j (token index in full sequence, 1 <= j) such that segment1 = [1, j), segment2 = [j, n-1)."""
    n = len(ids)
    cands = [j for j in range(2, n - 2) if ids[j - 1] in CONNECTOR_IDS and j >= 2 and n - 1 - j >= 1]
    if not cands:
        cands = list(range(2, n - 2))
    if len(cands) > max_splits:
        idx = np.linspace(0, len(cands) - 1, max_splits).round().astype(int)
        cands = [cands[i] for i in idx]
    return cands


def collate_latent(examples, pad_id, device, max_splits):
    """Fork rows: 0 = full (unsplit) CLS; then (2s+1, 2s+2) = the two segments of split s."""
    N = max(len(e["ids"]) for e in examples); B = len(examples); m_max = 1 + 2 * max_splits
    ids = torch.full((B, N), pad_id, dtype=torch.long); am = torch.zeros((B, N), dtype=torch.long)
    fm = torch.ones((B, m_max, N - 1), dtype=torch.bool)
    for b, e in enumerate(examples):
        n = len(e["ids"]); ids[b, :n] = torch.tensor(e["ids"]); am[b, :n] = 1
        for s, j in enumerate(e.get("splits", [])):
            eos = n - 1
            fm[b, 1 + 2 * s] = False; fm[b, 1 + 2 * s, 0:j - 1] = True; fm[b, 1 + 2 * s, eos - 1] = True
            fm[b, 2 + 2 * s] = False; fm[b, 2 + 2 * s, j - 1:eos - 1] = True; fm[b, 2 + 2 * s, eos - 1] = True
    return ids.to(device), am.to(device), fm.to(device)


def latent_route_loss(slot_logits, fork_logits, examples, counts, scored, margin=0.0):
    B, K, _ = slot_logits.shape; A = len(counts); C = counts[0]
    Ls = torch.log_softmax(slot_logits.view(B, K, A, C).float(), dim=-1)
    Lf = torch.log_softmax(fork_logits.view(B, -1, A, C).float(), dim=-1)
    total = 0.0; n = 0; accepted = 0; considered = 0
    for b, e in enumerate(examples):
        splits = e.get("splits", [])
        if not splits: continue
        considered += 1
        gold = [(a, int(e["y"][a])) for a in scored if e["y"][a] > 0]
        base = sum(float(Lf[b, 0, a, ya]) for a, ya in gold)
        best = (base + margin, None)
        for s in range(len(splits)):
            f1, f2 = 1 + 2 * s, 2 + 2 * s
            sc = sum(float(torch.maximum(Lf[b, f1, a, ya], Lf[b, f2, a, ya])) for a, ya in gold)
            if sc > best[0]: best = (sc, s)
        if best[1] is None: continue
        accepted += 1
        s = best[1]; forks = [1 + 2 * s, 2 + 2 * s]
        assign = {0: [], 1: []}
        for a, ya in gold:
            i = 0 if float(Lf[b, forks[0], a, ya]) >= float(Lf[b, forks[1], a, ya]) else 1
            assign[i].append(a)
        cost = torch.zeros(K, 2, device=slot_logits.device)
        for i in (0, 1):
            asp = assign[i]
            if not asp: cost[:, i] = 0.0; continue
            pf = Lf[b, forks[i], asp, :].exp()
            cost[:, i] = (pf[None] * (Lf[b, forks[i], asp, :][None] - Ls[b, :, asp, :])).sum(-1).sum(-1)
        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        for rk, ci in zip(r, c):
            if assign[ci]: total = total + cost[rk, ci]; n += 1
    return (total / max(n, 1)) if n else torch.zeros((), device=slot_logits.device), accepted, considered


def run_arm(task, K, beta, steps, seed, device, lr, noise_std, route_w, max_splits, multi_frac, log):
    model, tok, bundle, counts, ckpt = load_model("student_e8", task, device)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    scored = scored_aspects(task, bundle)
    tr_split = tr.split_field(bundle, "train"); va_split = tr.split_field(bundle, "validation")
    tr_texts = list(tr.split_texts(tr_split)); Ytr = np.array([list(l) for l in tr.split_labels(tr_split)])
    va_texts = list(tr.split_texts(va_split)); Yva = np.array([list(l) for l in tr.split_labels(va_split)])
    gold_va = tr.model_labels(bundle, "validation", va_split)
    rng = random.Random(seed)
    natural = [{"ids": tok(t, truncation=True, max_length=MAX_LEN)["input_ids"], "y": Ytr[i]} for i, t in enumerate(tr_texts)]
    multi = []
    for e in natural:
        if sum(1 for a in scored if e["y"][a] > 0) >= 2 and len(e["ids"]) <= 96:
            multi.append({**e, "splits": candidate_splits(e["ids"], max_splits)})
    se = SlotEncoder(model, K, beta, device)
    enc = encode(tok, va_texts[:64]); ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
    with torch.no_grad():
        lg0, _ = se.aggregate(se.forward(ids, am).float(), counts)
        ref = SlotEncoder(model, 1, 0.0, device)
        lg1, _ = ref.aggregate(ref.forward(ids, am).float(), counts)
    parity = {"max_abs_slot_vs_manualK1": float((lg0 - lg1).abs().max())}
    log(f"[{task} K={K} beta={beta} route={route_w} latent] parity {parity} multi_rows={len(multi)}")
    se.init_noise(noise_std, seed)
    opt = torch.optim.Adam(se.params(), lr=lr)
    t0 = time.time(); acc_tot = con_tot = 0
    for step in range(steps):
        n_multi = int(32 * multi_frac)
        batch = rng.sample(multi, n_multi) + rng.sample(natural, 32 - n_multi)
        ids, am, fm = collate_latent(batch, tok.pad_token_id, device, max_splits)
        labels = torch.tensor(np.stack([e["y"] for e in batch]), dtype=torch.long, device=device)
        if route_w > 0:
            lg, fk = se.forward(ids, am, fork_masks=fm)
            lr_, acc, con = latent_route_loss(lg, fk, batch, counts, scored); acc_tot += acc; con_tot += con
        else:
            lg = se.forward(ids, am); lr_ = torch.zeros((), device=device)
        agg, _ = se.aggregate(lg.float(), counts)
        loss = tr.compute_loss(agg, labels, num_aspects=len(counts), aspect_class_counts=counts) + route_w * lr_
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            log(f"  step {step} loss {loss.item():.4f} route {float(lr_):.4f} accepted_splits {acc_tot}/{con_tot} |delta| {se.delta.norm().item():.3f} |qb| {se.qbias.norm().item():.3f}")
    train_s = time.time() - t0
    lg_va, k_va = eval_logits(se, tok, va_texts, counts)
    P_va = np.array(tr._predictions(lg_va, aspect_class_counts=counts))
    f1 = tr.evaluate_predictions(task, bundle, gold_va, P_va.tolist(), split="validation")["selection_f1"]
    tag = f"{task}_latent_K{K}_b{beta:g}_r{route_w:g}_s{seed}"
    np.save(OUT / f"val_logits_{tag}.npy", lg_va.numpy())
    envelope = margin_envelope(lg_va.numpy(), Yva, scored)
    slot_usage = np.bincount(k_va[:, scored].reshape(-1).numpy(), minlength=K).tolist()
    singles, pairs = build_pairs(task, bundle, scored, tok, 200, 2026)
    rep = {}
    for order in ("A_first", "B_first"):
        texts = [A["text"] + " " + B["text"] if order == "A_first" else B["text"] + " " + A["text"] for A, B in pairs]
        lg, _ = eval_logits(se, tok, texts, counts); P = np.array(tr._predictions(lg, aspect_class_counts=counts))
        rep.update(composite_report(P, pairs, order))
    result = {"task": task, "K": K, "beta": beta, "steps": steps, "seed": seed, "lr": lr, "noise_std": noise_std, "route_w": route_w,
              "max_splits": max_splits, "multi_frac": multi_frac, "mode": "latent_split_natural_only",
              "checkpoint": str(ckpt.relative_to(ROOT)), "checkpoint_sha256": sha256(ckpt), "n_multi_rows": len(multi),
              "accepted_splits": acc_tot, "considered_splits": con_tot, "parity_zero_init": parity, "train_seconds": round(train_s, 1),
              "validation_pair_f1": round(100 * float(f1), 3), "validation_multi_row_missed_pct": multi_missed(P_va, Yva, scored),
              "validation_slot_usage": slot_usage, "validation_errors": error_decomposition(P_va, Yva, scored),
              "validation_margin_envelope": envelope, "composites": rep, "partition": partition_stats(se, tok, pairs, counts)}
    log(f"[{task} K={K} beta={beta} route={route_w} latent] valF1={result['validation_pair_f1']} multi_missed={result['validation_multi_row_missed_pct']} usage={slot_usage} errors={result['validation_errors']} envelope={envelope}")
    log("  " + json.dumps(rep))
    del model; torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", nargs="+", default=["absa_restaurant", "absa_hotel"])
    ap.add_argument("--arms", nargs="+", default=["2:0:1", "2:0:0"], help="K:beta:route_w")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--max-splits", type=int, default=6)
    ap.add_argument("--multi-frac", type=float, default=0.5)
    ap.add_argument("--tag", default="v3_latent")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logf = open(OUT / f"log_{args.tag}.txt", "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    results = []
    for task in args.tasks:
        for arm in args.arms:
            K, beta, rw = arm.split(":"); K = int(K); beta = float(beta); rw = float(rw)
            torch.manual_seed(args.seed)
            results.append(run_arm(task, K, beta, args.steps, args.seed, device, args.lr, args.noise_std, rw, args.max_splits, args.multi_frac, log))
            json.dump({"args": vars(args), "results": results}, open(OUT / f"results_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
