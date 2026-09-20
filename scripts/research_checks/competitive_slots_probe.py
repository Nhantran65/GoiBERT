"""Competitive-slot CLS probe on the frozen E8 student (validation-only diagnostic).

Mechanism under test: K CLS "slot" rows inside every encoder layer whose attention over
content tokens is gated by a competition softmax across slots (per key column, per head):

    w_kj  proportional to  exp(s_kj) * g_kj**beta,   g_kj = softmax_k(s_kj),  then normalised over j.

beta=0 reproduces plain per-slot softmax (Multi-CLS style, no competition); beta=1 is the
competitive gate.  With identical slots g_kj = 1/K for every j, so every slot reproduces the
original CLS row exactly -> the frozen checkpoint is function-preserved at zero init.

Token rows are unchanged (they see only the original CLS + tokens), so token states are exactly
those of the checkpoint.  Trainable parameters are tiny: a slot embedding offset delta[K,D] and a
per-layer per-slot query bias qb[L,K,D].  Encoder and head stay frozen.  Slot logits are decoded
by the frozen head and aggregated per aspect by the slot with the largest presence margin.

Training data: TRAIN rows + synthetic composites of TRAIN single-aspect rows (union labels).
Evaluation: real VALIDATION pair F1, the battery's 200 validation composites, partition stats.
No test split is loaded.
"""
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "independent_review"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mechanism_battery import load_model, build_pairs, scored_aspects, MAX_LEN, sha256  # noqa: E402
from goibert import train as tr  # noqa: E402

OUT = ROOT / "artifacts" / "research_diagnostics" / "competitive_slots"


class SlotEncoder:
    """Runs the frozen RoBERTa encoder with K slot rows and a competition-gated CLS attention."""

    def __init__(self, model, K: int, beta: float, device):
        self.model = model; self.K = K; self.beta = beta; self.device = device
        enc = model.encoder
        self.layers = enc.encoder.layer
        L = len(self.layers); D = enc.config.hidden_size
        self.H = enc.config.num_attention_heads; self.d = D // self.H
        self.delta = torch.zeros(K, D, device=device, requires_grad=True)
        self.qbias = torch.zeros(L, K, D, device=device, requires_grad=True)
        self.last_gate = None

    def params(self):
        return [self.delta, self.qbias]

    def init_noise(self, std: float, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.delta.copy_(torch.randn(self.delta.shape, generator=g) * std)

    def forward(self, ids, am, collect_gate: bool = False):
        model = self.model; K = self.K; H = self.H; d = self.d
        emb = model.encoder.embeddings(input_ids=ids)                     # B,N,D (eval: no dropout)
        B, N, D = emb.shape
        slots = emb[:, 0:1, :].expand(B, K, D) + self.delta[None]           # slot k = CLS + delta_k
        x = torch.cat([slots, emb[:, 1:, :]], dim=1)                        # B, K+N-1, D ; rows 0..K-1 slots
        M = x.shape[1]
        tok_valid = am[:, 1:].bool()                                        # B, N-1 (tokens incl </s>, excl CLS)
        gates = []
        for li, layer in enumerate(self.layers):
            sa = layer.attention.self
            q = sa.query(x); k = sa.key(x); v = sa.value(x)
            q = q.clone()
            q[:, :K, :] = q[:, :K, :] + self.qbias[li][None]                # per-layer per-slot query bias
            q = q.view(B, M, H, d).transpose(1, 2); k = k.view(B, M, H, d).transpose(1, 2); v = v.view(B, M, H, d).transpose(1, 2)
            # ---- token rows: keys = slot 0 (original CLS) + tokens, exactly the checkpoint path
            qt = q[:, :, K:, :]                                             # B,H,N-1,d
            kt_keys = torch.cat([k[:, :, 0:1, :], k[:, :, K:, :]], dim=2)   # B,H,N,d
            vt_vals = torch.cat([v[:, :, 0:1, :], v[:, :, K:, :]], dim=2)
            st = torch.matmul(qt, kt_keys.transpose(-1, -2)) / math.sqrt(d)  # B,H,N-1,N
            st = st.masked_fill(~am.bool()[:, None, None, :], float("-inf"))
            At = torch.softmax(st, dim=-1)
            ctx_t = torch.matmul(At, vt_vals)                               # B,H,N-1,d
            # ---- slot rows: keys = own slot column + tokens ; competition gate across slots
            qs = q[:, :, :K, :]                                             # B,H,K,d
            s_tok = torch.matmul(qs, k[:, :, K:, :].transpose(-1, -2)) / math.sqrt(d)   # B,H,K,N-1
            s_own = (qs * k[:, :, :K, :]).sum(-1, keepdim=True) / math.sqrt(d)          # B,H,K,1
            s_all = torch.cat([s_own, s_tok], dim=-1)                       # B,H,K,N  (col 0 = own)
            valid_cols = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=am.device), tok_valid], dim=1)  # B,N
            s_masked = s_all.masked_fill(~valid_cols[:, None, None, :], float("-inf"))
            if self.beta != 0.0:
                logg = torch.log_softmax(s_all, dim=2)                      # competition over slots, per column
                logits = s_masked + self.beta * logg
            else:
                logits = s_masked
            As = torch.softmax(logits, dim=-1)                              # B,H,K,N
            if collect_gate and self.beta != 0.0:
                gates.append(torch.softmax(s_all, dim=2).detach())          # B,H,K,N
            # own value for slot k is v of slot k
            v_own = v[:, :, :K, :]                                          # B,H,K,d
            ctx_s = As[:, :, :, 0:1] * v_own + torch.matmul(As[:, :, :, 1:], v[:, :, K:, :])
            ctx = torch.cat([ctx_s, ctx_t], dim=2).transpose(1, 2).reshape(B, M, H * d)
            attn_out = layer.attention.output(ctx, x)
            inter = layer.intermediate(attn_out)
            x = layer.output(inter, attn_out)
        self.last_gate = gates
        slot_states = x[:, :K, :]                                           # B,K,D
        head = model.head
        logits = head.output(head.activation(head.dense(slot_states)))     # B,K,A*4
        return logits

    def aggregate(self, slot_logits, counts):
        B, K, _ = slot_logits.shape
        A = len(counts); C = counts[0]
        L = slot_logits.view(B, K, A, C)
        margin = L[..., 1:].max(-1).values - L[..., 0]                     # B,K,A
        kstar = margin.argmax(1)                                            # B,A
        idx = kstar[:, None, :, None].expand(B, 1, A, C)
        return L.gather(1, idx).squeeze(1), kstar                           # B,A,C


def encode(tok, texts):
    return tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


def union_composites(rng, texts, Y, scored, singles, n_parts, n_attempts):
    out_t, out_y, out_parts = [], [], []
    for _ in range(n_attempts):
        idx = rng.sample(singles, n_parts)
        asps = [[a for a in scored if Y[i, a] > 0][0] for i in idx]
        if len(set(asps)) != n_parts: continue
        y = np.zeros_like(Y[idx[0]])
        for i, a in zip(idx, asps): y[a] = Y[i, a]
        out_t.append(" ".join(texts[i] for i in idx)); out_y.append(y); out_parts.append(idx)
    return out_t, np.array(out_y), out_parts


@torch.no_grad()
def eval_logits(se: SlotEncoder, tok, texts, counts, batch=64):
    outs, ks = [], []
    for i in range(0, len(texts), batch):
        enc = encode(tok, texts[i:i + batch])
        lg = se.forward(enc["input_ids"].to(se.device), enc["attention_mask"].to(se.device))
        agg, kstar = se.aggregate(lg.float(), counts)
        outs.append(agg.cpu()); ks.append(kstar.cpu())
    return torch.cat(outs, 0), torch.cat(ks, 0)


def composite_report(P, pairs, order_key):
    A_c = np.mean([P[i, A["aspect"]] == A["pol"] for i, (A, B) in enumerate(pairs)])
    A_p = np.mean([P[i, A["aspect"]] > 0 for i, (A, B) in enumerate(pairs)])
    B_c = np.mean([P[i, B["aspect"]] == B["pol"] for i, (A, B) in enumerate(pairs)])
    B_p = np.mean([P[i, B["aspect"]] > 0 for i, (A, B) in enumerate(pairs)])
    both = np.mean([(P[i, A["aspect"]] == A["pol"]) and (P[i, B["aspect"]] == B["pol"]) for i, (A, B) in enumerate(pairs)])
    return {f"{order_key}|A_correct": round(100 * float(A_c), 1), f"{order_key}|A_present": round(100 * float(A_p), 1),
            f"{order_key}|B_correct": round(100 * float(B_c), 1), f"{order_key}|B_present": round(100 * float(B_p), 1),
            f"{order_key}|both_correct": round(100 * float(both), 1)}


def multi_missed(P, Y, scored):
    mg = mm = 0
    for p, g in zip(P, Y):
        present = [i for i in scored if g[i] > 0]
        if len(present) >= 2:
            for i in present: mg += 1; mm += (p[i] == 0)
    return round(100 * mm / max(mg, 1), 1)


def error_decomposition(P, Y, scored):
    missed = spurious = wrong = tp = gold = pred = 0
    for p, g in zip(P, Y):
        for i in scored:
            gold += g[i] > 0; pred += p[i] > 0
            if g[i] > 0 and p[i] == 0: missed += 1
            elif g[i] == 0 and p[i] > 0: spurious += 1
            elif g[i] > 0 and p[i] > 0 and p[i] != g[i]: wrong += 1
            elif g[i] > 0 and p[i] == g[i]: tp += 1
    return {k: int(v) for k, v in {"missed": missed, "spurious": spurious, "wrong_polarity": wrong, "tp": tp, "gold": gold, "pred": pred}.items()}


@torch.no_grad()
def partition_stats(se: SlotEncoder, tok, pairs, counts, layer_ids=(7, 8, 9)):
    """Gate mass of A-part tokens vs B-part tokens on the slot selected for aspect A (last layers)."""
    if se.beta == 0.0 or se.K == 1: return None
    texts = [A["text"] + " " + B["text"] for A, B in pairs]
    a_len = [len(tok(A["text"], truncation=True, max_length=MAX_LEN)["input_ids"]) - 2 for A, B in pairs]  # content tokens of A (excl <s>,</s>)
    accA, accB, n = 0.0, 0.0, 0
    for i in range(0, len(texts), 50):
        enc = encode(tok, texts[i:i + 50]); ids = enc["input_ids"].to(se.device); am = enc["attention_mask"].to(se.device)
        lg = se.forward(ids, am, collect_gate=True)
        agg, kstar = se.aggregate(lg.float(), counts)
        for b in range(ids.shape[0]):
            A, B = pairs[i + b]
            k_sel = int(kstar[b, A["aspect"]])
            la = a_len[i + b]; n_tok = int(am[b].sum()) - 1
            for li in layer_ids:
                g = se.last_gate[li][b]                    # H,K,N (col0 own, cols 1.. tokens incl </s>)
                gA = g[:, k_sel, 1:1 + la].mean(); gB = g[:, k_sel, 1 + la:n_tok].mean()
                accA += float(gA); accB += float(gB); n += 1
    return {"gate_mass_on_selected_slot|A_tokens": round(accA / n, 3), "gate_mass_on_selected_slot|B_tokens": round(accB / n, 3)}


def run_arm(task, K, beta, steps, seed, device, lr, noise_std, mix_natural, log):
    model, tok, bundle, counts, ckpt = load_model("student_e8", task, device)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    scored = scored_aspects(task, bundle)
    tr_split = tr.split_field(bundle, "train"); va_split = tr.split_field(bundle, "validation")
    tr_texts = list(tr.split_texts(tr_split)); Ytr = np.array([list(l) for l in tr.split_labels(tr_split)])
    va_texts = list(tr.split_texts(va_split)); Yva = np.array([list(l) for l in tr.split_labels(va_split)])
    gold_va = tr.model_labels(bundle, "validation", va_split)
    rng = random.Random(seed)
    singles_tr = [i for i in range(len(tr_texts)) if sum(1 for a in scored if Ytr[i, a] > 0) == 1
                  and len(tok(tr_texts[i], truncation=True, max_length=MAX_LEN)["input_ids"]) <= 48]
    c2_t, c2_y, _ = union_composites(rng, tr_texts, Ytr, scored, singles_tr, 2, 3000)
    c3_t, c3_y, _ = union_composites(rng, tr_texts, Ytr, scored, singles_tr, 3, 1500)
    comp_t = c2_t + c3_t; comp_y = np.concatenate([c2_y, c3_y], 0)
    se = SlotEncoder(model, K, beta, device)
    # ---- parity at zero init: slot path (any K) must equal the K=1 manual path and be close to native bf16
    enc = encode(tok, va_texts[:64]); ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
    with torch.no_grad():
        lg0, _ = se.aggregate(se.forward(ids, am).float(), counts)
        ref = SlotEncoder(model, 1, 0.0, device)
        lg1, _ = ref.aggregate(ref.forward(ids, am).float(), counts)
        with tr.make_autocast("bf16", device):
            nat = model(ids, am).float().view(lg1.shape)
    parity = {"max_abs_slot_vs_manualK1": float((lg0 - lg1).abs().max()), "max_abs_manualK1_vs_native_bf16": float((lg1 - nat).abs().max())}
    log(f"[{task} K={K} beta={beta}] parity {parity}")
    se.init_noise(noise_std, seed)
    opt = torch.optim.Adam(se.params(), lr=lr)
    nat_idx = list(range(len(tr_texts))); comp_idx = list(range(len(comp_t)))
    t0 = time.time()
    for step in range(steps):
        n_nat = int(32 * mix_natural); n_comp = 32 - n_nat
        bi = rng.sample(nat_idx, n_nat); ci = rng.sample(comp_idx, n_comp)
        texts = [tr_texts[i] for i in bi] + [comp_t[i] for i in ci]
        labels = torch.tensor(np.concatenate([Ytr[bi], comp_y[ci]], 0), dtype=torch.long, device=device)
        enc = encode(tok, texts)
        lg = se.forward(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        agg, _ = se.aggregate(lg.float(), counts)
        loss = tr.compute_loss(agg, labels, num_aspects=len(counts), aspect_class_counts=counts)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            log(f"  step {step} loss {loss.item():.4f} |delta| {se.delta.norm().item():.3f} |qb| {se.qbias.norm().item():.3f}")
    train_s = time.time() - t0
    # ---- evaluation (validation only)
    lg_va, k_va = eval_logits(se, tok, va_texts, counts)
    P_va = np.array(tr._predictions(lg_va, aspect_class_counts=counts))
    f1 = tr.evaluate_predictions(task, bundle, gold_va, P_va.tolist(), split="validation")["selection_f1"]
    slot_usage = np.bincount(k_va[:, scored].reshape(-1).numpy(), minlength=K).tolist()
    singles, pairs = build_pairs(task, bundle, scored, tok, 200, 2026)
    rep = {}
    for order in ("A_first", "B_first"):
        texts = [A["text"] + " " + B["text"] if order == "A_first" else B["text"] + " " + A["text"] for A, B in pairs]
        lg, _ = eval_logits(se, tok, texts, counts); P = np.array(tr._predictions(lg, aspect_class_counts=counts))
        rep.update(composite_report(P, pairs, order))
    texts = [A["text"] + " " + " ".join([B["text"]] * 3) for A, B in pairs]
    lg, _ = eval_logits(se, tok, texts, counts); P = np.array(tr._predictions(lg, aspect_class_counts=counts))
    rep.update(composite_report(P, pairs, "A+3B"))
    lg, _ = eval_logits(se, tok, [A["text"] for A, B in pairs], counts); P = np.array(tr._predictions(lg, aspect_class_counts=counts))
    rep["A_alone|A_correct"] = round(100 * float(np.mean([P[i, A["aspect"]] == A["pol"] for i, (A, B) in enumerate(pairs)])), 1)
    result = {"task": task, "K": K, "beta": beta, "steps": steps, "seed": seed, "lr": lr, "noise_std": noise_std,
              "mix_natural": mix_natural, "checkpoint": str(ckpt.relative_to(ROOT)), "checkpoint_sha256": sha256(ckpt),
              "n_train_rows": len(tr_texts), "n_train_singles": len(singles_tr), "n_composites_2": len(c2_t), "n_composites_3": len(c3_t),
              "parity_zero_init": parity, "train_seconds": round(train_s, 1),
              "validation_pair_f1": round(100 * float(f1), 3), "validation_multi_row_missed_pct": multi_missed(P_va, Yva, scored),
              "validation_slot_usage": slot_usage, "validation_errors": error_decomposition(P_va, Yva, scored), "composites": rep,
              "partition": partition_stats(se, tok, pairs, counts)}
    log(f"[{task} K={K} beta={beta}] valF1={result['validation_pair_f1']} multi_missed={result['validation_multi_row_missed_pct']} usage={slot_usage} errors={result['validation_errors']}")
    log("  " + json.dumps(rep)); log("  partition " + json.dumps(result["partition"]))
    del model; torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tasks", nargs="+", default=["absa_restaurant", "absa_hotel"])
    ap.add_argument("--arms", nargs="+", default=["1:0", "2:0", "2:1", "3:1"], help="K:beta")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--mix-natural", type=float, default=0.5)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logf = open(OUT / f"log_{args.tag}.txt", "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    results = []
    for task in args.tasks:
        for arm in args.arms:
            K, beta = arm.split(":"); K = int(K); beta = float(beta)
            torch.manual_seed(args.seed)
            results.append(run_arm(task, K, beta, args.steps, args.seed, device, args.lr, args.noise_std, args.mix_natural, log))
            json.dump({"args": vars(args), "results": results}, open(OUT / f"results_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
