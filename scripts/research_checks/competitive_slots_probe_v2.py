"""Competitive-slot CLS probe v2: adds segment-fork routing targets (L_route) on self-composites.

v2 of competitive_slots_probe.py.  Composites are built at the token-id level so segment spans are exact;
fork rows (hard-masked CLS copies per segment, the fork_test operator) run in the same forward pass and
provide detached per-segment readouts; slots are matched to forks by Hungarian assignment on a KL cost.

Original v1 docstring follows.

Competitive-slot CLS probe on the frozen E8 student (validation-only diagnostic).

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
        self.absent_bias = torch.zeros((), device=device, requires_grad=True)   # optional single-scalar recalibration
        self.use_absent_bias = False
        self.last_gate = None

    def params(self):
        return [self.delta, self.qbias] + ([self.absent_bias] if self.use_absent_bias else [])

    def init_noise(self, std: float, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.delta.copy_(torch.randn(self.delta.shape, generator=g) * std)

    def forward(self, ids, am, collect_gate: bool = False, fork_masks=None):
        """fork_masks: optional bool tensor [B, m, N-1] over content columns (tokens incl </s>) for m fork rows.
        Fork rows are hard-masked CLS copies (no delta/qbias, plain softmax); their logits are returned detached."""
        model = self.model; K = self.K; H = self.H; d = self.d
        emb = model.encoder.embeddings(input_ids=ids)                     # B,N,D (eval: no dropout)
        B, N, D = emb.shape
        m = 0 if fork_masks is None else fork_masks.shape[1]
        slots = emb[:, 0:1, :].expand(B, K, D) + self.delta[None]           # slot k = CLS + delta_k
        parts = [slots]
        if m: parts.append(emb[:, 0:1, :].expand(B, m, D))                  # fork rows start from the original CLS
        parts.append(emb[:, 1:, :])
        x = torch.cat(parts, dim=1)                                         # B, K+m+N-1, D
        M = x.shape[1]; T0 = K + m                                          # first token row
        tok_valid = am[:, 1:].bool()                                        # B, N-1 (tokens incl </s>, excl CLS)
        gates = []
        for li, layer in enumerate(self.layers):
            sa = layer.attention.self
            q = sa.query(x); k = sa.key(x); v = sa.value(x)
            q = q.clone()
            q[:, :K, :] = q[:, :K, :] + self.qbias[li][None]                # per-layer per-slot query bias
            q = q.view(B, M, H, d).transpose(1, 2); k = k.view(B, M, H, d).transpose(1, 2); v = v.view(B, M, H, d).transpose(1, 2)
            # ---- token rows: keys = slot 0 (original CLS) + tokens, exactly the checkpoint path
            qt = q[:, :, T0:, :]                                            # B,H,N-1,d
            kt_keys = torch.cat([k[:, :, 0:1, :], k[:, :, T0:, :]], dim=2)  # B,H,N,d
            vt_vals = torch.cat([v[:, :, 0:1, :], v[:, :, T0:, :]], dim=2)
            st = torch.matmul(qt, kt_keys.transpose(-1, -2)) / math.sqrt(d)  # B,H,N-1,N
            st = st.masked_fill(~am.bool()[:, None, None, :], float("-inf"))
            At = torch.softmax(st, dim=-1)
            ctx_t = torch.matmul(At, vt_vals)                               # B,H,N-1,d
            # ---- slot rows: keys = own slot column + tokens ; competition gate across slots
            qs = q[:, :, :K, :]                                             # B,H,K,d
            s_tok = torch.matmul(qs, k[:, :, T0:, :].transpose(-1, -2)) / math.sqrt(d)  # B,H,K,N-1
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
            v_own = v[:, :, :K, :]                                          # B,H,K,d
            ctx_s = As[:, :, :, 0:1] * v_own + torch.matmul(As[:, :, :, 1:], v[:, :, T0:, :])
            ctx_parts = [ctx_s]
            if m:
                # ---- fork rows: plain softmax, hard segment mask (own column always allowed)
                qf = q[:, :, K:T0, :]                                       # B,H,m,d
                f_tok = torch.matmul(qf, k[:, :, T0:, :].transpose(-1, -2)) / math.sqrt(d)  # B,H,m,N-1
                f_own = (qf * k[:, :, K:T0, :]).sum(-1, keepdim=True) / math.sqrt(d)
                f_all = torch.cat([f_own, f_tok], dim=-1)
                f_valid = torch.cat([torch.ones(B, m, 1, dtype=torch.bool, device=am.device), fork_masks & tok_valid[:, None, :]], dim=2)  # B,m,N
                f_all = f_all.masked_fill(~f_valid[:, None, :, :], float("-inf"))
                Af = torch.softmax(f_all, dim=-1)
                ctx_f = Af[:, :, :, 0:1] * v[:, :, K:T0, :] + torch.matmul(Af[:, :, :, 1:], v[:, :, T0:, :])
                ctx_parts.append(ctx_f)
            ctx_parts.append(ctx_t)
            ctx = torch.cat(ctx_parts, dim=2).transpose(1, 2).reshape(B, M, H * d)
            attn_out = layer.attention.output(ctx, x)
            inter = layer.intermediate(attn_out)
            x = layer.output(inter, attn_out)
        self.last_gate = gates
        head = model.head
        slot_states = x[:, :K, :]                                           # B,K,D
        logits = head.output(head.activation(head.dense(slot_states)))     # B,K,A*4
        if m:
            fork_logits = head.output(head.activation(head.dense(x[:, K:T0, :]))).detach()   # B,m,A*4
            return logits, fork_logits
        return logits

    def aggregate(self, slot_logits, counts):
        B, K, _ = slot_logits.shape
        A = len(counts); C = counts[0]
        L = slot_logits.view(B, K, A, C)
        if self.use_absent_bias:
            L = torch.cat([L[..., :1] + self.absent_bias, L[..., 1:]], dim=-1)
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


def id_composites(rng, tok, texts, Y, scored, singles, n_parts, n_attempts):
    """Composites at the token-id level: ids = <s> + ids(P1) + ids(' '+P2) + ... + </s>; exact spans."""
    out = []
    for _ in range(n_attempts):
        idx = rng.sample(singles, n_parts)
        asps = [[a for a in scored if Y[i, a] > 0][0] for i in idx]
        if len(set(asps)) != n_parts: continue
        y = np.zeros_like(Y[idx[0]])
        ids = [tok.bos_token_id]; spans = []
        for j, (i, a) in enumerate(zip(idx, asps)):
            y[a] = Y[i, a]
            piece = tok(texts[i] if j == 0 else " " + texts[i], add_special_tokens=False)["input_ids"]
            spans.append((len(ids), len(ids) + len(piece))); ids += piece
        ids.append(tok.eos_token_id)
        out.append({"ids": ids, "y": y, "spans": spans, "present": [[a] for a in asps]})
    return out


def collate(examples, pad_id, device, m_max):
    N = max(len(e["ids"]) for e in examples); B = len(examples)
    ids = torch.full((B, N), pad_id, dtype=torch.long); am = torch.zeros((B, N), dtype=torch.long)
    fm = torch.ones((B, m_max, N - 1), dtype=torch.bool)      # default: full mask (== original CLS), ignored in loss
    for b, e in enumerate(examples):
        ids[b, :len(e["ids"])] = torch.tensor(e["ids"]); am[b, :len(e["ids"])] = 1
        if e.get("spans"):
            eos = len(e["ids"]) - 1
            for i, (s0, s1) in enumerate(e["spans"][:m_max]):
                fm[b, i] = False; fm[b, i, s0 - 1:s1 - 1] = True; fm[b, i, eos - 1] = True   # columns are token index-1
    return ids.to(device), am.to(device), fm.to(device)


def margin_envelope(logits, Y, scored, grid=np.linspace(-6, 6, 241)):
    """Diagnostic PR envelope: predict c*=argmax_{c>=1} when margin>delta else absent; report max pair F1 over delta.
    Not a deployable threshold (chosen on validation); mirrors READER_DIAGNOSTIC_SPEC oracle max-F1."""
    L = logits.reshape(logits.shape[0], -1, 4)[:, scored, :]
    G = Y[:, scored]
    m = L[..., 1:].max(-1) - L[..., 0]; cstar = L[..., 1:].argmax(-1) + 1
    gold = int((G > 0).sum()); best = (0.0, None); at0 = None
    for delta in grid:
        pred = m > delta
        tp = int(((pred) & (G > 0) & (cstar == G)).sum()); npred = int(pred.sum())
        f1 = 2 * tp / max(npred + gold, 1)
        if f1 > best[0]: best = (f1, float(delta))
        if abs(delta) < 1e-9: at0 = f1
    return {"oracle_max_pair_f1": round(100 * best[0], 3), "oracle_delta": best[1], "f1_at_delta0": round(100 * (at0 if at0 is not None else 0), 3)}


def route_loss(slot_logits, fork_logits, examples, counts):
    """Hungarian matching between slots and segment forks on KL(fork || slot) over aspects present in the segment."""
    from scipy.optimize import linear_sum_assignment
    B, K, _ = slot_logits.shape; A = len(counts); C = counts[0]
    Ls = torch.log_softmax(slot_logits.view(B, K, A, C).float(), dim=-1)
    Lf = torch.log_softmax(fork_logits.view(B, -1, A, C).float(), dim=-1)
    total = 0.0; n = 0
    for b, e in enumerate(examples):
        if not e.get("spans"): continue
        m = len(e["spans"]); cost = torch.zeros(K, m, device=slot_logits.device)
        for i in range(m):
            asp = e["present"][i]
            pf = Lf[b, i, asp, :].exp()                                     # |asp|,C
            kl = (pf[None] * (Lf[b, i, asp, :][None] - Ls[b, :, asp, :])).sum(-1).sum(-1)   # K
            cost[:, i] = kl
        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        total = total + cost[r, c].sum(); n += len(r)
    return total / max(n, 1)


def run_arm(task, K, beta, steps, seed, device, lr, noise_std, mix_natural, log, union_w=1.0, route_w=0.0, absent_bias=False):
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
    comps = id_composites(rng, tok, tr_texts, Ytr, scored, singles_tr, 2, 3000) + id_composites(rng, tok, tr_texts, Ytr, scored, singles_tr, 3, 1500)
    natural = [{"ids": tok(t, truncation=True, max_length=MAX_LEN)["input_ids"], "y": Ytr[i]} for i, t in enumerate(tr_texts)]
    se = SlotEncoder(model, K, beta, device)
    enc = encode(tok, va_texts[:64]); ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
    with torch.no_grad():
        lg0, _ = se.aggregate(se.forward(ids, am).float(), counts)
        ref = SlotEncoder(model, 1, 0.0, device)
        lg1, _ = ref.aggregate(ref.forward(ids, am).float(), counts)
        with tr.make_autocast("bf16", device):
            nat = model(ids, am).float().view(lg1.shape)
        # fork parity: a fork with the full mask must equal the original CLS logits
        fm = torch.ones((ids.shape[0], 1, ids.shape[1] - 1), dtype=torch.bool, device=device)
        _, fk = se.forward(ids, am, fork_masks=fm)
    parity = {"max_abs_slot_vs_manualK1": float((lg0 - lg1).abs().max()), "max_abs_manualK1_vs_native_bf16": float((lg1 - nat).abs().max()),
              "max_abs_fullmask_fork_vs_manualK1": float((fk[:, 0].view(lg1.shape) - lg1).abs().max())}
    log(f"[{task} K={K} beta={beta} union={union_w} route={route_w}] parity {parity}")
    se.init_noise(noise_std, seed)
    se.use_absent_bias = bool(absent_bias)
    opt = torch.optim.Adam(se.params(), lr=lr)
    t0 = time.time()
    for step in range(steps):
        n_nat = int(32 * mix_natural); n_comp = 32 - n_nat
        batch = rng.sample(natural, n_nat) + rng.sample(comps, n_comp)
        ids, am, fm = collate(batch, tok.pad_token_id, device, 3)
        labels = torch.tensor(np.stack([e["y"] for e in batch]), dtype=torch.long, device=device)
        if route_w > 0 and n_comp > 0:
            lg, fk = se.forward(ids, am, fork_masks=fm)
        else:
            lg = se.forward(ids, am); fk = None
        agg, _ = se.aggregate(lg.float(), counts)
        loss = union_w * tr.compute_loss(agg, labels, num_aspects=len(counts), aspect_class_counts=counts) if union_w > 0 else torch.zeros((), device=device)
        lr_ = route_loss(lg, fk, batch, counts) if fk is not None else torch.zeros((), device=device)
        loss = loss + route_w * lr_
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            log(f"  step {step} loss {loss.item():.4f} route {float(lr_):.4f} |delta| {se.delta.norm().item():.3f} |qb| {se.qbias.norm().item():.3f}")
    train_s = time.time() - t0
    lg_va, k_va = eval_logits(se, tok, va_texts, counts)
    P_va = np.array(tr._predictions(lg_va, aspect_class_counts=counts))
    f1 = tr.evaluate_predictions(task, bundle, gold_va, P_va.tolist(), split="validation")["selection_f1"]
    tag = f"{task}_K{K}_b{beta:g}_u{union_w:g}_r{route_w:g}_mix{mix_natural:g}_s{seed}" + ("_ab" if absent_bias else "")
    np.save(OUT / f"val_logits_{tag}.npy", lg_va.numpy())
    envelope = margin_envelope(lg_va.numpy(), Yva, scored)
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
    result = {"task": task, "K": K, "beta": beta, "steps": steps, "seed": seed, "lr": lr, "noise_std": noise_std, "union_w": union_w, "route_w": route_w,
              "absent_bias": (float(se.absent_bias) if absent_bias else None),
              "mix_natural": mix_natural, "checkpoint": str(ckpt.relative_to(ROOT)), "checkpoint_sha256": sha256(ckpt),
              "n_train_rows": len(tr_texts), "n_train_singles": len(singles_tr), "n_composites": len(comps),
              "parity_zero_init": parity, "train_seconds": round(train_s, 1),
              "validation_pair_f1": round(100 * float(f1), 3), "validation_multi_row_missed_pct": multi_missed(P_va, Yva, scored),
              "validation_slot_usage": slot_usage, "validation_errors": error_decomposition(P_va, Yva, scored), "composites": rep,
              "validation_margin_envelope": envelope, "partition": partition_stats(se, tok, pairs, counts)}
    log(f"[{task} K={K} beta={beta} union={union_w} route={route_w}] valF1={result['validation_pair_f1']} multi_missed={result['validation_multi_row_missed_pct']} usage={slot_usage} errors={result['validation_errors']} envelope={envelope}")
    log("  " + json.dumps(rep)); log("  partition " + json.dumps(result["partition"]))
    del model; torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tasks", nargs="+", default=["absa_restaurant", "absa_hotel"])
    ap.add_argument("--arms", nargs="+", default=["2:0:1:0", "2:0:1:1", "2:0:0:1", "2:1:1:1"], help="K:beta:union_w:route_w")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--mix-natural", type=float, default=0.5)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--absent-bias", action="store_true", help="train one scalar bias on the absent logit (minimal recalibration)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logf = open(OUT / f"log_{args.tag}.txt", "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    results = []
    for task in args.tasks:
        for arm in args.arms:
            K, beta, uw, rw = arm.split(":"); K = int(K); beta = float(beta); uw = float(uw); rw = float(rw)
            torch.manual_seed(args.seed)
            results.append(run_arm(task, K, beta, args.steps, args.seed, device, args.lr, args.noise_std, args.mix_natural, log, union_w=uw, route_w=rw, absent_bias=args.absent_bias))
            json.dump({"args": vars(args), "results": results}, open(OUT / f"results_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
