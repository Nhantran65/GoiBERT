"""D2 probe: MIL-max token readout vs CLS readout on a FROZEN fine-tuned encoder (validation-only).

Question: is the label-co-occurrence failure (rare label pairs missed / substituted, mixed-polarity errors) located in
the CLS pooling + head, or in the token representations themselves?  Train small heads on frozen final-layer states:
  - cls_probe : linear slot head on the CLS state (re-trained probe, same procedure -> fair comparison with MIL)
  - mean_probe: same head on the mean of content-token states
  - mil_max   : presence_a = max_t (w_a . h_t + b_a) over content tokens; polarity read by a 3-way head at argmax token
Metrics on real validation: pair micro F1, rare-pair missed %, mixed-polarity wrong %, plus per-group tables.
No encoder training, no test split.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "independent_review"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mechanism_battery import load_model, scored_aspects, MAX_LEN, sha256  # noqa: E402
from goibert import train as tr  # noqa: E402

OUT = ROOT / "artifacts" / "research_diagnostics" / "mil_max_probe"


@torch.no_grad()
def extract(model, tok, texts, device, batch=32):
    """Final-layer token states (bf16 on CPU) + masks; CLS at index 0."""
    states, masks = [], []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
        with tr.make_autocast("bf16", device):
            h = model.encoder(input_ids=ids, attention_mask=am, return_dict=True).last_hidden_state
        content = am.bool().clone(); content[:, 0] = False
        eos = am.sum(1) - 1; content[torch.arange(len(eos)), eos] = False   # exclude </s>
        states.append(h.to(torch.bfloat16).cpu()); masks.append(content.cpu())
    return states, masks


class Probe(torch.nn.Module):
    def __init__(self, D, A, C, mode):
        super().__init__(); self.mode = mode; self.A = A; self.C = C
        self.presence = torch.nn.Linear(D, A)
        self.polarity = torch.nn.Linear(D, A * (C - 1))        # per-aspect polarity head (C-1 classes: pos/neg/neu)

    def forward(self, h, content):
        """h: [B,N,D] float, content: [B,N] bool -> presence logits [B,A], polarity logits [B,A,C-1]."""
        B, N, D = h.shape
        if self.mode == "cls":
            v = h[:, 0]; pres = self.presence(v); pol = self.polarity(v).view(B, self.A, self.C - 1)
        elif self.mode == "mean":
            w = content.float()[..., None]; v = (h * w).sum(1) / w.sum(1).clamp_min(1)
            pres = self.presence(v); pol = self.polarity(v).view(B, self.A, self.C - 1)
        else:  # mil_max
            s = self.presence(h)                                              # B,N,A
            s = s.masked_fill(~content[..., None], -1e4)
            pres, tstar = s.max(1)                                            # B,A
            hv = h[torch.arange(B)[:, None], tstar]                           # B,A,D  (state at argmax token per aspect)
            pol = torch.einsum("bad,acd->bac", hv, self.polarity.weight.view(self.A, self.C - 1, D)) + self.polarity.bias.view(self.A, self.C - 1)
        return pres, pol

    def predict(self, h, content):
        pres, pol = self.forward(h, content)
        present = pres > 0
        polarity = pol.argmax(-1) + 1
        return torch.where(present, polarity, torch.zeros_like(polarity))


def train_probe(mode, S, M, Y, D, A, C, device, epochs=8, lr=1e-3, seed=1):
    torch.manual_seed(seed)
    probe = Probe(D, A, C, mode).to(device); opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    order = list(range(len(S)))
    for ep in range(epochs):
        rng = np.random.RandomState(seed + ep); rng.shuffle(order)
        for bi in order:
            h = S[bi].to(device).float(); c = M[bi].to(device)
            y = Y_batches[bi].to(device)
            pres, pol = probe(h, c)
            present = (y > 0).float()
            loss = F.binary_cross_entropy_with_logits(pres, present)
            if present.any():
                loss = loss + F.cross_entropy(pol[present.bool()], (y[y > 0] - 1))
            opt.zero_grad(); loss.backward(); opt.step()
    return probe


def group_metrics(P, Y, scored, pairs_train):
    A = Y.shape[1]; groups = {}
    for g, p in zip(Y, P):
        pres = [a for a in scored if g[a] > 0]; k = len(pres)
        mixed = len(set(int(g[a]) for a in pres)) > 1
        minpair = min((pairs_train[frozenset((pres[i], pres[j]))] for i in range(k) for j in range(i + 1, k)), default=None)
        keys = ["all", f"k={min(k,3)}"]
        if k >= 2: keys += ["k>=2|mixed" if mixed else "k>=2|same", "k>=2|rare_pair" if minpair <= 5 else "k>=2|freq_pair"]
        for key in keys:
            d = groups.setdefault(key, {"rows": 0, "gold": 0, "missed": 0, "wrong": 0, "spurious": 0, "tp": 0, "pred": 0})
            d["rows"] += 1
            for a in scored:
                if g[a] > 0:
                    d["gold"] += 1
                    if p[a] == 0: d["missed"] += 1
                    elif p[a] != g[a]: d["wrong"] += 1
                    else: d["tp"] += 1
                if p[a] > 0: d["pred"] += 1
                if g[a] == 0 and p[a] > 0: d["spurious"] += 1
    out = {}
    for key, d in groups.items():
        out[key] = {"rows": d["rows"], "gold": d["gold"], "missed_pct": round(100 * d["missed"] / max(d["gold"], 1), 1), "wrong_pct": round(100 * d["wrong"] / max(d["gold"], 1), 1),
                    "spurious_per_row": round(d["spurious"] / max(d["rows"], 1), 3), "pair_f1": round(100 * 2 * d["tp"] / max(d["pred"] + d["gold"], 1), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="bami_e14", help="mechanism_battery model kind (bami_e14 | student_e8)")
    ap.add_argument("--checkpoint", default=None, help="override checkpoint path (state['model'] must load into the kind's classifier)")
    ap.add_argument("--tasks", nargs="+", default=["absa_hotel", "absa_restaurant"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    device = torch.device(args.device); OUT.mkdir(parents=True, exist_ok=True)
    import mechanism_battery as mb
    if args.checkpoint:
        mb.MODELS[args.kind]["ckpt"] = lambda t, _c=Path(args.checkpoint): _c
    results = {}
    for task in args.tasks:
        model, tok, bundle, counts, ckpt = load_model(args.kind, task, device)
        model.eval(); scored = scored_aspects(task, bundle); A = len(counts); C = counts[0]
        tr_split = tr.split_field(bundle, "train"); va_split = tr.split_field(bundle, "validation")
        tr_texts = list(tr.split_texts(tr_split)); Ytr = np.array([list(l) for l in tr.split_labels(tr_split)])
        va_texts = list(tr.split_texts(va_split)); Yva = np.array([list(l) for l in tr.split_labels(va_split)])
        gold_va = tr.model_labels(bundle, "validation", va_split)
        pairs_train = Counter()
        for y in Ytr:
            pres = [a for a in scored if y[a] > 0]
            for i in range(len(pres)):
                for j in range(i + 1, len(pres)): pairs_train[frozenset((pres[i], pres[j]))] += 1
        t0 = time.time()
        S_tr, M_tr = extract(model, tok, tr_texts, device); S_va, M_va = extract(model, tok, va_texts, device)
        D = S_tr[0].shape[-1]
        global Y_batches
        Y_batches = [torch.tensor(Ytr[i:i + 32], dtype=torch.long) for i in range(0, len(tr_texts), 32)]
        # reference: the model's own CLS head predictions
        with torch.no_grad():
            ref = []
            for i in range(0, len(va_texts), 32):
                enc = tok(va_texts[i:i + 32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                with tr.make_autocast("bf16", device):
                    lg = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu()
                ref += tr._predictions(lg, aspect_class_counts=counts)
        ref = np.array(ref)
        res = {"reference_model_head": {"pair_f1": round(100 * float(tr.evaluate_predictions(task, bundle, gold_va, ref.tolist(), split="validation")["selection_f1"]), 3), "groups": group_metrics(ref, Yva, scored, pairs_train)}}
        for mode in ("cls", "mean", "mil_max"):
            probe = train_probe(mode, S_tr, M_tr, Ytr, D, A, C, device, epochs=args.epochs)
            probe.eval(); P = []
            with torch.no_grad():
                for h, c in zip(S_va, M_va):
                    P += probe.predict(h.to(device).float(), c.to(device)).cpu().tolist()
            P = np.array(P)
            f1 = tr.evaluate_predictions(task, bundle, gold_va, P.tolist(), split="validation")["selection_f1"]
            res[mode] = {"pair_f1": round(100 * float(f1), 3), "groups": group_metrics(P, Yva, scored, pairs_train)}
            print(task, mode, res[mode]["pair_f1"], {k: (v["missed_pct"], v["wrong_pct"]) for k, v in res[mode]["groups"].items() if k in ("all", "k>=2|rare_pair", "k>=2|mixed", "k>=2|freq_pair")}, flush=True)
        print(task, "reference", res["reference_model_head"]["pair_f1"], {k: (v["missed_pct"], v["wrong_pct"]) for k, v in res["reference_model_head"]["groups"].items() if k in ("all", "k>=2|rare_pair", "k>=2|mixed", "k>=2|freq_pair")}, flush=True)
        res["checkpoint"] = str(ckpt); res["checkpoint_sha256"] = sha256(Path(ckpt)); res["elapsed_s"] = round(time.time() - t0, 1)
        results[task] = res
        json.dump(results, open(OUT / f"results_{args.kind}_{args.tag}.json", "w"), indent=1)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
