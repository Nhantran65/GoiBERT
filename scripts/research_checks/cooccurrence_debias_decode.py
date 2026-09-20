"""Decision-level co-occurrence debiasing on E19 validation logits (inference only, no training).

Hypothesis under test: the rare-label-pair misses / frequent-partner substitutions come from a label co-occurrence
prior that the fine-tuned model applies at the decision level.  If so, subtracting an estimate of that prior from the
presence margins should rescue rare pairs.

Prior estimate from TRAIN label statistics (naive-Bayes lift over pairs, Laplace-smoothed):
    lift_a(S) = sum_{b in S} [ log P(a | b) - log P(a) ]     (S = set of OTHER aspects present)
Adjusted presence margin: m_a' = m_a - alpha * lift_a(S), with m_a = max_{c>=1} L[a,c] - L[a,0].
Two settings for S: oracle (gold other aspects; diagnostic upper bound) and predicted (model's own first-pass set,
one refinement iteration).  Reports pair F1 and the group table (rare-pair missed etc.) per alpha.
"""
from __future__ import annotations

import argparse, json, math, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "independent_review")); sys.path.insert(0, str(ROOT / "scripts" / "research_checks"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as tr  # noqa: E402
from goibert_e2.modeling import build_e2_teacher  # noqa: E402
from mil_max_probe import group_metrics  # noqa: E402
from mechanism_battery import scored_aspects, MAX_LEN  # noqa: E402

OUT = ROOT / "artifacts" / "research_diagnostics" / "cooccurrence_debias"


@torch.no_grad()
def validation_logits(task, ckpt, device):
    bundle = tr.load_task_bundle(task, ROOT, include_test=False)
    A = tr.task_num_aspects(bundle); counts = tr.task_aspect_class_counts(bundle)
    model, _ = build_e2_teacher(ROOT / "models" / "goibert-b-dapt", num_classes=tr.task_num_classes(bundle), num_aspects=A, aspect_class_counts=counts, attention_implementation="sdpa")
    state = torch.load(ckpt, map_location="cpu", weights_only=False); model.load_state_dict(state["model"], strict=True); model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / "goibert-b-dapt", local_files_only=True, use_fast=True)
    va = tr.split_field(bundle, "validation"); texts = list(tr.split_texts(va)); Y = np.array([list(l) for l in tr.split_labels(va)])
    gold = tr.model_labels(bundle, "validation", va)
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with tr.make_autocast("bf16", device):
            lg = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        outs.append(lg.float().cpu())
    L = torch.cat(outs, 0).view(len(texts), A, counts[0]).numpy()
    return bundle, L, Y, gold, counts


def train_stats(bundle, scored):
    trs = tr.split_field(bundle, "train"); Ytr = np.array([list(l) for l in tr.split_labels(trs)]); n = len(Ytr)
    single = Counter(); pair = Counter()
    for y in Ytr:
        pres = [a for a in scored if y[a] > 0]
        for a in pres: single[a] += 1
        for i in range(len(pres)):
            for j in range(len(pres)):
                if i != j: pair[(pres[i], pres[j])] += 1
    return n, single, pair, Ytr


def lift(a, S, n, single, pair, k=1.0):
    """sum over b in S of log P(a|b) - log P(a), Laplace-smoothed."""
    if not S: return 0.0
    pa = (single[a] + k) / (n + 2 * k)
    total = 0.0
    for b in S:
        if b == a: continue
        pab = (pair[(a, b)] + k) / (single[b] + 2 * k)
        total += math.log(pab) - math.log(pa)
    return total


def decode(L, scored, S_sets, alpha, n, single, pair):
    N, A, C = L.shape
    P = np.zeros((N, A), dtype=int)
    for i in range(N):
        for a in range(A):
            m = L[i, a, 1:].max() - L[i, a, 0]
            if a in scored and alpha:
                m = m - alpha * lift(a, [b for b in S_sets[i] if b != a], n, single, pair)
            P[i, a] = (L[i, a, 1:].argmax() + 1) if m > 0 else 0
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["absa_hotel", "absa_restaurant"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    args = ap.parse_args(); device = torch.device(args.device); OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in args.tasks:
        ckpt = ROOT / "artifacts" / "e19" / task / "seed_42" / "best_checkpoint.pt"
        bundle, L, Y, gold, counts = validation_logits(task, ckpt, device)
        scored = scored_aspects(task, bundle); n, single, pair, Ytr = train_stats(bundle, scored)
        pairs_train = Counter()
        for y in Ytr:
            pres = [a for a in scored if y[a] > 0]
            for i in range(len(pres)):
                for j in range(i + 1, len(pres)): pairs_train[frozenset((pres[i], pres[j]))] += 1
        base = decode(L, scored, [[] for _ in range(len(L))], 0.0, n, single, pair)
        res = {"baseline": {"pair_f1": round(100 * float(tr.evaluate_predictions(task, bundle, gold, base.tolist(), split="validation")["selection_f1"]), 3), "groups": group_metrics(base, Y, scored, pairs_train)}}
        # sign check on baseline errors in rare-pair rows
        neg_lift_missed = tot_missed = pos_lift_spur = tot_spur = 0
        for i in range(len(L)):
            pres = [a for a in scored if Y[i, a] > 0]
            if len(pres) < 2: continue
            if min(pairs_train[frozenset((pres[x], pres[y]))] for x in range(len(pres)) for y in range(x + 1, len(pres))) > 5: continue
            for a in scored:
                lf = lift(a, [b for b in pres if b != a], n, single, pair)
                if Y[i, a] > 0 and base[i, a] == 0: tot_missed += 1; neg_lift_missed += lf < 0
                if Y[i, a] == 0 and base[i, a] > 0: tot_spur += 1; pos_lift_spur += lf > 0
        res["sign_check_rare_rows"] = {"missed_with_negative_lift": f"{neg_lift_missed}/{tot_missed}", "spurious_with_positive_lift": f"{pos_lift_spur}/{tot_spur}"}
        gold_sets = [[a for a in scored if Y[i, a] > 0] for i in range(len(L))]
        pred_sets = [[a for a in scored if base[i, a] > 0] for i in range(len(L))]
        for setting, S_sets in (("oracle_S", gold_sets), ("predicted_S", pred_sets)):
            for alpha in args.alphas:
                if alpha == 0: continue
                P = decode(L, scored, S_sets, alpha, n, single, pair)
                f1 = 100 * float(tr.evaluate_predictions(task, bundle, gold, P.tolist(), split="validation")["selection_f1"])
                g = group_metrics(P, Y, scored, pairs_train)
                res[f"{setting}|alpha={alpha}"] = {"pair_f1": round(f1, 3), "groups": g}
                print(task, setting, alpha, round(f1, 2), "rare missed", g["k>=2|rare_pair"]["missed_pct"], "rare spur/row", g["k>=2|rare_pair"]["spurious_per_row"], "freq missed", g["k>=2|freq_pair"]["missed_pct"], "mixed wrong", g["k>=2|mixed"]["wrong_pct"], flush=True)
        print(task, "baseline", res["baseline"]["pair_f1"], "rare missed", res["baseline"]["groups"]["k>=2|rare_pair"]["missed_pct"], "sign", res["sign_check_rare_rows"], flush=True)
        results[task] = res
        json.dump(results, open(OUT / "e19_debias_decode.json", "w"), indent=1)


if __name__ == "__main__":
    main()
