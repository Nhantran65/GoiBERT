"""Segment-and-read decoding ceiling on the frozen E8 student (inference only, validation only).

For each validation sentence, enumerate binary split points (after punctuation/connectors, capped) and run two
hard-masked fork CLS rows per split in one forward pass (fork_test operator).  Decode each split by aggregating
the two forks per aspect with the max presence margin (the slot aggregation rule).  Report:
  - baseline: unsplit CLS (must reproduce E8 validation F1);
  - oracle split: per row, the split (or no split) whose decoded prediction maximises tp - fp - fn against gold
    (uses gold -> a CEILING for any one-split-per-row selector, not a result);
  - naive selectors without gold: max mean |margin|, and 'always split at the best-scoring connector'.
This measures the headroom of segment-restricted reading with the existing head before any training.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "independent_review")); sys.path.insert(0, str(ROOT / "scripts" / "research_checks"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mechanism_battery import load_model, scored_aspects, MAX_LEN, sha256  # noqa: E402
from goibert import train as tr  # noqa: E402
from competitive_slots_probe_v2 import SlotEncoder, OUT  # noqa: E402
from competitive_slots_probe_v3_latent import candidate_splits, collate_latent  # noqa: E402


def randomize_masks(fm, examples, seed):
    """Control: replace each split's two contiguous segment masks by two complementary RANDOM token subsets of the same
    sizes (contiguity destroyed, selection-bias preserved).  fm: [B, m, N-1] bool."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    fm = fm.clone().cpu()
    for b, e in enumerate(examples):
        n = len(e["ids"]); eos = n - 1; content = torch.arange(0, eos - 1)   # columns of content tokens (excl </s>)
        for s, j in enumerate(e.get("splits", [])):
            n1 = j - 1
            perm = content[torch.randperm(len(content), generator=g)]
            m1 = torch.zeros(fm.shape[2], dtype=torch.bool); m1[perm[:n1]] = True; m1[eos - 1] = True
            m2 = torch.zeros(fm.shape[2], dtype=torch.bool); m2[perm[n1:]] = True; m2[eos - 1] = True
            fm[b, 1 + 2 * s] = m1; fm[b, 2 + 2 * s] = m2
    return fm


def decode_pair(F, counts):
    """F: [m, A, C] fork logits for one row; aggregate forks by max presence margin -> prediction [A]."""
    margin = F[..., 1:].max(-1).values - F[..., 0]                      # m,A
    k = margin.argmax(0)                                                # A
    L = F[k, torch.arange(F.shape[1])]                                  # A,C
    return L.argmax(-1)


def row_score(pred, gold, scored):
    tp = fp = fn = 0
    for a in scored:
        g = int(gold[a]); p = int(pred[a])
        if g > 0 and p == g: tp += 1
        elif p > 0 and g == 0: fp += 1
        elif g > 0 and p != g: fn += 1; fp += (p > 0)
    return tp - fp - fn


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", nargs="+", default=["absa_restaurant", "absa_hotel"])
    ap.add_argument("--max-splits", type=int, default=8)
    ap.add_argument("--random-masks", action="store_true", help="selection-bias control: random complementary subsets instead of contiguous segments")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)
    out = {}
    for task in args.tasks:
        model, tok, bundle, counts, ckpt = load_model("student_e8", task, device)
        scored = scored_aspects(task, bundle); A = len(counts); C = counts[0]
        va_split = tr.split_field(bundle, "validation")
        va_texts = list(tr.split_texts(va_split)); Yva = np.array([list(l) for l in tr.split_labels(va_split)])
        gold_va = tr.model_labels(bundle, "validation", va_split)
        se = SlotEncoder(model, 1, 0.0, device)
        examples = []
        for i, t in enumerate(va_texts):
            ids = tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
            examples.append({"ids": ids, "y": Yva[i], "splits": candidate_splits(ids, args.max_splits) if len(ids) >= 6 else []})
        preds = {"baseline": [], "oracle_split": [], "naive_confidence": [], "always_split_first_connector": [], "oracle_gain_rows": 0}
        n_with_split = 0; n_oracle_split_chosen = 0
        for i in range(0, len(examples), 32):
            batch = examples[i:i + 32]
            ids, am, fm = collate_latent(batch, tok.pad_token_id, device, args.max_splits)
            if args.random_masks: fm = randomize_masks(fm, batch, args.seed + i).to(device)
            lg, fk = se.forward(ids, am, fork_masks=fm)
            Fk = fk.float().view(len(batch), -1, A, C)
            for b, e in enumerate(batch):
                base_pred = Fk[b, 0].argmax(-1)
                preds["baseline"].append(base_pred.tolist())
                S = len(e["splits"])
                if S == 0:
                    for k in ("oracle_split", "naive_confidence", "always_split_first_connector"): preds[k].append(base_pred.tolist())
                    continue
                n_with_split += 1
                cands = [("none", base_pred, float(Fk[b, 0][..., 1:].max(-1).values.sub(Fk[b, 0][..., 0]).abs().mean()))]
                for s in range(S):
                    F2 = Fk[b, 1 + 2 * s:3 + 2 * s]
                    p = decode_pair(F2, counts)
                    conf = float((F2[..., 1:].max(-1).values - F2[..., 0]).abs().mean())
                    cands.append((s, p, conf))
                base_sc = row_score(base_pred, e["y"], scored)
                best = max(cands, key=lambda c: (row_score(c[1], e["y"], scored), c[0] == "none"))
                if best[0] != "none" and row_score(best[1], e["y"], scored) > base_sc: n_oracle_split_chosen += 1
                preds["oracle_split"].append(best[1].tolist())
                preds["naive_confidence"].append(max(cands, key=lambda c: c[2])[1].tolist())
                preds["always_split_first_connector"].append(cands[1][1].tolist())
        res = {}
        for k in ("baseline", "oracle_split", "naive_confidence", "always_split_first_connector"):
            P = preds[k]
            f1 = tr.evaluate_predictions(task, bundle, gold_va, P, split="validation")["selection_f1"]
            res[k] = round(100 * float(f1), 3)
        res["rows"] = len(examples); res["rows_with_candidate_splits"] = n_with_split; res["rows_where_oracle_split_beats_unsplit"] = n_oracle_split_chosen
        res["checkpoint"] = str(ckpt.relative_to(ROOT)); res["checkpoint_sha256"] = sha256(ckpt); res["max_splits"] = args.max_splits
        out[task] = res
        print(task, json.dumps(res), flush=True)
        del model; torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / ("segment_read_ceiling_randomctrl.json" if args.random_masks else "segment_read_ceiling.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
