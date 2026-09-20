"""Clause readers with a decision rule calibrated on a TRAIN subset (thresholds chosen on train only), evaluated once on validation.
Rule: slot prediction = full reader's argmax; if full reader says absent, accept the most-confident clause reader's 'present'
only if its present-mass >= t_clause AND the full reader's absent-prob <= t_full."""
import sys, json, warnings, itertools, random
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch, numpy as np
from mechanism_battery import load_model, scored_aspects
from fork_test import MaskedCLS
from clause_readers import run_readers, metrics
from goibert import train as tr
device = torch.device('cuda:1'); OUT = ROOT/'artifacts'/'independent_review'/'battery'

def decode_rule(readers, K, t_clause, t_full):
    out = [0] * K
    for a in range(K):
        slot = [r[a] for r in readers]
        full = slot[0]; fp = int(torch.argmax(full))
        if fp > 0: out[a] = fp; continue
        best, best_mass = None, -1.0
        for s in slot[1:]:
            m = float(1 - s[0])
            if m > best_mass: best_mass, best = m, s
        if best is not None and best_mass >= t_clause and float(full[0]) <= t_full:
            out[a] = int(torch.argmax(best[1:])) + 1
    return out

GRID_C = [0.6, 0.7, 0.8, 0.9, 0.95]; GRID_F = [0.5, 0.7, 0.9, 0.97, 1.0]
res = {}
for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel', 'visfd'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle); K = tr.task_num_aspects(bundle)
        mc = MaskedCLS(model)
        trs = tr.split_field(bundle, 'train'); rng = random.Random(3)
        idx = rng.sample(range(len(list(tr.split_texts(trs)))), 1000)
        tr_texts = [list(tr.split_texts(trs))[i] for i in idx]; tr_Y = [list(l) for l in tr.model_labels(bundle, 'train', trs)]; tr_Y = [tr_Y[i] for i in idx]
        tr_probs, _ = run_readers(model, tok, tr_texts, mc, counts, 'clauses')
        base_tr = metrics([decode_rule(p, K, 2.0, 0.0) for p in tr_probs], tr_Y, scored)['pair_f1']
        best = (None, -1)
        for tc, tf in itertools.product(GRID_C, GRID_F):
            f1 = metrics([decode_rule(p, K, tc, tf) for p in tr_probs], tr_Y, scored)['pair_f1']
            if f1 > best[1]: best = ((tc, tf), f1)
        (tc, tf), f1_tr = best
        va = tr.split_field(bundle, 'validation'); va_texts = list(tr.split_texts(va)); va_Y = [list(l) for l in tr.model_labels(bundle, 'validation', va)]
        va_probs, _ = run_readers(model, tok, va_texts, mc, counts, 'clauses')
        r = {'train_subset_baseline_f1': base_tr, 'train_subset_best_f1': f1_tr, 'chosen_t_clause': tc, 'chosen_t_full': tf,
             'validation_baseline': metrics([decode_rule(p, K, 2.0, 0.0) for p in va_probs], va_Y, scored),
             'validation_calibrated': metrics([decode_rule(p, K, tc, tf) for p in va_probs], va_Y, scored)}
        res[f'{kind}__{task}'] = r
        print('=' * 100); print(kind, task, f"train: base {base_tr} -> best {f1_tr} at t_clause={tc}, t_full={tf}")
        print('   val baseline  ', r['validation_baseline']); print('   val calibrated', r['validation_calibrated'])
        mc.restore(); del model; torch.cuda.empty_cache()
json.dump(res, open(OUT / 'clause_readers_calibrated.json', 'w'), indent=1)
