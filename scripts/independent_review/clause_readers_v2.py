"""Training-free decoding with clause readers, two rules, thresholds calibrated on synthetic TRAIN composites only.
Rule P (polarity-only, no threshold): full reader decides presence; polarity of a present slot = argmax polarity of the
reader (full or clause) with the highest present-mass for that slot.
Rule R (recover misses): as P, plus accept a clause reader's present if its present-mass >= t_c and full absent-prob <= t_f;
(t_c, t_f) chosen on synthetic train composites (concatenations of train single-aspect rows, gold = union)."""
import sys, json, warnings, itertools, random
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch, numpy as np
from mechanism_battery import load_model, scored_aspects, MAX_LEN
from fork_test import MaskedCLS
from clause_readers import run_readers, metrics
from goibert import train as tr
device = torch.device('cuda:1'); OUT = ROOT/'artifacts'/'independent_review'/'battery'

def decode(readers, K, rule, t_c=2.0, t_f=0.0):
    out = [0] * K
    for a in range(K):
        slot = [r[a] for r in readers]; full = slot[0]
        fp = int(torch.argmax(full))
        masses = [float(1 - s[0]) for s in slot]
        kbest = int(np.argmax(masses)); best = slot[kbest]
        if fp > 0:
            out[a] = fp if rule == 'full' else int(torch.argmax(best[1:])) + 1
        elif rule == 'R' and kbest > 0 and masses[kbest] >= t_c and float(full[0]) <= t_f:
            out[a] = int(torch.argmax(best[1:])) + 1
    return out

GRID_C = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]; GRID_F = [0.5, 0.7, 0.9, 0.97, 1.0]
res = {}
for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel', 'visfd'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle); K = tr.task_num_aspects(bundle)
        mc = MaskedCLS(model)
        trs = tr.split_field(bundle, 'train'); tr_texts = list(tr.split_texts(trs)); Ytr = [list(l) for l in tr.model_labels(bundle, 'train', trs)]
        rng = random.Random(11)
        singles = [i for i in range(len(tr_texts)) if sum(1 for a in scored if Ytr[i][a] > 0) == 1 and len(tok(tr_texts[i], truncation=True, max_length=MAX_LEN)['input_ids']) <= 48]
        comp_t, comp_y = [], []
        while len(comp_t) < 1200:
            i, j = rng.sample(singles, 2)
            ai = [a for a in scored if Ytr[i][a] > 0][0]; aj = [a for a in scored if Ytr[j][a] > 0][0]
            if ai == aj: continue
            y = list(Ytr[i]); y[aj] = Ytr[j][aj]; comp_t.append(tr_texts[i] + ' ' + tr_texts[j]); comp_y.append(y)
        cp, _ = run_readers(model, tok, comp_t, mc, counts, 'clauses')
        base_c = metrics([decode(p, K, 'full') for p in cp], comp_y, scored)['pair_f1']
        P_c = metrics([decode(p, K, 'P') for p in cp], comp_y, scored)['pair_f1']
        best = ((2.0, 0.0), P_c)
        for tc, tf in itertools.product(GRID_C, GRID_F):
            f1 = metrics([decode(p, K, 'R', tc, tf) for p in cp], comp_y, scored)['pair_f1']
            if f1 > best[1]: best = ((tc, tf), f1)
        (tc, tf), R_c = best
        va = tr.split_field(bundle, 'validation'); va_t = list(tr.split_texts(va)); Yva = [list(l) for l in tr.model_labels(bundle, 'validation', va)]
        vp, _ = run_readers(model, tok, va_t, mc, counts, 'clauses')
        r = {'synthetic_train_composites': {'n': len(comp_t), 'full': base_c, 'ruleP': P_c, 'ruleR_best': R_c, 't_c': tc, 't_f': tf},
             'validation_full': metrics([decode(p, K, 'full') for p in vp], Yva, scored),
             'validation_ruleP': metrics([decode(p, K, 'P') for p in vp], Yva, scored),
             'validation_ruleR': metrics([decode(p, K, 'R', tc, tf) for p in vp], Yva, scored)}
        res[f'{kind}__{task}'] = r
        print('=' * 100); print(kind, task, 'synthetic composites:', r['synthetic_train_composites'])
        for k in ('validation_full', 'validation_ruleP', 'validation_ruleR'): print(f'   {k:18s} {r[k]}')
        mc.restore(); del model; torch.cuda.empty_cache()
json.dump(res, open(OUT / 'clause_readers_v2.json', 'w'), indent=1)
