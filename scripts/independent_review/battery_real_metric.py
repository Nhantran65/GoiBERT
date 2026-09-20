"""Inference-only: apply C1 (soft-multiplicity bias) at rho in {0,0.5,1,2} on frozen checkpoints and
measure (a) validation F1, (b) the real failure-mode metric: minority-polarity slots in mixed rows."""
import sys, json, collections
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch
from mechanism_battery import load_model, Battery, forward_texts, predictions, scored_aspects, MODELS
from goibert import train as tr
device = torch.device('cuda:1')
out = {}
for kind in ('student_e8', 'bami_e14'):
    for task in ('visfd', 'absa_restaurant', 'absa_hotel'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle)
        split = tr.split_field(bundle, 'validation'); texts = list(tr.split_texts(split)); gold = list(tr.split_labels(split))
        gold_model = tr.model_labels(bundle, 'validation', split)
        bat = Battery(model, MODELS[kind]['layers'], list(tok.all_special_ids))
        res = {}
        for name, rho, tau, perm in (('rho0', 0.0, 8.0, False), ('rho0.5', 0.5, 8.0, False), ('rho1', 1.0, 8.0, False), ('rho2', 2.0, 8.0, False), ('rho1_tau4', 1.0, 4.0, False), ('perm_rho1', 1.0, 8.0, True)):
            bat.rho, bat.tau, bat.perm = rho, tau, perm; bat.gen.manual_seed(777)
            lg = forward_texts(model, tok, texts, device, bat)
            pred = predictions(lg, counts)
            f1 = tr.evaluate_predictions(task, bundle, gold_model, pred, split='validation')['selection_f1']
            # minority-polarity metric on mixed rows (gold codes for scored slots identical between raw and model labels)
            n = wrong = tomaj = missed = 0
            for g, p in zip(gold, pred):
                present = [i for i in scored if g[i] > 0]
                if len(present) < 2: continue
                cnt = collections.Counter(g[i] for i in present)
                if len(cnt) < 2: continue
                maj, mn = cnt.most_common(1)[0]
                if sum(1 for c in cnt.values() if c == mn) > 1: continue
                for i in present:
                    if g[i] == maj: continue
                    n += 1
                    if p[i] == 0: missed += 1
                    elif p[i] != g[i]:
                        wrong += 1
                        if p[i] == maj: tomaj += 1
            res[name] = {'f1': round(100*f1, 3), 'minority_n': n, 'minority_wrongpol_pct': round(100*wrong/n, 1), 'to_majority_share_pct': round(100*tomaj/max(wrong,1), 1), 'minority_missed_pct': round(100*missed/n, 1)}
            print(kind, task, name, res[name], flush=True)
        out[f'{kind}__{task}'] = res
        bat.restore(); del model; torch.cuda.empty_cache()
json.dump(out, open(ROOT/'artifacts'/'independent_review'/'battery'/'real_metric_inference.json', 'w'), indent=1)
