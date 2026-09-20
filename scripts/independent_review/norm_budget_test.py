"""Is presence detection magnitude-thresholded? (inference only)
(i) scale CLS_A (single-aspect) by lambda<1 -> does A get 'missed'?
(ii) scale CLS(A+B) by lambda>1 -> does A/B presence recover?  (head and a linear probe trained on singles)
Also: per-aspect 'presence direction' projection: component of CLS along the probe weight vector for A, alone vs in A+B."""
import sys, json, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from mechanism_battery import load_model, build_pairs, predictions, scored_aspects, MAX_LEN
from goibert import train as tr
device = torch.device('cuda:1')
OUT = ROOT/'artifacts'/'independent_review'/'battery'

@torch.inference_mode()
def cls_of(model, tok, texts):
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        with tr.make_autocast('bf16', device):
            h = model.encoder(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device), return_dict=True).last_hidden_state
        outs.append(h[:, 0].float().cpu())
    return torch.cat(outs, 0)
@torch.inference_mode()
def head_pred(model, cls, counts):
    head = model.head; x = head.output(head.activation(head.dense(cls.to(device))))
    if model.num_aspects and len(set(model.aspect_class_counts)) == 1: x = x.reshape(x.shape[0], model.num_aspects, -1)
    return np.array(predictions(x.float().cpu(), counts))

res = {}
for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle)
        tr_split = tr.split_field(bundle, 'train')
        Xtr = cls_of(model, tok, list(tr.split_texts(tr_split))).numpy(); Ytr = np.array([list(l) for l in tr.split_labels(tr_split)])
        # presence probes WITHOUT standardisation (so that scaling x is a pure logit scaling), no class weighting
        pres_probe = {a: LogisticRegression(max_iter=5000, C=0.5).fit(Xtr, (Ytr[:, a] > 0).astype(int)) for a in scored}
        singles, pairs = build_pairs(task, bundle, scored, tok, 200, 2026)
        XA = cls_of(model, tok, [A['text'] for A, B in pairs]); XB = cls_of(model, tok, [B['text'] for A, B in pairs])
        XAB = cls_of(model, tok, [A['text'] + ' ' + B['text'] for A, B in pairs])
        r = {}
        for name, X in (('CLS_A', XA), ('CLS_A+B', XAB)):
            for lam in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0):
                Ph = head_pred(model, X * lam, counts)
                a_pres = float(np.mean([Ph[i, A['aspect']] > 0 for i, (A, B) in enumerate(pairs)])); a_ok = float(np.mean([Ph[i, A['aspect']] == A['pol'] for i, (A, B) in enumerate(pairs)]))
                b_pres = float(np.mean([Ph[i, B['aspect']] > 0 for i, (A, B) in enumerate(pairs)])) if name != 'CLS_A' else float('nan')
                spur = float(np.mean([(Ph[i] > 0).sum() - int(Ph[i, A['aspect']] > 0) - (int(Ph[i, B['aspect']] > 0) if name != 'CLS_A' else 0) for i, (A, B) in enumerate(pairs)]))
                # probe presence for A with scaled logits
                pa = float(np.mean([pres_probe[A['aspect']].predict((X[i:i+1] * lam).numpy())[0] == 1 for i, (A, B) in enumerate(pairs)]))
                r[f'{name}|lam={lam}'] = {'head_A_present': round(100*a_pres, 1), 'head_A_correct': round(100*a_ok, 1), 'head_B_present': round(100*b_pres, 1) if name != 'CLS_A' else None, 'head_spurious/row': round(spur, 2), 'probe_A_present': round(100*pa, 1)}
        # projection of CLS on A's presence direction (probe weight, unit norm): alone vs composite
        projA_alone = []; projA_comp = []; projB_comp = []; projB_alone = []
        for i, (A, B) in enumerate(pairs):
            wA = pres_probe[A['aspect']].coef_[0]; wA = wA / np.linalg.norm(wA); wB = pres_probe[B['aspect']].coef_[0]; wB = wB / np.linalg.norm(wB)
            projA_alone.append(float(XA[i].numpy() @ wA)); projA_comp.append(float(XAB[i].numpy() @ wA)); projB_alone.append(float(XB[i].numpy() @ wB)); projB_comp.append(float(XAB[i].numpy() @ wB))
        r['projection'] = {'A_dir: alone': round(float(np.mean(projA_alone)), 3), 'A_dir: in A+B': round(float(np.mean(projA_comp)), 3), 'B_dir: alone': round(float(np.mean(projB_alone)), 3), 'B_dir: in A+B': round(float(np.mean(projB_comp)), 3),
                           'ratio A': round(float(np.mean(projA_comp) / np.mean(projA_alone)), 3), 'ratio B': round(float(np.mean(projB_comp) / np.mean(projB_alone)), 3), 'CLS norm alone/comp': (round(float(XA.norm(dim=1).mean()), 2), round(float(XAB.norm(dim=1).mean()), 2))}
        res[f'{kind}__{task}'] = r
        print('=' * 100); print(kind, task)
        for k, v in r.items(): print(f'   {k:18s} {v}')
        del model; torch.cuda.empty_cache()
json.dump(res, open(OUT / 'norm_budget_test.json', 'w'), indent=1)
