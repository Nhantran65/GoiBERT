"""Is A's evidence still linearly present in CLS(A+B)? Train per-aspect linear probes (presence + polarity) on CLS of
TRAIN rows (frozen encoder), evaluate on (i) validation single-aspect rows, (ii) composites A+B / A+3B from the
battery pairs, (iii) real validation multi-aspect rows. Compare with the trained nonlinear head on the same inputs."""
import sys, json, collections
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mechanism_battery import load_model, build_pairs, predictions, scored_aspects, MAX_LEN
from goibert import train as tr

@torch.inference_mode()
def cls_of(model, tok, texts, device):
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        with tr.make_autocast('bf16', device):
            h = model.encoder(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device), return_dict=True).last_hidden_state
        outs.append(h[:, 0].float().cpu())
    return torch.cat(outs, 0)

@torch.inference_mode()
def head_logits(model, cls):
    head = model.head
    x = head.output(head.activation(head.dense(cls)))
    if model.num_aspects and len(set(model.aspect_class_counts)) == 1:
        x = x.reshape(x.shape[0], model.num_aspects, -1)
    return x.float().cpu()
device = torch.device('cuda:1')
OUT = ROOT/'artifacts'/'independent_review'/'battery'

def probe_predict(models, X):
    """models[a] = (presence clf, polarity clf or None). returns list of per-aspect codes."""
    K = max(models) + 1; P = np.zeros((X.shape[0], K), dtype=int)
    for a, (pc, qc) in models.items():
        pres = pc.predict(X)
        if qc is not None:
            pol = qc.predict(X)
            P[:, a] = np.where(pres == 1, pol, 0)
        else:
            P[:, a] = pres
    return P

res = {}
for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel', 'visfd'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle)
        tr_split = tr.split_field(bundle, 'train'); va_split = tr.split_field(bundle, 'validation')
        Xtr = cls_of(model, tok, list(tr.split_texts(tr_split)), device).numpy(); Ytr = np.array([list(l) for l in tr.split_labels(tr_split)])
        Xva = cls_of(model, tok, list(tr.split_texts(va_split)), device).numpy(); Yva = np.array([list(l) for l in tr.split_labels(va_split)])
        # linear probes per scored aspect: presence (gold>0) ; polarity among present (1/2/3)
        probes = {}
        for a in scored:
            pres = (Ytr[:, a] > 0).astype(int)
            pc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced')).fit(Xtr, pres)
            idx = np.where(Ytr[:, a] > 0)[0]
            qc = None
            if len(set(Ytr[idx, a])) >= 2:
                qc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(Xtr[idx], Ytr[idx, a])
            probes[a] = (pc, qc)
        # (i) validation: probe vs head, overall pair F1 and minority metric
        Pva = probe_predict(probes, Xva)
        Hva = np.array(predictions(head_logits(model, torch.tensor(Xva).to(device)), counts))
        def metrics(P, Y):
            tp = pred = gold = 0; n = wrong = tomaj = missed = 0; multi_missed = multi_gold = 0
            for p, g in zip(P, Y):
                present = [i for i in scored if g[i] > 0]
                for i in scored:
                    if g[i] > 0: gold += 1
                    if p[i] > 0: pred += 1
                    if g[i] > 0 and p[i] == g[i]: tp += 1
                if len(present) >= 2:
                    for i in present:
                        multi_gold += 1
                        if p[i] == 0: multi_missed += 1
                    cnt = collections.Counter(g[i] for i in present)
                    if len(cnt) >= 2:
                        maj, mn = cnt.most_common(1)[0]
                        if sum(1 for c in cnt.values() if c == mn) == 1:
                            for i in present:
                                if g[i] == maj: continue
                                n += 1
                                if p[i] == 0: missed += 1
                                elif p[i] != g[i]:
                                    wrong += 1; tomaj += int(p[i] == maj)
            return {'pair_f1': round(100*2*tp/(pred+gold), 2), 'multi_row_missed_pct': round(100*multi_missed/max(multi_gold,1), 1),
                    'minority_wrongpol_pct': round(100*wrong/max(n,1), 1), 'minority_to_majority_share': round(100*tomaj/max(wrong,1), 1), 'minority_missed_pct': round(100*missed/max(n,1), 1)}
        r = {'validation_head': metrics(Hva, Yva), 'validation_linear_probe': metrics(Pva, Yva)}
        # (ii) composites
        singles, pairs = build_pairs(task, bundle, scored, tok, 200, 2026)
        comps = {'A_alone': [A['text'] for A, B in pairs], 'A+B': [A['text']+' '+B['text'] for A, B in pairs], 'A+3B': [A['text']+' '+' '.join([B['text']]*3) for A, B in pairs]}
        for name, texts in comps.items():
            X = cls_of(model, tok, texts, device)
            Pp = probe_predict(probes, X.numpy()); Ph = np.array(predictions(head_logits(model, X.to(device)), counts))
            for lab, P in (('probe', Pp), ('head', Ph)):
                a_ok = np.mean([P[i, A['aspect']] == A['pol'] for i, (A, B) in enumerate(pairs)]); a_miss = np.mean([P[i, A['aspect']] == 0 for i, (A, B) in enumerate(pairs)])
                b_ok = np.mean([P[i, B['aspect']] == B['pol'] for i, (A, B) in enumerate(pairs)]) if name != 'A_alone' else float('nan')
                a_pres = np.mean([P[i, A['aspect']] > 0 for i, (A, B) in enumerate(pairs)])
                r[f'{name}_{lab}'] = {'A_correct': round(float(100*a_ok),1), 'A_present': round(float(100*a_pres),1), 'A_missed': round(float(100*a_miss),1), 'B_correct': round(float(100*b_ok),1) if name != 'A_alone' else None}
        res[f'{kind}__{task}'] = r
        print('='*100); print(kind, task)
        for k, v in r.items(): print(f'   {k:26s} {v}')
        del model; torch.cuda.empty_cache()
json.dump(res, open(OUT/'cls_linear_probe.json', 'w'), indent=1)
