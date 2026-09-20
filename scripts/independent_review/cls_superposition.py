"""Can the trained head decode two aspects from one CLS vector? Inference-only on frozen checkpoints.
For pairs (A,B): CLS_A, CLS_B from single forwards; composite CLS from A+B forward.
Feed the head with: CLS_A (sanity), CLS_comp, (CLS_A+CLS_B)/2, CLS_A+CLS_B, and the LN-renormalised sum.
Report A-correct%, B-correct% for each input; and cosine geometry of CLS_comp vs CLS_A / CLS_B."""
import sys, json
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch, torch.nn.functional as F
from mechanism_battery import load_model, build_pairs, predictions, scored_aspects, MAX_LEN
from goibert import train as tr
device = torch.device('cuda:1')
OUT = ROOT/'artifacts'/'independent_review'/'battery'

@torch.inference_mode()
def cls_of(model, tok, texts, device):
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        with tr.make_autocast('bf16', device):
            h = model.encoder(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device), return_dict=True).last_hidden_state
        outs.append(h[:, 0].float())
    return torch.cat(outs, 0)

@torch.inference_mode()
def head_logits(model, cls):
    # replicate ClassificationHead on a pooled vector (eval: dropout off)
    head = model.head
    x = head.output(head.activation(head.dense(cls)))
    if model.num_aspects and len(set(model.aspect_class_counts)) == 1:
        x = x.reshape(x.shape[0], model.num_aspects, -1)
    return x.float().cpu()

res = {}
for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel', 'visfd'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle)
        singles, pairs = build_pairs(task, bundle, scored, tok, 200, 2026)
        A_t = [A['text'] for A, B in pairs]; B_t = [B['text'] for A, B in pairs]
        comp1 = [A['text'] + ' ' + B['text'] for A, B in pairs]
        comp3 = [A['text'] + ' ' + ' '.join([B['text']]*3) for A, B in pairs]
        cA, cB, c1, c3 = (cls_of(model, tok, t, device) for t in (A_t, B_t, comp1, comp3))
        ln = model.encoder.encoder.layer[-1].output.LayerNorm
        inputs = {'CLS_A': cA, 'CLS_B': cB, 'CLS_A+B': c1, 'CLS_A+3B': c3, 'avg(CLS_A,CLS_B)': (cA + cB) / 2, 'sum(CLS_A,CLS_B)': cA + cB,
                  'LN(sum)': ln(cA + cB).float()}
        out = {}
        for name, v in inputs.items():
            pr = predictions(head_logits(model, v.to(device)), counts)
            a_ok = sum(1 for (A, B), p in zip(pairs, pr) if p[A['aspect']] == A['pol']) / len(pairs)
            b_ok = sum(1 for (A, B), p in zip(pairs, pr) if p[B['aspect']] == B['pol']) / len(pairs)
            a_miss = sum(1 for (A, B), p in zip(pairs, pr) if p[A['aspect']] == 0) / len(pairs)
            out[name] = {'A_correct': round(100*a_ok, 1), 'A_missed': round(100*a_miss, 1), 'B_correct': round(100*b_ok, 1)}
        geo = {'cos(comp1,A)': float(F.cosine_similarity(c1, cA).mean()), 'cos(comp1,B)': float(F.cosine_similarity(c1, cB).mean()),
               'cos(comp1,avgAB)': float(F.cosine_similarity(c1, (cA+cB)/2).mean()), 'cos(comp3,B)': float(F.cosine_similarity(c3, cB).mean()),
               'cos(A,B)': float(F.cosine_similarity(cA, cB).mean()), 'norm_A': float(cA.norm(dim=1).mean()), 'norm_comp1': float(c1.norm(dim=1).mean())}
        res[f'{kind}__{task}'] = {'head_inputs': out, 'geometry': {k: round(v, 3) for k, v in geo.items()}}
        print('='*90); print(kind, task)
        for k, v in out.items(): print(f'   {k:18s} {v}')
        print('   geometry', res[f'{kind}__{task}']['geometry'])
        del model; torch.cuda.empty_cache()
json.dump(res, open(OUT/'cls_superposition.json', 'w'), indent=1)
