"""Fork test (inference + linear probes only).
(i) Linear probes trained on CLS of synthetic composites built from TRAIN single-aspect rows -> is A linearly recoverable
    from CLS(A+B) on validation composites and on real multi-aspect validation rows?
(ii) Masked-CLS counterfactual: CLS row attends only to part A (B masked out for the CLS row in ALL layers; token rows
    unchanged) -> does the trained head recover A?  This is the 'per-aspect CLS on a shared token stack' upper bound."""
import sys, json, warnings, random, types, math, collections
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mechanism_battery import load_model, build_pairs, predictions, scored_aspects, MAX_LEN
from goibert import train as tr
device = torch.device('cuda:1'); OUT = ROOT/'artifacts'/'independent_review'/'battery'

class MaskedCLS:
    def __init__(self, model):
        self.model = model; self.on = False; self.mask2d = None; self.keep = None; self._orig = {}
        for idx, layer in enumerate(model.encoder.encoder.layer):
            sa = layer.attention.self; self._orig[idx] = sa.forward; sa.forward = types.MethodType(self._fwd(), sa)
    def restore(self):
        for idx, f in self._orig.items(): self.model.encoder.encoder.layer[idx].attention.self.forward = f
    def _fwd(self):
        st = self
        def forward(sa, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            B, N, _ = hidden_states.shape; H, d = sa.num_attention_heads, sa.attention_head_size
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                x = hidden_states.float()
                q = sa.query(x).view(B, N, H, d).transpose(1, 2); k = sa.key(x).view(B, N, H, d).transpose(1, 2); v = sa.value(x).view(B, N, H, d).transpose(1, 2)
                valid = st.mask2d.to(x.device).bool()
                s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
                s = s.masked_fill(~valid[:, None, None, :], float('-inf'))
                if st.on:
                    keep = st.keep.to(x.device).bool()            # B,N : keys the CLS row may attend to
                    s_cls = s[:, :, 0, :].masked_fill(~keep[:, None, :], float('-inf'))
                    s = s.clone(); s[:, :, 0, :] = s_cls
                A = torch.softmax(s, dim=-1)
                ctx = torch.matmul(A, v).transpose(1, 2).reshape(B, N, H * d)
            return (ctx.to(hidden_states.dtype),)
        return forward

@torch.inference_mode()
def cls_of(model, tok, texts, mc: MaskedCLS, keep_len=None):
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        mc.mask2d = enc['attention_mask']
        if keep_len is not None:
            K = torch.zeros_like(enc['input_ids'], dtype=torch.bool)
            for b in range(enc['input_ids'].shape[0]):
                K[b, :keep_len[i + b]] = True                  # <s> + A tokens
                K[b, int(enc['attention_mask'][b].sum()) - 1] = True   # keep </s> too
            mc.keep = K
        with tr.make_autocast('bf16', device):
            h = model.encoder(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device), return_dict=True).last_hidden_state
        outs.append(h[:, 0].float().cpu())
    return torch.cat(outs, 0)

@torch.inference_mode()
def head_pred(model, cls, counts):
    head = model.head; x = head.output(head.activation(head.dense(cls.to(device))))
    if model.num_aspects and len(set(model.aspect_class_counts)) == 1: x = x.reshape(x.shape[0], model.num_aspects, -1)
    return np.array(predictions(x.float().cpu(), counts))

def fit_probes(X, Y, scored):
    probes = {}
    for a in scored:
        pres = (Y[:, a] > 0).astype(int)
        if pres.sum() < 5 or (1 - pres).sum() < 5: continue
        pc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced')).fit(X, pres)
        idx = np.where(Y[:, a] > 0)[0]; qc = None
        if len(set(Y[idx, a])) >= 2: qc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X[idx], Y[idx, a])
        probes[a] = (pc, qc)
    return probes
def probe_pred(probes, X, K):
    P = np.zeros((X.shape[0], K), dtype=int)
    for a, (pc, qc) in probes.items():
        pres = pc.predict(X); P[:, a] = np.where(pres == 1, qc.predict(X), 0) if qc is not None else pres
    return P
def pair_metrics(P, Y, scored):
    tp = pred = gold = 0; mg = mm = 0
    for p, g in zip(P, Y):
        present = [i for i in scored if g[i] > 0]
        for i in scored:
            gold += g[i] > 0; pred += p[i] > 0; tp += (g[i] > 0 and p[i] == g[i])
        if len(present) >= 2:
            for i in present: mg += 1; mm += (p[i] == 0)
    return {'pair_f1': round(100 * 2 * tp / (pred + gold), 2), 'multi_row_missed_pct': round(100 * mm / max(mg, 1), 1)}

def _main():
  res = {}
  for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle); K = tr.task_num_aspects(bundle)
        mc = MaskedCLS(model); mc.on = False
        tr_split = tr.split_field(bundle, 'train'); va_split = tr.split_field(bundle, 'validation')
        tr_texts = list(tr.split_texts(tr_split)); Ytr = np.array([list(l) for l in tr.split_labels(tr_split)])
        va_texts = list(tr.split_texts(va_split)); Yva = np.array([list(l) for l in tr.split_labels(va_split)])
        Xtr = cls_of(model, tok, tr_texts, mc).numpy(); Xva = cls_of(model, tok, va_texts, mc).numpy()
        # synthetic composites from TRAIN single-aspect rows (disjoint aspects), labels = union
        rng = random.Random(7)
        singles_tr = [i for i in range(len(tr_texts)) if sum(1 for a in scored if Ytr[i, a] > 0) == 1 and len(tok(tr_texts[i], truncation=True, max_length=MAX_LEN)['input_ids']) <= 48]
        comp_texts, comp_Y = [], []
        for _ in range(3000):
            i, j = rng.sample(singles_tr, 2)
            ai = [a for a in scored if Ytr[i, a] > 0][0]; aj = [a for a in scored if Ytr[j, a] > 0][0]
            if ai == aj: continue
            comp_texts.append(tr_texts[i] + ' ' + tr_texts[j]); y = Ytr[i].copy(); y[aj] = Ytr[j, aj]; comp_Y.append(y)
        comp_Y = np.array(comp_Y); Xcomp = cls_of(model, tok, comp_texts, mc).numpy()
        probes_single = fit_probes(Xtr, Ytr, scored)
        probes_mixed = fit_probes(np.concatenate([Xtr, Xcomp]), np.concatenate([Ytr, comp_Y]), scored)
        # validation composites from battery pairs
        singles, pairs = build_pairs(task, bundle, scored, tok, 200, 2026)
        va_comp = [A['text'] + ' ' + B['text'] for A, B in pairs]
        Xvc = cls_of(model, tok, va_comp, mc).numpy()
        def ab(P):
            return {'A_correct': round(100 * float(np.mean([P[i, A['aspect']] == A['pol'] for i, (A, B) in enumerate(pairs)])), 1),
                    'A_present': round(100 * float(np.mean([P[i, A['aspect']] > 0 for i, (A, B) in enumerate(pairs)])), 1),
                    'B_present': round(100 * float(np.mean([P[i, B['aspect']] > 0 for i, (A, B) in enumerate(pairs)])), 1)}
        r = {}
        r['(i) probe_single | val composites'] = ab(probe_pred(probes_single, Xvc, K))
        r['(i) probe_mixed  | val composites'] = ab(probe_pred(probes_mixed, Xvc, K))
        r['(i) head         | val composites'] = ab(head_pred(model, torch.tensor(Xvc), counts))
        r['(i) probe_single | real validation'] = pair_metrics(probe_pred(probes_single, Xva, K), Yva, scored)
        r['(i) probe_mixed  | real validation'] = pair_metrics(probe_pred(probes_mixed, Xva, K), Yva, scored)
        r['(i) head         | real validation'] = pair_metrics(head_pred(model, torch.tensor(Xva), counts), Yva, scored)
        # (ii) masked-CLS counterfactual on validation composites
        a_len = [len(tok(A['text'], truncation=True, max_length=MAX_LEN)['input_ids']) - 1 for A, B in pairs]
        mc.on = True; Xmask = cls_of(model, tok, va_comp, mc, keep_len=a_len); mc.on = False
        XA = cls_of(model, tok, [A['text'] for A, B in pairs], mc)
        r['(ii) head | CLS sees only A-part'] = ab(head_pred(model, Xmask, counts))
        r['(ii) head | A alone'] = ab(head_pred(model, XA, counts))
        r['(ii) cos(maskedCLS, CLS_A)'] = round(float(torch.nn.functional.cosine_similarity(Xmask, XA).mean()), 3)
        res[f'{kind}__{task}'] = r
        print('=' * 100); print(kind, task, f'(train singles={len(singles_tr)}, synthetic train composites={len(comp_texts)})')
        for k, v in r.items(): print(f'   {k:36s} {v}')
        mc.restore(); del model; torch.cuda.empty_cache()
  json.dump(res, open(OUT / 'fork_test.json', 'w'), indent=1)
if __name__ == '__main__':
    _main()
