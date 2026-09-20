"""Oracle partition-normalised CLS aggregation (inference only). In the last layers, the CLS row's attention is replaced by
0.5*softmax over part A + 0.5*softmax over part B ('avg') or softmax_A + softmax_B ('sum'); token rows unchanged.
Single-aspect inputs are unaffected (one part). Evaluate A/B recovery on composites with (i) the trained head and
(ii) linear probes trained on train-split CLS of the unmodified encoder."""
import sys, json, types, math, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mechanism_battery import load_model, build_pairs, predictions, scored_aspects, MAX_LEN
from goibert import train as tr
device = torch.device('cuda:1')
OUT = ROOT/'artifacts'/'independent_review'/'battery'
LAYERS = {'student_e8': (7, 8, 9), 'bami_e14': (10, 11)}

class PartitionCLS:
    """Override self-attention in given layers: CLS row uses partition-normalised attention; other rows standard."""
    def __init__(self, model, layers):
        self.model = model; self.layers = layers; self.mode = 'off'; self.mask2d = None; self.part = None  # part: B,N long (0 = A-part, 1 = B-part, -1 pad)
        self._orig = {}
        for idx in layers:
            sa = model.encoder.encoder.layer[idx].attention.self
            self._orig[idx] = sa.forward; sa.forward = types.MethodType(self._fwd(), sa)
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
                A = torch.softmax(s, dim=-1)
                if st.mode != 'off':
                    part = st.part.to(x.device)
                    s_cls = s[:, :, 0, :]                                   # B,H,N
                    mA = (part == 0)[:, None, :]; mB = (part == 1)[:, None, :]
                    pA = torch.softmax(s_cls.masked_fill(~mA, float('-inf')), dim=-1)
                    pB = torch.softmax(s_cls.masked_fill(~mB, float('-inf')), dim=-1)
                    hasB = mB.any(-1, keepdim=True).float()
                    pB = torch.nan_to_num(pB, nan=0.0)
                    if st.mode == 'avg':
                        a_cls = torch.where(hasB.bool(), 0.5 * pA + 0.5 * pB, pA)
                    else:  # sum
                        a_cls = pA + pB
                    A = A.clone(); A[:, :, 0, :] = a_cls
                ctx = torch.matmul(A, v).transpose(1, 2).reshape(B, N, H * d)
            return (ctx.to(hidden_states.dtype),)
        return forward

@torch.inference_mode()
def cls_of(model, tok, texts, device, pc: PartitionCLS, parts=None):
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
        pc.mask2d = enc['attention_mask']
        if parts is not None:
            P = torch.full_like(enc['input_ids'], -1)
            for b in range(enc['input_ids'].shape[0]):
                la = parts[i + b]; n = int(enc['attention_mask'][b].sum())
                P[b, :la] = 0; P[b, la:n] = 1
            pc.part = P
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
        pc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight='balanced')).fit(X, pres)
        idx = np.where(Y[:, a] > 0)[0]; qc = None
        if len(set(Y[idx, a])) >= 2: qc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)).fit(X[idx], Y[idx, a])
        probes[a] = (pc, qc)
    return probes

def probe_pred(probes, X):
    K = max(probes) + 1; P = np.zeros((X.shape[0], K), dtype=int)
    for a, (pc, qc) in probes.items():
        pres = pc.predict(X)
        P[:, a] = np.where(pres == 1, qc.predict(X), 0) if qc is not None else pres
    return P

res = {}
for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel', 'visfd'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle)
        pc = PartitionCLS(model, LAYERS[kind]); pc.mode = 'off'
        tr_split = tr.split_field(bundle, 'train')
        Xtr = cls_of(model, tok, list(tr.split_texts(tr_split)), device, pc).numpy(); Ytr = np.array([list(l) for l in tr.split_labels(tr_split)])
        probes = fit_probes(Xtr, Ytr, scored)
        singles, pairs = build_pairs(task, bundle, scored, tok, 200, 2026)
        a_len = [len(tok(A['text'], truncation=True, max_length=MAX_LEN)['input_ids']) - 1 for A, B in pairs]   # A-part = <s> + A tokens (excluding </s>)
        comps = {'A+B': [A['text'] + ' ' + B['text'] for A, B in pairs], 'A+3B': [A['text'] + ' ' + ' '.join([B['text']] * 3) for A, B in pairs]}
        r = {}
        for name, texts in comps.items():
            for mode in ('off', 'avg', 'sum'):
                pc.mode = mode
                X = cls_of(model, tok, texts, device, pc, parts=a_len)
                for lab, P in (('head', head_pred(model, X, counts)), ('probe', probe_pred(probes, X.numpy()))):
                    a_ok = float(np.mean([P[i, A['aspect']] == A['pol'] for i, (A, B) in enumerate(pairs)])); a_pres = float(np.mean([P[i, A['aspect']] > 0 for i, (A, B) in enumerate(pairs)]))
                    b_ok = float(np.mean([P[i, B['aspect']] == B['pol'] for i, (A, B) in enumerate(pairs)])); b_pres = float(np.mean([P[i, B['aspect']] > 0 for i, (A, B) in enumerate(pairs)]))
                    n_spur = float(np.mean([(P[i] > 0).sum() - int(P[i, A['aspect']] > 0) - int(P[i, B['aspect']] > 0) for i, (A, B) in enumerate(pairs)]))
                    r[f'{name}|{mode}|{lab}'] = {'A_correct': round(100*a_ok, 1), 'A_present': round(100*a_pres, 1), 'B_correct': round(100*b_ok, 1), 'B_present': round(100*b_pres, 1), 'spurious_per_row': round(n_spur, 2)}
        pc.mode = 'off'
        XA = cls_of(model, tok, [A['text'] for A, B in pairs], device, pc)
        for lab, P in (('head', head_pred(model, XA, counts)), ('probe', probe_pred(probes, XA.numpy()))):
            r[f'A_alone|{lab}'] = {'A_correct': round(100*float(np.mean([P[i, A['aspect']] == A['pol'] for i, (A, B) in enumerate(pairs)])), 1), 'A_present': round(100*float(np.mean([P[i, A['aspect']] > 0 for i, (A, B) in enumerate(pairs)])), 1)}
        res[f'{kind}__{task}'] = r
        print('=' * 100); print(kind, task, 'layers', LAYERS[kind])
        for k, v in r.items(): print(f'   {k:20s} {v}')
        pc.restore(); del model; torch.cuda.empty_cache()
json.dump(res, open(OUT / 'partition_oracle.json', 'w'), indent=1)
