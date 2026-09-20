"""Training-free clause-masked multi-reader CLS on real validation (inference only).
Readers: full sentence + one per clause (clauses = punctuation / discourse-marker segments). Each reader = the same
forward pass but with the CLS row restricted to the clause's tokens (plus <s>, </s>) in every layer; token rows unchanged.
Decoding: per aspect slot, choose the reader with the highest 'present' probability mass; if no reader predicts present -> absent.
Compare pair F1 / precision / recall / multi-row miss with the single-CLS baseline. Validation only; no test."""
import sys, json, warnings, types, math, re, collections
warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch, numpy as np
from mechanism_battery import load_model, scored_aspects, MAX_LEN
from fork_test import MaskedCLS
from goibert import train as tr
device = torch.device('cuda:1'); OUT = ROOT/'artifacts'/'independent_review'/'battery'
MARKERS = [' nhưng ', ' còn ', ' tuy nhiên ', ' mà ', ' và ', ' nhưg ', ' nhg ', ' bù lại ', ' được cái ', ' chỉ có ', ' chỉ mỗi ', ' mỗi tội ', ' ngoại trừ ', ' trừ ', ' tuy ', ' song ']
SPLIT_RE = re.compile(r'([.,;:!?\n]+|' + '|'.join(re.escape(m) for m in MARKERS) + ')', re.IGNORECASE)

def clause_spans(text, min_chars=8):
    """Return list of (start,end) char spans of clauses; short pieces merged into the previous one."""
    pieces = []; pos = 0
    for m in SPLIT_RE.finditer(text):
        seg = (pos, m.start()); pos = m.end()
        if seg[1] > seg[0]: pieces.append(seg)
    if pos < len(text): pieces.append((pos, len(text)))
    merged = []
    for s, e in pieces:
        if text[s:e].strip() == '': continue
        if merged and (e - s) < min_chars: merged[-1] = (merged[-1][0], e)
        else: merged.append((s, e))
    return merged if merged else [(0, len(text))]

@torch.inference_mode()
def run_readers(model, tok, texts, mc: MaskedCLS, counts, mode):
    """mode: 'baseline' (full only) or 'clauses' (full + clause readers). Returns per-row list of per-reader slot-probabilities."""
    all_probs = []; n_readers = []
    for i in range(0, len(texts), 16):
        chunk = texts[i:i+16]
        enc = tok(chunk, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt', return_offsets_mapping=True)
        offs = enc.pop('offset_mapping')
        # build reader list: (row, keep mask)
        rows, keeps = [], []
        for b, t in enumerate(chunk):
            n = int(enc['attention_mask'][b].sum())
            full = torch.zeros(enc['input_ids'].shape[1], dtype=torch.bool); full[:n] = True
            rows.append(b); keeps.append(full)
            if mode == 'clauses':
                spans = clause_spans(t)
                if len(spans) >= 2:
                    for (s, e) in spans:
                        k = torch.zeros_like(full); k[0] = True; k[n - 1] = True
                        for ti in range(1, n - 1):
                            os_, oe = offs[b, ti].tolist()
                            if oe > s and os_ < e: k[ti] = True
                        if int(k.sum()) >= 4: rows.append(b); keeps.append(k)
        # forward all readers in sub-batches
        ids = enc['input_ids'][rows]; am = enc['attention_mask'][rows]; keep = torch.stack(keeps)
        probs = []
        for j in range(0, len(rows), 32):
            mc.mask2d = am[j:j+32]; mc.keep = keep[j:j+32]; mc.on = True
            with tr.make_autocast('bf16', device):
                lg = model(ids[j:j+32].to(device), am[j:j+32].to(device)).float().cpu()
            mc.on = False
            probs.append(lg)
        lg = torch.cat(probs, 0)
        # slot probabilities
        if lg.ndim == 3: P = torch.softmax(lg, -1)   # B,K,4
        else:
            pieces, off = [], 0
            for w in counts:
                pieces.append(torch.softmax(lg[:, off:off+w], -1)); off += w
            P = pieces
        per_row = collections.defaultdict(list)
        for r_idx, b in enumerate(rows):
            per_row[b].append(P[r_idx] if lg.ndim == 3 else [p[r_idx] for p in pieces])
        for b in range(len(chunk)):
            all_probs.append(per_row[b]); n_readers.append(len(per_row[b]))
    return all_probs, n_readers

def decode(readers, counts, K, rule):
    """readers: list of per-reader slot prob tensors. rule: 'full' (first reader only) | 'maxpresent' | 'full_or_confident'."""
    out = [0] * K
    for a in range(K):
        if isinstance(readers[0], list):
            slot = [r[a] for r in readers]          # variable widths (visfd)
        else:
            slot = [r[a] for r in readers]
        if rule == 'full':
            out[a] = int(torch.argmax(slot[0])); continue
        best = None; best_mass = -1
        for s in slot:
            mass = float(1 - s[0])
            if mass > best_mass: best_mass, best = mass, s
        if rule == 'maxpresent':
            out[a] = int(torch.argmax(best)) if best_mass > 0.5 else 0
        elif rule == 'full_or_confident':
            full_pred = int(torch.argmax(slot[0]))
            out[a] = full_pred if full_pred > 0 else (int(torch.argmax(best)) if best_mass > 0.7 else 0)
    return out

def metrics(P, Y, scored):
    tp = pred = gold = 0; mg = mm = 0; n = wrong = tomaj = miss = 0
    for p, g in zip(P, Y):
        present = [i for i in scored if g[i] > 0]
        for i in scored:
            gold += g[i] > 0; pred += p[i] > 0; tp += (g[i] > 0 and p[i] == g[i])
        if len(present) >= 2:
            for i in present: mg += 1; mm += (p[i] == 0)
            cnt = collections.Counter(g[i] for i in present)
            if len(cnt) >= 2:
                maj, mn = cnt.most_common(1)[0]
                if sum(1 for c in cnt.values() if c == mn) == 1:
                    for i in present:
                        if g[i] == maj: continue
                        n += 1; miss += (p[i] == 0)
                        if p[i] > 0 and p[i] != g[i]: wrong += 1; tomaj += (p[i] == maj)
    return {'pair_f1': round(100*2*tp/(pred+gold), 2), 'precision': round(100*tp/max(pred,1), 1), 'recall': round(100*tp/gold, 1),
            'multi_row_missed_pct': round(100*mm/max(mg,1), 1), 'minority_missed_pct': round(100*miss/max(n,1), 1), 'minority_wrongpol_pct': round(100*wrong/max(n,1), 1)}

def _main():
  res = {}
  for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel', 'visfd'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle); K = tr.task_num_aspects(bundle)
        mc = MaskedCLS(model)
        va = tr.split_field(bundle, 'validation'); texts = list(tr.split_texts(va)); Y = [list(l) for l in tr.model_labels(bundle, 'validation', va)]
        base_probs, _ = run_readers(model, tok, texts, mc, counts, 'baseline')
        cl_probs, n_readers = run_readers(model, tok, texts, mc, counts, 'clauses')
        r = {'rows': len(texts), 'mean_readers': round(float(np.mean(n_readers)), 2), 'rows_with_clauses': int(sum(1 for n in n_readers if n > 1))}
        r['baseline_full'] = metrics([decode(p, counts, K, 'full') for p in base_probs], Y, scored)
        r['clauses_maxpresent'] = metrics([decode(p, counts, K, 'maxpresent') for p in cl_probs], Y, scored)
        r['clauses_full_or_confident'] = metrics([decode(p, counts, K, 'full_or_confident') for p in cl_probs], Y, scored)
        res[f'{kind}__{task}'] = r
        print('=' * 100); print(kind, task, {k: r[k] for k in ('rows', 'mean_readers', 'rows_with_clauses')})
        for k in ('baseline_full', 'clauses_maxpresent', 'clauses_full_or_confident'): print(f'   {k:28s} {r[k]}')
        mc.restore(); del model; torch.cuda.empty_cache()
  json.dump(res, open(OUT / 'clause_readers.json', 'w'), indent=1)
if __name__ == '__main__':
    _main()
