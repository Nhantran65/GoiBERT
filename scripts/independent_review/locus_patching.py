"""Causal locus analysis (inference only, frozen checkpoints): where does the minority aspect's evidence get lost?
Composite = A + B + B + B (A-first). For each layer l, patch hidden states from the 'A alone' forward into the
composite forward at the output of layer l, for (i) CLS position only, (ii) A's token positions only, (iii) both;
then measure whether A's polarity prediction is restored. Also report logit-level A-evidence."""
import sys, json, random
from pathlib import Path
ROOT = Path('/home/kiettqa/GoiBERT'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'/'independent_review'))
import torch
from mechanism_battery import load_model, build_pairs, predictions, scored_aspects, MAX_LEN
from goibert import train as tr
device = torch.device('cuda:1')
OUT = ROOT/'artifacts'/'independent_review'/'battery'

def run_with_patch(model, enc, patch_layer, patch_src, positions_list, device):
    """patch_layer: -1 for embeddings output, l for output of encoder layer l. patch_src: list of tensors [Lsrc, d] per row."""
    layers = model.encoder.encoder.layer
    handles = []
    def make_hook():
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            for b, pos in enumerate(positions_list):
                if len(pos) == 0: continue
                hs[b, pos] = patch_src[b][pos].to(hs.device, hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
        return hook
    if patch_layer == -1:
        handles.append(model.encoder.embeddings.register_forward_hook(make_hook()))
    else:
        handles.append(layers[patch_layer].register_forward_hook(make_hook()))
    try:
        with torch.inference_mode(), tr.make_autocast('bf16', device):
            logits = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
    finally:
        for h in handles: h.remove()
    return logits.float().cpu()

@torch.inference_mode()
def hidden_states(model, enc, device):
    with tr.make_autocast('bf16', device):
        out = model.encoder(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device), output_hidden_states=True, return_dict=True)
    return [h.float().cpu() for h in out.hidden_states]   # L+1 tensors [B,N,d]

results = {}
for kind in ('student_e8', 'bami_e14'):
    for task in ('absa_restaurant', 'absa_hotel', 'visfd'):
        model, tok, bundle, counts, ckpt = load_model(kind, task, device)
        scored = scored_aspects(task, bundle)
        singles, pairs = build_pairs(task, bundle, scored, tok, 200, 2026)
        nL = len(model.encoder.encoder.layer)
        comp_texts = [A['text'] + ' ' + ' '.join([B['text']] * 3) for A, B in pairs]
        alone_texts = [A['text'] for A, B in pairs]
        # per-row A token positions in the composite (prefix match, exclude <s> and A's last token)
        a_ids = [tok(t, truncation=True, max_length=MAX_LEN)['input_ids'] for t in alone_texts]
        stats = {}
        BS = 16
        conds = ['none', 'cls', 'atok', 'both']
        agg = {(c, l): {'n': 0, 'A_correct': 0, 'A_flip': 0, 'A_missed': 0} for c in conds for l in range(-1, nL)}
        alone_ok = 0
        for i in range(0, len(pairs), BS):
            chunk = pairs[i:i+BS]
            enc_a = tok(alone_texts[i:i+BS], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
            enc_c = tok(comp_texts[i:i+BS], padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
            hs_a = hidden_states(model, enc_a, device)
            with torch.inference_mode(), tr.make_autocast('bf16', device):
                lg_alone = model(enc_a['input_ids'].to(device), enc_a['attention_mask'].to(device)).float().cpu()
            pa = predictions(lg_alone, counts)
            for b, (A, B) in enumerate(chunk):
                if pa[b][A['aspect']] == A['pol']: alone_ok += 1
            # positions: CLS = [0]; A tokens = 1..len(a_ids)-3 (exclude <s>, last A token, </s>)
            pos_cls = [[0] for _ in chunk]
            pos_atok = [list(range(1, max(1, len(a_ids[i+b]) - 2))) for b in range(len(chunk))]
            # sanity: composite prefix token ids equal A's ids on those positions
            for b in range(len(chunk)):
                cid = enc_c['input_ids'][b].tolist(); aid = a_ids[i+b]
                assert cid[:len(pos_atok[b])+1] == aid[:len(pos_atok[b])+1], (cid[:10], aid[:10])
            for c in conds:
                for l in range(-1, nL):
                    if c == 'none':
                        if l != -1: continue
                        with torch.inference_mode(), tr.make_autocast('bf16', device):
                            lg = model(enc_c['input_ids'].to(device), enc_c['attention_mask'].to(device)).float().cpu()
                    else:
                        src = hs_a[l + 1]   # [B, Na, d]; need to place into composite positions (same indices, prefix)
                        # pad src to composite length
                        Nc = enc_c['input_ids'].shape[1]
                        src_p = torch.zeros(src.shape[0], Nc, src.shape[2]); src_p[:, :src.shape[1]] = src
                        pl = pos_cls if c == 'cls' else pos_atok if c == 'atok' else [pos_cls[b] + pos_atok[b] for b in range(len(chunk))]
                        lg = run_with_patch(model, enc_c, l, [src_p[b] for b in range(len(chunk))], pl, device)
                    pr = predictions(lg, counts)
                    d = agg[(c, l if c != 'none' else -1)]
                    for b, (A, B) in enumerate(chunk):
                        d['n'] += 1
                        p = pr[b][A['aspect']]
                        if p == A['pol']: d['A_correct'] += 1
                        elif p == 0: d['A_missed'] += 1
                        elif p == B['pol']: d['A_flip'] += 1
        key = f'{kind}__{task}'
        results[key] = {'n_pairs': len(pairs), 'alone_A_correct_pct': round(100*alone_ok/len(pairs), 1), 'layers': nL,
                        'composite_no_patch': {k: round(100*v/agg[('none',-1)]['n'],1) for k, v in agg[('none',-1)].items() if k != 'n'},
                        'patch': {c: {str(l): {k: round(100*v/agg[(c,l)]['n'],1) for k, v in agg[(c,l)].items() if k != 'n'} for l in range(-1, nL)} for c in ('cls','atok','both')}}
        print('='*100); print(key, 'alone A correct%:', results[key]['alone_A_correct_pct'], '| composite(A+3B) no patch:', results[key]['composite_no_patch'])
        for c in ('cls', 'atok', 'both'):
            print(f'  patch {c:4s}: ' + ' '.join(f"L{l}:{results[key]['patch'][c][str(l)]['A_correct']:.0f}" for l in range(-1, nL)))
        del model; torch.cuda.empty_cache()
json.dump(results, open(OUT/'locus_patching.json', 'w'), indent=1)
