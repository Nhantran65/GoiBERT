"""Validation-only: break down minority-polarity aspect errors in mixed-polarity multi-aspect rows
by (gold minority polarity, majority polarity). No model loading; saved predictions only.
ABSA mapping (audited E12): 1 positive, 2 negative, 3 neutral. ViSFD OTHERS (index 5) excluded."""
import json, sys, collections
sys.path.insert(0, '/home/kiettqa/GoiBERT/src')
from goibert.data import load_task, VISFD_OTHERS_ASPECT_INDEX
ROOT = '/home/kiettqa/GoiBERT'
POL = {1: 'pos', 2: 'neg', 3: 'neu'}
FILES = {
 'visfd': {'E8_s42': 'artifacts/e8/visfd/seed_42/validation_predictions.json',
           'E8_s43': 'artifacts/e8/visfd/seed_43/validation_predictions.json',
           'E8_s44': 'artifacts/e8/visfd/seed_44/validation_predictions.json',
           'E14_bami': 'artifacts/e14/trials/visfd/lr_5e-05/seed_42/validation_predictions.json'},
 'absa_hotel': {'E8_s42': 'artifacts/e8/absa_hotel/seed_42/validation_predictions.json',
                'E14_bami': 'artifacts/e14/trials/absa_hotel/lr_5e-05/seed_42/validation_predictions.json'},
}
out = {}
for task, files in FILES.items():
    ds = load_task(task, ROOT, include_test=False)
    gold = {ex.id: tuple(ex.labels) for ex in ds.validation}
    scored = [i for i in range(ds.num_aspects) if not (task == 'visfd' and i == VISFD_OTHERS_ASPECT_INDEX)]
    print('=' * 80, '\n', task)
    out[task] = {}
    for m, path in files.items():
        rows = json.load(open(f'{ROOT}/{path}'))['rows']
        cell = collections.defaultdict(lambda: {'n': 0, 'missed': 0, 'wrong': 0, 'to_majority': 0})
        for r in rows:
            g = gold[r['id']]; p = r['prediction']
            present = [i for i in scored if g[i] > 0]
            if len(present) < 2: continue
            cnt = collections.Counter(g[i] for i in present)
            if len(cnt) < 2: continue
            maj_pol, maj_n = cnt.most_common(1)[0]
            if sum(1 for c in cnt.values() if c == maj_n) > 1: continue
            for i in present:
                if g[i] == maj_pol: continue
                c = cell[(POL[g[i]], POL[maj_pol])]; c['n'] += 1
                if p[i] == 0: c['missed'] += 1
                elif p[i] != g[i]:
                    c['wrong'] += 1
                    if p[i] == maj_pol: c['to_majority'] += 1
        print(f'  {m}')
        out[task][m] = {}
        for key in sorted(cell):
            c = cell[key]
            print(f'     minority={key[0]:3s} majority={key[1]:3s}: n={c["n"]:3d} missed={c["missed"]:3d} wrongpol={c["wrong"]:3d} of which ->majority={c["to_majority"]:3d}')
            out[task][m][f'{key[0]}|{key[1]}'] = dict(c)
json.dump(out, open(f'{ROOT}/artifacts/independent_review/minority_polarity_breakdown.json', 'w'), indent=1)
