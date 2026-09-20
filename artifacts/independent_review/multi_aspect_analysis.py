"""Validation-only analysis: does missed-aspect error concentrate in multi-aspect rows,
and is the missed aspect the 'minority' one? Uses saved validation predictions only
(no model, no test). Gold labels come from the frozen loader; predictions from artifacts.
Mapping (audited): ABSA 0 absent, 1 positive, 2 negative, 3 neutral. ViSFD OTHERS index 5
is detection-only, excluded from pair counts (model binary code 1 -> raw 4)."""
import json, sys, collections
sys.path.insert(0, '/home/kiettqa/GoiBERT/src')
from goibert.data import load_task, VISFD_OTHERS_ASPECT_INDEX

ROOT = '/home/kiettqa/GoiBERT'
MODELS = {
    'visfd': {
        'E8_s42': 'artifacts/e8/visfd/seed_42/validation_predictions.json',
        'E8_s43': 'artifacts/e8/visfd/seed_43/validation_predictions.json',
        'E8_s44': 'artifacts/e8/visfd/seed_44/validation_predictions.json',
        'E14_bami': 'artifacts/e14/trials/visfd/lr_5e-05/seed_42/validation_predictions.json',
        'E16_base': 'artifacts/e16/trials/baseline/visfd/seed_42/validation_predictions.json',
    },
    'absa_restaurant': {
        'E8_s42': 'artifacts/e8/absa_restaurant/seed_42/validation_predictions.json',
        'E8_s43': 'artifacts/e8/absa_restaurant/seed_43/validation_predictions.json',
        'E8_s44': 'artifacts/e8/absa_restaurant/seed_44/validation_predictions.json',
        'E14_bami': 'artifacts/e14/trials/absa_restaurant/lr_5e-05/seed_42/validation_predictions.json',
        'E16_base': 'artifacts/e16/trials/baseline/absa_restaurant/seed_42/validation_predictions.json',
    },
    'absa_hotel': {
        'E8_s42': 'artifacts/e8/absa_hotel/seed_42/validation_predictions.json',
        'E14_bami': 'artifacts/e14/trials/absa_hotel/lr_5e-05/seed_42/validation_predictions.json',
        'E16_base': 'artifacts/e16/trials/baseline/absa_hotel/seed_42/validation_predictions.json',
    },
}

def bucket(n):
    return '1' if n == 1 else '2' if n == 2 else '3' if n == 3 else '4+'

for task, files in MODELS.items():
    ds = load_task(task, ROOT, include_test=False)
    val = ds.validation
    gold_by_id = {ex.id: tuple(ex.labels) for ex in val}
    train_labels = ds.train.labels
    K = ds.num_aspects
    scored = [i for i in range(K) if not (task == 'visfd' and i == VISFD_OTHERS_ASPECT_INDEX)]
    train_presence = collections.Counter()
    for lab in train_labels:
        for i in scored:
            if lab[i] > 0: train_presence[i] += 1
    names = ds.aspect_names
    print('=' * 100)
    print(f'TASK {task}: val rows={len(val)} aspects scored={len(scored)}')
    # row distribution of #gold aspects
    dist = collections.Counter()
    for g in gold_by_id.values():
        n = sum(1 for i in scored if g[i] > 0)
        dist[bucket(n) if n > 0 else '0'] += 1
    print('gold-aspect-count distribution (rows):', dict(sorted(dist.items())))
    for mname, path in files.items():
        rows = json.load(open(f'{ROOT}/{path}'))['rows']
        # ViSFD predictions/gold in artifacts: gold in file uses model codes (OTHERS binary). Use loader gold for scored slots (identical codes 0-3 for scored aspects).
        per_bucket = collections.defaultdict(lambda: {'gold_slots': 0, 'missed': 0, 'wrongpol': 0, 'spurious': 0, 'rows': 0})
        minority = {'minority_gold': 0, 'minority_missed': 0, 'majority_gold': 0, 'majority_missed': 0}
        rare = {'rarer_gold': 0, 'rarer_missed': 0, 'commoner_gold': 0, 'commoner_missed': 0}
        mixed = {'mixed_rows': 0, 'mixed_wrongpol': 0, 'mixed_gold': 0, 'uniform_rows': 0, 'uniform_wrongpol': 0, 'uniform_gold': 0}
        tot = {'gold': 0, 'missed': 0, 'wrongpol': 0, 'spurious': 0, 'pred': 0}
        for r in rows:
            g = gold_by_id[r['id']]
            gfile = r['gold']
            for i in scored:
                assert g[i] == gfile[i], (task, r['id'], i, g[i], gfile[i])
            p = r['prediction']
            present = [i for i in scored if g[i] > 0]
            n = len(present)
            b = bucket(n) if n > 0 else '0'
            per_bucket[b]['rows'] += 1
            for i in scored:
                gi, pi = g[i], p[i]
                if gi > 0:
                    tot['gold'] += 1; per_bucket[b]['gold_slots'] += 1
                    if pi == 0:
                        tot['missed'] += 1; per_bucket[b]['missed'] += 1
                    elif pi != gi:
                        tot['wrongpol'] += 1; per_bucket[b]['wrongpol'] += 1
                if pi > 0:
                    tot['pred'] += 1
                    if gi == 0:
                        tot['spurious'] += 1; per_bucket[b]['spurious'] += 1
            if n >= 2:
                pols = [g[i] for i in present]
                cnt = collections.Counter(pols)
                if len(cnt) >= 2:
                    mixed['mixed_rows'] += 1
                    maj = cnt.most_common(1)[0][1]
                    for i in present:
                        mixed['mixed_gold'] += 1
                        if p[i] > 0 and p[i] != g[i]: mixed['mixed_wrongpol'] += 1
                        if cnt[g[i]] == maj and sum(1 for c in cnt.values() if c == maj) == 1:
                            minority['majority_gold'] += 1
                            if p[i] == 0: minority['majority_missed'] += 1
                        elif cnt[g[i]] < maj:
                            minority['minority_gold'] += 1
                            if p[i] == 0: minority['minority_missed'] += 1
                else:
                    mixed['uniform_rows'] += 1
                    for i in present:
                        mixed['uniform_gold'] += 1
                        if p[i] > 0 and p[i] != g[i]: mixed['uniform_wrongpol'] += 1
                # rarer-vs-commoner by train presence within the row
                freqs = sorted(present, key=lambda i: train_presence[i])
                rarer, commoner = freqs[0], freqs[-1]
                if train_presence[rarer] < train_presence[commoner]:
                    rare['rarer_gold'] += 1; rare['commoner_gold'] += 1
                    if p[rarer] == 0: rare['rarer_missed'] += 1
                    if p[commoner] == 0: rare['commoner_missed'] += 1
        tp = tot['gold'] - tot['missed'] - tot['wrongpol']
        prec = tp / tot['pred'] if tot['pred'] else 0; rec = tp / tot['gold']
        f1 = 2 * tp / (tot['pred'] + tot['gold'])
        print(f'--- {mname}: pairF1={100*f1:.2f} P={100*prec:.2f} R={100*rec:.2f} gold={tot["gold"]} missed={tot["missed"]} wrongpol={tot["wrongpol"]} spurious={tot["spurious"]}')
        for b in ['0', '1', '2', '3', '4+']:
            d = per_bucket.get(b)
            if not d or d['rows'] == 0: continue
            gs = d['gold_slots']
            mr = 100 * d['missed'] / gs if gs else float('nan')
            wr = 100 * d['wrongpol'] / gs if gs else float('nan')
            sr = d['spurious'] / d['rows']
            print(f'    n_gold={b:>2} rows={d["rows"]:4d} gold_slots={gs:4d} missed%={mr:5.1f} wrongpol%={wr:5.1f} spurious/row={sr:.3f}')
        if minority['minority_gold']:
            print(f'    mixed-polarity rows: minority-polarity aspect missed% = {100*minority["minority_missed"]/minority["minority_gold"]:.1f} (n={minority["minority_gold"]}); majority-polarity aspect missed% = {100*minority["majority_missed"]/minority["majority_gold"]:.1f} (n={minority["majority_gold"]})')
        if mixed['mixed_gold']:
            print(f'    wrongpol% in mixed-polarity multi rows = {100*mixed["mixed_wrongpol"]/mixed["mixed_gold"]:.1f} (rows={mixed["mixed_rows"]}); in uniform-polarity multi rows = {100*mixed["uniform_wrongpol"]/mixed["uniform_gold"]:.1f} (rows={mixed["uniform_rows"]})')
        if rare['rarer_gold']:
            print(f'    multi rows: rarer-aspect(train freq) missed% = {100*rare["rarer_missed"]/rare["rarer_gold"]:.1f}; commoner-aspect missed% = {100*rare["commoner_missed"]/rare["commoner_gold"]:.1f} (n={rare["rarer_gold"]})')
