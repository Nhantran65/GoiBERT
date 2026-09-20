"""In mixed-polarity multi-aspect rows: does the minority-polarity aspect's error rate
grow with the number of majority-polarity aspects in the same row? Validation-only."""
import json, sys, collections
sys.path.insert(0, '/home/kiettqa/GoiBERT/src')
from goibert.data import load_task, VISFD_OTHERS_ASPECT_INDEX
ROOT = '/home/kiettqa/GoiBERT'
FILES = {
 'visfd': {'E8_s42':'artifacts/e8/visfd/seed_42/validation_predictions.json',
           'E8_s43':'artifacts/e8/visfd/seed_43/validation_predictions.json',
           'E8_s44':'artifacts/e8/visfd/seed_44/validation_predictions.json',
           'E14_bami':'artifacts/e14/trials/visfd/lr_5e-05/seed_42/validation_predictions.json'},
 'absa_hotel': {'E8_s42':'artifacts/e8/absa_hotel/seed_42/validation_predictions.json',
                'E14_bami':'artifacts/e14/trials/absa_hotel/lr_5e-05/seed_42/validation_predictions.json'},
 'absa_restaurant': {'E8_s42':'artifacts/e8/absa_restaurant/seed_42/validation_predictions.json',
                'E8_s43':'artifacts/e8/absa_restaurant/seed_43/validation_predictions.json',
                'E8_s44':'artifacts/e8/absa_restaurant/seed_44/validation_predictions.json',
                'E14_bami':'artifacts/e14/trials/absa_restaurant/lr_5e-05/seed_42/validation_predictions.json'},
}
for task, files in FILES.items():
    ds = load_task(task, ROOT, include_test=False)
    gold = {ex.id: tuple(ex.labels) for ex in ds.validation}
    K = ds.num_aspects
    scored = [i for i in range(K) if not (task=='visfd' and i==VISFD_OTHERS_ASPECT_INDEX)]
    print('='*90); print(task)
    for m, path in files.items():
        rows = json.load(open(f'{ROOT}/{path}'))['rows']
        stat = collections.defaultdict(lambda: {'n':0,'missed':0,'wrongpol':0,'flip_to_majority':0})
        for r in rows:
            g = gold[r['id']]; p = r['prediction']
            present = [i for i in scored if g[i]>0]
            if len(present) < 2: continue
            cnt = collections.Counter(g[i] for i in present)
            if len(cnt) < 2: continue
            maj_pol, maj_n = cnt.most_common(1)[0]
            if sum(1 for c in cnt.values() if c==maj_n) > 1: continue  # tie: no unique majority
            for i in present:
                if g[i] == maj_pol: continue
                # minority-polarity aspect i; number of majority-polarity aspects = maj_n
                b = '1' if maj_n==1 else '2' if maj_n==2 else '3+'
                s = stat[b]; s['n'] += 1
                if p[i]==0: s['missed'] += 1
                elif p[i]!=g[i]:
                    s['wrongpol'] += 1
                    if p[i]==maj_pol: s['flip_to_majority'] += 1
        line = f'  {m:9s}'
        for b in ['1','2','3+']:
            s = stat[b]
            if s['n']==0: continue
            line += f' | maj={b}: n={s["n"]:3d} missed%={100*s["missed"]/s["n"]:5.1f} wrongpol%={100*s["wrongpol"]/s["n"]:5.1f} (->majority {s["flip_to_majority"]}/{s["wrongpol"]})'
        print(line)
