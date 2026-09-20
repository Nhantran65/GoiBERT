"""Step 0c: extract minority-polarity error cases (ViSFD validation, E8 seed 42) for human adjudication.
Validation-only; no model; gold unchanged. Categories: minority neutral in positive-majority rows,
minority neutral in negative-majority rows, minority negative in positive-majority rows."""
import json, sys, random, collections
sys.path.insert(0, '/home/kiettqa/GoiBERT/src')
from goibert.data import load_task, VISFD_OTHERS_ASPECT_INDEX
ROOT='/home/kiettqa/GoiBERT'
POL={0:'-',1:'pos',2:'neg',3:'neu',4:'OTHERS'}
ds=load_task('visfd', ROOT, include_test=False)
names=ds.aspect_names
val={ex.id:(ex.text, tuple(ex.labels)) for ex in ds.validation}
rows=json.load(open(f'{ROOT}/artifacts/e8/visfd/seed_42/validation_predictions.json'))['rows']
scored=[i for i in range(len(names)) if i!=VISFD_OTHERS_ASPECT_INDEX]
cases=collections.defaultdict(list)
for r in rows:
    text,g=val[r['id']]; p=r['prediction']
    present=[i for i in scored if g[i]>0]
    if len(present)<2: continue
    cnt=collections.Counter(g[i] for i in present)
    if len(cnt)<2: continue
    maj,mn=cnt.most_common(1)[0]
    if sum(1 for c in cnt.values() if c==mn)>1: continue
    for i in present:
        if g[i]==maj or p[i]==0 or p[i]==g[i]: continue
        key=f'{POL[g[i]]}|{POL[maj]}'
        cases[key].append({'id':r['id'],'text':text,'aspect':names[i],'gold_minority':POL[g[i]],'pred':POL[p[i]],'majority':POL[maj],
                           'gold_row':{names[j]:POL[g[j]] for j in range(len(names)) if g[j]>0},
                           'pred_row':{names[j]:POL[p[j]] for j in range(len(names)) if p[j]>0}})
rng=random.Random(20260913)
quota={'neu|pos':30,'neu|neg':15,'neg|pos':15}
sample=[]
for k,q in quota.items():
    pool=cases[k]; rng.shuffle(pool); sample+=pool[:q]
    print(k,'available',len(cases[k]),'sampled',min(q,len(pool)))
json.dump(sample, open(f'{ROOT}/artifacts/independent_review/adjudication_sample_visfd_e8s42.json','w'), ensure_ascii=False, indent=1)
with open(f'{ROOT}/artifacts/independent_review/adjudication_sample_visfd_e8s42.md','w') as f:
    f.write('# Mẫu adjudication 0c — ViSFD validation, E8 seed42, lỗi polarity của aspect thiểu số\n\n')
    for n,c in enumerate(sample):
        f.write(f"## {n+1}. {c['id']} | aspect {c['aspect']} | gold {c['gold_minority']} → pred {c['pred']} | majority {c['majority']}\n")
        f.write(f"Text: {c['text']}\n\nGold: {c['gold_row']}\n\nPred: {c['pred_row']}\n\n")
print('written', len(sample))
