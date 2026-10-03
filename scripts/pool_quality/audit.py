"""Prepare a blind 180-excerpt reading sheet and join human labels afterward."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/pool_quality'
TMP=Path('/tmp/goibert_pool_quality_20260928')
CATEGORIES=['tin tức/báo chí','bài giới thiệu/kiến thức','diễn đàn/UGC',
 'thương mại/quảng cáo/SEO','truyện/văn học','văn bản pháp luật/hành chính','rác/lỗi/máy dịch']

def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f: yield json.loads(line)

def prepare():
    src={'U':[],'R':[],'C':[]}
    for r in rows(OUT/'sample_raw.jsonl'): src[r['sample']].append(r)
    for r in rows(OUT/'sample_v1_cc100.jsonl'): src['C'].append(r)
    selected=[]
    for name in ('U','R','C'):
        for r in random.Random(7).sample(src[name],60):
            selected.append({'sample':name,'id':r.get('id'),'chunk_ordinal':r.get('chunk_ordinal'),
                'block_index':r.get('block_index'),'excerpt':r['text'][:600]})
    random.Random(7).shuffle(selected)
    with (TMP/'audit_blind.jsonl').open('w',encoding='utf-8') as f:
        for i,r in enumerate(selected):
            r['blind_id']=i+1
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (TMP/'audit_reading.txt').open('w',encoding='utf-8') as f:
        for r in selected: f.write(f"\n### {r['blind_id']}\n{r['excerpt']}\n")
    print(TMP/'audit_reading.txt')

def finish():
    labels={}
    for line in (TMP/'audit_labels.txt').read_text().splitlines():
        if not line.strip(): continue
        i,k=map(int,line.split())
        assert 1<=i<=180 and 0<=k<7 and i not in labels
        labels[i]=k
    assert len(labels)==180,len(labels)
    counts={n:{c:0 for c in CATEGORIES} for n in ('U','R','C')}
    with (OUT/'audit_180.jsonl').open('w',encoding='utf-8') as f:
        for r in rows(TMP/'audit_blind.jsonl'):
            r['label']=CATEGORIES[labels[r['blind_id']]]
            counts[r['sample']][r['label']]+=1
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (TMP/'audit_counts.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(counts,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','finish']);a=p.parse_args()
    {'prepare':prepare,'finish':finish}[a.phase]()
