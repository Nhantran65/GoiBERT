"""Measure sampled original text; no corpus construction or GPU use."""
from __future__ import annotations

import collections
import importlib.util
import json
import math
import os
import random
import re
import statistics
import sys
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/pool_quality'
TMP=Path('/tmp/goibert_pool_quality_20260928')
sys.path.insert(0,str(ROOT))
from scripts.corpus_v2 import build as v2

spec=importlib.util.spec_from_file_location('v1_build',ROOT/'reference/ai-foundary/experiments/goibert-family-v1/src/build_corpus.py')
v1=importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
URL=re.compile(r'(?:https?://|www\.|\b[a-z0-9.-]+\.(?:com|vn|net|org|edu|gov)\b)',re.I)
EMAIL=re.compile(r'\b[^\s@]+@[^\s@]+\.[^\s@]+\b')
PHONE=re.compile(r'(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)')

def jl(path):
    with path.open(encoding='utf-8') as f:
        for line in f: yield json.loads(line)

def q(xs,p):
    if not xs: return None
    ys=sorted(xs); pos=(len(ys)-1)*p; lo=math.floor(pos); hi=math.ceil(pos)
    return ys[lo]+(ys[hi]-ys[lo])*(pos-lo)

def ratio(a,b): return a/b if b else None

def run(names):
    os.environ['TOKENIZERS_PARALLELISM']='false'
    _,locks,_=v2.task_texts()
    dc=v2.Decontam(locks)
    records={name:[] for name in names}
    if 'U' in names or 'R' in names:
        for row in jl(OUT/'sample_raw.jsonl'):
            if row['sample'] in records: records[row['sample']].append(row)
    if 'C' in names:
        for row in jl(OUT/'sample_v1_cc100.jsonl'): records['C'].append(row)
    assert all(len(x)==20000 for x in records.values()),{k:len(v) for k,v in records.items()}
    target={r['pool_index'] for name in ('U','R') if name in records for r in records[name]}
    scores={}
    if target:
        for path in sorted((ROOT/'data/mmbert_vi_cpt_pilot/v2/spun_scores').glob('*.jsonl')):
            for row in jl(path):
                i=row['pool_index']
                if i in target: scores[i]=row
    assert len(scores)==len(target),(len(scores),len(target))
    result={}
    for name,rows in records.items():
        doc_lens=[]; par_counts=[]; par_lens=[]; freq=collections.Counter()
        sums=collections.Counter(); bins=collections.Counter(); host_tokens=collections.Counter()
        meta_vals=collections.defaultdict(list); per_doc=[]
        for i,row in enumerate(rows):
            pars=v2.split_paragraphs(row['text']); plen=[]; bad=[]
            for p in pars:
                n=v2.count(p); plen.append(n); par_lens.append(n); sums['tokens']+=n
                freq[v2.norm(p)]+=1; sums['paragraphs']+=1
                if v1.vietnameseish(p): sums['vietnameseish_tokens']+=n
                if len(v2.syllables(p))<5: sums['short_paragraphs']+=1; sums['short_tokens']+=n
                if URL.search(p) or EMAIL.search(p) or PHONE.search(p): sums['contact_paragraphs']+=1
                hit=bool(dc.hit(p)); bad.append(hit)
                if hit: sums['decontam_paragraphs']+=1; sums['decontam_tokens']+=n
            dn=sum(plen); doc_lens.append(dn); par_counts.append(len(pars))
            if pars and sum(bad)/len(pars)>0.2: sums['decontam_drop_docs']+=1
            per_doc.append((list(map(v2.norm,pars)),plen,bad))
            if name!='C':
                m=row['meta']; bins[str(m['bin'])]+=1
                host_tokens[m.get('host') or '']+=dn
                for key in ('op','re','mt','pii_n'):
                    if m.get(key) is not None: meta_vals[key].append(m[key])
                score=scores[row['pool_index']]
                for key in ('s1_mean','s2_prob'):
                    if score.get(key) is not None: meta_vals[key].append(score[key])
            if (i+1)%5000==0: print(name,i+1,flush=True)
        exact=sum(n-1 for n in freq.values() if n>1)
        boiler=sum(n for n in freq.values() if n>=5)
        exact_tok=0; boiler_tok=0; boiler_exact_tok=0; retained=0; seen=set()
        for pars,plen,bad in per_doc:
            if pars and sum(bad)/len(pars)>0.2: continue
            for k,n,hit in zip(pars,plen,bad):
                if hit: continue
                if freq[k]>=5: boiler_tok+=n
                if k in seen:
                    exact_tok+=n
                    if freq[k]>=5: boiler_exact_tok+=n
                else: seen.add(k); retained+=n
        data={'docs':len(rows),'tokens_bami':sums['tokens'],'paragraphs':sums['paragraphs'],
              'tokens_per_doc_mean':statistics.mean(doc_lens),'tokens_per_doc_median':q(doc_lens,.5),
              'tokens_per_doc_p90':q(doc_lens,.9),'paragraphs_per_doc_mean':statistics.mean(par_counts),
              'paragraphs_per_doc_median':q(par_counts,.5),'paragraph_tokens_mean':statistics.mean(par_lens),
              'paragraph_tokens_median':q(par_lens,.5),'paragraph_tokens_p90':q(par_lens,.9),
              'vietnameseish_token_rate':ratio(sums['vietnameseish_tokens'],sums['tokens']),
              'exact_duplicate_paragraph_rate':ratio(exact,sums['paragraphs']),
              'boilerplate_paragraph_rate':ratio(boiler,sums['paragraphs']),
              'short_paragraph_rate':ratio(sums['short_paragraphs'],sums['paragraphs']),
              'contact_paragraph_rate':ratio(sums['contact_paragraphs'],sums['paragraphs']),
              'decontam_paragraph_rate':ratio(sums['decontam_paragraphs'],sums['paragraphs']),
              'decontam_doc_drop_rate':ratio(sums['decontam_drop_docs'],len(rows)),
              'decontam_token_rate':ratio(sums['decontam_tokens'],sums['tokens']),
              'sequential_exact_drop_tokens':exact_tok,'sequential_boilerplate_exposure_tokens':boiler_tok,
              'sequential_boilerplate_exact_drop_tokens':boiler_exact_tok,
              'sequential_decontam_drop_tokens':sums['tokens']-retained-exact_tok,
              'sequential_retained_tokens':retained}
        if name!='C':
            mm=sum(row['meta']['n_tokens_mmbert'] for row in rows)
            data.update({'mmbert_tokens':mm,'bami_per_mmbert':ratio(sums['tokens'],mm),
                'bin_counts':dict(sorted(bins.items())),'host_top20_token_share':ratio(sum(n for _,n in host_tokens.most_common(20)),sums['tokens']),
                'metadata':{k:{'n':len(v),'mean':statistics.mean(v) if v else None,
                    'median':q(v,.5),'p90':q(v,.9)} for k,v in meta_vals.items()}})
        result[name]=data
        print('complete',name,flush=True)
    partial=OUT/('summary_'+''.join(names)+'.json')
    partial.write_text(json.dumps({'samples':result,'decontam_unique':dc.unique},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples',nargs='+',choices=['U','R','C'],default=['U','R','C'])
    a=p.parse_args();run(a.samples)
