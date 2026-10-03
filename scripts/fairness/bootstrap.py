#!/usr/bin/env python3
"""10,000 paired sentence bootstrap replicates, averaging three seeds per draw."""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/fairness/test'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.bench8.ner import spans
from summarize import TASKS,SEEDS,METRICS

MODELS=('bami_e14','bami_e7_kd','s_prime_ce','s_prime_kd','m_ce','m_kd')
PAIRS=(('m_kd','bami_e14'),('m_kd','bami_e7_kd'),('s_prime_kd','bami_e14'),
       ('m_kd','m_ce'),('s_prime_kd','s_prime_ce'))
N_REPLICATES=10000
SEED=20260930


def source(model,task,seed):
    if model=='bami_e14':return ROOT/f'artifacts/final_test/predictions/bami_{task}_seed_{seed}.json'
    if model=='s_prime_kd':return ROOT/f'artifacts/family_sm/test/predictions/s_prime_{task}_seed_{seed}.json'
    if model=='m_kd':return ROOT/f'artifacts/family_sm/test/predictions/m_{task}_seed_{seed}.json'
    return ART/'predictions'/f'{model}_{task}_seed_{seed}.json'


def records(model,task,seed):
    return json.loads(source(model,task,seed).read_text())['rows']


def sentence_stats(task,rows,metric):
    gold=[r['gold'] for r in rows];pred=[r['prediction'] for r in rows]
    n=len(rows)
    if metric in ('accuracy','macro_f1'):
        g=np.asarray(gold,dtype=np.int32);p=np.asarray(pred,dtype=np.int32)
        classes=int(max(g.max(),p.max())+1)
        if task=='vsfc_topic' or task=='vispam' or task=='vinli':classes=max(classes,4)
        if task=='vsfc_sentiment':classes=max(classes,3)
        out=np.zeros((n,1+3*classes),dtype=np.int32)
        out[:,0]=(g==p)
        for c in range(classes):
            out[:,1+c]=(g==c)&(p==c)
            out[:,1+classes+c]=(p==c)
            out[:,1+2*classes+c]=(g==c)
        return out
    if metric in ('detection_micro_f1','pair_micro_f1'):
        g=np.asarray(gold,dtype=np.int32);p=np.asarray(pred,dtype=np.int32)
        assert g.shape==p.shape and g.ndim==2
        pair_mask=np.ones(g.shape[1],dtype=bool)
        if task=='visfd':
            # The ViSFD OTHERS slot is detection-only in the frozen evaluator.
            from goibert import train as T
            bundle=T.load_task_bundle(task,ROOT,include_test=False)
            pair_mask=np.asarray([x!='OTHERS' for x in bundle.aspect_names],dtype=bool)
        out=np.zeros((n,3),dtype=np.int32)
        if metric=='detection_micro_f1':
            out[:,0]=((g>0)&(p>0)).sum(axis=1)
            out[:,1]=(p>0).sum(axis=1)
            out[:,2]=(g>0).sum(axis=1)
        else:
            gg=g[:,pair_mask];pp=p[:,pair_mask]
            out[:,0]=((gg>0)&(gg==pp)).sum(axis=1)
            out[:,1]=(pp>0).sum(axis=1)
            out[:,2]=(gg>0).sum(axis=1)
        return out
    assert metric=='f1' and task=='phoner'
    out=np.zeros((n,3),dtype=np.int32)
    for i,(g,p) in enumerate(zip(gold,pred)):
        gs,ps=Counter(spans(g)),Counter(spans(p))
        out[i]=[sum((gs&ps).values()),sum(ps.values()),sum(gs.values())]
    return out


def value(stats,metric):
    # stats: [..., sufficient statistics]
    if metric=='macro_f1':
        k=(stats.shape[-1]-1)//3
        tp=stats[...,1:1+k];pr=stats[...,1+k:1+2*k];gold=stats[...,1+2*k:]
        return np.divide(2*tp,pr+gold,out=np.zeros_like(tp,dtype=float),where=(pr+gold)!=0).mean(axis=-1)
    tp,pr,gold=stats[...,0],stats[...,1],stats[...,2]
    return np.divide(2*tp,pr+gold,out=np.zeros_like(tp,dtype=float),where=(pr+gold)!=0)


def metric_value(stats,metric,n):
    if metric=='accuracy':return stats[...,0]/n
    return value(stats,metric)


def main():
    manifest=ART/'locked_manifest.json'
    if not manifest.is_file() or not (ART/'test_results.json').is_file():
        raise RuntimeError('Bootstrap requires the locked manifest and completed test predictions')
    if json.loads((ART/'test_results.json').read_text())['completed']!=96:
        raise RuntimeError('Test predictions incomplete')
    rng=np.random.default_rng(SEED)
    output={'replicates':N_REPLICATES,'seed':SEED,'resampling_unit':'test sentence',
            'seed_aggregation':'mean of three run scores within each resample',
            'p_value_method':'two-sided sign-tail bootstrap with +1 correction',
            'pairs':{},'metric_results':{}}
    for task in TASKS:
        rows={(model,seed):records(model,task,seed) for model in MODELS for seed in SEEDS}
        ids=[r['id'] for r in rows[(MODELS[0],42)]]
        gold=[r['gold'] for r in rows[(MODELS[0],42)]]
        for key,rr in rows.items():
            if [r['id'] for r in rr]!=ids or [r['gold'] for r in rr]!=gold:
                raise AssertionError(f'Unpaired test rows: {task} {key}')
        n=len(ids)
        for metric in METRICS[task]:
            arrays={key:sentence_stats(task,rr,metric) for key,rr in rows.items()}
            shapes={a.shape for a in arrays.values()}
            if len(shapes)!=1:raise AssertionError((task,metric,shapes))
            point={model:np.mean([metric_value(arrays[(model,s)].sum(axis=0),metric,n)
                                  for s in SEEDS])*100 for model in MODELS}
            draws={model:np.empty(N_REPLICATES,dtype=float) for model in MODELS}
            order=[(model,seed) for model in MODELS for seed in SEEDS]
            width=next(iter(arrays.values())).shape[1]
            matrix=np.concatenate([arrays[key] for key in order],axis=1).astype(np.float64)
            probabilities=np.full(n,1/n)
            for start in range(0,N_REPLICATES,64):
                count=min(64,N_REPLICATES-start)
                # Multinomial sentence counts are exactly equivalent to
                # sampling n sentence indices with replacement.
                weights=rng.multinomial(n,probabilities,size=count).astype(np.float64)
                totals=weights@matrix
                for j,model in enumerate(MODELS):
                    seed_scores=[metric_value(totals[:,(j*3+k)*width:(j*3+k+1)*width],metric,n)
                                 for k in range(3)]
                    draws[model][start:start+count]=np.mean(seed_scores,axis=0)*100
            key=f'{task}/{metric}'
            for a,b in PAIRS:
                pair=f'{a} - {b}'
                delta=draws[a]-draws[b]
                lo,hi=np.quantile(delta,[.025,.975])
                p=min(1.,2*min((np.count_nonzero(delta<=0)+1)/(N_REPLICATES+1),
                                (np.count_nonzero(delta>=0)+1)/(N_REPLICATES+1)))
                output['metric_results'].setdefault(pair,{})[key]={
                    'delta':float(point[a]-point[b]),'ci_95':[float(lo),float(hi)],
                    'p_value_two_sided':float(p),'positive_ci_excludes_zero':bool((point[a]-point[b])>0 and lo>0),
                    'n_test_sentences':n}
            print('BOOTSTRAP',key,flush=True)
    for pair,results in output['metric_results'].items():
        positives=sum(v['positive_ci_excludes_zero'] for v in results.values())
        output['pairs'][pair]={'positive_ci_excludes_zero':positives,
                               'total_metrics':len(results),'fraction':positives/len(results)}
    target=ART/'bootstrap.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
