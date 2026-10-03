#!/usr/bin/env python3
"""Recompute all fairness validation metrics from saved predictions."""
from __future__ import annotations
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/fairness'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts/bench8_kd'), str(ROOT/'scripts/bench8')]
from goibert import train as T
import scripts.bench8_kd.common as B

OLD = ('vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel')
TASKS = OLD + ('vispam','vinli','phoner')
SEEDS = (42,43,44)
STUDENT_LR = {'vsfc_sentiment':1e-5,'vsfc_topic':1e-5,'visfd':1e-4,
              'absa_restaurant':1e-4,'absa_hotel':1e-4,'vispam':5e-5,
              'vinli':1e-4,'phoner':1e-4}
BAMI_LR = {'vsfc_sentiment':2e-5,'vsfc_topic':1e-5,'visfd':5e-5,
           'absa_restaurant':5e-5,'absa_hotel':5e-5,'vispam':2e-5,
           'vinli':1e-5,'phoner':5e-5}
METRICS = {'vsfc_sentiment':('accuracy','macro_f1'),'vsfc_topic':('accuracy','macro_f1'),
           'visfd':('detection_micro_f1','pair_micro_f1'),
           'absa_restaurant':('detection_micro_f1','pair_micro_f1'),
           'absa_hotel':('detection_micro_f1','pair_micro_f1'),
           'vispam':('accuracy','macro_f1'),'vinli':('accuracy','macro_f1'),
           'phoner':('f1',)}


def read(path): return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def run_dir(model,task,seed,lr=None):
    if model in ('bami_e7_ce','bami_e7_kd'):
        return ART/'runs'/model/task/f'lr_{lr:.0e}'/f'seed_{seed}'
    if model in ('s_prime_ce','m_ce'):
        return ART/'runs'/model/task/f'seed_{seed}'
    if model=='s_prime_insample':
        return ART/'runs'/model/task/f'seed_{seed}'
    raise ValueError(model)


def reference_dir(model,task,seed):
    if model=='bami_e14':
        if task in OLD and seed==42:
            return ROOT/f'artifacts/e14/trials/{task}/lr_{BAMI_LR[task]:.0e}/seed_42'
        if task in OLD:
            return ROOT/f'artifacts/cafe_oof_kd/bami/{task}/seed_{seed}'
        return ROOT/f'artifacts/bench8/runs/{task}/bami/lr_{BAMI_LR[task]:.0e}/seed_{seed}'
    if model in ('s_prime_kd','m_kd'):
        variant='s_prime' if model=='s_prime_kd' else 'm12'
        lr=STUDENT_LR[task] if variant=='s_prime' else read(ROOT/'artifacts/family_sm/decisions.json')['m_lr'][task]
        return ROOT/f'artifacts/family_sm/finetune/{variant}/{task}/lr_{lr:.0e}/seed_{seed}'
    raise ValueError(model)


def eval_dir(path,task,require_checkpoint=False):
    needed=('validation_predictions.json','best_validation_metrics.json','metadata.json','resolved_config.json')
    if not all((path/n).is_file() for n in needed): return None
    if require_checkpoint and not (path/'best_checkpoint.pt').is_file(): return None
    pred=read(path/'validation_predictions.json')['rows']
    saved=read(path/'best_validation_metrics.json')
    meta=read(path/'metadata.json')
    cfg=read(path/'resolved_config.json')
    checks=meta.get('postrun_checks',{})
    if meta['status']!='completed' or (not checks and ART in path.parents) or not all(x for x in checks.values() if isinstance(x,bool)):
        raise RuntimeError(f'Postrun check FAIL: {path}')
    gold=[r['gold'] for r in pred];guess=[r['prediction'] for r in pred]
    if task in OLD:
        bundle=T.load_task_bundle(task,ROOT,include_test=False)
        metrics=T.evaluate_predictions(task,bundle,gold,guess,split='validation')
    else:
        data=B.load(task);metrics=B.evaluate(task,tuple(data.label_names),gold,guess)
    for name in (*METRICS[task],'selection_f1'):
        if abs(float(metrics[name])-float(saved[name]))>=1e-9:
            raise AssertionError((path,name,metrics[name],saved[name]))
    if abs(float(metrics['selection_f1'])-float(meta['best_selection_f1']))>=1e-9:
        raise AssertionError(f'Selection mismatch: {path}')
    return {'metrics':metrics,'selection_f1':float(metrics['selection_f1']),
            'best_epoch':int(meta['best_epoch']),'learning_rate':float(meta['learning_rate']),
            'backbone':meta.get('backbone',cfg.get('backbone')),
            'elapsed_seconds':float(meta['elapsed_seconds']),'path':str(path)}


def stat(values):
    return {'mean':statistics.mean(values),'range':[min(values),max(values)],
            'seeds':values}


def main():
    d=read(ART/'decisions.json') if (ART/'decisions.json').is_file() else {'bami_lr':{}}
    models=('bami_e14','bami_e7_ce','bami_e7_kd','s_prime_ce','s_prime_kd','m_ce','m_kd')
    table={m:{} for m in models}
    for model in models:
        for task in TASKS:
            lr=None
            if model.startswith('bami_e7'):
                key=f'{model}/{task}'
                if key not in d['bami_lr']:continue
                lr=d['bami_lr'][key]['chosen_lr']
            runs=[]
            for seed in SEEDS:
                path=reference_dir(model,task,seed) if model in ('bami_e14','s_prime_kd','m_kd') else run_dir(model,task,seed,lr)
                runs.append(eval_dir(path,task))
            if not all(runs):continue
            for metric in METRICS[task]:
                table[model][f'{task}/{metric}']=stat([100*r['metrics'][metric] for r in runs])
    # Cafe teacher is a validation-only reference, averaged over existing folds.
    teacher={}
    for task in OLD:
        paths=[ROOT/f'artifacts/cafe_oof_kd/teacher/{task}/fold_{i}' for i in range(5)]
        vals=[eval_dir(p,task) for p in paths]
        if all(vals):
            for metric in METRICS[task]:
                teacher[f'{task}/{metric}']=stat([100*v['metrics'][metric] for v in vals])
    for task in ('vispam','vinli','phoner'):
        paths=[ROOT/f'artifacts/bench8_kd/teacher/{task}/fold_{i}' for i in range(5)]
        vals=[eval_dir(p,task) for p in paths]
        if all(vals):
            for metric in METRICS[task]:teacher[f'{task}/{metric}']=stat([100*v['metrics'][metric] for v in vals])
    table['cafe_teacher_validation']=teacher
    compare={}
    for task in ('vsfc_sentiment','absa_restaurant','vinli'):
        compare[task]={}
        for label,path_fn in [('oof',lambda seed:reference_dir('s_prime_kd',task,seed)),
                              ('insample',lambda seed:run_dir('s_prime_insample',task,seed))]:
            vals=[eval_dir(path_fn(seed),task) for seed in SEEDS]
            if all(vals):compare[task][label]={m:stat([100*v['metrics'][m] for v in vals]) for m in METRICS[task]}
    result={'metrics':METRICS,'validation_table':table,'insample_comparison':compare,
            'complete':all(len(table[m])==15 for m in models),
            'decisions':d}
    write(ART/'validation_summary.json',result)
    print(json.dumps({'complete':result['complete'],'metric_counts':{k:len(v) for k,v in table.items()}},indent=2))


if __name__=='__main__':main()
