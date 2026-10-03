#!/usr/bin/env python3
"""Recompute validation indicators from saved prediction rows."""
import statistics
from pathlib import Path
import common as C
from goibert import train as T

OLD=['vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel']
NEW=list(C.TASKS)
FIELDS={'vsfc_sentiment':['accuracy','macro_f1'],'vsfc_topic':['accuracy','macro_f1'],'visfd':['detection_micro_f1','pair_micro_f1'],'absa_restaurant':['detection_micro_f1','pair_micro_f1'],'absa_hotel':['detection_micro_f1','pair_micro_f1'],'vispam':['accuracy','macro_f1'],'vinli':['accuracy','macro_f1'],'phoner':['f1']}
BAMI_LR={'vsfc_sentiment':2e-5,'vsfc_topic':1e-5,'visfd':5e-5,'absa_restaurant':5e-5,'absa_hotel':5e-5}

def path(task,arm,seed):
    if task in OLD:
        if arm=='student':return C.ROOT/'artifacts/cafe_oof_kd/student/kd'/task/f'seed_{seed}'
        return C.ROOT/'artifacts/e14/trials'/task/f'lr_{BAMI_LR[task]:.0e}'/'seed_42' if seed==42 else C.ROOT/'artifacts/cafe_oof_kd/bami'/task/f'seed_{seed}'
    if arm=='student':return C.ART/'student'/task/f'seed_{seed}'
    selected=C.read(C.ROOT/'artifacts/bench8/selected_lrs.json')
    model='student' if arm=='ce' else 'bami'
    return C.ROOT/'artifacts/bench8/runs'/task/model/f'lr_{selected[f"{task}/{model}"]:.0e}'/f'seed_{seed}'

def recalc(task,p):
    rows=C.read(p/'validation_predictions.json')['rows'];saved=C.read(p/'best_validation_metrics.json')
    if task in OLD:
        bundle=T.load_task_bundle(task,C.ROOT,include_test=False)
        metric=T.evaluate_predictions(task,bundle,[r['gold'] for r in rows],[r['prediction'] for r in rows],split='validation')
    else:
        b=C.load(task);assert [r['id'] for r in rows]==b.validation.ids
        metric=C.evaluate(task,b.label_names,[r['gold'] for r in rows],[r['prediction'] for r in rows])
    for key,val in metric.items():
        if isinstance(val,(int,float)) and key in saved:assert abs(saved[key]-val)<1e-9,(p,key,saved[key],val)
    if (p/'metadata.json').exists():
        meta=C.read(p/'metadata.json');assert meta.get('status','completed')=='completed'
        checks=meta.get('postrun_checks',{})
        for k,v in checks.items():
            if isinstance(v,bool):assert v,(p,k)
    spec=C.read(p/'resolved_config.json').get('model_spec',{})
    if not spec and task in OLD:spec=C.read(p/'resolved_config.json').get('model',{})
    return {'path':str(p.relative_to(C.ROOT)),'metrics':metric,'parameters':spec.get('total_parameters')}

def stats(v):return {'mean':statistics.mean(v),'range':[min(v),max(v)]}

def main():
    runs={};table=[];kd_ce={}
    for task in OLD+NEW:
        runs[task]={arm:[recalc(task,path(task,arm,s)) for s in C.SEEDS] for arm in ('bami','student')}
        for field in FIELDS[task]:
            b=[100*r['metrics'][field] for r in runs[task]['bami']];s=[100*r['metrics'][field] for r in runs[task]['student']]
            table.append({'dataset':task,'indicator':field,'bami':stats(b),'student_e7_kd':stats(s),'student_minus_bami':statistics.mean(s)-statistics.mean(b),'bami_parameters':runs[task]['bami'][0]['parameters'],'student_parameters':runs[task]['student'][0]['parameters']})
    assert len(table)==15
    for task in NEW:
        ce=[recalc(task,path(task,'ce',s)) for s in C.SEEDS]
        kd=runs[task]['student'];field='f1' if task=='phoner' else 'macro_f1'
        cv=[100*r['metrics'][field] for r in ce];kv=[100*r['metrics'][field] for r in kd];delta=[a-b for a,b in zip(kv,cv)]
        kd_ce[task]={'indicator':field,'ce':stats(cv),'kd':stats(kv),'delta':stats(delta),'positive_seeds':sum(x>0 for x in delta),'per_seed_delta':dict(zip(map(str,C.SEEDS),delta)),'ce_runs':ce}
    teacher={t:C.read(C.ART/'soft'/f'{t}_teacher_stats.json') for t in NEW}
    gpu_seconds=0.
    for p in list(C.ART.glob('teacher/*/fold_*/metadata.json'))+list(C.ART.glob('student/*/seed_*/metadata.json'))+list(C.ART.glob('aborted/**/metadata.json')):
        gpu_seconds+=C.read(p).get('elapsed_seconds',0)
    for p in C.ART.glob('aborted/**/status.json'):
        if not (p.parent/'metadata.json').exists():gpu_seconds+=C.read(p).get('elapsed_seconds',0)
    smoke=C.read(C.ART/'phoner/smoke_checks.json');gpu_seconds+=smoke['teacher_elapsed_seconds']+smoke['student_elapsed_seconds']
    assert gpu_seconds<=18*3600,gpu_seconds/3600
    result={'scope':'validation only','strategy':'E7 LLRD gamma 0.8 + Cafe 5-fold OOF KD alpha 0.5 temperature 2','runs':runs,'kd_minus_ce':kd_ce,'teacher':teacher,'table':table,'gpu_seconds':gpu_seconds,'gpu_hours':gpu_seconds/3600}
    C.write(C.ART/'summary.json',result)
    lines=['# BENCH8 KD — validation only','','## KD − CE (3 task mới)','','| Task | Chỉ số | CE mean (range) | KD mean (range) | KD − CE mean (range) | Seed dương |','|---|---|---:|---:|---:|---:|']
    fmt=lambda x:f"{x['mean']} ({x['range'][0]}–{x['range'][1]})"
    for task,v in kd_ce.items():lines.append(f"| {task} | {v['indicator']} | {fmt(v['ce'])} | {fmt(v['kd'])} | {fmt(v['delta'])} | {v['positive_seeds']}/3 |")
    lines+=['','## 8 dataset / 15 chỉ số','','| Dataset | Chỉ số | Bami mean (range) | Student E7 + KD mean (range) | Student − Bami | Tham số Bami / Student |','|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['dataset']} | {r['indicator']} | {fmt(r['bami'])} | {fmt(r['student_e7_kd'])} | {r['student_minus_bami']} | {r['bami_parameters']} / {r['student_parameters']} |")
    (C.ART/'table_8datasets_kd.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
