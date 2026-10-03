#!/usr/bin/env python3
"""Recompute validation metrics and locked decision inputs; never select a recipe."""
import importlib.util,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'artifacts/gd_ablation'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/bench8_kd'),str(ROOT/'scripts/bench8')]
from goibert import train as T

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
B=load_module('bench_common',ROOT/'scripts/bench8_kd/common.py')
TASKS=('vsfc_sentiment','absa_restaurant','vispam','vinli','phoner');ARMS=('R0','A1','A2','A3','A4','A5');SEEDS=(42,43,44)
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def source(arm,task,seed):
    if arm!='R0':return ART/'finetune'/arm/task/f'seed_{seed}'
    if task in ('vsfc_sentiment','absa_restaurant'):return ROOT/'artifacts/cafe_oof_kd/student/kd'/task/f'seed_{seed}'
    return ROOT/'artifacts/bench8_kd/student'/task/f'seed_{seed}'
def eval_run(arm,task,seed):
    p=source(arm,task,seed);files=('validation_predictions.json','best_validation_metrics.json','metadata.json','status.json')
    if not all((p/f).exists() for f in files):return None
    pred=read(p/files[0]);saved=read(p/files[1]);meta=read(p/files[2]);status=read(p/files[3]);rows=pred['rows'];gold=[r['gold'] for r in rows];guess=[r['prediction'] for r in rows]
    if task in ('vsfc_sentiment','absa_restaurant'):
        bundle=T.load_task_bundle(task,ROOT,include_test=False)
        metric=T.evaluate_predictions(task,bundle,gold,guess,split='validation')
    else:
        data=B.load(task);metric=B.evaluate(task,tuple(data.label_names),gold,guess)
    value=float(metric['selection_f1']);assert math.isfinite(value)
    assert abs(value-float(saved['selection_f1']))<1e-9,(p,'metrics mismatch',value,saved['selection_f1'])
    assert abs(value-float(meta['best_selection_f1']))<1e-9,(p,'metadata mismatch')
    assert status['status']=='completed' and all(meta['postrun_checks'].values()),p
    return {'f1':value*100,'best_epoch':meta['best_epoch'],'elapsed_seconds':meta['elapsed_seconds'],'path':str(p)}
def main():
    per={};task_summary={};points={};curves={};missing=[]
    for arm in ARMS:
        per[arm]={};task_summary[arm]={}
        for task in TASKS:
            seed_data={str(s):eval_run(arm,task,s) for s in SEEDS};per[arm][task]=seed_data
            values=[v['f1'] for v in seed_data.values() if v]
            if len(values)<3:missing.append(f'{arm}/{task}');continue
            task_summary[arm][task]={'mean':statistics.mean(values),'range':[min(values),max(values)]}
        if len(task_summary[arm])==5:
            points[arm]={'P':statistics.mean(v['mean'] for v in task_summary[arm].values()),'P_without_ViNLI':statistics.mean(v['mean'] for t,v in task_summary[arm].items() if t!='vinli')}
    for arm in ARMS[1:]:
        path=ART/'backbones'/arm/'loss_curve.jsonl'
        if path.exists():curves[arm]=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    deltas={}
    for a,b in [('A2','A1'),('A3','A1'),('A4','A1'),('A5','A1'),('A1','R0')]:
        if a in points and b in points:deltas[f'{a}-{b}']=points[a]['P']-points[b]['P']
    decision={'port':None,'factors':{},'dapt':None}
    if 'A1-R0' in deltas:decision['port']={'abs_delta_P':abs(deltas['A1-R0']),'exceeds_1_point':abs(deltas['A1-R0'])>1.0}
    for arm in ('A2','A3','A4','A5'):
        if arm not in points or 'A1' not in points:continue
        drops={t:task_summary[arm][t]['mean']-task_summary['A1'][t]['mean'] for t in TASKS}
        decision['factors'][arm]={'delta_P':deltas[f'{arm}-A1'],'delta_by_task':drops,'delta_at_least_0_5':deltas[f'{arm}-A1']>=.5,'no_task_drop_below_minus_1':all(x>=-1 for x in drops.values()),'passes':deltas[f'{arm}-A1']>=.5 and all(x>=-1 for x in drops.values())}
    if 'A5' in points and 'A1' in points:decision['dapt']={'A5_minus_A1':deltas['A5-A1'],'A5_at_least_A1_minus_0_3':deltas['A5-A1']>=-.3}
    result={'unit':'F1 percentage points','per_seed':per,'task_summary':task_summary,'points':points,'deltas':deltas,'missing':missing,'gd_loss_curves':curves}
    write(ART/'summary.json',result);write(ART/'decision_inputs.json',decision)
    print(json.dumps({'points':points,'deltas':deltas,'missing':missing,'decision_inputs':decision},indent=2))
if __name__=='__main__':main()
