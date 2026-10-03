#!/usr/bin/env python3
"""Recompute family SM validation metrics from prediction files."""
import importlib.util,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/family_sm'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/bench8_kd'),str(ROOT/'scripts/bench8')]
from goibert import train as T

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
B=module('family_bench_common',ROOT/'scripts/bench8_kd/common.py')
OLD=('vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel')
NEW=('vispam','vinli','phoner')
TASKS=OLD+NEW
FIRST=('vsfc_sentiment','absa_restaurant','vispam','vinli','phoner')
SEEDS=(42,43,44)
STUDENT_LR={'vsfc_sentiment':1e-5,'vsfc_topic':1e-5,'visfd':1e-4,'absa_restaurant':1e-4,'absa_hotel':1e-4,'vispam':5e-5,'vinli':1e-4,'phoner':1e-4}
BAMI_LR={'vsfc_sentiment':2e-5,'vsfc_topic':1e-5,'visfd':5e-5,'absa_restaurant':5e-5,'absa_hotel':5e-5,'vispam':2e-5,'vinli':1e-5,'phoner':5e-5}
METRICS={'vsfc_sentiment':('accuracy','macro_f1'),'vsfc_topic':('accuracy','macro_f1'),'visfd':('detection_micro_f1','pair_micro_f1'),'absa_restaurant':('detection_micro_f1','pair_micro_f1'),'absa_hotel':('detection_micro_f1','pair_micro_f1'),'vispam':('accuracy','macro_f1'),'vinli':('accuracy','macro_f1'),'phoner':('f1',)}

def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def lr_tag(lr):return f'{lr:.0e}'
def run_dir(model,task,seed,lr):return ART/'finetune'/model/task/f'lr_{lr_tag(lr)}'/f'seed_{seed}'
def eval_dir(p,task,require_checkpoint=False):
 needed=('validation_predictions.json','best_validation_metrics.json','metadata.json','status.json','resolved_config.json')
 if not all((p/f).is_file() for f in needed):return None
 pred=read(p/'validation_predictions.json');saved=read(p/'best_validation_metrics.json');meta=read(p/'metadata.json');status=read(p/'status.json');cfg=read(p/'resolved_config.json')
 assert status['status']=='completed' and all(meta.get('postrun_checks',{}).values()),p
 if require_checkpoint:assert (p/'best_checkpoint.pt').is_file(),p
 rows=pred['rows'];gold=[r['gold'] for r in rows];guess=[r['prediction'] for r in rows]
 if task in OLD:
  bundle=T.load_task_bundle(task,ROOT,include_test=False)
  metrics=T.evaluate_predictions(task,bundle,gold,guess,split='validation')
 else:
  data=B.load(task);metrics=B.evaluate(task,tuple(data.label_names),gold,guess)
 for k,value in metrics.items():
  if k in saved and isinstance(value,(int,float)):
   assert abs(float(value)-float(saved[k]))<1e-9,(p,k,value,saved[k])
 score=float(metrics['selection_f1']);assert math.isfinite(score)
 assert abs(score-float(meta['best_selection_f1']))<1e-9,p
 return {'selection_f1':score,'f1_points':100*score,'metrics':metrics,'best_epoch':meta['best_epoch'],'learning_rate':meta['learning_rate'],'backbone':meta.get('backbone',cfg.get('backbone')),'elapsed_seconds':meta['elapsed_seconds'],'path':str(p)}
def summary(values):return {'mean':statistics.mean(values),'range':[min(values),max(values)],'seeds':values}
def main():
 decisions=read(ART/'decisions.json') if (ART/'decisions.json').exists() else {}
 groups={}
 for model in ('s_prime','a4','m6','m12'):
  groups[model]={}
  for task in TASKS:
   groups[model][task]={}
   for lr in sorted(set((STUDENT_LR[task],BAMI_LR[task]))):
    vals=[eval_dir(run_dir(model,task,seed,lr),task,False) for seed in SEEDS]
    if any(v is not None for v in vals):groups[model][task][lr_tag(lr)]={str(seed):v for seed,v in zip(SEEDS,vals)}
 validation={};metric_table={}
 def reference_path(model,task,seed):
  if model=='a4_reference':
   new=run_dir('a4',task,seed,STUDENT_LR[task])
   if (new/'status.json').is_file() and read(new/'status.json').get('status')=='completed':return new
   return ROOT/f'artifacts/gd_ablation/finetune/A4/{task}/seed_{seed}' if task in FIRST else None
  if task in OLD:
   if model=='s_original':return ROOT/f'artifacts/cafe_oof_kd/student/kd/{task}/seed_{seed}'
   if seed==42:return ROOT/f'artifacts/e14/trials/{task}/lr_{BAMI_LR[task]:.0e}/seed_42'
   return ROOT/f'artifacts/cafe_oof_kd/bami/{task}/seed_{seed}'
  return ROOT/f'artifacts/bench8_kd/student/{task}/seed_{seed}' if model=='s_original' else ROOT/f'artifacts/bench8/runs/{task}/bami/lr_{BAMI_LR[task]:.0e}/seed_{seed}'
 for model in ('bami','s_original','a4_reference'):
  validation[model]={}
  for task in TASKS:
   vals=[]
   for seed in SEEDS:
    p=reference_path(model,task,seed)
    v=eval_dir(p,task) if p is not None else None;vals.append(v['f1_points'] if v else None)
   if all(v is not None for v in vals):validation[model][task]=summary(vals)
 for label,model in [('s_prime',decisions.get('s_official','s_prime')),('m',decisions.get('m_shape'))]:
  if not model:continue
  validation[label]={}
  for task in TASKS:
   lr=(decisions.get('m_lr',{}).get(task) if label=='m' else STUDENT_LR[task])
   if lr is None:continue
   vals=[groups[model][task].get(lr_tag(lr),{}).get(str(seed)) for seed in SEEDS]
   if all(vals):validation[label][task]=summary([v['f1_points'] for v in vals])
 if len(validation.get('s_prime',{}))==8 and len(validation.get('m',{}))==8:
  validation['complete']=True
 for task in TASKS:
  for metric in METRICS[task]:
   key=f'{task}/{metric}';metric_table[key]={}
   for label in ('bami','s_original','a4_reference','s_prime','m'):
    vals=[]
    for seed in SEEDS:
     if label in ('bami','s_original','a4_reference'):
      p=reference_path(label,task,seed);v=eval_dir(p,task) if p is not None else None
     else:
      variant=decisions.get('s_official','s_prime') if label=='s_prime' else decisions.get('m_shape')
      lr=STUDENT_LR[task] if label=='s_prime' else decisions.get('m_lr',{}).get(task)
      v=eval_dir(run_dir(variant,task,seed,lr),task) if variant and lr else None
     vals.append(100*float(v['metrics'][metric]) if v else None)
    if all(v is not None for v in vals):metric_table[key][label]=summary(vals)
 write(ART/'validation_summary.json',{'groups':groups,'validation':validation,'metric_table':metric_table,'decisions':decisions})
 print(json.dumps({'complete':validation.get('complete',False),'tasks':{k:len(v) for k,v in validation.items() if isinstance(v,dict)}},indent=2))
if __name__=='__main__':main()
