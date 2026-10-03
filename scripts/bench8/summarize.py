#!/usr/bin/env python3
"""Recompute every validation metric and render 8-dataset, 15-indicator table."""
from __future__ import annotations
import json,sys,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from goibert import train as T
from data import load
from train import evaluate,read,write,ART,ROOT

OLD=['vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel']
NEW=['vispam','vinli','phoner']
FIELDS={'vsfc_sentiment':['accuracy','macro_f1'],'vsfc_topic':['accuracy','macro_f1'],'visfd':['detection_micro_f1','pair_micro_f1'],'absa_restaurant':['detection_micro_f1','pair_micro_f1'],'absa_hotel':['detection_micro_f1','pair_micro_f1'],'vispam':['accuracy','macro_f1'],'vinli':['accuracy','macro_f1'],'phoner':['f1']}
BAMI_LR={'vsfc_sentiment':2e-5,'vsfc_topic':1e-5,'visfd':5e-5,'absa_restaurant':5e-5,'absa_hotel':5e-5}

def old_path(task,model,seed):
 if model=='student':return ROOT/'artifacts/cafe_oof_kd/student'/('ce' if task.startswith('vsfc') else 'kd')/task/f'seed_{seed}'
 return ROOT/'artifacts/e14/trials'/task/f'lr_{BAMI_LR[task]:.0e}'/'seed_42' if seed==42 else ROOT/'artifacts/cafe_oof_kd/bami'/task/f'seed_{seed}'

def run_metric(task,model,seed,selected):
 lr_task='vispam' if task=='vispam_binary' else task
 p=old_path(task,model,seed) if task in OLD else ART/'runs'/task/model/f'lr_{selected[f"{lr_task}/{model}"]:.0e}'/f'seed_{seed}'
 saved=read(p/'best_validation_metrics.json');rows=read(p/'validation_predictions.json')['rows']
 if task in OLD:
  bundle=T.load_task_bundle(task,ROOT,include_test=False)
  metric=T.evaluate_predictions(task,bundle,[r['gold'] for r in rows],[r['prediction'] for r in rows],split='validation')
 else:metric=evaluate(task,load(task).label_names,[r['gold'] for r in rows],[r['prediction'] for r in rows])
 mismatches={k:[saved.get(k),v] for k,v in metric.items() if k in saved and isinstance(v,(int,float)) and (not isinstance(saved[k],(int,float)) or abs(v-saved[k])>1e-9)}
 if mismatches:raise AssertionError((p,mismatches))
 if task in NEW or task=='vispam_binary':
  meta=read(p/'metadata.json')
  if abs(meta['best_selection_f1']-metric['selection_f1'])>1e-9 or not all(meta['postrun_checks'].values()):raise AssertionError(p)
 config=read(p/'resolved_config.json')
 spec=config.get('model_spec',config.get('model',{}))
 return {'path':str(p.relative_to(ROOT)),'metrics':metric,'best_epoch':saved.get('epoch',read(p/'metadata.json').get('best_epoch') if (p/'metadata.json').exists() else None),'parameters':spec.get('total_parameters')}

def main():
 selected=read(ART/'selected_lrs.json');rows=[];runs={}
 all_new=[]
 for p in sorted((ART/'runs').glob('*/*/lr_*/seed_*/metadata.json')):
  folder=p.parent;meta=read(p);task=meta['task']
  saved=read(folder/'best_validation_metrics.json');pred=read(folder/'validation_predictions.json')['rows']
  recalculated=evaluate(task,load(task).label_names,[r['gold'] for r in pred],[r['prediction'] for r in pred])
  mismatches={k:[saved.get(k),v] for k,v in recalculated.items() if isinstance(v,(int,float)) and (not isinstance(saved.get(k),(int,float)) or abs(saved[k]-v)>1e-9)}
  if mismatches or abs(meta['best_selection_f1']-recalculated['selection_f1'])>1e-9 or not all(meta['postrun_checks'].values()) or read(folder/'status.json')['status']!='completed':raise AssertionError((folder,mismatches))
  all_new.append({'path':str(folder.relative_to(ROOT)),'selection_f1':recalculated['selection_f1'],'checks_pass':True})
 if len(all_new)!=36:raise AssertionError(f'expected 36 new runs, got {len(all_new)}')
 for task in OLD+NEW:
  runs[task]={}
  for model in ('bami','student'):
   rs=[run_metric(task,model,s,selected) for s in (42,43,44)];runs[task][model]=rs
  for field in FIELDS[task]:
   b=[100*r['metrics'][field] for r in runs[task]['bami']];s=[100*r['metrics'][field] for r in runs[task]['student']]
   rows.append({'dataset':task,'indicator':field,'bami_mean':statistics.mean(b),'bami_range':[min(b),max(b)],'student_mean':statistics.mean(s),'student_range':[min(s),max(s)],'student_minus_bami':statistics.mean(s)-statistics.mean(b),'bami_parameters':runs[task]['bami'][0]['parameters'],'student_parameters':runs[task]['student'][0]['parameters']})
 if len(rows)!=15:raise AssertionError(len(rows))
 binary={model:[run_metric('vispam_binary',model,s,selected) for s in (42,43,44)] for model in ('bami','student')}
 params={}
 for model in ('bami','student'):
  p=ART/'runs/vispam'/model/f'lr_{selected[f"vispam/{model}"]:.0e}'/'seed_42'/'resolved_config.json'
  params[model]=read(p)['model_spec']['total_parameters']
 gpu_seconds=sum(read(p)['elapsed_seconds'] for folder in ('runs','reproduction','aborted') for p in (ART/folder).rglob('metadata.json') if 'elapsed_seconds' in read(p))
 gpu_seconds+=sum(read(p)['elapsed_seconds'] for folder in ('runs','reproduction','aborted') for p in (ART/folder).rglob('status.json') if not (p.parent/'metadata.json').exists() and read(p).get('status')=='failed' and 'elapsed_seconds' in read(p))
 smoke=read(ART/'smoke_manifest.json') if (ART/'smoke_manifest.json').exists() else {}
 gpu_seconds+=sum(v['elapsed_seconds'] for v in smoke.values())
 truncation={}
 for task in NEW:
  truncation[task]={}
  for model in ('bami','student'):
   cfg=read(ART/'runs'/task/model/f'lr_{selected[f"{task}/{model}"]:.0e}'/'seed_42'/'resolved_config.json')
   truncation[task][model]=cfg['length_stats']
 out={'scope':'validation only','selected_lrs':selected,'parameter_counts':params,'rows':rows,'runs':runs,'all_new_runs_audit':all_new,'vispam_binary':binary,'truncation':truncation,'gpu_seconds':gpu_seconds,'gpu_hours':gpu_seconds/3600}
 write(ART/'summary.json',out)
 md=['# BENCH8 — validation, trung bình 3 seed','','| Dataset | Chỉ số | Bami TB (range) | Student S TB (range) | Student − Bami | Tham số Bami / Student |','|---|---|---:|---:|---:|---:|']
 for r in rows:
  fmt=lambda mean,range_:f'{mean} ({range_[0]}–{range_[1]})'
  md.append(f"| {r['dataset']} | {r['indicator']} | {fmt(r['bami_mean'],r['bami_range'])} | {fmt(r['student_mean'],r['student_range'])} | {r['student_minus_bami']} | {r['bami_parameters']} / {r['student_parameters']} |")
 (ART/'table_8datasets.md').write_text('\n'.join(md)+'\n')
 print('\n'.join(md))
if __name__=='__main__':main()
