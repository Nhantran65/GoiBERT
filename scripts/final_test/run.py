#!/usr/bin/env python3
"""Frozen 48-checkpoint validation gate, one-time test, and reporting."""
from __future__ import annotations
import csv, hashlib, json, os, sys, time, traceback, gc
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'scripts',ROOT/'scripts/bench8',ROOT/'scripts/bench8_kd'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import pandas as pd
from transformers import AutoTokenizer
from goibert import train as T
from goibert.modeling import build_classifier
from goibert_e14.modeling import build_bamibert_classifier
import run_e14 as E
import data as D
import train as B
import common as C
import scripts.bench8_kd.run as K

ART=ROOT/'artifacts/final_test'; DOC=ROOT/'docs/FINAL_TEST_RESULT.md'
OLD=('vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel')
NEW=('vispam','vinli','phoner'); TASKS=OLD+NEW; SEEDS=(42,43,44)
LR_OLD={'vsfc_sentiment':2e-5,'vsfc_topic':1e-5,'visfd':5e-5,'absa_restaurant':5e-5,'absa_hotel':5e-5}
LR_NEW={'vispam':2e-5,'vinli':1e-5,'phoner':5e-5}
TEST_SHA={'vinli':'9ad06bba2d622b282464076e1ccd797c3a47dcc478643d151381498968364188','phoner':'4b677028d6db5e1dab74d727d88b719454da60cfeaba681d9ee401fc83b04c77'}
PAPER={'vsfc_sentiment':(93.86,83.41),'vsfc_topic':(89.34,79.90),'visfd':(89.14,84.24),'absa_restaurant':(88.01,74.89),'absa_hotel':(79.99,72.65),'vispam':(90.76,78.20),'vinli':(81.01,81.15),'phoner':(94.90,)}
METRICS={'vsfc_sentiment':('accuracy','macro_f1'),'vsfc_topic':('accuracy','macro_f1'),'visfd':('detection_micro_f1','pair_micro_f1'),'absa_restaurant':('detection_micro_f1','pair_micro_f1'),'absa_hotel':('detection_micro_f1','pair_micro_f1'),'vispam':('accuracy','macro_f1'),'vinli':('accuracy','macro_f1'),'phoner':('f1',)}
CODE_COMMON=['src/goibert/modeling.py','src/goibert_e14/modeling.py','src/goibert/train.py','src/goibert/metrics.py','scripts/bench8/train.py','scripts/bench8/ner.py','scripts/bench8_kd/common.py','scripts/bench8/data.py']

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def read(p):return json.loads(p.read_text())
def code_paths(model,task,seed):
 if task in OLD:return ['scripts/cafe_oof_kd/student.py'] if model=='student' else (['scripts/run_e14.py'] if seed==42 else ['scripts/run_e14.py','scripts/cafe_oof_kd/bami_seed.py'])
 return ['scripts/bench8_kd/run.py'] if model=='student' else ['scripts/bench8/train.py']
def path_for(model,task,seed):
 if task in OLD:
  if model=='student':return ROOT/f'artifacts/cafe_oof_kd/student/kd/{task}/seed_{seed}'
  if seed==42:return ROOT/f'artifacts/e14/trials/{task}/lr_{LR_OLD[task]:.0e}/seed_42'
  return ROOT/f'artifacts/cafe_oof_kd/bami/{task}/seed_{seed}'
 if model=='student':return ROOT/f'artifacts/bench8_kd/student/{task}/seed_{seed}'
 return ROOT/f'artifacts/bench8/runs/{task}/bami/lr_{LR_NEW[task]:.0e}/seed_{seed}'
def specs():
 out=[]
 for task in TASKS:
  for model in ('bami','student'):
   for seed in SEEDS:
    folder=path_for(model,task,seed); meta=read(folder/'metadata.json'); saved=read(folder/'best_validation_metrics.json')
    cfg=read(folder/'resolved_config.json');checkpoint=folder/'best_checkpoint.pt'
    assert checkpoint.is_file() and meta['status']=='completed',(folder,meta.get('status'))
    assert int(meta['seed'])==seed and int(meta['best_epoch'])>0
    assert meta['learning_rate']==cfg['learning_rate']
    assert meta['precision']=='bf16' and int(meta['max_length'])==(512 if task=='phoner' else 256)
    assert abs(float(meta['best_selection_f1'])-float(saved['selection_f1']))<1e-12
    if model=='bami':assert abs(meta['learning_rate']-(LR_OLD[task] if task in OLD else LR_NEW[task]))<1e-15
    cfiles=code_paths(model,task,seed)+CODE_COMMON
    out.append({'model':model,'task':task,'seed':seed,'learning_rate':meta['learning_rate'],'best_epoch':meta['best_epoch'],'val_metric':saved['selection_f1'],'val_metrics':{k:saved[k] for k in METRICS[task]},'checkpoint':str(checkpoint.relative_to(ROOT)),'source_code_sha256':{p:sha(ROOT/p) for p in sorted(set(cfiles))},'checkpoint_sha256':sha(checkpoint),'parameters':int((cfg.get('model_spec') or cfg.get('model'))['total_parameters'])})
 assert len(out)==48
 return out

class Evaluator:
 def __init__(self,device):self.device=device;self.cache={};self.last=None
 def prepare(self,model,task,split,batch):
  key=(model,task,split,batch)
  if key in self.cache:return self.cache[key]
  backbone=ROOT/'models'/('goiturn-s-gd' if model=='student' else 'bamibert-base-original')
  tok=AutoTokenizer.from_pretrained(backbone,local_files_only=True,use_fast=True)
  if task in OLD:
   bundle=T.load_task_bundle(task,ROOT,include_test=(split=='test'))
   sp=T.split_field(bundle,'validation' if split=='validation' else 'test')
   loader=T._make_loader(tok,sp,max_length=256,batch_size=batch,shuffle=False,seed=42,labels=T.model_labels(bundle,split,sp),gold_labels=T.split_labels(sp))
   labels=None
  else:
   bundle=D.load(task)
   if split=='validation':sp=bundle.validation
   else:sp=load_new_test(task)
   labels=bundle.label_names
   loader,_=B.make_custom_loader(sp,task,tok,labels,batch,42,False,512 if task=='phoner' else 256)
  self.cache[key]=(bundle,loader,labels,sp)
  return self.cache[key]
 def model(self,model,task,bundle,labels):
  backbone=ROOT/'models'/('goiturn-s-gd' if model=='student' else 'bamibert-base-original')
  if task in OLD:
   kw=dict(num_classes=T.task_num_classes(bundle),num_aspects=T.task_num_aspects(bundle),aspect_class_counts=T.task_aspect_class_counts(bundle) if T.task_num_aspects(bundle) else None,attention_implementation='sdpa')
   if model=='student':m,_=build_classifier(backbone,local_files_only=True,**kw)
   else:m,_=build_bamibert_classifier(backbone,**kw)
  elif model=='student':m,_=K.model_for('student',task,len(labels))
  elif task=='phoner':m=B.TokenModel(backbone,len(labels))
  else:m,_=build_bamibert_classifier(backbone,num_classes=len(labels),attention_implementation='sdpa')
  return m
 def run(self,entry,split):
  task=entry['task'];model=entry['model']
  cfg=read(ROOT/entry['checkpoint'].replace('best_checkpoint.pt','resolved_config.json'))
  batch=int(cfg.get('micro_batch_size',cfg.get('micro_batch',cfg.get('batch_size',32))))
  bundle,loader,labels,sp=self.prepare(model,task,split,batch)
  m=self.model(model,task,bundle,labels)
  ck=ROOT/entry['checkpoint']; assert sha(ck)==entry['checkpoint_sha256']
  state=torch.load(ck,map_location='cpu',weights_only=False,mmap=True)
  assert int(state['epoch'])==entry['best_epoch']
  m.load_state_dict(state['model'],strict=True);del state;m.to(self.device).eval()
  if task in OLD:metrics,rows=T.run_validation(m,loader,self.device,'bf16',task,bundle,split_name=split)
  else:metrics,rows=B.validation(m,loader,task,labels,self.device,'bf16')
  assert [r['id'] for r in rows]==(list(T.split_ids(sp)) if task in OLD else sp.ids)
  del m;gc.collect();torch.cuda.empty_cache()
  return metrics,rows

def compare(saved,actual,task):
 keys=list(METRICS[task])+['selection_f1']
 return {k:abs(float(saved[k])-float(actual[k])) for k in keys}

def load_new_test(task):
 # Only called after locked_manifest.json has been written.
 assert (ART/'locked_manifest.json').is_file()
 if task=='vispam':
  p=ROOT/'data/raw/social/vispamdetection/test.csv'
  with p.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
  assert all('SpamLabel' in r for r in rows)
  return D.Split(D._ids(task,'test',len(rows)),[r['Comment'] for r in rows],[int(r['SpamLabel']) for r in rows])
 if task=='vinli':
  p=ROOT/'data/raw/vinli_mirror/test.parquet';frame=pd.read_parquet(p)
  return D.Split(D._ids(task,'test',len(frame)),None,[D.LABELS_NLI.index(x) for x in frame.gold_label],pairs=list(zip(frame.sentence1.astype(str),frame.sentence2.astype(str))))
 p=ROOT/'data/raw/phoner_covid19/test_syllable.json'
 with p.open(encoding='utf-8') as f:rows=[json.loads(line) for line in f if line.strip()]
 assert all(len(r['words'])==len(r['tags']) for r in rows)
 return D.Split(D._ids(task,'test',len(rows)),None,[r['tags'] for r in rows],words=[r['words'] for r in rows])

def test_hashes():
 assert (ART/'locked_manifest.json').is_file()
 files={'vispam':'data/raw/social/vispamdetection/test.csv','vinli':'data/raw/vinli_mirror/test.parquet','phoner':'data/raw/phoner_covid19/test_syllable.json'}
 result={task:{'path':p,'sha256':sha(ROOT/p)} for task,p in files.items()}
 for task,want in TEST_SHA.items():assert result[task]['sha256']==want,(task,result[task]['sha256'])
 return result

def from_file(task,model,p):
 rows=read(p)['rows'];gold=[r['gold'] for r in rows];pred=[r['prediction'] for r in rows]
 assert len(rows)==len({r['id'] for r in rows})
 if task in OLD:
  bundle=T.load_task_bundle(task,ROOT,include_test=True)
  return T.evaluate_predictions(task,bundle,gold,pred,split='test')
 labels=(['0','1','2','3'] if task=='vispam' else list(D.LABELS_NLI) if task=='vinli' else list(C.PHONER_LABELS))
 m=C.evaluate(task,labels,gold,pred);m['split']='test'
 return m

def main():
 ART.mkdir(parents=True,exist_ok=True)
 assert not (ART/'locked_manifest.json').exists(),'Manifest already exists; refusing second test run'
 started=time.time();device=torch.device('cuda:0');assert torch.cuda.mem_get_info(device)[0]>=12e9
 entries=specs();ev=Evaluator(device);checks=[]
 for i,e in enumerate(entries,1):
  actual,rows=ev.run(e,'validation');saved=read(ROOT/e['checkpoint'].replace('best_checkpoint.pt','best_validation_metrics.json'))
  diffs=compare(saved,actual,e['task'])
  oldrows=read(ROOT/e['checkpoint'].replace('best_checkpoint.pt','validation_predictions.json'))['rows']
  exact=rows==oldrows
  c={'model':e['model'],'task':e['task'],'seed':e['seed'],'metric_abs_diffs':diffs,'predictions_identical':exact,'n':len(rows),'passed':max(diffs.values())<1e-6 and exact}
  checks.append(c);write(ART/'validation_checks.json',{'completed':len(checks),'expected':48,'checks':checks})
  print(f"VALIDATION {i}/48 {e['model']} {e['task']} {e['seed']} maxdiff={max(diffs.values())} predictions={exact}",flush=True)
  if not c['passed']:raise RuntimeError('Validation FAIL; test remains locked')
 assert len(checks)==48 and all(c['passed'] for c in checks)
 locked={'protocol':'final-test-48-v1','locked_at':now(),'checkpoint_count':48,'validation_tolerance':1e-6,'validation_passed':True,'entries':entries}
 write(ART/'locked_manifest.json',locked)
 print('MANIFEST_LOCKED',flush=True)
 hashes=test_hashes();log={'locked_manifest_sha256':sha(ART/'locked_manifest.json'),'test_files':hashes,'entries':[]}
 write(ART/'test_access_log.json',log)
 results=[]
 for i,e in enumerate(entries,1):
  k=f"{e['model']}_{e['task']}_seed_{e['seed']}";pred=ART/'predictions'/f'{k}.json';met=ART/'metrics'/f'{k}.json'
  if pred.exists():raise RuntimeError(f'Refusing test repeat {k}')
  access={'model':e['model'],'task':e['task'],'seed':e['seed'],'checkpoint_sha256':e['checkpoint_sha256'],'accessed_at':now(),'status':'started','attempt':1}
  log['entries'].append(access);write(ART/'test_access_log.json',log)
  try:
   metrics,rows=ev.run(e,'test');write(pred,{'model':e['model'],'task':e['task'],'seed':e['seed'],'rows':rows})
   recomputed=from_file(e['task'],e['model'],pred)
   diffs=compare(metrics,recomputed,e['task']);assert max(diffs.values())<1e-12
   write(met,recomputed);access.update({'status':'completed','prediction_file':str(pred.relative_to(ROOT)),'prediction_sha256':sha(pred),'metric_file':str(met.relative_to(ROOT)),'n':len(rows),'completed_at':now()})
   results.append({**e,'test_metrics':recomputed})
   print(f"TEST {i}/48 {k} n={len(rows)} selection={recomputed['selection_f1']}",flush=True)
  except Exception as ex:
   access.update({'status':'failed','error':repr(ex),'traceback':traceback.format_exc(),'failed_at':now()});write(ART/'test_access_log.json',log);raise
  write(ART/'test_access_log.json',log)
 write(ART/'test_results.json',{'completed':len(results),'expected':48,'results':results,'elapsed_seconds':time.time()-started})
 print('TEST_COMPLETE',flush=True)

if __name__=='__main__':main()
