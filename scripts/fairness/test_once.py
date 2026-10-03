#!/usr/bin/env python3
"""Freeze 96 validated checkpoints, then access test once using final_test protocol."""
from __future__ import annotations
import csv,gc,hashlib,importlib.util,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
for p in (ROOT,ROOT/'src',ROOT/'scripts',ROOT/'scripts/bench8',ROOT/'scripts/bench8_kd',HERE):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
sys.path.remove(str(HERE));sys.path.insert(0,str(HERE))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from summarize import OLD,TASKS,SEEDS,STUDENT_LR,run_dir,eval_dir,read,write,B as KB
import torch,pandas as pd
from transformers import AutoTokenizer
from goibert import train as T
from goibert.modeling import build_classifier
from goibert_e14.modeling import build_bamibert_classifier
import data as D,train as B
ART=ROOT/'artifacts/fairness';OUT=ART/'test'
TEST_SHA={'vinli':'9ad06bba2d622b282464076e1ccd797c3a47dcc478643d151381498968364188','phoner':'4b677028d6db5e1dab74d727d88b719454da60cfeaba681d9ee401fc83b04c77'}
METRICS={'vsfc_sentiment':('accuracy','macro_f1'),'vsfc_topic':('accuracy','macro_f1'),'visfd':('detection_micro_f1','pair_micro_f1'),'absa_restaurant':('detection_micro_f1','pair_micro_f1'),'absa_hotel':('detection_micro_f1','pair_micro_f1'),'vispam':('accuracy','macro_f1'),'vinli':('accuracy','macro_f1'),'phoner':('f1',)}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def specs():
 assert read(ART/'launcher_summary.json')['status']=='validation_complete'
 d=read(ART/'decisions.json');out=[]
 for task in TASKS:
  for label in ('s_prime_ce','m_ce','bami_e7_ce','bami_e7_kd'):
   if label.startswith('bami'):
    arm=label.rsplit('_',1)[1];lr=d['bami_lr'][f'bami_e7_{arm}/{task}']['chosen_lr'];backbone=ROOT/'models/bamibert-base-original'
   else:
    variant='s_prime' if label=='s_prime_ce' else 'm12'
    lr=STUDENT_LR[task] if variant=='s_prime' else read(ROOT/'artifacts/family_sm/decisions.json')['m_lr'][task]
    backbone=ROOT/'artifacts/family_sm/backbones'/variant/'final'
   for seed in SEEDS:
    folder=run_dir(label,task,seed,lr);v=eval_dir(folder,task,True)
    assert v and Path(v['backbone']).resolve()==backbone.resolve(),folder
    checkpoint=folder/'best_checkpoint.pt';saved=read(folder/'best_validation_metrics.json')
    cfg=read(folder/'resolved_config.json')
    assert bool(cfg['kd']) == (label=='bami_e7_kd'),folder
    out.append({'model':label,'task':task,'seed':seed,'learning_rate':lr,'best_epoch':v['best_epoch'],'val_metric':v['selection_f1'],'val_metrics':{k:saved[k] for k in METRICS[task]},'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint),'backbone':str(backbone.resolve()),'parameters':int(cfg['model_spec']['total_parameters'])})
 assert len(out)==96
 return out
class Evaluator:
 def __init__(self,device):self.device=device;self.cache={}
 def prepare(self,e,split,batch):
  task=e['task'];backbone=Path(e['backbone']);key=(str(backbone),task,split,batch)
  if key in self.cache:return self.cache[key]
  tok=AutoTokenizer.from_pretrained(backbone,local_files_only=True,use_fast=True)
  if task in OLD:
   bundle=T.load_task_bundle(task,ROOT,include_test=(split=='test'));sp=T.split_field(bundle,split)
   loader=T._make_loader(tok,sp,max_length=256,batch_size=batch,shuffle=False,seed=42,labels=T.model_labels(bundle,split,sp),gold_labels=T.split_labels(sp));labels=None
  else:
   bundle=D.load(task);sp=bundle.validation if split=='validation' else load_new_test(task);labels=bundle.label_names
   loader,_=B.make_custom_loader(sp,task,tok,labels,batch,42,False,512 if task=='phoner' else 256)
  self.cache[key]=(bundle,loader,labels,sp);return self.cache[key]
 def run(self,e,split):
  task=e['task'];cfg=read(ROOT/e['checkpoint'].replace('best_checkpoint.pt','resolved_config.json'))
  batch=int(cfg.get('micro_batch_size',cfg.get('micro_batch',cfg.get('batch_size',32))))
  bundle,loader,labels,sp=self.prepare(e,split,batch);backbone=Path(e['backbone'])
  if task in OLD:
   n=T.task_num_aspects(bundle);kw=dict(num_classes=T.task_num_classes(bundle),num_aspects=n,aspect_class_counts=T.task_aspect_class_counts(bundle) if n else None,attention_implementation='sdpa')
   m,_=(build_bamibert_classifier(backbone,**kw) if e['model'].startswith('bami') else build_classifier(backbone,local_files_only=True,**kw))
  elif task=='phoner':m=B.TokenModel(backbone,len(labels))
  elif e['model'].startswith('bami'):m,_=build_bamibert_classifier(backbone,num_classes=len(labels),attention_implementation='sdpa')
  else:m,_=build_classifier(backbone,num_classes=len(labels),local_files_only=True,attention_implementation='sdpa')
  ck=ROOT/e['checkpoint'];assert sha(ck)==e['checkpoint_sha256']
  state=torch.load(ck,map_location='cpu',weights_only=False,mmap=True);assert int(state['epoch'])==e['best_epoch']
  m.load_state_dict(state['model'],strict=True);del state;m.to(self.device).eval()
  if task in OLD:metrics,rows=T.run_validation(m,loader,self.device,'bf16',task,bundle,split_name=split)
  else:metrics,rows=B.validation(m,loader,task,labels,self.device,'bf16')
  assert [r['id'] for r in rows]==(list(T.split_ids(sp)) if task in OLD else sp.ids)
  del m;gc.collect();torch.cuda.empty_cache();return metrics,rows
def compare(saved,actual,task):return {k:abs(float(saved[k])-float(actual[k])) for k in (*METRICS[task],'selection_f1')}
def load_new_test(task):
 assert (OUT/'locked_manifest.json').is_file()
 if task=='vispam':
  with (ROOT/'data/raw/social/vispamdetection/test.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
  return D.Split(D._ids(task,'test',len(rows)),[r['Comment'] for r in rows],[int(r['SpamLabel']) for r in rows])
 if task=='vinli':
  frame=pd.read_parquet(ROOT/'data/raw/vinli_mirror/test.parquet')
  return D.Split(D._ids(task,'test',len(frame)),None,[D.LABELS_NLI.index(x) for x in frame.gold_label],pairs=list(zip(frame.sentence1.astype(str),frame.sentence2.astype(str))))
 with (ROOT/'data/raw/phoner_covid19/test_syllable.json').open(encoding='utf-8') as f:rows=[json.loads(line) for line in f if line.strip()]
 return D.Split(D._ids(task,'test',len(rows)),None,[r['tags'] for r in rows],words=[r['words'] for r in rows])
def test_hashes():
 assert (OUT/'locked_manifest.json').is_file()
 files={'vispam':'data/raw/social/vispamdetection/test.csv','vinli':'data/raw/vinli_mirror/test.parquet','phoner':'data/raw/phoner_covid19/test_syllable.json'}
 result={task:{'path':p,'sha256':sha(ROOT/p)} for task,p in files.items()}
 for task,want in TEST_SHA.items():assert result[task]['sha256']==want
 return result
def from_file(task,p):
 rows=read(p)['rows'];gold=[r['gold'] for r in rows];pred=[r['prediction'] for r in rows]
 assert len(rows)==len({r['id'] for r in rows})
 if task in OLD:
  bundle=T.load_task_bundle(task,ROOT,include_test=True)
  return T.evaluate_predictions(task,bundle,gold,pred,split='test')
 labels=(['0','1','2','3'] if task=='vispam' else list(D.LABELS_NLI) if task=='vinli' else list(KB.PHONER_LABELS))
 metrics=KB.evaluate(task,labels,gold,pred);metrics['split']='test';return metrics
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'locked_manifest.json').exists():raise RuntimeError('Manifest already exists; refusing second test run')
 assert read(ART/'launcher_summary.json')['status']=='validation_complete'
 started=time.time();device=torch.device('cuda:0');assert torch.cuda.mem_get_info(device)[0]>=12e9
 entries=specs();ev=Evaluator(device);checks=[]
 for i,e in enumerate(entries,1):
  actual,rows=ev.run(e,'validation');saved=read(ROOT/e['checkpoint'].replace('best_checkpoint.pt','best_validation_metrics.json'))
  diffs=compare(saved,actual,e['task']);oldrows=read(ROOT/e['checkpoint'].replace('best_checkpoint.pt','validation_predictions.json'))['rows']
  c={'model':e['model'],'task':e['task'],'seed':e['seed'],'metric_abs_diffs':diffs,'predictions_identical':rows==oldrows,'passed':max(diffs.values())<1e-6 and rows==oldrows}
  checks.append(c);write(OUT/'validation_checks.json',{'completed':len(checks),'expected':96,'checks':checks})
  print('VALIDATION',i,'/96',e['model'],e['task'],e['seed'],max(diffs.values()),flush=True)
  if not c['passed']:raise RuntimeError('Validation FAIL; test remains locked')
  e['val_metric']=float(actual['selection_f1'])
  e['val_metrics']={k:float(actual[k]) for k in METRICS[e['task']]}
  e['validation_recomputed_from_checkpoint']=True
 locked={'protocol':'fairness-96-v1','locked_at':now(),'checkpoint_count':96,'validation_tolerance':1e-6,'validation_passed':True,'entries':entries}
 write(OUT/'locked_manifest.json',locked);print('MANIFEST_LOCKED',flush=True)
 log={'locked_manifest_sha256':sha(OUT/'locked_manifest.json'),
      'test_file_hash_started_at':now(),'entries':[]}
 write(OUT/'test_access_log.json',log)
 log['test_files']=test_hashes();log['test_file_hash_completed_at']=now();write(OUT/'test_access_log.json',log)
 results=[]
 for i,e in enumerate(entries,1):
  key=f"{e['model']}_{e['task']}_seed_{e['seed']}";pred=OUT/'predictions'/f'{key}.json';met=OUT/'metrics'/f'{key}.json'
  if pred.exists():raise RuntimeError(f'Refusing test repeat {key}')
  access={'model':e['model'],'task':e['task'],'seed':e['seed'],'checkpoint_sha256':e['checkpoint_sha256'],'accessed_at':now(),'status':'started','attempt':1}
  log['entries'].append(access);write(OUT/'test_access_log.json',log)
  try:
   metrics,rows=ev.run(e,'test');write(pred,{'model':e['model'],'task':e['task'],'seed':e['seed'],'rows':rows})
   recomputed=from_file(e['task'],pred);assert max(compare(metrics,recomputed,e['task']).values())<1e-12
   write(met,recomputed);access.update({'status':'completed','prediction_file':str(pred.relative_to(ROOT)),'prediction_sha256':sha(pred),'metric_file':str(met.relative_to(ROOT)),'n':len(rows),'completed_at':now()})
   results.append({**e,'test_metrics':recomputed});print('TEST',i,'/96',key,recomputed['selection_f1'],flush=True)
  except Exception as ex:
   access.update({'status':'failed','error':repr(ex),'traceback':traceback.format_exc(),'failed_at':now(),'technical_failure_before_predictions':not pred.exists()});write(OUT/'test_access_log.json',log);raise
  write(OUT/'test_access_log.json',log)
 write(OUT/'test_results.json',{'completed':len(results),'expected':96,'results':results,'elapsed_seconds':time.time()-started});print('TEST_COMPLETE',flush=True)
if __name__=='__main__':main()
