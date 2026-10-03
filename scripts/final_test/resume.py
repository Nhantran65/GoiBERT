#!/usr/bin/env python3
"""Resume only incomplete test entries after a technical interruption."""
from __future__ import annotations
import os,time,traceback
from datetime import datetime
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from run import ROOT,ART,Evaluator,read,write,sha,from_file,compare,now

def main():
 torch.set_num_threads(8);torch.set_num_interop_threads(8)
 started=time.time();locked=read(ART/'locked_manifest.json');log=read(ART/'test_access_log.json')
 assert sha(ART/'locked_manifest.json')==log['locked_manifest_sha256']
 assert len(locked['entries'])==48
 ev=Evaluator(torch.device('cuda:0'));results=[]
 for i,e in enumerate(locked['entries'],1):
  k=f"{e['model']}_{e['task']}_seed_{e['seed']}";pred=ART/'predictions'/f'{k}.json';met=ART/'metrics'/f'{k}.json'
  if met.exists():
   matches=[x for x in log['entries'] if x['model']==e['model'] and x['task']==e['task'] and x['seed']==e['seed']]
   assert len(matches)==1 and matches[0]['status']=='completed'
   results.append({**e,'test_metrics':read(met)});continue
  started_entries=[x for x in log['entries'] if x['model']==e['model'] and x['task']==e['task'] and x['seed']==e['seed']]
  if pred.exists():
   assert len(started_entries)==1 and started_entries[0]['status']=='started'
   rows=read(pred)['rows'];assert len(rows)==len({r['id'] for r in rows})
   m=from_file(e['task'],e['model'],pred);write(met,m)
   started_entries[0].update({'status':'completed','prediction_file':str(pred.relative_to(ROOT)),'prediction_sha256':sha(pred),'metric_file':str(met.relative_to(ROOT)),'n':len(rows),'completed_at':now(),'technical_interruption':'Original process stopped after saving complete prediction file; metric calculation resumed from file without another model inference or test data access.'})
   results.append({**e,'test_metrics':m});write(ART/'test_access_log.json',log)
   print('RECOVERED_FROM_PREDICTION',i,k,len(rows),flush=True);continue
  assert not started_entries or (len(started_entries)==1 and started_entries[0]['status']=='failed' and not pred.exists()),'Incomplete checkpoint needs explicit technical retry audit'
  attempt=2 if started_entries else 1
  access={'model':e['model'],'task':e['task'],'seed':e['seed'],'checkpoint_sha256':e['checkpoint_sha256'],'accessed_at':now(),'status':'started','attempt':attempt}
  if attempt==2:access['technical_retry_of']='Import collision before model inference on attempt 1; same locked checkpoint and configuration.'
  log['entries'].append(access);write(ART/'test_access_log.json',log)
  try:
   m,rows=ev.run(e,'test');write(pred,{'model':e['model'],'task':e['task'],'seed':e['seed'],'rows':rows})
   rec=from_file(e['task'],e['model'],pred);assert max(compare(m,rec,e['task']).values())<1e-12
   write(met,rec);access.update({'status':'completed','prediction_file':str(pred.relative_to(ROOT)),'prediction_sha256':sha(pred),'metric_file':str(met.relative_to(ROOT)),'n':len(rows),'completed_at':now()})
   results.append({**e,'test_metrics':rec});print('TEST',i,k,len(rows),rec['selection_f1'],flush=True)
  except Exception as ex:
   access.update({'status':'failed','error':repr(ex),'traceback':traceback.format_exc(),'failed_at':now()});write(ART/'test_access_log.json',log);raise
  write(ART/'test_access_log.json',log)
 assert len(results)==48 and len([x for x in log['entries'] if x['status']=='completed'])==48
 first=datetime.fromisoformat(log['entries'][0]['accessed_at']).timestamp()
 write(ART/'test_results.json',{'completed':48,'expected':48,'results':results,'elapsed_seconds':time.time()-first,'resume_elapsed_seconds':time.time()-started,'technical_interruption':'First process stopped after saving complete predictions for student ViNLI seed 42; no repeat inference. Student ViNLI seed 43 attempt 1 failed before inference due import collision; attempt 2 uses the same checkpoint.'})
 print('TEST_COMPLETE',flush=True)
if __name__=='__main__':main()
