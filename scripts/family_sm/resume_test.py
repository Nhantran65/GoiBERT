#!/usr/bin/env python3
"""Resume a locked test after a technical interruption without repeating predictions."""
import time,traceback
from datetime import datetime
import torch
from test import ROOT,OUT,Evaluator,read,write,sha,from_file,compare,now
def main():
 started=time.time();locked=read(OUT/'locked_manifest.json');log=read(OUT/'test_access_log.json')
 assert sha(OUT/'locked_manifest.json')==log['locked_manifest_sha256'] and len(locked['entries'])==48
 ev=Evaluator(torch.device('cuda:0'));results=[]
 for i,e in enumerate(locked['entries'],1):
  key=f"{e['model']}_{e['task']}_seed_{e['seed']}";pred=OUT/'predictions'/f'{key}.json';met=OUT/'metrics'/f'{key}.json'
  previous=[x for x in log['entries'] if x['model']==e['model'] and x['task']==e['task'] and x['seed']==e['seed']]
  if met.exists():
   assert len([x for x in previous if x['status']=='completed'])==1
   results.append({**e,'test_metrics':read(met)});continue
  if pred.exists():
   rows=read(pred)['rows'];assert len(rows)==len({r['id'] for r in rows})
   metrics=from_file(e['task'],pred);write(met,metrics)
   assert len(previous)==1
   previous[0].update({'status':'completed','prediction_file':str(pred.relative_to(ROOT)),'prediction_sha256':sha(pred),'metric_file':str(met.relative_to(ROOT)),'n':len(rows),'completed_at':now(),'technical_interruption':'Recovered metrics from complete predictions; no second inference.'})
   results.append({**e,'test_metrics':metrics});write(OUT/'test_access_log.json',log);continue
  assert not previous or (len(previous)==1 and previous[0]['status']=='failed' and previous[0].get('technical_failure_before_predictions') is True),'Retry requires an audited technical failure before predictions'
  attempt=2 if previous else 1
  access={'model':e['model'],'task':e['task'],'seed':e['seed'],'checkpoint_sha256':e['checkpoint_sha256'],'accessed_at':now(),'status':'started','attempt':attempt}
  if attempt==2:access['technical_retry_of']=previous[0]['error']
  log['entries'].append(access);write(OUT/'test_access_log.json',log)
  try:
   metrics,rows=ev.run(e,'test');write(pred,{'model':e['model'],'task':e['task'],'seed':e['seed'],'rows':rows})
   rec=from_file(e['task'],pred);assert max(compare(metrics,rec,e['task']).values())<1e-12
   write(met,rec);access.update({'status':'completed','prediction_file':str(pred.relative_to(ROOT)),'prediction_sha256':sha(pred),'metric_file':str(met.relative_to(ROOT)),'n':len(rows),'completed_at':now()})
   results.append({**e,'test_metrics':rec});print('TEST',i,key,rec['selection_f1'],flush=True)
  except Exception as ex:
   access.update({'status':'failed','error':repr(ex),'traceback':traceback.format_exc(),'failed_at':now(),'technical_failure_before_predictions':not pred.exists()});write(OUT/'test_access_log.json',log);raise
  write(OUT/'test_access_log.json',log)
 assert len(results)==48 and len([x for x in log['entries'] if x['status']=='completed'])==48
 first=datetime.fromisoformat(log['entries'][0]['accessed_at']).timestamp()
 write(OUT/'test_results.json',{'completed':48,'expected':48,'results':results,'elapsed_seconds':time.time()-first,'resume_elapsed_seconds':time.time()-started})
 print('TEST_COMPLETE',flush=True)
if __name__=='__main__':main()
