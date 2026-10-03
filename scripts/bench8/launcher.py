#!/usr/bin/env python3
"""Shared resumable GPU queue for BENCH8. Enable tasks by stage files."""
from __future__ import annotations
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'artifacts/bench8';STAGES=ART/'stages';MIN_FREE=12000;BUDGET=16*3600
LR={'bami':[1e-5,2e-5,5e-5],'student':[1e-5,5e-5,1e-4]}
def rd(p):return json.loads(p.read_text())
def wr(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def path(task,model,lr,seed):return ART/'runs'/task/model/f'lr_{lr:.0e}'/f'seed_{seed}'
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (OSError,TypeError,ValueError):return False
def gpu_free():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 return {x.strip():int(y.strip()) for line in out.splitlines() for x,y in [line.split(',')]}
def specs():
 jobs=[];selected={}
 for task in ('vispam','vinli','phoner'):
  if not (STAGES/f'{task}.ready').exists():continue
  for model in ('bami','student'):
   screen=[(task,model,lr,42) for lr in LR[model]];jobs.extend(screen)
   if all(path(*j).joinpath('status.json').exists() and rd(path(*j)/'status.json').get('status')=='completed' for j in screen):
    winner=min(LR[model],key=lambda lr:(-float(rd(path(task,model,lr,42)/'best_validation_metrics.json')['selection_f1']),lr))
    selected[f'{task}/{model}']=winner
    jobs.extend((task,model,winner,s) for s in (43,44))
    if task=='vispam' and (STAGES/'vispam_binary.ready').exists():jobs.extend(('vispam_binary',model,winner,s) for s in (42,43,44))
 wr(ART/'selected_lrs.json',selected)
 return jobs

def elapsed():
 total=0.
 for folder in ('runs','reproduction','aborted'):
  for p in (ART/folder).rglob('metadata.json') if (ART/folder).exists() else []:
   m=rd(p)
   if 'elapsed_seconds' in m:total+=float(m['elapsed_seconds'])
  for p in (ART/folder).rglob('status.json') if (ART/folder).exists() else []:
   if (p.parent/'metadata.json').exists():continue
   s=rd(p)
   if s.get('status')=='failed' and 'elapsed_seconds' in s:total+=float(s['elapsed_seconds'])
 smoke=ART/'smoke_manifest.json'
 if smoke.exists():total+=sum(float(x['elapsed_seconds']) for x in rd(smoke).values())
 return total

def main():
 ART.mkdir(parents=True,exist_ok=True)
 lock=(ART/'launcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpus=[g.strip() for g in os.environ.get('B8_GPUS','0,1,2,3').split(',') if g.strip()];running={};failures=[]
 while True:
  jobs=specs();now=time.time()
  for key,(proc,gpu,started) in list(running.items()):
   p=path(*key);st=rd(p/'status.json') if (p/'status.json').exists() else {}
   exited=(proc.poll() is not None) if proc is not None else not alive(st.get('pid'))
   if st.get('status') in ('completed','completed_with_check_failure') and exited:
    if st.get('status')!='completed':failures.append([key,st])
    del running[key]
   elif st.get('status')=='failed' and exited or proc is not None and proc.poll() is not None and st.get('status')!='completed':
    failures.append([key,st]);del running[key]
  for j in jobs:
   p=path(*j);s=p/'status.json'
   if j in running or not s.exists():continue
   status=rd(s)
   if status.get('status') in ('running','completed') and alive(status.get('pid')):
    running[j]=(None,str(status['gpu']),float(status.get('started_at_unix',now)))
   elif status.get('status')=='running':
    status.update(status='failed',error='worker disappeared; move directory to aborted before retry')
    wr(s,status);failures.append([j,status])
  used={g for _,g,_ in running.values()};free=gpu_free();budget=elapsed()+sum(now-t for _,_,t in running.values())
  for gpu in gpus:
   if gpu in used or free.get(gpu,0)<MIN_FREE or budget>=BUDGET:continue
   j=next((j for j in jobs if j not in running and (not (path(*j)/'status.json').exists() or rd(path(*j)/'status.json').get('status') not in ('completed','running','failed','completed_with_check_failure'))),None)
   if j is None:continue
   p=path(*j)
   if p.exists():
    failures.append([j,{'error':'stale directory requires manual move to aborted'}]);continue
   log=ART/'logs'/'__'.join(map(str,j));log.parent.mkdir(parents=True,exist_ok=True)
   with log.with_suffix('.log').open('a') as f:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',PYTHONPATH='src')
    cmd=[sys.executable,str(ROOT/'scripts/bench8/train.py'),'--task',j[0],'--model',j[1],'--lr',str(j[2]),'--seed',str(j[3]),'--out',str(p),'--gpu',gpu]
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   running[j]=(proc,gpu,now);used.add(gpu);print('launched',j,'gpu',gpu,'pid',proc.pid,flush=True)
  wr(ART/'launcher_state.json',{'pid':os.getpid(),'running':[{'job':j,'gpu':g,'pid':proc.pid if proc else rd(path(*j)/'status.json').get('pid')} for j,(proc,g,_) in running.items()],'completed':sum((path(*j)/'status.json').exists() and rd(path(*j)/'status.json').get('status')=='completed' for j in jobs),'expected':len(jobs),'gpu_seconds_completed':elapsed(),'failures':failures,'budget_exhausted':budget>=BUDGET,'updated_at':now})
  if not running and (STAGES/'all.ready').exists() and all((path(*j)/'status.json').exists() and rd(path(*j)/'status.json').get('status') in ('completed','failed','completed_with_check_failure') for j in jobs):break
  if budget>=BUDGET and not running:break
  time.sleep(10)
 print('launcher done',flush=True)
if __name__=='__main__':main()
