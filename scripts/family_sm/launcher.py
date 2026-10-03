#!/usr/bin/env python3
"""Resource-gated, resumable validation scheduler for the locked S-prime/M plan."""
import json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'artifacts/family_sm';PY=ROOT/'.venv-e0-cuda/bin/python';HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from summarize import FIRST,TASKS,SEEDS,STUDENT_LR,BAMI_LR,run_dir,eval_dir,read,write
GPUS=tuple(int(x) for x in os.environ.get('FSM_GPUS','0,1,2,3').split(','))
REG=ART/'launcher_jobs.json';DEC=ART/'decisions.json';LOGS=ART/'logs'
RETRY_HISTORY=ART/'retry_history.json'
PRE=read(ART/'preflight.json');SMOKE={v:read(ART/'smoke'/v/'final/distill_manifest.json') for v in ('s_prime','m6','m12')}
PEAK={v:max(json.loads(line)['peak_vram_bytes'] for line in (ART/'smoke'/v/'loss_curve.jsonl').read_text().splitlines()) for v in SMOKE}
THRESH={v:int(PEAK[v]*1.2+1_000_000_000) for v in PEAK};THRESH['a4']=THRESH['s_prime']
FORECAST_GD=sum(SMOKE[v]['elapsed_seconds']/200*120000/3600 for v in SMOKE)
FORECAST_TOTAL=FORECAST_GD+18+1
if FORECAST_TOTAL>45:raise RuntimeError(f'STOP forecast {FORECAST_TOTAL} > 45 GPU-hours')

def free_bytes():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(a.strip()):int(b.strip())*1024*1024 for a,b in (line.split(',') for line in raw.splitlines())}
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (OSError,ValueError,TypeError):return False
def gdjob(v):return f'GD:{v}'
def ftjob(v,t,seed,lr):return f'FT:{v}:{t}:{seed}:{lr:.0e}'
def parse(job):
 parts=job.split(':')
 return parts if parts[0]=='GD' else (parts[0],parts[1],parts[2],int(parts[3]),float(parts[4]))
def done(job):
 p=parse(job)
 if p[0]=='GD':
  m=ART/'backbones'/p[1]/'final/distill_manifest.json'
  return m.is_file() and read(m).get('completed_steps')==120000
 _,v,t,seed,lr=p
 try:return eval_dir(run_dir(v,t,seed,lr),t,False) is not None
 except Exception:return False
def score(v,t,seed,lr):return eval_dir(run_dir(v,t,seed,lr),t,True)['f1_points']
def initial_jobs():
 jobs=[gdjob(v) for v in ('m6','m12','s_prime')]
 if done(gdjob('s_prime')):
  jobs += [ftjob('s_prime',t,s,STUDENT_LR[t]) for t in FIRST for s in SEEDS]
 if done(gdjob('m6')) and done(gdjob('m12')):
  jobs += [ftjob(v,t,42,lr) for v in ('m6','m12') for t in FIRST for lr in (STUDENT_LR[t],BAMI_LR[t])]
 return jobs
def update_decisions(d):
 if 's_official' not in d and all(done(ftjob('s_prime',t,s,STUDENT_LR[t])) for t in FIRST for s in SEEDS):
  task_means={t:sum(score('s_prime',t,s,STUDENT_LR[t]) for s in SEEDS)/3 for t in FIRST}
  p=sum(task_means.values())/5
  d['s_gate']={'task_means':task_means,'P_s_prime':p,'P_a4':81.50748919256499,'threshold':81.00748919256499,'triggered':p<81.00748919256499}
  d['s_official']='a4' if d['s_gate']['triggered'] else 's_prime';write(DEC,d)
 if 'm_shape' not in d and all(done(ftjob(v,t,42,lr)) for v in ('m6','m12') for t in FIRST for lr in (STUDENT_LR[t],BAMI_LR[t])):
  table={v:{t:{f'{lr:.0e}':score(v,t,42,lr) for lr in (STUDENT_LR[t],BAMI_LR[t])} for t in FIRST} for v in ('m6','m12')}
  means={v:sum(max(table[v][t].values()) for t in FIRST)/5 for v in ('m6','m12')}
  shape='m6' if abs(means['m6']-means['m12'])<0.3 else max(means,key=means.get)
  d['shape_selection']={'scores':table,'means':means,'absolute_delta':abs(means['m6']-means['m12']),'tie_threshold':0.3,'winner':shape}
  d['m_shape']=shape
  d['m_lr']={t:max((STUDENT_LR[t],BAMI_LR[t]),key=lambda lr:table[shape][t][f'{lr:.0e}']) for t in FIRST}
  write(DEC,d)
 if 'm_shape' in d and 'remaining_lr_selection' not in d:
  v=d['m_shape'];remaining=('vsfc_topic','visfd','absa_hotel')
  if all(done(ftjob(v,t,42,lr)) for t in remaining for lr in sorted(set((STUDENT_LR[t],BAMI_LR[t])))):
   table={t:{f'{lr:.0e}':score(v,t,42,lr) for lr in sorted(set((STUDENT_LR[t],BAMI_LR[t])))} for t in remaining}
   d['remaining_lr_selection']=table
   d['m_lr'].update({t:max(sorted(set((STUDENT_LR[t],BAMI_LR[t]))),key=lambda lr:table[t][f'{lr:.0e}']) for t in remaining})
   write(DEC,d)
 return d
def desired_jobs(d):
 jobs=initial_jobs()
 if 's_official' in d:
  v=d['s_official'];tasks=TASKS if v=='a4' else tuple(t for t in TASKS if t not in FIRST)
  jobs += [ftjob(v,t,s,STUDENT_LR[t]) for t in tasks for s in SEEDS]
 if 'm_shape' in d:
  v=d['m_shape'];jobs += [ftjob(v,t,42,lr) for t in ('vsfc_topic','visfd','absa_hotel') for lr in sorted(set((STUDENT_LR[t],BAMI_LR[t])))]
 if 'remaining_lr_selection' in d:
  jobs += [ftjob(d['m_shape'],t,s,d['m_lr'][t]) for t in TASKS for s in (43,44)]
 return list(dict.fromkeys(jobs))
def command(job):
 p=parse(job)
 if p[0]=='GD':
  v=p[1];return ['scripts/family_sm/gd.py','--variant',v,'--corpus','data/corpus_v2/mlm_mix.txt','--teacher','models/bamibert-base-original','--steps','120000','--out',str(ART/'backbones'/v),'--resume']
 _,v,t,s,lr=p
 backbone=ROOT/'artifacts/gd_ablation/backbones/A4/final' if v=='a4' else ART/'backbones'/v/'final'
 return ['scripts/family_sm/finetune.py','--backbone',str(backbone),'--task',t,'--seed',str(s),'--lr',str(lr),'--out',str(run_dir(v,t,s,lr)),'--keep-checkpoint']
def ready(job):
 p=parse(job)
 return p[0]=='GD' or p[1]=='a4' or done(gdjob(p[1]))
def prune(d):
 if 'm_shape' not in d:return
 winner=d['m_shape']
 for loser in ({'m6','m12'}-{winner}):
  for t in FIRST:
   for lr in (STUDENT_LR[t],BAMI_LR[t]):
    p=run_dir(loser,t,42,lr)/'best_checkpoint.pt'
    if p.exists() and done(ftjob(loser,t,42,lr)):p.unlink()
 if 'm_lr' in d:
  for t,chosen in d['m_lr'].items():
   for lr in sorted(set((STUDENT_LR[t],BAMI_LR[t]))):
    if lr==chosen:continue
    p=run_dir(winner,t,42,lr)/'best_checkpoint.pt'
    if p.exists() and done(ftjob(winner,t,42,lr)):p.unlink()
 if d.get('s_official')=='a4':
  for t in FIRST:
   for s in SEEDS:
    p=run_dir('s_prime',t,s,STUDENT_LR[t])/'best_checkpoint.pt'
    if p.exists() and done(ftjob('s_prime',t,s,STUDENT_LR[t])):p.unlink()
def main():
 ART.mkdir(parents=True,exist_ok=True);LOGS.mkdir(exist_ok=True)
 reg=read(REG) if REG.exists() else {};d=read(DEC) if DEC.exists() else {'forecast_gpu_hours':{'gd':FORECAST_GD,'finetune_assumption':18,'test_assumption':1,'total':FORECAST_TOTAL},'vram_threshold_bytes':THRESH}
 write(DEC,d);active={};failed=set()
 for job,e in reg.items():
  if e.get('status')=='running' and alive(e.get('pid')):
   active[int(e['gpu'])]=(job,int(e['pid']),float(e['started']),None,None);print('adopt',job,e['gpu'],e['pid'],flush=True)
  elif e.get('status')=='failed':failed.add(job)
 while True:
  now=time.time()
  for gpu,(job,pid,started,proc,fh) in list(active.items()):
   if (proc is not None and proc.poll() is None) or (proc is None and alive(pid)):continue
   if fh:fh.close()
   ok=done(job);reg[job]={**reg[job],'status':'completed' if ok else 'failed','ended':now,'elapsed_seconds':now-started};write(REG,reg)
   if not ok:failed.add(job)
   del active[gpu];print('done' if ok else 'FAILED',job,'GPU',gpu,flush=True)
  d=update_decisions(d);prune(d)
  wanted=desired_jobs(d);pending=[j for j in wanted if not done(j) and j not in failed and j not in (x[0] for x in active.values())]
  retry_spent=sum(float(x.get('elapsed_seconds',0)) for x in (read(RETRY_HISTORY) if RETRY_HISTORY.exists() else []))
  spent=retry_spent+sum(float(x.get('elapsed_seconds',0)) for x in reg.values() if x.get('status') in ('completed','failed'))+sum(now-x[2] for x in active.values())
  if spent>=45*3600:
   write(ART/'launcher_summary.json',{'status':'budget_stop','used_gpu_hours':spent/3600,'failed':sorted(failed),'pending':pending});print('STOP budget',spent/3600,flush=True);return 1
  if not pending and not active:
   if failed:break
   if len(d.get('m_lr',{}))==8 and 's_official' in d and all(done(ftjob(d['s_official'],t,s,STUDENT_LR[t])) and done(ftjob(d['m_shape'],t,s,d['m_lr'][t])) for t in TASKS for s in SEEDS):
    write(ART/'validation_complete.json',{'completed_at_unix':time.time(),'gpu_hours':spent/3600,'decisions':d});print('VALIDATION_COMPLETE',flush=True);break
   raise RuntimeError('Scheduler stalled before validation complete')
  free=free_bytes()
  for gpu in GPUS:
   if gpu in active:continue
   candidates=[j for j in pending if ready(j) and free.get(gpu,0)>=THRESH[parse(j)[1]]]
   if not candidates:continue
   job=candidates[0];pending.remove(job);p=parse(job)
   if p[0]=='FT':
    out=run_dir(*p[1:]);
    if out.exists() and not done(job):
     aborted=ART/'aborted'/f'{out.relative_to(ART)}-{int(time.time())}';aborted.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(aborted))
   fh=(LOGS/(job.replace(':','_')+'.log')).open('a')
   env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'src','HF_HUB_OFFLINE':'1'}
   proc=subprocess.Popen([str(PY),*command(job)],cwd=ROOT,env=env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
   active[gpu]=(job,proc.pid,time.time(),proc,fh);reg[job]={'status':'running','pid':proc.pid,'gpu':gpu,'started':active[gpu][2]};write(REG,reg)
   print('launch',job,'GPU',gpu,'pid',proc.pid,flush=True)
   time.sleep(10);free=free_bytes()
  write(ART/'launcher_summary.json',{'status':'running','used_gpu_hours':spent/3600,'failed':sorted(failed),'pending':pending,'active':{str(g):x[0] for g,x in active.items()}})
  time.sleep(60)
 subprocess.run([str(PY),'scripts/family_sm/summarize.py'],cwd=ROOT,env={**os.environ,'PYTHONPATH':'src'},check=False)
 spent=sum(float(x.get('elapsed_seconds',0)) for x in (read(RETRY_HISTORY) if RETRY_HISTORY.exists() else []))+sum(float(x.get('elapsed_seconds',0)) for x in reg.values() if x.get('status') in ('completed','failed'))
 write(ART/'launcher_summary.json',{'status':'failed' if failed else 'validation_complete','used_gpu_hours':spent/3600,'failed':sorted(failed),'pending':[]})
 return bool(failed)
if __name__=='__main__':
 try:sys.exit(main())
 except Exception:traceback.print_exc();sys.exit(1)
