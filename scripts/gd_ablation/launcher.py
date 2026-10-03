#!/usr/bin/env python3
"""Resource-gated launcher for the locked five-arm GD ablation."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];ART=ROOT/'artifacts/gd_ablation';PY=ROOT/'.venv-e0-cuda/bin/python'
GPUS=tuple(int(x) for x in os.environ.get('GDA_GPUS','0,1,2,3').split(','))
ORDER=('A3','A1','A5','A4','A2');STEPS={'A1':30000,'A2':30000,'A3':120000,'A4':120000,'A5':30000}
TASKS=('vsfc_sentiment','absa_restaurant','vispam','vinli','phoner');SEEDS=(42,43,44)
REG=ART/'launcher_jobs.json';LOGS=ART/'logs'
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def read(path,default):return json.loads(path.read_text()) if path.exists() else default
def free_mib():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in raw.splitlines())}
def alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,TypeError,ValueError):return False
def done(job):
    if job.startswith('GD_'):
        arm=job[3:];p=ART/'backbones'/arm/'final'/'distill_manifest.json'
        return p.exists() and read(p,{}).get('completed_steps')==STEPS[arm]
    _,arm,task,seed=job.split(':');p=ART/'finetune'/arm/task/f'seed_{seed}'/'status.json'
    return p.exists() and read(p,{}).get('status')=='completed'
def jobs():
    for arm in ORDER:yield f'GD_{arm}'
    for arm in ORDER:
        for task in TASKS:
            for seed in SEEDS:yield f'FT:{arm}:{task}:{seed}'
def ready(job):
    if job.startswith('GD_'):
        arm=job[3:]
        return arm not in ('A2','A4') or (ROOT/'data/corpus_v2/READY').is_file()
    return done('GD_'+job.split(':')[1])
def command(job):
    if job.startswith('GD_'):
        arm=job[3:];v='v2' if arm in ('A2','A4') else 'v1';teacher='bamibert-base-original' if arm=='A5' else 'goibert-b-dapt'
        return ['scripts/gd_ablation/gd.py','--arm',arm,'--corpus',f'data/corpus_{v}/mlm_mix.txt','--teacher',f'models/{teacher}','--steps',str(STEPS[arm]),'--out',str(ART/'backbones'/arm),'--resume']
    _,arm,task,seed=job.split(':')
    return ['scripts/gd_ablation/finetune.py','--backbone',str(ART/'backbones'/arm/'final'),'--task',task,'--seed',seed,'--out',str(ART/'finetune'/arm/task/f'seed_{seed}')]
def main():
    ART.mkdir(parents=True,exist_ok=True);LOGS.mkdir(exist_ok=True)
    registry=read(REG,{});active={};failed=set(read(ART/'launcher_summary.json',{}).get('failed',[]));v2_wait=None
    for job,entry in registry.items():
        if entry.get('status')=='running' and alive(entry.get('pid')):
            active[int(entry['gpu'])]=(job,int(entry['pid']),float(entry['started']),None,None)
            print('adopt',job,entry['gpu'],entry['pid'],flush=True)
    while True:
        now=time.time()
        for gpu,(job,pid,started,proc,fh) in list(active.items()):
            if (proc is not None and proc.poll() is None) or (proc is None and alive(pid)):continue
            if fh:fh.close()
            ok=done(job);registry[job]={**registry[job],'status':'completed' if ok else 'failed','ended':now,'elapsed_seconds':now-started};write(REG,registry)
            if not ok:failed.add(job)
            del active[gpu];print('done' if ok else 'FAILED',job,'GPU',gpu,flush=True)
        pending=[j for j in jobs() if not done(j) and j not in failed and j not in (x[0] for x in active.values())]
        if not pending and not active:break
        v2_pending=[j for j in pending if j in ('GD_A2','GD_A4') and not ready(j)]
        other=[j for j in pending if j not in v2_pending]
        if v2_pending and not other and not active:
            if v2_wait is None:v2_wait=now
            if now-v2_wait>=6*3600:
                print('corpus v2 READY timeout; pending',v2_pending,flush=True);break
        spent=sum(float(x.get('elapsed_seconds',0)) for x in registry.values() if x.get('status') in ('completed','failed'))+sum(now-x[2] for x in active.values())
        if spent>=50*3600:print('STOP budget',spent/3600,flush=True);break
        free=free_mib()
        for gpu in GPUS:
            if gpu in active:continue
            candidates=[j for j in pending if ready(j) and free.get(gpu,0)>=(16384 if j.startswith('GD_') else 12288)]
            if not candidates:continue
            job=candidates[0];pending.remove(job)
            fh=(LOGS/(job.replace(':','_')+'.log')).open('a')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'src','HF_HUB_OFFLINE':'1'}
            proc=subprocess.Popen([str(PY),*command(job)],cwd=ROOT,env=env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=(job,proc.pid,time.time(),proc,fh)
            registry[job]={'status':'running','pid':proc.pid,'gpu':gpu,'started':active[gpu][2]};write(REG,registry)
            print('launch',job,'GPU',gpu,'pid',proc.pid,flush=True)
            time.sleep(20);free=free_mib()
        write(ART/'launcher_summary.json',{'used_gpu_hours':spent/3600,'failed':sorted(failed),'pending':pending,'active':{str(g):v[0] for g,v in active.items()}})
        time.sleep(60)
    spent=sum(float(x.get('elapsed_seconds',0)) for x in registry.values() if x.get('status') in ('completed','failed'))
    write(ART/'launcher_summary.json',{'used_gpu_hours':spent/3600,'failed':sorted(failed),'pending':[j for j in jobs() if not done(j) and j not in failed]})
    return bool(failed)
if __name__=='__main__':sys.exit(main())
