#!/usr/bin/env python3
"""Detached, restartable one-job-per-GPU BENCH8 KD queue."""
import json,os,shutil,subprocess,time
from pathlib import Path
import common as C

PY=str(C.ROOT/'.venv-e0-cuda/bin/python')
LOG=C.ART/'logs';LOG.mkdir(parents=True,exist_ok=True)
BUDGET=18*3600

def jobs():
    for task in C.TASKS:
        for k in range(5):yield task,'teacher',k,C.ART/'teacher'/task/f'fold_{k}'
        for s in C.SEEDS:yield task,'student',s,C.ART/'student'/task/f'seed_{s}'

def free():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(a):int(b) for a,b in (r.split(',') for r in raw.splitlines())}

def alive(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,ValueError,TypeError):return False

def status(path):
    p=path/'status.json';return json.loads(p.read_text()) if p.exists() else None

def spent():
    total=0.
    for p in C.ART.glob('**/status.json'):
        d=json.loads(p.read_text())
        if d.get('status') in ('completed','failed','completed_with_check_failure'):total+=float(d.get('elapsed_seconds',0))
        elif d.get('status')=='running':total+=max(0,time.time()-float(d['started_at_unix']))
    return total

def main():
    lock=C.ART/'launcher.lock'
    if lock.exists() and alive(lock.read_text().strip()):raise RuntimeError('launcher already running')
    lock.write_text(str(os.getpid()))
    failures=[];failed_ids=set()
    try:
        while True:
            queue=list(jobs());active={};pending=[]
            for task,kind,num,path in queue:
                if (task,kind,num) in failed_ids or (kind=='student' and any(t==task and k=='teacher' for t,k,n in failed_ids)):continue
                st=status(path)
                if st and st['status']=='completed':
                    meta=C.read(path/'metadata.json');assert all(meta['postrun_checks'].values());continue
                if st and st['status']=='running' and alive(st.get('pid')):
                    gpu=int(st['gpu']);assert gpu not in active,('two own jobs on GPU',gpu);active[gpu]=(task,kind,num,path);continue
                if st and st['status']!='completed':
                    aborted=C.ART/'aborted'/f'{task}_{kind}_{num}_{int(time.time())}'
                    aborted.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,aborted)
                    failures.append({'job':f'{task}_{kind}_{num}','aborted':str(aborted),'reason':st.get('error',st['status'])})
                    failed_ids.add((task,kind,num));continue
                pending.append((task,kind,num,path))
            if not pending and not active:break
            if spent()>=BUDGET:
                C.write(C.ART/'launcher_state.json',{'status':'budget_exhausted','spent_gpu_hours':spent()/3600,'pending':[str(p[3]) for p in pending],'failures':failures});return 2
            gpu_free=free()
            launched=0
            for gpu in range(4):
                if gpu in active or gpu_free.get(gpu,0)<20480:continue
                ready=next((j for j in pending if j[1]=='teacher' or (C.ART/'soft'/f'{j[0]}.npz').exists()),None)
                if ready is None:continue
                task,kind,num,path=ready
                if task=='phoner' and not (C.ART/'phoner/smoke_checks.json').exists():continue
                name=f'{kind}_{task}_{num}';cmd=[PY,'scripts/bench8_kd/run.py','--kind',kind,'--task',task,'--out',str(path)]
                cmd+=['--fold',str(num)] if kind=='teacher' else ['--seed',str(num)]
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                with open(LOG/f'{name}.log','a') as fh:
                    proc=subprocess.Popen(cmd,cwd=C.ROOT,env=env,stdout=fh,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                active[gpu]=ready;pending.remove(ready)
                launched+=1
                print(f'launch {name} GPU {gpu} PID {proc.pid} free_MiB {gpu_free[gpu]}',flush=True)
                time.sleep(20);gpu_free=free()
            for task in C.TASKS:
                soft=C.ART/'soft'/f'{task}.npz'
                if not soft.exists() and all(status(C.ART/'teacher'/task/f'fold_{k}') and status(C.ART/'teacher'/task/f'fold_{k}')['status']=='completed' for k in range(5)):
                    result=subprocess.run([PY,'scripts/bench8_kd/build_soft.py','--task',task],cwd=C.ROOT,env={**os.environ,'HF_HUB_OFFLINE':'1'},capture_output=True,text=True)
                    (LOG/f'build_soft_{task}.log').write_text(result.stdout+result.stderr)
                    if result.returncode:failures.append({'job':f'build_soft_{task}','reason':result.stderr});raise RuntimeError(result.stderr)
            C.write(C.ART/'launcher_state.json',{'status':'running','spent_gpu_hours':spent()/3600,'active':{str(k):v[0:3] for k,v in active.items()},'pending':[str(j[3]) for j in pending],'failures':failures})
            if pending and not active and not launched and all(j[1]=='student' and not (C.ART/'soft'/f'{j[0]}.npz').exists() for j in pending):
                C.write(C.ART/'launcher_state.json',{'status':'incomplete','spent_gpu_hours':spent()/3600,'pending':[str(j[3]) for j in pending],'failures':failures});return 1
            time.sleep(300)
        C.write(C.ART/'launcher_state.json',{'status':'completed','spent_gpu_hours':spent()/3600,'failures':failures})
        return 0
    finally:
        if lock.exists() and lock.read_text().strip()==str(os.getpid()):lock.unlink()

if __name__=='__main__':raise SystemExit(main())
