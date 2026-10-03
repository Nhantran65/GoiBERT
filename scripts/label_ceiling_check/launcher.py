#!/usr/bin/env python3
"""Two-GPU Cafe seed launcher. Run under nohup; never touches GPU 0/3."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/label_ceiling_check'
TASKS = ('vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel')
JOBS = [(t,s) for s in (43,44) for t in TASKS]
BUDGET_H = 15.0
PYTHON = ROOT / '.venv-e0-cuda/bin/python'

def free_mib():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
    return {int(p[0]): int(p[1]) for line in raw.splitlines() if (p := [x.strip() for x in line.split(',')]) and len(p) == 2}

def finished(job):
    t,s = job
    p = ART / 'cafe_seeds' / t / f'seed_{s}'
    status = p / 'status.json'; meta = p / 'metadata.json'; check = p / 'recipe_check.json'
    if not all(x.exists() for x in (status,meta,check,p/'validation_predictions.json')):
        return False
    a,b,c = [json.loads(x.read_text()) for x in (status,meta,check)]
    return a.get('status') == b.get('status') == 'completed' and all(b.get('postrun_checks',{}).values()) and c.get('pass') is True

def hours(job):
    t,s = job
    p = ART / 'cafe_seeds' / t / f'seed_{s}' / 'metadata.json'
    return float(json.loads(p.read_text())['elapsed_seconds']) / 3600

def main():
    ART.joinpath('logs').mkdir(parents=True,exist_ok=True)
    remaining = [j for j in JOBS if not finished(j)]
    active = {}
    completed_new = []
    launched_count = 0
    while remaining or active:
        for gpu,(proc,job,log) in list(active.items()):
            if proc.poll() is not None:
                log.close(); del active[gpu]
                if proc.returncode or not finished(job):
                    raise RuntimeError(f'Job failed: {job}, exit={proc.returncode}')
                completed_new.append(job)
                print(f'completed {job}, GPU-hours {hours(job):.6f}',flush=True)
        used = sum(hours(j) for j in completed_new)
        if remaining and len(completed_new) >= 2:
            estimate = used + sum(hours(j) for j in completed_new)/len(completed_new)*(len(remaining)+len(active))
            print(f'budget: used={used:.6f}, total estimate={estimate:.6f}, cap={BUDGET_H}',flush=True)
            if estimate > BUDGET_H:
                print('STOP: projected GPU hours exceed budget',flush=True)
                return 2
        if used >= BUDGET_H and remaining:
            print('STOP: GPU-hour cap reached',flush=True); return 2
        avail = free_mib()
        for gpu in (1,2):
            if launched_count >= 2 and len(completed_new) < 2:
                break
            if not remaining or gpu in active or avail.get(gpu,0) < 20480:
                continue
            job=remaining.pop(0); task,seed=job
            out=ART/'cafe_seeds'/task/f'seed_{seed}'
            if out.exists() and any(out.iterdir()):
                raise RuntimeError(f'Incomplete directory needs manual inspection: {out}')
            env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'src','HF_HUB_OFFLINE':'1'})
            log=open(ART/'logs'/f'{task}_seed_{seed}_gpu_{gpu}.log','a')
            cmd=[str(PYTHON),'-m','scripts.label_ceiling_check.seed_train','--task',task,'--seed',str(seed)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=(proc,job,log)
            launched_count += 1
            print(f'launched {job} on GPU {gpu}, pid={proc.pid}',flush=True)
        if remaining or active:
            time.sleep(300)
    print('all runs complete',flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
