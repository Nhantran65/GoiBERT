#!/usr/bin/env python3
"""Resumable, VRAM-gated scheduler for the locked fairness validation plan."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/fairness'
PY = ROOT / '.venv-e0-cuda/bin/python'
GPUS = tuple(int(x) for x in os.environ.get('FAIR_GPUS', '0,1,2,3').split(','))
REG, DEC = ART / 'launcher_jobs.json', ART / 'decisions.json'
RETRIES = ART / 'retry_history.json'
LOGS = ART / 'logs'
OLD = ('vsfc_sentiment', 'vsfc_topic', 'visfd', 'absa_restaurant', 'absa_hotel')
TASKS = OLD + ('vispam', 'vinli', 'phoner')
SEEDS = (42, 43, 44)
S_LR = {'vsfc_sentiment': 1e-5, 'vsfc_topic': 1e-5, 'visfd': 1e-4,
        'absa_restaurant': 1e-4, 'absa_hotel': 1e-4, 'vispam': 5e-5,
        'vinli': 1e-4, 'phoner': 1e-4}
B_LR = {'vsfc_sentiment': 2e-5, 'vsfc_topic': 1e-5, 'visfd': 5e-5,
        'absa_restaurant': 5e-5, 'absa_hotel': 5e-5, 'vispam': 2e-5,
        'vinli': 1e-5, 'phoner': 5e-5}
GRID = (1e-5, 2e-5, 5e-5, 1e-4)
S_BACKBONE = ROOT / 'artifacts/family_sm/backbones/s_prime/final'
M_BACKBONE = ROOT / 'artifacts/family_sm/backbones/m12/final'


def read(path): return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def candidates(task):
    base = B_LR[task]
    second = min(GRID, key=lambda x: (abs(x-2*base), x))
    return tuple(dict.fromkeys((base, second)))


def run_dir(job):
    parts = job.split(':')
    if parts[0] == 'bami':
        _, arm, task, seed, lr = parts
        return ART / 'runs' / f'bami_e7_{arm}' / task / f'lr_{float(lr):.0e}' / f'seed_{seed}'
    if parts[0] == 'student':
        _, model, task, seed = parts
        return ART / 'runs' / f'{model}_ce' / task / f'seed_{seed}'
    if parts[0] == 'teacher':
        return ART / 'teacher_insample' / parts[1]
    _, task, seed = parts
    return ART / 'runs' / 's_prime_insample' / task / f'seed_{seed}'


def done(job):
    p = run_dir(job)
    try:
        meta = read(p / 'metadata.json')
        checks=meta['postrun_checks']
        return meta['status'] == 'completed' and all(v for v in checks.values() if isinstance(v,bool)) and (
            job.startswith('teacher:') and (ART/'soft_insample'/f'{job.split(":")[1]}.npz').is_file()
            or not job.startswith('teacher:'))
    except (FileNotFoundError, KeyError, TypeError):
        return False


def score(job):
    return float(read(run_dir(job) / 'best_validation_metrics.json')['selection_f1'])


def update_decisions(d):
    for arm in ('kd', 'ce'):
        for task in TASKS:
            key = f'bami_e7_{arm}/{task}'
            if key in d['bami_lr']:
                continue
            jobs = [f'bami:{arm}:{task}:42:{lr}' for lr in candidates(task)]
            if not all(done(j) for j in jobs):
                continue
            vals = {str(lr): score(j) for lr,j in zip(candidates(task),jobs)}
            epochs={str(lr):read(run_dir(j)/'metadata.json')['best_epoch']
                    for lr,j in zip(candidates(task),jobs)}
            chosen = max(candidates(task), key=lambda lr: vals[str(lr)])
            d['bami_lr'][key] = {'candidates': vals, 'candidate_best_epochs':epochs,
                                 'chosen_lr': chosen,
                                 'criterion': 'validation selection_f1 seed 42; exact tie retains E14 base LR',
                                 'chosen_at_unix': time.time()}
            write(DEC, d)
            for lr,j in zip(candidates(task),jobs):
                if lr != chosen:
                    ck = run_dir(j) / 'best_checkpoint.pt'
                    if ck.is_file(): ck.unlink()
    return d


def jobs_for(d):
    # #2 and #2b always have priority over #1; #3 follows both.
    bami = [f'bami:{arm}:{task}:42:{lr}' for arm in ('kd','ce') for task in TASKS
            for lr in candidates(task)]
    for arm in ('kd','ce'):
        for task in TASKS:
            chosen = d['bami_lr'].get(f'bami_e7_{arm}/{task}')
            if chosen:
                bami += [f'bami:{arm}:{task}:{seed}:{chosen["chosen_lr"]}' for seed in (43,44)]
    if not all(f'bami_e7_{arm}/{task}' in d['bami_lr'] for arm in ('kd','ce') for task in TASKS):
        return bami
    if not all(done(j) for j in bami):
        return bami
    students = [f'student:{model}:{task}:{seed}' for model in ('s_prime','m')
                for task in TASKS for seed in SEEDS]
    if not all(done(j) for j in students):
        return bami + students
    teachers = [f'teacher:{task}' for task in ('vsfc_sentiment','absa_restaurant','vinli')]
    insample = [f'insample:{task}:{seed}' for task in ('vsfc_sentiment','absa_restaurant','vinli') for seed in SEEDS]
    return bami + students + teachers + insample


def command(job):
    parts = job.split(':'); out = run_dir(job)
    if parts[0] == 'bami':
        _, arm, task, seed, lr = parts
        script = 'bami_old.py' if task in OLD else 'bami_new.py'
        cmd = [f'scripts/fairness/{script}', '--arm', arm, '--task', task,
               '--seed', seed, '--lr', lr, '--out', str(out)]
        if task in OLD: cmd += ['--layers', '12']
        return cmd
    if parts[0] == 'student':
        _, model, task, seed = parts
        backbone = S_BACKBONE if model == 's_prime' else M_BACKBONE
        lr = S_LR[task] if model == 's_prime' else read(ROOT/'artifacts/family_sm/decisions.json')['m_lr'][task]
        return ['scripts/fairness/student_ce.py', '--backbone', str(backbone), '--task', task,
                '--seed', seed, '--lr', str(lr), '--out', str(out), '--keep-checkpoint']
    if parts[0] == 'teacher':
        task = parts[1]
        return ['scripts/fairness/teacher_vinli.py' if task == 'vinli' else
                'scripts/fairness/teacher_insample.py', *(['--task',task] if task != 'vinli' else []),
                '--out', str(out)]
    _, task, seed = parts
    return ['scripts/fairness/student_insample.py', '--task', task, '--seed', seed, '--out', str(out)]


def free_bytes():
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(a.strip()): int(b.strip())*1024*1024 for a,b in
            (line.split(',') for line in raw.splitlines())}


def alive(pid):
    try: os.kill(int(pid),0); return True
    except (OSError,ValueError,TypeError): return False


def threshold(job):
    if job.startswith('bami:'):
        peaks = [read(ART/'smoke'/name/'metadata.json')['peak_gpu_memory_allocated_bytes']
                 for name in ('bami_sentence','bami_absa','bami_ner_vram')]
        return int(max(peaks)*1.5+2e9)
    if job.startswith('student:') or job.startswith('insample:'):
        model = job.split(':')[1] if job.startswith('student:') else 's_prime'
        return int(read(ROOT/'artifacts/family_sm/decisions.json')['vram_threshold_bytes'][model if model!='m' else 'm12'])
    teacher_peak=max(read(ART/'smoke'/name/'metadata.json')['peak_gpu_memory_allocated_bytes']
                     for name in ('teacher_old','teacher_absa'))
    return int(max(teacher_peak*1.4+2e9,18e9))


def main():
    contract=read(ART/'optimizer_contract.json')
    if not all(v['all_trainable_once'] and v['actual_lr']==v['expected_lr'] for v in contract.values()):
        raise RuntimeError('STOP: Bami E7 optimizer contract FAIL')
    parity=read(ART/'parity_results.json')
    if not all(v['passed'] for v in parity.values()):
        raise RuntimeError('STOP: parity check FAIL')
    forecast = read(ART/'forecast.json')
    if not forecast['passed'] or forecast['total_gpu_hours']>65:
        raise RuntimeError('STOP forecast exceeds 65 GPU-hours')
    ART.mkdir(parents=True,exist_ok=True); LOGS.mkdir(exist_ok=True)
    preflight_names=('bami_sentence','bami_absa','bami_ner_vram','m_ce','s_prime_ce',
                     's_prime_ce_new','teacher_old','teacher_absa','teacher_vinli',
                     'student_insample','student_insample_vinli')
    preflight_seconds=sum(read(ART/'smoke'/name/'metadata.json')['elapsed_seconds'] for name in preflight_names)
    preflight_seconds+=sum(read(ART/'parity'/name/'metadata.json')['elapsed_seconds'] for name in
                           ('bami_e14_topic','bami_e14_vispam','student_ce'))
    for path in (ART/'aborted/parity_bami_e14_topic_micro32/metadata.json',
                 ART/'aborted/bami_ner_first_smoke/status.json'):
        if path.is_file():preflight_seconds+=float(read(path).get('elapsed_seconds',0))
    reg = read(REG) if REG.exists() else {}
    d = read(DEC) if DEC.exists() else {'bami_lr': {}}
    d.setdefault('forecast',forecast)
    d.setdefault('student_ce_locked_lr',{'s_prime':S_LR,
                                         'm':read(ROOT/'artifacts/family_sm/decisions.json')['m_lr']})
    d.setdefault('student_backbones',{'s_prime':str(S_BACKBONE.resolve()),
                                      'm':str(M_BACKBONE.resolve())})
    d.setdefault('bami_lr_rule',{'seed_for_selection':42,'base_lr':B_LR,
                                  'second_lr':'nearest grid point to twice base LR; tie chooses lower LR',
                                  'grid':GRID,'selection_metric':'validation selection_f1',
                                  'followup_seeds':(43,44)})
    d.setdefault('vram_threshold_bytes',{
        'bami':threshold('bami:kd:vsfc_sentiment:42:2e-05'),
        'student_s_prime':threshold('student:s_prime:vsfc_sentiment:42'),
        'student_m':threshold('student:m:vsfc_sentiment:42'),
        'teacher':threshold('teacher:vsfc_sentiment')})
    write(DEC,d)
    active = {}
    failed = set() if os.environ.get('FAIR_RETRY_FAILED')=='1' else {
        j for j,e in reg.items() if e.get('status')=='failed'}
    for job,e in reg.items():
        if e.get('status')=='running' and alive(e.get('pid')):
            active[int(e['gpu'])]=(job,int(e['pid']),float(e['started']),None,None)
            print('adopt',job,e['gpu'],e['pid'],flush=True)
        elif e.get('status')=='running' and done(job):
            reg[job].update(status='completed',ended=time.time(),
                            elapsed_seconds=time.time()-float(e['started']))
            write(REG,reg)
        elif e.get('status')=='running' and not done(job):
            failed.add(job);reg[job]['status']='failed';write(REG,reg)
    while True:
        now=time.time()
        for gpu,(job,pid,started,proc,fh) in list(active.items()):
            if (proc is not None and proc.poll() is None) or (proc is None and alive(pid)): continue
            if fh: fh.close()
            ok=done(job)
            reg[job].update(status='completed' if ok else 'failed',ended=now,elapsed_seconds=now-started)
            write(REG,reg)
            if not ok: failed.add(job)
            del active[gpu]
            print('done' if ok else 'FAILED',job,'GPU',gpu,flush=True)
        d=update_decisions(d)
        wanted=jobs_for(d)
        active_jobs={x[0] for x in active.values()}
        pending=[j for j in wanted if not done(j) and j not in failed and j not in active_jobs]
        retry_spent=sum(float(e.get('elapsed_seconds',0)) for e in (read(RETRIES) if RETRIES.exists() else []))
        spent=preflight_seconds+retry_spent+sum(float(e.get('elapsed_seconds',0)) for e in reg.values() if e.get('status') in ('completed','failed'))
        spent+=sum(now-x[2] for x in active.values())
        if spent+forecast['test_gpu_hours']*3600>=65*3600:
            write(ART/'launcher_summary.json',{'status':'budget_stop','used_gpu_hours':spent/3600,
                                                'failed':sorted(failed),'pending':pending})
            print('STOP budget 65 GPU-hours',flush=True);return 1
        if not pending and not active:
            status='failed' if failed else 'validation_complete'
            write(ART/'launcher_summary.json',{'status':status,'used_gpu_hours':spent/3600,
                                                'failed':sorted(failed),'pending':pending})
            print(status.upper(),flush=True)
            return int(bool(failed))
        free=free_bytes()
        for gpu in GPUS:
            if gpu in active: continue
            eligible=[j for j in pending if free.get(gpu,0)>=threshold(j) and
                      (not j.startswith('insample:') or done('teacher:'+j.split(':')[1]))]
            if not eligible: continue
            job=eligible[0];pending.remove(job);out=run_dir(job)
            if out.exists() and not done(job):
                dst=ART/'aborted'/f'{out.relative_to(ART)}-{int(time.time())}'
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out),str(dst))
            fh=(LOGS/(job.replace(':','_')+'.log')).open('a')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':'src',
                 'HF_HUB_OFFLINE':'1','FSM_KEEP_CHECKPOINT':'1' if job.startswith('bami:') or job.startswith('student:') else '0'}
            proc=subprocess.Popen([str(PY),*command(job)],cwd=ROOT,env=env,
                                  stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=(job,proc.pid,time.time(),proc,fh)
            if job in reg and reg[job].get('elapsed_seconds'):
                history=read(RETRIES) if RETRIES.exists() else []
                history.append({'job':job,'elapsed_seconds':reg[job]['elapsed_seconds'],
                                'previous_status':reg[job].get('status'),'recorded_at_unix':time.time()})
                write(RETRIES,history)
            reg[job]={'status':'running','pid':proc.pid,'gpu':gpu,'started':active[gpu][2]}
            write(REG,reg);print('launch',job,'GPU',gpu,'pid',proc.pid,flush=True)
            time.sleep(10);free=free_bytes()
        write(ART/'launcher_summary.json',{'status':'running','used_gpu_hours':spent/3600,
                                           'failed':sorted(failed),'pending':pending,
                                           'active':{str(g):x[0] for g,x in active.items()}})
        time.sleep(60)


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception: traceback.print_exc();raise
