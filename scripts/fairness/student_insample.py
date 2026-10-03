#!/usr/bin/env python3
"""S-prime E7 KD using the full-train CafeBERT soft labels."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LR = {'vsfc_sentiment': 1e-5, 'absa_restaurant': 1e-4, 'vinli': 1e-4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=tuple(LR), required=True)
    p.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--limit', type=int, default=None, help='smoke only, ViNLI')
    p.add_argument('--soft', type=Path, default=None)
    a = p.parse_args()
    soft = a.soft or ROOT / 'artifacts/fairness/soft_insample' / f'{a.task}.npz'
    if not soft.is_file():
        raise FileNotFoundError(soft)
    old = a.task != 'vinli'
    cmd = [sys.executable, str(ROOT / 'scripts/family_sm' / ('cafe_student.py' if old else 'bench_student.py'))]
    cmd += (['--arm', 'kd'] if old else ['--kind', 'student'])
    cmd += ['--task', a.task, '--seed', str(a.seed), '--epochs', str(a.epochs), '--soft', str(soft), '--out', str(a.out)]
    if a.limit is not None:
        if old:p.error('--limit is supported only for ViNLI')
        cmd += ['--limit',str(a.limit)]
    backbone = (ROOT / 'artifacts/family_sm/backbones/s_prime/final').resolve()
    env = {**os.environ, 'GDA_BACKBONE': str(backbone), 'FSM_LR': str(LR[a.task]),
           'FSM_KEEP_CHECKPOINT': '0', 'PYTHONPATH': str(ROOT / 'src')}
    result = subprocess.call(cmd, cwd=ROOT, env=env)
    if result:
        return result
    for name in ('metadata.json', 'resolved_config.json'):
        path = a.out / name
        d = json.loads(path.read_text())
        d.update({'soft_labels': str(soft), 'backbone': str(backbone), 'kd': True})
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
