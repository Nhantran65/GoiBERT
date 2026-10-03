#!/usr/bin/env python3
"""Run the existing E7 student trainer with CE only, preserving its recipe."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ('vsfc_sentiment', 'vsfc_topic', 'visfd', 'absa_restaurant', 'absa_hotel')
NEW = ('vispam', 'vinli', 'phoner')


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--task', choices=OLD + NEW, required=True)
    p.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--limit', type=int, default=None, help='smoke only, new tasks')
    p.add_argument('--keep-checkpoint', action='store_true')
    a = p.parse_args()
    backbone = a.backbone.resolve(strict=True)
    if not (backbone/'model.safetensors').is_file():
        raise FileNotFoundError(backbone/'model.safetensors')
    env = {**os.environ, 'GDA_BACKBONE': str(backbone), 'FSM_LR': str(a.lr),
           'FSM_KEEP_CHECKPOINT': '1' if a.keep_checkpoint else '0',
           'PYTHONPATH': str(ROOT / 'src')}
    old = a.task in OLD
    cmd = [sys.executable, str(ROOT / ('scripts/family_sm/cafe_student.py' if old else 'scripts/fairness/bench_student_ce.py'))]
    cmd += (['--arm', 'ce'] if old else ['--kind', 'student'])
    cmd += ['--task', a.task, '--seed', str(a.seed), '--epochs', str(a.epochs), '--out', str(a.out)]
    if a.limit is not None:
        if old: p.error('--limit is supported only for vispam, vinli and phoner')
        cmd += ['--limit', str(a.limit)]
    result = subprocess.call(cmd, cwd=ROOT, env=env)
    if result:
        return result
    for name in ('resolved_config.json', 'metadata.json'):
        path = a.out / name
        payload = json.loads(path.read_text())
        payload.update(kd=False, backbone=str(backbone), arm='ce')
        write(path, payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
