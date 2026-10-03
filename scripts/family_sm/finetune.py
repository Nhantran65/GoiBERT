#!/usr/bin/env python3
"""Locked E7 + Cafe OOF KD fine tuning for every family SM task."""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OLD=('vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel')
NEW=('vispam','vinli','phoner')
p=argparse.ArgumentParser()
p.add_argument('--backbone',type=Path,required=True)
p.add_argument('--task',choices=OLD+NEW,required=True)
p.add_argument('--seed',type=int,choices=(42,43,44),required=True)
p.add_argument('--lr',type=float,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--keep-checkpoint',action='store_true')
a=p.parse_args()
backbone=a.backbone.resolve()
if not (backbone/'model.safetensors').is_file():raise FileNotFoundError(backbone/'model.safetensors')
env={**os.environ,'GDA_BACKBONE':str(backbone),'FSM_LR':str(a.lr),'FSM_KEEP_CHECKPOINT':'1' if a.keep_checkpoint else '0','PYTHONPATH':str(ROOT/'src')}
script='cafe_student.py' if a.task in OLD else 'bench_student.py'
cmd=[sys.executable,str(Path(__file__).parent/script),'--task',a.task,'--seed',str(a.seed),'--out',str(a.out)]
cmd+=['--arm','kd'] if script=='cafe_student.py' else ['--kind','student']
raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))
