#!/usr/bin/env python3
"""Run locked E7 + KD fine tuning on a supplied student backbone."""
import argparse, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--backbone',type=Path,required=True);p.add_argument('--task',choices=['vsfc_sentiment','absa_restaurant','vispam','vinli','phoner'],required=True);p.add_argument('--seed',type=int,choices=[42,43,44],required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
backbone=a.backbone.resolve()
if not (backbone/'model.safetensors').is_file():raise FileNotFoundError(backbone/'model.safetensors')
env={**os.environ,'GDA_BACKBONE':str(backbone),'PYTHONPATH':str(ROOT/'src')}
script='cafe_student.py' if a.task in ('vsfc_sentiment','absa_restaurant') else 'bench_student.py'
cmd=[sys.executable,str(Path(__file__).parent/script),'--task',a.task,'--seed',str(a.seed),'--out',str(a.out)]
if script=='cafe_student.py':cmd+=['--arm','kd']
else:cmd+=['--kind','student']
raise SystemExit(subprocess.call(cmd,cwd=ROOT,env=env))
