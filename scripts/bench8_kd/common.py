"""Validation-only shared contracts for BENCH8 KD."""
from __future__ import annotations
import sys,importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'scripts/bench8',ROOT/'scripts/cafe_oof_kd'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from data import load,Split
from train import make_custom_loader,validation,evaluate as _evaluate,write,read,TokenModel
_spec=importlib.util.spec_from_file_location('cafe_oof_common',ROOT/'scripts/cafe_oof_kd/common.py')
_old=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_old)
fold_assignment=_old.fold_assignment
write_json=_old.write_json

ART=ROOT/'artifacts/bench8_kd'
TASKS=('vispam','vinli','phoner')
SEEDS=(42,43,44)
LR={'vispam':5e-5,'vinli':1e-4,'phoner':1e-4}
CAFE='uitnlp/CafeBERT'
CAFE_REV='af76fcf2a04096b2b54b348a3e4eb48253c93c5d'
TOK='FacebookAI/xlm-roberta-large'
TOK_REV='c23d21b0620b635a76227c604d44e43a9f0ee389'
PHONER_LABELS=tuple(load('phoner').label_names)
assert len(PHONER_LABELS)==21
assert PHONER_LABELS==tuple(['O']+sorted({tag for row in load('phoner').train.labels for tag in row if tag!='O'}))

def evaluate(task,labels,gold,pred):
    if task=='phoner':assert tuple(labels)==PHONER_LABELS
    return _evaluate(task,labels,gold,pred)

def subset(split,indices):
    return Split([split.ids[i] for i in indices],
                 [split.texts[i] for i in indices] if split.texts is not None else None,
                 [split.labels[i] for i in indices],
                 [split.pairs[i] for i in indices] if split.pairs is not None else None,
                 [split.words[i] for i in indices] if split.words is not None else None)

def folds(bundle,k):
    assignment=fold_assignment(len(bundle.train.ids))
    return subset(bundle.train,[i for i,v in enumerate(assignment) if v!=k]),subset(bundle.train,[i for i,v in enumerate(assignment) if v==k])
