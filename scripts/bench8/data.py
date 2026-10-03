"""Validation-only BENCH8 data contracts; test paths are never opened."""
from __future__ import annotations
import csv, hashlib, json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
LABELS_NLI=('entailment','neutral','contradiction','other')

@dataclass
class Split:
    ids:list[str]
    texts:list[str]|None
    labels:list
    pairs:list[tuple[str,str]]|None=None
    words:list[list[str]]|None=None

@dataclass
class Bundle:
    task_id:str
    train:Split
    validation:Split
    label_names:list[str]
    num_classes:int

def _ids(task,split,n):return [f'{task}:{split}:{i}' for i in range(n)]
def _verify(task):
    manifest=json.loads((ROOT/'artifacts/bench8/data_manifest.json').read_text())['files']
    paths={
        'vispam':['data/raw/social/vispamdetection/train.csv','data/raw/social/vispamdetection/dev.csv'],
        'vinli':['data/raw/vinli_mirror/train.parquet','data/raw/vinli_mirror/dev.parquet'],
        'phoner':['data/raw/phoner_covid19/train_syllable.json','data/raw/phoner_covid19/dev_syllable.json'],
    }
    selected=paths.get('vispam' if task=='vispam_binary' else task)
    if selected is None:raise ValueError(task)
    for name in selected:
        with (ROOT/name).open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=manifest[name]['sha256']:raise RuntimeError(f'data checksum changed: {name}')
def load(task):
    _verify(task)
    if task in ('vispam','vispam_binary'):
        column='SpamLabel' if task=='vispam' else 'Label'
        def one(split):
            with (ROOT/'data/raw/social/vispamdetection'/f'{split}.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
            return Split(_ids(task,split,len(rows)),[r['Comment'] for r in rows],[int(r[column]) for r in rows])
        return Bundle(task,one('train'),one('dev'),[str(i) for i in range(4 if task=='vispam' else 2)],4 if task=='vispam' else 2)
    if task=='vinli':
        def one(split):
            frame=pd.read_parquet(ROOT/'data/raw/vinli_mirror'/f'{split}.parquet')
            pairs=list(zip(frame.sentence1.astype(str),frame.sentence2.astype(str)))
            return Split(_ids(task,split,len(frame)),None,[LABELS_NLI.index(x) for x in frame.gold_label],pairs=pairs)
        return Bundle(task,one('train'),one('dev'),list(LABELS_NLI),4)
    if task=='phoner':
        def one(split):
            with (ROOT/'data/raw/phoner_covid19'/f'{split}_syllable.json').open(encoding='utf-8') as f: rows=[json.loads(line) for line in f if line.strip()]
            if any(len(r['words'])!=len(r['tags']) for r in rows):raise ValueError('word/tag length mismatch')
            return Split(_ids(task,split,len(rows)),None,[r['tags'] for r in rows],words=[r['words'] for r in rows])
        train,dev=one('train'),one('dev')
        labels=['O']+sorted({t for row in train.labels+dev.labels for t in row if t!='O'})
        return Bundle(task,train,dev,labels,len(labels))
    raise ValueError(task)
