#!/usr/bin/env python3
"""Confidence and disagreement of OOF versus in-sample teacher logits."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from goibert import train as T
from scripts.bench8.data import load as load_bench


def measure(path,ids,gold,widths):
    with np.load(path,allow_pickle=False) as z:
        file_ids=list(map(str,z['ids']))
        logits=np.asarray(z['logits'],dtype=np.float64)
    assert file_ids==list(map(str,ids)) and logits.shape==(len(ids),sum(widths))
    assert np.isfinite(logits).all()
    ent=[];conf=[];dis=[]
    offset=0
    for slot,width in enumerate(widths):
        x=logits[:,offset:offset+width];offset+=width
        x=x-x.max(axis=1,keepdims=True);p=np.exp(x);p/=p.sum(axis=1,keepdims=True)
        y=np.asarray(gold if len(widths)==1 else [row[slot] for row in gold])
        ent.extend((-np.sum(p*np.log(np.maximum(p,1e-300)),axis=1)).tolist())
        conf.extend((p.max(axis=1)>.99).tolist())
        dis.extend((p.argmax(axis=1)!=y).tolist())
    return {'entropy_mean_nats':float(np.mean(ent)),
            'max_probability_gt_0_99_fraction':float(np.mean(conf)),
            'argmax_disagrees_with_gold_fraction':float(np.mean(dis)),
            'slots':len(widths),'observations':len(ent),'temperature':1.0,
            'file':str(path.relative_to(ROOT))}


def main():
    out={}
    for task in ('vsfc_sentiment','absa_restaurant','vinli'):
        if task=='vinli':
            data=load_bench(task);split=data.train
            ids,gold,widths=split.ids,split.labels,(len(data.label_names),)
            oof=ROOT/f'artifacts/bench8_kd/soft/{task}.npz'
        else:
            bundle=T.load_task_bundle(task,ROOT,include_test=False)
            split=T.split_field(bundle,'train')
            ids,gold,widths=T.split_ids(split),T.model_labels(bundle,'train',split),(
                T.task_aspect_class_counts(bundle) if T.task_num_aspects(bundle) else (T.task_num_classes(bundle),))
            oof=ROOT/f'artifacts/cafe_oof_kd/soft/{task}.npz'
        ins=ROOT/f'artifacts/fairness/soft_insample/{task}.npz'
        if ins.is_file():
            out[task]={'oof':measure(oof,ids,gold,widths),
                       'insample':measure(ins,ids,gold,widths)}
    target=ROOT/'artifacts/fairness/confidence.json'
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
