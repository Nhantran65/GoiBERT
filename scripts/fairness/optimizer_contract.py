#!/usr/bin/env python3
"""Audit Bami E7 parameter coverage and the four required LR endpoints."""
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/bench8'),str(ROOT),str(ROOT/'src')]
from goibert_e14.modeling import build_bamibert_classifier
from goibert_e7.optim import build_layerwise_optimizer,validate_optimizer_groups
from scripts.bench8.train import TokenModel


def audit(model,lr):
    opt=build_layerwise_optimizer(model,lr,gamma=.8,num_layers=12)
    check=validate_optimizer_groups(model,opt,base_lr=lr,gamma=.8,num_layers=12)
    expected={'embeddings':lr*.8**12,'block_00':lr*.8**11,'block_11':lr,'head':lr}
    actual={}
    for region,want in expected.items():
        rates={float(g['lr']) for g in opt.param_groups if g['e7_region']==region}
        assert len(rates)==1 and math.isclose(next(iter(rates)),want,rel_tol=1e-12)
        actual[region]=next(iter(rates))
    assert check['all_trainable'] and check['duplicate_parameter_count']==0
    return {'trainable_parameter_tensors':check['parameter_count'],'groups':check['group_count'],
            'all_trainable_once':True,'expected_lr':expected,'actual_lr':actual}


def main():
    backbone=ROOT/'models/bamibert-base-original'
    sentence,_=build_bamibert_classifier(backbone,num_classes=4,attention_implementation='sdpa')
    ner=TokenModel(backbone,21)
    result={'sentence':audit(sentence,2e-5),'ner':audit(ner,5e-5)}
    out=ROOT/'artifacts/fairness/optimizer_contract.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
