"""Strict entity-level BIO evaluator. Invalid I begins a new entity."""
from __future__ import annotations
from collections import Counter

def spans(tags):
    result=[];start=None;kind=None
    def end(i):
        if kind is not None:result.append((kind,start,i))
    for i,tag in enumerate([*tags,'O']):
        if tag=='O':end(i);start=kind=None;continue
        prefix,typ=tag.split('-',1)
        if prefix=='B' or prefix=='I' and kind!=typ:
            end(i);start=i;kind=typ
        elif prefix!='I':raise ValueError(tag)
    return result

def metrics(gold,pred):
    if len(gold)!=len(pred):raise ValueError('sentence count mismatch')
    hit=ng=np_=0
    for a,b in zip(gold,pred):
        if len(a)!=len(b):raise ValueError('word count mismatch')
        ga,pb=Counter(spans(a)),Counter(spans(b))
        hit+=sum((ga&pb).values());ng+=sum(ga.values());np_+=sum(pb.values())
    p=hit/np_ if np_ else 0.;r=hit/ng if ng else 0.;f=2*hit/(ng+np_) if ng+np_ else 0.
    return {'n':len(gold),'entity_tp':hit,'entity_gold':ng,'entity_predicted':np_,'precision':p,'recall':r,'f1':f,'selection_f1':f,'selection_metric':'strict_entity_micro_f1'}
