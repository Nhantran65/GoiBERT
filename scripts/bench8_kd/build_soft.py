#!/usr/bin/env python3
"""Merge five Cafe folds and audit every OOF sentence/word."""
import argparse
from collections import Counter
import numpy as np
import common as C
from ner import spans

def metric_by_type(gold,pred):
    g=Counter();p=Counter();h=Counter()
    for a,b in zip(gold,pred):
        ga=set(spans(a));pb=set(spans(b))
        for x in ga:g[x[0]]+=1
        for x in pb:p[x[0]]+=1
        for x in ga&pb:h[x[0]]+=1
    return {k:{'gold':g[k],'predicted':p[k],'tp':h[k],'precision':h[k]/p[k] if p[k] else 0.,'recall':h[k]/g[k] if g[k] else 0.,'f1':2*h[k]/(g[k]+p[k]) if g[k]+p[k] else 0.} for k in sorted(set(g)|set(p))}

def main(task):
    data=C.load(task);ids=data.train.ids;index={s:i for i,s in enumerate(ids)};seen=set()
    fold_scores=[];fold_epochs=[];fold_trunc=[]
    if task=='phoner':matrix=[None]*len(ids);mask=[None]*len(ids)
    else:matrix=np.full((len(ids),4),np.nan,dtype=np.float32)
    for k in range(5):
        p=C.ART/'teacher'/task/f'fold_{k}'
        meta=C.read(p/'metadata.json');assert meta['status']=='completed' and all(meta['postrun_checks'].values())
        fold_scores.append(float(meta['best_selection_f1']));fold_epochs.append(meta['best_epoch']);fold_trunc.append(meta['length_stats'])
        z=np.load(p/'oof_logits.npz',allow_pickle=False)
        these=list(map(str,z['sentence_ids'] if task=='phoner' else z['ids']))
        _,held=C.folds(data,k);assert these==held.ids
        for j,s in enumerate(these):
            assert s in index and s not in seen;sidx=index[s];seen.add(s)
            if task=='phoner':
                a,b=int(z['offsets'][j]),int(z['offsets'][j+1]);n=len(data.train.words[sidx]);assert b-a==n
                matrix[sidx]=z['logits'][a:b];mask[sidx]=z['has_logit'][a:b]
            else:matrix[sidx]=z['logits'][j]
    assert len(seen)==len(ids) and seen==set(ids)
    out=C.ART/'soft';out.mkdir(parents=True,exist_ok=True)
    if task=='phoner':
        offsets=np.asarray([0]+list(np.cumsum([len(x) for x in matrix])),dtype=np.int64)
        logits=np.concatenate(matrix).astype(np.float32);has=np.concatenate(mask).astype(bool)
        assert logits.shape==(int(offsets[-1]),21) and np.isfinite(logits[has]).all() and np.isnan(logits[~has]).all()
        np.savez_compressed(out/f'{task}.npz',ids=np.asarray(ids),offsets=offsets,logits=logits,has_logit=has,label_names=np.asarray(C.PHONER_LABELS))
        pred=[];gold=[];correct=disagree=entity_disagree=entity_count=0;probs=[]
        for i in range(len(ids)):
            row=[]
            for w,tag in enumerate(data.train.labels[i]):
                if mask[i][w]:
                    v=matrix[i][w];e=np.exp(v-v.max());q=e/e.sum();guess=C.PHONER_LABELS[int(v.argmax())];probs.append(float(q.max()))
                    correct+=guess==tag;disagree+=guess!=tag
                    if tag!='O':entity_count+=1;entity_disagree+=guess!=tag
                else:guess='O'
                row.append(guess)
            pred.append(row);gold.append(data.train.labels[i])
        valid=int(has.sum());stats={'oof_entity_f1':C.evaluate(task,C.PHONER_LABELS,gold,pred)['f1'],'word_accuracy':correct/valid,'word_disagreement_rate':disagree/valid,'entity_gold_word_disagreement_rate':entity_disagree/entity_count if entity_count else 0.,'entity_gold_words_with_logit':entity_count,'valid_words':valid,'missing_words':int((~has).sum()),'f1_by_entity_type':metric_by_type(gold,pred)}
    else:
        assert np.isfinite(matrix).all();np.savez_compressed(out/f'{task}.npz',ids=np.asarray(ids),logits=matrix,label_names=np.asarray(data.label_names))
        gold=np.asarray(data.train.labels);pred=matrix.argmax(-1);q=np.exp(matrix-matrix.max(-1,keepdims=True));q/=q.sum(-1,keepdims=True);probs=q.max(-1)
        m=C.evaluate(task,data.label_names,gold.tolist(),pred.tolist())
        stats={'oof_macro_f1':m['macro_f1'],'oof_accuracy':m['accuracy'],'disagreement_rate':float((pred!=gold).mean()),'per_class_f1':{k:v['f1'] for k,v in m['per_class'].items()} if task=='vispam' else None}
    stats.update({'task':task,'train_examples':len(ids),'oof_unique_examples':len(seen),'oof_coverage':len(seen)/len(ids),'fold_validation_f1':fold_scores,'fold_validation_f1_mean':float(np.mean(fold_scores)),'fold_validation_f1_range':[min(fold_scores),max(fold_scores)],'fold_best_epochs':fold_epochs,'fold_length_stats':fold_trunc,'mean_max_prob':float(np.mean(probs)),'share_max_prob_ge_0.99':float(np.mean(np.asarray(probs)>=.99))})
    C.write(out/f'{task}_teacher_stats.json',stats)
    C.write(C.ART/task/'teacher_stats.json',stats)
    print(stats,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=C.TASKS,required=True);a=p.parse_args();main(a.task)
