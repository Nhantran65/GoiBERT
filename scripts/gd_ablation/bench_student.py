#!/usr/bin/env python3
"""Train one Cafe fold or one E7 KD seed, validation only."""
from __future__ import annotations
import argparse,math,os,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel,AutoTokenizer,get_linear_schedule_with_warmup
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"bench8_kd"))
import common as C
from goibert import train as T
from goibert.modeling import build_classifier
from goibert_e7.optim import build_layerwise_optimizer
from goibert_e14.modeling import BamiBERTClassifier

def model_for(kind,task,n):
    if kind=='student':
        backbone=Path(os.environ['GDA_BACKBONE']).resolve()
        if task=='phoner':m=C.TokenModel(backbone,n);spec={'total_parameters':sum(p.numel() for p in m.parameters())}
        else:m,s=build_classifier(backbone,num_classes=n,local_files_only=True,attention_implementation='sdpa');spec=s.to_dict()
        return m,spec
    encoder,info=AutoModel.from_pretrained(C.CAFE,revision=C.CAFE_REV,local_files_only=True,add_pooling_layer=False,attn_implementation='sdpa',output_loading_info=True)
    missing=info.get('missing_keys',[]);extra=info.get('unexpected_keys',[])
    assert all('pooler' in x or 'lm_head' in x for x in missing+extra),(missing,extra)
    assert int(encoder.config.hidden_size)==1024
    if task=='phoner':
        m=CafeToken(encoder,n)
    else:m=BamiBERTClassifier(encoder,num_labels=n,num_aspects=0,aspect_class_counts=(),head_hidden_size=1024)
    return m,{'total_parameters':sum(p.numel() for p in m.parameters()),'loading_info':{'missing':missing,'unexpected':extra}}

class CafeToken(nn.Module):
    def __init__(self,encoder,n):
        super().__init__();self.encoder=encoder;self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(1024,n))
    def forward(self,input_ids,attention_mask,token_type_ids=None):
        return self.head(self.encoder(input_ids=input_ids,attention_mask=attention_mask,return_dict=True).last_hidden_state)

def load_soft(task,split,path=None):
    z=np.load(path or C.ART/'soft'/f'{task}.npz',allow_pickle=False)
    if 'label_names' in z:
        expected=C.PHONER_LABELS if task=='phoner' else tuple(C.load(task).label_names)
        assert tuple(map(str,z['label_names']))==tuple(expected)
    ids=list(map(str,z['ids']))
    assert ids==split.ids and len(ids)==len(set(ids))
    if task=='phoner':
        offsets=z['offsets'];logits=z['logits'];mask=z['has_logit']
        assert len(offsets)==len(ids)+1 and logits.shape==(int(offsets[-1]),21) and len(mask)==len(logits)
        return {sid:(logits[offsets[i]:offsets[i+1]],mask[offsets[i]:offsets[i+1]]) for i,sid in enumerate(ids)}
    assert z['logits'].shape==(len(ids),4)
    return {sid:z['logits'][i] for i,sid in enumerate(ids)}

def token_pair(logits,batch,soft):
    ss=[];tt=[]
    for i,sid in enumerate(batch['ids']):
        tl,mask=soft[sid]
        assert len(tl)==batch['word_count'][i]
        for pos,wid in batch['positions'][i]:
            if mask[wid]:ss.append(logits[i,pos].float());tt.append(torch.as_tensor(tl[wid],device=logits.device))
    if not ss:return None,None
    return torch.stack(ss),torch.stack(tt)

def kd_loss(s,t):
    return F.kl_div(F.log_softmax(s.float()/2,dim=-1),F.softmax(t.float()/2,dim=-1),reduction='batchmean')*4

def extract_oof(model,loader,task,device,precision):
    ids=[];values=[];masks=[];offsets=[0]
    model.eval()
    with torch.inference_mode():
        for b in loader:
            x=T._move_batch(b,device)
            with T.make_autocast(precision,device):v=model(x['input_ids'],x['attention_mask'],token_type_ids=x.get('token_type_ids'))
            v=v.float().cpu().numpy();ids.extend(b['ids'])
            if task=='phoner':
                for i,pos in enumerate(b['positions']):
                    n=b['word_count'][i];a=np.full((n,21),np.nan,dtype=np.float32);m=np.zeros(n,dtype=bool)
                    for p,w in pos:a[w]=v[i,p];m[w]=True
                    values.append(a);masks.append(m);offsets.append(offsets[-1]+n)
            else:values.append(v)
    if task=='phoner':return {'sentence_ids':np.asarray(ids),'offsets':np.asarray(offsets),'logits':np.concatenate(values),'has_logit':np.concatenate(masks)}
    return {'ids':np.asarray(ids),'logits':np.concatenate(values)}

def run(a):
    out=a.out;out.mkdir(parents=True,exist_ok=False);started=time.time()
    C.write(out/'status.json',{'status':'running','pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'started_at_unix':started})
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        seed=42 if a.kind=='teacher' else a.seed
        T.seed_everything(seed);device=T.resolve_device('cuda:0');precision=T.resolve_precision('bf16',device)
        data=C.load(a.task);labels=C.PHONER_LABELS if a.task=='phoner' else tuple(data.label_names)
        assert tuple(data.label_names)==labels
        train,held=C.folds(data,a.fold) if a.kind=='teacher' else (data.train,None)
        tokenizer=AutoTokenizer.from_pretrained(C.TOK if a.kind=='teacher' else Path(os.environ['GDA_BACKBONE']).resolve(),revision=C.TOK_REV if a.kind=='teacher' else None,local_files_only=True,use_fast=True)
        model,spec=model_for(a.kind,a.task,len(labels));model.to(device)
        lr=2e-5 if a.kind=='teacher' else C.LR[a.task]
        optimizer=T.make_optimizer(model,lr) if a.kind=='teacher' else build_layerwise_optimizer(model,lr,gamma=.8)
        maxlen=512 if a.task=='phoner' else 256
        train_loader,ts=C.make_custom_loader(train,a.task,tokenizer,labels,32,seed,True,maxlen,a.limit)
        val_loader,vs=C.make_custom_loader(data.validation,a.task,tokenizer,labels,32,seed,False,maxlen)
        held_loader=None
        if held is not None:held_loader,hs=C.make_custom_loader(held,a.task,tokenizer,labels,32,seed,False,maxlen)
        epochs=a.epochs;steps=len(train_loader)*epochs;warmup=round(steps*.1)
        scheduler=get_linear_schedule_with_warmup(optimizer,warmup,steps)
        soft=load_soft(a.task,data.train,a.soft) if a.kind=='student' else None
        cfg={'task':a.task,'kind':a.kind,'fold':a.fold,'seed':seed,'learning_rate':lr,'epochs':epochs,'batch_size':32,'max_length':maxlen,'warmup_steps':warmup,'total_steps':steps,'precision':precision,'include_test':False,'model_spec':spec,'length_stats':{'train':ts,'validation':vs,**({'oof':hs} if held is not None else {})},'alpha':.5 if soft else None,'temperature':2 if soft else None,'labels':list(labels)}
        C.write(out/'resolved_config.json',cfg)
        best=-math.inf;best_epoch=0;curves=[]
        for epoch in range(1,epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True);total=ce_total=kd_total=weight=0.;t0=time.time()
            for bi,b in enumerate(train_loader):
                x=T._move_batch(b,device)
                with T.make_autocast(precision,device):
                    logits=model(x['input_ids'],x['attention_mask'],token_type_ids=x.get('token_type_ids'))
                    ce=F.cross_entropy(logits.reshape(-1,len(labels)),x['labels'].reshape(-1),ignore_index=-100) if a.task=='phoner' else T.compute_loss(logits,x['labels'],num_aspects=0)
                if soft is not None:
                    if a.task=='phoner':s,t=token_pair(logits,b,soft)
                    else:s=logits.float();t=torch.as_tensor(np.stack([soft[i] for i in b['ids']]),device=device)
                    kd=kd_loss(s,t) if s is not None else logits.sum()*0
                    loss=.5*ce+.5*kd
                else:kd=None;loss=ce
                if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss epoch {epoch} batch {bi}')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                n=len(b['ids']);total+=float(loss.detach())*n;ce_total+=float(ce.detach())*n;kd_total+=float(kd.detach())*n if kd is not None else 0;weight+=n
            vm,rows=C.validation(model,val_loader,a.task,labels,device,precision);score=float(vm['selection_f1'])
            curves.append({'epoch':epoch,'train_loss':total/weight,'train_ce':ce_total/weight,'train_kd':kd_total/weight if soft else None,'selection_f1':score,'validation':vm,'epoch_seconds':time.time()-t0});C.write(out/'epoch_curves.json',{'epochs':curves})
            if score>best:
                best=score;best_epoch=epoch;C.write(out/'best_validation_metrics.json',vm);C.write(out/'validation_predictions.json',{'epoch':epoch,'rows':rows});torch.save({'model':model.state_dict(),'epoch':epoch},out/'best_checkpoint.pt')
            print(f'{a.kind}/{a.task}/{a.fold if a.kind=="teacher" else seed} epoch {epoch} val {score} best {best}@{best_epoch}',flush=True)
        del model;torch.cuda.empty_cache()
        reload,_=model_for(a.kind,a.task,len(labels));reload.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=True)['model'],strict=True);reload.to(device)
        rm,rr=C.validation(reload,val_loader,a.task,labels,device,precision);saved=C.read(out/'validation_predictions.json')['rows'];fm=C.evaluate(a.task,labels,[r['gold'] for r in saved],[r['prediction'] for r in saved])
        checks={'reload_matches_best':abs(rm['selection_f1']-best)<1e-9,'prediction_file_matches_best':abs(fm['selection_f1']-best)<1e-9,'reload_predictions_identical':[r['prediction'] for r in rr]==[r['prediction'] for r in saved],'ids_stable_and_complete':[r['id'] for r in rr]==[r['id'] for r in saved]==data.validation.ids}
        if held_loader is not None:
            oof=extract_oof(reload,held_loader,a.task,device,precision);np.savez_compressed(out/'oof_logits.npz',**oof)
            assert (list(map(str,oof['sentence_ids'])) if a.task=='phoner' else list(map(str,oof['ids'])))==held.ids
        status='completed' if all(checks.values()) else 'completed_with_check_failure'
        C.write(out/'metadata.json',{**cfg,'status':status,'best_epoch':best_epoch,'best_selection_f1':best,'postrun_checks':checks,'elapsed_seconds':time.time()-started})
        C.write(out/'status.json',{'status':status,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'best_epoch':best_epoch,'best_selection_f1':best,'elapsed_seconds':time.time()-started})
        if status=='completed':(out/'best_checkpoint.pt').unlink()
        return 0 if status=='completed' else 1
    except Exception as e:
        C.write(out/'status.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')});traceback.print_exc();return 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['teacher','student'],required=True);p.add_argument('--task',choices=C.TASKS,required=True);p.add_argument('--fold',type=int,default=None);p.add_argument('--seed',type=int,default=None);p.add_argument('--epochs',type=int,default=30);p.add_argument('--limit',type=int);p.add_argument('--soft',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert (a.kind=='teacher' and a.fold in range(5)) or (a.kind=='student' and a.seed in C.SEEDS)
    raise SystemExit(run(a))
