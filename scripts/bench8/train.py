#!/usr/bin/env python3
"""BENCH8 validation-only trainer, preserving E14/E7 loop order."""
from __future__ import annotations
import argparse,json,math,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,RobertaModel,get_linear_schedule_with_warmup
from goibert import train as T
from goibert.modeling import build_classifier
from goibert_e14.modeling import build_bamibert_classifier
from goibert_e7.optim import build_layerwise_optimizer
from goibert.metrics import classification_metrics
from data import load
from ner import metrics as ner_metrics

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/bench8'

def write(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str)+'\n');temp.replace(path)
def read(path):return json.loads(path.read_text())

def evaluate(task,labels,gold,pred):
 if task=='phoner':return ner_metrics(gold,pred)
 if task=='vsfc_topic':
  bundle=T.load_task_bundle(task,ROOT,include_test=False)
  return T.evaluate_predictions(task,bundle,gold,pred,split='validation')
 m=classification_metrics(gold,pred,labels=tuple(range(len(labels))))
 return {**m,'task_id':task,'split':'validation','selection_f1':m['macro_f1'],'selection_metric':'fixed_label_macro_f1'}

class TokenModel(nn.Module):
 def __init__(self,backbone,classes):
  super().__init__()
  self.encoder=RobertaModel.from_pretrained(backbone,local_files_only=True,add_pooling_layer=False,attn_implementation='sdpa')
  self.head=nn.Sequential(nn.Dropout(0.1),nn.Linear(self.encoder.config.hidden_size,classes))
 def forward(self,input_ids,attention_mask,token_type_ids=None):
  kw={'input_ids':input_ids,'attention_mask':attention_mask}
  if token_type_ids is not None:kw['token_type_ids']=token_type_ids
  return self.head(self.encoder(**kw,return_dict=True).last_hidden_state)

class Records(Dataset):
 def __init__(self,rows):self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):return self.rows[i]

def make_custom_loader(split,task,tokenizer,label_names,batch,seed,shuffle,max_length,limit=None,show_pairs=False):
 rows=[];truncated=0;total_words=0;truncated_words=0
 n=len(split.ids) if limit is None else min(limit,len(split.ids))
 for i in range(n):
  if task=='phoner':
   words=split.words[i];tags=split.labels[i];total_words+=len(words)
   full=tokenizer(words,is_split_into_words=True,add_special_tokens=True,truncation=False)
   encoded=tokenizer(words,is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=max_length)
   word_ids=encoded.word_ids();seen=set();aligned=[];positions=[]
   for j,wid in enumerate(word_ids):
    if wid is None or wid in seen:aligned.append(-100)
    else:seen.add(wid);aligned.append(label_names.index(tags[wid]));positions.append((j,wid))
   truncated_words+=len(words)-len(seen);truncated+=len(full['input_ids'])>max_length
   row={'id':split.ids[i],'enc':dict(encoded),'label':aligned,'gold':tags,'positions':positions,'word_count':len(words)}
  else:
   args=split.pairs[i] if task=='vinli' else (split.texts[i],)
   full=tokenizer(*args,add_special_tokens=True,truncation=False)
   enc=tokenizer(*args,add_special_tokens=True,truncation=True,max_length=max_length)
   truncated+=len(full['input_ids'])>max_length
   if show_pairs and i<3:print('PAIR_EXAMPLE',i,'ids=',enc['input_ids'],'tokens=',tokenizer.convert_ids_to_tokens(enc['input_ids']),'sequence_ids=',enc.sequence_ids(),flush=True)
   row={'id':split.ids[i],'enc':dict(enc),'label':split.labels[i],'gold':split.labels[i]}
  rows.append(row)
 def collate(items):
  encoded=tokenizer.pad([r['enc'] for r in items],padding=True,return_tensors='pt')
  if task=='phoner':
   width=encoded['input_ids'].shape[1]
   encoded['labels']=torch.tensor([r['label']+[-100]*(width-len(r['label'])) for r in items])
   encoded['positions']=[r['positions'] for r in items];encoded['word_count']=[r['word_count'] for r in items]
  else:encoded['labels']=torch.tensor([r['label'] for r in items])
  encoded['gold_labels']=[r['gold'] for r in items];encoded['ids']=[r['id'] for r in items]
  return encoded
 loader=DataLoader(Records(rows),batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed) if shuffle else None,num_workers=0,pin_memory=torch.cuda.is_available(),collate_fn=collate)
 return loader,{'examples':n,'truncated_examples':truncated,'truncation_rate':truncated/n if n else 0.,'words':total_words,'truncated_words':truncated_words}

def validation(model,loader,task,labels,device,precision,bundle=None):
 if bundle is not None:return T.run_validation(model,loader,device,precision,task,bundle)
 model.eval();gold=[];pred=[];ids=[]
 with torch.inference_mode():
  for batch in loader:
   x=T._move_batch(batch,device)
   with T.make_autocast(precision,device):logits=model(x['input_ids'],x['attention_mask'],token_type_ids=x.get('token_type_ids'))
   if task=='phoner':
    predictions=logits.argmax(-1).cpu().tolist()
    for k,positions in enumerate(batch['positions']):
     tags=['O']*batch['word_count'][k]
     for pos,wid in positions:tags[wid]=labels[predictions[k][pos]]
     pred.append(tags)
    gold.extend(batch['gold_labels'])
   else:pred.extend(logits.argmax(-1).cpu().tolist());gold.extend(batch['gold_labels'])
   ids.extend(batch['ids'])
 return evaluate(task,labels,gold,pred),[{'id':i,'gold':g,'prediction':p} for i,g,p in zip(ids,gold,pred)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--model',choices=('bami','student'),required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--lr',type=float,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--limit',type=int);ap.add_argument('--gpu',required=True);a=ap.parse_args()
 out=a.out;out.mkdir(parents=True,exist_ok=False);started=time.time();write(out/'status.json',{'status':'running','pid':os.getpid(),'gpu':a.gpu,'started_at_unix':started})
 try:
  os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
  T.seed_everything(a.seed);device=T.resolve_device('cuda:0');precision=T.resolve_precision('bf16',device)
  backbone=ROOT/'models'/('bamibert-base-original' if a.model=='bami' else 'goiturn-s-gd')
  bundle=None
  if a.task=='vsfc_topic':
   bundle=T.load_task_bundle(a.task,ROOT,include_test=False);train=T.split_field(bundle,'train');valid=T.split_field(bundle,'validation');labels=list(bundle.label_names)
  else:
   data=load(a.task);train=data.train;valid=data.validation;labels=data.label_names
  tokenizer=AutoTokenizer.from_pretrained(backbone,local_files_only=True,use_fast=True)
  if a.model=='student' and a.task=='vsfc_topic':T.inspect_train_lengths(tokenizer,list(T.split_texts(train)))
  classes=len(labels)
  if a.task=='phoner':model=TokenModel(backbone,classes);spec={'total_parameters':sum(p.numel() for p in model.parameters()),'head':'dropout0.1 -> linear'}
  elif a.model=='bami':model,s=build_bamibert_classifier(backbone,num_classes=classes,attention_implementation='sdpa');spec=s.to_dict()
  else:model,s=build_classifier(backbone,num_classes=classes,local_files_only=True,attention_implementation='sdpa');spec=s.to_dict()
  model.to(device)
  optimizer=T.make_optimizer(model,a.lr) if a.model=='bami' else build_layerwise_optimizer(model,a.lr,gamma=0.8)
  batch=16 if a.model=='bami' else 32;accum=32//batch
  if bundle is not None:
   train_loader=T._make_loader(tokenizer,train,max_length=256,batch_size=batch,shuffle=True,seed=a.seed,labels=T.model_labels(bundle,'train',train),gold_labels=T.split_labels(train))
   valid_loader=T._make_loader(tokenizer,valid,max_length=256,batch_size=batch,shuffle=False,seed=a.seed,labels=T.model_labels(bundle,'validation',valid),gold_labels=T.split_labels(valid));lengths={}
  else:
   maxlen=512 if a.task=='phoner' else 256
   train_loader,ts=make_custom_loader(train,a.task,tokenizer,labels,batch,a.seed,True,maxlen,a.limit,a.task=='vinli')
   valid_loader,vs=make_custom_loader(valid,a.task,tokenizer,labels,batch,a.seed,False,maxlen,None)
   lengths={'train':ts,'validation':vs}
  steps=(math.ceil(len(train_loader)/accum) if a.model=='bami' else len(train_loader))*a.epochs
  warmup=round(steps*.1);scheduler=get_linear_schedule_with_warmup(optimizer,warmup,steps)
  cfg={'task':a.task,'model':a.model,'seed':a.seed,'learning_rate':a.lr,'epochs':a.epochs,'batch_size':32,'micro_batch_size':batch,'gradient_accumulation_steps':accum,'warmup_steps':warmup,'total_steps':steps,'max_length':512 if a.task=='phoner' else 256,'precision':precision,'attention_implementation':'sdpa','optimizer':'E14 AdamW' if a.model=='bami' else 'E7 LLRD gamma 0.8','include_test':False,'model_spec':spec,'length_stats':lengths,'pid':os.getpid(),'gpu':a.gpu}
  write(out/'resolved_config.json',cfg)
  best=-math.inf;best_epoch=0;curves=[];global_step=0
  for epoch in range(1,a.epochs+1):
   model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;count=0;t0=time.time()
   for bi,b in enumerate(train_loader):
    x=T._move_batch(b,device)
    with T.make_autocast(precision,device):
     logits=model(x['input_ids'],x['attention_mask'],token_type_ids=x.get('token_type_ids'))
     loss=nn.functional.cross_entropy(logits.reshape(-1,classes),x['labels'].reshape(-1),ignore_index=-100) if a.task=='phoner' else T.compute_loss(logits,x['labels'],num_aspects=0)
     if a.model=='bami':
      start=(bi//accum)*accum;end=min(start+accum,len(train_loader));window=sum(min(batch,len(train_loader.dataset)-j*batch) for j in range(start,end));scaled=loss*(len(b['ids'])/window)
     else:scaled=loss
    if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss epoch {epoch} batch {bi}')
    scaled.backward();loss_sum+=float(loss.detach())*len(b['ids']);count+=len(b['ids'])
    if a.model=='student' or (bi+1)%accum==0 or bi+1==len(train_loader):
     torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
   metrics,rows=validation(model,valid_loader,a.task,labels,device,precision,bundle);score=float(metrics['selection_f1'])
   curves.append({'epoch':epoch,'global_step':global_step,'train_loss':loss_sum/count,'selection_f1':score,'validation':metrics,'epoch_seconds':time.time()-t0});write(out/'epoch_curves.json',{'epochs':curves})
   if score>best:
    best=score;best_epoch=epoch;write(out/'best_validation_metrics.json',metrics);write(out/'validation_predictions.json',{'rows':rows,'epoch':epoch});torch.save({'model':model.state_dict(),'epoch':epoch},out/'best_checkpoint.pt')
   print(f'{a.model}/{a.task}/{a.seed}/{a.lr} epoch {epoch} f1 {score} best {best}@{best_epoch}',flush=True)
  del model
  if a.task=='phoner':reloaded=TokenModel(backbone,classes)
  elif a.model=='bami':reloaded,_=build_bamibert_classifier(backbone,num_classes=classes,attention_implementation='sdpa')
  else:reloaded,_=build_classifier(backbone,num_classes=classes,local_files_only=True,attention_implementation='sdpa')
  reloaded.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)['model'],strict=True);reloaded.to(device)
  rm,rr=validation(reloaded,valid_loader,a.task,labels,device,precision,bundle);saved=read(out/'validation_predictions.json')['rows'];fm=evaluate(a.task,labels,[r['gold'] for r in saved],[r['prediction'] for r in saved]);expected=list(T.split_ids(valid)) if bundle is not None else valid.ids
  checks={'reload_matches_best':abs(rm['selection_f1']-best)<1e-9,'prediction_file_matches_best':abs(fm['selection_f1']-best)<1e-9,'reload_predictions_identical':[r['prediction'] for r in rr]==[r['prediction'] for r in saved],'ids_stable_and_complete':[r['id'] for r in rr]==[r['id'] for r in saved]==expected}
  status='completed' if all(checks.values()) else 'completed_with_check_failure';write(out/'metadata.json',{**cfg,'status':status,'best_epoch':best_epoch,'best_selection_f1':best,'postrun_checks':checks,'elapsed_seconds':time.time()-started});write(out/'status.json',{'status':status,'pid':os.getpid(),'gpu':a.gpu,'best_epoch':best_epoch,'best_selection_f1':best});return 0 if status=='completed' else 1
 except Exception as ex:
  write(out/'status.json',{'status':'failed','pid':os.getpid(),'gpu':a.gpu,'error':repr(ex),'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-started});traceback.print_exc();return 1
if __name__=='__main__':sys.exit(main())
