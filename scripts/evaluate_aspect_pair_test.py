"""Locked Restaurant test: A42/A43/T43 inference; T42 and Bami42 reuse saved predictions."""
import os,sys,json,time,gc,importlib.util
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('aspect_eval_common',ROOT/'scripts/cafebert_aspect_pair/common.py');c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
import torch
out=ROOT/'artifacts/cafebert_aspect_pair_test';out.mkdir(exist_ok=True)
paths={'A42':ROOT/'artifacts/cafebert_aspect_pair/runs/A_absa_restaurant_seed42','A43':ROOT/'artifacts/cafebert_aspect_pair_confirmation/seed43/A_absa_restaurant_seed43','T43':ROOT/'artifacts/cafebert_aspect_pair_confirmation/seed43/T_absa_restaurant_seed43'}
if (out/'summary.json').exists():raise RuntimeError('Already evaluated; reuse outputs')
manifest={'created_at':c.now(),'authorization':'User requested test benchmark of arm A vs CafeBERT and BamiBERT; no training or reselection','selection':'existing best validation checkpoints; seeds42/43 fixed before test','script_sha256':c.sha256_file(Path(__file__)),'checkpoints':{}}
for key,p in paths.items():
 m=c.read_json(p/'metadata.json');cp=p/'best_checkpoint.pt';h=c.sha256_file(cp);assert h==m['checkpoint_sha256']
 if key.startswith('A'):assert m['aspect_mapping']==c.mapping_table('A')
 manifest['checkpoints'][key]={'path':str(cp.relative_to(ROOT)),'sha256':h,'seed':m['seed'],'epoch':m['best_epoch'],'validation_f1':m['best_selection_f1']}
c.write_json(out/'locked_manifest.json',manifest)
# Test is opened only after freezing checkpoint identities above.
bundle=c.e1_train.load_task_bundle('absa_restaurant',ROOT,include_test=True);split=c.e1_train.split_field(bundle,'test');reviews=c.reviews_of(split);assert len(reviews)==1938
ids=[r.id for r in reviews];gold=[list(r.gold) for r in reviews]
tok,_=c.load_tokenizer();pairs=c.encode_pairs(tok,reviews,'A');device=torch.device('cuda:0')
summary={};access=[]
def score(key,rows,meta):
 assert [str(r['id']) for r in rows]==ids and [r['gold'] for r in rows]==gold
 metrics=c.e1_train.evaluate_predictions('absa_restaurant',bundle,gold,[r['prediction'] for r in rows],split='test')
 c.write_json(out/(key+'_predictions.json'),{'rows':rows});c.write_json(out/(key+'_metrics.json'),metrics)
 summary[key]={'f1':metrics['selection_f1'],'metrics':metrics,**meta};print(key,metrics['selection_f1'],flush=True)
for key,p in paths.items():
 access.append({'model':key,'started_at':c.now(),'checkpoint_sha256':manifest['checkpoints'][key]['sha256']});c.write_json(out/'test_access_log.json',{'events':access})
 started=time.time()
 if key.startswith('A'):model,_=c.build_model()
 else:
  contract={'num_classes':48,'num_aspects':12,'aspect_class_counts':(4,)*12};model,_=c.cdbk.build_model('cafebert',contract)
 state=c.load_model_checkpoint(p/'best_checkpoint.pt',model);assert state['epoch']==manifest['checkpoints'][key]['epoch']
 model.to(device).eval()
 if key.startswith('A'):
  pp=c.predict_pairs(model,tok,pairs,device,'bf16',micro=64);pred=c.regroup(reviews,pp);rows=[{'id':r.id,'gold':list(r.gold),'prediction':v} for r,v in zip(reviews,pred)]
 else:
  loader=c.e1_train._make_loader(tok,split,max_length=256,batch_size=32,shuffle=False,seed=43,labels=c.e1_train.model_labels(bundle,'test',split),gold_labels=c.e1_train.split_labels(split))
  _,rows=c.e1_train.run_test(model,loader,device,'bf16','absa_restaurant',bundle)
 score(key,rows,{'seed':manifest['checkpoints'][key]['seed'],'elapsed_seconds':time.time()-started,'reused':False})
 access[-1]['finished_at']=c.now();c.write_json(out/'test_access_log.json',{'events':access});del model,state;gc.collect();torch.cuda.empty_cache()
reuse={'T42':ROOT/'artifacts/sealion_benchmark/test/cafebert/absa_restaurant','Bami42':ROOT/'artifacts/e14/trials/absa_restaurant/lr_5e-05/seed_42'}
for key,p in reuse.items():
 d=json.load(open(p/'test_predictions.json'));rows=d['rows'] if isinstance(d,dict) else d
 score(key,rows,{'seed':42,'reused':True,'source':str(p.relative_to(ROOT))})
summary['paired_deltas_pp']={str(s):100*(summary[f'A{s}']['f1']-summary[f'T{s}']['f1']) for s in [42,43]}
summary['mean_paired_delta_pp']=sum(summary['paired_deltas_pp'].values())/2
c.write_json(out/'summary.json',summary)
print('DONE',summary['paired_deltas_pp'],flush=True)
