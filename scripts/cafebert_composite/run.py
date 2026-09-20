"""One CafeBERT CE composite pilot; immutable historical recipes imported, not edited."""
import os,sys,json,hashlib,importlib.util,argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/cdbk_pilot'));sys.path.insert(0,str(ROOT/'src'))
import common as c
spec=importlib.util.spec_from_file_location('composite_cafe_finetune',ROOT/'scripts/cdbk_pilot/finetune.py');ft=importlib.util.module_from_spec(spec);spec.loader.exec_module(ft)
spec2=importlib.util.spec_from_file_location('e20_composite_source',ROOT/'scripts/run_e20.py');e20=importlib.util.module_from_spec(spec2);sys.modules[spec2.name]=e20;spec2.loader.exec_module(e20)
OUT=ROOT/'artifacts/cafebert_composite';OUT.mkdir(exist_ok=True)
original_contract=c.task_contract
source=ROOT/'artifacts/e20/d1_uniform/absa_restaurant/seed_42/composites_manifest.json'
data=json.load(open(source));comps=data['composites'];assert len(comps)==2343 and data['sampling_mode']=='uniform'
contract=original_contract();train=contract['train'];rows={str(x.id):x for x in train.examples};assert len(rows)==7028
for x in comps:
 a,b=rows[x['a_id']],rows[x['b_id']];ga=list(a.label);gb=list(b.label);pa=[i for i,v in enumerate(ga) if v];pb=[i for i,v in enumerate(gb) if v]
 assert len(pa)==len(pb)==1 and pa[0]!=pb[0]
 assert x['text']==a.text+' '+b.text and x['a_text']==a.text and x['b_text']==b.text
 assert x['label']==[u if u else v for u,v in zip(ga,gb)]
newbundle,newtrain=e20._inject_composites(contract['bundle'],comps)
newcontract={**contract,'bundle':newbundle,'train':newtrain}
assert len(newtrain)==9371 and newbundle.splits['validation']==contract['bundle'].splits['validation']
c.task_contract=lambda:newcontract
c.ARMS['C']={'init':'cafebert','learning_rate':2e-5,'kd':False,'head_hidden':1024}
c.PROTOCOL=ft.PROTOCOL='CafeBERT-composite-uniform-restaurant-seed42-v1'
basehash=c.source_hash();scriptsha=c.sha256_file(Path(__file__));c.source_hash=lambda:hashlib.sha256((basehash+scriptsha+c.sha256_file(source)).encode()).hexdigest()
original_build=c.build_model
head_hash=[]
def build(init,cont):
 model,ms=original_build(init,cont);h=hashlib.sha256()
 for k,v in model.head.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 head_hash.append(h.hexdigest());return model,ms
c.build_model=build
p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',type=int,default=0);p.add_argument('--device',default='cuda:0');args=p.parse_args()
if args.prepare:
 tok,_=c.load_tokenizer('cafebert');lens=[len(x) for x in tok([x['text'] for x in comps],truncation=False)['input_ids']];assert max(lens)<=256
 natural=set(c.e1_train.split_texts(contract['validation']));assert not any(x['text'] in natural for x in comps)
 info={'status':'passed','seed':42,'source_manifest':str(source.relative_to(ROOT)),'source_sha256':c.sha256_file(source),'natural_train':7028,'composites':2343,'augmented_train':9371,'validation':771,'max_composite_tokens_xlmr':max(lens),'truncation_count':0,'source_labels_union_exact':True,'all_sources_train_only':True,'validation_unchanged':True,'steps_per_epoch':293,'total_steps':8790,'warmup_steps':879,'baseline_steps':6600,'note':'Same30epochs; extra data adds33percent optimizersteps, practical augmentation screen not compute-matched.'}
 c.write_json(OUT/'preflight.json',info);print(json.dumps(info,indent=2));sys.exit(0)
assert c.read_json(OUT/'preflight.json')['status']=='passed'
runout=OUT/('smoke' if args.smoke else 'run_seed42')
meta=ft.run(SimpleNamespace(arm='C',out_dir=str(runout),device=args.device,micro_batch=32,teacher_cache=None,smoke=args.smoke))
meta.update({'augmentation':c.read_json(OUT/'preflight.json'),'head_init_sha256':head_hash[0],'inference':'identical CafeBERT48logit head, oneencoderpass; noaugmentation atprediction'})
c.write_json(runout/'metadata.json',meta)
if not args.smoke:
 c.write_json(OUT/'summary.json',{'status':meta['status'],'validation_f1':meta['best_selection_f1'],'delta_vs_cafe_pp':100*(meta['best_selection_f1']-0.7771971496437055),'best_epoch':meta['best_epoch'],'elapsed_seconds':meta['elapsed_seconds'],'postrun_checks':meta['postrun_checks']})
