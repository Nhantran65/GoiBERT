#!/usr/bin/env python3
"""Five-arm GoiTurn-S general distillation, with exact data/RNG resume."""
from __future__ import annotations
import argparse, hashlib, json, os, random, shutil, time
from pathlib import Path
import torch, transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaConfig, RobertaModel, get_linear_schedule_with_warmup
from port import DistillationAligner, MAPPING, iter_packed_batches

ROOT=Path(__file__).resolve().parents[2]
ARCH={'num_hidden_layers':10,'hidden_size':384,'num_attention_heads':6,'intermediate_size':1536,'max_position_embeddings':2050}
ARMS={'A1':('v1','dapt',30000),'A2':('v2','dapt',30000),'A3':('v1','dapt',120000),'A4':('v2','dapt',120000),'A5':('v1','bami',30000)}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def check_corpus(path):
    if path.parent.name=='corpus_v2':
        ready=path.parent/'READY'
        if not ready.is_file(): raise FileNotFoundError('corpus v2 READY missing')
        value=ready.read_text()
        expected=None
        for word in value.replace('"',' ').replace(':',' ').split():
            if len(word)==64 and all(c in '0123456789abcdef' for c in word.lower()): expected=word.lower();break
        if expected is None:
            try:
                obj=json.loads(value)
                expected=obj.get('mlm_mix_sha256') or obj.get('sha256')
            except json.JSONDecodeError: pass
        if expected is None: raise ValueError('READY lacks sha256')
        actual=sha(path)
        if actual!=expected:raise ValueError(f'corpus v2 hash mismatch {actual} != {expected}')
        return actual
    return sha(path)

def config():
    c=RobertaConfig(vocab_size=20481,type_vocab_size=1,pad_token_id=1,bos_token_id=0,eos_token_id=2,position_embedding_type='absolute',hidden_act='gelu',hidden_dropout_prob=.1,attention_probs_dropout_prob=.1,layer_norm_eps=1e-5,initializer_range=.02,use_cache=False,**ARCH)
    c._attn_implementation='eager';return c

def pca_initialize(student,teacher):
    tw=teacher.get_input_embeddings().weight.detach().float().cpu();sw=student.get_input_embeddings().weight
    if tw.shape[0]!=sw.shape[0]:raise ValueError(f'vocabulary mismatch: {tw.shape} vs {sw.shape}')
    centered=tw-tw.mean(0,keepdim=True)
    _,_,components=torch.pca_lowrank(centered,q=384,center=False,niter=2)
    projected=centered@components;projected=projected/projected.std(0,unbiased=False).clamp_min(1e-12)
    initialized=projected*sw.detach().float().std(0,unbiased=False).cpu()
    with torch.no_grad():sw.copy_(initialized.to(dtype=sw.dtype,device=sw.device))

def latest(out):
    candidates=[]
    for p in (out/'checkpoints').glob('checkpoint-*'):
        if (p/'training_state.pt').is_file() and (p/'config.json').is_file(): candidates.append(p)
    return sorted(candidates)[-1] if candidates else None

def save_checkpoint(out,step,student,aligner,opt,sched,tail,elapsed,corpus_sha):
    base=out/'checkpoints';base.mkdir(parents=True,exist_ok=True)
    tmp=base/f'.checkpoint-{step:08d}.tmp';final=base/f'checkpoint-{step:08d}'
    if tmp.exists():shutil.rmtree(tmp)
    tmp.mkdir()
    student.save_pretrained(tmp,safe_serialization=True)
    torch.save({'step':step,'aligner':aligner.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'tail':tail[-20:],'elapsed_seconds':elapsed,'corpus_sha256':corpus_sha,'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},tmp/'training_state.pt')
    tmp.rename(final)
    for old in sorted(base.glob('checkpoint-*'))[:-2]:shutil.rmtree(old)
    print(f'checkpoint={step}',flush=True)

def train(a):
    arm_corpus,arm_teacher,arm_steps=ARMS[a.arm]
    if not a.smoke_steps and a.steps!=arm_steps:raise ValueError('Steps differ from locked arm plan')
    if not a.smoke_steps and (a.corpus.resolve()!=ROOT/'data'/f'corpus_{arm_corpus}'/'mlm_mix.txt' or a.teacher.resolve()!=ROOT/'models'/('goibert-b-dapt' if arm_teacher=='dapt' else 'bamibert-base-original')):raise ValueError('Corpus/teacher differ from locked arm plan')
    corpus_sha=check_corpus(a.corpus)
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    device=torch.device('cuda:0');torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
    teacher=AutoModel.from_pretrained(a.teacher,local_files_only=True,attn_implementation='eager',add_pooling_layer=False).to(device).eval()
    if teacher.config.num_hidden_layers!=12 or teacher.config.hidden_size!=768:raise ValueError('Incorrect teacher architecture')
    for p in teacher.parameters():p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(a.teacher,local_files_only=True)
    resume=latest(a.out) if a.resume else None
    if resume:
        student=RobertaModel.from_pretrained(resume,local_files_only=True,attn_implementation='eager',add_pooling_layer=False)
    else:
        student=RobertaModel(config(),add_pooling_layer=False);pca_initialize(student,teacher)
    student.to(device);aligner=DistillationAligner().to(device)
    params=list(student.parameters())+list(aligner.parameters())
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=1000,num_training_steps=a.steps)
    step0=0;tail=[];elapsed0=0.
    if resume:
        st=torch.load(resume/'training_state.pt',map_location='cpu',weights_only=False)
        if st['corpus_sha256']!=corpus_sha:raise ValueError('Resume corpus hash mismatch')
        step0=st['step'];aligner.load_state_dict(st['aligner']);optimizer.load_state_dict(st['optimizer']);scheduler.load_state_dict(st['scheduler']);tail=st['tail'];elapsed0=st['elapsed_seconds']
        random.setstate(st['python_rng']);torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state_all(st['cuda_rng'])
    if step0>a.steps:raise ValueError('Checkpoint beyond requested steps')
    batches=iter_packed_batches(a.corpus,tok,32,256,skip_sequences=step0*32)
    logpath=a.out/'loss_curve.jsonl';a.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();window=start
    with logpath.open('a',encoding='utf-8') as log:
        for step in range(step0+1,a.steps+1):
            b=next(batches);ids=b['input_ids'].to(device);mask=b['attention_mask'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                t=teacher(input_ids=ids,attention_mask=mask,output_hidden_states=True,output_attentions=True,return_dict=True)
            student.train();aligner.train()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                s=student(input_ids=ids,attention_mask=mask,output_hidden_states=True,output_attentions=True,return_dict=True)
            with torch.autocast('cuda',enabled=False):
                h,att=aligner(s,t,mask);loss=h+att
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss step {step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            values={'total':float(loss.detach()),'hidden':float(h.detach()),'attention':float(att.detach())};tail.append(values);tail=tail[-20:]
            if step==1 or step%100==0 or step==a.steps or (a.smoke_steps and 101<=step<=110):
                torch.cuda.synchronize();now=time.perf_counter()
                row={'step':step,**values,'seconds_per_step':(now-window)/(100 if step%100==0 else 1),'peak_vram_bytes':torch.cuda.max_memory_allocated(0),'elapsed_seconds':elapsed0+now-start}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True);window=now
            if step%(100 if a.smoke_steps else 2000)==0 or step==a.steps:
                save_checkpoint(a.out,step,student,aligner,optimizer,scheduler,tail,elapsed0+time.perf_counter()-start,corpus_sha)
    final=a.out/'final';final.mkdir(parents=True,exist_ok=True)
    student.save_pretrained(final,safe_serialization=True);tok.save_pretrained(final)
    manifest={'model':'goiturn_s_gd','student':'goiturn_s','completed_steps':a.steps,'teacher_checkpoint':str(a.teacher.resolve()),'teacher_fallback_used':False,'corpus':str(a.corpus.resolve()),'corpus_stats_sha256':sha(a.corpus.parent/'stats.json'),'loss_curve_tail':tail,'seed':42,'architecture':ARCH,'max_length':256,'batch_size':32,'learning_rate':1e-4,'weight_decay':.01,'warmup_steps':1000,'grad_clip':1.,'checkpoint_every':100 if a.smoke_steps else 2000,'loss':'mapped hidden-state MSE + head-mean attention MSE; no labels or logits','layer_mapping':MAPPING,'pca_initialization':'selected teacher word embeddings','torch_version':torch.__version__,'transformers_version':transformers.__version__,'device_name':torch.cuda.get_device_name(0),'attention_implementation':'eager','amp_fp16':False,'smoke':bool(a.smoke_steps),'corpus_sha256':corpus_sha,'precision':'bf16 autocast; fp32 loss; no GradScaler','elapsed_seconds':elapsed0+time.perf_counter()-start,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES','0')}
    (final/'distill_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(f'final={final}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    a=p.parse_args()
    if a.smoke_steps:a.steps=a.smoke_steps
    train(a)
