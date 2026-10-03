#!/usr/bin/env python3
"""Inference speed, memory, and weight-only checkpoint footprint."""
from __future__ import annotations
import json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT/'scripts'):sys.path.insert(0,str(p))
import torch
from goibert.modeling import build_classifier
from goibert_e14.modeling import build_bamibert_classifier
from run_e14 import MODEL_DIR

ART=ROOT/'artifacts/final_test';DATA={
 'student':ROOT/'artifacts/cafe_oof_kd/student/kd/vsfc_sentiment/seed_42/best_checkpoint.pt',
 'bami':ROOT/'artifacts/e14/trials/vsfc_sentiment/lr_2e-05/seed_42/best_checkpoint.pt'}
def snapshot():
 def run(args):return subprocess.check_output(['nvidia-smi',*args],text=True).strip()
 return {'gpus':run(['--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),'processes':run(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])}
def model_for(name):
 if name=='student':m,_=build_classifier(ROOT/'models/goiturn-s-gd',num_classes=3,local_files_only=True,attention_implementation='sdpa')
 else:m,_=build_bamibert_classifier(MODEL_DIR,num_classes=3,attention_implementation='sdpa')
 state=torch.load(DATA[name],map_location='cpu',weights_only=False,mmap=True)
 m.load_state_dict(state['model'],strict=True);del state
 return m.eval()
def measure_gpu(m,batch,length):
 device=torch.device('cuda:0');m=m.to(device);ids=torch.randint(4,int(m.encoder.config.vocab_size),(batch,length),device=device);mask=torch.ones_like(ids)
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
 def one():
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):m(ids,mask)
 for _ in range(20):one()
 torch.cuda.synchronize();samples=[]
 for _ in range(200):
  t=time.perf_counter();one();torch.cuda.synchronize();samples.append((time.perf_counter()-t)*1000)
 median=statistics.median(samples)
 result={'median_ms':median,'p90_ms':sorted(samples)[179],'throughput_examples_per_second':batch*1000/median,'peak_vram_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'batch':batch,'length':length,'warmup':20,'iterations':200,'precision':'bf16'}
 del ids,mask;m.to('cpu');torch.cuda.empty_cache();return result
def measure_cpu(m,batch,length):
 m=m.cpu().eval();ids=torch.randint(4,int(m.encoder.config.vocab_size),(batch,length));mask=torch.ones_like(ids)
 def one():
  with torch.inference_mode():m(ids,mask)
 for _ in range(5):one()
 samples=[]
 for _ in range(50):
  t=time.perf_counter();one();samples.append((time.perf_counter()-t)*1000)
 return {'median_ms':statistics.median(samples),'batch':batch,'length':length,'warmup':5,'iterations':50,'precision':'fp32'}
def main():
 assert (ART/'test_results.json').is_file(),'Run only after test is complete'
 gpu=torch.device('cuda:0');free,total=torch.cuda.mem_get_info(gpu);assert free>=12e9
 torch.set_num_threads(8)
 out={'started_at_unix':time.time(),'gpu':'0','gpu_name':torch.cuda.get_device_name(gpu),'other_processes_at_start':snapshot(),'torch_num_threads_cpu':8,'models':{},'gpu':{},'cpu':{},'checkpoint_definition':'weight-only torch.save(state_dict), including encoder and sentence classification head'}
 with tempfile.TemporaryDirectory(prefix='goibert-final-eff-',dir='/tmp') as td:
  for name in ('bami','student'):
   m=model_for(name);params=sum(p.numel() for p in m.parameters());sizes={}
   for precision,dtype in [('fp32',torch.float32),('bf16',torch.bfloat16)]:
    p=Path(td)/f'{name}-{precision}.pt';torch.save({k:v.detach().to(dtype) for k,v in m.state_dict().items()},p);sizes[precision]=p.stat().st_size
   out['models'][name]={'parameters':params,'checkpoint_bytes':sizes,'original_checkpoint_bytes':DATA[name].stat().st_size}
   out['gpu'][name]={}
   for batch in (1,32):
    for length in (64,128,256):
     key=f'b{batch}_l{length}';out['gpu'][name][key]=measure_gpu(m,batch,length)
     print('GPU',name,key,out['gpu'][name][key],flush=True)
   out['cpu'][name]={}
   for length in (64,128):
    key=f'b1_l{length}';out['cpu'][name][key]=measure_cpu(m,1,length)
    print('CPU',name,key,out['cpu'][name][key],flush=True)
   del m
 out['other_processes_at_end']=snapshot();out['elapsed_seconds']=time.time()-out['started_at_unix']
 for device in ('gpu','cpu'):
  out[device+'_student_speedup']={key:out[device]['bami'][key]['median_ms']/v['median_ms'] for key,v in out[device]['student'].items()}
 birth=int(subprocess.check_output(['stat','-c','%W',str(ART)],text=True))
 out['gpu_hours_upper_bound']={'hours':(time.time()-birth)/3600,'method':'One GPU assigned; wall time from final_test directory creation before any job through efficiency output write, including inactive coding time and all validation/test/retry jobs.','start_unix_floor_seconds':birth,'end_unix_seconds':time.time()}
 (ART/'efficiency.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print('EFFICIENCY_COMPLETE',flush=True)
if __name__=='__main__':main()
