#!/usr/bin/env python3
"""Initialization invariants and step-zero loss before any long run."""
import json, sys
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer, RobertaModel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from gd import ARCHS, config, pca_initialize, teacher_layers_initialize
from port import DistillationAligner, MAPPINGS, iter_packed_batches

def main():
 torch.manual_seed(42)
 teacher=AutoModel.from_pretrained(ROOT/'models/bamibert-base-original',local_files_only=True,attn_implementation='eager',add_pooling_layer=False).eval()
 tok=AutoTokenizer.from_pretrained(ROOT/'models/bamibert-base-original',local_files_only=True)
 batch=next(iter_packed_batches(ROOT/'data/corpus_v2/mlm_mix.txt',tok,2,256))
 ids=batch['input_ids'];mask=batch['attention_mask']
 with torch.inference_mode():
  t=teacher(input_ids=ids,attention_mask=mask,output_hidden_states=True,output_attentions=True,return_dict=True)
 out={'backbones':{},'m6_copy_max_abs_diff':None,'m6_step0_loss':{}}
 for variant in ARCHS:
  s=RobertaModel(config(variant),add_pooling_layer=False).eval()
  count=sum(p.numel() for p in s.parameters())
  if variant=='m6':teacher_layers_initialize(s,teacher)
  else:pca_initialize(s,teacher)
  with torch.inference_mode():
   so=s(input_ids=ids,attention_mask=mask,output_hidden_states=True,output_attentions=True,return_dict=True)
  out['backbones'][variant]={'parameters':count,'architecture':ARCHS[variant]}
  if variant=='m6':
   extended=teacher.get_extended_attention_mask(mask,ids.shape)
   x=teacher.embeddings(input_ids=ids)
   diffs=[(so.hidden_states[0]-x).abs().max().item()]
   for k,layer_idx in enumerate((1,3,5,7,9,11),1):
    x=teacher.encoder.layer[layer_idx](x,attention_mask=extended)[0]
    diffs.append((so.hidden_states[k]-x).abs().max().item())
   out['m6_copy_max_abs_diff']=max(diffs)
   aligner=DistillationAligner(768,MAPPINGS['m6'],identity=True)
   with torch.no_grad():h,a=aligner(so,t,mask)
   out['m6_step0_loss']['teacher_layers']=float(h+a)
   random_s=RobertaModel(config('m6'),add_pooling_layer=False).eval()
   with torch.inference_mode():r=random_s(input_ids=ids,attention_mask=mask,output_hidden_states=True,output_attentions=True,return_dict=True)
   with torch.no_grad():h,a=aligner(r,t,mask)
   out['m6_step0_loss']['random']=float(h+a)
   assert out['m6_copy_max_abs_diff']<=1e-6,out
   assert out['m6_step0_loss']['teacher_layers']<out['m6_step0_loss']['random']*0.8,out
  del s,so
 dest=ROOT/'artifacts/family_sm/preflight.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))

if __name__=='__main__':main()
