"""GD alignment and sequence packing ported from family-v1 distill/mlm_pretrain."""
from pathlib import Path
import torch
from torch import nn

MAPPINGS = {
    's_prime': [0,1,2,4,5,6,7,8,10,11,12],
    'm6': [0,2,4,6,8,10,12],
    'm12': list(range(13)),
}

def iter_packed_sequences(corpus_path: Path, tokenizer, max_length=256):
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    payload_length = max_length-special_count
    if payload_length < 1: raise ValueError('max_length too small')
    separator_id = tokenizer.sep_token_id
    if separator_id is None: separator_id = tokenizer.eos_token_id
    while True:
        produced=0; buffer=[]
        with corpus_path.open('r',encoding='utf-8') as handle:
            for raw_line in handle:
                line=raw_line.strip()
                if not line: continue
                token_ids=tokenizer.encode(line,add_special_tokens=False)
                if not token_ids: continue
                if buffer and separator_id is not None: buffer.append(separator_id)
                buffer.extend(token_ids)
                while len(buffer)>=payload_length:
                    payload=buffer[:payload_length]; del buffer[:payload_length]
                    yield tokenizer.build_inputs_with_special_tokens(payload); produced+=1
        if buffer:
            yield tokenizer.build_inputs_with_special_tokens(buffer); produced+=1
        if produced==0: raise ValueError('Corpus has no tokenizable lines')

def iter_packed_batches(path, tokenizer, batch_size=32, max_length=256, skip_sequences=0):
    from itertools import islice
    iterator=iter_packed_sequences(path,tokenizer,max_length)
    if skip_sequences:
        for _ in islice(iterator,skip_sequences): pass
    pad_id=tokenizer.pad_token_id
    while True:
        sequences=[next(iterator) for _ in range(batch_size)]
        longest=max(map(len,sequences))
        ids=torch.full((batch_size,longest),pad_id,dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,seq in enumerate(sequences): ids[i,:len(seq)]=torch.tensor(seq);mask[i,:len(seq)]=1
        yield {'input_ids':ids,'attention_mask':mask}

def hidden_mse(student,teacher,mask):
    losses=(student.float()-teacher.detach().float()).pow(2).mean(-1)
    mask=mask.to(losses.dtype)
    return (losses*mask).sum()/mask.sum().clamp_min(1)

def attention_mse(student,teacher,mask):
    s=student.float().mean(1);t=teacher.detach().float().mean(1)
    pair=mask.float()[:, :, None]*mask.float()[:, None, :]
    return ((s-t).pow(2)*pair).sum()/pair.sum().clamp_min(1)

class DistillationAligner(nn.Module):
    def __init__(self, hidden_size, mapping, identity=False):
        super().__init__()
        self.mapping = mapping
        self.projections=nn.ModuleList(nn.Linear(hidden_size,768) for _ in mapping)
        if identity:
            assert hidden_size == 768
            for p in self.projections:
                nn.init.eye_(p.weight)
                nn.init.zeros_(p.bias)
    def forward(self,s,t,mask):
        assert len(s.hidden_states)==len(self.mapping) and len(s.attentions)==len(self.mapping)-1
        hidden=[hidden_mse(self.projections[i](s.hidden_states[i].float()),t.hidden_states[j],mask) for i,j in enumerate(self.mapping)]
        attention=[attention_mse(s.attentions[i],t.attentions[j-1],mask) for i,j in enumerate(self.mapping[1:])]
        return torch.stack(hidden).mean(),torch.stack(attention).mean()
