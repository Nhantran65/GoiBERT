#!/usr/bin/env python3
"""Count absent and truncated PhoNER syllables for both tokenizers."""
import random
from transformers import AutoTokenizer
import common as C

def inspect(tokenizer,split,max_length=512):
    missing=truncated=sentences=0;examples=[]
    for sid,words in zip(split.ids,split.words):
        full=tokenizer(words,is_split_into_words=True,add_special_tokens=True,truncation=False)
        cut=tokenizer(words,is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=max_length)
        all_words=set(w for w in full.word_ids() if w is not None);seen=set(w for w in cut.word_ids() if w is not None)
        absent=set(range(len(words)))-all_words;lost=all_words-seen
        missing+=len(absent);truncated+=len(lost);sentences+=len(full['input_ids'])>max_length
        for w in sorted(absent|lost):
            if len(examples)<20:examples.append({'id':sid,'word_index':w,'word':words[w],'reason':'no_subword' if w in absent else 'truncated'})
    return {'sentences':len(split.ids),'words':sum(map(len,split.words)),'words_without_subword':missing,'truncated_words':truncated,'truncated_sentences':sentences,'first_20_examples':examples}

def main():
    b=C.load('phoner');assert tuple(b.label_names)==C.PHONER_LABELS
    output={'labels':list(C.PHONER_LABELS),'max_length':512,'tokenizers':{}};tokenizers={}
    for name,source,rev in [('student',C.ROOT/'models/goiturn-s-gd',None),('cafe',C.TOK,C.TOK_REV)]:
        tok=AutoTokenizer.from_pretrained(source,revision=rev,local_files_only=True,use_fast=True)
        tokenizers[name]=tok
        output['tokenizers'][name]={'vocab_size':len(tok),'train':inspect(tok,b.train),'dev':inspect(tok,b.validation)}
    rng=random.Random(42);ix=rng.sample(range(len(b.train.ids)),20)
    cross=[]
    for i in ix:
        sid=b.train.ids[i];words=b.train.words[i];n=len(words)
        counts={name:len(set(w for w in tok(words,is_split_into_words=True,truncation=True,max_length=512).word_ids() if w is not None)) for name,tok in tokenizers.items()}
        ok=counts['student']==counts['cafe']==n
        cross.append({'id':sid,'teacher_words':counts['cafe'],'student_words':counts['student'],'gold_words':n,'pass':ok})
        assert ok
    output['cross_tokenizer_20_sentences']=cross
    C.write(C.ART/'phoner/alignment_check.json',output)
    print(output,flush=True)

if __name__=='__main__':main()
