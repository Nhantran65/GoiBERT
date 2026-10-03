"""Build the GD v2 corpus from pinned local text, using CPU only."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/corpus_v2'
ART = ROOT / 'artifacts/mmbert_vi_cpt_pilot/v2'
PILOT = ROOT / 'data/mmbert_vi_cpt_pilot'
RAW = ROOT / 'data/raw'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/bench8'))
from goibert.data import load_task
from transformers import AutoTokenizer
import pyarrow.parquet as pq

TOK = None
SPLITS = ('validation', 'test')
TASKS = ('vsfc', 'visfd', 'restaurant', 'hotel', 'vispam', 'vinli', 'phoner')
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
SPACE = re.compile(r'\s+')
ID_RE = re.compile(rb'"id"\s*:\s*"([0-9a-f]{32})"')


def tok():
    global TOK
    if TOK is None:
        TOK = AutoTokenizer.from_pretrained(str(ROOT/'models/goiturn-s-gd'), local_files_only=True)
        TOK.model_max_length = 10**9
    return TOK


def count(text):
    return len(tok().encode(text, add_special_tokens=False))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def norm(text):
    return SPACE.sub(' ', unicodedata.normalize('NFC', text).casefold()).strip()


def syllables(text):
    return PUNCT.sub(' ', unicodedata.normalize('NFC', text).lower()).split()


def task_texts():
    """Train through frozen loaders; val/test text only, with no test label access."""
    train, locks, paths = {}, {}, set()
    for key, task in [('vsfc','vsfc_sentiment'),('visfd','visfd'),('restaurant','absa_restaurant'),('hotel','absa_hotel')]:
        bundle = load_task(task, ROOT, include_test=False)
        train[key] = list(bundle.splits['train'].texts)
        locks[(key,'validation')] = list(bundle.splits['validation'].texts)
        paths.update(bundle.source_files)
    p = RAW/'uit-vsfc/test/sents.txt'
    locks[('vsfc','test')] = p.read_text(encoding='utf-8').splitlines(); paths.add(str(p))
    for key, filename, col in [('visfd','uit-visfd/Test.csv','comment'),('restaurant','uit-absa-public-candidate/dataset/UIT-ABSA-Restaurant/Test.csv','Review'),('hotel','uit-absa-public-candidate/dataset/UIT-ABSA-Hotel/Test.csv','Review')]:
        p = RAW/filename
        with p.open(encoding='utf-8', newline='') as f:
            locks[(key,'test')] = [r[col] for r in csv.DictReader(f)]
        paths.add(str(p))
    from data import load as bench_load
    for key in ('vispam','vinli','phoner'):
        b = bench_load(key)
        if key == 'vispam':
            train[key] = list(b.train.texts)
            locks[(key,'validation')] = list(b.validation.texts)
            for split, fname in [('test','test.csv')]:
                p = RAW/'social/vispamdetection'/fname
                with p.open(encoding='utf-8', newline='') as f: locks[(key,split)] = [r['Comment'] for r in csv.DictReader(f)]
                paths.add(str(p))
            paths.update(str(RAW/'social/vispamdetection'/f'{x}.csv') for x in ('train','dev'))
        elif key == 'vinli':
            train[key] = group_nli(b.train.pairs)
            locks[(key,'validation')] = [s for pair in b.validation.pairs for s in pair]
            p = RAW/'vinli_mirror/test.parquet'
            tb = pq.read_table(p, columns=['sentence1','sentence2'])
            locks[(key,'test')] = [s for pair in zip(tb['sentence1'].to_pylist(), tb['sentence2'].to_pylist()) for s in pair]
            paths.add(str(p)); paths.update(str(RAW/'vinli_mirror'/f'{x}.parquet') for x in ('train','dev'))
        else:
            train[key] = [' '.join(words) for words in b.train.words]
            locks[(key,'validation')] = [' '.join(words) for words in b.validation.words]
            p = RAW/'phoner_covid19/test_syllable.json'
            locks[(key,'test')] = [' '.join(json.loads(line)['words']) for line in p.open(encoding='utf-8') if line.strip()]
            paths.add(str(p)); paths.update(str(RAW/'phoner_covid19'/f'{x}_syllable.json') for x in ('train','dev'))
    return train, locks, paths


def group_nli(pairs):
    grouped = collections.OrderedDict()
    for a,b in pairs: grouped.setdefault(a, []).append(b)
    return ['\n'.join([a,*bs]) for a,bs in grouped.items()]


class Decontam:
    def __init__(self, locks):
        self.names = [(t,s) for t in TASKS for s in SPLITS]
        self.long = {}
        self.short = {}
        self.exact = {}
        self.unique = {}
        for bit, key in enumerate(self.names):
            texts = {norm(x) for x in locks[key] if norm(x)}
            self.unique[f'{key[0]}/{key[1]}'] = len(texts)
            mask = 1 << bit
            for x in texts:
                sy = syllables(x)
                self.exact[x] = self.exact.get(x,0)|mask
                if len(sy)<13:
                    k=' '.join(sy); self.short[k]=self.short.get(k,0)|mask
                else:
                    for i in range(len(sy)-12):
                        h=hash(tuple(sy[i:i+13])); self.long[h]=self.long.get(h,0)|mask

    def hit(self, text):
        sy=syllables(text)
        mask=self.short.get(' '.join(sy),0)
        for i in range(len(sy)-12):
            mask |= self.long.get(hash(tuple(sy[i:i+13])),0)
        return mask

    def names_of(self, mask):
        return [f'{t}/{s}' for i,(t,s) in enumerate(self.names) if mask & (1<<i)]


def split_paragraphs(text):
    return [x.strip() for x in text.splitlines() if x.strip()]


def clean_doc(text, source, dc, seen, counts, hits, max_tokens=2048):
    """Filter before exact dedup, then split at paragraph boundaries."""
    pars=split_paragraphs(text)
    counts[source]['docs_input']+=1; counts[source]['paragraphs_input']+=len(pars)
    if not pars: return []
    bad=[]
    for p in pars:
        m=dc.hit(p); bad.append(m)
        if m:
            for name in dc.names_of(m): hits[source][name]['paragraphs']+=1
    if sum(bool(x) for x in bad)/len(pars)>0.2:
        counts[source]['docs_decontam_dropped']+=1
        for m in set(bit for mask in bad for bit in range(len(dc.names)) if mask&(1<<bit)):
            name='/'.join(dc.names[m]); hits[source][name]['docs']+=1
        return []
    good=[]
    for p,m in zip(pars,bad):
        if m: counts[source]['paragraphs_decontam_dropped']+=1; continue
        k=norm(p)
        if k in seen:
            counts[source]['paragraphs_exact_dedup_dropped']+=1; continue
        seen.add(k); good.append(p)
    if not good: counts[source]['docs_empty_after_filter']+=1; return []
    chunks=[]; cur=[]; size=0
    for p in good:
        n=count(p)
        if n>max_tokens:
            counts[source]['paragraphs_over_2048_dropped']+=1
            continue
        if cur and size+n>max_tokens:
            chunks.append((cur,size)); cur=[];size=0
        cur.append(p); size+=n
    if cur: chunks.append((cur,size))
    counts[source]['docs_after_filter']+=len(chunks)
    counts[source]['paragraphs_after_filter']+=sum(len(c[0]) for c in chunks)
    return chunks


def verify_raw_sha(tmp):
    rec={}
    for line in (RAW/'hplt3_vie/sha256.txt').read_text().splitlines():
        if ' bytes=' in line and ' sha256=' in line:
            name=line.split()[0]; rec[name]=line.rsplit('sha256=',1)[1]
    cache=tmp/'verified_raw_sha.json'
    prior=json.loads(cache.read_text()) if cache.exists() else {}
    for name in ('9_2','9_1','8_4','7_1','10_1','5_1','6_1'):
        p=RAW/'hplt3_vie'/f'{name}.jsonl.zst'
        sig=[p.stat().st_size,p.stat().st_mtime_ns]
        actual=prior.get(name,{}).get('sha256') if prior.get(name,{}).get('stat')==sig else sha(p)
        if actual!=rec[name]: raise RuntimeError(f'raw SHA mismatch {p}')
        prior[name]={'sha256':actual,'stat':sig}
        save(cache,prior)
    return rec


def select_general(target_mm=540_000_000):
    rows=[json.loads(line) for line in (ART/'general_docs.jsonl').open(encoding='utf-8')]
    random.Random(42).shuffle(rows)
    selected=[]; n=0
    for r in rows:
        selected.append(r); n+=r['n_tokens']
        if n>=target_mm: break
    chosen={r['pool_index']:r for r in selected}
    ids={}
    for i,line in enumerate((PILOT/'pool/pool.jsonl').open(encoding='utf-8')):
        if i in chosen:
            row=json.loads(line)
            if int(row['kh64'])!=int(chosen[i]['kh64']): raise RuntimeError(f'pool ordinal mismatch at {i}')
            # pool_index is sourced from tokenized rows; check invariant rather than trusting order.
            ids[row['id']]={'source':'web','shard':row['shard'],'pool_index':i,'kh64':chosen[i]['kh64']}
    if len(ids)!=len(selected): raise RuntimeError(f'pool IDs {len(ids)} vs selected {len(selected)}')
    return ids, len(rows), n


def extract_raw(ids, ugc_ids, spool):
    wanted={**ids, **ugc_ids}
    got=collections.Counter()
    with spool.open('w',encoding='utf-8') as out:
        for shard in ('9_2','9_1','8_4','7_1','10_1','5_1','6_1'):
            keys={k for k,v in wanted.items() if v['shard']==shard}
            if not keys: continue
            p=RAW/'hplt3_vie'/f'{shard}.jsonl.zst'
            proc=subprocess.Popen(['zstd','-dc',str(p)],stdout=subprocess.PIPE,bufsize=8<<20)
            for raw in proc.stdout:
                m=ID_RE.search(raw)
                if not m: continue
                key=m.group(1).decode()
                if key not in keys: continue
                obj=json.loads(raw)
                if obj.get('id')!=key: raise RuntimeError('ID mismatch')
                info=wanted[key]
                out.write(json.dumps({'source':info['source'],'id':key,'text':obj['text'],'pool_index':info.get('pool_index')},ensure_ascii=False)+'\n')
                got[info['source']]+=1
            proc.stdout.close()
            if proc.wait(): raise RuntimeError(f'zstd failed: {p}')
    if got['web']!=len(ids) or got['ugc']!=len(ugc_ids):
        raise RuntimeError(f'raw missing: {dict(got)} expected web={len(ids)} ugc={len(ugc_ids)}')
    return dict(got)


def main():
    a=argparse.ArgumentParser();a.add_argument('--tmp',type=Path,required=True);args=a.parse_args()
    t0=time.time();OUT.mkdir(exist_ok=True);args.tmp.mkdir(parents=True,exist_ok=True)
    train,locks,task_paths=task_texts(); dc=Decontam(locks)
    print('locks',dc.unique,'13grams',len(dc.long),flush=True)
    counts=collections.defaultdict(collections.Counter)
    hits=collections.defaultdict(lambda:collections.defaultdict(lambda:{'paragraphs':0,'docs':0}))
    seen=set(); docs=[]
    # Historical CSVs are absent locally. Count physical v1 lines occurring an exact
    # multiple of 20; this also handles embedded newlines in historical ViSFD rows.
    counts_path=args.tmp/'v1_counts.txt'
    if not counts_path.exists(): raise FileNotFoundError(counts_path)
    old_tokens=0; old_lines=0
    with counts_path.open(encoding='utf-8') as f:
        for line in f:
            m=re.match(r'^\s*(\d+) (.*)$',line.rstrip('\n'))
            if not m: continue
            freq=int(m.group(1))
            if freq>=20 and freq%20==0 and m.group(2).strip():
                old_tokens+=freq*count(m.group(2).strip());old_lines+=freq
    total_v1=json.loads((OUT/'v1_measurement.json').read_text())['bami_tokens']
    ratio=old_tokens/total_v1
    print('v1',total_v1,'domain',old_tokens,'ratio',ratio,'identified_lines',old_lines,flush=True)
    target_domain=old_tokens
    domain_base=[]
    for source, texts in train.items():
        for text in texts:
            chunks=clean_doc(text,'domain',dc,seen,counts,hits)
            for pars,n in chunks: domain_base.append((source,pars,n))
    base_tokens=sum(n for _,_,n in domain_base)
    repeat=min(20,max(1,target_domain//max(1,base_tokens)))
    if base_tokens*20 < target_domain: print('WARNING domain shortfall',target_domain-base_tokens*20,flush=True)
    domain_docs=[('domain',pars,n) for _ in range(repeat) for _,pars,n in domain_base]
    domain_tokens=base_tokens*repeat
    if repeat<20:
        partial=list(domain_base);random.Random(42).shuffle(partial)
        for _,pars,n in partial:
            if domain_tokens+n>target_domain: continue
            domain_docs.append(('domain',pars,n));domain_tokens+=n
    print('domain prepared',base_tokens,repeat,domain_tokens,flush=True)
    # Wikipedia: original text selected by the v2 filtered IDs.
    wiki_meta=[json.loads(x) for x in (ART/'wiki_filtered.jsonl').open()]
    random.Random(42).shuffle(wiki_meta)
    wiki_mm_target=round(sum(r['n_tokens'] for r in wiki_meta)*(total_v1*0.15/60356544))
    wiki_ids=set();wiki_mm=0
    for r in wiki_meta:
        wiki_ids.add(r['id']);wiki_mm+=r['n_tokens']
        if wiki_mm>=wiki_mm_target:break
    wiki_docs=[]
    pf=pq.ParquetFile(RAW/'wikipedia_vi/wiki_vi_20231101_train_0000.parquet')
    for batch in pf.iter_batches(batch_size=2048,columns=['id','text']):
        d=batch.to_pydict()
        for id_,text in zip(d['id'],d['text']):
            if id_ in wiki_ids:
                for pars,n in clean_doc(text,'wiki',dc,seen,counts,hits): wiki_docs.append(('wiki',pars,n))
    wiki_tokens=sum(x[2] for x in wiki_docs)
    print('wiki',len(wiki_docs),wiki_tokens,flush=True)
    raw_sha=verify_raw_sha(args.tmp); print('raw SHA PASS',flush=True)
    web_ids,web_pool_docs,web_mm=select_general()
    ugc_meta=[json.loads(x) for x in (PILOT/'v2/ugc_refined/ugc.docs.jsonl').open()]
    ugc_ids={r['id']:{'source':'ugc','shard':r['shard']} for r in ugc_meta}
    spool=args.tmp/'raw_selected.jsonl'
    raw_count=extract_raw(web_ids,ugc_ids,spool)
    print('raw extracted',raw_count,flush=True)
    # UGC follows v2 extract/refine rules on original raw text. The refined manifest selects documents.
    sys.path.insert(0,str(ROOT/'scripts/mmbert_vi_cpt_pilot/v2'))
    from ugc_extract import extract_paragraphs
    from register_refine import refine_text
    ugc_docs=[];web_docs=[];web_freq=collections.Counter()
    for line in spool.open(encoding='utf-8'):
        r=json.loads(line); source=r['source']
        if source=='ugc': text='\n'.join(refine_text('\n'.join(extract_paragraphs(r['text']))))
        else: text=r['text']
        for pars,n in clean_doc(text,source,dc,seen,counts,hits):
            if source=='web':
                for p in pars:web_freq[norm(p)]+=1
                web_docs.append(('web',pars,n))
            else: ugc_docs.append(('ugc',pars,n))
    # Count web boilerplate before global exact dedup, separately below in manifest.
    # A paragraph kept by exact dedup cannot recur 5 times; assert this invariant.
    if any(n>=5 for n in web_freq.values()): raise RuntimeError('web boilerplate survived')
    ugc_tokens=sum(x[2] for x in ugc_docs)
    remaining=total_v1-domain_tokens-wiki_tokens-ugc_tokens
    random.Random(42).shuffle(web_docs)
    selected=[]; web_tokens=0
    for d in web_docs:
        selected.append(d);web_tokens+=d[2]
        if web_tokens>=remaining: break
    if web_tokens<remaining: raise RuntimeError(f'web shortfall {remaining-web_tokens} tokens')
    docs=domain_docs+wiki_docs+ugc_docs+selected
    random.Random(42).shuffle(docs)
    corpus=OUT/'mlm_mix.txt'
    with corpus.open('w',encoding='utf-8',newline='\n') as f, (OUT/'doc_sources.jsonl').open('w',encoding='utf-8') as sf:
        for source,pars,n in docs:
            f.write('\n'.join(pars)+'\n\n')
            sf.write(json.dumps({'source':source,'tokens_bami':n,'paragraphs':len(pars)})+'\n')
    corpus_sha=sha(corpus)
    actual=sum(n for _,_,n in docs)
    by_source={s:{'tokens_bami':sum(d[2] for d in docs if d[0]==s),'docs':sum(d[0]==s for d in docs)} for s in ('web','wiki','ugc','domain')}
    stats=json.loads((ROOT/'data/corpus_v1/stats.json').read_text())
    emitted_lines={s:sum(len(d[1]) for d in docs if d[0]==s) for s in by_source}
    source_lines={**emitted_lines,'domain':counts['domain']['paragraphs_after_filter']}
    stats.update({'source_line_counts':source_lines,'emitted_line_counts':emitted_lines,'total_lines':sum(emitted_lines.values()),'total_bytes':corpus.stat().st_size,'seed':42,'domain_repeat':repeat,'domain_partial_repeat_docs':len(domain_docs)-repeat*len(domain_base),'domain_max_document_repeat':repeat+int(len(domain_docs)>repeat*len(domain_base)),'token_bami_by_source':{s:v['tokens_bami'] for s,v in by_source.items()},'documents_by_source':{s:v['docs'] for s,v in by_source.items()},'total_tokens_bami':actual,'mlm_mix_sha256':corpus_sha})
    save(OUT/'stats.json',stats)
    report={'unique_texts':dc.unique,'by_source_task_split':{s:dict(v) for s,v in hits.items()},'normalization':'NFC lowercase punctuation removal whitespace syllables','long_rule':'any 13-syllable ngram','short_rule':'normalized exact match','doc_drop_threshold':0.2}
    save(OUT/'decontam_report.json',report)
    source_files=[ROOT/'data/corpus_v1/mlm_mix.txt', ROOT/'data/corpus_v1/stats.json',ART/'general_docs.jsonl',ART/'wiki_filtered.jsonl',PILOT/'pool/pool.jsonl',PILOT/'v2/ugc_refined/ugc.docs.jsonl',ROOT/'models/goiturn-s-gd/tokenizer.json',*map(Path,task_paths),*(RAW/'hplt3_vie'/f'{x}.jsonl.zst' for x in ('9_2','9_1','8_4','7_1','10_1','5_1','6_1')),RAW/'wikipedia_vi/wiki_vi_20231101_train_0000.parquet']
    source_sha={str(p.relative_to(ROOT)):(raw_sha[p.stem.split('.')[0]] if p.parent==RAW/'hplt3_vie' else sha(p)) for p in sorted(set(source_files))}
    manifest={'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_sha256':source_sha,'parameters':{'seed':42,'max_doc_tokens':2048,'wiki_target_fraction':0.15,'wiki_selected_mmbert_tokens':wiki_mm,'ugc_max_fraction':0.10,'domain_max_repeat':20,'host_cap':0.02,'web_manifest_pool_docs':web_pool_docs,'web_candidate_mmbert_tokens':web_mm,'web_selection':'seeded random from pinned general manifest','dedup':'NFC casefold whitespace; global exact paragraph','decontam_doc_threshold':0.2},'v1':{'tokens_bami':total_v1,'domain_tokens_bami':old_tokens,'domain_ratio':ratio,'domain_identification':'physical lines with frequency >=20 and divisible by 20 in sort|uniq -c','domain_identified_physical_lines':old_lines},'sources':by_source,'filter_counts':{s:dict(c) for s,c in counts.items()},'raw_extracted':raw_count,'total_tokens_bami':actual,'mlm_mix_sha256':corpus_sha,'elapsed_seconds':time.time()-t0,'known_limitations':['UGC raw paragraphs selected by v2 cues/refine rules using original text; v2 intermediate paragraph S1 decisions cannot be reproduced from refined manifest alone','A paragraph exceeding 2048 Bami tokens cannot be split at a paragraph boundary and is dropped','Historical v1 CSV inputs are absent; multiple-of-20 frequency can include incidental CC100 boilerplate and miss domain lines with added CC100 occurrences']}
    save(OUT/'manifest.json',manifest)
    # Audit samples are only sampled from actually selected documents.
    sample=random.Random(428).sample(docs,20)
    save(OUT/'audit_20_docs.json',[{'source':s,'excerpt':' '.join(pars)[:400]} for s,pars,_ in sample])
    print(json.dumps({'total':actual,'sources':by_source,'sha256':corpus_sha,'elapsed_s':manifest['elapsed_seconds']}),flush=True)


if __name__=='__main__': main()
