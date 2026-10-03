"""CPU-only, read-only measurements of v2 web candidates and v1 CC100 blocks.

Writes only artifacts/pool_quality and a dedicated /tmp directory.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/pool_quality'
TMP = Path('/tmp/goibert_pool_quality_20260928')
sys.path.insert(0, str(ROOT))
from scripts.corpus_v2 import build as v2

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def lines(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def prepare():
    OUT.mkdir(exist_ok=True); TMP.mkdir(exist_ok=True)
    rows = list(lines(v2.ART/'general_docs.jsonl'))
    shuffled = list(rows); random.Random(42).shuffle(shuffled)
    selected = []; mm = 0
    for row in shuffled:
        selected.append(row); mm += row['n_tokens']
        if mm >= 540_000_000: break
    chosen = {r['pool_index'] for r in selected}
    remain = [r for r in rows if r['pool_index'] not in chosen]
    r_sample = random.Random(1234).sample(remain, 20_000)
    # Output doc_sources lines align with blank-delimited mlm_mix documents.
    web_count = 0; web_tokens = 0
    for row in lines(ROOT/'data/corpus_v2/doc_sources.jsonl'):
        if row['source'] == 'web':
            web_count += 1; web_tokens += row['tokens_bami']
    assert web_count == 284991 and web_tokens == 232889660, (web_count, web_tokens)
    sampled_ord = set(random.Random(1234).sample(range(web_count), 20_000))
    first_hash = {}
    corpus = (ROOT/'data/corpus_v2/mlm_mix.txt').open(encoding='utf-8')
    source = (ROOT/'data/corpus_v2/doc_sources.jsonl').open(encoding='utf-8')
    current = []; web_ord = 0
    for line in corpus:
        if line != '\n':
            current.append(line.rstrip('\n')); continue
        row = json.loads(source.readline())
        if row['source'] == 'web':
            if web_ord in sampled_ord:
                key = v2.norm(current[0])
                first_hash.setdefault(key, []).append(web_ord)
            web_ord += 1
        current = []
    assert web_ord == web_count and not source.readline() and not current
    source.close(); corpus.close()
    wanted_index = chosen | {r['pool_index'] for r in r_sample}
    meta = {}
    for i, line in enumerate((v2.PILOT/'pool/pool.jsonl').open(encoding='utf-8')):
        if i in wanted_index:
            row = json.loads(line); meta[i] = row
    assert len(meta) == len(wanted_index)
    for row in rows:
        if row['pool_index'] in meta:
            assert int(meta[row['pool_index']]['kh64']) == int(row['kh64'])
            meta[row['pool_index']].update({'n_tokens_mmbert': row['n_tokens'], 'host': row['host']})
    selected_ids = {meta[i]['id']: i for i in chosen}
    r_ids = {meta[r['pool_index']]['id']: r['pool_index'] for r in r_sample}
    dump(TMP/'selection.json', {'pool_rows': len(rows), 'candidate_docs': len(selected),
        'candidate_mmbert_tokens': mm, 'v2_web_chunks': web_count, 'v2_web_bami_tokens': web_tokens,
        'remainder_docs': len(remain), 'u_sample_chunks': len(sampled_ord), 'r_sample_docs': len(r_sample)})
    # Keep large transient maps on disk only until extraction has finished.
    with (TMP/'selected_ids.jsonl').open('w') as f:
        for id_, i in selected_ids.items(): f.write(json.dumps([id_, i])+'\n')
    shutil.copyfile(TMP/'selected_ids.jsonl',OUT/'v2_selected_input_ids.jsonl')
    with (TMP/'r_ids.jsonl').open('w') as f:
        for id_, i in r_ids.items(): f.write(json.dumps([id_, i])+'\n')
    with (TMP/'first_paragraphs.jsonl').open('w') as f:
        for key, ords in first_hash.items(): f.write(json.dumps([key, ords],ensure_ascii=False)+'\n')
    with (TMP/'meta.jsonl').open('w') as f:
        for i,row in meta.items(): f.write(json.dumps([i,row],ensure_ascii=False)+'\n')
    print('prepared', len(selected), len(r_sample), len(first_hash), flush=True)

def extract():
    selected_ids = dict(lines_array(TMP/'selected_ids.jsonl'))
    r_ids = dict(lines_array(TMP/'r_ids.jsonl'))
    first = dict(lines_array(TMP/'first_paragraphs.jsonl'))
    meta = dict(lines_array(TMP/'meta.jsonl'))
    found_u = {}; found_r = {}; collisions = collections.Counter()
    with (OUT/'sample_raw.jsonl').open('w', encoding='utf-8') as out:
        for shard in ('9_2','9_1','8_4','7_1','10_1','5_1','6_1'):
            p = v2.RAW/'hplt3_vie'/f'{shard}.jsonl.zst'
            proc = subprocess.Popen(['zstd','-dc',str(p)], stdout=subprocess.PIPE, bufsize=8<<20)
            n = 0
            for raw in proc.stdout:
                m = v2.ID_RE.search(raw)
                if not m: continue
                id_ = m.group(1).decode()
                ri = r_ids.get(id_)
                si = selected_ids.get(id_)
                if ri is None and si is None: continue
                obj = json.loads(raw); assert obj['id'] == id_
                if ri is not None:
                    found_r[id_] = True
                    out.write(json.dumps({'sample':'R','id':id_,'pool_index':ri,'meta':meta[ri], 'text':obj['text']},ensure_ascii=False)+'\n')
                if si is not None:
                    hits = set()
                    for paragraph in v2.split_paragraphs(obj['text']):
                        hits.update(first.get(v2.norm(paragraph), []))
                    for ord_ in hits:
                        collisions[ord_] += 1
                        if ord_ not in found_u:
                            found_u[ord_] = id_
                            out.write(json.dumps({'sample':'U','chunk_ordinal':ord_, 'id':id_, 'pool_index':si,'meta':meta[si], 'text':obj['text']},ensure_ascii=False)+'\n')
                n += 1
            proc.stdout.close()
            if proc.wait(): raise RuntimeError(f'zstd failed: {p}')
            print('extracted shard', shard, n, len(found_u),len(found_r),flush=True)
            if len(found_u)==20_000 and len(found_r)==20_000:
                break
    dump(TMP/'extract_stats.json', {'u_found':len(found_u),'r_found':len(found_r),
         'u_collisions':sum(v>1 for v in collisions.values()),'u_missing':20_000-len(found_u),
         'r_missing':20_000-len(found_r)})
    if len(found_u)!=20_000 or len(found_r)!=20_000: raise RuntimeError('sample raw IDs unresolved')

def lines_array(path):
    with path.open(encoding='utf-8') as f:
        for line in f: yield json.loads(line)

def make_v1_counts():
    path=TMP/'v1_counts.txt'
    if path.exists(): return path
    ordered=TMP/'v1_sorted.txt'
    env={**os.environ,'LC_ALL':'C'}
    with ordered.open('wb') as out:
        subprocess.run(['sort','-T',str(TMP),str(ROOT/'data/corpus_v1/mlm_mix.txt')],stdout=out,env=env,check=True)
    with path.open('wb') as out:
        subprocess.run(['uniq','-c',str(ordered)],stdout=out,env=env,check=True)
    ordered.unlink()
    return path

def v1_blocks():
    counts=make_v1_counts()
    domain=set()
    with counts.open(encoding='utf-8') as f:
        for line in f:
            m=re.match(r'^\s*(\d+) (.*)$',line.rstrip('\n'))
            if m and int(m.group(1))>=20 and int(m.group(1))%20==0 and m.group(2).strip():
                domain.add(m.group(2).strip())
    # Reservoir sample 20k nonoverlapping 30-line CC100 blocks, exact uniform over blocks.
    rng=random.Random(1234); reservoir=[]; block=[]; n=0; excluded=0
    for line in (ROOT/'data/corpus_v1/mlm_mix.txt').open(encoding='utf-8'):
        s=line.strip()
        if not s: continue
        if s in domain: excluded+=1; continue
        block.append(s)
        if len(block)==30:
            if n<20_000: reservoir.append('\n'.join(block))
            else:
                j=rng.randrange(n+1)
                if j<20_000: reservoir[j]='\n'.join(block)
            n+=1; block=[]
    assert n>=20_000
    with (OUT/'sample_v1_cc100.jsonl').open('w',encoding='utf-8') as f:
        for i,text in enumerate(reservoir):
            f.write(json.dumps({'sample':'C','block_index':i,'text':text},ensure_ascii=False)+'\n')
    dump(TMP/'v1_stats.json',{'domain_distinct_lines':len(domain),'domain_physical_lines':excluded,
        'cc100_complete_30_line_blocks':n,'unused_tail_lines':len(block)})
    print('v1 blocks',n,len(domain),excluded,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['prepare','extract','v1'])
    a=p.parse_args()
    {'prepare':prepare,'extract':extract,'v1':v1_blocks}[a.phase]()
