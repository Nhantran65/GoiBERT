"""Independent final reader of the emitted corpus. READY is written only after all gates pass."""
from __future__ import annotations
import collections
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/corpus_v2'
sys.path.insert(0,str(ROOT/'scripts/corpus_v2'))
from build import task_texts, norm, syllables, tok, sha, save


def main():
    ready=OUT/'READY'
    if ready.exists(): ready.unlink()
    stats=json.loads((OUT/'stats.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    _,locks,_=task_texts()
    keys={f'{k[0]}/{k[1]}':{norm(x) for x in v if norm(x)} for k,v in locks.items()}
    found={k:set() for k in keys}
    web_seen=set();web_dup=web_n=0
    n_docs=n_lines=n_tokens=0
    source_lines=(OUT/'doc_sources.jsonl').open()
    cur=[]
    def consume(pars):
        nonlocal n_docs,web_n,web_dup
        if not pars:return
        n_docs+=1
        rec=json.loads(source_lines.readline())
        if rec['paragraphs']!=len(pars): raise RuntimeError('doc_sources paragraph count mismatch')
        for p in pars:
            k=norm(p)
            for label,ref in keys.items():
                if k in ref:found[label].add(k)
            if rec['source']=='web':
                web_n+=1
                if k in web_seen:web_dup+=1
                web_seen.add(k)
    with (OUT/'mlm_mix.txt').open(encoding='utf-8') as f:
        for raw in f:
            p=raw.strip()
            if not p:
                consume(cur);cur=[];continue
            cur.append(p);n_lines+=1;n_tokens+=len(tok().encode(p,add_special_tokens=False))
        consume(cur)
    if source_lines.readline():raise RuntimeError('doc_sources longer than corpus')
    source_lines.close()
    sys.path.insert(0,str(ROOT/'reference/ai-foundary/experiments/goibert-family-v1/src'))
    from mlm_pretrain import iter_packed_sequences
    packed=iter_packed_sequences(OUT/'mlm_mix.txt',tok())
    examples=[];packed_ok=True
    for _ in range(100):
        seq=next(packed);decoded=tok().decode(seq,skip_special_tokens=True)
        readable=bool(re.search(r'[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]',decoded,re.I)) and len(decoded.strip())>=100
        if len(seq)!=256 or not readable:packed_ok=False
        if len(examples)<5:examples.append({'length':len(seq),'decoded_excerpt':decoded[:300],'readable':readable})
    v1=manifest['v1']['tokens_bami']
    digest=sha(OUT/'mlm_mix.txt')
    checks={'exact_matches_zero':sum(len(x) for x in found.values())==0,
            'token_total_within_5pct':abs(n_tokens-v1)<=0.05*v1,
            'packed_first_100_full_readable_vi':packed_ok,
            'web_duplicate_ratio_below_1pct':(web_dup/max(1,web_n))<0.01,
            'sha_matches_manifest':digest==manifest['mlm_mix_sha256'],
            'stats_token_count_matches_reread':n_tokens==stats['total_tokens_bami'],
            'doc_count_matches_manifest':n_docs==sum(v['docs'] for v in manifest['sources'].values()),
            'ugc_le_10pct':manifest['sources']['ugc']['tokens_bami']/n_tokens<=0.10,
            'domain_repeat_le_20':stats['domain_repeat']<=20}
    result={'checks':checks,'passed':all(checks.values()),'sha256':digest,'tokens_bami_reread':n_tokens,'v1_tokens_bami':v1,'documents_reread':n_docs,'paragraphs_reread':n_lines,'web_duplicate_paragraphs':web_dup,'web_paragraphs':web_n,'web_duplicate_ratio':web_dup/max(1,web_n),'exact_match_table':{k:{'unique_texts':len(v),'matches':len(found[k])} for k,v in keys.items()},'packed_examples':examples,'checked_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save(OUT/'final_check.json',result)
    if result['passed']:
        ready.write_text(f'{digest}\n{result["checked_at"]}\n')
        print('PASS',digest,flush=True)
    else:
        print('FAIL',json.dumps(checks),flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
