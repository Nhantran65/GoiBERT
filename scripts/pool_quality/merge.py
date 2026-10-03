"""Join independently measured samples into the requested summary artifact."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/pool_quality'
TMP=Path('/tmp/goibert_pool_quality_20260928')

def read(p): return json.loads(p.read_text())

def run():
    u=read(OUT/'summary_UR.json'); c=read(OUT/'summary_C.json')
    r_path=OUT/'summary_R.json'
    if r_path.exists():
        r=read(r_path)
        assert u['decontam_unique']==r['decontam_unique']
        u['samples']['R']=r['samples']['R']
    assert u['decontam_unique']==c['decontam_unique']
    result={'samples':{**u['samples'],**c['samples']},
            'selection':read(TMP/'selection.json'),
            'extraction':read(TMP/'extract_stats.json'),
            'v1_blocks':read(TMP/'v1_stats.json'),
            'decontam_unique':u['decontam_unique']}
    unique_u=set(); pii={'U':0,'R':0}
    with (OUT/'sample_raw.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['sample']=='U': unique_u.add(r['id'])
            pii[r['sample']]+=int(r['meta'].get('pii_n',0)>0)
    result['selection']['u_unique_raw_ids']=len(unique_u)
    for name in ('U','R'):
        result['samples'][name]['metadata']['pii_n']['positive_n']=pii[name]
    assert set(result['samples'])=={'U','R','C'}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': run()
