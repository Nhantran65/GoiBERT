"""Transparent token capacity and source-mix projection from measured samples."""
from __future__ import annotations
import collections, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/pool_quality'
sys.path.insert(0,str(ROOT))
from scripts.corpus_v2 import build as v2

def run():
    path=OUT/'summary.json'; result=json.loads(path.read_text()); samples=result['samples']
    train,locks,_=v2.task_texts(); dc=v2.Decontam(locks)
    seen=set(); counts=collections.defaultdict(collections.Counter)
    hits=collections.defaultdict(lambda:collections.defaultdict(lambda:{'paragraphs':0,'docs':0}))
    domain_base=0
    for texts in train.values():
        for text in texts:
            domain_base+=sum(n for _,n in v2.clean_doc(text,'domain',dc,seen,counts,hits))
    wiki=list(json.loads(line) for line in (v2.ART/'wiki_filtered.jsonl').open())
    wiki_mm_total=sum(r['n_tokens'] for r in wiki)
    manifest=json.loads((ROOT/'data/corpus_v2/manifest.json').read_text())
    wiki_mm_used=manifest['parameters']['wiki_selected_mmbert_tokens']
    wiki_bami_used=manifest['sources']['wiki']['tokens_bami']
    wiki_bami_total_est=wiki_mm_total*wiki_bami_used/wiki_mm_used
    ugc=list(json.loads(line) for line in (v2.PILOT/'v2/ugc_refined/ugc.docs.jsonl').open())
    ugc_mm_total=sum(r['n_tokens'] for r in ugc)
    pool_mm=1_134_178_900
    r=samples['R']; raw_bami_est=pool_mm*r['bami_per_mmbert']
    cleaned_web_est=raw_bami_est*r['sequential_retained_tokens']/r['tokens_bami']
    nonweb_cap=domain_base*20+wiki_bami_total_est+manifest['sources']['ugc']['tokens_bami']
    target=1_000_000_000
    desired_web=target-nonweb_cap
    possible=min(desired_web,cleaned_web_est)
    projected_total=nonweb_cap+possible
    result['projection']={
      'target_bami_tokens':target,'pool_mmbert_tokens':pool_mm,
      'r_bami_per_mmbert':r['bami_per_mmbert'],
      'web_raw_bami_est':raw_bami_est,
      'r_decontam_drop_fraction':r['sequential_decontam_drop_tokens']/r['tokens_bami'],
      'r_exact_drop_fraction':r['sequential_exact_drop_tokens']/r['tokens_bami'],
      'r_boilerplate_exact_drop_fraction':r['sequential_boilerplate_exact_drop_tokens']/r['tokens_bami'],
      'r_nonboilerplate_exact_drop_fraction':(r['sequential_exact_drop_tokens']-r['sequential_boilerplate_exact_drop_tokens'])/r['tokens_bami'],
      'pool_decontam_drop_bami_est':raw_bami_est*r['sequential_decontam_drop_tokens']/r['tokens_bami'],
      'pool_exact_drop_bami_est':raw_bami_est*r['sequential_exact_drop_tokens']/r['tokens_bami'],
      'pool_boilerplate_exact_drop_bami_est':raw_bami_est*r['sequential_boilerplate_exact_drop_tokens']/r['tokens_bami'],
      'r_retention_after_doc_decontam_paragraph_decontam_exact':r['sequential_retained_tokens']/r['tokens_bami'],
      'web_cleaned_bami_est':cleaned_web_est,
      'web_needed_for_1b':desired_web,'web_capacity_shortfall':max(0,desired_web-cleaned_web_est),
      'web_capacity_headroom':max(0,cleaned_web_est-desired_web),
      'additional_raw_pool_loss_fraction_to_shortfall':max(0,cleaned_web_est-desired_web)/raw_bami_est,
      'wiki_mmbert_total':wiki_mm_total,'wiki_mmbert_used':wiki_mm_used,
      'wiki_mmbert_unused':wiki_mm_total-wiki_mm_used,'wiki_unused_docs':len(wiki)-manifest['filter_counts']['wiki']['docs_input'],
      'wiki_bami_total_est':wiki_bami_total_est,'ugc_mmbert_total':ugc_mm_total,
      'ugc_unused_docs':0,'ugc_bami_cap':manifest['sources']['ugc']['tokens_bami'],
      'domain_base_bami':domain_base,'domain_max20_bami':domain_base*20,
      'v2_large_estimated_total':projected_total,
      'v2_large_tokens_by_source':{'web':possible,'wiki':wiki_bami_total_est,
        'ugc':manifest['sources']['ugc']['tokens_bami'],'domain':domain_base*20},
      'v2_large_share_by_source':{'web':possible/projected_total,'wiki':wiki_bami_total_est/projected_total,
        'ugc':manifest['sources']['ugc']['tokens_bami']/projected_total,'domain':domain_base*20/projected_total},
      'v2_tokens_by_source':{k:v['tokens_bami'] for k,v in manifest['sources'].items()},
      'v1_tokens_by_source':{'web_cc100':307699789,'wiki':0,'ugc':0,'domain':20934020},
      'v1_total_bami':328633809,'v2_total_bami':328634392}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['projection'],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': run()
