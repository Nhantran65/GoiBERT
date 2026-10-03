#!/usr/bin/env python3
"""Compare dry-run Cafe configurations without loading any model."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.label_ceiling_check.seed_train import dry_config, original, config_diff
ART=ROOT/'artifacts/label_ceiling_check'
TASKS=('vsfc_sentiment','vsfc_topic','visfd','absa_restaurant','absa_hotel')
CORE=('learning_rate','epochs','effective_batch_size','micro_batch_size','gradient_accumulation_steps','max_length','precision','attention_implementation','weight_decay','warmup_fraction','gradient_clip_norm','loss','include_test')

def loaded(p):return json.loads(p.read_text())
def explained(differences):
    reasons={'seed':'Only training seed changes',
             'argv':'Dry-run invocation/output path replaces historical command',
             'dry_run_config':'Dry-run marker; not a training hyperparameter',
             'dry_run_source':'Historical fingerprint provenance for dry-run',
             'dry_run_note':'Dry-run provenance note',
             'source_hash':'Current frozen runner source hash; historical code revision differs',
             'protocol':'CDBK pilot versus SEA-LION benchmark wrapper name',
             'selection':'Same validation selection_f1; wording differs',
             'head':'CLS means first token; head dimensions/dropouts are identical',
             'pooling':'CDBK omitted explicit first-token pooling metadata',
             'model.benchmark_adapter':'SEA-LION adapter metadata only',
             'tokenizer.benchmark_adapter':'SEA-LION adapter metadata only'}
    return [{**d,'explanation':reasons.get(d['field'],'Historical runtime, timestamp, or metadata difference; inspect before training')} for d in differences]
def core(c):
    return {k:c.get(k) for k in CORE}|{'model':{k:c['model'].get(k) for k in ('backbone','hf_commit','hidden_size','head_hidden_size','head_dropout','num_labels','num_aspects','aspect_class_counts','weight_file_sha256')},'tokenizer':{k:c['tokenizer'].get(k) for k in ('source','hf_commit','files','vocab_size')}}

def main():
    report={'dry_runs':{},'restaurant':{}}
    for task in TASKS:
        base=loaded(original(task)/'resolved_config.json')
        report['dry_runs'][task]={}
        for seed in (43,44):
            c=dry_config(task,seed,ART/'cafe_seeds'/task/f'seed_{seed}')
            p=ART/'cafe_seeds'/task/f'seed_{seed}'
            # Prepared configs live outside the run directory so the launcher can
            # refuse all incomplete training directories safely.
            q=ART/'dry_run_configs'/task/f'seed_{seed}.json';q.parent.mkdir(parents=True,exist_ok=True)
            q.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
            report['dry_runs'][task][str(seed)]={'path':str(q.relative_to(ROOT)),'core_equal':core(c)==core(base),'differences':explained(config_diff(base,c))}
    t42=loaded(original('absa_restaurant')/'resolved_config.json')
    s43p=ROOT/'artifacts/cafebert_aspect_pair_confirmation/seed43/T_absa_restaurant_seed43'
    s43=loaded(s43p/'resolved_config.json')
    target=dry_config('absa_restaurant',43,ART/'cafe_seeds/absa_restaurant/seed_43')
    report['restaurant']['sealion_vs_cdbk_T42']={'core_equal':core(t42)==core(target),'differences':explained(config_diff(t42,target)),'explanation':'Same hyperparameters, backbone/head/tokenizer and validation evaluator. Protocol, head wording (CLS = first token), selection wording, benchmark_adapter, source hash and runtime metadata differ.'}
    report['restaurant']['existing_seed43_vs_T42']={'core_equal':core(s43)==core(t42),'differences':explained(config_diff(t42,s43)),'reuse_allowed':core(s43)==core(t42) and loaded(s43p/'status.json').get('status')=='completed' and all(loaded(s43p/'metadata.json').get('postrun_checks',{}).values()),'source':str(s43p.relative_to(ROOT))}
    (ART/'recipe_diff.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not all(x['core_equal'] for y in report['dry_runs'].values() for x in y.values()):raise RuntimeError('recipe blocker')
    print('recipe core PASS; existing Restaurant seed43 reusable=' + str(report['restaurant']['existing_seed43_vs_T42']['reuse_allowed']))
if __name__=='__main__':main()
