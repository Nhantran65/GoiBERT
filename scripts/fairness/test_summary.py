#!/usr/bin/env python3
"""Seven-row, 15-metric test table from existing and one-time new results."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/fairness/test'
METRICS={'vsfc_sentiment':('accuracy','macro_f1'),'vsfc_topic':('accuracy','macro_f1'),
         'visfd':('detection_micro_f1','pair_micro_f1'),
         'absa_restaurant':('detection_micro_f1','pair_micro_f1'),
         'absa_hotel':('detection_micro_f1','pair_micro_f1'),
         'vispam':('accuracy','macro_f1'),'vinli':('accuracy','macro_f1'),
         'phoner':('f1',)}
MODELS=('bami_e14','bami_e7_ce','bami_e7_kd','s_prime_ce','s_prime_kd','m_ce','m_kd')


def read(path):return json.loads(path.read_text())


def main():
    locked=read(OUT/'locked_manifest.json')
    if locked['checkpoint_count']!=96 or not locked['validation_passed']:
        raise RuntimeError('Manifest must pass all 96 checkpoint checks')
    sources={
        'new':read(OUT/'test_results.json')['results'],
        'bami_e14':read(ROOT/'artifacts/final_test/test_results.json')['results'],
        'family':read(ROOT/'artifacts/family_sm/test/test_results.json')['results'],
    }
    selected={}
    for label in MODELS:
        if label in ('bami_e14','s_prime_kd','m_kd'):
            key='bami_e14' if label=='bami_e14' else 'family'
            original='bami' if label=='bami_e14' else 's_prime' if label=='s_prime_kd' else 'm'
            selected[label]=[r for r in sources[key] if r['model']==original]
        else:selected[label]=[r for r in sources['new'] if r['model']==label]
        if len(selected[label])!=24:raise AssertionError((label,len(selected[label])))
    table={}
    for label,entries in selected.items():
        by={(e['task'],int(e['seed'])):e for e in entries}
        row={}
        for task,metrics in METRICS.items():
            for metric in metrics:
                values=[100*float(by[(task,seed)]['test_metrics'][metric]) for seed in (42,43,44)]
                row[f'{task}/{metric}']={'mean':statistics.mean(values),'range':[min(values),max(values)],
                                         'seeds':values}
        if len(row)!=15:raise AssertionError((label,len(row)))
        row['mean_15']=statistics.mean(v['mean'] for v in row.values())
        row['mean_13_without_vinli']=statistics.mean(v['mean'] for k,v in row.items()
                                                    if '/' in k and not k.startswith('vinli/'))
        params_by_task={task:int(by[(task,42)].get('parameters') or read(ROOT/by[(task,42)]['checkpoint'].replace('best_checkpoint.pt','resolved_config.json'))['model_spec']['total_parameters']) for task in METRICS}
        row['parameters']={'min':min(params_by_task.values()),'max':max(params_by_task.values()),
                           'by_task':params_by_task}
        table[label]=row
    result={'models':MODELS,'metric_order':[f'{task}/{metric}' for task,metrics in METRICS.items() for metric in metrics],
            'table':table,'provenance':{'new':'artifacts/fairness/test/test_results.json',
                                        'bami_e14':'artifacts/final_test/test_results.json',
                                        's_prime_kd_m_kd':'artifacts/family_sm/test/test_results.json'}}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{'mean_15':v['mean_15'],'mean_13_without_vinli':v['mean_13_without_vinli'],
                         'parameters':v['parameters']} for k,v in table.items()},indent=2))


if __name__=='__main__':main()
