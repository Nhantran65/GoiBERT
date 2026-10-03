#!/usr/bin/env python3
"""Compare fairness parity runs against frozen reference curves."""
import json
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/fairness'
CASES={
 'bami_e14_topic': (ART/'parity/bami_e14_topic',ROOT/'artifacts/e14/trials/vsfc_topic/lr_1e-05/seed_42'),
 'bami_e14_vispam': (ART/'parity/bami_e14_vispam',ROOT/'artifacts/bench8/runs/vispam/bami/lr_2e-05/seed_42'),
 'student_ce': (ART/'parity/student_ce',ROOT/'artifacts/cafe_oof_kd/student/ce/vsfc_sentiment/seed_42'),
}


def main():
    result={}
    curve_rows=[]
    for name,(actual,reference) in CASES.items():
        ma=json.loads((actual/'metadata.json').read_text())
        cfg=json.loads((actual/'resolved_config.json').read_text())
        refcfg=json.loads((reference/'resolved_config.json').read_text())
        a=json.loads((actual/'epoch_curves.json').read_text())['epochs']
        b=json.loads((reference/'epoch_curves.json').read_text())['epochs']
        assert len(a)==len(b)==30
        get=lambda row:float(row['selection_f1'])
        f1_delta=[abs(get(x)-get(y)) for x,y in zip(a,b)]
        loss_delta=[abs(float(x['train_loss'])-float(y['train_loss'])) for x,y in zip(a,b)]
        curve_rows.extend({'case':name,'epoch':i+1,'train_loss_new':x['train_loss'],
                           'train_loss_reference':y['train_loss'],
                           'validation_f1_new':get(x),'validation_f1_reference':get(y)}
                          for i,(x,y) in enumerate(zip(a,b)))
        best_delta=100*abs(max(map(get,a))-max(map(get,b)))
        checks={'first_five_f1_abs_max_le_0_05':max(f1_delta[:5])<=.05,
                'best_f1_point_delta_le_1':best_delta<=1.0,
                'micro_batch_matches_reference':int(cfg.get('micro_batch_size',cfg.get('micro_batch',cfg.get('batch_size'))))==int(refcfg.get('micro_batch_size',refcfg.get('micro_batch',refcfg.get('batch_size')))),
                'postrun_pass':ma['status']=='completed' and all(ma['postrun_checks'].values())}
        result[name]={'passed':all(checks.values()),'checks':checks,
                      'first_five_f1':[[get(x),get(y)] for x,y in zip(a,b)][:5],
                      'f1_abs_differences':f1_delta,
                      'train_loss_abs_differences':loss_delta,
                      'train_loss_first_five_max_abs':max(loss_delta[:5]),
                      'best_f1_point_delta':best_delta,
                      'best_epoch_actual':ma['best_epoch'],
                      'best_epoch_reference':json.loads((reference/'metadata.json').read_text())['best_epoch']}
    target=ART/'parity_results.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (ART/'parity_curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(curve_rows[0]))
        writer.writeheader();writer.writerows(curve_rows)
    print(json.dumps({k:v['checks'] for k,v in result.items()},indent=2))
    if not all(x['passed'] for x in result.values()):
        raise SystemExit('Parity FAIL; launcher must not start')


if __name__=='__main__':main()
