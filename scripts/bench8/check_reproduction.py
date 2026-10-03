#!/usr/bin/env python3
"""Apply the locked BENCH8 E14/E7 reproduction tolerances."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCES={'bami':ROOT/'artifacts/e14/trials/vsfc_topic/lr_1e-05/seed_42','student':ROOT/'artifacts/cafe_oof_kd/student/ce/vsfc_topic/seed_42'}
def rd(p):return json.loads(p.read_text())
def main():
 details={};passed=True
 for model,source in SOURCES.items():
  dest=ROOT/'artifacts/bench8/reproduction'/model
  old=rd(source/'epoch_curves.json')['epochs'];new=rd(dest/'epoch_curves.json')['epochs'];status=rd(dest/'status.json')
  if len(new)!=30 or status['status']!='completed':raise RuntimeError(f'{model} incomplete')
  gaps=[100*abs(old[i]['selection_f1']-new[i]['selection_f1']) for i in range(5)]
  old_best=rd(source/'best_validation_metrics.json')['selection_f1'];new_best=rd(dest/'best_validation_metrics.json')['selection_f1'];best_gap=100*abs(new_best-old_best)
  ok=max(gaps)<=.05 and best_gap<=1.0;passed &=ok
  details[model]={'first_five_f1_point_differences':gaps,'best_f1_old':old_best,'best_f1_new':new_best,'best_f1_point_difference':best_gap,'pass':ok}
 result={'passed':bool(passed),'models':details,'thresholds_f1_points':{'each_epoch_1_to_5':.05,'best':1.0}}
 out=ROOT/'artifacts/bench8/reproduction_check.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
