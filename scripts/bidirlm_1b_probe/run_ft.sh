#!/usr/bin/env bash
# Validation-only B1B screen: Restaurant LR grid, then ViSFD/Topic at selected LR.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-qwen-probe/bin/python
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
PHASE2=artifacts/bidirlm_1b_probe/phase2
LOG="$PHASE2/logs"
mkdir -p "$LOG"

decision=$("$PY" -c 'import json; print(json.load(open("artifacts/bidirlm_1b_probe/phase1/gate.json"))["decision"])')
[[ "$decision" == PROCEED ]] || { echo "phase1 gate is $decision"; exit 2; }
passed=$("$PY" -c 'import json; print(json.load(open("artifacts/bidirlm_1b_probe/phase2/preflight.json"))["passed"])')
[[ "$passed" == True ]] || { echo "preflight did not PASS"; exit 2; }
micro=$("$PY" -c 'import json; print(json.load(open("artifacts/bidirlm_1b_probe/phase2/preflight.json"))["selected_micro_batch"])')

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, '$1+0 <= 2 && $2+0 >= 20000 {if ($2+0 > best) {best=$2+0; idx=$1+0}} END {if (best>0) print idx}'
}

run_one() {
  local task="$1" lr="$2" gpu log dir
  gpu=$(pick_gpu)
  [[ -n "$gpu" ]] || { echo "no allowed GPU has >=20000 MiB free"; return 2; }
  dir=$("$PY" -c 'import sys; sys.path.insert(0,"scripts/bidirlm_1b_probe"); import common; print(common.arm_dir("B1B",sys.argv[1],float(sys.argv[2]),42))' "$task" "$lr")
  [[ ! -e "$dir/best_checkpoint.pt" && ! -e "$dir/metadata.json" ]] || { echo "refusing overwrite $dir"; return 2; }
  log="$LOG/ft_B1B_${task}_lr${lr}_seed42.log"
  echo "$(date -u +%FT%TZ) launch B1B/$task/lr$lr seed42 on GPU $gpu, micro $micro"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/bidirlm_1b_probe/finetune.py \
    --arm B1B --task "$task" --lr "$lr" --seed 42 --device cuda:0 --micro-batch "$micro" \
    > "$log" 2>&1 < /dev/null &
  local pid=$!
  wait "$pid"
  "$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["status"]=="completed",m["status"]; assert all(m["postrun_checks"][k] for k in ("reload_matches_best","prediction_file_matches_best","reload_predictions_identical","ids_stable_and_complete"))' "$dir/metadata.json"
  echo "$(date -u +%FT%TZ) completed B1B/$task/lr$lr; checks PASS"
}

budget_guard() {
  "$PY" - "$1" <<'PY'
import json, sys
from pathlib import Path
phase=Path('artifacts/bidirlm_1b_probe/phase2')
pre=json.loads((phase/'preflight.json').read_text())
first=json.loads((phase/'ft/B1B/absa_restaurant/lr_5e-06/seed_42/metadata.json').read_text())
first_s=first['elapsed_seconds']
expected_ratio=(3*7028+7786+11426)/7028
projected=(pre['elapsed_seconds']+first_s*expected_ratio*1.3)/3600
actual=pre['elapsed_seconds']/3600
for p in (phase/'ft/B1B').glob('*/lr_*/seed_42/metadata.json'):
    actual+=json.loads(p.read_text())['elapsed_seconds']/3600
next_task=sys.argv[1]
next_ratio={'absa_restaurant':1.3,'visfd':1.6,'vsfc_topic':2.2}.get(next_task,0)
next_estimated=first_s*next_ratio/3600
stop=projected>12 or actual+next_estimated>12
payload={'protocol':'bidirlm-1b-probe-budget-v1','first_restaurant_seconds':first_s,
         'projected_total_gpu_hours_conservative':projected,'actual_gpu_hours_conservative':actual,
         'next_task':next_task,'next_estimated_gpu_hours':next_estimated,'budget_gpu_hours':12,
         'decision':'STOP' if stop else 'GO'}
(phase/'budget_estimate.json').write_text(json.dumps(payload,indent=1)+'\n')
print(payload['decision'])
PY
}

run_one absa_restaurant 5e-6
if [[ $(budget_guard absa_restaurant) == STOP ]]; then
  "$PY" scripts/bidirlm_1b_probe/summarize.py
  echo "budget estimate exceeds 12 GPU-hours after first Restaurant; stopped"
  exit 3
fi
for lr in 1e-5 2e-5; do
  [[ $(budget_guard absa_restaurant) == GO ]] || { "$PY" scripts/bidirlm_1b_probe/summarize.py; exit 3; }
  run_one absa_restaurant "$lr"
done
best_lr=$("$PY" scripts/bidirlm_1b_probe/summarize.py --best-lr)
[[ -n "$best_lr" ]] || { echo "no Restaurant LR selected"; exit 4; }
for task in visfd vsfc_topic; do
  [[ $(budget_guard "$task") == GO ]] || { "$PY" scripts/bidirlm_1b_probe/summarize.py; exit 3; }
  run_one "$task" "$best_lr"
done
"$PY" scripts/bidirlm_1b_probe/summarize.py
