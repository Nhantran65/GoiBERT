#!/usr/bin/env bash
# 30 grid cells (2 models x 5 tasks x 3 LR), one job per GPU on GPUs 0 1 2 only, GPU chosen by free VRAM (>= 14 GB) and
# by no live baseline job on it.  Never touches other people's processes.  Re-runnable: skips done/running cells.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-e0-cuda/bin/python
LOG=artifacts/base_encoder_baseline/logs; mkdir -p "$LOG"
GPUS="0 1 2"
MODELS="mmbert_base xlmr_base"; TASKS="absa_restaurant vsfc_topic visfd absa_hotel vsfc_sentiment"; LRS="5e-5 2e-5 1e-5"
pilot_gpus() {
  for pid in $(pgrep -f "^\S*python scripts/base_encoder_baseline/finetune\.py"); do
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
  done | sort -u | tr '\n' ' '
}
pick_gpu() {
  while true; do
    used=" $(pilot_gpus) "
    while IFS=, read -r idx free; do
      idx=$(echo "$idx" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
      [[ " $GPUS " == *" $idx "* ]] || continue
      if [[ "$used" != *" $idx "* ]] && (( free >= 14000 )); then echo "$idx"; return; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    sleep 30
  done
}
cell_dir() { echo "artifacts/base_encoder_baseline/$1/$2/lr_$(printf '%.0e' "$3" | sed 's/+//')"; }
job_running() {
  local st="$1/status.json"; [[ -e "$st" ]] || return 1
  read -r status pid < <($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status',''), d.get('pid',0))" "$st")
  [[ "$status" == "running" ]] && [[ -e /proc/$pid ]]
}
for model in $MODELS; do for task in $TASKS; do for lr in $LRS; do
  d=$(cell_dir "$model" "$task" "$lr")
  if [[ -e "$d/metadata.json" ]]; then echo "skip $model/$task/$lr (done)"; continue; fi
  if job_running "$d"; then echo "skip $model/$task/$lr (running)"; continue; fi
  gpu=$(pick_gpu)
  echo "$(date -u +%FT%TZ) launch $model/$task/lr$lr on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/base_encoder_baseline/finetune.py --model "$model" --task "$task" --lr "$lr" --device cuda:0 --micro-batch 32 > "$LOG/${model}_${task}_lr${lr}.log" 2>&1 &
  sleep 60
done; done; done
wait
echo "$(date -u +%FT%TZ) grid finished"
