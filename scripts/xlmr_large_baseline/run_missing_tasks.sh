#!/usr/bin/env bash
# Five-task extension: seed 42, 3 LR per missing task, GPUs 2 and 3 only.
# Do not launch on a GPU with a pre-existing compute process. Re-runnable.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv-e0-cuda/bin/python
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=artifacts/xlmr_large_baseline/logs_five_task_extension
mkdir -p "$LOG"

gpu_uuid() {
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' -v target="$1" '$1 == target {print $2}'
}

gpu_ready() {
  local gpu="$1" uuid free
  uuid=$(gpu_uuid "$gpu")
  [[ -n "$uuid" ]] || return 1
  free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', *' -v target="$gpu" '$1 == target {print $2}')
  [[ -n "$free" ]] && (( free >= 20000 )) || return 1
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -Fxq "$uuid"
}

cell_done() {
  local task="$1" lr="$2" dir
  dir="artifacts/xlmr_large_baseline/xlmr_large/$task/lr_${lr}/seed_42"
  [[ -e "$dir/metadata.json" ]] && grep -Fq '"status": "completed"' "$dir/metadata.json"
}

launch() {
  local task="$1" lr="$2" gpu="$3" log
  LAST_PID=""
  log="$LOG/${task}_lr_${lr}_seed_42.log"
  if cell_done "$task" "$lr"; then
    echo "SKIP completed $task lr=$lr"
    return 0
  fi
  if ! gpu_ready "$gpu"; then
    echo "GPU $gpu has another compute process or less than 20 GB free; refusing to launch $task lr=$lr" >&2
    return 1
  fi
  echo "START $task lr=$lr GPU=$gpu log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/xlmr_large_baseline/finetune.py \
    --model xlmr_large --task "$task" --lr "$lr" --seed 42 \
    --device cuda:0 --micro-batch 32 > "$log" 2>&1 &
  LAST_PID="$!"
}

for lr in 1e-05 2e-05 5e-05; do
  launch vsfc_sentiment "$lr" 2
  p2="$LAST_PID"
  launch absa_hotel "$lr" 3
  p3="$LAST_PID"
  if [[ -n "$p2" ]]; then wait "$p2"; fi
  if [[ -n "$p3" ]]; then wait "$p3"; fi
  for task in vsfc_sentiment absa_hotel; do
    if ! cell_done "$task" "$lr"; then
      echo "FAILED $task lr=$lr; see $LOG/${task}_lr_${lr}_seed_42.log" >&2
      exit 1
    fi
    echo "DONE $task lr=$lr"
  done
done
