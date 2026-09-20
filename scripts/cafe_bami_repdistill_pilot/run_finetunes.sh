#!/usr/bin/env bash
# Launch the six fine-tunes (M/G x Restaurant/Topic/ViSFD), one job per GPU, choosing GPUs by free VRAM (>= 14 GB) at
# dispatch; waits for a free slot instead of stacking jobs.  Never touches other people's processes.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-e0-cuda/bin/python
LOG=artifacts/cafe_bami_repdistill_pilot/ft; mkdir -p "$LOG"
JOBS=("M absa_restaurant" "M vsfc_topic" "M visfd" "G absa_restaurant" "G vsfc_topic" "G visfd")
pilot_gpus() {   # GPUs occupied by live pilot processes (cpt_train / finetune), read from their CUDA_VISIBLE_DEVICES
  for pid in $(pgrep -f "^\S*python scripts/cafe_bami_repdistill_pilot/(cpt_train|finetune)\.py"); do
    tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
  done | sort -u | tr '\n' ' '
}
pick_gpu() {
  while true; do
    used=" $(pilot_gpus) "
    while IFS=, read -r idx free; do
      idx=$(echo "$idx" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
      if [[ "$used" != *" $idx "* ]] && (( free >= 14000 )); then echo "$idx"; return; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    sleep 30
  done
}
job_running() {   # status.json says running and its pid is alive
  local st="$LOG/$1/$2/status.json"
  [[ -e "$st" ]] || return 1
  local status pid
  read -r status pid < <($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('status',''), d.get('pid',0))" "$st")
  [[ "$status" == "running" ]] && [[ -e /proc/$pid ]]
}
for job in "${JOBS[@]}"; do
  set -- $job; arm=$1; task=$2
  if [[ -e "$LOG/$arm/$task/metadata.json" ]]; then echo "skip $arm/$task (done)"; continue; fi
  if job_running "$arm" "$task"; then echo "skip $arm/$task (running)"; continue; fi
  until [[ -e "artifacts/cafe_bami_repdistill_pilot/cpt/$arm/summary.json" ]]; do sleep 15; done   # wait for this arm's CPT export
  gpu=$(pick_gpu)
  echo "$(date -u +%FT%TZ) launch $arm/$task on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/cafe_bami_repdistill_pilot/finetune.py --arm $arm --task $task --device cuda:0 --micro-batch 32 > "$LOG/${arm}_${task}.log" 2>&1 &
  sleep 45   # let the job allocate before re-reading free memory
done
wait
echo "$(date -u +%FT%TZ) all fine-tunes finished"
