#!/usr/bin/env bash
# 15 §6 fine-tunes (E14 recipe, validation only) scheduled one job per GPU on GPUs 0 1 2 (GPU 3 untouched).
# Queue: Restaurant x {P_web,P_mix,C_mix} x seed {42,43}, ViSFD x 3 arms x seed 42, Topic x 3 arms x seed 42,
#        O (mmBERT-base) x {Restaurant,ViSFD,Topic} x seed 43  (O seed 42 is reused from the baseline, never re-trained).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-e0-cuda/bin/python}
LOG=artifacts/mmbert_vi_cpt_pilot/ft/logs; mkdir -p "$LOG"
GPUS=${GPUS:-"0 1 2"}
MICRO=${MICRO:-32}

run_dir() { echo "artifacts/mmbert_vi_cpt_pilot/ft/$1/$2/seed_$3"; }
queue() {
  for arm in P_web P_mix C_mix; do for seed in 42 43; do echo "$arm absa_restaurant $seed"; done; done
  for arm in P_web P_mix C_mix; do echo "$arm visfd 42"; done
  for arm in P_web P_mix C_mix; do echo "$arm vsfc_topic 42"; done
  for task in absa_restaurant visfd vsfc_topic; do echo "O $task 43"; done
}

pilot_gpus() {
  for pid in $(pgrep -f "scripts/mmbert_vi_cpt_pilot/finetune\.py" || true); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
  done | sort -u | tr '\n' ' '
}
pick_gpu() {
  local busy; busy=" $(pilot_gpus) "
  while IFS=, read -r idx freemib; do
    idx=$(echo "$idx" | tr -d ' '); freemib=$(echo "$freemib" | tr -d ' ')
    [[ " $GPUS " == *" $idx "* ]] || continue
    if [[ "$busy" != *" $idx "* ]] && (( freemib >= 20000 )); then echo "$idx"; return 0; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

while read -r arm task seed; do
  d=$(run_dir "$arm" "$task" "$seed")
  if [[ -f "$d/metadata.json" ]]; then echo "[ft] skip $arm/$task/seed$seed (done)"; continue; fi
  while true; do
    gpu=$(pick_gpu) && break
    echo "[ft] waiting for a free GPU in {$GPUS}"; sleep 60
  done
  echo "[ft] $(date -u +%FT%TZ) launch $arm/$task/seed$seed on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/mmbert_vi_cpt_pilot/finetune.py --arm "$arm" --task "$task" --seed "$seed" \
      --device cuda:0 --micro-batch "$MICRO" > "$LOG/${arm}_${task}_seed${seed}.log" 2>&1 &
  sleep 20
done < <(queue)
echo "[ft] all runs launched; waiting"
wait
echo "[ft] done"
