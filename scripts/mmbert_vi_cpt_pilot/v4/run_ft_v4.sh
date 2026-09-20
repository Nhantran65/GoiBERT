#!/usr/bin/env bash
# 13 v4 fine-tunes (E14 recipe, validation-only) scheduled one job per GPU on GPUs 0 1 2 (GPU 3 untouched).
# Queue: Restaurant x {P_web,P_inf,C_inf} x seed {42,43}; ViSFD x 3 arms x 42; Topic x 3 arms x 42; O x Restaurant x 43.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-e0-cuda/bin/python}
LOG=artifacts/mmbert_vi_cpt_pilot/v4/ft/logs; mkdir -p "$LOG"
GPUS=${GPUS:-"0 1 2"}
MICRO=${MICRO:-32}

dir_of() { case "$1" in O|P_web) echo "$1";; *) echo "$1_NC_research_only";; esac; }
queue() {
  for arm in P_web P_inf C_inf; do for seed in 42 43; do echo "$arm absa_restaurant $seed"; done; done
  for arm in P_web P_inf C_inf; do echo "$arm visfd 42"; done
  for arm in P_web P_inf C_inf; do echo "$arm vsfc_topic 42"; done
  echo "O absa_restaurant 43"
}
pilot_gpus() {
  for pid in $(pgrep -f "mmbert_vi_cpt_pilot/v4/finetune_v4\.py" || true); do
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
  d="artifacts/mmbert_vi_cpt_pilot/v4/ft/$(dir_of "$arm")/$task/seed_$seed"
  if [[ -f "$d/metadata.json" ]]; then echo "[ftv4] skip $arm/$task/seed$seed (done)"; continue; fi
  while true; do gpu=$(pick_gpu) && break; echo "[ftv4] waiting for a free GPU"; sleep 60; done
  echo "[ftv4] $(date -u +%FT%TZ) launch $arm/$task/seed$seed on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/mmbert_vi_cpt_pilot/v4/finetune_v4.py --arm "$arm" --task "$task" --seed "$seed" \
      --device cuda:0 --micro-batch "$MICRO" > "$LOG/${arm}_${task}_seed${seed}.log" 2>&1 &
  sleep 20
done < <(queue)
echo "[ftv4] launched; waiting"
wait
echo "[ftv4] done"
