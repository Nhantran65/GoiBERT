#!/usr/bin/env bash
# Launch the three v4 CPT arms, one job per GPU on GPUs 0 1 2 only (GPU 3 untouched), waiting for a free GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-e0-cuda/bin/python}
LOG=artifacts/mmbert_vi_cpt_pilot/v4/cpt/logs; mkdir -p "$LOG"
GPUS=${GPUS:-"0 1 2"}
MICRO=${MICRO:-32}

pilot_gpus() {
  for pid in $(pgrep -f "mmbert_vi_cpt_pilot/v4/cpt_train_v4\.py" || true); do
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

for arm in P_web P_inf C_inf; do
  case "$arm" in P_inf|C_inf) dir="${arm}_NC_research_only";; *) dir="$arm";; esac
  if [[ -f "artifacts/mmbert_vi_cpt_pilot/v4/cpt/$dir/summary.json" ]]; then echo "[cptv4] $arm already finished"; continue; fi
  while true; do
    gpu=$(pick_gpu) && break
    echo "[cptv4] waiting for a free GPU in {$GPUS}"; sleep 60
  done
  echo "[cptv4] $(date -u +%FT%TZ) launching $arm on GPU $gpu (dir $dir)"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/mmbert_vi_cpt_pilot/v4/cpt_train_v4.py --arm "$arm" --device cuda:0 \
      --micro "$MICRO" --resume > "$LOG/${arm}.log" 2>&1 &
  sleep 30
done
echo "[cptv4] launched; waiting"
wait
echo "[cptv4] all arms finished"
