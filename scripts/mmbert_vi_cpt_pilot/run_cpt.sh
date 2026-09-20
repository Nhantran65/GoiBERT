#!/usr/bin/env bash
# Three CPT arms, one job per GPU on GPUs 0 1 2 only (GPU 3 is never touched).  Waits for a free GPU (>= 20 GB free and
# no other process of this pilot on it) before launching, skips arms that already finished, and never stops other jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-e0-cuda/bin/python}
LOG=artifacts/mmbert_vi_cpt_pilot/cpt/logs; mkdir -p "$LOG"
GPUS=${GPUS:-"0 1 2"}
MICRO=${MICRO:-32}

for arm in P_web P_mix C_mix; do
  out="artifacts/mmbert_vi_cpt_pilot/cpt/$arm/summary.json"
  if [[ -f "$out" ]]; then echo "[cpt] $arm already finished"; continue; fi
  # wait for a GPU of our own set that is free
  while true; do
    busy=$(for pid in $(pgrep -f "scripts/mmbert_vi_cpt_pilot/cpt_train\.py" || true); do
             tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
           done | sort -u | tr '\n' ' ')
    pick=""
    while IFS=, read -r idx freemib; do
      idx=$(echo "$idx" | tr -d ' '); freemib=$(echo "$freemib" | tr -d ' ')
      [[ " $GPUS " == *" $idx "* ]] || continue
      if [[ " $busy " != *" $idx "* ]] && (( freemib >= 20000 )); then pick=$idx; break; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    [[ -n "$pick" ]] && break
    echo "[cpt] waiting for a free GPU in {$GPUS} (busy: $busy)"; sleep 60
  done
  echo "[cpt] $(date -u +%FT%TZ) launching $arm on GPU $pick"
  CUDA_VISIBLE_DEVICES=$pick nohup $PY scripts/mmbert_vi_cpt_pilot/cpt_train.py --arm "$arm" --device cuda:0 \
      --micro "$MICRO" --attn sdpa --resume > "$LOG/${arm}.log" 2>&1 &
  sleep 30
done
echo "[cpt] launched; waiting for all arms"
wait
echo "[cpt] all arms finished"
