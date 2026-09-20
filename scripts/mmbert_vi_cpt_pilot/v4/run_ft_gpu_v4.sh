#!/usr/bin/env bash
# Run a queue of fine-tune cells sequentially on one GPU (CUDA_VISIBLE_DEVICES must be set by the caller).
# Usage: CUDA_VISIBLE_DEVICES=1 run_ft_gpu_v4.sh arm:task:seed [arm:task:seed ...]
set -euo pipefail
REPO=${REPO_ROOT:-/home/kiettqa/GoiBERT}
cd "$REPO"
PY=${PY:-.venv-e0-cuda/bin/python}
LOG=artifacts/mmbert_vi_cpt_pilot/v4/ft/logs; mkdir -p "$LOG"
MICRO=${MICRO:-32}
dir_of() { case "$1" in O|P_web) echo "$1";; *) echo "$1_NC_research_only";; esac; }
for cell in "$@"; do
  IFS=: read -r arm task seed <<< "$cell"
  d="artifacts/mmbert_vi_cpt_pilot/v4/ft/$(dir_of "$arm")/$task/seed_$seed"
  if [[ -f "$d/metadata.json" ]]; then echo "[ftv4] skip $cell (done)"; continue; fi
  echo "[ftv4] $(date -u +%FT%TZ) start $cell on GPU ${CUDA_VISIBLE_DEVICES:-?}"
  $PY scripts/mmbert_vi_cpt_pilot/v4/finetune_v4.py --arm "$arm" --task "$task" --seed "$seed" --device cuda:0 \
      --micro-batch "$MICRO" 2>&1 | tee "$LOG/$(dir_of "$arm")_${task}_seed${seed}.log" | tail -3
done
echo "[ftv4] queue finished on GPU ${CUDA_VISIBLE_DEVICES:-?}"
