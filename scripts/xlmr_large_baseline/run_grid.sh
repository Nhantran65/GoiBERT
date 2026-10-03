#!/usr/bin/env bash
# XLM-R-large baseline launcher (copy of scripts/base_encoder_baseline/run_grid.sh; artifact paths, registry and
# the wave plan of docs/XLMR_LARGE_BASELINE_PLAN.md §4 changed).  One job per GPU, GPUs 0 1 2, free VRAM >= 20 GB,
# never stacks on a GPU that already runs a job of this task, never touches other people's processes or GPU 3.
# Waves: 1 = Restaurant x {5e-5,2e-5,1e-5} seed 42; 2 = ViSFD+Topic at the best Restaurant LR; 3 = ViSFD+Topic at
# 2e-5 (only when the best LR is not 2e-5); 4 = Restaurant seed 43 at the best LR; 5 = Restaurant 2e-5 seed 43
# (only when the best LR is not 2e-5).  Re-runnable: completed cells are skipped.
set -eo pipefail
cd "$(dirname "$0")/../.."
PY="${XL_PYTHON:-.venv-e0-cuda/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOG=artifacts/xlmr_large_baseline/logs
mkdir -p "$LOG"
GPUS="${XL_GPUS:-0 1 2}"
MIN_FREE_MB="${XL_MIN_FREE_MB:-20000}"
MICRO="${XL_MICRO:-32}"
WAVES="${XL_WAVES:-1,2,3,4,5}"
DRY_RUN="${XL_DRY_RUN:-0}"
MODEL=xlmr_large

declare -A SLOT_PID=()

probe_gpus() {
  for pid in $(pgrep -f "^\S*python scripts/xlmr_large_baseline/" || true); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p'
  done | sort -u | tr '\n' ' '
}

pick_gpu() {
  local used busy idx free
  used=" $(probe_gpus) "
  busy=" $(printf '%s\n' ${!SLOT_PID[@]} 2>/dev/null | tr '\n' ' ') "
  while IFS=, read -r idx free; do
    idx="${idx// /}"; free="${free// /}"
    [[ " $GPUS " == *" $idx "* ]] || continue
    [[ "$used" != *" $idx "* ]] || continue
    [[ "$busy" != *" $idx "* ]] || continue
    if (( free >= MIN_FREE_MB )); then echo "$idx"; return 0; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

cell_dir() { printf 'artifacts/xlmr_large_baseline/%s/%s/lr_%s/seed_%s' "$MODEL" "$1" "$(printf '%.0e' "$2" | sed 's/+//')" "$3"; }
cell_done() { [[ -e "$(cell_dir "$1" "$2" "$3")/metadata.json" ]]; }

launch_job() {  # task lr seed
  local task="$1" lr="$2" seed="$3" gpu
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$(date -u +%FT%TZ) DRY launch $MODEL/$task/lr$lr/seed$seed (micro $MICRO)"
    return 0
  fi
  gpu=$(pick_gpu) || return 1
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/xlmr_large_baseline/finetune.py \
    --model "$MODEL" --task "$task" --lr "$lr" --seed "$seed" --device cuda:0 --micro-batch "$MICRO" \
    > "$LOG/${MODEL}_${task}_lr${lr}_seed${seed}.log" 2>&1 &
  SLOT_PID[$gpu]=$!
  echo "$(date -u +%FT%TZ) launch $MODEL/$task/lr$lr/seed$seed on GPU $gpu (pid ${SLOT_PID[$gpu]}, micro $MICRO)"
}

run_queue() {
  local jobs=("$@") i=0 task lr seed launched gpu
  while (( i < ${#jobs[@]} )) || (( ${#SLOT_PID[@]} > 0 )); do
    launched=0
    while (( i < ${#jobs[@]} )); do
      IFS=: read -r task lr seed <<< "${jobs[$i]}"
      if cell_done "$task" "$lr" "$seed"; then
        echo "$(date -u +%FT%TZ) skip $task/lr$lr/seed$seed (done)"
        ((++i)); continue
      fi
      if launch_job "$task" "$lr" "$seed"; then ((++i)); launched=1; else break; fi
    done
    if (( ${#SLOT_PID[@]} > 0 )); then
      wait -n || true
      for gpu in "${!SLOT_PID[@]}"; do
        if ! kill -0 "${SLOT_PID[$gpu]}" 2>/dev/null; then
          wait "${SLOT_PID[$gpu]}" 2>/dev/null || true
          unset "SLOT_PID[$gpu]"
        fi
      done
    elif (( launched == 1 )); then
      continue
    else
      echo "$(date -u +%FT%TZ) waiting for a free GPU (need >= ${MIN_FREE_MB} MB on: $GPUS)"
      sleep 20
    fi
  done
}

wave_enabled() { [[ ",$WAVES," == *",$1,"* ]]; }

if wave_enabled 1; then
  echo "$(date -u +%FT%TZ) wave 1: Restaurant x {5e-5,2e-5,1e-5} seed 42"
  jobs=()
  for lr in 5e-5 2e-5 1e-5; do jobs+=("absa_restaurant:$lr:42"); done
  run_queue "${jobs[@]}"
fi

best_lr=$("$PY" scripts/xlmr_large_baseline/summarize.py --best-lr 2>/dev/null || true)
echo "$(date -u +%FT%TZ) best Restaurant LR = ${best_lr:-<none>}"

if wave_enabled 2; then
  if [[ -z "$best_lr" ]]; then
    echo "$(date -u +%FT%TZ) wave 2 skipped: no completed Restaurant cell"
  else
    echo "$(date -u +%FT%TZ) wave 2: ViSFD + Topic at LR $best_lr"
    run_queue "visfd:$best_lr:42" "vsfc_topic:$best_lr:42"
  fi
fi

if wave_enabled 3; then
  if [[ -n "$best_lr" && "$best_lr" != "2e-05" ]]; then
    echo "$(date -u +%FT%TZ) wave 3: ViSFD + Topic at 2e-5 (matched-LR comparison with CafeBERT)"
    run_queue "visfd:2e-5:42" "vsfc_topic:2e-5:42"
  else
    echo "$(date -u +%FT%TZ) wave 3 skipped (best LR is 2e-5 or unknown)"
  fi
fi

if wave_enabled 4; then
  if [[ -n "$best_lr" ]]; then
    echo "$(date -u +%FT%TZ) wave 4: Restaurant seed 43 at LR $best_lr"
    run_queue "absa_restaurant:$best_lr:43"
  fi
fi

if wave_enabled 5; then
  if [[ -n "$best_lr" && "$best_lr" != "2e-05" ]]; then
    echo "$(date -u +%FT%TZ) wave 5: Restaurant 2e-5 seed 43 (matched-LR, 2 seeds)"
    run_queue "absa_restaurant:2e-5:43"
  else
    echo "$(date -u +%FT%TZ) wave 5 skipped (best LR is 2e-5 or unknown)"
  fi
fi

"$PY" scripts/xlmr_large_baseline/summarize.py || true
echo "$(date -u +%FT%TZ) xlmr-large baseline grid finished"
