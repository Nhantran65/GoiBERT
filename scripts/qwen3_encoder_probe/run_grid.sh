#!/usr/bin/env bash
# Qwen3 encoder probe launcher (plan §7): one job per GPU, GPUs 0 1 2, free VRAM >= 20 GB, never reuses a
# GPU that already runs a probe job, never touches other people's processes or GPU 3.
# Wave 1: Restaurant x {5e-5,2e-5,1e-5} x {C,B0,B2}; Wave 2: ViSFD+Topic at each arm's best Restaurant LR;
# Wave 3: Restaurant seed 43 for the best bidirectional arm at its best LR.
# Re-runnable: completed cells are skipped.
set -eo pipefail
cd "$(dirname "$0")/../.."
PY="${PROBE_PYTHON:-.venv-qwen-probe/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOG=artifacts/qwen3_encoder_probe/logs
mkdir -p "$LOG"
GPUS="${PROBE_GPUS:-0 1 2}"
MIN_FREE_MB="${PROBE_MIN_FREE_MB:-20000}"
WAVES="${PROBE_WAVES:-1,2,3}"
DRY_RUN="${PROBE_DRY_RUN:-0}"

declare -A SLOT_PID=()

probe_gpus() {
  for pid in $(pgrep -f "^\S*python scripts/qwen3_encoder_probe/" || true); do
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

cell_dir() { echo "artifacts/qwen3_encoder_probe/$1/$2/lr_$(printf '%.0e' "$3" | sed 's/+//')/seed_$4"; }
cell_done() { [[ -e "$(cell_dir "$1" "$2" "$3" "$4")/metadata.json" ]]; }
micro_for() { case "$1" in B2) echo "${PROBE_MICRO_B2:-16}";; *) echo "${PROBE_MICRO_C:-32}";; esac; }

launch_job() {  # gpu arm task lr seed micro
  local gpu="$1" arm="$2" task="$3" lr="$4" seed="$5" micro="$6"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$(date -u +%FT%TZ) DRY launch $arm/$task/lr$lr/seed$seed on GPU $gpu (micro $micro)"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/qwen3_encoder_probe/finetune.py \
    --arm "$arm" --task "$task" --lr "$lr" --seed "$seed" --device cuda:0 --micro-batch "$micro" \
    > "$LOG/${arm}_${task}_lr${lr}_seed${seed}.log" 2>&1 &
  SLOT_PID[$gpu]=$!
  echo "$(date -u +%FT%TZ) launch $arm/$task/lr$lr/seed$seed on GPU $gpu (pid ${SLOT_PID[$gpu]}, micro $micro)"
}

run_queue() {
  local jobs=("$@") i=0 arm task lr seed micro gpu launched
  while (( i < ${#jobs[@]} )) || (( ${#SLOT_PID[@]} > 0 )); do
    launched=0
    while (( i < ${#jobs[@]} )) && gpu=$(pick_gpu); do
      IFS=: read -r arm task lr seed micro <<< "${jobs[$i]}"
      if cell_done "$arm" "$task" "$lr" "$seed"; then
        echo "$(date -u +%FT%TZ) skip $arm/$task/lr$lr/seed$seed (done)"
        ((++i)); continue
      fi
      launch_job "$gpu" "$arm" "$task" "$lr" "$seed" "$micro"
      ((++i)); launched=1
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
  echo "$(date -u +%FT%TZ) wave 1: Restaurant x {5e-5,2e-5,1e-5} x {C,B0,B2}"
  jobs=()
  for arm in C B0 B2; do
    for lr in 5e-5 2e-5 1e-5; do jobs+=("$arm:absa_restaurant:$lr:42:$(micro_for "$arm")"); done
  done
  run_queue "${jobs[@]}"
fi

"$PY" scripts/qwen3_encoder_probe/summarize.py >/dev/null || true

if wave_enabled 2; then
  mapfile -t best_lines < <("$PY" scripts/qwen3_encoder_probe/summarize.py --best-lr-per-arm)
  echo "$(date -u +%FT%TZ) wave 2: ViSFD+Topic at best Restaurant LR per arm (${best_lines[*]})"
  jobs=()
  for line in "${best_lines[@]}"; do
    read -r arm lr <<< "$line"
    for task in visfd vsfc_topic; do jobs+=("$arm:$task:$lr:42:$(micro_for "$arm")"); done
  done
  if (( ${#jobs[@]} > 0 )); then run_queue "${jobs[@]}"; fi
fi

"$PY" scripts/qwen3_encoder_probe/summarize.py >/dev/null || true

if wave_enabled 3; then
  best_arm_line=$("$PY" scripts/qwen3_encoder_probe/summarize.py --best-arm || true)
  if [[ -n "$best_arm_line" ]]; then
    read -r arm lr <<< "$best_arm_line"
    echo "$(date -u +%FT%TZ) wave 3: Restaurant seed 43 for $arm at lr $lr (noise floor)"
    run_queue "$arm:absa_restaurant:$lr:43:$(micro_for "$arm")"
  else
    echo "$(date -u +%FT%TZ) wave 3 skipped: no completed bidirectional arm"
  fi
fi

"$PY" scripts/qwen3_encoder_probe/summarize.py --table || true
echo "$(date -u +%FT%TZ) probe grid finished"
