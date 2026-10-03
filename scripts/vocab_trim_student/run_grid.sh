#!/usr/bin/env bash
# Vocabulary-trim student screen launcher (docs/VOCAB_TRIM_STUDENT_PLAN.md §2.2).  One job per GPU, GPUs 0 1 2,
# free VRAM >= 20 GB, never stacks on a GPU that already runs a job of this task, never touches other people's
# processes or GPU 3.  Waves: 1 = Restaurant x 3 LR x {m12,m6,b8,b12} (12 runs, per-model LR grid);
# 2 = ViSFD+Topic x 4 models at the best Restaurant LR (8 runs); 3 = Restaurant seed 43 for m12 and b8 (2 runs).
set -eo pipefail
cd "$(dirname "$0")/../.."
PY="${VT_PYTHON:-.venv-e0-cuda/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOG=artifacts/vocab_trim_student/logs
mkdir -p "$LOG"
GPUS="${VT_GPUS:-0 1 2}"
MIN_FREE_MB="${VT_MIN_FREE_MB:-20000}"
MICRO="${VT_MICRO:-32}"
WAVES="${VT_WAVES:-1,2,3}"
DRY_RUN="${VT_DRY_RUN:-0}"
MODELS="m12 m6 b8 b12"

declare -A SLOT_PID=()

probe_gpus() {
  for pid in $(pgrep -f "^\S*python scripts/vocab_trim_student/" || true); do
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

cell_dir() { printf 'artifacts/vocab_trim_student/%s/%s/lr_%s/seed_%s' "$1" "$2" "$(printf '%.0e' "$3" | sed 's/+//')" "$4"; }
cell_done() { [[ -e "$(cell_dir "$1" "$2" "$3" "$4")/metadata.json" ]]; }
keep_layers_for() { case "$1" in b8) echo 8;; *) echo 0;; esac; }
lrs_for() { case "$1" in m12|m6) echo "2e-5 5e-5 1e-4";; *) echo "1e-5 2e-5 5e-5";; esac; }

launch_job() {  # model task lr seed
  local model="$1" task="$2" lr="$3" seed="$4" gpu keep
  keep=$(keep_layers_for "$model")
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$(date -u +%FT%TZ) DRY launch $model/$task/lr$lr/seed$seed (keep-layers $keep, micro $MICRO)"
    return 0
  fi
  gpu=$(pick_gpu) || return 1
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/vocab_trim_student/finetune.py \
    --model "$model" --task "$task" --lr "$lr" --seed "$seed" --keep-layers "$keep" --device cuda:0 --micro-batch "$MICRO" \
    > "$LOG/${model}_${task}_lr${lr}_seed${seed}.log" 2>&1 &
  SLOT_PID[$gpu]=$!
  echo "$(date -u +%FT%TZ) launch $model/$task/lr$lr/seed$seed on GPU $gpu (pid ${SLOT_PID[$gpu]}, keep-layers $keep)"
}

run_queue() {
  local jobs=("$@") i=0 model task lr seed launched gpu
  while (( i < ${#jobs[@]} )) || (( ${#SLOT_PID[@]} > 0 )); do
    launched=0
    while (( i < ${#jobs[@]} )); do
      IFS=: read -r model task lr seed <<< "${jobs[$i]}"
      if cell_done "$model" "$task" "$lr" "$seed"; then
        echo "$(date -u +%FT%TZ) skip $model/$task/lr$lr/seed$seed (done)"
        ((++i)); continue
      fi
      if launch_job "$model" "$task" "$lr" "$seed"; then ((++i)); launched=1; else break; fi
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
  echo "$(date -u +%FT%TZ) wave 1: Restaurant x 3 LR x {$MODELS} (per-model LR grid)"
  jobs=()
  for model in $MODELS; do
    for lr in $(lrs_for "$model"); do jobs+=("$model:absa_restaurant:$lr:42"); done
  done
  run_queue "${jobs[@]}"
fi

if wave_enabled 2; then
  jobs=()
  while read -r model lr; do
    [[ -n "$model" && -n "$lr" ]] || continue
    jobs+=("$model:visfd:$lr:42" "$model:vsfc_topic:$lr:42")
  done < <("$PY" scripts/vocab_trim_student/summarize.py --best-lr-per-model || true)
  if (( ${#jobs[@]} )); then
    echo "$(date -u +%FT%TZ) wave 2: ViSFD + Topic for ${#jobs[@]} model-runs"
    run_queue "${jobs[@]}"
  else
    echo "$(date -u +%FT%TZ) wave 2 skipped: no completed Restaurant grid"
  fi
fi

if wave_enabled 3; then
  jobs=()
  while read -r model lr; do
    [[ "$model" == "m12" || "$model" == "b8" ]] || continue
    jobs+=("$model:absa_restaurant:$lr:43")
  done < <("$PY" scripts/vocab_trim_student/summarize.py --best-lr-per-model || true)
  if (( ${#jobs[@]} )); then
    echo "$(date -u +%FT%TZ) wave 3: Restaurant seed 43 for ${#jobs[@]} model(s)"
    run_queue "${jobs[@]}"
  else
    echo "$(date -u +%FT%TZ) wave 3 skipped"
  fi
fi

"$PY" scripts/vocab_trim_student/summarize.py || true
echo "$(date -u +%FT%TZ) vocab-trim student grid finished"
