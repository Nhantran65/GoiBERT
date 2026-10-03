#!/usr/bin/env bash
# Stage 1 §5 fine-tune screen launcher (copy of scripts/qwen3_encoder_probe/run_grid.sh).
# One job per GPU, GPUs 0 1 2, free VRAM >= 20 GB, never stacks on a GPU that already runs a stage-1 job,
# never touches other people's processes or GPU 3.  Waves: O = arm O Restaurant LR 5e-6 (can run while MNTP
# trains); 1 = P_vi Restaurant x {5e-6,1e-5,2e-5}; 2 = P_vi ViSFD + Topic at its best Restaurant LR.
set -eo pipefail
cd "$(dirname "$0")/../.."
PY="${FT_PYTHON:-.venv-qwen-probe/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOG=artifacts/bidirlm_vi_mntp/logs
mkdir -p "$LOG"
GPUS="${FT_GPUS:-0 1 2}"
MIN_FREE_MB="${FT_MIN_FREE_MB:-20000}"
WAVES="${FT_WAVES:-O,1,2}"
DRY_RUN="${FT_DRY_RUN:-0}"
EXPORT_DIR=artifacts/bidirlm_vi_mntp/mntp/hf_bidirlm

declare -A SLOT_PID=()

probe_gpus() {
  for pid in $(pgrep -f "^\S*python scripts/bidirlm_vi_mntp/" || true); do
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

cell_dir() { printf 'artifacts/bidirlm_vi_mntp/ft/%s/%s/lr_%s/seed_%s' "$1" "$2" "$(printf '%.0e' "$3" | sed 's/+//')" "$4"; }
cell_done() { [[ -e "$(cell_dir "$1" "$2" "$3" "$4")/metadata.json" ]]; }

launch_job() {  # gpu arm task lr seed
  local gpu="$1" arm="$2" task="$3" lr="$4" seed="${5:-42}"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$(date -u +%FT%TZ) DRY launch $arm/$task/lr$lr/seed$seed on GPU $gpu"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/bidirlm_vi_mntp/finetune.py \
    --arm "$arm" --task "$task" --lr "$lr" --seed "$seed" --device cuda:0 --micro-batch 16 \
    > "$LOG/ft_${arm}_${task}_lr${lr}_seed${seed}.log" 2>&1 &
  SLOT_PID[$gpu]=$!
  echo "$(date -u +%FT%TZ) launch $arm/$task/lr$lr/seed$seed on GPU $gpu (pid ${SLOT_PID[$gpu]}, micro 16)"
}

run_queue() {
  local jobs=("$@") i=0 job arm task lr seed gpu launched
  while (( i < ${#jobs[@]} )) || (( ${#SLOT_PID[@]} > 0 )); do
    launched=0
    while (( i < ${#jobs[@]} )) && gpu=$(pick_gpu); do
      IFS=: read -r arm task lr seed <<< "${jobs[$i]}"
      if cell_done "$arm" "$task" "$lr" "$seed"; then
        echo "$(date -u +%FT%TZ) skip $arm/$task/lr$lr/seed$seed (done)"
        ((++i)); continue
      fi
      launch_job "$gpu" "$arm" "$task" "$lr" "$seed"
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

if wave_enabled O; then
  echo "$(date -u +%FT%TZ) wave O: arm O Restaurant LR 5e-6 (completes the O grid {5e-6,1e-5,2e-5})"
  run_queue "O:absa_restaurant:5e-6:42"
fi

if wave_enabled 1 || wave_enabled 2; then
  if [[ ! -d "$EXPORT_DIR" ]]; then
    echo "$(date -u +%FT%TZ) P_vi export not found at $EXPORT_DIR - run mntp_train.py + export_bidirlm.py first"
    exit 1
  fi
fi

if wave_enabled 1; then
  echo "$(date -u +%FT%TZ) wave 1: P_vi Restaurant x {5e-6,1e-5,2e-5}"
  jobs=()
  for lr in 5e-6 1e-5 2e-5; do jobs+=("P_vi:absa_restaurant:$lr:42"); done
  run_queue "${jobs[@]}"
fi

"$PY" scripts/bidirlm_vi_mntp/summarize.py >/dev/null || true

if wave_enabled 2; then
  best_lr=$("$PY" scripts/bidirlm_vi_mntp/summarize.py --best-lr-per-arm | awk '$1=="P_vi"{print $2}')
  if [[ -z "$best_lr" ]]; then
    echo "$(date -u +%FT%TZ) wave 2 skipped: no completed P_vi Restaurant cell"
  else
    echo "$(date -u +%FT%TZ) wave 2: P_vi ViSFD + Topic at best Restaurant LR $best_lr"
    jobs=("P_vi:visfd:$best_lr:42" "P_vi:vsfc_topic:$best_lr:42")
    run_queue "${jobs[@]}"
  fi
fi

"$PY" scripts/bidirlm_vi_mntp/summarize.py --table || true
echo "$(date -u +%FT%TZ) stage-1 screen finished"
