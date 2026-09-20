#!/usr/bin/env bash
# E22 launcher: a 4-slot GPU queue (cuda:0..cuda:3) for the 12 E22 trials --
# E19 hotel/visfd seeds 43/44, d1_composite hotel seeds 43/44 + visfd seeds
# 42-44, d1_uniform absa_restaurant seeds 42-44. See docs/E20_PLAN.md,
# docs/E21_MULTISEED_D1.md, and the run_e19.py / run_e20.py module
# docstrings for the underlying frozen-loop recipe this only dispatches --
# scripts/run_e19.py and scripts/run_e20.py are never edited by this script.
#
# Ordered longest-first (visfd jobs first, then hotel, then restaurant; the
# composite arm -- ~33% more train rows -- before the plain e19 baseline
# within a task) so the 4-slot queue front-loads the biggest jobs. Each of
# the 4 slots is bound to one GPU for the whole run: as soon as a slot's job
# finishes, the next not-yet-completed job in the queue is dispatched on that
# same GPU (a plain polling loop -- no dependency on an interactive tty or
# bash job-control notifications), so this is safe to run detached:
#
#   nohup setsid /home/kiettqa/GoiBERT/scripts/e22_launch.sh \
#     > /home/kiettqa/GoiBERT/artifacts/e22/logs/e22_launch.out 2>&1 &
#
# Safe to re-run: a job whose output dir already has a status.json with
# status "completed" is skipped (not re-launched, not overwritten). A job
# whose output dir exists but is NOT completed (a stale/partial dir from an
# earlier crashed attempt) is still dispatched; run_e19.py's / run_e20.py's
# own run_worker() refuses to write into an existing dir (raises
# FileExistsError, so that job is logged as FAILED) rather than silently
# resuming or clobbering it -- remove or rename that dir first if you intend
# to redo it. At the end, runs scripts/eval_e22.py once (tolerant of any
# still-missing trial) to (re)write docs/E22_RESULT.md.
set -uo pipefail

ROOT="/home/kiettqa/GoiBERT"
PY="$ROOT/.venv-e0-cuda/bin/python"
LOG_DIR="$ROOT/artifacts/e22/logs"
mkdir -p "$LOG_DIR"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

LAUNCH_LOG="$LOG_DIR/e22_launch.out"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

ts() { date -u +%FT%TZ; }
echo "[$(ts)] E22 launcher starting (pid $$)"

GPUS=(cuda:0 cuda:1 cuda:2 cuda:3)

# runner;arm;task;seed;outdir;logname -- 12 jobs, longest-first.
JOBS=(
  "e20;d1_composite;visfd;42;$ROOT/artifacts/e20/d1_composite/visfd/seed_42;d1_composite__visfd__seed42"
  "e20;d1_composite;visfd;43;$ROOT/artifacts/e20/d1_composite/visfd/seed_43;d1_composite__visfd__seed43"
  "e20;d1_composite;visfd;44;$ROOT/artifacts/e20/d1_composite/visfd/seed_44;d1_composite__visfd__seed44"
  "e19;;visfd;43;$ROOT/artifacts/e19/visfd/seed_43;e19__visfd__seed43"
  "e19;;visfd;44;$ROOT/artifacts/e19/visfd/seed_44;e19__visfd__seed44"
  "e20;d1_composite;absa_hotel;43;$ROOT/artifacts/e20/d1_composite/absa_hotel/seed_43;d1_composite__absa_hotel__seed43"
  "e20;d1_composite;absa_hotel;44;$ROOT/artifacts/e20/d1_composite/absa_hotel/seed_44;d1_composite__absa_hotel__seed44"
  "e19;;absa_hotel;43;$ROOT/artifacts/e19/absa_hotel/seed_43;e19__absa_hotel__seed43"
  "e19;;absa_hotel;44;$ROOT/artifacts/e19/absa_hotel/seed_44;e19__absa_hotel__seed44"
  "e20;d1_uniform;absa_restaurant;42;$ROOT/artifacts/e20/d1_uniform/absa_restaurant/seed_42;d1_uniform__absa_restaurant__seed42"
  "e20;d1_uniform;absa_restaurant;43;$ROOT/artifacts/e20/d1_uniform/absa_restaurant/seed_43;d1_uniform__absa_restaurant__seed43"
  "e20;d1_uniform;absa_restaurant;44;$ROOT/artifacts/e20/d1_uniform/absa_restaurant/seed_44;d1_uniform__absa_restaurant__seed44"
)
TOTAL=${#JOBS[@]}
QUEUE_IDX=0
SKIPPED=0
declare -a COMPLETED_NAMES=()
declare -a FAILED_NAMES=()
declare -A PID_GPU=()
declare -A PID_DESC=()

job_is_completed() {
  # $1 = outdir. True iff outdir/status.json exists and status=="completed".
  local status_file="$1/status.json"
  [[ -f "$status_file" ]] && grep -q '"status": *"completed"' "$status_file"
}

dispatch_on() {
  # Launch the next not-yet-completed queued job on GPU $1 (skipping any
  # already-completed jobs it walks past). Returns 1 once the queue is
  # exhausted with nothing left to launch.
  local gpu="$1"
  while (( QUEUE_IDX < TOTAL )); do
    local spec="${JOBS[$QUEUE_IDX]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))
    local runner arm task seed outdir logname
    IFS=';' read -r runner arm task seed outdir logname <<< "$spec"
    if job_is_completed "$outdir"; then
      echo "[$(ts)] [skip] $logname already completed at $outdir"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
    mkdir -p "$(dirname "$outdir")"
    local logfile="$LOG_DIR/${logname}.log"
    echo "[$(ts)] [launch] $logname on $gpu -> $outdir (log: $logfile)"
    if [[ "$runner" == "e19" ]]; then
      "$PY" "$ROOT/scripts/run_e19.py" --worker --task-id "$task" --device "$gpu" \
        --output-dir "$outdir" --seed "$seed" > "$logfile" 2>&1 &
    else
      "$PY" "$ROOT/scripts/run_e20.py" --worker --arm "$arm" --task-id "$task" --device "$gpu" \
        --output-dir "$outdir" --seed "$seed" > "$logfile" 2>&1 &
    fi
    local pid=$!
    PID_GPU["$pid"]="$gpu"
    PID_DESC["$pid"]="$logname"
    return 0
  done
  return 1
}

# Fill all 4 slots.
for gpu in "${GPUS[@]}"; do
  dispatch_on "$gpu" || true
done

# Poll: reap any finished job, immediately backfill its GPU slot.
POLL_SECONDS=20
while (( ${#PID_GPU[@]} > 0 )); do
  sleep "$POLL_SECONDS"
  for pid in "${!PID_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      code=$?
      gpu="${PID_GPU[$pid]}"
      desc="${PID_DESC[$pid]}"
      if [[ $code -eq 0 ]]; then
        echo "[$(ts)] [done] $desc on $gpu (exit 0)"
        COMPLETED_NAMES+=("$desc")
      else
        echo "[$(ts)] [FAILED] $desc on $gpu (exit $code) -- see $LOG_DIR/${desc}.log"
        FAILED_NAMES+=("$desc")
      fi
      unset "PID_GPU[$pid]"
      unset "PID_DESC[$pid]"
      dispatch_on "$gpu" || true
    fi
  done
done

echo "[$(ts)] E22 queue drained. completed=${#COMPLETED_NAMES[@]} failed=${#FAILED_NAMES[@]} skipped=$SKIPPED"
if (( ${#FAILED_NAMES[@]} > 0 )); then
  echo "[$(ts)] Failed jobs: ${FAILED_NAMES[*]}"
fi

echo "[$(ts)] Running scripts/eval_e22.py"
"$PY" "$ROOT/scripts/eval_e22.py"
echo "[$(ts)] E22 launcher done."
