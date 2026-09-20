#!/usr/bin/env bash
# E20 autolaunch: designed to run fully detached (nohup setsid), independent
# of any interactive session.  Polls for E19 absa_hotel + absa_restaurant
# completion, then launches the E20 4-trial run-all controller, waits for it
# to exit, then runs the evaluation script that writes docs/E20_RESULT.md.
# All python calls use the pinned venv interpreter and absolute paths.
set -uo pipefail

ROOT="/home/kiettqa/GoiBERT"
PY="$ROOT/.venv-e0-cuda/bin/python"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

LOG="$ROOT/artifacts/e20/logs/autolaunch.log"
mkdir -p "$ROOT/artifacts/e20/logs"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"
}

log "=== e20_autolaunch.sh started, pid $$, session $(ps -o sid= -p $$ 2>/dev/null | tr -d ' ') ==="

HOTEL_STATUS="$ROOT/artifacts/e19/absa_hotel/seed_42/status.json"
REST_STATUS="$ROOT/artifacts/e19/absa_restaurant/seed_42/status.json"

is_completed() {
  local f="$1"
  [ -f "$f" ] || return 1
  "$PY" -c "
import json, sys
try:
    d = json.load(open('$f'))
except Exception:
    sys.exit(1)
sys.exit(0 if d.get('status') == 'completed' else 1)
" 2>>"$LOG"
}

log "polling every 120s for E19 absa_hotel AND absa_restaurant status=completed"
while true; do
  if is_completed "$HOTEL_STATUS" && is_completed "$REST_STATUS"; then
    log "both E19 absa_hotel and absa_restaurant are completed; proceeding to E20 dispatch"
    break
  fi
  sleep 120
done

log "launching: $PY $ROOT/scripts/run_e20.py --run-all --dispatch-approved --workers-per-gpu 1"
cd "$ROOT" || { log "FATAL: cannot cd to $ROOT"; exit 1; }
"$PY" "$ROOT/scripts/run_e20.py" --run-all --dispatch-approved --workers-per-gpu 1 >> "$LOG" 2>&1
RUN_ALL_RC=$?
log "run_e20.py --run-all exited with code $RUN_ALL_RC"

log "launching evaluation: $PY $ROOT/scripts/eval_e20.py (writes docs/E20_RESULT.md)"
"$PY" "$ROOT/scripts/eval_e20.py" >> "$LOG" 2>&1
EVAL_RC=$?
log "eval_e20.py exited with code $EVAL_RC"

log "=== e20_autolaunch.sh finished (run_all_rc=$RUN_ALL_RC eval_rc=$EVAL_RC) ==="
exit 0
