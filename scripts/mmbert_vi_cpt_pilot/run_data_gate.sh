#!/usr/bin/env bash
# Data gate stages: decompress one pinned HPLT shard, split it by lines into NPARTS raw parts, scan them in parallel
# (scan_hplt.py) and delete the raw parts.  Re-runnable: a shard whose scan counters exist is skipped.
# Usage: JOBS=32 NPARTS=32 run_data_gate.sh 10_1 9_2 9_1 [...]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-e0-cuda/bin/python}
JOBS=${JOBS:-32}
NPARTS=${NPARTS:-32}
SCAN=data/mmbert_vi_cpt_pilot/scan
RAW=data/mmbert_vi_cpt_pilot/raw_parts
mkdir -p "$SCAN" "$RAW"

for shard in "$@"; do
  done_marker="$SCAN/.$shard.scanned"
  if [[ -f "$done_marker" ]]; then echo "[gate] $shard already scanned"; continue; fi
  src="data/raw/hplt3_vie/$shard.jsonl.zst"
  [[ -f "$src" ]] || { echo "[gate] missing $src" >&2; exit 1; }
  have=$(ls "$RAW/$shard" 2>/dev/null | wc -l) || have=0
  recorded=$(cat "$RAW/$shard/.nparts" 2>/dev/null) || recorded=0
  if [[ "$have" == "$NPARTS" && "$recorded" == "$NPARTS" ]]; then
    echo "[gate] $shard raw parts already present ($have)"
  else
    rm -rf "$RAW/$shard"; mkdir -p "$RAW/$shard"
    echo "[gate] decompressing $shard -> $NPARTS raw parts"
    zstd -dc "$src" | split -n l/"$NPARTS" - "$RAW/$shard/part_"
    echo "$NPARTS" > "$RAW/$shard/.nparts"
  fi
  echo "[gate] scanning $shard with $JOBS workers (only parts without counters)"
  for part in $(ls "$RAW/$shard"); do
    [[ -f "$SCAN/${shard}_${part}.counters.json" ]] || echo "$part"
  done | xargs -P "$JOBS" -I{} "$PY" scripts/mmbert_vi_cpt_pilot/scan_hplt.py \
      --part "$RAW/$shard/{}" --shard "$shard" --out "$SCAN/${shard}_{}" --sample-mod "${SAMPLE_MOD:-8}" \
      > "$SCAN/${shard}_{}.log" 2>&1
  n_ok=$(ls "$SCAN"/${shard}_*.counters.json 2>/dev/null | wc -l)
  if [[ "$n_ok" != "$NPARTS" ]]; then echo "[gate] $shard: only $n_ok/$NPARTS parts scanned" >&2; exit 1; fi
  rm -rf "$RAW/$shard"
  touch "$done_marker"
  echo "[gate] $shard scanned"
done
