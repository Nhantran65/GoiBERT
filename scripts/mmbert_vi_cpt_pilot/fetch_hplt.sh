#!/usr/bin/env bash
# Range-parallel downloader for pinned HPLT 3.0 vie_Latn shards (plan §3).
# The server allows only ~3-4 MB/s per connection, so each shard is fetched as byte ranges and reassembled by
# concatenation (valid for byte ranges of one file).  Every chunk is size-verified and retried; the assembled file is
# checked with `zstd -t` (integrity) and sha256-summed.  Correctly sized chunks are reused, so the script resumes.
# Usage: JOBS=10 fetch_hplt.sh <shard> [<shard> ...]   with shard like 9_1 -> data/raw/hplt3_vie/<shard>.jsonl.zst
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=data/raw/hplt3_vie
BASE=https://data.hplt-project.org/three/sorted/vie_Latn
CHUNK=$((256 * 1024 * 1024))
JOBS=${JOBS:-10}
PARTS=$OUT/.parts
mkdir -p "$PARTS"

get_chunk() {  # <url> <start> <end> <out>
  local url=$1 a=$2 b=$3 out=$4 want=$((b - a + 1)) sz
  for attempt in 1 2 3 4 5 6; do
    if curl -sS --fail --max-time 3600 -r "$a-$b" -o "$out.tmp" "$url"; then
      sz=$(stat -c%s "$out.tmp" 2>/dev/null || echo 0)
      if [[ "$sz" == "$want" ]]; then mv "$out.tmp" "$out"; return 0; fi
      echo "[fetch] chunk $a-$b: got $sz want $want (attempt $attempt)" >&2
    fi
    rm -f "$out.tmp"; sleep $((attempt * 2))
  done
  echo "[fetch] chunk $a-$b FAILED" >&2
  return 1
}
export -f get_chunk

for shard in "$@"; do
  final="$OUT/$shard.jsonl.zst"
  url="$BASE/$shard.jsonl.zst"
  size=$(curl -sSI "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
  echo "[fetch] $shard size=$size bytes"
  if [[ -f "$final" ]] && [[ "$(stat -c%s "$final")" == "$size" ]] && zstd -t "$final" 2>/dev/null; then
    echo "[fetch] $shard already complete and valid"
    continue
  fi
  rm -f "$final"
  mkdir -p "$PARTS/$shard"
  for attempt in 1 2 3; do
    seq 0 "$CHUNK" "$size" | awk -v c="$CHUNK" -v s="$size" -v d="$PARTS/$shard" '{a=$1; b=a+c-1; if (b>=s) b=s-1; want=b-a+1; printf "%s %s %s %s\n", a, b, d"/"a, want}' |
      while read -r a b f want; do
        if [[ -f "$f" ]] && [[ "$(stat -c%s "$f")" == "$want" ]]; then continue; fi
        rm -f "$f"; printf '%s %s %s\n' "$a" "$b" "$f"
      done > "$PARTS/$shard/.todo"
    todo=$(wc -l < "$PARTS/$shard/.todo")
    echo "[fetch] $shard attempt $attempt: $todo chunks to fetch"
    if [[ "$todo" == "0" ]]; then break; fi
    xargs -P "$JOBS" -n 3 bash -c 'get_chunk "$0" "$1" "$2" "$3"' "$url" < "$PARTS/$shard/.todo" || true
  done
  rm -f "$PARTS/$shard/.todo"
  cat $(ls "$PARTS/$shard"/[0-9]* | sort -n) > "$final"
  now=$(stat -c%s "$final")
  if [[ "$now" != "$size" ]]; then echo "[fetch] $shard INCOMPLETE $now != $size"; exit 1; fi
  if ! zstd -t "$final"; then
    echo "[fetch] $shard zstd integrity check FAILED" >&2; exit 1
  fi
  sha=$(sha256sum "$final" | cut -d' ' -f1)
  echo "[fetch] $shard done bytes=$now sha256=$sha"
  printf '%s bytes=%s sha256=%s\n' "$shard" "$now" "$sha" >> "$OUT/sha256.txt"
  rm -rf "$PARTS/$shard"
done
