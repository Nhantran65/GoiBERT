#!/bin/bash
# Run commands from a queue file strictly one job per GPU: each GPU worker waits until no probe/CPT job of ours is
# using it, pops the next line, runs it with CUDA_VISIBLE_DEVICES set, then repeats. usage: gpu_queue.sh <queue.txt> "<gpus>"
q=$1; gpus=(${2:-0 1 2 3}); cd /home/kiettqa/GoiBERT; lock=$q.lock; log=${q%.txt}.log
pop() { ( flock 9; line=$(head -n1 $q); [ -n "$line" ] && sed -i '1d' $q; echo "$line" ) 9>$lock; }
busy() { for p in $(pgrep -f "probe_[c]sv_finetune|probe_[m]mbert_finetune|cpt_[t]rain.py"); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=$1$" && return 0; done; return 1; }
for g in "${gpus[@]}"; do
  ( while true; do
      while busy $g; do sleep 60; done
      line=$(pop); [ -z "$line" ] && break
      echo "$(date +%H:%M) gpu$g START $line" >> $log
      CUDA_VISIBLE_DEVICES=$g CUBLAS_WORKSPACE_CONFIG=:4096:8 bash -c "$line"
      echo "$(date +%H:%M) gpu$g END $line" >> $log
    done ) &
done; wait; echo "$(date +%H:%M) queue empty" >> $log
