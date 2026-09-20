#!/bin/bash
# Waits for each CPT arm checkpoint and evaluates it (3 passes of eval_arm.sh to fill OOM gaps). GPUs 1-3 only: GPU0
# carries the most foreign load and a probe next to the DDP rank does not fit there.
cd /home/kiettqa/GoiBERT
for arm in a c b2 b; do
  until [ -f artifacts/pilot_b/cpt/$arm/summary.json ]; do sleep 120; done
  for pass in 1 2 3; do
    scripts/pilot_b/eval_arm.sh $arm "1 2 3"
    n=$(grep -l '"epoch": 20' artifacts/pilot_b/probe/$arm/*.json 2>/dev/null | wc -l); [ "$n" -ge 8 ] && break
  done
  echo "arm $arm evaluated: $n/8 complete $(date)" >> artifacts/pilot_b/logs/eval_all.log
done
echo "eval_all done $(date)" >> artifacts/pilot_b/logs/eval_all.log
