#!/bin/bash
# After the main chain: arm b2 with 40x lr on the shared channels (b2 at base lr left them at ~0.3% of |E_tok|), then eval.
cd /home/kiettqa/GoiBERT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
until grep -q "chain done" artifacts/pilot_b/logs/chain.log; do sleep 120; done
echo "=== arm b2_hlr start $(date)" >> artifacts/pilot_b/logs/chain.log
.venv-e0-cuda/bin/python -m torch.distributed.run --nproc_per_node 4 --master_port 29572 scripts/pilot_b/cpt_train.py --arm b2 --tag _hlr --shared-lr-mult 40 --tokens 300e6 --micro 8 > artifacts/pilot_b/logs/cpt_b2_hlr.log 2>&1
echo "=== arm b2_hlr exit $? $(date)" >> artifacts/pilot_b/logs/chain.log
for pass in 1 2 3; do
  scripts/pilot_b/eval_arm.sh b2_hlr "0 1 2 3"
  n=$(grep -l '"epoch": 20' artifacts/pilot_b/probe/b2_hlr/*.json 2>/dev/null | wc -l); [ "$n" -ge 8 ] && break
done
echo "arm b2_hlr evaluated: $n/8 complete $(date)" >> artifacts/pilot_b/logs/eval_all.log
echo "followup done $(date)" >> artifacts/pilot_b/logs/eval_all.log
