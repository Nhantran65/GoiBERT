#!/bin/bash
# Sequential 4-GPU CPT arms for pilot B (a -> c -> b2 -> b), then nothing else; eval is launched separately.
cd /home/kiettqa/GoiBERT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p artifacts/pilot_b/logs
for arm in a c b2 b; do
  echo "=== arm $arm start $(date)" >> artifacts/pilot_b/logs/chain.log
  .venv-e0-cuda/bin/python -m torch.distributed.run --nproc_per_node 4 --master_port 29571 scripts/pilot_b/cpt_train.py --arm $arm --tokens 300e6 --micro 8 > artifacts/pilot_b/logs/cpt_$arm.log 2>&1
  echo "=== arm $arm exit $? $(date)" >> artifacts/pilot_b/logs/chain.log
done
echo "=== chain done $(date)" >> artifacts/pilot_b/logs/chain.log
