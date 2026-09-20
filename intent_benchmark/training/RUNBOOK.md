# Training runbook

The locked six-trial queues are in `training_config.json`:

- GPU 0: `goi_student@2e-5`, then `goi_student@5e-5`, then `bamibert@5e-5`.
- GPU 1: `goi_dapt@2e-5`, then `goi_dapt@5e-5`, then `bamibert@2e-5`.

Run smoke first on one free authorized GPU and wait for root review:

```bash
.venv-e0-cuda/bin/python intent_benchmark/training/run_smoke.py \
  --gpu 0 --output intent_benchmark/training/smoke
```

After smoke review, start one detached controller per authorized GPU. Recheck GPU
memory first and ensure no other workload is using that GPU:

```bash
mkdir -p intent_benchmark/runs/controller-gpu0 intent_benchmark/runs/controller-gpu1
nohup .venv-e0-cuda/bin/python intent_benchmark/training/controller.py \
  --gpu 0 --runs-root intent_benchmark/runs \
  > intent_benchmark/runs/controller-gpu0/controller.out 2>&1 &
nohup .venv-e0-cuda/bin/python intent_benchmark/training/controller.py \
  --gpu 1 --runs-root intent_benchmark/runs \
  > intent_benchmark/runs/controller-gpu1/controller.out 2>&1 &
```

The controllers run one worker per GPU at a time, write controller events, worker
PIDs, trial logs, `RUNNING`/`COMPLETED` markers, summaries, best checkpoints, and
dev predictions. A failed worker stops its queue for diagnosis; do not restart a
trial while its PID is still live.
