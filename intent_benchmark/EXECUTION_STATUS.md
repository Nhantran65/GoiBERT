# Intent benchmark execution status

Updated by the training controller/evaluation workflow. The locked runner and
configuration hashes are pinned in `training/root_approval.json`.

## Current state

- Smoke: complete and approved for full pilot.
- GPU queues: completed detached at 2026-09-13; controller PIDs from controller markers GPU0=4127475, GPU1=4127892.
- Full trials: 6/6 completed successfully; controller events contain no failed trial.
- Dev selection: locked in `results/selection.json` after all six trials.
- Test evaluation: exactly three dev-selected winners completed after selection lock.
- Timing: complete on shared NVIDIA L40S/GPU0 with 10 warmups and two batch-1 passes per winner.

## Required constraints

- GPU 0 queue: `goi_student@2e-5`, `goi_student@5e-5`, `bamibert@5e-5`.
- GPU 1 queue: `goi_dapt@2e-5`, `goi_dapt@5e-5`, `bamibert@2e-5`.
- Seed 42, batch 16, max 15 epochs, patience 3, unweighted eight-class CE.
- No new learning rates, seeds, filters, or test-based selection.

## Live locations

- GPU0 controller log/events/heartbeat: `runs/controller-gpu0/`.
- GPU1 controller log/events/heartbeat: `runs/controller-gpu1/`.
- Each trial writes its worker PID, log, marker, summary, checkpoint, and dev predictions in `runs/<model>-lr.../`.

## Final artifacts

- Dev lock: `results/selection.json`.
- Test metrics/predictions: `results/test_results.json`, `test_metrics_*.json`, `test_predictions_*.jsonl`.
- Saved-prediction audit: `results/saved_prediction_audit.json` (pass).
- Timing/benchmark: `results/timing.json`, `results/benchmark.csv`.
- Report: `results/REPORT.md`.
