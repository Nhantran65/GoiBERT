# E17 recovery record

## Scope and source gate

On 2026-09-13, the approved E17 source binding was rechecked before any recovery mutation. The current hashes match `artifacts/e17/root_approval.json` for `scripts/run_e17.py`, the E17 model/optimizer/config, frozen E2/E7 dependencies, and all six `goiturn-s-gd` backbone files. The approved recipe remains 3 arms × 5 tasks, seed 42, 30 epochs, effective batch 32 (`micro16 × accumulation2`), BF16, max length 256, E8 NC + LLRD gamma 0.8, with `include_test=false`.

The old controller snapshot at `artifacts/e17/controller_state.json` remained marked `running`, but its last heartbeat was `2026-09-12T19:51:50Z`; host inspection at `2026-09-13T05:58:14Z` found no controller or `run_e17.py` worker for the recorded PIDs 820459, 975600, and 975603. The three Hotel logs contain tokenizer and CUDA attention warnings only; no traceback, exception, or worker return code is available. The stopping cause is therefore unknown from retained evidence.

The frozen E2 loop records `resumable=false` and `resume_support=fresh_rerun_only; no resume CLI`. Its `best_checkpoint.pt` files contain optimizer/RNG state for the best epoch, but no latest-checkpoint/resume contract exists. The partial Hotel folders and logs were moved intact to [the recovery archive](../recovery/e17_hotel_20260913T060110Z/) before restart; the stale controller and dispatch snapshots are preserved there as well.

## Recovery launch

The detached recovery controller is [scripts/recover_e17_hotel.py](../scripts/recover_e17_hotel.py). It is a separate helper and does not modify the frozen runner, model, optimizer, or protocol. It source-checks the approval binding, rechecks `nvidia-smi` before each worker, and starts each frozen `run_e17.py --worker` with `Popen(start_new_session=True, stdin=DEVNULL, stdout=log, stderr=STDOUT)`, sequentially on one GPU.

Recovery run root: `recovery/e17_hotel_20260913T060110Z/`.

At launch, controller PID `1054226` was detached with PPID 1. Baseline Hotel worker PID `1055737` was launched under it on `cuda:0`; the launch snapshot recorded GPU0 free 40,866 MiB, utilization 0%, and the profile/headroom requirement 7,402 MiB. GPU1–3 were left untouched. The durable state and heartbeat are in [controller_state.json](../recovery/e17_hotel_20260913T060110Z/controller_state.json), with logs under the same run root.

## Audit status

The read-only audit implementation is [scripts/audit_e17_predictions.py](../scripts/audit_e17_predictions.py), and its current factual output is [recovery_audit.json](../artifacts/e17/recovery_audit.json). Before Hotel recovery completed, all 12 completed non-Hotel trials had exact validation IDs and gold labels against the frozen validation splits; recomputed selection F1 matched each saved metric within `1e-12`. Hotel rows are intentionally excluded while their status is `running`.

The audit and recovery outputs contain validation data only. No test split, extra seed, grid, or new model hypothesis is introduced.

## Final recovery handoff

The detached recovery completed at `2026-09-13T07:20:55Z` with return code 0 for all three Hotel workers. It used GPU0 sequentially with 40,866 MiB free at each launch and left GPU1–3 jobs untouched. The final Hotel validation selection F1 values are baseline `0.7337373737373737` (best epoch 25), full_context `0.7324171382376717` (best epoch 26), and cavity `0.7318255250403877` (best epoch 17).

The final read-only audit in [recovery_audit.json](../artifacts/e17/recovery_audit.json) covers all 15 completed trials: validation IDs and gold labels match the frozen splits for every arm/task, and every recomputed selection F1 matches its saved metric within `1e-12`. The recovery controller state is [completed](../recovery/e17_hotel_20260913T060110Z/controller_state.json); no E17 controller or worker remains live.

The separate reader diagnostics are under [artifacts/research_diagnostics/reader_operating_point](../artifacts/research_diagnostics/reader_operating_point), [decoder_feasibility](../artifacts/research_diagnostics/decoder_feasibility), and [shadow_reader](../artifacts/research_diagnostics/shadow_reader). They are validation-only measurements and did not modify E17 sources or checkpoints.
