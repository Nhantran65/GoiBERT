# E19 execution status

Written 2026-09-14 during the run; see `docs/E19_PLAN.md` for the locked spec.
This is a status snapshot at first-epoch verification, not a final readout.

## What was built

- `scripts/run_e19.py` -- preflight / worker / run-all controller, modeled on
  `scripts/run_e18.py` with the fork machinery dropped. It monkeypatches
  `goibert_e2.train.build_e2_student` (to build `goibert_e2.modeling.build_e2_teacher`
  over `models/goibert-b-dapt` instead of the E1 student backbone) and
  `goibert_e2.train.make_optimizer` (to build
  `goibert_e7.optim.build_layerwise_optimizer`, `num_layers=12`, `gamma=0.8`,
  `weight_decay=0.01`), then calls `goibert_e2.train._train_any(..., mode="nc")`
  directly -- **not** `train_student_trial`, which locks the LR to the E1/E2
  student winners (1e-5 / 1e-4) and would reject the teacher LRs used here.
  `src/goibert_e2` and `src/goibert_e7` are imported, unmodified.
- `configs/e19/protocol.json` -- locked recipe (dispatch order, per-task LR /
  NC lambda, LLRD params, epochs/batch/precision, references, gate).
- No new `src/goibert_e19` package -- not needed; every piece is either
  frozen E2/E7 code or a thin monkeypatch inside `scripts/run_e19.py`.

## Preflight (`artifacts/e19/preflight.json`, device `cuda:1`)

Status: **passed**, elapsed 10.4s.

1. Model builds for all 5 tasks, hidden size 768, finite logits:
   `absa_restaurant` [4,12,4], `absa_hotel` [4,34,4], `visfd` [4,42]
   (flat -- `aspect_class_counts` isn't uniform for visfd, `(4,4,4,4,4,2,4,4,4,4,4)`,
   so `E2TeacherClassifier.forward`'s reshape guard `len(set(counts))==1` is
   false and it stays flat; this is pre-existing frozen E2 behavior, not new
   to E19 -- the same code path already produced the referenced E2-teacher
   visfd 84.36 result), `vsfc_topic` [4,4], `vsfc_sentiment` [4,3]. All pass.
2. Optimizer coverage for every task's own LR: 28 groups
   (`embeddings_{decay,no_decay}` + `block_00`..`block_11` x
   `{decay,no_decay}` + `head_{decay,no_decay}`), every trainable parameter
   in exactly one group, 0 duplicates, 201 trainable tensors. Endpoint scales
   verified against `gamma**exponent` (e.g. absa_restaurant lr=2e-5: head=1.0
   -> 2e-5, block_09=0.64 -> 1.28e-5, block_00=0.0859 -> 1.72e-6,
   embeddings=0.0687 -> 1.37e-6).
3+4. One real NC micro-batch (16x256) step per task, forward+backward+optimizer
   step, loss and grad norm finite for all 5:

   | task | base LR | NC lambda | CE | consistency | total | grad norm | peak reserved |
   |---|---|---|---|---|---|---|---|
   | absa_restaurant | 2e-5 | 0.1 | 1.397 | 0.0041 | 1.397 | 2.24 | 1870 MiB |
   | absa_hotel | 5e-5 | 0.1 | 1.400 | 0.0054 | 1.400 | 1.42 | 2192 MiB |
   | visfd | 5e-5 | 0.5 | 1.356 | 0.0044 | 1.359 | 1.72 | 3430 MiB |
   | vsfc_topic | 1e-5 | 0.5 | 1.334 | 0.0039 | 1.336 | 5.52 | 1984 MiB |
   | vsfc_sentiment | 2e-5 | 0.1 | 1.095 | 0.0019 | 1.095 | 3.80 | 2014 MiB |

   Peak VRAM across all 5 tasks: **3.43 GiB max** (visfd) at micro 16 x 256,
   bf16, sdpa -- well inside GPU1's ~45 GiB free. No OOM risk; the micro
   8 x accumulation 4 fallback was not needed and was not used.
5. Strict `state_dict` round-trip into a fresh `build_e2_teacher` model
   (absa_restaurant): 0 missing keys, 0 unexpected keys, `strict_pass=True`.
6. Encoder directory hash unchanged before/after (`models/goibert-b-dapt`
   untouched).

## GPU dispatch

At dispatch time: GPU0 was mid-run by another job (up to 87% util, 26 GiB) at
the start of this session but its heavy processes exited partway through and
it settled at ~2.6 GiB / 0% util for the remainder; GPU2 was at 28.1 GiB
(vLLM engine) and GPU3 at 30.0 GiB, both belonging to other users. Only GPU1
was consistently idle (20 MiB used, 0% util) across the whole session, per
the round-4 research-loop memory note. Dispatch was pinned to `cuda:1` only
(`--devices cuda:1`) rather than opportunistically also claiming GPU0 the
moment its numbers dipped under the 5 GiB/5% guard -- two lingering low-usage
processes remained on GPU0 and could wake up mid-trial. This is a
conservative, single-GPU, strictly sequential run; the controller code
supports one-worker-per-eligible-GPU if a future invocation is given more
than one device.

## Launch

Controller launched detached (`nohup setsid ... &`, disowned) at
`2026-09-13T17:11:19Z` (`2026-09-14 00:11 ICT`):

- Launcher (run-all): PID 2698510, parent PID 1 (fully detached, survives
  this session).
- Worker 1 (`absa_restaurant`, `cuda:1`): PID 2698961.
- `artifacts/e19/controller_state.json` heartbeats every ~10s while a job is
  active; on resume it adopts any trial whose `status.json` already reads
  `"completed"` and refuses to re-launch a task with a live PID.
- Dispatch order enforced: `absa_restaurant, absa_hotel, visfd, vsfc_topic,
  vsfc_sentiment` (locked in `TASKS`), one task at a time on `cuda:1`.

## First-epoch verification

`absa_restaurant` (first in queue) completed epoch 1 at
`artifacts/e19/absa_restaurant/seed_42/epoch_curves.json`:

- epoch 1: 43.70 s, epoch 2: 43.31 s (train batches, batch32 = 220
  optimizer steps/epoch, 7028 train rows) -- includes validation time.
- train losses epoch 1: total 1.0436, CE 1.0432, consistency 0.0045 (all
  finite); `selection_f1` 0.0 (expected: LR is still deep in the 10%
  linear warmup at step 220/6600 total steps, effective LR ~4.6e-7).
- `best_checkpoint.pt` / `best_validation_metrics.json` /
  `validation_predictions.json` were written at epoch 1 (first score beats
  `-inf`), confirming the checkpoint/validation path executes end to end.

## Projected timing (approximate)

Per-epoch time scaled from the measured absa_restaurant rate
(~43.5s / 220 steps) by each task's steps/epoch (train rows / 32, rounded up)
-- an approximation; only absa_restaurant is empirically measured so far,
and ABSA aspect count (12 vs 34) may add some per-step cost the other tasks
don't scale-share with vsfc:

| task | train rows | steps/epoch | est. s/epoch | est. trial (30 ep) |
|---|---|---|---|---|
| absa_restaurant | 7028 | 220 | 43.5 (measured) | ~21.8 min |
| absa_hotel | 7180 | 225 | ~44.6 | ~22.3 min |
| visfd | 7786 | 244 | ~48.3 | ~24.1 min |
| vsfc_topic | 11426 | 358 | ~70.8 | ~35.4 min |
| vsfc_sentiment | 11426 | 358 | ~70.8 | ~35.4 min |

Sequential total on the single eligible GPU: **~139 minutes (~2h19m)**,
projected completion **~2026-09-14 02:30 ICT** (19:30 UTC 09-13), plus
per-task process-start overhead (tokenizer/model load, a few seconds each,
already folded into the gap observed before each task's first
`epoch_curves.json` write).

## Not done (by design)

Per the task instructions this run only verifies the first epoch of the
first trial and reports; it does not wait for all 5 trials, does not compute
final selection_f1 / gate outcome, and does not touch `artifacts/e1`..`e18`,
`src/`, or any other `scripts/run_e*.py`. Check
`artifacts/e19/controller_state.json` and each task's `epoch_curves.json` /
`status.json` for live progress; `artifacts/e19/logs/<task>.log` for full
worker stdout/stderr.
