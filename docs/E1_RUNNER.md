# E1 supervised runner

The E1 runner measures a fresh supervised fine-tune of
`models/goiturn-s-gd/`. Every trial loads that checkpoint independently and
initialises a new task head. The encoder is fully trainable. E1 does not use
the DAPT teacher, KD, noise consistency, corpus text, test examples, or any
historical task score.

## Data and metric interface

`src/goibert/train.py` deliberately uses one strict interface owned by the
data/evaluator agent:

```python
load_task(
    task_id: str,
    repo_root: pathlib.Path,
    *,
    include_test: bool,
    smoke: bool = False,
) -> TaskBundle
```

The returned `TaskBundle` exposes `train`, `validation` and (only when
`include_test=True`) `test` splits. Each split has `texts`, `labels` and
stable `ids`. A categorical task has `num_classes` and one integer label per
row. An ABSA task has `aspect_names`, `num_aspects`,
`aspect_class_counts`, and one integer label per aspect slot. `provenance` and
`metadata` are mappings containing source/file/evaluator hashes and the
locked label/slot policy. `aspect_class_counts` is usually four per slot; a
binary `OTHERS` slot may be represented as `10 * 4 + 1 * 2` instead of
inventing invalid polarity classes.

For tasks whose source labels differ from model-head labels, the bundle also
exposes `labels_for_model(split)`. The runner uses that adapter for CE labels
while preserving each split's raw labels for metric evaluation; this is how
ViSFD maps OTHERS raw code 4 to binary model code 1 and back in the evaluator.

The evaluator exposes exactly:

```python
evaluate_task(
    *,
    task_id: str,
    gold: Sequence[Any],
    predictions: Sequence[Any],
    aspect_names: list[str],
    class_names: list[str],
    aspect_class_counts: list[int],
    split: str,
) -> Mapping[str, Any]
```

It must return the audited task-specific `selection_f1` field. The runner does
not guess between macro/micro/conditional F1 aliases. Other named metrics are
saved unchanged. Selection is therefore locked before screening, and the
evaluator must keep absent ABSA labels in the CE/evaluation policy agreed in
the data audit.

## Fixed E1 recipe

- AdamW, betas `(0.9, 0.999)`, epsilon `1e-8`.
- Weight decay `0.01`, excluding bias and LayerNorm weights.
- 30 epochs, no early stopping, effective batch size 32.
- Learning rates `1e-5, 2e-5, 5e-5, 1e-4, 2e-4`.
- Ten percent linear warmup followed by linear decay.
- Gradient clipping at norm 1.0 with `error_if_nonfinite=True`.
- `CLS -> dropout(0.1) -> dense(384) -> tanh -> dropout(0.1) -> logits`.
- Cross entropy on labels. ABSA slots use the audited slot widths; absent
  labels are not silently dropped by the runner.
- Dynamic padding. Train-only token length statistics choose 256 when at most
  1% of train rows exceed it, otherwise 512. A rate above 1% at 512 is a hard
  task blocker and raises before a full run.
- Precision is explicit (`fp32` or smoke-approved `bf16`) and never inferred
  independently per trial.

The tail accumulation window is weighted by its actual number of examples, so
microbatching does not change the per-example gradient. The default
microbatch is 32 and no accumulation is used in the pilot.

## Stages and seed budget

The default `run_grid` command only writes a plan. `--execute` is required to
launch subprocesses, and each approved device has at most one trial at a time.
The launcher records a source hash at stage start and passes it to every
child; a child fails if the runner changes during a launch.

```bash
PYTHONPATH=src python -m goibert.run_grid \
  --stage screening --tasks vsfc_sentiment \
  --devices cuda:3 --precision fp32
```

For the actual pilot, add `--execute` only after the E0 gate and environment
smoke have passed. The stages are:

1. `screening`: seed 42 and all five learning rates for each requested task.
   It creates train/validation loaders only.
2. `shortlist`: selects the top two learning rates by seed-42 validation
   `selection_f1`, then runs seeds 43 and 44 for each. The selected learning
   rate is the highest mean over seeds 42–44; ties use the lower learning
   rate.
3. `final`: runs seeds 45 and 46 for the selected learning rate. This does not
   reopen the LR search.
4. `test`: is a separate evaluation-only command. It requires a locked
   `E1-test-gate-v1` selection manifest, `--confirm-test-gate`, and preselected
   checkpoint paths. It loads the test split only here; it never retrains.

Each trial writes its resolved arguments, source/data/evaluator provenance,
length statistics, model/optimizer metadata, epoch curve, best validation
checkpoint, validation predictions, and final status under
`artifacts/e1/<task>/<stage>/`. Existing directories are never overwritten;
reruns receive a suffix. A checkpoint contains optimizer, scheduler and RNG
state, but the runner does not claim resume support unless all of those states
are present.

## Test-gate manifest

The root agent creates this only after final validation selection is complete.
It is locked by the source hash and checkpoint paths; no external signing
service is required:

```json
{
  "protocol": "E1-test-gate-v1",
  "locked": true,
  "source_hash": "...",
  "selections": [
    {
      "task_id": "vsfc_sentiment",
      "seed": 45,
      "learning_rate": 0.00002,
      "artifact_dir": "/abs/path/artifacts/e1/vsfc_sentiment/final/2e-05/seed_45"
    }
  ]
}
```

The manifest must contain every frozen task/seed checkpoint to evaluate. The
test command checks the source hash, task metadata, checkpoint existence and
train-only max-length decision before reading test data. Test outputs are
stored beside the locked checkpoint and tagged with the evaluator provenance.

## Durable validation orchestration

For the long validation run, use
`scripts/execute_e1_validation.py`. It is an operator wrapper around the
frozen runner: it does not modify `src/` or `configs/`, never requests a test
split, and refuses to run if the runner hash is not
`456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.

The default command only writes a plan. Start it after the external VSFC
pilot has completed, or use `--wait-for-pilot` with the pilot output path so
the wrapper cannot duplicate that process:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python scripts/execute_e1_validation.py \
  --stage all \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --wait-for-pilot \
  --execute
```

Run it in a durable operator session (for example, a managed `tmux` or
`systemd-run` job) and keep the printed log/state paths. The wrapper checks
free memory immediately before each spawn and requires at least 8 GiB on the
approved device. It keeps one subprocess per device, persists PID, command,
device, log and attempt directory in `artifacts/e1/.orchestrator/state.json`,
and appends events to `events.jsonl`.

The wrapper scans all existing artifacts before launching. A record is reused
only when task, stage, learning rate, seed, 30 epochs, BF16, SDPA, effective
and micro batch 32, frozen source hash, complete checkpoint/prediction files,
and non-smoke status all match. The completed pilot may live under a `pilot/`
directory; its metadata stage and recipe still have to match. An occupied
path without an exact completed record blocks the run, so a running job is
never duplicated. Failed attempts receive unique attempt directories and are
listed in the failure state; they are excluded from the approved manifest and
the stage stops for review.

The stages are gated in order: 25 screening records, 20 new shortlist records
(seed-42 records are reused from screening), then 10 final records. A missing
seed, duplicate identity, mixed source hash, malformed metadata, or ambiguous
completed artifact stops the orchestration. No automatic retry or cherry-pick
is performed. The durable files `screening_approved.json`,
`shortlist_approved.json`, `final_approved.json`, and `approved_records.json`
are the only records used for later aggregation.
