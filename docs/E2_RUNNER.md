# E2 validation runner

E2 is the training ablation after the frozen E1 supervised baseline.  It
keeps the `goiturn-s-gd` architecture and asks whether task-specific teacher
logits or deterministic input noise improve the student on clean validation.
The runner never loads the test split and has no test CLI.

## Frozen contracts

Every student mode (`ce`, `kd`, `nc`, `kd_nc`) constructs a fresh
`goiturn-s-gd` model through the E1 `build_classifier` helper.  It therefore
uses the E1 384-wide head, trains the whole encoder, and does not initialise
from an E1 fine-tuned checkpoint.  E1's data loader, label adapter, evaluator,
cross-entropy loss, AdamW parameter groups, linear warmup, dynamic padding and
validation loop are imported directly.  The required E1 dependency hash is
`456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.

The DAPT teacher is loaded as a local `RobertaModel` from
`models/goibert-b-dapt` with the MLM decoder and pooler removed.  It receives a
fresh CLS head whose dense hidden size is 768.  Teacher screening uses
learning rates `{1e-5, 2e-5, 5e-5}`, seed 42, and validation `selection_f1`.
Students use the locked E1 winners: `1e-5` for VSFC sentiment/topic and
`1e-4` for ViSFD, Restaurant and Hotel.

All modes use 30 epochs, effective batch size 32, AdamW weight decay 0.01,
10% linear warmup, gradient clipping at 1.0, BF16/SDPA on the approved GPU,
and max length 256 from the E1 train-only audit.  `--smoke` is the only
shortened path and is never an experiment record.

## Modes and loss composition

`teacher` trains the 768-wide task teacher.  `ce` is the one-time student
replay gate and is an engineering control, not a new baseline candidate.
`kd` adds cached teacher-logit distillation at temperature 2 with
`alpha ∈ {0.25, 0.5, 0.75}`:

```text
(1 - alpha) * CE + alpha * T^2 * KL(teacher/T || student/T)
```

The loss helper implements the declared forward direction
`KL(teacher_T || student_T)` and already includes the single `T^2` factor;
the runner uses that returned value once.  ABSA logits are reduced
independently per slot, including ViSFD's binary `OTHERS` slot.

`nc` adds symmetric KL between clean and a deterministic noisy student view
with `lambda ∈ {0.1, 0.5}`.  Noise is derived independently from
`(seed, epoch, stable_id)`, so changing data-loader order cannot change a
row's corruption; operation counts and actual change rates are recorded per
epoch.  `kd_nc` is opened only by the root scheduler
after the separate KD and NC validation gates.  The loss implementation and
noise generator live in the sibling-owned `losses.py` and `noise.py` modules.

## Teacher cache

After a validation-selected teacher checkpoint is available, create one
train-only cache:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python -m goibert_e2.train \
  --task-id vsfc_sentiment --mode teacher --learning-rate 2e-5 \
  --device cuda:0 --output-dir artifacts/e2/teacher/vsfc_sentiment/2e-05

PYTHONPATH=src .venv-e0-cuda/bin/python - <<'PY'
from pathlib import Path
from goibert_e2.train import cache_teacher_checkpoint

cache_teacher_checkpoint(
    task_id="vsfc_sentiment",
    teacher_checkpoint=Path("artifacts/e2/teacher/vsfc_sentiment/2e-05/best_checkpoint.pt"),
    repo_root=Path.cwd(),
    output_path=Path("artifacts/e2/cache/vsfc_sentiment/teacher_logits.pt"),
    device_name="cuda:0",
)
PY
```

The cache stores fp32 logits and immutable metadata: task/split, ordered
source IDs and hashes, tokenizer hashes, max length, teacher checkpoint hash,
and the full aspect slot schema.  It contains no labels, validation rows or
test rows.  A path with another identity is rejected and an identical
existing cache is reused only after complete validation.  Teacher inference
restores Python, NumPy and Torch RNG state so cache creation cannot alter a
student initialisation seed.

The durable scheduler invokes the same runner with `--mode cache`, a selected
teacher checkpoint and `--cache-path <artifact-dir>/teacher_logits.pt`; the
cache path is the only cache artifact needed by the scheduler.  Direct users
can use `cache_teacher_checkpoint` as shown above.

Run a student mode with:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python -m goibert_e2.train \
  --task-id vsfc_sentiment --mode kd --alpha 0.5 \
  --cache-path artifacts/e2/cache/vsfc_sentiment/teacher_logits.pt \
  --seed 42 --device cuda:0 --output-dir artifacts/e2/student/vsfc_sentiment/kd_a0.5_s42
```

There is intentionally no `--test` or `include_test=True` path in this
runner.  The root agent may inspect clean validation curves and predictions;
test evaluation remains a separate future gate.

## Artifacts and failures

Each run gets a unique directory and writes `status.json` (`running`,
`completed` or `failed`), `resolved_config.json`, `epoch_curves.json`, the
best validation checkpoint, validation metrics/predictions and final
`metadata.json`.  Metadata includes separate E1 dependency and E2 source
hashes, data/evaluator provenance, model/tokenizer hashes, mode, seed,
hyperparameters and runtime details.  A run is not resumable; failed
directories are preserved and are never silently treated as completed.
The metadata keeps the scheduler's E2 code hash and a separate combined E2
source/config hash; the frozen E1 dependency hash remains explicit.
