# E2 validation orchestration

`scripts/execute_e2_validation.py` is the durable queue for the E2 training ablation. It owns stage identities, validation-only gates, selection manifests, source/config/script hashes, GPU admission, PID/log state and recovery. The training implementation remains in `src/goibert_e2/`.

The scheduler is dry-run by default. Planning does not import a dataset or start a GPU process:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python scripts/execute_e2_validation.py \
  --stage all --repo-root /home/kiettqa/GoiBERT \
  --output-root /home/kiettqa/GoiBERT/artifacts/e2 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

The generated plan records the E1 dependency hash, E2 source hash, E2 config hash, scheduler script hash, memory guards and `test_access=false`. A subprocess is launched only when the operator invokes `--execute` after the root smoke/replay gate. The scheduler never passes a test split to the runner and has no test command.

## Queue

The full queue contains 61 training trials and five cache passes:

| Stage | Trials | Identity | Gate or selection |
|---|---:|---|---|
| `stage0_teacher` | 15 | five tasks × LR `{1e-5, 2e-5, 5e-5}` × seed 42 | highest validation F1, then lower LR, then earlier best epoch |
| `stage0_cache` | 5 | selected teacher × task, train split only | one immutable `.pt` cache per task |
| `stagea_ce_replay` | 1 | VSFC sentiment, LR `1e-5`, seed 42 | E1 compatibility/replay gate |
| `stagea_kd` | 15 | five tasks × alpha `{.25,.5,.75}` × seed 42 | select alpha by validation F1; ties use lower alpha |
| `stagea_nc` | 10 | five tasks × lambda `{.1,.5}` × seed 42 | select lambda by validation F1; ties use lower lambda |
| `stageb` | 20 | selected KD and NC coefficients × seeds 43/44 | report paired validation results against E1 seeds 42–44 |

Stage B runs both selected branches for every task even when one branch is weaker at seed 42. The scheduler does not promote a combined KD+NC recipe, add seeds 45/46, or evaluate test. Those are root review decisions.

Student learning rates are fixed to the E1 winners: `1e-5` for VSFC sentiment/topic and `1e-4` for ViSFD, Restaurant and Hotel. Teacher strength is recorded against the available E1 validation reference when present; a weak teacher does not silently cancel the predeclared KD branch.

## Runner interface

Training calls the existing runner directly:

```text
python -m goibert_e2.train
  --mode teacher|ce|kd|nc|cache
  --task-id ... --learning-rate ... --seed ...
  --alpha ...                 # KD only
  --consistency-lambda ...    # NC only
  --cache-path ...            # KD only; immutable train cache
  --repo-root ... --output-dir ... --device ...
  --precision bf16 --attention-implementation sdpa
  --epochs 30 --effective-batch-size 32 --micro-batch-size 32  # teacher/cache use 8
  --stage ... --method ... --coefficient ...
  --protocol ... --revision ... --expected-e2-code-hash ...
  --expected-e2-config-hash ... --expected-orchestrator-hash ...
  --include-test false
```

The scheduler passes the stage/method and hash fields because the current runner accepts and records those exact flags. The runner writes `mode`, `stage`, `method`, `coefficient`, `hyperparameters.alpha` or `hyperparameters.consistency_lambda`, `source_hashes.e1_dependency/e2/e2_config/orchestrator`, `include_test=false`, the E1 artifact set and validation metrics.

Cache creation uses the runner's `cache` mode, which calls its public cache function:

```text
cache_teacher_checkpoint(
  task_id, teacher_checkpoint, repo_root, output_path,
  device_name, precision, attention_implementation, micro_batch_size
)
```

The resulting `.pt` file is validated by `goibert_e2.cache.load_teacher_logit_cache`. It must contain only train IDs/text hash/schema metadata and FP32 teacher logits. The selected teacher checkpoint hash, source IDs and E1 dependency hash must match; validation and test rows are forbidden.

## Recovery and GPU policy

There is one lock per output root, one active child per device, a deterministic canonical path for each identity, a PID/command/log entry and append-only events. Existing canonical artifacts are reused only when their mode, task, seed, LR, coefficient, source hashes, recipe, validation files and `include_test=false` match exactly. Smoke artifacts, malformed artifacts, occupied paths and duplicate identities stop the queue. A failed identity is recorded and is never automatically retried or replaced by the best rerun.

The scheduler checks `nvidia-smi` immediately before each spawn and leaves existing processes untouched. The frozen profile uses a common **5 GiB free guard** for teacher/cache micro-batch 8 and student CE/KD/NC micro-batch 32. A new root-approved memory profile can be supplied explicitly, for example:

```bash
--student-memory-guard-gib 5 --teacher-memory-guard-gib 5 --nc-memory-guard-gib 5
```

The selected values are recorded in the orchestration state and changing them mid-queue stops the run. Changing a guard is appropriate only after measuring the exact workload and keeping a safety margin; it does not authorize overcommit or process termination.

Approved records live under `artifacts/e2/.orchestrator/`: stage manifests, teacher selection, Stage A coefficient selection, paired E1 comparison, `state.json`, `events.jsonl`, logs and failures. No test predictions or test metrics belong in this tree.

## Pilot continuation controller

Once the operator has started the three VSFC teacher pilots and the CE replay at their canonical paths, use [continue_e2_after_pilots.py](../scripts/continue_e2_after_pilots.py). It waits for the supplied PIDs, validates completed metadata and checkpoint artifacts against the frozen release, compares CE replay validation F1 with the exact E1 seed-42/LR-1e-5 record (absolute tolerance `0.005`), registers the replay through the frozen scheduler, and then launches the remaining `--stage all` queue with the 5 GiB guards. It never launches while a pilot is incomplete and never retries a failed pilot or scheduler process.

The manifest is operator-owned and must use canonical paths:

```json
{
  "protocol": "E2-pilot-manifest-v1",
  "teacher_pilots": [
    {"learning_rate": 0.00001, "artifact_dir": "/abs/.../artifacts/e2/stage0_teacher/vsfc_sentiment/1e-05/seed_42", "pid": 123},
    {"learning_rate": 0.00002, "artifact_dir": "/abs/.../artifacts/e2/stage0_teacher/vsfc_sentiment/2e-05/seed_42", "pid": 124},
    {"learning_rate": 0.00005, "artifact_dir": "/abs/.../artifacts/e2/stage0_teacher/vsfc_sentiment/5e-05/seed_42", "pid": 125}
  ],
  "ce_replay": {"artifact_dir": "/abs/.../artifacts/e2/stageA/ce_replay/vsfc_sentiment/seed_42", "pid": 126},
  "additional_nc_pilots": [
    {"task_id": "vsfc_sentiment", "learning_rate": 0.00001, "coefficient": 0.1, "artifact_dir": "/abs/.../artifacts/e2/stageA/nc/vsfc_sentiment/coef_0p1/seed_42", "pid": 127}
  ]
}
```

Run it from a durable terminal:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python scripts/continue_e2_after_pilots.py \
  --pilot-manifest artifacts/e2/pilot_manifest.json \
  --root-smoke-gate artifacts/e2/preflight/smoke_gate.json \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

`additional_nc_pilots` is optional and may contain any preregistered NC screening identities; each entry is validated and waited on before the queue starts, so it can be reused by the frozen scheduler without overlap. Controller state, pilot validation, scheduler commands, PIDs, logs and timestamps are stored under `artifacts/e2/.controller/`. A live prior controller/scheduler or a prior failed controller blocks a duplicate continuation.
