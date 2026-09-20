# E18 execution status — preflight + profile (batch 1: b0/b1/b2)

Status: **preflight PASSED, profile PASSED**. The 6 full training trials
(b0/b1/b2 × {absa_restaurant, absa_hotel}) have **not** been launched — per
instructions, this stops here for root review.

## Files created

- `src/goibert_e18/__init__.py`
- `src/goibert_e18/modeling.py` — `E18Classifier`, `build_e18_student`,
  `candidate_splits`, `build_fork_masks`, `randomize_fork_masks`,
  `CONNECTOR_IDS`, `ARMS`.
- `src/goibert_e18/losses.py` — `compute_fork_ce` (isolated, testable fork-CE
  core), `make_patched_compute_loss` (the `goibert_e2.train.compute_loss`
  monkeypatch factory), `ForkTelemetry`.
- `configs/e18/protocol.json`
- `scripts/run_e18.py` — `--preflight` / `--profile` / `--worker --arm ARM
  --task-id TASK --device cuda:N --output-dir DIR` / `--run-all --arms ...
  --tasks ... --dispatch-approved`.
- Artifacts: `artifacts/e18/preflight.json`, `artifacts/e18/profile.json`,
  `artifacts/e18/source_manifest.json`,
  `artifacts/e18/preflight/strict_reload.pt`.

Nothing under `src/goibert`, `src/goibert_e2`, `src/goibert_e7`,
`src/goibert_e17`, other `scripts/run_e*.py`, `models/`, or
`artifacts/e1`..`artifacts/e17` was edited. `models/goiturn-s-gd` file
hashes and `.py` file hashes under `src/goibert*_e{1..17}`/`src/goibert` were
identical before and after every preflight/profile run.

Ran on GPU1 (`cuda:1`, confirmed idle: 20 MiB used, 0% util; GPU0/2/3 were
excluded — GPU0 had another job's 4.6 GiB resident, GPU2/3 were at 100%
util from other users' jobs).

## Preflight numbers (`artifacts/e18/preflight.json`, status: `passed`)

**1. Parity (64 Restaurant validation rows, b1_fork, dropout zeroed for
exact reproducibility — see "Implementation notes" below):**

| check | value | threshold | pass |
|---|---|---|---|
| CLS logits, fork-armed vs plain SDPA forward, FP32 | 2.98e-7 | ≤ 1e-4 | yes |
| CLS logits, fork-armed vs plain SDPA forward, BF16 | 3.91e-3 | ≤ 0.1 | yes |
| Fork row 0 (full mask) vs CLS, FP32 | 2.61e-7 | ≤ 1e-4 | yes |
| Fork row 0 (full mask) vs CLS, BF16 (extra, not required) | 3.91e-3 | — | — |

These numbers are consistent with the fact that CLS/token rows *never* read
a fork row as key/value (proven by construction in `modeling.py`'s
docstring), so the two paths are the same computation graph up to floating
point summation order in a differently-shaped matmul.

Alternation invariant, exercised over 4 consecutive forward calls on the
same batch (simulating repeated clean/noisy pairs): pattern was exactly
`[computes forks, plain, computes forks, plain]` — i.e. clean→forks,
noisy→plain, next clean→forks, ... An `model.eval()` forward afterward
computed no forks either. Both match the required invariant exactly.

**2. b0 with λ=0 reproduces the ordinary E2 NC loss** (same seed, fixed
Restaurant batch, dropout zeroed in both models for exact comparability):

| quantity | value |
|---|---|
| clean logits max\|Δ\| | 0.0 |
| noisy logits max\|Δ\| | 0.0 |
| CE max\|Δ\| | 0.0 |
| consistency max\|Δ\| | 0.0 |
| total loss max\|Δ\| | 0.0 |
| reference/b0 total loss | 1.46055198 / 1.46055198 |

b0's `last_forks` was `None` for every batch (`considered=0`), confirming b0
never enters the fork path regardless of `lambda_fork`.

**3. Gradient checks** (16-row fixed batch, ≥2-gold-aspect Restaurant train
rows, backprop only the fork CE):

| arm | encoder grad norm | encoder params w/ grad | head grad norm | head params w/ grad |
|---|---|---|---|---|
| b1_fork | 1.4524 (finite, >0) | 165 | 2.6109 (>0) | 4 |
| b3_fork_detach | 0.0 (0 params touched) | 0 | 2.2330 (>0) | 4 |

Matches spec exactly: b1 gradient reaches the encoder, b3's `.detach()`
before the head blocks it (encoder grad exactly 0, 0/165 encoder params
received any gradient); head grad is nonzero for both.

**4. Split acceptance / connector-position fraction**, 256 train rows with
≥2 gold aspects per task, fresh GD backbone + fresh head:

| task | considered | accepted | acceptance rate | connector fraction of accepted |
|---|---|---|---|---|
| absa_restaurant | 256 | 221 | **86.3%** | 95.0% |
| absa_hotel | 256 | 222 | **86.7%** | 96.8% |

**Deviation from the plan's expectation to flag explicitly:** `docs/E18_PLAN.md` says acceptance is "expected lower [than ~0.8] with a fresh GD checkpoint" than the ~0.8 reference measured on the *fine-tuned* E8 checkpoint. The measured rate here (86–87%) is not lower — it is essentially the same as or slightly higher than that reference. This is being reported as-is per instructions, not adjusted. A plausible reading: fork-row score(s\*) > score(fork 0) is a fairly weak bar for a fresh (undertrained) model, since neither side is well-calibrated yet, so segment fork scores beat the unsplit fork more often at this stage; whether that acceptance rate stays this high once the encoder is trained toward the fork objective is exactly one of the things batch-1 training will show. Root should sanity-check this reading before batch 1.

**5. Loader/shape checks, strict reload, optimizer coverage:**

- Restaurant: `num_aspects=12`, CLS logits shape `[B,12,4]`; Hotel:
  `num_aspects=34`, CLS logits shape `[B,34,4]`. Both finite, both eval and
  train-mode forward shapes match, eval always has `last_forks=None`.
- Strict checkpoint round-trip: an E18 `state_dict()` (169 tensors) loads
  with `strict=True` into both a plain `build_e2_student(...)` model **and**
  a fresh `build_e18_student(...)` model — zero missing/unexpected keys in
  either direction.
- Optimizer coverage (`goibert_e7.optim.build_layerwise_optimizer` +
  `validate_optimizer_groups`, unmodified): 24 groups (embeddings + 10
  blocks + head, × decay/no-decay), 169/169 parameters covered exactly
  once, 0 duplicates, for both tasks. Endpoint LR scales:
  `embeddings=0.107374`, `block_00=0.134218`, `block_09=1.0`, `head=1.0`
  (`gamma=0.8`, `num_layers=10`) — matches E7/E8's contract exactly.
- `models/goiturn-s-gd` file hashes and `src/goibert*` `.py` hashes
  (goibert, goibert_e2..e17) identical before/after the whole preflight run.

## Profile numbers (`artifacts/e18/profile.json`, status: `passed`)

Median of 3 measured full NC steps (forward clean+forks, noisy forward,
backward, optimizer step) after 2 warmup steps, fresh model per arm, same
GPU (`cuda:1`), Restaurant task, `micro_batch_size=16 × accumulation=2`:

| shape | arm | median step (s) | peak reserved VRAM | ratio to b0 |
|---|---|---|---|---|
| fixed_train_batch (actual 16×45) | b0_baseline | 0.1276 | 870 MB | 1.00 |
| fixed_train_batch | b1_fork | 0.2007 | 952 MB | **1.57×** |
| fixed_train_batch | b2_random_mask | 0.2137 | 946 MB | **1.67×** |
| synthetic_16×256 | b0_baseline | 0.1381 | 2053 MB | 1.00 |
| synthetic_16×256 | b1_fork | 0.2270 | 2747 MB | **1.64×** |
| synthetic_16×256 | b2_random_mask | 0.2447 | 2745 MB | **1.77×** |

Both shapes pass the plan's stop-gate (`b1/b0` and `b2/b0` ≤ 2.0×; measured
1.57–1.77×), and peak reserved VRAM (≤2.75 GB) is far under GPU1's 45 GB
free — no OOM risk at all for a single worker, and plenty of headroom for
two concurrent workers on GPU1 alone if that's ever wanted.

Fork telemetry sanity numbers from the profile's own fixed batches are also
consistent with preflight check 4 (i.e., the same acceptance/connector
mechanics, just on a much smaller sample so noisier — e.g. all 18
`considered` rows accepted on the tiny 16×45 fixed batch).

## Implementation notes / deviations from a literal reading of the plan

1. **Fork-forward mechanism**: implemented as a direct, simplified
   specialization of `SlotEncoder.forward`'s row-extension trick
   (`competitive_slots_probe_v2.py`) with `K=1` and no `delta`/`qbias` (the
   CLS row is literally the unmodified CLS, not a "slot" with an offset).
   Verified by construction (see the module docstring) and empirically
   (parity numbers above) that CLS/token rows are unaffected by the
   presence of fork rows.

2. **"Re-armed by the patched compute_loss" (forks_armed timing) — resolved
   differently from a literal reading, same behavioral outcome.** The plan
   says `forks_armed` "is set True at construction and re-armed by the
   patched `compute_loss`". Tracing `goibert_e2.train._train_loop` for
   `mode="nc"` shows `compute_loss(...)` is called *between* the clean
   forward and the noisy forward, not after both. If `compute_loss` sets
   `forks_armed = True` at that point (as a literal reading suggests), the
   very next forward call — the *noisy* one — would see `forks_armed=True`
   and incorrectly compute forks for the noisy batch too; and after that
   noisy forward disarms itself, the *next batch's clean* forward would
   start disarmed, so no batch after the first would ever compute forks
   correctly. I could not find a way to honor "compute_loss re-arms" and
   also satisfy the explicit outcome the plan requires ("exactly the clean
   forward of each training batch computes forks and the noisy forward does
   not; validation never does"), because `compute_loss` is the only hook
   patchable inside the frozen `_train_loop`, and it fires at the wrong
   point in the clean→noisy sequence for a simple set-true. Implemented
   instead as a self-contained toggle inside `E18Classifier.forward`: a
   fork-armed training call computes forks and disarms; the next
   training call (necessarily the noisy forward, since `mode="nc"` always
   alternates clean/noisy 1:1) re-arms *itself* for the following call
   without computing forks; eval-mode calls never touch the flag.
   `compute_loss`'s patch only reads and clears `model.last_forks`; it does
   not touch `forks_armed` at all. This reproduces the plan's stated
   outcome exactly (verified in preflight check 1's alternation test:
   `[compute, plain, compute, plain]` across 4 consecutive calls, and eval
   never computing forks) with no held ambiguity in the observable
   contract — flagging this because the *mechanism description* in
   §"Model và loss" differs from what's implemented, even though the
   *behavioral contract* elsewhere in the same section is met exactly.

3. **Dropout zeroed for the two exact-numeric preflight checks (1 and 2
   only)**: RoBERTa's SDPA self-attention applies attention-probability
   dropout as a raw float (`dropout_p=...if self.training else 0.0`) passed
   straight into `scaled_dot_product_attention`, not via an `nn.Dropout`
   submodule, and the fork path recomputes attention manually rather than
   calling that module's `forward`. Comparing a wider (fork-augmented) and
   a narrower (plain) forward pass with *any* active dropout is not a
   meaningful bitwise comparison — the two shapes draw different amounts of
   random noise, even from the same seed, since dropout RNG consumption is
   tied to tensor size, not logical row identity. Both of these specific
   preflight checks therefore run under a temporary zero-dropout context
   (`_zero_dropout` in `scripts/run_e18.py`, covering both `nn.Dropout`
   submodules and the raw `dropout_prob` floats), restored immediately
   after. This affects preflight-check methodology only; training itself
   uses ordinary dropout throughout (nothing in `modeling.py` or the worker
   changes dropout behavior).

4. **`root_approval.json` gate**: unlike `scripts/run_e17.py`, `run-all`
   here is gated only by an explicit `--dispatch-approved` CLI flag (plus
   the frozen/matching source manifest check), not by a separate
   `root_approval.json` artifact. The task's own CLI spec
   (`--run-all --arms ... --tasks ... `) didn't call for that extra file, so
   I kept the simpler gate. Say the word if you'd like the
   `root_approval.json` file-based gate added to match E17 exactly before
   launch.

5. **b3_fork_detach and b4_composite**: implemented in `modeling.py`
   (`b3_fork_detach` works, verified in preflight check 3) but, per the
   plan, not part of batch 1's dispatch list (`BATCH1_ARMS = (b0_baseline,
   b1_fork, b2_random_mask)`). `b4_composite` (E8 + composite union,
   data-only) is out of scope for this runner entirely — the plan says it's
   E8-checkpoint-based and data-only, not part of E18's training code.

No check failed; nothing needed correcting against the spec.

## Exact commands to launch batch 1 (6 trials) once root approves

```bash
cd /home/kiettqa/GoiBERT
CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/kiettqa/GoiBERT/.venv-e0-cuda/bin/python \
  scripts/run_e18.py --run-all \
  --arms b0_baseline b1_fork b2_random_mask \
  --tasks absa_restaurant absa_hotel \
  --dispatch-approved
```

This re-checks GPU eligibility (`<5 GiB used` and `≤5% util`) at every
single dispatch, refuses if the source manifest (`src/goibert_e18/*`,
`scripts/run_e18.py`, `configs/e18/protocol.json`, `src/goibert_e2/*`,
`src/goibert_e7/*`, backbone files) has changed since this preflight/profile
run, runs one worker per eligible GPU, and writes a durable
`artifacts/e18/controller_state.json` (heartbeat + PID map) that a restarted
`--run-all` will adopt completed trials from. Individual trials can also be
run/resumed one at a time, e.g.:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/kiettqa/GoiBERT/.venv-e0-cuda/bin/python \
  scripts/run_e18.py --worker --arm b1_fork --task-id absa_restaurant \
  --device cuda:1 --output-dir artifacts/e18/b1_fork/absa_restaurant/seed_42
```

Re-run `--preflight`/`--profile` first if any of the above source files
change, or if you want fresh GPU-availability numbers before dispatch (GPU0
and GPU2/3 were busy with other users' jobs at the time of this run; only
GPU1 was confirmed idle).
