# E2 protocol review

Review owner: independent E2 reviewer. Review date: 2026-09-08.

This is a read-only protocol review. It does not run a model and does not open
the test split. The implementation review remains pending until the E2 runner
and orchestration files are present.

## Current decision

The research question is well scoped: keep the GoiTurn-S-GD student and its
E1 head fixed, then test task-specific learning signals. The planned E2a/E2b
comparisons are interpretable if the CE replay, teacher cache identity, and
validation-only gates below are enforced.

No E2 training run should be treated as protocol-approved solely from the
plan. The following gates are required before the first teacher or student
full run.

## Required protocol gates

1. **CE replay fidelity.** The E2 `alpha=0`/CE path must be checked against
   the frozen E1 `compute_loss`, batch weighting, data order, scheduler steps,
   seed initialization, head initialization, and checkpoint selection. A
   numerical loss-only unit test is insufficient: the smoke/replay artifact
   must record the same resolved recipe and show no unexplained difference in
   loss/validation behavior. If E2 deliberately computes CE in FP32 while E1
   relies on its BF16 autocast behavior, record the exact reason and prove the
   resulting difference is within a declared tolerance.

2. **Teacher identity and tokenizer contract.** Before a cache is accepted,
   verify that teacher and student use byte-identical tokenizer files and the
   same `max_length`, tokenization options, class order, slot order, and
   `aspect_class_counts`. The cache must bind to the complete selected teacher
   checkpoint (weights plus task-head/config identity), not an informal or
   partial hash. A missing/mismatched identity must fail closed.

3. **Cache provenance.** The current cache helper binds IDs/text hash,
   tokenizer hashes, max length, teacher hash, and slot schema. The runner
   must also persist and validate the data-file/evaluator/source-version hash
   and inference precision required by the E2 plan. Reusing a cache after any
   of these inputs changes would invalidate the KD comparison. Cache creation
   must remain train-only, ordered-ID checked, immutable, and free of labels,
   validation rows, or test rows.

4. **KD direction and slot reduction.** Keep the declared forward KL
   `KL(teacher || student) * T^2`, with teacher detached, and apply softmax
   independently per ABSA slot. ViSFD must use ten four-class slots plus the
   binary OTHERS slot; OTHERS is included in detection/CE/KD but excluded from
   the polarity-pair metric. Do not flatten all 42 outputs into one softmax.

5. **NC confound accounting.** E2b uses two stochastic forwards and, with the
   declared 50% unchanged operation, partly measures dropout consistency. Log
   clean/noisy operation counts, actual change rate, elapsed time and peak
   memory. Do not describe a gain as text-noise-only unless a later clean/clean
   ablation separates the dropout component. Keep effective optimizer updates
   equal to E1 and report that E2b is not equal-FLOP.

6. **Selection and split boundary.** Teacher LR/epoch and student alpha or
   lambda may be selected from train/validation only. E2 runner code must call
   `load_task(..., include_test=False)` and must not inspect test paths. E1
   comparison values must come from the locked validation records for the same
   task, seed, LR and evaluator. The E2 confirmation seeds 43/44 are an
   evidence stage, not a five-seed final estimate.

## Non-blocking interpretation limits

- A single seed-42 teacher is a fixed condition; student seed variation does
  not include teacher-seed variance.
- The predeclared 0.3-point / two-of-three heuristic is useful for screening,
  but is not a significance test.
- E2c should remain closed until E2a and E2b are reviewed separately; its
  coefficient combination must be recorded as a new condition rather than
  silently folded into either branch.

## Implementation review — integration pass (`src/goibert_e2/train.py`)

The integration has resolved the first draft's KD scaling, packed-slot, noise
stream, runtime-device, and scheduler CLI/layout defects. The remaining
blockers below must be fixed and covered by tests before a teacher pilot or CE
replay is called valid.

### KD scaling (resolved)

`logits_kd_loss` already returns `T² * KL(teacher || student)` and the current
runner uses that value once.

### Blocker B — categorical KD/NC receives an invalid empty slot schema

The latest runner now passes explicit slot counts, fixing the packed ViSFD
42-output case. It still passes `counts=()` for categorical VSFC tasks;
`losses._normalise_slot_counts` rejects an empty sequence, so categorical KD
and NC fail before their first update. Pass `None` for categorical tasks (or
make the loss treat an empty tuple as categorical) and add a categorical KD/NC
smoke test. Keep explicit `(4,)*10 + (2,)` for ViSFD and the declared counts
for Restaurant/Hotel, with a regression test that changing an invalid class
cannot affect the loss.

### NC stable-ID stream (resolved)

The current helper uses `noisy_batch` with a separate record-local RNG per
stable ID, removes the stale `batch_index` call, and records operation counts
and actual change rate.

### Blocker D — cache validation is weaker than the plan

`_validate_cache_for_task` checks IDs, tokenizer hashes, length and slot
schema, but not the current train text hash, data/evaluator provenance,
precision, or the E1 dependency hash. The cache helper also does not persist
all of those fields. A changed source file with unchanged row IDs could
silently reuse stale teacher logits. Add the current train text/data/evaluator
identity to the cache contract and fail closed on mismatch.

The cache now stores those data/provenance fields and the E2 source-hash map,
but the scheduler's stage-0 cache validator only checks IDs and the frozen E1
hash. It does not compare cache `source_hashes` (E2 code/config/orchestrator)
with the current launch hashes. Cache mode also accepts expected-hash CLI flags
without validating them before creation. Add both checks so a stale cache
cannot be approved after an E2 code change.

### Blocker G — Stage B applies KD cache binding to NC records

`stageb` mixes the selected KD and NC confirmation records, then calls
`_validate_teacher_binding(..., require_cache=True)` on the combined list.
NC metadata intentionally has no teacher cache, so the validation will reject
every NC confirmation after the 20 runs. Validate KD confirmation records with
the cache requirement and NC records without it (while retaining selected
teacher identity checks where applicable).

### Runtime metadata (resolved)

The call site now passes the resolved device, and source/config hashes include
`configs/e2`. Keep the metadata contract covered by the integration test.

### Blocker E — orchestration and runner contracts do not match

The scheduler now launches `goibert_e2.train` with the canonical `teacher`,
`ce`, `kd`, `nc`, and `cache` modes, and the current tests parse every
generated command. Cache identity is a single immutable `teacher_logits.pt`
payload; the scheduler and runner agree on that layout. Keep the dry-run and
artifact validation tests in the preflight gate.

### Blocker F — attention backend must stay matched (resolved in current draft)

An earlier orchestration draft set `ATTENTION_IMPLEMENTATION = "eager"`,
which would have confounded the E1 replay. The current file now says `sdpa`,
matching E1 and the plan. Keep this value frozen and include it in the source
and artifact identity checks.

No model execution or test-split access was performed for this review. The
current focused/orchestration tests pass (`21 passed`), including generated
CLI parsing and cache-path layout. They still do not cover categorical KD/NC
with the empty slot schema or cache source-hash freshness; those two tests are
required before launch.

The durable execution log still contains earlier planned guard values of
8/14 GiB and the pre-fix 29-test status. Refresh that record to the current
5 GiB guard and 21-test integration result before treating the handoff as the
final source of truth.

One additional documentation mismatch is present: `docs/E2_RUNNER.md` writes
`KL(student/T || teacher/T)`, while `docs/E2_EXECUTION_PLAN.md` and the loss
implementation specify forward `KL(teacher/T || student/T)`. Freeze one
direction in the runner contract and update the documentation/tests together;
the current wording is insufficient evidence for an auditable run.
