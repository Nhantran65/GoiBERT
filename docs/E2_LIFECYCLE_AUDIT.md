# E2 lifecycle audit

Audit date: 2026-09-08. Scope: read-only review of the scheduler and pilot
controller lifecycle paths. No model execution or test-split access was
performed.

## Confirmed blocker

The frozen scheduler has a hash-map API mismatch on the process-completion and
process-recovery paths.

`source_hashes()` returns four keys:

```text
e1_dependency_hash, e2_code_hash, e2_config_hash, orchestrator_hash
```

`metadata_matches()` accepts only the last three keys. Four call sites pass the
whole map with `**hashes`:

| Path | Line | Effect |
|---|---:|---|
| `_validate_completed_artifact()` cache branch | 991 | cache completion raises `TypeError` |
| `_validate_completed_artifact()` train branch | 1005 | train completion raises `TypeError` |
| `_recover_running_jobs()` cache branch | 1039 | recovery raises before reusing a completed cache |
| `_recover_running_jobs()` train branch | 1047 | recovery raises before reusing a completed trial |

The observed failure is the train branch at line 1005:

```text
TypeError: metadata_matches() got an unexpected keyword argument 'e1_dependency_hash'
```

The candidate-reuse paths at lines 630 and 664 already pass the three accepted
arguments explicitly, so this is specifically a completion/recovery bug. The
minimal safe repair is to adapt the four calls to the accepted keyword set (or
add a narrow adapter), while preserving the frozen release identity used by
existing artifacts.

## Recovery state observed

At the audit point:

- `.controller/state.json` is `blocked` because the `all` scheduler exited with
  return code 1.
- `.orchestrator/state.json` contains four jobs marked `running`, even though
  their recorded PIDs are no longer present in `/proc`.
- One of those four jobs (`visfd`, LR `1e-5`) has a completed canonical artifact;
  the three VSFC Topic artifacts still have `status=running` and should be
  inspected before any relaunch. The scheduler's durable recovery path is
  intended to validate each artifact and mark it completed or failed.
- Approved state contains the three completed VSFC Sentiment teachers and the
  CE replay. No failed identity was recorded, and no stage status was written
  before the exception.

After the adapter is in place, recovery must first handle these four stale
`running` jobs. A direct scheduler restart can reuse valid canonical artifacts;
it must not launch a duplicate while an actually live child remains.

The current job list is ordered with the three incomplete Topic jobs before the
completed ViSFD job. `_recover_running_jobs()` marks the first incomplete
identity as failed and stops; because `_job_failed()` then refuses automatic
retry, an operator must explicitly resolve or archive those three incomplete
attempts before resuming. This is the declared fail-closed policy, not a reason
to silently rerun them. If the children are confirmed dead and no valid
artifact exists, a root-approved fresh attempt (with its identity and reason
recorded) is required.

## Release-hash constraint

The existing completed E2 metadata binds `orchestrator_hash` to the frozen
scheduler hash. Editing `scripts/execute_e2_validation.py` changes that hash and
would make the existing completed artifacts fail the exact-reuse contract,
causing occupied canonical paths to block relaunch. Recovery should therefore
use an external/in-memory compatibility adapter or an explicitly versioned
release migration that preserves and records the old artifact contract. Do not
silently change `REVISION` or overwrite canonical artifacts.

## Other lifecycle paths reviewed

- CLI arguments emitted by `build_command()` match the current runner parser,
  including `--coefficient`, `--alpha`, `--consistency-lambda`, cache paths,
  `--include-test false`, and stage/method fields.
- The CE replay CLI omission is already isolated in
  `scripts/run_e2_ce_replay_compat.py`; it constructs the existing scheduler
  namespace and records the wrapper hash.
- Teacher selection, cache selection, Stage A coefficient selection, and Stage
  B confirmation use consistent task/method/coefficient identities and have
  complete-grid checks. The selected-teacher and cache binding checks are
  present, though in `all` mode the cache binding check occurs only after the
  cache stage and should be kept as a post-stage gate before KD starts.
- PID-dead recovery, failed-identity refusal, canonical-path checks, and
  test-artifact rejection are present. Recovery is deliberately fail-closed;
  it should not automatically retry a failed identity.
- E1 reference scans enforce completed validation metadata and reject smoke
  records. They do not independently require the frozen E1 source hash in the
  scheduler's `e1_reference_score()`/`paired_e1_comparison()` helpers; this is
  an integrity hardening item for a future release, not the cause of the
  current stop. The pilot controller's `metadata_ok_e1()` does enforce the
  release hash.

## Minimal handoff

1. Repair only the hash-key adaptation for completion/recovery without
   invalidating the frozen artifact contract.
2. Validate that the four stale jobs are either completed with exact artifacts
   or explicitly failed; do not classify a `status=running` artifact as
   complete solely from a PID record.
3. Resume the scheduler from the existing `.orchestrator` state. Reuse the
   already approved CE/teacher records and continue the preregistered queue.
4. Before KD begins, run the existing teacher/cache binding gate; if adding the
   early all-mode check, record it as a controller hardening change with a new
   release rather than mutating this frozen run in place.
