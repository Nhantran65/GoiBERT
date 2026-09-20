# E2 progress and durable recovery v3

At the 2026-09-08 progress check, the host had no live E2 controller or
training child. The old v2 JSON heartbeat still said `running`, so that file
was stale. The four KD jobs left in the legacy ledger as `running` had valid,
completed canonical artifacts:

- VSFC Topic, alpha 0.5 and 0.75;
- ViSFD, alpha 0.25 and 0.5.

They each have 30 epochs, a completed metadata contract, validation
predictions, and `include_test=false`. The recovery adapter is therefore
expected to adopt them by identity. They must not be retrained.

The immutable completed inventory before resuming is 25 training artifacts:
15 teacher trials, one CE replay, eight KD trials, and one NC trial, plus five
train-logit caches. The remaining preregistered work is the unfinished KD/NC
screening and then Stage B confirmation. No test split is opened by this
queue.

`scripts/supervise_e2_validation.py` is the v3 runtime supervisor. It writes
`artifacts/e2/.controller/recovery_v3_status.json`, keeps a real heartbeat,
launches the existing recovery adapter in a separate session, and reattaches
to a live child instead of starting a duplicate. Its child output is kept in
`artifacts/e2/.controller/recovery_v3_child.log`.

The launch is intentionally durable:

```bash
setsid nohup .venv-e0-cuda/bin/python scripts/supervise_e2_validation.py \
  --repo-root /home/kiettqa/GoiBERT \
  --output-root /home/kiettqa/GoiBERT/artifacts/e2 \
  --root-smoke-gate /home/kiettqa/GoiBERT/artifacts/e2/preflight/smoke_gate.json \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  > artifacts/e2/.controller/recovery_v3_supervisor.log 2>&1 < /dev/null &
```

The current cause of the old controller exit is unknown: host process checks
showed the controller and its parent gone, while detached children completed.
No persisted training error was found. v3 records the live supervisor and
child separately so a later status check can distinguish a live queue from a
stale historical heartbeat.

The v3 command does not pass `--fresh-rerun-interrupted`. A dead partial job
therefore blocks for review instead of being retried automatically. The four
completed KD artifacts described above are adopted by contract and are not
rerun; the three old interrupted Topic attempts remain preserved in the v2
archive.
