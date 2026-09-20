# TASK PACKET — family v1: seed-suffixed outputs (small, surgical)

GOAL: Make multi-seed runs safe — when `--seed` != 42, every output filename (result JSONs,
best JSONs, checkpoints, summary markdown) gets a `__seed<k>` suffix so concurrent/repeat
runs never overwrite the seed-42 artifacts.

BUSINESS REASON: The VLSP win (+10.7 F1 over BamiBERT) is single-seed; we must run seeds
43/44 to harden the claim without destroying the seed-42 evidence.

FILES (ONLY these, in experiments/goibert-family-v1/src/): `run_family.py`, `train.py`,
`distill.py`. Do not touch anything else (another worker is active in goiturn-synthetic-v0/).

SPEC:
- seed == 42 → filenames EXACTLY as today (backward compatible, nothing moves).
- seed != 42 → append `__seed{seed}` before the extension for: per-run result JSONs, the
  `*__best.json`, checkpoint `.pt` files, and `summary_<dataset>*.md` (both plain and
  `__stage_p` variants).
- The summary table itself should state the seed in the header line.
- Concurrent different-seed jobs on the same dataset must have fully disjoint output paths.

DONE WHEN:
- py_compile passes on the three files.
- `run_family.py --dataset vlsp_restaurant --dry-run --seed 43 --init-from-b X
  --init-from-students Y` prints the planned filenames WITH the seed suffix; the same
  command with --seed 42 prints today's names unchanged.

VALIDATE (run, report):
- py_compile each file
- both dry-runs above (plus a no-flag vsfc --dry-run to prove default behavior unchanged)

STOP RULES: standard (ambiguity/hack → report, don't guess; unsure → self-test).
ROLLBACK: 3-file diff, flag-gated by seed value.
