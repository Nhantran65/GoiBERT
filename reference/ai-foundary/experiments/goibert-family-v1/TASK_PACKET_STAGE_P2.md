# TASK PACKET — Stage P follow-up: finish --init-from integration (scope expanded)

GOAL: Complete the blocked part of TASK_PACKET_STAGE_P.md — implement `--init-from-b` and
`--init-from-students` in run_family.py properly, now WITH authorization to modify
`experiments/goibert-family-v1/src/train.py`, `src/distill.py`, and `src/modeling.py`.

CONTEXT: You (a previous worker run) implemented build_corpus.py, mlm_pretrain.py,
general_distill.py, lumi_job_pretrain.sh and reported the hooks blocked on file scope.
Scope is now granted. Read your own TASK_PACKET_STAGE_P.md for the full spec of the flags
and row-naming rules.

SUCCESS CRITERIA (same as Stage P spec):
- `--init-from-b <dir>`: baseline row unchanged (pure BamiBERT fine-tune, the measuring
  stick); NEW row `goibert_b_dapt` fine-tuned from the DAPT checkpoint over the same lr
  grid; when the flag is present the students' teacher becomes the best goibert_b_dapt.
- `--init-from-students <dir>`: students initialize from `<dir>/goiturn_s_gd` /
  `<dir>/goiturn_xs_gd` HF checkpoints instead of PCA init; rows named `goiturn_s_gd`,
  `goiturn_xs_gd`; result filenames carry the suffix so Stage A files are never overwritten.
- No flags => byte-identical behavior to today (verify: the vsfc --dry-run output must be
  unchanged vs current).

FILES: the three newly authorized files + run_family.py + README.md. Nothing else.

CONSTRAINTS:
- Refactor minimally: prefer adding an optional `init_dir=None` parameter threaded through
  existing functions over restructuring. Keep names/style.
- No monkeypatching, no duplicated training loops.
- Stage A result JSONs/checkpoint filenames must remain untouched by flagged runs.

DONE WHEN:
- py_compile all touched files.
- `run_family.py --dataset vsfc --dry-run` (no flags) output unchanged.
- `run_family.py --dataset vsfc --dry-run --init-from-b X --init-from-students Y` prints the
  v2 run matrix (goibert_b_dapt grid + gd students) without importing torch.
- README Stage P section updated with the final flagged commands.

VALIDATE (run, report output):
- `python -m py_compile experiments/goibert-family-v1/src/*.py`
- `python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --dry-run`
- `python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --dry-run --init-from-b results/pretrained/goibert_b_dapt --init-from-students results/pretrained`

STOP RULES: as team standard — missing prerequisite/ambiguity/hack-only => report, don't
guess; unsure => add a self-test.

ROLLBACK: git working tree diff review by supervisor; changes are flag-gated.
