# TASK PACKET — family v1: noisy-slice evaluation (robustness table §7.4)

GOAL: One new script `src/eval_noisy.py` that scores saved task checkpoints on CORRUPTED
copies of each dataset's test split, producing the robustness table nobody in Vietnamese
NLP has published — and the verdict on whether noise-consistency training earns its keep.

BUSINESS REASON: Customer chat is noisy (no diacritics, teencode). NC ablation on clean
tests was inconclusive by design; §13.8 says NC must be judged on noisy slices. This also
yields the §7.4 robustness baseline table for the technical report.

FILES (ONLY): NEW `experiments/goibert-family-v1/src/eval_noisy.py`, `README.md` append.
Nothing else. Reuse noise.py, data.py, evaluate/metrics code via imports — no duplication.

SPEC:
- CLI: `python src/eval_noisy.py --dataset {vsfc,vlsp_restaurant,visfd,phoatis}
  [--models m1,m2,...] [--seed 42] [--slices all|diacritics|teencode|combined]`
- Slices, built deterministically (random.Random(seed)) from the CLEAN test split texts:
  * `clean` (control, unchanged)
  * `no_diacritics`: strip_diacritics on every text
  * `teencode`: teencode() with p_word=0.6 (heavier than training's 0.3)
  * `combined`: corrupt() as in training
- Checkpoints: discover in `results/checkpoints/` for the dataset+seed, matching the known
  naming (baseline, goibert_b_dapt, goiturn_s_gd, goiturn_xs_gd, and *_nc variants when
  present; compose with __seed<k>). `--models` filters. Load each onto the available
  device, batch-eval each slice, macro F1 + accuracy (reuse existing metrics code).
- Output: `results/noisy_eval_<dataset>[__seed<k>].md` — one table: rows = models, columns
  = slices (macro F1), plus a delta column (clean − combined) so robustness is readable at
  a glance. Also a JSON alongside with full per-class metrics.
- Must run all models of one dataset in ONE process (load, eval, free) — a single LUMI job
  per dataset. Progress prints with flush.
- `--smoke`: cap test rows at 200, still all slices.

NOTE ON CHECKPOINT FORMAT: task checkpoints are torch.save dicts (see how run_family/
train.py/distill.py save them — model_state/state_dict + config). Read those files' save
code first and load accordingly; validate by rebuilding the architecture from the stored
config and load_state_dict strict=True.

DONE WHEN:
- py_compile passes; `--dry-run` lists discovered checkpoints + planned slices without
  torch.
- `--smoke` runs end-to-end IF torch importable; torch is NOT importable on this host →
  dry-run + py_compile, state which checks ran.
- README documents the LUMI command per dataset.

VALIDATE (run, report):
- `python -m py_compile experiments/goibert-family-v1/src/eval_noisy.py`
- `python experiments/goibert-family-v1/src/eval_noisy.py --dataset vsfc --dry-run`

STOP RULES: standard; if checkpoint naming/format is ambiguous, STOP and report exactly
what you found instead of guessing a loader.
ROLLBACK: one additive file.
