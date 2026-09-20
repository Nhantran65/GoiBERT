# E21 -- Multi-seed check: does the E20 `d1_composite` gain survive across seeds?

Generated 2026-09-14T02:27Z. Execution-only follow-up to `docs/E20_RESULT.md`: the E20
seed-42 result showed `d1_composite` (75.46 pair F1) beating the `e19` baseline (74.51
pair F1) on `absa_restaurant` by +0.95 pts. This doc reruns both arms at seeds 43 and 44
(seed 42 numbers are the existing E19/E20 artifacts, unchanged) to check whether that
gain is a seed-42 fluke or holds up. **Only `absa_restaurant`, only `e19` and
`d1_composite`** -- no `cti`, no `Hotel`, no other tasks.

Runner changes (both frozen-loop-preserving; see module docstrings for details):

- `scripts/run_e19.py` / `scripts/run_e20.py`: added a `--seed` CLI option (default 42)
  threaded into `_train_any(..., seed=<seed>)`, the artifact dir (`seed_<seed>`), and
  metadata (`e19_seed` / `e20_seed`, plus the frozen loop's own `metadata.json["seed"]`).
  In E20, the composite-construction RNG seed is now derived from (equal to) the run
  seed, so seeds 43/44 draw a different composite set than seed 42 (confirmed: seed 43's
  first composite pairs train row `absa_restaurant:train:005094`, seed 44's pairs
  `absa_restaurant:train:005505`; both build 2343 composites). The E20
  `composites_manifest.json` write was also moved from *before* `_train_any` creates the
  trial dir to *after* (using the real `result["artifact_dir"]`), which is what let seeds
  43/44 land directly in `seed_43` / `seed_44` instead of being bumped to `__rerun1` the
  way the original seed-42 `d1_composite` run was (that one still lives at
  `artifacts/e20/d1_composite/absa_restaurant/seed_42__rerun1`, pre-existing, untouched).
- `scripts/eval_e20.py`: path helpers (`_checkpoint_path`, `_predictions_path`,
  `_metrics_path`, `_metadata_path`) now take a `seed` (default 42) and resolve via a new
  `_artifact_dir(arm, task, seed)` that tries the plain `seed_<seed>` dir first and falls
  back to `seed_<seed>__rerun1` when the plain dir has no `status.json` -- reproducing the
  old hardcoded seed-42 behavior by default and generalizing it for 43/44.

All four new trials completed cleanly (`status.json` -> `"completed"`) with no process
failures; LRs, NC lambda (0.1), LLRD (`gamma=0.8`, 12 layers), epochs (30), and
micro-batch (16, accum 2 -> effective 32) are identical to the seed-42 runs.

## Validation pair F1 (best epoch), absa_restaurant

| seed | e19 pair F1 | e19 best epoch | d1_composite pair F1 | d1_composite best epoch | paired diff (d1 - e19) |
|---|---|---|---|---|---|
| 42 | 74.51 | 22 | 75.46 | 21 | +0.95 |
| 43 | 74.07 | 16 | 75.31 | 14 | +1.24 |
| 44 | 73.72 | 26 | 75.00 | 18 | +1.28 |

**Mean +/- sd per arm (n=3 seeds):**

| arm | values (%) | mean (%) | sd (pts) |
|---|---|---|---|
| e19 | 74.51, 74.07, 73.72 | 74.10 | 0.40 |
| d1_composite | 75.46, 75.31, 75.00 | 75.26 | 0.23 |

**Paired differences (d1_composite - e19), per seed:** +0.95, +1.24, +1.28 pts
**Mean paired difference:** +1.16 pts
**SD of paired differences (seed noise):** 0.18 pts

**Verdict: the mean paired difference (+1.16 pts) exceeds the seed noise (sd of paired
differences = 0.18 pts) by a wide margin (~6.5x).** The gain is not a seed-42 fluke --
`d1_composite` beat `e19` on all 3 seeds, by a similar margin each time (+0.95 to +1.28
pts), and the paired-difference variance across seeds is much smaller than the mean
effect.

## Rare-pair (`k>=2|rare_pair(<=5)`) group: missed% / wrong%

(9 rows / 24 gold-aspect slots in every seed -- the group is defined by train-set pair
frequency <=5, which is seed-independent; only the model's predictions vary by seed.)

| seed | arm | missed% | wrong% |
|---|---|---|---|
| 42 | e19 | 45.8 | 12.5 |
| 42 | d1_composite | 20.8 | 16.7 |
| 43 | e19 | 41.7 | 20.8 |
| 43 | d1_composite | 33.3 | 16.7 |
| 44 | e19 | 45.8 | 12.5 |
| 44 | d1_composite | 37.5 | 16.7 |

`d1_composite` lowers rare-pair missed% vs `e19` in every seed (-25.0, -8.4, -8.3 pts),
though the size of the improvement varies more than the main pair-F1 gain does (this
group is only 24 gold slots, so it's noisier).

## Mixed-polarity (`k>=2|mixed`) group: missed% / wrong%

(109 rows / 237 gold-aspect slots in every seed -- rows with >=2 present aspects whose
gold polarities disagree.)

| seed | arm | missed% | wrong% |
|---|---|---|---|
| 42 | e19 | 19.0 | 19.4 |
| 42 | d1_composite | 16.0 | 16.0 |
| 43 | e19 | 20.3 | 19.0 |
| 43 | d1_composite | 16.5 | 18.1 |
| 44 | e19 | 20.7 | 19.8 |
| 44 | d1_composite | 18.6 | 17.3 |

`d1_composite` lowers both missed% and wrong% vs `e19` on the mixed-polarity group in
every seed (missed%: -3.0, -3.8, -2.1 pts; wrong%: -3.4, -0.9, -2.5 pts) -- a smaller but
consistent effect, same direction as the main result.

## Sources

- Seed 42: `artifacts/e19/absa_restaurant/seed_42`,
  `artifacts/e20/d1_composite/absa_restaurant/seed_42__rerun1` (pre-existing, unchanged).
- Seed 43: `artifacts/e19/absa_restaurant/seed_43`,
  `artifacts/e20/d1_composite/absa_restaurant/seed_43` (new).
- Seed 44: `artifacts/e19/absa_restaurant/seed_44`,
  `artifacts/e20/d1_composite/absa_restaurant/seed_44` (new).
- Logs: `artifacts/e19/logs/absa_restaurant__seed{43,44}.log`,
  `artifacts/e20/logs/d1_composite__absa_restaurant__seed{43,44}.log`.
- Group-decomposition and pair-F1 numbers computed via
  `scripts/eval_e20.py`'s `train_pair_frequency` / `error_decomposition_groups` /
  seed-aware path helpers (see `_artifact_dir`), read directly from each trial's
  `best_validation_metrics.json` and `validation_predictions.json` -- no retraining, no
  changes to `src/`, `models/`, or any pre-existing artifact.
