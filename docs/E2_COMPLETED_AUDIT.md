# E2 completed validation audit

Audit date: 2026-09-09. The audit is validation-only. It did not read the test split, test predictions, or model weights. The E2 supervisor completed with return code 0 after 61 training trials and 5 train-logit cache passes.

## Execution integrity

The approved ledger contains exactly 61 full training identities:

| Stage | Completed records | Metadata | Best checkpoint |
|---|---:|---:|---:|
| Teacher, 5 tasks × 3 LR | 15 | 15 | 15 |
| CE replay | 1 | 1 | 1 |
| KD screening, 5 tasks × 3 alpha | 15 | 15 | 15 |
| NC screening, 5 tasks × 2 lambda | 10 | 10 | 10 |
| Confirmation, selected KD + NC × seeds 43/44 | 20 | 20 | 20 |
| **Total** | **61** | **61** | **61** |

All 61 records have `status=completed`, `include_test=false`, `smoke=false`, and an existing `best_checkpoint.pt`. All five approved train-logit caches exist. The three interrupted VSFC Topic teacher attempts are stored under `artifacts/e2/recovery/` and are excluded from the completed count. One NC seed-42 artifact appears under an alias duplicate directory with the same checkpoint hash; the approval ledger counts that identity once.

## Pairing and selection checks

I checked the selected KD and NC metadata for all five tasks and seeds 42, 43, and 44, together with the E1 record at the same task, seed, and fixed learning rate. This covers 30 E2 records and 30 E1 baseline records. Metadata values, validation scores, stage/method/seed/coefficient, and checkpoint presence match the paired comparison file with zero mismatches.

The coefficients are the validation-only seed-42 winners under the declared rule (score descending, then coefficient ascending): KD alpha `{0.25, 0.75, 0.25, 0.25, 0.25}` for Sentiment, Topic, ViSFD, Restaurant, and Hotel; NC lambda `{0.1, 0.5, 0.5, 0.1, 0.1}` in the same order. No test access is recorded in the selection manifest, comparison file, or checked metadata.

## Paired validation results

Scores below are fractions in the artifacts; the table shows percentages. The mean and sample SD use seeds 42–44. Delta is E2 minus E1 in percentage points.

| Task | Method | Coefficient | E1 mean ± SD | E2 mean ± SD | Delta | Positive seeds | Gate ≥0.3 pp and ≥2/3 |
|---|---|---:|---:|---:|---:|---:|---|
| VSFC Sentiment | KD | 0.25 | 84.49 ± 0.09 | 84.40 ± 0.33 | −0.10 | 2/3 | No |
| VSFC Sentiment | NC | 0.1 | 84.49 ± 0.09 | 84.93 ± 0.22 | **+0.44** | 3/3 | **Yes** |
| VSFC Topic | KD | 0.75 | 80.10 ± 0.66 | 81.06 ± 0.24 | **+0.96** | 3/3 | **Yes** |
| VSFC Topic | NC | 0.5 | 80.10 ± 0.66 | 79.80 ± 0.08 | −0.31 | 2/3 | No |
| ViSFD | KD | 0.25 | 80.25 ± 0.38 | 80.26 ± 0.38 | +0.02 | 1/3 | No |
| ViSFD | NC | 0.5 | 80.25 ± 0.38 | 81.23 ± 0.28 | **+0.99** | 3/3 | **Yes** |
| UIT-ABSA Restaurant | KD | 0.25 | 68.98 ± 0.13 | 68.40 ± 0.44 | −0.59 | 0/3 | No |
| UIT-ABSA Restaurant | NC | 0.1 | 68.98 ± 0.13 | 69.39 ± 1.19 | **+0.41** | 2/3 | **Yes** |
| UIT-ABSA Hotel | KD | 0.25 | 72.07 ± 0.33 | 71.64 ± 0.33 | −0.43 | 0/3 | No |
| UIT-ABSA Hotel | NC | 0.1 | 72.07 ± 0.33 | 72.68 ± 0.81 | **+0.61** | 2/3 | **Yes** |

The independent recomputation is stored at [artifacts/e2/validation_final_audit.json](../artifacts/e2/validation_final_audit.json). It records the exact per-seed scores, deltas, means, sample SDs, positive-seed counts, and gate decisions.

## Interpretation boundary

The strongest validation evidence is NC for VSFC Sentiment and ViSFD, and KD for VSFC Topic: each improves on all three paired seeds. NC also passes the predeclared mean-gain gate on Restaurant and Hotel, but each has one negative seed and Restaurant has high between-seed spread, so these are weaker confirmation candidates. KD is negative on both ABSA tasks and does not pass on Sentiment or ViSFD under this fixed teacher, temperature, and learning-rate recipe.

These results support changing the training objective for several tasks while keeping the student architecture fixed. They do not establish test-set improvement, statistical significance, a new shared backbone checkpoint, or a win over BamiBERT. The next research decision remains outside this audit: confirm the three stable candidates with additional seeds before opening test evaluation or KD+NC combination runs.
