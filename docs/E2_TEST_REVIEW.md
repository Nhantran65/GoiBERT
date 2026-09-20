# E2 test evaluation review

Review scope: the validation-selected E2 winners only. This review does not
run model inference and does not use test results to select a branch,
coefficient, epoch, or seed.

## Locked selection audit

The selection was checked against
`artifacts/e2/.orchestrator/stageB_paired_e1_comparison.json`,
`stageA_selection.json`, and the canonical checkpoint metadata. The 15
records below are complete, `include_test=false`, `smoke=false`, use the
`goiturn-s-gd` student with the E1 384-dimensional head, and have a matching
`best_checkpoint.pt` SHA-256. Seed 42 is from Stage A; seeds 43 and 44 are
from Stage B. The duplicate Sentiment seed-42 alias has the same metadata and
checkpoint hash and is counted once using the canonical `seed_42` path.

| Task | Method/coefficient | Seed 42 best epoch / F1 | Seed 43 best epoch / F1 | Seed 44 best epoch / F1 |
|---|---|---:|---:|---:|
| VSFC Sentiment | NC / lambda 0.1 | 13 / 84.7722 | 9 / 84.8352 | 6 / 85.1750 |
| VSFC Topic | KD / alpha 0.75 | 20 / 80.7842 | 27 / 81.2274 | 10 / 81.1809 |
| ViSFD | NC / lambda 0.5 | 15 / 81.0884 | 30 / 81.0544 | 27 / 81.5563 |
| UIT-ABSA Restaurant | NC / lambda 0.1 | 21 / 69.0600 | 30 / 68.3987 | 27 / 70.7119 |
| UIT-ABSA Hotel | NC / lambda 0.1 | 27 / 73.5776 | 26 / 72.4502 | 26 / 72.0157 |

The coefficients agree with the validation-only selection rule and paired
comparison file. The frozen dependency, E2 code/config, and orchestrator
hashes match all 15 records. No test split or test prediction was read during
this preflight.

## Execution requirements for the test worker

- Use exactly these 15 canonical artifact paths and evaluate each once. Do
  not evaluate the other E2 branch or choose a checkpoint from test results.
- E2 metadata is not E1 metadata: it does not provide E1's top-level
  `source_hash` contract. Bind the worker to the E2 fields
  `e1_dependency_source_hash`, `source_hashes`, `e2_code_hash`,
  `e2_config_hash`, `orchestrator_hash`, and the recorded backbone/tokenizer
  hashes instead of copying the E1 manifest validator unchanged.
- Build the test provenance with `load_task(..., include_test=True)` while
  keeping the train-only length audit. Test outputs must include stable IDs
  and raw gold labels.
- Rebuild the exact student architecture from metadata and the frozen
  `goiturn-s-gd` backbone before loading each checkpoint. Check the checkpoint
  state and model specification; do not silently load an E1 or teacher
  architecture.
- Use BF16, SDPA, max length 256, evaluation mode, no gradients, and the
  recorded batch configuration. Do not retrain, run teacher inference, add
  seeds, or overwrite existing outputs.
- For ViSFD, raw gold `OTHERS` is label 4 while model-space predictions use
  the binary `OTHERS` code 0/1. Recompute with
  `goibert.metrics.evaluate_task(task_id="visfd", ...)`; keep OTHERS in
  detection and exclude it from polarity-pair scoring.

## Post-run audit requirements

After the 15 prediction files exist, verify the expected row counts and unique
stable IDs: 3,166 for each VSFC task, 2,224 for ViSFD, 1,938 for Restaurant,
and 2,030 for Hotel. Recompute every reported metric from saved predictions,
requiring exact agreement with its `test_metrics.json` (up to JSON float
serialization), and verify checkpoint/metadata/data/backbone/tokenizer hashes
remain unchanged.

The paired E1 baseline must be read from the existing E1 test outputs for
seeds 42, 43, and 44; do not run new E1 inference and do not substitute the
historical five-seed E1 aggregate.

## Final independent audit — PASS

The CUDA rerun produced complete output for all 15 selected checkpoints. The
CPU run is retained under
`artifacts/e2/cpu_diagnostic_20260909/` for diagnostic provenance only and is
excluded from this readout. The canonical access log records CUDA devices
`cuda:1`, `cuda:2`, and `cuda:3`, BF16, SDPA, and inference-only execution.

I recomputed every metric from the saved GPU prediction rows with
`goibert.metrics.evaluate_task`, including the explicit ViSFD `OTHERS` mapping
(raw gold 4; model prediction 0/1). All 15 prediction files have the expected
row count and exactly matching stable IDs; the saved gold rows match the
current test split. Recomputed metrics match each `e2_test_metrics.json`, and
the E2 readout aggregates match the per-seed values. Checkpoint, metadata,
backbone, tokenizer, evaluator, data, and selection-manifest hashes are
unchanged. E1 values below were read from existing E1 test artifacts for the
same seeds 42–44; E1 inference was not rerun.

| Task | Metric | E1 test 42–44 | E2 GPU test 42–44 | Δ E2−E1 | BamiBERT paper |
|---|---|---:|---:|---:|---:|
| VSFC Sentiment | Accuracy | 92.22994 ± 0.35172 | 92.37734 ± 0.14927 | +0.14740 | 93.86 |
| VSFC Sentiment | Macro F1 | 81.25550 ± 0.74944 | 81.30612 ± 0.78816 | +0.05062 | 83.41 |
| VSFC Topic | Accuracy | 88.48179 ± 0.20060 | 89.30301 ± 0.19299 | +0.82123 | 89.34 |
| VSFC Topic | Macro F1 | 79.16025 ± 0.51647 | 79.44863 ± 0.37450 | +0.28838 | 79.90 |
| ViSFD | Detection micro F1 | 86.76303 ± 0.21143 | 87.10587 ± 0.30054 | +0.34284 | 89.14 |
| ViSFD | Aspect-polarity pair micro F1 | 79.55407 ± 0.43374 | 80.61568 ± 0.25657 | +1.06162 | 84.24 |
| UIT-ABSA Restaurant | Detection micro F1 | 84.21116 ± 0.35166 | 84.51332 ± 0.54173 | +0.30216 | 88.01 |
| UIT-ABSA Restaurant | Aspect-polarity pair micro F1 | 68.83163 ± 0.67511 | 69.85670 ± 0.62583 | +1.02506 | 74.89 |
| UIT-ABSA Hotel | Detection micro F1 | 78.74862 ± 0.19028 | 79.19666 ± 0.29034 | +0.44804 | 79.99 |
| UIT-ABSA Hotel | Aspect-polarity pair micro F1 | 71.88761 ± 0.39640 | 72.27967 ± 0.39244 | +0.39206 | 72.65 |

All ten E2 rows improved over the paired E1 mean. E2 remains below every
reported BamiBERT target in this table; F1 equivalence with the paper is still
unverified, and the UIT-ABSA files remain the public candidate mirror.

Evidence: `artifacts/e2/test_readout.json`,
`artifacts/e2/test_access_log.json`, the 15 per-checkpoint
`e2_test_predictions.json` and `e2_test_metrics.json` files, and the archived
CPU diagnostic record above.
