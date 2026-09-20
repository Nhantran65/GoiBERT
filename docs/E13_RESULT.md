# E13 Result: Centered pair-value residual versus uncentered control

E13 trained centered and uncentered block-6 pair-value residual variants on all five locked tasks with seed 42, 30 epochs, effective/micro batch 32, max length 256, BF16 SDPA, NC loss and E7 layer-wise AdamW decay. Validation selected each checkpoint independently; test was run for all ten locked checkpoints after manifest creation.

The numeric gold labels and prediction integers are unchanged. Semantic report metadata uses the corrected polarity overlay: Restaurant/Hotel and ViSFD polarity 0 absent, 1 positive, 2 negative, 3 neutral; ViSFD OTHERS raw code 4 is model-binary and pair-masked.

Artifacts: `artifacts/e13/.orchestrator/preflight.json`, `runtime_profile.json`, `validation_summary.json/.csv`, `locked_checkpoints.json`, `test_summary.json/.csv`, and `test_access_log.json`.

The locked manifest binds the model/test source hash `42143f183085e54b69839b7f596ee46e973f2eb24c1945398d946cc86b9db4ba`. A report-only writer update was made after test completion; it did not change the model constructor, optimizer, data loader, evaluator, or any checkpoint.

## Validation

| Task | E8 | Centered | Uncentered | Centered−E8 (pp) | Centered−Control (pp) |
|---|---:|---:|---:|---:|---:|
| vsfc_sentiment | 0.851604 | 0.852047 | 0.851454 | +0.044 | +0.059 |
| vsfc_topic | 0.803678 | 0.803526 | 0.804173 | -0.015 | -0.065 |
| visfd | 0.824421 | 0.824421 | 0.824474 | +0.000 | -0.005 |
| absa_restaurant | 0.712766 | 0.715596 | 0.713597 | +0.283 | +0.200 |
| absa_hotel | 0.730491 | 0.729100 | 0.727711 | -0.139 | +0.139 |

Best epochs are recorded in `validation_summary.json`; the table reports the validation-selected centered and uncentered scores. The E13 investment gate did not pass: centered has fewer than two gains of at least +0.3 pp versus E8. The centering subcheck is positive on 3/5 tasks and has no drop larger than 0.3 pp versus the uncentered control, but this is a one-seed result.


Selection F1 is reported internally as a fraction in [0,1]; percentage-point deltas are computed as 100 × (new − baseline).

## Test

| Task | E8 test | Centered | Uncentered | Centered−E8 (pp) |
|---|---:|---:|---:|---:|
| vsfc_sentiment | 0.812141 | 0.812141 | 0.818314 | +0.000 |
| vsfc_topic | 0.793788 | 0.793078 | 0.792319 | -0.071 |
| visfd | 0.817502 | 0.816788 | 0.816998 | -0.071 |
| absa_restaurant | 0.715203 | 0.720233 | 0.715867 | +0.503 |
| absa_hotel | 0.726164 | 0.725434 | 0.725229 | -0.073 |

All ten locked checkpoints were evaluated. E8 test seed42 and paper BamiBERT comparisons preserve their split/evaluator lineage; test scores were not used for checkpoint selection.
