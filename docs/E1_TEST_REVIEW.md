# E1 test evaluation review

This is a read-only code-path review before the locked test inference. No
model evaluation was run by this review.

## Required checks for the test agent

- Use the immutable `E1-test-gate-v1` manifest and exactly 25 rows: each of
  the five task IDs with seeds 42, 43, 44, 45, and 46. Sort any aggregate by
  `(task_id, seed)` because `run_test_records()` collects futures in completion
  order.
- Pass `--precision bf16 --attention-implementation sdpa` explicitly. The
  runner checks the manifest/source/checkpoint hashes, but it does not verify
  that the CLI precision or attention implementation matches the selected
  row's metadata. The default CLI precision is fp32, so omitting the flag can
  silently violate the locked recipe.
- Build each manifest row's `data_provenance` with
  `load_task(..., include_test=True)`. Validation artifacts were created with
  `include_test=false` and therefore contain only train/dev source files;
  `evaluate_locked_checkpoint()` recomputes provenance after loading the test
  split and compares it to the manifest row. Copying validation provenance
  verbatim will fail the guard.
- Preserve ViSFD raw-label semantics when recomputing metrics from saved
  predictions. Test rows store raw gold labels (`OTHERS=4`) while saved model
  predictions use the binary model-space OTHERS code (`0/1`). Recompute via
  `metrics.evaluate_task(task_id="visfd", ...)`, which maps model `1` back to
  raw code `4` and excludes OTHERS from pair classification while retaining it
  for detection. Do not flatten the matrix or score OTHERS as polarity.
- Verify each prediction file has the expected test row count and stable IDs:
  VSFC 3,166; ViSFD 2,224; Restaurant 1,938; Hotel 2,030. Recompute the
  reported `selection_f1` from the saved rows and require exact agreement (up
  to JSON floating-point serialization) with `test_metrics.json`.
- Aggregate mean and sample standard deviation (`ddof=1`) over the five seed
  values per task. Include each per-seed score and the exact metric name in the
  readout; do not infer significance against a single BamiBERT paper score.

## Paper comparison caveat

The locked evaluator reports fixed-label macro F1 for VSFC and explicit
detection micro F1 plus end-to-end aspect-polarity pair micro F1 for ABSA.
The BamiBERT paper does not publish enough evaluator/averaging detail to
certify exact equivalence, and the UIT-ABSA files are a public third-party
mirror. A comparison table is useful, but any difference should be labelled
as a measured gap under the stated protocol rather than a definitive win/loss.

## Implementation observations

`evaluate_locked_checkpoint()` refuses pre-existing test output files, checks
the locked source/checkpoint/metadata/backbone/tokenizer hashes, and chooses
max length using train-only statistics before reading the test loader. The
test function has no optimizer or training path. These guards should remain
untouched.

## Independent audit of completed test artifacts

Audit status: **PASS**. This audit read the 25 saved prediction files and
recomputed metrics with `goibert.metrics.evaluate_task`; it performed no model
inference. The locked manifest contains exactly 25 unique `(task, seed)` rows,
all five task IDs, and seeds 42--46 for each task. The current source hash
matches `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.
Every artifact has matching metadata/checkpoint SHA256 values, BF16/SDPA
runtime metadata, complete test provenance including test files, and stable
IDs with the expected row count. Recomputed metrics match each
`test_metrics.json` and the aggregate values in `artifacts/e1/test_readout.json`.

The row counts checked were VSFC sentiment/topic 3,166 each, ViSFD 2,224,
Restaurant 1,938, and Hotel 2,030. For ViSFD, the audit compared raw gold
labels (including OTHERS code 4) with the loader and scored saved model-space
predictions through the explicit OTHERS conversion path.

The ten aggregate rows below are percentages; SD is the sample standard
deviation over seeds 42--46. The per-seed order in parentheses is 42, 43, 44,
45, 46.

| Task | Metric | Mean ± SD | Per-seed values |
|---|---|---:|---|
| VSFC sentiment | Accuracy | 92.21099 ± 0.28914 | 92.54580, 91.85092, 92.29311, 91.97726, 92.38787 |
| VSFC sentiment | Fixed-label macro F1 | 80.59974 ± 1.04267 | 81.94422, 80.45737, 81.36491, 79.62346, 79.60873 |
| VSFC topic | Accuracy | 88.49021 ± 0.24629 | 88.59760, 88.59760, 88.25016, 88.78711, 88.21857 |
| VSFC topic | Fixed-label macro F1 | 78.62203 ± 0.82387 | 79.73685, 78.74008, 79.00382, 77.88157, 77.74782 |
| ViSFD | Detection micro F1 | 86.85838 ± 0.20016 | 86.69835, 86.59150, 86.99925, 86.96487, 87.03793 |
| ViSFD | End-to-end pair micro F1 | 79.83323 ± 0.50044 | 79.08077, 79.64887, 79.93257, 80.10875, 80.39519 |
| UIT-ABSA Restaurant | Detection micro F1 | 84.31633 ± 0.29949 | 84.02344, 84.61685, 83.99318, 84.59341, 84.35478 |
| UIT-ABSA Restaurant | End-to-end pair micro F1 | 69.08006 ± 0.59868 | 69.38930, 69.02453, 68.08108, 69.62483, 69.28055 |
| UIT-ABSA Hotel | Detection micro F1 | 78.63631 ± 0.24854 | 78.96794, 78.62764, 78.65029, 78.26774, 78.66792 |
| UIT-ABSA Hotel | End-to-end pair micro F1 | 71.78450 ± 0.31705 | 72.33776, 71.73434, 71.59073, 71.56630, 71.69337 |

Cross-check artifacts: `test_selection_manifest.json` SHA256
`bd8e8855fc3507c3c88ed8797671ca7ed9f1edc246e0363ab17e398ea98c82c4`,
`test_readout.json` SHA256
`2dfd61761fe5fe19f5a206a82aafc17524e703abb035b92db61af46348bfcb17`, and
the executed test index/access log are recorded in the test report.
