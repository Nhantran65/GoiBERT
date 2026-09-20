# GoiBERT intent benchmark report

The six locked trials used seed 42, the frozen 353/77/77 grouped split, batch 16, max 15 epochs, patience 3, unweighted eight-class cross entropy, AdamW with uniform learning rate, BF16 autocast/SDPA where supported, and no test-based selection. All 507 reviewed rows were used normally, including the 82 uncertainty-marked rows. This is a single-seed pilot on 77 test examples, not a universal SOTA claim.

## Dev trial results

| Model | LR | Best epoch | Best dev macro-F1 | Dev accuracy | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| bamibert | 2e-05 | 11 | 0.502859 | 0.753247 | 2.478 |
| bamibert | 5e-05 | 15 | 0.275081 | 0.610390 | 2.478 |
| goi_dapt | 2e-05 | 9 | 0.500858 | 0.766234 | 2.478 |
| goi_dapt | 5e-05 | 11 | 0.722344 | 0.805195 | 2.478 |
| goi_student | 2e-05 | 11 | 0.626394 | 0.792208 | 0.842 |
| goi_student | 5e-05 | 13 | 0.724106 | 0.844156 | 0.842 |

The locked winner per backbone maximized dev macro-F1. Within a trial, strict improvement retained the earlier epoch on ties; across equal learning-rate scores the 2e-5 trial wins.

| Selected model | LR | Epoch | Dev macro-F1 |
|---|---:|---:|---:|
| bamibert | 2e-05 | 11 | 0.502859 |
| goi_dapt | 5e-05 | 11 | 0.722344 |
| goi_student | 5e-05 | 13 | 0.724106 |

## Frozen test results

Exactly the three dev-selected winners were evaluated on the 77-row test split. Saved prediction files were independently re-read and checked by `saved_prediction_audit.json`.

| Model | Accuracy | Macro-F1 | Prediction file |
|---|---:|---:|---|
| bamibert | 0.792208 | 0.474469 | `test_predictions_bamibert.jsonl` |
| goi_dapt | 0.896104 | 0.832981 | `test_predictions_goi_dapt.jsonl` |
| goi_student | 0.883117 | 0.820869 | `test_predictions_goi_student.jsonl` |

### Test per-intent metrics

| Model | Intent | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| bamibert | GREETING | 0.000000 | 0.000000 | 0.000000 | 2 |
| bamibert | CONFIRMATION | 0.666667 | 0.857143 | 0.750000 | 7 |
| bamibert | GRATITUDE | 0.000000 | 0.000000 | 0.000000 | 2 |
| bamibert | MENU_QUERY | 0.863636 | 0.904762 | 0.883721 | 21 |
| bamibert | COMPLAINT | 0.400000 | 0.400000 | 0.400000 | 5 |
| bamibert | ADDRESS | 1.000000 | 0.833333 | 0.909091 | 6 |
| bamibert | PHONE | 0.000000 | 0.000000 | 0.000000 | 2 |
| bamibert | OTHER | 0.805556 | 0.906250 | 0.852941 | 32 |
| goi_dapt | GREETING | 1.000000 | 0.500000 | 0.666667 | 2 |
| goi_dapt | CONFIRMATION | 0.875000 | 1.000000 | 0.933333 | 7 |
| goi_dapt | GRATITUDE | 1.000000 | 1.000000 | 1.000000 | 2 |
| goi_dapt | MENU_QUERY | 0.909091 | 0.952381 | 0.930233 | 21 |
| goi_dapt | COMPLAINT | 1.000000 | 0.400000 | 0.571429 | 5 |
| goi_dapt | ADDRESS | 1.000000 | 1.000000 | 1.000000 | 6 |
| goi_dapt | PHONE | 1.000000 | 0.500000 | 0.666667 | 2 |
| goi_dapt | OTHER | 0.857143 | 0.937500 | 0.895522 | 32 |
| goi_student | GREETING | 1.000000 | 0.500000 | 0.666667 | 2 |
| goi_student | CONFIRMATION | 0.875000 | 1.000000 | 0.933333 | 7 |
| goi_student | GRATITUDE | 1.000000 | 1.000000 | 1.000000 | 2 |
| goi_student | MENU_QUERY | 0.904762 | 0.904762 | 0.904762 | 21 |
| goi_student | COMPLAINT | 0.666667 | 0.400000 | 0.500000 | 5 |
| goi_student | ADDRESS | 1.000000 | 1.000000 | 1.000000 | 6 |
| goi_student | PHONE | 1.000000 | 0.500000 | 0.666667 | 2 |
| goi_student | OTHER | 0.857143 | 0.937500 | 0.895522 | 32 |

## Timing and resource results

Timing is model-forward only: tokenizer, padding, and host-to-device transfer are excluded. Each selected model ran 10 warmup forwards and two passes over all 77 test inputs at batch size 1 on the same NVIDIA L40S/GPU0.

| Model | Params | Mean ms | Median ms | P95 ms | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| bamibert | 102,957,320 | 4.025368 | 3.974666 | 4.342140 | 0.575 |
| goi_dapt | 102,957,320 | 4.016316 | 3.977986 | 4.182496 | 0.576 |
| goi_student | 26,548,616 | 3.466640 | 3.439778 | 3.681270 | 0.146 |

## Artifacts and caveats

- `selection.json` records the dev-only lock and checkpoint hashes; no test score influenced selection.
- `test_results.json`, `test_metrics_*.json`, and `test_predictions_*.jsonl` contain the three test evaluations.
- `timing.json` and `benchmark.csv` contain the shared-GPU latency/resource measurements.
- `saved_prediction_audit.json` recomputes metrics from saved prediction files.
- The PHONE and ADDRESS classes contain anonymized placeholders and small supports; the report should not be read as real-world generalization evidence.
- The input snapshot contains customer-only history and no assistant replies; buffered-burst source labels are preserved as provided.
