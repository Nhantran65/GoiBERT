# E1 completed validation audit

Audit date: 2026-09-08. This is a validation-only audit; no test split was read.

## Execution integrity

- Approved records: 25 screening + 20 shortlist + 10 final = 55 records.
- 54 jobs were launched and completed; the VSFC sentiment pilot was reused once.
- Failed jobs: 0. Every selected task has exactly seeds 42, 43, 44, 45, and 46.
- All 55 selected records have 30 epochs, `include_test=false`, source hash
  `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`, a
  completed `metadata.json`, and an existing `best_checkpoint.pt`.
- Selection used validation only. VSFC uses fixed-label macro F1; ABSA uses
  end-to-end pair micro F1. ABSA detection micro F1 is supplementary.

## Learning-rate selection

The selected learning rate is the higher mean validation score over seeds 42--44
among the two shortlisted rates. Values are percentages; SD is sample SD.

| Task | Selected LR | Selection metric | Seeds 42--44 mean ± SD |
|---|---:|---|---:|
| VSFC sentiment | 1e-5 | fixed-label macro F1 | 84.4910 ± 0.0851 |
| VSFC topic | 1e-5 | fixed-label macro F1 | 80.1043 ± 0.6649 |
| ViSFD | 1e-4 | end-to-end pair micro F1 | 80.2470 ± 0.3780 |
| UIT-ABSA Restaurant | 1e-4 | end-to-end pair micro F1 | 68.9839 ± 0.1288 |
| UIT-ABSA Hotel | 1e-4 | end-to-end pair micro F1 | 72.0695 ± 0.3321 |

## Five-seed validation readout

These are the five selected-seed validation results after fixing the LR. They are
not test results and cannot establish a comparison with BamiBERT paper scores.

| Task | Pair/macro F1 mean ± SD | Detection F1 mean ± SD |
|---|---:|---:|
| VSFC sentiment (macro) | 84.2727 ± 0.3875 | — |
| VSFC topic (macro) | 79.7355 ± 0.7047 | — |
| ViSFD (pair micro) | 80.3662 ± 0.3648 | 87.0518 ± 0.3772 |
| UIT-ABSA Restaurant (pair micro) | 68.7288 ± 0.7109 | 83.4492 ± 0.7038 |
| UIT-ABSA Hotel (pair micro) | 71.9181 ± 0.3249 | 77.9134 ± 0.5396 |

## Audit issues and limits

- `state.json` reports 54 launched jobs because the pilot is represented in the
  separate `reused` list; the approved manifests correctly total 55 records.
- Older execution status/interim documents still describe the queue as running
  and should be refreshed by the root agent.
- BF16/SDPA is valid for these runs, but the existing CUDA attention warning
  means exact bitwise reproduction is not guaranteed.
- The validation scores are evidence that E1 completed and produced stable
  candidates. They do not answer whether GoiBERT beats BamiBERT until the
  selected checkpoints are evaluated once on the untouched test splits.
