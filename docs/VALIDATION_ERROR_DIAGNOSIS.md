# Validation error diagnosis — E1 versus selected E2

Generated 2026-09-09 from locked validation predictions. Only train and validation were loaded; no test split, checkpoint, teacher cache, model, or new inference was used. The report covers seeds 42–44 for the selected E1 and E2 artifacts.

E1 uses supervised CE at LR 1e-4. E2 uses selected NC: lambda=0.5 for ViSFD and lambda=0.1 for Restaurant. Stable IDs and source gold labels match all prediction rows (1,112 ViSFD rows and 771 Restaurant rows per seed).

ViSFD serialized gold uses raw OTHERS=4 and predictions use model OTHERS=1. The diagnostic maps model 1 to raw 4 for comparison. OTHERS counts for detection and is excluded from polarity/pair metrics. Restaurant uses 0=absent, 1=negative, 2=neutral, 3=positive.

## Aggregate errors

Counts are mean aspect-cell counts over three seeds. Missed = gold present/prediction absent; spurious = gold absent/prediction present; polarity = both present but label differs. Polarity accuracy uses pair-scored cells where both are present.

| Task | System | Pair F1 | Detection F1 | Missed | Spurious | Polarity errors | Pair-scored both present | Polarity accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ViSFD | E1 | 80.247 | 86.920 | 439.3 | 502.0 | 320.0 | 3,002.7 | 89.343% |
| ViSFD | E2 NC | 81.233 | 87.605 | 441.0 | 443.7 | 307.3 | 3,014.7 | 89.806% |
| Restaurant | E1 | 68.984 | 83.201 | 171.3 | 184.7 | 150.7 | 881.7 | 82.913% |
| Restaurant | E2 NC | 69.390 | 83.926 | 168.0 | 171.0 | 153.3 | 885.0 | 82.682% |

ViSFD pair gains are consistent: +1.016, +1.066, and +0.876 points for seeds 42, 43, and 44. The main change is fewer spurious aspects. Restaurant changes are +0.043, -0.443, and +1.619 points; NC reduces spurious predictions but does not improve polarity consistently.

## Imbalance and support

Train-only slot statistics:

| Task | Total slots | Absent slots | Absent fraction |
|---|---:|---:|---:|
| ViSFD | 85,646 | 60,080 | 70.149% |
| Restaurant | 84,336 | 74,878 | 88.785% |

The following excludes ViSFD OTHERS. Validation support is summed over the three seeds; F1 is mean per-aspect F1 over the three seeds.

| Task | Class | Train support | Validation support ×3 | E1 F1 | E2 F1 |
|---|---|---:|---:|---:|---:|
| ViSFD | positive | 13,505 | 5,772 | 83.49 | 82.42 |
| ViSFD | negative | 7,464 | 3,033 | 64.72 | 64.83 |
| ViSFD | neutral | 2,903 | 1,143 | 34.90 | 34.93 |
| Restaurant | positive | 5,179 | 1,737 | 69.35 | 69.82 |
| Restaurant | negative | 1,746 | 588 | 53.69 | 54.01 |
| Restaurant | neutral | 2,533 | 834 | 56.29 | 57.65 |

Representative sparse slots; errors are totals over three seeds:

| Task / aspect | Train present | Train absent | E1 pair F1 | E2 pair F1 | E1 miss/spurious/polarity | E2 miss/spurious/polarity |
|---|---:|---:|---:|---:|---:|---:|
| ViSFD / STORAGE | 91 | 98.83% | 54.96 | 39.44 | 18 / 6 / 4 | 27 / 9 / 2 |
| ViSFD / SCREEN | 949 | 87.81% | 72.20 | 73.59 | 50 / 72 / 43 | 50 / 58 / 43 |
| ViSFD / DESIGN | 1,378 | 82.30% | 78.69 | 81.42 | 86 / 75 / 38 | 77 / 58 / 35 |
| Restaurant / DRINKS#PRICES | 170 | 97.58% | 66.70 | 66.04 | 8 / 11 / 10 | 10 / 14 / 8 |
| Restaurant / FOOD#PRICES | 401 | 94.30% | 71.54 | 72.22 | 28 / 29 / 10 | 30 / 23 / 10 |
| Restaurant / AMBIENCE#GENERAL | 815 | 88.40% | 74.63 | 78.45 | 27 / 25 / 43 | 23 / 25 / 35 |

The STORAGE collapse under NC is a warning that aggregate gains hide sparse-slot failures. It supports testing class/slot imbalance; it does not establish an encoder-capacity bottleneck.

## Length and truncation

Whitespace buckets are short 0–10, medium 11–20, and long 21+. Mean pair F1 E1→E2 is 81.82→93.80 (short), 85.68→86.50 (medium), and 79.34→80.31 (long) for ViSFD. Restaurant is 79.70→78.50, 66.09→67.88, and 64.51→64.21.

The local tokenizer was run without truncation. Maximum validation length is 241 tokens for ViSFD and 73 for Restaurant; zero rows would truncate at max length 256. Truncation is not supported as the current explanation.

## Next hypothesis

The measurable weakness is rare polarity learning under a large absent fraction. Keep encoder and head fixed and screen a conditional polarity loss:

- retain original E1 CE for detection and absent labels;
- add 0.5 times CE over gold-present polarity-bearing slots only;
- exclude ViSFD OTHERS;
- run seed 42 on ViSFD and Restaurant, same model/head, LR 1e-4, 30 epochs, effective batch 32, max length 256, BF16/SDPA;
- select by validation pair micro F1 and monitor detection, polarity errors, and sparse-slot performance.

This tests one loss hypothesis; it does not claim imbalance is causal.

## Reproducibility

Executable: [scripts/diagnose_validation_errors.py](/home/kiettqa/GoiBERT/scripts/diagnose_validation_errors.py).

Output: [artifacts/diagnostics/validation_errors.json](/home/kiettqa/GoiBERT/artifacts/diagnostics/validation_errors.json).

The script loads include_test=False, resolves locked E1/E2 paths, checks IDs and gold labels, computes train-only support and validation tokenizer lengths, and recomputes metrics from saved predictions.
