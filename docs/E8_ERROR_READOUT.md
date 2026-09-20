# E8 validation error readout

This readout was refreshed from the saved validation predictions for E8, the
selected E2 noise-consistency student, and the selected E2 teacher. The
analysis uses `include_test=False`, the same validation IDs and gold labels,
and does not load a checkpoint or a test artifact. The machine-readable
output is `artifacts/e8/error_analysis/e8_validation_error_analysis.json`.

The new E8 seeds 43 and 44 are admitted by `e8_ready` only when the artifact
has completed metadata, `include_test=false`, the validation prediction file,
and exactly 30 records in `epoch_curves.epochs`. All six E8 task/seed artifacts
pass this gate; their metadata also records `epochs=30`. E2 NC comparison
artifacts for seeds 42, 43, and 44 pass the validation-only row and metadata
checks as well.

## E8 versus E2 NC, same seed

`M/S/P` means pair-scored missed aspects, pair-scored spurious aspects, and
wrong polarity. Pair FN = M + P and pair FP = S + P. All values below exclude
ViSFD `OTHERS` from pair counts.

| Task | Seed | E8 pair F1 | E2 NC pair F1 | Delta (pp) | E8 M/S/P; delta E8-NC M/S/P | E8 pair FN/FP |
|---|---:|---:|---:|---:|---|---:|
| ViSFD | 42 | 82.44 | 81.09 | +1.35 | 323/292/272; +32/−56/−41 | 595/564 |
| ViSFD | 43 | 82.17 | 81.05 | +1.11 | 306/280/296; +5/−66/−13 | 602/576 |
| ViSFD | 44 | 82.13 | 81.56 | +0.57 | 308/285/294; −4/−26/−6 | 602/579 |
| Restaurant | 42 | 71.28 | 69.06 | +2.22 | 176/138/140; +16/−33/−22 | 316/278 |
| Restaurant | 43 | 71.31 | 68.40 | +2.91 | 166/155/140; −2/−24/−21 | 306/295 |
| Restaurant | 44 | 73.18 | 70.71 | +2.47 | 176/147/117; 0/−16/−20 | 293/264 |

The repeated result is therefore specific: pair F1 improves 3/3 seeds, and
spurious and polarity errors decrease 3/3 seeds. Missed aspects decrease only
on ViSFD seed 44 and Restaurant seed 43; Restaurant seed 44 is unchanged.
Pair FN is larger than pair FP
for every E8 run, so missed-aspect recall remains the larger error side even
after the precision-side gains.

ViSFD detection totals include `OTHERS`. Its raw code is normalized from model
0/1 to semantic presence 0/4, then excluded from pair and conditional
polarity counts. E8 `OTHERS` missed/spurious counts are 138/99, 139/96, and
136/87 for seeds 42/43/44; these are not included in the M/S/P columns above.

## Seed-42 teacher gap

The teacher is the selected E2 DAPT fine-tuned checkpoint for seed 42, not the
original BamiBERT model. On the same validation IDs, E8 has 129 fewer exact
pair hits on ViSFD and 51 fewer on Restaurant. The pair-FN gap decomposes as
follows:

| Task | Teacher pair FN | E8 pair FN | E8−teacher FN | Missed pair aspects | Wrong polarity | Teacher/E8 pair precision | Teacher/E8 pair recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViSFD | 466 | 595 | +129 | +121 | +8 | 82.82% / 82.83% | 85.95% / 82.06% |
| Restaurant | 265 | 316 | +51 | +40 | +11 | 74.41% / 72.61% | 74.83% / 69.99% |

The paired positive-gold transitions support the same direction: teacher-hit /
E8-miss versus teacher-miss / E8-hit is 241/112 on ViSFD and 120/69 on
Restaurant. This is a descriptive comparison of saved argmax rows, not a
causal attribution.

The largest pair-FN gap contributors are below. Train presence support and the
rarest polarity support are computed from the full train split only; they are
included to distinguish broad-support slots from genuinely sparse ones.

| Task | Aspect | Train presence support; rarest class support | E8−teacher missed | E8−teacher wrong polarity | E8−teacher pair FN |
|---|---|---:|---:|---:|---:|
| ViSFD | SER&ACC | 1,995; neutral 107 | +33 | 0 | +33 |
| ViSFD | BATTERY | 3,604; neutral 349 | +6 | +19 | +25 |
| ViSFD | PERFORMANCE | 4,140; neutral 391 | +14 | +7 | +21 |
| ViSFD | GENERAL | 4,866; neutral 290 | +16 | −2 | +14 |
| Restaurant | FOOD#QUALITY | 1,995; positive 309 | +10 | +4 | +14 |
| Restaurant | FOOD#STYLE&OPTIONS | 1,571; negative 234 | +1 | +12 | +13 |
| Restaurant | SERVICE#GENERAL | 1,093; positive 92 | +8 | +1 | +9 |
| Restaurant | AMBIENCE#GENERAL | 815; positive 126 | +1 | +6 | +7 |

These top four slots account for 93/129 (72%) of the ViSFD teacher pair-FN
gap and 43/51 (84%) of the Restaurant gap. They are mostly medium- and
high-support slots rather than the rarest slots: for reference, the rarest
train presence support is 91 on ViSFD (`STORAGE`) and 170 on Restaurant
(`DRINKS#PRICES`), while those slots contribute only +5 and +1 pair-FN to the
seed-42 teacher gap. `STORAGE` does have seed-sensitive pair F1 because of its
small support, but it does not explain the aggregate gap.

## Interpretation for the next validation decision

The dominant observed error pattern is aspect-presence recall: E8 reliably
removes spurious predictions and many polarity errors, but it does not
reliably reduce missed aspects, and the teacher gap is concentrated in missed
aspects on broad-support slots (ViSFD SER&ACC/GENERAL/PERFORMANCE and
Restaurant FOOD#QUALITY/SERVICE). A recall-targeted validation change is a
supported next hypothesis to evaluate against this decomposition; the rows do
not establish a neural bottleneck or causality. No new experiment,
calibration, or logit extraction is launched by this readout.
