# E5 mean-pooling screening result

E5 tested one controlled change: the original GoiTurn-S-GD student and E1 classification head were kept unchanged, while the head received the arithmetic mean of non-padding, non-special hidden states. The first-token CLS vector was used only as an explicit fallback for an input with no valid content token.

All five tasks ran with seed 42, the E1 learning-rate choice, CE loss, 30 epochs, effective batch size 32, max length 256, BF16 and SDPA. The run selected checkpoints on validation only; no test split was loaded and no additional seeds were run.

| Task | E1 validation F1 | E5 validation F1 | E5 − E1 | Best epoch | Primary metric |
|---|---:|---:|---:|---:|---|
| VSFC sentiment | 84.51 | 84.83 | +0.32 pp | 5 | macro F1 |
| VSFC topic | 79.77 | 79.48 | −0.29 pp | 7 | macro F1 |
| ViSFD | 80.07 | 79.93 | −0.14 pp | 15 | pair micro F1 |
| UIT-ABSA Restaurant | 69.02 | 67.49 | −1.53 pp | 22 | pair micro F1 |
| UIT-ABSA Hotel | 72.04 | 71.95 | −0.09 pp | 30 | pair micro F1 |

Mean pooling therefore does not improve the student as a general replacement for CLS. It gives a small screening gain on VSFC sentiment, is essentially flat on Hotel and ViSFD, and is harmful on Topic and Restaurant. The result does not justify multi-seed confirmation or test evaluation at this stage.

The preflight passed: the adapter excluded padding and tokenizer special IDs, handled the empty-content fallback without NaNs, preserved E1 state keys and parameter counts (26,546,691 for the sentiment classifier), and produced finite BF16 forward/backward values. The saved-row audit also passed for all five tasks: validation IDs were in order, gold labels matched semantically, and recomputed selection F1 matched the saved value.

Provenance: E1 dependency hash `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`; E5 source hash `d63a5dafbeaf4d9c07305f580dc41f5924e71172949dc1d15a5384a679f75c29`. Full metrics and predictions are in [`artifacts/e5/summary.json`](../artifacts/e5/summary.json), with the saved-row audit in [`artifacts/e5/postrun_audit.json`](../artifacts/e5/postrun_audit.json).

Metadata wording correction: the field `pooling_fallback_recording` in the per-trial metadata was broader than the evidence supports. The preflight covered synthetic inputs and two short examples; it did not measure fallback frequency over the full validation splits. The correction is recorded in [`artifacts/e5/metadata_correction.json`](../artifacts/e5/metadata_correction.json); training artifacts and source code were not changed after launch.

## Root decision

Keep CLS as the default. E5 sentiment84.8285 is only0.0563 points above the prior NC seed42 score84.7722; this is a small single-seed candidate, not an established win. No additional seeds, test evaluation, or NC combination are launched. The four other task scores do not support switching their pooling. All comparisons concern validation and this fixed CE recipe.
