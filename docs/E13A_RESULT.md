# E13A reviewed binding diagnostic

E13A ran the 30 root-reviewed pairs from [the binding spec](E13A_BINDING_DIAGNOSTIC_SPEC.md) on the saved E8 CLS, E11 signed, and E11 convex seed-42 checkpoints. It performed inference only: no training, no test prediction/metric access, and no E1–E11 source or checkpoint mutation.

The runtime was CPU FP32 because CUDA was unavailable (`nvidia-smi` could not communicate with a driver). All three model families used the same CPU FP32 execution path, so these numbers are diagnostic comparisons and are not benchmark BF16 results. Full logits, raw texts, corrected numeric expectations, predictions, checkpoint hashes, and per-pair scores are in [artifacts/e13a/results.json](../artifacts/e13a/results.json). The isolated runner is [run_e13a_binding_diagnostic.py](../scripts/run_e13a_binding_diagnostic.py).

There are 60 occurrences and 56 unique texts because the spec intentionally repeats four control/swap texts. No occurrence matched the loaded train or validation rows. Expectations were explicit: VSFC sentiment `0/1/2 = negative/neutral/positive`, VSFC topic `0/1/2/3 = lecturer/training_program/facility/others`, and Restaurant/Hotel `0/1/2/3 = absent/positive/negative/neutral`. ViSFD OTHERS was stored after raw-code conversion and excluded from scoring.

## Correct slots and pair failures

`correct/expected` counts all explicitly expected slots on both sides. `pair-both-correct` requires every expected slot on A and B to be correct; unspecified slots are never assigned absent.

| Task | E8 correct | E8 pair-both | E11 signed correct | signed pair-both | E11 convex correct | convex pair-both |
|---|---:|---:|---:|---:|---:|---:|
| VSFC sentiment | 10/12 | 4/6 | 10/12 | 4/6 | 10/12 | 4/6 |
| VSFC topic | 12/12 | 6/6 | 12/12 | 6/6 | 12/12 | 6/6 |
| ViSFD | 15/19 | 4/6 | 13/19 | 4/6 | 13/19 | 4/6 |
| Restaurant | 16/17 | 5/6 | 16/17 | 5/6 | 16/17 | 5/6 |
| Hotel | 12/17 | 3/6 | 11/17 | 3/6 | 12/17 | 3/6 |

Per-pair failures are stable across the three checkpoints:

| Task | E8 failures | Signed failures | Convex failures |
|---|---|---|---|
| VSFC sentiment | S3, S4 | S3, S4 | S3, S4 |
| VSFC topic | none | none | none |
| ViSFD | V3, V5 | V3, V5 | V3, V5 |
| Restaurant | R1 | R1 | R1 |
| Hotel | H1, H2, H6 | H1, H2, H6 | H1, H2, H6 |

The repeated failures point to the reviewed target/category cases: accessory versus phone battery (V3/V5), food/service binding (R1), and room/service or room-price target binding (H1/H2/H6). They do not establish prevalence or a causal neural mechanism.

## Candidate versus E8

The regression denominator is the number of E8-correct expected slots, and the numerator is `E8 correct AND candidate wrong`. It does not count every model disagreement. Corrections are `E8 wrong AND candidate right`.

| Task | Signed regression | Signed corrections | Convex regression | Convex corrections |
|---|---:|---:|---:|---:|
| VSFC sentiment | 0/10 | 0 | 0/10 | 0 |
| VSFC topic | 0/12 | 0 | 0/12 | 0 |
| ViSFD | 2/15 | 0 | 2/15 | 0 |
| Restaurant | 0/16 | 0 | 0/16 | 0 |
| Hotel | 1/12 | 0 | 0/12 | 0 |

For the label-preserving shared slots, neither E11 variant produced a correct-to-wrong false flip or a wrong-to-correct flip in this sample. Expected-change shared-slot transition accuracy was 0.50 for Sentiment, 1.00 for Topic, 0.60 for ViSFD, and 0.667 for both Restaurant and Hotel for all three models.

## Checkpoint provenance

The runner strictly restored each `best_checkpoint.pt`; saved checkpoint epoch/global-step records were:

| Model | Sentiment | Topic | ViSFD | Restaurant | Hotel |
|---|---|---|---|---|---|
| E8 | 21/7518 | 25/8950 | 28/6832 | 25/5500 | 26/5850 |
| E11 signed | 8/2864 | 26/9308 | 28/6832 | 28/6160 | 27/6075 |
| E11 convex | 8/2864 | 29/10382 | 25/6100 | 28/6160 | 28/6300 |

Torch was `2.6.0+cu124` and Transformers `4.49.0`; source/checkpoint/tokenizer hashes are recorded in `results.json`. The E11 signed/convex gate values remain in the checkpoint state and full model outputs are preserved for review.

The diagnostic lowers priority for a shared attention or character branch as the immediate explanation: E11 never corrects an E8-wrong reviewed slot and introduces 2 ViSFD plus 1 Hotel regressions in the E8-correct denominator. A narrower target/category binding or explicit paired-consistency direction remains a hypothesis for root review, with this sample treated as directional evidence only.

