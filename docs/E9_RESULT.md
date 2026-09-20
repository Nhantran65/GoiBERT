# E9 Result: Label-aware Attention

Five original GoiTurn-S-GD students use one learned query per VSFC class or ABSA aspect, with the fixed E8 NC plus E7 LLRD recipe. Validation only; seed 42; no test split.

F1 is stored as a fraction in artifacts and displayed as percent below. Delta is `(E9 fraction - E8 fraction) * 100` percentage points.

| Task | E8 baseline (%) | E9 (%) | Delta (pp) | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| vsfc_sentiment | 85.1604 | 85.1923 | +0.032 | 13 |
| vsfc_topic | 80.3678 | 80.2676 | -0.100 | 26 |
| visfd | 82.4421 | 82.0986 | -0.343 | 28 |
| absa_restaurant | 71.2766 | 72.3018 | +1.025 | 27 |
| absa_hotel | 73.0491 | 72.4333 | -0.616 | 21 |

Screening gate: FAIL (2/5 increased; 1 increased by at least 0.3pp; largest decrease -0.616pp).

Every task has completed 30 epochs. Validation IDs/order and saved-vs-recomputed E9 F1 were verified; E8 baselines came from the actual E8 best-validation metric files.
