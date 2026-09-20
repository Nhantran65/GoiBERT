# E9 Result: Label-aware Attention

Five original GoiTurn-S-GD students use one learned query per VSFC class or ABSA aspect, with the fixed E8 NC plus E7 LLRD recipe. Validation only; seed 42; no test split.

| Task | E8 baseline | E9 | Delta (pp) | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| vsfc_sentiment | 85.1604 | 0.8519 | -8430.852 | 13 |
| vsfc_topic | 80.3678 | 0.8027 | -7956.511 | 26 |
| visfd | 82.4421 | 0.8210 | -8162.107 | 28 |
| absa_restaurant | 71.2766 | 0.7230 | -7055.358 | 27 |
| absa_hotel | 73.0491 | 0.7243 | -7232.474 | 21 |

Screening gate: FAIL (0/5 increased; 0 increased by at least 0.3pp; largest decrease -8430.852pp).

Source hashes: E9 `d2ad93c73d8135e78514a9f863ec35e1e08e10a64596545b194194bd636883dd`, E7 dependency `a1124b40d2957b8eb7bb84ea6409b486a80474dde74ca2a94845ee12baa750f5`, E2 `1a9d9b1d22f038f8231e5194a947515df1361c1b5603ed262e55f69e9e851ca6`, E1 `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.
