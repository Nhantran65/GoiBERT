# E16 validation results

Protocol: `E16-BamiBERT-original-identity-gated-ffn-v1`

All 15 validation trials completed with fractions in [0, 1]. Display deltas use percentage points. E14 values are historical context; E16 baseline is the primary reference.

The audit observed source hashes `ec68892d5381c67022476a0a3f9a2adfe3c2536132bd8f73267b2ef7e418ee1e` for the initial baseline trials and `8f37d926c3cdf8dc6ad44df53a79e012010637511099f4d6039ddc37aaf18faa` after the postrun checkpoint-validation repair; the training recipe and model path were unchanged, and all 15 checkpoints pass strict constructor roundtrip.

| Arm | Task | Best epoch | Validation selection F1 | Delta vs E16 baseline (pp) | Delta vs E14 historical (pp) |
|---|---|---:|---:|---:|---:|
| baseline | vsfc_sentiment | 5 | 0.85193157 | 0.0000 | -1.1321 |
| baseline | vsfc_topic | 8 | 0.79547631 | 0.0000 | 0.0148 |
| baseline | visfd | 23 | 0.83970829 | 0.0000 | -0.4126 |
| baseline | absa_restaurant | 21 | 0.74484053 | 0.0000 | -0.2577 |
| baseline | absa_hotel | 27 | 0.73035439 | 0.0000 | -0.2008 |
| attention_update_gate | vsfc_sentiment | 24 | 0.84560684 | -0.6325 | -1.7645 |
| attention_update_gate | vsfc_topic | 23 | 0.79591343 | 0.0437 | 0.0585 |
| attention_update_gate | visfd | 24 | 0.83813218 | -0.1576 | -0.5702 |
| attention_update_gate | absa_restaurant | 27 | 0.74015009 | -0.4690 | -0.7268 |
| attention_update_gate | absa_hotel | 26 | 0.73042808 | 0.0074 | -0.1934 |
| ffn_input_gate | vsfc_sentiment | 4 | 0.85442580 | 0.2494 | -0.8826 |
| ffn_input_gate | vsfc_topic | 5 | 0.80587507 | 1.0399 | 1.0547 |
| ffn_input_gate | visfd | 26 | 0.84038604 | 0.0678 | -0.3448 |
| ffn_input_gate | absa_restaurant | 30 | 0.74098361 | -0.3857 | -0.6434 |
| ffn_input_gate | absa_hotel | 24 | 0.73636017 | 0.6006 | 0.3998 |

## Predeclared gate facts

```json
{
  "predeclared_attention_vs_baseline": {
    "gain_tasks": 2,
    "gain_ge_0_3pp_tasks": 0,
    "drop_worse_than_0_3pp_tasks": 2,
    "passes": false
  },
  "predeclared_attention_vs_ffn": {
    "win_tasks": 0,
    "loss_worse_than_0_3pp_tasks": 3,
    "passes": false
  },
  "test_review_eligible": false
}
```

Test split access: false.
