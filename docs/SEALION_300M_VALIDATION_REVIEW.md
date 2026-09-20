# SEA-LION-300M: root review validation

14/14 new fine-tuning trials completed 30 epochs. Root independently recomputed primary F1 from saved predictions for all10 SEA runs,5 Cafe runs (including reused Restaurant),15 Bami E14 grid runs. Ordered validation IDs/gold match frozen splits; recomputed metrics match artifacts within1e-12. No training or model inference performed in this review.

| Task | BamiBERT | CafeBERT | SEA-LION | SEA−Bami | SEA−Cafe |
|---|---:|---:|---:|---:|---:|
| vsfc_sentiment | 86.33 | 87.69 | 82.26 | -4.06 | -5.43 |
| vsfc_topic | 79.53 | 81.74 | 77.59 | -1.94 | -4.15 |
| visfd | 84.38 | 85.47 | 79.43 | -4.96 | -6.04 |
| absa_restaurant | 74.74 | 77.72 | 66.44 | -8.30 | -11.28 |
| absa_hotel | 73.24 | 73.34 | 71.35 | -1.89 | -2.00 |

## Interpretation

SEA-LION loses on all five validation tasks under the tested recipe. Its selected LR is5e-5 on every task. Differences are descriptive for seed42; no significance claim.

This benchmark uses a shared first-token task-head family; SEA pools BOS2, while its config advertises native mean pooling. This was an explicit design choice, not a newly discovered implementation bug. Results do not establish the best attainable SEA-LION quality. Before further investment, investigate task-head/pooling compatibility with a bounded control rather than infer that modern encoders generally fail. No pooling experiment is authorized or launched by this review.

Tuning budgets differ: Bami3LR, SEA2LR, Cafe fixed2e-5; microbatch differs in reused E14 runs. Same downstream splits, metrics and30epochs. All tables here are validation, not test. Full benchmark report/test evaluation is handled separately by executor.

Evidence: artifacts/sealion_benchmark/root_validation_audit.json, controller_state.json, per-run metadata/curves/predictions, docs/SEALION_300M_BENCHMARK_PLAN.md.
