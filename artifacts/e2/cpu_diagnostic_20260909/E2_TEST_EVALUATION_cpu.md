# E2 test evaluation and BamiBERT comparison

Generated `2026-09-09T01:55:51.658119+00:00`. The locked E2 selection evaluated **15/15 student checkpoints**: five tasks × seeds 42–44. Seed 42 comes from stageA and seeds 43–44 from stageB. The run is inference-only: BF16, SDPA, max length 256, batch size 32, dropout disabled by `model.eval()`, strict model-state loading, and no teacher/KD/noise computation at inference.

The NVIDIA driver was unavailable on the execution host, so the wrapper used its recorded CPU BF16 fallback. The device and precision are stored per checkpoint in the evaluation metadata and access log; no training or checkpoint changes were made.

The main baseline is E1 on the same seeds 42–44. E1 test predictions and metrics were reused from its locked artifacts; E1 was not rerun. Values are percentages and mean ± sample standard deviation across the three paired seeds.

| Task | Metric | E1 test (42–44) | E2 test (42–44) | Δ E2−E1 | BamiBERT paper | Δ E2−paper |
|---|---|---:|---:|---:|---:|---:|
| VSFC Sentiment | Accuracy | 92.23 ± 0.35 | 92.38 ± 0.17 | +0.15 | 93.86 | -1.48 |
| VSFC Sentiment | Macro F1 | 81.26 ± 0.75 | 81.30 ± 0.93 | +0.05 | 83.41 | -2.11 |
| VSFC Topic | Accuracy | 88.48 ± 0.20 | 89.29 ± 0.20 | +0.81 | 89.34 | -0.05 |
| VSFC Topic | Macro F1 | 79.16 ± 0.52 | 79.44 ± 0.38 | +0.28 | 79.90 | -0.46 |
| ViSFD | Detection micro F1 | 86.76 ± 0.21 | 87.11 ± 0.31 | +0.35 | 89.14 | -2.03 |
| ViSFD | Aspect-polarity pair micro F1 | 79.55 ± 0.43 | 80.64 ± 0.29 | +1.09 | 84.24 | -3.60 |
| UIT-ABSA Restaurant | Detection micro F1 | 84.21 ± 0.35 | 84.52 ± 0.54 | +0.30 | 88.01 | -3.49 |
| UIT-ABSA Restaurant | Aspect-polarity pair micro F1 | 68.83 ± 0.68 | 69.89 ± 0.57 | +1.05 | 74.89 | -5.00 |
| UIT-ABSA Hotel | Detection micro F1 | 78.75 ± 0.19 | 79.17 ± 0.27 | +0.42 | 79.99 | -0.82 |
| UIT-ABSA Hotel | Aspect-polarity pair micro F1 | 71.89 ± 0.40 | 72.26 ± 0.38 | +0.38 | 72.65 | -0.39 |

## Reading the result

This is the requested direct test evaluation of the validation-selected E2 branches. The E2–E1 delta measures whether the new training objective transferred to held-out test data; it does not retroactively change the branch selection. Accuracy is directly comparable by definition. The F1 rows are reported for the project benchmark, but the paper does not publish enough evaluator and averaging detail to establish exact equivalence for every F1 row.

UIT-ABSA Restaurant and Hotel use the public candidate mirror already recorded in the project. The paper target remains a useful reported reference, while exact split lineage and evaluator equivalence remain caveats.

## Provenance

- Selection manifest: `/home/kiettqa/GoiBERT/artifacts/e2/test_selection_manifest.json` (SHA-256 `313b7b3fae986bac0e659e18010410b0d0739bb6e61237fd8b3beea6b832ee38`).
- Wrapper smoke: `/home/kiettqa/GoiBERT/artifacts/e2/preflight/e2_test_smoke.json` (SHA-256 `82c5a82e3205906bcbe512098a2954a037fcd2d715046ecd44f89e36322e8490`).
- Test access log: `/home/kiettqa/GoiBERT/artifacts/e2/test_access_log.json`. Each task-seed identity has its own evaluation metadata, metrics, and stable-ID predictions under the E2 checkpoint artifact directory.
- Paper targets: `/home/kiettqa/GoiBERT/benchmarks/paper_targets.csv`; source [BamiBERT paper](https://arxiv.org/html/2607.02259).
- E1 paired metrics/predictions are referenced from `artifacts/e1/.orchestrator/attempts/` and were read without rerunning inference.
