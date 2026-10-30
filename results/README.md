# Results

Raw records behind the GoiBERT, BamiBERT and ablation numbers in the paper and the main README. Other baselines (CafeBERT, XLM-R, ViSoBERT, PhoBERT-v2) are reported in the paper.

| File | Content |
|---|---|
| `test_runs.csv` | Test metrics of every run: 72 main runs (GoiBERT-base, GoiBERT-large, BamiBERT; 8 tasks; seeds 42, 43, 44) and 54 ablation runs. Columns: setting, learning rate, selected epoch, metrics, and the SHA-256 of the evaluated checkpoint. |
| `test_protocol.json` | Test-access record: when each manifest was locked, the SHA-256 of the locked manifest and of the test files, and the time each checkpoint was evaluated. |
| `efficiency.json` | Raw latency, throughput and peak-memory measurements behind the efficiency table. |

Settings in `test_runs.csv`:

| `setting` | Meaning |
|---|---|
| `full` | GoiBERT recipe: general distillation, CafeBERT out-of-fold KD, LLRD |
| `plain_ce` | BamiBERT baseline: cross-entropy, plain AdamW |
| `gd_task_kd` | Ablation: general distillation and KD, plain AdamW |
| `gd_ce` | Ablation: general distillation, cross-entropy, plain AdamW |
| `ce_only` | Ablation: PCA-initialised student without general distillation, cross-entropy, plain AdamW |

**Protocol.**
- Learning rates and epochs were selected on validation data.
- Each set of checkpoints was frozen in a manifest before the test split was read.
- Each checkpoint was then evaluated on test once.
- Metrics are fractions (multiply by 100 for the percentages in the tables).
- Primary metrics: macro-F1 for VSFC, ViSpamReviews and ViNLI; aspect–sentiment pair micro-F1 for the ABSA tasks; strict entity-level F1 for PhoNER.
