# mDeBERTa-v3-base five-task pilot

Validation-only screen: seed 42, LR 2e-5, 30 epochs, effective batch 32, max length 256. Test was not opened.

| Task | Bami E14 val F1 (%) | Cafe val F1 (%) | mDeBERTa val F1 (%) | Δ Bami (pp) | Δ Cafe (pp) | Best epoch | Runtime (min) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| VSFC sentiment | 86.3252 | 87.6924 | 85.9183 | -0.4069 | -1.7741 | 4 | 36.0 | completed |
| VSFC topic | 79.5328 | 81.7446 | 80.0796 | 0.5468 | -1.6650 | 12 | 32.9 | completed |
| ViSFD | 84.3834 | 85.4665 | 79.8485 | -4.5349 | -5.6180 | 29 | 70.0 | completed |
| ABSA restaurant | 74.7418 | 77.7197 | 70.5598 | -4.1820 | -7.1599 | 22 | 35.2 | completed |
| ABSA hotel | 73.2362 | 73.3438 | 70.8249 | -2.4113 | -2.5189 | 28 | 43.1 | completed |

The model uses the native DebertaV2 ContextPooler and classifier: first hidden state, pooler dropout, native pooler dense/activation, classifier dropout, then task logits. ABSA uses independent per-slot CE; ViSFD `OTHERS` is binary and excluded from pair scoring. Encoder weights load as FP32 master parameters with BF16 autocast and native eager attention. ABSA prediction rows retain the frozen numeric labels; their corrected semantics are `0=absent, 1=positive, 2=negative, 3=neutral` (see E12_LABEL_MAPPING_AUDIT.md). The evaluator scores exact numeric pair agreement. The original report accidentally repeated stale semantic names; this text-only correction does not change training, labels, checkpoints or metrics.

## Root review — 2026-09-18

Independently recomputed all five validation metrics from saved predictions; IDs/order and numeric gold match the frozen validation splits, and every F1 matches saved results within1e-12. Audit: `artifacts/mdeberta_pilot/root_validation_audit.json`. All five runs completed30epochs; no new model inference/training/test was performed in this review.

Verdict: the fixed-LR pilot improves Topic by0.55pp over Bami, loses to Bami on the other four tasks, and loses to Cafe on all five. No broad improvement signal under this recipe. One seed and oneLR do not measure best attainable mDeBERTa performance. Reused Bami baseline had3LR selection; Cafe used fixed2e-5. No additional runs launched.
