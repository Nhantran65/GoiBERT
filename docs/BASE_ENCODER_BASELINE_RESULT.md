# Baseline mmBERT-base / XLM-R-base trên 5 task — kết quả (validation seed 42)

Plan `docs/BASE_ENCODER_BASELINE_PLAN.md`; tạo 2026-09-19T06:49:20Z; recipe E14 (3 LR, 30 epoch, head CLS E14, eff. batch 32, bf16); không test. Artifacts `artifacts/base_encoder_baseline/`. F1×100; ABSA = pair micro F1, VSFC = fixed-label macro F1.

## mmbert_base — `jhu-clsp/mmBERT-base @ c5955035435e` (ModernBertModel, 22L/768, 110.3M không-embedding / 306.9M encoder, vocab 256000)

| Task | LR 1e-5 | LR 2e-5 | LR 5e-5 | Chọn (LR, epoch) | Bami E14 (LR) | Cafe 2e-5 | Δ vs Bami | Δ vs Cafe |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Restaurant | 71.67@29 | 72.48@20 | 75.63@24 | **75.63** (5e-05, 24) | 74.74 (5e-05) | 77.72 | +0.88 | -2.09 |
| Topic | 80.58@3 | 81.18@11 | 79.53@2 | **81.18** (2e-05, 11) | 79.53 (1e-05) | 81.74 | +1.65 | -0.56 |
| ViSFD | 77.71@21 | 82.42@27 | 83.64@16 | **83.64** (5e-05, 16) | 84.38 (5e-05) | 85.47 | -0.74 | -1.83 |
| Hotel | 70.52@16 | 72.32@15 | 73.70@12 | **73.70** (5e-05, 12) | 73.24 (5e-05) | 73.34 | +0.46 | +0.35 |
| Sentiment | 86.70@15 | 85.10@19 | 84.39@4 | **86.70** (1e-05, 15) | 86.33 (2e-05) | 87.69 | +0.37 | -1.00 |

Gate (đăng ký trước): ≥ Bami−0,5 trên ≥4/5: **True**; Restaurant ≥ Bami−0,5: **True**; Topic hoặc ViSFD ≥ +0,5: **True** → **PASS**. Task sát biên (|Δ| ≤ 0,5): Hotel, Sentiment.

## xlmr_base — `FacebookAI/xlm-roberta-base @ e73636d4f797` (XLMRobertaModel, 12L/768, 85.5M không-embedding / 277.5M encoder, vocab 250002)

| Task | LR 1e-5 | LR 2e-5 | LR 5e-5 | Chọn (LR, epoch) | Bami E14 (LR) | Cafe 2e-5 | Δ vs Bami | Δ vs Cafe |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Restaurant | 66.96@24 | 71.09@21 | 73.61@28 | **73.61** (5e-05, 28) | 74.74 (5e-05) | 77.72 | -1.13 | -4.11 |
| Topic | 81.89@17 | 81.22@15 | 80.34@23 | **81.89** (1e-05, 17) | 79.53 (1e-05) | 81.74 | +2.35 | +0.14 |
| ViSFD | 79.96@28 | 82.20@24 | 83.82@28 | **83.82** (5e-05, 28) | 84.38 (5e-05) | 85.47 | -0.56 | -1.65 |
| Hotel | 67.76@27 | 70.84@25 | 73.05@25 | **73.05** (5e-05, 25) | 73.24 (5e-05) | 73.34 | -0.18 | -0.29 |
| Sentiment | 86.34@16 | 85.83@8 | 83.77@22 | **86.34** (1e-05, 16) | 86.33 (2e-05) | 87.69 | +0.02 | -1.35 |

Gate (đăng ký trước): ≥ Bami−0,5 trên ≥4/5: **False**; Restaurant ≥ Bami−0,5: **False**; Topic hoặc ViSFD ≥ +0,5: **True** → **FAIL**. Task sát biên (|Δ| ≤ 0,5): Hotel, Sentiment.

Ghi chú: Bami E14 chọn LR theo cùng grid; CafeBERT chỉ chạy LR 2e-5 (Restaurant reuse từ CDBK). Một seed; |Δ| < 1 điểm nằm trong nhiễu run-to-run đã quan sát. Mọi run có reload/prediction/ID checks trong `metadata.json`.
