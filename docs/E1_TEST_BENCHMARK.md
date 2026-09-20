# E1 — Benchmark test so với BamiBERT

Đã đánh giá 25 checkpoint được chọn bằng validation: 5 task × seeds42–46. Điểm trên thang0–100; GoiBERT báo mean ± sample std (ddof=1). Không retrain, không chọn lại checkpoint theo test.

GoiBERT E1 là `goiturn-s-gd` fine-tune supervised-only, toàn bộ encoder và head, không task KD/NC hoặc thay kiến trúc. VSFC chọn LR1e-5; ViSFD/Restaurant/Hotel chọn LR1e-4 bằng validation.

| Benchmark | Metric | BamiBERT paper | GoiBERT E1 test, 5 seeds |
|---|---|---:|---:|
| VSFC Sentiment | Accuracy | 93.86 | 92.21 ± 0.29 |
| VSFC Sentiment | F1 (macro GoiBERT) | 83.41 | 80.60 ± 1.04 |
| VSFC Topic | Accuracy | 89.34 | 88.49 ± 0.25 |
| VSFC Topic | F1 (macro GoiBERT) | 79.90 | 78.62 ± 0.82 |
| ViSFD | F1 Detection (micro GoiBERT) | 89.14 | 86.86 ± 0.20 |
| ViSFD | F1 Classification (pair micro GoiBERT) | 84.24 | 79.83 ± 0.50 |
| UIT-ABSA Restaurant | F1 Detection (micro GoiBERT) | 88.01 | 84.32 ± 0.30 |
| UIT-ABSA Restaurant | F1 Classification (pair micro GoiBERT) | 74.89 | 69.08 ± 0.60 |
| UIT-ABSA Hotel | F1 Detection (micro GoiBERT) | 79.99 | 78.64 ± 0.25 |
| UIT-ABSA Hotel | F1 Classification (pair micro GoiBERT) | 72.65 | 71.78 ± 0.32 |

Mốc BamiBERT: [Table 2, arXiv:2607.02259v1](https://arxiv.org/html/2607.02259#S3). Đây là điểm được báo cáo của tác giả, không phải BamiBERT được chạy lại trong harness này.

## Diễn giải và giới hạn

- Các giá trị GoiBERT hiện đều thấp hơn các mốc tương ứng trong paper. Baseline này chưa cung cấp bằng chứng đạt mục tiêu vượt BamiBERT. Accuracy VSFC thấp hơn lần lượt khoảng1.65 và0.85 điểm.
- F1 chỉ được đặt cạnh nhau để tham khảo: evaluator/averaging của BamiBERT chưa được công bố đủ để xác nhận tương đương. CSV có numeric_difference_only, không được coi hiệu số đó là verified quality gap cho F1.
- UIT-ABSA dùng bản sao public chưa xác minh hoàn toàn lineage với release gốc. ViSFD OTHERS tính trong detection, không tính polarity pair. Các duplicate/overlap và nguồn corpus kế thừa đã ghi trong E0_DATA_METRIC_AUDIT.md.
- Đây là test readout của baseline đã chốt; không dùng test errors để chọn phương án E2. Chẩn đoán và thiết kế cải tiến tiếp tục dựa validation. Test không còn là tập chưa từng được đánh giá sau đợt này.

## Bằng chứng

- Manifest khóa trước inference: `artifacts/e1/test_selection_manifest.json`.
- Mỗi artifact trong manifest có `test_metrics.json`, `test_predictions.json`, `test_evaluation_metadata.json`.
- Dữ liệu bảng: `benchmarks/e1_test_comparison.csv`.
- Báo cáo operator: `docs/E1_TEST_EVALUATION.md`; kiểm tra độc lập: `docs/E1_TEST_REVIEW.md`.
