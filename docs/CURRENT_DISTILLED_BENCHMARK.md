# Distilled model hiện tại và mốc BamiBERT

Tổng hợp từ artifacts hiện có; đã cập nhật sau lượt test E8 được người dùng cho phép. Không train lại model. “Ứng viên hiện tại” là lựa chọn theo validation trong vòng screening cùng seed42, không phải lấy seed có test cao nhất. Các cấu hình đều dùng student GD gốc, cùng kiến trúc; không dùng FFN mở rộng hoặc backbone MLM E6.

## Các ứng viên hiện tại — seed42

Val dùng chọn cấu hình. Cột BamiBERT là test từ paper, chỉ so với cột TEST của chúng ta; không lấy delta giữa val và paper test.

| Task | Cấu hình | Val F1 | Test F1 đã có | BamiBERT paper test F1 |
|---|---|---:|---:|---:|
| Sentiment | E8: NC + LLRD | 85.16 | 81.21 | 83.41 |
| Topic | E2: KD | 80.78 | 79.83 | 79.90 |
| ViSFD | E8: NC + LLRD | 82.44 | 81.75 | 84.24 |
| Restaurant | E7: LLRD | 71.52 | Chưa chạy | 74.89 |
| Hotel | E2: NC | 73.58 | 72.70 | 72.65 |

NC là noise consistency, LLRD là learning rate theo layer, KD là task-logit distillation. Hệ số giữ từ các vòng đã chốt: E8 gamma0.8 với NC lambda0.1 Sentiment/lambda0.5 ViSFD; E7 gamma0.8 Restaurant; E2 KD alpha0.75/T2 Topic; E2 NC lambda0.1 Hotel.

E8 Sentiment/ViSFD đã được chấm test; E7 Restaurant trong bảng ứng viên theo task vẫn chưa test. Topic và Hotel dùng đúng test của cùng checkpoint seed42 đã chọn theo validation. Hotel72.7045 nhỉnh hơn mốc paper72.65 khoảng0.0545điểm ở seed này; chưa chứng minh hơn ổn định (E2 NC Hotel mean3seeds72.2797). Topic seed42 test79.8347, dưới mốc79.90 khoảng0.0653điểm.

## Bảng test nhiều seed đã có: E2, mean ± sample SD của seeds42–44

Bảng này là các model E2 cũ, không phải kết quả test của E7/E8 mới. Chọn branches bằng validation trước khi mở test; sử dụng canonical CUDA/BF16/SDPA outputs, không dùng CPU diagnostic đã lưu riêng.

| Task | Metric | E2 test (3 seeds) | BamiBERT paper | Delta |
|---|---|---:|---:|---:|
| VSFC Sentiment | accuracy | 92.38 ± 0.15 | 93.86 | -1.48 |
| VSFC Sentiment | macro_f1 | 81.31 ± 0.79 | 83.41 | -2.10 |
| VSFC Topic | accuracy | 89.30 ± 0.19 | 89.34 | -0.04 |
| VSFC Topic | macro_f1 | 79.45 ± 0.37 | 79.90 | -0.45 |
| ViSFD | detection_micro_f1 | 87.11 ± 0.30 | 89.14 | -2.03 |
| ViSFD | pair_micro_f1 | 80.62 ± 0.26 | 84.24 | -3.62 |
| UIT-ABSA Restaurant | detection_micro_f1 | 84.51 ± 0.54 | 88.01 | -3.50 |
| UIT-ABSA Restaurant | pair_micro_f1 | 69.86 ± 0.63 | 74.89 | -5.03 |
| UIT-ABSA Hotel | detection_micro_f1 | 79.20 ± 0.29 | 79.99 | -0.79 |
| UIT-ABSA Hotel | pair_micro_f1 | 72.28 ± 0.39 | 72.65 | -0.37 |

## Giới hạn so sánh và phần còn thiếu

VSFC dùng fixed-label macro F1; ABSA dùng end-to-end pair micro F1 (ViSFD OTHERS được tính detection nhưng loại khỏi polarity pairs). F1 averaging/evaluator của paper chưa được xác minh hoàn toàn; lineage của mirror UIT-ABSA có giới hạn đã ghi. Không tuyên bố significance hay thắng BamiBERT chỉ từ chênh lệch nhỏ.

E8 đã được đánh giá test cả5task theo yêu cầu người dùng; xem [E8_TEST_RESULT.md](E8_TEST_RESULT.md) cho bảng một recipe chung. Trong bảng ứng viên riêng từng task ở trên, chỉ E7 Restaurant còn chưa test; không nhầm kết quả E8 Restaurant thành test của E7.

Nguồn paper: [BamiBERT Table2](https://arxiv.org/html/2607.02259). Nguồn local: per-trial metadata/testmetrics, benchmarks/paper_targets.csv và artifacts/e2/test_readout.json. Snapshot máy đọc được: benchmarks/current_distilled_candidates.json.
