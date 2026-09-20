# E2 — Kết quả validation sau khi hoàn tất

Đợt E2 kết thúc bình thường lúc 00:08:08 ngày 09/09/2026 (UTC+7), supervisor returncode=0. Đã hoàn tất 61 training trials và 5 train-logit caches. Ba attempt teacher Topic từng bị gián đoạn được lưu riêng; không tính là completed hoặc dùng chọn kết quả.

Bảng dùng cùng seeds 42,43,44 cho E1 và E2; mean ± sample std trên thang 100. VSFC dùng fixed-label macro F1; ABSA dùng end-to-end pair micro F1. Đây hoàn toàn là validation, chưa E2 test.

| Task | Nhánh | Hệ số chọn từ seed42 | E1 mean | E2 mean ± SD | Delta (điểm) | Seeds tăng |
|---|---|---:|---:|---:|---:|---:|
| VSFC Sentiment | KD | 0.25 | 84.49 | 84.40 ± 0.33 | -0.10 | 2/3 |
| VSFC Sentiment | NC | 0.1 | 84.49 | 84.93 ± 0.22 | +0.44 | 3/3 |
| VSFC Topic | KD | 0.75 | 80.10 | 81.06 ± 0.24 | +0.96 | 3/3 |
| VSFC Topic | NC | 0.5 | 80.10 | 79.80 ± 0.08 | -0.31 | 2/3 |
| ViSFD | KD | 0.25 | 80.25 | 80.26 ± 0.38 | +0.02 | 1/3 |
| ViSFD | NC | 0.5 | 80.25 | 81.23 ± 0.28 | +0.99 | 3/3 |
| UIT-ABSA Restaurant | KD | 0.25 | 68.98 | 68.40 ± 0.44 | -0.59 | 0/3 |
| UIT-ABSA Restaurant | NC | 0.1 | 68.98 | 69.39 ± 1.19 | +0.41 | 2/3 |
| UIT-ABSA Hotel | KD | 0.25 | 72.07 | 71.64 ± 0.33 | -0.43 | 0/3 |
| UIT-ABSA Hotel | NC | 0.1 | 72.07 | 72.68 ± 0.81 | +0.61 | 2/3 |

## Nhận định của root

- NC trên VSFC Sentiment (+0.44 điểm) và ViSFD (+0.99), KD trên VSFC Topic (+0.96) đều tăng ở cả ba seeds: ứng viên rõ nhất để xác nhận tiếp.
- NC trên Restaurant (+0.41) và Hotel (+0.61) tăng trung bình nhưng chỉ 2/3 seeds; variance và sự khác nhau giữa seeds cần được giữ rõ. Chưa gọi đây là gain ổn định như ba ứng viên trên.
- KD không đem lại gain có ý nghĩa trên Sentiment/ViSFD và giảm cả ba seeds trên Restaurant/Hotel. Kết quả này áp dụng cho logits KD, teacher cố định, T=2 và LR E1 trong đợt này; không bác bỏ mọi cách distillation.
- Các coefficient được chọn bằng seed42; mean validation này không phải một estimate độc lập, và ba seeds chưa chứng minh statistical significance. NC bao gồm cả text noise và dropout consistency, với compute cao hơn một forward CE.
- Kết quả chứng minh có thể cải thiện một số task mà giữ nguyên kiến trúc student; không chứng minh đã cải thiện một backbone pretrained dùng chung mới hoặc vượt BamiBERT trên test.

## Bước đề xuất, chưa khởi chạy

Xác nhận branches thắng bằng seeds45/46 trước; ưu tiên ba ứng viên tăng 3/3 seeds, đồng thời làm rõ Restaurant/Hotel. Chưa tự mở kết hợp KD+NC, test hoặc architecture expansion. Mọi lựa chọn tiếp theo dùng validation evidence.

Nguồn: `artifacts/e2/.orchestrator/stageB_paired_e1_comparison.json`, approved records và per-trial metadata. Review độc lập sẽ nằm tại `docs/E2_COMPLETED_AUDIT.md`.
