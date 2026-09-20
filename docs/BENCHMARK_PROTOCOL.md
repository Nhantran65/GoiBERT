# Benchmark mục tiêu

Nguồn paper: [BamiBERT, arXiv:2607.02259v1, bảng 1–2 và mục 3.1](https://arxiv.org/html/2607.02259).

| Dataset/task | Train / valid / test theo paper | Mốc BamiBERT, thang 100 |
|---|---|---|
| UIT-VSFC sentiment | 11,426 / 1,583 / 3,166 | Accuracy 93.86; F1 83.41 |
| UIT-VSFC topic | 11,426 / 1,583 / 3,166 | Accuracy 89.34; F1 79.90 |
| UIT-ViSFD | 7,786 / 1,112 / 2,224 | Detection F1 89.14; Classification F1 84.24 |
| UIT-ABSA Restaurant | 7,028 / 771 / 1,938 | Detection F1 88.01; Classification F1 74.89 |
| UIT-ABSA Hotel | 7,180 / 795 / 2,030 | Detection F1 79.99; Classification F1 72.65 |

UIT-ViSFD là smartphone feedback, không phải food dataset. UIT-ABSA Restaurant là mục tiêu F&B trực tiếp. UIT-ABSA Hotel được bổ sung vào phạm vi chính ngày 2026-09-08. Phạm vi gồm 4 dataset, tương ứng 5/8 benchmark trong paper (VSFC sentiment/topic tính riêng), 8 mốc F1 và 2 mốc accuracy. VLSP2018 Restaurant KHÔNG thay thế UIT-ABSA Restaurant khi so điểm paper. PhoATIS chỉ là lịch sử tham khảo, không thuộc mục tiêu mới.

## Quyết định đã thống nhất

- Dùng điểm paper làm mốc chính. KHÔNG bắt buộc chạy lại BamiBERT.
- So đúng dataset/version/split, nhãn và metric; số lượng split là sanity check, không đủ xác nhận hai phiên bản là một.
- Paper mô tả AdamW, batch 32, LR grid {1e-5,2e-5,5e-5}, 30 epochs, chọn checkpoint bằng validation F1. BamiBERT nhận raw text. Các tham số không công bố phải ghi rõ là lựa chọn của GoiBERT.
- GoiBERT được phép thay training/architecture. Khi so paper, gọi đúng là vượt kết quả được báo cáo; không coi đó là đối chứng cùng ngân sách tính toán.

## Metric audit bắt buộc trước claim

Paper ghi F1 nhưng chưa cung cấp evaluator đầy đủ trong tài liệu đã kiểm tra. Xác minh averaging, label inclusion, aspect absence, end-to-end vs conditional sentiment bằng nguồn dataset/evaluator chính thức. Không mặc định F1 ABSA là macro hoặc micro.

Hai báo cáo cũ đang có metric khác nhau:

- Family report: ViSFD classification **macro** F1 baseline 0.7548, S-GD 0.6990.
- Architecture-scaling report: ViSFD **micro** detection 0.8832, classification 0.7887 cho baseline; cấu hình và training khác Stage P.

Do đó khoảng cách 84.24 − 75.48 = 8.76 chỉ là hiệu số hai số được báo cáo, chưa chứng minh khoảng cách chất lượng trên metric tương đương. Không xếp chung hai bảng thành leaderboard.

## Quy trình đánh giá mới

1. Lưu nguồn dataset, revision, file hashes, official split IDs, label map, duplicates và mọi thay đổi preprocessing.
2. Dùng train để học. Không đưa test vào DAPT, distillation, tokenizer training, augmentation hoặc chọn cấu hình; kiểm tra corpus kế thừa.
3. Tách prediction generation khỏi evaluator. Có fixture cho TP/FP/FN và aspect absent; lưu evaluator version/hash cùng kết quả.
4. Screening bằng validation seed 42. Shortlist chạy seeds 42,43,44; final dùng 42,43,44,45,46. Chọn phương án trước khi xem test.
5. Report test từng seed và mean±std; không chọn seed tốt nhất làm headline. Lưu predictions để bootstrap paired giữa các biến thể mới nếu cần. Không suy ra statistical significance so với paper single score khi thiếu predictions/variance của tác giả.
6. Headline ưu tiên mean vượt các F1 mục tiêu; nếu chỉ thắng một số task thì ghi đúng phạm vi. Accuracy báo riêng.

## Trạng thái triển khai 2026-09-08

Code mới tại `src/goibert/` đã có loaders và evaluator riêng cho cả năm task. Xem E0_DATA_METRIC_AUDIT.md: VSFC dùng fixed-label macro F1 selection, ABSA dùng end-to-end pair micro F1 selection nội bộ; ViSFD OTHERS là binary slot riêng và không tính polarity pair. Source/evaluator tests và smoke được ghi trong EXECUTION_STATUS.md.

Paper không ghi rõ averaging/evaluator nên chưa xác nhận các F1 nội bộ tương đương paper. Public Restaurant/Hotel còn giới hạn lineage như DATA_STATUS.md. Screening được chạy trên train/validation theo E0_GATE_DECISIONS.md; không tự claim vượt paper. Code family trong reference vẫn là workflow cũ, không được dùng đổi tên VLSP thành UIT-ABSA.
