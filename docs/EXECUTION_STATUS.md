# Trạng thái thực thi GoiBERT

Cập nhật 2026-09-08 sau yêu cầu đánh giá test của người dùng.

## Đã hoàn tất

- E0: environment/checkpoint smoke, loaders và evaluator. Bốn NVIDIA L40S, Python3.12, PyTorch2.6+cu124, Transformers4.49; BF16/SDPA.
- E1 validation: 55 records (25 screening +20 shortlist +10 final), gồm pilot được tái dùng. Không trial training thất bại. Chọn LR bằng validation42–44, kiểm tra đủ seeds42–46.
- E1 test: user đã mở gate trong E1_TEST_GATE_APPROVAL.md; đã evaluate25/25checkpoint, không retrain hoặc chọn lại checkpoint theo test. Test hiện đã được dùng để đánh giá; trạng thái “chưa test” trong các readout cũ là lịch sử.

## Kết quả và bằng chứng

- [Bảng benchmark test so BamiBERT](E1_TEST_BENCHMARK.md).
- [Báo cáo operator test](E1_TEST_EVALUATION.md), [review độc lập](E1_TEST_REVIEW.md).
- [Audit validation hoàn tất](E1_COMPLETED_AUDIT.md).
- CSV so sánh: `benchmarks/e1_test_comparison.csv`.
- Manifest khóa25checkpoint: `artifacts/e1/test_selection_manifest.json`.
- Kết quả test: `artifacts/e1/test_readout.json`; predictions/metrics per-seed nằm cạnh từng checkpoint trong manifest.

Nguồn khởi tạo của mọi trial là `models/goiturn-s-gd/`. Đây là baseline supervised-only, chưa task KD/NC hoặc thay kiến trúc. Các giá trị test hiện đều thấp hơn mốc paper; F1 equivalence chưa được xác minh đầy đủ, không suy verified quality gap từ phép trừ các F1 khác protocol. Accuracy VSFC có định nghĩa phù hợp để đối chiếu các điểm được báo cáo.

## Tiếp tục

Root review kết quả agents rồi thiết kế E2 từ validation evidence; chưa tự động khởi chạy E2. Không dùng test errors để chọn kiến trúc, LR, loss hoặc seed. Các giới hạn lineage của UIT-ABSA public copy, overlap dữ liệu và corpus kế thừa nằm trong E0_DATA_METRIC_AUDIT.md và DATA_STATUS.md.

## Lịch sử vận hành

Plan gốc: E0_E1_EXECUTION_PLAN.md. Nhật ký pilot/queue: E1_EXECUTION_LOG.md. Pilot VSFC sentiment LR2e-5 seed42 hoàn tất30epochs trong638.06s, best-validation epoch7 F1=84.34%; đây là pilot, không thay bảng test final. PID4157853 và các PID con trong nhật ký chỉ là snapshot lịch sử, không được coi là process hiện còn chạy.

Frozen E1 runner source hash: `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`. Root điều phối và tổng hợp; các subagents triển khai, train và đánh giá.
