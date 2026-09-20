# E7 — learning rate giảm dần theo độ sâu

Người dùng duyệt thử sau D2. Root thiết kế/đọc kết quả; subagent e7_llrd_run triển khai và chạy. Năm trial seed42, không test/multi-seed hoặc quét hệ số.

## Giả thuyết

D2 cho thấy student ABSA đổi biểu diễn ở các tầng dưới nhiều hơn teacher, trong khi khả năng phân loại ở tầng trên chưa tăng tương ứng. Chưa chứng minh đây là forgetting hoặc nguyên nhân gap. E7 kiểm chứng một cách phân bố cập nhật khác: giữ LR block cuối/head như E1, giảm dần ở các block dưới.

## Cấu hình cố định

Fresh models/goiturn-s-gd gốc, FFN1536; không E6 MLM hoặc E4 expanded. Kiến trúc, CLS head, CE, tokenizer, dataset splits, head initialization giữ nguyên E1. Mỗi task seed42,30epochs,micro/effective batch32,max_length256,BF16/SDPA.

Base LR:1e-5 cho VSFC Sentiment/Topic;1e-4 cho ViSFD/Restaurant/Hotel. Gamma0.8 duy nhất:

- Block i (đánh số1..10): LR = base ×0.8^(10-i).
- Embeddings: LR = base ×0.8^10.
- Head: LR = base.

| Nhóm | Hệ số so với base | LR VSFC | LR ABSA |
|---|---:|---:|---:|
| Embeddings | 0.107374 | 1.0737418e-06 | 1.0737418e-05 |
| Block 1 | 0.134218 | 1.3421773e-06 | 1.3421773e-05 |
| Block 4 | 0.262144 | 2.62144e-06 | 2.62144e-05 |
| Block 7 | 0.512000 | 5.12e-06 | 5.12e-05 |
| Block 10 / head | 1.000000 | 1e-05 | 0.0001 |

Tất cả tham số vẫn trainable, mỗi parameter xuất hiện đúng một optimizergroup. AdamW betas0.9/0.999,eps1e-8,weight decay0.01; bias/LayerNorm không decay như E1. Global clip1. Scheduler warmup10% rồi linear decay dùng chung một factor cho mọi group, giữ tỷ lệ LR. LR nhỏ cũng làm giảm tác dụng AdamW decay theo bước; đây là một phần của recipe, không quy toàn bộ effect cho gradient.

## Thực thi tối thiểu

Tái sử dụng frozen E1 train_trial qua optimizer adapter riêng. Kiểm tra coverage/no duplicates/LR endpoints/scheduler ratios/no-decay; gamma1 update equivalence nếu rẻ. Không lặp bộ test cũ hoặc làm lại GPUprofile khi model không đổi. Ghi nhóm LR vào metadata/checkpoint và lưu LR mọi nhóm theo epoch (ghi rõ nếu tái dựng từ scheduler).

Source/model E1–E6/D1/D2 không sửa. E7 có hash riêng, original E1 dependency hash456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873.

CUBLAS_WORKSPACE_CONFIG=:4096:8 trướcCUDA, đúng PYTHONPATH. GPU2/1/3 khi≥5GiBfree, một worker trên mỗi GPU thực sự trống; queue theo free-device set. Durable logs/status, từ chối launch lặp. Không kill workload khác.

## Đọc kết quả và dừng

Mốc chính E1 CE cùng seed42; thêm mốc NC/KD/E3 đã có để đánh giá thực dụng. Không so với mean3/5seeds. Ngưỡng screening≥0.3điểm F1/task; xem cả độ rộng và task bị giảm. Không gọi significance từ mộtseed.

Đối chiếu predictions IDs/gold/metrics một lần sau đủ5trial. Nếu rẻ, tái dùng D2 drift baseline và checkpoint E7 để đo early/mid/late relative weight changes, không cần encoderinference mới. Giảm drift mà F1 không tăng sẽ không ủng hộ giả thuyết cải thiện chất lượng. Chưa tự thêm gamma,seed,NC,MLM hoặc test.

Đầu ra: artifacts/e7, code/config adapter, docs/E7_RESULT.md; root tổng hợp sau khi hoàn tất cả5task.
