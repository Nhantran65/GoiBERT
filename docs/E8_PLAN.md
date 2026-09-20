# E8 — kết hợp noise consistency và LR theo layer

Người dùng đã duyệt thử E8; implementator được tạo rõ bằng `gpt-5.6-luna`, reasoning `xhigh`, không kế thừa phiên chính, theo AGENTS.md. Root phụ trách kế hoạch và đọc kết quả.

## Phạm vi cố định

Một cấu hình kết hợp, student GD gốc, seed42 trên cả5task. Không thêm seed, test, quét hệ số, KD, MLM hoặc kiến trúc mới.

Loss dùng nguyên E2 NC: clean CE + lambda × symmetric KL giữa clean/noisy, T1, gradient hai phía, cùng text noise và dropout độc lập. Lambda: Sentiment0.1,Topic0.5,ViSFD0.5,Restaurant0.1,Hotel0.1.

Optimizer dùng nguyên E7 gamma0.8: head/block cuối giữ base LR; block index i=0..9 dùng base×0.8^(9-i), embeddings base×0.8^10. Base LR1e-5 cho VSFC,1e-4 cho ABSA. Mọi tham số trainable, weight decay/bias/LayerNorm, AdamW/scheduler/clip theo E1/E7.

Fresh models/goiturn-s-gd + head CLS gốc mỗi trial,30epochs,batch32,max256,BF16/SDPA. Không khởi tạo từ E7 fine-tuned checkpoint. E2 NC và E7 LLRD là hai đối chứng có sẵn cùngseed42.

## Thực thi

Tái sử dụng frozen E2 runner và E7 optimizer bằng adapter riêng, không sửa E1–E7. Ghi E8 recipe/source và các dependency hashes đúng nghĩa; kiểm tra coverage/LR groups và một combined forward/backward hữu hạn, rồi được phép chạy ngay. Không lặp bộ test cũ.

Worker detached/file logs; GPU2/1/3 khi có ít nhất5GiB free, một worker/GPU thực sự trống. CUBLAS trướcCUDA, đúng PYTHONPATH. Không chạy trùng, không kill workload khác. Khi đủ5trial, tính lại validation metrics/IDs/gold từ predictions một lần.

## Đánh giá

| Task | E2 NC seed42 | E7 LLRD seed42 |
|---|---:|---:|
| Sentiment | 84.7722 | 84.0638 |
| Topic | 79.8524 | 79.9539 |
| ViSFD | 81.0884 | 81.4927 |
| Restaurant | 69.0600 | 71.5175 |
| Hotel | 73.5776 | 73.2909 |

Báo E8 với E1 và cảhai thành phần. Chỉ có giá trị thêm nếu vượt thành phần tốt hơn; không cộng cơ học mức tăng của NC/LLRD. Gain dưới0.3điểm ở mộtseed là tín hiệu yếu. Có thể regularize quá mức; nếu không hơn thì giữ phương pháp riêng và không tự quét lại hệ số.

Outputs: artifacts/e8, docs/E8_RESULT.md. Đây là screening validation, chưa phải test hoặc kết quả multi-seed.
