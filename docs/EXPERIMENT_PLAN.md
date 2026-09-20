# GoiBERT experiment plan v1

Trạng thái cập nhật 2026-09-08: E0 đã có audit dữ liệu/evaluator và environment smoke PASS; pilot E1 VSFC sentiment đang chạy. Kế hoạch thực thi chi tiết và gate nằm tại E0_E1_EXECUTION_PLAN.md, E0_GATE_DECISIONS.md và EXECUTION_STATUS.md. 4×L40S, không áp trần tham số hoặc GPU-hours. Không có yêu cầu chứng minh novelty trước khi làm. Chất lượng trên benchmark đã chọn là ưu tiên.

## E0 — Nhận tài sản và chuẩn hóa benchmark

Nhận backbone GD-S, tokenizer/config, provenance/log/teacher identity; ghi SHA256 và parameter count. Xác minh dataset, splits và evaluator theo BENCHMARK_PROTOCOL.md. Bổ sung VSFC-topic và UIT-ABSA Restaurant/Hotel. Chạy forward/backward smoke và load/save round-trip; không khởi động full grid khi dataset/metric chưa rõ.

Đầu ra: manifest dữ liệu/model, evaluator có fixtures, baseline runner, environment lock thực tế trên NVIDIA server. Checkpoint/data còn thiếu chỉ chặn những task phụ thuộc, không chặn xây evaluator hoặc tài liệu.

Phạm vi chính cập nhật 2026-09-08: VSFC sentiment, VSFC topic, ViSFD, UIT-ABSA Restaurant và UIT-ABSA Hotel. Mỗi benchmark có head/checkpoint task riêng từ cùng backbone; các bước screening, shortlist và final đều bao gồm Hotel.

## E1 — Baseline student

Từ cùng backbone bước 5, gắn head mới cho từng task. Train supervised-only (CE phù hợp task), chưa KD hoặc NC. Một checkpoint backbone chung làm nguồn cho tất cả task; mỗi task có checkpoint fine-tuned riêng.

Khởi đầu AdamW, batch hiệu dụng 32, 30 epochs, LR {1e-5,2e-5,5e-5,1e-4,2e-4}. LR mở rộng là lựa chọn mới, không phải protocol nguyên xi paper. Max length xác định qua phân bố token train, ghi tỷ lệ truncation; có thể kiểm tra 256/512 bằng validation. Chốt warmup, weight decay, loss và pooling bằng config, áp dụng nhất quán cho đối chứng.

Đầu ra: quality floor của backbone hiện có, per-class/aspect errors, validation learning curves. Không lấy historical task-distilled F1 thay cho E1.

## E2 — Training ablation, giữ nguyên architecture

| ID | Thay đổi | Đối chứng |
|---|---|---|
| E2a | Task KD từ teacher fine-tuned train-only | E1 |
| E2b | Noise consistency | E1 |
| E2c | KD + NC | E2a và E2b |
| E2d | Continued general distillation trên corpus đã audit | Backbone E1, cùng task recipe |

KD cần teacher task checkpoint; không cần chạy BamiBERT test benchmark. Nếu teacher chưa có, E1/NC vẫn chạy độc lập. Khi thêm data hoặc bước train, có đối chứng student gốc nhận cùng data/update budget. Đánh giá clean validation làm chính; noisy validation là phụ, không thay mục tiêu paper.

## E3 — Mở rộng architecture từ GD-S

Các cấu hình dưới đây là giả thuyết screening, chưa phải kiến trúc đã chọn:

| ID | Layers / hidden / heads / FFN | Mục đích |
|---|---|---|
| S0 | 10 / 384 / 6 / 1536 | Gốc |
| F1 | 10 / 384 / 6 / 2304 | Thêm dung lượng FFN |
| F2 | 10 / 384 / 6 / 3072 | Kiểm tra saturation FFN |
| D1 | 12 / 384 / 6 / 1536 | Thêm chiều sâu |
| D2 | 16 / 384 / 6 / 1536 | Kiểm tra chiều sâu lớn hơn |
| W1 | 10 / 512 / 8 / 2048 | Tăng độ rộng |
| W2 | 12 / 768 / 12 / 3072 | Ứng viên dung lượng lớn nếu cần |

Không copy tensors khác shape một cách tùy tiện. FFN expansion có thể thiết kế neuron replication và chia output weights để giữ hàm gần ban đầu; layer expansion và width expansion cần mapping/init được kiểm tra riêng. Khi không giữ được hàm, ghi rõ approximation/reinitialization và đo drift trước/sau surgery.

Mỗi biến thể: load/migrate weights → verify shape/finite outputs và init drift → continued distillation hoặc MLM thích hợp → task training. Lưu mapping weights, phần mới khởi tạo và adaptation budget. So với S0 được train thêm tương đương và đối chứng có số tham số gần nhau khi khả thi. Không hứa expansion sẽ thắng.

## E4 — Kết hợp và xác nhận

Chỉ kết hợp thay đổi thắng validation ổn định. Tái kiểm tra interaction bằng ablation, tránh quy mọi gain cho architecture. Shortlist trên toàn bộ task; không tạo backbone đặc biệt cho từng test set. Nếu các task xung đột, giữ bảng Pareto theo validation, chốt ưu tiên trước test.

Final: một backbone chung + các head/checkpoint theo task, 5 seeds, mean±std, predictions, configs, exact commands, git commit, hashes, môi trường, parameter count, training time và latency đo thật. So với paper theo phạm vi metric đã xác minh.

## Scheduling trên 4 GPU

Trước hết profile memory/throughput thực tế. Với model nhỏ, có thể chạy 4 trials độc lập, mỗi GPU một process. Multi-GPU DDP chỉ khi code đã hỗ trợ và có nhu cầu; code tham chiếu hiện không phải pipeline DDP hoàn chỉnh. Dùng effective batch rõ ràng để không vô tình thay recipe khi thay số GPU. Các thông số AMP/attention phải smoke-test, nhất là KD cần materialize attention.

## Quy tắc giữ/loại

- Screening chưa dùng test. Loại biến thể lỗi số học hoặc kém đối chứng nhất quán; ghi cả kết quả âm.
- Lợi ích nhỏ chỉ một seed cần multi-seed trước khi kết luận.
- Không loại model chỉ vì lớn hơn 26.5M; latency/params là thông tin phụ.
- Không mặc định tăng GPU-hours, thêm layers hay distillation sẽ đảm bảo vượt BamiBERT.
