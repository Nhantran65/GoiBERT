# E18 — Kết quả screening 4B (fork-consistency), đợt 1

Protocol `E18-segment-fork-consistency-screening-4B-v1`, seed 42, 30 epochs, recipe E8 (NC λ=0,1 + LLRD γ=0,8), micro 16 × accum 2, validation-only. Plan: [E18_PLAN.md](E18_PLAN.md); preflight/profile: [E18_EXECUTION_STATUS.md](E18_EXECUTION_STATUS.md). Hai sửa đổi trước launch được ghi trong `configs/e18/protocol.json`: attention dropout trên đường fork (fix 1), ramp λ_fork 0,1→0,3 trong 3 epoch đầu (fix 2). Controller đợt đầu dừng ở job 3 vì gate utilization (5%) không tương thích với 2 worker/GPU; resume tuần tự, nhận lại 2 trial đã xong (status.json), không rerun.

## Validation pair F1 (best epoch)

| Task | b0 baseline | b1 fork (4B) | b2 random-mask control | E8 gốc cùng seed (tham chiếu) |
|---|---:|---:|---:|---:|
| Restaurant | **72,26** (ep22) | 71,37 (ep23) | 71,15 (ep29) | 71,28 |
| Hotel | 72,47 (ep16) | 72,97 (ep20) | **73,16** (ep21) | 73,05 |

F1 tính lại từ `best_validation_metrics.json`; IDs/gold đã được E2 loop kiểm trong run. Nhiễu giữa các lần chạy cùng recipe ≈ 1 điểm (b0 Restaurant 72,26 so E8 71,28; b0 Hotel 72,47 so E8 73,05).

**Gate: không đạt.** b1 không hơn b0 trên cả hai task (Restaurant −0,9; Hotel +0,5 trong nhiễu); b2 ≈ b1 ở cả hai task (Hotel b2 còn cao hơn b1). Phần chênh nếu có là hiệu ứng regularization/đọc fragment, không phải cấu trúc đoạn.

## Cơ chế có đạt không? Có — nhưng không đổi F1

Trần đọc-theo-đoạn (oracle chọn split theo gold; đối chứng random-mask cùng số ứng viên) tính trên chính 6 checkpoint (`segment_read_ceiling_summary.json` trong từng thư mục trial):

| Task | Arm | Oracle liên tục | Oracle random-mask | Phần riêng của cấu trúc đoạn |
|---|---|---:|---:|---:|
| Restaurant | b0 | 79,31 | 78,66 | 0,65 |
| Restaurant | b1 | 80,59 | 78,86 | **1,74** |
| Restaurant | b2 | 80,28 | 79,46 | 0,82 |
| Hotel | b0 | 78,45 | 77,15 | 1,29 |
| Hotel | b1 | 79,08 | 77,71 | 1,37 |
| Hotel | b2 | 78,52 | 77,74 | 0,78 |

Telemetry b1: fork CE trên train giảm 0,71 → 0,01; tỉ lệ split được chấp nhận 0,85 → 0,52 khi CLS tự giải thích gold tốt dần. Trên Restaurant, huấn luyện fork làm phần đoạn của trần tăng gấp đôi (0,65 → 1,74) đúng dự đoán (i) của plan, nhưng F1 không tăng — đúng kịch bản (iii): **đọc được theo đoạn không phải nút thắt của pair F1 với student này.** Hotel gần như không đổi gap.

## Quyết định

Đóng hướng "đoạn/reader/định tuyến đọc" (E9, C4, 4A, 4B) với bằng chứng sạch: trần thấp (proposal §4.4), huấn luyện đạt cơ chế mà không đổi quality, control ngẫu nhiên ngang ứng viên. Không chạy b3/b4, không multi-seed, không test. Artifacts: `artifacts/e18/`.

Không có kết quả nào trong E18 vượt E8 ngoài biên nhiễu; mục tiêu thắng BamiBERT chưa đạt.
