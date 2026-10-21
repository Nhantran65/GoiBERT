# GoiBERT — Server research handoff

Ngày đóng gói: 2026-09-07. Ngôn ngữ tài liệu: tiếng Việt.

Cập nhật thực thi 2026-09-08: đã tải VSFC, ViSFD và bản sao public UIT-ABSA Restaurant/Hotel. Bắt đầu E0/E1 qua các subagents; xem [plan thực thi](E0_E1_EXECUTION_PLAN.md), [trạng thái công việc](EXECUTION_STATUS.md) và [trạng thái dữ liệu](DATA_STATUS.md). Root điều phối/review, subagents triển khai và chạy thí nghiệm. Gói ZIP là snapshot cũ, không tự cập nhật theo working directory.

Mục tiêu: phát triển một encoder tiếng Việt chung, fine-tune riêng theo task để vượt kết quả BamiBERT trong paper trên UIT-VSFC, UIT-ViSFD và UIT-ABSA Restaurant/Hotel. Project độc lập với TwoHearts và turn detection.

Điểm xuất phát: backbone **GoiTurn-S sau general distillation** (`goiturn_s_gd`), encoder 26,397,696 tham số; 26,546,691 nếu cộng head 3 nhãn. Được phép tăng kích thước. Compute dự kiến: 4 GPU L40S trên server; chưa kết nối hoặc chạy thử server.

## Đọc theo thứ tự

1. [Bối cảnh và quy trình tạo backbone](BACKBONE.md).
2. [Benchmark và các điểm cần xác minh](BENCHMARK_PROTOCOL.md).
3. [Kế hoạch experiment](EXPERIMENT_PLAN.md).
4. [Handoff server và tài sản còn thiếu](SERVER_HANDOFF.md).
5. [Nguồn gốc và giới hạn bằng chứng](PROVENANCE.md).

## Nội dung gói

- `benchmarks/paper_targets.csv`: mốc BamiBERT trong paper, thang 0–100.
- `benchmarks/historical_family.csv`: kết quả lịch sử chọn lọc, giữ nguyên thang 0–1 và loại metric.
- `reference/ai-foundary/`: bản sao tài liệu, code/config family và architecture-scaling, tokenizer/config BamiBERT. Đây là code tham chiếu, chưa phải pipeline GoiBERT mới đã triển khai.
- `MANIFEST.sha256`: checksum các file trong gói, không gồm chính manifest và file ZIP.

Đã nhận model tại `models/goiturn-s-gd/` và `models/goibert-b-dapt/`, gồm weights, config, tokenizer và training manifests. Raw datasets đã tải như DATA_STATUS.md; corpus cũ còn thiếu, không chặn baseline supervised. Per-run family JSON/predictions chưa có. Kết quả lịch sử chưa được rerun trên server mới. Xem docs/MODEL_STATUS.md.

Sau khi copy lên server, đứng trong thư mục gói và chạy `sha256sum -c MANIFEST.sha256`. Sau đó làm theo SERVER_HANDOFF.md. Không chạy tự động các script LUMI cũ.
