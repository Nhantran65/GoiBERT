# Model đã nhận

Cập nhật 2026-09-08: student/teacher load và forward trên L40S đã PASS; student backward/optimizer/save-load và BF16 train-step smoke cũng PASS. Xem E0_ENVIRONMENT_REPORT.md và EXECUTION_STATUS.md. Nội dung bên dưới ghi trạng thái lúc đóng gói ban đầu; không thay thế báo cáo smoke mới.

- `models/goiturn-s-gd/`: RobertaModel 10 layers, hidden 384, 6 heads, FFN 1536. distill_manifest.json ghi 30,000 steps, teacher goibert_b_dapt, fallback false.
- `models/goibert-b-dapt/`: RobertaForMaskedLM 12 layers, hidden 768, 12 heads, FFN 3072. pretrain_manifest.json ghi 40,000 MLM steps, mask 0.3.

Cả hai có weights safetensors, config, tokenizer và special token files. Vocab 20,481, max positions 2050, float32; max length train 256 theo manifest. Student là encoder, không phải model 3 nhãn turn completion. Teacher có MLM head, chưa là classifier cho từng task.

Hai manifests cùng corpus_stats_sha256 `1fa34627adddf9206f850e4595ecb0f88fdb47477eebb5beadb191ad20b26c80`. Đây là hash stats.json, không phải mlm_mix.txt; kiểm tra khi nhận corpus và hash corpus riêng.

Corpus và raw datasets đang chờ người dùng bổ sung. Không có optimizer/scheduler state: có thể train run mới từ weights, không coi là exact resume run cũ.

Đã kiểm tra config/manifest và cấu trúc safetensors (shapes, offsets, byte sizes), tạo checksum. Chưa chạy Transformers load/forward hoặc benchmark trên L40S. Metadata không chứng minh lại toàn bộ lịch sử training. Dùng PyTorch CUDA trên server mới dù manifest lịch sử ghi ROCm MI250X.
