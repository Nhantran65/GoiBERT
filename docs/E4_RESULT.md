# E4 — kết quả screening mở rộng FFN

E4 mở FFN của cả 10 encoder layer từ 1536 lên 3072, giữ hidden size 384, 10 layer, 6 attention heads và head E1. Với head Sentiment, có 38,358,531 tham số, tăng 11,811,840 so với E1 (26,546,691). Chạy một seed 42, 30 epoch, CE gốc, learning rate E1 đã chọn, effective batch 32, max length 256, BF16/SDPA trên cả năm task.

| Task | E1 CE seed42 | E4 FFN3072 | Δ E4−E1 |
|---|---:|---:|---:|
| VSFC Sentiment (macro-F1) | 84.5076 | 84.6135 | +0.1059 |
| VSFC Topic (macro-F1) | 79.7709 | 78.9927 | −0.7782 |
| ViSFD (pair micro-F1) | 80.0722 | 79.9515 | −0.1207 |
| UIT-ABSA Restaurant (pair micro-F1) | 69.0174 | 67.8472 | −1.1702 |
| UIT-ABSA Hotel (pair micro-F1) | 72.0375 | 71.4059 | −0.6317 |

Kết quả được recompute một lần từ `validation_predictions.json`; toàn bộ ID, thứ tự và gold khớp split validation. E4 không dùng test và chưa chạy multi-seed. Artifact tổng hợp nằm tại [artifacts/e4/summary.json](../artifacts/e4/summary.json).

Preflight chuyển trọng số pass: FP32 max absolute difference 2.86e-6 với tolerance 1e-4, non-FFN weights và head initialization giữ nguyên, gradient hữu hạn, và memory batch32×256 đạt peak reserved 2.26 GiB trên GPU2. E4 source hash: `84831243f59d918d7659974335f303130c7d8e9432260889916a222090a8bec7`.

Attempt Hotel đầu tiên bị dừng có chủ ý ở epoch5 sau khi controller cấp trùng GPU3 cho Restaurant và Hotel; thư mục được giữ tại `seed_42__interrupted_sigterm`. Hotel được chạy lại độc lập trên GPU2 và hoàn tất. Bốn worker đầu thiếu biến `CUBLAS_WORKSPACE_CONFIG=:4096:8`, nên không tuyên bố parity runtime deterministic tuyệt đối với E1; không rerun bốn kết quả đó.

## Quyết định của root

Dừng cấu hình E4 này: không task nào đạt ngưỡng screening +0.3 điểm so với E1, trong khi số tham số tăng khoảng44.5%. Cả năm điểm còn dưới các mốc phương pháp học đã chọn trước: NC Sentiment84.7722, KD Topic80.7842, NC ViSFD81.0884, E3 Restaurant70.1299, NC Hotel73.5776 (đều validation seed42).

Kết luận chỉ áp dụng cho FFN3072 với cách chuyển trọng số và CE/LR đã thử. Chưa chứng minh student nhỏ đã đạt giới hạn, cũng không bác bỏ mọi cách tăng kiến trúc. Không tự chạy thêm seed, quét LR hoặc ghép NC vào E4. Giữ model gốc và các baseline E2/E3 cho vòng nghiên cứu tiếp theo.

Lỗi chọn GPU của launcher đã được ghi nhận; không tái sử dụng cơ chế xếp hàng này cho vòng khác nếu chưa sửa. Source được giữ nguyên để bảo toàn provenance của E4 đã chạy.
