# E5 — content mean pooling, seed42 screening

Người dùng duyệt thử mean pooling. Mục tiêu: thay cách lấy biểu diễn câu của student gốc, giữ nguyên số tham số và cách fine-tune.

- Model: fresh models/goiturn-s-gd, FFN1536, 10 layers, hidden384; không dùng model mở rộng E4 hoặc checkpoint đã fine-tune.
- Pooling: trung bình final hidden states tại vị trí attention_mask=1 và input_id không thuộc tokenizer.all_special_ids. Loại padding và token đặc biệt. Nếu không còn token hợp lệ, fallback về vector CLS để tránh NaN.
- Pool trước dropout, rồi giữ đúng head dropout/dense384/tanh/dropout/output E1. Head khởi tạo và global RNG phải giống E1. Decoder và ánh xạ nhãn giữ nguyên.
- Năm task: VSFC Sentiment, VSFC Topic, ViSFD, Restaurant, Hotel. Mỗi task seed42, 30 epochs; LR1e-5 cho VSFC, 1e-4 cho ABSA; CE gốc, batch32, max_length256, BF16/SDPA, cùng AdamW/scheduler/selection như E1.
- Không NC/KD/E3, test, multi-seed hoặc quét LR. Không sửa source/artifacts E1–E4.

## Thực thi tối thiểu

Subagent triển khai adapter riêng dùng lại E1 runner. Chỉ kiểm tra mask/empty fallback, head/RNG và gradient hữu hạn; không chạy lại toàn bộ test cũ. Được phép launch sau các kiểm tra này, không thêm gate.

GPU2 ưu tiên, GPU1/3 khi trống và >=5GiB free; một worker/GPU. Queue phải chọn theo tập GPU thực sự đang bận, tránh lỗi devices[len(running)] của E4. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 trước khởi tạo CUDA, PYTHONPATH và file logs cho worker detached.

Sau đủ năm trial, tính lại validation metrics/IDs/gold từ predictions một lần. Source hash, pooling/mask/fallback và backbone phải được lưu để dựng lại checkpoint đúng cách; trọng số có cùng shape với E1 không có nghĩa forward vẫn là CLS.

## Đánh giá

Mốc chính: E1 CE cùng seed42. Mốc thực dụng: NC Sentiment84.7722, KD Topic80.7842, NC ViSFD81.0884, E3 Restaurant70.1299, NC Hotel73.5776, đều validation seed42. VSFC dùng macro F1, ABSA dùng pair micro F1.

Chỉ coi tăng >=0.3 điểm so E1 là tín hiệu screening đáng chú ý, đồng thời xem delta với mốc thực dụng. Không tuyên bố significance hoặc tự thêm seed. Kết quả áp dụng cho pooling và recipe này, không phải mọi cách pooling.

Deliverables: code adapter tối thiểu, artifacts/e5, docs/E5_RESULT.md. Root tổng hợp và chốt hướng sau khi có đủ kết quả.
