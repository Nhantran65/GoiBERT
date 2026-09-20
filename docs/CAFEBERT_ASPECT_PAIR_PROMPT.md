# Prompt — thực thi pilot CafeBERT aspect-pair

Hãy triển khai và chạy pilot theo `docs/CAFEBERT_ASPECT_PAIR_PLAN.md` trong `/home/kiettqa/GoiBERT`. Đọc toàn bộ plan và AGENTS.md trước; tự thực hiện, không tự tạo thêm agent. Chỉ hai lượt Restaurant, seed42, validation-only.

**Câu hỏi:** đưa aspect vào đầu vào encoder có giúp hơn CafeBERT baseline 77,7197 validation không?

1. Arm A dùng cặp `(review nguyên văn, mã aspect "mục 01"…"mục 12")`; arm B dùng `(review nguyên văn, mô tả aspect)` đúng mapping đã khóa trong plan.
2. Mỗi review tạo đủ 12 cặp; lấy nhãn từ vector gold gốc. Không cắt câu, không gán nhãn lại, không bỏ absent. Gộp dự đoán theo stable ID và slot thành vector 12 nhãn rồi dùng evaluator pair micro F1 gốc.
3. Cả hai khởi tạo từ CafeBERT pretrained gốc, gắn shared head 4 lớp mới; không dùng checkpoint đã fine-tune task. LR 2e-5, 30 epochs, seed42, CE, BF16/SDPA theo plan. Không DAPT, KD, NC, LLRD hoặc composite.
4. **Batch hiệu dụng 32 reviews = 384 pairs**. Accumulate microbatches trong từng review group; không coi 32 pairs là batch 32 reviews. Tổng 6.600 optimizer steps trong 30 epochs; group cuối chuẩn hóa theo số pairs thực. A/B cùng head-init, thứ tự review và ngân sách cập nhật.
5. Reuse baseline `artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42/`, không train lại. Source mới `scripts/cafebert_aspect_pair/`, artifacts mới `artifacts/cafebert_aspect_pair/`. Giữ nguyên source, gold và artifacts cũ.
6. Kiểm expand→regroup gold chính xác, tokenizer pair/special tokens/truncation, finite gradients và cách weighting loss. Profile một optimizer step trên GPU trước full run và báo ETA: chi phí khoảng 12 encoder passes/review, không hứa nhanh như baseline. Dùng GPU còn tài nguyên, không dừng jobs khác.
7. Chạy A/B, lưu best validation checkpoint và predictions đã gộp; reload/recompute để kiểm tra. Báo `docs/CAFEBERT_ASPECT_PAIR_RESULT.md`: reference/A/B, các delta, best epoch, runtime và giới hạn diễn giải.

Xong báo cáo và dừng. Gain ít nhất 0,5 điểm so baseline mới là tín hiệu đáng xác nhận ở lượt sau. Không tự thêm seed, mở test, quét wording/LR hoặc chạy task khác. Nếu cả A/B tăng tương tự, không quy gain cho ý nghĩa tên aspect. Tự tiến hành các bước đã chốt trong scope; báo blocker nếu không thể giữ đúng protocol.
