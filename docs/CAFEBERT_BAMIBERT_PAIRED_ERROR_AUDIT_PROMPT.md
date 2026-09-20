# Prompt bàn giao — audit lỗi ghép cặp CafeBERT–BamiBERT

Hãy đọc `AGENTS.md` và toàn bộ `docs/CAFEBERT_BAMIBERT_PAIRED_ERROR_AUDIT_PLAN.md` tại `/home/kiettqa/GoiBERT`. **Tự thực thi phần trích xuất/kiểm số, không tạo subagent.** Đây là audit read-only từ validation predictions đã lưu; không chạy model, không dùng GPU/test và không sửa nhãn.

1. Dùng đúng **6 prediction files** và loader text validation tại §2: Restaurant, VSFC Topic, ViSFD; BamiBERT gốc E14 và CafeBERT seed 42. Hash nguồn, join theo stable ID, kiểm gold/shape/row counts. Tính lại **sáu F1** bằng evaluator và phép đếm độc lập; nếu lệch, dừng báo blocker.
2. Tạo `aggregate.json` và `events.jsonl` đúng §3: Topic per-class/confusion/paired outcomes; ABSA missed, spurious, wrong polarity, TP/FP/FN, support/F1 theo aspect/polarity. Giữ mapping ABSA đúng `0/1/2/3 = absent/positive/negative/neutral`; ViSFD OTHERS raw4↔model1 chỉ detection, không vào pair F1.
3. Lấy **72 mẫu** theo quota/cách chọn cố định §4, gồm cả Cafe-only, Bami-only, both-wrong và both-correct. Mỗi mẫu có raw text, gold và hai predictions đầy đủ, event target, ID/hash; không tự diễn giải nguyên nhân hoặc đổi nhãn. Báo quota thiếu nếu có.
4. Viết `docs/CAFEBERT_BAMIBERT_PAIRED_ERROR_AUDIT_PACKET.md` và các artifacts mới ở §5. Báo rõ những phép kiểm pass, số liệu mô tả và giới hạn của mẫu chọn có điều kiện. **Dừng và bàn giao cho root đọc/diễn giải.**

Không đề xuất kiến trúc hoặc train pilot từ packet; phần chọn giả thuyết nghiên cứu là việc của root theo `AGENTS.md`. Giữ nguyên toàn bộ source/gold/predictions/checkpoints lịch sử.
