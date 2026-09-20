# Prompt bàn giao — CafeBERT → BamiBERT representation distillation

Hãy đọc `AGENTS.md` và toàn bộ `docs/CAFEBERT_TO_BAMIBERT_REP_DISTILL_PILOT_PLAN.md` tại `/home/kiettqa/GoiBERT`, rồi **tự triển khai pilot đúng scope; không tạo subagent**. Đây là phép kiểm một seed để quyết định có nên đầu tư vào general representation distillation, **khác task-logit KD đã thất bại**. Model triển khai phải là BamiBERT ~103M, một encoder pass; CafeBERT frozen chỉ dùng khi train.

1. Dùng đúng shard FineWeb2 tiếng Việt local trong plan; tokenize lại cho Bami, lọc trùng/decontam text-only với cả năm benchmark, tách holdout theo document, khóa **≥31M vị trí Bami non-pad trong corpus train đã dedup** để hai arm xử lý 30M token và thứ tự M/G như nhau. Không lọc web thành review giả, không dùng corpus review nhỏ ở pilot trước, không tải nguồn khác. Nếu gate dữ liệu §3 fail, viết báo cáo và dừng.
2. Kiểm load sạch BamiBERT MLM head/tied và CafeBERT teacher/tokenizer pinned; căn span ký tự của **cùng raw text** qua hai tokenizer theo §4. Nếu coverage/offset/gradient gate fail, dừng; không tự thay objective.
3. Chạy hai arm học trung gian **M = plain MLM** và **G = cùng MLM + frozen CafeBERT relational hidden-state loss** đúng phương trình, layer, λ, dữ liệu, seed, token/update budget ở §5. Profile GPU, không dừng job khác; lưu checkpoint cuối budget và correctness/metadata. Không thêm teacher hoặc loss khác.
4. Fine-tune **Restaurant, Topic, ViSFD × M/G**, seed 42, recipe E14/validation-only §6; reuse E14 Bami O và CafeBERT references. M/G cùng head-init và data order. Reload best checkpoint, kiểm IDs/gold/F1 từ predictions.
5. Viết `docs/CAFEBERT_TO_BAMIBERT_REP_DISTILL_PILOT_RESULT.md` với O/M/G/Cafe, deltas, data/alignment/gradient checks, runtime và quyết định gate. **Dừng**: không seed khác, Sentiment/Hotel, test, grid, multi-teacher hoặc sửa protocol để cứu kết quả.

Source/data/artifacts mới theo §7; giữ nguyên source, gold và artifacts cũ. Nếu không thể giữ một điều kiện trọng yếu, báo blocker và dừng đúng gate. Không tuyên bố thắng CafeBERT từ validation một seed.
