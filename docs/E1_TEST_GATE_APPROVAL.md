# Mở test gate — 2026-09-08

Người dùng đã yêu cầu rõ: “chạy trên test, rồi tạo bảng kết quả benchmark so sánh với BamiBERT”. Root phê duyệt test inference cho đúng25checkpoint đã được chọn bằng validation, năm task × seeds42–46. Các quyết định HOLD test trước đây được thay thế bởi gate này cho phạm vi dưới đây.

## Selection đã khóa trước khi đọc kết quả test

- VSFC sentiment: LR1e-5, seeds42/43/44/45/46.
- VSFC topic: LR1e-5, seeds42/43/44/45/46.
- ViSFD: LR1e-4, seeds42/43/44/45/46.
- UIT-ABSA Restaurant: LR1e-4, seeds42/43/44/45/46.
- UIT-ABSA Hotel: LR1e-4, seeds42/43/44/45/46.

Mỗi checkpoint là best-validation epoch của run30epochs đã hoàn tất. Không lấy last epoch, không chọn lại LR/seed và không retrain. Bằng chứng selection: E1_COMPLETED_AUDIT.md, artifacts/e1/validation_readout.json và approved records. Frozen runner hash `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.

## Thực thi

Subagent test_evaluation tạo manifest immutable đầy đủ checkpoint/metadata/source/data/tokenizer/evaluator hashes trước inference; dùng eval-only runner, BF16/SDPA, max_length và head theo metadata đã khóa. Giữ source/config nguyên trạng. Nếu cần wrapper eval riêng, ghi hash wrapper, kiểm tra contract rồi mới chạy. Không đổi metric để làm đẹp test. GPU0–3 được phép chia sẻ workload hiện có, không kill process. Chỉ evaluate mỗi checkpoint một lần; nếu outputs đã có thì kiểm tra và tái sử dụng thay vì ghi đè.

Đầu ra: test predictions có stable IDs, per-seed metrics, access log, hashes; test_readout.json và báo cáo markdown. Kiểm tra đủ test row counts,25unique(task,seed), sample std ddof1, cùng recipe/source; không suy paired significance từ điểm paper đơn.

## Cách so sánh

Bảng đặt BamiBERT paper reported scores cạnh GoiBERT E1 measured test mean±sample std trên5seeds. Accuracy VSFC có định nghĩa rõ. F1 VSFC nội bộ là fixed-label macro; F1 ABSA detection micro và end-to-end pair micro theo evaluator đã khóa (ViSFD OTHERS detection-only). Paper chưa công bố đủ evaluator/averaging, UIT-ABSA mirror còn hạn chế lineage; phải ghi rõ điều này cạnh bảng. Không gọi hiệu số số học là bằng chứng thắng/thua khi metric equivalence chưa xác nhận.

Test access không còn untouched sau lần này. Không dùng test errors để chọn thay đổi E2; các giả thuyết tiếp theo tiếp tục dựa validation evidence. Root tổng hợp báo cáo từ kết quả agent, không trực tiếp chạy thí nghiệm.
