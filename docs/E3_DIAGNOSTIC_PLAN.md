# Chẩn đoán trước E3 — 2026-09-09

Người dùng đồng ý thực hiện chẩn đoán teacher–student và lỗi validation trước khi chốt thí nghiệm tiếp theo. Root điều phối, đọc thiết lập và tổng hợp; hai subagents thực hiện phân tích từ predictions đã lưu. Giai đoạn này không train, không chạy inference mới.

## Câu hỏi và phân công

1. `teacher_gap`: xác định teacher thực sự được chọn để tạo cache từ selection/approved records; so sánh validation teacher seed42, E1 và E2 KD/NC seed42 trên năm task. Bổ sung mean student seeds42–44, ghi rõ teacher chỉ có một seed. Đếm teacher đúng/student sai và chiều ngược lại, phân biệt đơn vị câu, slot, aspect-polarity pair. Kiểm tra liệu KD sửa được lỗi ở nơi teacher đúng và có đồng thời mất các dự đoán đúng của baseline hay không.
2. `validation_errors`: ưu tiên ViSFD và Restaurant, đối chiếu E1 với NC được chọn ở cùng seeds42–44. Phân rã missed aspect, spurious aspect và wrong polarity khi cả gold/prediction đều có aspect. Thống kê theo aspect, polarity, support train và độ dài câu. Ngưỡng nhóm hiếm/dài dựa trên train và công bố rõ; không suy diễn độ dài từ whitespace thành token hay tỷ lệ truncation.
3. Root: kiểm tra loss/head/decoder hiện tại, rà soát mẫu số của chẩn đoán và viết kết luận cùng một đề xuất thử nghiệm tiếp theo. Dung lượng encoder chỉ là giả thuyết, chưa phải kết luận.

## Hợp đồng dữ liệu và kiểm tra

- Chỉ dùng train để tính support/ngưỡng nhóm và validation để đo lỗi; không phân tích thêm test để chọn phương pháp.
- Dùng artifacts hoàn tất được duyệt; loại smoke, interrupted attempts; không đếm đôi symlink NC sentiment seed42.
- Căn chỉnh stable IDs và gold giữa các mô hình, kiểm tra số hàng và tính lại metric từ predictions.
- VSFC: macro F1 fixed labels. ABSA: detection micro và end-to-end pair micro. ViSFD OTHERS dùng raw gold4/model1, có trong detection nhưng không trong polarity pairs.
- Wrong polarity đóng góp một false positive và một false negative vào pair F1. Không cộng đơn giản tỷ lệ loại lỗi với mẫu số khác nhau.
- Không sửa source/config/checkpoints E1/E2 đã đóng băng. Lưu đường dẫn và hashes đầu vào của phân tích.

## Sản phẩm

- `scripts/diagnose_teacher_gap.py`, `artifacts/diagnostics/teacher_gap.json`, `docs/TEACHER_STUDENT_DIAGNOSIS.md`.
- `scripts/diagnose_validation_errors.py`, `artifacts/diagnostics/validation_errors.json`, `docs/VALIDATION_ERROR_DIAGNOSIS.md`.
- Root tổng hợp tại `docs/E3_DIAGNOSTIC_DECISION.md` sau khi nhận và rà soát hai báo cáo.

## Quy tắc quyết định

Chọn một giả thuyết có dấu hiệu quan sát được, thiết kế so sánh seed42 trên hai task trước; không tự thêm multi-seed hoặc thay kiến trúc trong giai đoạn chẩn đoán. Báo cáo phải phân biệt mô tả lỗi, suy luận nguyên nhân và giả thuyết cần kiểm chứng. Hiệu ứng validation đã qua selection không phải ước lượng độc lập và không chứng minh ý nghĩa thống kê.
