# Quyết định sau chẩn đoán validation — 2026-09-09

Trạng thái: quyết định nghiên cứu của root sau khi đọc kết quả chẩn đoán; chưa khởi chạy E3.

## Cập nhật thực thi

Người dùng đã duyệt triển khai ngày09/09/2026 sau khi xác nhận chỉ thay loss. Kế hoạch thực thi hai trial tại [E3_EXECUTION_PLAN.md](E3_EXECUTION_PLAN.md). Các câu “chưa triển khai” bên dưới ghi trạng thái lúc chốt đề xuất; trạng thái mới theo E3 execution/readout.

## Bằng chứng đã đọc

Teacher được chọn từ E2 có validation pair F1 84.3570 trên ViSFD và 74.6212 trên Restaurant; student NC trung bình seeds42–44 lần lượt 81.2330 và 69.3902. Teacher chỉ một seed và đã chọn LR bằng validation; đây là khoảng cách mô tả, không phải bằng chứng thống kê hay kết luận về dung lượng student.

Train có 60,080/85,646 slot absent trên ViSFD (70.1492%) và 74,878/84,336 trên Restaurant (88.7853%). Loss hiện tại tính trung bình các slot, bao gồm absent. Tỷ lệ nhãn không tự chứng minh tỷ trọng gradient hoặc nguyên nhân KD kém.

Trên Restaurant, số aspect dự đoán thừa của E1 → NC theo seeds42/43/44 là 184→171, 184→179, 186→163. Lỗi polarity khi cả gold và prediction đều present là 156→162, 150→161, 146→137. NC cải thiện detection nhưng chưa cải thiện polarity nhất quán.

Nguồn: `artifacts/diagnostics/teacher_gap.json` và `artifacts/diagnostics/validation_errors.json`. Chỉ train/validation được dùng cho chẩn đoán. Root kiểm tra source/config E1 vẫn có hash `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.

## Quyết định và đề xuất E3 screening

Chưa tăng FFN. Teacher tốt hơn student không tách được ảnh hưởng của pretraining, kích thước, tối ưu và task loss. Argmax predictions cũng không chứng minh gradient KD bị chi phối bởi absent.

**Giả thuyết:** bổ sung giám sát polarity chỉ trên các aspect có mặt trong nhãn train giúp student tăng end-to-end pair F1 mà giữ nguyên kiến trúc. Đây chưa phải kết quả E3.

Thiết kế đề xuất, chưa triển khai:

1. Hai trial: ViSFD và Restaurant, seed42; fresh `models/goiturn-s-gd`, cùng head/tokenizer/decoder argmax/max_length256 và optimizer/scheduler E1. LR1e-4, 30epochs, effective batch32, BF16/SDPA. Đối chứng chính E1 CE seed42 đã có; NC seed42 là mốc phụ về lợi ích thực dụng.
2. Loss: `L = CE_E1 + 0.5 * CE_polarity_present`. Với mỗi slot gold>0 có polarity, lấy ba logits polarity (bỏ logit absent), tính CE rồi trung bình trên tất cả slot hợp lệ trong batch. ViSFD OTHERS bị loại khỏi term phụ; batch không có slot hợp lệ dùng zero có kết nối gradient. Giữ CE gốc để vẫn học phát hiện aspect/absent. Không thêm KD/NC để cô lập một thay đổi loss.
3. Mapping theo dataset: ViSFD1=positive,2=negative,3=neutral; Restaurant1=negative,2=neutral,3=positive. Ba logits giữ thứ tự gốc, nhãn phụ=gold-1. Không áp polarity mapping chung.
4. Chọn checkpoint bằng validation pair micro F1 và tie-break E1. Lưu CE gốc/term phụ riêng, detection precision/recall/F1, pair F1 và confusion polarity. Conditional polarity accuracy chỉ là chẩn đoán vì tập predicted-present thay đổi giữa mô hình.
5. Ngưỡng screening trước chạy: pair F1 tăng ít nhất 0.3 điểm so với E1 cùng seed; báo delta với NC để tránh tiếp tục hướng chỉ bắt kịp E2. Chỉ gọi là hướng chung có tín hiệu nếu tăng ở cả hai task. Đây là tiêu chí thực dụng, không phải significance. Chỉ thêm seeds43/44 ở task hứa hẹn, không quét nhiều hệ số.
6. Nếu polarity tốt hơn nhưng pair F1 không tăng, chưa đạt mục tiêu; xem detection trade-off. Nếu cả hai task không cải thiện, dừng nhánh và cân nhắc capacity hoặc cách truyền teacher có kiểm soát. Không dùng test để chọn coefficient/checkpoint.

Phép thử không cần corpus hay teacher inference, chỉ thêm loss nhỏ vào một forward. Nó kiểm tra một yếu tố có dấu hiệu trong chẩn đoán, không hứa vượt BamiBERT. Việc triển khai/chạy E3 là bước tiếp theo riêng, chưa thực hiện trong đợt chẩn đoán được giao.

## Kiểm tra hoàn tất

Hai subagents đã bàn giao báo cáo. Predictions được căn chỉnh stable IDs/gold và tính lại metric, không inference mới. Root phát hiện và yêu cầu sửa mẫu số conditional polarity accuracy ViSFD để loại OTHERS; bản cuối đã sửa (E1 seed42:2662/(2662+318)=89.3289%). Pair F1 chính không bị lỗi mẫu số này.

Báo cáo chi tiết: [TEACHER_STUDENT_DIAGNOSIS.md](TEACHER_STUDENT_DIAGNOSIS.md), [VALIDATION_ERROR_DIAGNOSIS.md](VALIDATION_ERROR_DIAGNOSIS.md). ViSFD/Restaurant validation không có hàng vượt256 tokenizer tokens, nên không có bằng chứng truncation là nguyên nhân lỗi hiện tại. Teacher-gap agent báo10 validation contract tests passed; error-diagnostic script compile thành công.
