# Quyết định gate của root — 2026-09-08

Root review plan, [E0_ENVIRONMENT_REPORT.md](E0_ENVIRONMENT_REPORT.md), [E0_DATA_METRIC_AUDIT.md](E0_DATA_METRIC_AUDIT.md), artifacts và read-only protocol review. Subagents thực hiện mọi smoke/audit/model run.

## Đã chốt

1. Environment/checkpoint gate PASS. Precision E1 = BF16 trên NVIDIA L40S, effective batch32. Profile full train-step synthetic length256/512 đạt peak allocated khoảng2.6/3.8GiB. GPU0–3 được người dùng cho phép dùng chung workload hiện có; không kill process. Kiểm tra >=8GiB free trước mỗi launch, pilot GPU3.
2. VSFC sentiment/topic: cho phép supervised validation screening bằng fixed-label macro F1. F1 paper chưa xác minh averaging nên không báo numeric gap hay claim beat paper lúc này; accuracy có thể đối chiếu đúng phạm vi sau test gate.
3. ViSFD: phê duyệt policy internal gồm 10 slot4-class và OTHERS binary riêng. Raw OTHERS4 đổi thành model1 và đổi ngược khi score; loss học cả absent0. Detection gồm OTHERS, pair micro/macro bỏ OTHERS; presence F1 OTHERS báo riêng. Không gắn policy nội bộ với evaluator paper.
4. Restaurant/Hotel: phê duyệt internal validation training trên nguyên các CSV public đã hash, cố định số0..3 và0=absent. Semantic polarity names theo mapping audit còn provisional; không đổi labels theo suy đoán. Provenance tới release gốc và metric equivalence vẫn chưa được xác nhận. Không công bố paper-equivalent F1.
5. Giữ nguyên duplicate/conflicting rows: VSFC train một cặp xung đột sentiment; Hotel có trùng trong train và overlap train/dev, train/test như audit. Không tự sửa official splits để cải thiện score. Ghi rõ dữ liệu mirror không được coi là independent holdout đã chứng minh.
6. Mọi trial lưu giới hạn provenance: backbone kế thừa validation text từ corpus domain cũ; test không được domain builder cố ý nhập nhưng overlap CC100 chưa audit. Không gọi validation là chưa từng được encoder tiếp xúc.

## Full pilot được thông qua

Operator đã trả `15 passed` ở integration tests và BF16 VSFC smoke hoàn tất2epochs trên24train/24validation, không test access. Root đọc artifacts và thông qua pilot VSFC sentiment30epochs seed42 LR2e-5, BF16/SDPA, batch32, GPU3. Code hash được smoke: `d3dff1d2a70fd50ebe00aa9c4e96ea63507c2e0b1bfdc4d68ad6a479819bbded`.

Operator phối hợp pipeline agent khóa source/config trước launch. Nếu còn thay đổi mã sau hash trên, chạy lại focused tests/smoke trên final hash rồi được tự launch theo gate này, không cần chờ root hỏi lại. Cấu hình không được thay khác recipe. Ghi exact command/output/PID/logs. Pilot hợp lệ được tái dùng như một screening trial; nếu code/protocol thay sau pilot thì không trộn vào grid phiên bản mới. Sau epoch đầu full train/validation hợp lệ, báo root để mở các trial khác.

## Test hiện chưa được mở

Hiện chỉ cho phép training/validation. Sau đủ screening/shortlist/final validation, root mới khóa selection manifest. Không có test inference, test F1 hoặc E2 tự động được phép từ tài liệu gate này. Root sẽ review evidence mới trước các bước đó, không cần hỏi lại người dùng cho phần đã được giao.

## GO cho toàn bộ E1 validation

Pilot đã launch trên GPU3, PID4083334, frozen source `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`. Root đọc epoch artifacts tới epoch5: full1583validation rows, loss hữu hạn/giảm, best fixed-label macro F1 0.842571 tại epoch4. Đây chỉ là interim validation một seed, không dùng để claim paper.

Cho phép agent vận hành nối toàn bộ screening→shortlist→final VALIDATION cho năm task theo plan tối đa55trials, BF16/SDPA batch32 trên GPU0–3. Orchestrator phải đợi pilot hoàn tất30epochs và kiểm metadata, tái dùng pilot hợp lệ, không chạy trùng hoặc tính partial pilot như completed. Tạo stage indexes mới đúng frozen hash; bỏ index/smoke cũ khỏi selection. Dừng khi run lỗi, dữ liệu/source hash đổi, thiếu seed hoặc tài nguyên không đủ. Không được tự mở test/E2.
