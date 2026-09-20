# Chuẩn bị benchmark intent 8 lớp

## Phạm vi đã duyệt

Chỉ trích xuất, rà nhãn/trùng lặp và khóa split. Không training, inference qua API, hoặc sửa nghiên cứu kiến trúc GoiBERT. Toàn bộ công việc nằm trong intent_benchmark/. Bundle nguồn được giữ nguyên.

Nguồn: ../tmp/prompt-bench-v1-250-standalone/scripts/ai-verify/packs/prompt-bench-v1/cases.jsonl. Snapshot giới thiệu250cases,512target turns; kiểm trực tiếp có507lượt được gán expectedIntent và5lượt thiếu. Cả250cases có ít nhất một lượt được gán nhãn.

## Input contract

Mỗi sample dự đoán intent của current message. Input gồm các messages có sẵn trước lượt đó trong cùng case và current message, theo thứ tự. Không chứa expectedIntent, primaryIntent, assertions, fixture state, outcomes, source ID hoặc turn tương lai. Metadata truy vết lưu tách khỏi model_input. Không dựng assistant responses không có trong nguồn. Lượt thiếu nhãn có thể làm history nhưng không thành training/evaluation target.

## Nhãn

Giữ8labels theo schema nguồn. Mọi sample cần ledger original/reviewed label, confidence và rationale. Review do AI thực hiện, không gọi human gold. Chỉ sửa khi ngữ nghĩa và rubric hỗ trợ rõ; trường hợp cần history không có phải đánh dấu uncertain và giữ conservative label hoặc quyết định root được ghi lại. Không tự map primaryIntent sang expectedIntent.

Production classifier prompt nằm trong DB không có trong snapshot. Rubric của benchmark phải nêu hạn chế đó, dựa vào taxonomy, fast-path/tests hiện có và quy tắc diễn giải nhất quán. Không tuyên bố phục hồi chính xác prompt production.

## Trùng và split

Seed42, mục tiêu xấp xỉ70/15/15 theo số lượt; grouping theo case, shared source, exact full-input duplicates và các substantive duplicate families đã xác nhận. Không ép tỷ lệ bằng cách tách nhóm. Cố gắng có đủ8labels mỗi split; báo support thực tế và trade-off.

Các câu generic như 'ok' hoặc 'giao [ADDRESS_1]' với history khác nhau phải được báo riêng thay vì tự động nối mọi case thành một nhóm khổng lồ. Model_input trùng hoàn toàn phải cùng split. Mọi overlap current-text được công khai; không gọi split này là text-unique hoặc chứng minh không có mọi dạng leakage.

Placeholder được giữ để bảo toàn anonymization; không khôi phục PII. Đây là shortcut tiềm năng cho PHONE/ADDRESS và phải được ghi trong README. Bộ507lượt vẫn là benchmark nhỏ, không bảo đảm hiệu quả với chat production.

## Điều kiện hoàn tất

507stable sample IDs,5excluded targets có lý do, review ledger đủ507rows, source hashes/version, corrected data và split exports tái lập được, thống kê original/reviewed/per-split, duplicate manifest, kiểm tra group/input/history isolation. Root review các sửa nhãn và trường hợp uncertainty trước khi bàn giao. Không đụng docs/E* hoặc artifacts/ của nhánh research cũ.
