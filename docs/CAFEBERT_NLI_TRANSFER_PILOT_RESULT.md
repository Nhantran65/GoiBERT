# CafeBERT NLI transfer pilot v1 — Gate D result

Ngày 2026-09-22. Pilot dừng ở Gate D trước mọi preflight, training, model inference, hoặc downstream run.

## Quyết định

**Gate D: KHÔNG ĐẠT.** Mẫu audit cố định 300 dòng có 56 dòng `questionable` và 41 dòng `unclear`, tổng 97 dòng, vượt ngưỡng vận hành đã khóa là 30. `questionable` riêng cũng vượt ngưỡng 30.

Đây là một lượt sàng lọc bởi **một người đọc LLM duy nhất** trên mẫu phân tầng 100 dòng mỗi lớp. Lượt đọc mù prediction nhưng không mù gold; không có người adjudicate và không được diễn giải như tỷ lệ lỗi nhãn của toàn bộ ViANLI. Không sửa, lọc, hoặc đổi nhãn gold theo audit. Kết quả dừng cấu hình pilot này vì rủi ro ngữ nghĩa; nó **không kiểm định và không bác bỏ giả thuyết NLI transfer**.

Các bước Gate C, Gate M, downstream, và mọi GPU job không được chạy. Chi phí GPU là **0.0 GPU-giờ**.

## Nguồn và tính toàn vẹn cơ học

Nguồn là `uitnlp/ViANLI@0fec8d6ecb043a61c609f9b51f80401fdf1e84d3`. Chỉ tải README, train (`8012`) và dev (`1000`); không tải hoặc mở test. SHA256 raw được lưu trong [provenance_manifest.json](../artifacts/cafe_nli_transfer/provenance_manifest.json) và [download_manifest.json](../artifacts/cafe_nli_transfer/download_manifest.json).

Schema `uid/premise/hypothesis/label` chính xác; nhãn chỉ là `entailment/neutral/contradiction` với mapping `0/1/2`. Null, UID trùng, exact duplicate pair, conflicting gold pair, và cross-split exact pair overlap đều bằng 0. Dev có 888 premise chuẩn hóa duy nhất, trong đó 866 cũng xuất hiện trong train.

Grouping train theo premise exact hoặc word-5-shingle Jaccard từ 0.8, nối thêm hypothesis exact, tạo 3311 component từ 8012 dòng. Decontamination so với train/dev của năm benchmark, tổng 50 690 dòng text, tìm thấy 0 match và loại 0 component. Split hash cố định đạt:

| Split | Rows | Components | Gate |
|---|---:|---:|---|
| fit | 6453 | 2661 | đạt |
| select | 752 | 312 | đạt |
| audit | 807 | 338 | đạt |

Audit có 178 premise groups chứa cả entailment và contradiction. Mechanical review xác nhận sample khớp raw, split IDs rời nhau, và toàn bộ 8012 train IDs được tính đúng. Chi tiết nằm trong [root_mechanical_review.json](../artifacts/cafe_nli_transfer/root_mechanical_review.json) và [summary.json](../artifacts/cafe_nli_transfer/summary.json).

## Một số dòng làm rõ lý do dừng

Các dòng dưới đây giữ nguyên ID và text của sample; chúng chỉ minh họa rủi ro được root ghi nhận, không phải gold mới:

| Sample | UID | Gold | Nhận xét |
|---:|---|---|---|
| 15 | `uit_Adver_1849_3_11_10` | entailment | Premise nói điều kiện “nếu anh vẫn ở lại”, nhưng hypothesis kết luận Ronaldo không ở lại. Root đánh dấu questionable. |
| 161 | `uit_Adver_787_2_31_04` | neutral | Hypothesis lặp lại mệnh đề được nêu rõ trong premise về ghi hình 21/4 và phát sóng 24/4. Root đánh dấu questionable. |
| 216 | `uit_Adver_308_1_21_02` | contradiction | Premise nói các hãng bay giảm giá tới 80%; hypothesis nói có giảm giá của hãng máy bay. Root đánh dấu questionable. |
| 286 | `uit_Adver_1_4_21_01` | contradiction | Premise chỉ nhắc một hãng máy ảnh Đức; hypothesis là một sự thật không liên quan về chủ quyền của Đức. Root đánh dấu questionable. |

Toàn bộ 300 nhận xét, lý do, và số đếm theo lớp nằm trong [root_semantic_audit.json](../artifacts/cafe_nli_transfer/root_semantic_audit.json). Sample cố định nằm trong [audit_sample.jsonl](../artifacts/cafe_nli_transfer/audit_sample.jsonl) và [audit_sample.txt](../artifacts/cafe_nli_transfer/audit_sample.txt).

## Phạm vi và kết thúc

Pilot không truy cập test, không thay đổi nhãn, không mở rộng ngân sách, và không chạy runner. Không có kết quả NLI hoặc downstream để diễn giải; vì vậy không có cơ sở nói NLI transfer có ích hay vô ích. Nếu nghiên cứu tiếp tục, cần một quyết định dữ liệu mới của root sau khi xử lý nguyên nhân provenance/semantic này, với ngưỡng Gate D hiện tại được giữ nguyên cho pilot này.

Machine-readable record: [artifacts/cafe_nli_transfer/summary.json](../artifacts/cafe_nli_transfer/summary.json).
