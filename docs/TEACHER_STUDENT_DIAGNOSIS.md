# Chẩn đoán khoảng cách teacher–student trên validation

Ngày phân tích: 2026-09-09.

Phân tích này trả lời câu hỏi: teacher đang cung cấp tín hiệu có ích đến mức nào, và student đang mất những dự đoán nào khi học từ teacher? Tất cả số liệu bên dưới chỉ dùng validation.

## Phạm vi và provenance

Phân tích đọc các `validation_predictions.json` và `metadata.json` đã lưu:

- Teacher: checkpoint tốt nhất theo validation trong ba learning rate E2, seed 42, chọn riêng cho từng task.
- E1: checkpoint seed 42 ở learning rate đã chốt bằng trung bình seeds 42–44.
- E2: KD và noise consistency (NC), mỗi phương pháp dùng coefficient được chọn trên validation; báo cáo cả seed 42 và trung bình seeds 42–44.
- Dataloader được gọi với `include_test=False`. Script không mở checkpoint, không chạy model và không đọc test prediction artifact nào.

Script tái lập là [`scripts/diagnose_teacher_gap.py`](/home/kiettqa/GoiBERT/scripts/diagnose_teacher_gap.py). Kết quả đầy đủ, gồm per-aspect counts và các ví dụ validation, nằm ở [`artifacts/diagnostics/teacher_gap.json`](/home/kiettqa/GoiBERT/artifacts/diagnostics/teacher_gap.json).

## Metric và định nghĩa transition

Với VSFC, một hàng đúng khi class dự đoán bằng nhãn gold. “Teacher đúng, student sai” và “teacher sai, student đúng” là các transition trên cùng một hàng validation.

Với ViSFD và UIT-ABSA:

- Detection đúng khi trạng thái có aspect khớp: `label > 0`. Slot OTHERS của ViSFD được tính ở detection.
- Positive pair hit khi gold là aspect dương và nhãn polarity dự đoán trùng chính xác gold. Đây là đơn vị tương ứng với pair micro F1; OTHERS của ViSFD bị loại khỏi pair vì không có polarity.
- Các cặp trong bảng là `teacher hit → student miss` và `teacher miss → student hit`. Nhãn “miss” ở pair chỉ nói rằng student không tạo được positive pair hit; true negative không được tính là pair hit.

## Mức teacher đạt được

Teacher được chọn theo validation ở các learning rate: Sentiment 2e-5, Topic 1e-5, ViSFD 5e-5, Restaurant 2e-5, Hotel 5e-5. Bảng dùng teacher seed 42; E1/KD/NC bên cạnh là trung bình seeds 42–44 nên đây là chẩn đoán định hướng, không phải so sánh multi-seed đối xứng.

| Task | Teacher val | E1 mean | KD mean | NC mean | Metric |
|---|---:|---:|---:|---:|---|
| VSFC sentiment | 85.923 | 84.491 | 84.395 | 84.927 | macro F1 |
| VSFC topic | 82.173 | 80.104 | 81.064 | 79.796 | macro F1 |
| ViSFD | 84.357 | 80.247 | 80.262 | 81.233 | pair micro F1 |
| Restaurant | 74.621 | 68.984 | 68.397 | 69.390 | pair micro F1 |
| Hotel | 73.216 | 72.070 | 71.638 | 72.681 | pair micro F1 |

Teacher có khoảng cách đáng kể so với student ở ViSFD và Restaurant. Điều này cho thấy teacher fine-tune được tín hiệu task tốt hơn, nhưng chưa nói khoảng cách đó hoàn toàn do kiến trúc; teacher là model lớn hơn và đang được fine-tune riêng.

## Transition trực tiếp trên seed 42

Các số dưới đây so sánh với cùng teacher seed 42 và cùng các hàng validation.

| Task / unit | Student | Teacher đúng → student sai | Teacher sai → student đúng | Net student |
|---|---|---:|---:|---:|
| Sentiment / row | E1 | 29 | 21 | -8 |
|  | KD | 21 | 19 | -2 |
|  | NC | 36 | 20 | -16 |
| Topic / row | E1 | 53 | 30 | -23 |
|  | KD | 26 | 20 | -6 |
|  | NC | 69 | 37 | -32 |
| ViSFD / detection slot | E1 | 435 | 273 | -162 |
|  | KD | 420 | 262 | -158 |
|  | NC | 393 | 268 | -125 |
| ViSFD / positive pair | E1 | 302 | 114 | -188 |
|  | KD | 274 | 115 | -159 |
|  | NC | 252 | 114 | -138 |
| Restaurant / detection slot | E1 | 158 | 89 | -69 |
|  | KD | 167 | 87 | -80 |
|  | NC | 142 | 89 | -53 |
| Restaurant / positive pair | E1 | 127 | 73 | -54 |
|  | KD | 134 | 66 | -68 |
|  | NC | 132 | 75 | -57 |
| Hotel / detection slot | E1 | 138 | 141 | +3 |
|  | KD | 163 | 178 | +15 |
|  | NC | 138 | 150 | +12 |
| Hotel / positive pair | E1 | 80 | 47 | -33 |
|  | KD | 112 | 57 | -55 |
|  | NC | 72 | 62 | -10 |

Với ViSFD, NC giảm số lần student làm mất một detection đúng của teacher từ 435 xuống 393 và giảm số lần sửa lỗi teacher từ 273 xuống 268; tuy nhiên student vẫn mất ròng 125 slot. Ở pair, NC giảm mất hit từ 302 xuống 252 nhưng chỉ tạo thêm 114 hit ở nơi teacher sai. Restaurant có cùng xu hướng yếu hơn: NC tốt hơn E1 ở detection transition (142/89 so với 158/89), nhưng pair vẫn mất ròng 57 hit.

## E1 → E2: phương pháp có thực sự sửa lỗi cũ không?

Bảng này đổi reference: cột đầu là số đơn vị E1 đang đúng nhưng E2 làm sai; cột sau là số đơn vị E1 đang sai mà E2 sửa được. Đây là transition seed42 giữa hai student trên cùng validation rows, không phải so sánh với teacher.

| Task / unit | E2 method | E1 đúng → E2 sai | E1 sai → E2 đúng | Net E2 |
|---|---|---:|---:|---:|
| Sentiment / row | KD | 5 | 11 | +6 |
|  | NC | 27 | 19 | -8 |
| Topic / row | KD | 28 | 45 | +17 |
|  | NC | 37 | 28 | -9 |
| ViSFD / detection slot | KD | 202 | 206 | +4 |
|  | NC | 239 | 276 | +37 |
| ViSFD / positive pair | KD | 112 | 141 | +29 |
|  | NC | 122 | 172 | +50 |
| Restaurant / detection slot | KD | 83 | 72 | -11 |
|  | NC | 82 | 98 | +16 |
| Restaurant / positive pair | KD | 67 | 53 | -14 |
|  | NC | 73 | 70 | -3 |
| Hotel / detection slot | KD | 124 | 136 | +12 |
|  | NC | 92 | 101 | +9 |
| Hotel / positive pair | KD | 75 | 53 | -22 |
|  | NC | 42 | 65 | +23 |

NC có bằng chứng rõ nhất ở ViSFD: số positive pair được sửa nhiều hơn số pair bị phá 50 hit. Ở Restaurant, NC chỉ gần cân bằng ở pair (70 sửa, 73 phá), nên điểm tăng trung bình nhỏ. KD nổi bật ở Topic và Sentiment seed42, nhưng dấu hiệu này cần giữ cách diễn giải thận trọng vì phương pháp và coefficient đã được chọn bằng validation.

JSON cũng lưu transition E1→E2 được tách theo trạng thái của teacher. Ở ViSFD positive pair, NC có 82 phá/132 sửa trong các slot teacher đúng và 40 phá/40 sửa trong các slot teacher sai. Ở Restaurant, NC có 48/43 khi teacher đúng và 25/27 khi teacher sai. Điều này củng cố rằng NC có tín hiệu thật trên ViSFD, còn Restaurant chưa có cải thiện pair rõ ràng.

Các slot ViSFD còn gây mất positive pair nhiều nhất dưới NC là SER&ACC (29 mất, 7 sửa), GENERAL (43 mất, 21 sửa), PERFORMANCE (44 mất, 25 sửa) và FEATURES (28 mất, 14 sửa). Với Restaurant là FOOD#QUALITY (25/10), SERVICE#GENERAL (16/2), FOOD#STYLE&OPTIONS (26/16) và RESTAURANT#GENERAL (16/13). Đây là các vùng lỗi có thể dùng để kiểm tra sau khi thử giả thuyết mới.

## Một vài ví dụ VSFC

Các ví dụ này chỉ minh họa transition, không được dùng để chọn model:

- Sentiment, teacher đúng student sai: “truyền đạt kiến thức .” — gold neutral (1), teacher 1, E1 student 2.
- Sentiment, teacher sai student đúng: “cô rất nghiêm .” — gold negative (0), teacher 1, E1 student 0.
- Topic, teacher đúng student sai: “đồng thời , bài giảng , theo em nhìn nhận , thực sự rất chi tiết .” — gold training program (1), teacher 1, E1 student 0.
- Topic, teacher sai student đúng: “giáo trình chưa cụ thể .” — gold training program (1), teacher 0, E1 student 1.

## Nhận định và hướng tiếp theo

Kết quả không ủng hộ giả thuyết rằng student chỉ cần một loss KD tốt hơn là sẽ tự động bắt kịp teacher. Teacher đúng hơn student trên phần lớn transition quan trọng, nhất là ViSFD và Restaurant. NC có tín hiệu tốt hơn KD trên hai task này, nhưng phần cải thiện chủ yếu là giảm một phần lỗi mất dự đoán của teacher.

Chẩn đoán này chưa chứng minh nguyên nhân là thiếu năng lực biểu diễn. Nó chỉ cho thấy khoảng cách còn nằm ở các lỗi task cụ thể, đặc biệt là positive pair ở ViSFD và Restaurant. Hai giả thuyết có thể kiểm tra tiếp là: (1) điều chỉnh cách học detection/polarity để xử lý mất cân bằng giữa absent và positive labels; (2) giữ nguyên recipe và tăng FFN encoder từ 1536 lên 3072 để kiểm tra giới hạn capacity. Cả hai cần được chạy như screening seed 42 trên ViSFD và Restaurant, có E1 CE làm đối chứng; chỉ mở rộng task hoặc thêm seeds nếu có tín hiệu rõ.

## Giới hạn

- Teacher là một checkpoint seed 42 được chọn theo validation; chưa có variance của teacher.
- E2 coefficient và checkpoint đều được chọn dựa trên validation, nên các transition là chẩn đoán conditional trên lựa chọn đó.
- Chỉ dùng argmax prediction; chưa đo soft-logit calibration hay mức teacher uncertainty được truyền qua KD.
- Dataset ABSA là public candidate mirror đã được audit trong project; diễn giải vẫn phụ thuộc metric contract hiện tại.
