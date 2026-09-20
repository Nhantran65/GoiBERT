# Rà lại E3 và cách tính presence

## E3 đã kiểm gì

Source `src/goibert_e3/losses.py` giữ head và CE bốn lớp gốc, rồi cộng0.5 CE trên ba logits polarity của gold-present slots. Nó không thay kiến trúc thành hai head độc lập. Kết quả audit cũ: Restaurant giảm wrong-polarity156→123 nhưng missed tăng163→201; ViSFD wrong-polarity318→295, missed460→498. Đây là trade-off của objective/recipe E3, không bác bỏ mọi kiến trúc tách presence và polarity.

Với z0 là absent, z1..3 là present, đặt r=logsumexp(z1..3)-z0 và q=softmax(z1..3). Phân phối bốn lớp tách chính xác thành P(absent)=1-sigmoid(r), P(c)=sigmoid(r)q_c. CE gốc do đó bằng detection BCE cộng conditional polarity CE khi gold present. E3 chỉ tăng hệ số term polarity trong phân tích này. Việc factorize probability là một đồng nhất thức, không phải phát minh.

Tham số hóa r và q bằng hai nhánh độc lập là một thay đổi khác với E3; nó có các tiền lệ hierarchical classification. Không suy rằng chưa chạy local thì mới về khoa học.

## Kiểm độ nhạy của decoder từ cache

Root dùng six-case FP32 logits đã lưu và tính score bằng float64 để so max-present-minus-absent với logsumexp-present-minus-absent. Polarity argmax giữ nguyên. Quét global threshold chỉ để chẩn đoán, không chọn threshold cho test hoặc báo benchmark win.

| Model/task | Max-margin oracle F1 | Logsumexp-presence oracle F1 |
|---|---:|---:|
|Student Restaurant|71.730|71.596|
|Student Hotel|73.049|72.894|
|Student ViSFD|82.497|82.538|
|Bami Restaurant|75.167|75.060|
|Bami Hotel|73.554|73.318|
|Bami ViSFD|84.627|84.641|

Artifact `artifacts/research_diagnostics/reader_operating_point/root_presence_score_sensitivity.json`. Không có gain rộng từ đổi score này. Những chênh lệch nhỏ không có ý nghĩa thống kê được xác lập. Kết quả không loại learned calibration hoặc một tham số hóa khác, nhưng không hỗ trợ việc chỉ đổi rule decode để giải thích/sửa lỗi đã đo.

## Quyết định

Sửa cách diễn giải lịch sử: E3 kiểm auxiliary polarity loss, không phải independent-head architecture. Đồng thời không chọn hierarchical decode quen thuộc làm ý tưởng mới. Chưa launch training từ việc rà lại này.
