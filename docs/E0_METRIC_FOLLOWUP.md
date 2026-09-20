# E0 metric follow-up: VSFC F1

Ngày: 2026-09-08. Phạm vi: đọc nguồn trực tiếp liên quan đến UIT-VSFC và BamiBERT; không chạy lại BamiBERT, không thay đổi loader/config.

## Kết quả chính

Đã tìm thấy bằng chứng mạnh cho evaluator/report của **bài UIT-VSFC gốc**: dòng `Average` trong Table IX (sentiment) và Table X (topic) là support-weighted average, không phải macro F1.

Trong bài gốc, Table IX báo F1 theo lớp:

| Sentiment | F1 (%) | Số mẫu test hiện có trong split đã tải |
|---|---:|---:|
| Positive | 91.32 | 1,590 |
| Negative | 90.52 | 1,409 |
| Neutral | 33.99 | 167 |

Support-weighted F1 tính từ các số được công bố là:

```text
(1590×91.32 + 1409×90.52 + 167×33.99) / 3166 = 87.9399% ≈ 87.94%
```

Đây khớp chính xác dòng `Average 87.94`. Macro F1 của cùng ba số là khoảng 71.94%, nên dòng này không thể là macro F1.

Table X cũng dùng nhãn `Average` cho bốn topic classes và không gọi là macro/micro. Phần experimental setting nói dùng accuracy, precision, recall và F1 nhưng không đặt tên averaging; vì vậy bằng chứng xác định được phép tính của report gốc qua support-weighted reconstruction, không phải tên API cụ thể.

Kiểm tra tương tự với topic F1 `91.12, 67.19, 88.73, 38.10` và test supports hiện có `2290, 572, 145, 159` cho ra `84.0244% ≈ 84.03%`, đúng dòng `Average 84.03` của Table X.

Nguồn full text công khai do tác giả upload: [UIT-VSFC paper trên ResearchGate](https://www.researchgate.net/publication/329645066_UIT-VSFC_Vietnamese_Students%27_Feedback_Corpus_for_Sentiment_Analysis), xem phần experimental setting và Table IX/X. Trang dữ liệu chính thức chỉ mô tả baseline F1 tổng quát, không có evaluator code: [UIT-NLP datasets](https://sites.google.com/uit.edu.vn/uit-nlp/datasets). Repository của tác giả cũng chỉ cung cấp mô tả/dataset link, không kèm metric implementation: [kietnv/uit-vsfc](https://github.com/kietnv/uit-vsfc).

## Liên hệ với BamiBERT

Không tìm thấy benchmark repository hoặc evaluator riêng của BamiBERT. Model card chính thức chỉ trỏ đến paper và hướng dẫn fine-tuning chung của Transformers; không chứa code tính F1 cho VSFC: [Qualcomm-AI-Research/BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT). Paper BamiBERT chỉ ghi chọn learning rate trên validation, chọn checkpoint theo F1 sau mỗi epoch, rồi báo `F1` trong Table 2; không ghi `macro`, `weighted`, `micro`, công thức, label inclusion hoặc implementation: [BamiBERT paper](https://arxiv.org/html/2607.02259), đặc biệt mục 3.1 và Table 2.

Vì vậy:

- **Đã xác minh:** convention của bài UIT-VSFC gốc là support-weighted average cho dòng `Average`.
- **Chưa xác minh:** BamiBERT 83.41 (sentiment) và 79.90 (topic) có dùng đúng support-weighted convention của bài gốc hay dùng evaluator khác.
- **Không được suy ra:** không dùng việc các target score “trông hợp lý” để kết luận evaluator BamiBERT.

## Tác động lên protocol

E0/E1 hiện vẫn giữ `fixed-label macro F1` làm selection metric nội bộ theo protocol đã khóa, để tránh âm thầm thay đổi thí nghiệm giữa chừng. Nếu root muốn claim VSFC F1 so trực tiếp với BamiBERT, cần một protocol amendment rõ ràng: báo cáo cả `macro_f1` và `weighted_f1`, dùng weighted F1 làm cột candidate cho comparison với target, và đánh dấu rằng equivalence với BamiBERT vẫn là **probable by benchmark convention, not directly verified**.

Accuracy của VSFC không phụ thuộc averaging và có thể đối chiếu trực tiếp hơn. Các điểm F1 paper vẫn giữ trạng thái chưa đủ bằng chứng evaluator trong [E0_DATA_METRIC_AUDIT.md](E0_DATA_METRIC_AUDIT.md).
