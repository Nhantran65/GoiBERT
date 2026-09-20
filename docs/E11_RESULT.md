# E11 — Review kết quả signed attention

Hoàn tất 10/10 trials, 30 epochs mỗi trial, seed42; validation-only. Root đối chiếu trực tiếp status/epoch_curves/best_validation_metrics của cả hai variant với metric E8. Summary có metric tính lại từ predictions theo báo cáo agent. Không chạy test.

| Task | E8 | Signed | Convex | Signed−E8 (pp) | Signed−Convex (pp) |
|---|---:|---:|---:|---:|---:|
| Sentiment | 85.1604 | 84.1369 | 84.1369 | −1.0235 | 0.0000 |
| Topic | 80.3678 | 80.2018 | 80.1303 | −0.1660 | +0.0715 |
| ViSFD | 82.4421 | 82.0272 | 82.0412 | −0.4148 | −0.0140 |
| Restaurant | 71.2766 | 72.4485 | 72.2649 | +1.1719 | +0.1836 |
| Hotel | 73.0491 | 73.4824 | 73.4563 | +0.4334 | +0.0261 |

F1 hiển thị theo thang100, chênh lệch là điểm F1. Không so validation với paper test.

## Nhận định

Signed tăng đủ lớn để chú ý ở Restaurant/Hotel, nhưng giảm ba task còn lại, đặc biệt Sentiment. Không đạt gate cải thiện rộng. Đối chứng convex cũng tăng gần tương tự ở hai task đó: hiện chưa đủ bằng chứng quy phần lớn gain cho khả năng attention có trọng số âm. Chênh signed−convex nhỏ, chỉ một seed; không gọi là khác biệt ổn định hay có ý nghĩa thống kê.

Đây là kết quả âm cho việc thay E8 bằng cấu hình E11 chung, không phải mọi differential architecture đều bị bác bỏ. Không có cơ sở kết luận model đã học loại bỏ distractor. Diagnostic synthetic đã được agent thực thi nhưng chưa được root audit đầy đủ về nhãn/định nghĩa metric; không dùng nó để đưa ra claim cơ chế trong review này.

Giữ E8 làm baseline chung. Lưu hai nhánh Restaurant/Hotel như tín hiệu task-specific. Không mở thêm test/multiseed/grid trong vòng screening thất bại này. Nếu nghiên cứu sâu tiếp, cần lý do mới để phân biệt lợi ích của second attention path với signed combination, thay vì chỉ dựa vào điểm Restaurant cao hơn.

## Đính chính phạm vi diagnostic sau review E12

Root phát hiện bất nhất giữa tên polarity UIT-ABSA trong metadata và cách mã hóa được gợi ý bởi raw examples. `scripts/build_e11_diagnostic.py` dùng positive=3,negative=1; đồng thời gán “phòng sạch sẽ” vào FACILITIES#CLEANLINESS thay vì ROOMS#CLEANLINESS. Vì vậy **không dùng diagnostic synthetic Restaurant/Hotel hiện tại để suy luận chất lượng/cơ chế** cho đến khi audit nguồn và nhãn kỳ vọng xong. Không sửa raw gold/checkpoints. Bảng validation trên được đọc từ numeric predictions/gold, là đường đánh giá khác; không tự suy rằng bảng này sai chỉ vì tên nhãn hoặc diagnostic sai. Xem [review E12](E12_SEMANTIC_REVIEW.md).

Nguồn: `artifacts/e11/summary.json`, `artifacts/e11/{signed,convex}/<task>/seed_42/{status.json,epoch_curves.json,best_validation_metrics.json}`, `artifacts/e8/<task>/seed_42/best_validation_metrics.json`. E11 source hash: `59a09275f66f1dc27ec902a4a728eb086c335ea29eb098c320f73adc97146c38`.
