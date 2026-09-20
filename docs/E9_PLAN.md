# E9 — Label-aware attention, screening trên cả 5 task

## Giả thuyết và phạm vi

Đọc token theo từng nhãn/aspect, đồng thời giữ thông tin CLS, có thể giúp student thu nhận bằng chứng tốt hơn head CLS chung. Đây là giả thuyết về readout, không phải kết luận đã tìm ra bottleneck hay đề xuất attention mới.

Một subagent GPT 5.6 Luna xhigh triển khai và thực thi; agent chính review trước khi chạy đầy đủ và tổng hợp kết quả. Không tạo agent phụ khác.

## Thiết kế cố định

- Khởi tạo encoder từ `models/goiturn-s-gd` nguyên bản; toàn bộ encoder vẫn trainable. Không dùng checkpoint E6 hoặc checkpoint đã fine-tune.
- VSFC: một query học được cho mỗi class. ABSA: một query cho mỗi aspect, dự đoán các trạng thái của aspect đó.
- Query có chiều 384, khởi tạo normal std 0.02. Attention = softmax(q·h / sqrt(384)), tính softmax FP32; chỉ đọc token hợp lệ, loại padding và special tokens. Nếu không có content token, dùng CLS làm token duy nhất.
- Context là tổng token có trọng số. Readout cho mỗi query = (CLS + context)/2; không thêm gate hay projection key/value.
- Giữ cấu trúc dense/dropout của head cũ: dropout 0.1 → dense 384 → tanh → dropout 0.1, dùng chung dense giữa các query. Mỗi class VSFC có một output row; mỗi aspect ABSA có các output row riêng. VSFC vẫn softmax categorical; ABSA vẫn joint state classification, không đổi loss/decoder.
- ViSFD giữ OTHERS 2 lớp và packed 42 logits; Restaurant/Hotel giữ 4 trạng thái/aspect. Tham số mới nằm dưới `head.*` để nhận đúng LR head.
- Không dùng gold để chọn query; mọi query hoạt động ở train và inference. Lưu đầy đủ architecture metadata để checkpoint có thể được load đúng bằng constructor E9.

## Recipe và ngân sách

Giữ nguyên E8: seed 42, 30 epochs, max length 256, effective/micro batch 32, BF16/SDPA, AdamW, warmup 10%, LLRD gamma 0.8 và NC T=1. LR VSFC 1e-5, ABSA 1e-4; NC lambda lần lượt Sentiment 0.1, Topic 0.5, ViSFD 0.5, Restaurant 0.1, Hotel 0.1. Cùng split, metrics và quy tắc chọn checkpoint trên validation như E8.

Chạy đúng 5 trial, không grid, không thêm seed, không test trong vòng này. Ưu tiên GPU2, kiểm tra tài nguyên trước khi chạy, không can thiệp tiến trình khác. Tách source/config/artifacts E9; không sửa E1–E8.

## Gate trước khi chạy

Kiểm tra nhỏ có ý nghĩa: shape của cả 5 task, mask padding/special và empty fallback, finite loss/gradient đến queries và encoder, encoder init giống GD, optimizer chứa mỗi tham số đúng một lần với LR đúng, checkpoint strict round-trip và metadata. Một smoke ngắn được phép, lưu riêng; agent chính review code và preflight trước khi duyệt chạy 5 trial. Không xây thêm test framework.

## Review kết quả

So sánh validation seed42 với E8 cùng task: Sentiment 85.1604428; Topic 80.3677890; ViSFD 82.4420542; Restaurant 71.2765957; Hotel 73.0490748. Báo cả năm delta, số task tăng, mức giảm lớn nhất, thời gian và peak VRAM. Không dùng trung bình F1 để che task bị giảm.

Ngưỡng screening được chốt trước: tín hiệu cải thiện rộng khi ít nhất 4/5 task tăng, không task nào giảm quá 0.3 điểm và ít nhất 2 task tăng từ 0.3 điểm. Đây là heuristic ưu tiên vòng sau, không phải kiểm định thống kê. Nếu chỉ ABSA tăng, gọi đúng là cải thiện có tính task-specific. Chưa kết luận thắng BamiBERT từ validation; chỉ quyết định có đáng xác nhận/test tiếp hay không.
