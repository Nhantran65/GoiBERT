# E4 — mở rộng FFN, screening seed42

Người dùng duyệt chạy ngày09/09/2026. Root thiết kế/đọc kết quả; subagent thực hiện chuyển trọng số và train. Giữ quy trình experimental tối thiểu: năm trial, một seed, không test, không multi-seed hoặc quét LR.

## Giả thuyết và cấu hình

Tăng FFN1536 lên3072 trên cả10layer giúp student học task tốt hơn với cùng CE recipe. Hidden384,6attentionheads,10layers,tokenizer/head và decoder giữ nguyên. Tăng thêm11,811,840tham số; phải đo tổng thực tế. Đây là kiểm chứng mở rộng bằng một cách khởi tạo và recipe cụ thể, không phải chứng minh tuyệt đối về capacity nếu thất bại.

Chạy VSFC Sentiment,VSFC Topic,ViSFD,Restaurant,Hotel; seed42,30epochs,LR1e-5 cho VSFC và1e-4 cho ABSA,effbatch32,max_length256,BF16/SDPA,CE gốc. E1 optimizer/scheduler/checkpoint selection/tie-break giữ nguyên. Fresh student gốc mỗi trial, không khởi tạo từ E2/E3 fine-tuned checkpoints.

## Chuyển trọng số

FFN gốc: W2*GELU(W1*x+b1)+b2. Đặt W1'=[W1;W1],b1'=[b1;b1], W2'=[W2/2+D,W2/2-D], b2'=b2. Hai phần D triệt tiêu trước dropout ở khởi tạo, nên hàm FFN giữ nguyên về đại số; sai số số thực cần đo. D là Gaussian độc lập từng phần tử, std0.01*std(W2) mỗi layer, CPUgenerator riêng seed314159. D giúp tránh hai bản sao luôn học giống nhau. Không nhân/chia activation GELU.

Đây là cách mở rộng dựa trên nguyên tắc chuyển trọng số giữ hàm của [Net2Net](https://arxiv.org/abs/1511.05641); perturbation đối nhau là lựa chọn cụ thể của phép thử này, không tuyên bố sao chép toàn bộ protocol paper.

Tạo layer mới dưới fork_rng hoặc bảo toàn RNG: không làm lệch seed/head initialization/shuffle chỉ vì allocation thêm trọng số. Chỉ FFN weights/bias/configintermediate_size thay đổi. Ghi conversionseed/noisescale/actualparametercount/dependencyhash/E4sourcehash trong artifact, không mô tả model như backbone cũ chưa đổi.

## Kiểm tra tối thiểu và chạy

Một preflight: shapes/unchanged weights, FP32 eval encoder trên vài TRAIN câu ngắn/dài có padding với atol1e-4,rtol1e-4 và maxdiff được ghi; gradient hữu hạn và xác nhận symmetry có thể phá khi học; đo memory batch32. Nếu pass, worker được phép chạy ngay, không chờ vòng phê duyệt nữa. Không chạy lại bộ test E1/E2/E3 đã qua.

Worker dùng GPU2 ưu tiên, GPU1/3/0 khi đủ5GiBfree; một worker/GPU, không kill workload khác. Chạy detached có file logs/status, xếp hàng đủ5trial. Frozen E1/E2/E3 và artifacts cũ không sửa. Sau chạy kiểm tra IDs/gold và tính lại validation metrics một lần.

## Đọc kết quả

So sánh E4 với E1 CE cùngseed42 để đo thay đổi kiến trúc; thêm mốc tốt nhất sẵn có từ các nhánh E2/E3 ởseed42 để đánh giá lợi ích thực dụng (ghi rõ nhánh, không dùng test để chọn). VSFCmacroF1,ABSApairmicroF1. Báo params,elapsed,peakmemory; không xem model lớn hơn là cải thiện miễn phí.

Không tự thêm seed. Tín hiệu screening thực dụng là >=0.3điểm trên từng task; đánh giá độ rộng bằng số task cải thiện và mức giảm ở task khác. Chỉ kết luận cho cấu hình khởi tạo/LR này. Nếu đáng tiếp tục mới chọn vòng thử tiếp theo.
