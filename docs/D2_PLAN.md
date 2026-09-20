# D2 — cùng phép đo trước và sau fine-tune

Người dùng duyệt thực thi ngày10/09/2026. Root thiết kế/đọc kết quả; d2_extract lấy đặc trưng sau fine-tune; d2_measure tái sử dụng probes/CKA và tạo báo cáo. Không cập nhật encoder, thêm seed, mở test hoặc chạy kiến trúc/loss mới.

## Câu hỏi

Khoảng cách teacher–student thay đổi ở đâu khi thích nghi task? D1 cho thấy frozen CLS của student không thua rõ teacher trong protocol probe, trong khi teacher fine-tuned có điểm cao hơn. D2 kiểm tra sự thay đổi này, không mặc định nguyên nhân là optimizer, forgetting hoặc capacity.

## Bốn trạng thái so sánh

BEFORE student/teacher dùng nguyên cache D1 (GD student và DAPT teacher trước task fine-tune). AFTER dùng checkpoint E1 CE student seed42 và checkpoint teacher E2 stage0 seed42 đã chọn theo validation, riêng từng task. Không dùng E2 KD/NC hoặc E3–E6. Teacher là BamiBERT derivative DAPT, không phải bản gốc paper.

Cả5task: VSFC Sentiment, Topic, ViSFD, Restaurant, Hotel. Cohort lấy nguyên D1dataset.json:2.000trainrows với cùng indices/IDs/texts và full validation. Không lấy lại sample. Tokenizer/hash/masks/max_length256 và precisionBF16/SDPA giữ như D1.

## Trích đặc trưng

Nạp encoder weights từ đúng checkpoint fine-tuned, strip prefix encoder. rồi strictload vào kiến trúc gốc, không nạp đầu phân loại vào feature cache. Kiểm tra checkpointseed/task/backbone/selection và không missing/mismatched weights. Encoder.eval + inference_mode, không optimizer.

Lấy allstates: student11×H384,teacher13×H768; lưu CLS view chính và contentmean view phụ cùng schemaD1 tại artifacts/d2/cache. D1không bị ghi đè. Manifest ghi actual fine-checkpoint SHA/path/epoch/LR và nguồn trước fine-tune; cùng input fingerprints giữa BEFORE/AFTER.

Khi đã có hai bộ trọng số trong bộ nhớ, đo relative L2 update theo nhóm attention/FFN/LayerNorm từng layer và embeddings. Baseline norm0 báo riêng absolute delta. Đây là mô tả mức đổi trọng số, không phải thước đo trực tiếp tầm quan trọng.

## Probes và CKA

Probes AFTER tái sử dụng nguyên D1fixed200steps/AdamW/LR0.01/wd0.01/seed42/train-only normalization và officialmetrics, tất cả layer. Majority/shuffled-final controls giữ cùng contract. BEFORE200 lấy kết quả D1, không fit lại.

Sensitivity cùngbudget: fit AFTERfinal layer mỗi model/task với1.000steps (10heads), đối chiếu BEFOREfinal1.000 có sẵn từ D1. Không chọn bestepoch; không cần quét lại1000steps mọi layer. Báo convergence/trainingloss để không gọi probe chưa học đủ là thiếu thông tin của encoder.

CKA TRAIN-only1.024 cùngsampleindicesseed123. Đo BEFORE–AFTER trong cùngmodel (matched layer và crossdepth); crossstudent–teacher AFTER nếu rẻ để đối chiếu D1. CLS view chính, mean phụ. EmbeddingCLSconstant=>undefined; không coiNaN là collapse. CKA similarity/weightdrift không tự chỉ ra layer cần sửa.

## Diễn giải và giới hạn

Đặt cạnh nhau: delta probeF1, CKA trước–sau, relative weightchange. Chỉ khoanh vùng nếu tín hiệu nhất quán; một layer thay đổi nhiều hoặc similarity thấp chưa chứng minh learning hỏng. Teacher/student có width và probeparametercount khác nhau, LR fine-tune có thể khác. Các encoder AFTER đã học nhãn từ fulltrain và được chọn checkpoint bằngvalidation: đây là chẩn đoán có điều kiện trên các checkpoint đó, không phải đánh giá generalization độc lập hoặc dự đoán test.

Nếu teacher tăng mạnh hơn student ở các layer sau, đó là dấu hiệu chênh lệch gắn với taskadaptation; chưa phân biệt được capacity,head,regularization hoặc tối ưu. Nếu không thấy vùng nhất quán, kết luận chưa khoanh vùng được. Không tự thay layer, đóng băng encoder hoặc chạy lại fine-tune sau D2.

## Bàn giao

Scripts d2_*.py tối thiểu, artifacts/d2/cache/probes/cka và hình MatplotlibPNG/SVG; root docs/D2_READOUT.md. Chạy tuần tự theo cache từng task để extraction/analysis song song. GPU2/1 cho extraction, GPU3 hoặc GPUtrống cho linearheads; không kill workload khác, CUBLAS trướcCUDA. Không lặp whole-test-suite cũ.
