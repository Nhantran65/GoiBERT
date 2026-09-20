# Sau E5: chẩn đoán trước khi chọn kiến trúc tiếp theo

Ngày09/09/2026. Người dùng đề nghị nghiên cứu BamiBERT và tìm điểm nghẽn thay vì tiếp tục các thay đổi phỏng đoán. Lượt này chỉ đọc paper/config/code/manifests, chưa chạy probe hoặc thí nghiệm GPU mới.

## Điều đã xác minh

BamiBERT chính thức là RoBERTa/BERT-base:12layers,hidden768,12heads,FFN3072,vocab20481,max_position_embeddings2050. Paper mô tả MLM dynamic masking,129GB text,20epochs. Không có layerwise ablation xác định bottleneck. Nguồn: https://arxiv.org/html/2607.02259 và https://huggingface.co/Qualcomm-AI-Research/BamiBERT/raw/main/config.json .

Student hiện tại:10layers,hidden384,6heads,FFN1536. Embeddings khởi tạo PCA; transformer weights khởi tạo ngẫu nhiên rồi general distillation30000steps từ teacher DAPT. GD loss là hidden-state MSE qua projection học được + head-mean attention MSE, không MLM/logits. Nguồn: models/goiturn-s-gd/distill_manifest.json; reference/ai-foundary/experiments/goibert-family-v1/src/general_distill.py và distill.py.

Teacher local models/goibert-b-dapt là BamiBERT đã DAPT40000steps, không phải checkpoint BamiBERT gốc. Chỉ có hai backbone local. Cần phân biệt khi diễn giải kết quả; không gọi điểm của derivative là reproduce BamiBERT paper.

Loss tail ghi hidden khoảng0.081,attention khoảng0.00008. Độ lớn scalar này không chứng minh attention gradient quá yếu; số chiều/reduction khác nhau. Artifact student không chứa training_state/projections nên không thể tái tạo chính xác gradient GD lịch sử từ export hiện có.

E4/E5 cho biết hai biến thể cụ thể chưa có lợi ích rộng, không xác định được nguyên nhân là attention, FFN hay pretraining. Tăng FFN bằng chuyển trọng số rồi supervised fine-tune không tương đương pretrain model lớn từ đầu.

## Vòng chẩn đoán đề xuất, chưa chạy

1. Cache hidden states của student GD và teacher DAPT trên cùng một tập train/validation nhỏ có đủ5task; tokenizer/IDs/masks cố định, không test. Lấy tất cả layer trong mỗi forward. BamiBERT gốc là đối chứng riêng nếu cần phân biệt ảnh hưởng DAPT; tải checkpoint gốc khi thực thi, không thay teacherlocal bằng tên Bami gốc.
2. Linear probes: đóng băng encoder, fit bộ phân loại tuyến tính trên TRAIN features từnglayer, đánh giá VALIDATION cùng metric. Dùng cùngbudget/regularization trong mỗi model, giữ CLS làmviewchính để không trộnthêm giảthuyếtpooling. Báo support và meanmajority/controlbaselines; teacher768 vsstudent384 có probeparametercounts khác, nên không quy tất cả gap cho representation. Có thể dùng train-only reduction làm sensitivitycheck nếu chênhlệch quyếtđịnh lựa chọn.
3. CKA trên representations cùngsample (không cần hiddenwidth bằng nhau) để xem layeralignment GD có dấu hiệu lệch, kết hợp probe curves. CKA đo similarity, không đo trực tiếp kiến thức hay chất lượng. Không lấy layerCKAthấp nhất làm bottleneck tựđộng. Nguồnmethod: https://proceedings.mlr.press/v97/kornblith19a.html . Probeinterpretation: https://aclanthology.org/D19-1275/ .
4. Chỉ khi hai loại bằng chứng cùng chỉ vào một vùng, chọn một can thiệp có đối chứng: ví dụ thay mapping/distill vùng đó, hoặc chỉ cập nhật attention/FFN tại vùng nghi ngờ với budget kiểm soát. Layer removal làm scoregiảm chỉ thể hiện tầm quantrọng, chưa chứng minh layer đó là bottleneck có thể cải thiện.

Ưu tiên đọc chất lượng backbone trước taskfine-tune. Nếu cần phân biệt mất kiến thức lúc GD hay lúc taskadaptation, bổ sung cùngphépđo trên checkpointE1 đã có; không lập tức trainlại5grid.

Đầu ra mong muốn là đồ thị layer→probeF1,teacher–student CKA và một giả thuyết có dự đoán kiểm chứng được. Chưa khởi chạy probe, chưa tuyên bố tìm ra layer nghẽn.

## Bổ sung từ audit code

Mapping GD là [0,1,2,4,5,6,7,8,10,11,12]. Teacher layers3/9 không được dùng làm target trực tiếp; điều này không chứng minh thông tin của chúng bị mất vì các layer sau đã nhận đầu ra của chúng. Không đổi mapping chỉ dựa vào danh sách này.

11 projection384→768 được học cùng student, khoảng3.25M tham số phụ và không nằm trong encoder export. Đây là cơ chế alignment hợp lệ; chưa có bằng chứng projection làm hỏng distillation. Phải đo khả năng đọc thông tin task từ student và mức alignment thay vì suy diễn từ loss nhỏ.

Không có bug padding hiển nhiên trong các MSE đã đọc: hiddenmask theo token hợp lệ, attentionmask theo cặp token hợp lệ. Các giả thuyết hiện tại là về chất lượng tín hiệu GD, chưa phải bug đã được chứng minh. Corpus GD cũ và aligner training_state không nằm trong model export; dùng train texts hiện có cho diagnostic, không hứa tái tạo gradient lịch sử.
