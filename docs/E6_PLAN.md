# E6 — pilot tiếp tục MLM trên student GD

Người dùng duyệt triển khai hướng pretraining. Root thiết kế và tổng hợp; e6_mlm xây corpus/train MLM, e6_finetune dùng encoder cuối để fine-tune CE. Một biến thể backbone, một seed, không test/multi-seed/gridsearch.

## Giả thuyết

Student có thể hưởng lợi từ học trực tiếp dự đoán token bị che, bổ sung cho GD hidden/attention đã có. D1 chưa chứng minh thiếu MLM là bottleneck; đây là phép thử dựa trên khác biệt mục tiêu học, không phải kết luận nhân quả từ probe.

## Corpus và giới hạn dữ liệu

Chỉ lấy text từ TRAIN của VSFC, ViSFD, UIT-ABSA Restaurant và Hotel. Không đếm VSFC Topic lần nữa vì cùng văn bản với Sentiment. Strip whitespace ngoài, exact case-sensitive dedup toàn bộ tập, giữ provenance mọi source ID; loại rỗng/không có token maskable. Không thay đổi các split fine-tune hiện có.

Chia5% text duy nhất thành holdout MLM bằng hash xác định seed42, còn lại train MLM. Holdout này lấy từ downstream TRAIN, không phải validation/test của benchmark. Không dùng nhãn, không thêm corpus ngoài hoặc corpus GD cũ. Sampling đều trên unique texts, không nhân bản theo miền; không nối các review thành đoạn chéo.

## MLM cố định trước chạy

- Init encoder: models/goiturn-s-gd, kiến trúc10L/384H/6heads/FFN1536 giữ nguyên.
- Thêm Roberta MLM head tạm thời, decoder tied word embeddings. Head mới chưa học; encoder load đủ weights, không missing key ngoài MLM head dự kiến.
- Tokenization max256, dynamic padding. Dynamic15% masking:80% mask,10% random,10% unchanged; không chọn original special/pad làm target. Loss trung bình trên tổng số masked tokens, không lấy trung bình batch loss khi số target khác nhau.
- Warmup head:500optimizersteps, encoder frozen và eval, tied decoder embeddings cũng frozen; chỉ train dense/LayerNorm/outputbias của MLM head. AdamW LR5e-4,weight_decay0.01, bias/LayerNorm no decay. Xác nhận encoder bitwise unchanged sau giai đoạn này.
- Joint MLM:2000optimizersteps encoder+head, AdamW LR2e-5,weight_decay0.01 (bias/LN no decay),10%linearwarmup và linear decay,clip1, effective batch32,BF16/SDPA,seed42. Một ngân sách cố định, không chọn thời lượng theo điểm task.
- Tính logits chỉ tại masked positions để tiết kiệm VRAM; kiểm tra tương đương CE ignore_index trên một batch nhỏ. Cách này không thay đổi loss MLM.
- Holdout mask cố định seed1729, tạo không làm lệch RNG training. Đo sau headwarmup và jointsteps500/1000/1500/2000; optional đo head chưa học. Loss MLM giảm không tự chứng minh benchmark tăng.
- Export FINAL encoder sau jointstep2000 cùng tokenizer tại models/goiturn-s-gd-mlm-e6. Không chọn checkpoint bằng benchmark validation. Bỏ MLM head khỏi encoder dùng downstream; số tham số inference giữ nguyên. Ghi ancestry/source/data/model hashes.

## Fine-tune đối chứng

Tất cả5task dùng CÙNG encoder export E6, mỗi task headCLS mới, CE gốc, seed42,30epochs,batch32,max256,BF16/SDPA. LR1e-5 cho VSFC Sentiment/Topic;1e-4 cho ViSFD/Restaurant/Hotel. E1 optimizer/scheduler/selection và decoder giữ nguyên. Không NC/KD/E3/pooling/FFN mới trong phép thử.

E1 source/config/checkpoints và models/goiturn-s-gd không sửa. E6 adapter phải ghi đúng path/hash encoder đã MLM trong resolved config/checkpoint, không mang metadata của encoder GD cũ.

## Thực thi và kiểm tra tối thiểu

GPU2 cho MLM, ≥5GiB free; xác nhận actual peak first batch. Fine-tune dùng GPU2/1/3 khi trống, một worker/GPU, queue theo free-device set. Không kill workload khác. CUBLAS_WORKSPACE_CONFIG=:4096:8 trước CUDA và logs/status durable.

Chỉ kiểm tra corpus provenance/mask/head tie/freeze/finite gradients/encoder export shape trước chạy. Không lặp bộ test cũ. Được phép chạy năm fine-tune ngay khi encoder cuối được xác nhận, không thêm gate. Sau chạy đối chiếu IDs/gold và metrics từ predictions một lần.

## Quyết định

So E6 với E1 cùngseed42 để đo tác dụng của bước MLM; thêm mốc thực dụng NC/KD/E3 đã có, ghi rõ từng nhánh. Ngưỡng screening chú ý: tăng≥0.3điểm F1 trên từng task; xem độ rộng và mức giảm ở task khác, không gọi significance từ mộtseed. Chỉ khi có tín hiệu mới đầu tư corpus lớn hơn hoặc kết hợp cách học khác. Không tự mở thêm trial, seed hoặc test.

Deliverables: corpusmanifest, MLM curves/finalencoder,5fine-tuneartifacts, summarytable và docs/E6_RESULT.md. Root bàn giao kết quả hoàn tất, không chỉ báo đã launch.
