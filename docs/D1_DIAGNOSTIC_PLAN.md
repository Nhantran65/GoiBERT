# D1 — chẩn đoán biểu diễn teacher/student theo layer

Người dùng duyệt lập plan và thực thi ngày09/09/2026; ước tính ban đầu45–90phút, không phải deadline đảm bảo. Root thiết kế/đọc kết quả. Ba subagents: d1_extract trích đặc trưng, d1_probes fit linear probes, d1_cka đo similarity. Không fine-tune encoder, không test, không multi-seed, không tải BamiBERT gốc trong vòng này.

## Câu hỏi

Student GD mất khả năng biểu diễn thông tin task ở vùng layer nào so với teacher DAPT? Mapping GD hiện tại có dấu hiệu không phù hợp không? Chỉ tìm vùng/giả thuyết đáng kiểm chứng, không tuyên bố chẩn đoán nhân quả bằng một heatmap.

Teacher dùng models/goibert-b-dapt (BamiBERT derivative sau DAPT), không phải checkpoint paper gốc. Student dùng models/goiturn-s-gd trước task fine-tuning, không E1/E2/E3/E4/E5 task checkpoints. DAPT12layers768dim và GD10layers384dim, cùng tokenizer. Export thiếu GD aligner/training_state; không tái tạo gradient GD lịch sử hay đòi corpus cũ.

## Dữ liệu cố định

Đủ5tasks: VSFC Sentiment,VSFC Topic,ViSFD,Restaurant,Hotel. Mỗi task lấy min(2000,Ntrain) hàng train, uniform without replacement bằng numpy.default_rng(42), sort indices về thứ tự nguồn; dùng toàn bộ validation. Hai model dùng đúng cùng rows/tokenization/labels. Báo class/slot support và lớp không xuất hiện trong subset. VSFC hai task có thể dùng chung features khi IDs/textmapping được xác minh; không làm phức tạp cache nếu không cần.

Chỉ load include_test=False. Không xem test để chọn lớp, pooling, mapping hoặc model. Backbone có provenance tiếp xúc validation không nhãn lịch sử, nên không gọi kết quả là hoàn toàn độc lập với mọi pretraining.

## Cache một lần

Eval/inference_mode, BF16/SDPA, max_length256, studentbatch32/teacherbatch16. Tất cả hidden states gồm embedding layer0 trong cùng forward. Lưu FP32 CLS view chính và content-mean view phụ (validattention, loại all_special_ids; emptyfallbackCLS). CLS layer0 là vector hằng theo câu, đây là điều dự kiến chứ không phải encoder collapse.

Paths: artifacts/d1/cache/<task>/<student|teacher>/{train_cls,validation_cls,train_mean,validation_mean}.npy, shape[L,N,H]; manifest completed sau khi write atomic đủ4arrays. Dataset metadata lưu trainindices,IDs,raw/modelgold,classschema; cache lưu source/checkpoint/tokenizer/data hashes,precision/masks. Student11states,teacher13states.

GPU2/1 cho extraction, GPU3 cho linearheads nếu đủ5GiB free; không kill workload khác. CUBLAS_WORKSPACE_CONFIG=:4096:8 trước CUDA. CPU BLAS giới hạn4threads cho analyses.

## Linear probes

Chỉ CLS view, tất cả layer và5tasks. Một linear classifier, không hidden/dropout. TRAIN-only featurestandardization (epsilon1e-6) rồi chia sqrt(hidden width), lưu policy. Fit fixed200full-batch AdamW steps, LR0.01,weight_decay0.01,seed42; dùng cùng initializer/policy cho mọi model/layer. Không chọn epoch/LR bằng validation. CE thông thường cho VSFC, average-slot CE cho ABSA; giữ nativeclassorder, ViSFD OTHERS slot5width2. Evaluate bằng frozen evaluator.

Báo finaltrainCE, curve và cải thiện20steps cuối; flag chưa hội tụ nếu còn giảm đáng kể (gợi ý >1%relative), không đánh đồng underfitprobe với thôngtinencoderthiếu. Majority baseline từtrain. Labelshuffle control chỉ layercuối mỗi model/task, shufflewholelabelrows cùngseed314159 để giữjointfrequencies; cùngtrainingbudget. Native768/384 tạo probeparametercounts khác: chênhlệchscore không thuần do qualityrepresentation. Chưa thêm PCA/controlgrid; root xem kết quả rồi quyết định kiểm tra giới hạn nào thật sự cần.

## CKA

TRAIN-only, min(1024,Nsubset) cùngsampleindices cho cảhai, seed123. LinearCKA float64centered; cross11×13 vàwithinmodel. CLS primary, contentmean secondary, không chọnpooling từheatmap. Centerednormzero/nearzero =>undefined/NaN có ghi rõ, không gán0. Minimalanalyticchecks: identity1,orthogonal/scaleinvariance,constantundefined. So với GDmapping[0,1,2,4,5,6,7,8,10,11,12], nhưng teacher3/9 khônglàdirecttarget không có nghĩa thôngtinbị bỏ.

Đo similarity không trực tiếp đo taskquality. CKA thấp không chứng minh bottleneck; layer cóCKAcao cũng có thể mã hóa tínhiệu không hữuích.

## Bàn giao và quyết định

Cache bắt đầu dùng ngay khi từng task/model hoàn tất. Deliverables: scripts/d1_*.py, artifacts/d1/{cache,probes,cka}, chartsPNG/SVG và báo cáo root docs/D1_READOUT.md. Root đọc probeF1 theo layer cùng CKA, kiểm tra hội tụ/control và chọn tối đa một giảthuyết có dự đoán kiểm chứng được. Chưa tự chạy thí nghiệm sửa layer, đổi GD hoặc fine-tune encoder.

Nếu không có vùng nhất quán giữa task/method, kết luận chưa khoanh vùng được; không chọn layer tùy ý. Bằng chứng có thể chỉ vào pretraining/GD toàncục hoặc probe hạnchế. Một intervention cóđối chứng sẽ cần để xác nhận cơchế.
