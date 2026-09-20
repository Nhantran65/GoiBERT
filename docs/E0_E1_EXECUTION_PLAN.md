# GoiBERT — Kế hoạch thực thi E0/E1

Ngày: 2026-09-08. Người dùng đã cho phép triển khai và chạy theo kế hoạch. Root là cộng sự nghiên cứu/điều phối: viết giả thuyết, khóa protocol, review bằng chứng và tổng hợp. Subagents triển khai, kiểm thử và chạy thí nghiệm. Không yêu cầu lại phê duyệt cho các bước thuộc phạm vi này.

## 1. Câu hỏi và phạm vi

H0 thực nghiệm: backbone `models/goiturn-s-gd/` fine-tune supervised-only đạt chất lượng nào trên năm benchmark? E1 tạo baseline đo mới, không kiểm định kiến trúc mới và không lấy kết quả task-distilled lịch sử làm baseline này.

Giữ cùng checkpoint nguồn cho mọi task, mỗi task có toàn bộ encoder fine-tuned và head riêng. Teacher DAPT chỉ smoke trong E0, không tham gia loss E1. Chưa chạy KD, noise consistency, MLM, architecture expansion hoặc train lại BamiBERT. Corpus không cần cho E1.

| Task ID | Train / validation / test | Head | F1 BamiBERT paper (0–100) |
|---|---|---|---|
| vsfc_sentiment | 11426 / 1583 / 3166 | 3 lớp | 83.41 |
| vsfc_topic | 11426 / 1583 / 3166 | 4 lớp | 79.90 |
| visfd | 7786 / 1112 / 2224 | 10 aspect × 4 lớp + OTHERS nhị phân riêng | Detection 89.14; classification 84.24 |
| absa_restaurant | 7028 / 771 / 1938 | 12 aspect slots × 4 lớp | Detection 88.01; classification 74.89 |
| absa_hotel | 7180 / 795 / 2030 | 34 aspect slots × 4 lớp | Detection 79.99; classification 72.65 |

Accuracy VSFC báo riêng: sentiment 93.86, topic 89.34. Các điểm paper là reported reference, không phải đối chứng cùng compute.

## 2. Phân công và hợp đồng bàn giao

- `data_evaluator`: loader, evaluator, fixtures; `docs/E0_DATA_METRIC_AUDIT.md` và `artifacts/e0/data_audit.json`. Xác minh label mapping/OTHERS, source lineage, shape/split, các khác biệt preprocessing và phạm vi metric equivalence. Không train.
- `environment_smoke`: môi trường `.venv`, pin dependencies, GPU inventory, smoke/checkpoint round-trip và profile; `docs/E0_ENVIRONMENT_REPORT.md`, machine-readable environment/smoke artifacts. Không thay weights nguồn, không chạy grid.
- `baseline_pipeline`: runner/config/grid/tests, `docs/E1_RUNNER.md`. Tách prediction và evaluation; triển khai contract validation-only. Chờ root thông qua gate trước full run.
- Root review code/config/audit và giao agent thực thi E1 theo task đã READY. Agents trao đổi interface trực tiếp, không ghi đè ownership. Phát hiện bất nhất phải báo root trước thay protocol.

Không chạy notebook/script tải về từ bên thứ ba; chỉ đọc để kiểm tra nguồn và logic.

## 3. E0 — Gate bắt buộc trước full training

### Dữ liệu và metric

1. Verify downloaded SHA256, số hàng/cột, nhãn hợp lệ, text không trống, text/label alignment và IDs ổn định. Giữ official splits, không shuffle lại split hoặc xóa mẫu để đạt điểm đẹp. Báo duplicates/conflicting labels riêng.
2. Dùng label map cố định từ nguồn; không suy vocabulary bằng test. VSFC có sentiment và topic trên cùng text. ABSA phải xác minh số nào là absent/negative/neutral/positive; ViSFD OTHERS không được mặc định thành positive.
3. Audit có thể kiểm tra cấu trúc/hash/nhãn test; không phân tích lỗi model hay dùng test để chọn preprocessing, max_length, loss, LR hoặc architecture. Screening runner không load test.
4. Metric fixtures phải kiểm tra trường hợp predicted aspect dư/thiếu, sai polarity, absent, zero division; phân biệt micro và macro và conditional với end-to-end.
5. VSFC selection metric: macro F1 trên toàn bộ label IDs. ABSA local selection metric: `pair_micro_f1`, end-to-end aspect-polarity micro F1 trên các aspect có polarity, bỏ OTHERS khỏi tập pair. OTHERS presence F1 báo riêng; detection phải ghi rõ có/không tính OTHERS. Khóa field/key API tương ứng sau audit trước screening. Lưu detection micro, pair micro, per-aspect và macro dưới tên riêng, không tự gắn mọi F1 với paper.
6. Mỗi task ghi `training_ready` và `paper_comparison_ready` riêng. Full ABSA screening yêu cầu mapping và local selection metric đã xác định, nguồn đủ tin cậy để train. Nếu paper averaging chưa được xác minh, chỉ được chạy internal exploratory validation sau quyết định root ghi lại; không tính gap/claim tương đương paper. Task blocked không chặn VSFC.
7. Corpus kế thừa từng chứa validation text. Ghi rõ provenance; không tuyên bố validation chưa từng được encoder tiếp xúc hoặc train-only pretraining. Không có corpus chưa xác minh được contamination cũ.

### Môi trường/model

1. Inspect GPU/driver, active process/memory, disk, Python. Người dùng đã cho phép chia sẻ cả GPU0/2 đang có workload khác (bổ sung 2026-09-08): allocation E1 là GPU0,1,2,3, tối đa một GoiBERT trial/GPU. Giữ nguyên process hiện có, không kill. Trước launch kiểm tra free memory so với peak profile cộng khoảng dự phòng; GPU chưa đủ bộ nhớ thì chờ hoặc dùng microbatch phù hợp giữ effective batch32.
2. Pin môi trường thực tế dùng được, lưu Python/torch/transformers/CUDA/cuDNN/GPU và freeze. Python >=3.13 trong scaffold có thể chỉnh theo Python server được kiểm chứng, ghi lý do.
3. Smoke local teacher và student: tokenizer/config compatibility, finite forward. Student+head: finite loss/grad, optimizer thực sự cập nhật weights, save/load logits round-trip ở eval mode. Weights nguồn bất biến.
4. Profile synthetic batch32 max_length256 và512, peak memory và throughput. Chọn bf16 nếu smoke pass, fp32 fallback nếu không; giữ cố định và ghi precision. Không tự dùng fp16.
5. Runner smoke ít bước trên TRAIN/VALIDATION, save/load và artifact completeness. Kiểm thử không sử dụng test và mỗi trial khởi đầu cùng backbone. Agent thực thi và trả logs cho root.

## 4. Recipe baseline cố định

| Thành phần | Giá trị |
|---|---|
| Optimizer | AdamW, betas (0.9,0.999), epsilon 1e-8 |
| Weight decay | 0.01; bias và LayerNorm weights không decay |
| Epochs | 30 đầy đủ; chưa early stopping |
| Effective batch | 32; microbatch32 mặc định, accumulation chỉ khi cần và ghi rõ |
| Learning rates | 1e-5, 2e-5, 5e-5, 1e-4, 2e-4 |
| Schedule | Linear warmup 10% tổng optimizer steps rồi linear decay |
| Gradient clipping | Norm 1.0 |
| Pooling/head | CLS token → dropout0.1 → dense hidden384 → tanh → dropout0.1 → logits |
| Loss | CE nhãn thật, không class weights/label smoothing; ABSA CE trung bình tất cả slots, bao gồm absent0. ViSFD OTHERS là binary CE presence/absence, không có polarity |
| Padding | Dynamic padding, attention mask; tokenizer checkpoint nguồn |
| Max length | Dựa train-only: 256 nếu >256 tokens chiếm <=1%, nếu không 512. Nếu >512 vẫn >1%, báo root trước full task run |
| Precision | bf16 sau smoke pass, nếu không fp32; khóa cho tất cả trial đối chứng |
| Checkpoint selection | F1 validation tốt nhất mỗi epoch; hòa chọn epoch sớm hơn |

Batch32/30epochs/ba LR thấp theo mô tả paper; các quyết định head, schedule, decay, max_length, precision và hai LR cao là lựa chọn GoiBERT, không khẳng định là config đầy đủ của tác giả.

## 5. E1 — Thứ tự và seed budget

### E1a Screening

Sau E0, chạy VSFC sentiment trước để xác minh luồng thực tế; sau pilot hợp lệ, phân phối các LR/task READY lên GPUs rảnh. Seed42, 5 LR × 5 task = tối đa 25 trials, mỗi trial 30 epochs. Chỉ train/validation. Không chọn trial bằng training loss hoặc test.

### E1b Shortlist

Với mỗi task, lấy top2 LR theo best validation F1 seed42. Chạy thêm seeds43/44 cho cả hai LR (4 trials/task); tái dùng seed42 tương ứng. Chọn LR bằng mean best-validation F1 trên seeds42/43/44. Nếu bằng nhau chọn LR nhỏ hơn. Tối đa20 trials mới.

Các LR cùng seed phải dùng cùng khởi tạo head và thứ tự data shuffle (seed trước model/head và generator dataloader riêng). Đây là selection có điều kiện từ seed42, không phải validation estimate độc lập; không bỏ seed thất bại khỏi mean mà phải rerun hoặc đánh dấu incomplete.

### E1c Xác nhận baseline

Giữ LR thắng mỗi task, chạy thêm seeds45/46 (10 trials). Không mở lại LR search theo kết quả seeds45/46; mỗi seed chọn epoch bằng validation. Tối đa55 full trials cho năm task, không tính smoke/failure reruns. Lưu cả kết quả âm, failure và config.

### Test gate

Trước evaluation test, root ghi selection manifest khóa backbone, head, data/evaluator/code hashes, LR, max_length, seeds, checkpoint paths/hashes và lý do selection. Agent chỉ evaluate 5 checkpoint seed đã khóa cho mỗi task, tổng tối đa 25 checkpoint cho 5 task. Paper comparison chỉ cho task có equivalence đủ xác minh; các task khác báo metric nội bộ rõ tên. Không dùng E1 test errors để thiết kế E2; chẩn đoán và hypotheses tiếp theo dựa train/validation. Sau test baseline, test không còn là tập chưa từng xem của cả dự án; mọi báo cáo cuối phải ghi lịch sử truy cập test.

Nếu metric/provenance chưa đủ cho test gate, bàn giao kết quả validation và blocker cụ thể, không tự quyết định lấy số paper trừ metric khác định nghĩa.

## 6. Artifacts, recovery và điều kiện dừng

Mỗi trial dưới `artifacts/e1/<task>/<stage>/<lr>/seed_<n>/` (hoặc layout tương đương được ghi trong runner docs): resolved config, exact command, source/data/evaluator/environment hashes, seed, epoch curves, best epoch, best validation predictions với IDs, model checkpoint, time, peak GPU memory, status/failure reason. Ghi git commit nếu có; workspace hiện không có Git checkout thực thì lưu source hash và `git_commit: null`.

Epoch summaries flush ra disk để theo dõi job dài. Tên output không được ghi đè run trước. Chỉ exact-resume khi đã lưu optimizer/scheduler/scaler/RNG và sampler state đầy đủ; nếu chưa hỗ trợ phải fresh rerun có ID mới. Không coi load model weights là resume.

Dừng trial khi loss/gradient/output không finite, split/hash/config mismatch, OOM không giải quyết được bằng microbatch giữ effective batch, hoặc artifacts/checkpoint lỗi. Không giảm epochs hay đổi LR âm thầm. Hạ microbatch/precision phải ghi protocol amendment và chạy lại đối chứng cần thiết.

Agent trả root: completed/running/blocked trials, commands, artifact links, seed-wise validation, mean/std (sample std ddof=1), runtime, GPU use, failures và giới hạn bằng chứng. Root kiểm tra rồi tổng hợp hypotheses tiếp theo; chưa tự động mở E2/E3 trong đợt này.

## 7. Tiêu chí hoàn tất đợt

E0 artifacts và runner kiểm chứng; E1 có kết quả cho các task vượt gate, tối đa năm benchmark; bảng riêng cho measured validation/test và paper reference, không lẫn historical scores. Nếu cần chạy lâu qua phiên, phải ghi PID/session, log path, command và cách tiếp tục theo dõi, không báo completed khi process còn running.
