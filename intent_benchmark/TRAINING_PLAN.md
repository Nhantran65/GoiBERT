# Pilot intent 8 lớp — protocol khóa trước

## Phạm vi

Người dùng đã duyệt chuẩn bị model, training, chọn cấu hình, test và so sánh ba model. Mọi507mẫu được xử lý bình thường; uncertainty chỉ là metadata, không lọc, không giảm trọng số hoặc tách metric. Không sửa split/labels. Tất cả code, logs, checkpoints, results ở intent_benchmark/. Không dùng chung artifacts của nghiên cứu kiến trúc GoiBERT.

## Models và input

Backbones local: models/goibert-b-dapt (goi_dapt), models/goiturn-s-gd (goi_student), models/bamibert-base-original (bamibert). Load pretrained encoder weights thực sự, không task-finetuned E* checkpoints. Gắn head mới cùng kiểu CLS -> dropout0.1 -> dense(hidden_size) -> tanh -> dropout0.1 -> linear8. Width head theo native hidden_size. Fine-tune toàn bộ encoder+head. Kiểm loading report: chỉ missing/new task-head hoặc omitted MLM/pooler đúng thiết kế; lỗi thiếu backbone weights phải dừng.

LABELS theo label_map.json; target=label đã rà, không original_label. Input chỉ model_input.messages. Serialize prior messages với role/text, latest message tách rõ; không metadata/caseID/labels. Native tokenizer của mỗi backbone, max_length512, dynamic padding. Ưu tiên giữ toàn current message, cắt oldest history trước; nếu current vượt giới hạn thì truncate có ghi count. Giữ thuật toán/format giống nhau ba models; lưu config và hashes. Không dùng test để chọn preprocessing.

## Sáu trials

Mỗi model chạy learning rate2e-5 và5e-5, seed42. AdamW, weight_decay0.01 cho weights (bias/LayerNorm0), betas0.9/0.999, eps1e-8. Uniform LR, unweighted cross entropy, không NC/KD/LLRD/class weights. Batch16, drop_last=False; nếu phải giảm microbatch vì OOM, dùng gradient accumulation để giữ effective16 và ghi deviation trước rerun. BF16 autocast/SDPA khi hỗ trợ, FP32 master weights. Clip gradient norm1.0. Linear warmup10% planned training steps rồi linear decay; max15epochs. Fixed seed Python/NumPy/Torch và shuffle; same trial freshly initialized, không warm-start từ LR khác.

Evaluate dev sau mỗi epoch. Metric chọn: Macro-F1 trên fixed8labels, zero_division0. Chỉ replace best khi strict improvement, tie giữ epoch sớm hơn. Early stop sau3epochs liên tiếp không cải thiện (min_delta0). Ties LR chọn2e-5. Không thêm LR/seed/loss sau thấy kết quả. Lưu train loss, dev metrics, best weights/epoch/config/provenance và dev predictions.

## GPU và execution

Snapshot kiểm lúc giao task: GPU1 L40S46GB dùng20MiB/util0%; GPU0 dùng4593MiB/util0%; GPU2–3 util100%. Chỉ dùng0/1 cho pilot, một worker/GPU. Queue cố định:

- GPU0: student2e-5 -> student5e-5 -> Bami5e-5.
- GPU1: DAPT2e-5 -> DAPT5e-5 -> Bami2e-5.

Recheck free memory trước launch; không kill workload khác. Dùng detached controller/worker với PID, log, heartbeat, exit codes; không phụ thuộc PTY. Mỗi trial có marker running/completed/failed. Khi process mất, kiểm log/process thực tế trước restart; không nhân đôi jobs. Root review smoke trước full launch. Một agent GPT5.6Luna xhigh làm implementation/execution, root review.

## Kiểm tra trước launch

Pin hashes train/dev/test, label map và corrected data. Audit no model-input metadata/future leak, counts353/77/77 và label order. Training loader chỉ train/dev, không parse test labels. Kiểm1batch forward/backward mỗi backbone, trainable parameter counts, finite loss/gradients và loading info; smoke không ghi ngược backbone gốc. Restore/reload fresh model/seed cho full trials. Lưu smoke artifacts đủ root review.

## Chọn model và test

Sau khi cả6trials terminal và mọi lỗi được giải quyết, ghi selection.json dựa dev, checkpoint SHA và tiêu chí đã khóa trước khi load test. Chạy test cho đúng3winners, không chọn lại theo test. Báo Macro-F1, accuracy, per-intent precision/recall/F1/support, confusion matrix (rows gold/columns prediction), predictions cùng stable sample IDs. Tính lại metrics từ saved predictions để kiểm.

Đo params, training peak torch memory và inference latency/peak memory của3winners trên cùngGPU rảnh, không concurrent training. Batch1, cùng77testtexts, warmup10forwards, synchronize CUDA khi timing; ít nhất2passes. Báo mean/median/p95 và phạm vi timing (model-forward only, tokenizer không tính); không ngụ ý end-to-end production latency. Ghi torch/transformers/CUDA/GPU, precision và sequence lengths. Không cần broad speed grid.

## Bàn giao

intent_benchmark/training/ chứa implementation/config vàsmoke; intent_benchmark/runs/ chứa6trials; intent_benchmark/results/ chứa selection, test predictions/metrics, timing và benchmark.csv/REPORT.md. EXECUTION_STATUS.md ở folder riêng ghi current PIDs/stage/completed runs và kết quả cuối. Không training hoặc sửa kiến trúc ngoài phạm vi này.
