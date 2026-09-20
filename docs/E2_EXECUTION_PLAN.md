# GoiBERT E2 — Giữ kiến trúc, kiểm tra cách học

Ngày lập: 2026-09-08. User đã đồng ý hướng nghiên cứu và yêu cầu root lập plan, giao subagents thực thi. Root thiết kế/review/tổng hợp; subagents triển khai và chạy. Đợt này chỉ dùng train/validation, không chạy thêm test. E1 test đã được xem; không dùng các lỗi/điểm test đó để chọn cấu hình E2.

## 1. Giả thuyết và các đối chứng

| Nhánh | Giả thuyết | Thay đổi so với E1 |
|---|---|---|
| E2a KD | Teacher task cung cấp tín hiệu mềm hữu ích ngoài nhãn cứng | CE + logits KD |
| E2b NC | Consistency giữa hai lượt forward giúp regularize student | CE + symmetric KL clean/noisy |
| E2c KD+NC | Hai tín hiệu có thể bổ trợ nhau | Chỉ chạy sau review hai nhánh riêng |

Kiến trúc student, tokenizer, dữ liệu, split, head, max_length, effective batch và optimizer/update budget giữ như E1. Chưa hidden-state/attention KD, chưa corpus, chưa mở rộng FFN/layers, chưa thêm dữ liệu ngoài. Teacher `goibert-b-dapt` phải fine-tune theo task trước khi dùng làm teacher; không coi MLM head là teacher classifier.

Các kết quả E1 validation ở docs/E1_COMPLETED_AUDIT.md là đối chứng. Student mỗi trial load lại backbone `models/goiturn-s-gd/`, không bắt đầu từ checkpoint E1 đã fine-tune. Không sửa `src/goibert/` hoặc `configs/e1/`; dùng namespace `src/goibert_e2/` và cấu hình `configs/e2/`. Hash E1 đã khóa `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`; dependency hash mới phải khớp trước khi chạy.

## 2. Task, dữ liệu, selection metric

Năm task: `vsfc_sentiment`, `vsfc_topic`, `visfd`, `absa_restaurant`, `absa_hotel`. Dùng nguyên loader E1 `include_test=False`, fixed IDs và labels; chỉ hash/file metadata test từ audit cũ nếu cần, không mở test text/predictions trong E2 runner.

VSFC selection = fixed-label macro F1. ABSA selection = end-to-end `pair_micro_f1`; detection micro và per-aspect chỉ bổ sung. ViSFD gồm10slot4-class và OTHERS binary2-class (raw4 ↔ model1). OTHERS nằm trong detection, không trong polarity pair. Teacher và student cùng slot schema, class order, tokenizer và max_length256 đã chốt từ E1 train-only stats. CE và KD đều bao gồm absent0; không mask bỏ absent.

Student LR cố định theo E1 đã chọn: sentiment/topic1e-5, ViSFD/Restaurant/Hotel1e-4. Đây là kiểm tra thay đổi loss ở recipe E1, không phải search LR tối ưu cho mọi loss mới; kết quả âm không chứng minh KD/NC vô ích ở mọi cấu hình.

## 3. Teacher task training

- Từ `models/goibert-b-dapt/`, load encoder RoBERTa12L/768H; bỏ MLM head, gắn head CLS→dropout0.1→dense768→tanh→dropout0.1→logits. Cùng schema nhãn/task với student. Train toàn bộ encoder và head, CE thật.
- Mỗi task: seed42, LR{1e-5,2e-5,5e-5}, 30epochs, effective batch32. Tổng15teacher trials.
- AdamW betas(.9,.999), epsilon1e-8, wd.01 trừ bias/LayerNorm, warmup10% + linear decay, gradclip1 finite-check; BF16/SDPA. Microbatch có thể giảm giữ effective batch32, báo rõ tail weighting. Không early stopping.
- Chọn best-validation epoch trong trial; chọn LR theo validation F1, hòa chọn LR nhỏ hơn rồi epoch sớm hơn. Không chọn bằng test.
- Báo bảng teacher và student E1 bằng cùng validation evaluator. Teacher không mặc định mạnh hơn. Vẫn chạy KD predeclared ngay cả khi teacher yếu để đo tác động, nhưng phải ghi teacher quality, không gọi teacher là quality ceiling.
- Một teacher seed42 được cố định dùng chung cho mọi student seed trong task. Các độ lệch chuẩn student là có điều kiện theo teacher này; không bao gồm teacher-seed variance.

## 4. Train-logit cache

Sau khi teacher được chọn, chạy teacher eval/no_grad trên TRAIN CLEAN duy nhất, không dropout, tokenizer/max_length giống student. Lưu logits (FP32 storage từ BF16 inference), stable IDs, thứ tự class/slot, text/input hashes, checkpoint hash, tokenizer/data/evaluator hashes, max_length, precision. Một cache/task; không validation/test soft labels cho training.

Cache phải kiểm tra đủ IDs, không trùng/mất/lệch dòng, tensor hữu hạn, shape đúng, cùng text/truncation với student. Training shuffle lấy logits theo IDs/indexmapping, không dựa vào thứ tự batch của file. Hash mismatch dừng, không fallback teacher khác. Sample cache logits so trực tiếp teacher forward trên TRAIN để xác nhận alignment trước full KD.

## 5. Student losses

Giữ nguyên E1 classifier/head và initialization. Cả teacher cache và augmentation không được làm thay đổi RNG toàn cục dùng khởi tạo head/shuffle/dropout của baseline.

### E2a: logits KD

`L = (1-alpha) * CE(y, student_logits) + alpha * T² * KL(teacher_T || student_T)`

T=2; alpha{0.25,0.5,0.75}. Teacher xác suất detach; KL tính FP32 log-softmax. Với ABSA, softmax từng slot theo đúng số lớp rồi trung bình theo batch và slots; không softmax chung42logits của ViSFD hoặc coi OTHERS là sentiment. KL direction phải có test bất đối xứng; đây là forward KL được chọn mới, khác code reference reverse KL trước kia.

### E2b: noise consistency

`L = CE(y, clean_logits) + lambda * 0.5 * [KL(p_clean || p_noisy) + KL(p_noisy || p_clean)]`

Tconsistency=1; lambda{0.1,0.5}. Cả clean/noisy logits nhận gradient; CE chỉ trên clean để không ép nhãn cứng cho câu có thể bị biến đổi ngữ nghĩa. KL tính FP32 từng slot như KD, gồm OTHERS binary.

Augmentation dùng RNG riêng từ(seed,epoch,stableID), không phụ thuộc Python hash ngẫu nhiên và không chạm global RNG. Mỗi sample chọn:50% giữ text,25% bỏ dấu (đ→d),20% thay teencode từ điển nhỏ có boundary,5% một typo bên trong từ. Typo không sửa digits/punctuation hoặc từ phủ định; tránh tạo text rỗng. Không chỉnh nguồn dữ liệu. Lưu policy/version/hash và tỷ lệ thực sự thay text.

Hai forward có dropout độc lập, nên nhánh này kiểm tra kết hợp text-noise + dropout consistency; chưa quy gain riêng cho noise. Nếu thắng, cần clean/clean consistency ablation để phân biệt hai phần. NC dùng khoảng2forward/backward thay vì1; giữ cùng số optimizer updates, không khẳng định equal-FLOP/compute budget. Lưu walltime và memory.

### E2c: combination (chưa mở gate)

Sau E2a/b, root review evidence rồi quyết định kết hợp. Nếu chạy: `(1-alpha)*CE + alpha*KD + lambda*NC` với coefficient đã chọn từ nhánh riêng, đối chiếu cả E1/KD/NC. Không mặc định kết hợp sẽ thắng; không tự chạy trong queue E2a/b hiện tại.

## 6. Matrix, seeds và quyết định

1. E0-E2 smoke/unit tests và CE compatibility replay:1student VSFCsentiment LR1e-5 seed42 full30epochs. So cùng E1 recipe/source và initialization/loss semantics; đây là engineering gate, không thêm candidate để chọn best-of-reruns. E1 record vẫn là baseline chính. Nếu khác bất thường, dừng và giải thích trước KD/NC.
2. Teacher15trials +5train caches. Có thể chạy NC độc lập sau student smoke/replay gate, nhưng ưu tiên teacher pilot và replay trước; scheduler không vượt1job/GPU.
3. Screening seed42: KD3alpha×5task=15student trials; NC2lambda×5task=10student trials. Cả hai nhánh phải được báo dù không thắng.
4. Mỗi task/nhánh chọn coefficient top1 trên seed42, hòa chọn hệ số nhỏ hơn. Chạy seeds43/44 cho cả hai nhánh:2seeds×2nhánh×5task=20trials mới. Không bỏ nhánh chỉ vì seed42 thấp; không thay coefficient sau xem43/44.
5. Báo paired validation deltas với E1 cùng(task,seed,LR), mean±sample std trên42–44; hiển thị từng seed và baseline. Không báo validation-selected mean là independent generalization estimate.

Tổng tối đa61full training trials hiện được thiết kế:15teacher +1CEreplay +25student screening +20student confirmation.5cache passes và smoke không tính training trial. Seeds45/46, E2c, test hoặc architecture experiments chỉ mở sau root review kết quả; không tự chạy từ queue này.

Tiêu chí để gọi là ứng viên đáng xác nhận tiếp (quyết định nghiên cứu được chốt trước kết quả): mean paired validation gain≥0.3điểm F1 và gain dương ít nhất2/3seeds của task. Gain dương nhỏ hơn ngưỡng giữ ở trạng thái inconclusive; gain âm báo rõ. Đây là heuristic screening, không phải statistical significance. Không đổi các threshold theo test.

Một backbone nguồn chung cho5task; E2 task-specific training có thể dùng coefficient khác nhau theo validation, không biến checkpoint task thành backbone chung mới. Khi chọn recipe dùng chung toàn bộtask, root xem bảng mean delta và worst-task tradeoff; không gộp F1 khác task để gọi overall accuracy.

## 7. Gates, thực thi và phân công

- `e2_losses`: losses/noise độc lập, tests và E2_LOSS_CONTRACT.md.
- `e2_runner`: teacher/student CE/KD/NC runner, cache, adapter tái dùng E1, tests/E2_RUNNER.md.
- `e2_orchestration`: config/matrix, durable queue, stage selection/readout, tests/E2_ORCHESTRATION.md.
- `e2_operator`: preflight GPU/env, integration smoke, teacherprofile, CEreplay/teacherpilot/fullqueue và logs E2_EXECUTION_LOG.md.
- Agent reviewer độc lập: protocol và implementation/data/cache audits. Root quyết định GO sau đọc bằng chứng, không chạy model trực tiếp.

Gate smoke: unit loss KL/T²/alpha0, slotwiseOTHERS, augmentation deterministic; teacherstudentload, finite forward/backward/optimizer, save/load; loader không mởtest; cache exactIDs/shuffledalignment; E1dependencyhash nguyên vẹn; runneralpha0 CE gradients so E1; profileteacher CUDA memory. Teacherpilot VSFC LR2e-5 seed42 và CEreplay VSFC LR1e-5seed42 là fullruns đầu tiên sau rootGO. Cache smoke dùng smallTRAIN teacher smoke riêng, không lẫn cache/fullteacher outputs.

CE fidelity gate cụ thể: so E1/E2 student initial state với cùng seed và RNG reset phải khớp tensor/hash; smoke FP32 cùng batch/RNG phải khớp loss và gradient (`atol=1e-6`, `rtol=1e-5`). Kiểm tra cùng số optimizer steps, scheduler và thứ tự IDs. Replay full30epochs chỉ là diagnostic, cho phép BF16/SDPA sai khác không bitwise; nếu best-validation F1 lệch E1 quá0.5điểm hoặc recipe/initialization khác thì dừng mở student KD/NC để review. Không dùng replay để thay thế baseline E1 hoặc chọn run tốt hơn. Ngưỡng0.5điểm là engineering alarm, không phải tiêu chí scientific gain.

Venv `.venv-e0-cuda`; user cho phép GPU0,1,2,3 dù có workload khác. Không kill process ngoài. Memory guard ban đầu≥8GiBstudent,≥14GiBteacher; điều chỉnh theo actualprofile với margin trước launch. Student batch32 mặc định, teacher giảmmicro khi cần. Set CUBLAS_WORKSPACE_CONFIG=:4096:8 trước init, ghi SDPA nondeterminism; không hứa bitwise reproducibility.

Bổ sung theo chỉ đạo người dùng: GPU2 với khoảng7.5GiB free vẫn được dùng nếu model fit. Student guard cuối phải dựa trên peak CUDA reserved đo ở full train-step của từng mode CE/KD/NC cộng1GiB dự phòng, không giữ cứng8GiB. Đo NC đủ hai forward/backward và optimizer; nếu peak+margin fit thì dùng GPU2. Teacher ưu tiên GPU0/1/3; chỉ dùngGPU2 nếu teacher profile cũng fit. Snapshot free memory ngay trước launch và ghi profile/threshold trong manifest.

Chốt sau profile thực tế: student microbatch32/accum1, length256, BF16/SDPA đạt peak reserved1.9004GiB (CE) và3.2559GiB (NC); teacher microbatch8/accum4 đạt2.7734GiB. Root chọn guard chung5GiB free cho các mode (dư hơn peak+1GiB), cho phép dùng cảGPU2. Teacher dùng microbatch8 nhất quán mọi trial; student giữmicro32 như E1. Nguồn profile: `artifacts/e2/preflight/profile_student_batch32_20260908T150818+0700.json`, `profile_teacher_train_20260908T150331+0700.json`. Quyết định này thay các guard tạm8/14GiB ở trên.

E1source/config giữ nguyên. E2source/config hash riêng và E1 dependencyhash khóa trướcfullrun. Không sửa code/config giữa queue; mismatch dừng và version/rerun có chủ đích, không trộn kết quả. Orchestration scripts lưu hash snapshot riêng, PID/command/log/state/events. Dừng khi nonfinite/hashmismatch/OOM không giải quyết theo microbatchpolicy, missingseed, duplicate identity, cache misalignment hoặc artifacts lỗi. Không auto-retry chọn run đẹp; không overwrite hoặc coi weights-only load là exactresume.

## 8. Artifacts và bàn giao

`artifacts/e2/`: recipe manifest, teacher trials/selection, train-logit caches, student trials/predictions/epoch curves, source/environment/checkpoint/data/cache hashes, PIDlogs, failures, validated aggregate readout. Lưu lossCE/KD/NC riêng, learning curves train/validation, time, peakmemory, maxlen, seeds, augmentation stats, teacherquality và selectedalpha/lambda.

Root nhận report thực nghiệm để so E1→E2 trên VALIDATION, chỉ rõ gain có lặp lại hay không và giới hạn teacherseed/searchcompute/noise. Chưa có E2 results khi viết plan này; không coi giả thuyết là kết quả.
