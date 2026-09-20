# E14 — BamiBERT gốc trong pipeline GoiBERT

## Mục tiêu và ngân sách

Đo BamiBERT gốc trên đúng split và evaluator đang dùng cho student, thay vì chỉ dựa vào số paper. Đây là baseline model lớn, không phải kiến trúc mới hoặc phép kiểm định riêng tác động kích thước.

Đúng một execution agent GPT5.6Luna xhigh phụ trách lấy checkpoint, code adapter, preflight, training, validation, test và report; root review. GPU0–3 theo free VRAM, không ảnh hưởng tiến trình khác. Tự chạy sau checks, không đợi thêm xác nhận user.

Single seed42. Dùng **ba LR nhỏ đúng tập paper {1e-5,2e-5,5e-5}**, 5task: tối đa15trials,30epochs mỗi trial. Đây là budget chọnLR giới hạn để tránh dùng tùy ý LRstudent cho model lớn, không mởgrid khác/multiseed. Mỗi task chọn LR+epoch bằng validation, khóa5checkpoint rồi test đúng5. Không test mọiLR để chọn setting.

## Nguồn model và tính tương thích

Official `Qualcomm-AI-Research/BamiBERT`; kiểm tra localcache trước, lấy revision cố định nếu cần download. Lưu HFcommit,config/tokenizer/weightsSHA, thông tin nguồn trongmanifest. Thư mục `models/bamibert-base-original` hoặc localcache pinned, không overwrite `models/goibert-b-dapt` hayGDstudent. Không dùng teacherDAPT và gắn nhãn BamiBERTgốc.

Load encoder `RobertaModel(add_pooling_layer=False)`, bỏ MLMhead nếu checkpoint có. Xác minh12layersH768,12heads,FFN3072. Fresh standardCLS head: dropout.1→dense768→tanh→dropout.1→tasklogits; cùnghọhead E2teacher, ghi rõ khác studentdense384. Tất cảparams trainable, không branchE9–E13.

Sử dụng tokenizer chính checkpointgốc. So sánh tokenization vớiGD trên train/validation, không mặc định giống. Nếu khác, vẫn dùng tokenizer chính model và ghi mismatch; không tráo tokenizer để ép công bằng. Không chạy externalwordsegmenter.

## Recipe cố định

Theo điểm xuất phát paper: AdamW, batch32,30epochs, LR được chọn trong3giátrị trên, bestvalidationF1. Các chi tiết paper không nói rõ được chốt theo pipeline: wd.01,betas.9/.999,eps1e-8,warmup10%linear+decay,clip1,maxlen256,BF16SDPA. Micro32 nếu vừaVRAM; khi không vừa, ghi rõmicro16/8+accumulation đểeff32, đồng nhất cácLRcùngtask.

Loss CE supervised như baseline E1/E2teacher; **không NC/KD/LLRD** trong lượtBamiBERTbaseline này. So vớiE8 là so hai hệ thống fine-tune khácrecipe cùngdata/evaluator, không phải kiến trúc-only. Thêm cột studentE1CEseed42 đãcó nếu xác định đượcLRval-selection provenance; khôngtrainlại student.

5task: vsfc_sentiment,vsfc_topic,visfd,absa_restaurant,absa_hotel. Giữ nguyên rawtexts/splits, IDs, numericgold, evaluator. VSFC macroF1/accuracy; ABSA detectionmicroF1 vàend-to-endpairmicroF1; ViSFDOTHERSbinarypresence,pair-excluded.

Polaritynames E14 dùng overlayđúng0absent1positive2negative3neutral; VSFCsent0negative1neutral2positive. Không sửa integercodes hoặc frozenE1–E13 source. BảnmirrorABSAlineage chưa được xác nhận byteequivalencevớipaper, nên gọi là localbaseline, không exactpaperreproduction.

## Kiểm tra và điều phối

Tái sử dụng frozenloader/loss/evaluator qua isolated E14runner/factory, không fork cảframework lớn. Factory, metadata và weightshashes phải cùng chỉ BamiBERTgốc; tránh E2runner hardcodehashDAPT dùbuilderđãpatch. Hash nguồn/model đúng ngay từresolvedconfig/checkpoint, không chỉ sửa report cuối.

Preflight nhỏ: freshweights identity sourceencoder, all5outputschemas, tokenizercheck, finitegradient/loss, original12layercoverage, numericmetricrecompute từsavedpredictions ởevaluation, strictcheckpoint load vớihead768, actualCUDAforward/backward profile. CUBLAS_WORKSPACE_CONFIG=:4096:8; .venv-e0-cuda/bin/python. Nếu sandboxkhôngthấyCUDA dùng hostexecescalated, không fallbackCPUtraining.

Durablecontrollerheartbeat, mỗiGPU1workerbanđầu, dynamicfree-memoryguard, khôngtrùngtrial; recover/adoptliveworkers/completedartifacts nếucontrollerdừng. Freeze source trướcfullrun. Báo rootactualPIDmapping và lỗi ngay. Có thể chuẩn bị evaluator trongkhi trainingchạy, khôngdelaylaunchđểpolishdocs.

## Selection, test và readout

Metricinternalfraction0..1. LRselection bằngmaxvalidationselectionF1; tie chọnLRnhỏhơn, epoch dùng earliestbest. Khóa manifesttask/LR/epoch/seed/checkpointSHA/sourceSHA trước test. Test đúng5selected, savepredictions/metrics/accesslog, verify IDs/gold/counts và strictstate. Không đọc test để chọn LR.

Report tables táchvalidation/test: studentE8seed42, BamiBERTlocal, paperBamiBERT; thêmstudentE1CE vàteacherDAPT đãcó dưới dạng reference rõrecipe/checkpoint nếu provenancekhớp. Các delta test chỉ so test. F1display100×F1, delta_pp100×difference; khôngso85.16với.8516.

Đọc kết quả theo phạm vi: localBamiBERT khácpaper bao nhiêu; student còn gapbao nhiêu trong evaluatorchung; recipe/pretraining/size/head vẫn confounded. Single-seed chưa có confidenceinterval ổn định; chưaclaim SOTA. Không tự mởarchitecturemod/multiseed saukhibaselinexong.

Deliver `docs/E14_BAMIBERT_BASELINE_RESULT.md`, validationgrid CSV/JSON, lockedselectionmanifest, testsummaries, runtimestats, sourcecheckpointmanifest; báoactualcompletioncho root.

Nguồn: [BamiBERT paper §3.1](https://arxiv.org/html/2607.02259), [official checkpoint](https://huggingface.co/Qualcomm-AI-Research/BamiBERT), [config](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/raw/main/config.json). Paper specifies3LR,batch32,30epochs vàbestvalidationcheckpoint; cácmặcđịnh bổsung được ghiởrecipe để không giảvờ exactreproduction.
