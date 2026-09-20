# E15 — Chuyển E13 sang BamiBERT gốc

## Câu hỏi và ngân sách

Kiểm tra centered pair-value residual có cải thiện BamiBERT gốc và hơn uncentered control không. Đây là phép thử chuyển cơ chế sang backbone khác, không cô lập riêng tác động model size.

Đúng một subagent GPT5.6Luna xhigh thực thi code/preflight/train/evaluation; root review. Tận dụng GPU0–3 theo VRAM khả dụng, không dừng công việc khác. 10 trials: 2 variants × 5 tasks, seed42,30epochs. Không grid hoặc multi-seed. Tự chạy sau preflight đạt và không còn lỗi root flag, không hỏi lại user.

## Backbone và vị trí

Dùng official BamiBERT gốc đã pin ở E14: `models/bamibert-base-original`, HFcommit `57bc1340debbe4e348ec549047a763caebe4a977`, weightSHA `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`. Không dùng teacherDAPT hoặc checkpointE14 đã fine-tune để khởi tạo.

12 layers, hidden768,12heads,headdim64. Chèn vào **block8/index7**: tương ứng khoảng2/3 độ sâu, đã chốt trước khi xem kết quả. E13 student dùng block7/index6 trong10layers; vị trí tương đối chỉ gần nhau, không giả định chức năng layer tương đương.

## Cơ chế

Giữ nguyên công thức E13, rank64: q_i=U h_i,k_j=W h_j. Centered r_ij=GELU(q_i+k_j)−GELU(q_i)−GELU(k_j); control r_ij=GELU(q_i+k_j). U/W/P no bias, thêm3×768×64=147456params cho mỗi variant. U/Wnormalstd.02,Pzero,fork_rng giữ initialization head/encoder và RNG cũ.

Routing nhánh phụ là mean12attentionmaps gốc tại targetlayer, softmaxFP32 với autocastdisabled, dropoutoff; giữ gradients quaQ/K. OriginalSDPApath giữ nguyên. Cộng PΣa_ijr_ij sauWo trước sharedoutputdropout vàresidualLN; deltaquerypadding=0. Không thêm gate/loss/normalization/tokenizer.

Chunk queries64 mặc định, có thể activationcheckpoint nhánh phụ nếu cần memory, giống cảhai variants. Chunk không tự giải phóng graph backward. Không sửa frozenE13; tạo isolated E15module hỗ trợH768 thay vì ép constantsH384.

## Recipe E14

Fresh CLS head768, dropout.1→dense768→tanh→dropout.1→logits. CE plain, uniform AdamW tất cảlayers, không NC/KD/LLRD. LR cố định từ E14validation: Sentiment2e-5,Topic1e-5,ViSFD/Restaurant/Hotel5e-5. Maxlen256,eff/microbatch32,BF16SDPA,warmup10%linear,clip1,wd.01,betas.9/.999,eps1e-8. Dùng chính tokenizer E14, data/evaluator/IDs/splits không đổi.

Nếu microbatch32 không vừaVRAM, dùng accumulation với hiệu dụng32 giốngE14 và ghi rõ; tránh thay recipe riêng từngvariant. Tham số nhánh nhận cùng uniformLR, không mangLLRDstudent sang. Numericgold giữ nguyên; semanticoverlayABSA1positive2negative3neutral,0absent; ViSFDOTHERS binary/detectiononly/pairmasked.

## Preflight và vận hành

Reuse E14frozen pipeline qua isolated E15adapter, source/weightsmetadata phải trỏ officialBamiBERT. Kiểm tra sourceidentity, all5shapes, exactevalinitparity E14same seed/head, counts147456 vàsameparams, FP32routingunderBF16, nullidentity, chunkedforward/gradientparity, Pgradientstep1/UWstep2, uniformoptimizercoverage vàstrictcheckpointrestore.

Profile actualCEforward/backward batch32 cho model768, không dựa profilestudent. .venv-e0-cuda/bin/python vàCUBLAS_WORKSPACE_CONFIG=:4096:8. Nếu sandboxkhôngthấyCUDA, thửhostexecescalated trước khi báo thiếuGPU; không fallbackCPUtraining. Durablecontrollerheartbeat/adoptworkers/no duplicate,freeVRAMcheck mỗidispatch, sourcefreeze trướcfullrun. Báo lỗi nhanh và tiếp tục sửa trong scope.

## Validation, test và quyết định

Chọn epoch từngtrial bằng validationF1; không chọn lạiLR. Recompute savedpredictions với IDs/goldcheck. SoE15centered/E15control/E14 ở cùngtask, internalfraction0..1,display100×F1,delta_pp100×difference. Báoparams/time/VRAM vàcorrectionnorm ngoài hotloop.

Theo đề xuất đã duyệt, validation là screening trước: centered phải tăng≥4/5taskvsE14,≥2gain≥.3pp,nodrop>.3pp và hơncontrol≥3/5 để coi là tín hiệu cải thiệnrộng. Đây là heuristic đầu tư, không significance.

Nếu gate đạt, root review khóa10bestvalidationcheckpoints, rồi cùngagenttest cảhai variants trên5task trong phạmvi evaluation đãduyệt; không hỏi lại user. Nếu không đạt, dừng sauvalidation và báo root, không tự mởtest/multiseed để tìm thắng. Bảnproposal E13 từng cótestall được duyệt riêng choE13, không mặc định ngân sách đó áp sangE15.

Không tuyên bố kiến trúc phải hiệu quả hơn vì modellớn. Nếu hai nhánh cùng tăng, chưa chứng minhcentering; nếu không có tín hiệu, dừng cơchế này. Report `docs/E15_RESULT.md`, summaryCSV/JSON,source/provenance/preflight/controller vàcheckpointmanifest. Không tự mở thí nghiệm tiếp.
