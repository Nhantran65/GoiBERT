# E13 — Centered pair-value residual versus uncentered control

## Ủy quyền và ngân sách

User đã duyệt coding, tự bắt đầu training sau kiểm tra kỹ thuật, validation và test. Đúng một execution subagent `gpt-5.6-luna`, `xhigh`; root plan/review/tổng hợp. Không tạo helpers. Dùng GPU0–3 theo tài nguyên thực, không dừng tiến trình khác.

10 training trials = centered/uncentered × 5 task, seed42, 30epochs. Chọn checkpoint bằng validation. Sau khi cả10 hoàn tất và kiểm tra integrity đạt, khóa manifest10checkpoints rồi **chạy test cả10**, kể cả gate cải thiện validation không đạt, theo yêu cầu test mới nhất. Không dùng test để chọn lại epoch/cấu hình. Không grid, seed mới, corpus mới, thêm BamiBERT training hoặc sửa failuremode sau khi nhìn điểm.

Nguồn hypothesis: [E13 proposal](E13_VALUE_INTERACTION_PROPOSAL.md); [diagnostic E13A đã hoàn tất](E13A_RESULT.md). E13 là thử nghiệm kiến trúc, không reproduction QVI/DAT hoặc claim novelty/SOTA đã xác nhận.

## Thiết kế chốt trước kết quả

- Original `models/goiturn-s-gd`, encoder10layersH384 và headCLS E1/E8 nguyên bản. Không dùng E9/E10/E11finetuned init.
- Chỉ sửa attention output block7/index6. Giữ originalQ/K/V/Wo và pathSDPA/dropout/LN. Thêm U,W:384→64, P:64→384, tất cả no-bias. U/W normalstd.02; Pzero. Thêmmodule trong fork_rng sau originalmodel/head để giữ oldinit và trainingRNG.
- q_i=U h_i, k_j=W h_j từ cùng hidden đầu vào attention, trước projectionQKV. Centered `r_ij=GELU(q_i+k_j)-GELU(q_i)-GELU(k_j)`; uncentered `r_ij=GELU(q_i+k_j)`. GELU exact (`approximate=none`) trong cảhai. Tổng thêm73,728parameters mỗivariant.
- Tính attention map gốc từ Q/K của block này: softmaxFP32(qhead kheadᵀ/sqrt64+additivemask), không attentiondropout trong nhánhphụ. Mean qua6heads, giữ gradients (khôngdetach). Không có QK mới cho routing.
- delta_i=P sum_j meanA_ij r_ij. Cộng vào outputdenseWo trước **cùng dropout hiện hành** và residual+LayerNorm. Không dropout/gate/normalization mới. NhánhoriginalSDPA vẫn giữ attentiondropout nhưE8.
- Mask keys padding nhưoriginal; không hardfilter specialtokens củaoriginalattention. Mask delta trên querypadding để không thêm signal ởpad. Không rawoffset/tokenizer mới.
- Chunk queries cố định32 hoặc64 để tránh temporary quá lớn. Lưu ý chunk đơnthuần KHÔNG xóa tensors autograd giữ qua backward: nếu cần dùng non-reentrant activationcheckpoint riêngnhánhphụ, giống cả2variant, recordoverhead. Không giảm batch/length âmthầm.
- Newparams dưới `encoder.encoder.layer.6...` nhận baseLR*.8**3, cùng decaypredicateE8 (weightswd.01, noaddedbias). Không headLR. Oldparamnames giữ nguyên hoặc có explicitmapping được verify.

## Recipe E8 giữ nguyên

NC loss CEclean+λNC symKL(clean,noisy), actualT1; deterministicnoise keyedseed/epoch/ID; allencodertrainable. seed42, maxlen256, eff/microbatch32, BF16SDPA, AdamW, wd.01,beta.9/.999,eps1e-8, warmup10%linear,clip1,LLRDgamma.8. LR VSFC1e-5,ABSA1e-4; λNC Sent.1,Topic.5,ViSFD.5,Restaurant.1,Hotel.1.

Reuse E2runner via isolatedfactory/optimizer adapters, E7optimizer mapping. Không copyE10rawalignment machinery. E1–E12/e13a sources/artifacts frozen. E13metadata polaritycorrectedoverlay: ABSA/ViSFD1positive2negative3neutral,0absent; ViSFDOTHERS raw4/modelbinary, pairmasked. Không đổi integerlabels. Recordwronghistoricalnames correction separately; không sửa frozenE1hash/cachecontract.

## Preflight và review trước full run

Agent gửi filepaths và kếtquả kiểm tra để rootreview song song, rồi tựlaunch khi các kiểm tra bắt buộc đạt và không còn lỗi rootflag; không đợi thêm userconfirmation.

Kiểm tra có ý nghĩa: all5shapes; encoder/head init và logits t0 soE8; cùngparams2variants; mask/padding; FP32softmax trong outerBF16autocast; centerednullidentity(q0/k0) vàlinearactivationidentity trênsynthetictensors; chunkedvsdense forward/gradient mộtcase nhỏ; gradient tớiPstep1 vàU/Wstep2; optimizerunique/branchLR; strictcheckpointload bằng E13constructor. Không branchcompute shortcut khiến Pzero khôngcógradient.

Profile vàiactualtrainsteps batch32 trên ViSFD để đo VRAM/time. CUDA unavailable trong sandbox phải thử hostexec escalated trước khi kếtluận thiếuGPU. Không fallbackCPU chofulltraining. Nếu vẫn không cóCUDA báo blocker rõ. `.venv-e0-cuda/bin/python`; CUBLAS_WORKSPACE_CONFIG=:4096:8 trướcimportCUDA.

## Điều phối và lưu kết quả

Durablecontroller detached,state/heartbeat, PID+livenesschecks,freeVRAM từngdispatch, mộtworker/GPUbanđầu. Khôi phục/adoptworkerđangchạy vàcompletedartifacts khi controllerrestart, không chạytrùng. Logerror vànotifyroot khi jobfail hoặcqueuekhôngtiến. Freezehashes trướcfulltrain; sửa lỗi cần version/history và quarantinepartial không overwritesilent.

## Validation, test và diagnostic

Recompute validationmetrics từpredictions,exactIDs/gold/order,30epochcompletion. F1internal0..1;display100×F1;delta_pp=100×(new−baseline). ReadactualE8baselinefiles, không hardcode85.16 nhưfraction. Reportcentered−E8 vàcentered−uncentered toàn5task, cùngparams, bestepoch,time,VRAM,gradient/correctionnorm (diagnostic ngoài hotloop).

Test: lockcheckpointsha/architecture/task/seed/bestepoch/fullsource manifest trướcinference. Strictload sameconstructor,bf16sdpa, fixedtestsplit/decoder. Savefulllogits nếu storagehợp lý, predictions, metrics, testaccesslog. Numericgold giữnguyên; metadata/reportnames dùngcorrectoverlay. SoE8testseed42 vàpaperBamiBERT riêng cột, ghi split/evaluatorlineage caveat. Không labelpermutation hoặc thresholdtuning.

Sau fulltrain có thể đánh giáE13A30cặpcốđịnh trên10checkpointbest để xem cơchế, lưu fulllogits/expectations/provenance. Không dùngdiagnosticchọncheckpoint hoặcđổicâu/nhãn.

Gate nhận định (khôngchặntest đãduyệt): ≥4/5gainvsE8,≥2gain≥.3pp,nodrop>.3pp; centered hơncontrol≥3/5 vàkhôngloss>.3pp. Heuristic đầu tư, không significance. Nếu chỉ một tasktăng hoặccontrolngang, báođúng là chưacó cải thiệnrộng/chưachứngminhcentering. Rootreviewresults vàđềxuấttiếp, khôngagenttựmởtrainingmới.

Deliver: scripts/src/configsE13, preflight+runtimeprofile, manifests, validation/testsummaryCSV+JSON, reportdocs/E13_RESULT.md và checkpoint/provenance. Hoàn tất cả code→train→val→test, không dừng ở bàngiao code.
