# E17 — Screening recipient-excluded keys trên student

Người dùng duyệt bước3 kiểm chứng ứng viên. Root lập plan/review; đúng một execution agent GPT5.6Luna/xhigh code, profile, training và trả artifacts. Không giao research. Phạm vi: preflight GPU và tối đa15trials = baseline/full_context/cavity ×5task,seed42,30epochs. Không LRgrid,multiseed,teacher/corpus mới. Chưa mở test; quyết định sau validation thuộc root.

## Model và operator khóa

Fresh `models/goiturn-s-gd`, CLS head384 nhưE8; block index6/layer7. E1–E16 và research-check artifacts không sửa. Tạo src/goibert_e17,configs/e17,scripts/run_e17.py,artifacts/e17 riêng; có thể reuse immutable E2loop/E7optimizer theo adapter.

Operator theo `RECIPIENT_EXCLUDED_KEYS_PROPOSAL.md` và hai `CAVITY_KEY_*SPEC.md`. q,k,v gốc cóbias. Preview A=softmax(QKᵀ/sqrt64+mask), trước dropout. Content messages u_i^t=Wo^t(Wv^t h_i), gộp bv chung vào b_eff=bo+ΣWo^t bv^t. z_j=h_j+b_eff+ΣA_jk^t u_k^t. LN dùng đúng attention-output LayerNorm của targetblock, gamma/beta/eps1e-5; không FFN trong preview.

Cavity C_ji=z_j−Σ_t A_ji^t u_i^t choi≠j; diagonalgiữz_j. Không renormalize saudeletion. Lcav=q_iᵀ(WkLN(C_ji)+bk)/sqrt64. Fullcontextcontrol dùng cùng phép tính nhưng coefficientsdeletion=0. Q vẫn từh_i đầu block, không previewQ. Tất cảweights dùng lại, khôngdetach.

Hai variants có eta[6] unbounded,init0. Lnew=Lorig+eta*(Lvariant−Lorig) trên validfinite logits; maskáp saublend. Baseline dùng Lorig và khôngeta. Finalsoftmax/dropout/AV dùng **V gốc gồm bias**, rồi Wo/dropout/residual/LN/FFN gốc đúngmột lần. Preview khôngdropout, không tiêu RNG; finaldropout cùngrecipe. Không đăngkýtrùng modules hoặc nhânđôi outputbias.

## Precision, memory, correctness

Targetblock cả3arms dùng cùng explicitattention FP32scores/softmax và cáchcastprobability/AV đãghi; cácblocks khácBF16SDPA. Baselinereplay mới cần vì targetkernel vàmicrokhác E8lịchsử. Lưu policyactualdtypes. Eta0 comparison against matchedbaseline phải đạt1e-5abs+1e-4rel logits trướctraining; thêm FP32reference tightercheck1e-5abs. Preserveoldweights/headinit/RNGvới fork_rng.

Factorization/guard theo researchspec, nhưng **không copy Python loop từngpair hoặcCPUtransfer mỗiforward** vào training. Vectorized querytiles/fallbackchunks64; không fullN²d. GuardE=sumabsoluteterms trước tổng. Nếuinputs/intermediatesnonfinite: failclosed ghierror; khôngclaim guardcứuoverflow. Fastdivision cho flaggedpairs phải dùng denominator hữu hạn để tránh0*NaNbackward. Nếu cần activationcheckpoint, áp cùng policy chofullcontext/cavity và báobaselinepolicy. Không đổioperator đểtăngtốc.

Preflight bắtbuộc: actualstudentbatchshapes/all5heads; literalvsfactorized tạiNnhỏ forward/backward; biasrelocation; targetLNlocation; pad/diagonal; eta0parity; firstbackward eta gradient finite/nonzero trênbatchcósignal; gradientoldweights; optimizeruniqueLR; strictcheckpointreload; nofrozen-source mutation. Check eta parameter receives layer6 LR (gamma^3), zero weightdecay. Đây là scalar interpolation, khôngnângLRriêng.

## Recipe và ngân sách

E8 NC+LLRDgamma.8, CEclean+lambda symmetricKLclean/noisy, NC cảhaiforwards attached, T1. Sent/TopicLR1e-5; Vi/Rest/Hotel1e-4. NCλ=.1/.5/.5/.1/.1 theo thứtự trên.30epochs,seed42,warmup10%,AdamWwd.01,clip1,maxlen256. **micro16×accum2=effective32 cho mọiarm/task**; accumulationweightedbysamples includinglastwindow. Không tựđổimicrotheoGPU. Numericdata/splits/evaluator frozen; ABSA0absent1pos2neg3neutral; ViOTHERSindex5binary,detectiononly,pair-excluded.

Profile thực trênfixedtrainbatch cóchiều dài vàpadding đượcghi, thêm syntheticshape16×256; full NCoptimizerstep timing sauwarmup, resetfreshmodel trướcfulltraining. Báo3arms throughput/allocated/reservedVRAM,fallbackrate,eta/logitstatistics. DùngsameGPU đểratio meaningful. Ngưỡng đầu tưpilot: cavity<=3×matchedbaseline medianstepwalltime, khôngOOMmicro16 vàreservekhôngvượt freeVRAMtrừ4GiB. Đây là budgetgate, khôngclaim productionfast. Nếu vượt: chỉ tối ưuimplementationkhôngđổiformula; nếuvẫnfail báo root, khônglaunch15trials.

## Dispatch và review

GPU0–3 đãđược cho phép, sửdụngfreeVRAM-aware; khôngdừngjobkhác, không2workerchungGPU. `.venv-e0-cuda/bin/python`, CUBLAS_WORKSPACE_CONFIG=:4096:8. HostCUDAescalation nếu sandboxkhôngthấyGPU. Durablecontroller/logs/PIDmap/heartbeat, adoptcompleted/active saugiánđoạn, khôngrerun tựđộngtrialsđãxong. Freezehash trướcfullrun, preflight/profile artifacts trướcdispatch.

Agent gửi rootcodepaths/preflight/profile đểreview. **Root review là bước nội bộ, không cần hỏi lại người dùng.** Sau rootpass, chínhagentlaunch vàgiám sát15trials; khôngdừng ở “code xong”. Mỗiworker lưustatus/epochcurves/checkpoint/valpredictions vàeta. Log gradients/etanorm/fallback đầu vàepochends, tránh logfulltensor.

Ghi thêm magnitude của eta*(Lvariant−Lorig) trên valid offdiagonal logits, so với magnitude Lorig, và mức thay đổi attention probabilities. Eta0preservation không đảm bảo nhánh được sử dụng đủ mạnh trong30epochs; nếu screenfail và effectivecorrection nhỏ, chỉ kết luận về recipe/parameterization đãchạy, không bác bỏ toàn bộ cavityoperator. Không tự nângLReta hoặc schedulesau khi thấyđiểm.

## Đánh giá

Chọnbestepoch bằngvalidationprimaryF1,tiesearliest; full30epochs. Recompute IDs/gold/count/F1 từngarm. Primarycomparison cavity−full_context vàcavity−baseline; E8historical chỉreference. Gate thực dụng: cavity tăng≥4/5vsbaseline,≥2taskgain≥.3pp,nodrop>.3pp; vsfullcontext thắng≥3/5 vàít nhất1gain≥.3pp,nodrop>.3pp. Khôngsignificance hoặc guaranteeSOTA. Báoactual5task ngaycảgatefail. Nếucontroltăngngangcandidate, khônggánchoexclusion.

Deliver `docs/E17_RESULT.md`, preflight/profile/source manifests,15trialsvalsummary, resource/correctionstats. Root tổnghợp vàquyết địnhtest/confirmation; khôngtự mởtest/grid đểcứu điểm.
