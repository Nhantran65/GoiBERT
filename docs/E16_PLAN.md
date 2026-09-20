# E16 — Identity-gated FFN trên BamiBERT gốc

## Scope đã duyệt

Đúng một execution agent gpt-5.6-luna xhigh; chỉ code/run/return artifacts, root tự research/review/kết luận. 3arms baseline, attention_update_gate, ffn_input_gate ×5tasks =15trainingtrials, seed42,30epochs. Không grid/multiseed/corpus/MLM/KD/NC/LLRD. Bắt đầu tự động sau preflight đạt, không đợi user nhắc. GPU0–3freeVRAM-aware, không dừng jobs khác. Không dùng student trong vòng này.

## Model và công thức

Pinned original BamiBERT E14 `models/bamibert-base-original`, commit57bc1340debbe4e348ec549047a763caebe4a977, weightSHA c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627. Freshhead768 giốngE14. Chỉ targetblock8/index7 của12; mọi tham số encoder/head trainable.

Giữ FFN z=GELU(W1x+b1), f0=W2z+b2. x là đầu vào FFN sau attention/residual/LN. c là output attention denseWo **trước dropout/residual/LN** cùngblock. Không đưa gold/category vào c.

Gate G:3072×768,nobias,zero init, thêm2,359,296params cho mỗi gatearm. s=tanh(Gv), delta=W2(z⊙s) KHÔNG bias. fnew=f0+delta. Attention_update_gate dùngv=c/sc; ffn_input_gate dùngv=x/sx. BaselinekhôngG,delta=0. Không thêmP ngẫu nhiên, khônggatesigmoidkhác côngthức, khônglearnedLN/newdropout.

Capture c bằng wrapper có explicit per-forward lifetime, khôngdetach (gradient tớiattention), khôngregisterduplicate modules. Forward clear cache saukhiFFN đã dùng; khôngretain graph quasangbatch. Baselinecapturec cùngpath nhưngkhôngdùnggate. Checksourceparamnamesvàweightsidentity; fork_rng cho mọi constructor bổsung để giữhead và RNG cũ.

## Precision và initialization

Originalencoder và f0 dùng BF16/SDPA nhưE14. Tại targetFFN, Gv,tanh, z.float()*s và F.linear(...,W2.float(),bias=None) tínhFP32 với autocastdisabled. f0.float()+delta rồi sharedFFNoutputdropout và residual/LN. **Baseline cũng cast f0.float() tại chínhđiểm đó**; cả3arms cùngpolicy, không dùng E14khácprecision/micro làm đối chứngduynhất. Biasb2 chỉ trongf0 mộtlần.

G0→s0→delta0: evalparity baseline/cả2gate phải đạt tolerance chốt trước (FP32exact khi kernelpathgiống). Kiểm tra gradientG ngay backwardđầu hữu hạn, khác0 trênbatch cóloss; khôngoptimizerstep trongparitycheck. Không cần mộtwarmup riêng chohead/gate, không nângLRgate riêng.

## Calibration train-only cố định

Trước bất kỳ downstreamtraining, dùng officialpretrainedencoder eval/no dropout, khôngmodelhead đểlấy sc vàsx RMS scalar trên2048traintexts,512/domain VSFC/ViSFD/Restaurant/Hotel,seed16042; VSFCtext tínhmộtlần. Lấy khônghoànlại, dedup exacttrong/phốihợpdomains, filltheo seed từtrain đến đủquota nếu có; báoactualcounts vàIDs/hash. Khôngđọcval/test đểcalibrate. TínhRMS qua validnonpad/non-specialtokens, FP32accumulation; sc/sx≥1e-6.

Hai scalar chung cho tấtcảtask/arms, lưu cùngcheckpoint buffers/metadata vàcalibrationmanifest. Không thaytheo task/epoch; nếuscaler0/nonfinite dừng kỹthuật. Calibration khôngoptimizerstep vàkhôngMLM. Baseline dùng chungmanifest dù khôngcầnscalar đểtínhf0.

## Recipe đồng nhất

LRfixedE14selected: Sent2e-5,Topic1e-5,ViSFD/Restaurant/Hotel5e-5. CEplain, uniformAdamW betas.9/.999eps1e-8wd.01, exclude bias/LN, warmup10%linear+decay,clip1,maxlen256. Seed42,30epochs, **micro16×accum2=effective32 cho cả15trials** trênmọiGPU; không tựchuyểnmicro theoVRAM. NếuOOM dùngactivationcheckpointmatching3arms hoặcbáoroot, khôngâmthầmđổi recipe.

Data/evaluator integercodes frozen, overlayđúngABSA1pos2neg3neutral0absent; VSFCsent0neg1neutral2pos; ViSFDOTHERSbinary/detectiononly/pair-excluded. Giữsplit/IDs vàtokenizerBamiBERTgốc. Không sửaE1–E15.

## Kiểm tra, log và vận hành

Isolated E16source/config/script/artifacts. Preflight: pinnedsource/head768identity, all5shapes, baseline/gatesinitparity, correctc capture/xlocation, scalarcheckpointrestore, sameparams giữa2gatearms, gradientGfirststep vàoldweights, allparamsunique/optimizerLR, dtypeFP32actual underBF16, querypadmaskforstats, strictcheckpointconstructor. ProfileactualCE backward micro16accum2 trênmodel768.

Log G weight/update/grad norms tạistep0,1,10,100 vàepochend, gatequantiles/RMS/correctionnorm trênfixedtrainprobe, khôngvaltuning hoặcfulltensorlog mọistep. Phânbiệt gradienttrướcclip/sauclip. Nếu khôngdễcapturestep0 thìghi rõpreflightprobe thayvìlịch sử. Khôngclaimnovelty chỉtừnorm.

CUDAhostexec require_escalated nếu sandboxkhôngthấyGPU; .venv-e0-cuda/bin/python,CUBLAS_WORKSPACE_CONFIG=:4096:8. Durablequeueheartbeat vàPIDmap,explicitfreeGPUpool/no duplicate,adoptactive/completed saugiánđoạn. Saveincrementally,freezehash trướcfulltrain. Agentgửirootprogress vàactualPIDs,khôngdừng ởcode.

## Evaluation và dừng

Chọnepoch bằngbestvalidationF1, tiesearliest. Recompute predictionsIDs/gold/numericmetrics. BaselineE16mới là primaryreference; E14 là historicalreferencekhácmicro/precision. DisplayF1×100 vàdelta_pp100×difference,khôngnhầm thang.

Gate ưu tiên attention_update_gate: tăng≥4/5vsbaseline,≥2gain≥.3pp,nodrop>.3pp; hơn ffn_input_gate ≥3/5 vàkhôngloss>.3pp. Báo cả3arms và5task,khôngpickper-taskmixture. Heuristicchưaphảisignificance.

Nếu có tínhiệurộng rootreview vàkhóa15bestvalcheckpoints đểcùngagenttesttheophạmvi evaluation; khôngxinlạiuser. Nếu khôngđạt, dừng sauvalidation báođúngkếtquả,khôngtựtest/grid/multiseed đểcứu. Khôngmởstudenttransfer khi chưađượcreview.

Deliver docs/E16_RESULT.md, calibrationmanifest/scalars,preflight/runtime,alltrialmetadata/curves/gradientlogs/checkpoints/predictions,summaryCSV/JSON,sourceprovenance. Rootđọc vàquyếtđịnhtiếp; agentkhôngđềxuấtresearch.
