# E17 — review triển khai của root

## Trạng thái

CPU và GPU preflight đã đạt, profile NC micro16×accum2 đạt; **root đã duyệt full pilot15trials** bằng `artifacts/e17/root_approval.json`. Output contracts đủ5task đã được kiểm trên GPU2. Startup/firstepoch sẽ được xác nhận từ tiến trình thật, không chỉ approvalfile.

## Các vấn đề đã phát hiện khi đọc code

- Eta ban đầu bị viết thành scalar thay vì6head coefficients; đã sửa shape và broadcast.
- Preview tổng hợp content values thiếu Wo projection; đã sửa về residual-space messages.
- Chiều Aji ban đầu nhầm Aij; đã sửa transpose.
- Hai einsum có sai query index; đã sửa.
- Fallback query chưa index riêng receiver; đã sửa.
- `.float()` bên trong autocast chưa đủ đảm bảo FP32 GEMM; đã thêm context disabled ở preview/blend.
- Mask phải áp sau blend trên finite logits; đã bỏ cách kiểm nonfinite bằng việc thay chúng thành0.
- Literal reference ban đầu broadcast z_i thay vì z_j; yêu cầu sửa trước kiểm GPU.
- Optimizer cần eta zeroWD ở đúng block6 LR, không đổi frozenE7; baseline cũng phải có fullcoverage metadata.
- Telemetry attention difference phải trước dropout và trên valid offdiagonal entries; không đồng nhất dropout noise với hiệu ứng kiến trúc.

Những lỗi trên thuộc bản nháp chưa training, không phải kết quả thí nghiệm E17. Root đọc implementation, agent thực hiện sửa và chạy checks.

## Bằng chứng CPU đã review

`artifacts/e17/preflight.json` lúc CPU run ghi status blocked_no_cuda do sandbox, không phải kết luận host thiếuGPU. Host nvidia-smi đã xác nhận4L40S; GPU1/2 trống tại thời điểm kiểm, GPU0 còn khoảng40GiB, GPU3busy.

- Literal/factorized forwardmaxabs1.67e-6; backwardgroupsmax1.43e-6.
- Bias relocationmaxabs9.54e-7.
- Eta0 full_context/cavity vs matched baseline: maxabs0.
- Eta gradientfinite/nonzero, oldquerygradientfinite, eta[6], zeroWD, LR5.12e-6 khi baseLR1e-5.
- Pad key probability và cavity diagonal coefficient bằng0.
- Strictcheckpointreload pass, backbonefilehash khôngđổi.

Đây là CPU/FP32 evidence, chưa thay thế kiểm BF16 GPU. Profile phải dùng median các optimizersteps thực sauwarmup, có2micro mỗioptimizerstep; không dùng mộtforward hoặc mộtmicrostep để tính budgetgate.

## Các điều kiện còn phải kiểm

1. Actual GPU BF16 eta0/gradient, đủ5head shapes vàslotcounts: **đã đạt**.
2. GPU profile hai shape: fixedtrainbatch và16×256; cavity/baseline ratio≤3 và memoryheadroom: **đã đạt**.
3. Durablequeue task-major, mộtworker/GPU, savedactivePIDs/commands/heartbeat; không rerun completed/live jobs.
4. Telemetry được ghi thật, không chỉ metadata nói đãghi.
5. Source hashes preflight/profile/rootapproval khớp source launch cuối; không sửa nguồn đang chạy.

Root sẽ cập nhật quyết định dựa trên artifacts cuối; không cần hỏi lại người dùng cho các bước pilot đã duyệt trong E17_PLAN.md.

## GPU review đã chốt

Root đọc `preflight.json` và `profile.json` trên cuda:2: tất cảchecks True, BF16eta0errors đều0, gradientfinite/nonzero, cả5headschema đúng (3/4/42/12×4/34×4). Model/optimizer/configSHA thực khớp preflight.

Profile2warmup+3measured NCoptimizersteps, mỗi step2micro, cùngGPU:

| Shape | Baseline median s | Fullcontext median s | Cavity median s | Cavity/baseline | Cavity peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| Trainbatch16×26 | .11955 | .13237 | .14121 | 1.181 | .8203 |
| Synthetic16×256 | .11793 | .16240 | .14727 | 1.249 | 3.2520 |

Đây là timingoptimizerstep, không inference latency; chỉ3samples mỗiarm/shape, khôngclaim cavity nhanh hơncontrol từthứhạng nhỏ. Baselineworstshapereserved2.0039GiB, nên candidate cómemoryoverheadthật.

CoremodelSHA `132fc2aee7565022582fb97b7675c1c4587f6852bd63191134c7c2fd9e27a16d`; optimizerSHA `9bcd2f23ba20a03191a20b9c160fd89a0cd1c8fe7800f039cbbc6d3efe445fb7`; configSHA `4514b0fc95c9ea8ea6083a4fdcb606263e4f5ea7dee1291a6db478f08424e071`. Runner plumbing cònđược hoànthiện trướcsourcefreezelaunch; chưa cótrainingtrialnào để thaynguồn giữa chừng.

## Dispatch approval cuối

Root đã đọc và yêu cầu sửa counter step-hook (không dùng số logrows làm stepcount), atomicwrite telemetry tại callbacks, source-bound approval trướccontroller/workerstartup, live-job refusal và per-launchVRAMcheck. Plumbingreport ghi rõ runtimehook/approvalpath mới structuralreview, không còn hardcodedpass giả; actual firstepoch là bước xác nhận tiếp.

RunnerSHA launch khóa: `5e241c0d0e5999a1005f84a8d5475aa86f437de49fe7b2630c24e4cabcc78561`. Approval chứa stable sourcebinding, scope15trials/seed42/30epochs/no test/no multiseed vàhash của preflight/profile/plumbing artifacts. Root kiểm lạiactualsource/backbonehash trướcghiapproval.

Lượt profile cuối cùng được khóa cho dispatch cho ratio cavity/baseline1.549 ởfixedbatch và2.123 ở16×256, vẫnpass3×gate. Các timingtrước ởbảngtrên làlượtđotrước đó, khôngđược coi như tốcđộ bảođảm; chênh lệch giữa các đợt đo nhắc rằng3samples chỉđủbudgetscreen, chưaperformancebenchmark chuẩn. Coremodel khôngđổi.

## Xác nhận training đã bắt đầu

Root kiểm tra process host và artifacts sau launch: controller PID2418485 (`run_e17.py --run-all --dispatch-approved`), ba workers live PID2420246 baseline/cuda2,2420253 full_context/cuda1,2420259 cavity/cuda0, cùngtaskVSFCsentiment. Mỗiworker đã hoàn tất epoch1/globalstep358, có epoch_curves và epoch_telemetry ghi thật; hooks cósteps1/10/100. Hai nhánh có eta gradients hữu hạn và khác0. Queue còn12trials ở thời điểm snapshot.

Đây là startup confirmation, khôngkếtluậnquality. Runner/modelsource đãkhóa; khôngsửanguồnđangchạy. Theo dõi `artifacts/e17/controller_state.json`, per-trialstatus/epochcurves vàlogs; phải revalidatePIDs/cmdlines khi kiểm tiến độ sau này, không coi snapshotPID này làlivevĩnhviễn.
