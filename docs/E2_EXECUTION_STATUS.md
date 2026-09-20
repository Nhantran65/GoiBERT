# Trạng thái E2

Ngày bắt đầu: 2026-09-08. User đã phê duyệt thí nghiệm giữ nguyên kiến trúc, thay đổi cách học. Plan: [E2_EXECUTION_PLAN.md](E2_EXECUTION_PLAN.md).

## E3 đã hoàn tất — 09/09/2026

Đúng hai trial loss-only seed42 đã chạy đủ30epochs, audit PASS, không mở test. ViSFD pair F1 80.223 (+0.151 so E1, -0.865 so NC); Restaurant70.130 (+1.112 so E1, +1.070 so NC). So sánh cùng seed42. Xem [E3_VALIDATION_READOUT.md](E3_VALIDATION_READOUT.md). Chưa chạy thêm seeds; đề xuất chỉ xác nhận Restaurant nếu tiếp tục.

## Chẩn đoán sau E2 — 09/09/2026

Đã phân tích predictions validation để so teacher–student và phân rã lỗi aspect/polarity, không train hoặc inference mới. Root chọn đề xuất kiểm chứng loss polarity trên gold-present slots trước khi tăng FFN; thiết kế hai trial seed42 ViSFD/Restaurant tại [E3_DIAGNOSTIC_DECISION.md](E3_DIAGNOSTIC_DECISION.md). E3 chưa được triển khai hoặc khởi chạy. Kế hoạch chẩn đoán: [E3_DIAGNOSTIC_PLAN.md](E3_DIAGNOSTIC_PLAN.md).

## Cập nhật mới nhất — E2 test đã hoàn tất

User đã mở test gate ngày09/09;15checkpoint của5validation-selected branches × seeds42/43/44 đã được đánh giá bằng CUDA/BF16/SDPA trênGPU1–3. Không thêmseed hoặc train lại. E1 comparison dùng lại testresults của cùng3seeds, không dùng aggregate5seeds cũ.

Canonical test F1classification: Sentiment81.31 (+0.05 soE1), Topic79.45 (+0.29), ViSFD80.62 (+1.06), Restaurant69.86 (+1.03), Hotel72.28 (+0.39). Tất cả là mean3seeds; fullmean/std và2accuracy/8F1rows nằm trong [E2_TEST_EVALUATION.md](E2_TEST_EVALUATION.md) và `artifacts/e2/test_readout.json`.

LượtCPU đầu do sandbox không thấyCUDA đã được giữ riêng tại `artifacts/e2/cpu_diagnostic_20260909/`, không dùng trong bảng canonical. HostCUDA đã được xác minh và chính15checkpoint cố định được đo lại; không chọn checkpoint hoặc backend theo điểm test. Xem E2_TEST_GATE_APPROVAL.md và accesslogs cho lịch sử cảhai lượt.

Các mục bên dưới nói “E2 chưa test” là lịch sử trước yêu cầu mới. Source/recipe training không đổi; F1so paper vẫn cần đọc kèm định nghĩa evaluator và giới hạn lineage đã ghi.

## Hoàn tất E2 — cập nhật 09/09/2026

Supervisor kết thúc bình thường lúc 00:08:08 ngày09/09 (UTC+7), returncode0. Đủ61/61trainingtrials và5/5caches; không còn E2process trênhost. Các trạng thái running/PID ở dưới là lịch sử, không phải lỗi mới.

Đã hoàn tất:15teacher,1CEreplay,15KDscreening,10NCscreening,20confirmationseeds43/44. Các3attempt Topic từng bị gián đoạn được lưu riêng, không tính completed. E2 chưa chạy test, chưa mởKD+NC hoặc seeds45/46.

Đọc [E2_VALIDATION_READOUT.md](E2_VALIDATION_READOUT.md) để xem kết quả và nhận định; nguồn máy đọc được là `.orchestrator/stageB_paired_e1_comparison.json`. Ứng viên tăng ở cả3seeds: NC Sentiment (+0.44điểm), KD Topic (+0.96), NC ViSFD (+0.99). Restaurant/Hotel NC tăng mean nhưng chưa ổn định giữa seeds.

## Cập nhật mới nhất — đã vào Stage B

Kiểm tra host và artifacts lúc heartbeat2026-09-08T16:19:23Z: supervisor2686110 và scheduler2669874 vẫn chạy, không có lỗi mới. NC Hotel cuối cùng đã hoàn tất30epochs; đủ41/61trainingtrials (15teacher+1CEreplay+15KDscreening+10NCscreening) và5/5caches.

Scheduler đã tự chuyển sang20confirmationtrials seeds43/44. Bốn job đầu trênGPU0–3 là VSFCsentiment KDseed43/44 và NCseed43/44; đã có1–2epochs thực tế. PID snapshot2864801/2864881/2864962/2865036. Các GPU0–2 tạm nghỉ ở cuối Stage A vì chờ NC Hotel trênGPU3; đó là ranh giới đồng bộ giai đoạn, không phải process chết.

Nguồn theo dõi hiện tại: `artifacts/e2/.controller/recovery_v3_status.json` và `.orchestrator/state.json`, kết hợp hostPID và epochfiles. E2 vẫn validation-only. Các con số25/61 hoặc13/61 ở dưới là snapshot cũ.

## Trạng thái mới nhất — supervisor v3

Đã hoàn tất25/61trainingtrials:15teacher,1CEreplay,8KDscreening,1NCscreening;5/5train-logit caches đã có. Chưa có confirmation43/44. Bốn KD completions còn bị ledger cũ đánh dấu running đã được xác minh và nhận lại, không train lại.

Process điều phối v2 không còn trênhost dù heartbeat cũ ghi running; chưa xác định được nguyên nhân từ logs còn lại. Supervisor mới chạy tách phiên với heartbeat thật và logchild: `artifacts/e2/.controller/recovery_v3_status.json`. Tại handoff, supervisorPID2686110 theo dõi recoverychild2669874; các PID chỉ là snapshot, phải kiểm tra host/liveheartbeat.

Bốn job KD mới đang chạy trênGPU0–3: ViSFDalpha0.75, Restaurantalpha0.25/0.5/0.75. Epoch artifacts đã thể hiện tiến độ10–13epochs tại lúc root kiểm tra. Legacyv2 được đánh dấu superseded; không dùng heartbeatv2 để kết luậnqueue hiện tại.

Kết quả screeningseed42: KDsentiment tốt nhất84.5078F1, gần như E1same-seed84.5076; KDtopic tốt nhất80.7842 soE1same-seed79.7709 (+1.0133điểm); ViSFD mới2/3alpha hoàn tất, tốt nhất80.3284 soE1same-seed80.0722 (+0.2562điểm). Tất cả là validation, chưa phải kết luận đa seed haytest.

Xem [E2_PROGRESS_RECOVERY_V3.md](E2_PROGRESS_RECOVERY_V3.md). Supervisor ghi đúng command của child đang có; launch kế tiếp bỏ cờ fresh-rerun-interrupted. Không tự retry các partial/numeric failures. Các snapshotv2 ở dưới là lịch sử.

## Trạng thái mới nhất — runtime recovery v2

Queue lại dừng khi tiếp nhận job hoàn tất vì lỗi keyword argument của validator. Đã khôi phục qua `scripts/execute_e2_recovery_v2.py`, giữ nguyên mãtraining/config/frozenscheduler; adapter được ghi hash riêng. Xem [E2_RUNTIME_RECOVERY_V2.md](E2_RUNTIME_RECOVERY_V2.md).

- Hoàn tất6/61trainingtrials:4teacher,1CEreplay,1NC. Chưa có taskKD.
- Ba teacher VSFCtopic trước bị gián đoạn tại21/22/22epochs đã được lưu nguyên artifacts và hashes trong recovery archive; không coi là exactresume hoặc kết quả được chọn.
- Bốn jobs đang chạy thật trênGPU0–3: ba VSFCtopic chạy lại cùng recipe/seed từ đầu, và ViSFD LR2e-5 mới. Tại handoff, Topic đã đến5/5/5epochs và ViSFD7epochs.
- Parent mới PID1023543; children1026602/1026759/1026903/1026981. Kiểm tra PID trênhost và epochfile mới khi hỏi tiến độ, không dựa duy nhất vào snapshotPID.
- State mới: `artifacts/e2/.controller/recovery_v2_status.json`, cùng `.orchestrator/state.json`. Controller cũ đã được đánh dấu `superseded`, các lỗi cũ vẫn giữ làm lịch sử.
-57testsPASS, trong đó5tests lifecycle. E2 test vẫn đóng.

Các snapshot ở các mục bên dưới là lịch sử trước recoveryv2 và không đại diện trạng thái hiện tại.

## Cập nhật sau kiểm tra tiến độ và khôi phục controller

Đã hoàn tất5/61trainingtrials:3teacherVSFCsentiment,1CEreplay,1NCscreeninglambda0.1. Controller trước bị chặn bởi contractmetadata và CLI nên queue chưa đi tiếp; đã sửa lớpcontroller/compatibility, giữ nguyên mãtraining/config/frozenscheduler. Các state/lỗi cũ được lưu trong recovery audit, không chạy lại5trials hoàn tất.

Controller mới PID806382, scheduler PID807077 đang chạy. Bốn jobs teacher tiếp theo đã khởi chạy thật: VSFCtopic LR1e-5/2e-5/5e-5 trênGPU0/1/2 và ViSFD LR1e-5 trênGPU3. Live state: `artifacts/e2/.controller/state.json` và `artifacts/e2/.orchestrator/state.json`. PID497795 và các snapshotpilot bên dưới là lịch sử.

Kết quả seed42 trênVSFCvalidation: E1/CEreplay84.5076; teacherbest LR2e-5 đạt85.9226; studentNC lambda0.1 đạt84.7722 (+0.2646điểm soE1). Chưa có taskKD hoặc kết quả E2 đa seed; không dùng bảng này để claim paper/test. Recovery: [E2_RECOVERY_REPORT.md](E2_RECOVERY_REPORT.md),52testsPASS sau sửa,5artifacts được xác thực và tái dùng.

| Agent | Trách nhiệm | Trạng thái ban đầu |
|---|---|---|
| e2_losses | KL/KD/consistency, augmentation deterministic và fixtures | Hoàn tất |
| e2_runner | Teacher/student runner, train-logit cache, CE compatibility | Hoàn tất, source đã khóa |
| e2_orchestration | Recipe/matrix, selection, durable GPU queue | Hoàn tất scheduler/controller, đã bàn giao |
| e2_operator | Preflight, integration smoke, pilot/replay và vận hành | Full teacher grid VSFC và CE replay đã khởi chạy |
| e2_review | Review độc lập plan và implementation | Đã review, các blocker đã được sửa |

Root giữ vai trò nghiên cứu và review; mọi model execution do subagents thực hiện. Hiện có full-run teacher/CE replay learning curves, chưa có kết quả KD/NC đa seed.

## Gates

1. Bảo toàn E1 dependency/source hash, dữ liệu/evaluator và các checkpoint E1.
2. Unit tests + small GPU smoke teacher/student/loss/cache; profile teacher memory.
3. Root mở teacher pilot VSFC LR2e-5 seed42 và CE replay VSFC LR1e-5seed42, mỗi run30epochs.
4. Sau review smoke/replay/teacher pilot, mở queue validation-only tối đa61trials như plan. Mỗi GPU tối đa1job, cho phép chia sẻ GPU0–3 với workload hiện có mà không dừng process khác.
5. Root tổng hợp E2a/E2b so với E1 validation; chưa tự mở KD+NC, seeds45/46, test hoặc architecture expansion.

PID/commands/log paths và measured status sẽ được cập nhật trong E2_EXECUTION_LOG.md khi có run thật. Các bảng E1 test chỉ là báo cáo lịch sử; không được dùng chọn cấu hình E2.

## Resource gate đã xác minh

Root đã đọc full train-step BF16/SDPA profiles, length256: student micro32 CE peak reserved1.9004GiB; student micro32 NC3.2559GiB; teacher micro8accum4 đạt2.7734GiB. Chốt free-memory guard5GiB cho các mode, teacher micro8/studentmicro32, effective batch32. User cho phépGPU2 nên có thể sử dụng ngay khi fresh preflight free≥5GiB; không giữ guardtạm8/14GiB nữa. Teacher/student tiny smoke đã hoàn tất; cache/runner fidelity và finalsourcefreeze vẫn phải kiểm tra trước fullpilot.

## Full runs đã mở

Sau khi initialstate169keys E1/E2 khớp hash/maxdiff0, CE loss/gradient khớp0diff, và GPU cache→KD scalar/ViSFD cùng NC smoke PASS, root đã mở các fullrun đúng release trong E2_GATE_DECISIONS.md:

- GPU2: CE replay VSFCsentiment LR1e-5 seed42,30epochs,micro32. Artifact `artifacts/e2/stageA/ce_replay/vsfc_sentiment/seed_42/`.
- GPU1: teacher VSFCsentiment LR2e-5 seed42,30epochs,micro8acc4.
- GPU0: teacher VSFCsentiment LR1e-5 seed42,cùng recipe.
- GPU3: teacher VSFCsentiment LR5e-5 seed42,cùng recipe.

Teacher artifacts ở `artifacts/e2/stage0_teacher/vsfc_sentiment/<LR>/seed_42/`. Đây là3trials trong grid15teacher, không phải trials ngoài plan. Chỉ `metadata.json` statuscompleted đủ30epochs mới được tính completed; xem liveepoch_curves/status thay vì PID cũ.

Controller saupilots sẽ kiểm gateCEreplay, đăng ký tái dùng replay và3teacher, rồi khởi chạy phần còn lại của61trials+5caches. E2c/seeds45–46/test vẫn đóng. Lệnh/PID/controllerlog cụ thể ở E2_EXECUTION_LOG.md khi operator bàn giao.

CE replay đã hoàn tất30epochs: best-validation F1 `0.8450763457314355`, bằng E1 same(task,LR1e-5,seed42), delta0. Root đã đánh dấu fidelity gate PASS. Ba teacher VSFC còn running; không coi các điểm interim của teacher là kết quả được chọn cuối cùng.

## Controller và NC đã khởi chạy

Controller PID497795, live state `artifacts/e2/.controller/state.json`, controller SHA256 `5bd87dc2d4e88c8f8b782fd550d52606dc6022db0a2a9a9f47fc2ab2abb0a5b6`. Controller chờ metadata hoàn tất của các manual runs, validate gate, đăng ký CE replay rồi chuyển sang frozen scheduler. Không cần giữ phiên tool của root để các process tiếp tục.

Snapshot process: teacher LR1e-5 PID436790 GPU0; teacher LR2e-5 PID387894 GPU1; teacher LR5e-5 PID436972 GPU3. GPU2 đang chạy NC VSFCsentiment lambda0.1 seed42 PID488258 sau khi CE replay PID387430 hoàn tất. NC là một phần của10screeningNC, không phải trial ngoài plan. Xem E2_EXECUTION_LOG.md cho đường dẫn output/log thực tế và mọi sửa lỗi đường dẫn vận hành. Luôn kiểm tra live state/process thay vì tái sử dụng PID lịch sử.

Hiện chỉ mới hoàn tất engineering replay; teacher/KD/NC experiment matrix chưa hoàn tất. Khi đủ kết quả, root sẽ review paired validation deltas theo từng task/seed để quyết định mở E2c hoặc xác nhận5seeds. Test vẫn đóng cho E2.
