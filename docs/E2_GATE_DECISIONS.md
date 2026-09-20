# E2 gates của root

2026-09-08. User đã yêu cầu plan và agents thực thi E2, và xác nhận dùng GPU2 nếu model fit. Không cần hỏi lại user cho các bước thuộc phạm vi này.

## Resource gate PASS

Root đã đọc profile full train-step BF16/SDPA length256. Student micro32 CE peak reserved1.9004GiB, NC3.2559GiB; teacher micro8acc4 đạt2.7734GiB. Chốt effective batch32, studentmicro32, teachermicro8; guard chung5GiB free. Cho phép GPU0/1/2/3, kiểm tra fresh free memory trước launch; không dừng workload khác.

## Release và integration đã được root kiểm tra

- E1 dependency: `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.
- E2 code: `efb81ac232de2f0ae4287688ad0125a1d84fb1e689e2b1566ef26d8d6e2b60b5`.
- E2 config: `45e6c36196fe95ae0eb9193e9197224bdf2980515c15ddf01998267b668b61b8`.
- Orchestrator: `f626112d163a4c9ed476d9811bb74447b337d23bd6e87d9e105ab8e0a66807b8`.

Combined E2 code+config hash `1a9d9b1d22f038f8231e5194a947515df1361c1b5603ed262e55f69e9e851ca6` là field khác code-only hash; không coi khác tên field này là drift.

Root đã đọc metadata `freeze_kd_vsfc_s42_gpu2`, `freeze_kd_visfd_s42_gpu3` và `final_nc_vsfc_s42_gpu2`: status completed, hashes khớp release. CE fidelity probe ghi loss/gradient FP32 sai khác0, optimizer/helper contracts khớp. Small-cache ID mismatches ở các smoke trước đã được phát hiện và không dùng vào run chính; cache mới và KD smoke phải cùng full/capped train scope. Chỉ giữ các smoke PASS của release cuối làm bằng chứng gate.

## GO có điều kiện cho hai full run đầu tiên

Operator được tự khởi chạy sau khi xác nhận các điều kiện bên dưới, không phải chờ thêm một vòng hỏi phép root:

1. Final E2 source/config/adapter/orchestrator release được các agent chốt; E1 dependency hash nguyên vẹn. Lưu hashes và exact commands trước launch.
2. Unit và integration tests PASS, đặc biệt KD T² đúng một lần, slotwiseKL, ViSFD OTHERS, stable-IDnoise, cache identity, CLI/metadata contracts.
3. Real GPU teacher/student smoke finite/backward/save-load; CE construction/gradient fidelity với E1 theo tolerance trong plan. Cache→KD/NC small-train smoke không lỗi trước khi mở các mode student tương ứng.
4. Các lỗi review ở E2_REVIEW.md đã được sửa hoặc được ghi rõ không áp dụng cho release cuối.

Hai run được phép: teacher VSFC sentiment LR2e-5 seed42 micro8acc4,30epochs; student CE replay VSFC sentiment LR1e-5 seed42 micro32,30epochs. Cả hai BF16/SDPA, train/validation-only, mỗi run GPU riêng. CE replay có thể dùng GPU2 nếu còn≥5GiB; teacher ưu tiên GPU0/1/3. Không lấy CE replay làm candidate thay E1. Teacher pilot hợp lệ được tính đúng một lần trong grid15teacher.

Operator báo ngay PID/log/metadata/epoch đầu. Nếu source/config còn đang đổi hoặc smoke fails, giữ fullrun HOLD và báo đúng lỗi; không launch bằng hash tạm để rồi trộn kết quả.

## Các bước sau hai run

Root review replay và teacher artifacts để mở queue còn lại theo plan61trainingtrials. Replay best-validation delta so E1 không được vượt0.5điểm mà chưa giải thích; recipe/init/step count phải đúng. Teacher strength không phải điều kiện bắt buộc cho KD, nhưng phải báo rõ.

E2c combination, seeds45/46, test và sửa kiến trúc chưa được mở. Không đọc E1 test predictions để thiết kế/cài đặt E2.

## GO bổ sung sau khi root đọc các full-run epoch đầu

Root đã xác minh `initial_state_fidelity_20260908T153034+0700.json`: state169keys của E1/E2 giống hệt, SHA256`bc787512489b6d1e8209d69f1715257aae5372b8efba23d1f9ead897920f77cf`, maxdiff0. CE loss/gradient FP32 cũng khớp0diff.

Hai fullrun đã khởi chạy đúng release: teacher VSFC LR2e-5 GPU1 đã có epoch2 finite; CE replay GPU2 có8epochs và best-validation F1 0.8450763457 khớp E1 cùng seed42/LR1e-5. Chưa gọi run chưa đủ30epochs là completed.

Phê duyệt mở thêm hai teacher VSFC LR1e-5 trênGPU0 và5e-5 trênGPU3, cùng seed42/micro8/30epochs. Cả ba teacher VSFC là ba thành viên của grid15 đã preregister, không phải trials bổ sung.

Phê duyệt durable controller thực hiện tiếp theo mà không cần hỏi lại khi các điều kiện đủ:

1. Chờ ba teacher VSFC và CE replay hoàn tất30epochs; validate exact recipe/hash/status/uniqueidentity và checkpoint đầy đủ. Nếu process chết hoặc artifact lỗi, dừng và ghi lý do.
2. CE replay so đúng E1(task=vsfc_sentiment,LR1e-5,seed42) trênvalidation. Nếu absdelta>0.005 F1 hoặc init/recipe không đúng, dừng studentKD/NC và báo root. Không thay E1 reference bằng replay.
3. Đăng ký/reuse CE replay qua stagea_ce_replay của scheduler; không chạy thêm một replay khác. Tái dùng đúng ba teacher VSFC đã hoàn tất.
4. Khởi chạy scheduler `--stage all` cho phần còn lại của plan61trainingtrials +5cachepasses, tất cảGPU0–3,guard5GiB,không test. Top1alpha/lambda được chọn theo seed42 và xác nhận43/44 nhưplan; không tự mở E2c/seeds45–46/test.

Controller riêng trong scripts/ được phép thêm để chờ và thực thi các gate trên, nhưng phải ghi hash/controllercommand/PID và không sửa frozen runner/config/orchestrator. Điều phối gốc, metadata và cache hash phải còn khớp release đã khóa. Chỉ bàn giao trạng thái running/queued khi process đã thực sự tồn tại; root sẽ tổng hợp kết quả validation khi hoàn tất.

## CE replay full gate PASS

Root đã đọc `artifacts/e2/stageA/ce_replay/vsfc_sentiment/seed_42/metadata.json`: completed30epochs, best-validation F1 `0.8450763457314355`, bằng đúng E1 cùng task/LR1e-5/seed42. Delta0, nhỏ hơn engineering alarm0.005. Initialstate và FP32 CEgradient fidelity cũng đã PASS như trên. Cho phép controller đăng ký/reuse replay ngay; không chạy lại. FullE2KD/NC gate baseline fidelity đã đạt, còn chờ teacher grid/caches theo dependency của kế hoạch.

## NC độc lập trên GPU2 sau replay

Để dùng GPU2 trong khi ba teacher VSFC còn chạy, root cho phép một full NC screening: VSFCsentiment, lambda0.1, seed42,LR1e-5,micro32,30epochs,BF16/SDPA,GPU2 nếu free≥5GiB. Đây là1/10NCscreening đã preregister, không phải trial bổ sung. NC không cần teacher/cache và CEfidelity gate đã đạt nên có thể chạy độc lập.

Operator thêm NC path/PID vào manifest của controller; controller phải chờ hoặc xác minh NC hoàn tất trước khi dùng lạiGPU2/chạyqueueall, và tái dùng đúng record này ở stagea_nc. Không launch một NC duplicate. Các stage khác và test gates giữ như trên.
