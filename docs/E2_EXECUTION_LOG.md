# E2 execution log

Ngày bắt đầu vận hành: 2026-09-08. Operator: `e2_operator`. Root giữ quyền mở gate nghiên cứu và quyết định khi nào chạy full queue; operator không tự mở E2 ngoài gate.

## Preflight — completed

Preflight được ghi bằng snapshot thật, không đặt thời điểm giả. Các artifact liên quan:

- [GPU inventory](../artifacts/e2/preflight/gpu_inventory_20260908T144813+0700.json)
- [E1 artifact inventory](../artifacts/e2/preflight/e1_artifact_inventory_20260908T144839+0700.json)
- [Runtime/disk/memory](../artifacts/e2/preflight/runtime_disk_memory_20260908T144839+0700.json)

Kết quả hiện tại:

- Máy thấy đủ 4 GPU NVIDIA L40S, driver `580.126.20`, mỗi GPU 46,068 MiB. PyTorch `2.6.0+cu124` thấy CUDA và cả 4 GPU; BF16 được hỗ trợ trên cả 4 GPU.
- Snapshot lúc `2026-09-08T14:48:13+07:00`: GPU0 còn 32,509 MiB, GPU1 còn 36,771 MiB, GPU2 còn 7,569 MiB, GPU3 còn 36,727 MiB.
- Theo guard khởi đầu, GPU0/1/3 đủ cho student và teacher; GPU2 không đủ guard 8 GiB student hoặc 14 GiB teacher nên được để nguyên cho workload hiện tại. Không process nào bị dừng hoặc chiếm dụng.
- Disk còn khoảng 2.97 TB (62% đã dùng); RAM khả dụng khoảng 873 GiB.
- E1 selection metadata đã đầy đủ: 55/55 records completed, mỗi task có 11 records, `include_test=false`, đầy đủ metadata/checkpoint/config/validation predictions, cùng source hash `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.
- Preflight này không mở test split, test predictions hoặc test metrics.

GPU2 đang có VLLM process dùng khoảng 29,148 MiB; các worker khác trên GPU0/1/2/3 cũng được giữ nguyên. Mọi launch E2 phải snapshot lại memory ngay trước launch và dừng nếu không đạt guard.

## Chờ source/gate

E2 runner/loss/orchestration agents đang hoàn thiện namespace `src/goibert_e2/`, `configs/e2/` và các test/contract liên quan. Operator không sửa source hoặc config. Trước khi chạy smoke, cần xác nhận:

1. loss unit tests và runner/orchestration tests đã PASS;
2. source/config freeze cùng E1 dependency hash đã được ghi;
3. teacher/student loader, train-logit cache hash/ID checks và masked ViSFD slot-OTHERS tests đã có artifact;
4. root gửi GO cho các full runs.

### Integration check snapshot

The integration check was recorded at `2026-09-08T14:54:48+07:00`; the available E2 loss/noise and cache/model
unit tests were run with `PYTHONPATH=src .venv-e0-cuda/bin/python -m pytest -q
tests`; the result was **29 passed**. This is only a software check; no GPU
model run was launched. The runner review still has launch blockers recorded
in [E2_REVIEW.md](E2_REVIEW.md): KD temperature scaling, explicit ABSA slot
counts, stable-ID noise streams, cache provenance/identity, and runtime device
metadata must be resolved and retested before GPU smoke.

## Planned operator actions after smoke gate

Theo [E2 execution plan](E2_EXECUTION_PLAN.md), operator sẽ chạy các smoke nhỏ trên GPU thật cho teacher/student, sau đó profile teacher với batch hiệu dụng 32, microbatch 8, accumulation 4, sequence length 256. Student guard là 8 GiB; teacher guard là 14 GiB. Smoke/profile phải lưu command, PID, stdout/stderr, observed start/end timestamps, device, peak memory, finite loss/gradient và artifact hash.

### GPU smoke/profile snapshot

Hai smoke validation-only `vsfc_sentiment`, seed 42, BF16/SDPA, một epoch trên
subset smoke, microbatch 8/accumulation 4 đã hoàn tất thành công trên GPU0
teacher và GPU1 student CE. Metadata ghi đúng device (`cuda:0`/`cuda:1`),
`include_test=false`, finite loss/gradient, và không có test access. Teacher
smoke peak reserved là 2,348,810,240 bytes; student CE smoke peak reserved là
631,242,752 bytes. Các artifact nằm ở
`artifacts/e2/smoke/teacher_vsfc_s42_gpu0_run2` và
`artifacts/e2/smoke/student_ce_vsfc_s42_gpu1_run2`.

Teacher train profile với synthetic sequence 256, effective batch 32,
microbatch 8, accumulation 4, BF16/SDPA đã hoàn tất trên GPU0. Một warmup và
hai measured optimizer updates đều finite; peak allocated 2.632 GiB, peak
reserved 2.773 GiB. Bằng chứng JSON và stdout nằm ở
[profile_teacher_train_20260908T150331+0700.json](../artifacts/e2/preflight/profile_teacher_train_20260908T150331+0700.json)
và file `.stdout.log` cùng thư mục. Đây là capacity profile, chưa phải full
training result.

Student NC profile với hai forward mỗi microbatch, cùng batch/sequence và
BF16/SDPA, đã hoàn tất trên GPU3. Peak allocated là 1.176 GiB, peak reserved
là 1.252 GiB; mọi loss/gradient đều finite. Bằng chứng nằm ở
[profile_student_nc_train_20260908T150441+0700.json](../artifacts/e2/preflight/profile_student_nc_train_20260908T150441+0700.json)
và file `.stdout.log` cùng thư mục. Theo snapshot trước profile, GPU2 còn
7,571 MiB; profile này cho thấy student NC có margin lớn hơn 1 GiB ở GPU2,
nhưng scheduler vẫn phải kiểm tra memory tươi ngay trước mỗi launch.

### Final-freeze smoke and CE fidelity

Sau final freeze (`efb81ac2…`, config `45e6c361…`, orchestrator
`f626112d…`), CE và NC smoke với microbatch 32/accumulation 1 đều hoàn tất
trên GPU3/GPU2; mỗi run có finite loss/gradient và `include_test=false`.
Scalar VSFC KD và packed ViSFD KD với train-only cache cũng hoàn tất trên
GPU2/GPU3. Cache validation đã xác nhận source IDs, text/provenance hashes,
tokenizer, slot schema (ViSFD 10×4 + binary OTHERS), E1 dependency và final
E2 source hash.

CE fidelity probe cho thấy E2 dùng cùng callable `compute_loss`, optimizer,
loader và validation với E1; loss và gradient sai khác tuyệt đối bằng 0 trên
fixture FP32, optimizer groups giống nhau. Bằng chứng ở
[ce_fidelity_probe_20260908T152745+0700.json](../artifacts/e2/preflight/ce_fidelity_probe_20260908T152745+0700.json).

Initial student state fidelity cũng đã được kiểm tra với seed 42: E1
`build_classifier` và E2 `build_e2_student` có 169 state keys, cùng 26,546,691
tham số, state SHA256 giống hệt nhau và max tensor difference bằng 0. Bằng
chứng ở
[initial_state_fidelity_20260908T153034+0700.json](../artifacts/e2/preflight/initial_state_fidelity_20260908T153034+0700.json).

Để khớp recipe E1 mặc định, operator đo lại student với microbatch 32,
accumulation 1, sequence 256. CE peak reserved **1.900 GiB**; NC hai forward
peak reserved **3.256 GiB**; các loss/gradient đều finite. Kết quả đầy đủ ở
[profile_student_batch32_20260908T150818+0700.json](../artifacts/e2/preflight/profile_student_batch32_20260908T150818+0700.json)
và file `.stdout.log` cùng thư mục. Vì NC cần khoảng 3.256 GiB reserved,
GPU2 chỉ đủ theo snapshot nếu còn hơn khoảng 4.256 GiB free tại thời điểm
launch; scheduler phải dùng snapshot tươi và giữ margin 1 GiB.

Sau khi root mở gate, hai full run đầu tiên được preregistered:

- teacher pilot: `vsfc_sentiment`, LR `2e-5`, seed `42`, 30 epochs, GPU riêng;
- CE compatibility replay: student `vsfc_sentiment`, LR `1e-5`, seed `42`, 30 epochs, GPU khác.

Hai run này không được chạy trước GO, không đọc test và phải giữ E1 update recipe/initialization. Teacher task grid và student KD/NC queue chỉ được mở sau khi replay/smoke evidence được root review.

Mỗi job dài phải có PID/log sống được sau handoff. Không báo `completed` khi process còn chạy; không retry/cherry-pick run đẹp và không overwrite artifact identity.

### Full pilot handoff

Four GO-approved jobs are tracked in
[pilot_process_manifest_20260908T153824+0700.json](../artifacts/e2/preflight/pilot_process_manifest_20260908T153824+0700.json):
teacher LR2e-5 on GPU1 (PID 387894), CE replay on GPU2 (PID 387430), and the
additional teacher grid jobs LR1e-5 on GPU0 (PID 436790) and LR5e-5 on GPU3
(PID 436972). Root's `continue_e2_after_pilots.py` owns adoption and queue
continuation after these four canonical artifacts complete.

The NC pilot initially collided with its pre-created canonical directory: that
directory contained only the tee launch log, so the runner safely allocated
`seed_42__rerun1`. While PID 488258 was still running, the log-only directory
was moved to a diagnostic backup and canonical `seed_42` was made a relative
symlink to the active output. The process and original active path were left
untouched; the audit is recorded in
[nc_reconcile_audit_20260908T155121+0700.json](../artifacts/e2/preflight/nc_reconcile_audit_20260908T155121+0700.json).

### Runtime recovery v2 — 2026-09-08

Host-level inspection after the first scheduler failure found no live E2
controller, scheduler, or training child. PIDs 807589, 807657, and 807725 were
dead incomplete VSFC Topic attempts (21/22/22 epochs); PID 807793 was dead but
its ViSFD LR1e-5 artifact was a valid completed 30-epoch artifact. The exact
fault was the frozen scheduler passing `e1_dependency_hash` to a validator
whose declared API accepts only `e2_code_hash`, `e2_config_hash`, and
`orchestrator_hash`.

The versioned adapter [execute_e2_recovery_v2.py](../scripts/execute_e2_recovery_v2.py)
was added without changing the frozen scheduler or training source. It checked
the host PIDs, adopted the completed ViSFD artifact, archived the three partial
attempts with per-file SHA256 hashes under
`artifacts/e2/recovery/runtime_v2_20260908T101454Z_b0323759/`, and recorded
the repair hash `4ea4943a4b015861954d39102b3be5fba9d4db8f1b5c0771d6cef909453a5f3a`.
The frozen recipe/scheduler hash remains
`f626112d163a4c9ed476d9811bb74447b337d23bd6e87d9e105ab8e0a66807b8`.

The recovery command was started with `--fresh-rerun-interrupted` and the
approved 5 GiB free-memory guards. It reused the completed artifacts already
registered in the scheduler and launched fresh Topic trials with PIDs 1026602,
1026759, and 1026903 on GPU0/1/2, then filled GPU3 with the next planned
ViSFD LR2e-5 trial PID 1026981. All four children have their own session and
durable scheduler logs; the queue still passes `--include-test false`.
