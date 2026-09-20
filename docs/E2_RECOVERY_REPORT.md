# E2 controller recovery report

Ngày 2026-09-08, pilot controller E2 dừng ở bước chờ artifact. Đây là lỗi điều phối, không phải lỗi huấn luyện. Cả ba teacher VSFC (`1e-5`, `2e-5`, `5e-5`), CE replay và NC pilot đều đã ghi artifact hoàn tất 30 epoch, `include_test=false`, đúng release hash và checkpoint; không trial nào được chạy lại.

## Nguyên nhân

- Controller cũ coi basename `seed_42` là bắt buộc. NC pilot hợp lệ dùng thư mục collision-safe `seed_42__rerun1`, được trỏ bởi symlink canonical `seed_42`; validator từ chối target sau khi resolve symlink.
- Controller phải chấp nhận `coefficient=learning_rate` của teacher. Validator recovery hiện kiểm tra đúng identity này; bản controller bị lỗi trước đó đã không đồng nhất với metadata teacher thực tế.
- `release_hashes` dùng tên key khác với `source_hashes()` (`e1_dependency` so với `e1_dependency_hash`), khiến một lần recovery bị block dù giá trị hash đúng.
- `wait_pilot` nuốt lỗi validator và báo lỗi PID chung. Bản sửa thêm grace period và đưa lỗi validator cuối cùng vào thông báo; đây là hardening cho khả năng race khi artifact đang được ghi, không phải kết luận rằng race là nguyên nhân đã được chứng minh của lần block này.
- Hai metadata chẩn đoán E1 cũ trong thư mục `vsfc_sentiment/smoke/` thiếu trường `smoke`. Scheduler E2 quét metadata theo task/LR nên có thể coi chúng là reference hợp lệ hoặc báo ambiguous. Chỉ hai metadata diagnostic này được thêm `smoke=true`; bản gốc được backup và checksum trong [e1_smoke_metadata_migration.json](../artifacts/e2/recovery/controller_20260908T093016Z/e1_smoke_metadata_migration.json). Không sửa 55 approved E1 records, checkpoint, prediction, metric hoặc dataset.

## Kiểm tra sau sửa

- `PYTHONPATH=src .venv-e0-cuda/bin/python -m pytest -q tests`: **51 passed**.
- Controller regression tests: **14 passed**, gồm teacher/CE/NC metadata thực tế, NC symlink `seed_42 -> seed_42__rerun1`, completed artifact sau PID exit và lỗi validator được hiển thị. Sau khi thêm CE compatibility wrapper, toàn bộ suite hiện tại là **52 passed**.
- 15/15 truy vấn E1 reference cho 5 task × 3 teacher learning rates trả đúng một record.
- 15/15 paired E1 reference cho 5 task × seeds 42/43/44 trả đúng một record.
- Không đọc test split trong các kiểm tra recovery; E1 reference chỉ đọc metadata/validation score.

## CLI compatibility recovery

The frozen scheduler's `run_orchestration` already contains the intended `stagea_ce_replay` branch, but its CLI choices omit that stage. The controller now routes only this stage through [run_e2_ce_replay_compat.py](../scripts/run_e2_ce_replay_compat.py), which constructs the scheduler Namespace and calls the existing branch directly. The frozen scheduler source/config/orchestrator hashes remain unchanged. The wrapper hash, command log, and reuse result are recorded in [ce_compatibility_evidence.json](../artifacts/e2/.controller/ce_compatibility_evidence.json).

The wrapper reused the existing CE replay artifact (`F1=0.8450763457314355`, no new training job), then launched the original frozen `--stage all` CLI. At handoff, the real host controller was PID `806382`, the stage-all scheduler was PID `807077`, and four teacher jobs were visible on GPUs 0–3 for new task identities.

## Tiếp tục queue

State controller cũ và lần thử CLI lỗi được giữ nguyên tại [state.blocked.json.orig](../artifacts/e2/recovery/controller_20260908T093016Z/controller_state/state.blocked.json.orig) và `artifacts/e2/.controller/state.blocked_20260908T093501Z_cli_choice.json`. Một state tạm có PID sandbox `2` sinh ra bởi lần launch thử trong namespace cô lập cũng đã được archive ở `state.recovery_launch_stale_20260908T093359Z.json`; host-level `ps` xác nhận không có controller/training process tương ứng trước khi khởi động lại.

Hiện controller host PID `806382` và scheduler `--stage all` PID `807077` đang chạy; bốn teacher jobs đầu tiên đã được launch trên GPU0–3. Queue tiếp tục với guard 5 GiB, reuse exact artifact identity/hash và không truy cập test.
