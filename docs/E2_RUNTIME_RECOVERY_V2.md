# E2 runtime recovery v2

Ngày 2026-09-08, queue E2 dừng sau khi scheduler phát hiện artifact đầu tiên hoàn tất. Đây là lỗi runtime điều phối, không phải lỗi huấn luyện:

```text
TypeError: metadata_matches() got an unexpected keyword argument 'e1_dependency_hash'
```

`source_hashes()` của scheduler trả bốn khóa (`e1_dependency_hash`, `e2_code_hash`, `e2_config_hash`, `orchestrator_hash`), còn validator trong scheduler đóng băng chỉ nhận ba khóa sau. Một số đường gọi truyền nguyên mapping bốn khóa bằng `**hashes`, nên queue dừng khi child đầu tiên hoàn tất.

## Host state trước recovery

Kiểm tra host lúc 2026-09-08 xác nhận không còn controller, scheduler hoặc training child nào sống:

| PID | Vai trò | Kết quả |
|---:|---|---|
| 807589 | VSFC Topic teacher, LR 1e-5 | dead, partial 21 epochs |
| 807657 | VSFC Topic teacher, LR 2e-5 | dead, partial 22 epochs |
| 807725 | VSFC Topic teacher, LR 5e-5 | dead, partial 22 epochs |
| 807793 | ViSFD teacher, LR 1e-5 | dead, artifact hoàn tất 30 epochs |
| 806382 | controller cũ | dead |
| 807077 | frozen scheduler cũ | dead |

Ba teacher VSFC Sentiment và một teacher ViSFD đã hoàn tất, cùng CE replay và NC pilot; tổng cộng sáu artifact đã hoàn tất trước recovery. Ba artifact Topic dở dang được giữ lại riêng, không nhận là checkpoint có thể resume và không chọn best-of-rerun.

## Adapter đã triển khai

[scripts/execute_e2_recovery_v2.py](../scripts/execute_e2_recovery_v2.py) bao quanh scheduler đóng băng mà không sửa `src/goibert_e2/`, `configs/e2/` hoặc [scripts/execute_e2_validation.py](../scripts/execute_e2_validation.py).

- Kiểm tra `e1_dependency_hash` riêng, sau đó gọi validator đóng băng bằng đúng ba keyword mà nó khai báo.
- Giữ nguyên `orchestrator_hash` trong recipe và metadata training: đó là SHA256 của scheduler đóng băng, `f626112d...` ở release E2 này.
- Ghi riêng SHA256 của adapter vào recovery report/state dưới `runtime_repair_hash`; không giả mạo hoặc ghi đè recipe hash.
- Trước khi handoff queue, kiểm tra PID host. PID còn sống luôn block duplicate launch.
- Artifact dead và hợp lệ được adopt/reuse. Artifact dead nhưng thiếu contract được chuyển vào `artifacts/e2/recovery/runtime_v2_*/interrupted/` cùng log và SHA256 từng file.
- Fresh rerun cùng planned identity chỉ được mở với cờ tường minh `--fresh-rerun-interrupted`; runner không có exact resume vì không lưu sampler/RNG checkpoint đủ để chứng minh resume.
- Runner child được mở bằng `start_new_session=True`, để lỗi controller/shell không làm mất các child còn sống. PID, command và log vẫn do scheduler ghi.

Recovery report nằm ở `artifacts/e2/.orchestrator/recovery_v2_last_preflight.json`; state lưu `interrupted_attempts` và `runtime_repair`. Artifact hoàn tất vẫn giữ canonical path và được scheduler reuse theo contract.
Trạng thái live của parent và các child nằm ở
`artifacts/e2/.controller/recovery_v2_status.json`. State controller cũ được
đánh dấu `superseded` và giữ nguyên lỗi gốc trong
`artifacts/e2/.controller/state.json`; không dùng state cũ để suy ra queue mới.

## Kiểm thử

`tests/test_e2_recovery_v2.py` kiểm tra:

- adapter truyền đúng ba keyword tới validator đóng băng và từ chối E1 hash sai;
- adoption completed artifact và archive interrupted artifact;
- live child chặn recovery trước khi archive/launch;
- cùng contract argument cho cache, KD và NC;
- child subprocess thật hoàn tất trong session riêng.

Kết quả sau khi thêm adapter: `57 passed` cho toàn bộ suite `tests/`. Các fixture không chạy training đầy đủ và không mở test split.

Recovery command phải chạy từ durable terminal, với memory guard đã được root chốt:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python scripts/execute_e2_recovery_v2.py \
  --stage all \
  --repo-root /home/kiettqa/GoiBERT \
  --output-root /home/kiettqa/GoiBERT/artifacts/e2 \
  --root-smoke-gate /home/kiettqa/GoiBERT/artifacts/e2/preflight/smoke_gate.json \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --poll-seconds 5 \
  --student-memory-guard-gib 5 \
  --teacher-memory-guard-gib 5 \
  --nc-memory-guard-gib 5 \
  --execute --fresh-rerun-interrupted
```

Recovery giữ `include_test=false`; test vẫn đóng hoàn toàn.
