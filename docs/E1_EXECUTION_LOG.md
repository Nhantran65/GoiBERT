# E1 execution log

Ngày bắt đầu: 2026-09-08. Operator: `environment_smoke`. Log này chỉ ghi các lệnh E1 đã được chuẩn bị/chạy và trạng thái artifact; root giữ quyền mở gate nghiên cứu.

## Preflight hiện tại

- E0 environment/checkpoint smoke: PASS, xem [E0_ENVIRONMENT_REPORT.md](E0_ENVIRONMENT_REPORT.md).
- CUDA venv: `.venv-e0-cuda`, Python 3.12.3, PyTorch 2.6.0+cu124, Transformers 4.49.0.
- `uv pip check`: PASS cho `.venv-e0-cuda` và `.venv-e0`.
- GPU inventory lúc preflight: GPU3 NVIDIA L40S còn 45,327 MiB; GPU0 còn 41,283 MiB; GPU1 còn 45,434 MiB; GPU2 còn 16,281 MiB. Không dừng process hiện có.
- `vsfc_sentiment` được data audit đánh dấu `training_ready=true`; pilot validation-only không đọc test.
- Root gate decisions hiện tại: [E0_GATE_DECISIONS.md](E0_GATE_DECISIONS.md). Root đã mở full pilot sau runner integration/unit/BF16 smoke gate.
- Runner source hash khi tạo plan cũ: `a3c9a4216079fe7fb691e2bde75f1319e872feaf20fd170352a387b9191c4a6a`. Sau các pipeline fixes, final frozen source hash là `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.

Lệnh dry-run đã kiểm tra queue (không launch):

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python -m goibert.run_grid \
  --stage screening --tasks vsfc_sentiment --devices cuda:3 \
  --precision bf16 --attention-implementation sdpa \
  --min-free-memory-gib 8 --dry-run
```

Dry-run tạo plan 5 learning rates tại [screening_index_20260907T174002Z.json](../artifacts/e1/screening_index_20260907T174002Z.json). Đây chỉ là queue plan; chưa có trial completed.

## Pilot chờ GO gate

Target: `vsfc_sentiment`, screening, seed `42`, learning rate `2e-5`, 30 epochs, effective/micro batch 32, BF16, GPU3. Artifact directory được giữ đúng layout để tái dùng trong screening queue:

`/home/kiettqa/GoiBERT/artifacts/e1/vsfc_sentiment/screening/2e-05/seed_42`

Exact command sau khi root gửi GO gate và baseline_pipeline xác nhận task-level runner smoke:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python -m goibert.train \
  --task-id vsfc_sentiment \
  --learning-rate 2e-05 \
  --seed 42 \
  --repo-root /home/kiettqa/GoiBERT \
  --output-dir /home/kiettqa/GoiBERT/artifacts/e1/vsfc_sentiment/screening/2e-05/seed_42 \
  --device cuda:3 \
  --precision bf16 \
  --attention-implementation sdpa \
  --stage screening \
  --expected-source-hash 456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873
```

Trạng thái pilot 30 epochs: **GO / LAUNCHING**. Không launch bằng `run_grid --execute` cho pilot vì lệnh đó sẽ chạy cả 5 learning rates. Không xem test trong pilot. Preflight GPU3: 45,327 MiB free, 132 MiB used, no compute process on GPU3; existing workloads on GPU0/GPU2 left untouched. Final-hash focused smoke passed at [bf16_seed_42_final_hash456e3020](../artifacts/e1/vsfc_sentiment/smoke/bf16_seed_42_final_hash456e3020). Pilot stdout/epoch artifacts will be appended below.

## Integration/unit và task-level smoke đã chạy

Lệnh kiểm thử:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python -m pytest -q tests
```

Kết quả: **15 passed**. Sau test, source hash ổn định là `d3dff1d2a70fd50ebe00aa9c4e96ea63507c2e0b1bfdc4d68ad6a479819bbded`.
Output tóm tắt được lưu tại [integration_pytest.txt](../artifacts/e1/integration_pytest.txt).

Task smoke validation-only đã chạy trên GPU3, không đọc test:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python -m goibert.train \
  --task-id vsfc_sentiment --learning-rate 2e-05 --seed 42 \
  --repo-root /home/kiettqa/GoiBERT \
  --output-dir /home/kiettqa/GoiBERT/artifacts/e1/vsfc_sentiment/smoke/bf16_seed_42 \
  --device cuda:3 --precision bf16 --attention-implementation sdpa \
  --stage screening \
  --expected-source-hash d3dff1d2a70fd50ebe00aa9c4e96ea63507c2e0b1bfdc4d68ad6a479819bbded \
  --smoke
```

Kết quả: **completed**, 2 epochs trên 24 train + 24 validation rows, `include_test=false`, max length 256, best validation fixed-label macro F1 `0.2342342342` ở epoch 1, elapsed `1.77s`. Artifact đầy đủ gồm resolved config, epoch curves, validation predictions, checkpoint và metadata tại [vsfc_sentiment/smoke/bf16_seed_42](../artifacts/e1/vsfc_sentiment/smoke/bf16_seed_42). GPU3 sau run đã trở về mức 45,327 MiB free.
Stdout/command snapshot: [stdout.log](../artifacts/e1/vsfc_sentiment/smoke/bf16_seed_42/stdout.log).

Smoke với `sdpa` phát cảnh báo rằng CUBLAS workspace chưa được đặt và memory-efficient attention backward không bitwise-deterministic dù runner bật `torch.use_deterministic_algorithms(..., warn_only=True)`. Đây là giới hạn reproducibility cần ghi khi chạy pilot; không có test access hoặc non-finite loss/gradient.

## Full pilot và validation orchestrator

Full pilot dùng final frozen source hash `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`, chạy PID `4083334` trên GPU3 với `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Trial completed 30/30 epochs trong `638.06s`; metadata xác nhận `include_test=false`, batch/effective batch 32, BF16/SDPA, best epoch 7 và validation macro-F1 `0.8433919944789509`. Epoch 30 macro-F1 là `0.8081654422479545`; checkpoint được giữ theo best validation epoch. Artifact canonical nằm tại [vsfc_sentiment/screening/2e-05/seed_42](../artifacts/e1/vsfc_sentiment/screening/2e-05/seed_42).

Orchestrator [execute_e1_validation.py](../scripts/execute_e1_validation.py) ban đầu chặn do thấy task smoke artifacts có cùng task/stage/LR/seed identity. Đã bổ sung exclusion rõ ràng cho đường dẫn chứa `smoke`; không thay `src` hoặc frozen source hash. `py_compile` PASS. Lần launch sau đang chạy durable dưới parent PID `4157853` (PPID `3711987`), state tại [state.json](../artifacts/e1/.orchestrator/state.json), log từng job tại [orchestrator logs](../artifacts/e1/.orchestrator/logs/).

Orchestrator đã reuse đúng pilot (`reused=1`) và khởi chạy bốn trial screening còn lại của VSFC sentiment:

| GPU | PID | LR | Epochs đã flush tại handoff | Validation macro-F1 gần nhất |
|---:|---:|---:|---:|---:|
| cuda:0 | 4158616 | 1e-5 | 2 | 0.707137 |
| cuda:1 | 4158703 | 5e-5 | 2 | 0.808425 |
| cuda:2 | 4158774 | 1e-4 | 2 | 0.800739 |
| cuda:3 | 4158858 | 2e-4 | 2 | 0.802340 |

Tại handoff, GPU free memory lần lượt khoảng 41,283 / 43,371 / 14,218 / 43,263 MiB; tất cả vượt guard 8 GiB, dù GPU2 còn workload VLLM. Không có test job nào được tạo. Orchestrator sẽ tiếp tục screening các task còn lại, sau đó shortlist và final theo state/manifest; không rerun pilot.

## Final operator handoff checks

The smoke-collision fix is at [execute_e1_validation.py](../scripts/execute_e1_validation.py), inside `candidate_records()` immediately after the `.orchestrator` exclusion (the `if "smoke" in artifact_dir.parts: continue` guard). It excludes capped task-smoke directories before identity matching, while the canonical completed pilot remains discoverable. Script SHA256 at handoff: `990a7a94218217ee4876d6dc8e1de294b26aa7aec13e7bd26213f6cccff9a3c3`; no edits occurred after launch.

Focused post-fix verification, without model execution:

```bash
PYTHONPATH=src .venv-e0-cuda/bin/python -m pytest -q tests/test_execute_e1_validation.py
# 5 passed in 3.31s
```

Manual candidate scan returned exactly one record for the pilot identity, `/home/kiettqa/GoiBERT/artifacts/e1/vsfc_sentiment/screening/2e-05/seed_42`; both smoke directories were ignored.

Durable orchestrator launch was recorded at `2026-09-07T18:06:59Z` (`2026-09-08T01:06:59+07:00`) with parent PID `4157853`, PPID `3711987`:

```bash
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
  .venv-e0-cuda/bin/python scripts/execute_e1_validation.py \
  --stage all --repo-root /home/kiettqa/GoiBERT \
  --output-root /home/kiettqa/GoiBERT/artifacts/e1 \
  --tasks vsfc_sentiment vsfc_topic visfd absa_restaurant absa_hotel \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 --poll-seconds 30 \
  --wait-for-pilot \
  --pilot-path /home/kiettqa/GoiBERT/artifacts/e1/vsfc_sentiment/screening/2e-05/seed_42 \
  --pilot-pid 4083334 --execute
```

At final handoff, PID `4157853` and its four screening children remained alive, state reported `reused=1`, `approved.screening=1`, `failed=0`, and the queue had no test jobs.
