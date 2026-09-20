# CafeBERT aspect-pair — xác nhận seed 43 trên Restaurant

Ngày 2026-09-18. Tiếp `docs/CAFEBERT_ASPECT_PAIR_RESULT.md` (seed 42). Câu hỏi: gain của arm A (`(review, "mục NN")`, shared head 4 lớp)
so với reference T (review → 48 logits) có lặp lại ở seed 43 không? Đúng 2 lượt mới, cùng seed 43 cho T và A; không chạy B, seed 44,
test, task khác, quét LR/prompt hay thêm loss. Agent chính (Claude Opus 5) tự thực hiện theo `AGENTS.md`, không tạo subagent.

## 1. Kết quả

| Seed | Arm | Best epoch | Val pair micro-F1 | Pair P / R | Detection F1 | Runtime | GPU | PID | Trạng thái |
|---|---|---|---|---|---|---|---|---|---|
| 42 | T | 21 | 77.7197 | 77.76 / 77.68 | 87.89 | 23.3 min | cuda:2 (chia sẻ) | 3632388 | completed (đã có) |
| 42 | A | 23 | 78.4536 | 77.90 / 79.01 | 89.20 | 113.5 min | cuda:3 (chia với B) | 3828667 | completed (đã có) |
| 43 | T | 23 | **76.3961** | 75.51 / 77.30 | 87.56 | 19.3 min | cuda:2 (cạnh ray worker idle) | 1032629 | completed, all checks ok |
| 43 | A | 20 | **78.0442** | 77.28 / 78.82 | 88.86 | 67.1 min | cuda:3 (riêng) | 1032630 | completed, all checks ok |

| Delta A − T | pp |
|---|---|
| seed 42 | **+0.7338** |
| seed 43 | **+1.6481** |
| **mean paired delta (2 seed)** | **+1.1910** |
| mean T / mean A | 77.058 / 78.249 |

Mọi F1 được tính lại từ `validation_predictions.json` (12 slot đã gộp) bằng evaluator gốc, khớp số đã lưu đến 1e-9.

**Đọc theo tiêu chí đã đặt:** cả 2 seed dương và mean delta ≥ 0.5 → **tín hiệu đủ mạnh để cân nhắc mở rộng ở lượt sau.** Ngưỡng là
heuristic, không phải significance (n = 2 seed). Lưu ý: gain seed 43 lớn hơn chủ yếu vì **T rớt** (77.72 → 76.40, −1.32) trong khi A
giảm ít (78.45 → 78.04, −0.41); với 2 seed chưa nói được A ổn định hơn T theo seed.

## 2. Đường cong seed 43 (chỉ mô tả; tiêu chí chọn checkpoint không đổi — best validation pair F1, tie chọn epoch sớm)

| Epoch | 1 | 2 | 5 | 7 | 10 | 12 | 15 | 17 | 19 | 20 | 21 | 23 | 24 | 27 | 30 | TB 10–30 | TB 17–30 (bỏ best) | Second-best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T43 | 26.42 | 58.28 | 72.83 | 75.59 | 74.99 | 76.25 | 74.74 | 75.49 | 75.31 | 75.33 | 76.11 | **76.40** | 75.80 | 75.93 | 75.51 | 75.57 | 75.62 | 76.25 |
| A43 | 0.19 | 55.02 | 76.26 | 76.50 | 75.09 | 77.03 | 76.92 | 76.70 | 77.27 | **78.04** | 75.40 | 77.21 | 77.39 | 76.38 | 75.41 | 76.33 | 76.39 | 77.39 |

- A43 ≥ T43 ở **24/30** epoch cùng chỉ số; A43 vượt best của T43 (76.40) ở **10 epoch** (7, 12, 15, 17–20, 22–24) — khác seed 42, nơi A
  chỉ vượt best T ở 1 epoch. Plateau 17–30 (bỏ best): A43 76.39 vs T43 75.62 (+0.77); second-best A43 77.39 vs T43 76.25 (+1.14).
  Nghĩa là ở seed 43 gain không phụ thuộc vào một epoch đỉnh đơn lẻ.
- Tuy vậy A43 vẫn dao động mạnh (78.04 → 75.40 giữa epoch 20 và 21); T43 phẳng hơn (75.3–76.4 từ epoch 12).
- A43 epoch 1 = 0.19: head chung 4 lớp seed 43 dự đoán gần như toàn absent sau epoch đầu (recall ~0), rồi bắt kịp từ epoch 2; không phải lỗi
  pipeline (loss epoch 1 = 0.670, giảm bình thường; validation epoch 2 = 55.0).
- Train loss cuối: T43 0.0020, A43 0.0004 (như seed 42).

Seed 42 để đối chiếu (từ report trước): T42 TB 17–30 bỏ best 76.15 / second-best 76.99; A42 76.79 / 77.26.

## 3. Kiểm correctness seed 43 (`artifacts/cafebert_aspect_pair_confirmation/seed43/{checks/pre_launch.json, summary.json, */seed_checks.json}`)

| Kiểm | Kết quả |
|---|---|
| Seed thực sự vào RNG/head-init/sampler | Adapter `confirm_seed.py` nạp `scripts/cdbk_pilot/{common,finetune}.py` (T) và `scripts/cafebert_aspect_pair/{common,finetune}.py` (A) theo path, gán lại `finetune.SEED` và `common.SEED` = 43 (mọi chỗ dùng seed đều đọc hai global này lúc gọi), assert trước khi chạy; `resolved_config.json`/`metadata.json`/`best_checkpoint.pt.metadata` ghi `seed: 43` |
| Head init | Hash head sau khởi tạo (spy quanh `build_model` cho T; `model.head_init_sha256` cho A) **bằng** hash kỳ vọng tính độc lập trước launch với `seed_everything(43)`: T `9812829d…`, A `1439ac69…`; khác hash seed 42 (A42 `8d1ceb77…`, tái tạo được bằng cùng phương pháp → phương pháp hợp lệ); reproducible |
| Cùng thứ tự review T/A seed 43 | Hash order mỗi epoch (spy collate của train loader T; log epoch của A) **trùng nhau cả 30 epoch** và trùng hash kỳ vọng pre-launch (`2027c2ab…` toàn epoch); khác seed 42 ở mọi epoch; mỗi epoch là một hoán vị đầy đủ 7028 review |
| Encoder khởi tạo | CafeBERT pretrained gốc (`uitnlp/CafeBERT@af76fcf2`, cùng file safetensors với seed 42), không từ checkpoint fine-tuned seed 42 (code path `build_model` không nhận checkpoint) |
| Budget cập nhật | Cả hai: 220 step/epoch, `total_steps` 6600, `warmup` 660, `global_step` cuối 6600, LR cuối 0.0; T micro 32 review (accum 1), A group 32 review = 384 pair, micro 64 pair, weighting `n_pairs_micro / n_pairs_group` (group cuối 240 pair) |
| Checkpoint reload | Reload best (T epoch 23 / A epoch 20) → F1 = best; F1 tính lại từ file predictions = best; predictions reload giống hệt |
| ID/gold | Hàng predictions của cả 4 run khớp ID và gold của loader validation (771 × 12); A gộp từ 9 252 cặp theo `(id, slot)` |
| Data | Giữ nguyên câu; A: 84 336 pair train, 9 252 validation, 0 truncation, đủ 12 aspect kể cả absent; mapping "mục 01"…"mục 12" không đổi |
| Source seed 42 | Không sửa `scripts/cdbk_pilot/*`, `scripts/cafebert_aspect_pair/{common,finetune,preflight,summarize}.py` hay artifacts seed 42; file mới: `confirm_seed.py`, `confirm_check.py`, `confirm_summarize.py` |

## 4. Vận hành

- Mỗi run một GPU, chọn theo free VRAM lúc launch (07:18 UTC): T → cuda:2 (24 GB trống cạnh ray worker idle 21 GB, không dừng), A → cuda:3
  (45 GB trống, không có process ngoài). cuda:0 (22 GB trống nhưng 62 % util của job khác) và cuda:1 (9.5 GB trống) không dùng.
  Ghi trong `checks/launch_decision.json`.
- T43: 37.8 s/epoch, peak 10.8 GB, 07:18:44 → 07:37:55 UTC. A43: 133 s/epoch (chạy một mình, so 226 s khi chia GPU ở seed 42), peak 16.9 GB,
  07:18:48 → 08:25:45 UTC. Smoke adapter trước launch: `../smoke/{T,A}` (checkpoint smoke đã xóa).

## 5. Kết luận và dừng

- Seed 43 lặp lại chiều dương của A − T với biên lớn hơn (+1.65); mean paired delta 2 seed **+1.19 pp**, cả hai dương → theo tiêu chí:
  *đủ mạnh để cân nhắc mở rộng ở lượt sau*. Chưa phải significance; biến thiên T giữa hai seed (1.32 pp) cùng cỡ delta seed 42.
- Vẫn giữ các giới hạn của report seed 42: không cô lập compute/head (A đọc review 12 lần, head chung 4 lớp), chi phí inference 12×,
  B (mô tả) không hơn A nên gain không thuộc về ngữ nghĩa tên aspect.
- **Dừng tại đây** theo scope: không chạy seed 44, B, test, task khác, quét LR/prompt hay thêm loss. Việc mở rộng (thêm seed, Hotel/ViSFD,
  hoặc đối chứng compute-matched) là quyết định của root ở lượt sau.

## 6. Artifacts

- `artifacts/cafebert_aspect_pair_confirmation/seed43/{T,A}_absa_restaurant_seed43/` — `resolved_config.json`, `epoch_curves.json`,
  `best_checkpoint.pt` (model state_dict, 2.2 GB; sha256 T `d07f20fe…`, A `2a8181a2…`), `best_validation_metrics.json`,
  `validation_predictions.json`, `metadata.json`, `status.json`, `seed_checks.json`, `train.log`.
- `artifacts/cafebert_aspect_pair_confirmation/seed43/checks/{pre_launch.json, launch_decision.json}`, `summary.json`
  (`python scripts/cafebert_aspect_pair/confirm_summarize.py`).
- Code mới: `scripts/cafebert_aspect_pair/{confirm_seed,confirm_check,confirm_summarize}.py`. Seed 42: `artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42/`,
  `artifacts/cafebert_aspect_pair/runs/A_absa_restaurant_seed42/` (không đổi).
