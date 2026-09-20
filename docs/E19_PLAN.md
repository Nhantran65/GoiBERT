# E19 — DAPT-BamiBERT (goibert-b-dapt) + recipe E8 (NC + LLRD), screening 5 task

Người dùng duyệt 2026-09-14 (bước A trong [NEXT_DIRECTION_AFTER_E18.md](NEXT_DIRECTION_AFTER_E18.md)). Mục tiêu: kiểm nhanh xem recipe đã thắng trên student (NC + LLRD) có nâng model 110M đã DAPT lên trên E2 teacher CE và BamiBERT gốc không. Validation-only, seed 42, không test, không grid.

## Model và recipe (khóa)

- Encoder `models/goibert-b-dapt` (RoBERTa 12L×768, tokenizer trùng student), head 768 như E2 teacher (`goibert_e2.modeling.build_e2_teacher`). Không tham số mới.
- Loss: CE + λ_NC · symmetric KL(clean, noisy) đúng E2 `_train_loop` mode `nc` (noise của `goibert_e2.noise`), T=1. λ_NC theo E8: Sent 0,1; Topic 0,5; ViSFD 0,5; Restaurant 0,1; Hotel 0,1.
- Optimizer: `goibert_e7.optim.build_layerwise_optimizer` với `num_layers=12`, γ=0,8, wd 0,01 (validate coverage 12 blocks + embeddings + head).
- LR base = LR E2 teacher đã chọn bằng validation: Sent 2e-5, Topic 1e-5, ViSFD 5e-5, Restaurant 2e-5, Hotel 5e-5.
- 30 epochs, warmup 10%, clip 1, max_len 256, BF16/SDPA, effective batch 32 (micro 16 × accum 2; nếu OOM micro 8 × 4, báo rõ).
- Data/split/evaluator frozen (`include_test=False`); chọn best epoch theo selection_f1 validation.

## Arms

Một arm `dapt_nc_llrd` × 5 task. Tham chiếu (không rerun): E2 teacher CE (`artifacts/e2/stage0_teacher/<task>/<lr>/seed_42`: Sent 85,92 / Topic 82,17 / ViSFD 84,36 / Restaurant 74,62 / Hotel 73,22) và E14 BamiBERT gốc CE (86,33 / 79,53 / 84,38 / 74,74 / 73,24).

## Gate

≥3/5 task tăng ≥0,3 so E2 teacher CE, không task giảm >0,3. Đạt → seeds 43–44 rồi mở test một lần cho checkpoint chọn bằng validation. Không đạt → dừng, ghi kết quả.

## Vận hành

`src/goibert_e19/` (nếu cần), `scripts/run_e19.py` theo mẫu `scripts/run_e18.py` (preflight nhỏ / worker / run-all, controller_state, logs, source manifest). Không sửa E1–E18. Preflight tối thiểu: head 768 + LLRD coverage 12 blocks; parity forward với `build_e2_teacher`; một bước NC finite; strict checkpoint round-trip. GPU: chỉ GPU <5 GiB đã dùng và util ≤5% lúc dispatch, 1 worker/GPU, không chạm job khác. Thứ tự: absa_restaurant, absa_hotel, visfd, vsfc_topic, vsfc_sentiment.
