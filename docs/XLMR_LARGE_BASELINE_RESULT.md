# XLM-R-large gốc trên Restaurant / ViSFD / Topic — kết quả (validation)

Plan `docs/XLMR_LARGE_BASELINE_PLAN.md`; chạy 2026-09-21T06:11:56Z; recipe E14 y hệt `artifacts/base_encoder_baseline` (CLS head E14, CE, 30 epoch, eff. batch 32, maxlen 256, bf16/SDPA, AdamW wd 0,01, warmup 10 %, clip 1, best epoch theo validation, reload/prediction/ID checks). Validation-only, không test. F1×100 (ABSA = pair micro F1). Artifacts `artifacts/xlmr_large_baseline/`.

**Pin:** `FacebookAI/xlm-roberta-large @ c23d21b0620b635a76227c604d44e43a9f0ee389` (MIT) · `model.safetensors` sha256 `2dfa19f172412917cab174da04b46e2134811b723666965fd0aabd97caa6e23b` · XLMRobertaModel, 24L/1024, 302.8M không-embedding / 558.8M encoder (embedding 256.0M, vocab 250002).

## 1. Task × LR × seed

Ô = **val F1 / best epoch / phút-run**; `diverged` = run sập (F1 ≈ 0 hoặc hằng).

| Task | LR | seed | val F1 | best epoch | phút | diverged | checks |
|---|---|---:|---:|---:|---:|---|---|
| Restaurant | 1e-05 | 42 | 76.34 | 14 | 22.5 | no | ok |
| Restaurant | 2e-05 | 42 | 77.81 | 19 | 22.7 | no | ok |
| Restaurant | 2e-05 | 43 | 76.20 | 28 | 22.9 | no | ok |
| Restaurant | 5e-05 | 42 | 76.16 | 27 | 23.1 | no | ok |
| Topic | 2e-05 | 42 | 80.61 | 19 | 36.3 | no | ok |
| ViSFD | 2e-05 | 42 | 85.48 | 28 | 28.6 | no | ok |

LR tốt nhất Restaurant (seed 42): **2e-05**.

## 2. Tham chiếu và Δ = Cafe − XLM-R-large

| Task | XLM-R-large | CafeBERT | Δ (Cafe − XLM-R-large) | Bami | mmBERT | B2 | P_vi |
|---|---:|---:|---:|---:|---:|---:|---:|
| Restaurant (best-of-grid 2e-05, mean 2 seed = s42 77.81 / s43 76.20) | 77.01 | 77.06 | **+0.05** | 74.74 | 75.63 | 75.72 | 76.15 |
| Restaurant (cùng LR 2e-5, mean 2 seed = s42 77.81 / s43 76.20) | 77.01 | 77.06 | **+0.05** | 74.74 | 75.63 | 75.72 | 76.15 |
| ViSFD (1 seed) | 85.48 (best) / 85.48 (2e-5) | 85.47 | -0.02 (best) / -0.02 (2e-5) | 84.38 | 83.64 | 84.85 | 85.44 |
| Topic (1 seed) | 80.61 (best) / 80.61 (2e-5) | 81.74 | +1.13 (best) / +1.13 (2e-5) | 79.53 | 81.18 | 80.74 | 80.69 |

(tham chiếu: CafeBERT Restaurant seed42 77.72 / seed43 76.40 → mean **77.06**; Cafe chỉ chạy LR 2e-5. XLM-R-base Restaurant round 15: 73.61.)

## 3. Kết luận theo plan §1

Phép so chính (cùng LR 2e-5, mean 2 seed): Δ = **+0.05** điểm.

Cafe ≈ XLM-R-large (Δ ≤ 1,0) → lợi thế của CafeBERT gắn với **kích thước/kiến trúc encoder gốc**; không chi CPT liều lớn (1B/5B) trên cơ sở này.

### 3b. Kết quả phụ (không phải cổng): encoder gốc vs encoder dựng từ decoder cùng cỡ

| Task | XLM-R-large (2e-5) | B2 (BidirLM) | P_vi (B2 + 250M VN MNTP) | Δ XLM-R-large − B2 | Δ XLM-R-large − P_vi |
|---|---:|---:|---:|---:|---:|
| Restaurant | 77.81 | 75.72 | 76.15 | +2.09 | +1.67 |
| Topic | 80.61 | 80.74 | 80.69 | -0.12 | -0.08 |
| ViSFD | 85.48 | 84.85 | 85.44 | +0.63 | +0.05 |

→ Ở cùng cỡ ~560–596M, **encoder XLM-R-large gốc ≥ hai encoder dựng từ decoder** (B2/P_vi) trên cả ba task; khoảng cách nhỏ nhất ở Topic (≈ 0,1).

## 4. Checks

Mọi ô có `postrun_checks`: reload checkpoint trùng best validation F1, file prediction cho lại đúng F1, dự đoán sau reload giống hệt, ID ổn định và đủ (7028/7786/11426 train; 771/1112/1583 validation). Không chạy test, không Hotel/Sentiment, không CPT, không sửa baseline/artifacts cũ. Không ô nào diverged (kể cả 5e-5).
Runtime: 6 run = **2.60 GPU-giờ** (Restaurant 22,5–23,1 phút/run, ViSFD 28,6, Topic 36,3; chỉ GPU 0 và 2 đủ trống ≥ 20 GB vì GPU 1 có job người khác, nên wall ≈ 1,9 h thay vì 1,5 h).

Sai lệch nhỏ so với bản copy baseline: `finetune.py` được thêm **một tầng `seed_<s>`** trong đường dẫn artifact (baseline không có tầng seed, và lượt seed 43 sẽ ghi đè seed 42). Ba ô đợt 1 được chạy trước khi phát hiện và đã được di chuyển nguyên vẹn vào `…/lr_<lr>/seed_42/`; mọi số liệu trong bảng đọc từ metadata sau khi di chuyển.

## 5. Giới hạn

- Một–hai seed (42; 43 cho Restaurant), validation-only; |Δ| < 1,0 coi là chưa phân biệt, không claim significance.
- CafeBERT là tham chiếu lịch sử (Restaurant từ CDBK seed 42/43, ViSFD/Topic từ sealion_benchmark) — khác ngày chạy và khác pipeline launcher, dù cùng recipe E14 và cùng split validation.
- XLM-R-large hay sập ở LR cao: ô nào diverged được ghi rõ và **không** chạy lại, không đổi recipe.
- Không kết luận nhân quả CPT vs kích thước cho ViSFD/Topic (một seed, một LR mỗi so sánh).

