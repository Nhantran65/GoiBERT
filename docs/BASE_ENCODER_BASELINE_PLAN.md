# Baseline mmBERT-base và XLM-R-base trên 5 task — plan đăng ký trước

Ngày 2026-09-19. Mục đích: quyết định có đầu tư pretraining tiếng Việt cho một base-size encoder hay không (bước 1–2 của lộ trình sau `docs/CAFEBERT_BAMIBERT_STEP0_BLIND_READING.md`). Chỉ fine-tune, validation-only, seed 42; **không test**, không CPT.

## Model (pin)
| Key | Repo @ commit | Kiến trúc | Ghi chú |
|---|---|---|---|
| `mmbert_base` | `jhu-clsp/mmBERT-base @ c5955035435e2bf121cde7f3c8863ef52ff35d82` | ModernBERT 22L/768, vocab 256K | ứng viên chính; báo cả tham số không-embedding |
| `xlmr_base` | `FacebookAI/xlm-roberta-base @ e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` | XLM-R 12L/768, vocab 250K | đối chứng cùng họ CafeBERT ở cỡ base |

## Recipe (= E14, không đổi)
Encoder HF (`AutoModel`, không pooler) + head E14 `CLS → dropout .1 → dense(H) → tanh → dropout .1 → logits` (`goibert_e14.modeling.BamiBERTClassifier`, `head_hidden_size = H = 768`); loader/split/evaluator E1; CE; AdamW wd 0,01, betas (0,9, 0,999), eps 1e-8; warmup 10% + linear; clip 1; effective batch 32 (micro 32); maxlen 256; bf16/SDPA; 30 epoch; best epoch theo metric chọn của task trên validation; reload + recompute từ predictions.

Grid cố định: **5 task × 3 LR {1e-5, 2e-5, 5e-5} × 2 model = 30 run**, seed 42. Chọn LR mỗi task = max validation, hòa → LR nhỏ hơn (như E14). Báo đủ 15 ô/model, không chỉ số max.

## Gate (viết trước, so với Bami E14 validation cùng grid; Cafe để tham chiếu)
mmBERT **đi tiếp** nếu: (i) ≥ Bami − 0,5 điểm trên ≥4/5 task; (ii) Restaurant ≥ Bami − 0,5; (iii) hơn Bami ≥ +0,5 ở Topic hoặc ViSFD. Kết quả sát biên (±0,5) → chạy seed 43 chỉ trên task quyết định trước khi kết luận. mmBERT thua Bami rõ → dừng hướng base-size CPT. Nếu mmBERT ≈ Bami nhưng cả hai kém Cafe ~3 điểm Restaurant → ghi nhận khoảng cách là size/CPT-large, cân nhắc XLM-R-large/CafeBERT làm base.

## Tài nguyên
GPU 0–2 (không dùng GPU 3), một job/GPU. Source `scripts/base_encoder_baseline/`; artifacts `artifacts/base_encoder_baseline/<model>/<task>/lr_<lr>/`; báo cáo `docs/BASE_ENCODER_BASELINE_RESULT.md`. Không sửa source/gold/artifacts cũ.
