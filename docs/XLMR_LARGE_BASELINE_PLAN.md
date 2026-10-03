# XLM-R-large gốc trên Restaurant / ViSFD / Topic — CafeBERT hơn nhờ kích thước hay nhờ CPT 4–5B token?

**Trạng thái:** plan bàn giao, chưa chạy. Ngày chốt 2026-09-21. Chi phí ~3,5 GPU-giờ, ~1,5 giờ wall trên 3 GPU. Validation-only, không test.

## 1. Câu hỏi và vì sao quan trọng

CafeBERT = **XLM-R-large (560M) + continued pretraining ~4–5B token tiếng Việt**. Ta chưa bao giờ đo XLM-R-large **gốc** trên benchmark, nên không biết phần hơn của Cafe (Restaurant val 77,72 / ViSFD 85,47 / Topic 81,74) đến từ đâu. Ba lần CPT liều 250M trên ba backbone (Bami, mmBERT, BidirLM) đều ≈ 0 ở ABSA; câu hỏi còn lại là **liều lớn (gấp 20 lần) có khác không** — và cách rẻ nhất để trả lời là đo chính nền của Cafe trước khi CPT.

| Kết quả | Kết luận | Hệ quả |
|---|---|---|
| XLM-R-large ≈ Cafe (Δ Restaurant ≤ 1,0) | lợi thế của Cafe là kích thước/kiến trúc encoder gốc | **không** chi LUMI cho CPT 1B/5B; dồn sức vào phía dữ liệu có nhãn trên nền B2/P_vi |
| XLM-R-large ≪ Cafe (Δ ≥ 2,0) | CPT liều lớn có tác dụng thật trên benchmark này | mở dose-response 1B → 5B trên BidirLM ở LUMI |
| 1,0 < Δ < 2,0 | CPT giúp vừa phải | root quyết; nghiêng về LUMI chỉ nếu rảnh |

Kết quả phụ (không phải gate): XLM-R-large gốc so với B2/P_vi (cùng cỡ ~560–596M, cùng "chưa/ít CPT tiếng Việt") — cho biết encoder xây từ decoder có hơn encoder gốc cùng cỡ không.

## 2. Model và môi trường

- `FacebookAI/xlm-roberta-large @ c23d21b0620b635a76227c604d44e43a9f0ee389` (MIT), 24 lớp × 1024, 560M. Cache local hiện **chỉ có tokenizer/config**, cần tải `model.safetensors` (~2,2 GB) đúng commit; ghi sha256 vào manifest.
- Env `.venv-e0-cuda` (transformers 4.49.0) như baseline round 15. Copy `scripts/base_encoder_baseline/{common.py,finetune.py,run_grid.sh,summarize.py}` sang `scripts/xlmr_large_baseline/`, đổi `ART` → `artifacts/xlmr_large_baseline`, thêm registry `xlmr_large` (family `xlm-roberta`, `add_pooling_layer=False`, expected 24 lớp / hidden 1024). Không sửa script/artifacts baseline cũ.

## 3. Recipe (E14, y hệt baseline round 15 và các tham chiếu)

CLS → dropout 0,1 → dense 1024 → tanh → dropout 0,1 → logits; CE; 30 epoch; eff. batch 32 (micro 32; hạ micro 16 nếu VRAM không đủ); maxlen 256; bf16 autocast, SDPA; AdamW wd 0,01, betas (0,9, 0,999); warmup 10 %; clip 1,0; best epoch theo validation; reload/prediction/ID checks. XLM-R-large hay **sập ở LR cao** (F1 kẹt ~0 hoặc hằng): nếu xảy ra, ghi nhận là "diverged", không chạy lại, không đổi recipe.

## 4. Run (seed 42 trừ khi ghi khác)

| # | Task | LR | Seed | Mục đích |
|---|---|---|---|---|
| 1–3 | Restaurant | 1e-5, 2e-5, 5e-5 | 42 | grid như Bami/mmBERT |
| 4–5 | ViSFD, Topic | LR tốt nhất Restaurant | 42 | so với Cafe/Bami/mmBERT |
| 6–7 | ViSFD, Topic | 2e-5 | 42 | **chỉ nếu** LR tốt nhất ≠ 2e-5 — so cùng LR với Cafe (Cafe chạy cố định 2e-5) |
| 8 | Restaurant | LR tốt nhất | 43 | mean 2 seed, so với Cafe 2 seed (77,72 / 76,40 → **77,06**) |
| 9 | Restaurant | 2e-5 | 43 | chỉ nếu LR tốt nhất ≠ 2e-5 |

Ước tính: Restaurant ~23 phút/run (như Cafe), ViSFD ~35, Topic ~40 → ≤ 3,5 GPU-giờ. Launcher như `run_grid.sh` (một job/GPU, VRAM trống ≥ 20 GB, không stack, pgrep neo `^\S*python scripts/xlmr_large_baseline/`, GPU theo chỉ dẫn root — mặc định 0–2, không dừng job người khác).

## 5. Tham chiếu (validation)

| Model | Restaurant | ViSFD | Topic | Ghi chú |
|---|---:|---:|---:|---:|
| CafeBERT (2e-5) | 77,72 (s42) / 76,40 (s43) → 77,06 | 85,47 | 81,74 | XLM-R-large + CPT 4–5B |
| BamiBERT | 74,74 | 84,38 | 79,53 | grid |
| mmBERT-base | 75,63 | 83,64 | 81,18 | grid |
| B2 (BidirLM-0.6B-Base) | 75,72 (s42) / 75,72 (s43) | 84,85 | 80,74 | 1e-5 |
| P_vi (B2 + 250M VN MNTP) | 76,15 | 85,44 | 80,69 | 1e-5 |
| XLM-R-base | 73,61 | — | — | round 15 |

## 6. Đọc kết quả (§1) — ghi rõ hai phép so

- **So best-of-grid:** Δ_Rest = Cafe mean 2 seed (77,06) − XLM-R-large mean 2 seed (LR tốt nhất).
- **So cùng LR 2e-5:** Δ_Rest(2e-5) = 77,06 − XLM-R-large 2e-5 mean 2 seed. Nếu hai phép so cho kết luận khác nhau, lấy phép **cùng LR** làm chính (cùng điều kiện với Cafe) và ghi cả hai.
- ViSFD/Topic: Δ một seed, chỉ mô tả.
- Không claim significance; |Δ| < 1,0 coi là chưa phân biệt.

## 7. Artifacts, báo cáo, điểm dừng

`artifacts/xlmr_large_baseline/{manifest.json, <task>/lr_<lr>/seed_<s>/…, summary.json}`; báo cáo `docs/XLMR_LARGE_BASELINE_RESULT.md`: pin/sha/tham số (kỳ vọng ~560M, embedding 250K×1024 = 256M), bảng task × LR × seed (val F1, best epoch, phút, diverged?), bảng so tham chiếu §5, Δ theo §6 (cả hai phép so), kết luận theo §1, checks, giới hạn (một–hai seed, validation-only). **Không test, không Hotel/Sentiment, không CPT, không sửa artifacts cũ.** Blocker (không tải được weights đúng commit; mọi LR đều diverged; VRAM không đủ micro 16) → báo cụ thể, dừng.
