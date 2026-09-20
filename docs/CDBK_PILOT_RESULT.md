# CDBK pilot — Milestone 1: CafeBERT → BamiBERT KD trên Restaurant

Ngày 2026-09-17. Thực thi theo `docs/CAFEBERT_DAPT_BAMIBERT_KD_PLAN.md` (bản 2026-09-17) và
`docs/CAFEBERT_DAPT_BAMIBERT_KD_HANDOFF.md`, chỉ Milestone 1. Task `absa_restaurant`, seed 42, train 7028 / validation 771,
không chạm test. Ba lượt fine-tune, không grid, không DAPT, không E8, không seed thêm.

**Ghi chú thực thi:** cấu hình execution agent `gpt-5.6-luna`/`xhigh` trong AGENTS.md không tồn tại trong tooling của phiên này
(chỉ có sonnet/opus/haiku/fable); agent nhận bàn giao (Claude Opus 5) tự thực thi trực tiếp và ghi rõ giới hạn tại đây.

## Kết quả

| Arm | Khởi tạo | Loss | LR | Best epoch | Val pair micro-F1 | Pair P / R | Detection F1 | Runtime | GPU peak |
|---|---|---|---|---|---|---|---|---|---|
| T | uitnlp/CafeBERT (550M) | CE | 2e-5 | 21 | **77.72** | 77.76 / 77.68 | 87.89 | 23.3 min (cuda:2, chia sẻ) | 11.6 GB |
| B | BamiBERT gốc (103M) | CE | 5e-5 | 29 | **74.74** | 73.91 / 75.59 | 86.48 | 7.8 min (cuda:3, chia sẻ) | 3.1 GB |
| K | BamiBERT gốc (103M) | 0.5 CE + 0.5 KD(T) | 5e-5 | 24 | **73.94** | 73.91 / 73.98 | 85.24 | 8.9 min (cuda:2, chia sẻ) | 3.1 GB |
| E14 B (đối chiếu) | BamiBERT gốc | CE | 5e-5 | 29 | 74.74 | – | – | 7.1 min | – |

- **T − B = +2.98 pp** → qua gate "T hơn B ≥ 0.5" (bước 2 của plan), nên đã train K.
- **K − B = −0.80 pp** → rơi vào nhánh gate "gain < 0.2 hoặc giảm": **dừng pilot, kiểm correctness từ artifacts, không mở grid**.
  Không mở Milestone 2 (seed 43) vì nhánh đó chỉ dành cho gain 0.2–0.5. Các ngưỡng là quyết định đầu tư, không phải significance.
- K − T = −3.78 pp. Probe CafeBERT cũ (20 epochs, head khác, không lưu weights) cho 77.14; T mới 30 epochs cho 77.72, cùng bậc.

## Đường cong (validation pair F1 theo epoch, đầy đủ trong `epoch_curves.json`)

| Epoch | 2 | 5 | 10 | 14 | 20 | 21 | 24 | 29 | 30 | trung bình 10–30 |
|---|---|---|---|---|---|---|---|---|---|---|
| T | 56.71 | 72.83 | 76.39 | 76.99 | 76.94 | **77.72** | 76.60 | 76.06 | 76.10 | 76.14 |
| B | 32.69 | 63.52 | 71.00 | 74.00 | 74.10 | 73.22 | 73.56 | **74.74** | 74.74 | 73.30 |
| K | 12.99 | 56.73 | 68.88 | 70.82 | 72.84 | 73.46 | **73.94** | 72.67 | 73.16 | 71.95 |
| K − B | −19.70 | −6.79 | −2.12 | −3.18 | −1.26 | +0.24 | +0.39 | −2.07 | −1.58 | **−1.35** |

K thấp hơn B ở 25/30 epoch; 5 epoch K ≥ B (gồm epoch 1 cả hai = 0) đều ≤ +0.4. KD làm học chậm rõ ở epoch đầu (−19.7 ở epoch 2). Không phải artifact của việc
chọn best epoch. Train KD term (đã nhân T²) giảm 3.07 → 0.016 (KL thô ≈ 0.004/slot): student khớp teacher trên train gần hoàn toàn.

## Kiểm correctness từ artifacts (tất cả pass)

| Kiểm | Kết quả | File |
|---|---|---|
| Runner tái tạo E14 B | Pilot B trùng E14 B **bit-exact** cả 30 epoch (max diff loss 0.0, F1 0.0), predictions best epoch giống hệt | `checks/b_vs_e14_reproduction.json` |
| Loss contract (synthetic) | 12/12: KD(t,t)=0; KD = T²·KL(teacher‖student) thủ công (khác chiều ngược); T² đúng; gradient chỉ vào student; layout 12 slot × 4 đúng; CE gồm absent 0 và bằng E2 helper; total = 0.5 CE + 0.5 KD; bf16 autocast → loss fp32 hữu hạn | `checks/loss_checks.json` |
| Cache teacher | 9/9: IDs = train IDs có thứ tự; text hash khớp; shape [7028,12,4] fp32 hữu hạn; lookup qua loader shuffle của **student tokenizer** = index trực tiếp, labels khớp theo ID; teacher forward lại (fp32) lệch ≤ 5.5e-6 so với cache; không chứa labels/validation | `checks/cache_checks.json` |
| Checkpoint | Mỗi arm: reload best checkpoint → F1 bằng best; F1 tính lại từ `validation_predictions.json` bằng best; predictions reload giống hệt | `runs/*/metadata.json → postrun_checks` |
| B/K cùng init & data order | Smoke K epoch-1 CE = 1.4069 = smoke B; cùng code path E14 (`build_bamibert_classifier`, `seed_everything(42)`, loader seed 42); loss K đầu = 0.5·CE + 0.5·KD đúng số | log smoke, `runs/K_*/train.log` |
| Cache đúng teacher | `kd.teacher_hash` trong K = sha256 `T/best_checkpoint.pt`; cache ghi run T, epoch 21, F1 | `runs/K_*/metadata.json → kd` |

`finetune.py`/`common.py` không đổi giữa các lượt (mtime trước khi T/B launch); `source_hash` của K khác T/B chỉ vì 3 script phụ
(`check_losses.py`, `cache_teacher_logits.py`, `summarize.py`) được thêm sau khi T/B đã chạy.

## Quan sát mức prediction (validation, `checks/validation_prediction_analysis.json`)

- **KD không kéo K về phía T trên validation.** Slot agreement với T: B 96.87 %, K 96.65 %; trên 1279 slot "active": B 77.3 %, K 75.8 %;
  example-level (12 slot đúng hết): B 69.0 %, K 69.1 %. K gần B (97.3 % slot) hơn gần T.
- Slot T đúng/B sai: 153, K sửa được 63. Slot B đúng & T đúng nhưng K sai: 82. Net âm.
- K dự đoán ít cặp non-absent hơn (1054 so với B 1077; gold 1053) → recall giảm 1.6 pp, precision bằng B, detection F1 −1.2 pp.
- Per-aspect K − B: +5.47 DRINKS#QUALITY, +2.33 LOCATION#GENERAL; −11.19 DRINKS#PRICES (19 gold), −6.22 RESTAURANT#PRICES (47),
  −5.13 RESTAURANT#MISC (58), −3.30 AMBIENCE. Lỗ lớn nhất ở aspect nhỏ → cỡ nhiễu seed (0.5–1 F1 theo E21/E22).
- **Teacher logits trên train gần one-hot:** T đạt train pair-F1 99.97 %, argmax trùng gold 99.995 %; mean max-prob softmax T=2 là 0.947
  (T=1: 0.998). Với soft target như vậy KD trên train mang rất ít thông tin ngoài nhãn cứng, phù hợp với việc K khớp teacher trên train
  (KD → 0.016) nhưng không gần T hơn trên validation. Đây là quan sát từ artifacts, không phải kết luận về KD nói chung.

## Quyết định

Theo gate của plan: **dừng Milestone 1 tại đây, không mở M2/DAPT/E8, không sweep alpha/T/LR.** Cấu hình pilot (KD logits từ teacher
đã fine-tune trên chính train, T=2, alpha 0.5, 1 seed) chưa có tín hiệu để đầu tư tiếp. Root quyết định bước kế; các điểm root có
thể cân nhắc (chưa chạy, không nằm trong scope): teacher logits gần one-hot trên train là giới hạn cấu trúc của cách cache này; seed
đơn với nhiễu ~0.5–1 F1. GPU trống có sẵn lúc kết thúc nhưng không launch thêm vì ngoài scope.

## Artifacts

- Runs: `artifacts/cdbk_pilot/runs/{T,B,K}_absa_restaurant_seed42/` — `resolved_config.json`, `epoch_curves.json`,
  `best_checkpoint.pt` (model state_dict, không resumable; T 2.2 GB, B/K 0.4 GB), `best_validation_metrics.json`,
  `validation_predictions.json`, `metadata.json`, `status.json`, `train.log`.
- Teacher cache: `artifacts/cdbk_pilot/teacher_cache/absa_restaurant_train_logits_T_seed42.pt` (schema E2, fp32, chỉ train, join theo ID
  `absa_restaurant:train:NNNNNN`; sha256 `ce3cfd3d…`; teacher checkpoint sha256 `21c1094e…`).
- Checks: `artifacts/cdbk_pilot/checks/{loss_checks,cache_checks,b_vs_e14_reproduction,validation_prediction_analysis}.json`.
- Tổng hợp: `artifacts/cdbk_pilot/summary.json` (`python scripts/cdbk_pilot/summarize.py`).
- Code: `scripts/cdbk_pilot/{common,finetune,check_losses,cache_teacher_logits,summarize}.py`. Reuse: E1 loader/loss/evaluator
  (`src/goibert/train.py`), head + builder E14 (`src/goibert_e14/modeling.py`), `logits_kd_loss` và `build_teacher_logit_cache` E2.
  Không sửa gold/split/source/artifacts cũ.
- Teacher: `uitnlp/CafeBERT@af76fcf2` (XLMRobertaModel 24L/1024, pooler + MLM head bỏ, head E14 với H=1024), tokenizer
  `FacebookAI/xlm-roberta-large@c23d21b0` (sentencepiece sha256 trùng CafeBERT). Student: `models/bamibert-base-original`
  (HF `57bc1340`, safetensors `c8c8abe5…`). Recipe chung: 30 epochs, eff batch 32 (micro 32, accum 1), maxlen 256, AdamW wd 0.01,
  warmup 10 % + linear decay, clip 1.0, bf16 autocast + SDPA, torch 2.6.0+cu124, transformers 4.49.0, L40S dùng chung với tiến trình khác.
