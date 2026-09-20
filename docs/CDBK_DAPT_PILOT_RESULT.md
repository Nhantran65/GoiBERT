# CDBK lượt D1 — CafeBERT-DAPT (social) → teacher TD → student KD_D trên Restaurant

Ngày 2026-09-17. Thực thi theo `docs/CAFEBERT_DAPT_BAMIBERT_KD_PLAN.md` (bản D1) và `docs/CAFEBERT_DAPT_BAMIBERT_KD_HANDOFF.md`.
Task `absa_restaurant`, seed 42, validation-only; **1 DAPT + 2 fine-tune + 1 cache**, không E8/NC/LLRD/sweep/seed khác/test.
T/B/K tái dùng nguyên từ M1 (`docs/CDBK_PILOT_RESULT.md`), không rerun. Agent chính tự thực thi, không tạo subagent.

## Kết quả

| Arm | Khởi tạo | Loss | LR | Best ep | Val pair micro-F1 | Pair P / R | Detection F1 | Runtime |
|---|---|---|---|---|---|---|---|---|
| T | CafeBERT gốc | CE | 2e-5 | 21 | 77.72 | 77.76 / 77.68 | 87.89 | 23.3 min (M1) |
| B | BamiBERT gốc | CE | 5e-5 | 29 | **74.74** | 73.91 / 75.59 | 86.48 | 7.8 min (M1) |
| K | BamiBERT gốc | 0.5 CE + 0.5 KD(T) | 5e-5 | 24 | 73.94 | 73.91 / 73.98 | 85.24 | 8.9 min (M1) |
| **TD** | CafeBERT + DAPT 300M social | CE | 2e-5 | 23 | **77.58** | 77.29 / 77.87 | 88.17 | 19.0 min |
| **KD_D** | BamiBERT gốc | 0.5 CE + 0.5 KD(TD) | 5e-5 | 22 | **73.34** | 72.89 / 73.79 | 85.61 | 8.1 min |

| Delta | pp | Đọc theo §7 plan |
|---|---:|---|
| TD − T | **−0.14** | teacher DAPT ngang teacher gốc ở best epoch (trong nhiễu seed 0.5–1) |
| KD_D − K | **−0.61** | không "bớt hại" hơn KD cũ |
| **KD_D − B** | **−1.41** | **< 0.2 pp: chưa có tín hiệu; student vẫn dưới baseline 74.7418** |
| KD_D − TD | −4.24 | |

Kết luận D1: **teacher DAPT không tăng, student không tăng.** Cấu hình DAPT-social + task-logit KD (T=2, alpha 0.5) chưa
giúp BamiBERT vượt B. Dừng theo plan; không chạy seed 43/E8/composite/general distillation/tăng token.

## Bước A — Corpus và DAPT

**Corpus** (`data/cdbk_dapt_pilot/manifest.json`): `phucdev/ViSoBERT@74dbc64e` — corpus pretraining của ViSoBERT (EMNLP 2023):
Facebook posts, TikTok/YouTube comments 2020–2022; điều kiện: "research purposes only" (card `uitnlp/visobert`), CC-BY-NC-4.0
(`SEACrowd/visobert`). **Không có nguồn review nhà hàng/ẩm thực tiếng Việt công khai đủ ≥100M token** (các tập review tìm được
trên HF ≤ ~26k dòng ≈ 3M token); đây là "social rộng" theo plan, **domain mismatch rõ**: comment ngắn (median 9 từ sau dedup),
nhiều tên tag/tục/teencode, chỉ **2.9 %** văn bản chứa từ khoá review/ăn uống/sản phẩm. 50 mẫu kiểm tay nằm trong manifest.

| Bước lọc | Docs |
|---|---:|
| raw (3 parquet, 862 MB, sha256 trong manifest) | 15,737,126 |
| − rỗng / không chữ-số sau html-unescape + NFC + gộp khoảng trắng | −169,186 / −54,334 |
| − trùng exact (key casefold) | −6,588,591 |
| − decontam vs train/dev/test 5 task, **text-only** (exact key: 24,687; 8-word shingle: 1,992) | −26,679 |
| còn lại | **8,898,336** |

Decontamination chỉ đọc cột text (`sents.txt`, `comment`, `Review`), không đọc nhãn/không chạy model trên test; ghi số và sha256
tập key. Split theo document trước packing (shuffle seed 42): holdout 54,322 docs → 2,000 seq (1.024M token), train 8,844,014
docs → **325,067 seq × 512 = 166.4M token positions** (sàn 100M đạt). 300M target = **1.80 pass** (≤ 3). Không dùng FineWeb-2.

**DAPT** (`artifacts/cdbk_dapt_pilot/dapt/{summary,resolved_config}.json`, `log.jsonl`): adapter `scripts/cdbk_dapt_pilot/dapt_train.py`
tái dùng `scripts/pilot_b/cpt_train.py` (mask_batch 15 % 80/10/10 không special/pad, SparseMLM tied-embedding, ShardSampler,
param_groups, lr_lambda, evaluate). Init `uitnlp/CafeBERT@af76fcf2` `XLMRobertaForMaskedLM`: **missing/unexpected/mismatched keys
= 0, MLM head pretrained, decoder tied với embedding** (kiểm trước train, fail-closed). Tokenizer XLM-R-large@c23d21b0 (spm trùng CafeBERT).
seq 512, global batch 256 (2×L40S DDP, micro 16, accum 8), bf16, lr 5e-5, AdamW(.9,.98) eps 1e-6 wd .01, warmup 137/2289 step (6 %)
+ linear decay, clip 1, seed 42(+rank). **300,023,808 token** (2289 batch; +0.18 batch so với 300M), 42.46M vị trí mask (14.15 %),
1.80 pass, 1.53 h, 54.5k tok/s, peak 24.8 GB/rank. Checkpoint = **cuối budget**, không chọn theo downstream; export
`dapt/hf/model.safetensors` sha256 `0134a914…`, reload kiểm tra bằng và decoder vẫn tied.

Holdout MLM loss (masking cố định seed 1234, 2,000 seq): **3.665 → 2.670 (500) → 2.540 (1000) → 2.470 (1500) → 2.435 (2000) → 2.426 (2289)**.
Train loss 3.55 (50 step đầu) → 2.56 (50 step cuối). Loss đầu 3.67 so với ~1.1 của CafeBERT trên FineWeb (Pilot B) cho thấy
corpus social này rất khác phân phối pretrain; DAPT giảm 1.24 nats nhưng vẫn cách xa mức web.

## Bước B — TD, Bước C — cache + KD_D

- TD: encoder từ export DAPT (chỉ bỏ `lm_head.*`, kiểm architecture 24L/1024), head mới `goibert_e14.ClassificationHead` H=1024, cùng
  thứ tự khởi tạo/RNG như T (smoke epoch-1 CE 1.4510 vs T 1.4501). Recipe T giữ nguyên (CE, lr 2e-5, 30 ep, eff batch 32/micro 32,
  maxlen 256, bf16/SDPA, AdamW(.9,.999) eps 1e-8 wd .01, warmup 10 %, clip 1). Best 77.58@23; reload/predictions khớp.
- Cache TD (`teacher_cache/absa_restaurant_train_logits_TD_seed42.pt`, schema E2, fp32 `.eval()`, chỉ train): 8/8 kiểm pass —
  IDs = train IDs, text hash khớp, shape [7028,12,4] hữu hạn, lookup qua loader shuffle của student tokenizer = index trực tiếp
  và labels khớp theo ID, teacher forward lại lệch ≤ 6.2e-6, không chứa labels/validation (`checks/cache_checks_TD.json`).
- KD_D: student khởi tạo lại từ `models/bamibert-base-original` qua đúng code path E14 (không từ B/K), recipe K nguyên vẹn
  (lr 5e-5, 30 ep, eff 32, `0.5 CE + 0.5 T²KL(TD‖S)` T=2, E2 helper nhân T² một lần, FP32). Smoke qua adapter cho số y hệt K smoke M1
  (loss 2.9869 / ce 1.4069 / kd 4.5668) → cùng head-init/data order. Best 73.34@22; reload/predictions khớp.

## Đường cong (validation pair F1)

| Epoch | 5 | 10 | 15 | 20 | 22 | 23 | 24 | 29 | 30 | TB 10–30 | TB 5 cuối |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T | 72.83 | 76.39 | 75.25 | 76.94 | 76.24 | 76.15 | 76.60 | 76.06 | 76.10 | 76.14 | 75.96 |
| TD | 73.63 | 74.85 | 76.60 | 76.57 | 76.62 | **77.58** | 76.91 | 77.00 | 76.86 | 76.52 | 77.02 |
| B | 63.52 | 71.00 | 71.44 | 74.10 | 74.58 | 73.98 | 73.56 | **74.74** | 74.74 | 73.30 | 74.37 |
| K | 56.73 | 68.88 | 71.51 | 72.84 | 73.11 | 72.64 | **73.94** | 72.67 | 73.16 | 71.95 | 73.00 |
| KD_D | 52.58 | 68.74 | 71.60 | 72.03 | **73.34** | 73.06 | 71.70 | 72.11 | 71.98 | 71.64 | 72.02 |

KD_D ≥ B ở **2/30** epoch; KD_D ≥ K ở 13/30 (hai đường KD gần như trùng nhau, cùng chậm ở epoch đầu: −8.9 và −13.8 pp so với B ở
epoch 2–3). TD cao hơn T ở giai đoạn muộn (TB 5 cuối +1.06) nhưng best epoch ngang nhau; đây là quan sát một seed.

## Kiểm correctness (tất cả pass)

| Kiểm | Kết quả | File |
|---|---|---|
| Nguồn DAPT | CafeBERT MLM load sạch (0 missing/unexpected/mismatched), head MLM pretrained, tied; export reload bằng & tied | `dapt/summary.json → source_checks, export` |
| Loss DAPT | hữu hạn suốt 2289 step (fail-closed nếu non-finite), clip 1, holdout giảm đơn điệu | `dapt/log.jsonl`, `summary.json` |
| TD encoder | load từ export, sha256 khớp summary, chỉ bỏ `lm_head.*`, architecture khớp | `TD/metadata.json → model` |
| Cache TD | 8/8 flags; teacher train pair-F1 99.99 % | `checks/cache_checks_TD.json` |
| Loss KD | contract M1 (`artifacts/cdbk_pilot/checks/loss_checks.json`, 12/12) không đổi; smoke KD_D = smoke K | log smoke |
| Checkpoint | TD, KD_D: reload best → F1 bằng best; F1 tính lại từ `validation_predictions.json` bằng best; predictions giống hệt | `*/metadata.json → postrun_checks` |

Checkpoint fine-tune là model-only (không resume); DAPT chỉ có export cuối. Source M1 (`scripts/cdbk_pilot/`, `artifacts/cdbk_pilot/`)
không sửa; adapter D1 patch in-process (`common_d1.install()`), `source_hash` D1 = hash M1 + adapter files.

## Quan sát mức prediction (`checks/validation_prediction_analysis_D1.json`)

- **Teacher train confidence T=2** (mô tả, không chứng minh nguyên nhân): TD mean max-prob 0.953 (absent 0.959 / present 0.907),
  T 0.947 (0.953 / 0.902); ở T=1 cả hai ≈ 0.998; argmax train TD ≡ T **99.993 %**, ≡ gold 99.998 %. DAPT không đổi "độ mềm" của
  soft target trên train; tín hiệu KD mà student nhận từ TD gần như giống từ T.
- Trên validation, **KD_D không gần TD hơn B gần TD**: slot agreement trên 1,279 slot active — B–TD 77.5 %, KD_D–TD 76.0 %;
  KD_D–K 81.7 %, KD_D–B 80.2 % (hai student KD giống nhau hơn giống teacher). TD–T 85.3 %.
- Slot TD đúng / B sai: 151, KD_D sửa 64; slot B và TD cùng đúng nhưng KD_D sai: 88 → net âm, cùng dạng với K ở M1.
- KD_D: precision −1.0, recall −1.8 pp so với B; detection F1 85.61 (B 86.48). Per-aspect KD_D − B: +3.63 SERVICE#GENERAL,
  −12.78 DRINKS#PRICES (19 gold), −6.88 RESTAURANT#MISC (58), −3.36 FOOD#QUALITY (222); lỗ ở aspect nhỏ cỡ nhiễu seed.
- TD vs T per-aspect: +11.3 DRINKS#PRICES, +4.3 DRINKS#QUALITY, −3.3 RESTAURANT#GENERAL, −2.8 FOOD#STYLE — dao động trong nhiễu.

## Cách đọc và điểm dừng (§7 plan)

KD_D − B = −1.41 pp → **chưa có tín hiệu đạt mục tiêu**; KD_D < K nên cũng không "bớt hại". TD − T = −0.14 → teacher không tăng,
student không tăng. Heuristic một seed, không significance; không có web-CPT control nên không quy nguyên nhân cho domain; không suy
mọi general distillation thất bại. Điểm cần ghi cho lượt sau (không thực thi): (1) corpus là comment social, không phải review — mismatch
lớn với Restaurant (holdout loss đầu 3.67); (2) logits teacher fine-tune trên chính train gần one-hot ở cả T và TD, nên "teacher tốt
hơn" hầu như không đổi target KD; (3) cả hai student KD chậm học rõ ở epoch đầu với alpha 0.5.

**Kết thúc D1.** Không tự chạy seed 43, E8, composite, general distillation, test hay tăng token.

## Artifacts và code

- Corpus: `data/cdbk_dapt_pilot/{manifest.json,prepare_corpus.log,visobert_xlmr_512_{train,holdout}.npy}`, raw parquet `data/cdbk_dapt_pilot/raw/`.
- DAPT: `artifacts/cdbk_dapt_pilot/dapt/{hf/,summary.json,resolved_config.json,log.jsonl,train.log,status.json}` (2×L40S cuda:2,3).
- TD: `artifacts/cdbk_dapt_pilot/TD/` (checkpoint sha256 `8002fa8e…`, 2.2 GB, peak 11.6 GB, cuda:2); KD_D: `artifacts/cdbk_dapt_pilot/KD_D/`
  (checkpoint `3e58fd6a…`, 0.4 GB, peak 3.1 GB, cuda:2); mỗi run có resolved_config/epoch_curves/best_checkpoint/best_validation_metrics/
  validation_predictions/metadata/status/train.log.
- Cache: `artifacts/cdbk_dapt_pilot/teacher_cache/absa_restaurant_train_logits_TD_seed42.pt` (sha256 `f437a057…`).
- Checks/tổng hợp: `artifacts/cdbk_dapt_pilot/checks/{cache_checks_TD,validation_prediction_analysis_D1}.json`, `artifacts/cdbk_dapt_pilot/summary.json`
  (`python scripts/cdbk_dapt_pilot/summarize_d1.py`).
- Code: `scripts/cdbk_dapt_pilot/{prepare_corpus,dapt_train,common_d1,finetune_d1,cache_td_logits,summarize_d1}.py`; reuse
  `scripts/cdbk_pilot/*` (runner/loss/cache/checks M1), `scripts/pilot_b/cpt_train.py` (MLM helpers), E1/E2/E14 modules. Không sửa gold/split.
- Chi phí: DAPT 1.53 h × 2 GPU, TD 19 min, cache 1 min, KD_D 8 min (GPU dùng chung ở phần đầu, trống ở phần sau).
