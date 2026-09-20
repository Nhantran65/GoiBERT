# Audit lỗi ghép cặp CafeBERT–BamiBERT trên validation — packet bàn giao (cơ học)

Plan `docs/CAFEBERT_BAMIBERT_PAIRED_ERROR_AUDIT_PLAN.md`; tạo 2026-09-18T17:27:15Z; seed 42; **read-only** (không model/GPU/test, không sửa nhãn). Source `scripts/cafe_bami_paired_error_audit/run_audit.py` (sha256 `f260a5c15aa7fb07…`); artifacts `artifacts/cafe_bami_paired_error_audit/{manifest.json, aggregate.json, events.jsonl, packet.jsonl, packet.md}`. Báo cáo này chỉ trình bày số liệu và phép kiểm; **diễn giải nguyên nhân và chọn giả thuyết là việc của root**.

Mã nhãn: Topic `0 lecturer / 1 training_program / 2 facility / 3 others`; ABSA `0 absent / 1 positive / 2 negative / 3 neutral` (theo `docs/E12_LABEL_MAPPING_AUDIT.md`, không dùng tên trong `data.py`); ViSFD `OTHERS` slot 5 detection-only (gold raw 4 ↔ model 1), loại khỏi pair F1 và packet polarity.

## 1. Nguồn và gate correctness

| Task | Model | Prediction file | sha256 | best epoch | F1 evaluator | F1 đếm độc lập | Khớp artifact ≤1e-12 |
|---|---|---|---|---:|---:|---:|:--:|
| absa_restaurant | BamiBERT E14 | `artifacts/e14/trials/absa_restaurant/lr_5e-05/seed_42/validation_predictions.json` | `d34adfba8f92…` | 29 | 0.7474178404 | 0.7474178404 | ✓ |
| absa_restaurant | CafeBERT | `artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42/validation_predictions.json` | `349e4de9bad0…` | 21 | 0.7771971496 | 0.7771971496 | ✓ |
| vsfc_topic | BamiBERT E14 | `artifacts/e14/trials/vsfc_topic/lr_1e-05/seed_42/validation_predictions.json` | `32fdf14f4b1a…` | 13 | 0.7953280822 | 0.7953280822 | ✓ |
| vsfc_topic | CafeBERT | `artifacts/sealion_benchmark/cafe_runs/vsfc_topic/lr_2e-05/seed_42/validation_predictions.json` | `d35136b0282a…` | 23 | 0.8174458365 | 0.8174458365 | ✓ |
| visfd | BamiBERT E14 | `artifacts/e14/trials/visfd/lr_5e-05/seed_42/validation_predictions.json` | `32388579e4c0…` | 23 | 0.8438339803 | 0.8438339803 | ✓ |
| visfd | CafeBERT | `artifacts/sealion_benchmark/cafe_runs/visfd/lr_2e-05/seed_42/validation_predictions.json` | `b42661951279…` | 19 | 0.8546650718 | 0.8546650718 | ✓ |

Gate: mỗi task 771/1.583/1.112 hàng ở loader và cả hai file; ID duy nhất, cùng tập và **cùng thứ tự** với loader; gold giống hệt giữa loader và hai file; shape 12/11 slot, mã trong tập cho phép; ViSFD OTHERS gold ∈ {0,4}, prediction ∈ {0,1}. Sáu F1 tính lại bằng `goibert.metrics.evaluate_task` và bằng phép đếm TP/FP/FN độc lập trong script: khớp nhau và khớp `metadata.json`/`best_validation_metrics.json` của từng run ≤1e-12; khớp mốc 4 chữ số trong plan ≤1e-6. `gate_all_ok = True`. Evaluator sha256 `6248547c17f3…`, loader `aeff87536a81…`. Không đọc `test_predictions.json`.

## 2. VSFC Topic — toàn validation (1.583 câu)

| Lớp | Support | Bami P / R / F1 | Cafe P / R / F1 | Bami TP/FP/FN | Cafe TP/FP/FN |
|---|---:|---|---|---|---|
| lecturer | 1151 | 92.12 / 95.48 / 93.77 | 94.69 / 94.53 / 94.61 | 1099/94/52 | 1088/61/63 |
| training_program | 267 | 78.35 / 74.53 / 76.39 | 77.35 / 83.15 / 80.14 | 199/55/68 | 222/65/45 |
| facility | 70 | 94.12 / 91.43 / 92.75 | 92.54 / 88.57 / 90.51 | 64/4/6 | 62/5/8 |
| others | 95 | 66.18 / 47.37 / 55.21 | 67.50 / 56.84 / 61.71 | 45/23/50 | 54/26/41 |
| **macro F1 / accuracy** | | **79.53 / 88.88** | **81.74 / 90.08** | | |

Confusion bami (hàng = gold, cột = dự đoán, thứ tự lecturer, training_program, facility, others):

| gold \ pred | lecturer | training_program | facility | others |
|---|---:|---:|---:|---:|
| lecturer | 1099 | 40 | 1 | 11 |
| training_program | 56 | 199 | 1 | 11 |
| facility | 2 | 3 | 64 | 1 |
| others | 36 | 12 | 2 | 45 |

Confusion cafe (hàng = gold, cột = dự đoán, thứ tự lecturer, training_program, facility, others):

| gold \ pred | lecturer | training_program | facility | others |
|---|---:|---:|---:|---:|
| lecturer | 1088 | 46 | 3 | 14 |
| training_program | 32 | 222 | 1 | 12 |
| facility | 2 | 6 | 62 | 0 |
| others | 27 | 13 | 1 | 54 |

Paired outcomes theo gold class (both_correct / cafe_only / bami_only / both_wrong_same / both_wrong_different):

| Gold class | both_correct | cafe_only | bami_only | both_wrong_same | both_wrong_different |
|---|---:|---:|---:|---:|---:|
| lecturer | 1072 | 16 | 27 | 33 | 3 |
| training_program | 193 | 29 | 6 | 34 | 5 |
| facility | 62 | 0 | 2 | 4 | 2 |
| others | 42 | 12 | 3 | 35 | 3 |
| **tổng** | 1369 | 57 | 38 | 106 | 13 |

Cross-tab (bami_pred | cafe_pred) theo gold class, các ô ≥5:

- **lecturer** (n=1151): `bami=lecturer|cafe=lecturer` 1072; `bami=training_program|cafe=training_program` 26; `bami=lecturer|cafe=training_program` 19; `bami=training_program|cafe=lecturer` 12; `bami=lecturer|cafe=others` 7; `bami=others|cafe=others` 6
- **training_program** (n=267): `bami=training_program|cafe=training_program` 193; `bami=lecturer|cafe=lecturer` 27; `bami=lecturer|cafe=training_program` 24; `bami=others|cafe=others` 6; `bami=lecturer|cafe=others` 5; `bami=training_program|cafe=lecturer` 5; `bami=others|cafe=training_program` 5
- **facility** (n=70): `bami=facility|cafe=facility` 62
- **others** (n=95): `bami=others|cafe=others` 42; `bami=lecturer|cafe=lecturer` 24; `bami=lecturer|cafe=others` 11; `bami=training_program|cafe=training_program` 10

## 3. UIT-ABSA Restaurant (771 review, 12 aspect) — toàn validation

Gold-present pair-scored slots: **1053**; phân bố số aspect gold-present/review: {"1": 521, "2": 219, "3": 30, "4": 1}.

| Model | TP | FP | FN | Pair micro F1 | correct | miss | wrong_polarity | spurious |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bami | 796 | 281 | 257 | 74.74 | 796 | 132 | 125 | 156 |
| Cafe | 818 | 234 | 235 | 77.72 | 818 | 128 | 107 | 127 |

Theo gold polarity (gold-present slots): support · correct / miss / wrong_polarity — Bami vs Cafe; spurious theo polarity dự đoán:

| Gold polarity | Support | Bami correct/miss/wrong | Cafe correct/miss/wrong | Bami spurious→pol | Cafe spurious→pol |
|---|---:|---|---|---:|---:|
| positive | 579 | 466/66/47 | 474/70/35 | 66 | 63 |
| negative | 196 | 133/32/31 | 150/28/18 | 30 | 26 |
| neutral | 278 | 197/34/47 | 194/30/54 | 60 | 38 |

Per-aspect (support = gold-present; F1 = pair F1 của aspect; events của từng model). Support thấp (<30) chỉ để đếm, không xếp hạng:

| Aspect | Support | Bami F1 | Bami miss/wrong/spur | Cafe F1 | Cafe miss/wrong/spur | Δ F1 (Cafe−Bami, pp) |
|---|---:|---:|---|---:|---|---:|
| AMBIENCE#GENERAL | 91 | 81.08 | 6/10/9 | 78.92 | 9/9/12 | -2.16 |
| DRINKS#PRICES | 19 | 76.19 | 2/1/6 | 61.90 | 2/4/6 | -14.29 |
| DRINKS#QUALITY | 81 | 69.82 | 7/15/14 | 77.11 | 6/11/10 | +7.29 |
| DRINKS#STYLE&OPTIONS | 52 | 64.58 | 15/6/7 | 73.08 | 10/4/10 | +8.49 |
| FOOD#PRICES | 45 | 79.57 | 5/3/8 | 82.98 | 5/1/9 | +3.41 |
| FOOD#QUALITY | 222 | 80.53 | 14/24/27 | 83.41 | 17/19/19 | +2.88 |
| FOOD#STYLE&OPTIONS | 175 | 70.66 | 32/19/33 | 75.15 | 32/16/20 | +4.49 |
| LOCATION#GENERAL | 42 | 79.07 | 4/4/6 | 77.65 | 2/7/3 | -1.42 |
| RESTAURANT#GENERAL | 100 | 61.78 | 26/15/17 | 63.32 | 20/17/19 | +1.54 |
| RESTAURANT#MISCELLANEOUS | 58 | 75.21 | 8/6/9 | 77.97 | 8/4/10 | +2.75 |
| RESTAURANT#PRICES | 47 | 65.31 | 6/9/10 | 65.12 | 12/7/4 | -0.19 |
| SERVICE#GENERAL | 121 | 82.45 | 7/13/10 | 89.26 | 5/8/5 | +6.81 |

Paired outcomes trên gold-present slots: both_correct 734, cafe_only 84, bami_only 62, both_wrong_same 148, both_wrong_different 25. Trên gold-absent slots: both_correct_absent 8003 (không dùng để nói model "đúng hầu hết"), cafe_only (Bami spurious) 69, bami_only (Cafe spurious) 40, cả hai spurious cùng polarity 77, khác polarity 10.

Paired outcomes theo aspect × gold polarity (cafe_only / bami_only / both_correct / both_wrong_same / both_wrong_different):

| Aspect | polarity | cafe_only | bami_only | both_correct | bw_same | bw_diff |
|---|---|---:|---:|---:|---:|---:|
| AMBIENCE#GENERAL | negative | 2 | 2 | 8 | 1 | 0 |
| AMBIENCE#GENERAL | neutral | 2 | 3 | 8 | 2 | 0 |
| AMBIENCE#GENERAL | positive | 1 | 2 | 52 | 7 | 1 |
| DRINKS#PRICES | negative | 0 | 1 | 0 | 1 | 0 |
| DRINKS#PRICES | neutral | 0 | 1 | 12 | 1 | 0 |
| DRINKS#PRICES | positive | 0 | 1 | 1 | 1 | 0 |
| DRINKS#QUALITY | negative | 3 | 0 | 12 | 6 | 2 |
| DRINKS#QUALITY | neutral | 0 | 3 | 8 | 4 | 0 |
| DRINKS#QUALITY | positive | 6 | 1 | 35 | 1 | 0 |
| DRINKS#STYLE&OPTIONS | negative | 0 | 0 | 2 | 3 | 0 |
| DRINKS#STYLE&OPTIONS | neutral | 3 | 2 | 17 | 3 | 1 |
| DRINKS#STYLE&OPTIONS | positive | 6 | 0 | 10 | 5 | 0 |
| FOOD#PRICES | negative | 2 | 0 | 5 | 2 | 0 |
| FOOD#PRICES | neutral | 0 | 0 | 28 | 1 | 0 |
| FOOD#PRICES | positive | 0 | 0 | 4 | 2 | 1 |
| FOOD#QUALITY | negative | 3 | 1 | 25 | 2 | 2 |
| FOOD#QUALITY | neutral | 4 | 5 | 18 | 9 | 1 |
| FOOD#QUALITY | positive | 4 | 3 | 132 | 11 | 2 |
| FOOD#STYLE&OPTIONS | negative | 3 | 0 | 10 | 5 | 0 |
| FOOD#STYLE&OPTIONS | neutral | 11 | 6 | 31 | 4 | 3 |
| FOOD#STYLE&OPTIONS | positive | 3 | 8 | 69 | 20 | 2 |
| LOCATION#GENERAL | negative | 1 | 0 | 3 | 0 | 0 |
| LOCATION#GENERAL | neutral | 2 | 4 | 12 | 3 | 0 |
| LOCATION#GENERAL | positive | 1 | 1 | 14 | 0 | 1 |
| RESTAURANT#GENERAL | negative | 2 | 1 | 13 | 6 | 0 |
| RESTAURANT#GENERAL | neutral | 3 | 1 | 10 | 5 | 4 |
| RESTAURANT#GENERAL | positive | 4 | 3 | 31 | 17 | 0 |
| RESTAURANT#MISCELLANEOUS | negative | 0 | 0 | 9 | 3 | 1 |
| RESTAURANT#MISCELLANEOUS | neutral | 2 | 1 | 3 | 3 | 0 |
| RESTAURANT#MISCELLANEOUS | positive | 4 | 3 | 28 | 1 | 0 |
| RESTAURANT#PRICES | negative | 0 | 1 | 6 | 1 | 0 |
| RESTAURANT#PRICES | neutral | 0 | 3 | 14 | 5 | 2 |
| RESTAURANT#PRICES | positive | 2 | 2 | 6 | 4 | 1 |
| SERVICE#GENERAL | negative | 9 | 2 | 32 | 3 | 0 |
| SERVICE#GENERAL | neutral | 0 | 1 | 6 | 3 | 0 |
| SERVICE#GENERAL | positive | 1 | 0 | 60 | 3 | 1 |

## 4. ViSFD (1.112 comment, 10 slot polarity + OTHERS) — toàn validation

Gold-present pair-scored slots: **3316**; phân bố số aspect gold-present/review: {"2": 261, "5": 112, "4": 209, "1": 156, "3": 301, "0": 18, "6": 46, "7": 9}.

| Model | TP | FP | FN | Pair micro F1 | correct | miss | wrong_polarity | spurious |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bami | 2826 | 556 | 490 | 84.38 | 2826 | 237 | 253 | 303 |
| Cafe | 2858 | 514 | 458 | 85.47 | 2858 | 225 | 233 | 281 |

Theo gold polarity (gold-present slots): support · correct / miss / wrong_polarity — Bami vs Cafe; spurious theo polarity dự đoán:

| Gold polarity | Support | Bami correct/miss/wrong | Cafe correct/miss/wrong | Bami spurious→pol | Cafe spurious→pol |
|---|---:|---|---|---:|---:|
| positive | 1924 | 1750/86/88 | 1770/82/72 | 121 | 123 |
| negative | 1011 | 837/120/54 | 846/111/54 | 149 | 128 |
| neutral | 381 | 239/31/111 | 242/32/107 | 33 | 30 |

Per-aspect (support = gold-present; F1 = pair F1 của aspect; events của từng model). Support thấp (<30) chỉ để đếm, không xếp hạng:

| Aspect | Support | Bami F1 | Bami miss/wrong/spur | Cafe F1 | Cafe miss/wrong/spur | Δ F1 (Cafe−Bami, pp) |
|---|---:|---:|---|---:|---|---:|
| BATTERY | 504 | 87.91 | 3/50/21 | 90.82 | 2/37/18 | +2.91 |
| CAMERA | 296 | 84.59 | 6/32/24 | 84.87 | 5/33/21 | +0.28 |
| DESIGN | 187 | 83.99 | 16/11/23 | 84.10 | 21/10/18 | +0.11 |
| FEATURES | 381 | 82.77 | 36/28/40 | 85.23 | 27/25/37 | +2.47 |
| GENERAL | 689 | 88.46 | 56/20/64 | 88.68 | 52/22/61 | +0.22 |
| PERFORMANCE | 582 | 83.43 | 38/53/51 | 84.46 | 45/45/46 | +1.03 |
| PRICE | 252 | 80.84 | 9/32/27 | 79.54 | 11/35/25 | -1.31 |
| SCREEN | 121 | 77.73 | 16/9/21 | 78.43 | 10/11/23 | +0.70 |
| SER&ACC | 290 | 79.28 | 53/17/28 | 80.78 | 49/14/31 | +1.50 |
| STORAGE | 14 | 64.29 | 4/1/4 | 76.92 | 3/1/1 | +12.64 |

Paired outcomes trên gold-present slots: both_correct 2698, cafe_only 160, bami_only 128, both_wrong_same 300, both_wrong_different 30. Trên gold-absent slots: both_correct_absent 7421 (không dùng để nói model "đúng hầu hết"), cafe_only (Bami spurious) 102, bami_only (Cafe spurious) 80, cả hai spurious cùng polarity 185, khác polarity 16.

- OTHERS detection-only, bami: gold-present 251, detected 135, missed 116, spurious 111.
- OTHERS detection-only, cafe: gold-present 251, detected 139, missed 112, spurious 122.

Paired outcomes theo aspect × gold polarity (cafe_only / bami_only / both_correct / both_wrong_same / both_wrong_different):

| Aspect | polarity | cafe_only | bami_only | both_correct | bw_same | bw_diff |
|---|---|---:|---:|---:|---:|---:|
| BATTERY | negative | 4 | 4 | 136 | 5 | 1 |
| BATTERY | neutral | 10 | 7 | 20 | 12 | 2 |
| BATTERY | positive | 14 | 3 | 281 | 5 | 0 |
| CAMERA | negative | 3 | 4 | 75 | 5 | 1 |
| CAMERA | neutral | 3 | 0 | 19 | 14 | 0 |
| CAMERA | positive | 1 | 3 | 157 | 8 | 3 |
| DESIGN | negative | 3 | 3 | 26 | 6 | 2 |
| DESIGN | neutral | 0 | 1 | 1 | 9 | 1 |
| DESIGN | positive | 1 | 4 | 125 | 4 | 1 |
| FEATURES | negative | 11 | 4 | 205 | 11 | 2 |
| FEATURES | neutral | 2 | 3 | 13 | 10 | 5 |
| FEATURES | positive | 9 | 3 | 89 | 13 | 1 |
| GENERAL | negative | 8 | 4 | 100 | 15 | 0 |
| GENERAL | neutral | 3 | 1 | 19 | 11 | 0 |
| GENERAL | positive | 13 | 17 | 472 | 25 | 1 |
| PERFORMANCE | negative | 10 | 16 | 157 | 26 | 1 |
| PERFORMANCE | neutral | 7 | 8 | 12 | 15 | 3 |
| PERFORMANCE | positive | 17 | 9 | 289 | 12 | 0 |
| PRICE | negative | 2 | 5 | 21 | 8 | 0 |
| PRICE | neutral | 6 | 8 | 121 | 9 | 0 |
| PRICE | positive | 3 | 3 | 53 | 11 | 2 |
| SCREEN | negative | 4 | 1 | 35 | 7 | 0 |
| SCREEN | neutral | 0 | 1 | 4 | 6 | 1 |
| SCREEN | positive | 3 | 1 | 54 | 4 | 0 |
| SER&ACC | negative | 13 | 8 | 33 | 24 | 0 |
| SER&ACC | neutral | 2 | 1 | 0 | 8 | 2 |
| SER&ACC | positive | 7 | 6 | 172 | 13 | 1 |
| STORAGE | negative | 0 | 0 | 0 | 2 | 0 |
| STORAGE | neutral | 0 | 0 | 0 | 1 | 0 |
| STORAGE | positive | 1 | 0 | 9 | 1 | 0 |

## 5. Packet 72 trường hợp (`packet.jsonl`, `packet.md`)

Quy tắc chọn: sha256('42|task|bucket|error_type|stable_id|aspect_index') ascending; event quotas 3/3/2 miss/wrong_polarity/spurious with fallback miss->wrong_polarity->spurious; <=2 per aspect per exclusive column when enough candidates; unique review id per task; both_wrong prefers gold-present; both_correct requires a correct gold-present pair; Topic round-robin over gold classes.

| Task | cafe_only | bami_only | both_wrong | both_correct | ID duy nhất |
|---|---:|---:|---:|---:|---:|
| absa_restaurant | 8 | 8 | 4 | 4 | 24 |
| vsfc_topic | 8 | 8 | 4 | 4 | 24 |
| visfd | 8 | 8 | 4 | 4 | 24 |

Ứng viên và shortfall theo bucket:

- absa_restaurant/cafe_only: quota 8, lấy 8, ứng viên 153; event lấy {'miss': 3, 'wrong_polarity': 3, 'spurious': 2}; ứng viên theo event {'spurious': 69, 'miss': 36, 'wrong_polarity': 48}.
- absa_restaurant/bami_only: quota 8, lấy 8, ứng viên 102; event lấy {'miss': 3, 'wrong_polarity': 3, 'spurious': 2}; ứng viên theo event {'miss': 31, 'spurious': 40, 'wrong_polarity': 31}.
- absa_restaurant/both_wrong: quota 4, lấy 4; ứng viên gold-present 173, gold-absent 87.
- absa_restaurant/both_correct: quota 4, lấy 4; ứng viên gold-present 734.
- vsfc_topic/cafe_only: quota 8, lấy 8, ứng viên 57; gold classes phủ ['lecturer', 'training_program', 'others']; ứng viên theo class {'lecturer': 16, 'training_program': 29, 'facility': 0, 'others': 12}.
- vsfc_topic/bami_only: quota 8, lấy 8, ứng viên 38; gold classes phủ ['lecturer', 'training_program', 'facility', 'others']; ứng viên theo class {'lecturer': 27, 'training_program': 6, 'facility': 2, 'others': 3}.
- vsfc_topic/both_wrong: quota 4, lấy 4, ứng viên 119; gold classes phủ ['lecturer', 'training_program', 'facility', 'others']; ứng viên theo class {'lecturer': 36, 'training_program': 39, 'facility': 6, 'others': 38}.
- vsfc_topic/both_correct: quota 4, lấy 4, ứng viên 1369; gold classes phủ ['lecturer', 'training_program', 'facility', 'others']; ứng viên theo class {'lecturer': 1072, 'training_program': 193, 'facility': 62, 'others': 42}.
- visfd/cafe_only: quota 8, lấy 8, ứng viên 262; event lấy {'miss': 3, 'wrong_polarity': 3, 'spurious': 2}; ứng viên theo event {'wrong_polarity': 79, 'spurious': 102, 'miss': 81}.
- visfd/bami_only: quota 8, lấy 8, ứng viên 208; event lấy {'miss': 3, 'wrong_polarity': 3, 'spurious': 2}; ứng viên theo event {'spurious': 80, 'wrong_polarity': 59, 'miss': 69}.
- visfd/both_wrong: quota 4, lấy 4; ứng viên gold-present 330, gold-absent 201.
- visfd/both_correct: quota 4, lấy 4; ứng viên gold-present 2698.

Shortfall về phủ lớp: Topic `cafe_only` không có ứng viên gold `facility` (0 câu Cafe đúng–Bami sai ở lớp này), nên cột này chỉ phủ 3/4 lớp; các cột Topic khác phủ đủ 4 lớp. Mọi quota số lượng đều đủ. Mỗi hàng packet có text nguyên văn, gold/pred hai model (số + tên), event target, vector đầy đủ (ABSA), số aspect gold-present, path/sha256 nguồn, khóa chọn, và bốn cột trống `gold_supported / evidence_span / possible_failure / ambiguity_note` cho root.

## 6. Giới hạn diễn giải

- Packet chọn **có điều kiện** theo đúng/sai và theo quota event; không dùng để ước lượng tỷ lệ nguyên nhân lỗi trong dân số. Bảng toàn validation ở §2–§4 mới là số mô tả đầy đủ.
- Một seed, một checkpoint mỗi model (best validation epoch), validation đã dùng để chọn LR/epoch; disagreement cafe_only/bami_only gồm cả nhiễu run-to-run (~1 điểm F1 quan sát trong project). Không có test.
- Đơn vị ABSA là (review, slot); `both_correct` chỉ nói về slot target; các ô support thấp (DRINKS#PRICES 19, STORAGE 14, FOOD#PRICES 45…) chỉ để đếm.
- Macro F1 Topic và micro F1 ABSA không cộng trung bình. Tên polarity ABSA theo E12 (`1 positive / 2 negative / 3 neutral`); gold có thể sai — cột `gold_supported` để root đánh giá, subagent không relabel.
- Không đề xuất kiến trúc hay pilot từ packet này.
