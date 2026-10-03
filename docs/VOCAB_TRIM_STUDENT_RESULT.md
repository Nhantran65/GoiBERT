# Student nhỏ từ XLM-R-large — kết quả bước 1 (cắt vocab) + bước 2 (đo sàn)

Plan `docs/VOCAB_TRIM_STUDENT_PLAN.md`; chạy 2026-09-21T09:39:43Z; recipe E14 như `scripts/base_encoder_baseline` (CLS head E14, CE, 30 epoch, eff. batch 32, maxlen 256, bf16/SDPA, AdamW wd 0,01, warmup 10 %, clip 1, best epoch theo validation, reload/prediction/ID checks). Validation-only (test chỉ dùng **text** để đo `<unk>`). Artifacts `artifacts/vocab_trim_student/`; model cắt ở `models/trimmed/`.

**Kết luận ngắn.** Bước 1 thành công: |K| = **45000** token, phủ **99.9978 %** lượt token của pool và **100 %** train+dev benchmark, `<unk>` chỉ **0.00449 %** token trên validation, khớp tokenization (B)↔(A) **100.00 %**, và trên checkpoint đã fine-tune thì F1 **không đổi** (77.8147 → 77.8147, max|Δ logits| = 0.0e+00) → phát hành được 4 model cắt gọn (XLM-R-large 558,8M → 348,9M; XLM-R-base 277,5M → 120,0M; mMiniLMv2-L12 117,5M → 38,8M; L6 106,8M → 28,1M). Bản 12 lớp trên XLM-R-base cắt vocab đạt **73.54** Restaurant, so với **73.61** của XLM-R-base **chưa cắt** ở round 15 → Δ **-0.06**, xác nhận lần nữa là cắt vocab không mất điểm. Bước 2 (đo sàn) **không** đạt mốc nào của plan §2.3 cho hướng student ≤100M: student công khai mMiniLMv2 chỉ **61.96** (M12) / **62.93** (M6) Restaurant — quá xa BamiBERT 74,74; nền 8 lớp B8 = **71.09** (< 72, dưới mốc 74,0 cho bước 3); không ứng viên nào ≥ BamiBERT ở ≥ 2/3 task. Kết luận: muốn có student nhỏ cạnh tranh thì phải **tự distill (bước 3)** trên XLM-R-large với dữ liệu của ta — các student công khai và việc bỏ lớp đơn thuần là chưa đủ.

## 1. Bước 1 — cắt vocab

- |K| = **45000** (cap 45000, mandatory 15642, single-char 2272, frequent 50556, dropped-for-cap 7232)
- Token đếm: pool 300.2M (target 300M), wiki 64.7M, benchmark train+dev 1.17M, social train+dev 1.54M
- Độ phủ của K: pool **99.9978 %**, wiki 99.9311 %, benchmark train+dev **100.0000 %**, social 100.0000 %

### 1.2 `<unk>` và độ khớp tokenization

| Đo | (A) token/unk | (A) câu bị ảnh hưởng | (B) token/unk | (B) câu bị ảnh hưởng |
|---|---:|---:|---:|---:|
| validation 5 task | 0.00449 % | 0.103 % | 0.00449 % | 0.103 % |
| text test 5 task | 0.01132 % | 0.256 % | 0.00652 % | 0.144 % |
| text test social | 0.10976 % | 1.199 % | 0.06699 % | 0.734 % |

Khớp tokenization (B) vs (A) trên validation: **100.00 %** chuỗi trùng id (5844/5844); 0 ví dụ khác nhau được lưu trong `trim_report.json`.

### 1.3 Bằng chứng cắt vocab không mất điểm (không train)

Checkpoint XLM-R-large Restaurant 2e-5 seed 42 (val 77.81): không cắt **77.8147** → (A) id-map + embedding cắt **77.8147** (Δ +0.0000; max|Δ logits| **0.00e+00**) → (B) re-tokenize bằng tokenizer cắt **77.8147** (Δ +0.0000) — ĐẠT.

### Tham số trước/sau khi cắt (4 model, `models/trimmed/`)

| Model | Kiến trúc | Tham số trước | Tham số sau | Embedding trước → sau |
|---|---|---:|---:|---:|
| `xlmr_large_vi` | XLMRobertaModel, 24L/1024 | 558.8M | **348.9M** | 256.0M → 46.1M |
| `xlmr_base_vi` | XLMRobertaModel, 12L/768 | 277.5M | **120.0M** | 192.0M → 34.6M |
| `mminilmv2_l12_vi` | XLMRobertaModel, 12L/384 | 117.5M | **38.8M** | 96.0M → 17.3M |
| `mminilmv2_l6_vi` | XLMRobertaModel, 6L/384 | 106.8M | **28.1M** | 96.0M → 17.3M |

## 2. Bước 2 — đo sàn (validation, seed 42)

Ô = **val F1 / best epoch / phút-run**; mỗi model dùng LR grid riêng (M12/M6: 2e-5, 5e-5, 1e-4; B8/B12: 1e-5, 2e-5, 5e-5).

### m12 — mMiniLMv2-L12-H384 (student của XLM-R-large)

| Task | LR 2e-05 | LR 5e-05 | LR 1e-04 | chọn |
|---|---|---|---|---|
| Restaurant | 46.12 / 30 / 6.3 | 61.85 / 30 / 6.4 | 61.96 / 28 / 6.3 | **61.96** @1e-04 · seed43 65.22 |
| Topic | — | — | 79.68 / 19 / 10.4 | **79.68** @1e-04 |
| ViSFD | — | — | 80.04 / 29 / 7.6 | **80.04** @1e-04 |

### m6 — mMiniLMv2-L6-H384

| Task | LR 2e-05 | LR 5e-05 | LR 1e-04 | chọn |
|---|---|---|---|---|
| Restaurant | 44.43 / 25 / 3.8 | 57.58 / 29 / 3.8 | 62.93 / 27 / 3.8 | **62.93** @1e-04 |
| Topic | — | — | 78.78 / 8 / 6.5 | **78.78** @1e-04 |
| ViSFD | — | — | 78.57 / 28 / 4.7 | **78.57** @1e-04 |

### b8 — XLM-R-base, giữ 8 lớp dưới

| Task | LR 1e-05 | LR 2e-05 | LR 5e-05 | chọn |
|---|---|---|---|---|
| Restaurant | 65.31 / 30 / 6.3 | 71.09 / 27 / 6.1 | 71.08 / 30 / 6.2 | **71.09** @2e-05 · seed43 69.75 |
| Topic | — | 79.65 / 26 / 10.0 | — | **79.65** @2e-05 |
| ViSFD | — | 79.72 / 25 / 7.3 | — | **79.72** @2e-05 |

### b12 — XLM-R-base, đủ 12 lớp

| Task | LR 1e-05 | LR 2e-05 | LR 5e-05 | chọn |
|---|---|---|---|---|
| Restaurant | 66.57 / 29 / 8.3 | 71.93 / 23 / 8.2 | 73.54 / 26 / 8.3 | **73.54** @5e-05 |
| Topic | — | — | 80.24 / 10 / 13.2 | **80.24** @5e-05 |
| ViSFD | — | — | 83.68 / 24 / 9.8 | **83.68** @5e-05 |

### Tham chiếu (validation, seed 42)

| Task | BamiBERT | mmBERT-base | XLM-R-large gốc | B2 | P_vi |
|---|---:|---:|---:|---:|---:|
| Restaurant | 74.74 | 75.63 | 77.81 | 75.72 | 76.15 |
| Topic | 79.53 | 81.18 | 80.61 | 80.74 | 80.69 |
| ViSFD | 84.38 | 83.64 | 85.48 | 84.85 | 85.44 |

### Đọc kết quả theo plan §2.3

- B12 (XLM-R-base cắt vocab, 12 lớp) Restaurant = **73.54** (kỳ vọng ~73.6 ± 0.5) → khớp: cắt vocab trước fine-tune không mất điểm.
- B8 (XLM-R-base 8 lớp dưới) Restaurant = **71.09** → < 72 → cân nhắc giữ 10 lớp/khởi tạo khác.
- M12 (mMiniLMv2-L12, ~39M sau cắt) Restaurant = **61.96** → < 73,0: mục tiêu student ~30M khó.
- m12: ≥ BamiBERT ở 1/3 task (Topic)
- m6: ≥ BamiBERT ở 0/3 task (không)
- b8: ≥ BamiBERT ở 1/3 task (Topic)
- b12: ≥ BamiBERT ở 1/3 task (Topic)
- B12 (cắt vocab) 73.54 vs XLM-R-base **chưa cắt** round 15 73.61 → Δ -0.06: cắt vocab pre-FT không mất điểm (khớp §1.3).
- Sàn nhiễu seed ở cỡ nhỏ (Restaurant, cùng LR): m12 seed42 61.96 vs seed43 65.22, b8 seed42 71.09 vs seed43 69.75 → chênh ~2–3 điểm, nên ở cỡ này |Δ| < 1 **không** có nghĩa chưa phân biệt; mọi so sánh với BamiBERT phải coi là thô.

## 3. Checks, license, giới hạn

Mọi ô có `postrun_checks` (reload trùng best F1, file prediction cho lại đúng F1, dự đoán sau reload giống hệt, ID ổn định và đủ). Không chạy test (ngoài đo `<unk>` trên text), không Hotel/Sentiment, không distill, không sửa artifacts cũ.

- License: XLM-R (base/large) MIT; mMiniLMv2 phát hành theo UniLM/MIT (card `nreimers/mMiniLMv2-*`); dữ liệu đếm vocab: pool v3 CC0 (không NC), Wikipedia CC BY-SA 4.0 (chỉ đếm tần suất, không phân phối lại text).
- Giới hạn: 1 seed (42) + seed 43 cho M12/B8; validation-only; các student **chưa distill** trên dữ liệu của ta — đây là sàn, không phải kết quả cuối; ở cỡ model này sàn nhiễu seed đo được ~2–3 điểm nên **không** dùng ngưỡng |Δ| < 1,0 của các model lớn, và không claim significance.
- Chi phí: 2.60 GPU-giờ cho 22 ô fine-tune.

