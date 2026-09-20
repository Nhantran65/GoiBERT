# mmBERT + tiếng Việt văn nói (corpus ViSoBERT, research-only) — pilot v4: 3 arm × 250 M token

**Trạng thái:** pilot v4 **đã chạy xong toàn bộ** (gate dữ liệu PASS, 3 lượt CPT 250 M, 13 fine-tune). **Dừng ở đây**:
không bản đầy đủ (3 epoch / 1,13 B+ / 5 task × 2 seed), không thêm seed, không Hotel/Sentiment. Ngày chạy 2026-09-19/20.
Model gọi đúng: **mmBERT-base 307 M tổng (110 M không-embedding)**.

> **Bổ sung ngày 2026-09-20 — test mở theo yêu cầu trực tiếp của người dùng.** Plan v4 ghi "không test" và gate đã kết
> luận dừng; người dùng sau đó yêu cầu chạy test. Đã đánh giá **đúng checkpoint mà validation đã chọn** (best epoch on
> validation, không chọn lại theo test) cho cả 13 lượt v4 + O seed 42 (baseline) + O seed 43, và ghi riêng vào
> `<ft dir>/test_eval/`. Vì test đã được mở ngoài gate đã chốt, các số dưới đây là **mô tả một lần**, không phải gate
> xác nhận; test không còn là tập kín cho bất kỳ bản đầy đủ nào sau này. Xem §6b.

**Kết luận một câu:** trên validation, (Q1) học thêm 250 M token tiếng Việt **sạch** (P_web) **không** giúp — ngang hoặc
thấp hơn mmBERT gốc ở cả ba task (−0,46 … −0,97); (Q2) thêm **30 % bình luận mạng xã hội** (ViSoBERT, P_inf) **không** đạt
tiêu chí đã chốt: Restaurant −1,29 điểm so với P_web (ViSFD +1,16, Topic −0,11); (Q3) dồn văn nói vào chặng decay (C_inf)
**có** tín hiệu so với trộn đều (+1,05 Restaurant, +0,66 Topic; −0,92 ViSFD) và gần bằng P_web ở Restaurant (−0,25),
nhưng Q3 chỉ được diễn giải khi Q2 đạt. Theo gate v3 §6: **không gate nào đạt → dừng, không đầu tư thêm**.

## 0. License — đọc trước (ràng buộc NC)

Bucket văn nói là corpus **ViSoBERT** (`phucdev/ViSoBERT` @ `74dbc64e92c627c70670dd91344b776c8500a104`), điều khoản
**research purposes only** (card `uitnlp/visobert`) / **CC BY-NC 4.0** (SEACrowd). Hệ quả, ghi ở mọi manifest/báo cáo:

* **P_inf và C_inf là checkpoint NC research-only**: thư mục `artifacts/mmbert_vi_cpt_pilot/v4/cpt/P_inf_NC_research_only/`
  và `…/C_inf_NC_research_only/`; mọi `manifest.json`/`summary.json` có `research_only: true` + `license`. **Không phát
  hành, không dùng thương mại.**
* **P_web và O không chứa dữ liệu NC** (general v3 pool + Wikipedia v2) → dùng được cho model phát hành.
* Mục đích duy nhất của P_inf/C_inf: kiểm giả thuyết "văn nói người dùng có giúp mmBERT trên ABSA/phản hồi sinh viên".
  Nếu sau này cần bản phát hành, phải tìm nguồn social **có license** và không lấy checkpoint pilot làm nền.
* v3 review (`phamluan/vn_msa_sar_v3`, CC BY 4.0) chỉ dùng làm **holdout MLM**, không đưa vào train (§2).

## 1. Chuẩn bị ViSoBERT cho mmBERT (§2)

Đọc 3 parquet shard đã pin (sha256 khớp `data/cdbk_dapt_pilot/raw/download.json`), chuẩn hoá NFC/HTML-unescape/khoảng
trắng, exact dedup casefold, decontam text-only với train/dev/test của 5 benchmark (exact key + shingle 8 từ, đúng logic
D1 `scripts/cdbk_dapt_pilot/prepare_corpus.py`), tokenize bằng **tokenizer mmBERT (Gemma 256K)** — không dùng mảng XLM-R
cũ — tách trước 2 000 bình luận held-out, đóng gói 512 token tách bằng EOS, mỗi chuỗi thuần một nguồn.

| Bước | Số bình luận |
|---|---:|
| Đọc thô | 15 737 126 |
| Bỏ rỗng | 169 186 |
| Bỏ không có chữ/số | 54 334 |
| Bỏ exact duplicate (casefold) | 6 588 591 |
| Còn sau normalize/dedup | 8 925 015 |
| Bỏ decontam — exact key | 24 687 |
| Bỏ decontam — shingle 8 từ | 1 992 |
| **Còn dùng được** | **8 898 336** (train 8 896 336 + holdout 2 000) |

Decontam theo split: `vsfc/train` 11 425 · `vsfc/test` 3 166 · `vsfc/dev` 3 · `visfd/Train` 7 065 · `visfd/Dev` 1 007 ·
`visfd/Test` 2 013 · `absa_hotel/Train` 5 · `absa_hotel/Test` 2 · `absa_restaurant/Test` 1 · (shingle: 1 992, không gắn
split). **Không có văn bản benchmark trong bucket** (đã loại ở cả hai mức).

Token: **167 946 752 token mmBERT non-pad** trong **328 021 chuỗi 512** (train) và holdout 2 000 bình luận → 73 chuỗi
(37 376 token). Ngưỡng lượng ≥75 M: **đạt** (gấp 2,24 lần, không cần lặp).

## 2. Audit và gate dữ liệu (§2)

Đọc tay 100 bình luận (seed 42):

| Ô | Ngưỡng | Kết quả | Đạt |
|---|---|---|---|
| Tiếng Việt thật | ≥90 % | **99 %** (1 mẫu là tiếng Anh) | ✅ |
| **Văn nói người dùng** | **≥70 %** | **95 %** | ✅ |
| Spam/quảng cáo/người lớn | ≤10 % | **4 %** (2 mẫu quảng cáo, 1 mẫu rao bán BĐS, 1 mẫu chỉ tag tên) | ✅ |
| Lượng | ≥75 M token | **167,9 M token** | ✅ |

Verdict: `artifacts/mmbert_vi_cpt_pilot/v4/DATA_GATE_VERDICT_V4.json` = **PASS**.

**Độ phủ teencode** (chỉ mô tả, không phải điều kiện dừng): sau khi loại các mục mơ hồ (dạng ≤2 ký tự và danh sách đa
nghĩa đã ghi ở v3: `vs, a, s, e, t, d, i, ô, dừa, bia, seo, âu, mí, kon, …`), top-100 dạng teencode trong bucket **đều
đạt ≥100 lần** khi quy về 75 M token văn nói của P_inf (100/100) — khác hẳn v3, nơi chỉ 52/100 vì bảng bị chi phối bởi
các mục mơ hồ.

## 3. Ba arm (250 M non-pad token mỗi arm)

| Arm | general (v3 pool) | Wikipedia v2 | ViSoBERT (NC) | Tỷ lệ văn nói | tokens hash | order hash | multiset hash |
|---|---:|---:|---:|---:|---|---|---|
| **P_web** | 95 % | 5 % | 0 | 0,000 | `2766f8b9d20a…` | … | — |
| **P_inf** | 65 % | 5 % | **30 %** | **0,300** | `d97e99582c4e…` | … | `…` |
| **C_inf** | 65 % | 5 % | 30 % (cùng multiset) | **0,300** | `b395ae6ba04c…` | staged | bằng P_inf |

C_inf giữ **đúng multiset chuỗi đã đóng gói của P_inf** (multiset hash bằng nhau, kiểm trong preflight) và chỉ đổi thứ tự
batch: chặng 1 (0–40 % token) văn nói 10 %, chặng 2 (40–70 %) 20 %, chặng 3 (70–100 %, trùng decay LR) 66,7 % → trung
bình 0,4·0,10 + 0,3·0,20 + 0,3·0,667 = 0,300. Nhật ký huấn luyện cho thấy đúng hai mốc chuyển chặng (loss nhảy ở step
1 526 và 2 670, mốc sau trùng lúc LR bắt đầu decay).

## 4. Preflight (§4)

`artifacts/mmbert_vi_cpt_pilot/v4/preflight.json` — **all_ok = true**: `ModernBertForMaskedLM` load sạch 0
missing/unexpected/mismatched, MLM head pretrained, decoder tied, **307,79 M tổng / 111,18 M không-embedding**; tokenizer
Gemma 256K (pad 0, eos 1, mask 4); attention `sdpa` (flash_attn không có trong venv) dùng chung ba arm; mỗi arm đúng
`[3815, 128, 512]`, 250 019 840 token, **0 pad id**; lưới mask theo `(step, micro)` **giống nhau ở cả ba arm**; dựng lại
arm cho hash trùng khít; smoke 50 step loss/grad hữu hạn, 62,7 k token/s, peak 21,0 GB, ETA 1,11 h/arm.

## 5. CPT 250 M và holdout MLM

| Arm | wall (h) | GPU-giờ | tok/s | loss (100 step đầu → cuối) | a_hlpt 0→50→100 % | b_fineweb2 | c_visobert (NC) | d_v3review |
|---|---:|---:|---:|---|---|---|---|---|
| P_web | 1,22 | 1,22 | 57 067 | 1,046 → 0,850 | 1,368 → 1,218 → **1,162** | 1,352 → 1,217 → **1,158** | 4,064 → 3,744 → 3,652 | 2,555 → 2,355 → 2,236 |
| P_inf (NC) | 1,09 | 1,09 | 63 487 | 1,869 → 1,443 | 1,368 → 1,231 → 1,176 | 1,352 → 1,221 → 1,167 | 4,064 → 2,926 → **2,783** | 2,555 → 2,228 → **2,145** |
| C_inf (NC) | 1,10 | 1,10 | 63 251 | 1,326 → 2,185 | 1,368 → 1,223 → 1,188 | 1,352 → 1,217 → 1,179 | 4,064 → 3,076 → 2,798 | 2,555 → 2,242 → 2,171 |

Đọc bảng này: P_inf/C_inf cải thiện mạnh trên **holdout đúng miền của nó** (`c_visobert`, −1,28 và −1,27 so với P_web) và
nhỉnh hơn một chút trên `d_v3review` (−0,09) nhưng **kém hơn P_web** trên HPLT và FineWeb-2 (+0,01…+0,02) — tức không có
chuyển giao miền rộng, chỉ học miền văn nói. Checkpoint cuối ngân sách, export encoder-only, reload khớp, không chứa
head/projection lạ, smoke inference hữu hạn (đã kiểm trong `summary.json` của từng arm; P_inf/C_inf có cờ `research_only`).

## 6. Fine-tune (13 lượt, E14, validation-only)

Recipe E14 như `scripts/base_encoder_baseline/finetune.py`: head CLS E14, CE, 30 epoch, eff. batch 32, maxlen 256, bf16/SDPA,
best epoch theo validation, LR khóa (Restaurant 5e-5 · ViSFD 5e-5 · Topic 2e-5), không test. O seed 42 reuse từ baseline.

| Arm | Restaurant (seed 42 / 43 / **pair-mean**) | ViSFD (42) | Topic (42) |
|---|---|---|---|
| **O** (mmBERT gốc) | 75,63 / 76,42 / **76,02** | 83,64 | 81,18 |
| **P_web** | 76,31 / 74,50 / **75,40** | 83,18 | 80,21 |
| **P_inf** (NC) | 74,05 / 74,17 / **74,11** | **84,34** | 80,10 |
| **C_inf** (NC) | 76,09 / 74,22 / **75,16** | 83,42 | 80,76 |
| Cafe (tham chiếu) | 77,72 | 85,47 | 81,74 |
| Bami (tham chiếu) | 74,74 | 84,38 | 79,53 |

Delta (điểm F1 validation; Restaurant = pair-mean, task khác seed 42):

| So sánh | Restaurant | ViSFD | Topic |
|---|---:|---:|---:|
| **Q1** P_web − O | **−0,62** | **−0,46** | **−0,97** |
| **Q2** P_inf − P_web | **−1,29** | +1,16 | −0,11 |
| **Q3** C_inf − P_inf | **+1,05** | −0,92 | +0,66 |

Best epoch: P_web 25/29 (Restaurant), 14 (ViSFD), 28 (Topic) · P_inf 8/11, 18, 26 · C_inf 15/12, 28, **2** (Topic — đỉnh
rất sớm, dấu hiệu nhiễu). Mọi run `status=completed` với `reload_matches_best` và `ids_stable_and_complete` = true
([metadata.json](/home/kiettqa/GoiBERT/artifacts/mmbert_vi_cpt_pilot/v4/ft/)).

## 6b. Test (mở theo yêu cầu người dùng, 2026-09-20)

Checkpoint = đúng checkpoint validation đã chọn (`best_checkpoint.pt`, best epoch on validation); không chọn lại model,
epoch hay LR sau khi nhìn test; metric và loader y hệt validation nhưng với split `test`; mỗi lượt chạy 2 lần để kiểm
prediction tái lập (tất cả `reload_prediction_repeatable = true`). Chi tiết:
`artifacts/mmbert_vi_cpt_pilot/v4/test_summary.json` + `test_tables.md` + `<ft dir>/test_eval/test_evaluation.json`.

| Arm | Restaurant (seed 42 / 43 / **pair-mean**) | ViSFD (42) | Topic (42) |
|---|---:|---:|---:|
| **O** (mmBERT gốc) | 74,38 / 74,57 / **74,48** | **84,38** | **80,81** |
| **P_web** | 74,20 / 73,72 / **73,96** | 83,59 | 78,88 |
| **P_inf** (NC) | 74,28 / 74,20 / **74,24** | 83,45 | 78,50 |
| **C_inf** (NC) | 75,75 / 74,44 / **75,09** | 83,61 | 79,24 |
| CafeBERT (test nội bộ, Restaurant, tham chiếu) | 77,39 / 76,85 / **77,12** | — | — |
| BamiBERT (test nội bộ, Restaurant, tham chiếu) | 74,15 (seed 42) | — | — |
| BamiBERT (test theo paper, thang/metric của paper) | 74,89 | 84,24 | 79,90 |

| Delta test | Restaurant | ViSFD | Topic |
|---|---:|---:|---:|
| **Q1** P_web − O | **−0,52** | **−0,79** | **−1,93** |
| **Q2** P_inf − P_web | +0,28 | −0,14 | −0,38 |
| **Q3** C_inf − P_inf | **+0,85** | +0,16 | +0,74 |

Đọc kết quả test:

* **P_web thấp hơn O ở cả ba task** (−0,52 … −1,93) — kết luận Q1 "tiếng Việt sạch không giúp" giữ nguyên trên test.
* **P_inf ≈ P_web** (Q2: +0,28 Restaurant nhưng −0,14 ViSFD, −0,38 Topic) — thêm 30 % văn nói **không** cải thiện; khác
  với validation (Restaurant −1,29), trên test chênh lệch nhỏ hơn nhiều, tức tín hiệu âm ở validation phần lớn là nhiễu
  seed/kỳ vọng chứ không phải mất mát lớn.
* **C_inf > P_inf ở cả ba task** (+0,85 / +0,16 / +0,74) và là arm **duy nhất vượt O ở Restaurant test** (+0,61 so với
  O pair-mean 74,48; +1,13 so với P_web) — tín hiệu curriculum (Q3) dương đều trên test, dù vẫn dưới CafeBERT
  (−2,03) trên Restaurant.
* ViSFD/Topic: mọi arm CPT đều dưới O trên test (ViSFD −0,77…−0,93; Topic −1,57…−2,31) → không có lợi ích ngoài
  Restaurant; ViSFD test của C_inf (83,61) chỉ kém BamiBERT paper 0,63 và kém O 0,77.

**Cảnh báo sử dụng**: test đã dùng cho các model này; không gọi là holdout kín nữa. Nếu sau này muốn xác nhận bản đầy
đủ, cần giao thức test mới (ví dụ seed/holdout khác hoặc chấp nhận test đã "cháy" và ghi rõ).

## 7. Gate quyết định (v3 §6) và quyết định

* **Q1 — MLM tiếng Việt sạch có giúp?** **Không.** P_web − O = −0,62 (Restaurant), −0,46 (ViSFD), −0,97 (Topic): không
  task nào ≥ +0,5 và Restaurant còn thấp hơn O quá 0,5 → **Q1 fail**.
* **Q2 — thêm 30 % văn nói người dùng?** **Không đạt tiêu chí.** Restaurant −1,29 so với P_web (điều kiện chính là
  Restaurant ≥ +0,5) → **Q2 fail**, dù ViSFD tăng +1,16.
* **Q3 — dồn văn nói vào chặng decay?** Restaurant +1,05 (≥ +0,5) và Topic +0,66 → **có tín hiệu tích cực**, nhưng theo
  plan Q3 chỉ diễn giải khi Q2 đạt → không dùng để mở bản đầy đủ.
* **An toàn:** C_inf ≥ O − 0,5 ở ViSFD (83,42 ≥ 83,14) và Topic (80,76 ≥ 80,68) ✓; **P_inf vi phạm** ở Topic
  (80,10 < 80,68; −1,08 so với O).

**Quyết định:** không gate nào đạt → **dừng, không đầu tư bản đầy đủ** (không 3 epoch / 1,13 B+, không 5 task × 2 seed,
không test). Không có đề xuất dùng P_inf/C_inf cho bất kỳ mục đích phát hành nào (ràng buộc NC §0).

## 8. Chi phí

| Pha | GPU-giờ |
|---|---:|
| CPT 3 arm × 250 M token | 1,22 + 1,09 + 1,10 = **3,41** |
| 13 fine-tune (3 GPU song song) | ≈ **3,5** (tổng 211 phút GPU) |
| Hậu kiểm/preflight (smoke + profile) | ≈ 0,15 |
| **Tổng** | **≈ 7,1 GPU-giờ** (kế hoạch ước ~4; phần chênh là 30 epoch × 13 lượt fine-tune) |

## 9. Giới hạn

* **Nhiễu seed lớn hơn hầu hết delta**: P_web Restaurant seed 42 so với 43 chênh 1,81 điểm; vì vậy ViSFD/Topic (một seed)
  chỉ là tín hiệu yếu, và ngưỡng 0,5 của gate là heuristic chứ không phải kiểm định ý nghĩa.
* **Lệch miền**: ViSoBERT là bình luận mạng xã hội nói chung, không phải review nhà hàng/khách sạn; kết quả Restaurant
  (−1,29) là phép thử khắc nghiệt nhất và đúng như tiền nghiệm round 10 (CafeBERT + 300 M ViSoBERT + KD không tăng
  Restaurant).
* **NC**: mọi kết luận về P_inf/C_inf chỉ có giá trị nghiên cứu; bản phát hành (nếu có) phải làm lại với nguồn có license.
* Audit 100 mẫu do một người đọc; teencode coverage là bảng mô tả (đếm trên mẫu, quy về tỷ trọng văn nói của P_inf).
* Không đánh giá trên test; mọi số là validation.

## 10. Artifact

| Đường dẫn | Nội dung |
|---|---|
| `scripts/mmbert_vi_cpt_pilot/v4/` | `common_v4.py`, `prep_visobert.py`, `audit_v4.py`, `build_arms_v4.py`, `prep_holdouts_v4.py`, `preflight_v4.py`, `cpt_train_v4.py`, `finetune_v4.py`, `run_ft_gpu_v4.sh`, `summarize_v4.py` |
| `data/mmbert_vi_cpt_pilot/v4/visobert/` | `visobert_train.npy` (328 021×512), `visobert_holdout.npy`, `comments.jsonl`, `manifest.json` |
| `artifacts/mmbert_vi_cpt_pilot/v4/` | `DATA_GATE_VERDICT_V4.json`, `data_audit_v4/`, `preflight.json`, `arms/` (P_web, P_inf_NC_research_only, C_inf_NC_research_only), `holdouts/`, `cpt/`, `ft/`, `summary.json`, `report_tables.md` |
