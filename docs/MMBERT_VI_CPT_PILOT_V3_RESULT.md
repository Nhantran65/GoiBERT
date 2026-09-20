# mmBERT + tiếng Việt văn nói — pilot v3: **DỪNG Ở GATE DỮ LIỆU (không train)**

**Trạng thái:** v3 dừng ở gate dữ liệu §3.3 của plan v3. **Không** materialize arm, **không** preflight, **không** CPT
250 M, **không** fine-tune 13 lượt (đúng yêu cầu "chỉ sau verdict đạt"), không dùng GPU, không sửa v1/v2. Ngày chạy
2026-09-19.

**Tóm tắt một câu:** vá được phần "general" (general bin 7–9 sau lọc mới: 1 132 387 854 token, tiếng Việt 100 %, spam
1 %, spun 0 % — đạt ≥1 B) nhưng **bucket văn nói không đạt chất lượng**: audit 100 bài cho thấy chỉ **13 %** là văn nói
thật (ngưỡng ≥70 %) và **13 % là spam** (ngưỡng ≤5 %); bộ lọc mới vẫn **loại 76,5 %** bài văn nói đã đọc tay là thật
(ngưỡng ≤15 %); độ phủ teencode chỉ **52/100 dạng** đạt ≥100 lần trong P_inf (vì "teencode" bắt được phần lớn là các
mục mơ hồ trong `teencode_map.json` như `vs`, `dừa`, `a`, `s`, `seo`, `bia`, `e`, `âu`). Verdict: **FAIL** → dừng, không
hạ ngưỡng, không thêm nguồn.

## 1. Bối cảnh: v2 dừng vì "register" từ web crawl, v3 đổi sang "văn nói"

v2 (`docs/MMBERT_VI_CPT_PILOT_V2_RESULT.md`) cho pool general sạch 1,134 B token nhưng bucket register theo nền tảng UGC
chỉ 12–26 % là review/ý kiến người dùng (ngưỡng ≥70 %) và chỉ 37,5 M token (ngưỡng ≥50 M). v3 thay giả thuyết: **cho
mmBERT học 250 M token tiếng Việt trong đó ≥30 % là văn nói (diễn đàn, hỏi đáp, teencode, ngôi thứ nhất) có giúp
Restaurant/ViSFD/Topic hơn là chỉ học văn bản báo chí sạch?** Câu trả lời chỉ được đo nếu gate dữ liệu §3.3 đạt.

## 2. Sửa bộ lọc spam (§3.1)

**Nhãn đọc tay.** Đã đọc 1 000 ứng viên (400 mẫu batch 1 + 600 mẫu bổ sung, seed 42) qua 4 tầng: `discussion`
(ID ≥ 0,3 + cue), host UGC whitelist v2, general có ≥3 teencode, và bài bị S2 v2 loại có tín hiệu mạnh. Kết quả:
**345 bài là văn nói thật** (221 discussion, 114 host UGC, 5 general, 5 v2-loại-oan) — đủ mức ≥300 mà plan yêu cầu; 655
bài còn lại không đạt định nghĩa văn nói (SEO, tin tức, truyện dịch, sản phẩm, spam).

**S1 v3 — LM âm tiết Kneser-Ney (pure NumPy)** dựng lại với tập tham chiếu mở rộng (báo chính thống bin 7–8 + câu
review v3 CC-BY + document `discussion` đã audit):

| Tham số S1 v3 | Giá trị |
|---|---|
| Document tham chiếu | 201 758 |
| Âm tiết | 70 000 000 (cap) · vocab 346 108 |
| Bigram / trigram unique | 4 638 231 / 21 779 886 |
| Ngưỡng p5 / p2 | **−6,983** / −8,753 |

**S2 v3 — logistic regression char 3–5-gram + S1**, huấn luyện trên **1 200 nhãn v2 + 345 nhãn văn nói mới (đều là
"không spam")** = 1 545 hàng, 367 positive:

| Tham số S2 v3 | Giá trị |
|---|---|
| AUC 5-fold | **0,873** (các fold 0,850–0,903) |
| Ngưỡng khóa (recall ≥ 0,90) | S2 ≥ **0,3065** → recall 0,902 · precision 0,402 |
| Quy tắc | general pool: `S2 ≥ ngưỡng` **hoặc** `S1 < p5`; ngưỡng khoá trước khi chấm pool |

**Gate tác dụng phụ (§3.1) — FAIL.** Trên 345 bài đã đọc tay là văn nói thật, bộ lọc mới **loại 264 bài = 76,5 %**
(ngưỡng ≤15 %). Theo plan, biện pháp khắc phục được áp dụng: **bucket văn nói dùng luật S1-only** (`S1 ≥ p5`), còn
general pool vẫn dùng S1+S2 (để giữ spam ≤5 % ở general); đã ghi rõ trong `informal_bucket/manifest.json`.

## 3. Bucket văn nói (§3.2) và general v3 (§3.3)

Bucket lấy từ HPLT (không thêm nguồn web mới), mọi bài đều qua filter/dedup/decontam 5 benchmark của v1:

| Nguồn | Ứng viên (doc / token) | Vào bucket (token) |
|---|---:|---:|
| `v2_dropped` — bài S2 v2 loại oan, có cue, S1 ≥ p5 | 411 643 / 654 921 032 | 146 791 499 |
| `discussion` — ID ≥ 0,3 + cue + S1 ≥ p5 | 8 587 / 8 913 928 | 8 913 928 |
| host UGC whitelist v2 + cue + S1 ≥ p5 | 3 332 / 4 294 824 | 4 294 824 |
| **Tổng bucket (cap 160 M theo thứ tự ưu tiên + mật độ tín hiệu)** | | **160 000 251** (116 922 doc) |

Xếp hạng trong bucket: ưu tiên `discussion` → host UGC → bài bị v2 loại → general, rồi theo mật độ tín hiệu văn nói
(teencode + 3×emoji + 0,2×đại từ)/1000 token. Có chặn thủ công 22 host spam (cá cược/người lớn) đã thấy ở audit v1/v2
(loại 527 doc). Held-out: **2 000 bài văn nói** tách trước khi train (`artifacts/mmbert_vi_cpt_pilot/v3/holdouts/c_informal.npy`,
4 000 chuỗi 512).

**General v3** (bin 7–9, filter mới S1+S2, trừ bài đã vào bucket, cap host 2 %):
**1 132 387 854 token / 1 226 854 doc** (bin 9 628 371 177 · bin 8 262 451 063 · bin 7 241 565 614) → **đạt ≥1,0 B**;
bin 10 bị bỏ (4 911 doc), filter mới loại 1 659 767 doc, bucket lấy 6 136 doc.

## 4. Audit v3 (100 bài/ô, seed 42, một người đọc) và verdict

| Ô | Tiếng Việt | Văn nói thật | Spam | Boilerplate | Spun/MT | Ngưỡng | Đạt |
|---|---:|---:|---:|---:|---:|---|---|
| **bucket văn nói** | 100 % | **13 %** | **13 %** | 13 % | 2 % | văn nói ≥70 · spam ≤5 · boil ≤15 · VN ≥90 | ❌ |
| general bin 9 (filter mới) | 100 % | — | **1 %** | 4 % | 0 % | spam ≤5 % | ✅ |
| tác dụng phụ bộ lọc mới | — | — | — | — | — | loại ≤15 % bài văn nói thật | ❌ (76,5 %) |
| lượng bucket | — | — | — | — | — | ≥75 M token | ✅ (160,0 M) |
| lượng general | — | — | — | — | — | ≥1,0 B token | ✅ (1,132 B) |
| độ phủ teencode (top-100 trong P_inf) | — | — | — | — | — | mỗi dạng ≥100 lần | ❌ (52/100) |

**Teencode coverage — nguyên nhân gốc.** Bảng "top-100 teencode" trong bucket bị chi phối bởi các mục **mơ hồ** của
`data/cpt/teencode_map.json` (384 mục), tức các từ/âm tiết bình thường hoặc chữ cái đơn cũng khớp:

| Dạng khớp | Nghĩa trong map | Ước lượng số lần trong P_inf |
|---|---:|---:|
| `vs` | với | 129 081 |
| `dừa` | vừa | 82 612 |
| `a` | anh | 77 449 |
| `s` | sao | 69 704 |
| `seo` | sao | 46 469 |
| `bia` | bê | 43 888 |
| `e` | em | 41 306 |
| `âu` | đâu | 36 143 |

(Ước lượng = đếm trên 100 bài audit của bucket rồi scale lên tỷ trọng văn nói của P_inf; xem
`data_audit_v3/teencode_coverage.json`.) Vì các mục này xuất hiện trong cả văn bản SEO/tin tức, tiêu chí "≥3 teencode =
văn nói" bắt sai rất nhiều — đúng với kết quả audit 13 % văn nói thật.

**Verdict:** `artifacts/mmbert_vi_cpt_pilot/v3/DATA_GATE_VERDICT_V3.json` = **FAIL**; các check fail:
`bucket_informal_ge_70`, `bucket_spam_le_5`, `side_effect_le_15`, `teencode_top100_each_ge_100_in_arm`.

## 5. Quyết định

Theo plan v3 §3.3: **không đạt → báo cáo và dừng; không hạ ngưỡng, không thêm nguồn web.** Do đó:

* Không materialize P_web/P_inf/C_inf, không preflight, không CPT 250 M, không fine-tune; Q1/Q2/Q3 chưa được đo.
* Không mở bản đầy đủ (3 epoch, 1,13 B+, 5 task × 2 seed), không Hotel/Sentiment, không seed thêm, không test.
* Đã kiểm license ViLexNorm: dataset tương ứng trên Hugging Face là `SEACrowd/vilexnorm` (bản ViLexNorm của bài EACL
  2024); theo card là research-only ⇒ chỉ được dùng làm holdout; pilot **không** dùng vì dừng ở gate. Shard `5_1`/`6_1`
  (bin 5–6) đã bắt đầu tải nhưng không dùng vì bucket đã đạt cap token mà chất lượng vẫn trượt.

**Giá trị thu được:** (i) một LM âm tiết + classifier spun/MT đã hiệu chuẩn (AUC 0,873) và bộ lọc mới vẫn giữ general
1,132 B token sạch (spam 1 %, spun 0 %); (ii) bằng chứng định lượng rằng **định nghĩa "văn nói" của plan (teencode map
384 mục + emoji + đại từ) không đủ phân biệt** trên HPLT tiếng Việt, và **S2 hiện tại loại oan 76,5 % văn nói thật**;
(iii) bucket văn nói chất lượng cao thực tế chỉ có ~13 M token (discussion ID ≥ 0,3 + host UGC), dưới cả ngưỡng 37,5 M.

**Đề xuất cho root** (không tự thực hiện): (1) xây **lexicon teencode đã kiểm định** (loại các mục mơ hồ như `a`, `s`,
`e`, `t`, `dừa`, `bia`, `seo`) hoặc classifier văn nói huấn luyện trên ~1 000 nhãn đọc tay, rồi audit lại; (2) nếu vẫn
không đủ 37,5 M token, cần **nguồn social mới có license** (ví dụ bộ social/comments công khai) — là plan mới; (3) hoặc
chạy Q1 với pool general v3 đã đạt gate như một nhánh riêng.

## 6. Giới hạn

* Audit một người đọc, 100 bài/ô, đọc 1 200 ký tự đầu; nhãn "văn nói thật" là phán đoán chủ quan nhưng khoảng cách
  ngưỡng rất lớn (13 % so với 70 %).
* Nhãn "văn nói" cho S2 (345 bài) lấy từ 4 tầng ứng viên, không phải mẫu ngẫu nhiên toàn pool; 655 ứng viên còn lại
  trong 1 000 mẫu đã đọc nhưng **không** đưa vào huấn luyện (chỉ 345 bài đạt định nghĩa văn nói được gán "không spam").
* Bảng teencode là **ước lượng** (đếm trên 100 bài audit × hệ số scale lên tỷ trọng P_inf), không phải đếm toàn arm
  (arm chưa được dựng vì verdict fail).
* Bucket dùng luật S1-only theo biện pháp khắc phục của plan; đây là lựa chọn đã ghi rõ, không phải thay phương pháp.

## 7. Artifact

| Đường dẫn | Nội dung |
|---|---|
| `scripts/mmbert_vi_cpt_pilot/v3/` | `common_v3.py`, `informal_scan.py`, `informal_sample.py`, `spun_filter_v3.py`, `informal_bucket.py`, `audit_v3.py` |
| `artifacts/mmbert_vi_cpt_pilot/v3/spun_filter_v3/` | `lm_counts.npz`, `vocab.json`, `s2_model.pkl`, `cv_report.json` (AUC 0,873), `thresholds.json` (khoá trước khi chấm pool) |
| `data/mmbert_vi_cpt_pilot/v3/` | `informal_scan/` (2,9 M doc), `spun_scores_v3/` (2,9 M doc), `informal_bucket/` (160,0 M token + manifest) |
| `artifacts/mmbert_vi_cpt_pilot/v3/data_audit_v3/` | `informal_labels.jsonl` (1 000 mẫu đọc tay, 345 văn nói thật), `samples.jsonl`/`labels.jsonl` (audit 2 ô), `side_effect_report.json`, `teencode_coverage.json` |
| `artifacts/mmbert_vi_cpt_pilot/v3/general_pool_v3.json` | general 1 132 387 854 token theo bin/host/counters |
| `artifacts/mmbert_vi_cpt_pilot/v3/holdouts/c_informal.npy` | holdout văn nói 2 000 bài (4 000 chuỗi 512) |
| `artifacts/mmbert_vi_cpt_pilot/v3/DATA_GATE_VERDICT_V3.json` | verdict FAIL + lý do |
