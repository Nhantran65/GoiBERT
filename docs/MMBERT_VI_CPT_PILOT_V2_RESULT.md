# mmBERT Vietnamese CPT pilot v2 — vá gate dữ liệu: **DỪNG Ở GATE v2 (không train)**

**Trạng thái:** v2 dừng ở gate dữ liệu §3 của plan v2. **Không** materialize arm, **không** preflight, **không** CPT,
**không** fine-tune (đúng yêu cầu "chỉ khi mọi ô đạt mới chạy tiếp"; không hạ ngưỡng, không thêm nguồn). Ngày chạy
2026-09-19.

**Tóm tắt:** vá được vấn đề "văn bản xoay từ/dịch máy" của HPLT (general bin 7–9 sau lọc: 0–1 % spun, 100 % tiếng Việt,
boilerplate ≤4 %, spam ≤1 %, **1,134 B token ≥ ngưỡng 1,0 B**) và Wikipedia sau lọc đạt (68,9 M token, 100 % bài có nội
dung thật). Nhưng **bucket register vẫn trượt**: đọc tay 100 đoạn/ô cho thấy chỉ **20 %** (diễn đàn/hỏi đáp) và **12 %**
(review/thương mại) là ý kiến/review/thảo luận ở ngôi người dùng — so với ngưỡng **≥70 %**; bản trích đoạn "lỏng" (trước
khi siết) chỉ đạt 26 %. Đồng thời, khi siết để tăng độ chính xác thì register chỉ còn **37,5 M token < 50 M**, và phép
kiểm tác dụng phụ cho thấy vá 1 loại **38 %** doc UGC đã audit là thật (>15 %). Vì vậy verdict là **FAIL**, dừng trước
CPT theo đúng plan v2 §3.

## 1. Vì sao phải vá (nhắc lại kết quả v1)

v1 dừng ở gate dữ liệu: audit 800 doc cho thấy general bin 9 có 16 % và bin 10 có 41 % văn bản xoay từ/dịch máy; bucket
`max(OP,re) ≥ 0,5` chỉ 33–41 % là ý kiến người dùng (`docs/MMBERT_VI_CPT_PILOT_RESULT.md`). v2 vá ba chỗ: lọc spun
(vá 1), lấy register theo nền tảng UGC ở mức đoạn (vá 2), và chỉnh general/Wikipedia (vá 3).

Không file nào của v1 bị sửa: toàn bộ code v2 nằm ở `scripts/mmbert_vi_cpt_pilot/v2/`, artifact ở
`artifacts/mmbert_vi_cpt_pilot/v2/` và `data/mmbert_vi_cpt_pilot/v2/`. Model vẫn gọi đúng: **mmBERT-base 307 M tổng
(110 M không-embedding)**; v2 không train nên không dùng GPU nào.

## 2. Vá 1 — lọc văn bản xoay từ/dịch máy (S1 + S2)

**S1 — LM âm tiết Kneser-Ney (pure NumPy; `kenlm` không có trong venv).** Tập tham chiếu sạch = document HPLT bin 7–8
thuộc danh sách báo chính thống cố định trong `common_v2.MAINSTREAM_NEWS_HOSTS` (36 host) **cộng** câu review tiếng Việt
của `phamluan/vn_msa_sar_v3 @ 6f3f17b3` (CC BY 4.0):

| Tham số S1 | Giá trị |
|---|---|
| Document tham chiếu | 197 758 (báo + câu v3) |
| Âm tiết huấn luyện | 40 000 000 (cap) |
| Vocab | 331 442 âm tiết |
| Bigram / trigram unique | 3 354 796 / 14 452 227 |
| Chiết khấu KN | D2 = 0,685 · D3 = 0,761 |
| Ngưỡng S1 (percentile 5 / 2 của tham chiếu) | **−7,232** / −8,169 (trung bình tham chiếu −5,094) |

**S2 — logistic regression char 3–5-gram TF-IDF + đặc trưng S1** (1 200 ký tự đầu, đúng cửa sổ audit v1):

| Tham số S2 | Giá trị |
|---|---|
| Hàng huấn luyện | 1 200 = 800 nhãn v1 (`machine_translated_or_generated`) + **400 doc gán tay mới** (seed 42, bin 7–9, 26 positive) |
| AUC 5-fold | **0,819** (các fold 0,791–0,845); số đặc trưng 43 329 |
| Ngưỡng khóa (recall ≥ 0,90) | **S2 ≥ 0,2564** → recall 0,902 · precision 0,397 |
| Quy tắc loại | `S2 ≥ ngưỡng` **hoặc** `S1 < p5` (khóa **trước** khi chấm toàn pool; `thresholds.json`) |

**Tác dụng của vá 1 trên pool** (2 897 668 doc v1 đã chấm):

| Bin | Token trước | Token bị vá 1 loại | Tỷ lệ loại |
|---|---:|---:|---:|
| 9 | 1 826 033 295 | 1 174 902 517 | 64,3 % |
| 8 | 678 303 982 | 424 678 431 | 62,6 % |
| 7 | 739 305 579 | 495 259 682 | 67,0 % |
| 10 (bỏ theo vá 3) | 16 098 389 | 13 108 154 | 81,4 % |

(Tỷ lệ loại theo doc: bin 9 56,1 %, bin 8 60,0 %, bin 7 61,8 %; tổng thể 58,1 %.) Top host bị loại nhiều nhất:
truyenfull.vn 27,2 M token, webtruyen.com 9,5 M, vietgiaitri.com 7,7 M, truyenhay.com 6,4 M, vietbao.com 6,2 M,
truyenyy.com 5,9 M, 123doc.org 5,8 M, gacsach.com 5,2 M — tức bộ lọc bắt mạnh thể loại truyện dịch/SEO, đúng mục tiêu.

**Tác dụng phụ (theo plan phải ≤15 %):** lấy mẫu 150 doc UGC *trước* vá 1 (shard 9_2, `side_effect_sample.jsonl`), đọc
tay 100 mẫu: **13 doc là ý kiến/review thật**, trong đó **5 bị vá 1 loại = 38 %** (>15 %). Nguyên nhân là **S2**, không
phải S1: hạ S1 về percentile 2 (đúng biện pháp plan đề) **không đổi** kết quả (vẫn 5/13) vì các doc đó đều có S2 ≥ 0,256.
Đây là một trong các lý do verdict FAIL.

## 3. Vá 2 — census host và bucket register theo nền tảng người dùng

**Census** (`host_census.py`, đọc `u` từ 5 shard raw, không JSON-parse toàn bộ): **43 572 811 doc, 1 026 249 host**,
trong đó pool v1 (1/8) có 2 897 668 doc / 3,26 B token. Whitelist seed (37 host) chiếm **293 304 doc raw** ≈ 0,67 % doc và
12,96 M token pool (~104 M token nếu tính toàn bộ shard). TripAdvisor–Agoda–Booking bị **loại** theo plan
(`tripadvisor.com.vn` 52 293 doc, `agoda.com` 11 971, `booking.com` 304).

**Trích đoạn UGC** (`ugc_extract.py` + `register_refine.py`), bin 7–9, có decontam benchmark:

| Bước | Doc | Token |
|---|---:|---:|
| Doc thuộc whitelist (bin 7–9) | 410 505 | — |
| Qua lọc v1 + trích đoạn có cue ngôi thứ nhất/đánh giá | 323 504 | 87,2 M (sau vá 1) |
| Siết thêm (đoạn ≤700 ký tự, cue ở 60 % đầu hoặc ≥2 hit, loại marker pháp lý/kỹ thuật/thương mại) | 78 886 | 30,4 M |

**Bucket register cuối (bản siết, dùng cho gate)**: UGC 29,6 M + `discussion` (ID ≥ 0,5, đã qua vá 1) 7,9 M =
**37,5 M token** (21 % là discussion, dưới cap 30 %) → **không đạt ≥50 M**; cũng không đạt mức 240 M của chương trình 1 B.

## 4. Vá 3 — general và Wikipedia

* **General = bin 7–9 sau vá 1**, bỏ bin 10, trừ doc dành cho register (UGC + discussion), **cap host 2 %**:
  **1 134 178 900 token / 1 163 152 doc** (bin 9 642,6 M · bin 8 250,5 M · bin 7 241,1 M) → **đạt ≥1,0 B**.
  Cap host không phải cắt host nào (host lớn nhất `baomoi.com` chỉ 0,91 % tổng token).
* **Wikipedia**: từ 128 462 doc / 91,4 M token của v1, giữ bài ≥300 từ (≈330 mmBERT token) và không khớp mẫu stub →
  **36 551 doc / 68,9 M token ≥ 60 M** → **giữ Wikipedia**, không cần chuyển 5 % về general.

## 5. Audit v2 (100 mẫu/ô, seed 42, một người đọc) và verdict

| Ô | Tiếng Việt | Boilerplate | Spun/MT | Spam | Ý kiến người dùng | Ngưỡng | Đạt |
|---|---:|---:|---:|---:|---:|---|---|
| general bin 9 | 100 % | 4 % | 1 % | 0 % | — | VN ≥90 · boil ≤10 · spun ≤5 · spam ≤3 | ✅ |
| general bin 8 | 100 % | 4 % | 1 % | 1 % | — | nt | ✅ |
| general bin 7 | 100 % | 4 % | 0 % | 0 % | — | nt | ✅ |
| register UGC — diễn đàn/hỏi đáp | 100 % | 9 % | 0 % | 0 % | **20 %** | **≥70 %** | ❌ |
| register UGC — review/thương mại | 100 % | 24 % | 0 % | 0 % | **12 %** | **≥70 %** | ❌ |
| register UGC — bản "lỏng" (tham chiếu) | 100 % | 34 % | 0 % | 0 % | 26 % | ≥70 % | ❌ |
| `discussion` sau vá 1 | 100 % | 18 % | 0 % | 0 % | 82 % | ≥70 % | ✅ |
| Wikipedia sau lọc | 100 % | 0 % | — | 0 % | — | bài thật ≥70 % | ✅ (100 %) |
| Tác dụng phụ vá 1 | — | — | — | — | 13 doc thật, 5 bị loại (**38 %**) | ≤15 % | ❌ |

**Lượng:** general 1,134 B token ✅ · register 37,5 M token ❌ (cần ≥50 M; 240 M ❌).

**Verdict: FAIL** (`artifacts/mmbert_vi_cpt_pilot/v2/DATA_GATE_VERDICT_V2.json`). Các check fail:
`register_forum_qa_opinion_ge_70`, `register_review_commerce_opinion_ge_70`, `side_effect_le_15`,
`register_tokens_ge_50M` (+ mức 240 M không đạt).

## 6. Vì sao register trượt — bằng chứng đọc tay

Các đoạn được giữ trong bucket register (kể cả sau khi siết) phần lớn **không phải** review/ý kiến người dùng:

* `thuvienphapluat.vn`, `luatminhkhue.vn`, `danluat.*`, `banan.*` (chiếm 11,2 M + 3,6 M + 1,8 M token): HPLT bắt được
  **trang văn bản pháp luật/bản án/tư vấn luật**, không phải câu hỏi của người dùng; cue ngôi thứ nhất xuất hiện rải rác
  trong đoạn dài nên bộ lọc theo cue vẫn giữ lại.
* `fptshop.com.vn`, `thegioididong.com`, `cellphones.com.vn`, `tiki.vn`, `lazada.vn`: HPLT bắt được **trang sản phẩm,
  bài tư vấn, nội dung pháp lý/hành chính** thay vì mục đánh giá của người mua.
* `cuasotinhyeu.vn`, `edoctor.io`, `webtretho.com`, `voz.vn`, `tinhte.vn`: có câu hỏi/bài tâm sự thật, nhưng là thiểu
  số trong mẫu (khoảng 1/5) và bị pha loãng bởi phần lớn nội dung là câu trả lời của chuyên gia, bài biên tập hoặc
  nội dung tổng hợp.

Nói cách khác: **tập URL của các nền tảng UGC không tương ứng với tập trang review trong HPLT**. Đây là hạn chế của
nguồn (HPLT 3.0 `vie_Latn` + cách chọn theo host), không phải của ngưỡng: kể cả bản trích đoạn lỏng (87 M token) cũng
chỉ 26 % là ý kiến người dùng, nên không thể vừa đạt ≥70 % precision vừa đạt ≥50 M token.

## 7. Quyết định và đề xuất

Theo plan v2 §3: **không đạt → viết báo cáo và dừng; không hạ ngưỡng, không thêm nguồn.** Do đó:

* Không `build_arms_v2`, không preflight, không CPT 250 M, không fine-tune; G1–G3 không được đo.
* Không mở 1 B, không seed thêm, không test, không Hotel/Sentiment, không mean-pooling.
* Giá trị lượt này: (i) một bộ lọc spun/MT đã hiệu chuẩn cho tiếng Việt (AUC 0,819; ngưỡng khóa; loại đúng nhóm
  truyện dịch/SEO), (ii) bằng chứng định lượng rằng **các nền tảng UGC tiếng Việt trong HPLT không cho đủ văn bản
  ý kiến/review chất lượng cao** ở mức 50 M token.
* Đề xuất cho root (không tự thực hiện): (1) lấy register từ nguồn **trang review có cấu trúc** (ví dụ API/CSV của
  Foody/Tiki/Shopee hoặc bộ dữ liệu review công khai) thay vì crawl host — đây là *nguồn mới*, cần plan mới; hoặc
  (2) chấp nhận Q1 (MLM tiếng Việt chung) với pool general đã đạt gate và tách câu hỏi register thành một nhánh khác;
  hoặc (3) mở rộng census sang bin 5–6 và các forum nhỏ rồi audit lại — nhưng phải là plan mới, vì hiện tại pool
  7–9 đã cạn phần review.

## 8. Giới hạn

* Audit do **một người đọc**, 100 mẫu/ô, đọc phần đầu đoạn (register) / 1 200 ký tự đầu (general, discussion, wiki);
  nhãn "ý kiến người dùng" là phán đoán chủ quan nhưng áp dụng nhất quán và ngưỡng cách xa kết quả (12–20 % so với 70 %).
* S1 là LM thuần NumPy (không KenLM) trên 40 M âm tiết cap; S2 dùng 1 200 hàng gán nhãn (800 nhãn v1 + 400 đọc tay mới,
  26 positive) nên precision tại ngưỡng recall-90 % chỉ 0,397 — bộ lọc **thô** ở phía "loại nhầm" (xem tác dụng phụ 38 %).
* Census dựa trên regex trường `u` (bỏ qua URL có ký tự escape); doc-level token của UGC được ước lượng bằng cách nhân
  8 từ pool 1/8 khi báo sản lượng, còn số liệu gate dùng token thật sau trích đoạn.
* Vì verdict FAIL, mọi số của v2 §4 (mixture, holdout thứ tư, CPT/fine-tune) **không được sản sinh** — đúng yêu cầu
  "không khởi động CPT trước verdict".

## 9. Artifact

| Đường dẫn | Nội dung |
|---|---|
| `scripts/mmbert_vi_cpt_pilot/v2/` | `common_v2.py`, `host_census.py`, `spun_filter.py`, `ugc_extract.py`, `register_refine.py`, `register_pool.py`, `general_pool.py`, `wiki_filter.py`, `audit_v2.py`, `gate_v2.py` |
| `artifacts/mmbert_vi_cpt_pilot/v2/host_census.json` | census 43,57 M doc / 1,03 M host; top 300 + host seed |
| `artifacts/mmbert_vi_cpt_pilot/v2/spun_filter/` | `lm_counts.npz`, `vocab.json`, `s2_model.pkl`, `cv_report.json` (AUC 0,819), `thresholds.json` (khóa trước khi chấm pool) |
| `data/mmbert_vi_cpt_pilot/v2/spun_scores/` | điểm S1/S2 + quyết định cho 2 897 668 doc pool |
| `artifacts/mmbert_vi_cpt_pilot/v2/general_pool.json` | general 1,134 B token, theo bin/host, tỷ lệ loại |
| `data/mmbert_vi_cpt_pilot/v2/{ugc,ugc_refined}/` + `artifacts/.../register/` | trích đoạn UGC (87,2 M → 30,4 M token) và bucket register 37,5 M token |
| `artifacts/mmbert_vi_cpt_pilot/v2/wiki_filter.json` | Wikipedia 36 551 bài / 68,9 M token |
| `artifacts/mmbert_vi_cpt_pilot/v2/data_audit_v2/{samples,labels,s2_labels}.jsonl` + `DATA_GATE_VERDICT_V2.json` | 900 mẫu audit + 400 nhãn S2, verdict FAIL (kèm độ nhạy S1 p2) |
