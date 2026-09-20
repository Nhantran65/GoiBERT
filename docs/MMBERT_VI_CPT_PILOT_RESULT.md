# mmBERT-base + Vietnamese CPT — pilot 250 M token: DỪNG Ở GATE DỮ LIỆU (§3)

**Trạng thái:** pilot **dừng ở gate dữ liệu §3** theo đúng quy định của plan ("Không đạt → viết báo cáo lý do và
dừng"). Không chạy 15 fine-tune, không mở test, không hạ ngưỡng, không đổi nguồn. Ngày chạy: 2026-09-19.

**Tóm tắt một câu:** pool HPLT 7–10 tiếng Việt vượt cả hai ngưỡng về *lượng* (3,27 B token unique; 367 M token
register), nhưng **audit tay cho thấy bucket "register ý kiến/review" không đạt độ chính xác yêu cầu** (precision giọng
ý kiến người dùng 41 % ở ô bin 9–10 và 33 % ở ô bin 7–8, so với ngưỡng ≥70 %) — cờ `web-register.OP`/`re` của HPLT 3.0
trên tiếng Việt chủ yếu bắt nội dung SEO/viết lại, thông tin/giáo lý, tạp văn, chứ không phải văn bản ý kiến/review ở
ngôi người dùng. Vì vậy can thiệp Q2/Q3 (thêm 20 % register, dồn vào chặng decay) không còn đúng đặc tả, và pilot dừng
trước khi tiêu tốn phần compute còn lại.

## 1. Provenance, license, nguồn đã pin

| Hạng mục | Giá trị |
|---|---|
| HPLT 3.0 `vie_Latn` | HF dataset revision `3394d6ba8dae4da834e3b11771daf95028a960b1` (card HPLT/HPLT3.0), shards `10_1`, `9_2`, `9_1`, `8_4`, `7_1` từ `https://data.hplt-project.org/three/sorted/vie_Latn/<shard>.jsonl.zst` |
| SHA256 shards (đo sau tải, kiểm `zstd -t`) | `10_1` 294 598 179 B `d82ba990c2b6b2667d8ac628630da351b084f72968f4fdfea0db169d29b5c62d`; `9_2` 9 960 740 669 B `31d5c0303f3afc1bd91e4625d626ffe235675304578b809446c49ca4b3962071`; `9_1` 26 320 991 139 B `32c7ab8b33079d390fa6e4e79bef0d73fb36b5c77fe6b7acecb71fec485fd4b8`; `8_4` 16 045 366 519 B `f7072e1776753dc4121c2a0acfc694611b51a86560926ae2a0bb89143c00cdf7`; `7_1` 26 555 021 262 B `94fe72985e64b077f69229ba40800164dfefe452c33e9d19467c4582e7d357fa` |
| License HPLT | CC0 1.0 cho phần packaging của HPLT; pilot **không** sở hữu trang gốc, không giải quyết quyền cấp trang, không phát hành lại raw text |
| Wikipedia tiếng Việt | `wikimedia/wikipedia`, config `20231101.vi`, dataset revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, parquet `20231101.vi/train/0000.parquet` sha256 `ec53cc9899be9287b10785c794745eeef94b2df52970ba2ebd1808bfa8d1599a`; CC BY-SA 4.0 |
| FineWeb-2 (holdout b) | `HuggingFaceFW/fineweb-2@af9c1333…`, `data/vie_Latn/train/000_00000.parquet`, sha256 `337df3c3…`; ODC-By 1.0 |
| mmBERT-base | `jhu-clsp/mmBERT-base @ c5955035435e2bf121cde7f3c8863ef52ff35d82` — **307 M tổng / 110,3 M không-embedding** (đo lại: 307,79 M tổng, 111,18 M không-embedding) |

Ghi chú provenance: một phiên tải cũ (bash) để lại một bản ghép `9_2` sai (sha `cd670206…`) — bản đó bị
`fetch_hplt.py` từ chối (fail `zstd -t`) và tải lại; **bản được dùng để scan/tokenize là bản sha `31d5c030…`**. Về sau
file `9_2.jsonl.zst` bị một tiến trình lạ cắt về 0 byte; nó đã được tải lại và hash được xác minh lại đúng
`31d5c030…` (xem `data/raw/hplt3_vie/sha256.txt`).

## 2. Bộ lọc, dedup, decontam, holdout (§3)

Quy trình (code trong `scripts/mmbert_vi_cpt_pilot/`, artifact dưới `data/mmbert_vi_cpt_pilot/` và
`artifacts/mmbert_vi_cpt_pilot/`):

1. `scan_hplt.py`: `filter=="keep"`; `lang[0]=="vie_Latn"` và `prob[0]≥0,9`; ≥90 % segment `vie_Latn`; loại
   `web-register.MT ≥ 0,5`; loại doc >5 span PII hoặc >1 % ký tự PII; kiểm tiếng Việt theo nội dung (≥2 % ký tự đặc
   trưng hoặc ≥15 % từ chức năng); chuẩn hoá NFC + gộp khoảng trắng, giữ dấu; decontam text-only với train/dev/test của
   **cả năm benchmark** (exact key hoặc shingle 8 từ; loại đoạn dính, loại doc nếu >20 % đoạn dính); subsample đồng nhất
   theo hash id (1/8). Bucket: `opinion_review = max(OP, re) ≥ 0,5`; `discussion = ID ≥ 0,5` (và không thuộc opinion);
   `general` phần còn lại.
2. `dedup_pool.py`: exact (sha1 NFC-casefold) **1** doc; near-dup shingle 8 từ lấy mẫu (hash%16) với bloom 2^32 bit, loại
   **1 249 957** doc (nhiều trang dùng chung template/nav); holdout theo document: HPLT 5 000 doc, register 467 doc.
3. `tokenize_pool.py`: token hoá mmBERT (Gemma 256K) 2 903 124 doc → **3 269 873 509 token unique**; loại 2 055 doc
   >20 000 token, 0 doc <50 token.
4. `prepare_wiki.py`: Wikipedia vi → 128 462 doc / 91 363 372 token (exact-dedup chéo với toàn bộ doc HPLT đã scan,
   near-dup trong Wikipedia, decontam benchmark).
5. `prepare_holdouts.py`: holdout MLM ba tập — (a) HPLT 4 990 doc → 4 000 chuỗi; (b) FineWeb-2 5 000 doc → 4 000 chuỗi;
   (c) register 466 doc → 1 365 chuỗi (mask cố định seed 1234).

Counters thô (stage 1, tổng 5 shard): đọc vào 43 572 811 doc; giữ 4 155 137 doc; loại theo cờ MT 2 663 623 doc (6,1 %);
seg_langs <90 % 7 222 231; `lang/prob` 25 935; PII (quá nhiều span hoặc >1 % ký tự) 196 070; không phải tiếng Việt theo
nội dung 204 433; tiền lọc độ dài 9 643; decontam loại 4 341 doc (và cắt 236 406 đoạn dính benchmark); subsample đồng
nhất 1/8 loại 29 091 384 doc. Stage 3 đọc lại 4 155 137 doc, bỏ 1 249 958 doc không thuộc pool, giữ 2 903 124 doc và
loại 2 055 doc >20 000 token (0 doc <50 token).

## 3. Gate dữ liệu (§3)

| Tiêu chí | Ngưỡng | Kết quả | Đạt |
|---|---|---|---|
| Pool token unique (sau lọc/dedup/decontam) | ≥ 1,2 B | **3,27 B** (trong đó 2,67 B thuộc các nhóm dùng được cho arm: general + register + bin8 + wiki; 0,68 B là bin-7 general chỉ đọc để lấy register) | ✅ |
| Register token (bucket ý kiến) | ≥ 240 M (1 B) / ≥ 50 M (pilot) | **367,4 M** | ✅ (cả mức 1 B) |
| General token (bins 9–10) | ≥ 200 M (pilot) | **1,59 B** (+ bin 8 626 M) | ✅ |
| Audit `opinion_review` precision (giọng người dùng) | ≥ 70 % | **41 %** (bin 9–10) và **33 %** (bin 7–8) | ❌ |
| Audit general ≥90 % tiếng Việt thật | ≥ 90 % | 92–94 % ở ba ô general (99 % ở ô opinion) | ✅ |
| Audit general ≤10 % boilerplate | ≤ 10 % | 3–5 % | ✅ |
| Audit cờ MT ≥70 % đúng | ≥ 70 % | 100 % mẫu MT≥0,5 là văn bản dịch/viết lại hoặc spam | ✅ |
| **Gate tổng** | tất cả | **không đạt (audit register fail)** | ❌ |

Pool theo nhóm (token mmBERT): general (bin 9–10) 1 588 719 574; register 367 401 674; bin-8 general 626 247 593;
bin-7 general 677 372 404 (đọc chỉ để lấy register, không dùng cho arm nào); Wikipedia 91 363 372.

## 4. Audit tay 100 doc/ô (một người đọc, seed 42, đọc 1 200 ký tự đầu)

| Ô | n | tiếng Việt thật | boilerplate chiếm đa số | dịch máy/văn bản sinh | người lớn/spam | giọng ý kiến người dùng |
|---|---:|---:|---:|---:|---:|---:|
| bin10_general | 100 | 0,94 | 0,04 | 0,41 | 0,05 | — |
| bin9_general | 100 | 0,93 | 0,03 | 0,16 | 0,04 | — |
| bin8_general | 100 | 0,92 | 0,05 | 0,12 | 0,03 | — |
| opinion_review (bin 9–10) | 100 | 0,99 | 0,04 | 0,14 | 0,04 | **0,41** |
| opinion_review (bin 7–8) | 100 | 0,96 | 0,11 | 0,34 | 0,10 | **0,33** |
| discussion (ID≥0,5) | 100 | 0,90 | 0,06 | 0,24 | 0,08 | 0,71 |
| MT ≥ 0,5 (kiểm cờ) | 100 | 0,61 | 0,25 | 1,00 | 0,53 | — |
| Wikipedia | 100 | 1,00 | 0,00 | 1,00 (stub dịch/bot) | 0,00 | — |

Đọc mẫu cho thấy bucket `max(OP,re)≥0,5` gồm: bài SEO/viết lại (thường là máy dịch hoặc AI spin: "phim kinh khủng",
"bài bác giới thiệu sách", "Sữa Rửa Mặt Con Bò"), trang thông tin/how-to, bài suy niệm tôn giáo, tử vi/giải mộng, và
một phần thật sự là review/ý kiến người dùng (đánh giá điện thoại/mỹ phẩm/quán ăn, blog cá nhân, bình luận diễn đàn).
Ô `discussion` (ID≥0,5) tốt hơn (0,71) nhưng chủ yếu là hỏi–đáp y tế/pháp lý, không phải thảo luận về sản phẩm/dịch vụ.

## 5. Những gì đã chạy trước khi verdict audit có (và đã dừng)

Để audit có mẫu, `scan_hplt.py` phải chạy xong (excerpt 1/64 doc của mọi shard), nên thứ tự thực tế là: preflight →
materialize arm → CPT chạy ~6 % → audit đọc xong → **dừng**. Đây là sai lệch thứ tự so với plan (§3 yêu cầu audit đạt
trước khi materialize arm); nó được ghi lại đầy đủ ở đây và **không** dùng để ra bất kỳ quyết định gate nào.

* **Preflight (§4): pass toàn bộ** — `ModernBertForMaskedLM` load sạch 0 missing/unexpected/mismatched, MLM head
  pretrained, decoder tied; tokenizer Gemma 256K (pad 0, eos 1, bos 2, unk 3, mask 4); attention **sdpa** (flash_attn
  không có trong venv) dùng chung mọi arm; đóng gói đúng `[3815, 128, 512]`, **0 pad id**, 250 019 840 token/arm (sai số
  +0,008 % ngân sách, << 1 batch); mask 15 % 80/10/10 sinh bằng generator theo `(step, micro)` — **lưới chọn vị trí
  trùng nhau giữa ba arm** và tái lập được; dựng lại arm cho hash trùng khít; smoke 50 step loss/grad hữu hạn,
  62 k token/s, peak 21,0 GB, ETA 1,12 h/arm.
* **Arm đã materialize (mixture + thứ tự khoá):**

| Arm | general | register | bin8 | wiki | token | tokens hash | order hash | multiset hash |
|---|---:|---:|---:|---:|---:|---|---|---|
| P_web | 0,950 | 0 | 0 | 0,050 | 250 019 840 | `e05253ffe6bc…` | `f4bfd6bff855…` | — |
| P_mix | 0,650 | 0,200 | 0,100 | 0,050 | 250 019 840 | `5e6427fc54ac…` | `a404b6cc24b2…` | `2724a0607797…` |
| C_mix | 0,650 | 0,200 | 0,100 | 0,050 | 250 019 840 | `bd760da8c53a…` | `cbc277a122c4…` | `2724a0607797…` |

P_mix và C_mix dùng **cùng multiset** (cùng document, cùng số lần xuất hiện; multiset hash bằng nhau), chỉ khác thứ tự
batch. Về curriculum: với tỉ lệ register 20 % cố định bởi P_mix, chặng 2 không thể là "50/50" như plan ghi (0,3·r₂ +
0,3·0,6 = 0,20 ⇒ r₂ ≈ 6,7 %); bản dựng giữ **bất biến multiset** (điều kiện mạnh hơn) và ghi rõ sai lệch số học này.
* **CPT:** ba arm đã khởi động (một job/GPU trên GPU 0, 1, 2; GPU 3 không đụng tới), chạy tới step ~220/3815
  (≈ 5,8 % ngân sách, ~0,2 GPU-giờ mỗi arm) rồi **được dừng (SIGTERM)** ngay khi verdict audit hoàn tất. Không có
  checkpoint cuối, không có export, **không chạy fine-tune nào**, không dùng holdout MLM vào bất kỳ kết luận nào.

## 6. Quyết định và đề xuất

Theo §3 và §6 của plan: **không đạt gate dữ liệu ⇒ dừng, không hạ ngưỡng, không đổi nguồn, không chạy tiếp**. Do đó:

* G1/G2/G3 **không được đo** (không có fine-tune nào; O seed 42 vẫn là mốc baseline cũ).
* Không mở 1 B token, không seed thêm, không Hotel/Sentiment, không mean-pooling, không test.
* Giá trị thu được của lượt này là **bằng chứng phủ định về nguồn dữ liệu**: cờ register `OP`/`re` của HPLT 3.0 cho
  tiếng Việt không đủ chính xác để làm can thiệp "register ý kiến/review/thảo luận", dù sản lượng token rất lớn
  (367 M token bucket, 3,27 B token pool). Đây là thông tin trực tiếp cho thiết kế 1 B sau này.
* Đề xuất cho root (không tự thực hiện): (i) thay cách chọn register bằng tín hiệu khác (ví dụ lexicon miền
  nhà hàng/khách sạn/điện thoại/học tập + mẫu ngôi thứ nhất, hoặc một classifier register tự huấn luyện trên dữ liệu
  gán tay), rồi **audit lại đúng gate**; (ii) hoặc giữ nguyên câu hỏi Q1 (MLM tiếng Việt chung) với arm P_web — phần
  pool/audit của nó đã đạt — nhưng như vậy là một plan khác, cần root chốt trước khi chạy.

## 7. Giới hạn

* Audit do **một người đọc** trên 1 200 ký tự đầu của mỗi document (không đọc toàn văn), seed 42; nhãn "giọng ý kiến
  người dùng" là đánh giá chủ quan nhưng được áp dụng nhất quán, và ngay cả khi nới lỏng (tính cả tạp văn/khuyên
  nhủ/suy niệm có quan điểm) precision cũng chỉ ~50–55 %.
* Near-dup dùng bloom 2^32 bit, 1 hash/shingle (FPR ~21 % ở quy mô này) và subsample đồng nhất 1/8 theo id — các con số
  pool là của **mẫu 1/8** đó, không phải toàn bộ bins 7–10 (bins 7–10 đầy đủ ước tính ~20–26 B token).
* Cờ MT của HPLT bắt tốt văn bản dịch/spam (ô MT: 100 %), nhưng không bắt được văn bản "viết lại" vẫn qua bộ lọc —
  đây chính là phần lớn nội dung trong bucket register.
* Chất lượng ô `discussion` (0,71) gần ngưỡng; nếu vòng sau chỉ dùng discussion làm bucket register thì cần audit
  riêng với mẫu lớn hơn.

## 8. Artifact

| Đường dẫn | Nội dung |
|---|---|
| `scripts/mmbert_vi_cpt_pilot/` | `common.py`, `fetch_hplt.py`, `run_data_gate.sh`, `scan_hplt.py`, `dedup_pool.py`, `tokenize_pool.py`, `prepare_wiki.py`, `prepare_holdouts.py`, `audit.py`, `build_arms.py`, `preflight.py`, `cpt_train.py`, `finetune.py`, `run_cpt.sh`, `run_ft.sh`, `summarize.py` |
| `data/mmbert_vi_cpt_pilot/{scan,pool,tokens,wiki}/` | excerpt + meta + docs của stage 1, `pool.jsonl`, `holdout_hlpt.jsonl`, token array theo part, `*_manifest.json` |
| `artifacts/mmbert_vi_cpt_pilot/data_audit/` | `samples.jsonl` (800 mẫu), `labels.jsonl` (nhãn), `audit_report.json` |
| `artifacts/mmbert_vi_cpt_pilot/holdouts/` | `a_hlpt.npy`, `b_fineweb2.npy`, `c_register.npy`, `holdout_manifest.json` |
| `artifacts/mmbert_vi_cpt_pilot/arms/` | `P_web/`, `P_mix/`, `C_mix/` (`tokens.npy`, `manifest.json`), `arms_manifest.json` |
| `artifacts/mmbert_vi_cpt_pilot/preflight.json` | kết quả §4 (model/tokenizer/packing/mask/smoke/rebuild) |
| `artifacts/mmbert_vi_cpt_pilot/cpt/{P_web,P_mix,C_mix}/` | `status.json` (`stopped_at_data_gate`, step ~220, kèm lý do), `logs/` |
| `artifacts/mmbert_vi_cpt_pilot/DATA_GATE_VERDICT.json` | verdict máy-đọc-được của gate §3 |

Không file nào thuộc raw gold/split, source hay artifact cũ của project bị sửa.
