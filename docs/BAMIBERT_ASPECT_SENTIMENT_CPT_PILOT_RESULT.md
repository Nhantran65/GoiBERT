# BamiBERT aspect–sentiment CPT pilot — kết quả: **dừng ở gate dữ liệu (§3)**

Protocol: `BamiBERT-AS-CPT-pilot-v1`; plan: `docs/BAMIBERT_ASPECT_SENTIMENT_CPT_PILOT_PLAN.md`; ngày: 2026-09-18. Source: `scripts/bami_as_cpt_pilot/` (source hash `2fb7efe2…`); dữ liệu/manifest: `data/bami_as_cpt_pilot/`; quyết định gate: `artifacts/bami_as_cpt_pilot/data_gate.json`.

**Quyết định:** gate dữ liệu §3 **không đạt** trên mọi điều kiện định lượng và trên audit chất lượng của nguồn FineWeb2. Theo plan, pilot dừng **trước khi dùng GPU**: không tạo lexicon, không CPT R/S, không fine-tune. Không có số O/R/S/Cafe nào được sinh ra ở lượt này; các cột đó trong plan §6 để trống. GPU 0–2 không được dùng cho training (snapshot lưu trong `data_gate.json`).

## 1. Tóm tắt gate

| Điều kiện §3 | Ngưỡng | Đo được (tier strict) | Đạt |
|---|---:|---:|:--:|
| Audit tay ≥70% firsthand, mọi nhóm source×domain | ≥0,70 | v3/hotel **0,91**, v3/restaurant **0,94**, fw2/hotel **0,39**, fw2/restaurant **0,41** | ✗ (FineWeb2) |
| Train ≥4M BamiBERT non-special tokens duy nhất | ≥4.000.000 | **346.359** (v3 208.488 + fw2 137.871) | ✗ |
| Mỗi miền ≥0,5M | ≥500.000 | restaurant 170.183, hotel 176.176 | ✗ |
| 20M token ≤5 passes | ≤5 | **57,7** passes | ✗ |
| License/quyền dùng | — | v3 CC BY 4.0; FineWeb2 ODC-By 1.0 (quyền trang gốc chưa rõ, không phát hành lại) | ✓ |

Vì FineWeb2 trượt audit, corpus hợp lệ chỉ còn **v3: 208.488 token train** (hotel 141.693 / restaurant 66.795) → **95,9 passes** ở ngân sách 20M. Tier "relaxed" (chỉ để ước lượng trần, §3.3) cũng chỉ đạt 659.906 token (30,3 passes) và chưa được audit — kỳ vọng precision còn thấp hơn strict.

Không có điều kiện nào có thể cứu bằng cách hạ ngưỡng hay đổi bộ lọc trong phạm vi plan: khoảng cách token là ~11,5× (strict) hoặc ~19× (v3-only).

## 2. Nguồn dữ liệu đã pin

| Nguồn | Revision / hash | License | Ghi chú |
|---|---|---|---|
| `phamluan/vn_msa_sar_v3`, split **train** | revision `6f3f17b390dcf1bf411804e1bc8246bf88ea1bf4`; `data/train-00000-of-00001.parquet` sha256 `c7fe8ab1…d36601ce1` (khớp LFS) | CC BY 4.0 (card) | tải về `data/bami_as_cpt_pilot/raw/vn_msa_sar_v3/`; validation/test không tải |
| FineWeb2 `vie_Latn` shard `000_00000.parquet` | commit `af9c13333eb981300149d5ca60a8e9d659b276b9`; sha256 `337df3c3…6219acf` (khớp metadata tải của Pilot B, đã hash lại) | ODC-By 1.0 (card) | 2.319.000 document, 4,84 GB; chỉ trích xuất đoạn review ngôi người dùng |

Loại theo plan: ViSoBERT social (research-only/NC), ViHoRec/CAMEO (NC), `pqthinh232/…` (quyền không rõ), text train của năm benchmark, web chung. Thêm: trang `tripadvisor.*` bị loại toàn bộ vì review tiếng Việt trên đó là bản dịch máy (`"Các đánh giá này được dịch máy từ tiếng Anh"`); document chứa dấu hiệu dịch máy cũng bị loại (10.247 doc).

## 3. Audit v3 train

- 60.406 dòng = 14.536 review, 51.199 câu duy nhất; tag `language`: en 21.050, zh 14.663, ja 12.549, **vi 12.144**. Card ghi CC BY 4.0, chia split theo `review_id`.
- **Tag ngôn ngữ không tin được** (đúng như plan cảnh báo): kiểm theo nội dung (không CJK; ≥2% ký tự đặc trưng tiếng Việt hoặc ≥15% từ chức năng phổ biến) cho **16.128 dòng tiếng Việt**: 255 dòng tag `vi` không phải tiếng Việt (Anh, Ý, Bồ, Tonga, tiếng Việt không dấu bị chặn bởi ngưỡng), 4.239 dòng tag `en`/`zh` thực ra là tiếng Việt. Dùng kiểm nội dung, không dùng tag.
- Phần tiếng Việt: **4.790 review** (hotel 3.590 / restaurant 1.200; booking 1.874, google_maps 1.518, foody 1.015, other 383), 13.773 câu duy nhất, trung bình ~16 BamiBERT token/câu. Review được ghép lại từ các câu cùng `review_id` theo thứ tự xuất hiện; **thứ tự câu gốc không xác nhận được** từ bản phát hành, nên review ghép là xấp xỉ. 58 review >254 token bị cắt theo ranh giới câu.
- Dedup: 40 exact, 202 near-dup (≥50% shingle 8 từ đã gặp). **Decontam text-only** loại 68 review v3 trùng benchmark: absa_hotel/Train 41, absa_hotel/Test 10, visfd/Train 11, absa_hotel/Dev 3, visfd/Dev 3, absa_restaurant Train/Dev/Test 2/1/2, visfd/Test 1 — tức v3 có chồng lấn thật với UIT-ABSA Hotel (cùng loại nguồn booking/google), đã loại trước khi đếm.
- Audit tay 100 review/miền (seed 42): hotel **91/100** firsthand (9 other: review điểm đến/bãi biển/công viên nước 3, tóm tắt bên thứ ba 1, văn quảng cáo 1, chung cư 1, mảnh rỗng/header 3); restaurant **94/100** (6 other: giọng quảng cáo 1, chủ quán trả lời 1, tiếng Anh 2, khiếu nại sản phẩm cửa hàng 1, header 1). Một số review google_maps đọc như dịch máy (vd. "Người chồng tốt bụng… Chủ khách sạn sẽ gọi lại cho bạn") nhưng vẫn là giọng khách hàng nên tính firsthand; ghi nhận là rủi ro chất lượng.
- Thống kê nhãn phần tiếng Việt (chỉ để định cỡ lexicon §4, **không** dùng): aspect service 3.246, food 2.892, overall 1.926, amenities 1.489, location 1.384, room 1.344, ambiance 1.342, price 1.332, cleanliness 1.173; dấu sentiment theo câu duy nhất pos 8.361 / neg 4.932 / neu 480.

## 4. Trích xuất review từ FineWeb2 (`fw2_filter.py`, sha256 `be790b41…`)

Quy tắc cố định trước khi nhìn bất kỳ số downstream nào; chỉ chỉnh bằng cách đọc mẫu để tăng precision (2 vòng: vòng đầu marker miền quá lỏng — `món`, `vị`, `giường` — cho ~10% firsthand; vòng hai siết như dưới).

- Đơn vị: paragraph (tách theo newline) → tách câu → gán mỗi câu `neg` (giọng doanh nghiệp/quảng cáo/mô tả, tin tức/sức khỏe, công thức, khuyên bảo "bạn nên/hãy" không có ngôi thứ nhất, tiểu thuyết, số điện thoại/URL/bảng giá), `rev` (ngôi thứ nhất không phản thân + đánh giá/trải nghiệm), `neu`. Giữ run liên tiếp dài nhất không chứa `neg`, cắt về câu `rev` hai đầu. Không viết lại câu.
- Gate đoạn (strict): có **danh từ địa điểm mạnh** (khách sạn/ks/resort/homestay/nhà nghỉ/lễ tân/nhận-trả phòng/đặt phòng… hoặc nhà hàng/quán/tiệm/buffet/gọi món/menu/đồ ăn…), ≥1 ngôi thứ nhất (mình/tui/tôi/tớ/bọn mình…, trừ "của mình/cho mình/bản thân mình"…), ≥1 dấu trải nghiệm (đã ăn/ở, ghé, hôm…, lần…, ở đây, quay lại, order, check in…), ≥2 từ đánh giá, ≥2 danh từ review, 12–320 từ, tiếng Việt theo nội dung. Loại document: host truyện/story, dấu dịch máy, mật độ marker tiểu thuyết ≥1,5/100 từ, host tripadvisor.
- Tier relaxed (diagnostic): ngôi thứ nhất **hoặc** khẩu ngữ; trải nghiệm **hoặc** ≥3 đánh giá; ≥1 danh từ review.
- Chạy toàn shard: 2.319.000 doc, 42.027.962 paragraph, 48 worker, 369 s. Loại: 12.658.241 quá ngắn, 213.409 không phải tiếng Việt, 27.866.050 không có marker địa điểm, 1.245.234 không có run review, 43.074 trượt gate đoạn, 330.258 doc tiểu thuyết, 68.646 host truyện, 10.247 doc dịch máy, 581 tripadvisor. **Giữ: strict 1.954 đoạn** (restaurant 1.457 / hotel 497; 123.801 từ; 1.225 host, 943 host chỉ có 1 đoạn — top: khongsolac.com 50, grab.com 32, wesave.vn 25), **relaxed 9.020**.
- Sau dedup/decontam (strict): 1.925 đoạn (15 exact, 8 near-dup, 6 decontam). Train: 137.871 token (hotel 34.483 / restaurant 103.388); đoạn strict trung bình ~63 từ (~80 token); toàn bộ train sequence (v3+fw2) trung bình 55 token, median 42, p95 151.
- **Audit tay 100 đoạn/miền: hotel 39/100, restaurant 41/100 firsthand.** Phần "other" chủ yếu là: tường thuật du lịch có nhắc khách sạn/quán nhưng không đánh giá, bài khuyên/cẩm nang, văn quảng cáo, trích dẫn phỏng vấn/tin tức, tùy bút ẩm thực/hồi ký, tiểu thuyết/tâm sự, nấu ăn tại nhà, địa điểm khác (app, xe khách, rạp phim, hồ bơi công cộng, cửa hàng), và **trang nội dung người lớn** lọt qua vì có "khách sạn/nhà nghỉ" + ngôi thứ nhất. Review khách hàng được trích nguyên văn trong bài báo được tính firsthand.

Kết luận về FineWeb2: shard này gần như không chứa review thật ở ngôi người dùng về nhà hàng/khách sạn (≈0,08% document cho ra đoạn strict). Nếu ngoại suy theo yield strict (~138K token/shard) và precision ~40%, cần cỡ **70 shard (~330 GB)** chỉ để có 3,8M token *firsthand thô* — và vẫn trượt gate ≥70% trừ khi bộ lọc tốt hơn hẳn. Đây không phải đường đi khả thi trong ràng buộc của plan (không crawl/nguồn mới, không web chung).

## 5. Vì sao không chạy CPT với corpus nhỏ

Plan §3 nêu rõ ngưỡng 4M/0,5M/≤5 passes là **tối thiểu cho pilot** và cấm tự hạ ngưỡng hay thay bằng social/web chung. Với v3-only (208K token), 20M token là ~96 lượt lặp trên cùng ~4.400 review: kết quả sẽ đo hiệu ứng lặp/overfit corpus nhỏ chứ không đo giả thuyết "che có ưu tiên aspect–ý kiến có hơn MLM ngẫu nhiên". Lexicon §4 (`min_doc_freq=10`, top 50/aspect, top 200/dấu) cũng chỉ được khai thác từ ~13.700 câu; chưa tạo vì không có CPT để dùng.

## 6. Artifacts

- `data/bami_as_cpt_pilot/corpus_manifest.json` (tier strict; gồm revision/hash nguồn, license, counts trước/sau, token theo source/domain/split, audit tay theo cell, decontam theo split, sha256 tập ID holdout `holdout_doc_ids_sha256`), `tier_relaxed/corpus_manifest.json`.
- `data/bami_as_cpt_pilot/fw2_extract_stats.json`, `fw2_passages_{strict,relaxed}.jsonl` (đoạn + host/url + điểm; **không phát hành lại**), `corpus_{train,holdout}.jsonl` (đã token hóa, thứ tự train đã khóa seed 42 — R/S sẽ cùng ID/thứ tự nếu có vòng sau), `manual_audit_sample.jsonl` (sha256 `7f6be080…`), `manual_audit_labels.json` (sha256 `32209ae3…`, 400 nhãn, tiêu chí + ghi chú từng cell).
- `artifacts/bami_as_cpt_pilot/data_gate.json` (quyết định, hash script, snapshot GPU), `fw2_extract.log`.
- Không sửa raw gold/split, không sửa source/artifacts E1–E22, Pilot B, CDBK. Không có checkpoint mới.

## 7. Đề xuất cho root (không tự thực hiện)

1. Giả thuyết vẫn chưa được kiểm; kết luận đúng là *"không có corpus review restaurant/hotel tiếng Việt đủ lớn và đủ quyền dùng trong các nguồn plan cho phép"*, không phải selective masking sai.
2. Nếu muốn mở vòng sau cần root duyệt một trong: (a) nguồn review có license rõ đủ ≥4M token (cần tìm/đánh giá riêng; `phamluan/vn_hotel_res_aspect_v3` chia cùng review nên **không** thêm text mới); (b) hạ mục tiêu thành pilot *rất nhỏ* trên v3-only với ngân sách ~1M token (≤5 passes) — chỉ để kiểm cơ chế mask/pair và tín hiệu MLM, không kỳ vọng downstream; (c) đổi giả thuyết sang miền có corpus lớn hơn.
3. Bộ lọc FineWeb2 hiện tại (precision ~40%) không nên dùng làm nguồn CPT; nếu vẫn muốn khai thác web cần classifier có nhãn tay thay cho regex và phải kiểm lại quyền dùng trang gốc.
