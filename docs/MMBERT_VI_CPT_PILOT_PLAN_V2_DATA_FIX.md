# mmBERT Vietnamese CPT pilot — v2: vá gate dữ liệu HPLT rồi mới train

**Trạng thái:** plan bàn giao, chưa chạy. Bổ sung cho `docs/MMBERT_VI_CPT_PILOT_PLAN.md` (v1); §4–§7 của v1 giữ nguyên trừ chỗ ghi rõ dưới đây. Ngày chốt 2026-09-19.

## 1. Vì sao phải vá (kết quả v1, `docs/MMBERT_VI_CPT_PILOT_RESULT.md`)

Pool HPLT 3.0 `vie_Latn` bin 7–10 đủ lượng (3,27B token unique) nhưng audit tay 800 doc cho thấy:

| Ô | Tiếng Việt thật | Spam xoay từ / dịch máy / sinh | Đúng loại |
|---|---:|---:|---|
| general bin 10 / 9 / 8 | 94 / 93 / 92% | **41 / 16 / 12%** | — |
| `opinion_review` (OP/re ≥ 0,5) bin 9–10 / 7–8 | 99 / 96% | 14 / 34% | ý kiến người dùng **41 / 33%** (gate ≥70%) |
| `discussion` (ID ≥ 0,5) | 90% | 24% | 71% |
| Wikipedia vi | 100% | — | 100% stub bot (loài/đô thị) |

Census host (mẫu 150K doc/shard): top host của bin 9 là truyện + SEO (truyenfull, xemanhnong, ruousanh, napthungxebantai…), của bin 7–8 là báo chính thống; các nền tảng người dùng viết (webtretho, voz, tinhte, tripadvisor, ivivu, mytour, agoda, foody, tiki, fptshop, edoctor, hoidap…) chiếm 0,4–1,2% doc, ước 0,2–0,3B token toàn pool. Kết luận: **điểm chất lượng WDS và nhãn register của HPLT không dùng được trực tiếp cho tiếng Việt**; phải thêm hai bộ lọc riêng và đổi cách lấy bucket register.

Hỗn hợp P_mix như v1 ước có ~16% spam xoay từ, ~12% văn bản nhầm loại, 5% stub. Chuẩn để train: **spam xoay từ/MT ≤5%**, **register đúng loại ≥70%**.

## 2. Ba bản vá (làm tuần tự, không GPU, không dùng benchmark text để chọn document)

### Vá 1 — Bộ lọc văn bản xoay từ/dịch máy (document-level)

Hai tín hiệu, cả hai **khóa trước khi chấm toàn pool**:

- **S1 — mô hình ngôn ngữ âm tiết**: bigram/trigram Kneser-Ney (pure-Python hoặc KenLM nếu cài được) trên **tập tham chiếu sạch** = document HPLT bin 7–8 từ whitelist báo chính thống (tuoitre, thanhnien, vnexpress, dantri, laodong, vietnamnet, tienphong, nld, sggp, baomoi… danh sách cố định ghi trong source) sau lọc boilerplate, cộng 13,7K câu review v3 (CC BY). Điểm = trung bình log-prob/âm tiết và tỷ lệ bigram chưa từng gặp. Văn bản xoay từ ("bây chừ say mê được một lượng phệ") sẽ rất thấp.
- **S2 — classifier có giám sát**: logistic regression (char 3–5-gram TF-IDF + S1 làm đặc trưng) huấn luyện trên **800 doc đã gán nhãn** (`data_audit/labels.jsonl`, cột `machine_translated_or_generated`) + tối thiểu **400 doc gán thêm bằng đọc tay** lấy ngẫu nhiên từ bin 7–9 (seed 42) để cân bằng; báo AUC cross-validation 5-fold; ngưỡng chọn để recall(spun) ≥90% trên CV, ghi precision tương ứng.
- Quy tắc loại: `S2 ≥ ngưỡng` **hoặc** `S1` dưới percentile 5 của tập tham chiếu. Ghi tỷ lệ loại theo bin và theo host (top 30 host bị loại nhiều nhất).
- **Kiểm tác dụng phụ**: trên 100 doc thuộc whitelist nền tảng người dùng (vá 2) đã audit là ý kiến thật, bộ lọc được phép loại **≤15%** (teencode/không dấu không được bị coi là spam). Nếu vượt, hạ ngưỡng S1 về percentile 2 và báo.

### Vá 2 — Bucket register lấy theo nền tảng người dùng, ở mức đoạn

- **Census host toàn pool** (đọc lại URL từ 5 shard raw đã tải, không tokenize): số doc/token theo host; xuất top 300 host và toàn bộ host khớp seed list dưới đây.
- **Whitelist UGC** (seed, có thể mở rộng bằng census nhưng phải ghi lý do từng host): diễn đàn/hỏi đáp — webtretho, voz.vn/vozforums, tinhte.vn, otofun, lamchame, hoidap247, hoidapphapluat, edoctor.io, cuasotinhyeu; review/du lịch/ẩm thực — foody.vn, diadiemanuong.com, riviu.vn, pasgo.vn, ivivu.com, mytour.vn, gody.vn, toidi.net, dulichchat, wesave.vn; thương mại điện tử có mục đánh giá — tiki.vn, shopee.vn, lazada.vn, thegioididong.com, cellphones.com.vn, fptshop.com.vn, hoanghamobile. **Loại** tripadvisor.*, agoda.*, booking.* (bản Việt chủ yếu dịch máy — đã kiểm ở pilot trước) trừ khi audit riêng ≥70%.
- **Trích đoạn**: trong document thuộc whitelist, tách đoạn; giữ đoạn có ≥12 đơn vị, qua vá 1, và có cue ngôi thứ nhất/đánh giá (lexicon FIRST_PERSON/EVALUATION/COLLOQUIAL tái dùng từ `scripts/bami_as_cpt_pilot/fw2_filter.py`, không dùng marker miền); loại đoạn listing/giá/điều hướng (PRICE_LIST, URL, PHONE). Đoạn giữ được ghép lại theo document (không xuyên document).
- Bucket register = UGC-paragraphs (chính) ∪ `discussion` ID ≥ 0,5 đã qua vá 1 (phụ, tối đa 30% bucket). Báo token theo host và theo nhóm.

### Vá 3 — General và Wikipedia

- General = **bin 7, 8, 9** sau vá 1 (bỏ bin 10: 41% spun, chỉ ~1% token); **cap mỗi host ≤2% token general** (chống truyenfull/vietbao chiếm pool); giữ tỷ lệ bin theo token thực có, ghi lại.
- Wikipedia: giữ bài ≥300 từ và không khớp mẫu stub (`là một loài`, `là một đô thị`, `là một xã`, `thị trấn thống kê`, `là một chi`…); nếu sau lọc <60M token → bỏ Wikipedia, chuyển 5% về general và ghi rõ.

## 3. Audit lại và gate dữ liệu v2 (100 doc/đoạn mỗi ô, seed 42, đọc tay, cùng bộ cột nhãn như v1)

| Ô | Ngưỡng |
|---|---|
| general bin 9 / bin 8 / bin 7 sau vá 1 | tiếng Việt thật ≥90%; boilerplate ≤10%; **spun/MT ≤5%**; người lớn/spam ≤3% |
| register UGC-paragraphs (2 ô: diễn đàn/hỏi đáp; review/thương mại) | **ý kiến/review/thảo luận ngôi người dùng ≥70%**; spun/MT ≤5% |
| `discussion` sau vá 1 | ≥70% (đã 71%; kiểm lại sau lọc) |
| Wikipedia sau lọc | bài có nội dung thật ≥70% |
| tác dụng phụ vá 1 | loại ≤15% doc UGC đã audit là thật |

Số lượng: general ≥1,0B token unique sau vá 1 và cap host; register ≥ **50M** (pilot) — báo có đạt 240M (1B) không; Wikipedia theo vá 3. **Mọi ô đạt → mới được materialize arm và train.** Không đạt → viết báo cáo, dừng; không hạ ngưỡng.

## 4. Thay đổi so với v1 khi được train

- **Trình tự bắt buộc**: vá → audit → verdict ghi `DATA_GATE_VERDICT_V2.json` → **rồi mới** `build_arms` → preflight → CPT. Không khởi động CPT trước verdict (lỗi trình tự của v1).
- **Lịch C_mix sửa số học** (tổng register 20%): chặng 1 (0–40% token) **5%**, chặng 2 (40–70%) **15%**, chặng 3 (70–100%, trùng decay) **45%** → 0,4·0,05 + 0,3·0,15 + 0,3·0,45 = 0,20. P_mix cùng document/cùng số lần xuất hiện, trộn đều.
- Mixture P_mix: general (bin 7–9, đã lọc, cap host) 70% + register 20% + Wikipedia 5% + `discussion` nằm trong register; nếu Wikipedia bị bỏ, general 75%. P_web: general 95% + Wikipedia 5% (hoặc 100% general).
- Còn lại (§4 preflight, §5 recipe 250M/WSD/lr 5e-5/mask 15%, §6 15 fine-tune và gate G1–G3, §7 artifacts) **y như v1**. Bổ sung holdout MLM thứ tư: 2K đoạn UGC held-out.

## 5. Deliverables

- Source mới `scripts/mmbert_vi_cpt_pilot/v2/` (hoặc module mới cùng thư mục, ghi hash; không sửa kết quả v1): `host_census.py`, `spun_filter.py` (S1+S2, model/threshold lưu file), `ugc_extract.py`, `wiki_filter.py`, `audit_v2.py`, `gate_v2.py`.
- Artifacts `artifacts/mmbert_vi_cpt_pilot/v2/{host_census.json, spun_filter/{model, cv_report.json, thresholds.json, drop_by_bin_host.json}, data_audit_v2/{samples, labels, audit_report.json}, DATA_GATE_VERDICT_V2.json}`; nếu qua gate: `arms_v2/`, `cpt/`, `ft/` như v1.
- Báo cáo `docs/MMBERT_VI_CPT_PILOT_V2_RESULT.md`: tác dụng của từng bộ lọc (trước/sau theo bin và host), AUC/threshold, census UGC theo host và token, bảng audit v2, verdict; nếu train: đủ mục §7 v1. Gọi model đúng: **mmBERT-base 307M tổng (110M không-embedding)**. Không test, không 1B, không thêm nguồn ngoài HPLT/Wikipedia/v3 (chỉ dùng làm tham chiếu LM).
