# Bước 0 — Rẽ nhánh: post-training có mục tiêu hay backbone mới?

Ngày 2026-09-19. Read-only, không GPU, không test, không sửa nhãn. Đầu vào: `artifacts/cafe_bami_paired_error_audit/events.jsonl` (audit ghép cặp CafeBERT–BamiBERT, validation seed 42). Source `scripts/cafe_bami_paired_error_audit/{step0_counts,build_blind_sets,compare_blind}.py`; artifacts `artifacts/cafe_bami_paired_error_audit/{step0_counts.json, blind_reading_comparison.json, blind_dispute_kinds.json, blind_reading/}`.

**Câu hỏi:** phần hơn của CafeBERT có đến từ một loại lỗi cụ thể của BamiBERT (→ có thể nhắm bằng cơ chế post-training) hay là "giỏi đều hơn" (→ phải xuống tầng pretraining)? Và bao nhiêu phần của khoảng cách nằm trên nhãn đáng ngờ?

## 1. Hai phép đếm không cần người đọc (`step0_counts.json`)

| Task | Gap Cafe−Bami (pp) toàn validation | Bỏ câu/slot **cả hai cùng sai giống nhau** | Bỏ mọi câu/slot cả hai cùng sai |
|---|---:|---:|---:|
| Restaurant (pair F1) | +2,98 | +3,61 | +3,69 |
| ViSFD (pair F1) | +1,08 | +1,21 | +1,23 |
| Topic (macro F1) | +2,21 | +2,22 | +2,88 |

Khoảng cách không co lại khi bỏ vùng nhãn đáng ngờ nhất: phần hơn của Cafe nằm trong vùng hai model bất đồng.

"Cảm xúc đúng, aspect sai" (cùng review: miss ở aspect A + spurious cùng polarity ở aspect B): Restaurant Bami 65 event / Cafe 62; ViSFD 75 / 69. Swap thuần (review 1 aspect gold, model dự đoán 1 aspect khác cùng polarity): Restaurant 37 / 30, ViSFD 6 / 11. Cặp aspect lặp nhiều nhất chỉ 4–5 lần (RESTAURANT#GENERAL→FOOD#QUALITY, GENERAL→FEATURES), xuất hiện ở **cả hai** model. **Không có lỗi lặp riêng của BamiBERT.**

## 2. Đọc mù bằng LLM (`blind_reading/`, `blind_reading_comparison.json`)

- Người đọc: 3 subagent Claude Sonnet 5 (một/task), chỉ nhận text + guideline gốc (VLSP 2018 ABSA / UIT-ViSFD / UIT-VSFC) + danh sách nhãn; ID ẩn danh, thứ tự xáo; **không thấy gold, không thấy dự đoán, không được đọc repo**. Mỗi agent tự xác nhận tuân thủ; mapping uid↔id giữ ngoài tầm với của agent tới khi đọc xong.
- Đơn vị: toàn bộ review có ≥1 slot bất đồng exclusive (cafe_only/bami_only): Restaurant 216 review (255 slot), ViSFD 371 (470 slot), Topic 95 câu; + đối chứng 40 review "cả hai cùng sai giống nhau" và 40 "cả hai đúng" mỗi task. 922 đơn vị, 100% được đọc, 0 dòng lỗi.
- Giới hạn: **một** người đọc LLM, không phải hai người thật → không có κ giữa người đọc; nhóm đối chứng "cả hai đúng" cho mức trần đồng thuận của reader với gold.

| | Restaurant | ViSFD | Topic |
|---|---:|---:|---:|
| Reader = gold trên đối chứng **cả hai đúng** (slot gold-present / câu) | 79% | 90% | 95% |
| Reader = gold trên đối chứng **cả hai cùng sai giống nhau** | 46% | 63% | 30% |
| … reader = model ở nhóm đó | 49% | 74% | 60% |
| Slot/câu bất đồng: **gold-supported** | 148/255 (58%) | 290/470 (62%) | 55/95 (58%) |
| … reader đứng về model bị tính sai | 74 (29%) | 136 (29%) | 31 (33%) |
| … không theo ai | 33 (13%) | 44 (9%) | 9 (9%) |
| Trong gold-supported: Cafe đúng : Bami đúng | **94 : 54** | **165 : 125** | **33 : 22** |
| Gap toàn validation → bỏ slot/câu tranh chấp | +2,98 → **+2,61** | +1,08 → **+0,87** | +2,21 → **+0,68** |
| Reader đánh dấu mơ hồ (nhóm bất đồng) | 8% | 3% | 44% |

Đọc: (i) khi cả hai model cùng bất đồng với gold, reader mù đứng về phía model nhiều hơn phía gold ở cả ba task → nhóm `both_wrong_same` đúng là nhãn đáng ngờ, nhưng nó không tạo ra khoảng cách Cafe−Bami; (ii) ~30% slot bất đồng là gold bị tranh chấp, chia gần đều cho hai bên (Restaurant 59 cafe_only vs 48 bami_only; ViSFD 97 vs 83), nên **khoảng cách Restaurant/ViSFD sống sót qua adjudication**; (iii) **Topic thì không**: bỏ 40 câu tranh chấp, gap còn 0,7 — và trong gold-supported Cafe vẫn hơn (33:22) nhưng phần lớn lợi thế ban đầu là do Cafe khớp *quy ước* nhãn tốt hơn.

## 3. Bản chất tranh chấp (`blind_dispute_kinds.json`)

- Topic: gần như toàn bộ ở ranh giới **lecturer ↔ training_program** (17 câu gold=lecturer reader=training_program, 7 ngược lại; ví dụ gold gán "nội dung bài giảng khó hiểu", "cho ít đồ án thôi" là *lecturer*). Quy ước của bộ dữ liệu, không phải lỗi hiểu ngôn ngữ.
- Restaurant: 53 slot gold-present mà reader không thấy aspect, 24 slot reader thấy aspect gold không có, 30 tranh chấp polarity (chủ yếu positive↔neutral 13, neutral↔negative 12). Tập trung ở FOOD#STYLE&OPTIONS (27) và FOOD#QUALITY (20) — ranh giới "món gì/khẩu phần" vs "vị".
- ViSFD: 68 + 68 tranh chấp có/không aspect, 44 polarity (neutral→positive 21). Tập trung GENERAL 37, PERFORMANCE 36, PRICE 25.
- Chung: tranh chấp ABSA là **ranh giới aspect và ngưỡng neutral**, rải đều nhiều aspect, không gắn với model nào.

## 4. Kết luận rẽ nhánh

1. Không tìm thấy lỗi cụ thể, lặp lại, riêng của BamiBERT để nhắm bằng một cơ chế one-pass. Cả hai model mắc cùng các loại lỗi với tỷ lệ gần nhau; Cafe chỉ mắc ít hơn một chút ở khắp nơi.
2. Sau khi bỏ nhãn tranh chấp, khoảng cách thật còn: Restaurant ≈ +2,6, ViSFD ≈ +0,9, Topic ≈ +0,7 (Topic phần lớn là quy ước nhãn). Đây là headroom thực tế cho một model 100–600M muốn thắng cả hai; các delta <1 điểm trên một seed vẫn nằm trong nhiễu.
3. → **Nhánh B (backbone/pretraining)**, không phải nhánh A. Adjudication thêm bằng người thật chỉ cần thiết nếu sau này định thiết kế cơ chế quanh một nhóm câu cụ thể.

Giới hạn: một reader LLM (Sonnet 5) thay cho hai người; reader ABSA chỉ khớp gold 79–90% ngay cả ở câu dễ nên "gold-supported" là điều kiện có nhiễu; mọi số là validation seed 42; không suy ra tỷ lệ nhãn sai toàn benchmark từ mẫu chọn có điều kiện.
