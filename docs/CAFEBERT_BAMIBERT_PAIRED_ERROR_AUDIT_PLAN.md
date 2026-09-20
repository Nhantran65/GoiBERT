# Audit lỗi ghép cặp CafeBERT–BamiBERT trên validation

**Trạng thái:** plan bàn giao, chưa thực thi. Không training, không model inference, không GPU, không đọc test. Ngày chốt 2026-09-19.

## 1. Câu hỏi và phạm vi

Trước khi chọn một kiến trúc khác, cần biết **CafeBERT hơn BamiBERT ở những trường hợp nào** trên cùng câu và cùng nhãn. Chỉ dùng ba task validation, seed 42: UIT-ABSA Restaurant, VSFC Topic, ViSFD. Đây là ba task mà CafeBERT đang hơn BamiBERT ở local validation và bao gồm hai kiểu metric (pair micro F1 / macro F1). Hotel và VSFC Sentiment không thuộc lượt này; không suy kết luận năm task từ ba task.

Audit phải trả lời bằng hai tầng bằng chứng:

1. **Toàn validation, định lượng:** số câu/slot Cafe đúng–Bami sai, Bami đúng–Cafe sai, cả hai đúng/sai; các thay đổi missed/spurious/wrong-polarity theo aspect, polarity hoặc lớp Topic. So đúng metric đã dùng, không đổi nhãn.
2. **Mẫu cố định để root đọc:** trích văn bản nguyên bản cho các nhóm trên, kèm dự đoán hai model, nhãn và vị trí lỗi. Người đọc sau đó tự xem gold có hợp lý không và có một kiểu lỗi lặp lại đáng thiết kế cơ chế mới không.

**Subagent thực thi chỉ dựng dữ liệu, kiểm số và packet.** Root tự đọc mẫu, đánh giá nhãn, diễn giải nguyên nhân và quyết định giả thuyết/kiến trúc theo `AGENTS.md`. Không giao kết luận nghiên cứu cho subagent.

## 2. Nguồn prediction và text khóa cứng

| Task | BamiBERT gốc E14, seed 42 | CafeBERT, seed 42 | Số validation | Metric / mốc Bami → Cafe |
|---|---|---|---:|---|
| Restaurant | `artifacts/e14/trials/absa_restaurant/lr_5e-05/seed_42/validation_predictions.json` | `artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42/validation_predictions.json` | 771 | Pair micro F1: 74,7418 → 77,7197 |
| Topic | `artifacts/e14/trials/vsfc_topic/lr_1e-05/seed_42/validation_predictions.json` | `artifacts/sealion_benchmark/cafe_runs/vsfc_topic/lr_2e-05/seed_42/validation_predictions.json` | 1.583 | Fixed-label macro F1: 79,5328 → 81,7446 |
| ViSFD | `artifacts/e14/trials/visfd/lr_5e-05/seed_42/validation_predictions.json` | `artifacts/sealion_benchmark/cafe_runs/visfd/lr_2e-05/seed_42/validation_predictions.json` | 1.112 | Pair micro F1: 84,3834 → 85,4665 |

Prediction JSON có `rows` gồm `id`, `gold`, `prediction`; E14 có thể **không có** trường `epoch` ở cấp file. Text validation lấy qua loader `src/goibert/train.py::load_task_bundle(task_id, ROOT, include_test=False)` và `split_field`/`split_texts`, hoặc đọc cột text raw validation tương đương rồi kiểm order/ID. Ghi SHA256 của cả sáu predictions, file text nguồn, evaluator/source, checkpoint metadata và best epoch từ metrics/metadata tương ứng. Không dùng checkpoint để forward lại.

**Gate correctness trước mọi phân tích:** cùng tập ID, không trùng/mất; mỗi ID có gold giống hệt ở hai prediction files và loader; thứ tự đã kiểm hoặc join theo ID; 771/1.583/1.112 hàng; class/slot shape đúng. Tính lại F1 từ `rows` bằng evaluator gốc **và** phép đếm độc lập, khớp từng artifact ≤1e-12. Nếu không khớp, dừng và báo blocker; không tự sửa gold/split.

## 3. Mã nhãn và đơn vị lỗi

- Topic: `0 lecturer`, `1 training_program`, `2 facility`, `3 others`; fixed-label macro F1 tính đủ bốn lớp dù lớp ít mẫu.
- Restaurant và các slot polarity ViSFD: **`0 absent, 1 positive, 2 negative, 3 neutral`** theo `docs/E12_LABEL_MAPPING_AUDIT.md`. Đừng lấy tên Restaurant từ `src/goibert/data.py` cũ: metadata đó đã được xác nhận đặt sai tên, nhưng mã số và F1 numeric vẫn đúng.
- ViSFD `OTHERS` ở slot index 5 là **detection-only**: gold raw code 4, prediction model code 1; tính detection có slot này sau mapping phù hợp, **loại khỏi pair micro F1 và packet lỗi polarity**. Báo riêng missed/spurious OTHERS để không lẫn nó với sentiment.
- Với ABSA, đơn vị thống kê chính là **(review ID, aspect slot)**. Trên gold-present pair-scored slot: `correct` = prediction đúng mã non-absent; `miss` = prediction 0; `wrong_polarity` = prediction nonzero khác gold. Trên gold-absent slot: `spurious` = prediction nonzero. Không dùng số lượng absent-correct khổng lồ để gọi model “đúng hầu hết aspect”.
- Ghi confusion/cross-tab toàn validation: Topic theo gold class và cặp dự đoán; ABSA theo aspect + gold polarity, tách `Cafe_only`, `Bami_only`, `both_correct`, `both_wrong_same`, `both_wrong_different`. Các event `miss`, `wrong_polarity`, `spurious` phải đếm riêng cho từng model. Báo TP/FP/FN và pair F1 toàn task, per-aspect support + F1; khi support thấp ghi số lượng, không xếp hạng bằng tỷ lệ đơn lẻ. Không cộng macro F1 Topic với micro F1 ABSA thành một mean.

## 4. Packet 72 trường hợp để root đọc

Lấy mẫu **sau** khi đã tính bảng toàn validation. Với từng bucket/quota, sắp ứng viên theo khóa SHA256 của chuỗi `42|task|bucket|error_type|stable_id|aspect_index` (Topic dùng `aspect_index=-1`), rồi lấy lần lượt theo thứ tự hash; không chọn ví dụ vì câu “trông hay” hoặc vì hợp giả thuyết. Không lặp cùng một review ID trong cùng task nếu có đủ hàng; ghi shortfall nếu nhóm thiếu. Packet được chọn có điều kiện theo đúng/sai, **không là mẫu ngẫu nhiên để ước lượng tỷ lệ lỗi dân số**.

| Task | Cafe đúng, Bami sai | Bami đúng, Cafe sai | Cả hai sai | Cả hai đúng | Tổng |
|---|---:|---:|---:|---:|---:|
| Restaurant | 8 | 8 | 4 | 4 | 24 |
| ViSFD | 8 | 8 | 4 | 4 | 24 |
| Topic | 8 | 8 | 4 | 4 | 24 |

Với mỗi cột **exclusive** của Restaurant/ViSFD, ưu tiên theo ba loại sự kiện `miss / wrong_polarity / spurious` với quota `3 / 3 / 2`; thiếu loại nào thì bổ sung từ loại còn lại theo thứ tự `miss → wrong_polarity → spurious`, vẫn dùng thứ tự hash. Giới hạn tối đa hai event cùng aspect trong mỗi cột nếu đủ ứng viên, để không chỉ lấy FOOD#QUALITY/BATTERY. `both_wrong` ưu tiên gold-present; `both_correct` phải có ít nhất một gold-present pair đúng, không lấy review toàn absent. Với ABSA, đúng/sai ở bảng packet là **tại target aspect slot**, không nhất thiết đúng/sai toàn bộ review. Topic lấy theo đúng/sai ở cấp câu; cố gắng phủ cả bốn gold classes, ghi thiếu support nếu không thể. Đây là quota **packet đọc**, không reweight bảng tổng thể.

Mỗi hàng packet: task, stable ID, nguyên văn text validation, target aspect/slot nếu ABSA, gold numeric + tên đúng, prediction Bami/Cafe numeric + tên đúng, loại event, toàn bộ vector gold/pred cho ABSA, số gold-present aspects, và path/hash nguồn. Chừa cột trống `gold_supported`, `evidence_span`, `possible_failure`, `ambiguity_note` cho **root** điền sau; subagent không tự gán nguyên nhân neural, không relabel gold. Không cần logits hay explanation model; artifacts chỉ lưu hard predictions.

## 5. Deliverables và cách đọc kết quả

Viết source mới `scripts/cafe_bami_paired_error_audit/`, outputs mới `artifacts/cafe_bami_paired_error_audit/{manifest.json,aggregate.json,events.jsonl,packet.jsonl,packet.md}`, và báo cáo cơ học `docs/CAFEBERT_BAMIBERT_PAIRED_ERROR_AUDIT_PACKET.md`. Báo cáo gồm input provenance/correctness, bảng toàn validation, quota packet, dữ liệu thiếu nếu có và **giới hạn diễn giải**. Giữ nguyên source, gold, checkpoints, prediction artifacts cũ. Không dùng GPU, không inference/training, không test, không tạo subagent.

**Điểm dừng:** subagent bàn giao packet và dừng. Root đọc mẫu và tự viết kết luận nghiên cứu riêng sau đó. Một mẫu chọn theo disagreement không chứng minh tỷ lệ nguyên nhân; một lỗi ngôn ngữ trong câu không chứng minh cơ chế nội bộ gây sai. Nếu không có kiểu lỗi chung, kết luận trung thực có thể là chưa có kiến trúc một-pass nào đủ căn cứ để train. Vì hướng tiếp theo sẽ được chọn từ validation, mọi claim cải thiện sau này cần vòng xác nhận độc lập; không dùng lại cùng validation làm bằng chứng quyết định cuối cùng.
