# Lỗi tổ hợp nhãn tồn tại ở cả 26M và 110M: chẩn đoán sau E18

Ngày 2026-09-14. Root phân tích trên checkpoint có sẵn (E8 student, E14 BamiBERT gốc LR 5e-5, E2 teacher goibert-b-dapt CE), validation-only, không training mới. Scripts tạm: scratchpad `err_decomp.py`, `substitution.py`, `ceiling_bami.py`; artifacts trần BamiBERT: `artifacts/research_diagnostics/competitive_slots/bami_e14/`.

## 1. Trần đọc-theo-đoạn của BamiBERT (E14), có đối chứng random-mask

| Task | Baseline | Oracle split liên tục | Oracle random-mask | Phần đoạn | Bộ chọn naive |
|---|---:|---:|---:|---:|---:|
| Restaurant | 74,72 | 76,61 | 75,35 | 1,26 | 74,48 |
| Hotel | 73,29 | 75,41 | 74,18 | 1,23 | 72,88 |

So với student E8: oracle random-mask của student +6,4/+4,2 điểm, của BamiBERT chỉ +0,6/+0,9 — **head của student cực kỳ bất ổn khi che token ngẫu nhiên, BamiBERT thì ổn định**; bộ chọn naive của BamiBERT ≈ baseline thay vì −3 như student. Phần đoạn ~1,2 ở cả hai model: hướng đọc-theo-đoạn cũng bị trần thấp ở 110M.

## 2. Phân rã lỗi validation theo cấu trúc nhãn (missed = % gold slot bị bỏ; wrong = % gold slot sai polarity)

| Nhóm | Student E8 | DAPT teacher (CE) | BamiBERT E14 |
|---|---|---|---|
| **Restaurant** k=1 | missed 12,9 / wrong 15,2 | 9,8 / 13,1 | 10,0 / 12,7 |
| k≥2 cùng polarity | 16,6 / 8,8 | 13,2 / 6,4 | 12,2 / 7,1 |
| k≥2 **trái polarity** (109 rows) | 25,3 / 14,8 | 19,4 / 17,7 | 18,6 / 16,0 |
| k≥2 **cặp hiếm** (≤5 lần train; 9 rows) | 54,2 / 4,2 | 41,7 / 16,7 | 33,3 / 16,7 |
| **Hotel** k=1 | 21,6 / 5,0 | 18,8 / 5,6 | 18,1 / 5,8 |
| k≥2 cùng polarity | 25,4 / 3,0 | 21,5 / 3,3 | 21,1 / 3,9 |
| k≥2 trái polarity (58 rows) | 40,3 / 12,3 | 30,5 / 14,9 | 30,5 / 14,9 |
| k≥2 cặp hiếm (42 rows, 129 gold) | 55,8 / 3,1 | 43,4 / 4,7 | 46,5 / 4,7 |

Model lớn thu hẹp lỗi ở nhóm thường (k=1, cùng polarity) nhưng **giữ nguyên hai nhóm lỗi cấu trúc**: cặp nhãn hiếm bị bỏ sót 33–47% (gấp 2–3 lần nhóm cặp thường) và câu trái polarity sai polarity 15–18% (gấp 2–4 lần). Hotel có 42/795 câu cặp hiếm → riêng nhóm này mất ~60/1309 gold slot (4,6% gold).

## 3. Lỗi cặp hiếm là "thay thế bằng bạn đồng hành quen"

Trong các câu cặp hiếm của Hotel, spurious ≈ 1,0/câu (BamiBERT). Với mỗi spurious c, so tần suất train của (c, aspect còn lại) với tần suất của (aspect bị bỏ sót, aspect còn lại): 37/42 (BamiBERT) và 26/30 (student) spurious có tần suất cặp cao hơn aspect bị bỏ. Ví dụ gold ROOM_AMENITIES#MISCELLANEOUS bị bỏ, dự đoán ROOM_AMENITIES#GENERAL hoặc SERVICE#GENERAL; gold ROOM_AMENITIES#CLEANLINESS → dự đoán ROOMS#CLEANLINESS. Chưa có đối chứng base-rate cho phép đo này (nhãn thường có tần suất cặp cao với mọi nhãn), nên đọc là dấu hiệu mạnh chứ chưa phải chứng minh; control tương tự cho student ở vòng trước (`frequent_atoms_control.json`) đã loại giải thích "chỉ vì nhãn hiếm".

## 4. Ý nghĩa

- Hai nhóm lỗi cấu trúc này là phần còn lại đáng kể sau khi đã dùng model 110M + DAPT; chúng không phải lỗi "đọc theo đoạn" (E18) và không phải lỗi bất ổn (BamiBERT ổn định).
- Cơ chế giả định: head/CLS học cấu trúc đồng xuất hiện của tập nhãn train (đặc trưng của aspect b vừa là bằng chứng dương cho bạn đồng hành quen vừa là bằng chứng âm cho aspect ít đi cùng), nên tổ hợp hiếm bị kéo về tổ hợp quen. Đây là "label co-occurrence bias" đã có tiền lệ trong multi-label text classification (counterfactual de-bias 2023, CoDeNet, causal cooperative game 2025), chưa thấy nghiên cứu riêng cho ABSA category×polarity tiếng Việt và chưa kết hợp với nhóm trái polarity.
- Câu hỏi quyết định cho ý tưởng công bố: cơ chế trên **sửa được ở lớp dữ liệu** (composite/pair-balanced augmentation, có tiền lệ) hay chỉ **ở lớp biểu diễn/head** (cần phương pháp mới)? Hai phép kiểm nhanh ở [NEXT_DIRECTION_AFTER_E18.md](NEXT_DIRECTION_AFTER_E18.md) bước D sẽ trả lời.
