# Kiểm nhanh: trừ prior đồng xuất hiện nhãn ở lớp quyết định (E19, không train)

Ngày 2026-09-14. Script `scripts/research_checks/cooccurrence_debias_decode.py`; artifact `artifacts/research_diagnostics/cooccurrence_debias/e19_debias_decode.json`. Logits validation của E19 (seed 42), prior = lift naive-Bayes theo cặp từ train (Laplace), m'_a = m_a − α·lift_a(S); S = các aspect khác (oracle: gold; predicted: dự đoán lần đầu).

| Task | Setting | α | pair F1 | cặp hiếm missed | spurious/row (cặp hiếm) | cặp thường missed | trái polarity wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| Hotel | baseline | 0 | 73,58 | 48,1 | 0,74 | 19,8 | 16,9 |
| Hotel | oracle S | 0,25 | 73,69 | 45,7 | 0,79 | 20,2 | 16,9 |
| Hotel | oracle S | 1,0 | 71,19 | 45,0 | 1,24 | 21,2 | 19,5 |
| Hotel | predicted S | 1,0 | 72,11 | 44,2 | 1,14 | 20,9 | 18,2 |
| Restaurant | baseline | 0 | 74,60 | 45,8 | 0,44 | 15,6 | 19,4 |
| Restaurant | oracle S | 1,0 | 68,76 | 29,2 | 1,89 | 16,1 | 19,4 |
| Restaurant | predicted S | 0,25 | 74,69 | 45,8 | 0,44 | 15,2 | 19,8 |

Sign check trên câu cặp hiếm (baseline): Hotel aspect bị bỏ sót có lift âm 40/62, spurious có lift dương chỉ 12/31; Restaurant 11/11 và 0/4.

**Kết luận: bác bỏ giả thuyết "bias tập nhãn nằm ở prior đồng xuất hiện tại lớp quyết định".** Ngay cả khi biết trước các aspect khác (oracle), trừ prior không cứu được cặp hiếm ở Hotel (48 → 45%), và ở Restaurant chỉ cứu khi α lớn đến mức phá F1 (−6 điểm). Margin của các aspect bị bỏ sót nằm rất sâu dưới 0, không phải lệch nhẹ do prior.

Chuỗi đã đóng: đọc theo đoạn (E18), cách đọc token (D2), nhiễm token (E20), prior quyết định (đây) — không cái nào đổi lỗi cặp hiếm/trái polarity. Phần còn lại: độ khó nội tại/nhãn mơ hồ của chính các câu đó (adjudication mẫu trước đây: 55–60% lỗi thiểu số có gold mơ hồ). Không đề xuất thêm phương pháp encoder cho nhóm lỗi này.
