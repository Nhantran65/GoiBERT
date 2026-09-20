# E19 — Kết quả: goibert-b-dapt (103M) + NC + LLRD, 5 task, seed 42, validation

Protocol `E19-dapt-nc-llrd`; plan [E19_PLAN.md](E19_PLAN.md); runner `scripts/run_e19.py`; artifacts `artifacts/e19/<task>/seed_42/`. Chạy 2026-09-13 17:11–18:30 UTC, 4 GPU song song sau khi người dùng yêu cầu. Không test, không grid, không multi-seed.

| Task | E19 (best epoch) | E2 teacher CE (cùng encoder) | Δ | BamiBERT gốc E14 CE | Δ |
|---|---:|---:|---:|---:|---:|
| VSFC Sentiment (macro F1) | 85,71 (ep 4) | 85,92 | −0,21 | 86,33 | −0,62 |
| VSFC Topic (macro F1) | 81,77 (ep 8) | 82,17 | −0,40 | 79,53 | +2,24 |
| ViSFD (pair F1) | **85,08** (ep 26) | 84,36 | **+0,72** | 84,38 | +0,70 |
| Restaurant (pair F1) | 74,51 (ep 22) | 74,62 | −0,11 | 74,74 | −0,23 |
| Hotel (pair F1) | **73,63** (ep 29) | 73,22 | +0,41 | 73,24 | +0,39 |

**Gate (≥3/5 task tăng ≥0,3, không task giảm >0,3): không đạt** — chỉ ViSFD và Hotel tăng, Topic giảm 0,4. Recipe NC + LLRD (đã cho +1–2,5 điểm trên student 26M) **không chuyển thành gain rộng ở 103M**: model lớn ít cần regularization/bảo vệ lớp dưới hơn. Chênh lệch ±0,4 nằm trong nhiễu seed (≈0,5–1 điểm cho recipe này).

Ý nghĩa: (1) E19 là baseline "model lớn + recipe tốt nhất" hợp lệ cho các so sánh sau (E20 dùng nó làm nền); (2) không có "gain miễn phí" từ recipe để vượt paper — muốn vượt Restaurant (paper test 74,89) cần đúng phần lỗi cấu trúc còn lại (cặp nhãn hiếm, trái polarity), tức phụ thuộc E20; (3) DAPT vẫn là đòn bẩy lớn nhất đã thấy ở model lớn (Topic +2,2 so BamiBERT gốc).

Không multi-seed/test cho E19 vì gate không đạt; nếu sau này cần số test cho baseline, chạy seeds 43–44 trước.
