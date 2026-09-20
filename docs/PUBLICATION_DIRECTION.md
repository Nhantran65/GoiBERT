# Hướng công bố sau E1–E20 (bản nháp 2026-09-14, chờ multi-seed d1)

## Bằng chứng đã có

1. Student 26M: mọi can thiệp lúc fine-tune (E2–E18, 12 nhánh kiến trúc/loss) ≤ ±1 điểm ≈ nhiễu seed; chỉ LLRD/NC là tín hiệu thật.
2. Chẩn đoán có đối chứng: locus lỗi composition khác nhau giữa 26M (CLS trộn) và 110M (token nhiễm); trần đọc-theo-đoạn với đối chứng random-mask chỉ +1–2 F1 ở cả hai quy mô; composite battery phóng đại headroom.
3. Ba cơ chế được "sửa" thành công mà quality không đổi: đọc theo đoạn (E18: gap đoạn ×2,7), nhiễm token (E20: 16,5 → 2,0%), prior đồng xuất hiện ở quyết định (oracle) — lỗi cặp hiếm/trái polarity giữ nguyên ở 110M.
4. Lỗi cấu trúc tồn tại ở mọi quy mô: cặp nhãn hiếm missed 33–48%, trái polarity wrong 15–19%; adjudication trước: 55–60% lỗi thiểu số có gold mơ hồ.
5. Baseline thực dụng: DAPT (+2,6 Topic ở 110M) và ghép câu nhãn hợp (Restaurant +0,95 một seed; chờ 3 seed).

## Ba lựa chọn

**A. Bài phân tích + benchmark chẩn đoán (khuyến nghị).** "Mechanisms are not the bottleneck: a controlled diagnostic study of multi-aspect errors in Vietnamese ABSA encoders." Đóng góp: (i) bộ chẩn đoán có đối chứng (fork-restricted reads, ceiling với selection-bias control, fixture can thiệp, composite battery, phân tầng cặp hiếm/trái polarity); (ii) phát hiện locus khác nhau theo quy mô; (iii) ba can thiệp cơ chế bị bác bỏ có đối chứng; (iv) lỗi còn lại gắn với thống kê nhãn/ontology, có adjudication; (v) baseline mạnh + số test. Cần thêm: chạy bộ chẩn đoán trên 2–3 encoder Việt khác (PhoBERT, XLM-R, ViDeBERTa) để không chỉ là BamiBERT; adjudication 100 lỗi có 2 người; test cho baseline. Rủi ro: venue thích "phương pháp"; phù hợp Findings/BlackboxNLP/*SEM/VLSP.

**B. Phương pháp ở lớp dữ liệu/ontology.** Ghép câu ưu tiên cặp hiếm (nếu multi-seed xác nhận) + huấn luyện có nhận biết ontology (nhãn kề nhau ROOMS#X / ROOM_AMENITIES#X). Tiền lệ mạnh (C3DA, Chai 2023, hierarchical label smoothing); novelty thấp; chỉ nên là một phần của A.

**C. Phương pháp ở giai đoạn pretraining/distillation.** Đòn bẩy lớn nhất đã thấy là DAPT; student được distill bằng attention trung bình head. Một recipe distill/pretrain mới cho tiếng Việt (ví dụ MiniLM-style + MLM domain) có thể đưa student nhỏ lên gần 110M — nhiều ngày GPU, novelty ở mức recipe/ngôn ngữ; SCAPT/SKEP/SentiLARE đã chiếm "sentiment-aware pretraining". Chỉ làm khi cần một "model mới" cho paper.

## Đề nghị

Chọn A làm khung bài; đưa d1 (nếu xác nhận) và DAPT vào phần baseline/practical findings; giữ E18/E20 như hai can thiệp có đối chứng. Việc cần làm tiếp theo (mỗi việc ≤ 1 ngày): (1) chạy bộ chẩn đoán trên PhoBERT-base và XLM-R-base cho Restaurant/Hotel; (2) adjudication 100 lỗi cặp hiếm/trái polarity; (3) test một lần cho baseline được chọn bằng validation (E19 hoặc d1) với 3 seed.
