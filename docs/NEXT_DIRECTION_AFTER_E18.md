# Hướng tiếp theo sau E18

Ngày 2026-09-13. Root tổng hợp sau khi đóng hướng readout/segment.

## Điều đã chắc sau E1–E18

1. Mọi đòn bẩy lúc fine-tune trên student 26M (loss KD/NC, LLRD, pooling, FFN rộng, MLM ngắn, label-attention, span/conv, signed attention, pair-value, FFN gate, cavity keys, fork-consistency) cho ≤ ±1 điểm, bằng nhiễu seed (b0 Restaurant 72,26 so E8 71,28 cùng recipe). Chỉ LLRD (+1,3–2,5 ABSA) và NC (+1 ViSFD) là tín hiệu thật, và đã nằm trong E8.
2. Trần đọc-theo-đoạn có đối chứng và E18 cùng nói: phần lỗi composition sửa được ở lớp đọc chỉ ~+1–2 F1 oracle, và làm đoạn đọc được cũng không đổi F1. Nút thắt còn lại nằm ở **backbone** (biểu diễn token-level, pretraining): D2 cho thấy teacher tăng nhiều hơn khi fine-tune; E2 teacher (goibert-b-dapt, chỉ CE) đạt val Restaurant 74,62 / Hotel 73,22 / ViSFD 84,36 / Topic 82,17 so student E8 71,3 / 73,0 / 82,4 / 80,4.
3. BamiBERT gốc chạy local (E14, test) đã vượt paper ở Hotel (73,94 vs 72,65) và ViSFD (84,50 vs 84,24), gần ở Sentiment, thua ở Restaurant (74,15 vs 74,89) và Topic (78,18 vs 79,90). DAPT (goibert-b-dapt) cải thiện Topic +2,6 điểm val so BamiBERT gốc.

## Đề xuất: hai bước xác minh nhanh, đầu tư thêm chỉ khi có tín hiệu

### Bước A (chính, ~5 h GPU với 2 GPU): DAPT-BamiBERT + recipe E8

Fine-tune `models/goibert-b-dapt` (110M, đã DAPT trên domain) với đúng recipe đã thắng trên student: NC + LLRD γ=0,8, LR theo E2 teacher đã chọn (Sent 2e-5, Topic 1e-5, ViSFD/Hotel 5e-5, Restaurant 2e-5), 30 epochs, seed 42, cả 5 task, validation-only. So sánh với E2 teacher CE (cùng checkpoint, không NC/LLRD) và E14 BamiBERT gốc.

- Kỳ vọng nếu đúng: +0,5–1 điểm val trên ABSA so E2 teacher; khi đó là ứng viên thực tế đầu tiên có khả năng vượt paper ở cả 5 task (Restaurant cần +0,7 trên test, Topic đã có DAPT bù).
- Gate: ≥3/5 task tăng ≥0,3 so E2 teacher, không task giảm >0,3. Đạt → chạy seeds 43–44 rồi mới mở test một lần cho checkpoint chọn bằng validation.
- Cần: một runner mới tái dùng E2 teacher loop + E7 optimizer (num_layers=12); không sửa E1–E18. Đây là công việc kỹ thuật thuần, giao subagent.

Lưu ý trung thực: bước này không phải phát minh; đóng góp mới của project sẽ là chuỗi chẩn đoán (locus khác nhau student/teacher, trần có đối chứng, E18 tách cơ chế khỏi quality). Nếu mục tiêu công bố cần novelty phương pháp, phải chấp nhận đây là "analysis paper + strong baseline", hoặc chuyển sang bước C.

### Bước B (phụ, ~30 phút, chạy song song): ensemble đọc-nhiễu lúc decode

Oracle random-mask (+6 điểm) cho thấy quyết định của head không ổn định dưới nhiễu đầu vào. Kiểm nhanh trên checkpoint E8/E18 có sẵn: trung bình logits của lần đọc thường + K lần đọc với mask ngẫu nhiên 10–20% token (hoặc MC-dropout), K=8. Nếu +0,5 val ổn định trên cả Restaurant/Hotel: một gain rẻ, cộng dồn được cho mọi model (kể cả bước A). Không novelty; chỉ kiểm có/không.

### Bước C (chỉ khi A không đủ hoặc cần novelty): backbone student mới

Student hiện được distill 30k bước bằng hidden-MSE + **attention trung bình theo head**, chưa từng MLM, corpus cũ đã mất. Nếu vẫn muốn một encoder nhỏ thắng BamiBERT: distill lại student rộng hơn (ví dụ 12L×512 hoặc 6L×768) từ goibert-b-dapt với distillation theo head/relation (MiniLM-style) + MLM trên corpus dựng lại (CC100-vi + text task), rồi E8 recipe. Đây là dự án nhiều ngày GPU, kết quả không chắc, và novelty cũng chỉ ở mức "recipe cho tiếng Việt". Không bắt đầu trước khi có kết quả bước A.

## Không làm

- Không thêm biến thể reader/segment/attention operator trên student.
- Không multi-seed hoặc test cho E18.
- Không mở test cho bước A trước khi qua gate validation và multi-seed.

## Bổ sung 2026-09-14 — bước D (sau khi có E19): hai phép kiểm nhanh nhắm lỗi tổ hợp nhãn

Chẩn đoán mới ở [SCALE_PERSISTENT_LABEL_SET_BIAS.md](SCALE_PERSISTENT_LABEL_SET_BIAS.md): ở cả 26M và 110M, cặp nhãn hiếm bị bỏ sót 33–47% và câu trái polarity sai polarity 15–18%; lỗi cặp hiếm là thay thế bằng bạn đồng hành quen. Đây là đích cho ý tưởng công bố. Hai phép kiểm, mỗi cái ≤ 2–3 h GPU, trên nền E19 (DAPT + NC + LLRD), Hotel trước (42 câu cặp hiếm):

- **D1 (lớp dữ liệu)**: E19 + composite augmentation ưu tiên cặp hiếm (ghép hai câu train đơn-aspect sao cho phân bố cặp phẳng hơn; nhãn hợp). Đo missed cặp hiếm, wrong polarity nhóm trái polarity, F1 tổng. Nếu cặp hiếm giảm mạnh mà F1 không giảm → cơ chế sửa được bằng dữ liệu (có tiền lệ C3DA/Chai 2023); đóng góp công bố phải là benchmark + phân tích, không phải phương pháp.
- **D2 (lớp đọc/head, không train encoder)**: probe MIL-max trên token states cuối của E19/E14 đóng băng (presence_a = max_t σ(w_a·h_t), polarity đọc tại token argmax), huấn luyện vài phút. Nếu MIL-max giảm missed cặp hiếm rõ so head CLS cùng encoder → bias nằm ở pooling/head và một readout theo bằng chứng có thể là phương pháp; nếu không → bias nằm trong biểu diễn, phương pháp phải ở mức representation/objective (ví dụ debias đồng xuất hiện kiểu counterfactual, đối chiếu tiền lệ MLTC 2023–2025).

Quyết định ý tưởng công bố chỉ sau D1/D2; không mở thêm nhánh nào khác trước đó.
