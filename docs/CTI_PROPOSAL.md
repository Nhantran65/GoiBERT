# Ứng viên vòng 5: bất biến token-state dưới ghép câu (CTI — Compositional Token-state Invariance)

Ngày 2026-09-14. Root đề xuất sau chẩn đoán [SCALE_PERSISTENT_LABEL_SET_BIAS.md](SCALE_PERSISTENT_LABEL_SET_BIAS.md) và probe D2. Chưa training; chờ E19 làm baseline.

## 1. Chuỗi bằng chứng dẫn tới ứng viên

1. Ở cả 26M và 110M, cặp nhãn hiếm bị bỏ sót 33–47% và bị **thay bằng bạn đồng hành quen** (37/42 spurious ở Hotel BamiBERT có tần suất cặp cao hơn nhãn bị bỏ); câu trái polarity sai polarity 15–18%.
2. **D2** (`scripts/research_checks/mil_max_probe.py`, `artifacts/research_diagnostics/mil_max_probe/results_bami_e14_bami.json`): trên BamiBERT E14 đóng băng, đổi cách đọc (CLS probe / mean / **MIL-max theo token**) không đổi lỗi cặp hiếm (missed 43,4 / 47,3 / 42,6% so 46,5% head gốc) và chỉ giảm nhẹ wrong polarity nhóm trái polarity (14,9 → 11,7). ⇒ bias nằm **trong token states** của encoder đã fine-tune, không ở pooling/head.
3. fork_test (vòng trước): với BamiBERT, CLS chỉ đọc đoạn A vẫn báo aspect của B có mặt 30,5% ⇒ token states của A **bị nhiễm** nội dung B. Với student thì không (6,5%) — student hỏng ở CLS, đã đóng.
4. Trần đọc-theo-đoạn của BamiBERT thấp (phần đoạn 1,2) và head ổn định dưới nhiễu ⇒ không phải lỗi đọc.

Giả thuyết cơ chế: khi fine-tune trên câu đa aspect, encoder 110M nhúng **ngữ cảnh tập nhãn** vào token của từng đoạn (token của "phòng" mang thông tin "câu này cũng nói về nhân viên"); head học dùng ngữ cảnh đó như prior đồng xuất hiện. Gặp tổ hợp hiếm, token states mang ngữ cảnh "lạ" → kéo về tổ hợp quen (thay thế) hoặc bỏ sót; polarity của đoạn này nhiễm polarity đoạn kia.

## 2. Phương pháp

Với hai câu train đơn-aspect A, B (aspect khác nhau — ghép có "hợp đồng không tương tác" đúng theo cấu trúc), tạo composite x = A ⊕ B (nhãn hợp). Trong cùng bước huấn luyện chạy ba forward: x, A, B. Loss:

L = CE(x, G_A ∪ G_B) + CE(A) + CE(B) + μ · [ mean_{t∈A} ‖h_t(x) − sg(h_t(A))‖² + mean_{t∈B} ‖h_t(x) − sg(h_t(B))‖² ] / D

với h_t là token state lớp cuối (có thể thêm 1–2 lớp giữa), sg = stop-gradient trên standalone, μ chọn 0,1–1 (chuẩn hóa theo D). Câu tự nhiên giữ CE (+ NC như E19). Inference không đổi.

Tính chất: (i) ràng buộc chỉ áp ở nơi "không tương tác" đúng theo xây dựng (hai câu độc lập), nên không ép bất biến ở chỗ ngữ cảnh thật sự cần (phủ định xuyên vế, tham chiếu) — khác các invariance objective áp lên mọi input; (ii) đo được trực tiếp bằng fork_test: B_present khi CLS chỉ đọc A phải giảm; (iii) không tham số mới, không đổi inference.

Dự đoán có thể bác bỏ (Hotel, E19 làm nền): (a) nhiễm fork giảm rõ (30% → <15%); (b) missed cặp hiếm giảm ≥ 10 điểm và wrong polarity nhóm trái polarity giảm, **hơn** arm chỉ dùng composite nhãn hợp (D1); (c) F1 tổng không giảm. Nếu (a) đúng mà (b) sai: nhiễm token không phải nguyên nhân của bias tập nhãn → ứng viên dừng, kết luận là bias ở mức phân bố huấn luyện (D1 quyết định). Nếu D1 ≈ CTI ≫ baseline: đóng góp là dữ liệu (tiền lệ C3DA/Chai), không claim phương pháp.

## 3. Đối chiếu tính mới (sơ bộ)

| Tiền lệ | Phần phải ghi nhận | Khác biệt |
|---|---|---|
| C3DA (2022), LWEDA (2024), Chai et al. 2023 | Sinh/ghép câu đa aspect; contrastive/union label ở **đầu ra** | CTI ràng buộc **token states** của từng đoạn bằng standalone của chính nó; có control D1 tách phần dữ liệu |
| UDA, consistency/invariance losses (ℓ2 consistency, R-Drop) | Bất biến dưới augmentation | Augmentation ở đây là ghép câu độc lập; bất biến áp theo đoạn với hợp đồng không tương tác; động cơ từ phép đo nhiễm fork |
| IIT / causal distillation | Can thiệp hidden state theo mô hình nhân quả | Không swap, không mô hình nhân quả; target là trạng thái standalone |
| Label co-occurrence debiasing MLTC (2023–2025, counterfactual/TDE) | Bias tập nhãn ở classifier | D2 cho thấy bias nằm trong representation; CTI tác động ở encoder, không ở classifier |
| ARTS AddDiff, MAMS | Đo robustness khi thêm câu aspect khác | Dùng làm phép đo, không phải phương pháp |

Chưa tìm thấy nguồn trùng cho "token-state invariance theo đoạn dưới ghép câu độc lập" trong ABSA; cần audit thêm trước claim.

## 4. Screening đề xuất (E20, sau E19)

Nền: E19 (DAPT + NC + LLRD), Hotel trước, seed 42, 30 epochs, validation-only.

| Arm | Mô tả |
|---|---|
| e19 | baseline (đã chạy) |
| d1_composite | + composite nhãn hợp, tỉ lệ 1 composite / 3 câu tự nhiên, ưu tiên cặp hiếm (sampling phẳng theo cặp) |
| cti | như d1 + L_inv (μ=0,3), standalone forward có stop-gradient |

Đo: pair F1; missed cặp hiếm / wrong trái polarity (bảng nhóm); fork_test B_present (nhiễm); composite battery. Gate: cti hơn d1 ≥ 5 điểm missed cặp hiếm và không giảm F1 > 0,3; nhiễm fork giảm.
