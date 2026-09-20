# E20 — Screening CTI (bất biến token-state dưới ghép câu) trên nền E19

Người dùng duyệt 2026-09-14: tự động chạy sau khi E19 xong; tổng kết cho người dùng đọc sáng hôm sau. Ý tưởng: [CTI_PROPOSAL.md](CTI_PROPOSAL.md). Nền: E19 (goibert-b-dapt 103M + NC + LLRD), cùng kernel/loop/micro-batch. Validation-only, seed 42, 30 epochs, không test, không grid.

## Arms (× absa_hotel, absa_restaurant)

| Arm | Mô tả |
|---|---|
| e19 (baseline) | trial E19 đã chạy, không rerun |
| d1_composite | E19 + composite nhãn hợp: thêm vào tập train một tập cố định C composite (|C| ≈ ⅓ số câu train), mỗi composite = câu đơn-aspect A + " " + câu đơn-aspect B (A, B aspect khác nhau), nhãn = hợp; chọn cặp aspect **ưu tiên cặp hiếm** (xác suất chọn cặp (a,b) ∝ 1/(1 + tần suất cặp trong train)), A/B lấy ngẫu nhiên trong câu đơn-aspect ≤48 token của aspect đó. NC áp lên composite như câu thường. |
| cti | như d1_composite **+** L_inv: với mỗi composite trong micro-batch, forward thêm A và B đứng riêng (cùng model, stop-gradient), L_inv = mean_{t∈A} ‖h_t(x) − sg(h_t(A))‖²/D + mean_{t∈B} ‖h_t(x) − sg(h_t(B))‖²/D trên token state lớp cuối (chỉ token nội dung; </s> và <s> bỏ), μ = 0,3, chỉ trên forward "clean" (không trên noisy, không khi eval). Vị trí biên B trong câu ghép = 1 + len(ids(A)) (đã kiểm: tokenize(A+" "+B) = ids(A) + ids(" "+B)); kiểm lại bằng assert trên từng composite lúc dựng. |

Kỹ thuật: composite rows được nối vào train split qua patch `_task_contract` (id có tiền tố `e20comp:`), boundary/standalone ids lưu theo id; `_move_batch` patch nhận diện hàng composite trong micro-batch và stash metadata lên model; forward của cti chạy standalone forwards (no_grad) rồi tính L_inv lưu `model.last_inv_loss`; `compute_loss` patch cộng μ·L_inv rồi xóa. Trial d1 dùng cùng code với μ=0.

## Preflight tối thiểu

1. Composite: kiểm assert boundary với 200 composite; thống kê phân bố cặp (số cặp hiếm được phủ) và ghi manifest (ids, cặp, boundary).
2. cti: L_inv ≥ 0 hữu hạn, gradient vào encoder khác 0; với μ=0 loss trùng d1; forward eval không đổi so E19 (parity 1e-4 FP32 trên 32 câu validation).
3. Một bước NC + composite hữu hạn; VRAM.

## Đánh giá sau khi xong (tự động, ghi `docs/E20_RESULT.md`)

Cho e19 / d1 / cti trên mỗi task:
- pair F1 validation (best epoch) và bảng nhóm: missed/wrong theo k=1, k≥2 cùng/trái polarity, cặp thường/hiếm (logic như scratchpad `err_decomp.py`: cặp hiếm = tần suất cặp train ≤5).
- **Nhiễm fork**: theo `scripts/independent_review/fork_test.py` mục (ii): trên 200 composite validation (build_pairs seed 2026), CLS chỉ đọc phần A ở mọi layer → A_correct, B_present; và A_alone. Báo B_present cho ba arm.
- Composite battery A+B: A_correct / B_correct / both.

Gate (Hotel là chính): cti hơn d1 ≥ 5 điểm missed cặp hiếm **và** B_present (nhiễm) giảm rõ so e19, F1 tổng không giảm > 0,3. d1 ≈ cti → đóng góp là dữ liệu (có tiền lệ). Không multi-seed, không test.
