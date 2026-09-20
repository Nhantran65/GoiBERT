# E10 — Residual span composition trong student encoder

## Phạm vi đã chốt

Theo hướng nghiên cứu trong [báo cáo](ENCODER_ARCHITECTURE_RESEARCH.md), thử hai cấu hình trên đủ 5 task, seed42, 30 epochs: span có lựa chọn theo nội dung và convolution đa kernel làm đối chứng. Tổng cộng 10 training trials, không grid hoặc thêm seed. E10 ở đây là thí nghiệm kiến trúc, thay thế đề xuất KD chưa triển khai trước đó.

Đúng một subagent `gpt-5.6-luna`, `xhigh` làm coding/training/evaluation/kiểm tra. Root chốt thiết kế, review và nhận định. Dùng GPU0–3 theo tài nguyên khả dụng, một worker/GPU trước; không chiếm hay dừng tiến trình khác. Agent tiếp tục chạy sau preflight và root review, không đợi thêm xác nhận của người dùng.

## Thiết kế và đối chứng

- Cả hai bắt đầu từ `models/goiturn-s-gd` nguyên bản, giữ head CLS E1 và mọi transformer weights. Chèn residual sau block2 (index1), trước block3. Tất cả encoder trainable.
- Đơn vị chính tả lấy từ raw input theo khoảng trắng/punctuation; giữ dấu và hoa/thường. Punctuation ngăn span vượt qua. Alignment bằng overlap ký tự với offsets tokenizer thật; chỉ phần visible sau truncation. Special/pad không vào nhánh.
- Tính lại alignment cho noisy text trong NC; không decode rồi đoán lại raw input, không dùng raw clean alignment cho noisy tokens. Có thể cache tokenization+alignment theo chính chuỗi đầu vào; không dùng gold để dựng span.
- Pool hidden về đơn vị bằng overlap chuẩn hóa. Sinh span dài1–4 không vượt separator. Hợp thành có thứ tự bằng `z[a,w]=GELU(sum_t W[w,t] U[a+t]+b[w])`.
- **span:** r128. Score cho span chứa s: `(Wq U[s]) dot z[a,w] / sqrt(r) + b_position[w,s-a]`. Softmax FP32 giữa mọi span chứa s và null; null score là scalar học được khởi tạo0, null value0. Tạo weighted message rồi distribute overlap-normalized về tokens.
- **conv:** ngân hàng convolution valid kernel1–4 tương đương các W[w,t] nêu trên, mask cùng separator. Trung bình đồng đều mọi output kernel bao phủ s; không query/content routing/null. Chọn r140 để số tham số gần r128+query của span; ghi số thực và chênh lệch, không thêm tham số giả chỉ để match.
- Output projection về384 khởi tạo zero cả weight/bias; các ma trận bên trong nonzero. Giữ đường hidden gốc. Head khởi tạo giống E8 với cùng seed bằng bảo toàn RNG khi thêm block mới; không làm đổi RNG stream khởi tạo head cũ.
- Branch tham số được LR của block2: baseLR * .8**8, cùng quy tắc decay/no-decay E8. Không gán branch vào head LR. Không đổi LR trong lúc thấy kết quả.

## Recipe cố định

Reuse frozen E2 NC loop và E7 LLRD bằng adapter isolated; không sửa E1–E9. E8: seed42, maxlen256, batch32 (micro32), BF16 SDPA, warmup10%, clip1, AdamW wd.01, gamma.8. LR Sent/Topic1e-5, baABSA1e-4; lambda NC .1/.5/.5/.1/.1, actualT1. Giữ metrics/labels: VSFC fixed-label macroF1; ABSA pair microF1, ViSFD packed42logits với OTHERS2classes.

## Gate review và kiểm tra tối thiểu

Agent gửi root đường dẫn code và báo preflight trước full training. Kiểm tra identity init cả logits với baseline same seed; all5 shapes; unicode offsets, overlap, punctuation, empty, truncation; clean/noisy alignment; optimizer allparams đúng một lần; gradient Wout ở bước1 và inner ở bước2; checkpoint strict roundtrip bằng constructor mới. Audit nhỏ từ predictions E8 về fragmentation/rare-span nếu khả thi, không dựng cả dự án diagnostic hoặc đợi annotation thủ công để khởi chạy.

Profile một bước train thực batch32 ở task dài để xác định VRAM guard, kiểm tra tài nguyên mỗi lần dispatch. Controller detached có state/log/PID, không rerun active/completed trial khi gián đoạn. Freeze code hashes trước full runs, ghi dependency/provenance riêng. Bất kỳ lỗi report phải sửa bằng adapter riêng, không làm mất provenance source train.

## Evaluation và tiêu chí

Mỗi trial chọn best validation epoch. Recompute metrics từ saved predictions, kiểm tra IDs và gold. Đọc baseline từ E8 metric files theo thang0–1; display100×F1, delta_pp=100×(new-old). Báo cả5 task, candidate−E8 và candidate−conv, bestepoch, thời gian, VRAM, số tham số.

Screening signal: ≥4/5 task tăng vsE8, ≥2 gain≥.3pp, không giảm quá.3pp. Đồng thời span cần thắng conv ở≥3/5 và không thua conv quá.3pp ở task nào để ưu tiên cơ chế routing. Đây là heuristic chưa phải significance. Nếu không đạt, dừng sau validation và báo root; không mở grid/seed/test để cứu kết quả.

Nếu đạt gate, root review khóa10 checkpoint và cho cùng agent thực hiện test cả2cấu hình trên5task trong phạm vi evaluation đã cho phép, không cần hỏi lại người dùng. Test không chọn cấu hình hoặc epoch. Chưa tự chạy BamiBERT mới/multi-seed trong vòng này; paper comparison chỉ có caveat protocol, không coi val là test.
