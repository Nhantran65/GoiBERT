# Plan: distill CafeBERT → GoiTurn-S bằng nhãn mềm ngoài fold (pilot Restaurant + ViSFD)

Ngày chốt: 2026-09-25. Người dùng duyệt pilot 2 task và cho dùng song song **GPU 0, 1, 2**; không dùng GPU 3. Agent chính tự triển khai.

## Mục tiêu và bối cảnh

Mục tiêu là một model **nhỏ hơn BamiBERT (103M) mà không thua** Bami.

Ứng viên là **GoiTurn-S** (`models/goiturn-s-gd`, 26,4M, distill từ Bami-DAPT). Ở recipe E7 (LLRD γ = 0,8), validation seed 42 cho thấy GoiTurn-S đã ngang Bami ở Topic và Hotel, nhưng thua 2–3 điểm ở ba task:

| Task | GoiTurn-S | Bami | Chênh |
|---|---:|---:|---:|
| Restaurant | 71,52 | 74,74 | −3,22 |
| ViSFD | 81,49 | 84,38 | −2,89 |
| Sentiment | 84,06 | 86,33 | −2,27 |

[LABEL_CEILING_CHECK_RESULT.md](LABEL_CEILING_CHECK_RESULT.md) cho thấy nhãn gold có nhiễu đáng kể, và nhiễu seed khoảng 0,4–1,4 điểm.

**Giả thuyết:** cho student học thêm nhãn mềm từ teacher mạnh hơn Bami (CafeBERT) sẽ thu hẹp khoảng cách này. Nhãn mềm phải là **dự đoán ngoài fold (OOF)** trên tập train, để teacher không đơn giản lặp lại nhãn đã thuộc; nhờ vậy cũng giảm tác động của nhãn train sai.

Khác các lần KD trước (CDBK, rep-distill): teacher không còn cho xác suất gần one-hot trên dữ liệu đã thấy, và student là model nhỏ đã distill trong miền.

## Thiết kế

1. **Teacher 5-fold (GPU).**
   - Chia train mỗi task thành 5 fold cố định: hoán vị `default_rng(42)`, fold = thứ hạng mod 5.
   - Với mỗi fold k, train CafeBERT trên 4 fold còn lại theo đúng recipe `sealion_benchmark`: LR 2e-5, 30 epoch, batch 32, max length 256, bf16, seed 42. Chọn best epoch trên validation như mọi run trong dự án.
   - Dự đoán logits cho fold k. Tổng cộng 10 run.
2. **Soft labels.** Ghép OOF thành logits cho **mọi** câu train, kiểm phủ đủ và không trùng. Báo:
   - F1 của teacher OOF so với gold train;
   - tỷ lệ slot teacher bất đồng với gold;
   - entropy trung bình, để xác nhận nhãn không one-hot.
3. **Student (GPU).** Vòng train mới bám sát E1/E7: `build_classifier(goiturn-s-gd)`, optimizer LLRD γ = 0,8 của E7, LR 1e-4, 30 epoch, micro/effective batch 32, bf16, warmup 10%, clip 1, max length theo `inspect_train_lengths`.
   - **CE:** chỉ học gold (đối chứng).
   - **KD:** loss = 0,5 · CE(gold) + 0,5 · T² · KL(teacher_T ‖ student_T) cho từng slot, T = 2. α và T chốt trước, không quét.
   - Seed 42, 43, 44 cho mỗi nhánh × 2 task, tổng 12 run.
   - Kiểm tái tạo: CE seed 42 nên gần E7 (71,52 / 81,49).
4. **Mốc Bami:** chạy thêm BamiBERT seed 43/44 theo recipe E14, ở LR đã chọn 5e-5 cho cả 2 task (4 run), để so bằng trung bình 3 seed.

## Gate (chốt trước khi đo)

Áp cho từng task, trên trung bình 3 seed validation:

- **PASS** nếu cả hai điều kiện đúng:
  - `mean(KD) − mean(CE) ≥ 1,0`;
  - `mean(KD)` thu hẹp ít nhất **một nửa** khoảng cách `mean(Bami) − mean(CE)`.
- Pilot **PASS** nếu cả Restaurant và ViSFD PASS. Khi đó đề xuất mở rộng sang Sentiment, Topic, Hotel.
- Một task PASS: báo cáo và để người dùng quyết.
- Không task nào PASS: đóng hướng.

Báo thêm "không thua Bami" (`mean(KD) ≥ mean(Bami) − 0,5`) như thông tin, không phải gate.

## Ràng buộc

- Validation-only, `include_test=False`; không mở test.
- GPU 0/1/2, mỗi GPU một job của mình:
  - job Cafe cần ≥ 20 GB trống, job student/Bami cần ≥ 10 GB;
  - không dùng GPU 3; không kill process của người khác.
- Không sửa script hay artifact cũ. Code mới trong `scripts/cafe_oof_kd/`, artifacts trong `artifacts/cafe_oof_kd/`.
- Ngân sách: trần **12 GPU-giờ**. Ước tính khoảng 3,5 cho teacher, 3 cho student, 1 cho Bami.

## Báo cáo

`docs/CAFE_OOF_KD_STUDENT_RESULT.md`, gồm:

- thống kê teacher OOF;
- bảng CE / KD / Bami theo seed, mean và range;
- kiểm tái tạo E7;
- quyết định gate;
- GPU-giờ;
- giới hạn: validation-only; teacher chọn epoch trên validation; một cấu hình α/T.
