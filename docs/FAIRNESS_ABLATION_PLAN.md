# Plan: ablation "phương pháp" cho paper — KD, baseline Bami cùng recipe, OOF so với in-sample

Chốt ngày 2026-09-30. Người dùng duyệt và cho phép Codex (gpt-6-sol, reasoning high) chạy ngoài sandbox trên GPU. Root lập plan, review và tổng kết.

## 1. Mục đích

Paper được định vị là **phương pháp distill hai teacher**:

- **Teacher biểu diễn:** Bami. Dùng cho GD mức token, vì cùng tokenizer với student.
- **Teacher dự đoán:** CafeBERT. Dùng cho KD mức đầu ra, qua nhãn mềm out-of-fold. Cách này không phụ thuộc tokenizer.

Cần bằng chứng cho ba câu hỏi:

1. **#1** KD từ teacher dự đoán đóng góp bao nhiêu? So S′ và M có KD với không KD.
2. **#2** Ở cùng recipe fine-tune, Bami đứng ở đâu trên đường cong kích thước–chất lượng?
3. **#3** Nhãn mềm out-of-fold có đáng hơn in-sample không?

Riêng #2b (Bami + LLRD, không KD) root thêm vào để khép kín ma trận recipe. Không có #2b thì không tách được hiệu ứng của LLRD với hiệu ứng của KD trên Bami.

## 2. Ma trận recipe

Ký hiệu: E14 = AdamW thường (recipe gốc của Bami); E7 = LLRD γ 0,8; KD = nhãn mềm CafeBERT out-of-fold, α 0,5, T 2.

| Backbone | E14, CE | E7, CE | E7 + KD |
|---|---|---|---|
| Bami 103M | **có** (baseline hiện tại) | **#2b mới** | **#2 mới** |
| M 50M | — | **#1 mới** | có |
| S′ 26M | — | **#1 mới** | có |

## 3. Thiết kế

**Chung cho mọi arm mới:**

- 8 task, seed 42/43/44.
- Chọn epoch và LR trên validation.
- 30 epoch, batch 32, bf16.
- Head và loader giống các run đã có của từng backbone.

**#1: S′ và M không KD** (E7, CE).

- Dùng đúng LR đã khóa của mỗi model ở arm có KD:
  - **S′:** Sentiment 1e-5, Topic 1e-5, ViSFD 1e-4, Restaurant 1e-4, Hotel 1e-4, ViSpam 5e-5, ViNLI 1e-4, PhoNER 1e-4.
  - **M:** theo `artifacts/family_sm/decisions.json::m_lr`.
- Chỉ bỏ số hạng KD, giữ α = 1 cho CE. Như vậy đây là ablation một yếu tố.

**#2: Bami E7 + KD** và **#2b: Bami E7, CE.**

- Head Bami E14 (`build_bamibert_classifier`); optimizer E7 với 12 lớp.
- LR quét 2 giá trị trên seed 42, cho từng task và từng arm:
  - LR E14 tốt nhất đã có của task đó;
  - và 2× giá trị đó, lấy giá trị gần nhất trong {1e-5, 2e-5, 5e-5, 1e-4}. LLRD giảm LR ở các lớp dưới, nên thường cần LR gốc cao hơn.
- Chọn LR theo validation, rồi chạy seed 43 và 44.

**#3: nhãn mềm OOF so với in-sample** (S′, E7 + KD).

- **Task:** Sentiment, Restaurant, ViNLI (một task phân loại câu, một ABSA, một cặp câu).
- **Teacher in-sample:** CafeBERT fine-tune trên **toàn bộ train**, cùng recipe với các teacher fold (LR 2e-5, batch 32, 30 epoch, warmup 10%, chọn epoch trên validation), seed 42. Sinh nhãn mềm trên chính tập train đó.
- **So sánh:** S′ + KD với nhãn in-sample, 3 seed, đối chiếu S′ + KD với nhãn OOF (đã có).
- **Đo thêm:** độ "tự tin" của nhãn mềm (entropy trung bình, tỷ lệ max-prob > 0,99) cho cả in-sample và OOF.
- **Chỉ dùng validation;** không cần test.

## 4. Test

1. Sau khi validation của #1, #2 và #2b đã xong, **khóa một manifest chung** cho 96 checkpoint mới: S′-CE, M-CE, Bami-E7-CE, Bami-E7-KD, mỗi model 8 task × 3 seed.
2. Test **đúng một lần**, theo protocol `scripts/final_test/`.
3. **Bảng cuối gồm 8 dòng:**
   - Bami E14 (có sẵn), Bami E7-CE, Bami E7-KD;
   - S′-CE, S′-KD (có sẵn);
   - M-CE, M-KD (có sẵn);
   - CafeBERT teacher: mean của 5 fold trên validation, chỉ để tham khảo; không test.
4. **Paired bootstrap** (10.000 mẫu, lấy mẫu lại theo câu; trung bình 3 seed trong mỗi mẫu) trên predictions test, cho các cặp:
   - M-KD so với Bami-E14 và so với Bami-E7-KD;
   - S′-KD so với Bami-E14;
   - M-KD so với M-CE; S′-KD so với S′-CE.

   Báo khoảng tin cậy 95% và p-value cho từng chỉ số.
5. Báo đủ **15 chỉ số, gồm ViNLI**.

## 5. Ngân sách và vận hành

- **Ước tính:**
  - #1: khoảng 10 GPU-giờ;
  - #2 và #2b: khoảng 32 GPU-giờ;
  - #3: khoảng 10 GPU-giờ;
  - test và bootstrap: khoảng 2 GPU-giờ.
- **Trần: 65 GPU-giờ.**
- **GPU:** 0–3, mỗi GPU một job của mình, launch khi còn đủ VRAM trống. Không kill, dừng hay renice process của người khác.
- **Launcher:** chạy tách phiên, có resume.
- **Checkpoint:**
  - chỉ giữ checkpoint của các run có thể vào test (các arm #1, #2, #2b ở LR đã chọn);
  - run chọn LR bị thua và toàn bộ run của #3: xóa checkpoint sau khi postrun check PASS;
  - riêng teacher in-sample của #3: giữ tới khi đã sinh xong nhãn mềm, rồi xóa.

## 6. Giới hạn

- LR của #1 không quét lại. Mục đích là giữ ablation đúng một yếu tố.
- LR của Bami E7 chỉ quét 2 giá trị.
- #3 chỉ chạy 3 task, trên S′.
- Test đã được dùng ở hai lượt trước. Mỗi model mới vẫn được test đúng một lần, và paper sẽ khai báo điều này.
