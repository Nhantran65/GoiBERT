# Kết quả: distill CafeBERT → GoiTurn-S bằng nhãn mềm ngoài fold (pilot Restaurant + ViSFD)

Ngày 2026-09-26. Theo [plan](CAFE_OOF_KD_STUDENT_PLAN.md).

- Chỉ dùng validation, không mở test.
- Chạy trên GPU 0/1/2, tổng **khoảng 4,97 GPU-giờ** (4,36 cho teacher/student, 0,61 cho Bami).
- Artifacts: `artifacts/cafe_oof_kd/`. Code: `scripts/cafe_oof_kd/`.

## Kết luận

**Gate PASS ở cả hai task.** Student 26,4M học thêm nhãn mềm OOF từ CafeBERT tăng **+2,00 điểm ở Restaurant** và **+1,81 điểm ở ViSFD** so với chính nó chỉ học gold. Ở cả hai task, run KD kém nhất vẫn cao hơn run CE tốt nhất.

So với BamiBERT 103M, tính trung bình 3 seed:

- **Restaurant:** KD **ngang Bami** (73,45 so với 73,36).
- **ViSFD:** KD còn kém **0,89 điểm**.

## Bảng validation F1 (seed 42 / 43 / 44)

| Task | CE: GoiTurn-S, chỉ gold | KD: GoiTurn-S + Cafe OOF | BamiBERT 103M | KD − CE | KD − Bami |
|---|---|---|---|---:|---:|
| Restaurant | 71,52 / 71,19 / 71,64 → **71,45** (range 0,45) | 73,51 / 73,40 / 73,43 → **73,45** (range 0,11) | 74,74 / 71,83 / 73,51 → **73,36** (range 2,91) | **+2,00** | **+0,09** |
| ViSFD | 81,62 / 81,14 / 81,09 → **81,28** (range 0,53) | 83,32 / 82,69 / 83,28 → **83,10** (range 0,63) | 84,38 / 83,88 / 83,70 → **83,99** (range 0,68) | **+1,81** | **−0,89** |

## Áp gate (chốt trước khi đo)

| Task | KD − CE ≥ 1,0 | Thu hẹp ≥ ½ khoảng cách Bami − CE | Kết quả | Không thua Bami (≥ Bami − 0,5), chỉ để tham khảo |
|---|---|---|---|---|
| Restaurant | 2,00 ✓ | khoảng cách 1,91, thu hẹp 2,00 (105%) ✓ | **PASS** | ✓ |
| ViSFD | 1,81 ✓ | khoảng cách 2,71, thu hẹp 1,81 (67%) ✓ | **PASS** | ✗ (−0,89) |

Pilot PASS. Theo plan, bước tiếp theo là mở rộng sang Sentiment, Topic và Hotel.

## Teacher OOF (CafeBERT 5-fold, 10 run, tất cả PASS)

| | Restaurant | ViSFD |
|---|---:|---:|
| F1 OOF so với gold train | 76,51 | 85,04 |
| Slot teacher bất đồng với gold | 3,98% | 8,11% |
| Xác suất lớn nhất trung bình | 0,988 | 0,977 |
| Tỷ lệ slot có xác suất lớn nhất ≥ 0,99 | 90,3% | 81,3% |

- F1 OOF xấp xỉ F1 validation của Cafe, đúng kỳ vọng với dự đoán ngoài tập.
- Nhãn mềm vẫn khá tự tin: phần lớn slot gần one-hot. Như vậy phần tăng chủ yếu đến từ khoảng 4–8% slot mà teacher bất đồng với gold, cộng phần xác suất mềm còn lại. Điều này phù hợp với phát hiện nhãn train có nhiễu trong [LABEL_CEILING_CHECK_RESULT.md](LABEL_CEILING_CHECK_RESULT.md).
- Chưa tách được hai nguồn tăng này. Muốn tách cần thêm đối chứng, ví dụ chỉ thay nhãn gold bằng argmax của teacher.

## Kiểm tra

- 26/26 run hoàn tất. Student có 4 check: reload khớp best, file dự đoán khớp best, dự đoán sau reload giống hệt, ID ổn định và đủ. Teacher có postrun check của `sealion_benchmark` và phủ OOF đủ, không trùng.
- Root tính lại F1 từ `validation_predictions.json` cho mọi run student và Bami; tất cả khớp metadata.
- Tái tạo: CE seed 42 Restaurant = **71,52, trùng chính xác E7**. ViSFD = 81,62 so với 81,49 của E7.
- Smoke test trước khi chạy thật: chia fold, suy luận OOF, ghép nhãn mềm với teacher giả trùng gold (F1 = 1,0), và KD 1 epoch.
- Sự cố: 4 run Bami lần đầu lỗi vì `bami_seed.py` tạo sẵn thư mục output, trong khi E14 yêu cầu thư mục chưa tồn tại. Đã sửa, xóa 4 thư mục rỗng, chạy lại; cả 4 PASS.

## Giới hạn

- Chỉ validation; chưa mở test.
- Teacher chọn epoch trên validation, nên có một liên hệ gián tiếp giữa validation và nhãn mềm.
- Một cấu hình α = 0,5, T = 2, không quét.
- Bami Restaurant dao động mạnh giữa các seed (range 2,91): seed 42 dùng làm mốc trước đây là seed cao.
- So sánh dùng số tham số (26,4M so với 103M), chưa đo tốc độ suy luận.

## Mở rộng 5 task (2026-09-26)

Chạy thêm Sentiment, Topic và Hotel với cùng recipe:

- Student: LR theo E7, 1e-5 cho VSFC và 1e-4 cho Hotel.
- Bami: LR đã khóa ở E14.
- 39 job hoàn tất, không lỗi, khoảng 5,48 GPU-giờ, chạy trên GPU 0–3. Người dùng cho thêm GPU 3 giữa chừng.
- Lần chạy đầu bị ngắt khi phiên Claude kết thúc. 3 fold dở dang được chuyển sang `artifacts/cafe_oof_kd/aborted/`, rồi chạy lại bằng launcher tách khỏi phiên.
- Root tính lại F1 của mọi run từ file dự đoán; tất cả khớp metadata và mọi check PASS.
- Bảng số: `artifacts/cafe_oof_kd/five_task_summary.json`.

| Task | CE (TB 3 seed) | KD (TB 3 seed) | Bami 103M (TB 3 seed) | KD − CE | KD − Bami | Gate |
|---|---:|---:|---:|---:|---:|---|
| Sentiment | 84,50 | 84,34 | 86,83 | −0,16 | −2,48 | FAIL |
| Topic | 80,25 | 79,92 | 79,53 | −0,33 | +0,38 | FAIL (student vốn đã ≥ Bami) |
| ViSFD | 81,29 | 83,10 | 83,99 | +1,81 | −0,89 | PASS |
| Restaurant | 71,45 | 73,45 | 73,36 | +2,00 | +0,08 | PASS |
| Hotel | 72,49 | 73,05 | 73,69 | +0,56 | −0,64 | FAIL (thu hẹp 47%) |

Teacher OOF trên 3 task mới:

| | Sentiment | Topic | Hotel |
|---|---:|---:|---:|
| F1 OOF so với gold train | 83,43 | 79,41 | 74,12 |
| Tỷ lệ bất đồng với gold | 5,7% | 11,1% | 2,15% |
| Xác suất lớn nhất trung bình | 0,996 | 0,965 | 0,989 |

Đọc kết quả:

- KD từ Cafe giúp rõ ở hai task ABSA là Restaurant và ViSFD, giúp ít ở Hotel, và không giúp ở hai task phân loại cả câu là Sentiment và Topic. Teacher Sentiment gần như one-hot (97% trường hợp có xác suất lớn nhất ≥ 0,99).
- So với Bami, student 26M (chọn nhánh tốt hơn cho mỗi task) **không thua ở Topic và Restaurant**, còn **kém 0,6–0,9 điểm ở Hotel và ViSFD**, và **kém 2,5 điểm ở Sentiment**.
