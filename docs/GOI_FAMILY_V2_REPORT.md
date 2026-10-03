# Báo cáo tổng hợp: họ encoder Goi thế hệ 2 (S′ 26M, M 50M) so với BamiBERT 103M

Ngày 2026-09-28. Phạm vi: toàn bộ chuỗi thí nghiệm từ lần test chính thức của student S tới khi có họ model S′/M. Mọi số test đều đã được root **tính lại độc lập từ file dự đoán**.

---

## 1. Kết quả chính

| | BamiBERT | **S′** | **M** |
|---|---:|---:|---:|
| Tham số | 103,0M | 26,5M (**¼**) | 49,6M (**½**) |
| TB 15 chỉ số, **test** | 83,61 | 82,98 (−0,63) | **84,02 (+0,41)** |
| TB 13 chỉ số bỏ ViNLI, **test** | 84,15 | 84,08 (−0,07) | **84,88 (+0,73)** |
| Số chỉ số test ≥ Bami | — | 6/15 | **11/15** |
| Tốc độ GPU theo lô (32 × 256) | 1× | **3,5×** | 2,0× |
| Tốc độ CPU (1 × 128) | 1× | **2,2×** | 1,6× |
| Peak VRAM (32 × 256) | 801 MB | 248 MB | 410 MB |

- **M** (12 lớp × 512, nhỏ bằng một nửa Bami) **vượt Bami về điểm trung bình trên tập test** và hơn Bami ở 11/15 chỉ số. M còn thua ở ViNLI (khoảng −1,7), PhoNER (−0,85) và Sentiment F1 (−0,33).
- **S′** (26M, nhỏ bằng ¼ Bami) **gần ngang Bami khi bỏ ViNLI** (−0,07), hơn Bami ở 6/15 chỉ số, và nhanh hơn Bami 2–3,5 lần.
- **Hai thay đổi quyết định:**
  - GD **dài gấp 4 lần**: từ 30K lên 120K bước.
  - **Họ model lớn hơn:** thêm bản M.

  Corpus sạch (v2) chỉ giúp ít về điểm, nhưng loại bỏ hoàn toàn việc tiếp xúc với text validation.

---

## 2. Quá trình và các quyết định

| Bước | Câu hỏi | Kết quả | Tài liệu |
|---|---|---|---|
| 1. Test chính thức S (E7 + KD) | Student 26M so với Bami | Hơn Bami ở 4/15 chỉ số; TB 81,72 so với 83,61; ViNLI −10,8 | `docs/FINAL_TEST_RESULT.md` |
| 2. Kiểm corpus gốc (`tmp/corpus_v1.zip`) | Có đúng corpus đã dùng để dựng S không? | Đúng: hash `stats.json` khớp manifest; số byte khớp. Chứa text **validation** của VSFC/ViSFD, **không** chứa text test | mục 3 |
| 3. Dựng corpus v2 | Corpus sạch, giữ mạch văn bản | 328,6M token, bằng v1; không trùng val/test | `docs/CORPUS_V2_RESULT.md` |
| 4. Ablation công thức GD (A1–A5) | Corpus, độ dài GD hay DAPT giúp? | **GD dài +1,7 đến +2,0**; corpus +0,2 đến +0,3; DAPT ≈ 0 | `docs/GD_RECIPE_ABLATION_RESULT.md` |
| 5. Chất lượng pool HPLT khoảng 1,1B token | v2-large có đáng làm không? | Chất lượng như web của v2; chỉ đáng làm khi GD > 120K bước | `docs/POOL_QUALITY_RESULT.md` |
| 6. So teacher DAPT với Bami gốc | Hai teacher lệch nhau bao nhiêu? | CKA theo lớp 0,86–0,99; khác biệt lớn nhất ở lớp cuối | mục 5 |
| 7. Họ S′/M, test một lần | Áp công thức thắng cho 2 kích thước | M 11/15; S′ 6/15 | `docs/FAMILY_SM_RESULT.md` |

---

## 3. Corpus: v1 (dùng cho S gốc) so với v2 (dùng cho S′/M)

| | v1 | v2 |
|---|---|---|
| Tổng | 328,6M token Bami | 328,6M token Bami |
| Web | 307,7M (94%): phần đầu CC100, lọc sơ bộ | 232,9M (71%): HPLT bin 7–9, đã lọc spun/MT, cap host 2%, near-dedup |
| Wikipedia | 0 | 49,4M (15%) |
| UGC / bình luận | 0 | 25,5M (8%) |
| Text task | 20,9M (6%): VSFC, ViSFD, VLSP2018, PhoATIS; **train + validation**, lặp 20 lần | 20,9M (6%): 8 task; **chỉ train**, lặp tối đa khoảng 10 lần |
| Trùng lặp đoạn web | khoảng 9% (mẫu); boilerplate lặp tới hơn 5.000 lần | 0% |
| Mạch văn bản | xáo **từng dòng** | xáo **theo bài**, đoạn trong bài liền nhau |
| Trùng val/test | 100% val VSFC, 89% val ViSFD; test 0 | **0 / 0** (đã khử 30.334 đoạn, 2.630 bài) |
| sha256 `mlm_mix.txt` | `fb0b4dd4…` | `3ba6537c…` |

**Pool HPLT đầy đủ (khoảng 1,1B token):**

- Phần chưa dùng có chất lượng như phần đã vào v2. Trên mẫu 20K bài: tỷ lệ tiếng Việt 99,5%; trùng đoạn 3,5–4,9%; boilerplate 2,3–2,5%.
- Cả hai phần đều sạch hơn CC100 của v1 khoảng 2–2,5 lần.
- Nếu dựng **v2-large** khoảng 1B token, tỷ lệ web sẽ tăng lên khoảng 87%, còn Wikipedia và UGC giảm tỷ lệ vì đã dùng gần hết.

---

## 4. Ablation công thức GD (validation, student 26M, trung bình 3 seed)

**P** = trung bình của 5 chỉ số: ViNLI F1, PhoNER F1, Sentiment F1, ViSpam F1 và Restaurant cls.

| Arm | Corpus | Teacher | Số bước | Sentiment | Restaurant | ViSpam | ViNLI | PhoNER | **P** | ΔP so với A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S gốc | v1 | DAPT | 30K | 84,34 | 73,45 | 75,44 | 71,07 | 93,20 | 79,50 | — |
| A1 | v1 | DAPT | 30K | 84,35 | 73,27 | 76,68 | 70,18 | 93,07 | 79,51 | 0 |
| A2 | v2 | DAPT | 30K | 84,20 | 73,11 | 75,88 | 72,17 | 93,19 | 79,71 | +0,20 |
| A3 | v1 | DAPT | 120K | 85,47 | 74,55 | 77,55 | 74,38 | 94,13 | 81,21 | **+1,70** |
| **A4** | v2 | DAPT | 120K | 84,77 | 74,44 | 78,64 | 75,60 | 94,08 | **81,51** | **+2,00** |
| A5 | v1 | Bami gốc | 30K | 83,31 | 73,44 | 76,57 | 71,54 | 93,36 | 79,65 | +0,13 |
| **S′** | v2 | Bami gốc | 120K | 84,47 | 73,66 | 77,08 | 75,87 | 94,13 | 81,04 | +1,53 |

- **Port pipeline chuẩn:** A1 − S gốc = +0,01.
- **Độ dài GD là đòn bẩy chính.** Loss GD đi ngang từ khoảng 75K bước (khoảng 3 vòng qua corpus 329M token).
- **S′ kém A4 0,46.** Mức này nằm trong ngưỡng 0,5 đã khóa, nên theo quy tắc S′ được chọn làm bản chính thức vì sạch.

---

## 5. Teacher DAPT so với Bami gốc

Đo trên 512 đoạn text của v2, khoảng 34K token.

| | Kết quả |
|---|---|
| Trọng số | Thay đổi tương đối khoảng 10% ở mọi lớp |
| CKA theo lớp | 0,99 (embedding), 0,96 (lớp 3), 0,92 (lớp 6–9), 0,86 (lớp 12) |
| Tác động lên student | A5 so với A1 là +0,13; S′ so với A4 là −0,46 |

**Ảnh hưởng của việc teacher DAPT từng đọc text validation:**

- Nhóm chỉ số bị dính lạc quan hơn nhóm sạch khoảng 0,4 điểm, nằm trong mức nhiễu.
- Không ảnh hưởng tới test.
- S′ và M dùng Bami gốc nên không có vấn đề này.

---

## 6. Các model

| Model | Kiến trúc | Tham số | Khởi tạo | Teacher GD | Corpus / số bước | Ghi chú |
|---|---|---:|---|---|---|---|
| Bami | 12L × 768 | 103,0M | — | — | — | Baseline; số chạy lại tại máy |
| S gốc | 10L × 384 | 26,5M | PCA embedding | DAPT | v1 / 30K | Test 4/15 |
| A4 | 10L × 384 | 26,5M | PCA embedding | DAPT | v2 / 120K | Student tốt nhất trên validation; **chưa test** |
| **S′** | 10L × 384 | 26,5M | PCA embedding | Bami gốc | v2 / 120K | Chính thức, sạch |
| M-6×768 | 6L × 768 | 59,8M | Chép lớp 2,4,…,12 của Bami | Bami gốc | v2 / 120K | Thua chọn hình dạng (82,50) |
| **M** | 12L × 512 | 49,6M | PCA embedding | Bami gốc | v2 / 120K | Thắng chọn hình dạng (82,86); chính thức |

**Hai giai đoạn học của mọi student:**

1. **GD (không dùng nhãn):** student bắt chước hidden state và attention của teacher GD trên text corpus. Với S′ và M, teacher là **Bami gốc giữ nguyên**, không DAPT.
2. **Fine-tune từng task:** E7 (LLRD γ 0,8) cộng KD từ nhãn mềm **CafeBERT** out-of-fold (α 0,5, T 2), 30 epoch, chọn epoch trên validation.
   - CafeBERT (560M) chỉ xuất hiện ở giai đoạn này, dưới dạng xác suất đáp án cho từng câu train.
   - Bami baseline fine-tune chỉ với nhãn thật (recipe E14), **không có KD**.

- LR của S′ cố định theo recipe đã khóa.
- LR của M quét {LR student, LR Bami} trên seed 42. LR thắng: Sentiment 1e-5, Topic 1e-5, ViSpam 5e-5, còn lại 1e-4.

---

## 7. Validation: 8 dataset, 15 chỉ số (trung bình 3 seed, %)

| Dataset | Chỉ số | Bami | S gốc | A4 | S′ | **M** |
|---|---|---:|---:|---:|---:|---:|
| Sentiment | Acc | 94,69 | 94,29 | 94,57 | 94,57 | **94,99** |
| | F1 | **86,83** | 84,34 | 84,77 | 84,47 | 86,27 |
| Topic | Acc | 88,82 | 89,66 | **89,77** | 89,45 | 89,49 |
| | F1 | 79,53 | 79,92 | **80,08** | 79,88 | 80,04 |
| ViSFD | det | 89,15 | 89,18 | 89,68 | 89,45 | **89,84** |
| | cls | 83,99 | 83,10 | 83,94 | 83,67 | **84,69** |
| Restaurant | det | 85,57 | 85,76 | 86,14 | 85,57 | **86,29** |
| | cls | 73,36 | 73,45 | 74,44 | 73,66 | **75,20** |
| Hotel | det | **79,46** | 77,88 | 78,71 | 78,89 | 79,14 |
| | cls | 73,69 | 73,05 | 73,63 | 73,66 | **74,13** |
| ViSpam | Acc | 90,10 | 90,21 | 90,40 | 89,73 | **90,59** |
| | F1 | 77,38 | 75,44 | **78,64** | 77,08 | 78,32 |
| ViNLI | Acc | **80,95** | 71,00 | 75,67 | 75,89 | 78,03 |
| | F1 | **80,94** | 71,07 | 75,60 | 75,87 | 78,02 |
| PhoNER | F1 | **95,00** | 93,20 | 94,08 | 94,13 | 94,58 |
| **TB 15** | | 83,96 | 82,10 | 83,34 | 83,06 | **83,97** |
| **≥ Bami** | | — | 6/15 | 7/15 | 5/15 | **10/15** |

---

## 8. Test: 8 dataset, 15 chỉ số (trung bình 3 seed, %)

| Dataset | Chỉ số | Bami | S gốc | S′ | **M** | M − Bami |
|---|---|---:|---:|---:|---:|---:|
| Sentiment | Acc | 93,50 | 92,45 | 92,70 | **93,56** | +0,05 |
| | F1 | **82,36** | 80,54 | 81,50 | 82,03 | −0,33 |
| Topic | Acc | 88,82 | 89,06 | 89,36 | **89,73** | +0,92 |
| | F1 | 78,88 | 78,71 | 80,17 | **80,42** | +1,54 |
| ViSFD | det | 89,26 | 88,96 | 89,30 | **89,81** | +0,55 |
| | cls | 84,41 | 83,24 | 83,94 | **84,78** | +0,37 |
| Restaurant | det | 87,03 | 86,78 | 86,91 | **88,14** | +1,11 |
| | cls | 73,53 | 73,69 | 74,15 | **76,00** | +2,47 |
| Hotel | det | 80,15 | 80,38 | 80,66 | **80,96** | +0,81 |
| | cls | 73,94 | 74,87 | 75,01 | **75,67** | +1,73 |
| ViSpam | Acc | 90,19 | 89,93 | 89,88 | **90,75** | +0,56 |
| | F1 | 77,55 | 76,40 | 76,52 | **78,13** | +0,58 |
| ViNLI | Acc | **80,07** | 69,23 | 75,75 | 78,39 | −1,68 |
| | F1 | **80,18** | 69,46 | 75,86 | 78,52 | −1,67 |
| PhoNER | F1 | **94,32** | 92,18 | 92,97 | 93,47 | −0,85 |
| **TB 15** | | 83,61 | 81,72 | 82,98 | **84,02** | +0,41 |
| **TB bỏ ViNLI** | | 84,15 | 83,63 | 84,08 | **84,88** | +0,73 |
| **≥ Bami** | | — | 4/15 | 6/15 | **11/15** | |

Số Bami và S gốc lấy từ lượt test ngày 2026-09-27 (`artifacts/final_test/`). Số S′ và M lấy từ lượt test ngày 2026-09-28 (`artifacts/family_sm/test/`).

---

## 9. Hiệu năng

Root đo lại trên GPU 2 khi GPU trống (L40S). Chỉ đo encoder, 2 lượt với thứ tự đảo ngược. GPU dùng bf16, 200 lượt đo. CPU dùng fp32, 8 luồng.

| | Bami | S′ | M | M-6×768 |
|---|---:|---:|---:|---:|
| GPU 1 × 128 (ms) | 2,93 | 2,52 | 3,26 | 1,56 |
| GPU 32 × 128 (ms) | 6,09 | 2,43 | 3,28 | 3,12 |
| GPU 32 × 256 (ms) | 13,35 | 3,86 | 6,63 | 6,82 |
| Throughput 32 × 256 (câu/giây) | 2.397 | 8.290 | 4.827 | 4.692 |
| CPU 1 × 128 (ms) | 31,0 | 14,2 | 19,4 | 15,3 |
| Peak VRAM 32 × 256 | 801 MB | 248 MB | 410 MB | 536 MB |
| Trọng số bf16 | 206 MB | 53 MB | 99 MB | 120 MB |

- **M chạy từng câu trên GPU thì ngang Bami**, vì cùng 12 lớp và ở batch 1 thời gian bị chi phối bởi chi phí khởi chạy từng lớp.
- **M chạy theo lô thì nhanh khoảng 2 lần.**
- Nếu cần độ trễ thấp nhất khi xử lý từng câu, M-6×768 nhanh gấp khoảng 1,9 lần nhưng thấp hơn 0,36 điểm validation.
- Số liệu gốc: `artifacts/family_sm/efficiency_root_recheck.json`.
- Máy chủ có tải CPU khoảng 60–74 / 128 lõi, nên số CPU có nhiễu.

---

## 10. Các kiểm tra đã làm

- **Corpus v1:** hash `stats.json` khớp manifest của S và của teacher DAPT; số byte khớp; chênh số dòng giải thích được chính xác (2.093 × 20).
- **Corpus v2:** hash khớp `READY`; quét độc lập cho thấy 0 trùng với val/test của 8 task.
- **Ablation GD:** tính lại 90 run từ file dự đoán, khớp tới 3 chữ số thập phân.
- **Test S′/M:**
  - manifest khóa lúc 12:26:36.331, trước lần mở test đầu tiên lúc 12:26:36.334;
  - hash manifest khớp log truy cập;
  - mỗi checkpoint chỉ chạy inference test một lần;
  - 15 chỉ số tính lại khớp chính xác.
- **Code:** phần sửa optimizer E7 cho số lớp tùy ý đã được xem lại, đúng. Khởi tạo M-6×768 khớp fp32 tuyệt đối với các lớp teacher.

---

## 11. Giới hạn và điểm reviewer có thể hỏi

1. **KD từ CafeBERT (560M) chỉ áp cho student, không áp cho Bami.** So sánh là "student + KD" với "Bami fine-tune thường". Bằng chứng duy nhất về Bami + KD là pilot trên 2 task (validation): Sentiment −1,7, Restaurant +1,7, không giúp đều. Nên báo rõ quy trình này.
2. **Corpus v2 chứa text train (không nhãn) của 8 task.** Đây là kiểu task-adaptive pretraining, hợp lệ vì chỉ dùng train, nhưng Bami không có bước này.
3. **Nhiễu:** mỗi backbone chỉ có 1 GD seed. Hình dạng M được chọn trên 5 task, 1 seed; chênh 0,36 chỉ vừa vượt ngưỡng 0,3.
4. **Tập test đã được dùng qua 2 lượt** (S rồi S′/M). Mỗi model chỉ được test đúng một lần, và mọi lựa chọn công thức đều dựa trên validation. Tuy vậy, việc quyết định làm thêm bản M có một phần xuất phát sau khi đã thấy kết quả test của S.
5. **Dữ liệu:**
   - ViNLI là mirror `presencesw/vinli_4_label`, chưa xác minh với bản gốc.
   - PhoNER và các dataset UIT chỉ dùng cho nghiên cứu, không phát tán lại.
6. **Số Bami là bản chạy lại tại máy**, thấp hơn số trong paper Bami khoảng 0–1 điểm ở phần lớn chỉ số.
7. **A4 chưa có số test.** A4 tốt hơn S′ khoảng 0,3 điểm trên validation, nhưng teacher của A4 từng đọc text validation.
8. **Tốc độ đo chỉ gồm encoder.** Head và tiền xử lý không được tính.

---

## 12. Tài nguyên đã dùng (giai đoạn này)

| Việc | GPU-giờ |
|---|---:|
| Test S và đo hiệu năng (27/09) | khoảng 0,3 |
| Ablation GD (5 arm, 75 fine-tune) | 29,1 |
| Họ S′/M: 3 GD, fine-tune, test | 28,4 |
| Fine-tune bổ sung A4 (3 task) | khoảng 1,3 |
| **Tổng** | **khoảng 59** |

Corpus v2 và phần đo pool HPLT chỉ dùng CPU.

---

## 13. Hướng tiếp theo (không bắt buộc)

1. **ViNLI:** mở rộng KD từ Cafe sang dữ liệu NLI chưa có nhãn, theo kiểu augmentation của TinyBERT. Đây là chỗ còn thua lớn nhất.
2. **GD dài hơn cho M:** dùng v2-large, lịch WSD, và đánh giá tại 120K, 180K, 240K. Dung lượng của M lớn hơn S nên có thể còn "đói dữ liệu".
3. **Bami + KD trên đủ 8 task,** để trả lời điểm 1 ở mục 11 cho paper.
4. **Công bố:** Bami là BSD-3-Clause-Clear + Qualcomm RAIL. S′ và M là derivative, cần rà soát pháp lý trước khi phát hành.

---

## 14. Chỉ mục file

- **Plan:** `docs/GD_RECIPE_ABLATION_PLAN.md`, `docs/FAMILY_SM_PLAN.md`.
- **Kết quả:** `docs/FINAL_TEST_RESULT.md`, `docs/CORPUS_V2_RESULT.md`, `docs/GD_RECIPE_ABLATION_RESULT.md`, `docs/POOL_QUALITY_RESULT.md`, `docs/FAMILY_SM_RESULT.md`.
- **Backbone:**
  - `artifacts/family_sm/backbones/{s_prime,m12,m6}/final/`
  - `artifacts/gd_ablation/backbones/A1…A5/final/`
- **Test:** `artifacts/family_sm/test/` gồm `locked_manifest.json`, `test_access_log.json`, `test_results.json`, `predictions/`.
- **Corpus:** `data/corpus_v1/`, `data/corpus_v2/`. Đã git-ignore.
