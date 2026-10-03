# Plan: họ model S′/M — GD sạch từ Bami gốc, chọn hình dạng M, test một lần

Chốt ngày 2026-09-28. Người dùng duyệt và cho Codex (gpt-6-sol, reasoning high) chạy ngoài sandbox trên GPU 0–3, kết thúc bằng **một lần test**. Root lập plan, review và tổng kết.

## 1. Cơ sở

Kết quả từ `docs/GD_RECIPE_ABLATION_RESULT.md` (validation):

- GD 120K bước tốt hơn 30K: ΔP +1,70 trên corpus v1, +2,00 trên corpus v2.
- Corpus v2 sạch, chỉ hơn v1 một chút.
- Bỏ DAPT không làm kém đi: A5 − A1 = +0,13.
- Teacher DAPT và Bami gốc có biểu diễn gần nhau (CKA theo lớp 0,86–0,99; root đo).

**Công thức chung cho cả họ model:** teacher `models/bamibert-base-original`, corpus `data/corpus_v2`, GD 120K bước, fine-tune E7 + KD Cafe OOF. Không còn phụ thuộc nào vào text validation.

## 2. Ba backbone

GD dùng lại pipeline `scripts/gd_ablation/`: hidden MSE + attention MSE, LR 1e-4, wd 0,01, warmup 1000, decay tuyến tính tới 120K, batch 32, độ dài 256, bf16, attention eager, seed 42.

| Backbone | Kiến trúc | Khởi tạo | Layer mapping (student → teacher) | Tham số ước |
|---|---|---|---|---|
| **S′** | 10L × 384, 6 head, FFN 1536 (giống S) | PCA 768→384 từ embedding Bami gốc; các lớp khởi tạo ngẫu nhiên (giống S) | [0,1,2,4,5,6,7,8,10,11,12] | 26,5M |
| **M-6×768** | 6L × 768, 12 head, FFN 3072 | Chép embedding, LayerNorm embedding và các lớp **2,4,6,8,10,12** (đánh số từ 1) của Bami gốc | [0,2,4,6,8,10,12] | khoảng 60M |
| **M-12×512** | 12L × 512, 8 head, FFN 2048 | PCA 768→512 embedding; các lớp khởi tạo ngẫu nhiên | [0,1,…,12] | khoảng 49M |

Với M-6×768, projection của aligner (768→768) khởi tạo **bằng ma trận đơn vị**, để loss ban đầu phản ánh đúng chất lượng khởi tạo. Các backbone khác giữ khởi tạo gốc.

## 3. Fine-tune (validation-only)

Recipe khóa: E7 (LLRD γ 0,8) + KD (α 0,5, T 2) từ nhãn mềm Cafe OOF đã có, 30 epoch. Các lựa chọn khác giống các run student trước.

LR của student đã khóa:

| Task | LR student |
|---|---|
| Sentiment | 1e-5 |
| Topic | 1e-5 |
| ViSFD | 1e-4 |
| Restaurant | 1e-4 |
| Hotel | 1e-4 |
| ViSpam | 5e-5 |
| ViNLI | 1e-4 |
| PhoNER | 1e-4 |

**S′:**

1. Dùng LR student, không quét lại.
2. Chạy **5 task trước** (Sentiment, Restaurant, ViSpam, ViNLI, PhoNER) × seed 42/43/44.
3. **Gate dự phòng:** nếu P(S′) < P(A4) − 0,5, trong đó P(A4) = 81,50748919256499 (xem `artifacts/gd_ablation/summary.json`), thì S′ chính thức là **A4**:
   - fine-tune thêm A4 cho Topic, ViSFD, Hotel × 3 seed;
   - ghi chú trong báo cáo rằng teacher DAPT từng đọc text validation.

   Nếu không kích hoạt gate: chạy tiếp Topic, ViSFD, Hotel × 3 seed cho S′.

**M, gồm hai bước chọn:**

1. **Chọn hình dạng.**
   - Seed 42, 5 task nêu trên, mỗi task 2 LR: LR student và LR Bami.

     | Task | LR Bami |
     |---|---|
     | Sentiment | 2e-5 |
     | Restaurant | 5e-5 |
     | ViSpam | 2e-5 |
     | ViNLI | 1e-5 |
     | PhoNER | 5e-5 |

   - Điểm của một hình dạng = trung bình 5 task, mỗi task lấy LR tốt hơn.
   - Chọn hình dạng có điểm cao hơn. Nếu chênh < 0,3 thì chọn **6×768** vì chạy nhanh hơn.
2. **Chọn LR cho 3 task còn lại** (hình dạng đã chọn, seed 42):
   - ViSFD và Hotel: 1e-4 so với 5e-5.
   - Topic: chỉ 1e-5, vì LR student và LR Bami trùng nhau.

   Sau đó chạy seed 43 và 44 cho cả 8 task ở LR đã chọn. Run seed 42 được dùng lại.

Chỉ số P trong plan này là trung bình của 5 task sau (tính theo %):

- ViNLI macro F1
- PhoNER entity F1
- Sentiment macro F1
- ViSpam macro F1
- Restaurant cls

## 4. Test một lần

Sau khi mọi lựa chọn đã chốt trên validation:

1. **Khóa manifest** cho 48 checkpoint: S′ và M, mỗi model 8 task × 3 seed. Manifest gồm sha256, LR, epoch, và chỉ số validation được tính lại.
2. **Chạy test đúng một lần**, theo đúng protocol và evaluator của `scripts/final_test/`. Ghi log truy cập test.
3. **Bảng kết quả 15 chỉ số**, gồm:
   - Bami và S gốc, lấy từ `artifacts/final_test/test_results.json` (không chạy lại);
   - S′ và M.
4. **Đo hiệu năng** Bami, S′ và M trong cùng một phiên, cùng cách đo với `scripts/final_test/efficiency.py`.

## 5. Ngân sách và vận hành

**Ước tính:**

- GD: S′ khoảng 4–5 GPU-giờ; mỗi bản M khoảng 7 GPU-giờ.
- Fine-tune: khoảng 18 GPU-giờ.
- Test: khoảng 1 GPU-giờ.
- **Trần 45 GPU-giờ.**

**Thứ tự:** GD M-6×768, GD M-12×512 và GD S′ khởi động song song. Fine-tune S′ chạy ngay khi backbone S′ xong. Fine-tune M bắt đầu khi cả hai bản M đã GD xong.

**Quy tắc GPU:**

- Mỗi GPU đúng một job của mình.
- Ngưỡng VRAM trống để launch lấy theo peak đo được ở bước smoke, cộng dư.
- Không kill, dừng hay renice process của người khác.
- Launcher chạy tách phiên và nhận lại được job đang chạy dở.

## 6. Giới hạn đã biết

- Mỗi backbone chỉ có 1 GD seed.
- LR của S′ không quét lại. LR của M chỉ quét 2 giá trị.
- Hình dạng M được chọn trên 5 task, 1 seed.
- Test dùng hết sau lượt này; cải tiến tiếp theo cần một lượt test mới.
