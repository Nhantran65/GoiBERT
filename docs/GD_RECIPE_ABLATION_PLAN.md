# Plan: tìm công thức GD tốt nhất cho student (corpus, độ dài GD, DAPT) trước khi làm bản M

Chốt ngày 2026-09-27. Người dùng yêu cầu làm và cho phép Codex (gpt-6-sol, reasoning high) chạy ngoài sandbox. Root lập plan, giao việc, review và chọn công thức.

## 1. Câu hỏi

Student S (GoiTurn-S 26M) được tạo theo 3 bước:

1. **Corpus v1:** 1,67 GB. Gồm phần đầu CC100-vi, xáo trộn **từng dòng**, và text train + validation của VSFC, ViSFD, VLSP2018 Restaurant, PhoATIS lặp 20 lần.
2. **DAPT:** Bami chạy MLM 40K bước trên corpus v1, cho ra teacher `goibert-b-dapt`.
3. **GD:** 30K bước × batch 32 × 256 token, khoảng 244M token. Chưa đi hết một vòng corpus.

Plan này trả lời **3 câu hỏi**, mỗi câu tương ứng một yếu tố, không có dữ liệu mới nào được dùng để chọn model:

- (a) Corpus sạch hơn và giữ mạch văn bản có giúp student không?
- (b) GD dài hơn có giúp không?
- (c) Bước DAPT có đáng giữ không?

Công thức thắng sẽ dùng để dựng bản M (6 lớp × 768) ở vòng sau.

## 2. Corpus v1 (đã xác minh)

- **Vị trí:** `data/corpus_v1/{mlm_mix.txt,stats.json}`, giải nén từ `tmp/corpus_v1.zip`.
- **Hash:**
  - `stats.json` sha256 `1fa34627…` khớp manifest của S và của teacher DAPT;
  - `mlm_mix.txt` sha256 `fb0b4dd4f2a8bc6912bf13bb6a6d4f2af9d3271a6cfe49941eed0650f178c63f`;
  - số byte khớp `total_bytes`.
- **Số dòng vật lý:** 9.627.914, stats ghi 9.586.054. Chênh 41.860 = 20 × 2.093, do review ViSFD có chứa ký tự xuống dòng.
- **Đặc điểm đã đo:**
  - khoảng 19% dòng CC100 là bản lặp (boilerplate lặp tới hơn 5.000 lần);
  - chứa 100% text validation VSFC và 89% text validation ViSFD;
  - không chứa text test của task nào.

## 3. Corpus v2 (Codex A dựng, chỉ dùng CPU)

Mục tiêu: **cùng tổng số token Bami** như v1 (±5%), chỉ đổi chất lượng và thành phần.

| Thành phần | Nguồn local | Tỷ lệ token (Bami) |
|---|---|---|
| Web tổng quát | HPLT 3.0 pool đã lọc ở vòng mmBERT v2 (bin 7–9, đã lọc spun/MT, cap host 2%, near-dedup). Text lấy lại **đúng nguyên văn** từ `data/raw/hplt3_vie/*.jsonl.zst` theo doc id. | phần còn lại, khoảng 65–70% |
| Wikipedia | Bài đã lọc ở vòng mmBERT v2 (`data/mmbert_vi_cpt_pilot/wiki/`), text lấy từ `data/raw/wikipedia_vi` | khoảng 15% |
| UGC / ý kiến người dùng | Bucket `v2/ugc_refined` (khoảng 37,5M token mmBERT) | lấy hết, tối đa 10% |
| Text task (không nhãn) | **Chỉ split train** của 8 task: VSFC, ViSFD, Restaurant, Hotel, ViSpam, ViNLI (mỗi doc = 1 premise + các hypothesis của nó), PhoNER | **Bằng đúng tỷ lệ token domain của v1** (Codex đo), lặp tối đa 20 lần |

**Quy tắc:**

- **Giữ mạch văn bản:**
  - các đoạn của cùng một doc nằm liền nhau, mỗi đoạn một dòng, doc cách nhau bằng dòng trống;
  - xáo ở **mức doc** (seed 42);
  - doc dài được cắt ở ranh giới đoạn, tối đa 2.048 token Bami.
- **Dedup:**
  - exact theo đoạn (chuẩn hóa NFC + casefold + gộp khoảng trắng);
  - bỏ đoạn xuất hiện ≥ 5 lần trong phần web (boilerplate).
- **Khử nhiễm (decontam):**
  - khóa đối chiếu là text **validation và test** của cả 8 task;
  - text ≥ 13 âm tiết: loại đoạn chứa bất kỳ 13-gram âm tiết nào của nó (chuẩn hóa: chữ thường, bỏ dấu câu);
  - text < 13 âm tiết: loại đoạn trùng nguyên văn sau chuẩn hóa;
  - loại cả doc nếu > 20% đoạn bị dính;
  - text train của task trùng nguyên văn với một text val/test (trùng lặp nội bộ dataset) cũng bị loại.
- **Kiểm cuối:** quét lại v2 như đã làm với v1. Số khớp nguyên văn với val/test của 8 task phải bằng **0**.
- **Đầu ra:**
  - `data/corpus_v2/{mlm_mix.txt,stats.json,manifest.json,decontam_report.json,READY}`;
  - `READY` ghi sau cùng, chỉ khi mọi kiểm tra PASS;
  - báo cáo `docs/CORPUS_V2_RESULT.md`, kèm 20 doc ngẫu nhiên để audit bằng mắt.

Corpus v2 gộp nhiều thay đổi thành **một gói**: sạch hơn, mạch văn, đủ domain và không có validation. Nếu gói này thắng thì dùng cả gói. Plan này không tách riêng đóng góp của từng thành phần.

## 4. Các arm GD (Codex B, GPU)

Kiến trúc, PCA-init, loss (hidden MSE + attention MSE), layer mapping, LR 1e-4, warmup 1000, batch 32 và độ dài 256 đều giữ **y hệt S**. Mỗi arm chạy một GD seed (42).

| Arm | Corpus | Teacher GD | Số bước | Trả lời |
|---|---|---|---:|---|
| **A1 ctrl** | v1 | goibert-b-dapt | 30K | Mốc so sánh, đồng thời kiểm việc port pipeline |
| **A2 v2** | v2 | goibert-b-dapt | 30K | (a) corpus |
| **A3 long** | v1 | goibert-b-dapt | 120K | (b) độ dài GD |
| **A4 v2-long** | v2 | goibert-b-dapt | 120K | (a) × (b) |
| **A5 noDAPT** | v1 | bamibert-base-original | 30K | (c) DAPT |
| R0 | — | — | — | S gốc, dùng kết quả E7+KD 3 seed đã có |

- Arm 120K dùng lịch LR tuyến tính decay trên 120K bước.
- Precision: bf16 autocast cho cả teacher và student, loss tính fp32, attention eager (cần xác suất attention). S gốc chạy fp32 trên MI250X. Vì mọi arm dùng chung pipeline mới, so sánh giữa các arm vẫn công bằng; khoảng cách giữa A1 và R0 đo đúng độ lệch do port.

## 5. Đánh giá (validation-only, không mở test)

- **Recipe khóa:** E7 (LLRD γ 0,8) + KD từ nhãn mềm Cafe OOF (α 0,5, T 2), 30 epoch. LR cố định theo recipe đã khóa của S, không quét lại:

  | Task | LR |
  |---|---|
  | Sentiment | 1e-5 |
  | Restaurant | 1e-4 |
  | ViSpam | 5e-5 |
  | ViNLI | 1e-4 |
  | PhoNER | 1e-4 |

- **5 task:** ViNLI (macro F1), PhoNER (entity F1), Sentiment (macro F1), ViSpam (macro F1), Restaurant (cls, pair micro F1). ABSA/Topic khác được bỏ qua vì đã chạm trần nhãn.
- **Seed fine-tune:** 42, 43, 44 cho mỗi arm.
- **Điểm chính P:** trung bình 5 task của trung bình 3 seed. Báo thêm P khi bỏ ViNLI và từng task riêng kèm range.

**Quy tắc quyết định (khóa trước khi có số):**

1. **Kiểm port:** nếu |P(A1) − P(R0)| > 1,0 thì báo rõ. So sánh giữa các arm vẫn dựa trên A1.
2. **Một yếu tố "giúp"** khi ΔP so với A1 ≥ +0,5 **và** không task nào giảm quá 1,0 điểm (tính trên trung bình 3 seed).
3. **DAPT:**
   - nếu P(A5) ≥ P(A1) − 0,3: DAPT không cần thiết, bản M có thể GD thẳng từ Bami gốc (đơn giản hơn, không dính corpus v1);
   - ngược lại: giữ DAPT.
4. **Công thức cho M:** arm có P cao nhất trong số các arm thỏa quy tắc 2. Nếu không arm nào thỏa, giữ công thức S gốc.

Codex tính số, **root ra quyết định**.

## 6. Ngân sách và vận hành

- **GD:** 3 × 30K + 2 × 120K = 330K bước. Tốc độ sẽ đo ở bước smoke; ước 0,15–0,3 giây/bước, tức khoảng 14–28 GPU-giờ.
- **Fine-tune:** 5 arm × 3 seed × khoảng 1 GPU-giờ (5 task), khoảng 15 GPU-giờ.
- **Trần 50 GPU-giờ.** Nếu phép đo tốc độ cho thấy tổng vượt trần, dừng và báo trước khi chạy arm 120K.
- **GPU:**
  - chỉ dùng GPU 0–3, mỗi GPU đúng một job của mình (không chồng job);
  - chỉ launch khi GPU còn ≥ 16 GB trống với GD, ≥ 12 GB với fine-tune;
  - không kill, dừng hay renice process của người khác.
- **Launcher:**
  - chạy tách phiên (`setsid nohup`);
  - nhận lại được job đang chạy dở;
  - GD resume được từ checkpoint;
  - giữ 2 checkpoint gần nhất cộng bản cuối.
- **Thứ tự ưu tiên:** A3 → A1 → A5, rồi A4 → A2 khi `data/corpus_v2/READY` có. Fine-tune được đưa vào hàng đợi ngay khi backbone tương ứng xong.
- **Hai Codex chạy song song, không đụng file của nhau:**
  - **Codex A:** `scripts/corpus_v2/`, `data/corpus_v2/`, `docs/CORPUS_V2_RESULT.md`. Chỉ dùng CPU.
  - **Codex B:** `scripts/gd_ablation/`, `artifacts/gd_ablation/`, `docs/GD_RECIPE_ABLATION_RESULT.md`. Codex B chỉ **đọc** `data/corpus_v2/` sau khi có `READY`.
- Không commit, không push, không cài hay gỡ gói, không sửa code hay artifact cũ.

## 7. Giới hạn đã biết

- Mỗi arm chỉ có 1 GD seed, nên chưa đo được độ biến thiên do GD.
- LR fine-tune không quét lại cho backbone mới.
- v2 là một gói nhiều thay đổi, không tách được từng thành phần.
- Kết quả là validation. Công thức thắng sẽ dựng S'/M, và mỗi model đó cần một lần test riêng, có manifest khóa.
