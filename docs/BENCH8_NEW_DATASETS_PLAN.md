# Plan: bổ sung 3 dataset (ViSpamReviews, ViNLI, PhoNER_COVID19) để so BamiBERT và Student S theo protocol 8 dataset của paper BamiBERT

Chốt ngày 2026-09-26. Người dùng yêu cầu làm toàn bộ. Root lập plan, giao **một Codex** (gpt-6-sol, reasoning high) làm tuần tự theo skill codex-handoff, root review và tổng kết.

## 1. Mục tiêu

Paper BamiBERT đánh giá trên 8 dataset / 15 chỉ số. GoiBERT đã so Bami và Student S (GoiTurn-S 26M) trên 5 dataset. Bảng validation, trung bình 3 seed, nằm ở `artifacts/cafe_oof_kd/five_task_summary.json` và các run tương ứng.

Plan này bổ sung 3 dataset còn thiếu **với cùng một protocol**, để có bảng đủ 8 dataset:

- Bami 103M so với Student S 26M;
- validation, trung bình 3 seed;
- **không mở test**.

## 2. Dữ liệu (root đã tải, ghim nguồn; tất cả chỉ dùng cho nghiên cứu)

| Task id | Nguồn local | Split | Nhãn | Chỉ số (chọn epoch theo chỉ số in đậm) |
|---|---|---|---|---|
| `vispam` (chính) | `data/raw/social/vispamdetection/{train,dev,test}.csv`, cột `SpamLabel` | 14.306 / 1.590 / 3.974 | 4 lớp: 0 = không spam (10.585 train), 1 (195), 2 (1.087), 3 (2.439) | Accuracy, **macro F1** |
| `vispam_binary` (phụ) | cùng file, cột `Label` | như trên | 2 lớp | Accuracy, **macro F1** |
| `vinli` | `data/raw/vinli_mirror/{train,dev,test}.parquet`: mirror `presencesw/vinli_4_label@a885fd1bdeb00099f2215b601ddb0df255503cf1`, chưa xác minh nguồn; người dùng chọn dùng | 24.376 / 3.009 / 2.991 | 4 nhãn cân bằng: entailment / neutral / contradiction / other | Accuracy, **macro F1** |
| `phoner` | `data/raw/phoner_covid19/{train,dev,test}_syllable.json`: VinAI commit `9af190aa76adcfad0a2d14fab9bcab2b252c9c2e`, **bản theo âm tiết** vì Bami không tách từ; không được phát tán lại | 5.027 / 2.000 / 3.000 câu | 21 nhãn BIO, 10 loại thực thể | **F1 mức thực thể, micro, khớp chặt** (kiểu conlleval/seqeval); kèm precision/recall |

Sha256 các file mới: xem `artifacts/bench8/data_manifest.json` do Codex tạo. Root đã ghi sẵn trong spec.

**Vì sao chọn SpamLabel 4 lớp làm chỉ số chính:**

- Paper Bami báo ViSpamReviews Acc 90,76 / F1 78,20, lấy theo số của ViSoBERT (90,99 / 79,06).
- Với nhãn nhị phân (26% spam), Acc khoảng 91 sẽ đi kèm macro-F1 khoảng 88. Chỉ task 4 lớp, có lớp hiếm 195 mẫu, mới cho mức F1 khoảng 78–79.
- Báo thêm bản nhị phân để tham khảo.

## 3. Model và recipe (giữ nhất quán với các kết quả 5 task đã có)

| Model | Nguồn | Head phân loại câu | Optimizer / LR grid |
|---|---|---|---|
| **BamiBERT** | `models/bamibert-base-original`, dựng qua `goibert_e14.modeling.build_bamibert_classifier` | head E14: CLS → dropout → dense 768 → tanh → dropout → logits | recipe E14 (`scripts/run_e14.py::_run_trial`): AdamW `goibert.train.make_optimizer`; LR ∈ {1e-5, 2e-5, 5e-5} |
| **Student S** | `models/goiturn-s-gd`, dựng qua `goibert.modeling.build_classifier` | head E1 | recipe E7: LLRD γ = 0,8 (`goibert_e7.optim.build_layerwise_optimizer`); LR ∈ {1e-5, 5e-5, 1e-4} |

**Chung cho cả hai:**

- 30 epoch, batch 32, warmup 10% tuyến tính, clip 1,0, bf16 autocast, sdpa, chọn best epoch theo validation.
- **Chỉ fine-tune thường (CE), không KD.** Quy tắc đã chốt: KD từ Cafe chỉ dùng cho ABSA.
- max length 256 cho phân loại câu. ViNLI mã hóa **theo cặp câu** (premise, hypothesis) bằng tokenizer pair encoding. Báo tỷ lệ bị cắt.
- **PhoNER:**
  - Head token = dropout 0,1 → linear, trên hidden state cuối của **cùng encoder pretrained**.
  - Nhãn gán cho **subword đầu tiên** của mỗi âm tiết; subword còn lại là −100.
  - Mã hóa với `is_split_into_words=True`, max length 512 để tránh cắt.
  - Chấm điểm ở mức từ, với từ bị cắt dự đoán là O; báo số lượng từ bị cắt.
  - Cùng optimizer và LR grid như trên.

**Seed:** quét LR ở seed 42 → chọn LR tốt nhất theo validation, riêng cho từng model × task → chạy thêm seed 43 và 44 ở LR đó. `vispam_binary`: 3 seed ở LR tốt nhất của `vispam`, không quét riêng.

**Kiểm tái tạo** (trước khi chạy dataset mới): trainer phân loại câu mới phải tái tạo được run cũ trên `vsfc_topic` seed 42.

- Bami LR 1e-5, so với `artifacts/e14/trials/vsfc_topic/lr_1e-05/seed_42`.
- Student LR 1e-5, so với `artifacts/cafe_oof_kd/student/ce/vsfc_topic/seed_42`.
- Tiêu chí: val F1 **epoch 1–5 lệch ≤ 0,05 điểm** so với `epoch_curves.json` cũ, và best F1 lệch ≤ 1,0. GPU không tất định có thể làm các epoch sau trôi dần. Không đạt thì dừng và báo.

## 4. Khối lượng và ngân sách

- Quét LR seed 42: 3 task × 2 model × 3 LR = 18.
- Seed 43/44: 12.
- `vispam_binary`: 6.
- Tái tạo: 2.
- **Tổng 38 run.** Ước tính khoảng 12 GPU-giờ; **trần 16 GPU-giờ**. Khoảng 3,5–4 giờ chạy thực trên 4 GPU.

## 5. Thứ tự thực hiện (một Codex, tuần tự; GPU dùng chung một hàng đợi)

1. Trainer chung + kiểm tái tạo trên `vsfc_topic`.
2. `vispam`: code → smoke test → đưa vào hàng đợi.
3. `vinli` (hỗ trợ cặp câu): code → smoke test → đưa vào hàng đợi.
4. `phoner` (NER): code → smoke test, gồm unit test cho evaluator thực thể trên ví dụ tự tạo → đưa vào hàng đợi.

Job nào sẵn sàng thì chạy ngay, không chờ cả ba task code xong. Seed 43/44 tự lên hàng đợi khi quét LR của model × task đó xong.

## 6. Ràng buộc

- **Validation-only**; không đọc hay chạy test.
- Không sửa code hay artifact cũ. Code mới trong `scripts/bench8/`, có thể import `src/` và `scripts/` cũ nhưng không sửa. Artifacts trong `artifacts/bench8/`. Báo cáo `docs/BENCH8_NEW_DATASETS_RESULT.md`.
- **GPU 0–3, mỗi GPU đúng một job của mình** (không chồng job), cần ≥ 12 GB trống lúc launch. **Không kill, dừng hay renice process không do mình tạo.**
- Launcher chạy tách phiên (`setsid nohup`) và nhận lại được job đang chạy dở.
- Không commit hay push. Không cài hay gỡ gói. Evaluator NER tự viết, không cần seqeval.

## 7. Đầu ra

**Bảng 8 dataset, 15 chỉ số, theo format paper Bami:** Bami (TB 3 seed) | Student S (TB 3 seed) | chênh lệch | số tham số.

- 5 dataset cũ: lấy từ artifacts đã có, theo cùng quy tắc (KD cho ABSA, CE cho VSFC).
  - Student: `artifacts/cafe_oof_kd/student/{ce,kd}/<task>/seed_<s>/best_validation_metrics.json`.
  - Bami seed 42: `artifacts/e14/trials/<task>/lr_<LR>/seed_42/`.
  - Bami seed 43/44: `artifacts/cafe_oof_kd/bami/<task>/seed_{43,44}/`.
- 3 dataset mới: từ plan này.
- Kèm theo: range theo seed, LR đã chọn, tỷ lệ bị cắt, GPU-giờ.
- Giới hạn cần ghi rõ: ViNLI là mirror; test chưa mở; đây là validation; dataset chỉ dùng cho nghiên cứu.
