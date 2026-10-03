# Plan: student E7 + KD (nhãn mềm Cafe ngoài fold) cho 3 dataset mới — ViSpam, ViNLI, PhoNER_COVID19

Chốt ngày 2026-09-27. Người dùng đã chốt **một strategy duy nhất cho student trên cả 8 dataset: E7 (LLRD γ 0,8) + KD từ CafeBERT, dùng nhãn mềm ngoài fold (OOF), α = 0,5, T = 2 cố định**.

Plan này làm các phần **bắt buộc** còn thiếu:

1. Nhãn mềm Cafe 5-fold cho ViSpam, ViNLI, PhoNER.
2. Student E7 + KD, 3 seed, trên 3 dataset đó.
3. Cập nhật bảng 8 dataset / 15 chỉ số theo đúng một strategy.

**Không mở test.** Việc test cần người dùng duyệt riêng. Thực thi: một Codex (gpt-6-sol, reasoning high) theo skill codex-handoff; root review.

## 0. Tái dùng — đọc trước, không sửa file cũ

| Thành phần | Nguồn |
|---|---|
| Dữ liệu và loader 3 task, id ổn định `<task>:<split>:<i>` | `scripts/bench8/data.py` (đã kiểm sha256) |
| Trainer student/Bami, CE, LLRD, căn nhãn NER, `TokenModel` | `scripts/bench8/train.py` |
| Evaluator thực thể NER, micro, khớp chặt + unit test | `scripts/bench8/ner.py`, `scripts/bench8/test_ner.py` |
| Chia fold (`default_rng(42)`, thứ hạng mod 5), `kd_term`, `write_json` | `scripts/cafe_oof_kd/common.py`, `scripts/cafe_oof_kd/student.py` |
| Recipe Cafe teacher: LR 2e-5, 30 epoch, batch 32, max length 256, bf16, head first-token → dense1024 → tanh; chọn best epoch theo validation | `scripts/sealion_benchmark/{common,train}.py`, `scripts/cafe_oof_kd/teacher_fold.py` |
| Checkpoint Cafe: `uitnlp/CafeBERT@af76fcf2a04096b2b54b348a3e4eb48253c93c5d`; tokenizer `FacebookAI/xlm-roberta-large@c23d21b0620b635a76227c604d44e43a9f0ee389` | HF cache local, `HF_HUB_OFFLINE=1` |
| LR student đã chọn ở bước CE (giữ nguyên, không quét lại) | ViSpam **5e-5**, ViNLI **1e-4**, PhoNER **1e-4** |
| Kết quả CE, Bami và 5 task cũ | `artifacts/bench8/`, `artifacts/cafe_oof_kd/` |

Code mới đặt trong `scripts/bench8_kd/`, được import code cũ. Artifacts đặt trong `artifacts/bench8_kd/`. Báo cáo: `docs/BENCH8_KD_RESULT.md`.

## 1. ViSpam (4 lớp, `SpamLabel`) và ViNLI (cặp câu)

**Teacher Cafe 5-fold:**

- Fold gán theo **thứ tự train của `bench8/data.py`**, dùng đúng `fold_assignment` của `cafe_oof_kd/common.py`.
- Mỗi fold train Cafe trên 4/5 train với recipe Cafe ở mục 0. ViNLI mã hóa **cặp câu** bằng tokenizer XLM-R, cùng cách `bench8` làm cho Bami/student. Max length 256; báo tỷ lệ bị cắt.
- Chọn best epoch theo validation chính thức, như các teacher đã chạy.
- Xuất logits cho fold giữ lại. Ghép lại phải **phủ đủ 100% id train, không trùng**.
- Báo chất lượng teacher:
  - val F1 của từng fold model, mean và range;
  - F1 OOF so với gold train;
  - tỷ lệ bất đồng với gold;
  - xác suất lớn nhất trung bình, và tỷ lệ ≥ 0,99;
  - riêng ViSpam, thêm F1 theo từng lớp OOF, vì lớp 1 rất hiếm.

**Student KD:**

- Loss = 0,5 · CE(gold) + 0,5 · T² · KL(softmax(teacher/T) ‖ log_softmax(student/T)), T = 2, tra nhãn mềm theo id ví dụ.
- LLRD γ 0,8; LR như mục 0; seed 42/43/44; recipe còn lại **giống hệt nhánh CE của `bench8`**, để KD − CE chỉ khác ở loss.
- `vispam_binary` không làm KD: đây là chỉ số phụ, ghi rõ trong báo cáo.

## 2. PhoNER_COVID19 — plan chi tiết

### 2.1 Thách thức chính

1. **Hai tokenizer khác nhau.** Student dùng tokenizer 20.481 token; Cafe dùng SentencePiece XLM-R 250K. Nhãn mềm theo subword **không so được trực tiếp** giữa hai model.
   → Làm KD ở **mức âm tiết (word-level)**, vì danh sách âm tiết là chung.
2. **Căn nhãn:** mỗi model lấy logits tại **subword đầu tiên** của mỗi âm tiết, đúng như `bench8` đang dùng cho CE.
3. **Âm tiết có thể không sinh subword nào**, hoặc bị cắt do max length. Hai trường hợp phải được phát hiện và loại khỏi KD; không được lệch chỉ số.
4. **Không gian nhãn** phải trùng tuyệt đối giữa teacher và student: 21 nhãn BIO, đúng thứ tự.

### 2.2 Không gian nhãn

- Định nghĩa **một danh sách nhãn chuẩn** (21 nhãn, thứ tự như `bench8` đang dùng: sắp xếp từ tag trong train) ở một chỗ duy nhất.
- Teacher, student và evaluator đều import danh sách đó. Assert độ dài 21 và nội dung trùng với `bench8` hiện tại.

### 2.3 Teacher Cafe cho NER

- **Model:** encoder CafeBERT, nạp bằng `AutoModel.from_pretrained` từ snapshot đã pin, `add_pooling_layer=False`, sdpa. Head token giống `TokenModel` của `bench8`: dropout 0,1 → linear(1024 → 21).
  - Assert nạp strict: không có key thiếu hoặc thừa ngoài pooler/lm_head.
- **Mã hóa:** tokenizer XLM-R, `is_split_into_words=True`, max length 512, căn nhãn subword đầu bằng `word_ids()`, các subword còn lại gán −100. Tái dùng hoặc sao chép logic `make_custom_loader` của `bench8`.
- **Kiểm trước khi train** (ghi `artifacts/bench8_kd/phoner/alignment_check.json`), trên cả train và dev, cho **cả hai tokenizer**:
  - số âm tiết không có subword nào;
  - số âm tiết bị cắt;
  - số câu bị cắt;
  - danh sách 20 ví dụ đầu nếu có.

  Kỳ vọng cả ba là 0. Bước CE trước đã đo 0 với tokenizer student; với tokenizer Cafe phải đo mới.
  Nếu khác 0: âm tiết đó **không có nhãn mềm** (mask khỏi KD, vẫn tính CE nếu student có subword). Ghi số lượng, **không** tự chế nhãn mềm.
- **Recipe:** như Cafe ở mục 0 nhưng cho bài gán nhãn token.
  - AdamW với LR 2e-5 cố định (giống các teacher Cafe khác, không quét), 30 epoch, batch 32, warmup 10%, clip 1, bf16.
  - CE trung bình trên các vị trí có nhãn.
  - Chọn best epoch theo **F1 thực thể trên dev**, dùng `bench8/ner.py`.
- **5-fold theo câu:** `fold_assignment` trên thứ tự câu train của `bench8/data.py`.
- **Xuất nhãn mềm** cho fold giữ lại: với mỗi câu, ma trận **logits theo âm tiết `[n_words, 21]`**, lấy tại subword đầu. Âm tiết không có vị trí thì để NaN và kèm cờ `has_logit`.
  - Lưu dạng ragged: `sentence_ids`, `offsets`, `logits` nối liền, `has_logit`, float32.
  - Ghép 5 fold; **assert** mỗi câu train xuất hiện đúng 1 lần, và `n_words` trùng độ dài `words` của câu.
- **Chất lượng teacher** (ghi `teacher_stats.json`):
  - dev F1 của từng fold model, mean và range;
  - **F1 thực thể OOF trên train** (lấy argmax teacher theo âm tiết → nhãn → `ner.py`);
  - accuracy theo âm tiết;
  - tỷ lệ bất đồng với gold, tách riêng các âm tiết gold là thực thể;
  - xác suất lớn nhất trung bình, tỷ lệ ≥ 0,99;
  - bảng F1 theo loại thực thể.

### 2.4 Student KD cho NER

- Model `TokenModel` với `goiturn-s-gd`, như `bench8`. Căn nhãn subword đầu như cũ.
- Trong mỗi batch, gom logits student tại vị trí subword đầu → ma trận `[n_words_valid, 21]`, rồi ghép với nhãn mềm teacher của đúng (câu, âm tiết).
  - Tra theo id câu và chỉ số âm tiết, **không** theo vị trí trong batch.
- **Loss:**
  - CE: như `bench8` hiện tại, trung bình trên các vị trí subword đầu có nhãn.
  - KD: T² · KL(softmax(teacher/T) ‖ log_softmax(student/T)), **trung bình trên đúng tập âm tiết** có cả vị trí student lẫn `has_logit` của teacher.
  - Tổng = 0,5 · CE + 0,5 · KD. Ghi riêng CE và KD vào `epoch_curves` cho từng epoch.
- LLRD γ 0,8, LR 1e-4, seed 42/43/44, max length 512, 30 epoch, chọn best theo dev F1 thực thể, postrun checks như `bench8`.

### 2.5 Kiểm tra bắt buộc cho NER (unit test + smoke test)

1. **Unit test gom vị trí:** batch 2 câu có padding, một âm tiết tách thành nhiều subword, một câu bị cắt (tự tạo). Ma trận gom ra phải đúng thứ tự âm tiết, đúng số lượng, và bỏ âm tiết bị cắt.
2. **Unit test ánh xạ nhãn mềm:** 3 câu giả, logits giả biết trước. Tra đúng (câu, âm tiết) sau khi batch bị xáo trộn thứ tự.
3. **Unit test loss:** nếu teacher = one-hot của gold nhân một hằng số lớn, thì KD ≈ CE × hệ số tương ứng, gradient hữu hạn. Nếu teacher logits bằng chính student logits, KD ≈ 0.
4. **Smoke teacher:** 1 epoch, vài batch, xuất OOF của 1 fold ra `/tmp` riêng; kiểm shape, `has_logit` và `n_words`.
5. **Smoke student KD:** 1 epoch với nhãn mềm giả (one-hot gold); loss hữu hạn, postrun checks PASS.
6. **Kiểm chéo kích thước:** với 20 câu ngẫu nhiên, số âm tiết teacher = số âm tiết student = `len(words)`.

Bất kỳ kiểm tra nào FAIL thì dừng NER và báo; ViSpam và ViNLI vẫn chạy tiếp.

## 3. Thứ tự thực hiện và hàng đợi GPU

1. **ViSpam:** teacher 5-fold → ghép → student KD 3 seed.
2. **ViNLI:** giống ViSpam, thêm hỗ trợ cặp câu cho Cafe.
3. **PhoNER:** kiểm căn nhãn → unit test → smoke → teacher 5-fold → ghép → student KD.

Launcher dùng chung một hàng đợi:

- Job nào sẵn sàng thì chạy ngay. Student KD của một task chỉ vào hàng đợi khi file nhãn mềm của task đó đã ghép xong và qua kiểm.
- Chạy tách phiên (`setsid nohup`), **GPU 0–3, mỗi GPU đúng một job của mình, cần ≥ 20 GB trống** (teacher Cafe cần khoảng 16 GB).
- Nhận lại job đang chạy dở; bỏ qua ô đã xong; job lỗi không làm dừng cả loạt.
- **Không kill, dừng hay renice process không do mình tạo.**
- Ngân sách: **ước tính khoảng 12 GPU-giờ, trần 18 GPU-giờ**.

## 4. Báo cáo và bảng cuối

- **Chất lượng teacher** cho từng task, gồm tách riêng lớp hiếm của ViSpam và từng loại thực thể của NER.
- **KD − CE** cho student, 3 seed (mean, range, và số seed dương) ở cả 3 task. **Không chọn giữa CE và KD** theo từng task: strategy đã chốt là KD, còn CE chỉ để minh bạch.
- **Bảng 8 dataset / 15 chỉ số, student E7 + KD so với Bami**, trung bình 3 seed validation, kèm range và số tham số:
  - 5 task cũ: nhánh KD trong `artifacts/cafe_oof_kd/student/kd/`, gồm cả Sentiment và Topic;
  - 3 task mới: plan này.
- Ghi rõ: **validation-only**; ViNLI là mirror chưa xác minh; dataset chỉ dùng cho nghiên cứu; teacher chọn epoch theo validation; α và T không tinh chỉnh.
