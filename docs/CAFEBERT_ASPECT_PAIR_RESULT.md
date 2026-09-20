# CafeBERT aspect-pair — kết quả pilot Restaurant

Ngày 2026-09-18. Thực thi theo `docs/CAFEBERT_ASPECT_PAIR_PLAN.md`. Task `absa_restaurant`, seed 42, train 7028 / validation 771,
không chạm test. Hai lượt fine-tune (A, B), một LR, không DAPT/KD/NC/LLRD/composite, không quét prompt/LR, không thêm seed.

**Ghi chú thực thi:** theo chỉ dẫn mới nhất trong `AGENTS.md` (2026-09-17), agent chính (Claude Opus 5) tự triển khai và chạy toàn bộ,
không tạo subagent. Câu hỏi của pilot: *đưa aspect vào đầu vào encoder có giúp CafeBERT vượt reference 77.7197 validation không?*

## 1. Kết quả

| Arm | Input encoder | Head | Best epoch | Val pair micro-F1 | Pair P / R | Detection F1 | Pair macro-F1 | Runtime | GPU peak |
|---|---|---|---|---|---|---|---|---|---|
| Reference T | review → 48 logits (12 slot × 4) | 48 | 21 | 77.7197 | 77.76 / 77.68 | 87.89 | 75.49 | 23.3 min (cuda:2, chia sẻ) | 10.8 GB |
| **A** — aspect ID | `(review, "mục 01"…"mục 12")` | shared 4 | 23 | **78.4536** | 77.90 / 79.01 | 89.20 | 76.71 | 113.5 min (cuda:3, chia với B) | 17.0 GB |
| **B** — aspect description | `(review, mô tả tiếng Việt)` | shared 4 | 17 | **75.9008** | 74.82 / 77.02 | 87.41 | 74.41 | 117.6 min (cuda:3, chia với A) | 17.3 GB |

Mọi F1 ở trên được **tính lại** từ `validation_predictions.json` (12 slot đã gộp) bằng evaluator gốc `goibert.metrics.evaluate_task`
(`scripts/cafebert_aspect_pair/summarize.py`), khớp số đã lưu trong `status.json` đến 1e-9.

| Delta | pp | Đọc theo gate của plan §7 |
|---|---|---|
| A − T | **+0.73** | ≥ 0.5 → *tín hiệu đáng xác nhận seed ở lượt sau*; chưa tự chạy xác nhận |
| B − T | **−1.82** | giảm → không có tín hiệu trong cấu hình này |
| B − A | **−2.55** | mô tả có ngữ nghĩa **không** hơn mã ID → không quy gain cho tên aspect |

Trả lời câu hỏi pilot: **có, với arm A (mã aspect)**, +0.73 pp trên validation seed đơn; **không, với arm B (mô tả aspect)**. Vì B < A rõ,
phần gain của A không thể quy cho nội dung ngữ nghĩa của mô tả; nó chỉ có thể đến từ cách đặt bài toán (phân loại từng aspect với head
chung 4 lớp, encoder đọc review 12 lần) hoặc nhiễu seed/best-epoch (xem §2).

## 2. Đường cong và độ ổn định (validation pair F1 theo epoch, đầy đủ trong `epoch_curves.json`)

| Epoch | 1 | 2 | 5 | 10 | 13 | 17 | 18 | 21 | 22 | 23 | 25 | 30 | TB 10–30 | TB 17–30 (bỏ best) | Second-best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T | 28.20 | 56.71 | 72.83 | 76.39 | 75.15 | 74.57 | 76.70 | **77.72** | 76.24 | 76.15 | 76.15 | 76.10 | 76.14 | 76.15 (bỏ ep 21) | 76.99 |
| A | 19.63 | 57.41 | 73.03 | 74.19 | 76.24 | 76.72 | 77.22 | 76.80 | 77.26 | **78.45** | 77.21 | 76.64 | 76.41 | 76.79 (bỏ ep 23) | 77.26 |
| B | 40.77 | 69.92 | 75.45 | 74.75 | 75.13 | **75.90** | 75.57 | 75.64 | 75.09 | 75.85 | 75.31 | 75.50 | 75.25 | 75.55 | 75.85 |

- **A:** vượt 77.72 chỉ ở **một epoch (23)**; 13 epoch còn lại của giai đoạn 17–30 nằm 76.4–77.3 (TB 76.79, max 77.26). So với T cùng
  giai đoạn (TB 76.15, max 76.94 bỏ best), A cao hơn ~0.3–0.6 pp trên plateau và second-best A (77.26) > second-best T (76.99). A ≥ T ở
  19/30 epoch cùng chỉ số. Nghĩa là: có xu hướng nhất quán nhỏ ở plateau, nhưng **con số +0.73 của best-epoch chủ yếu là một đỉnh đơn
  lẻ** (epoch 24 rớt về 76.68) — đúng cỡ nhiễu seed 0.5–1 F1 đã ghi nhận ở E21/E22. Không coi là xác nhận.
- **B:** không vượt T ở bất kỳ epoch nào; plateau 74.5–75.9 từ epoch 5; B ≥ T chỉ 8/30 epoch (chủ yếu epoch đầu). B học nhanh hơn
  hẳn ở epoch 1–2 (40.8 / 69.9 so với A 19.6 / 57.4 và T 28.2 / 56.7) — mô tả ngữ nghĩa giúp khởi đầu, không giúp đích.
- Train loss cuối: A 0.0002, B 0.0006, T 0.0019 — cả hai arm pair fit train sâu hơn T (mỗi review-slot là một ví dụ riêng).
- A dự đoán 1068 cặp non-absent (gold 1053, T 1052), B 1084: A/B nghiêng recall hơn T; A tăng cả detection F1 (+1.3) lẫn pair macro (+1.2).

## 3. Per-aspect pair F1 (validation, best epoch)

| Aspect | gold | T | A | B | A − T | B − T |
|---|---:|---:|---:|---:|---:|---:|
| AMBIENCE#GENERAL | 91 | 78.92 | 80.66 | 81.97 | +1.74 | +3.05 |
| DRINKS#PRICES | 19 | 61.90 | 71.79 | 68.42 | +9.89 | +6.52 |
| DRINKS#QUALITY | 81 | 77.11 | 73.81 | 72.94 | −3.30 | −4.17 |
| DRINKS#STYLE&OPTIONS | 52 | 73.08 | 74.51 | 66.02 | +1.43 | −7.06 |
| FOOD#PRICES | 45 | 82.98 | 80.46 | 79.57 | −2.52 | −3.41 |
| FOOD#QUALITY | 222 | 83.41 | 84.62 | 80.53 | +1.21 | −2.88 |
| FOOD#STYLE&OPTIONS | 175 | 75.15 | 75.43 | 68.79 | +0.28 | −6.36 |
| LOCATION#GENERAL | 42 | 77.65 | 79.07 | 84.71 | +1.42 | +7.06 |
| RESTAURANT#GENERAL | 100 | 63.32 | 69.57 | 67.00 | +6.25 | +3.68 |
| RESTAURANT#MISCELLANEOUS | 58 | 77.97 | 73.33 | 74.60 | −4.63 | −3.36 |
| RESTAURANT#PRICES | 47 | 65.12 | 68.04 | 58.06 | +2.92 | −7.05 |
| SERVICE#GENERAL | 121 | 89.26 | 89.26 | 90.32 | 0.00 | +1.07 |

Gain của A tập trung ở RESTAURANT#GENERAL (+6.3, 100 gold) và các aspect nhỏ (DRINKS#PRICES 19 gold: +9.9); lỗ ở RESTAURANT#MISC, DRINKS#QUALITY,
FOOD#PRICES. B mất nhiều nhất ở hai aspect STYLE&OPTIONS (−7.1, −6.4) và RESTAURANT#PRICES (−7.1) — đúng các aspect có mô tả dài
(9 token "cách trình bày và sự lựa chọn …") hoặc dễ nhầm với FOOD/DRINKS#PRICES; đây là quan sát, không phải kết luận nhân quả. Slot agreement
với T: A 97.50 %, B 97.41 %; A với B 97.27 %.

## 4. Kiểm correctness (tất cả pass; `artifacts/cafebert_aspect_pair/checks/preflight.json`)

| Kiểm | Kết quả |
|---|---|
| Mapping | 12 entry/arm, thứ tự = `RESTAURANT_ASPECT_NAMES` của loader; A = "mục 01"…"mục 12"; B đúng bảng plan §3; không token mới |
| Expand → regroup | train 7028 → 84 336 pairs, validation 771 → 9 252; mỗi review đủ 12 slot (absent giữ nguyên, 74 878 / 8 199 pair absent); key `(id, slot)` duy nhất; regroup (kể cả sau shuffle) **bằng chính xác** vector gold; thiếu/thừa key → raise |
| Nhãn | target = `gold[slot]` integer 0..3 từ loader, không recode; histogram train 74878/5179/1746/2533 |
| Pair encoding | `tokenizer(review, aspect)` chính thức → `<s> review </s></s> aspect </s>`; kiểm cấu trúc trên 4 800 pair mẫu; **0 pair bị truncation** ở cả 4 (split × arm) (max 131 token < 256) → yêu cầu giữ nguyên câu được thỏa; không `<unk>` trong aspect text; prompt A 2 token, B 3–9 token (mean len pair A 27.2, B 30.5) |
| Review order | sampler = `DataLoader(shuffle, Generator(42))` như reference → hoán vị mỗi epoch **trùng bit** reference sampler (so 3 epoch); 220 group/epoch (219 × 32 + 1 × 20 review = 240 pair); A và B cùng hash order toàn 30 epoch (`ed771c9b…`) |
| Init | encoder = `uitnlp/CafeBERT@af76fcf2` cùng file safetensors với T (`a7306a56…`), state **khác** encoder của `T/best_checkpoint.pt` (không dùng checkpoint fine-tuned); head init reproducible và giống nhau A/B (`8d1ceb77…`), đổi seed thì đổi; 559 894 532 tham số trainable (T: 559 939 632, chênh đúng head 48 → 4 logit) |
| Numerics | bf16 autocast forward/backward hữu hạn, mọi tham số có grad; **weighting accumulate**: Σ CE_mean·n_micro/n_group = CE_mean toàn group (lệch 1e-8), gradient lệch L2 tương đối 2.5e-5 so với full-batch fp32 (padding) |
| Budget cập nhật | 220 step/epoch, 6 600 tổng, warmup 660, LR cuối 0.0, `global_step` = 6 600 ở cả hai arm (khớp reference, không nhân 12) |
| Checkpoint | model-only (`cafebert-aspect-pair-model-only-v1`), reload → predictions giống hệt; post-run mỗi arm: reload best → F1 bằng best, F1 tính lại từ file predictions bằng best, predictions reload giống hệt, ID/gold trong file khớp loader |
| Smoke | `finetune.py --smoke 3` (3 group, 12 review validation) chạy hết pipeline + post-run checks trước full run |

`source_hash` của hai run (`07614b12…`) khác preflight (`e796a45a…`) chỉ vì `summarize.py` được thêm khi preflight đang chạy;
`common.py`/`finetune.py` không đổi từ preflight tới khi cả hai run kết thúc (mtime trước launch).

## 5. Triển khai và vận hành

- Batch hiệu dụng **32 review = 384 pair**: mỗi group expand đủ 12 slot, chia microbatch, mỗi microbatch đóng góp `CE_mean · n_pairs_micro / n_pairs_group`,
  một lần clip/optimizer/scheduler sau cả group; group cuối 20 review chuẩn hóa theo 240 pair.
- **Microbatch 64 pair** (plan đề xuất 16; đổi theo profile, ghi trước launch trong `checks/launch_decision.json`, dùng chung A/B).
  Profile 1 step 384 pair trên cuda:3 rảnh (arm B): 16 → 2.02 s, 32 → 1.04, 48 → 0.71, 64 → 0.55 (16.3 GiB), 96 → 0.46. Gradient là
  cùng weighted mean bất kể microbatch (đã kiểm §4); chỉ dropout mask/kernel numerics khác, plan không hứa bitwise.
- ETA từ profile: ~2.0 min train + 3 s validation/epoch khi chạy một mình (~1 h/arm). Thực tế: hai arm chạy **song song trên cuda:3**
  (GPU duy nhất không có process người khác; GPU 0/1/2 đang có job ngoài, plan yêu cầu không dừng): 221 s + 6 s/epoch (A), 236 s + 7 s (B)
  → 113.5 / 117.6 min, tức **~5× reference** (23.3 min) cho ~12× số encoder pass/review nhờ microbatch lớn và review ngắn (p50 25–28 token).
- Peak GPU: A 17.0 GB allocated / 18.7 GB reserved; B 17.3 / 19.8 GB. Recipe còn lại đúng plan §5: LR 2e-5, 30 epoch, AdamW (0.9, 0.999,
  eps 1e-8, wd 0.01, bias/LayerNorm không decay), warmup 10 % + linear, clip 1.0, FP32 master + BF16 autocast, SDPA, CE 4 lớp.

## 6. Giới hạn diễn giải

1. **Seed đơn, chọn best-epoch trên validation**: +0.73 của A đến từ một epoch đỉnh; plateau chỉ hơn T ~0.3–0.6. Nhiễu seed đã đo ở
   E21/E22 cỡ 0.5–1 F1. Gate ≥ 0.5 là heuristic đầu tư, không phải significance.
2. **Không cô lập compute/head**: A/B đọc review 12 lần và dùng head 4 lớp chung; reference đọc 1 lần với head 48. Không tách được phần
   gain do "aspect trong input" khỏi phần do "12 ví dụ/review + head chung".
3. **B < A** loại bỏ giả thuyết "ngữ nghĩa mô tả tạo gain" *cho wording đã khóa*; không nói gì về wording khác (không quét).
   Mã "mục NN" là định danh học được, không phải đối chứng không thông tin.
4. Chi phí inference 12× reference; runtime train ~5× trên GPU chia sẻ.
5. Chỉ Restaurant, chỉ validation; không suy ra Hotel/ViSFD hay test.

## 7. Quyết định

Theo gate của plan §7: **dừng sau 2 run**. max(A, B) − T = +0.73 ≥ 0.5 → *A là ứng viên đáng xác nhận seed ở lượt sau* (seed 43/44 cho
A và, để cô lập, có thể cân nhắc cùng lúc reference T seed 43/44); **không** tự chạy xác nhận, không mở test, không quét wording/LR,
không mở task khác. B không có tín hiệu trong cấu hình này. Nếu xác nhận seed cho thấy A chỉ ngang T, thì kết luận là cách đặt bài toán
pair không thay đổi mức CafeBERT trên Restaurant với chi phí 12×.

## 8. Artifacts

- Runs: `artifacts/cafebert_aspect_pair/runs/{A,B}_absa_restaurant_seed42/` — `resolved_config.json` (kèm mapping đã khóa),
  `epoch_curves.json` (kèm hash review-order mỗi epoch), `best_checkpoint.pt` (model state_dict, 2.2 GB, không resumable; sha256 A `b1067607…`,
  B `d61ed2a1…`), `best_validation_metrics.json`, `validation_predictions.json` (771 hàng `{id, gold, prediction}` 12 slot đã gộp),
  `metadata.json` (postrun_checks), `status.json`, `train.log`.
- Checks: `artifacts/cafebert_aspect_pair/checks/{preflight.json, preflight.log, launch_decision.json}`; smoke: `artifacts/cafebert_aspect_pair/smoke/A/`
  (checkpoint smoke đã xóa).
- Tổng hợp: `artifacts/cafebert_aspect_pair/summary.json` (`python scripts/cafebert_aspect_pair/summarize.py`) — F1 tính lại, deltas, gate,
  per-aspect, agreement, kiểm cùng head-init/review-order A–B.
- Code: `scripts/cafebert_aspect_pair/{common,finetune,preflight,summarize}.py`. Reuse: loader/evaluator/optimizer E1 (`src/goibert/train.py`,
  `data.py`, `metrics.py`), `ClassificationHead` E14 (`src/goibert_e14/modeling.py`), tokenizer/source contract CafeBERT của
  `scripts/cdbk_pilot/common.py`. Không sửa source, gold hay artifacts cũ; reference T không train lại.

## 9. Root review độc lập

Root tính lại F1 từ saved predictions của T/A/B, kiểm đúng771IDs và numeric gold theo frozen validation loader; tất cả khớp đến1e-12. Không chạy model inference hoặc training mới. Số đếm pair đúng T818/A832/B811; số pair dự đoán T1052/A1068/B1084; gold1053. So với T, A thêm14true positives và2false positives tại các best checkpoints.

A chỉ vượt best T ở epoch23. Tuy nhiên, T cũng có một đỉnh riêng: second-best T76.99, A77.26. Trung bình epoch17–30 là T76.2581/A76.9131; loại best của từngarm là T76.1457/A76.7946. Đây là mô tả hậu kiểm đường cong, không dùng chọn lại checkpoint và không phải ước lượng nhiễu seed. Dao động qua epoch không chứng minh gain do seed noise; cần paired fresh seed để kiểm tra.

B−A âm không hỗ trợ lợi ích của bộ mô tả đã khóa; chưa chứng minh nguyên nhân là độ dài hoặc ngữ nghĩa mơ hồ. So A với T còn khác head/input/compute.12encoder passes không tự có nghĩa latency chính xác12lần; chưa có benchmark inference latency khớp điều kiện. Thời gian train chia sẻGPU không dùng để tuyên bố speedup/slowdown kiến trúc tuyệt đối.

Quyết định đề xuất: giữ A là ứng viên, dừng B. Bước xác nhận nhỏ nhất là T vàA cùngseed43, giữ cấu hình tương ứng; chưa launch và không mởtest/wordinggrid. Evidence audit: `artifacts/cafebert_aspect_pair/root_review.json`.
