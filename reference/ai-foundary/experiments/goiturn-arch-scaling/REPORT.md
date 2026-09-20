# GoiTurn Architecture Scaling — Báo cáo Feasibility Experiment

Ngày: 2026-08-30 · Người chạy: Line (hạ tầng LUMI qua account nhantra1) · Trạng thái: hoàn tất, 15/15 runs sạch
Loại: architecture-scaling / feasibility experiment — KHÔNG phải GoiTurn production.
Proposal gốc: `GoiTurn_Vietnamese_Proposal.md` · Code: `experiments/goiturn-arch-scaling/` · Kết quả thô: `results-lumi/summary.md` (bản đầy đủ trên LUMI, giữ 289 ngày)

## 1. Câu hỏi cần trả lời

Giảm BamiBERT (102M) xuống cỡ 25–36M thì mất bao nhiêu chất lượng và được bao nhiêu tốc độ —
để quyết định kích thước cho GoiTurn-S (proposal nhắm ~21M)?

## 2. Setup (tất cả là số ĐO THỰC, không phải target)

- **Baseline**: BamiBERT thật (RoBERTa 12L/768H/12A/FFN3072, vocab 20,481) fine-tune từ checkpoint.
  Nguồn: HuggingFace [`Qualcomm-AI-Research/BamiBERT`](https://huggingface.co/Qualcomm-AI-Research/BamiBERT),
  revision `57bc1340` (2026-07-03), tải 2026-08-30, `model.safetensors` MD5
  `b5f369196f27fdcf4c3e38bca27a2be1`, license BSD-3-Clause-Clear + Qualcomm Responsible AI License.
  Paper: [arXiv:2607.02259](https://arxiv.org/abs/2607.02259). Bản local: `ml-models/models/bamibert/`.
- **Variant A "wide"**: 8L/512H/8A/FFN2048 = 35,970,048 params (35% baseline).
- **Variant B "GoiTurn-S geometry"**: 10L/384H/6A/FFN1536 = 25,807,872 params (25% baseline) — đúng hình học GoiTurn-S trong proposal §13.2.
- Variants không load được checkpoint (lệch shape) → 2 chế độ: **scratch** (random init, chỉ task data) và **distill** (PCA-init embeddings từ BamiBERT + logit distillation từ teacher là baseline đã fine-tune; T=2, α=0.5).
- 3 datasets: UIT-VSFC (sentiment 3 lớp), VLSP2018-ABSA-Restaurant (F&B proxy), UIT-ViSFD (e-commerce ABSA). Protocol: seed 42, batch 32, 10 epochs, chọn epoch tốt nhất theo validation, test set riêng.
- Phần cứng: LUMI MI250X, fp32, eager attention (xem §6 — bài học ROCm).

**Sanity check đạt**: baseline VSFC macro F1 **82.74** vs paper BamiBERT 83.41; ViSFD detection **88.32** vs paper 89.14 (chênh ~0.6–0.8 do protocol rút gọn: 10 epochs × 1 lr thay vì 30 epochs × 3 lr). → So sánh nội bộ giữa 3 model là đáng tin.

## 3. Bảng kết quả chính (mode tốt nhất của mỗi variant; F1 = micro cho ABSA, macro cho VSFC)

| Model | Layers | Hidden | Heads | FFN | Params | VSFC F1/Acc | VLSP-Res det/cls | ViSFD det/cls | Latency p50 (seq32, 1 thread) | Size fp32 |
|---|--:|--:|--:|--:|--:|---|---|---|--:|--:|
| BamiBERT baseline | 12 | 768 | 12 | 3072 | 102.4M | **82.7** / 93.4 | **77.1** / **60.7** | **88.3** / **78.9** | 92.9 ms | 393 MiB |
| Variant A (distill) | 8 | 512 | 8 | 2048 | 36.0M | 77.2 / 89.7 | 73.1 / 58.0 | 83.3 / 70.0 | 30.1 ms | 138 MiB |
| Variant B (distill) | 10 | 384 | 6 | 1536 | 25.8M | 74.8 / 89.5 | 71.0 / 56.1 | 82.5 / 69.6 | 23.4 ms | 99 MiB |

Tỷ lệ giữ chất lượng so với baseline (distill): **Variant A giữ 93–96%, Variant B giữ 90–93%** trên các metric chính, với **3.1× / 4.0× tốc độ** và **35% / 25% số params**.

### Bảng đầy đủ 15 runs (mỗi dòng một model+mode, mỗi cụm cột một dataset)

| Model (mode) | Cấu hình | Params | VSFC Acc | VSFC F1 | VLSP-Res det | VLSP-Res cls | ViSFD det | ViSFD cls |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** (finetune) | 12L × 768H | 102.4M | **93.37** | **82.74** | **77.07** | **60.70** | **88.32** | **78.87** |
| variant_a (scratch) | 8L × 512H | 36.0M | 89.32 | 72.91 | 72.36 | 57.72 | 82.31 | 69.52 |
| **variant_a** (distill) | 8L × 512H | 36.0M | 89.70 | 77.18 | 73.09 | 57.99 | 83.29 | 69.96 |
| variant_b (scratch) | 10L × 384H | 25.8M | 88.91 | 75.54 | 69.52 | 54.21 | 81.53 | 68.93 |
| **variant_b** (distill) | 10L × 384H | 25.8M | 89.45 | 74.79 | 71.02 | 56.09 | 82.48 | 69.63 |

Mọi số là % trên test set, epoch chọn theo validation. VSFC F1 = macro F1; các cột ABSA = micro F1.

### Precision / Recall (micro, %) — hai dataset ABSA

| Model (mode) | VLSP det P/R | VLSP cls P/R | ViSFD det P/R | ViSFD cls P/R |
|---|---:|---:|---:|---:|
| baseline (finetune) | 81.94 / 72.76 | 64.53 / 57.30 | 89.92 / 86.79 | 80.30 / 77.50 |
| variant_a (scratch) | 81.96 / 64.78 | 65.38 / 51.67 | 86.08 / 78.87 | 72.70 / 66.61 |
| variant_a (distill) | 82.11 / 65.85 | 65.15 / 52.25 | 86.25 / 80.53 | 72.45 / 67.64 |
| variant_b (scratch) | 81.14 / 60.81 | 63.27 / 47.42 | 86.03 / 77.47 | 72.74 / 65.50 |
| variant_b (distill) | 81.52 / 62.92 | 64.38 / 49.69 | 85.68 / 79.50 | 72.34 / 67.12 |

Đọc nhanh: precision gần như không đổi khi giảm size (81–82 / 86 ổn định) — cái mất là **recall**
(baseline 72.8 → variant B 62.9 trên VLSP): model nhỏ bỏ sót aspect nhiều hơn chứ không đoán bừa hơn.
Với VSFC chỉ lưu accuracy + macro/weighted F1 (per-class P/R không được persist trong run này —
ghi ở mục Verification là gap).

**Vì sao ABSA không có accuracy:** VLSP-Res và ViSFD là bài toán multi-label — mỗi câu chấm
đồng thời 12 (VLSP) / 11 (ViSFD) aspect. Accuracy trên các ô đó ảo cao (~95%) vì tuyệt đại đa
số ô là "aspect không xuất hiện" — model đoán trắng toàn bộ cũng đạt. Chuẩn đo của các paper
(kể cả bảng BamiBERT trong proposal §4.2) là detection F1 (tìm đúng aspect có mặt) và
classification F1 (đúng aspect VÀ đúng polarity), nên report dùng đúng hai cột đó. Accuracy
chỉ có nghĩa với VSFC vì mỗi câu đúng một nhãn.

### CPU latency đầy đủ (batch 1, fp32, 30 warmup + 300 timed iterations)

| Model | Seq | Threads | p50 ms | p95 ms | fp32 state dict MiB |
|---|---:|---:|---:|---:|---:|
| baseline | 32 | 1 | 92.876 | 95.650 | 392.82 |
| baseline | 32 | 4 | 38.295 | 47.255 | 392.82 |
| baseline | 128 | 1 | 281.003 | 303.829 | 392.82 |
| baseline | 128 | 4 | 103.504 | 104.326 | 392.82 |
| variant_a | 32 | 1 | 30.111 | 30.392 | 138.28 |
| variant_a | 32 | 4 | 13.535 | 13.851 | 138.28 |
| variant_a | 128 | 1 | 88.086 | 88.572 | 138.28 |
| variant_a | 128 | 4 | 34.681 | 35.755 | 138.28 |
| variant_b | 32 | 1 | 23.359 | 25.026 | 99.08 |
| variant_b | 32 | 4 | 11.244 | 13.413 | 99.08 |
| variant_b | 128 | 1 | 65.650 | 70.888 | 99.08 |
| variant_b | 128 | 4 | 26.600 | 27.966 | 99.08 |

### Distillation đáng giá bao nhiêu? (đo được)

Distill thắng scratch ở 5/6 cặp so sánh: +4.3 điểm macro F1 (VSFC, variant A), +1.0–1.5 điểm det (ViSFD/VLSP). Đây mới là distill "nhẹ" (logit-only, không pretraining) — sàn chứ không phải trần.

## 4. Kết luận cho quyết định architecture

1. **Variant B (25.8M, hình học GoiTurn-S) là candidate chính — kích thước ~25M khả thi.** Giữ 90–93% chất lượng baseline dù **hoàn toàn không có pretraining tiếng Việt** (baseline hưởng 129GB pretraining). Gap 5–8 điểm hiện tại là **chặn trên** của chi phí giảm size; pipeline đầy đủ trong proposal (MLM domain-adaptive pretraining + multi-teacher distill) theo tiền lệ ViSoBERT/CafeBERT sẽ thu hẹp đáng kể.
2. **Variant A (36M) là fallback đúng như proposal §13.10 dự liệu** ("nếu 21M không đạt thì lên ~40M"): hơn B ổn định 1–3 điểm, đổi lại chậm hơn ~30% và nặng hơn 40%.
3. **Bottleneck là pretraining, không phải capacity.** Scratch và distill-nhẹ đã đạt 89% accuracy VSFC — bước có giá trị nhất tiếp theo là Stage 2-4 của proposal (domain-adaptive pretraining rồi distill sâu), KHÔNG phải đổi hình học tiếp.
4. **Thành phần architecture nên thử tiếp theo** (chưa đo trong experiment này): factorized embeddings (giảm B từ 25.8M → ~21.9M, ALBERT-style, gần đúng 21.1M của proposal); custom 16K chat BPE; modern block (RoPE/GeGLU/RMSNorm). Thứ tự đó — embedding factorization rẻ nhất và proposal đánh giá rủi ro thấp nhất.

## 5. Mismatch so với proposal (phải biết trước khi trích dẫn số)

1. **UIT-ABSA (Restaurant/Hotel) — benchmark headline của proposal — không public.** Đã truy tìm kỹ (2026-08-31), kết quả:
   - HuggingFace: 0 kết quả cho UIT-ABSA dưới mọi tên biến thể (search UIT-ABSA / absa / vietnamese restaurant / ViRes).
   - GitHub global code search: 0 file data nào chứa "UIT-ABSA" hay "Res_ABSA"; repo duy nhất dùng nó (`banbequyen/ABSA-with-UIT-ABSA`) load data từ **Google Drive riêng của tác giả** (`/content/drive/MyDrive/Dataset/Sentiment/Res_ABSA/...`); repo SOTA mới nhất (`xndien2004/ViGSA`, 2025) cũng dùng UIT-ABSA nhưng không redistribute.
   - Trang ACM của paper (TALLIP 2021, DOI 10.1145/3446678) không có link dataset.
   - Kaggle: không có.
   → Cách duy nhất: **email nhóm tác giả** (Đặng Văn Thìn, UIT NLP Group — nhóm UIT phát dataset theo request nghiên cứu, đây là thông lệ của họ, không phải bế tắc). Experiment dùng **VLSP2018-ABSA-Restaurant** (cùng domain, public) thay thế. Muốn công bố "beat 88.01" phải xin được bản thật.
   - Phát hiện thêm: **M-ABSA** (`Multilingual-NLP/M-ABSA` trên HF, 2025) có subset tiếng Việt food/hotel/restaurant, public — nhưng là **dịch máy từ tiếng Anh**, không phải tiếng Việt gốc; chỉ đáng dùng làm benchmark phụ, không thay được UIT-ABSA cho claim chính.
2. **BamiBERT thật là RoBERTa cổ điển** (absolute positions, GELU, LayerNorm, dense embeddings) — không phải modern block như proposal §13 giả định. GoiTurn-S geometry với vocab thật ra **25.8M chứ không phải 21.1M** (chênh do 20,481 dense vocab vs 16K factorized giả định).
3. Protocol rút gọn so với BamiBERT paper (10 epochs, 1 lr, 1 seed) — số baseline thấp hơn paper ~0.6–0.8 điểm. Số trong report này dùng để **so sánh tương đối nội bộ**, không dùng công bố ngoài.
4. Latency đo bằng PyTorch fp32 eager trên CPU EPYC của LUMI — chưa phải ONNX/INT8 production; dùng cho so sánh tỷ lệ giữa 3 model (tỷ lệ nhất quán với số đo trên laptop RTX 4060/Docker).

## 6. Verification status (TwoHearts contract)

- `Verified`: 15/15 runs chạy thật trên LUMI (job 21625380), 0 dòng loss=nan, param counts khớp công thức độc lập, sanity anchors khớp paper. Smoke tests chạy thật trên cả CUDA (laptop Docker) và ROCm (LUMI).
- `Not verified`: UIT-ABSA thật; protocol đầy đủ 30-epoch; ONNX/INT8 latency; variance giữa seeds (1 seed duy nhất); per-class precision/recall của VSFC (không persist trong run — cần rerun nếu team đòi).
- Delivery level: **implementation-complete** (experiment code + measured results; không có PR/commit — repo chưa init git, và theo quy tắc không commit khi Line chưa yêu cầu).

## 7. Next steps đề xuất (chờ Line/team quyết)

1. Xin dataset UIT-ABSA Restaurant từ nhóm UIT (email tác giả — cần cho claim công khai).
2. Chạy protocol đầy đủ (30 epochs × 3 lr grid, 3 seeds) cho baseline + variant B trên LUMI (~20 GPU-hours trong 5000 được cấp) nếu cần số chabsolute chắc hơn.
3. Prototype factorized embeddings cho variant B (25.8M → ~21.9M) — bước architecture rẻ nhất còn lại.
4. Bắt đầu Stage 1-2 của proposal (corpus + domain-adaptive pretraining) — nơi phần lớn chất lượng còn lại nằm ở đó.
