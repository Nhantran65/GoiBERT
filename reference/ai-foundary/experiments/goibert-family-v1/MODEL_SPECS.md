# GoiBERT Family v1 — Đặc tả 3 model

Cập nhật: 2026-09-01. Số param là số đếm thật từ code (`sum(p.numel())`), không ước lượng.
Cả 3 dùng chung: tokenizer BamiBERT (vocab 20,481), block RoBERTa cổ điển (absolute
positions, GELU, LayerNorm), max_position 2050, max_length train 256, seed 42.

## Bảng tổng

| | **GoiTurn-XS** | **GoiTurn-S** | **GoiBERT-B** |
|---|---|---|---|
| **Vai trò** | Edge / burst traffic — rẻ nhất có thể | **Production default** — cân bằng chất lượng/tốc độ | Teacher + contender benchmark — trần chất lượng của family |
| Params (encoder+emb) | 10,507,264 | 26,397,696 | 102,360,576 |
| Params tổng (kèm head 3 lớp) | **10,573,827** | **26,546,691** | **102,953,475** |
| Tỷ lệ so với B | ~1/10 | ~1/4 | 1× |
| Layers | 6 | 10 | 12 |
| Hidden size | 256 | 384 | 768 |
| Attention heads | 4 (d=64) | 6 (d=64) | 12 (d=64) |
| FFN intermediate | 1024 | 1536 | 3072 |
| **Nguồn weights** | PCA(BamiBERT emb→256) + random | PCA(BamiBERT emb→384) + random | **Nguyên weights BamiBERT** |
| **Cách train** | Task distillation từ B | Task distillation từ B | Fine-tune trực tiếp |
| lr grid | {5e-5, 1e-4, 2e-4}, 15 epochs | {5e-5, 1e-4, 2e-4}, 15 epochs | {1e-5, 2e-5, 5e-5}, 10 epochs |
| Latency CPU (seq32/1thread, đo arch-scaling) | chưa đo (sẽ đo) | 23.4ms (geometry cũ 514-pos) | 92.9ms |
| Size F32 ước theo params | ~40MB | ~101MB | ~393MB |

## Khác biệt cốt lõi giữa 3 model

**GoiBERT-B** — là BamiBERT "đội mũ mình": lấy nguyên checkpoint BamiBERT rồi fine-tune
từng task. v1 chưa thêm gì khác (xem Lộ trình differentiator bên dưới — cố tình, để có sàn
đo ablation). Vai trò thật: (1) con số chuẩn mọi gap% tính từ nó, (2) thầy cho S/XS,
(3) chỗ gắn các differentiator sau này.

**GoiTurn-S** — model định mệnh của dòng Goi. Không có weights BamiBERT (hình dạng khác,
không copy được), chỉ thừa hưởng: (a) tri thức embedding qua phép chiếu PCA 768→384,
(b) toàn bộ "cách suy nghĩ" của thầy qua distillation 3 kênh:
  - logit: khớp phân bố xác suất đầu ra (KL, T=2, trộn 50/50 với CE nhãn thật)
  - hidden states: khớp biểu diễn từng tầng qua projection học được (student tầng i ↔
    teacher tầng round(i·12/10)), MSE có mask
  - attention: khớp bản đồ attention (trung bình theo head vì 6 head ↔ 12 head)
Các projection distill (3.25M params) chỉ tồn tại lúc train, vứt đi khi deploy.

**GoiTurn-XS** — bản thu nhỏ cực đoan để tìm đáy: cùng công thức distill như S nhưng
6L/256H. Trả lời câu hỏi "rẻ đến mức nào thì gãy" — số của nó quyết định model nào đi vào
tier edge/burst (§6.1 Tier 1 dự phòng).

## Khác biệt so với spec giấy §13.2 (khai báo trước, không giấu)

| Spec §13.2 | Thực tế v1 | Lý do |
|---|---|---|
| XS 6.9M / S 21.1M / B 97.6M | 10.6M / 26.5M / 103M | Spec giả định vocab 16K + factorized embeddings; thực tế dùng vocab BamiBERT 20,481 dense + bảng position 2050 (bắt buộc để kế thừa weights/tokenizer) |
| Modern block (RoPE, GeGLU, RMSNorm) | Block RoBERTa cổ điển | BamiBERT thật là block cổ điển — kế thừa weights và modern block loại trừ nhau. Scratch modern block không pretraining chắc chắn thua >5% (đã đo). Modern block quay lại khi có corpus pretrain (fork #1) |
| Recurrent deliberation, FiLM timing | Chưa có | Thuộc GoiTurn turn-detection (cần timing features từ logs), không thuộc benchmark classification |
| Tokenizer 16K chat-Việt | Tokenizer BamiBERT | Cần corpus chat để train tokenizer riêng (P0 #7) |

## Lộ trình differentiator (thứ tự thêm, mỗi món một ablation §13.8)

1. **Stage B — Noise-augmented consistency training** (§13.5 Stage 5): bản sạch vs bản bẩn
   (bỏ dấu/teencode/typo) phải dự đoán giống nhau. Không cần data mới. BamiBERT không có.
2. **Stage C — Multi-task fine-tuning** (§13.5 Stage 3): 1 backbone học chung 4 datasets.
3. **Khi có corpus — Domain-adaptive MLM 30%** (§13.5 Stage 2): món chính, kỳ vọng
   +2.5–3.5 điểm theo cơ chế ViSoBERT/CafeBERT. Bản "lite" với messenger data (20K tin)
   chạy được sớm hơn nếu được duyệt sanitize.

## Chống gian lận (cài trong code, không phải lời hứa)

- Chọn model theo **validation**, chấm **test đúng 1 lần** trên bản đã chọn.
- Cùng protocol 2 phía: cùng data, cùng seed, cùng cách chọn; baseline cũng được grid lr.
- Mọi JSON kết quả ghi kèm config + param count + version torch/transformers + device.
