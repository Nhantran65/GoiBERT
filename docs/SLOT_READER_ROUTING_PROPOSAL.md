# Vòng 4: từ slot readers (4A) đến huấn luyện đọc-được-theo-đoạn (4B)

Ngày: 2026-09-13. Root tự chẩn đoán, đề xuất, đối chiếu tính mới và chạy một phép kiểm nhỏ trên encoder đóng băng. Không training encoder, không test split, không sửa E1–E17.

## 0. Kết luận ngắn

Vòng này đi qua ba bước: chẩn đoán → đề xuất 4A → kiểm novelty → phép kiểm nhỏ trên encoder đóng băng → phép đo trần có đối chứng → điều chỉnh thành đề xuất 4B. Kết quả:

1. **Chẩn đoán được siết lại bằng hai phép đo mới.** (a) Reader probe: K CLS "slot" trong encoder với ~8,4k tham số đọc (encoder/head đóng băng) tách composite ghép (both-correct 9%→32%) nhưng không cấu hình nào vượt PR envelope của E8 trên validation thật; reader chỉ phân phối lại lỗi (missed↓, spurious↑). (b) Trần đọc-theo-đoạn có đối chứng random-mask (§4.4): phần headroom thuộc về cấu trúc đoạn trên câu tự nhiên chỉ ~+2,0 (Restaurant) / +1,5 (Hotel) F1 ở mức oracle; composite battery (39%→66%) phóng đại headroom. Kết luận: **với student hiện tại, fix ở lớp đọc/định tuyến có trần thấp; thiếu hụt trên câu tự nhiên nằm ở token-level representation của từng đoạn.** Đây là bằng chứng mới, giải thích chuỗi null E9–E17 và C4 theo cùng một cách.

2. **Đề xuất 4A (slot readers + định tuyến từ composite) bị hạ cấp**, không đề nghị screening 15 trial. Nó vẫn được ghi đầy đủ ở §2–§3 vì đã qua đối chiếu novelty và có phép kiểm; cổng CGSA (§2.2) chỉ là ablation. Composite là điều kiện cần để reader tách nhưng đồng thời làm mất thông tin tự nhiên (§4.2); định tuyến từ split ẩn tự nhiên (§4.3) không mất envelope nhưng không tăng.

3. **Đề xuất 4B — huấn luyện "đọc được theo đoạn" (fork-consistency) cho toàn encoder, inference giữ CLS thường.** Lấy đúng phép đo trần ở §4.4 làm mục tiêu huấn luyện: với câu train ≥2 aspect, chọn split ẩn liên tục mà cặp fork (CLS bị mask theo đoạn, chạy trong cùng forward) giải thích gold tốt nhất; loss = CE(CLS, gold) + λ·CE(fork_i, gold gán cho đoạn i) + μ·[fork không được thêm aspect ngoài phần được gán]. Không composite, không nhãn scope, không thay kiến trúc lúc inference, chi phí train ≈ +2S hàng attention/câu. Tính chất: đẩy token states của mỗi đoạn về trạng thái đọc được độc lập (binding tại token), điều mà §4.4 cho thấy đang thiếu; phép đo tiến bộ trực tiếp là chính oracle/random-control gap ở §4.4 và composite battery. Novelty: nguyên lý MIL theo đoạn/latent attribution có tiền lệ (MilNet, AC-MIMLLN, aspect-based sentence segmentation 2010, latent sentiment-aspect attribution 2020); phép chọn split bằng fork của chính model trong encoder, dùng làm giám sát biểu diễn rồi bỏ fork khi inference, chưa tìm thấy nguồn trùng. Chi tiết và screening ở §6.

Chưa có kết quả training nào; không claim thắng BamiBERT. Kết quả vòng này là một chẩn đoán mới có đối chứng và một đề xuất được đo trước bằng chính oracle của nó.

## 1. Chẩn đoán tổng hợp

Bằng chứng local đã có, đọc lại theo một trục duy nhất — *ai trộn thông tin của các aspect*:

| Phép đo | Student E8 | BamiBERT E14 | Đọc |
|---|---:|---:|---|
| Composite A+B (200 cặp val), A_correct | 66,0 đơn lẻ → 39,0 | 72,5 → 51,5 | Cả hai mất, student mất nhiều hơn |
| A+3B (B lặp 3 lần) | 29,0 | 36,0 | Nội dung lặp không thêm thông tin vẫn làm mất A |
| fork_test: CLS row chỉ đọc phần A ở mọi layer | A_correct 65,5 (≈ đơn lẻ), B_present 6,5 | A_correct 60,0 (< đơn lẻ 72,5), **B_present 30,5** | Student: token states A sạch, lỗi ở CLS trộn. Bami: token states A đã nhiễm B |
| partition_oracle: một CLS, attention chia đều A/B | 37,5 (không cứu) | 48,0 | Một vector CLS không giữ được cả hai; cần nhiều reader |
| C1 (dedup key multiplicity) | không cứu | không cứu | Không phải chỉ dilution do trùng lặp |
| C4 clause readers (regex, frozen) | recall↑, precision↓, F1↓ | tương tự | Scope cứng sai ở phủ định; operating point đổi |
| Natural val, k=2 mixed polarity vs không mixed (Restaurant) | missed 23,1% vs 14,1% | — | Binding trái chiều là phần lỗi tự nhiên |
| Natural val, pair chưa gặp trong train (k=2) | missed 37,5% vs 16,0% | — | Prior đồng xuất hiện |

Kết luận chẩn đoán: với student, đường lỗi chủ đạo trên composition là **CLS đọc trộn nhiều đoạn**; token states đủ tốt để một reader được giới hạn đúng đoạn phục hồi gần mức đơn lẻ. Các thí nghiệm E9–E17 sửa encoder (span/conv, signed attention, pair-value, FFN gate, cavity keys) đều gần null; điều này nhất quán với chẩn đoán trên: chúng không sửa định tuyến đọc.

Phần chưa được E9 giải quyết: E9 dùng query theo nhãn chỉ ở layer cuối, trộn 50% CLS gốc vào readout, và không có tín hiệu huấn luyện nào buộc query đọc đúng đoạn. Phép kiểm §4 cho thấy reader ở **mọi layer** với tham số rất nhỏ đã tách composite; do đó khác biệt so E9 là vị trí (trong encoder) và tín hiệu định tuyến, không phải "thêm query".

## 2. Cơ chế

### 2.1. Slot rows bảo toàn hàm

Cho embedding sau LayerNorm của chuỗi e_0..e_{N-1} (e_0 là CLS). Tạo K slot: c_k^0 = e_0 + δ_k, δ_k ∈ R^D. Ở mỗi layer l, token rows tính attention như checkpoint (key/value gồm slot 1 và các token; không thấy slot 2..K). Slot k tính query q_k = W_Q c_k + b_Q + β_{l,k} với bias query riêng theo layer/slot, và attend lên các cột {own_k} ∪ tokens hợp lệ. FFN/LN dùng chung. Head dùng chung: L_k = head(c_k^L) ∈ R^{A×C}.

Gộp: m_k(a) = max_{c≥1} L_k[a,c] − L_k[a,0]; k*(a) = argmax_k m_k(a); logits cuối cho aspect a là L_{k*(a)}[a,:]. Presence là hàm không giảm theo số slot.

Tính chất P1 (bảo toàn hàm): khi δ_k = 0, β_{l,k} = 0 với mọi k, mọi slot trùng CLS gốc theo từng layer, nên logits gộp bằng logits checkpoint. Đã kiểm FP32: max |Δ| = 4,6e-6 (K=2), 3,3e-6 (K=3) so với đường tính thủ công K=1; đường thủ công so native BF16 SDPA 0,047 (cùng mức parity của battery cũ).

### 2.2. Cổng cạnh tranh giữa slot (CGSA), giữ làm ablation

Với điểm s_kj = q_k·k_j/√d cho slot k và cột j, đặt g_kj = softmax_k(s_kj) (cạnh tranh giữa slot cho từng cột, từng head) và

w_kj = exp(s_kj + β·log g_kj) / Σ_{j'} exp(s_kj' + β·log g_kj').

- β=0: softmax thường của từng slot.
- Slot trùng nhau: g_kj = 1/K với mọi j ⇒ w = softmax_j(s_kj), tức P1 vẫn đúng với mọi β. Đây là điểm khác DNAS và slot attention: hai phương pháp đó dùng w ∝ g_kj (chuẩn hóa theo query rồi theo key), nên với slot trùng nhau cho trọng số **đều** trên token và không bảo toàn checkpoint.
- β=1: w_kj ∝ e^{2s_kj}/Σ_{k'} e^{s_k'j}; cột bị slot khác chiếm rời khỏi mẫu số của slot k (loại trừ), nên nội dung được slot khác nhận không pha loãng slot k. Đây là họ một tham số nối softmax thường (β=0) với một dạng "DNAS bình phương điểm" (β=1); DNAS gốc không thuộc họ này vì thiếu thừa số e^{s_kj}.

Kết quả phép kiểm §4: β=1 hơn β=0 khoảng +2 điểm A_correct và +2 điểm both_correct trên 200 composite (trong biên nhiễu), gate mass của slot được chọn trên token A là 0,60 so 0,39 trên token B (K=2), 0,50 so 0,30 (K=3). **Không dùng CGSA làm claim chính**; nếu screening cho thấy β=1 không hơn β=0 sau khi train đầy đủ, bỏ cổng và giữ slot rows thường.

### 2.3. Giám sát định tuyến sinh từ ghép câu

Composite x = P_1 ⊕ … ⊕ P_m từ m câu train mỗi câu đúng một aspect, aspect khác nhau, nhãn hợp là hợp nhãn. Trong cùng forward, thêm m hàng "fork" f_i có attention bị mask cứng vào {own} ∪ tokens(P_i) ∪ {</s>} ở mọi layer; đây là chính operator fork_test, đã đo phục hồi gần mức đơn lẻ trên student. Fork không được gradient (detach), không dùng ở inference.

Loss trên composite:

- L_union: CE trên logits gộp của slot với nhãn hợp.
- L_route: ghép cặp Hungarian giữa slot k và đoạn i với cost KL(softmax L_k[a,:] ‖ softmax F_i[a,:]) cộng trên các aspect a có mặt trong P_i; tối thiểu hóa tổng cost của các cặp được ghép. Slot không được ghép không chịu L_route.

Trên câu tự nhiên chỉ có L_union (chính là CE thường). Recipe nền giữ E8 (NC + LLRD) để không đổi hai thứ cùng lúc; NC áp lên logits gộp.

Điều L_route thêm mà L_union không có: L_union chỉ nói *cái gì có mặt*, cho phép slot đọc polarity từ đoạn sai nếu head khai thác prior đồng xuất hiện; L_route buộc từng slot tái tạo readout của đúng một đoạn, tức học định tuyến theo đoạn. Giả định để chuyển sang văn bản tự nhiên: mệnh đề tự nhiên có cấu trúc gần đoạn ghép; đây là gamble phải đo, không phải điều đã chứng minh (C3DA/LWEDA đã ghi nhận composite ghép thiếu mạch lạc).

## 3. Đối chiếu tính mới

| Tiền lệ | Phần phải ghi nhận | Khác biệt cụ thể trong ứng viên |
|---|---|---|
| Multi-CLS BERT (Chang et al. 2023) | Nhiều CLS trong encoder; ghi nhận CLS dễ trùng nhau | Không reparameterization/inserted layers; khởi tạo bảo toàn hàm chính xác; gộp theo aspect bằng max margin; giám sát định tuyến từ composite |
| Label-prompt/label-token tuning cho multi-label | Token riêng cho từng nhãn trong encoder | Slot không gắn nhãn (K ≪ số aspect), phân vùng theo nội dung; head chung |
| Slot Attention (Locatello 2020), DNAS (Ding 2020), Sinkformer, GroupViT, Soft MoE | Cạnh tranh giữa slot/query cho token; chuẩn hóa hai chiều | CGSA giữ softmax theo key rồi nhân cổng cạnh tranh; giảm chính xác về softmax gốc khi slot trùng; chỉ ở slot rows; là ablation phụ |
| SPARO (2024) | Read-out nhiều slot, mỗi slot một head; compositional robustness (vision/CLIP) | Slot chạy trong mọi layer của encoder pretrained, không phải read-out sau encoder; bằng chứng fork cho thấy cần trong-encoder |
| Query2Label / ML-Decoder / E9 | Query theo nhãn ở layer cuối | E9 đã screening fail; khác vị trí, không trộn CLS, có L_route |
| CAN (Hu 2019) | Attention theo aspect với ràng buộc sparse/orthogonal | Không ràng buộc trực giao; định tuyến học từ target fork thay vì regularizer |
| MIL / noisy-OR / max pooling | Presence đơn điệu theo instance | Gộp theo slot, không theo token; polarity đọc từ slot |
| C3DA (2022), LWEDA, concat augmentation, Chai et al. 2023 | Sinh câu đa aspect; contrastive/union label | Không chỉ union label: dùng đoạn biết trước để sinh target định tuyến |
| IIT / causal distillation (Geiger 2022; Wu et al. 2022) | Giám sát bằng output phản thực dưới can thiệp | Target là readout của cùng model bị mask theo đoạn, không cần mô hình nhân quả hay swap hidden state |
| Allchin 2026 (causal evidence sets, masked frozen teacher) | Selector được giám sát bằng masked teacher | Pruning/retrieval, không phải readers ABSA; ở đây mask theo đoạn biết trước của composite, teacher là chính model đang train (detach) |
| Attention supervision bằng rationale (Bao 2018 …), Aspect-to-Scope (Chai 2023) | Giám sát attention/scope | Không nhãn người; scope = đoạn ghép |
| ARTS (Xing 2020), MAMS | Kiểm robustness khi thêm câu aspect khác | Dùng làm phép đo, không phải phương pháp |
| DETR / set prediction | Ghép Hungarian slot–target | Dùng như công cụ ghép, không claim |

Kết luận novelty: không tìm thấy nguồn trùng cho (a) target định tuyến từ fork theo đoạn của cùng model trên self-composite, ghép Hungarian với slot rows trong encoder; (b) công thức CGSA. Nguyên lý nền của từng thành phần đều có tiền lệ, đã liệt kê. Search absence không phải chứng nhận; cần đọc full SPARO và Multi-CLS §2.5 trước khi viết claim công bố. Đóng góp dễ bảo vệ nhất là **chẩn đoán cơ chế (định tuyến đọc, khác giữa student và teacher) cộng fix có đối chứng**, đúng hướng root đã ghi ở INDEPENDENT_BATTERY_ROOT_REVIEW mục 5.4.

## 4. Phép kiểm nhỏ đã chạy (encoder và head đóng băng)

Script: `scripts/research_checks/competitive_slots_probe.py`; artifacts: `artifacts/research_diagnostics/competitive_slots/`. Checkpoint E8 seed42, GPU1, FP32 đường thủ công. Tham số học: δ ∈ R^{K×384} (init N(0,0,05²)), bias query ∈ R^{10×K×384} (init 0); Adam lr 5e-3, 500 bước, batch 32, trộn 50% câu train tự nhiên + 50% composite 2–3 đoạn từ câu train đơn-aspect ≤48 token (Restaurant: 4841 singles, 2654 + 1290 composite). Đánh giá chỉ trên validation: pair F1 thật, 200 composite validation của battery (cùng seed 2026), A+3B, và gate mass.

| Arm (Restaurant) | val pair F1 | multi-row missed | A_correct A+B | B_correct | both | A+3B A_correct | A đơn lẻ |
|---|---:|---:|---:|---:|---:|---:|---:|
| E8 gốc (battery) | 71,30 | 20,5 | 39,0 / 40,0 | 33,5 | 9,0 | 29,5 | 66,0 |
| K=1 chỉ prompt | 69,73 | 16,7 | 42,5 / 44,0 | 40,0 | 12,0 | 30,0 | 66,5 |
| K=2 β=0 | 69,65 | 16,9 | 56,0 / 56,0 | 58,0 / 52,5 | 30,5 | 47,5 | 68,5 |
| K=2 β=1 | 70,34 | 17,3 | 58,5 / 59,0 | 59,5 / 54,0 | 32,5 | 48,0 | 68,5 |
| K=3 β=1 | 70,51 | 17,9 | 60,0 / 58,5 | 59,0 / 52,5 | 33,0 / 30,5 | 50,5 | 68,0 |

| Arm (Hotel) | val pair F1 | multi-row missed | A_correct A+B | B_correct | both | A+3B A_correct | A đơn lẻ |
|---|---:|---:|---:|---:|---:|---:|---:|
| E8 gốc (battery) | 73,05 | 28,1 | 41,5 / 42,0 | — | — | 22,5 | 59,5 |
| K=1 chỉ prompt | 71,25 | 24,7 | 47,0 / 47,5 | 46,0 / 50,0 | 18,5 | 29,5 | 59,0 |
| K=2 β=0 | 71,51 | 24,5 | 54,0 / 55,0 | 64,0 / 63,5 | 30,5 | 48,0 | 56,5 |
| K=2 β=1 | 71,51 | 24,6 | 54,0 / 53,0 | 64,0 / 64,5 | 32,5 / 30,5 | 43,5 | 58,5 |
| K=3 β=1 | 71,74 | 23,8 | 53,0 / 55,0 | 64,0 / 62,5 | 30,5 / 31,0 | 50,0 | 58,5 |

Hai cột A_correct là thứ tự A trước / B trước. Đọc:

- Với ~8,4k tham số đọc và encoder/head đóng băng, hai slot tách composite tốt hơn hẳn một CLS có prompt (both 12,0→30,5; A+3B 30,0→47,5). Vì K=1 chỉ prompt gần như không tách, phần hiệu quả đến từ **nhiều reader**, không từ việc tinh chỉnh một prompt.
- Cổng cạnh tranh thêm khoảng +2 điểm, trong biên nhiễu 200 cặp; gate mass cho thấy phân vùng mềm (0,60/0,39). Giữ làm ablation.
- pair F1 validation giảm ở mọi arm có composite, kể cả K=1. Phân tích tiếp ở §4.2.

### 4.2. Đối chứng dữ liệu, giám sát định tuyến từ fork và PR envelope

Bổ sung (`competitive_slots_probe_v2.py`, `results_v2.json`, `results_v3_*.json`): composite xây ở mức token-id với span chính xác; fork rows theo đoạn chạy trong cùng forward (parity fork full-mask = CLS gốc 3,8e-6); L_route = KL(fork ‖ slot) ghép Hungarian; envelope = max pair F1 trên validation khi quét ngưỡng margin toàn cục (chỉ chẩn đoán, như READER_DIAGNOSTIC_SPEC). Oracle envelope của E8 gốc: Restaurant 71,73; Hotel 73,05.

| Arm (Restaurant, K=2 β=0) | tỷ lệ natural | val F1 | oracle envelope | missed / spurious / wrong | tp | both A+B |
|---|---:|---:|---:|---|---:|---:|
| E8 gốc | – | 71,30 | 71,73 | 176 / 133 / 142 | 735 | 9,0 |
| K=1 chỉ prompt | 0,5 | 69,73 | 70,05 | 146 / 194 / 156 | 751 | 12,0 |
| union | 0,5 | 69,65 | 70,23 | 148 / 187 / 158 | 747 | 30,5 |
| union + route | 0,5 | 68,68 | 69,69 | 110 / 252 / 171 | 772 | 32,5 / 35,0 |
| route only | 0,5 | 68,26 | 69,50 | 108 / 267 / 172 | 773 | 32,0 / 35,5 |
| union + route, CGSA β=1 | 0,5 | 68,74 | – | 125 / 233 / 167 | 761 | 35,0 / 32,5 |
| union | 0,85 | 70,62 | 70,95 | 162 / 163 / 147 | 744 | 24,5 / 24,0 |
| union + route | 0,85 | 69,35 | 70,16 | 120 / 229 / 165 | 768 | 29,5 / 31,5 |
| union, không composite | 1,0 | 71,74 | 71,74 | 162 / 147 / 141 | 750 | 8,0 |
| K=1, không composite | 1,0 | 71,26 | – | 176 / 133 / 142 | 735 | 6,5 / 7,5 |

| Arm (Hotel, K=2 β=0) | tỷ lệ natural | val F1 | oracle envelope | missed / spurious / wrong | tp | both A+B |
|---|---:|---:|---:|---|---:|---:|
| E8 gốc | – | 73,05 | 73,05 | – | – | – |
| K=1 chỉ prompt | 0,5 | 71,25 | 71,87 | 300 / 287 / 81 | 928 | 18,5 |
| union | 0,5 | 71,51 | 72,58 | 301 / 287 / 77 | 931 | 30,5 |
| union + route | 0,5 | 69,21 | 70,65 | 245 / 427 / 95 | 969 | 34,0 / 33,5 |
| route only | 0,5 | 68,72 | 70,46 | 242 / 450 / 96 | 971 | 34,0 / 33,5 |
| union + route, CGSA β=1 | 0,5 | 70,17 | – | 269 / 373 / 85 | 955 | 34,5 / 33,0 |
| union | 0,85 | 72,65 | 72,87 | 313 / 236 / 73 | 923 | 26,5 / 25,5 |
| union + route | 0,85 | 69,93 | 71,01 | 259 / 381 / 92 | 958 | 31,0 / 31,0 |
| union, không composite | 1,0 | 72,79 | 72,90 | 338 / 212 / 64 | 907 | 16,5 / 15,5 |

Đọc:

- **L_route hoạt động đúng cơ chế**: missed trên validation thật giảm mạnh (Restaurant 176→108–120; Hotel ~340→242–259) và tp tăng (+35–40), composite tách tốt hơn. Nhưng spurious tăng gấp đôi và **oracle envelope thấp hơn baseline** (Restaurant 69,5–70,2 so 71,73), tức không phải chỉ lệch operating point: reader học từ composite ghép mất thông tin trên câu tự nhiên, đơn điệu theo tỷ lệ composite (1,0 → 0,85 → 0,5).
- Không composite: slot không tách composite (both ≈ 8%) và envelope ≈ baseline. Vậy trong chế độ đóng băng, **composite là điều kiện cần để tách, đồng thời là nguồn mất thông tin tự nhiên**; hai điều này chưa tách rời được bằng tham số đọc nhỏ.
- Kết quả này lặp lại cảnh báo của fork_test probe_mixed (70,25→67,74) ở một cơ chế khác. Nó hạ prior của gamble "composite ghép → mệnh đề tự nhiên". Nó không loại trừ rằng head/encoder trainable sẽ hấp thụ phần lệch, nhưng screening phải coi A1 (data-only) là đối chứng bắt buộc và phải có kill criterion trên spurious.

### 4.3. Định tuyến từ split ẩn trên câu tự nhiên (không composite)

Vì lỗi ở §4.2 nằm ở phong cách composite, root thử biến thể lấy target định tuyến từ chính câu train đa aspect: liệt kê split nhị phân tại dấu câu/liên từ (tối đa 6), chạy hai fork theo đoạn cho mỗi split trong cùng forward, chọn split có Σ_a max_i log p_fork_i(y_a) cao hơn CLS chưa split, gán aspect cho fork có xác suất cao hơn, rồi L_route như trên. Script `competitive_slots_probe_v3_latent.py`, `results_v3_latent.json`. Kết quả ở bảng dưới (điền sau khi chạy xong).

| Arm (K=2 β=0, chỉ câu tự nhiên, 50% batch là câu ≥2 aspect) | val F1 | oracle envelope | missed / spurious / wrong | tp | multi-row missed | both A+B (composite, tham khảo) |
|---|---:|---:|---|---:|---:|---:|
| Restaurant, E8 gốc | 71,30 | 71,73 | 176 / 133 / 142 | 735 | 20,5 | 9,0 |
| Restaurant, không L_route | 71,30 | 71,35 | 166 / 147 / 143 | 744 | 18,8 | 7,5 / 11,0 |
| Restaurant, L_route λ=1 | **71,66** | 71,77 | 142 / 171 / 146 | 765 | 15,4 | 15,0 / 15,5 |
| Restaurant, L_route λ=3 | 71,04 | 71,73 | 140 / 182 / 150 | 763 | 15,6 | 16,5 / 15,5 |
| Hotel, E8 gốc | 73,05 | 73,05 | – | – | 28,1 | – |
| Hotel, không L_route | 72,76 | 72,82 | 331 / 220 / 66 | 912 | 27,4 | 18,5 / 16,5 |
| Hotel, L_route λ=1 | 71,10 | 72,07 | 275 / 321 / 87 | 947 | 22,8 | 23,0 / 20,0 |
| Hotel, L_route λ=3 | 70,51 | 71,83 | 270 / 355 / 86 | 953 | 22,6 | 24,5 / 20,0 |

Split được chấp nhận (fork pair giải thích gold tốt hơn CLS chưa split) ở ~80% (Restaurant) và ~84% (Hotel) lượt lấy mẫu câu đa aspect. Đọc: định tuyến từ split ẩn tự nhiên không làm mất envelope như composite (Restaurant 71,77 ≈ 71,73), giảm missed và tăng tp, cho +0,36 F1 ở δ=0 trên Restaurant nhưng −1,95 trên Hotel (spurious tăng). Trong chế độ đóng băng, không cấu hình reader nào **vượt envelope của E8 gốc**; các reader chỉ phân phối lại lỗi.

### 4.4. Trần của "đọc theo đoạn" trên câu tự nhiên, có đối chứng selection bias

Câu hỏi quyết định: nếu có một bộ chọn split hoàn hảo, đọc theo đoạn với head hiện tại được bao nhiêu? Script `segment_read_ceiling.py` (inference-only): mỗi câu validation liệt kê ≤8 split tại dấu câu/liên từ, chạy 2 fork/split, gộp hai fork theo max presence margin; oracle chọn (theo gold) split hoặc không-split tốt nhất từng câu. Đối chứng **random-mask**: thay hai đoạn liên tục bằng hai tập token ngẫu nhiên bù nhau cùng kích thước — giữ nguyên số ứng viên (selection bias) nhưng phá cấu trúc đoạn.

| Validation | E8 gốc | Oracle split liên tục | Oracle random-mask (control) | Phần riêng của cấu trúc đoạn | Bộ chọn naive (max |margin|) |
|---|---:|---:|---:|---:|---:|
| Restaurant | 71,31 | 79,68 | 77,70 | **+1,98** | 67,93 |
| Hotel | 73,08 | 78,75 | 77,23 | **+1,52** | 68,87 |

Artifact: `segment_read_ceiling.json`, `segment_read_ceiling_randomctrl.json`. Đọc:

- Phần lớn "trần" +8,4/+5,7 là selection bias: chọn tốt nhất trong 9 lần đọc nhiễu bằng gold đã cho +6,4/+4,2 ngay cả khi mask ngẫu nhiên. Phần thuộc về cấu trúc đoạn chỉ ~+2/+1,5 F1, và đó vẫn là oracle.
- Bộ chọn không dùng gold theo confidence kém hơn baseline 3–4 điểm: chọn split là bài toán khó, và composite battery (39%→66% khi tách) **phóng đại** headroom trên dữ liệu thật. Câu đa aspect tự nhiên không tách sạch thành đoạn đọc được với token states hiện tại (ngữ cảnh chung, phủ định, aspect ngầm).
- Hệ quả cho project: các fix chỉ ở lớp đọc (E9, slot readers, C4, 4A) có trần thấp với student hiện tại; thiếu hụt trên câu tự nhiên nằm ở **biểu diễn token-level** (đoạn không tự đọc được) và/hoặc head chưa được huấn luyện cho fragment. Điều này nhất quán với D2 (teacher tăng nhiều hơn khi fine-tune) và với chuỗi null E10–E17.


## 5. Điều kiện ứng viên (theo COMPOSITION_DIAGNOSIS_RESEARCH §5) — áp cho 4B

1. Failure mode và comparator: composite battery + fork oracle + C4 calibrated + K=1 prompt control + **trần §4.4 với random-mask control** (ngưỡng, prompt, hay chọn-trong-nhiễu đều không giải thích phần đoạn). ✔
2. Forward/objective cụ thể: §6.1; assumptions (câu đa aspect có split liên tục đọc được ở ~80% lượt lấy mẫu theo chính fork của model) và trường hợp thất bại (phủ định xuyên đoạn, aspect ngầm, câu 1 aspect không có split → chỉ CE thường). ✔
3. Claim chart: §3 cộng MilNet/AC-MIMLLN/latent attribution/aspect-based segmentation; cần đọc full MilNet và Multi-Aspect Latent Attribution trước claim công bố. ✔ có điều kiện
4. Init/adaptation/compute: không tham số mới lúc inference; train thêm 2S fork rows/câu (S≤6), ≈ +10–15% attention cost; recipe E8 giữ nguyên. ✔
5. Phép kiểm nhỏ có thông tin: §4.3–4.4 đo trước rằng (i) split đọc được tồn tại cho ~80% câu đa aspect, (ii) phần đoạn của oracle ≈ +2/+1,5 khi head/encoder đóng băng. 4B nhắm nâng chính con số (ii) bằng huấn luyện encoder; nếu sau huấn luyện oracle-gap không tăng và F1 không tăng, giả thuyết "thiếu đọc được theo đoạn" bị bác. ✔

## 6. Screening đề xuất (chưa launch)

### 6.1. Objective 4B

Câu x có gold G = {(a, y_a)}. Nếu |G| ≥ 2: liệt kê split j ∈ J(x) (sau dấu câu/liên từ, ≤6); trong cùng forward thêm 2|J| fork rows f_{j,1}, f_{j,2} (mask cứng theo đoạn, không có gradient vào việc chọn). Chọn j* = argmax_j Σ_{a∈G} max_i log p(y_a | f_{j,i}); nếu j* không hơn CLS chưa split thì bỏ qua thành phần fork cho câu đó. Gán mỗi a ∈ G cho fork có log p cao hơn: G_1 ∪ G_2 = G.

L = CE(CLS, G) + λ Σ_i CE(f_{j*,i}, G_i ∪ {absent cho a ∉ G_i}) , λ ≈ 0,3–0,5; gradient chảy qua fork rows vào encoder (fork không detach). NC của E8 áp lên CLS như cũ. Inference: CLS thường.

Điểm khác 4A: không slot, không composite, không Hungarian với target mềm; target là gold, split là latent hard-EM tính bằng chính model (không gradient qua argmax).

### 6.2. Arms

| Arm | Mô tả | Mục đích |
|---|---|---|
| B0 | E8 replay cùng kernel | baseline |
| B1 | 4B, λ=0,3 | ứng viên |
| B2 | 4B nhưng split ngẫu nhiên (random-mask, cùng kích thước) thay vì j* | đối chứng: gain do "đọc fragment" hay do cấu trúc đoạn |
| B3 | 4B nhưng fork detach (chỉ head học) | đối chứng: gain ở encoder hay ở head |
| B4 | E8 + composite union (data-only) | đối chứng augmentation cũ |

Restaurant + Hotel trước, seed 42, 30 epochs, recipe E8, validation-only. Đo: pair F1; phân rã missed/spurious/wrong theo k và mixed-polarity; **oracle-gap §4.4 tính lại trên checkpoint mới** (kỳ vọng B1 tăng gap và tăng F1 đồng thời); composite battery; fixture controlled interventions. Gate: B1 > B0 và B1 > B2 ≥ 0,5 điểm trên cả hai task, spurious không vượt B0 quá 10%. Nếu B2 ≈ B1: gain là "đọc fragment/regularization", không phải cấu trúc đoạn — vẫn có thể hữu ích nhưng không claim cơ chế.

Dự đoán có thể bác bỏ: (i) B1 tăng oracle-gap phần đoạn (§4.4) ≥ +1 so B0; (ii) B1 giảm missed ở k≥2 mixed-polarity mà spurious tăng ≤ 10%; (iii) nếu (i) đúng mà F1 không tăng, đọc được theo đoạn không phải nút thắt của F1 → dừng hướng reader/segment hoàn toàn.

## 7. Giới hạn và điều không được nói quá

- 200 composite lấy từ pool nhỏ, không phải 200 quan sát độc lập; chênh ±2 điểm giữa β là nhiễu.
- Mọi probe dùng head đóng băng; F1 giảm ở đó không phải bằng chứng cơ chế hại, cũng không phải bằng chứng nó có lợi. Oracle ở §4.4 dùng gold để chọn, chỉ là trần; random-mask control là bắt buộc khi trích dẫn.
- Không tuyên bố slot/fork học "mệnh đề" hay "scope ngữ nghĩa"; chỉ đo phân vùng trên composite và split liên tục tại dấu câu/liên từ.
- Không gọi CGSA là đóng góp; không gọi search absence là chứng nhận novelty.
- Mục tiêu thắng BamiBERT chưa đạt; 4B nhắm phần lỗi composition/binding trên ABSA, không tự động cải thiện VSFC.

## Nguồn

- Multi-CLS BERT: Chang et al., ACL 2023, https://aclanthology.org/2023.acl-long.48.pdf (đã đọc §2.5–2.6, 3.1–3.3 ở vòng trước).
- Slot Attention: Locatello et al., NeurIPS 2020. DNAS: Ding et al., "Attention is not all you need: doubly-normalized attention", ICLR 2020 submission, https://openreview.net/pdf?id=r1g_R6EKvr (đọc trích dẫn qua search; chưa audit toàn bộ). Sinkformers: https://arxiv.org/pdf/2110.11773.
- SPARO: Vani et al. 2024, https://arxiv.org/abs/2404.15721 (đọc abstract; chưa xác minh chi tiết cơ chế cạnh tranh).
- C3DA: Wang et al. 2022, https://arxiv.org/abs/2204.07832 (abstract). LWEDA: Neurocomputing 2024 (chỉ metadata).
- Query2Label: Liu et al. 2021. CAN: Hu et al., EMNLP 2019. AC-MIMLLN: Li et al., EMNLP 2020.
- IIT: Geiger et al., ICML 2022. Causal Distillation: Wu et al., NAACL 2022 (chưa đọc lại lượt này).
- Allchin 2026: https://arxiv.org/abs/2607.21692 (abstract). ARTS: Xing et al., EMNLP 2020. MAMS: Jiang et al., EMNLP 2019.
- Battery và fork_test local: `artifacts/independent_review/battery/`, `docs/INDEPENDENT_BATTERY_ROOT_REVIEW.md`.
- MilNet: Angelidis & Lapata, TACL 2018, https://arxiv.org/pdf/1711.09645 (abstract/metadata qua search). Aspect-based sentence segmentation: Zhai et al. 2010, https://dl.acm.org/doi/pdf/10.1145/1651461.1651474 (metadata). Multi-Aspect Sentiment Analysis with Latent Sentiment-Aspect Attribution, 2020, https://arxiv.org/pdf/2012.08407 (metadata). Chưa đọc full text ba nguồn này.
- Probe artifacts vòng này: `artifacts/research_diagnostics/competitive_slots/` (results_v1/v2/v3_*.json, segment_read_ceiling*.json, log_*.txt); scripts `scripts/research_checks/competitive_slots_probe*.py`, `segment_read_ceiling.py`.
