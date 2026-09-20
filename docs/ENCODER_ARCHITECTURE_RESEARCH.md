# Kiến trúc encoder tiếng Việt: cơ hội cải tiến GoiBERT để cạnh tranh BamiBERT

## 1. Quyết định nghiên cứu

**Ưu tiên một block tổng hợp cụm có thứ tự, có lựa chọn theo ngữ cảnh, đặt bên trong encoder của student hiện tại.** Giữ checkpoint distilled làm điểm khởi đầu và giữ recipe E8 làm đối chứng. Không tiếp tục lấy thay loss, KD hoặc hyperparameter làm đóng góp chính. Kích thước không bị khóa, nhưng chỉ tăng khi phép thử cho thấy có ích.

Tên mô tả tạm thời là **block hợp thành span với đường residual**. Đây là tên làm việc, không phải tên một phương pháp đã được chứng minh mới. Đóng góp tiềm năng nằm ở cơ chế hợp thành và phân phối thông tin cụm cho từng token, khả năng hoạt động trên checkpoint byte-BPE sẵn có, và bằng chứng rằng cơ chế đó cải thiện nhiều task. Thay một nhánh tính toán trong encoder có ý nghĩa nghiên cứu hơn việc thay head chỉ khi vượt được đối chứng đơn giản cùng ngân sách.

Tại mốc rà soát 11/09/2026, có cơ sở để ưu tiên nhóm vấn đề **đơn vị biểu diễn, hợp thành cục bộ và nhiễu bề mặt**. Chưa có bằng chứng công khai hoặc nội bộ đủ để chỉ định “layer k của GoiBERT/BamiBERT là bottleneck”. Các chẩn đoán cũ không đo trực tiếp những hiện tượng này. Vì vậy, kết quả của báo cáo là một giả thuyết kiến trúc có thể bác bỏ, cùng thiết kế và kế hoạch kiểm chứng; không phải lời khẳng định đã tìm ra lỗi của BamiBERT.

Lộ trình đề nghị gồm ba bước: kiểm tra nhóm lỗi liên quan đến hợp thành; thử một block cùng một đối chứng cục bộ trên đủ năm task; nếu có tín hiệu rộng, kiểm chứng cơ chế và chuyển cùng thiết kế sang BamiBERT gốc. Không mặc định rằng student phải bắt chước teacher để thắng. Nhãn thật vẫn trực tiếp tối ưu toàn bộ encoder đã sửa.

## 2. Định nghĩa mục tiêu và đối thủ

BamiBERT là mốc mạnh, nhưng “thắng BamiBERT” và “SOTA tiếng Việt” là hai tuyên bố khác nhau. Paper báo cáo tám hàng benchmark, 15 metric; model dẫn 11 metric, không phải tất cả. Nó dùng BERT-base 12 block, byte-BPE 20.481 mục, context 2.048 và MLM trên 129 GB trong 20 epoch. Đóng góp không phải một block attention mới.[^1]

Config chính thức ghi hidden 768, 12 attention heads, FFN 3.072 và positional embedding tuyệt đối.[^2] Như vậy, có không gian sửa phép tính trong encoder, nhưng việc một kiến trúc cũ vẫn đạt điểm cao cũng cho thấy không được suy luận “cũ = bottleneck”.

Ba mức kết luận nên tách biệt:

| Mức | Bằng chứng cần có | Cách diễn đạt hợp lệ |
|---|---|---|
| Cải thiện student | Cùng GD checkpoint, recipe, dữ liệu và seed; chỉ khác block | Kiến trúc A cải thiện baseline E8 trong protocol này |
| Vượt BamiBERT | Test cùng split/evaluator và baseline BamiBERT gốc chạy tại chỗ | A vượt BamiBERT trên các task được đo |
| SOTA rộng | Thêm đối thủ hiện đại, nhiều họ task, nhiều seed ở cấu hình cuối | SOTA trong phạm vi model/task/metric được công bố |

Để mở rộng đối thủ, mmBERT là một encoder đa ngôn ngữ hiện đại cần được xét ở giai đoạn chốt kết quả. Công trình dùng khoảng 3T token và hơn 1.800 ngôn ngữ; kết quả đa ngôn ngữ của nó không tự chứng minh hơn BamiBERT trên năm task hiện tại.[^3] Không cần chạy ngay mọi model, nhưng bảng paper cuối không nên chỉ so với những đối thủ có trước BamiBERT.

Năm task hiện tại thuộc bốn dataset: hai task VSFC dùng chung văn bản. Vì thế, thắng 4/5 task chưa tương đương tổng quát hóa trên bốn nguồn dữ liệu độc lập. Khi một hướng có tín hiệu, bổ sung NLI và NER sẽ có giá trị hơn tiếp tục thêm nhiều biến thể ABSA. ViNLI là nguồn mở rộng hợp lý cho suy luận câu; đó là một bộ đánh giá riêng, không được trộn điểm với ABSA.[^4]

## 3. Bằng chứng nội bộ: điều đã biết và chưa biết

Student GD có 10 block, hidden 384, sáu head, FFN 1.536. Head dimension bằng 64, cũng bằng 768/12 của teacher DAPT. Student không đơn giản là BamiBERT cắt đôi mọi kích thước. Teacher DAPT trong project là derivative đã huấn luyện thêm, không phải checkpoint BamiBERT gốc.

E8 là baseline hợp lý vì đã có tín hiệu lặp lại trên ViSFD và Restaurant. E9 thay head sang attention theo nhãn không đạt cải thiện rộng. Bảng dưới là validation seed 42, F1 hiển thị theo thang 0–100; không so với paper test.

| Task | E8 | E9 | Chênh lệch điểm F1 |
|---|---:|---:|---:|
| VSFC Sentiment | 85,1604 | 85,1923 | +0,0319 |
| VSFC Topic | 80,3678 | 80,2676 | −0,1002 |
| ViSFD | 82,4421 | 82,0986 | −0,3435 |
| Restaurant | 71,2766 | 72,3018 | +1,0252 |
| Hotel | 73,0491 | 72,4333 | −0,6158 |

Nguồn: [E9_RESULT.md](E9_RESULT.md), metric gốc trong `artifacts/e8` và `artifacts/e9`. Lỗi đơn vị của bản summary đầu đã được sửa; không dùng số chênh lệch từ bản lỗi.

[D1](D1_READOUT.md) và [D2](D2_READOUT.md) chưa xác định được một block lỗi chung. Probe chủ yếu đọc CLS, số mẫu probe hạn chế và kết quả nhạy với tối ưu. CKA giống nhau không chứng minh layer dư; drift lớn không chứng minh mất kiến thức. Việc chưa tìm thấy bottleneck bằng các phép đo ấy không loại trừ một vấn đề ở cách hợp thành token.

[Phân tích lỗi E8](E8_ERROR_READOUT.md) cho thấy khoảng cách FN với teacher trên ViSFD/Restaurant chủ yếu đến từ missed aspects. Tuy nhiên, missed aspect có thể do nhiều nguyên nhân: hiểu nghĩa ngầm, nhãn thiếu, imbalance, readout, hoặc thành phần encoder. Không thể chuyển trực tiếp thống kê đó thành kết luận “thiếu phrase block”.

Một phép đọc metadata hữu ích loại bớt hướng không phù hợp:

| Tập train | Token p50 | Token p99 | Max token | Tỷ lệ cắt ở 256 |
|---|---:|---:|---:|---:|
| VSFC, dùng chung hai task | 14 | 56 | 164 | 0% |
| ViSFD | 39 | 139 | 241 | 0% |
| Restaurant | 20 | 60 | 115 | 0% |
| Hotel | 20 | 68 | 229 | 0% |

Nguồn: `artifacts/e8/<task>/seed_42/metadata.json`, trường `length_stats`. Đây là thống kê **train**, không phải kiểm tra mới toàn bộ validation/test. Nó đủ để giảm ưu tiên sửa long-context trong vòng đầu, không chứng minh dependency dài luôn dễ.

Tokenizer hiện có không khai báo normalizer, và sử dụng chuỗi pre-tokenization có bước ByteLevel. Cần đo fragmentation và offset thực tế, không đoán số subword từ số dấu cách. Không đồng nhất một token BPE, một đơn vị chính tả, một âm tiết và một từ.

## 4. Các nhóm bottleneck và mức độ bằng chứng

### 4.1. Đơn vị biểu diễn không khớp đơn vị hợp thành

PhoBERT dựa vào đầu vào đã word-segmented.[^5] Nhưng nghiên cứu AAAI 2025 về ViWordFormer phản biện việc bắt buộc segmentation và thiết kế attention có thông tin phrasal lexeme.[^6] Đây là bất đồng thực nghiệm và cách định nghĩa đơn vị, không phải một quy tắc ngôn ngữ đã được mọi bên thống nhất.

Suy luận nghiên cứu: nên để model cân nhắc nhiều cụm chồng lấp thay vì ép một phân đoạn duy nhất. Trong raw text, các cụm như “không đáng tiền”, “pin dùng lâu”, hoặc “giảng viên nhiệt tình” cần thứ tự và ngữ cảnh. Các ví dụ này minh họa thiết kế, không phải lỗi đã được quan sát ở checkpoint.

Điểm cần kiểm tra là model có khó hơn với tổ hợp ít gặp của các thành phần quen thuộc hay không, sau khi kiểm soát độ dài và nhãn. Một biểu diễn cụm hữu ích có thể giúp nhiều task. Tuy nhiên, nếu chỉ cải thiện vài aspect tên dài, đó vẫn có thể là lợi ích chuyên biệt.

### 4.2. Nhiễu chính tả, dấu và biến thể mạng xã hội

ViSoBERT cung cấp bằng chứng rằng biểu diễn phù hợp văn bản mạng xã hội có ích; nhưng công trình thay dữ liệu/pretraining nên không tách riêng nguyên nhân kiến trúc.[^7] ViLexNorm cho thấy chuẩn hóa giúp một số task nhưng có thể làm giảm spam detection; bỏ dấu nhiều cũng gây suy giảm và chuẩn hóa không luôn cứu được.[^8]

Suy luận nghiên cứu: một nhánh xử lý ký tự hoặc dấu có thể tăng độ bền, song không nên cưỡng ép mọi biến thể thành cùng một vector. Dấu thanh có thể thay nghĩa; viết kéo dài, emoji, cách viết bất thường cũng có thể là tín hiệu nhãn. “Bất biến với mọi nhiễu” là mục tiêu sai.

Nhóm này có bằng chứng ngôn ngữ tương đối tốt, nhưng chưa có số đo rằng nó giải thích phần lớn khoảng cách E8–BamiBERT. Vì vậy, đây là hướng dự phòng sau khi audit, không thêm đồng thời vào block cụm.

### 4.3. Attention cục bộ, toàn cục và chọn lọc thông tin

ConvBERT chỉ ra một phần head học quan hệ cục bộ và dùng dynamic convolution để xử lý local dependencies.[^9] Điều này gợi ý một đối chứng local mixer rất cần thiết: nếu convolution thường cũng đạt gain như module mới, chưa có bằng chứng cho cơ chế đặc thù tiếng Việt.

Differential Transformer dùng hiệu hai attention maps; DEX nghiên cứu cách đưa lợi ích liên quan vào model pretrained.[^10][^11] Các kết quả chính được đọc thuộc LLM, không xác nhận hiệu quả trên GoiBERT. Không thể gọi attention vào token ngoài aspect là “noise” nếu chưa thử can thiệp: token đó có thể chứa phủ định, đối lập hoặc quan hệ cần thiết.

Một phép thử hữu ích là thêm mệnh đề không đổi gold, và một phép thử đối nghịch là đổi phủ định làm đổi gold. Model tốt cần bỏ qua distractor nhưng vẫn nhạy với thay đổi nghĩa. Chỉ đo độ tập trung attention hoặc entropy thấp là không đủ.

### 4.4. Rank, head redundancy và đường truyền qua layer

Low-Rank Bottleneck in Multi-head Attention nghiên cứu hạn chế biểu diễn liên quan đến kích thước head.[^12] Nhưng student và teacher ở đây cùng head dimension 64. Không có cơ sở gọi chính kích thước đó là nguyên nhân teacher tốt hơn, hoặc tăng số head sẽ giải quyết được.

Tương tự, ưu thế của depth, normalization hay activation phải được đo bằng đối chứng. NeoBERT kết hợp RoPE, SwiGLU, RMSNorm và các thay đổi khác; trong ablation, dữ liệu và quy mô cũng đóng góp lớn.[^13] Không được gán toàn bộ gain cho một thay đổi kiến trúc.

ViDeBERTa đã đưa kiến trúc DeBERTa vào tiếng Việt và báo cáo kết quả tốt trên POS, NER, QA.[^22] Vì vậy, “đổi sang disentangled attention cho tiếng Việt” không phải đóng góp mới. Kết quả trên các task đó cũng không đảm bảo lợi ích trên sentiment/ABSA; khác biệt giữa các bảng paper không tự tách được tác động của kiến trúc, dữ liệu và fine-tuning.

### 4.5. Chi phí và độ dài context

ModernBERT kết hợp kiến trúc và tối ưu thực thi, gồm local/global attention và hỗ trợ context dài.[^14] Đây là nguồn thiết kế và baseline hiệu năng tốt. Tuy nhiên, hiệu quả hệ thống không đồng nghĩa tăng F1 của câu ngắn.

Đổi absolute positions sang RoPE trên checkpoint sẵn có cũng thay đổi hàm đã học; không thể coi là thao tác tương đương về đầu ra. Với số liệu độ dài hiện tại, sửa context được xếp sau hợp thành và robustness.

## 5. Tiền lệ gần: nơi novelty dễ bị đánh giá quá cao

| Công trình | Cơ chế liên quan | Hệ quả cho đề xuất mới |
|---|---|---|
| HuTieuBERT, ACL 2026 | Gated BMES fusion và fixed structural attention bias ở layer sớm | Thêm boundary embedding/bias vào encoder tiếng Việt đã có tiền lệ trực tiếp[^15] |
| ViWordFormer, AAAI 2025 | Phrasal scores điều chỉnh attention; co-text truyền thông tin qua tầng | Học phrase và truyền thông tin phrase qua layer không còn là ý tưởng trống[^6] |
| Tree Transformer, 2019 | Constituent attention khuyến khích cấu trúc cây | Soft grouping và hierarchical attention đã được nghiên cứu[^16] |
| ZEN, 2020 | Nhánh n-gram kết hợp vào encoder token | Chỉ thêm vector n-gram chưa đủ mới[^17] |
| Lattice-BERT, 2021 | Nhiều đơn vị chồng lấp và lattice position attention | “Nhiều cách phân đoạn cùng tồn tại” cũng có tiền lệ[^18] |
| Charformer, 2022 | Học soft subword blocks bằng gradient | Learned grouping từ đơn vị nhỏ cần đối chiếu GBST[^19] |
| CANINE, 2022 | Character input, downsampling, contextual encoder | Bỏ tokenizer hoặc thêm character path không tự là novelty[^20] |

HuTieuBERT là đối chứng ưu tiên vì chỉnh encoder pretrained và có đánh giá tiếng Việt. Trang tác giả mô tả structural bias có các hệ số cố định và lợi ích nổi bật ở tác vụ cú pháp; chưa có bằng chứng từ nguồn đã đọc rằng nó đánh bại BamiBERT trong protocol năm task này.[^15] Không lấy điểm từ bảng của họ so trực tiếp với bảng project khi task setup chưa khớp.

ViWordFormer dùng tích xác suất liên kết lân cận để điều chỉnh attention. Tính tích trong log-space cải thiện cách tính số, không loại sự suy giảm theo độ dài về mặt toán học; co-text là một phần khác của thiết kế. Thí nghiệm đọc được dùng Transformer sáu tầng hidden 512 và single run, không phải cùng initialization với GD/BamiBERT.[^6]

Đáng lưu ý, nghiên cứu phản biện Tree Transformer không tìm thấy bằng chứng mạnh rằng cấu trúc học được thực sự tương ứng tốt với constituency.[^21] Vì thế, block đề xuất nên được mô tả là **học span hữu ích cho task**, không tuyên bố tự khám phá đúng từ/cây cú pháp nếu chưa có đánh giá độc lập.

## 6. Xếp hạng hướng kiến trúc

Đây là đánh giá định tính phục vụ quyết định, không phải xác suất thắng hoặc điểm benchmark.

| Hướng | Liên hệ tiếng Việt | Tận dụng GD | Rủi ro trùng ý tưởng | Ưu tiên |
|---|---|---|---|---|
| Residual span composition, chọn span theo từng vị trí | Rõ: hợp thành nhiều đơn vị và thứ tự | Tốt, thêm block với init bảo toàn hàm | Cao; cần đối chứng HuTieu/ViWordFormer/ZEN | 1 |
| Nhánh ký tự/dấu có gate, giữ nguyên lexical path | Rõ: lỗi bề mặt và dấu | Khá, cần alignment chuỗi | Cao với character-aware/token-free | 2, nếu audit chỉ ra nhóm lỗi đủ lớn |
| Attention có thành phần signed/differential | Gián tiếp: chọn thông tin, scope | Khá nếu adapter tương thích | Cao với DIFF/DEX | 3, chỉ khi distractor audit có tín hiệu |
| RoPE + gated FFN + pre-normalization | Không đặc thù tiếng Việt | Khó hơn do đổi hàm checkpoint | Rất cao | Baseline hiện đại hóa, không đóng góp chính |
| Tăng width/depth hoặc MoE | Chủ yếu tăng capacity | Tùy cách chuyển trọng số | Rất cao | Đối chứng scale về sau |
| Thay toàn bộ tokenizer bằng byte/character | Có thể hữu ích cho noise | Kém hơn, thay interface nền | Cao và cần pretraining đáng kể | Chưa ưu tiên |

Không có hướng nào được bảo đảm vượt BamiBERT. Hướng 1 được chọn vì vừa có động cơ ngôn ngữ, vừa có phép thử phân biệt với “chỉ thêm tham số”, vừa triển khai được trên student sẵn có.

## 7. Thiết kế đề xuất: block hợp thành span trong encoder

### 7.1. Vị trí và đường dữ liệu

Vòng đầu đặt **một block sau transformer block 2, đánh số từ 1**, trước block 3. Đây là lựa chọn thiết kế sớm để còn tám block phía sau xử lý biểu diễn mới; không phải kết quả D1/D2 chỉ ra layer 2 bị lỗi. Không quét mười vị trí ngay từ đầu.

```text
Raw text → tokenizer hiện tại → encoder blocks 1–2
                                      │
                                      ├── đường hidden gốc ──────────┐
                                      │                             │
                                      └── gom theo offset           │
                                          → span có thứ tự          │
                                          → chọn span theo vị trí   │
                                          → chiếu về token ── cộng ─┘
                                                     │
                                  encoder blocks 3–10 → head CLS cũ
```

Không đổi label space, head, loss hay tokenizer vocabulary. Block chạy giống nhau cho VSFC và ABSA, không có query theo label. Đây là khác biệt thiết kế với E9: thay cách tạo contextual representation trước head, thay vì đổi cách đọc layer cuối.

### 7.2. Alignment không giả định từ chuẩn

Từ raw text xác định các đơn vị chính tả bằng khoảng trắng và punctuation, giữ nguyên dấu, hoa/thường và ký tự. Gọi chúng là **đơn vị chính tả**, không cam kết tất cả đều là âm tiết/từ/hình vị. Foreign words, emoji, số và chuỗi lỗi có thể không khớp ngôn ngữ chuẩn.

Dùng tokenizer offsets lập ma trận overlap giữa subword và đơn vị chính tả. Nếu một subword chạm nhiều đơn vị, dùng trọng số overlap đã chuẩn hóa; không gán cưỡng ép vào một đơn vị. Special tokens không vào span. Phần bị truncation chỉ dùng những ký tự còn hiện diện trong input; không cho branch đọc phần văn bản encoder không thấy.

Với hidden H ở đầu vào block, gom các token thành U bằng trung bình overlap. Việc gom này chỉ khởi tạo nhánh phụ: đường H gốc luôn còn, nên không làm mất toàn bộ tín hiệu vị trí của các subword.

### 7.3. Hợp thành có thứ tự

Với mỗi span liên tiếp dài 1, 2, 3 hoặc 4 đơn vị, xây vector từ phép chiếu riêng theo vị trí trong span:

`z[a,w] = GELU(sum(t=0..w-1, W[w,t] · U[a+t]) + b[w])`.

W ánh xạ từ hidden dimension d sang r=128 trong bản student đầu tiên. Không dùng mean đơn thuần làm vector cụm cuối, vì phép hợp thành cần phân biệt thứ tự. Các span không vượt punctuation separator đã chốt. Phủ định hoặc từ chức năng không bị lọc bằng stopword list.

Độ dài 4 chỉ là phạm vi local của pilot. Quan hệ xa vẫn do encoder gốc xử lý. Một thất bại ở phạm vi này không đủ loại bỏ mọi thiết kế hợp thành dài/hierarchical, nhưng cũng không được dùng lý do ấy để mở rộng vô hạn khi không có tín hiệu.

### 7.4. Cạnh tranh giữa các span chứa một vị trí

Tại đơn vị s, tập C(s) chứa tất cả span hợp lệ bao phủ s và một lựa chọn rỗng. Tính score từ U[s], vector z và vị trí tương đối của s trong span. Softmax trên C(s) tạo trọng số pi[s,c]. Vector của lựa chọn rỗng bằng zero, cho phép bỏ qua nhánh phụ.

`m[s] = sum(c in C(s), pi[s,c] · z[c])`.

Các trọng số này là lựa chọn của module, **không phải xác suất ranh giới ngôn ngữ đã được hiệu chuẩn**. Không gọi entropy của chúng là uncertainty đáng tin nếu chưa đo calibration. Không bắt các span trên toàn câu tạo thành một segmentation duy nhất.

Nhánh tạo delta cho token bằng phép phân phối overlap ngược, đã chuẩn hóa theo từng token:

`H_new = H + W_out · distribute(m)`.

Khởi tạo W_out và bias bằng zero, các ma trận bên trong bằng giá trị ngẫu nhiên khác zero. Khi bắt đầu, đầu ra đúng bằng H trong sai số số học; gradient bước đầu đi vào W_out, sau đó mới đi vào các tầng bên trong. Không đồng thời zero mọi projection vì có thể làm nhánh không học được.

### 7.5. Kích thước, chi phí và tính tương thích

Riêng mười ma trận W[w,t] cho w=1..4 có `10 × 384 × 128 = 491.520` trọng số. Cộng projection output, score và bias dự kiến dưới khoảng 0,7M cho cấu hình mô tả; phải đếm lại từ code thực. Đây là ước lượng thiết kế, không phải kết quả profiler. Không có lý do bắt buộc tăng lên hàng trăm triệu tham số để thử giả thuyết đầu tiên.

Số span là O(SK), số quan hệ vị trí–span là O(SK²) với K cố định bằng 4; không thêm global attention trên mọi cặp span. Không materialize tensor quá lớn khi có thể gather/scatter theo danh sách. FLOPs thấp hơn không bảo đảm latency thấp hơn, vì gather/scatter và padding có chi phí thực tế.

Giữ nguyên attention gốc giúp hạn chế thay đổi kernel và tận dụng weights đã học. Nhưng mới chỉ là lựa chọn triển khai, chưa bảo đảm tốc độ hoặc ổn định. Cần đo peak VRAM, thời gian train, inference batch1 và batch32 ở độ dài thực của dataset.

Các tham số block mới thuộc encoder, được gán LR của vị trí chèn trong cả hai nhánh đối chứng; không vô tình nhận LR head chỉ vì tên module. Giữ E8 optimizer cho tham số cũ. Nếu cần LR riêng cho module mới, phải khai báo trước và dùng chung cho đối chứng, không chọn riêng theo task sau khi thấy kết quả.

### 7.6. Đóng góp có thể bảo vệ và giới hạn novelty

Tổ hợp cụ thể được đề xuất là: span có thứ tự trên raw-offset units; cạnh tranh giữa span chứa từng vị trí với lựa chọn rỗng; phân phối residual về subword; giữ nguyên đường global pretrained. Chưa tìm thấy trong nguồn đã đọc một hệ thống đúng toàn bộ hợp đồng này trên student GoiBERT. **Việc chưa tìm thấy không chứng minh novelty toàn cầu.**

Ngay cả khi công thức không trùng nguyên văn, reviewers vẫn có thể coi đây là kết hợp incremental của các cơ chế đã biết. Muốn nâng thành đóng góp có sức nặng, cần chứng minh cơ chế mới giải quyết một failure mode đo được, vượt local mixer cùng tham số và một prior structural đã có, đồng thời chuyển được sang backbone/task khác.

Nếu chỉ hơn E8 nhưng ngang đối chứng convolution, kết luận phù hợp là local inductive bias có ích. Nếu hơn convolution nhưng ngang bản span uniform, chưa chứng minh lựa chọn span theo ngữ cảnh cần thiết. Nếu chỉ hơn trên dữ liệu nhiễu nhân tạo, chưa thể gọi thắng benchmark sạch.

## 8. Chẩn đoán trước khi đầu tư training

Không lặp lại toàn bộ CKA/probe. Chỉ thực hiện audit phục vụ giả thuyết mới trên train/validation và checkpoint có sẵn.

| Kiểm tra | Cách làm | Điều phân biệt được |
|---|---|---|
| Fragmentation | Đếm BPE trên mỗi đơn vị chính tả; chia nhóm bằng ngưỡng lấy từ train | Lỗi có tập trung ở input phân mảnh không |
| Tổ hợp ít gặp | Đếm span train; tìm tổ hợp hiếm gồm thành phần phổ biến | Có vấn đề hợp thành ngoài memorization không |
| Scope | Đọc mẫu phủ định/đối lập, kiểm tra gold phù hợp từng task | Model nhầm từ khóa hay quan hệ giữa chúng |
| Distractor | Cặp có/không có mệnh đề không đổi gold, được kiểm tra thủ công | Khả năng bỏ qua thông tin không liên quan |
| Nghĩa thay đổi | Cặp tối thiểu đổi phủ định hoặc quan hệ, gold được xác nhận lại | Tránh tối ưu nhầm invariance |
| Bề mặt | Lỗi dấu, viết tắt, ký tự lặp có thật trong validation | Có nên ưu tiên branch ký tự thay branch cụm |

Mẫu audit đầu có thể khoảng 100–200 câu phân bố trên các nhóm văn bản, không chọn toàn câu model sai. Nếu lấy mẫu theo lỗi, phải tách tập đối chứng đúng và không dùng tỷ lệ thô để suy ra prevalence toàn tập. Cặp tối thiểu là một diagnostic set, không thay thế benchmark và không được nhập vào train.

So sánh các nhóm cần kiểm soát độ dài, tần suất nhãn và domain. Có thể bắt đầu bằng bảng stratified đơn giản; chưa cần fit mô hình giải thích phức tạp. Tương quan chỉ giúp chọn hướng, còn can thiệp architecture và ablation mới kiểm tra đóng góp.

Một gate thực dụng: nếu fragmentation/rare-composition không liên quan lỗi, scope cũng không cho tín hiệu, hạ ưu tiên block cụm trước khi chạy nhiều vòng. Nếu chỉ có noise bề mặt, chuyển sang hướng ký tự/dấu. Không có nghĩa phải chờ diagnostic “chứng minh nhân quả” rồi mới được thử kiến trúc.

## 9. Kế hoạch thực nghiệm theo giai đoạn

### Giai đoạn A: tính đúng và screening

Mỗi trial khởi tạo từ `models/goiturn-s-gd`, không từ checkpoint E9/E6 hoặc checkpoint đã học task. Recipe E8 giữ nguyên: NC, LLRD, seed42, split và metric. Đầu tiên kiểm tra init parity, alignment, shape, gradient, padding và strict checkpoint reconstruction. Có kiểm tra riêng cho noisy branch: phải tính lại offsets theo noisy text, không tái dùng alignment của clean text.

Hai cấu hình kiến trúc trên đủ năm task:

1. **Local control:** nhánh convolution cục bộ trên cùng đơn vị chính tả, cùng vị trí chèn, residual init và số tham số gần tương đương; không có cạnh tranh span theo nội dung.
2. **Span candidate:** block mô tả ở mục 7, không thêm loss phụ hay KD.

Như vậy ngân sách screening đề xuất là **10 training trials seed42**, cộng tối đa một smoke nhỏ. E8 đã có kết quả nên dùng lại làm reference; chỉ replay khi code/runtime/data thay đổi đủ để làm mất tính đối chiếu. GPU1/2/3 chạy độc lập theo hàng đợi. Đây là kế hoạch đề xuất, chưa được triển khai bởi báo cáo này.

Ngưỡng ưu tiên vòng sau: ít nhất 4/5 task tăng so với E8, ít nhất hai task tăng từ 0,3 điểm F1, không task giảm quá 0,3; candidate cũng cần có lợi thế quan sát được so với local control. Báo vector năm delta, không chỉ mean. Các ngưỡng là heuristic về giá trị đầu tư, không là kiểm định thống kê.

Nếu candidate và control đều thất bại, dừng nhánh ở mức screening. Nếu chỉ Restaurant tăng, lưu kết quả task-specific nhưng không mở nhiều seed chỉ để biến nó thành một câu chuyện thắng rộng. Nếu candidate ngang control, ưu tiên model đơn giản hơn hoặc kiểm tra thêm một ablation hẹp trước khi đưa ra claim novelty.

### Giai đoạn B: phân biệt cơ chế khi có tín hiệu

Chỉ sau gate A, chọn các đối chứng có khả năng thay đổi kết luận:

- Bỏ cạnh tranh span, lấy uniform aggregation: kiểm tra vai trò content-dependent selection.
- Span dài 1: kiểm tra phần gain có phải chỉ do thêm nonlinear capacity.
- Structural bias kiểu HuTieuBERT trên cùng backbone: kiểm tra có vượt một giải pháp đã có không. Nếu không tái tạo đầy đủ thì ghi “inspired control”, không gán tên bản gốc.
- Can thiệp ranh giới/span trong một bản đối chứng riêng, giữ phân bố độ dài: kiểm tra ý nghĩa alignment. Không coi riêng việc làm hỏng module đã train là bằng chứng nhân quả đủ.
- Chỉ một kiểm tra vị trí sớm so với giữa encoder nếu cơ chế cần định vị; không mở grid toàn layer.

Nhiều seed chỉ dành cho cấu hình đã chọn và baseline tương ứng. Có thể dùng seeds42/43/44 như các vòng trước. Không cho phép mỗi task một thiết kế block khác nhau rồi gọi chung là một setting.

### Giai đoạn C: đối đầu BamiBERT và độ rộng

Chạy BamiBERT **gốc** bằng evaluator đã thống nhất, tách khỏi teacher DAPT. Phần recipe cần hai góc nhìn: cùng recipe để tách kiến trúc, và ngân sách tuning công bằng khi tuyên bố chất lượng tốt nhất. Không ép LR tối ưu của student sang model lớn rồi coi thất bại của model lớn là bằng chứng thắng.

Nếu student mới gần vượt hoặc vượt, chốt test các checkpoint chọn bằng validation. Nếu student mới vẫn thua nhưng có cải thiện rộng, chuyển cùng block lên BamiBERT gốc để thử khả năng tổng quát của cơ chế. Đây là mở rộng có lý do, không đồng nghĩa student đã bị chứng minh chạm trần.

Khi hướng đủ mạnh, bổ sung ViNLI, PhoNER_COVID19 và ViSpamReviews để bao phủ tám hàng benchmark. Vượt BamiBERT trên năm task hiện tại vẫn là thành quả, nhưng không thay thế bằng chứng trên suy luận và gán nhãn token. Trong vòng công bố cuối nên thêm mmBERT và đối thủ liên quan khi có tài nguyên.

## 10. Những rủi ro cần xử lý trong thiết kế, không giấu trong kết quả

**Thay kiến trúc thường làm đổi initialization.** Residual zero-output giúp bảo toàn hàm ban đầu, nhưng sau cập nhật còn khác optimization. Parameter-matched local control và giữ recipe mới giúp diễn giải gain.

**E9 không loại bỏ mọi attention-based architecture.** Nó thay head, trong khi block mới sửa hidden states và giữ head. Ngược lại, không được viện lý do “lần này ở trong encoder” để bỏ qua một thất bại thực nghiệm.

**Thông tin nhãn có thể là giới hạn.** Một câu có nhiều cách diễn giải hoặc annotation không nhất quán không tự được chữa bằng layer mới. Error audit phải phân biệt lỗi model và lỗi/ambiguity của gold, nhưng không sửa test gold để tăng điểm.

**Provenance GD cũ chưa hoàn toàn sạch về unlabeled validation.** Khi so các biến thể từ cùng checkpoint, yếu tố này được chia sẻ; vẫn cần ghi rõ khi công bố, không khẳng định pretraining holdout sạch. Tránh bổ sung train data lấy từ test qua một pipeline khác.

**Test đã được mở trong lịch sử project.** Các vòng tiếp theo cần ghi nhận mức độ thích nghi nghiên cứu với benchmark, khóa cấu hình trước test mới và bổ sung task/holdout chưa dùng để tăng sức nặng. Không giả vờ đây là lần đầu tiếp xúc test.

**BamiBERT paper chưa đủ để xác nhận mọi chi tiết reproduction.** Head, cách aggregate metric, seed và lineage bản mirror ABSA phải khớp trước khi tuyên bố hơn vài phần mười điểm. Đây là công việc cần làm cho claim cuối, không lý do để dừng screening kiến trúc.

**Một kiến trúc không tự có ngữ nghĩa khi chưa được học.** Fine-tune toàn encoder với nhãn thật là bước đầu phù hợp. Nếu muốn train thêm để ổn định block, phải cho baseline/control cùng budget; khi đó đóng góp vẫn là block nếu đã tách riêng được hiệu quả. Không đổi corpus, objective và kiến trúc đồng thời rồi quy gain cho kiến trúc.

## 11. Tiêu chí dừng và quyết định cuối

Ưu tiên chất lượng và novelty không có nghĩa phải bắt đầu bằng model lớn hơn. Student hiện tại cho một nền kiểm chứng nhanh, có nhiều kết quả đối chứng. Một thiết kế tốt nên tận dụng nền này trước, đồng thời giữ khả năng chuyển sang hidden768 nếu dữ liệu ủng hộ.

Nhánh span được coi là đáng tiếp tục khi nó vượt E8 trên nhiều task, có lợi thế so với local control và cải thiện đúng nhóm lỗi dự đoán. Nếu không, dừng hoặc đổi hypothesis, không thêm gate/loss/module vô hạn. Nhánh ký tự chỉ ưu tiên nếu lỗi bề mặt được đo; signed attention chỉ ưu tiên nếu kiểm tra distractor/scope cho tín hiệu.

Mục tiêu khoa học có thể bảo vệ là: **một phép hợp thành trong encoder cải thiện khả năng khái quát từ các thành phần quen thuộc sang tổ hợp mới, với bằng chứng qua task, đối chứng và backbone**. Mục tiêu sản phẩm là vượt BamiBERT trên protocol tương thích. Cần đạt cả hai để vừa có chất lượng, vừa có đóng góp kiến trúc đáng tin.

## 12. Nguồn và giới hạn truy cập

Các nguồn dưới là paper hoặc công bố trực tiếp của tác giả. Với HuTieuBERT, trang ACL và repository đọc được, nhưng endpoint PDF không tải thành công trong lượt rà soát; vì thế không khẳng định đã kiểm chứng đầy đủ mọi phương trình và phụ lục. ViWordFormer, ViLexNorm, ModernBERT, NeoBERT và DEX có nội dung toàn văn truy cập được. Việc tìm literature không thể bảo đảm bao phủ mọi công trình hoặc xác nhận tuyệt đối tính mới.

[^1]: Dat Quoc Nguyen, Thinh Pham, Chi Tran, Linh The Nguyen. [BamiBERT: A New BERT-based Language Model for Vietnamese](https://arxiv.org/html/2607.02259). arXiv v1, 02/07/2026. Nguồn cho kiến trúc, pretraining, phạm vi benchmark và protocol paper.
[^2]: Qualcomm AI Research. [BamiBERT config.json](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/raw/main/config.json). Cấu hình checkpoint chính thức, truy cập 11/09/2026.
[^3]: Marc Marone và cộng sự. [mmBERT: A Modern Multilingual Encoder with Annealed Language Learning](https://arxiv.org/abs/2509.06888). 2025. Nguồn cho đối thủ encoder đa ngôn ngữ hiện đại.
[^4]: Tin Van Huynh, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen. [ViNLI: A Vietnamese Corpus for Studies on Open-Domain Natural Language Inference](https://aclanthology.org/2022.coling-1.339/). COLING 2022.
[^5]: Dat Quoc Nguyen, Anh Tuan Nguyen. [PhoBERT: Pre-trained language models for Vietnamese](https://aclanthology.org/2020.findings-emnlp.92/). Findings EMNLP 2020; [repository chính thức](https://github.com/VinAIResearch/PhoBERT) mô tả preprocessing.
[^6]: Nghia Hieu Nguyen, Dat Tien Nguyen, Ngan Luu-Thuy Nguyen. [Vietnamese Words Are Not Constructed from Syllables: Rethinking the Role of Word Segmentation in Natural Language Processing for Vietnamese Texts](https://ojs.aaai.org/index.php/AAAI/article/view/34581/36736). AAAI 2025, pp. 24069–24077. Phần phương pháp và thiết lập thí nghiệm.
[^7]: Nam Nguyen, Thang Phan, Duc-Vu Nguyen, Kiet Nguyen. [ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing](https://aclanthology.org/2023.emnlp-main.315/). EMNLP 2023.
[^8]: [ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social Media Text](https://aclanthology.org/2024.eacl-long.85.pdf). EACL 2024. Đặc biệt §5 và Tables 4–5 về downstream và bỏ dấu.
[^9]: Zihang Jiang và cộng sự. [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://proceedings.neurips.cc/paper/2020/hash/96da2f590cd7246bbde0051047b0d6f7-Abstract.html). NeurIPS 2020.
[^10]: Tianzhu Ye và cộng sự. [Differential Transformer](https://arxiv.org/abs/2410.05258). 2024, ICLR 2025. Cơ chế attention signed; không phải bằng chứng riêng cho tiếng Việt.
[^11]: Chaerin Kong, Jiho Jang, Nojun Kwak. [Understanding Differential Transformer Unchains Pretrained Self-Attentions](https://arxiv.org/html/2505.16333v1). 2025. DEX và vấn đề sử dụng trọng số pretrained.
[^12]: Srinadh Bhojanapalli và cộng sự. [Low-Rank Bottleneck in Multi-head Attention Models](https://proceedings.mlr.press/v119/bhojanapalli20a/bhojanapalli20a.pdf). ICML 2020.
[^13]: Lola Le Breton và cộng sự. [NeoBERT: A Next-Generation BERT](https://arxiv.org/html/2502.19587v1). 2025. Kiến trúc và ablation; dùng phiên bản v1 được đọc.
[^14]: Benjamin Warner và cộng sự. [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/html/2412.13663v1). 2024; [bản ACL 2025](https://aclanthology.org/2025.acl-long.127.pdf).
[^15]: Anh Trac Duc Dinh, Khang Hoang Nhat Vo, Tai Tien Ta, Vinh Cong Doan, Tho Quan. [When Morphology Hides in Plain Sight: Breaking the Isolation in Vietnamese and Beyond](https://aclanthology.org/2026.acl-long.472/). ACL 2026. [Repository HuTieuBERT](https://github.com/TracDucAnh/HuTieuBERT). Trang ACL/repo đọc được; PDF chưa truy cập thành công.
[^16]: Yaushian Wang, Hung-Yi Lee, Yun-Nung Chen. [Tree Transformer: Integrating Tree Structures into Self-Attention](https://aclanthology.org/D19-1098/). EMNLP-IJCNLP 2019.
[^17]: Shizhe Diao và cộng sự. [ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations](https://aclanthology.org/2020.findings-emnlp.425/). Findings EMNLP 2020.
[^18]: Yuxuan Lai và cộng sự. [Lattice-BERT: Leveraging Multi-Granularity Representations in Chinese Pre-trained Language Models](https://aclanthology.org/2021.naacl-main.137/). NAACL 2021.
[^19]: Yi Tay và cộng sự. [Charformer: Fast Character Transformers via Gradient-based Subword Tokenization](https://arxiv.org/abs/2106.12672). 2021, ICLR 2022.
[^20]: Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting. [Canine: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://aclanthology.org/2022.tacl-1.5/). TACL 2022.
[^21]: [Tree Transformers are an Ineffective Model of Syntactic Constituency](https://arxiv.org/abs/2411.16993). 2024. Bằng chứng phản biện việc đồng nhất cấu trúc attention với cây cú pháp.
[^22]: Cong Dao Tran, Nhut Huy Pham, Anh Tuan Nguyen, Truong Son Hy, Tu Vu. [ViDeBERTa: A powerful pre-trained language model for Vietnamese](https://aclanthology.org/2023.findings-eacl.79/). Findings EACL 2023. Tiền lệ DeBERTa cho tiếng Việt và phạm vi đánh giá.

Nguồn nội bộ bổ sung: [D1_READOUT.md](D1_READOUT.md), [D2_READOUT.md](D2_READOUT.md), [E8_CONFIRMATION_RESULT.md](E8_CONFIRMATION_RESULT.md), [E8_ERROR_READOUT.md](E8_ERROR_READOUT.md), [E9_RESULT.md](E9_RESULT.md), cấu hình model và metadata E8. Conversation đính kèm được dùng để xác định lịch sử/quyết định; số liệu ưu tiên artifacts hiện hành.
