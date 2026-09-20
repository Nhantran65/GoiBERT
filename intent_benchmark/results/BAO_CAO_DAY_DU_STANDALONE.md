# Báo cáo thực nghiệm: Phân loại 8 intent tiếng Việt bằng Goi DAPT, Goi student và BamiBERT

**Ngày tổng hợp:** 13/09/2026  
**Phiên bản báo cáo:** 1.0 — kết quả pilot đã hoàn tất và kiểm toán độc lập  
**Đơn vị dự đoán:** một lượt khách hàng, có thể kèm lịch sử các lượt khách hàng trước đó  
**Phạm vi:** benchmark riêng cho intent classification; không gộp với các thí nghiệm cải tiến kiến trúc hoặc benchmark sentiment/ABSA khác của GoiBERT.

Báo cáo này chứa phương pháp, số liệu và diễn giải cần thiết để đọc độc lập khi sao chép sang nơi khác. Đường dẫn artifact ở cuối chỉ phục vụ truy xuất/tái lập, không phải điều kiện để hiểu các kết luận. Muốn chạy lại thí nghiệm vẫn cần dữ liệu và trọng số tương ứng; một file Markdown không thay thế gói tái lập.

## Mục lục đọc nhanh

- [Tóm tắt kết quả](#1-tóm-tắt-kết-quả)
- [Dataset và định nghĩa nhãn](#3-dữ-liệu-và-định-nghĩa-bài-toán) · [Split và chống trùng lặp](#4-chia-traindevtest-và-kiểm-soát-trùng-lặp)
- [Nguồn gốc ba model](#5-ba-checkpoint-và-nguồn-gốc-học-biểu-diễn) · [Protocol fine-tune](#6-protocol-fine-tune-và-lựa-chọn-model)
- [Kết quả dev](#9-kết-quả-validation-và-lựa-chọn-checkpoint) · [Kết quả test và confusion matrices](#10-kết-quả-test-chính)
- [Giải thích chênh lệch](#11-vì-sao-bamibert-có-điểm-thấp-hơn) · [Audit độc lập](#12-audit-độc-lập-và-kiểm-tra-độ-nhạy-nhãn)
- [Latency và memory](#13-kích-thước-latency-và-memory) · [Giới hạn kết luận](#14-giới-hạn-và-các-claim-có-thể-bảo-vệ)
- [Tái lập và checksums](#16-tái-lập-và-truy-xuất-artifact)
- [Phụ lục nhãn](#phụ-lục-a-target-thiếu-nhãn-và-các-nhãn-đã-thay-đổi) · [77 predictions test](#phụ-lục-b-predictions-trên-toàn-bộ-77-mẫu-test) · [Đường học sáu trials](#phụ-lục-c-đường-học-của-sáu-trials)

## 1. Tóm tắt kết quả

Trên một split cố định gồm **353 mẫu train, 77 mẫu dev và 77 mẫu test**, ba encoder được fine-tune cho cùng bài toán 8 lớp. Mỗi model thử hai learning rate với seed 42, tổng cộng sáu lượt train. Checkpoint được chọn bằng dev Macro-F1 trước khi đánh giá test.

**Goi DAPT đạt chất lượng cao nhất trong pilot:** Macro-F1 83,30%, accuracy 89,61% (69/77 mẫu đúng). **Goi student đạt Macro-F1 82,09%, accuracy 88,31% (68/77)** với 26,55 triệu tham số, so với 102,96 triệu ở hai model base. BamiBERT đạt Macro-F1 47,45%, accuracy 79,22% (61/77).

Kiểm toán độc lập chưa phát hiện lỗi nghiêm trọng trong extraction, input, chia nhóm/split, loading checkpoint, lựa chọn bằng dev hay tính metrics. Tuy nhiên, kết quả chỉ xác nhận lợi thế của các checkpoint Goi trong **pilot một seed trên bộ dữ liệu này**. Chưa xác định được trần hiệu năng BamiBERT, độ ổn định qua seed, hay nguyên nhân nhân quả của khoảng cách.

Ba lớp hiếm GREETING, GRATITUDE và PHONE có đúng hai mẫu test mỗi lớp. BamiBERT không dự đoán ra ba lớp này trên test; chúng đóng góp khoảng **84,2% chênh lệch Macro-F1 student–BamiBERT**. Vì vậy phải đọc Macro-F1 cùng accuracy, support từng lớp và số mẫu đúng.

## 2. Mục tiêu và phạm vi so sánh

Các câu hỏi thực nghiệm:

1. Các checkpoint Goi đã có có phù hợp để làm intent classifier cho hội thoại đặt món/dịch vụ bằng tiếng Việt không?
2. Trong cùng ngân sách fine-tune pilot, Goi DAPT và student so với BamiBERT gốc như thế nào?
3. Student đánh đổi bao nhiêu chất lượng để giảm kích thước và tài nguyên inference?
4. Chênh lệch kết quả có thể xuất phát từ lỗi pipeline, nhãn, split hoặc cách đo hay không?

Đây là **phân loại đơn nhãn, tập nhãn đóng**: mỗi mẫu nhận đúng một trong tám intent. Chưa đánh giá multi-label, intent ngoài tập nhãn, cơ chế từ chối dự đoán, calibration, hoặc hiệu quả vận hành bot đầu cuối. Nhãn CONFIRMATION cũng không đồng nghĩa với quyền thực hiện giao dịch hoặc tạo đơn thực tế.

Mọi số trong bảng benchmark chính là kết quả model chạy lại trong pipeline này, **không phải số trích từ paper BamiBERT**. Paper chỉ dùng để giải thích phạm vi tuyên bố SOTA và sự khác biệt về protocol [1].

## 3. Dữ liệu và định nghĩa bài toán

### 3.1. Nguồn và quy mô

Nguồn là bundle `prompt-bench-v1-250-standalone`, phiên bản `2.0.5-g6.8`, với dữ liệu tại:

```text
tmp/prompt-bench-v1-250-standalone/scripts/ai-verify/packs/prompt-bench-v1/cases.jsonl
```

Bundle này là tập case phục vụ kiểm thử hệ thống hội thoại; không nên coi nó là mẫu ngẫu nhiên đại diện cho mọi hội thoại khách hàng ngoài thực tế. Nguồn audit ghi các target turns có role `customer`, origin `sanitized_customer`. Không suy từ đó rằng toàn bộ tập đã được con người gán nhãn độc lập.

| Hạng mục | Số lượng |
|---|---:|
| Cases nguồn | 250 |
| Lượt khách hàng có thể làm target | 512 |
| Lượt có nhãn `expectedIntent` | 507 |
| Lượt thiếu nhãn, không dùng làm target | 5 |
| Mẫu sau rà nhãn | 507 |
| Mẫu được thay nhãn | 19 |
| Mẫu có cờ uncertainty, vẫn dùng bình thường | 82 |

Một case có thể sinh nhiều mẫu: mỗi lượt có nhãn là một target riêng, dùng các lượt khách hàng trước nó làm history. Các lượt thiếu nhãn không được dự đoán/đánh giá, nhưng vẫn có thể xuất hiện trong history của lượt sau nếu có trong nguồn. Năm target bị loại được liệt kê ở Phụ lục A.

### 3.2. Tám nhãn và quy tắc diễn giải

Nhãn lấy từ `expectedIntent`, không lấy từ `primaryIntent`. Trường `primaryIntent` thuộc một taxonomy nghiệp vụ khác (23 giá trị thực tế trong bundle) và không phải đầu vào model.

| ID | Nhãn | Ý nghĩa | Ví dụ minh họa từ dữ liệu/rubric | Ranh giới |
| --- | --- | --- | --- | --- |
| 0 | GREETING | Chào/gọi mở hoặc nối lại hội thoại. | “chào em” | Không lấn át yêu cầu cụ thể chỉ vì câu có lời chào. |
| 1 | CONFIRMATION | Đồng ý hoặc xác nhận rõ trong ngữ cảnh. | “đúng rồi” | Không mặc định yêu cầu tạo summary là xác nhận. |
| 2 | GRATITUDE | Cảm ơn hoặc bày tỏ appreciation. | “Cảm ơn em 🌷” | Câu mixed intent có thể cần ưu tiên hành động chính. |
| 3 | MENU_QUERY | Menu, món, giá, availability, gợi ý, lựa chọn món mới. | “tổng mới bao nhiêu” | Theo rubric đã chốt, câu hỏi tổng tiền vẫn thuộc lớp này; không phải mọi chỉnh sửa đơn đều thuộc lớp này. |
| 4 | COMPLAINT | Bất mãn, vấn đề dịch vụ hoặc tranh chấp hoàn tiền. | “Ship cao quá…” | Hủy đơn trung tính không tự động là complaint. |
| 5 | ADDRESS | Thông tin/cập nhật địa chỉ giao hàng. | “giao [ADDRESS_1]” | Không bao gồm địa chỉ quán hay địa chỉ xuất hóa đơn theo rubric này. |
| 6 | PHONE | Thông tin, sửa đổi, yêu cầu hoặc tham chiếu số điện thoại trong phone flow. | “số mình là [PHONE_1] nha” | Không yêu cầu chuỗi số phải hợp lệ; có thể là từ chối cung cấp số trong flow. |
| 7 | OTHER | Các hành vi ngoài bảy lớp trên. | “tóm tắt lại nha” | Gồm trạng thái đơn, summary, thanh toán, hóa đơn, vị trí/giờ mở cửa và hủy trung tính. |

Các ví dụ ngắn không phải định nghĩa đủ cho mọi ngữ cảnh. Các nhãn được giữ theo rubric thực nghiệm đã chốt; không sửa lại theo lỗi model sau khi xem test.

### 3.3. Rà nhãn và provenance

Quy trình rà sử dụng taxonomy và bản tài liệu `INTENT_CLASSIFIER_PROMPT.md` trong source snapshot. Prompt vận hành thực tế được quản lý trong database; snapshot không chứa bản ghi prompt đó, nên không thể xác nhận rubric tài liệu trùng từng byte với prompt từng tạo nhãn.

Đã chấp nhận 19 thay đổi: 12 mẫu train, 2 dev, 5 test. Nhãn gốc, nhãn đã rà và lý do được lưu trong ledger. Audit độc lập đọc lại 19 thay đổi: 12 có căn cứ trực tiếp theo rubric/ngữ cảnh; 7 có căn cứ nhưng vẫn phụ thuộc ngữ cảnh hoặc có cách hiểu cạnh tranh. Đây là review có AI hỗ trợ, không phải quy trình hai annotator con người và adjudication.

Theo quyết định phạm vi, **cả 82 mẫu có cờ uncertainty được train và chấm điểm như mọi mẫu khác**: không lọc, giảm trọng số hay tách metric chính. Việc rà nhãn không làm giảm số mẫu ngoài năm target thiếu nhãn.

### 3.4. Đầu vào thực tế của model

Mỗi mẫu chỉ chứa text của khách hàng, theo thứ tự thời gian. Không tạo giả câu trả lời assistant, không đưa trạng thái đơn từ fixture vào để bù context thiếu. Không đưa case ID, source ID, `expectedIntent`, `primaryIntent`, assertions hoặc future turns vào input.

Ví dụ định dạng (minh họa cấu trúc, không phải một dự đoán thực nghiệm):

```text
[HISTORY] customer: cho mình một tô phở
[HISTORY] customer: không hành nhé
[CURRENT] customer: cảm ơn em
```

Các dấu xuống dòng là ký tự newline thực. Mỗi đoạn được tokenize không kèm special tokens rồi ghép theo thứ tự; sau đó thêm BOS/EOS của tokenizer. Marker `[HISTORY]` và `[CURRENT]` là text input, không phải nhãn hay các embedding đặc biệt mới được thêm vào vocabulary.

Native tokenizer của cả ba checkpoint tạo cùng token IDs trên toàn bộ 507 serialized inputs. BOS/CLS=0, PAD=1, EOS/SEP=2, MASK=20480. API `vocab_size` báo 20.480 token cơ sở; vocabulary cấu hình model là 20.481 khi tính token mask bổ sung, không phải lỗi lệch một token.

Max length đặt 512, padding động theo batch. Khi vượt giới hạn, cắt token history cũ nhất trước; chỉ cắt đuôi current nếu riêng current đã vượt ngân sách. Thực tế train/dev không có mẫu bị cắt. Các input test trong phép timing dài 15–125 token, trung vị 36, cũng nằm dưới giới hạn.

Placeholder đã ẩn danh như `[ADDRESS_1]`, `[PHONE_1]` được giữ nguyên. Chúng cung cấp tín hiệu loại thực thể; kết quả trên dạng dữ liệu này không tự chứng minh chất lượng trên địa chỉ/số điện thoại raw trong production.

## 4. Chia train/dev/test và kiểm soát trùng lặp

Seed chia dữ liệu là 42. Tỷ lệ mục tiêu 70/15/15 được cân bằng cùng ràng buộc nhóm, nên số mẫu thực tế không đúng tỷ lệ tuyệt đối. Tập train dùng cập nhật trọng số; dev chọn epoch/LR; test chỉ đánh giá ba winner sau khi khóa selection.

| Split | Mẫu | Tỷ lệ | Cases |
| --- | --- | --- | --- |
| train | 353 | 69,63% | 175 |
| dev | 77 | 15,19% | 37 |
| test | 77 | 15,19% | 38 |

### 4.1. Phân bố nhãn

| Intent | Train | Dev | Test | Tổng |
| --- | --- | --- | --- | --- |
| GREETING | 8 | 2 | 2 | 12 |
| CONFIRMATION | 30 | 7 | 7 | 44 |
| GRATITUDE | 7 | 2 | 2 | 11 |
| MENU_QUERY | 101 | 22 | 21 | 144 |
| COMPLAINT | 25 | 5 | 5 | 35 |
| ADDRESS | 26 | 6 | 6 | 38 |
| PHONE | 8 | 2 | 2 | 12 |
| OTHER | 148 | 31 | 32 | 211 |
| Tổng | 353 | 77 | 77 | 507 |

Train có 148 mẫu OTHER và 101 MENU_QUERY: hai lớp chiếm 249/353, khoảng 70,54%. GREETING/GRATITUDE/PHONE chỉ có 7–8 mẫu train mỗi lớp. Đây là mất cân bằng đáng kể dù cả tám lớp đều xuất hiện ở mọi split.

### 4.2. Grouping và giới hạn chống leakage

250 cases được ghép thành 235 nhóm liên thông theo case/source, input trùng và các quan hệ gần trùng đã xác định. Quá trình chuẩn bị ghi nhận bốn họ full-input duplicate và chín họ template gần trùng được xác nhận sau rà candidates. Audit độc lập xác nhận:

- Không có case hoặc source vượt qua hai split.
- Cả 235 nhóm và chín họ template đã xác nhận nằm trong một split.
- Không có normalized full-input duplicate giữa các split.
- Vẫn có **10 họ current-text trùng** giữa các split khi history khác nhau, ví dụ “giao [ADDRESS_1]”, “ok”, “tổng mới bao nhiêu”.

Như vậy, split kiểm soát các nhóm đã biết; chưa chứng nhận mọi paraphrase/template tiềm ẩn đều độc lập. Lặp một câu ngắn với history khác không phải trùng toàn bộ input, nhưng có thể cho phép học thuộc tín hiệu từ vựng.

77 mẫu test thuộc **38 cases**, không phải 77 hội thoại độc lập. Đặc biệt hai mẫu PHONE test cùng một case. Các phép suy luận thống kê tương lai cần tôn trọng phụ thuộc theo case/group, không mặc định mọi turn độc lập.

## 5. Ba checkpoint và nguồn gốc học biểu diễn

### 5.1. Quan hệ giữa các model

```text
BamiBERT gốc
    │ tiếp tục học MLM, 40.000 bước (theo manifest)
    ▼
Goi DAPT: cùng kiến trúc base, trọng số đã học thêm
    │ distillation hidden states + attention, 30.000 bước
    ▼
Goi student: encoder nhỏ hơn

Sau đó: mỗi checkpoint được fine-tune RIÊNG với nhãn intent trong pilot này.
```

DAPT là *domain-adaptive pretraining*, nghĩa là tiếp tục pretraining trên corpus được lựa chọn. Với MLM, model đoán token bị che, ví dụ “Cho mình tô phở không [MASK] nhé”. Ví dụ này chỉ minh họa cơ chế học, không khẳng định câu đó thuộc corpus lịch sử. Một bước là một lần cập nhật trọng số theo batch, không phải một câu hoặc một epoch.

Student học cách tạo hidden states và attention gần teacher DAPT. Manifest ghi loss khớp hidden-state MSE và attention MSE; không dùng nhãn intent hoặc logits intent trong giai đoạn general distillation đó. Khi fine-tune intent hiện tại, cả ba dùng CE nhãn thật, không dùng teacher/KD.

Đây là so sánh checkpoint gốc với checkpoint đã học thêm và bản nén của nó. Không thể dùng thí nghiệm này để tách riêng lợi ích của kiến trúc, kích thước, domain adaptation và distillation.

### 5.2. Cấu hình model

| Thuộc tính | BamiBERT | Goi DAPT | Goi student |
| --- | --- | --- | --- |
| Thư mục backbone | models/bamibert-base-original | models/goibert-b-dapt | models/goiturn-s-gd |
| Số Transformer layers | 12 | 12 | 10 |
| Hidden size | 768 | 768 | 384 |
| Attention heads | 12 | 12 | 6 |
| FFN intermediate size | 3072 | 3072 | 1536 |
| Vocabulary cấu hình | 20481 | 20481 | 20481 |
| Max position embeddings | 2050 | 2050 | 2050 |
| Tham số encoder + head intent | 102957320 | 102957320 | 26548616 |

Đây là checkpoint encoder pretrained, không phải checkpoint đã fine-tune sentiment/ABSA hoặc model turn-completion của những thí nghiệm khác. Tham số báo cáo bao gồm head intent mới; không dùng số tham số của bản MLM có LM head để so sánh.

BamiBERT local có checksum khớp file chính thức trên Hugging Face [2]. Cấu hình base của DAPT giống BamiBERT; student có chiều rộng và số layer nhỏ hơn. Không thêm block nghiên cứu mới trong pilot này.

### 5.3. Provenance của DAPT/student

Theo manifest checkpoint:

| Thuộc tính lịch sử | DAPT | Student |
|---|---|---|
| Số bước đã hoàn tất | 40.000 | 30.000 |
| Điểm khởi đầu / teacher | BamiBERT | Goi DAPT, không fallback |
| Max length lịch sử | 256 | 256 |
| Batch size lịch sử | 32 | 32 |
| Learning rate lịch sử | 5e-5 | 1e-4 |
| Weight decay | 0,01 | 0,01 |
| Warmup steps | 1.000 | 1.000 |
| Seed | 42 | 42 |
| Objective | MLM, mask probability 0,3 | Hidden-state và head-mean attention MSE |

Manifest hai model cùng ghi hash thống kê corpus `1fa34627adddf9206f850e4595ecb0f88fdb47477eebb5beadb191ad20b26c80`. Đây là hash stats, **không phải checksum trực tiếp corpus**. Corpus lịch sử đầy đủ không có để kiểm toán độc lập overlap với bộ intent này. Các con số lịch sử là metadata kế thừa, không phải giai đoạn được chạy lại trong benchmark hiện tại; không có bằng chứng ở đây để kết luận bộ intent đã bị lộ vào pretraining, cũng không đủ chứng nhận hoàn toàn không overlap.

## 6. Protocol fine-tune và lựa chọn model

### 6.1. Head phân loại

```text
Pretrained encoder → hidden state của token đầu (CLS/BOS)
                  → Dropout(0,1)
                  → Linear(hidden, hidden)
                  → Tanh
                  → Dropout(0,1)
                  → Linear(hidden, 8)
```

Pretrained pooler bị tắt (`add_pooling_layer=False`). Head mới dùng chiều rộng native của backbone; toàn bộ encoder và head được cập nhật. Linear layers dùng khởi tạo mặc định PyTorch; đây là chi tiết khác với việc bắt buộc dùng Gaussian initialization của head Hugging Face. Không có bằng chứng nó gây lỗi riêng cho BamiBERT; recipe này được áp dụng cùng kiểu cho cả ba model.

### 6.2. Hyperparameters đã khóa

| Tham số | Giá trị |
| --- | --- |
| Trials | 3 model × 2 LR = 6; khởi tạo lại từ backbone cho mỗi trial |
| Seed | 42 cho Python / NumPy / Torch; seed cố định không đồng nghĩa bảo đảm bitwise trên mọi máy |
| Learning rates | 2e-5 và 5e-5 |
| Batch size | 16, không gradient accumulation, drop_last=False |
| Epoch tối đa | 15 |
| Early stopping | 3 epoch liên tiếp không tăng dev Macro-F1; min_delta=0 |
| Loss | Cross-entropy 8 lớp, không class weights; không lọc uncertainty |
| Optimizer | AdamW, betas=(0.9, 0.999), eps=1e-8 |
| Weight decay | 0,01 cho weights; 0 cho bias/LayerNorm |
| LR theo layer | Uniform; không LLRD |
| Scheduler | Linear warmup 10% tổng bước dự kiến, sau đó linear decay |
| Gradient clipping | Norm tối đa 1,0 |
| Precision | BF16 autocast, master weights FP32 |
| Attention | SDPA trên CUDA |
| Sequence length | Tối đa 512; padding động |
| Augmentation / KD / NC | Không áp dụng trong fine-tune intent |
| Selection metric | Macro-F1 fixed 8 labels, zero_division=0 |

Với 353 mẫu và batch 16, mỗi epoch có 23 optimizer steps; lịch dự kiến 15 epochs tương ứng 345 steps. Code làm tròn 10% thành 34 warmup steps bằng quy tắc `round` của Python. Early stopping không tính lại lịch LR theo số epoch thực tế.

### 6.3. Quy tắc selection và test

1. Evaluate dev sau mỗi epoch; lưu checkpoint khi Macro-F1 tăng nghiêm ngặt.
2. Nếu bằng điểm trong một trial, giữ epoch sớm hơn.
3. Sau khi cả sáu trial hoàn tất, chọn LR có best dev Macro-F1 cao nhất cho từng backbone. Nếu hai LR bằng điểm chính xác, chọn 2e-5.
4. Ghi selection và checksum checkpoint trước khi evaluator đọc test để chấm điểm.
5. Chỉ đánh giá test cho ba winner. Không chọn lại theo điểm test, không mở rộng grid sau khi thấy kết quả trong pilot.

Test đã được xử lý ở giai đoạn chuẩn bị/audit dữ liệu và hash được pin trước training; “không dùng test để selection” không có nghĩa không ai từng đọc text test khi rà nhãn. Sau benchmark, saved predictions được đọc lại cho audit và phân tích lỗi, không dùng để sửa checkpoint hoặc nhãn của kết quả chính.

## 7. Môi trường và thực thi

| Hạng mục | Cấu hình |
|---|---|
| GPU | NVIDIA L40S |
| Training | GPU 0 và GPU 1, một worker mỗi GPU |
| GPU khác | GPU 2/3 đang có workload khác, không sử dụng cho pilot |
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime ghi nhận | 12.4 |
| Transformers | 4.49.0 |
| Precision | BF16 autocast với trọng số FP32 |

Queue cố định: GPU0 chạy student@2e-5 → student@5e-5 → BamiBERT@5e-5; GPU1 chạy DAPT@2e-5 → DAPT@5e-5 → BamiBERT@2e-5. Controllers chạy detached, có PID, log, heartbeat và exit code. Cả sáu lượt hoàn tất thành công, không có trial thất bại trong queue cuối.

Trước training đã kiểm smoke forward/backward từng backbone: loss/gradients hữu hạn, có gradient khác 0 ở encoder và head, không thiếu backbone weights. BamiBERT/DAPT chỉ bỏ các MLM-head keys đúng thiết kế; student không có missing/unexpected keys. Smoke không thay đổi backbone nguồn và mỗi trial được nạp mới.

Người điều phối lập protocol, review và diễn giải kết quả; implementation/training/evaluation do execution agent GPT-5.6 Luna xhigh thực hiện. Một agent Luna xhigh riêng thực hiện audit độc lập, không chạy lại training hoặc sửa artifact gốc. Đây là kiểm toán có AI hỗ trợ, không phải đánh giá độc lập của con người.

## 8. Metrics và cách đọc bảng

Với mỗi lớp c:

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
Macro-F1    = (F1_1 + ... + F1_8) / 8
Accuracy    = số mẫu dự đoán đúng / tổng số mẫu
```

Quy ước chia cho 0 trả về 0. Support là số mẫu gold thuộc lớp đó. Mọi bảng dùng cùng thứ tự tám nhãn; confusion matrix có **hàng là gold, cột là prediction**. Tất cả phần trăm được tính từ số liệu không làm tròn rồi mới hiển thị; “điểm” chênh lệch là điểm phần trăm, không phải phần trăm tương đối.

Accuracy ưu tiên theo tần suất mẫu; Macro-F1 cho mỗi lớp trọng số ngang nhau. Do đó chỉ sáu mẫu thuộc ba lớp hiếm có thể tác động lớn đến Macro-F1. Đây là tính chất đúng của metric, không phải lỗi tính toán. Không báo confidence interval hay kiểm định ý nghĩa thống kê trong pilot này; chưa có multi-seed.

## 9. Kết quả validation và lựa chọn checkpoint

| Model | LR | Epoch tốt nhất | Epoch đã chạy | Dev Macro-F1 | Dev accuracy | Train loss cuối |
| --- | --- | --- | --- | --- | --- | --- |
| BamiBERT | 2e-05 | 11 | 14 | 50,29% | 75,32% | 0,3657 |
| BamiBERT | 5e-05 | 15 | 15 | 27,51% | 61,04% | 0,8109 |
| Goi DAPT | 2e-05 | 9 | 12 | 50,09% | 76,62% | 0,3028 |
| Goi DAPT | 5e-05 | 11 | 14 | 72,23% | 80,52% | 0,0362 |
| Goi student | 2e-05 | 11 | 14 | 62,64% | 79,22% | 0,2554 |
| Goi student | 5e-05 | 13 | 15 | 72,41% | 84,42% | 0,0219 |

Student@5e-5 có best dev Macro-F1 72,41%, chỉ nhỉnh DAPT@5e-5 (72,23%) khoảng 0,18 điểm. Checkpoint thắng của hai model đều dùng LR 5e-5; BamiBERT chọn LR 2e-5.

DAPT@2e-5 và BamiBERT@2e-5 gần như ngang dev Macro-F1. Khi tăng LR lên5e-5, DAPT tăng mạnh còn BamiBERT giảm. Vì vậy lợi thế checkpoint có tương tác mạnh với recipe. BamiBERT@5e-5 đạt best ở epoch cuối 15, train loss cuối 0,8109; chưa có bằng chứng model đã đạt trần có thể đạt dưới một recipe khác.

Đường học đầy đủ của sáu trial được đưa vào Phụ lục C để không chỉ báo các run tốt nhất.

## 10. Kết quả test chính

| Model | LR | Epoch | Macro-F1 | Accuracy | Đúng / 77 | Tham số |
| --- | --- | --- | --- | --- | --- | --- |
| Goi DAPT | 5e-05 | 11 | 83,30% | 89,61% | 69 | 102,957,320 |
| Goi student | 5e-05 | 13 | 82,09% | 88,31% | 68 | 26,548,616 |
| BamiBERT | 2e-05 | 11 | 47,45% | 79,22% | 61 | 102,957,320 |

DAPT hơn student **1,21 điểm Macro-F1 và một mẫu đúng**. Student hơn BamiBERT **34,64 điểm Macro-F1 và bảy mẫu đúng**; DAPT hơn BamiBERT **35,85 điểm Macro-F1 và tám mẫu đúng**.

Test tốt hơn dev ở cả ba winner trong lượt chạy này. Do hai tập nhỏ và khác cases, khác độ khó, không thể suy ra test đã được dùng để chọn model chỉ từ hiện tượng đó. Code và selection artifacts được kiểm tra riêng cho điều kiện không dùng test selection. Đồng thời, không nên chọn narrative chỉ dựa test thuận lợi; các bảng dev được giữ đầy đủ ở trên.

### 10.1. Precision / Recall / F1 từng intent

#### Goi DAPT

| Intent | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| GREETING | 2 | 100,00% | 50,00% | 66,67% |
| CONFIRMATION | 7 | 87,50% | 100,00% | 93,33% |
| GRATITUDE | 2 | 100,00% | 100,00% | 100,00% |
| MENU_QUERY | 21 | 90,91% | 95,24% | 93,02% |
| COMPLAINT | 5 | 100,00% | 40,00% | 57,14% |
| ADDRESS | 6 | 100,00% | 100,00% | 100,00% |
| PHONE | 2 | 100,00% | 50,00% | 66,67% |
| OTHER | 32 | 85,71% | 93,75% | 89,55% |

#### Goi student

| Intent | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| GREETING | 2 | 100,00% | 50,00% | 66,67% |
| CONFIRMATION | 7 | 87,50% | 100,00% | 93,33% |
| GRATITUDE | 2 | 100,00% | 100,00% | 100,00% |
| MENU_QUERY | 21 | 90,48% | 90,48% | 90,48% |
| COMPLAINT | 5 | 66,67% | 40,00% | 50,00% |
| ADDRESS | 6 | 100,00% | 100,00% | 100,00% |
| PHONE | 2 | 100,00% | 50,00% | 66,67% |
| OTHER | 32 | 85,71% | 93,75% | 89,55% |

#### BamiBERT

| Intent | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| GREETING | 2 | 0,00% | 0,00% | 0,00% |
| CONFIRMATION | 7 | 66,67% | 85,71% | 75,00% |
| GRATITUDE | 2 | 0,00% | 0,00% | 0,00% |
| MENU_QUERY | 21 | 86,36% | 90,48% | 88,37% |
| COMPLAINT | 5 | 40,00% | 40,00% | 40,00% |
| ADDRESS | 6 | 100,00% | 83,33% | 90,91% |
| PHONE | 2 | 0,00% | 0,00% | 0,00% |
| OTHER | 32 | 80,56% | 90,62% | 85,29% |

DAPT và student cùng chỉ nhận đúng 1/2 GREETING, 1/2 PHONE; cả hai chưa giải quyết hoàn toàn các lớp hiếm. Recall COMPLAINT đều 40% (2/5), nên đây cũng là điểm yếu còn lại ngay cả ở hai checkpoint Goi.

### 10.2. Confusion matrices

Các cột viết tắt theo bảng: G=GREETING, C=CONFIRMATION, T=GRATITUDE, M=MENU_QUERY, P=COMPLAINT, A=ADDRESS, H=PHONE, O=OTHER. Mỗi ô là **số mẫu**, không phải phần trăm.

#### Goi DAPT

| Gold \ Pred | G | C | T | M | P | A | H | O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GREETING | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| CONFIRMATION | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| GRATITUDE | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| MENU_QUERY | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 1 |
| COMPLAINT | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 3 |
| ADDRESS | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| PHONE | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| OTHER | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 30 |

#### Goi student

| Gold \ Pred | G | C | T | M | P | A | H | O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GREETING | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| CONFIRMATION | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| GRATITUDE | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| MENU_QUERY | 0 | 0 | 0 | 19 | 0 | 0 | 0 | 2 |
| COMPLAINT | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 3 |
| ADDRESS | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| PHONE | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| OTHER | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 30 |

#### BamiBERT

| Gold \ Pred | G | C | T | M | P | A | H | O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GREETING | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| CONFIRMATION | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 1 |
| GRATITUDE | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| MENU_QUERY | 0 | 0 | 0 | 19 | 1 | 0 | 0 | 1 |
| COMPLAINT | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 3 |
| ADDRESS | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 1 |
| PHONE | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| OTHER | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 29 |

### 10.3. So sánh paired trên cùng mẫu

| Cặp model (A / B) | Cả hai đúng | Chỉ A đúng | Chỉ B đúng | Cả hai sai |
| --- | --- | --- | --- | --- |
| BamiBERT / DAPT | 61 | 0 | 8 | 8 |
| BamiBERT / student | 59 | 2 | 9 | 7 |
| DAPT / student | 67 | 2 | 1 | 7 |

DAPT giữ đúng toàn bộ 61 mẫu BamiBERT làm đúng và sửa thêm 8 mẫu. Student sửa 9 lỗi BamiBERT nhưng sai thêm 2 mẫu khác. DAPT và student không tạo cùng predictions hoàn toàn: DAPT đúng riêng 2 mẫu, student đúng riêng 1 mẫu.

## 11. Vì sao BamiBERT có điểm thấp hơn?

### 11.1. Cơ chế số học đã xác nhận: lớp hiếm chi phối chênh lệch Macro-F1

Ở test, BamiBERT không dự đoán mẫu nào thành GREETING, GRATITUDE hoặc PHONE. F1 cả ba bằng 0. DAPT và student cùng đạt F1 tương ứng 2/3, 1 và 2/3.

Đóng góp của ba lớp này vào khoảng cách Macro-F1 là:

```text
[(2/3 − 0) + (1 − 0) + (2/3 − 0)] / 8 = 0,291667
```

Tức **29,17 điểm**, chiếm khoảng 81,4% khoảng cách DAPT–BamiBERT và 84,2% student–BamiBERT. BamiBERT vẫn có F1 MENU_QUERY 88,37%, ADDRESS 90,91% và OTHER 85,29%; kết quả không phải suy giảm đồng đều trên mọi intent.

Đây là phân rã chính xác của kết quả đã quan sát. Nó chưa chứng minh tại sao BamiBERT không học được các lớp đó trong lượt train này.

### 11.2. Bằng chứng về optimization

Cùng LR 2e-5, dev Macro-F1 BamiBERT 50,29% và DAPT 50,09%. LR 5e-5 giúp DAPT lên 72,23% nhưng BamiBERT chỉ 27,51%. Kết quả này không phù hợp với lời giải thích đơn giản rằng “DAPT luôn tốt hơn bất kể cách train”.

Cả ba có cùng ngân sách thử nghiệm downstream nhưng cùng ngân sách không bảo đảm cùng mức độ tối ưu. Dữ liệu rất ít, head mới, seed khởi tạo, learning rate, lịch decay và early stopping có thể tương tác với từng checkpoint. Các artifact chỉ cho thấy độ nhạy recipe; chưa có thí nghiệm can thiệp để xác định yếu tố gây ra khoảng cách. Không thể hứa giảm LR hoặc tăng epoch chắc chắn giúp BamiBERT bắt kịp.

### 11.3. Lợi ích học thêm là giả thuyết hợp lý, chưa phải kết luận nhân quả

DAPT bắt đầu từ BamiBERT và được học thêm MLM; student kế thừa tín hiệu từ DAPT qua distillation. Kiến thức và phân bố biểu diễn trước fine-tune có thể thuận lợi hơn cho dữ liệu hội thoại này. Tuy nhiên, thiếu control tách riêng corpus, compute, objective, seed và recipe; corpus lịch sử cũng chưa có để audit trực tiếp. Do đó báo cáo chỉ ghi **khả năng đóng góp**, không khẳng định adaptation là nguyên nhân đã được chứng minh.

### 11.4. Vì sao không mâu thuẫn với SOTA của BamiBERT?

Paper BamiBERT báo cáo tốt nhất 11/15 metrics trên tám benchmark tiếng Việt thuộc NLI, NER, sentiment/topic, spam và ABSA; bộ intent hội thoại này không nằm trong đó [1]. SOTA có phạm vi benchmark, nhóm model và protocol cụ thể, không bảo đảm đứng đầu trên mọi dataset.

Paper dùng AdamW, batch 32, grid LR {1e-5, 2e-5, 5e-5}, train 30 epochs và chọn checkpoint bằng validation [1]. Pilot này dùng batch 16, hai LR, tối đa 15 epochs, patience 3 và một seed. Vì vậy đây là common-pipeline pilot, không phải tái lập đầy đủ điều kiện fine-tune của paper.

### 11.5. Những giải thích chưa được bằng chứng ủng hộ

- Không thấy tải nhầm BamiBERT: checksum local khớp weights chính thức [2].
- Không thấy tokenizer bất lợi riêng: IDs giống nhau trên 507 inputs.
- Không thấy thiếu backbone weights hoặc dùng nhầm task checkpoint.
- Không thấy dùng test để lựa chọn LR/epoch.
- Không thể quy lợi thế Goi cho năm nhãn test đã sửa: kiểm tra dưới đây cho thấy đổi nhãn test không làm đảo thứ hạng.

## 12. Audit độc lập và kiểm tra độ nhạy nhãn

### 12.1. Phạm vi và kết quả audit

Audit đọc raw bundle, corrected/split JSONL, ledger, script training/evaluation, manifest/checksum model, summaries, selection và saved predictions. Audit không train lại hoặc chạy model mới; tokenization và tính lại metrics trên artifact là các kiểm tra dữ liệu/code, không phải thí nghiệm model mới.

| Hạng mục | Kết quả | Giới hạn diễn giải |
| --- | --- | --- |
| Extraction | 250 cases / 512 targets / 507 nhãn / 5 thiếu khớp nguồn | Không tự chứng nhận nhãn là human gold |
| Input/history | Không phát hiện metadata hoặc future-turn leakage | Thiếu assistant context là thuộc tính của snapshot |
| Groups/split | 235 groups, 9 họ template split-local; full-input overlap=0 | Không loại hết paraphrase/template chưa biết |
| Tokenization | Cùng IDs trên 507 mẫu, vocab/special IDs khớp | Không đo lợi ích một tokenizer khác |
| Checkpoint | Bami SHA khớp chính thức, không missing backbone | Corpus lịch sử chưa kiểm được |
| Dev predictions | Cả 6 files khớp IDs/gold/metrics | Không tái chạy stochastic training |
| Test predictions | 3 files × 77 IDs duy nhất; gold/label/probabilities/argmax khớp | Chưa đo độ ổn định seed/split |
| Selection | Winner theo dev, checkpoint hashes nhất quán | Không chứng minh recipe đã tối ưu |
| Timing | Cùng GPU, warmup, synchronization, forward-only | Không phải latency phục vụ production |
| Tài liệu | README ban đầu còn mô tả phạm vi chỉ chuẩn bị dữ liệu | Lỗi diễn đạt phạm vi, không ảnh hưởng execution |

Không phát hiện lỗi nghiêm trọng làm sai headline metrics trong phạm vi kiểm tra. Một audit dựa artifact không thể chứng minh không tồn tại mọi loại lỗi hoặc xác nhận lịch sử pretraining ngoài dữ liệu có sẵn.

### 12.2. Năm nhãn test đã sửa có tạo lợi thế Goi không?

Giữ nguyên saved predictions rồi chấm lại bằng nhãn raw source là phép sensitivity hậu kiểm, **không dùng chọn model**, không thay thế benchmark chính. Trên năm mẫu test đã sửa, BamiBERT/DAPT đúng 5/5 theo nhãn rà, student đúng 4/5; theo nhãn nguồn cả ba đúng 0/5.

| Model | Macro-F1 theo nhãn rà (chính) | Macro-F1 theo nhãn nguồn (sensitivity) |
| --- | --- | --- |
| Goi DAPT | 83,30% | 78,04% |
| Goi student | 82,09% | 77,20% |
| BamiBERT | 47,45% | 42,41% |

Nhãn thay đổi ảnh hưởng điểm tuyệt đối nhưng không giải thích lợi thế Goi trong chấm test. Kết quả này không xác định tác động của 12 nhãn train đã sửa lên quá trình học; muốn kiểm điều đó cần một thí nghiệm mới. Không lọc bất kỳ mẫu uncertainty nào trong phép đánh giá chính.

## 13. Kích thước, latency và memory

### 13.1. Protocol timing

Cả ba winner chạy lần lượt trên cùng NVIDIA L40S/GPU0, batch 1, BF16 autocast với master weights FP32. Mỗi model có 10 warmup forwards và 2 lượt qua 77 test inputs, tổng 154 lần forward được đo. CUDA synchronize trước/sau mỗi lần đo.

Thời gian được đo là **model-forward-only**, không gồm tokenizer, padding, chuyển CPU→GPU, mạng, hàng đợi hoặc logic ứng dụng. Length distribution: min 15, median 36, max 125 tokens. Đo cùng input không có nghĩa đại diện workload dài 512 tokens hay throughput batch lớn.

| Model | Params | Mean ms | Median ms | P95 ms | Inference allocated GiB | Inference reserved GiB |
| --- | --- | --- | --- | --- | --- | --- |
| Goi DAPT | 102,957,320 | 4,016 | 3,978 | 4,182 | 0,576 | 0,594 |
| Goi student | 26,548,616 | 3,467 | 3,440 | 3,681 | 0,146 | 0,168 |
| BamiBERT | 102,957,320 | 4,025 | 3,975 | 4,342 | 0,575 | 0,594 |

### 13.2. Peak memory khi training

| Model | LR | Peak allocated GiB | Peak reserved GiB |
| --- | --- | --- | --- |
| BamiBERT | 2e-05 | 2,478 | 2,770 |
| BamiBERT | 5e-05 | 2,478 | 2,770 |
| Goi DAPT | 2e-05 | 2,478 | 2,770 |
| Goi DAPT | 5e-05 | 2,478 | 2,770 |
| Goi student | 2e-05 | 0,842 | 0,973 |
| Goi student | 5e-05 | 0,842 | 0,973 |

GiB=2³⁰ bytes. Các số allocated/reserved là peak do PyTorch báo, không phải toàn bộ VRAM process/device và không phải cam kết cấu hình VRAM tối thiểu để deployment. CUDA context, allocator, tiến trình khác và length/batch khác có thể làm tổng bộ nhớ khác đi.

Student có khoảng 25,79% số tham số của model base, tức giảm 74,21% hay khoảng 3,88 lần nhỏ hơn. Trong phép đo này, mean latency student thấp hơn BamiBERT khoảng 13,88% (speedup khoảng 1,16×); **không nhanh hơn 3,9×** chỉ vì ít tham số 3,9×. Chi phí launch/kernel, phần cố định và workload câu ngắn có thể ảnh hưởng tốc độ; chưa có profiling để phân rã nguyên nhân.

Inference allocated peak student khoảng 0,146 GiB so với 0,575–0,576 GiB ở base. Timing một phiên trên máy dùng chung chưa đủ để chứng nhận latency production hoặc khác biệt tốc độ nhỏ giữa DAPT và BamiBERT có ý nghĩa ổn định. Không suy từ mean latency ra throughput phục vụ nhiều người dùng.

## 14. Giới hạn và các claim có thể bảo vệ

### 14.1. Giới hạn

1. **Một seed, một split:** chưa có độ lệch chuẩn, confidence interval hoặc kiểm định paired theo group.
2. **Test nhỏ:** 77 turns, 38 cases; lớp hiếm có 2 mẫu. Một sai khác có thể đổi đáng kể F1 lớp.
3. **Nguồn là bundle kiểm thử:** không đại diện đầy đủ production, dialect, typing noise, lịch sử assistant hoặc intent mới.
4. **Gold chưa độc lập bằng con người:** 19 thay đổi được giải thích theo rubric; 82 mẫu mơ hồ giữ nguyên vai trò bình thường.
5. **Tín hiệu placeholder và template:** có thể giúp học từ khóa; không chứng minh tổng quát hóa sang dữ liệu raw mới.
6. **Ngân sách fine-tune hẹp:** recipe chung có thể chưa phù hợp cho BamiBERT; không gọi nó là baseline đã tune đến trần.
7. **Lịch sử checkpoint khác nhau:** DAPT/student đã nhận thêm training; không tách được kiến trúc khỏi data/compute/objective.
8. **Corpus lịch sử thiếu:** không audit được overlap pretraining với dataset hiện tại.
9. **Latency phạm vi hẹp:** model-forward-only trên L40S, batch 1, câu ngắn; không gồm model startup hoặc application overhead.
10. **Chưa đánh giá production behavior:** không có calibration, abstention, out-of-domain, robustness hoặc chi phí sai intent thực tế.

### 14.2. Cách diễn đạt kết luận

Có thể viết:

> Trong pilot phân loại 8 intent tiếng Việt với 353/77/77 mẫu và protocol fine-tune thống nhất, Goi DAPT đạt Macro-F1 test 83,30%, Goi student 82,09%, so với 47,45% của checkpoint BamiBERT gốc. Student dùng 26,55 triệu tham số so với 102,96 triệu ở base và chỉ kém DAPT một mẫu đúng trên 77 mẫu test. Chênh lệch chủ yếu tập trung ở ba lớp hiếm. Kết quả chưa xác nhận thứ hạng ngoài pilot hoặc nguyên nhân nhân quả của lợi thế.

Chưa đủ bằng chứng để viết: “Goi đạt SOTA tiếng Việt”, “student thông minh hơn BamiBERT nói chung”, “DAPT chắc chắn gây ra toàn bộ cải thiện”, “BamiBERT không phù hợp intent classification” hoặc “student nhanh hơn gần 4 lần”.

## 15. Hướng kiểm chứng tiếp theo — chưa thực thi

Ưu tiên độ tin cậy của baseline trước khi sửa kiến trúc:

1. Khóa protocol bổ sung trên dev: thêm LR 1e-5 và ngân sách epoch dài hơn; giữ input/head cố định để đọc được tác động, không đổi hàng loạt yếu tố cùng lúc.
2. Sau diagnostic, áp ngân sách so sánh tương đương cho cả ba backbone. Cấu hình chọn hoàn toàn bằng dev, không sửa để đuổi theo test đã thấy.
3. Kiểm tra seed mới cho các cấu hình có tín hiệu; nếu cần claim ổn định, báo mean/std và phân tích theo case/group.
4. Thu thập tập đánh giá độc lập, tăng support GREETING/GRATITUDE/PHONE và COMPLAINT, có hội thoại gần production; giữ bộ test hiện tại như benchmark đã quan sát, không gọi nó là test mới chưa từng xem.
5. Chỉ khi cần attribution khoa học, xây control cùng data/compute để tách adaptation, distillation và recipe; cần corpus provenance để đánh giá overlap.

Về lựa chọn thực dụng hiện tại, student là ứng viên phù hợp để thử tích hợp vì chất lượng gần DAPT và giảm tài nguyên; đây là lựa chọn pilot, chưa phải xác nhận sẵn sàng deployment. Nếu chỉ ưu tiên điểm test trong lượt chạy này, DAPT cao nhất, nhưng chênh một mẫu chưa đủ khẳng định nó luôn tốt hơn student.

## 16. Tái lập và truy xuất artifact

Để tái lập cần cùng source snapshot, nhãn đã rà, assignments theo group, tokenizer/backbone bytes, code và môi trường; chỉ đặt seed 42 rồi random split lại không bảo đảm tạo đúng tập đã dùng. GPU/library khác có thể cho kết quả khác dù cùng seed. Không dùng bảng test dưới đây để tune một run mới rồi gọi là đánh giá độc lập.

Các artifact của lần chạy:

| Nội dung | Đường dẫn tương đối trong project |
|---|---|
| Dữ liệu và provenance | `intent_benchmark/corrected_dataset.jsonl`, `provenance.json` |
| Nhãn và ledger | `label_map.json`, `review_ledger.jsonl`, `review_changes.csv` trong `intent_benchmark/` |
| Splits | `intent_benchmark/train.jsonl`, `dev.jsonl`, `test.jsonl` |
| Grouping | `intent_benchmark/group_manifest.json`, `split_assignments.jsonl`, `duplicate_analysis.json` |
| Protocol | `intent_benchmark/TRAINING_PLAN.md`, `training/training_config.json` |
| Runner | `intent_benchmark/training/train_intent.py` |
| Logs / best weights | `intent_benchmark/runs/<trial>/` |
| Selection | `intent_benchmark/results/selection.json` |
| Test metrics / predictions | `intent_benchmark/results/test_metrics_*.json`, `test_predictions_*.jsonl` |
| Timing | `intent_benchmark/results/timing.json` |
| Audit độc lập | `intent_benchmark/audit_independent/AUDIT_REPORT.md`, `evidence.json`, `audit.py` |

File Markdown này không nhúng trọng số hay corpus. Các bảng bên dưới cung cấp đủ kết quả để kiểm tra tổng, đọc lỗi và trích dẫn mà không cần mở các artifact đó.

### 16.1. Checksum dữ liệu/code

| Artifact | SHA256 |
| --- | --- |
| Source cases.jsonl, version 2.0.5-g6.8 | bd29e71cc49a7ccf92f9287d0d8c74d6d1240cd157cc42027a62e9f5c7cdcc19 |
| intent_benchmark/train.jsonl | 775475397d9bb28d006a842a6201efe532b0da65d3562b97cad425b34cf9331c |
| intent_benchmark/dev.jsonl | 184c0efb674aefae99854b2090bd693990f62a188739131a60378ff0df13b7ae |
| intent_benchmark/test.jsonl | 33eb33cc8292650c320b167d52eb5c979e6591acd36a5b27a3336d6b6b54f4a5 |
| intent_benchmark/label_map.json | 23063af163dd52c2d9afddd3d0f527b732a00380f6592c8c74d0568a96dc86a2 |
| intent_benchmark/training/train_intent.py | 845fc50b94beadb184a6ff57cb28d1b90b29e092846b4dffb2b1b3059756c3eb |
| intent_benchmark/training/training_config.json | 6fd483168dc3cff84f64a8df800121a3038f28f43bfa244ffb9a7869329fcda9 |
| intent_benchmark/results/selection.json | 5344ff6261357731768c1ca2fe1eb467e5841566cdcebda11c5b389fa6e9cb90 |

### 16.2. Checkpoint đã dùng để test

| Model | Checkpoint | SHA256 |
| --- | --- | --- |
| Goi DAPT | intent_benchmark/runs/goi_dapt-lr5em5/best.pt | 1154f032eb3702316d37b82ca04c5653eae01d03c2eaa6f8ddc3e57626a6474f |
| Goi student | intent_benchmark/runs/goi_student-lr5em5/best.pt | b320d32a3d1e93e2361a349481b9fea22cc3351b62fa5ae6a376495a600be93c |
| BamiBERT | intent_benchmark/runs/bamibert-lr2em5/best.pt | 55cc8ca041d0ea7c8ff0d9bd38fd027fe936bb8b14cd7482f42130ea15ae465c |

## 17. Nguồn tham khảo

[1] Dat Quoc Nguyen, Thinh Pham, Chi Tran, Linh The Nguyen. **BamiBERT: A New BERT-based Language Model for Vietnamese**. arXiv:2607.02259v1, 2026. [Bản paper trên arXiv](https://arxiv.org/html/2607.02259v1). Dùng để đối chiếu phạm vi benchmark và protocol; không lấy điểm paper thay cho kết quả intent thực nghiệm.

[2] Qualcomm AI Research. **BamiBERT — official model repository và weights**. [Model card](https://huggingface.co/Qualcomm-AI-Research/BamiBERT) · [Weights và checksum chính thức](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/blob/main/model.safetensors). SHA256 file weights đối chiếu: `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`.

[3] **Nguồn nội bộ của thực nghiệm:** bundle `prompt-bench-v1`, version `2.0.5-g6.8`; manifests của `goibert-b-dapt` và `goiturn-s-gd`; logs/summaries/predictions của sáu trials; audit độc lập. Đây là artifacts project, không phải các nguồn công bố độc lập. Các số liệu chính và phương pháp từ chúng được đưa trực tiếp vào báo cáo này.

## Phụ lục A. Target thiếu nhãn và các nhãn đã thay đổi

### A.1. Năm target không đưa vào benchmark

| Sample ID | Text |
| --- | --- |
| pb-v1-0001:t1 | Gần tới chưa em |
| pb-v1-0004:t1 | Nãy c trả 170k |
| pb-v1-0059:t1 | Ok thanks b |
| pb-v1-0059:t2 | Tới chưa b? |
| pb-v1-0061:t1 | Gần tới chưa em |

### A.2. Danh sách đầy đủ 19 thay đổi nhãn

| Sample ID | Split | Nhãn nguồn → nhãn rà | Current text |
| --- | --- | --- | --- |
| pb-v1-0019:t0 | train | ADDRESS → OTHER | Xuất hoá đơn e [ADDRESS_1] |
| pb-v1-0043:t0 | train | COMPLAINT → OTHER | Hủy đơn trên Shopee F dùm nhe. |
| pb-v1-0045:t0 | train | COMPLAINT → OTHER | Chị có việc nên em hủy bàn giúp chị. Thanks em |
| pb-v1-0047:t0 | dev | OTHER → COMPLAINT | Ship cao quá để khi khác ra ăn tại quán, cám ơn. Cho cô không mua nhe |
| pb-v1-0062:t0 | test | ADDRESS → OTHER | Shop ơi cho mình hỏi chỗ mình đi đường nào vô cho dễ, mình đi kiếm mà kg ra |
| pb-v1-0063:t0 | train | MENU_QUERY → OTHER | Đia chi quán |
| pb-v1-0181:t3 | test | OTHER → CONFIRMATION | đủ rồi |
| pb-v1-0183:t3 | train | OTHER → CONFIRMATION | xong |
| pb-v1-0184:t3 | train | OTHER → CONFIRMATION | được rồi |
| pb-v1-0185:t3 | test | OTHER → CONFIRMATION | vậy thôi |
| pb-v1-0189:t3 | train | OTHER → CONFIRMATION | vậy nha |
| pb-v1-0215:t4 | dev | MENU_QUERY → OTHER | tóm tắt lại nha |
| pb-v1-0218:t4 | train | MENU_QUERY → OTHER | làm summary mới |
| pb-v1-0220:t4 | train | MENU_QUERY → OTHER | summary lại nha |
| pb-v1-0222:t2 | test | CONFIRMATION → OTHER | 19h, COD, lên summary giúp mình |
| pb-v1-0222:t3 | test | CONFIRMATION → OTHER | 19h, COD, lên summary giúp mình nha em |
| pb-v1-0228:t7 | train | MENU_QUERY → OTHER | tóm tắt giúp |
| pb-v1-0229:t7 | train | MENU_QUERY → OTHER | cho summary |
| pb-v1-0240:t1 | train | COMPLAINT → OTHER | à với hủy giùm mình cái đơn đặt hồi trưa nha, mình đặt nhầm — còn đơn cơm đang đặt đây thì vẫn lấy đó |

Lý do theo nhóm: địa chỉ quán/hóa đơn khác địa chỉ giao; hủy trung tính khác complaint; phản đối phí ship có thể là complaint; lời chốt sau lựa chọn có thể là confirmation; yêu cầu summary thuộc thao tác đơn/OTHER. Những câu mixed act hoặc ngắn vẫn có thể có cách hiểu khác; danh sách không khẳng định mọi thay đổi là gold tuyệt đối.

## Phụ lục B. Predictions trên toàn bộ 77 mẫu test

Bảng chứa current text và số lượt history để truy xuất lỗi; model thực tế còn đọc nội dung history. Không nên chấm lại các câu phụ thuộc ngữ cảnh chỉ dựa current text trong bảng. Dấu ✓ nghĩa prediction khớp nhãn đã rà; nhãn gold là target thực nghiệm, không phải khẳng định khách quan cho mọi cách diễn giải. Không đưa bảng này vào dữ liệu train cho lần đánh giá độc lập khác.

| Sample ID | History turns | Current text | Gold | BamiBERT | Goi DAPT | Goi student |
| --- | --- | --- | --- | --- | --- | --- |
| pb-v1-0005:t0 | 0 | Chị nhận được tiền hoàn rồi nha. Mai chị ck cho em nhaz | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0008:t0 | 0 | Cảm ơn em 🌷 | GRATITUDE | OTHER | GRATITUDE ✓ | GRATITUDE ✓ |
| pb-v1-0010:t0 | 0 | Bắt đầu | GREETING | MENU_QUERY | OTHER | MENU_QUERY |
| pb-v1-0037:t0 | 0 | Phí ship tính như thế nào? | MENU_QUERY | COMPLAINT | OTHER | OTHER |
| pb-v1-0046:t0 | 0 | ad ơi chắc thui đông quá add cancel đơn giúp mình nhen | COMPLAINT | OTHER | OTHER | OTHER |
| pb-v1-0049:t0 | 0 | Thôi C huỷ nha | OTHER | COMPLAINT | OTHER ✓ | OTHER ✓ |
| pb-v1-0062:t0 | 0 | Shop ơi cho mình hỏi chỗ mình đi đường nào vô cho dễ, mình đi kiếm mà kg ra | OTHER | OTHER ✓ | OTHER ✓ | COMPLAINT |
| pb-v1-0064:t0 | 0 | Ngoài cơm tắm còn có món gì | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0067:t0 | 0 | deal mì quảng 59k ko còn hả e | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0074:t0 | 0 | chắc chắn em quên order của chị nên đơn mới lâu vậy | COMPLAINT | COMPLAINT ✓ | COMPLAINT ✓ | COMPLAINT ✓ |
| pb-v1-0086:t0 | 0 | Em ui, mì râmen có nhiều size ha em? | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0093:t0 | 0 | Chị đã trả tiền trước theo gói combo khuyến mãi rồi | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0098:t0 | 0 | Rồi xong… vội quá chuyển nhầm ngân hàng vào người khác. Mất x 2 tiền cho 1 đơn hàng😭 | COMPLAINT | OTHER | OTHER | OTHER |
| pb-v1-0109:t0 | 0 | Mình muốn đặt bàn 2 người cho tối nay tầm 18h-19h ngày hôm nay | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0112:t0 | 0 | Joi ơi mình đặt bàn 12g45 dòn chỗ ko ạ? | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0122:t0 | 0 | Hi ad, cho mình đặt bàn 2 người trưa nay 11h15 nhé | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0135:t0 | 0 | E ơi, cho chị đặt bàn ở chỗ gần cửa ra vào chị hay ngồi đươc ko :) | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0138:t0 | 0 | Thông tin xuất hoá đơn e vẫn còn pk | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0142:t0 | 0 | tiệm oei cho mình đặt bàn 2 người lúc 30p nữa được hong ạ | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0143:t0 | 0 | e oi mai c dat ban 12 nguoi 6h chieu nhe e | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0149:t0 | 0 | bạn ơi, lát mình định đến quán ăn trưa, bạn cho mình đặt bàn 2 ng, lúc 1pm nha | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0153:t0 | 0 | Cho mình hỏi mình muốn đặt bàn qua đây được không ạ | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0165:t0 | 0 | Cho em đặt bàn 3 người vào 19h hôm nay được không ạ | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0168:t0 | 0 | Cho chị đặt bàn 3 người 11h15 nhé | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0169:t0 | 0 | Mình liên hệ đặt bàn cho 1 người | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0180:t0 | 0 | cho Nước Ép Cam | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0180:t1 | 1 | thêm Chả Giò | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0180:t2 | 2 | à nước đổi qua Nước Ép Thơm nha | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0180:t3 | 3 | chả giò vẫn giữ | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0181:t0 | 0 | cho mình 1 phở | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0181:t1 | 1 | Phở Trộn Trứng Cuộn nha | OTHER | MENU_QUERY | MENU_QUERY | OTHER ✓ |
| pb-v1-0181:t2 | 2 | thêm Chả Giò | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0181:t3 | 3 | đủ rồi | CONFIRMATION | CONFIRMATION ✓ | CONFIRMATION ✓ | CONFIRMATION ✓ |
| pb-v1-0185:t0 | 0 | 1 bánh mì nha em | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0185:t1 | 1 | loại Pa Tê - Chả | OTHER | MENU_QUERY | MENU_QUERY | MENU_QUERY |
| pb-v1-0185:t2 | 2 | thêm Nước Ép Cam | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0185:t3 | 3 | vậy thôi | CONFIRMATION | CONFIRMATION ✓ | CONFIRMATION ✓ | CONFIRMATION ✓ |
| pb-v1-0187:t0 | 0 | cho Phở Trộn Trứng Cuộn, Chả Giò với cà phê đen đá | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0187:t1 | 1 | bỏ Chả Giò đi em | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0187:t2 | 2 | 2 món kia giữ nguyên nha | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0201:t0 | 0 | Cơm Tấm Joi Original với Trà Lài Xiêm Xanh | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0201:t1 | 1 | cơm ko hành nha em | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0201:t2 | 2 | trà giữ nguyên đó nha | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0202:t0 | 0 | 2 Phở Trộn Trứng Cuộn với 1 Coca | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0202:t1 | 1 | 2 tô phở đều ít cay nha | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0202:t2 | 2 | Coca bình thường | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0202:t3 | 3 | chuẩn | CONFIRMATION | CONFIRMATION ✓ | CONFIRMATION ✓ | CONFIRMATION ✓ |
| pb-v1-0203:t0 | 0 | 1 Bún Riêu Mê Ly nha | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0203:t1 | 1 | giao số 101 đường Nguyễn Cơ Thạch | ADDRESS | OTHER | ADDRESS ✓ | ADDRESS ✓ |
| pb-v1-0203:t2 | 2 | phường An Khánh, TP.HCM nha | ADDRESS | ADDRESS ✓ | ADDRESS ✓ | ADDRESS ✓ |
| pb-v1-0203:t3 | 3 | giao ngay, COD, chốt giúp mình | CONFIRMATION | CONFIRMATION ✓ | CONFIRMATION ✓ | CONFIRMATION ✓ |
| pb-v1-0213:t0 | 0 | Cơm Lứt Joi Original với Trà Lài Xiêm Xanh | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0213:t1 | 1 | giao [ADDRESS_1] | ADDRESS | ADDRESS ✓ | ADDRESS ✓ | ADDRESS ✓ |
| pb-v1-0213:t2 | 2 | giao ngay | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0213:t3 | 3 | chốt luôn em | CONFIRMATION | CONFIRMATION ✓ | CONFIRMATION ✓ | CONFIRMATION ✓ |
| pb-v1-0219:t0 | 0 | 2 Cơm Tấm Nhẹ với Trà Lài Xiêm Xanh | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0219:t1 | 1 | giao [ADDRESS_1] | ADDRESS | ADDRESS ✓ | ADDRESS ✓ | ADDRESS ✓ |
| pb-v1-0219:t2 | 2 | 19h, COD, tổng giúp | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0219:t3 | 3 | cơm còn 1 phần thôi | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0219:t4 | 4 | tổng mới bao nhiêu | MENU_QUERY | OTHER | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0222:t0 | 0 | Cơm Tấm Original với cà phê đen đá | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0222:t1 | 1 | giao [ADDRESS_1] | ADDRESS | ADDRESS ✓ | ADDRESS ✓ | ADDRESS ✓ |
| pb-v1-0222:t2 | 2 | 19h, COD, lên summary giúp mình | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0222:t3 | 3 | 19h, COD, lên summary giúp mình nha em | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0223:t0 | 0 | cho 1 Gyoza Size Nhẹ với 1 Coca nha | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0223:t1 | 1 | cho 1 Gyoza Size Nhẹ với 1 Coca nha | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | OTHER |
| pb-v1-0223:t2 | 2 | giỏ đang có gì rồi em? | OTHER | OTHER ✓ | OTHER ✓ | OTHER ✓ |
| pb-v1-0239:t0 | 0 | chào em | GREETING | COMPLAINT | GREETING ✓ | GREETING ✓ |
| pb-v1-0239:t1 | 1 | hôm qua mình đặt phở mà giao qua bún riêu, sao kỳ vậy | COMPLAINT | COMPLAINT ✓ | COMPLAINT ✓ | COMPLAINT ✓ |
| pb-v1-0239:t2 | 2 | đơn đó mình đặt lúc 7h tối đó, giao thiếu cả khăn giấy nữa | COMPLAINT | OTHER | OTHER | OTHER |
| pb-v1-0239:t3 | 3 | ừ nhờ em báo lại, cảm ơn nha | GRATITUDE | CONFIRMATION | GRATITUDE ✓ | GRATITUDE ✓ |
| pb-v1-0246:t0 | 0 | 1 Bún Riêu Mê Ly nha | MENU_QUERY | MENU_QUERY ✓ | MENU_QUERY ✓ | MENU_QUERY ✓ |
| pb-v1-0246:t1 | 1 | giao [ADDRESS_1] | ADDRESS | ADDRESS ✓ | ADDRESS ✓ | ADDRESS ✓ |
| pb-v1-0246:t2 | 2 | giao ngay, COD, chốt nha | CONFIRMATION | OTHER | CONFIRMATION ✓ | CONFIRMATION ✓ |
| pb-v1-0246:t3 | 3 | thôi khỏi số được ko, mình ngại cho số lắm | PHONE | CONFIRMATION | CONFIRMATION | CONFIRMATION |
| pb-v1-0246:t4 | 4 | ừ thôi được rồi, [PHONE_1] nha | PHONE | CONFIRMATION | PHONE ✓ | PHONE ✓ |
| pb-v1-0246:t5 | 5 | ok chốt | CONFIRMATION | CONFIRMATION ✓ | CONFIRMATION ✓ | CONFIRMATION ✓ |

## Phụ lục C. Đường học của sáu trials

Train loss là CE trung bình theo mẫu trong epoch ở training mode; dev loss đo ở evaluation mode. Hai đại lượng không nên diễn giải như cùng điều kiện dropout. Bảng giữ các epoch thực tế đến khi dừng; LR cuối epoch được ghi để hiểu lịch decay.

### BamiBERT — LR 2e-05

| Epoch | Train loss | Dev loss | Dev Macro-F1 | LR cuối epoch |
| --- | --- | --- | --- | --- |
| 1 | 1,9435 | 1,6741 | 7,18% | 1.35294e-05 |
| 2 | 1,6104 | 1,5519 | 7,18% | 1.92283e-05 |
| 3 | 1,4969 | 1,4596 | 13,87% | 1.77492e-05 |
| 4 | 1,3775 | 1,3837 | 13,94% | 1.62701e-05 |
| 5 | 1,2459 | 1,3430 | 14,69% | 1.4791e-05 |
| 6 | 1,0912 | 1,2382 | 24,07% | 1.33119e-05 |
| 7 | 0,9448 | 1,2039 | 27,86% | 1.18328e-05 |
| 8 | 0,7996 | 1,0562 | 45,25% | 1.03537e-05 |
| 9 | 0,6831 | 0,9769 | 48,27% | 8.8746e-06 |
| 10 | 0,5820 | 1,0521 | 44,46% | 7.3955e-06 |
| 11 | 0,4958 | 0,9403 | 50,29% | 5.9164e-06 |
| 12 | 0,4359 | 0,9611 | 48,06% | 4.4373e-06 |
| 13 | 0,3912 | 0,8641 | 50,21% | 2.9582e-06 |
| 14 | 0,3657 | 0,8808 | 48,80% | 1.4791e-06 |

### BamiBERT — LR 5e-05

| Epoch | Train loss | Dev loss | Dev Macro-F1 | LR cuối epoch |
| --- | --- | --- | --- | --- |
| 1 | 1,8338 | 1,6141 | 7,18% | 3.38235e-05 |
| 2 | 1,5978 | 1,5923 | 7,18% | 4.80707e-05 |
| 3 | 1,5248 | 1,5220 | 11,19% | 4.4373e-05 |
| 4 | 1,4800 | 1,6998 | 10,21% | 4.06752e-05 |
| 5 | 1,5046 | 1,6577 | 10,09% | 3.69775e-05 |
| 6 | 1,4339 | 1,4720 | 12,66% | 3.32797e-05 |
| 7 | 1,3786 | 1,4617 | 13,46% | 2.9582e-05 |
| 8 | 1,2597 | 1,4060 | 14,31% | 2.58842e-05 |
| 9 | 1,2060 | 1,4120 | 19,94% | 2.21865e-05 |
| 10 | 1,1161 | 1,4067 | 17,47% | 1.84887e-05 |
| 11 | 0,9824 | 1,3587 | 19,52% | 1.4791e-05 |
| 12 | 0,9173 | 1,3549 | 22,71% | 1.10932e-05 |
| 13 | 0,8603 | 1,3368 | 22,54% | 7.3955e-06 |
| 14 | 0,8313 | 1,3247 | 24,25% | 3.69775e-06 |
| 15 | 0,8109 | 1,3336 | 27,51% | 0 |

### Goi DAPT — LR 2e-05

| Epoch | Train loss | Dev loss | Dev Macro-F1 | LR cuối epoch |
| --- | --- | --- | --- | --- |
| 1 | 1,9763 | 1,6961 | 7,18% | 1.35294e-05 |
| 2 | 1,6119 | 1,5379 | 8,33% | 1.92283e-05 |
| 3 | 1,4482 | 1,3822 | 16,07% | 1.77492e-05 |
| 4 | 1,2594 | 1,2904 | 15,05% | 1.62701e-05 |
| 5 | 1,0867 | 1,3468 | 25,02% | 1.4791e-05 |
| 6 | 0,9054 | 1,0970 | 36,81% | 1.33119e-05 |
| 7 | 0,7432 | 1,0359 | 42,16% | 1.18328e-05 |
| 8 | 0,5625 | 0,9401 | 46,24% | 1.03537e-05 |
| 9 | 0,4745 | 0,8786 | 50,09% | 8.8746e-06 |
| 10 | 0,4034 | 0,8625 | 48,60% | 7.3955e-06 |
| 11 | 0,3421 | 0,8232 | 49,67% | 5.9164e-06 |
| 12 | 0,3028 | 0,8411 | 49,68% | 4.4373e-06 |

### Goi DAPT — LR 5e-05

| Epoch | Train loss | Dev loss | Dev Macro-F1 | LR cuối epoch |
| --- | --- | --- | --- | --- |
| 1 | 1,8602 | 1,6205 | 7,18% | 3.38235e-05 |
| 2 | 1,5380 | 1,4713 | 14,81% | 4.80707e-05 |
| 3 | 1,3685 | 1,3044 | 16,81% | 4.4373e-05 |
| 4 | 1,2005 | 1,2482 | 29,89% | 4.06752e-05 |
| 5 | 0,9579 | 1,2252 | 34,62% | 3.69775e-05 |
| 6 | 0,7059 | 1,0107 | 42,93% | 3.32797e-05 |
| 7 | 0,4831 | 1,0194 | 51,78% | 2.9582e-05 |
| 8 | 0,3025 | 0,9685 | 53,18% | 2.58842e-05 |
| 9 | 0,2067 | 1,1029 | 67,60% | 2.21865e-05 |
| 10 | 0,1440 | 0,9934 | 71,31% | 1.84887e-05 |
| 11 | 0,1040 | 0,8709 | 72,23% | 1.4791e-05 |
| 12 | 0,0811 | 1,0304 | 69,39% | 1.10932e-05 |
| 13 | 0,0414 | 1,0407 | 69,98% | 7.3955e-06 |
| 14 | 0,0362 | 1,0572 | 69,50% | 3.69775e-06 |

### Goi student — LR 2e-05

| Epoch | Train loss | Dev loss | Dev Macro-F1 | LR cuối epoch |
| --- | --- | --- | --- | --- |
| 1 | 1,9777 | 1,7339 | 7,18% | 1.35294e-05 |
| 2 | 1,6192 | 1,5502 | 7,18% | 1.92283e-05 |
| 3 | 1,4361 | 1,3620 | 15,50% | 1.77492e-05 |
| 4 | 1,2354 | 1,3400 | 22,59% | 1.62701e-05 |
| 5 | 1,0597 | 1,1980 | 21,57% | 1.4791e-05 |
| 6 | 0,9040 | 1,1121 | 26,98% | 1.33119e-05 |
| 7 | 0,7453 | 0,9812 | 35,28% | 1.18328e-05 |
| 8 | 0,6333 | 0,8868 | 55,65% | 1.03537e-05 |
| 9 | 0,4933 | 0,7811 | 61,29% | 8.8746e-06 |
| 10 | 0,4335 | 0,7910 | 55,45% | 7.3955e-06 |
| 11 | 0,3574 | 0,7358 | 62,64% | 5.9164e-06 |
| 12 | 0,3109 | 0,7437 | 61,72% | 4.4373e-06 |
| 13 | 0,2766 | 0,7132 | 60,82% | 2.9582e-06 |
| 14 | 0,2554 | 0,7175 | 61,77% | 1.4791e-06 |

### Goi student — LR 5e-05

| Epoch | Train loss | Dev loss | Dev Macro-F1 | LR cuối epoch |
| --- | --- | --- | --- | --- |
| 1 | 1,8569 | 1,6394 | 7,18% | 3.38235e-05 |
| 2 | 1,5471 | 1,4643 | 13,89% | 4.80707e-05 |
| 3 | 1,2624 | 1,2826 | 15,80% | 4.4373e-05 |
| 4 | 1,0228 | 1,2346 | 28,23% | 4.06752e-05 |
| 5 | 0,8203 | 1,0793 | 42,92% | 3.69775e-05 |
| 6 | 0,5288 | 0,9550 | 56,87% | 3.32797e-05 |
| 7 | 0,3359 | 0,9621 | 57,16% | 2.9582e-05 |
| 8 | 0,2513 | 0,8937 | 58,02% | 2.58842e-05 |
| 9 | 0,1542 | 0,6533 | 70,91% | 2.21865e-05 |
| 10 | 0,0871 | 0,7840 | 70,23% | 1.84887e-05 |
| 11 | 0,0559 | 0,7355 | 71,93% | 1.4791e-05 |
| 12 | 0,0366 | 0,6687 | 70,08% | 1.10932e-05 |
| 13 | 0,0302 | 0,7524 | 72,41% | 7.3955e-06 |
| 14 | 0,0242 | 0,7511 | 72,41% | 3.69775e-06 |
| 15 | 0,0219 | 0,7504 | 72,41% | 0 |
