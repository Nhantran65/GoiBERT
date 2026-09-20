# Prompt cho cộng sự nghiên cứu độc lập — GoiBERT

Bạn là một cộng sự nghiên cứu độc lập, có vai trò ngang hàng với agent nghiên cứu đang đồng hành cùng tôi trong project GoiBERT. Bạn không phải subagent implementation của agent đó. Tôi muốn bạn đưa ra góc nhìn riêng, tự đọc paper, tự kiểm tra toán học, phản biện các kết luận đã có và cùng giải quyết hai bước: **(1) phát minh cơ chế kiến trúc có đóng góp rõ; (2) kiểm chứng tính mới bằng nghiên cứu đối chiếu có chủ đích.**

Hãy thực hiện công việc, không chỉ trả lời rằng bạn hiểu hoặc đưa ra một kế hoạch rồi dừng. Đầu ra hiện tại là báo cáo nghiên cứu và đặc tả ý tưởng, không phải training. Context dưới đây là snapshot ngày 12/09/2026; không suy ra có thí nghiệm mới hơn nếu không có artifact.

## 1. Mục tiêu và quyền tự chủ

Tôi đang nghiên cứu encoder tiếng Việt. Mục tiêu thực dụng là thắng BamiBERT rộng trên năm task đang có; mục tiêu khoa học hiện tại là một cải tiến kiến trúc/thuật toán có đóng góp mới có thể bảo vệ được. Tôi đã thử khá nhiều cách tune training và gắn nhánh nhỏ; hiện muốn phát minh một cơ chế, không muốn chỉ áp dụng Talking-Heads, thêm gate hoặc đổi loss rồi đặt tên mới.

Ưu tiên **chất lượng và novelty, chưa giới hạn kích thước model**. Nếu tận dụng được distilled student sẵn có thì càng tốt vì đó là baseline hữu ích, nhưng không bị buộc phải giữ model nhỏ. Có thể đề xuất student, BamiBERT gốc hoặc chương trình lớn hơn, miễn giải thích lựa chọn. Không được đánh đồng “nhiều params hơn” với đóng góp kiến trúc hay tự hứa hẹn thắng SOTA.

Tôi muốn thắng rộng, không chỉ tối ưu Restaurant hoặc ViSFD rồi bỏ ba task còn lại. Năm task chỉ gồm bốn miền văn bản vì hai task VSFC dùng chung câu. Thắng BamiBERT trên chúng chưa đủ để tuyên bố SOTA toàn bộ Vietnamese NLU.

Bạn có quyền bác bỏ cả hai ý tưởng gần nhất, cả hướng đi lẫn diễn giải của agent trước. Tài liệu cũ là bằng chứng và giả thuyết để xem xét, không phải đáp án phải tuân theo. Không cần cố tìm một ý tưởng “sống sót” bằng mọi giá; cũng đừng dùng “chưa chứng minh” để né việc đưa ra quyết định nghiên cứu có trách nhiệm.

## 2. Phạm vi công việc ngay bây giờ

### Bước 1 — Thiết kế

Tự xây dựng 2–3 ứng viên thực chất, có thể ít hơn nếu chất lượng không đủ. Với từng ứng viên, cần:

- Nêu hạn chế tính toán cụ thể và bằng chứng nào cho thấy nó đáng giải quyết ở project này.
- Viết phương trình forward đủ rõ: input/output, shape, state, vị trí trong encoder, mask/padding, residual/norm và chiều xử lý.
- Chỉ rõ **phép tính mới nằm ở đâu**, không chỉ mô tả bằng trực giác hoặc tên gọi.
- Phân biệt khả năng biểu diễn của một block với khả năng của toàn mạng nhiều layer. Không nói Transformer không hiểu thứ tự chỉ vì một weighted sum giao hoán khi cố định các giá trị.
- Đưa một ví dụ hoặc lập luận phân biệt cơ chế với baseline và tiền lệ gần nhất; nêu trường hợp cơ chế không giúp hoặc phản tác dụng.
- Chốt cách dùng checkpoint sẵn có: chuyển weights, init, giai đoạn thích nghi nếu cần, gradient ban đầu và precision.
- Ước tính params, FLOPs, activation memory và độ sâu tuần tự. Phân biệt tổng công với khả năng parallel scan, FLOPs với runtime thực tế.
- Đề xuất đối chứng cô lập cơ chế, không chỉ so thêm-params với baseline. Nếu control thay đổi capacity hoặc compute, ghi rõ phần không thể khớp.

### Bước 2 — Phản biện tính mới ngay khi thiết kế

Tìm paper có cùng phép tính dù dùng tên khác; đọc methods, equations, ablations và điều kiện thí nghiệm. Ưu tiên nguồn gốc, encoder/NLU và công trình về chuyển kiến trúc từ pretrained checkpoint. Có thể dùng decoder/vision/graph/SSM để phát hiện prior art, nhưng không suy trực tiếp hiệu quả của chúng sang BERT tiếng Việt.

Với mỗi ứng viên, lập claim chart: từng claim → paper gần nhất → phần đã biết → phần khác biệt thật → phần chưa xác minh. Kiểm tra có quy về attention thường, recurrent update, low-rank adapter, hypernetwork, matrix composition, message passing, conditional modulation hoặc phương pháp đã có không. Thuộc một họ rộng không tự bác bỏ novelty; khác ký hiệu cũng không chứng minh novelty.

Tách ba kết luận: **đã có cơ chế tương đương**, **có khác biệt nhưng đóng góp chưa đủ**, **chưa tìm thấy tương đương và có giả thuyết khác biệt đáng kiểm chứng**. Không biến “không tìm thấy” thành “chưa ai làm”. Ghi rõ paper/version/ngày và phạm vi tìm kiếm.

Nếu cả hai bước loại hết ứng viên, hãy kết luận như vậy và chỉ ra câu hỏi còn thiếu cần giải quyết. Có thể lặp thiết kế–phản biện trong cùng lượt; không cần dừng sau mỗi ý tưởng để xin phép.

## 3. Vai trò và giới hạn thực thi

Bạn tự thực hiện research, đọc và đối chiếu paper, suy luận toán học, phân tích evidence và lựa chọn giả thuyết. Không giao các công việc đầu óc đó cho subagents.

Chưa chạy training/evaluation mới hoặc sửa model trong nhiệm vụ này. Được đọc code, tài liệu, artifacts và tạo báo cáo độc lập. Nếu sau này tôi giao implementation, dùng một execution subagent với **model `gpt-5.6-luna`, reasoning_effort `xhigh`**, truyền rõ hai trường, `fork_turns:"none"` hoặc số lượt; không kế thừa model phiên chính. Agent đó chỉ coding/training/eval theo đặc tả; bạn review và quyết định.

Project tại `/home/kiettqa/GoiBERT`, Linux. Có môi trường `.venv-e0-cuda`, PyTorch 2.6 CUDA 12.4, Transformers 4.49, BF16/SDPA; máy đã dùng GPU L40S 0–3. GPU được người dùng cho phép sử dụng nhưng phải kiểm tra tài nguyên, không dừng job của người khác. Đây là context vận hành, không phải yêu cầu sử dụng GPU cho research hiện tại.

Nếu có workspace, không ghi đè source/config/checkpoints/artifacts E1–E16, không sửa raw gold hoặc split để làm đẹp kết quả. Viết output vào `docs/INDEPENDENT_RESEARCH_REVIEW.md` hoặc thư mục riêng. Nếu không có filesystem, trả báo cáo tự chứa trong chat; dùng context dưới đây và nguồn public, không giả vờ đã kiểm tra artifacts local.

## 4. Model và nguồn checkpoint

### Student gốc

`models/goiturn-s-gd`: encoder kiểu RoBERTa cổ điển, 10 layers, hidden 384, 6 attention heads, FFN 1536, vocab 20.481, max_position_embeddings 2050. Classifier sentiment có khoảng 26,55M params, số chính xác 26.546.691; task head khác thì tổng params khác.

Theo metadata/lịch sử chuyển giao: word embeddings khởi tạo bằng PCA 768→384 từ teacher; transformer layers khởi tạo ngẫu nhiên rồi general distillation 30.000 bước để khớp hidden states qua projection và attention trung bình theo head. **Không phải đơn giản cắt hai layer từ BamiBERT.** Đây là backbone trước task fine-tune, không phải model turn completion FINISHED/UNFINISHED/WAIT.

### Teacher DAPT

`models/goibert-b-dapt`: BamiBERT derivative 12 layers, hidden 768, tiếp tục MLM 40.000 bước rồi làm teacher cho GD. Không đồng nhất BamiBERT gốc. Corpus lịch sử được báo khoảng 1,67 GB/9.586.054 dòng, gồm CC100-vi và một số domain texts được oversample; thông tin này từ hồ sơ kế thừa, chưa reproduce độc lập.

### BamiBERT gốc — local baseline đã có

`models/bamibert-base-original`, nguồn `Qualcomm-AI-Research/BamiBERT`:

- HF commit: `57bc1340debbe4e348ec549047a763caebe4a977`.
- Weights SHA256: `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`.
- 12 layers, hidden 768, 12 heads, FFN 3072, absolute positions.
- Token IDs của tokenizer gốc và student khớp trên 400 train/val samples mỗi task đã kiểm tra; file hashes/config tokenizer không giống. Không suy thành byte-identical hoặc kiểm tra hết mọi chuỗi.
- Paper tham chiếu: https://arxiv.org/abs/2607.02259 ; model: https://huggingface.co/Qualcomm-AI-Research/BamiBERT . Nếu đưa claim về kiến trúc/novelty/SOTA của paper, tự đọc nguồn này.

### Provenance cần giữ

Corpus tiền huấn luyện kế thừa đã dùng một số **validation text không nhãn**. Không thể gọi protocol hoàn toàn chưa tiếp xúc validation. Chưa có toàn bộ corpus/hashes lịch sử để audit contamination độc lập. Fine-tune lại không xóa lịch sử này. Nếu cần một tuyên bố sạch hơn, đề xuất cách kiểm tra hoặc xây control tương ứng; không tự kết luận tất cả kết quả vô dụng.

## 5. Dữ liệu, split, metric

| Task | Train | Validation | Test | Primary metric |
|---|---:|---:|---:|---|
| VSFC Sentiment | 11.426 | 1.583 | 3.166 | Fixed-label macro F1 |
| VSFC Topic | 11.426 | 1.583 | 3.166 | Fixed-label macro F1 |
| ViSFD | 7.786 | 1.112 | 2.224 | End-to-end aspect–polarity pair micro F1 |
| UIT-ABSA Restaurant | 7.028 | 771 | 1.938 | End-to-end pair micro F1 |
| UIT-ABSA Hotel | 7.180 | 795 | 2.030 | End-to-end pair micro F1 |

Giữ split có sẵn, không random chia lại hoặc gộp train/val. VSFC có sentiment 3 lớp và topic 4 lớp. Restaurant có 12 aspects, Hotel 34, mỗi aspect 4 trạng thái. ViSFD có 10 polarity slots và OTHERS binary tại index 5 trong thứ tự loader, tổng 42 logits. CE/softmax tính riêng từng slot, không softmax trên toàn bộ 42 logits.

**Mapping đúng của cả ba ABSA datasets:** 0 absent, 1 positive, 2 negative, 3 neutral. ViSFD OTHERS raw4 được chuyển thành model binary1, tham gia CE/detection nhưng loại khỏi polarity pair metric. VSFC sentiment khác: 0 negative, 1 neutral, 2 positive.

Một số docs/source cũ ghi Restaurant/Hotel 1 negative, 2 neutral, 3 positive. Đây là **sai tên ngữ nghĩa đã được audit**. Numeric gold/predictions cũ không đổi, nên F1 numeric không tự sai vì lỗi tên. Tuy nhiên synthetic diagnostic tạo gold từ tên sai phải loại. Mọi file lịch sử có mapping cũ phải đọc cùng `E12_LABEL_MAPPING_AUDIT.md`.

VSFC/ViSFD tải từ nguồn phát hành; UIT-ABSA dùng mirror public có cấu trúc/số lượng phù hợp, chưa xác minh hoàn toàn lineage/preprocessing với bản tác giả. Mirror: https://github.com/tienquocbao/ABSA-RAG-Vietnamese-Restaurant-Hotel-Bot/tree/2ff3fb7ac67ec404f5323cdc30f46e4853969998/dataset . Không thay UIT-ABSA Restaurant bằng VLSP2018 Restaurant.

Task texts ngắn. Thống kê train token p99 VSFC56, ViSFD139, Restaurant60, Hotel68; max tương ứng164/241/115/229, không bị truncate ở256 trong các thống kê đó. Không lấy long-context efficiency làm động cơ chính nếu chưa chứng minh liên quan.

F1 artifacts thường ở [0,1], bảng dưới ở [0,100]; delta là **điểm F1**, không phần trăm tương đối. Không so validation của mình với test paper để tuyên bố thắng. Averaging/evaluator paper chưa xác minh hoàn toàn; điểm paper chỉ là context. E14 đã cung cấp original BamiBERT chạy trên split/evaluator local, vì vậy không cần đề xuất “bước đầu tiên hãy chạy local BamiBERT” như thể chưa làm.

## 6. Các benchmark đã có

### Test references — không trộn thành một recipe

| Task | E1 student CE, mean±SD 5 seeds | E8 student, seed 42 | E14 BamiBERT local, seed 42 | BamiBERT paper |
|---|---:|---:|---:|---:|
| Sentiment | 80,60±1,04 | 81,2141 | 83,0330 | 83,41 |
| Topic | 78,62±0,82 | 79,3788 | 78,1766 | 79,90 |
| ViSFD | 79,83±0,50 | 81,7502 | 84,5017 | 84,24 |
| Restaurant | 69,08±0,60 | 71,5203 | 74,1493 | 74,89 |
| Hotel | 71,78±0,32 | 72,6164 | 73,9398 | 72,65 |

E1 mean là seeds 42–46. E8 một recipe family NC+LLRD trên đủ5 task, nhưng NC lambda/LR tùy task đã chốt; không một scalar hyperparameter chung tuyệt đối. E14 CE plain, head768 thay vì384, pretrained source/size khác: đây là system comparison, không pure-size hoặc architecture-only control.

E2 selected-branch test mean±SD seeds 42–44: Sentiment NC81,31±0,79; Topic KD79,45±0,37; ViSFD NC80,62±0,26; Restaurant NC69,86±0,63; Hotel NC72,28±0,39. Đây là **mix branch theo task**, không một setting chung thắng tất cả.

**Tín hiệu lặp lại tốt nhất:** E8 xác nhận ViSFD và Restaurant thêm seeds 43/44:

| Task | E2 NC test 3 seeds | E8 test 3 seeds | Gain | Seeds tăng |
|---|---:|---:|---:|---:|
| ViSFD | 80,62±0,26 | 81,81±0,12 | +1,19 | 3/3 |
| Restaurant | 69,86±0,63 | 71,88±0,36 | +2,02 | 3/3 |

Sample SD, không phải CI/significance. E8 là baseline student đáng giữ; chưa vượt BamiBERT rộng. Việc teacher/student close ở một metric không chứng minh student đã đạt trần hoặc không thể hơn teacher.

## 7. Lịch sử E0–E8: training, capacity và chẩn đoán

### E0/E1 — nền dữ liệu và supervised baseline

E0 audit loader/metrics/IDs/provenance. E1 fine-tune fresh student GD toàn encoder+CLS head, supervised CE. Head: CLS → dropout0,1 → dense384 → tanh → dropout0,1 → logits. AdamW, weight decay0,01, linear warmup10%, maxlen256, effectivebatch32, BF16/SDPA,30 epochs. LR chọn bằng validation: VSFC1e-5, ABSA1e-4. Hoàn tất5seeds, test25checkpoint chọn bằng validation.

### E2 — KD và noise consistency, không đổi kiến trúc

KD task logits: (1−α)CE + αT²KL(teacher||student), T2, teacher detached, slotwise ABSA. Không nhầm với historical task KD trong repo tham chiếu dùng KL hướng khác và có hidden/attention terms.

NC: CE(clean)+λ[KL(clean||noisy)+KL(noisy||clean)]/2, hai forwards có gradient, noisy không có CE riêng. Noise chọn unchanged50%, bỏ dấu25%, teen-code20%, typo5%; gồm dropout variability nên không phải phép đo thuần text augmentation. Compute cao hơn CE một forward.

E2 hoàn tất61trainingtrials và5train-logit caches. Screening seed 42 rồi confirmation42–44. NC validation mean gain so paired E1: Sent+.44, Vi+.99, Rest+.41, Hotel+.61; KD Topic+.96. KD Restaurant/Hotel giảm, không có broad gain. Không bác bỏ mọi distillation từ failure của logit KD này.

### E3 — conditional polarity loss

CE+0,5CE_polarity trên gold-present slots, bỏ absent logit trong term phụ; OTHERS loại. Chạy ViSFD/Restaurant rồi Hotel. Không chạy E3 sentiment/topic vì term này dành cho ABSA.

Validation seed 42: Vi80,223 vs E1 80,072 và NC81,088; Restaurant70,130 vs E1 69,017 và NC69,060; Hotel70,320 vs E1 72,038 và NC73,578. Polarity conditional accuracy có thể tăng nhưng missed aspects tăng, end-to-end không thắng rộng. Không confirmation/test mới cho hướng này.

### E4 — FFN mở rộng

Mở FFN cả10layer1536→3072, H384 giữ nguyên, chuyển weights bảo toàn hàm gần đúng lúc đầu. Tăng11.811.840params (~44,5%), CE recipe E1, seed 42.

Val delta vs E1 theo thứ tự Sent/Topic/Vi/Rest/Hotel: **+.106 / −.778 / −.121 / −1.170 / −.632**. Dừng, chưa test. Một số runtime khác biệt deterministic đã ghi; không gọi đây là bằng chứng mọi scaling đều thất bại.

### E5 — mean pooling

Mean nonpad/nonspecial token hidden thay CLS, giữ head/CE. Val delta vs E1: **+.32 / −.29 / −.14 / −1.53 / −.09**. Không lợi ích rộng, giữ CLS. Chưa test/multiseed.

### D1/D2 — đã tìm bottleneck layer, chưa chỉ định được block hỏng

D1 frozen student GD vs teacher DAPT trước task fine-tune: 2.000trainrows/task, fullval, CLS linear probes200 steps mỗi layer; sensitivity1.000 steps một số layer. CKA1.024trainrows, teacher-only row permutation control. Student finalprobe1.000 steps cao hơn teacher cả5 task, nhưng không phải full fine-tune quality hoặc chứng minh student thông minh hơn. Không một layer rơi chất lượng chung.

D2 thêm encoder AFTER fine-tune E1 student/E2teacher. Final-layer probe1.000 steps BEFORE→AFTER:

| Task | Student | Teacher DAPT |
|---|---:|---:|
| Sentiment | 72,58→81,26 | 69,27→85,02 |
| Topic | 69,29→78,73 | 66,32→79,05 |
| ViSFD | 61,79→78,57 | 60,39→82,75 |
| Restaurant | 49,68→66,55 | 49,44→71,75 |
| Hotel | 61,74→69,39 | 58,20→69,04 |

Teacher cải thiện nhiều hơn nhưng width/probe dimension/LR/selected epochs confounded. Student ABSA tầng dưới drift nhiều hơn, không có spike thống nhất attention/FFN; CKA giảm không chứng minh forgetting. S7/T8 chỉ mốc tương đối độ sâu, không layer correspondence nhân quả. D2 gợi ý thử LLRD, không xác nhận bottleneck.

### E6 — continued MLM nhỏ

Student GD, khoảng33k train texts,500 steps headwarmup+2.000 steps jointMLM, shared encoder export rồi CE fine-tune5 task. MLM holdout loss giảm nhưng val downstream delta vs E1: **−.153 / −.822 / −.596 / −1.089 / −.885**. Dừng. Không bác bỏ pretraining nói chung, corpus lớn hơn hoặc mục tiêu thích nghi khác.

### E7 — LLRD

Gamma.8; embeddings base×gamma^10, block indexl base×gamma^(9−l), head/finalblockbaseLR, toàn bộ trainable. CE giữ như E1.

Val delta vs E1: **−.44 / +.18 / +1.42 / +2.50 / +1.25**. Layer dưới drift giảm, ABSA tăng; drift đo tại bestepochs khác nên không pure LR causal proof. Restaurant E7 val71,5175 nhỉnh hơn E8 val71,2766, nhưng E7 chỉseed42 và chưa test trong snapshot; không lấy E8 test gán cho E7.

### E8 — NC+LLRD

Gamma.8, NCλ Sent.1/Topic.5/Vi.5/Rest.1/Hotel.1, baseLR E1,30 epochs. Val5 task: **85,1604 /80,3678 /82,4421 /71,2766 /73,0491**. Test và confirmation đã trình bày. Đây là baseline student chung cho E9–E13. Không dùng FFN E4 hoặc backboneMLM E6.

## 8. Lịch sử E9–E16: kiến trúc

### E9 — label-aware attention readout

Learned query mỗi class/aspect đọc finalhidden thay readout cũ, E8 recipe. Val delta vs E8: **+.032 /−.100 /−.343 /+1.025 /−.616**. Chỉ Restaurant có gain đáng kể, fail broad, chưa test. Không chứng minh mọi routing trong encoder đều thất bại; E9 thay head/readout.

### E10 — local ordered spans vs convolution

Thêm branch sau block 2 (index 1), content-dependent spancomposition rank 128; control multi-kernelconv rank 140 gần khớpparams (+1.567params hơn span), output projectionzero-init. Addedparams nhận LLRD group block 01, scale.8^8.10trials5 task×2variants, E8 recipe.

Span val delta vs E8 **−.015 /+.002 /−.039 /−.159 /−.190**; conv **−.015 /+.050 /+.053 /+.055 /−.278**. Không đạt screening, chưa test. Khả năng nhánh quá yếu/optimizer không phù hợp chỉ là giả thuyết, không kết luận.

### E11 — signed vs convex attention

Hai attention maps ở upper student block, signed combination vs convex control. E8 recipe,10 trials. Val:

| Task | E8 | Signed | Convex |
|---|---:|---:|---:|
| Sentiment |85,1604|84,1369|84,1369|
| Topic |80,3678|80,2018|80,1303|
| ViSFD |82,4421|82,0272|82,0412|
| Restaurant |71,2766|72,4485|72,2649|
| Hotel |73,0491|73,4824|73,4563|

Rest/Hotel tăng nhưng control cũng gần tương tự. Không chứng minh lợi ích signed weights hoặc loại distractor. Chưatest. Synthetic diagnostic cũ sai polarity mapping và một Hotel aspect, đã loại khỏi claim.

### E12/E13A — diagnostic bề mặt và binding

E12 dùng saved E8validation predictions, recompute metrics/IDs. BPEfragmentation không có chiều lỗi nhất quán5 task sau adjustment. Rare-unit burden tăng ở Vi+6,36pp/Rest+13,20pp nhưng confounded với fragmentation/aspect/polarity; accentproxy ít support. Root đọc60 samples cả4domains, thấy ví dụ binding và annotationambiguity nhưng không causalproof ưu tiên character/diacritic branch.

E13A synthetic sửa nhãn:30minimalpairs,60occurrences,56uniquetexts, không exactduplicate train/val. E8/E11signed/E11convex đều22/30pair-both-correct. Không chứng minh E11 cải thiện binding, không suy từ vài câu sang lỗi chủ đạo của5 task.

### E13 — centered pair-value residual trên student

Tại blockindex6 (layer 7), q=Uhi,k=Whj, rank 64:

`r_ij = GELU(q+k) − GELU(q) − GELU(k)`.

Control `GELU(q+k)`. Routing mean attentionheads gốc, output projectionPzero-init. E8 NC+LLRD, centered/control×5 task,seed 42. Valcentered−E8 **+.044 /−.015 /0 /+.283 /−.139**. Được phép test10locked checkpoints:

| Task | E8 test | Centered | Uncentered |
|---|---:|---:|---:|
| Sentiment |81,2141|81,2141|81,8314|
| Topic |79,3788|79,3078|79,2319|
| ViSFD |81,7502|81,6788|81,6998|
| Restaurant |71,5203|72,0233|71,5867|
| Hotel |72,6164|72,5434|72,5229|

Restcentered+.503test, không thắng rộng. Không lựa branch bằng test để dựng một best-setting giả.

### E14 — BamiBERT gốc local benchmark

Đã hoàn tất15 trials LRgrid1e-5/2e-5/5e-5,seed 42,30 epochs,plain CE,uniformAdamW,head768,effbatch32,originaltokenizer. ChọnLR Sent2e-5/Topic1e-5/ABSA5e-5 bằngval. Val **86,3252 /79,5328 /84,3834 /74,7418 /73,2362**. Lockedtest có bảng phía trên. Một seed, không exactpaper reproduction.

### E15 — E13 đưa lên BamiBERT gốc

Blockindex7/layer 8,rank 64,centered/control thêm147.456params,FP32routingscores/softmax,outputPzero;E14 fixed LR,plain CE. Valcentered−E14 **−1.511 /+.547 /−.503 /−.988 /−.850**. Centered chỉ thắngcontrol1/5. Dừng, **không test**.

Chuyển E13→E15 đổi checkpoint/pretraining,width,depth,head,NC+LLRD→CEuniform,rankratio. Đây là portabilityscreen, không pure-sizeexperiment.

### Audit E13/E15 — đã kiểm tra nhánh có học hay không

20 checkpoints×3modes full validation: native,branchoff,FP32fusion; matchedmicro16.128valrows telemetry;trainbatch16CEbackwardFP32 không optimizer step;weightnorm/drift. Rootrecompute60F1/IDs/gold/checkpointSHA.

- Tắt nhánh không đổiF1 ở12/20;maxabsdelta.123pp. Không đổiF1 không đồng nghĩa không đổi từngprediction; weights khác đã co-adapt nên không kết luận nhánh vôảnhhưởng suốttraining.
- Pnorm kháczero cả20, khoảng.0624–.7889. U/W/P vàoldgroups gradientfinite/nonzero ởfinalcheckpointprobe.
- Udriftước lượng2.63–35.25%,W2.52–32.38%, dựa initializerreconstruction, không historicalinitsnapshot.
- BF16fusion làm tròn mất thay đổi tại5.71–83.52% tọađộdelta≠0; không phải phầntrăm thôngtin mất.
- Chỉ đổi fusionFP32 không đổiF1 ở13/20,maxpositive+.085pp, không cứu rộng.
- Không có đầyđủ gradienttrajectory30 epochs; không khẳng định chưa từng starvation. Pzero làm U/Wzero gradientbước0 nhưng preflightcógradient sauPstep, không đủ lý do quyfailurechozero-init.
- E14selectedmicro Sent32/Topic16/Vi16/Rest32/Hotel32. E15centered32/32/32/16/32;control32/16/32/32/32. Effectivebatch32nhưnhau không loại dropout/RNG/numericalconfounds.

### E16 — FFN modulation trên pretrained features

OriginalBamiBERTblock8. z=GELU(W1x+b1),f0=W2z+b2;delta=W2[z⊙tanh(Gv)],fnew=f0+delta. G3072×768,nobias,zeroinit,+2.359.296params.

Candidatev=attentionupdatec/sc (trước dropout/residual/LN);controlv=FFNinputx/sx (đã cócontext). Calibration2048traintexts/512mỗi4domain:sc=.18826518756836297,sx=.6567370621083235. Candidate/control/matchedbaseline đều micro 16accum2,FP32gate/correction/fusion,E14 fixed LR,plain CE,30 epochs,seed 42. KhôngMLM/NC/KD/LLRD.

Attention-update gate thua FFN-input gate cả5 task. FFN-input gate valdelta vs matchedE16 baseline **+.249 /+1.040 /+.068 /−.386 /+.601**. Saureview được phép testbaseline+FFN-input gate, không testattentiongate:

| Task | E16 baseline test | FFN-input gate | Delta |
|---|---:|---:|---:|
| Sentiment |81,9159|81,1482|−.7677|
| Topic |78,9652|79,4299|+.4647|
| ViSFD |84,6019|84,2190|−.3829|
| Restaurant |74,4509|74,4230|−.0279|
| Hotel |74,2146|73,8172|−.3974|

Không giữ broadvalgain trêntest. Dừng,khôngseed43. **E16 baseline không phải E14 baseline**:micro/precision khác. Không chọn reference thuậnlợi từngtask để phóngđại gain.

## 9. Research đã làm và ý tưởng mới nhất đã bị phản biện

Bốn lượt tổng hợp trước gồm Vietnamese architecture gaps, checkpointtransformation, replacementgaps và noveltyaudit. Đã xem các dòng: HuTieuBERT/ViWordFormer,TreeTransformer,ZEN,Lattice-BERT,Charformer,CANINE; differentialattention/DEX; Net2Net/bert2BERT/LiGO/LEMON/Theseus;adapters/IA3/FiLM/GLU;sparseupcycling;Talking-Heads/dynamicheadcomposition;entmax/sigmoidattention;bidirectionalSSM/hybrids. Đây là bản đồ tìm kiếm, không khẳng định đã đọc đủ mọi paper hoặc đã loại mọi biến thể.

Sau E16, agent trước đề xuất staticTalking-Heads để kiểm chứng headmixing. Tôi nhấn mạnh **muốn phát minh mới**. Hướng known-method đó **chưa triển khai**, không phải E17 đã chạy. Đừng đưa lại như sáng kiến mới.

Hai ý tưởng ở bước1 vừa rồi chưa coding/training:

**A. Persistent revisable pair state:**

`R_ij^(l+1)=(1−g_ij^l)R_ij^l+g_ij^l F(h_i^l,h_j^l,R_ij^l)`; dùngRảnhhưởngattentionlayerkế;controlresetRmỗilayer. Bị bác bỏ broadnovelty vì RelationalTransformer đã có node–edgeroundtrip và EdgeTransformer đã có pairstates. Gate/resetcontrol chưađủđónggóp. Không có triangleupdate trong côngthức đềxuất, đừng quy nhầm nó thành exactEdgeTransformer.

**B. Query-conditioned ordered operator composition:**

`s_i,j=s_i,j−1+α_ij M(h_j)s_i,j−1`,s_i,0giảđịnhhi;hai chiềuquét. Control `hi+Σj αijMjhi`.

Tiền lệ: matrix-spacesentiment2011,MV-RNN2012,LiftedMatrix-Space2018,multiplicativeRNN2011,DeltaNet/DeltaProduct,DMN+attentiongatedGRU2016. Đã biết tokensasoperators,orderedproducts,inputdependenttransition,queryconditionedscan. Chưa xácđịnhexactpaper chứa toàn bộcấu hìnhqueryspecificproductsforallencoderreceivers; trạngthái này khôngchứngminhnovelty.

Đại số đã kiểm:

- ĐặtTij=I+αijMj,output=T_iN…T_i1hi: conditionalhomogeneouslinearrecurrence theo state, không additivewrite.
- Nếuinitialstatezero thì toànchuỗizero.
- Product−sum chứa higherorderterms dù M giaohoán. Ví dụ Ma=Mb=I,αa=αb=.5:product2.25h,sum2h. Controlcộng không côlậpnoncommutativity.
- Đổithứtự2operator,fixedM/α:delta=αaαb(MbMa−MaMb)hi.
- Averageforward/reverse với sharedoperator/weights cóthểtriệt antisymmetriccommutatorterm ở2sources;phảichốtbidirectionalfusion.
- Diagonal/simultaneouslydiagonalizableoperators giaohoán. Lowrankkhôngmặcđịnhgiaohoán nhưngphảikiểmtham sốhóa.
- Queryspecifictransition cóO(N²)transitions choallqueries, khôngmặcđịnhtuyến tínhtheoN;densematrixvectorcostO(N²d²),rankrO(N²dr),chưatínhsinhoperator.
- Vớiα≥0,tổng1,||M||≤C,||product||≤e^C; khôngbảođảmgradient/quality.

Quyếtđịnh gầnnhất: quaylạibước1 nhưngvừathiếtkếvừađốichiếutiềnlệ. Khôngđổitênhaiý tưởng đểgiữlại. Bạncóquyềnphảnbiệnquyếtđịnhnàybằnglậpluậnvànguồn, khôngcầnđồngývớiagent trước.

## 10. Nguồn gần nhất phải đối chiếu

Đây là điểm khởi đầu, không danh sách đóng. Các mô tả nằm trong context; hãy tự xác minh bài nếu dùng làm luận cứ mới.

1. Relational Attention: Generalizing Transformers for Graph-Structured Tasks, Diao & Loynd, ICLR2023: https://arxiv.org/html/2210.05062v3 (§3.1–3.2).
2. Systematic Generalization with Edge Transformers, Bergen etal.,NeurIPS2021: https://proceedings.neurips.cc/paper/2021/file/0a4dc6dae338c9cb08947c07581f77a2-Paper.pdf .
3. RealFormer,Heetal.,2021: https://aclanthology.org/2021.findings-acl.81/ .
4. Compositional Matrix-Space Models for Sentiment Analysis,Yessenalina&Cardie,2011: https://aclanthology.org/D11-1016/ .
5. Semantic Compositionality through Recursive Matrix-Vector Spaces,Socheretal.,2012: https://aclanthology.org/D12-1110.pdf .
6. The Lifted Matrix-Space Model for Semantic Composition,Chungetal.,2018: https://aclanthology.org/K18-1049/ .
7. Generating Text with Recurrent Neural Networks,Sutskeveretal.,2011: https://icml.cc/2011/papers/524_icmlpaper.pdf .
8. DeltaProduct,Siemsetal.,2025: https://arxiv.org/html/2502.10297v3 . Bản đầu có tiêu đề khác, cùngarXiv2502.10297.
9. Dynamic Memory Networks for Visual and Textual Question Answering,Xiongetal.,2016: https://proceedings.mlr.press/v48/xiong16.pdf (§3.3,§5).

## 11. Những suy luận cần tránh

- “Student không thể vượt teacher do distillation”: không có định lý như vậy cho downstream training có gold labels/architecture khác.
- “E4/E16failed chứng minh capacity đã đủ”: các screen có recipe/init/budget riêng, không exhaustive.
- “CKA thấp nên layer hỏng”, “driftcao nênforgetting”, “gradientnonzero nên tối ưu đủ tốt”: bằng chứng không hỗ trợ các bước nhảy đó.
- “Tắt nhánh cuối khôngđổiF1 nêntrainingnhánhvôích”: coadaptation khôngđược loại.
- “Có lỗi phủ định nên cầnoperatornoncommutative”: vídụ chỉ là độngcơ, cần comparator và kiểm chứng.
- “Thêmattentiongate/inputconditionalmatrix là mới”: priorart đãrấtđông.
- “Control có context còn candidate mới có context”: inputFFN đãcontextual.
- “Tăng meanF1 là thắngrộng”: metric/taskkhác nhau; báo từngtask,win/loss,vàworstregression.
- “Test chỉ mởmộtlần nên chưa dùngđểresearch”: nhiều vòng đãquan sáttest, phảibáotrungthực adaptiveexposure; khônggọi test hiện tại virginholdout.
- “Mọi hướng cầnmulti-seed ngay”: giaiđoạnexperimental ưu tiên1seed, đo rẻ,cơchế rõ; chỉcandidatehứa hẹn mớixácnhận kỹ. Khônghy sinh correctness,IDs/gold/metrics/parity cầnthiết đểtiếtkiệm.

## 12. Đầu ra mong muốn từ bạn

Trả lời bằng tiếng Việt, dễ hiểu nhưng đủ phương trình và bằng chứng để một cộng sự kiểm tra. Báo cáo cần có:

1. **Nhận định độc lập:** phần nào trong diễn giải lịch sử bạn đồng ý, phản đối hoặc chưa đủ nguồn; chỉ nêu vấn đề ảnh hưởng quyết định.
2. **Bản đồ khoảng trống:** vấn đề có bằng chứng local, vấn đề chỉ có literaturemotivation và phần chưa biết.
3. **2–3 candidate cards** theo đặc tả bước1, có verdict bước2 từngcái. Khôngnhét nhiều trick vào mộtcandidate để khó biếtgain từđâu.
4. **Claimchart và tài liệu gần nhất**: equation/section,điểmgiống/khác,encoderNLUevidence vàgiới hạntransfer.
5. **Một lựa chọn ưu tiên**, hoặc kết luận chưa cócandidateđạt; nêu lýdo vàđiềukiện khiếnđổiý.
6. **Đặc tả tối thiểu có thể triển khai** choứng viênđượcgiữ: nơi gắn,shapes,mask,init,FP32/BF16,gradientflow,complexity,checkpointconversion,adaptationbudgetvàbaselineequivalence.
7. **Kế hoạch kiểm chứng tiếtkiệm:** analytic/syntheticmechanismcheck→single-seedscreen→broad5 tasks→confirmationkhi cótínhiệu. Chốtdựđoán,control,killcriteria vàcách táchcapacity/compute/learningbenefit. Các ngưỡng+.3pp trướcđây làheuristiclịchsử, khôngđịnhluật hoặc significance.
8. **Bản tóm tắt để hai cộng sự đối chiếu**: topidea,claimmới,priorartgầnnhất,risks,thôngtincầnđo,disagreements. Không yêu cầu agentcũ chấpnhận; đưa bằngchứng đểcùngquyếtđịnh.

Nếu chưa có dữ liệu để ước tính giờGPU/thànhcông, ghi giảđịnh vàkhoảngước lượng; khôngfabricate runtimehoặc xácsuấtwin. Khôngđưa ra bảngkếtquả giả củaexperimentchưachạy.

## 13. File map nếu có workspace

Ưu tiên `*_RESULT.md`, finalreviews vàartifactsthực tế hơn những đoạn “nextstep/chưa test” trong plan cũ. Không một file nào mặc nhiên lànguồn hiệntrạng tuyệtđối.

- `docs/E1_TEST_BENCHMARK.md`, `E2_VALIDATION_READOUT.md`, `E2_TEST_REVIEW.md`: baseline/learning.
- `docs/E3_REVIEW_FINAL.md`, `E3_HOTEL_RESULT.md`, `E4_RESULT.md`…`E8_RESULT.md`: screenđầu.
- `docs/E8_TEST_RESULT.md`, `E8_CONFIRMATION_RESULT.md`, `E8_ERROR_READOUT.md`: studentreference cậpnhật. `CURRENT_DISTILLED_BENCHMARK.md` cóbảngmixper-tasklịchsử, khôngdùngđểgọibestsettingchung.
- `docs/D1_READOUT.md`, `D2_READOUT.md`: probe/CKA đãlàm.
- `docs/E9_RESULT.md`…`E15_RESULT.md`, `E14_BAMIBERT_BASELINE_RESULT.md`: architecturevàoriginalbaseline. Khôngdùng `E9_RESULT_buggy_units.md`.
- `docs/E12_LABEL_MAPPING_AUDIT.md`, `E12_SEMANTIC_REVIEW.md`, `E13A_RESULT.md`: nhãn,manualreview,bindingdiagnostic.
- `docs/ARCHITECTURE_TRANSFER_FINAL_REVIEW.md`: auditchi tiếtnhánhE13/E15; phầnproposalE16cuốifileđãđược triểnkhaisauđó, khôngpending.
- `artifacts/e16/E16_RESULT.md`: validation3arms. `docs/E16_FFN_TEST_RESULT.md`: kếtluậntestmớihơn,trướcsnapshotnày.
- `docs/ENCODER_ARCHITECTURE_RESEARCH.md`, `PRETRAINED_ENCODER_TRANSFORMATION_RESEARCH.md`, `ENCODER_REPLACEMENT_GAP_RESEARCH.md`, `IDEA_NOVELTY_AUDIT.md`: cácresearch trước, cócitation; recommendationlịchsửkhôngphải nhiệm vụhiệntại.
- `docs/BACKBONE.md`, `PROVENANCE.md`, `E0_DATA_METRIC_AUDIT.md`: lịch sử/data; đọc mappingcùngcorrectionE12.
- `src/`, `scripts/`, `configs/`, `artifacts/e*/`, `artifacts/d*/`, `artifacts/architecture_transfer_audit/`: implementation/artifacts nếu cần kiểm chứng.

**Bắt đầu từ việc thiết kế và phản biện độc lập. Mục tiêu của lượt này là tạo ra một đề xuất có thể đứng vững trước phản biện, không phải chứng minh agent trước đúng, cũng không phải vội chạy thêm một thí nghiệm.**
