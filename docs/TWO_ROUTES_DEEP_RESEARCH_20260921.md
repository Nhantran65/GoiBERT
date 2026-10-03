# Hai hướng GoiBERT: ≤100M vượt BamiBERT hoặc ≤1B vượt BamiBERT/CafeBERT

Ngày nghiên cứu: 2026-09-21. Root tự đọc báo cáo local, paper/model card/repo tác giả và đếm tham số trên meta device. Không chạy training, inference, test hay tạo subagent trong lượt này. Các pilot dưới đây là đề xuất, chưa được launch.

## 1. Kết luận để ra quyết định

Không có bằng chứng đủ mạnh để hứa một hướng sẽ thắng cả năm task. Sau khi đối chiếu trực tiếp RemBERT–XLM-R và mmBERT-small–base trong paper gốc, thứ tự ưu tiên là:

1. **Supervision đúng task trên nền đã mạnh** cho mục tiêu ≤1B: Cafe/XLM-R-large đã ở gần mốc cần vượt; dữ liệu có nhãn là giả thuyết mới nhất nhưng vẫn phải chứng minh đủ năm task, đặc biệt Sentiment và Hotel.
2. **BamiBERT giữ 12 lớp, giảm FFN 3072→2880** cho mục tiêu ≤100M: còn **99,51M gồm head Hotel**; đây là cách giữ gần nền local 103M. Cần kiểm loss do pruning và tạo gain bằng supervision mới, chưa có điểm sau pruning.
3. **Student ~98,9M giữ 12 lớp + distillation** là chương trình rủi ro cao hơn, cần đối chứng pruning-only/MLM trước khi scale. **RemBERT và mmBERT-small là probe dự phòng với cơ sở cụ thể nhưng tín hiệu yếu cho mục tiêu thắng toàn diện**, không nên mở ngay thành 18-run campaign. EuroBERT-610M còn thấp ưu tiên hơn.

Không ưu tiên lặp lại plain web MLM, decoder→encoder Qwen/BidirLM liều nhỏ, thêm module attention/head, hoặc cứu MiniLM 28–39M bằng một grid dài. Cắt vocab Large chỉ giúp kích thước, không phải cơ chế tăng F1. Không cần bắt buộc hoàn tất mọi ô của Large cắt vocab trước khi screening backbone khác.

## 2. Định nghĩa thắng và phạm vi còn giữ

- Hiểu mục tiêu theo ngữ cảnh: **bốn dataset, năm task** (VSFC sentiment/topic, ViSFD, Restaurant, Hotel); nhánh nhỏ phải vượt Bami trên từng task, nhánh lớn phải vượt cả hai trên từng task. Thắng trung bình nhưng giảm một task chưa hoàn thành mục tiêu nghiêm ngặt này.
- Một pretrained backbone, fine-tune riêng mỗi task như baseline hiện tại. Nếu dùng một checkpoint multitask dùng chung, báo thêm như một setting khác. Không chọn backbone khác cho từng task rồi gọi đó là một encoder mới.
- Tham số = tất cả weights cần tại inference, kể cả embedding và task head; không phải non-embedding, số weights khác zero, số trainable hay active params. Teacher chỉ dùng lúc train không nằm trong budget deployment, nhưng tính vào chi phí huấn luyện.
- Giữ mục tiêu tính mới từ bối cảnh. Chỉ fine-tune checkpoint công khai, vocabulary trimming, CoFi hay MiniLMv2 thông thường đều là baseline/phương pháp đã có. Nếu đạt điểm bằng những cách đó thì có kết quả thực dụng, chưa mặc nhiên có novelty về phương pháp.
- **Vượt baseline local ≠ SOTA công khai.** Phải kiểm đúng version/split/metric, so thêm công trình cùng setting; targeted sentiment khi biết aspect không tương đương end-to-end aspect–polarity pair F1. Protocol local vẫn ghi giới hạn lineage/evaluator và test đã được quan sát qua nhiều vòng. [Protocol](BENCHMARK_PROTOCOL.md), [tổng hợp](EXPERIMENTS_SUMMARY_STANDALONE.md).

## 3. Bằng chứng local thực sự nói gì?

| Quan sát | Suy luận có cơ sở | Điều chưa được chứng minh |
|---|---|---|
| XLM-R-large Restaurant 77,01 vs Cafe 77,06, mean 2 seed; ViSFD 85,48 vs 85,47, 1 seed | Cafe không thể hiện lợi thế rõ trên hai phép đối chứng này | CPT vô ích ở mọi liều/task; năng lực tiếng Việt đã bão hòa hoàn toàn |
| Cafe Topic 81,74 vs XLM-R-large 80,61, 1 seed | Có tín hiệu khác biệt theo task | Gain ổn định, hoặc nguyên nhân nhân quả chỉ do CPT |
| Large đã FT, cắt vocab: 77,8147→77,8147, logits bằng nhau | Compression giữ đầu ra trên Restaurant đã kiểm | Giữ mọi task, mọi input, mọi lượt fine-tune |
| B12 cắt 73,54; B8 71,09 Restaurant | Bỏ bốn lớp làm mất chất lượng trong recipe này | Mọi student <100M đều yếu; bắt buộc dùng 120M |
| M12 61,96, M6 62,93 Restaurant | Hai checkpoint và recipe hiện tại không cạnh tranh ABSA | Student công khai chỉ được train cho XNLI; 39M là bất khả thi tuyệt đối |
| RepKD Cafe→Bami 30M token fail, relation gradient co về ~1% MLM | Objective cosine/span này không tạo gain | Mọi representation KD đã thất bại |
| Composite +1,16 mean 3 seed trên control E19; Cafe −1,02 một seed | Gain phụ thuộc nền/recipe/task | +1,16 là gain so Bami gốc, hoặc transferable sang Cafe |
| Aspect-pair Cafe +0,34 test mean 2 seed, 12 passes/review | Có lợi ích nhỏ, đổi bằng compute | Một đột phá one-pass/SOTA |

Nguồn: [Large baseline](XLMR_LARGE_BASELINE_RESULT.md), [vocab/student](VOCAB_TRIM_STUDENT_RESULT.md), [repKD](CAFEBERT_TO_BAMIBERT_REP_DISTILL_PILOT_RESULT.md), [E22](E22_RESULT.md), [Cafe composite](CAFEBERT_COMPOSITE_PILOT_RESULT.md), [aspect-pair test](CAFEBERT_ASPECT_PAIR_TEST_RESULT.md).

Các sửa đổi quan trọng khi đọc lịch sử:

- **Topic/ViSFD của student dùng LR thắng trên Restaurant**, không phải best LR riêng từng task. Không quy mức giảm Topic B12 1,65 điểm cho trimming.
- MiniLMv2 là task-agnostic distillation; XNLI là benchmark. Paper mô tả chuyển quan hệ Q-Q/K-K/V-V; teacher XLM-R-large dùng layer 19 (đếm từ 1), không đơn giản luôn layer cuối. [Paper gốc](https://aclanthology.org/2021.findings-acl.188.pdf).
- Khác tokenizer không làm **task-logit KD** tự động sai: cùng text và label space vẫn so logits được. Khác tokenizer làm token-level representation alignment khó hơn. Teacher quá tự tin là giả thuyết cần đo, không phải chứng minh duy nhất cho nguyên nhân các run KD âm.
- Hai seed chưa cho phép ấn định “sàn nhiễu 2–3 điểm”. Khoảng chênh quan sát ở M12/B8 không phải độ lệch chuẩn được ước lượng chắc chắn.
- Khoảng 30% tranh chấp là trên **mẫu bất đồng được chọn và do LLM đọc**, không phải 30% nhãn benchmark sai, không suy ra trần F1. [Audit](CAFEBERT_BAMIBERT_STEP0_BLIND_READING.md).

## 4. Nhánh ≤100M

### 4.1. mmBERT-small + vocabulary trimming — probe dự phòng, xác suất thắng toàn diện thấp

Checkpoint chính thức có **22 lớp, hidden 384**, khoảng **140M tổng/42M ngoài embedding**, vocab Gemma 2 khoảng 256K. Với vocab giả định 45K, ước tính ~59M; thậm chí giữ 100K vẫn chỉ khoảng 80M. Có khoảng trống để ưu tiên coverage thay vì ép vocab quá nhỏ. Đây là model pretrained riêng, không phải lấy mmBERT-base rồi bỏ lớp. [Model card](https://huggingface.co/jhu-clsp/mmBERT-small).

Lý do duy nhất để giữ trong shortlist: đây là checkpoint nhỏ pretrained độc lập, khác MiniLM 12×384; trimming có thể đưa nó sâu dưới 100M mà không phải bỏ lớp. **Không có bằng chứng nó có dư địa để thắng Bami trên năm task.** mmBERT-base local là 75,63/83,64/81,18 ở Restaurant/ViSFD/Topic, trong đó ViSFD đã dưới Bami 0,74 điểm; Sentiment và Hotel hơn Bami chỉ 0,37/0,46. Paper mmBERT báo bản small dưới base trên XTREME trung bình **68,6 vs 72,8** và XNLI **73,6 vs 77,1**; trên GLUE tiếng Anh **84,7 vs 86,3**. Đây không phải dữ liệu tiếng Việt in-domain, song nó là prior âm rõ ràng cho việc thu nhỏ. [Paper, Tables 2–3](https://arxiv.org/pdf/2509.06888), [baseline local](BASE_ENCODER_BASELINE_RESULT.md).

**Chỉ nếu cần probe nhanh sau ưu tiên chính:** kiểm tokenization/coverage bằng corpus và train-only; benchmark bản gốc 140M trên Restaurant và ViSFD trước, 2–3 LR riêng mỗi task. Nếu bản gốc đã yếu rõ, dừng trước việc cắt tokenizer. Nếu có tín hiệu, thêm ba task, rồi mới cắt vocab và kiểm logits/fine-tune lại. Bản 140M chỉ là diagnostic, không tính đạt mục tiêu.

Không dùng tập K XLM-R cho tokenizer Gemma. Phải giữ special tokens, byte fallback và tính đúng các dependency của tokenizer. Có thể giữ tokenizer gốc + map ID để kiểm chức năng trước; bản tokenizer độc lập cần kiểm riêng. Tổng ~59–80M là **ước tính**, chưa đếm checkpoint mmBERT-small thực trong lượt này.

**Đánh giá:** phép thử có thể bác bỏ nhanh nhưng prior thắng cả năm thấp; novelty của trimming đơn thuần thấp. Tiền lệ trimming đã có từ [Ushio et al., 2023](https://aclanthology.org/2023.findings-emnlp.981/).

### 4.2. BamiBERT giảm nhẹ FFN — đường ngắn nhất để giữ nền sát baseline

Đếm cấu hình local bằng transformers 4.49 trên meta device, không allocate weights:

| Cấu hình | Encoder | Gồm head E14 136 output |
|---|---:|---:|
| Bami gốc: 12L, H768, FF3072, vocab20481 | 102.361M | 103.056M |
| **12L, H768, FF2880, vocab20481** | **98.819M** | **99.515M** |

Chỉ giảm 6,25% chiều FFN, giữ tokenizer và attention. Prune **có cấu trúc** rồi export tensor nhỏ thật; mask zero không được tính là giảm tổng tham số. Chọn neuron bằng calibration train-only, không chọn bằng validation/test. Phải đối chứng random pruning cùng budget; có thể recovery ngắn với teacher Bami chưa prune.

Tiền lệ structured pruning + distillation: [CoFi](https://aclanthology.org/2022.acl-long.107/). CoFi là task-specific; dùng nó không tự tạo backbone tổng quát. Với mục tiêu encoder chung, chọn cấu trúc trên corpus/train mix rồi cố định trước fine-tune từng task.

**Cái còn thiếu là tăng điểm.** Nén nhẹ chỉ có khả năng giữ gần Bami. Sau đó mới thử supervision tốt hơn ở §6. E22 hỗ trợ thử lại composite như control dữ liệu trên nhánh Bami, nhưng hiệu quả đã biết không bao phủ Sentiment/Topic và không được cộng cơ học vào Bami gốc. Nhánh này khả thi về size, chưa chứng minh thắng năm task; novelty thấp nếu chỉ ghép recipe cũ.

### 4.3. Student XLM-R ~99M + distillation — chương trình nghiên cứu lớn hơn

Không cần đồng nhất “≤100M” với “bỏ xuống tám lớp”. Hai cấu hình gần như bằng nhau về budget đã được đếm:

| Cấu hình đề xuất từ XLM-R-base | Encoder | Gồm head 136 output |
|---|---:|---:|
| **12L, H768, FF2304, vocab35K** | 98.167M | **98.862M** |
| **10L, H768, FF3072, vocab35K** | 98.156M | **98.851M** |
| 12L, H768, FF3072, vocab19K | 100.044M | **100.739M: không đạt** |

Hai cấu hình đầu cho đối chứng có ý nghĩa: giữ depth hay giữ width với cùng budget, cùng vocab và cùng dữ liệu. Pruning FFN phải có selection/recovery; không cắt cột tùy tiện rồi quy lỗi cho kích thước. Vocab35K phải đo lại coverage độc lập, không thừa hưởng kết quả K45K. Cắt chỉ vocab để giữ toàn bộ 12L/FF3072 cần khoảng ≤18K khi tính head lớn; nên chưa ưu tiên vì rủi ro tokenization.

**Thiết kế distillation:** teacher pretrained Cafe hoặc XLM-R-large, student khởi tạo từ XLM-R-base; shared token sequence, remap ID đúng. Teacher frozen. Đối chứng (a) pruning-only, (b) MLM cùng dữ liệu/budget, (c) relation distillation. Dùng quan hệ attention theo MiniLMv2 làm baseline, không dùng lại cosine/span objective đã fail. Khóa layer mapping và recipe trước downstream screening. Không trộn đồng thời relation KD, task KD, LLRD và augmentation trong pilot đầu vì không giải thích được nguyên nhân.

Pilot ngân sách đề xuất: 100M token, checkpoint cuối khóa trước; mở lên 250M/1B chỉ sau khi hơn pruning-only và MLM control. Đây là mốc quản lý rủi ro, **không có bằng chứng 100M là đủ để tái hiện MiniLMv2**. Paper dùng training lớn hơn rất nhiều. Profile actual tokens/s trước khi báo GPU-hours; relation attention T² có thể phá đường fast attention.

**Điểm dừng:** student chỉ giảm KD loss mà không cải thiện downstream thì không scale chỉ vì “cần thêm token”. Phải thấy mean gain qua seed, đặc biệt Restaurant/ViSFD. Mục tiêu student 75,5 Restaurant là aspiration, không phải điểm dự báo.

## 5. Nhánh ≤1B

### 5.1. RemBERT và EuroBERT-610M — probe backbone có điều kiện

**RemBERT:** chính thức có 32 lớp/H1152, input embedding256. Đếm config bằng meta device: **574.592M encoder**, bảng token64.077M, phần còn lại **510.515M**. Nền XLM-R-large local chỉ ~302.8M ngoài bảng token. Paper gốc báo RemBERT vượt XLM-R-large trên XTREME **75,4 vs 71,4 trung bình**, gồm XNLI **80,8 vs 79,2** và PAWS-X **87,5 vs 86,4**; đây là chuyển từ train tiếng Anh sang ngôn ngữ khác, không phải fine-tune tiếng Việt có nhãn như GoiBERT. RemBERT cũng dùng khoảng **995M tham số lúc pretraining** vì output embedding lớn được bỏ ở inference; không phải chứng minh 575M inference tự hiệu quả hơn XLM-R-large ở cùng điều kiện. Chênh lớn nhất trong paper có cả QA/NER; chúng không nằm trong năm task local. [Paper gốc, Tables 7/16](https://arxiv.org/pdf/2010.12821), [repo tác giả](https://github.com/google-research/google-research/tree/master/rembert), [config](https://huggingface.co/google/rembert/blob/main/config.json).

**EuroBERT-610M:** inventory local trước đây đếm **607.875M encoder**; ~755.63M stored MLM có output head untied, bỏ head này khi classification. Trần cũ600M loại nó; trần mới 1B cho phép. Pretraining có tiếng Việt. Tuy nhiên paper nói classification nhìn chung ngang model cùng cỡ, ưu thế retrieval/code không đảm bảo ABSA. [Paper §3.2](https://arxiv.org/html/2503.05500v2), [inventory local](../artifacts/research_20260918/parameter_accounting.json).

**Ưu tiên:** cả hai thấp hơn cải thiện supervised trên Cafe/XLM-R-large. Nếu cần bác bỏ ý tưởng backbone nhanh thì RemBERT có bằng chứng ngoài miền nhỉnh hơn EuroBERT, nhưng chưa đủ để dự báo vượt Cafe. Kiểm pooling chính thức, tokenizer fertility/truncation và implementation; không ép first-token pooling vào mọi kiến trúc nếu recipe gốc khác. Với RemBERT lưu ý dropout/optimizer recipe; một LR cố định dễ cho kết luận âm giả.

**Pilot nếu mở:** RemBERT trên Restaurant và ViSFD, LR nhỏ quanh khoảng paper dùng và LR E14, seed 42; chỉ mở Topic/Hotel/Sentiment, seed43/44 nếu sát hoặc hơn Cafe ở hai task đầu. EuroBERT dùng quy trình tương tự sau đó. Candidate thua rõ cả Restaurant và ViSFD không mở CPT để cứu ngay.

### 5.2. CafeBERT + supervision đúng task — hướng cải thiện có liên hệ local rõ nhất

Cafe đã có khoảng cách tốt với Bami ở phần lớn task; tăng nhãn hữu ích không tăng params inference. Có thể cắt vocab Cafe riêng về khoảng 349M bằng pipeline tương tự XLM-R, nhưng phải kiểm lại, chưa coi đó là kết quả đã có. Vì Cafe 560M đã dưới1B, trimming là tùy chọn triển khai, không phải điều kiện để nghiên cứu gain.

Đầu tiên thử **huấn luyện chung Restaurant+Hotel với encoder shared, head riêng**, rồi fine-tune riêng. Hai task có 12 và 34 aspect, không ghép numeric label ID trực tiếp, không coi aspect không có annotation ở miền kia là absent. Dữ liệu thêm là khoảng 7K review train miền lân cận, không phải tăng gấp đôi dữ liệu cùng phân phối. Phải giữ target-only control cùng optimizer steps và số lần thấy target. Transfer sang ViSFD hoặc VSFC chưa được đảm bảo; VSFC sentiment/topic có thể shared encoder hai head nhưng cũng cần control.

Phương án thứ hai là dữ liệu sinh có nhãn ở §6, ưu tiên hơn nối thẳng hai review vốn đã fail trên Cafe. Cafe + dữ liệu sinh và Bami99.5M + cùng dữ liệu tạo ablation xem lợi ích có thật sự cần model lớn hay không.

### 5.3. Ứng viên dự phòng, không mở một cuộc quét vô hạn

| Ứng viên | Lý do còn hợp lệ | Vì sao chưa ưu tiên |
|---|---|---|
| InfoXLM-large, khoảng559M | Cùng kiến trúc/tokenizer XLM-R; thêm cross-lingual contrastive learning; code dễ tái sử dụng | Chứng cứ chủ yếu cross-lingual transfer, chưa có local ABSA |
| LFM2.5-Encoder-350M | Encoder bidirectional có tiếng Việt, khác kiến trúc, có public FT evaluation | Bảng tác giả không hơn XLM-R-large ở multilingual tasks được nêu; lợi thế dài/CPU không trúng benchmark ngắn |
| mGTE multilingual MLM, ~306M | Backbone MLM đa ngữ công khai | Không lấy MTEB của bản embedding làm bằng chứng thắng classification FT |
| PhoBERT-large, ~368M | Monolingual encoder chưa có baseline local đầy đủ trong nhánh hiện tại | Cần segmentation đúng; paper Bami chủ yếu so PhoBERT-base, không loại large; nhưng thiếu tín hiệu local để xếp đầu |
| ViT5-large, ~791M full encoder-decoder | Có thể sinh nhãn theo schema dưới1B | Đổi cả objective/inference/parser; không tự thắng và không còn encoder-only setting |

Nguồn: [InfoXLM chính thức](https://github.com/microsoft/unilm/blob/master/infoxlm/README.md), [LFM model card và bảng đầy đủ](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M), [mGTE](https://huggingface.co/Alibaba-NLP/gte-multilingual-base), [PhoBERT](https://github.com/VinAIResearch/PhoBERT). Size ViT5 và mGTE từ inventory local ngày18; cần audit lại deploy head khi thực thi.

Ettin1B/ModernBERT-large tiếng Anh không được xếp cao chỉ vì leaderboard tiếng Anh. Qwen3/BidirLM đã có phản chứng local cùng cỡ; chưa có lý do ưu tiên adaptation tiếp. [Ettin](https://arxiv.org/abs/2507.11412) cũng cho thấy encoder/decoder có lợi thế theo mục tiêu train, không phải decoder hiện đại luôn thay encoder tốt hơn.

## 6. Hướng dữ liệu dùng được cho cả hai budget

**Giả thuyết:** phần thiếu nằm ở ví dụ supervised phân biệt aspect, absent/neutral và cảm xúc ngầm; có thể bổ sung bằng dữ liệu có kiểm chứng. Đây là giả thuyết dựa trên audit, không phải chẩn đoán nhân quả đã chốt.

Tiền lệ gần nhất: [LACA, ACL2025](https://aclanthology.org/2025.acl-long.41/) sinh câu phù hợp pseudo-label cho cross-lingual ABSA. Setting này khác supervised Vietnamese category ABSA của ta. [Semantic-preserved augmentation, 2025](https://arxiv.org/abs/2506.07148) gần ACSA hơn. Hai nguồn hỗ trợ thử data augmentation, không cung cấp con số gain dự báo cho GoiBERT. SKEP cũng là tiền lệ dùng sentiment supervision khác plain MLM: [paper](https://aclanthology.org/2020.acl-main.374/).

**Thiết kế pilot có thể bác bỏ:**

1. Chỉ lấy train thật và ontology để viết schema/prompt. Sinh khoảng2–5K mẫu đạt chất lượng trước, chưa mua/sinh 100K. Mẫu gồm paraphrase giữ nhãn, thay một polarity có kiểm soát, và thêm/bỏ một aspect. Gắn lại toàn bộ vector nhãn; không giả định các aspect khác tự giữ nguyên.
2. Dùng review/đánh giá độc lập theo guideline, audit phân tầng neutral/absent/multi-aspect. Không lọc chỉ bằng “Cafe đồng ý” vì như vậy dễ giữ nguyên blind spots của Cafe. Không dùng gold validation/test để tạo hoặc lọc mẫu.
3. Bốn arm cùng compute: real-only lặp thêm steps; composite cũ; synthetic thông thường; synthetic có chỉnh sửa kiểm soát. Giữ số mẫu/nhãn marginal gần nhau để tách lợi ích cách sinh khỏi oversampling.
4. Thử trước trên Restaurant, xác nhận ViSFD hoặc Hotel; sau đó thiết kế dữ liệu riêng cho VSFC. Không dùng gain ABSA để claim thắng Topic/Sentiment.
5. Train teacher/student dùng cùng split; LLM chỉ chạy offline lúc tạo dữ liệu. Không cần LLM ở inference, nhưng phải tính chi phí generation và annotation.

**Nếu làm task KD:** so hard label synthetic với soft logits teacher; đo entropy và lỗi teacher trên train folds trước. Out-of-fold teacher là một control tránh học từ teacher đã thuộc chính mẫu đó, không phải thuốc chữa chắc chắn: teacher fold ít dữ liệu có thể yếu hơn. Cross-tokenizer task logits hợp lệ. Nếu soft KD không hơn hard labels cùng data/compute thì bỏ KD.

**Đóng góp nghiên cứu tiềm năng:** phương pháp chọn/kiểm chứng supervision giữ quan hệ aspect–polarity khi nén, với ablation ngang compute và kiểm ngoài miền. Tuy nhiên counterfactual augmentation, confidence filtering, relation KD đều có tiền lệ; chưa được gọi một cách phối hợp là mới chỉ vì áp dụng tiếng Việt. Cần review prior art cụ thể sau khi có pilot dương.

## 7. Các hướng không nên đầu tư tiếp lúc này

| Hướng | Quyết định |
|---|---|
| Thêm250M token web MLM/MNTP cùng kiểu cũ | Dừng; nhiều nền đã không có gain ABSA |
| Tăng thẳng lên10–60B token CPT | Không ưu tiên; liều lớn chưa bị bác bỏ nhưng thiếu tín hiệu/ROI, không mặc định cấm vĩnh viễn |
| Train từ đầu100M–1B bằng corpus Việt | Rủi ro và compute lớn; chưa có bằng chứng vượt khởi tạo pretrained với nguồn lực hiện tại |
| MiniLM28–39M tiếp tục FT/KD không đổi giả thuyết | Gap Restaurant quá lớn, chưa đáng ưu tiên |
| B8 bỏ lớp rồi tự hứa KD bù4điểm | Chưa có bằng chứng; đo cấu hình giữ depth/cùng budget trước |
| Các module reader/gate/cavity/pair-value cũ | Không mở lại nếu không có giả thuyết mới và control khác lịch sử |
| Aspect-pair Cafe12/34passes | Gain test nhỏ, compute cao; baseline tham khảo, không trục chính |
| Quantization để tuyên bố≤100M | Không hợp lệ: giảm bytes, không giảm params |
| Hai Cafe cắt vocab ensemble≈698M | Có thể hợp budget hệ thống, nhưng không phải một encoder mới; không ưu tiên vì latency/novelty |

Không dùng xác suất thành công40–70% như báo cáo cũ: chưa có dữ liệu để hiệu chuẩn các xác suất đó.

## 8. Lộ trình đề xuất và cổng dừng

**Vòng A — ưu tiên gắn với kết quả local:**

- Nhánh nhỏ: Bami FFN2880 được đếm đúng budget; đo loss sau pruning/recovery trước, rồi chọn một cải thiện supervised có control. Nếu model này không giữ được chất lượng, chuyển sang student 12 lớp ~98,9M; mmBERT-small chỉ là probe dự phòng.
- Nhánh lớn: thử một can thiệp supervised trên Cafe/XLM-R-large với target-only matched control. RemBERT là probe dự phòng nếu muốn kiểm xem backbone khác có lợi thế tự thân; EuroBERT tiếp sau.
- Bami/Cafe tham chiếu phải có ngân sách tuning tương đương ở vòng xác nhận; bảng lịch sử dùng để screen, không dùng để chứng nhận win.

Không chốt trước tổng số runs/GPU-hours cho vòng này vì cần profile pruning và kiểm chất lượng dữ liệu. Neo thực tế là Large cũ 22–36 phút/run; RemBERT chưa có throughput local.

**Gate đầu tư, không phải significance:** với3 task screening, candidate nên hơn baseline mục tiêu≥0,5 pp ở≥2 task, task còn lại không thấp hơn quá0,5 pp. Nhánh nhỏ so Bami; nhánh lớn so max(Bami,Cafe) mỗi task. Vùng sát ngưỡng chỉ đáng thêm seed nếu còn cơ chế/data cụ thể để cải thiện. Thua>1 pp cả hai ABSA thì giảm ưu tiên, không tự scale CPT.

**Vòng B — xác nhận:** khóa LR riêng từng task; seeds 42/43/44, đủ 5 task. Báo từng task và paired delta. Candidate nhỏ chỉ đạt goal khi bản deploy≤100M, không dùng điểm bản gốc140M. Nếu chưa có win rộng, chọn một nhánh supervision §6 với control rõ; không cộng các gain lịch sử.

**Vòng C — nếu chọn nghiên cứu distillation:** hai student98.85M / 98.86M là cặp đối chứng depth/width; bắt đầu không KD, sau đó KD và MLM matched. Chỉ mở ngân sách lớn khi có lợi ích downstream lặp lại. Không tiến hành song song với một cuộc quét backbone/data không giới hạn.

**Final:** theo protocol hiện có, ưu tiên 5 seed cho kết quả cuối, đăng ký trước mọi recipe trước test; test cũ được gọi đúng là đã quan sát. Báo uncertainty theo seed và paired bootstrap theo review/câu, không bootstrap từng slot độc lập. Muốn claim SOTA phải audit leaderboard/split/metric và có holdout độc lập hoặc đánh giá ngoài miền để giảm nguy cơ chọn phương pháp theo test lịch sử.

## 9. Kiểm toán giới hạn của chính nghiên cứu này

- Chỉ đọc paper/card/repo công khai, không chứng nhận đã bao quát mọi công trình 2026 hoặc mọi checkpoint ≤1B.
- Không tải weights mới. Config mmBERT-small chưa fetch được qua URL config trong lượt này; dùng architecture/size công bố, vì vậy size sau trim chỉ là ước tính.
- RemBERT được đếm từ config chính thức; Bami/XLM-R dùng config local. Đây là tính toán shape, không phải checkpoint đã prune/reload thành công. Task head 136 là head Hotel lớn trong setting hiện tại, không phải tuyên bố hỗ trợ mọi head ≤100M.
- EuroBERT count tái dùng inventory local ngày18; deployment cần pin revision và đếm lại. Bản model card/HTML online là bằng chứng tra cứu, chưa snapshot toàn bộ source theo SHA.
- Không sửa dữ liệu, nhãn hay kết quả cũ. Tài liệu này thu hẹp các kết luận quá mạnh trong bảnsummary trước, không ghi đè lịch sử.

**Khuyến nghị sau khi phản biện trực tiếp bằng paper:** ưu tiên **supervision đúng nhãn trên Cafe/XLM-R-large** cho nhánh ≤1B và **Bami FFN2880 + supervised recovery** cho nhánh ≤100M. Distillation vào student giữ 12 lớp ≈98,9M là nghiên cứu tiếp theo nếu pruning nhẹ không đủ. RemBERT có một prior dương ngoài miền nhưng chưa đủ để xếp trước nhánh supervised; mmBERT-small có prior âm từ khoảng cách với base và từ kết quả mmBERT-base local. EuroBERT-610M vẫn dự phòng. Không có bằng chứng nào hiện tại biện minh cho lời hứa thắng cả 5 task.

Kết quả đếm tham số máy đọc: [parameter_counts.json](../artifacts/research_20260921_two_routes/parameter_counts.json).
