# Các khoảng trống còn khả thi khi thay thế thành phần encoder

## 1. Kết luận và hướng ưu tiên

**Hướng ưu tiên để kiểm chứng tiếp là thay multi-head attention độc lập bằng attention có trao đổi thông tin giữa các head, bắt đầu từ Talking-Heads tĩnh và khởi tạo identity.** Đây là phương pháp đã có, không phải đề xuất một block hoàn toàn mới. Khoảng trống có thể nghiên cứu nằm ở khả năng chuyển cơ chế này vào BamiBERT pretrained, sự cần thiết của các liên kết chéo giữa head, và lợi ích trên nhiều tác vụ tiếng Việt trong cùng pipeline.

Chưa có bằng chứng chỉ ra head nào hoặc layer nào của BamiBERT là bottleneck. Cũng chưa có cơ sở khẳng định “chưa ai làm cho tiếng Việt” là novelty đủ mạnh. Đề xuất là một phép kiểm chứng có chi phí giới hạn, trước khi quyết định có đáng thực hiện thêm một vòng fine-tune đầy đủ hay không.

Hai hướng khác còn mở nhưng xếp sau là **thay chuẩn hóa attention bằng phân bố sparse/independent** và **thay token mixer bằng SSM/hybrid**. Hướng đầu có tiền lệ chuyển RoBERTa nhưng thường cần continuation đáng kể; hướng sau có bằng chứng encoder nhưng mục tiêu nổi bật là chi phí/context dài. Chúng chưa phù hợp bằng hướng head composition với mục tiêu tăng chất lượng trên các câu ngắn đang đánh giá.

Lượt rà soát này xét tài liệu tới 12/09/2026, ưu tiên paper, repository và công bố trực tiếp của tác giả. Không chạy model, không tạo subagent và không triển khai thí nghiệm mới. Kết luận “chưa được kiểm chứng trong project” được tách khỏi “gap chưa được giải quyết trong literature”.

## 2. Các kết quả cũ thực sự loại trừ điều gì?

E16 FFN-input gate tăng Topic nhưng giảm Sentiment, ViSFD và Hotel trên test. Kết quả này đủ để dừng cấu hình đã thử theo mục tiêu thắng rộng. Nó không bác bỏ mọi conditional computation hoặc mọi gated architecture. E13/E15 tương tự: cơ chế pair-value residual không cho cải thiện rộng trong protocol đã chạy.

| Nhóm thử nghiệm | Đã can thiệp | Chưa được kiểm chứng bởi nhóm đó |
|---|---|---|
| E9 | Đọc layer cuối bằng query theo nhãn/aspect | Trao đổi score giữa các head bên trong encoder |
| E10 | Nhánh hợp thành span cục bộ | Token mixer khác hoặc topology liên kết giữa head |
| E11 | Signed/convex combination của hai attention maps cùng head | Phối hợp score/value bank giữa nhiều head pretrained |
| E13/E15 | Thêm message tương tác phi tuyến vào output | Thay toàn bộ phép tính routing tại block được chọn |
| E16 | Điều biến các kênh FFN | Phân phối attention khác hoặc head composition |

Đây là phân loại phạm vi phép thử, không là lập luận rằng những phần chưa thử chắc chắn tốt hơn. Thay thế một thành phần cũng có thể thất bại như thêm nhánh. Điểm cần cải thiện trong cách nghiên cứu là có đối chứng phân biệt được cơ chế, không chỉ đổi tên từ “adapter” thành “replacement”.

[Audit E13/E15](ARCHITECTURE_TRANSFER_FINAL_REVIEW.md) đã đo20checkpoint với60lượtvalidation. Tắt nhánh không đổi F1 ở12/20; FP32 fusion không đổi F1 ở13/20; gradient hiện tại hữu hạn và P khác0. Vì vậy, không tiếp tục lấy chết gradient hay một lỗi fusion đơn lẻ làm lời giải thích mặc định để cứu các nhánh cũ.

Năm task hiện tại thuộc bốn nguồn văn bản, với hai task VSFC dùng chung câu. Tất cả đều là phân loại câu hoặc aspect-category sentiment. “Thắng rộng trong project” chưa đồng nghĩa “encoder tiếng Việt tốt hơn trên mọi NLU”. BamiBERT paper còn đánh giá NLI, NER và spam; đây là phần kiểm chứng rộng cần bổ sung sau khi có candidate, không phải điều đã thực hiện.[^1]

## 3. Gap A — Phối hợp giữa attention heads

### Bằng chứng và phản bằng chứng

Talking-Heads thêm phép chiếu qua chiều head trước và sau softmax. Trong thí nghiệm BERT12head của paper, MNLI tăng82,6→83,6 và SQuAD1.1 F1 tăng88,51→89,2. Nhưng model được pretrain và dùng relative positions; đây không phải chuyển trực tiếp BamiBERT bằng fine-tuning. Appendix còn thử dynamic projections: pretraining loss tốt hơn không luôn chuyển thành downstream gain.[^2]

DCMHA tiếp tục hướng phối hợp head theo input; bằng chứng chính là language modeling và các model kiểu Pythia.[^3] Vì vậy, dynamic composition không phải phần mới nếu chỉ áp lại cho tiếng Việt, và kết quả decoder không đủ làm lý do ưu tiên nó trước static composition.

Nghiên cứu low-rank bottleneck cung cấp động cơ về giới hạn do kích thước từng head.[^4] Song student và BamiBERT đều có head dimension64. Không được suy rằng đây là nguyên nhân student thua model lớn hoặc tăng head count sẽ tăng chất lượng.

Head pruning cho thấy nhiều head có thể bị loại mà không gây giảm lớn trong một số thiết lập.[^5] Các nghiên cứu về head chuyên biệt cũng có những giới hạn theo tác vụ và mô hình.[^6] “Có redundancy” không chứng minh head cần giao tiếp nhiều hơn: redundancy có thể hữu ích, và một số head nhỏ về norm vẫn quan trọng cho nhãn hiếm. CKA/cosine của map chỉ là mô tả, không phải tiêu chí causal để chọn layer.

### Khoảng trống có thể khai thác

Câu hỏi phù hợp là: **những score spaces đã pretrained có bổ sung được cho nhau khi được kết hợp trước nonlinear normalization, hơn việc chỉ rescale từng head hay không?** Đây là khả năng biểu diễn khác với gate FFN hoặc nhánh tạo feature random. Nó có thể dùng chung cho các task mà không cần gold target span hoặc parser.

Trong BamiBERT hiện tại, mỗi head tính attention map riêng trước khi output được ghép. Head composition can thiệp ngay vào cách chọn nguồn thông tin, thay vì chỉ thêm correction sau khi lựa chọn đã xong. Lợi ích tiềm năng là dùng nhiều kiểu so khớp đã học để quyết định một luồng truy hồi; chưa có dữ liệu project chứng minh cần điều này.

Khoảng trống thực nghiệm còn lại gồm ba phần: chuyển từ checkpoint có sẵn mà không phá hàm ban đầu; tách cross-head interaction khỏi temperature/output rescaling; và xác nhận lợi ích có giữ trên nhiều domain, không chỉ một task nhạy seed. Nếu không làm được ba phần này, chỉ có một application của phương pháp đã có, chưa phải đóng góp kiến trúc mạnh.

## 4. Gap B — Chuẩn hóa và cạnh tranh trong routing

Softmax tạo một phân phối tổng bằng1 trên keys hợp lệ. Khi thay bằng sigmoid hoặc entmax, điều thay đổi là cách lựa chọn và chuẩn hóa các đóng góp, không chỉ thêm một gate output. Giả thuyết có thể liên quan đến nhiều bằng chứng cùng lúc hoặc distractors, nhưng chưa có kiểm tra cho thấy đó là nguyên nhân lỗi BamiBERT trên năm task.

Paper về sigmoid attention có lý thuyết, thực nghiệm và hướng dẫn ổn định; phần language modeling đọc được dùng thiết lập autoregressive, không phải BERT fine-tuning. Length-dependent bias và kiểm soát norm là các chi tiết quan trọng.[^7] Không thể lấy “drop-in replacement” trong abstract thành cam kết chỉ đổi một dòng trên BamiBERT rồi giữ nguyên mọi hyperparameter.

AdaSplash có bằng chứng gần hơn: chuyển RoBERTa/ModernBERT sang entmax bằng2Btoken continuation; parameter sparsity được anneal trong giai đoạn đầu, có softmax continuation control. GLUE nhìn chung tương đương. Appendix ghi khởi đầu alpha1,01, trong khi phần mô tả chính nói gầnsoftmax1,0; chi tiết implementation phải lấy đúng phiên bản khi tái lập.[^8]

Sparsefinder có MLM encoder-only, nhưng trọng tâm là dự đoán đồ thị attention thưa và trade-off hiệu năng.[^9] Repository AdaSplash hiện phân biệt phiên bản1/2; không trộn kernel mới vào kết luận của paper cũ hoặc gọi cải thiện tốc độ là cải thiện F1.[^10]

**Gap còn mở:** BamiBERT có đang chịu một bất lợi thực sự từ normalization trên short-text multi-label/category tasks không, và có chuyển được với budget nhỏ hơn các paper hay không? Đây là giả thuyết cần đo, không phải lỗ hổng đã xác nhận. Với mục tiêu chất lượng, bằng chứng “giữ accuracy trong khi chạy nhanh hơn” chưa đủ để đặt hướng này lên đầu.

Nếu đi hướng này, phải đếm key hợp lệ thay vì chiều padding khi bias phụ thuộc độ dài. Mọi thay đổi alpha/temperature cần được khóa trước khi xem validation. Không dùng một sweep nhiều biến thể rồi chọn test đẹp. Cần runtime control cùng implementation, bởi kernel và masking có thể tạo thay đổi số học độc lập với distribution.

## 5. Gap C — Thay token mixer bằng SSM hoặc hybrid

BiGS cho thấy mô hình bidirectional state-space có thể đạt chất lượng GLUE ngang BERT trong thiết lập pretraining của công trình.[^11] Monarch Mixer cũng là tiền lệ encoder không dựa hoàn toàn vào attention, với mục tiêu chất lượng/hiệu năng.[^12] Các kết quả đó chứng minh có lựa chọn kiến trúc khác, không chứng minh chuyển một layer BamiBERT sang SSM sẽ thắng ngay.

MaBERT là preprint2026 có GLUE và ablation padding/mixing. Công trình dùng MLM budget1Mstep, báo cáo nhiều seed và không strict parameter-match vì thêm SSM capacity. Các kết quả long-context chủ yếu là hiệu năng thực thi; paper thừa nhận GLUE không đo long-context reasoning trực tiếp.[^13]

Một gap kỹ thuật thật của SSM bidirectional là bảo toàn trạng thái khi padding và khi đảo chiều chuỗi. Nhưng đó là vấn đề của kiến trúc đích, không bằng chứng BamiBERT hiện có lỗi padding. Mang một vấn đề mới vào rồi giải quyết nó chưa chắc giúp vượt model nguồn.

Trên các train split project, metadata ghi0% truncation ở256token. Vì vậy, lợi ích asymptotic theo chiều dài không khớp tốt với mục tiêu gần. Nếu chuyển nhiệm vụ sang văn bản dài, retrieval hoặc constraints latency thì thứ tự ưu tiên có thể khác; trong phạm vi hiện tại, SSM/hybrid là chương trình lớn hơn cần budget chuyển weights và corpus riêng.

## 6. Gap D — Thay FFN bằng các phân vùng computation

MoEfication chia FFN pretrained thành các experts và router, chủ yếu phục vụ giảm computation.[^14] G-MoEfication xử lý các model không thỏa giả định sparsity kiểuReLU, phù hợp hơn khi cân nhắc GELU của BamiBERT.[^15] Sparse Upcycling sao chép FFN vào experts và học tiếp, có kết quả T5/SuperGLUE nhưng sử dụng continuation đáng kể.[^16]

Chúng khác E16: E16 rescale channel trong một FFN, còn partitioned computation quyết định nhóm parameters nào xử lý token. Tuy vậy, lợi ích quality không tự suy ra từ routing thưa. Nếu chỉ giữ95%performance để giảmFLOPs, đó không đáp ứng mục tiêu vượt baseline về F1.

Khoảng trống đáng nghiên cứu về lâu dài là phân chia/chia sẻ chuyên môn mà vẫn giữ kiến thức gốc và tổng quát hóa qua domain. Nhưng đó dễ biến thành thêm capacity hoặc domain specialization, trái mục tiêu một setting thắng rộng nếu không có control. Vòng đầu chưa nên mở nhiều experts chỉ vì các gate trước thất bại.

## 7. Gap E — Cách chuyển weights, không phải bản thân block mới

BERT-of-Theseus cho một tiền lệ encoder về thay module dần.[^17] Attention Editing mới hơn nghiên cứu chuyển attention ở model đã huấn luyện, nhưng bằng chứng được đọc thuộc LLM, không có đánh giá BERT trong phần văn bản.[^18] Các công trình này hữu ích cho protocol chuyển đổi; không chứng minh kiến trúc đích tốt hơn model nguồn.

Có ba trạng thái cần tách: operator mới chứa operator cũ như một trường hợp đặc biệt; operator mới chỉ xấp xỉ được hàm cũ sau calibration; và operator mới cần học tiếp đáng kể. Chọn một khởi tạo identity mà sau đó chỉ có một đường gradient hoạt động cũng không đủ để khẳng định đã xử lý mọi vấn đề tối ưu.

Với head composition cùng số head, hai ma trận trộn identity cho một điểm xuất phát rõ. Vì đó là trường hợp đặc biệt của operator mới, không cần thêm feature-matching chỉ để match một hàm đã trùng. Nếu sau này cần continuation, baseline phải nhận cùng corpus/objective/budget. Không mặc định huấn luyện teacher imitation là đủ để vượt teacher; nhãn task hoặc objective dữ liệu vẫn phải quyết định lợi ích cuối.

## 8. Ma trận ưu tiên

Các mức dưới là đánh giá kỹ thuật có điều kiện, không phải xác suất thành công.

| Hướng | Bằng chứng encoder/NLU | Khả năng dùng weights BamiBERT | Độ khớp short-text F1 | Rủi ro novelty | Ưu tiên |
|---|---|---|---|---|---|
| Static cross-head composition | Có trực tiếp | Identity conversion tương đối rõ | Khá | Cao: phương pháp đã có | 1 |
| Dynamic head composition | Mạnh hơn ở decoder; kết quả dynamic không đồng nhất | Phức tạp hơn | Có thể | Rất cao | Sau static, không thử ngay |
| Entmax/sigmoid replacement | Có encoder ở sparse line; sigmoid phần lớn khác thiết lập | Cần calibration/continuation | Chưa có bằng chứng gain rộng | Cao | 2 |
| FFN partition/upcycling | Có NLU/encoder-decoder | Tận dụng weights tốt nhưng thêm routing | Chưa rõ quality | Cao | 3 |
| Bidirectional SSM/hybrid | Có encoderGLUE | Chuyển đổi khó hơn | Lợi thế chính chưa khớp dữ liệu ngắn | Cao | Chương trình riêng |
| Thêm gate/correction kiểu đã thử | Project đã có negative screens | Dễ | Không có tín hiệu rộng | Cao | Dừng |

Ưu tiên1 không có nghĩa chọn ngay15fulltrials. Nó có nghĩa đáng thực hiện một phép kiểm tra khả năng tận dụng head bank hiện tại trước. Nếu capability probe không có tín hiệu, không cần triển khai dynamic composition để cứu.

## 9. Đề xuất kiểm chứng: identity-anchored cross-head replacement

### Mục tiêu và vị trí

Thay phép MHA trong **bốn block trên cùng của BamiBERT**, blocks9–12/index8–11, bằng một operator có trao đổi score giữa head. Đây là lựa chọn phạm vi thử trước kết quả, không xuất phát từ tuyên bố đã tìm được layer lỗi. Tám block dưới và toàn bộ Q/K/V/Wo được giữ từ checkpoint.

Phạm vi bốn block giúp thử một thay đổi operator liên tiếp ở vùng trên, khác việc đặt một correction nhỏ ở một block. Đồng thời, nó là lựa chọn mới cần đối chứng; nếu thất bại không được tự mở sweep tất cả vị trí. Static composition là bước đầu, không thêm input-conditioned gates.

### Hợp đồng operator

Với vector score qua12heads tại từng cặp token là s, dùng hai ma trận12×12: L cho logits, R cho weights. Tính score sau trộn, softmax theo keys, rồi trộn weights qua heads trước khi nhân V. Khởi tạo L=I,R=I; giữ12head vàdimension64.[^2]

Đây là adaptation của Talking-Heads đã biết. Hai ma trận thêm288parameters mỗi block, tổng1.152 cho bốn block. Đó là phép đếm thiết kế, không phải số đếm từ code đã chạy. Số parameters nhỏ không đồng nghĩa overhead nhỏ: có thêm phép nhân theo head trên tensor attention.

Phải trộn **logits hữu hạn trước khi áp mask**. Trộn tensor chứa−inf có thể gâyNaN với coefficient0 hoặc âm. Sau softmax và weight mixing, keys padding vẫn phảizero. Không tự renormalize sauR, vì như vậy sẽ đổi operator; nếu muốn ràng buộc R phải coi là variant khác, không sửa âm thầm.

Không nhầm việc head trao đổi score với việc model tạo thêm Q/K/V random. Những vector cũ đã được pretrained và luôn nằm trong đường xử lý chính. Tuy nhiên, identity initialization không bảo đảm matrices sẽ học ra một tổ hợp tốt; đó là điều cần kiểm tra.

### Control phân biệt cơ chế

Ba arms: MHA gốc; hai ma trận chỉ có diagonal học được; hai ma trận full học được. Diagonal control kiểm tra temperature/output scaling mà không trao đổi thông tin giữa head. Nó có ít trainable parameters hơn full mixing; phải báo đúng, không thêm parameters không dùng để tạo cảm giác parameter-match.

Tất cả arms cần cùng implementation tại các block được thử, microbatch và precision. Nếu baselineSDPA còn candidateeager thì không thể gán các thay đổi nhỏ cho mixing. Vòng kỹ thuật phải đo evalinitparity và gradient, không chỉ kiểm tra shapes.

Full mixing hơn baseline nhưng ngang diagonal chưa ủng hộ hypothesis cross-head composition. Full hơn diagonal mới là tín hiệu thích hợp để đầu tư, nhưng vẫn phải xác nhận trên nhiều task và seed. Không dùng entropy thấp hoặc matrix khác identity làm bằng chứng thay thế F1.

## 10. Cách sàng lọc để không lặp lại chuỗi thử mù

### Bước A — Capability probe nhỏ

Dùng các checkpoint E14 đã fine-tune theo task, đóng băng weights và head gốc; chỉ tối ưu matricesmixing hoặcdiagonal trên train. Đây là diagnostic adaptation của model đã chọn bằngvalidation, không phải một baseline pretrained mới hoặc test độc lập.

Đề xuất giới hạn200optimizersteps cho từng arm có học trên mỗi task, một seed, fixedtrainorder. Không mở LRgrid. Learning rate và bound update phải được khóa trước khi nhìn validation, sau một kiểm tra train-only về tính ổn định. Chưa chọn giá trị cuối khi chưa có profilegradient/runtime; không giao implementation tự chọn theo validation.

Đánh giá trainloss, validationF1, số predictions đổi, norm L−I/R−I và cost. So cả2controls với baseline đóng băng. Nếu chỉ fittrain nhưng không có lợi thế validation hoặc full không hơn diagonal, dừng ở đây. Không tự tạo câu synthetic khó hơn đến khi có lỗi để biện minh tiếp.

Probe này có selection bias vì E14checkpoint đã được chọn trênvalidation. Do đó nếu nó có tín hiệu, cần fulltraining từ originalBamiBERT ở bướcB; không dùng ngay probecheckpoint để tuyên bố vượt baseline hoặcSOTA. Nhãn integer và semantics đã audit vẫn phải giữ chính xác.

### Bước B — Fine-tune có đối chứng

Chỉ nếu bướcA có tín hiệu nhiều domain, dựng lại từ officialBamiBERT pretrained, cùngtaskheadinit và cùngrecipe. Baseline, diagonal vàfullmixing đều đi qua cùngoperatorimplementation; cố địnhmicro16/eff32. Cả15trials trên5tasks là budget tối đa đề xuất cho bước này, không phải đã được thực thi.

Giữ LRselection củaE14 để tránhsearchrộng, nhưng báo rõ đây là screening và không tối ưuhyperparameters riêng cho mỗiarchitecture. Ghi gradient/matrixupdate từ các bước đầu, tránh lặp thiếu log ở các vòng trước. Không thêmMLM/NC/KD cùnglúc. Nếu có continuation ở một vòng khác, dùng architecture×adaptation controls với cùngdata/compute.

Chọncheckpoint bằngvalidation; test là bước xác nhận cấu hình đã khóa. Không tiếp tục đổi thiết kế sau khi mởtest. Chỉ có tín hiệu rộng mới chạy thêmseed; taskTopic vàSentiment dùng cùngvăn bản cần được xem như haioutcomes của mộtdomain, không hai bằng chứngdomain độc lập.

### Tiêu chí kết luận

Mục tiêu đầu tư vẫn là tăng trên ít nhất4/5task, ít nhất hai gain≥0,3điểm và khôngtask giảm quá0,3điểm, đồng thời fullmixing hơndiagonal đủ rõ. Đây là heuristic budget, không significance. Các chênh lệch tiny mộtseed không đủ nhậnnovelty hay lợi ích ổn định.

Nếu fullmixing tăng một task, chỉ có tín hiệu task-specific. Nếu tất cảmixing thất bại, dừng hướngheadcomposition trước khi chuyểndynamic, sparse hayMoE. Không lấy “có thể cần nhiều hơn4blocks” làm lý do mở thêm mộtgrid saukhi thấy kết quả.

## 11. Các gap phải giữ ở trạng thái chưa xác nhận

**Head redundancy của BamiBERT:** chưa đo causal head ablation trong protocolnày. Literature không thay thế phép đo tại chỗ. Ma trận tương đồng không đủ nói cần pruning hoặcmixing.

**Softmax khiến bỏ sót aspect:** plausible nhưng chưa được chỉ ra trên dữ liệu. Lỗi category binding, absentprior, ontology và annotation ambiguity có thể tạo hiện tượng tương tự. Không đặt tên bottleneck theo đúng block muốn thử.

**Thiếu pretraining khiến mọi nhánh thất bại:** chưa có architecture×adaptation study ởproject. Đây là một khả năng, không kết luận. AdaSplash cho biết continuation có thể lớn; không thể cam kết mộtstage ngắn sẽ đủ.

**Kiến trúc nhỏ là trần chất lượng:** E14 hơnstudent4/5, nhưng recipe/head/pretraining khác. E15 thất bại trênBamiBERT cũng không chứng minh capacity đã đủ cho mọitask. Chưa có phép so scaling kiểm soát toàn bộ confounds.

**Novelty tiếng Việt:** chưa tìm được một gap có thể tuyên bố chắc chắn chưaai giải quyết. Các nghiên cứu Việt như HuTieuBERT/ViWordFormer đã chiếm những ý tưởng đơn giản về boundary/morphology; các operatorheadmixing/sparse/SSM cũng đã có literature. Đóng góp khả thi phải nằm ở một cơ chế hoặc quy trình chuyển đổi được kiểm chứng vượt các prior controls, không chỉ ở đổi ngôn ngữ.

## 12. Độ tin cậy và phạm vi kết luận

E9–E16 chủ yếu là single-seed screens. Nghiên cứu về fine-tuning BERT đã chỉ ra ảnh hưởng của initialization, data order và số lần thử lên kết quả tốt nhất.[^19] Điều này không yêu cầu chạy nhiều seed cho mọicandidate; nó yêu cầu chỉ gọi negative screen là quyết định dừngbudget, và positive screen là tín hiệu cần xác nhận.

Test đã được mở nhiều lần trong lịch sửproject. Những lựa chọn tiếp theo đều có mức thích nghi với benchmark; cần ghi nhận điều đó trongpublication. Một candidate thắng nămtask cần được kiểm tra thêm NLI/NER hoặc holdout ngoài vòng phát triển trước khi nhận “encoder tiếng Việt tốt hơn” trên diện rộng.

Các nguồn2026 được dùng theo trạng thái công bố truy cập được. MaBERT vàAttentionEditing làpreprints trong lượt rà soát; không gọi họ là chuẩn đã được tái lập độc lập. AdaSplash-2 được thấy qua repositorytác giả, endpointOpenReview không đọc thành công nên không dùng các claimđịnh lượng của bản2 làm căn cứ xếp hạng.

**Kết quả cuối:** có các hướng operator replacement còn mở; static cross-head composition có sự kết hợp tốt nhất giữa bằng chứngencoder/NLU, khả năng identityconversion và relevance với mục tiêu chất lượng hiện tại. Nhưng nó là một phương pháp đã biết cần kiểm chứng tại chỗ. LượtDeepResearch này không phát hiện được một “bottleneck đã được chứng minh” hay một block mới bảo đảm thắngBamiBERT.

## Nguồn

[^1]: Dat Quoc Nguyen et al. [BamiBERT: A New BERT-based Language Model for Vietnamese](https://arxiv.org/html/2607.02259). 2026. Phạm vi benchmark; kết quả local ưu tiên [E14](E14_BAMIBERT_BASELINE_RESULT.md).
[^2]: Noam Shazeer, Zhenzhong Lan, Youlong Cheng, Nan Ding, Le Hou. [Talking-Heads Attention](https://arxiv.org/html/2003.02436). 2020, §7.3 và AppendixA. BERT/ALBERT/T5 và static/dynamic controls.
[^3]: Da Xiao, Qingye Meng, Shengping Li, Xingyuan Yuan. [Improving Transformers with Dynamically Composable Multi-Head Attention](https://proceedings.mlr.press/v235/xiao24d.html). ICML2024; [toàn văn](https://arxiv.org/html/2405.08553).
[^4]: Srinadh Bhojanapalli et al. [Low-Rank Bottleneck in Multi-head Attention Models](https://proceedings.mlr.press/v119/bhojanapalli20a/bhojanapalli20a.pdf). ICML2020. Động cơ biểu diễn, không phải auditBamiBERT.
[^5]: Paul Michel, Omer Levy, Graham Neubig. [Are Sixteen Heads Really Better than One?](https://proceedings.neurips.cc/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf). NeurIPS2019; [code tác giả](https://github.com/pmichel31415/are-16-heads-really-better-than-1).
[^6]: Elena Voita et al. [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://aclanthology.org/P19-1580/). ACL2019.
[^7]: [Theory, Analysis, and Best Practices for Sigmoid Self-Attention](https://arxiv.org/html/2409.04431). 2024/2025; [trang tác giả Apple](https://machinelearning.apple.com/research/sigmoid-self-attention). Lý thuyết, scaling, stability; không coi là BERTtransfer evidence.
[^8]: Nuno Gonçalves, Marcos Treviso, André F. T. Martins. [AdaSplash: Adaptive Sparse Flash Attention](https://arxiv.org/html/2502.12082). ICML2025, bảnv2; §4.2 và AppendixB.1.
[^9]: Marcos Treviso et al. [Predicting Attention Sparsity in Transformers](https://aclanthology.org/2022.spnlp-1.7/). SPNLP2022. Sparsefinder vàMLMencoder.
[^10]: DeepSPIN/SARDINE authors. [AdaSplash repository](https://github.com/deep-spin/adasplash). Bản1/2 và các implementation; truy cập12/09/2026. Không dùng claimđịnh lượng bản2 chưa đọcpaper.
[^11]: [Pretraining Without Attention](https://aclanthology.org/2023.findings-emnlp.5/). FindingsEMNLP2023. BiGS; [code](https://github.com/jxiw/BiGS).
[^12]: [Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture](https://arxiv.org/abs/2310.12109). NeurIPS2023; [code M2-BERT](https://github.com/HazyResearch/m2/blob/main/bert/README.md).
[^13]: Jinwoong Kim, Sangjin Park. [MaBERT: A Padding-Safe Interleaved Transformer–Mamba Hybrid Encoder for Efficient Extended-Context Masked Language Modeling](https://arxiv.org/html/2603.03001). Preprint2026, §4–6.
[^14]: Zhengyan Zhang et al. [MoEfication: Transformer Feed-forward Layers are Mixtures of Experts](https://aclanthology.org/2022.findings-acl.71/). FindingsACL2022.
[^15]: [Breaking ReLU Barrier: Generalized MoEfication for Dense Pretrained Models](https://aclanthology.org/2024.emnlp-main.563/). EMNLP2024.
[^16]: [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/html/2212.05055). ICLR2023. T5/ViT, continuation budget; đã đối chiếu trong [research chuyển weights](PRETRAINED_ENCODER_TRANSFORMATION_RESEARCH.md).
[^17]: Canwen Xu et al. [BERT-of-Theseus: Compressing BERT by Progressive Module Replacing](https://aclanthology.org/2020.emnlp-main.633/). EMNLP2020.
[^18]: Zhen Cheng et al. [Attention Editing: A Versatile Framework for Cross-Architecture Attention Conversion](https://arxiv.org/html/2604.05688). Preprint2026.
[^19]: Jesse Dodge et al. [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305). 2020.

Nguồn nội bộ: [E16 test](E16_FFN_TEST_RESULT.md), [audit20checkpoints](ARCHITECTURE_TRANSFER_FINAL_REVIEW.md), [review60mẫu](E12_SEMANTIC_REVIEW.md), [audit mapping](E12_LABEL_MAPPING_AUDIT.md), [E13A](E13A_RESULT.md). Không thay đổi các kết quả hoặc thí nghiệm lịch sử trong lượt rà soát này.
