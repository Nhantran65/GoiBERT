# Chẩn đoán giới hạn của các reader và hướng học biểu diễn hợp thành

## Kết luận hiện tại

Chưa có ứng viên mới được thông qua. Bằng chứng hiện tại đủ để xác nhận một số checkpoint mất độ bền khi ghép thêm nội dung và các reader theo mệnh đề có trade-off precision–recall khác. Chưa đủ để kết luận thay đổi kiến trúc cụ thể nào sẽ khắc phục nguyên nhân, hoặc chỉ có một hướng còn khả thi.

Hai nhánh thường được đề xuất — contrastive alignment giữa câu và tổ hợp cụm từ; kiểm soát false positive của multiple-instance pooling — đã có prior art trực tiếp. Các nguyên lý này có thể làm baseline hoặc thành phần của một phương pháp khác, nhưng không tự tạo novelty khi áp vào tiếng Việt.

Phần thiếu có ảnh hưởng đến quyết định tiếp theo là so sánh single-reader và C4 ở cùng operating point, cùng checkpoint/precision. Nếu thiếu đối chứng này, việc giảm missed aspects dễ bị nhầm với một cải thiện representation.

## 1. Bằng chứng local mới cần đưa vào nhận định

Các kết quả `clause_readers_calibrated` đã thử chọn ngưỡng trên TRAIN, rồi đo validation. Chúng không cho một cải thiện rộng:

| Model/task | Baseline validation pairF1 | C4 calibrated pairF1 |
|---|---:|---:|
| Student Restaurant |71.35|71.30|
| Student Hotel |73.05|72.92|
| Student ViSFD |82.39|82.43|
| BamiBERT Restaurant |74.68|74.68|
| BamiBERT Hotel |73.31|73.13|
| BamiBERT ViSFD |84.44|84.41|

Các số thuộc protocol inference/calibration của chính bảng này, không trộn với E14/E8 gốc khác batching/attention path. Ngưỡng được chọn trên dữ liệu model đã fit rất cao, nên đây không phải out-of-fold calibration. Kết quả gầnbaseline không chứng minh mọi calibration đều bất khả thi.

C4 maxpresent trước calibration tăng recall nhưng giảm F1 rõ. Operator chọn một reader cho mỗi aspect từ nhiều distribution khác nhau; nó không chỉ thêm thông tin mà còn thay đổi cơ hội chọn dự đoán positive. Số reader trung bình khác giữa dataset cũng là confound khi so độ cải thiện hoặc suy giảm cross-domain.

Nguồn local: `artifacts/independent_review/battery/clause_readers_calibrated.txt`, `clause_readers.txt`, và `scripts/independent_review/clause_readers.py`.

## 2. Phép chẩn đoán đã chốt

Đặc tả triển khai: [READER_DIAGNOSTIC_SPEC.md](READER_DIAGNOSTIC_SPEC.md). Backbone đóng băng, không task training hoặc test mới. Thu per-reader logits trên validation thật của ba ABSA tasks, cả studentE8 và BamiBERT E14. Fullreader vàC4 phải dùng cùng numerical path.

Với một slot và reader được chọn, đặt:

\[
m=\max_{c\in\{1,2,3\}}z_c-z_0,
\quad c^*=\arg\max_{c\in\{1,2,3\}}z_c.
\]

Predict c* khi m>delta, còn lạiabsent. Với delta0, đây là argmaxdecode với tieabsent. C4 chọnreader bằng đúngmaxpresence của prototype trước khi quét delta; không âm thầm đổi chính sách chọnreader.

Tạo empirical PRcurve bằng cách thêm các slots theo margin giảm dần, nhóm các ties. Goldpositivecount cố định; TP chỉ tăng khi polarity đúng. So precision tốt nhất mà mỗi policy đạt tại recall của policy kia, vàghi cả oraclevalidationmaxF1.

Đây là **diagnostic envelope**, không phải một ngưỡng được phép chọn để báo benchmark mới. Nếu later cần ngưỡng deployable, phải dùng calibration tách biệt hoặc out-of-fold trong TRAIN; không dùng những điểm tối ưu trên validation như held-out estimates.

Đồng thời lưu các chuyển đổi full→C4: missed được cứu, newlyspurious, polarity bị làm sai, polarity được sửa. Các nhóm single/multi-aspect, mixedpolarity, minority vàreader-count cần có supportcounts. Root sẽ đọc ví dụ ở các nhóm này trước khi chọn một operator hoặc objective.

### Kết quả nào thay đổi hướng?

### Kết quả đầu tiên: student Restaurant

Report `artifacts/research_diagnostics/reader_operating_point/student_e8__absa_restaurant.json`, 771 validation rows, 1053 gold pairs. Root tính lại độc lập từ FP32 tensor cache và gold index, xác nhận oracle maxF1 của hai policy trùng report.

| So sánh | Baseline full | C4 maxpresent |
|---|---:|---:|
| F1 tại delta0 |71.346|67.604|
| Precision tại recall khoảng73% |69.495 (R73.124)|62.930 (R73.029)|
| Precision tại recall khoảng70% |72.754 (R69.991)|66.546 (R70.085)|
| Oracle validation maxF1 |71.730|68.831|

Các điểm recall gần nhau nhưng không chính xác bằng nhau do slot scores rời rạc. Oracle maxF1 chỉ dùng chẩn đoán, không phải kết quả benchmark mới hoặc ngưỡng cho test. Kết quả chưa chứng minh toàn bộ PR curve của baseline trội ở mọi điểm.

Ở delta0, missed177→122, wrong polarity139→162, spurious137→291. Như vậy giảm missed đi kèm thêm154 spurious và23 wrong-polarity, chỉ tăng32 true positives. Đây là bằng chứng trực tiếp chống việc coi giảm missed đơn lẻ là lợi ích representation đã xác nhận.

Quyết định: chưa ưu tiên training C4 dựa vào recall rescue. Hoàn thành các checkpoint/task còn lại và đọc chuyển đổi theo trường hợp để tìm phần thông tin C4 có mà baseline thiếu. Chưa loại mọi multi-reader architecture hoặc khả năng học lại reader; kết quả chỉ phủ frozen prototype và global-margin policies đã đo.

### Đọc lỗi và đối chứng chọn reader từ cache

Sáu report đã có, tất cả native delta0 khớp reference trong0.1pp. Oracle validation maxF1 full→C4: student Restaurant71.730→68.831; Hotel73.049→70.869; ViSFD82.497→80.677; BamiBERT Restaurant75.167→74.028; Hotel73.554→73.814; ViSFD84.627→84.082. Chỉ Bami Hotel có lợi thế nhỏ ở phép chọn ngưỡng oracle, chưa phải gain held-out. Root đã recompute cache Restaurant độc lập; các hàng còn lại hiện đọc từ report, cần audit cache trước khi dùng để chốt.

Root đọc sáu chuyển đổi đầu tiên theo thứ tự validation của mỗi nhóm missed-corrected, newly-spurious, correct-polarity-broken và wrong-polarity-fixed (24 slot transitions; không phải mẫu ngẫu nhiên để ước lượng tỷ lệ nguyên nhân). Có47 missed được sửa đúng,154 newly-spurious,37 correct-polarity-broken và22 wrong-polarity-fixed trên toàn tập; các nhóm này không đồng nhất với tổng net errors.

Ví dụ validation row36: “Món xôi nấm cũng không còn ngon như trước.” Reader chọn “ngon như trước”, chuyển dự đoán từ gold neutral sang positive. Gold neutral có thể còn cần adjudication, nhưng phần phủ định bị loại khỏi vùng CLS được đọc trực tiếp là sự thật của mask. Row69: phàn nàn “ko bao h đặt hàng mà giao đúng yêu cầu…” có reader chọn “giao đúng yêu cầu là sao vậy nè”, đổi negative đúng thành positive. Row22: câu về cafe kết thúc “nói chung là chán”, reader của phần này thêm RESTAURANT#GENERAL negative không có trong gold. Đây là dấu hiệu scope/referent có thể bị mất; chưa phải chứng minh causal cho toàn bộ154 spurious.

Ở chiều tốt, row61 “Quán ấm cúng, nhân viên dễ thương…” reader “Quán ấm cúng” cứu AMBIENCE#GENERAL positive; row72 reader “nhà hàng thì chật chội” cứu AMBIENCE negative. Một cơ chế hữu ích phải giữ được lợi ích tách đối tượng mà không tách mất phủ định hoặc đối tượng được nhắc trước đó.

Phân tích exploratory không inference mới, lưu `root_selection_policy_diagnostic.json`: chọn reader bằng margin lớp positive cao nhất thay max total-presence cho oracle validation F1=69.117; giữ margin presence của C4 nhưng lấy polarity từ full reader cho70.005; vẫn dưới full71.730. Hai đối chứng này là cách decode quen thuộc, không phải candidate mới. Chúng không bao phủ mọi cách kết hợp reader, nhưng chống giải thích rằng chỉ đổi tiêu chí argmax đơn giản là đủ.

Prior art liên quan: Joshi, Pan và He phân biệt đặc trưng thực sự không liên quan với đặc trưng có tác dụng phụ thuộc ngữ cảnh (ví dụ phủ định).[^10] Vì vậy không đặt objective bất biến với mọi phần ngoài reader: nó có thể buộc model bỏ qua modifier cần thiết. Giả thuyết tiếp theo phải phân biệt competitor không liên quan và context làm thay đổi nghĩa; chỉ thêm masking hoặc consistency không giải quyết định nghĩa đó.

- Nếu C4 không hơn single-reader trên phần PRenvelope liên quan: không ưu tiên một architecture đa reader chỉ dựa vào gainrecall. Cần tìm loại bằng chứng nào missing mà calibration không khôi phục được.
- Nếu C4 có PRadvantage nhưng threshold trênTRAIN khôngtransfer: nghiên cứu phân phối confidence/calibration và sự thay đổi theo reader-count.
- Nếu C4 cứu được các trường hợp riêng biệt nhưng tạo falsepositive ở các mệnh đề sai scope: nghiên cứu tính đầy đủ của ngữ cảnh và ràng buộc evidence, thay vì gộp views bằngmax đơn giản.

Không trường hợp nào tự xác nhận novelty. Chẩn đoán chọn câu hỏi; literature và phép rút gọn phải kiểm tra candidate tiếp theo.

## 3. Prior art trực tiếp và điều bị loại khỏi broad novelty

### 3.1. Embedding câu khớp với tổ hợp các thành phần

Chanchani và Huang đề xuất contrastive alignment giữa một văn bản với composition của các phrasal constituents. Có thí nghiệm sentence similarity trên các pretrained encoders; các biến thể không hiệu quả đồng nhất giữa BERT và RoBERTa. Đây là tiền lệ gần cho ý tưởng “continued learning buộc pooledcode có tính compositional”.[^1]

Vì vậy, chỉ thêm loss để f(A+B) gần mean/sum(f(A),f(B)) chưa phải một nguyên lý mới. Một đề xuất cần chỉ ra cấu trúc thiếu nào không được mục tiêu đó xử lý, phép tính khác ở đâu và control tương ứng. STS trong paper không chứng minh hiệu quả ABSA.

### 3.2. False positive khi max-pooling instances

*Avoiding False Positive in Multi-Instance Learning* đã phân tích hạn chế của việc chỉ quan tâm positive witnesses mà bỏ qua các nonwitnesses trong positive bags, và đề xuất projection constraints. Đây là tiền lệ cho mối lo “MIL chỉ học chọn một instance mạnh và bỏ qua các tín hiệu âm”.[^2]

*Certainty Pooling* dùng độ chắc chắn để xử lý khó khăn của pooling trong MIL.[^3] Các cách thêm confidence/uncertainty trướcmax vì vậy phải được đối chiếu thay vì gọi mới. Đây là priorart; chưa đánh giá phương pháp đó trên GoiBERT hoặc quy trực tiếp các kết luận domain khác vào tiếng Việt.

### 3.3. Sửa prediction bằng constraints rồi distill

Hu et al. đã xây teacher từ rule-regularized projection của neural predictions rồi distill trở lại student, có ví dụ quy tắc sentiment. Ý tưởng “teacher tổng hợp bằng ràng buộc thay vì một model lớn khác” không mới ở cấp khung.[^4]

Semantic Loss cũng đã nối logical constraints với trainingobjective.[^5] Việc dùng mục tiêu tập hợp/logic thay CE cần đối chiếu những dòng này; không đủ chỉ đổi tên loss thành “compositional” hoặc “set-structured”.

### 3.4. Trừ response nền và counterfactual pooling

Contextual calibration dùng content-free inputs để đo bias đã có tiền lệ.[^6] Counterfactual regularization cho attention-based MIL cũng đã có công trình mới, gồm CAR-MIL2026.[^7] Do đó “thêm nullreader rồi trừlogits” hoặc “thêm counterfactual branch” chưa đủ làm novelty. Trước khi đề xuất cần chứng minh null can thiệp đúng yếu tố nào và khác các phép debiasing/regularization sẵn có ra sao.

## 4. Một hướng hình thành giả thuyết, chưa phê duyệt candidate

Một câu hỏi cụ thể hơn “CLS chỉ chứa một ý” là **head hiện tại có vùng biểu diễn cho tổ hợp nhãn mong muốn hay không?** Head có dạng affine–tanh–affine. Trong không gian sau tanh v∈(−1,1)^d, các điều kiện một lớp thắng trong từng slot là các bất đẳng thức tuyến tính:

\[
(w_{a,y_a}-w_{a,c})^\top v +(b_{a,y_a}-b_{a,c})\geq t.
\]

Có thể dùng linearprogram để tìm commonmargin lớn nhất trong box v. Đây là phương pháp diagnostic quen thuộc, không được gọi là thuật toán mới. Nếu không có positivemargin cho một label combination ngay cả trong box, đó là bằng chứng mạnh về giới hạn decoder đã học đối với combination đó. Nếu có, chưa chứng minh encoder tạo được v: reachableCLS bị giới hạn bởi backbone và LayerNorm, box chỉ là outerbound.

Một hướng từng được đặt ra là target adaptation trong geometry của decoder thay vì cộng CLS một cách tùy ý: minimum-change projection của latent code vào intersection của class regions rồi dùng làm target học. Feasibility đã được đo ở phần dưới, nhưng projection và teacher-student constraint distillation đều có prior art ở trên. Chưa có bằng chứng utility hoặc audit đủ để gọi mục tiêu này là ý tưởng mới khả thi; chưa đưa sang training.

Prior art bổ sung: *Taming the Sigmoid Bottleneck* phân tích các tổ hợp nhãn không thể được dự đoán bởi output layer hạng thấp trong multi-label classification.[^9] Điều này xác nhận câu hỏi về vùng quyết định có tiền lệ trực tiếp. Tuy nhiên các ABSA head ở đây có số chiều hidden lớn hơn số đầu ra và dùng softmax theo slot; không được viện định lý low-rank sigmoid để khẳng định bottleneck tại đây. Nếu tiếp tục hướng feasibility, mục đích là bác bỏ hoặc xác nhận một giả thuyết checkpoint cụ thể, không tạo claim mới từ LP.

### Kết quả decoder: mọi tổ hợp đều có miền quyết định trong post-tanh box

Sáu checkpoint đã qua numerical certificate của `DECODER_FEASIBILITY_DIAGNOSTIC_SPEC.md`. Ma trận margin full row rank: Restaurant36, Hotel102, ViSFD30 (không gồm OTHERS). LP tìm được điểm mọi lớp hòa nhau sâu trong box, maxabs từ0.0045 đến0.0428. Right inverse tạo được target margin dương cho mọi tổ hợp bằng chứng đủ đã nêu trong spec.

| Checkpoint | Margin bảo đảm xấp xỉ |
|---|---:|
| Student Restaurant |0.1833|
| Student Hotel |0.0555|
| Student ViSFD |0.2364|
| BamiBERT Restaurant |0.2577|
| BamiBERT Hotel |0.0850|
| BamiBERT ViSFD |0.3414|

Root phát hiện report ban đầu dùng max-entry residual thay induced infinity norm; đã yêu cầu sửa. Root kiểm độc lập bằng cận bảo thủ ||E||inf<=m*maxentry(E), error=e_center+epsilon*m*e_inverse. Margin giữa hai lớp còn ít nhất epsilon-2*error; cả sáu vẫn dương, error chỉ khoảng1.7e-13–3.5e-13. Artifact: `artifacts/research_diagnostics/decoder_feasibility/root_conservative_certificate_audit.json`. Đây là numerical certificate với float64, không formal exact-arithmetic proof.

Diễn giải: output affine không loại trừ tổ hợp nhãn nào trong box latent sau tanh. Không còn cơ sở dùng failure của mean/sum CLS làm bằng chứng về incapacity của output layer. Kết quả không chứng minh encoder/LayerNorm hay affine trước tanh tạo được mọi latent được xây dựng, không giải thích toàn bộ lỗi dữ liệu và không tạo sẵn một phương pháp cải thiện quality.

Hướng diagnostic tiếp theo đã khóa: `SHADOW_READER_DIAGNOSTIC_SPEC.md` tách tác động giới hạn CLS read mask khỏi phản hồi làm đổi content states. Đây là đối chứng để định vị can thiệp tiếp theo, không gọi shadow reader là phát minh mới.

## 5. Điều kiện để chốt một ứng viên

### Phân tích tần suất tổ hợp trên validation thật

Root dùng E8 saved validation predictions và train labels từ frozen loader, không encoder forward hoặc test. Artifact `artifacts/research_diagnostics/composition_frequency/student_validation.json` đếm exact label-set frequency và pair frequency (pair có thể xuất hiện trong câu train nhiều hơn2aspect). Phân tầng theo số gold aspects để tránh nhầm ảnh hưởng cardinality với novelty của combination.

Ở câu validation có đúng2aspect: Restaurant missed rate của pair chưa xuất hiện=37.5% (4rows), pair1–5lần=31.0% (21rows), pair>5lần=16.0% (194rows). Hotel tương ứng70.0% (15rows),46.1% (38rows),19.7% (157rows). ViSFD không có pair<=5 trong261rows hai aspect, nên không có contrast tương đương; không suy ra ViSFD giải quyết được mọi unseen combination.

Control bổ sung giới hạn từng atomic label=(aspect,polarity) xuất hiện>=100lần trong train, vẫn giữ đúng2aspect. Restaurant pair<=5 có22rows/44goldslots, missed31.8%; pair>5 có167rows/334slots, missed15.3%. Hotel chỉ6rows trong rare group, missed41.7% so18.1% ở144frequent rows, chưa đủ cohort để diễn giải mạnh. Artifact `frequent_atoms_control.json`.

Đây là phân tích exploratory tương quan. Chưa khớp từng category/polarity, độ dài, lexical difficulty hay annotation ambiguity; cutoff100 và5 do root đặt để chẩn đoán, không pre-registered hypothesis test. Tín hiệu Restaurant chống giải thích đơn giản rằng mọi lỗi rare combinations chỉ vì atomic labels hiếm, nhưng chưa cô lập causal mechanism. Điều phù hợp để kiểm tiếp là transfer của bằng chứng giữa known atomic labels trong unseen/rare combinations, không kết luận phải có operator giữa stack.

Control trong cùng atomic label đã được tính thêm: giữ cardinality2 và atomic train frequency>=100; chỉ giữ label có ít nhất3validation slots trong cả rare-pair và frequent-pair groups. Restaurant còn6labels, mỗi rare group chỉ3–6slots. Bốn label có missed cao hơn trong rare pairs, hai label thấp hơn; chênh lệch trung bình đều trọng số theo label=+15.71pp. Hotel và ViSFD không đủ matched groups. Artifact `within_atomic_label_control.json`. Không lấy6nhóm nhỏ làm xác nhận mạnh hoặc sáu thí nghiệm độc lập; nhiều slots có thể thuộc cùng sentence. Kết quả duy trì hướng liên hệ, đồng thời cho thấy giới hạn power của validation hiện có.

Kết luận sau control: chưa đủ cơ sở để chọn một architecture chuyên chống rare-pair bias. Muốn đo causal transfer, cần can thiệp nội dung có label-preserving contract và đối chứng độ dài/polarity, thay vì tiếp tục chia nhỏ validation tới khi thấy gap lớn. Không đổi official splits hoặc tuyên bố test độc lập sau các phân tích này.

Root đã tự viết fixture `CONTROLLED_ASPECT_INTERVENTIONS.json` và giao execution spec cùng tên hậu tố `_SPEC.md`:6pairs trong ba domain, mỗi pair có standalone, full factorial polarity/order/connector và filler controls (168inputs toàn bộ). Chỉ annotate slots được nói rõ, không tự gán absent cho phần còn lại. Đây là dữ liệu diagnostic nhân tạo mới, không training/test corpus hoặc benchmark win. Nó kiểm tra một missing control của battery cũ: đổi polarity của mệnh đề khác trong khi giữ nguyên target, đồng thời kiểm model có hiểu polarity của chính target đứng riêng. Kết quả và giới hạn được trình bày bên dưới.

Kết quả đã có: mỗi checkpoint chạy56inputs trong domain, tổng extraction khoảng8.3giây. Student Restaurant standalone8/8, factorial joint23/32; BamiRestaurant8/8 và32/32. Student ViSFD8/8 và27/32; BamiViSFD8/8 và30/32. Hotel standalone student7/8, Bami6/8, joint đều14/32: không quy toàn bộ Hotel failures cho interference khi standalone đã sai. Không coi32variations là32templates độc lập.

Case Restaurant rõ nhất: 'Nhân viên phục vụ tận tình' được đọc positive khi đứng riêng; ghép 'Món ăn không ngon' khiến service thànhnegative, trong cả2thứ tự và2connectors, trong khi food vẫn đúngnegative. Cặp lexical cũng có5missedtargets, bao gồm cả hai aspect negative; do đó không quy mọi thất bại cho opposite polarity. StudentRestaurant filler controls không đổi prediction ở16trường hợp. Những kết luận này dựa trên raw rows, không chỉ summary.

Root phát hiện và yêu cầu sửa summary B-flip: khi B đổi gold1→2, code cũ vẫn so output mới với1 và gọi một số correct switches là lỗi. Execution agent đã recompute metrics từ savedoutputs; không rerun inference. Standalone/joint counts không bị lỗi này.

Đã chốt phép định vị tiếp theo trong `CONTROLLED_SCOPE_PATCHING_SPEC.md`: hai BAinputs giữ nguyên service prefix/positions, patch riêng service token states từ source(foodpositive) sang base(foodnegative), không patchCLS, chấm đồng thời servicepositive vàfoodnegative. Đây là kiểm đường truyền lỗi trong backbone trước khi chọn block; chưa có kết quả hoặc candidate được thông qua.

Patching đã hoàn tất: FP32 source/base tái hiện; replay maxlogiterror0, final-layer token patch không đổi logits. Không site0..9 nào cứu joint FOODnegative/SERVICEpositive ở hai connectors. Dot case tại block6 chuyển cả hai sang neutral, nên không xem service margin tăng là rescue. Kết quả không chứng minh service tokens sạch hoặc encoder không liên quan; chỉ loại việc thay toàn service-state group từ source như một can thiệp đủ trong hai case.

Vòng đề xuất tiếp theo xét dual-frame readout R=(AAᵀ+lambdaI)^(-1)AH. Audit `DUAL_FRAME_READOUT_NOVELTY_AUDIT.md` chỉ ra nó đúng là Intention với K=Aᵀ,V=H,Q=I. Loại vì trùng phép tính trước khi code; chưa có baseline trial mới.

Root phân rã margin service positive-minus-negative trên quartet của từng order/connector. Own-polarity effect là trung bình thay đổi margin khi flip SERVICE; cross effect là trung bình thay đổi khi flip FOOD. Student own6.16–8.10, cross6.50–8.03; Bami own13.54–14.05, cross0.79–1.06. Tỷ lệ cross/own của student khoảng0.81–1.30, Bami0.058–0.078. Đây là logit-scale descriptive analysis trong từng model; không so magnitude logit như metric calibrated xuyên model. Mixed finite difference của student5.02–10.17, Bami0.078–1.27: không phù hợp giải thích toàn bộ leakage bằng một offset cộng cố định không phụ thuộc polarity của service. Artifact `root_factorial_margin_decomposition.json`.

Prior art về phương pháp định vị: Zhao và Bethard kiểm hypothesis attention-to-negation-cue bằng downstream controls và consistency giữa datasets.[^21] Họ không chứng minh head có attention cao là causal source của lỗi. Ở đây không chọn head chỉ bằng attention heatmap; token-state patch và joint target metric là phép can thiệp cụ thể đang chờ kết quả.

Mục tiêu dài hạn vẫn cả5tasks. Phát hiện ABSA này không tự hỗ trợ claim thắng rộng, và VSFC không được coi là thắng nếu giữ nguyên full reader.

### Khả năng khai thác các reader cần được đo bằng một probe học nhỏ

Root kiểm hindsight availability trong cache: Restaurant full reader bỏ sót47gold-neutral slots, có16slot mà ít nhất một clause reader dự đoán đúngneutral theo argmax bốn lớp. Hotel tương ứng33và2. Với gold positive/negative, Restaurant130missed/32córeaderđúng; Hotel305/79. Đây là kiểm có dùng gold để nhận diện reader đúng, không phải routing deployable hoặc một benchmark score. Nó cũng không phải giới hạn của mọi combiner: thay calibration hoặc kết hợp logits có thể tạo prediction mà không reader đơn lẻ nào có.

Artifact `root_reader_label_availability.json`. Đã giao `READER_LEARNABILITY_DIAGNOSTIC_SPEC.md`: probe nhỏ trên full/clause/random views, encoder đóng băng, 3-fold cross-fit trên validation chỉ cho exploratory diagnosis. Vì model/checkpoint đã dùng validation trước đó, OOF này không được coi như held-out independent benchmark. Không thay mục tiêu novelty bằng một combiner quen thuộc; phép đo nhằm kiểm assumption “học được cách dùng các views thì có ích” trước khi thiết kế block mới.

Probe đã hoàn tất: Restaurant full-only/clause/random pairF1 xấp xỉ67.9/67.2/65.1; Hotel68.6/68.3/68.1. Root kiểm cách tạo random subset đúng số content tokens và fit StandardScaler trong training fold. Clause hơn random nhưng chưa hơn full-only; mọi probe đều dưới original E8 decoder. Đây là kết quả của logistic summary-features trên ít calibration data, không chứng minh mọi learned reader bất khả thi. Ngừng mở rộng probe/readout family này vì chưa có tín hiệu rõ. Ghi chú nghiên cứu hướng khác ở `COUNTERFACTUAL_GROWTH_RESEARCH_NOTE.md`, chưa xác nhận novelty hoặc triển khai growth.

### Vòng ý tưởng: truyền residual của reader nhưng bảo vệ các aspect khác

Đề xuất được xét rồi loại bằng đại số trước implementation: với post-tanh full code v0 và reader code va, đặt Da là các hàng margin của aspect a. Chiếu delta_a=va-v0 vào nullspace của D_-a (mọi aspect khác), Pa=I-D_-a^+D_-a, rồi cộng v'=v0+sum_a Pa delta_a. D_-a Pa=0 bảo đảm can thiệp dành cho a không đổi margin các aspect khác trong arithmetic lý tưởng.

Tuy nhiên Da v'+beta_a = Da v0+beta_a+Da Pa(va-v0). Toàn bộ operator ở tầng affine rút gọn thành readout riêng từng aspect với trọng số hiệu dụng Da Pa; ghép các residual không tạo tương tác mới giữa aspect. Nó cũng không biết reader nào đã bỏ mất phủ định, và v' có thể ra ngoài box tanh. Chiếu/clamp trở lại box phá bảo đảm nullspace. Vì vậy không chọn hướng này làm phát minh mới hoặc giải pháp scope; đây là một linear readout constraint có thể làm diagnostic, không cần đầu tư training.

### Prior art cụ thể cho các biến thể reader tiếp theo

CAN dùng attention riêng theo aspect và regularization sparse/orthogonal để hạn chế chồng lấn, đồng thời mở rộng sang task detection.[^11] Thêm exclusivity hay orthogonality cho C4 không đủ tạo nguyên lý mới; shared evidence giữa các aspect cũng không luôn là nhiễu.

AC-MIMLLN tách attention tìm key instances qua aspect detection khỏi bộ dự đoán sentiment trên contextual word states, rồi tổng hợp sentiment theo trọng số đó.[^12] Do đó “tìm aspect cục bộ nhưng đọc sentiment có ngữ cảnh” đã có tiền lệ trực tiếp. Mô hình paper dùng các nhánh LSTM/BiLSTM; đổi backbone thành BERT chưa đủ chứng minh novelty. Task ACSA trong paper cho trước các category được nhắc tới, nên không so trực tiếp accuracy đó với pairF1 end-to-end ở đây.

### Tiền lệ trực tiếp của shadow reader và học nhiều CLS

RPO dùng attention mask để prompt đọc token features mà không làm thay đổi chúng, và khởi tạo prompt từ special tokens.[^13] Đây là prior art trực tiếp cho nguyên lý read-only, không chỉ liên hệ rộng với pooling. Shadow diagnostic ở đây dùng CLS checkpoint đã fine-tune, tự thay K/V vị trí0 và clause masks; chưa chứng minh exact equivalence mọi chi tiết với RPO, nhưng các khác biệt đó không tự tạo novelty. Kết quả CLIP trong paper cũng không phải bằng chứng quality trên BERT tiếng Việt.

Multi-CLS BERT ghi nhận chỉ thêm nhiều CLS dễ dẫn đến các CLS giống nhau; họ dùng biến đổi riêng trong encoder và reparameterization khi fine-tune để tăng diversity, cùng continued pretraining.[^14] Ablation §3.3 có bỏ inserted layers, bỏ reparameterization và thay số CLS. BERT-base full-data GLUE trong Table1: MTL83.30 và K5/lambda0.1=83.47, nên không diễn giải mọi gain của hệ thống là lợi ích nhiều reader. Đây là comparator thích hợp nếu sau này đề xuất learned readers; chưa được chạy local.

Aspect-to-Scope Multi-view Contrastive Learning đã kết hợp scope theo aspect và contrastive learning để xử lý interference.[^15] Đọc kỹ §2.1–2.5 cho thấy task cho trước aspect span; scope có nhãn BIO từ Xu et al.2022, được học bằng CRF/NLL. Scope được định nghĩa là vùng liên tục nhỏ nhất chứa aspect và sentiment expression; graph contrastive kéo các node trong scope gần aspect và xem node ngoài scope là negatives. GoiBERT chỉ có category/polarity cấp câu, không có aspect spans hoặc BIO scope, nên đây không phải công thức có thể chép sang mà không bổ sung supervision. Tuy nhiên bỏ annotation hay thay CRF chưa tự tạo novelty: phải đối chiếu thêm weakly supervised rationalization và causal evidence learning. Broad claim “scope + contrastive” vẫn có tiền lệ.

Một lead mới cần audit methods trước khi dùng: Allchin2026 phân biệt attention imitation với selector được giám sát bằng causal evidence sets, có masking frozen teacher.[^16] Đây là công trình về pruning/retrieval, không bằng chứng encoder NLU; không suy kết quả reported sang ABSA. Nó đặt thêm đối chứng cho ý tưởng tự sinh supervision bằng can thiệp thay vì teacher attention.

### Không dùng subtraction làm mặc định cho scope

Một biến thể đang được cân nhắc ở mức giả thuyết là trừ representation của context không có target khỏi representation đầy đủ, nhằm giữ phần phụ thuộc target. Chưa triển khai: nếu bỏ target làm thay đổi tham chiếu hoặc cấu trúc phủ định, chênh lệch không chỉ là “thông tin target”; nếu target implicit thì không có span đúng để loại. Vì vậy không được gọi phép trừ là causal effect nếu thiếu mô hình can thiệp và assumptions rõ.

DINER đã có tiền lệ cụ thể về debiasing ABSA bằng kết hợp review/context adjustment và counterfactual aspect effect.[^17] Đọc introduction cho thấy paper còn dùng review-only/aspect-only probes để phân biệt các loại bias. Điều đó phản đối một broad novelty claim kiểu “thêm nhánh counterfactual để trừ nhiễu aspect”; chưa chứng minh exact equivalence với mọi subtraction ở hidden states. Chưa chọn biến thể này làm candidate hoặc mở phép đo mới.

### Trạng thái implementation shadow diagnostic

Lần chạy đầu dừng ở lỗi device indexing trước khi có metric; không có kết luận quality từ lần này. Root yêu cầu sửa trước relaunch: attention cùng device với index, snapshot K/V trước can thiệp, so argmax từng slot, EOS theo valid length, FP32 parity gate trước BF16 smoke, và kiểm provenance cache. Không nới gate để có kết quả nhanh. Các sửa này đang được áp dụng bởi execution agent; chỉ đọc utility sau khi parity được xác minh.

Smoke report07:12:40UTC đã được root review: 16 Restaurant rows, FP32 max layer error2.861e-6, logit error2.384e-6, không slot argmax mismatch; K/V invariance pass. BF16 có logit max error0.03125 do shape arithmetic, không đổi slot predictions trên smoke. Root cho tiếp tục extraction R/H dùng full-shadow làm baseline cùng numerical path; chưa có utility result. Đây là xác minh implementation, không bằng chứng cho cơ chế.

Extraction đã hoàn tất. Restaurant: full-shadow F1=71.346, original C4=67.604, shadow C4=67.516; oracle validation maxF1 tương ứng71.764/68.831/68.807. Hotel:73.049/68.577/68.318; oracle73.049/70.869/70.747. Root đọc report và tự decode cache cho hai câu mục tiêu: row36 FOOD#QUALITY vẫn neutral→positive sai, row69 SERVICE#GENERAL vẫn negative→positive sai với cùng reader được chọn. Artifact root: `artifacts/research_diagnostics/shadow_reader/root_readout.json`.

Trên toàn Restaurant có1 slot argmax mismatch giữa full-shadow và full pass BF16 dù aggregate F1 trùng; không coi chúng bit-identical. Các delta rất nhỏ shadow-vs-original không được dùng làm bằng chứng suy giảm có ý nghĩa thống kê. Điều đủ vững cho quyết định là khoảng cách lớn với matched full baseline và không cứu hai lỗi scope cụ thể.

Quyết định dừng diagnostic shared-stack: không có tín hiệu đầu tư cho giả thuyết “chỉ loại CLS→content feedback sẽ sửa C4”. Điều này không loại trained RPO hoặc mọi learned reader. Không mở rộng GPU hoặc training nhánh này. Cần tìm một cách chọn/huấn luyện evidence khác có đóng góp riêng, thay vì tiếp tục biến thể phản hồi trên cùng hard clause masks.

### Weak supervision không tự tạo khoảng trống novelty

Lei et al. đã học rationales từ nhãn dự đoán mà không cần nhãn span.[^18] Yu et al. nêu rõ selector và predictor có thể dùng các đoạn không đủ ngữ cảnh, thậm chí mã hóa label bằng lựa chọn từ/vị trí; họ thêm complement predictor đối kháng để kiểm soát thông tin bị bỏ lại.[^19] Vì vậy “scope không cần BIO” và “thêm nhánh complement” đều có prior art trực tiếp.

Điểm cần thận trọng về lý luận: sufficiency đối với predictor không đồng nghĩa đầy đủ ngữ nghĩa theo ground truth. Nếu model gốc dự đoán sai, bảo toàn dự đoán của nó có thể bảo toàn shortcut. Đây là giới hạn của cách dùng tiêu chí, không phải claim rằng paper bỏ qua mọi kiểu degeneration. Một candidate mới phải chỉ rõ supervision nào phân biệt context modifier cần giữ với aspect competitor cần loại; chưa có thiết kế được thông qua cho yêu cầu này.

Một nhánh tiếp theo được xét là ghép các câu có nhãn rồi can thiệp/swap các biểu diễn aspect, yêu cầu output thay đổi theo phép ghép nhãn đúng thay vì bắt chước prediction cũ. IIT đã có khung trực tiếp: align biến của mô hình nhân quả với neural representation, swap states giữa base/source, rồi train theo counterfactual output của mô hình nhân quả.[^20] Do đó “hidden-state swap + compositional gold” chưa phải một thuật toán mới. Với ABSA, còn phải xác minh phép ghép bảo toàn nhãn và xử lý conflict trong cùng category; không mặc định union nhãn luôn hợp lệ. Nhánh này chưa được đưa sang implementation.

### Bổ sung: tiền lệ trực tiếp về tổ hợp nhãn hiếm

Chai et al. nghiên cứu compositional generalization trong multi-label text classification trên ba benchmark. Họ quan sát khó khăn với tổ hợp cảm xúc khác polarity hiếm trong train, và thử augmentation có sinh văn bản theo tổ hợp nhãn, disentangled label representations, cùng bước lọc chất lượng.[^8] Đây là tiền lệ sát hơn STS cho giả thuyết hiện tại. Tuy nhiên paper không tự chứng minh cơ chế CLS trên checkpoint tiếng Việt: label-composition shift và activation interventions là hai mức bằng chứng khác nhau.

Hệ quả: không định vị đóng góp mới đơn thuần là phát hiện “model học tương quan nhãn và hụt tổ hợp trái chiều”, hoặc dùng augmentation để sửa. Cần xác định phần can thiệp, phép đo hoặc thuật toán cụ thể vượt khỏi khung đã có. Không thay official splits để tạo một benchmark mới rồi gọi đó là thắng trên benchmark gốc.

### Kiểm soát tính đúng của phép đo PR

Review mã phát hiện `cstar` ban đầu lấy từ dự đoán đã threshold ở delta=0. Điều này làm cstar thành absent khi margin âm, nên đường PR không thể ghi nhận true positive khi hạ ngưỡng. Định nghĩa đúng phải luôn lấy argmax trong các lớp positive, độc lập với quyết định absent. Agent thực thi đã được yêu cầu sửa và kiểm fixture CPU có positive gold nhưng negative margin. Chưa dùng kết quả PR trước sửa để quyết định hướng.

Cache phải giữ logits/probabilities FP32; chuyển half có thể đổi ties và reader selection. Lỗi tên biến khi lưu cache cũng đang được sửa. Các lỗi triển khai này không phải kết quả âm của C4.

Một ứng viên cần đồng thời có:

1. Failuremode và comparator cho thấy vấn đề không chỉ là đổi threshold hoặcnhãn mơhồ.
2. Forward/objective cụ thể, không chỉ một tên cơ chế; chỉ rõ assumptions vàtrườnghợp thấtbại.
3. Claimchart với priorart gần và phép rút gọn, phân biệt exactequivalence với cùnghọ rộng.
4. Init/adaptation và compute khảthi từcheckpoint đãcó, cócontrol cùngdata/budget.
5. Phép kiểm nhỏ có thông tin về lợi ích cơ chế; không chỉ tensorparity/gradientnonzero nhưE17.

Hiện các điều kiện này chưa được đáp ứng đầy đủ cho một candidate mới. Mục tiêu nghiên cứu vẫn đang mở; không chốt train C4 hoặc một objective đã biết để thay thế yêu cầu novelty.

## Nguồn

[^1]: Sachin Chanchani, Ruihong Huang. *Composition-contrastive Learning for Sentence Embeddings*. ACL2023. [Paper](https://aclanthology.org/2023.acl-long.882.pdf). Đã đọc nguồn và phần kết quả/ablation; không reproduce.
[^2]: Yanjun Han, Qing Tao, Jue Wang. *Avoiding False Positive in Multi-Instance Learning*. NeurIPS2010. [Paper](https://papers.neurips.cc/paper_files/paper/2010/file/ccb0989662211f61edae2e26d58ea92f-Paper.pdf), §3.1–3.2 về nonwitnesses vàprojectionconstraints.
[^3]: *Certainty Pooling for Multiple Instance Learning*. 2020preprint. [Nguồn](https://arxiv.org/abs/2008.10548). Lượt hiện tại đọc abstract; PDFtruycập thấtbại, chưa kết luận exactmethodmatching.
[^4]: Zhiting Hu et al. *Harnessing Deep Neural Networks with Logic Rules*. ACL2016. [Paper](https://aclanthology.org/P16-1228.pdf). Tiền lệ rule-projection teacher vàdistillation, chưa phân tích toàn bộ derivation trong lượt này.
[^5]: Jingyi Xu et al. *A Semantic Loss Function for Deep Learning with Symbolic Knowledge*. ICML2018. [Nguồn](https://proceedings.mlr.press/v80/xu18h.html). Broadconstraintobjectivepriorart, chưa exactmatching candidate.
[^6]: Tony Z. Zhao et al. *Calibrate Before Use: Improving Few-Shot Performance of Language Models*. 2021. [Nguồn](https://arxiv.org/abs/2102.09690). Content-free calibration, không phải bằng chứng ABSAquality.
[^7]: Imane Chraki et al. *CAR-MIL: Counterfactual Attention Regularization for Multiple Instance Learning*. arXiv2026. [Nguồn](https://arxiv.org/abs/2609.08419). Abstract-level lead, cầnđọc methods trước exactequivalenceclaim.
[^8]: Yuyang Chai, Zhuang Li, Jiahui Liu, Lei Chen, Fei Li, Donghong Ji, Chong Teng. *Compositional Generalization for Multi-label Text Classification: A Data-Augmentation Approach*. 2023 preprint. [Nguồn](https://arxiv.org/html/2312.11276v2), phần problem setting, discussion và compositional data augmentation. Đã đọc các phần này; chưa reproduce kết quả hoặc xác nhận toàn bộ ablation.
[^9]: Andreas Grivas, Antonio Vergari, Adam Lopez. *Taming the Sigmoid Bottleneck: Provably Argmaxable Sparse Multi-Label Classification*. AAAI2024. [Nguồn chính thức](https://ojs.aaai.org/index.php/AAAI/article/view/29110). Đã đọc abstract và đoạn mở đầu từ PDF chính thức; cần đọc proof trước khi áp dụng một định lý cụ thể.
[^10]: Nitish Joshi, Xiang Pan, He He. *Are All Spurious Features in Natural Language Alike? An Analysis through a Causal Lens*. EMNLP2022. [Nguồn](https://arxiv.org/abs/2210.14011). Đã đọc abstract; dùng để định vị nguyên lý phân biệt context-dependent features, chưa áp dụng estimator hoặc kết quả thực nghiệm của paper.
[^11]: Mengting Hu et al. *CAN: Constrained Attention Networks for Multi-Aspect Sentiment Analysis*. EMNLP-IJCNLP2019. [Paper](https://aclanthology.org/D19-1467.pdf). Đọc introduction, kiến trúc và problem formulation; chưa audit ablation đầy đủ.
[^12]: Yuncong Li, Cunxiang Yin, Sheng-hua Zhong, Xu Pan. *Multi-Instance Multi-Label Learning Networks for Aspect-Category Sentiment Analysis*. EMNLP2020. [Paper](https://aclanthology.org/2020.emnlp-main.287.pdf), §3.1–3.2. Đã đối chiếu formulation và hai nhánh detection/sentiment; chưa reproduce kết quả.
[^13]: Dongjun Lee et al. *Read-only Prompt Optimization for Vision-Language Few-shot Learning*. ICCV2023. [Nguồn](https://arxiv.org/abs/2308.14960). Đọc abstract và primary PDF excerpts về Figure2/initialization; full PDF fetch không ổn định.
[^14]: Haw-Shiuan Chang, Ruei-Yao Sun, Kathryn Ricci, Andrew McCallum. *Multi-CLS BERT: An Efficient Alternative to Traditional Ensembling*. ACL2023. [Paper](https://aclanthology.org/2023.acl-long.48.pdf), §2.5–2.6, §3.1–3.3 và Tables1–2; đã đọc các phần này, chưa reproduce.
[^15]: Heyan Chai et al. *Aspect-to-Scope Oriented Multi-view Contrastive Learning for Aspect-based Sentiment Analysis*. Findings EMNLP2023. [Paper](https://aclanthology.org/2023.findings-emnlp.727.pdf). Đã đọc §2.1–2.5 về supervision, CRF, graph và contrastive; chưa audit toàn bộ ablation.
[^16]: James E. Allchin. *Learning What Matters: Supervising Global Context Pruning with Causal Evidence Sets*. arXiv2607.21692v4, 28Aug2026. [Nguồn](https://arxiv.org/abs/2607.21692). Chỉ đọc abstract/version metadata, chưa xác minh kết quả hay methods. Search index còn dùng title cũ “Sparse Attention Routing”; ở đây dùng title v4.
[^17]: Jialong Wu, Linhai Zhang, Deyu Zhou, Guoqiang Xu. *DINER: Debiasing Aspect-based Sentiment Analysis with Multi-variable Causal Inference*. Findings ACL2024. [Paper](https://aclanthology.org/2024.findings-acl.208.pdf). Đọc abstract/introduction và đầu formulation; chưa audit estimator và ablation đầy đủ.
[^18]: Tao Lei, Regina Barzilay, Tommi Jaakkola. *Rationalizing Neural Predictions*. EMNLP2016. [Nguồn](https://aclanthology.org/D16-1011/). Đọc metadata và abstract/PDF excerpt; chưa audit toàn bộ methods.
[^19]: Mo Yu, Shiyu Chang, Yang Zhang, Tommi S. Jaakkola. *Rethinking Cooperative Rationalization: Introspective Extraction and Complement Control*. EMNLP-IJCNLP2019. [Paper](https://aclanthology.org/D19-1420.pdf), §1–2.3. Đọc setup, sufficiency/comprehensiveness và degeneration example; chưa reproduce hoặc audit toàn bộ equilibrium proof.
[^20]: Atticus Geiger et al. *Inducing Causal Structure for Interpretable Neural Networks*. ICML2022. [Nguồn](https://proceedings.mlr.press/v162/geiger22a.html). Đọc abstract chính thức, có NLI/reasoning evaluations; chưa audit methods và các control chi tiết.
[^21]: Yiyun Zhao, Steven Bethard. *How does BERT’s attention change when you fine-tune? An analysis methodology and a case study in negation scope*. ACL2020. [Nguồn](https://aclanthology.org/2020.acl-main.429/). Đọc abstract và primary PDF excerpts về methodology §3–4; chưa reproduce.
