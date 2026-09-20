# Root review: bộ chẩn đoán độc lập và quan hệ với E17

## Kết luận

Bộ phép đo bổ sung có giá trị: đưa ra một failure mode thực nghiệm cụ thể về độ bền của dự đoán aspect khi thêm mệnh đề khác, và tách được bỏ sót aspect khỏi đổi polarity. Nó có cơ sở local trực tiếp hơn động cơ hiện tại của E17. Tuy nhiên, các kết luận “token A vẫn đúng”, “CLS chỉ chứa được một ý”, “không attention nào sửa được” và “nguyên nhân đã được chứng minh là phân bố train” vượt quá bằng chứng.

Root đã đọc báo cáo, scripts và outputs, không rerun inference. Đặc biệt, các artifacts `fork_test` và `clause_readers` hiện có bổ sung bằng chứng chưa được phản ánh đầy đủ trong bản tóm tắt người dùng gửi. Không sửa nguồn hoặc kết quả của agent độc lập. Không thay đổi thí nghiệm E17 đang chạy.

## 1. Những kết quả được hỗ trợ

- Restaurant student: A_correct66% khi đứng riêng,39% khi ghép một B,29% khi ghép3B; missed tăng18,5→35→45,5%. BamiBERT cũng có suy giảm tương ứng72,5→51,5→36%.
- C1 chưa tạo cải thiện hữu ích trong inference screen đã chạy; một số settings còn làm kém hơn. Dừng đầu tư C1 ở protocol này là hợp lý. “Không cải thiện” chính xác hơn “không tác động/không khác baseline” vì có thay đổi dự đoán và mức giảm đo được.
- Student chịu interference trong cách tạo/đọc CLS của composite. Patch CLS ở các layer trung gian có rescue tăng dần là một quan sát về đường truyền đáng nghiên cứu tiếp.
- Mean/sum của CLS từ hai lần chạy riêng không phải một mã hợp thành mà head hiện tại đọc tốt.
- ViSFD student ít suy giảm ở A+B hơn Restaurant/Hotel trong cohort này. Đây là một tương phản hữu ích, chưa phải thí nghiệm kiểm soát nguyên nhân dữ liệu.

Nguồn local: `mechanism_battery.py`, `locus_patching.py`, `cls_superposition.py`, `cls_linear_probe.py` và các outputs cùng tên dưới `artifacts/independent_review/battery/`.

## 2. Những bước suy luận cần sửa

### CLS cuối được patch không định vị duy nhất nguyên nhân

`ClassificationHead.forward` chỉ lấy `hidden_states[:,0]`. Vì vậy thay CLS cuối bằng CLS(A) rồi đo lại nhãn A gần như tất yếu trả về dự đoán của A một mình. Đây là sanity check cho can thiệp, không bằng chứng đủ rằng các token khác đúng hoặc chỉ CLS là nguyên nhân.

Patch sớm/trung gian có giá trị hơn nhưng vẫn là can thiệp lớn, thay đồng thời nhiều thông tin. Bộ patch chỉ chấm recovery của A, chưa báo joint A-and-B preservation. Một can thiệp khôi phục A bằng cách xóa B không giải quyết đầy đủ ABSA.

### Patch token A không cứu không có nghĩa token A đúng

Các token được patch có thể không được CLS sử dụng, thiếu một thành phần cần thiết hoặc lại bị context B tác động ở layer sau. Script còn chủ ý bỏ token nội dung cuối của A để giữ prefix alignment. Kết quả âm của phép patch một phần không chứng minh thông tin nguyên vẹn tại mọi token/layer.

Bản chi tiết cũng có phản ví dụ với câu tóm tắt chung: BamiBERT Restaurant patch token A ở layer8 tăng A_correct36→52%. Do đó “token patch không cứu được ở cả student và teacher” không đúng với chính outputs.

### Superposition thất bại không phải giới hạn dung lượng một vector

Head có dense+bias+tanh+output+bias, không được học rằng mean hai CLS phải là hợp của hai tập aspect. CLS(A) còn chứa bằng chứng các aspect khác absent; trộn nó với CLS(B) có thể làm cả hai presence margins yếu đi. Đây là vấn đề của phép hợp thành/decoder đã thử, không một định lý rằng vector384/768chiều chỉ chứa được một ý.

ViSFD cùng dạng CLS/head xử lý composite tốt hơn cũng phản đối một giới hạn phổ quát về kiểu vector. Kết luận phù hợp là “các checkpoint chưa có tính hợp thành tuyến tính đủ tốt theo các phép trộn này”.

Norm scaling chỉ ít tác động trong khoảng đã thử. Bảng `norm_budget_test` vẫn có thay đổi nhỏ; tanh và bias khiến head không có tính argmax-invariant theo scale nói chung. Không gọi sự ổn định quan sát được là một định lý.

### Oracle partition không phải oracle tối ưu mọi attention

`partition_oracle.py` thử hai chính sách CLS attention cụ thể: chia khối lượng50/50 hoặc cộng hai distributions đã chuẩn hóa, với những phạm vi layer xác định. Biết ranh giới mệnh đề là oracle về partition, không phải oracle biết weights hoặc value transformation tối ưu.

Không cứu bằng hai chính sách đó chỉ loại hai interventions trên checkpoint đóng băng. Nó không loại learned routing khác, feature changes sau training, nhiều reader vectors hoặc E17.

**Artifact bổ sung trực tiếp:** `fork_test` chỉ cho hàng CLS đọc phầnA ở mọi layer, không patch CLS từ bên ngoài. Restaurant student A_correct39→65,5%, gần Aalone66%; đồng thời B_present56→6,5%. Như vậy đổi attention có thể rescueA, nhưng phải trả giá mấtB. Kết quả này gợi ý selective readers và nhu cầu giữ đồng thời các aspect, thay vì kết luận aggregation không thể sửa.

## 3. Những giới hạn thiết kế cần giữ rõ

`build_pairs` lấy các câu validation ngắn, một aspect được gán nhãn, polarity1/2; ghép hai câu khácaspect/ngượcpolarity. Có200pairs nhưng lấy lặp từ pool, ViSFD chỉ66eligible singles. Không coi200pairs là200observations độc lập. Các kết luận cross-domain còn chịu khác biệt cohort, ontology, sốaspect và chất lượng nhãn.

Can thiệp thêm B có hiệu ứng trên output của model đóng băng. Nhưng nó đồng thời đổi độ dài, nội dung và cấu trúc discourse; chưa có matched controls cùng-polarity/neutral/context-length để quy toàn bộ hiệu ứng riêng cho polarity đối lập. Gold single-aspect không tự bảo đảm văn bản không ngầm nói đến aspect khác. Một mẫu composites cần được review ngữ nghĩa độc lập trước khi gọi toàn bộ biến đổi là label-preserving.

Một B gây phần lớn suy giảm không chứng minh bội số hoàn toàn không liên quan: thêm B từ1→3 vẫn làm Restaurant student giảm39→29%. Điều được hỗ trợ là multiplicity correction C1 đã thử không giải quyết đủ vấn đề.

Các linear probes được fit trên train CLS dù backbone đóng băng; mô tả chính xác là “không fine-tune encoder”, không hoàn toàn không có học. Presence probe dùng class_weight balanced trong khi head gốc tối ưu slotCE; tăng presence recall có thể kèm spurious predictions. Trên Restaurant validation thật, linear probe pairF1 70,19 thấp hơn head71,28; không lấy presence recall riêng làm thắng benchmark.

Trong `fork_test`, fit probe bằng thêm synthetic composites tăng một số tỷ lệ recovery trên composite, nhưng Restaurant realvalF1 70,25→67,74; Hotel69,52→68,07. Các con số hỗ trợ tính phụ thuộc cách đọc/dữ liệu, đồng thời cảnh báo synthetic rescue chưa chắc chuyển thành gain thực.

Adjudication60errors một người đọc có ích để tách gold mơ hồ khỏi lỗi rõ, nhưng sample đã chọn có điều kiện trên lỗi thiểu số ViSFD. Không ngoại suy tỷ lệ55–60% thành trần cải thiện0,6–0,7điểm cho mọi task.

## 4. So với E17

Cùng quan tâm interference giữa các thông tin, nhưng không cùng giả thuyết. C1 giảm tác động multiplicity; E17 loại explicit recipient contribution khỏi preview key. Battery chưa can thiệp trực tiếp vào đường E17 nhắm tới. Patching và partition experiments trên model đóng băng cũng không tương đương với fine-tune một architecture có control.

Root đồng ý với phê bình quan trọng: E17 chưa có bằng chứng local cho thấy đường message quay về là một nguyên nhân lớn. Vì vậy chưa có cơ sở kỳ vọng nó chắc chắn cứu failure mode này. Giữ pilot đã chốt, không mở rộng grid/budget để bảo vệ ý tưởng.

Về novelty, cavity và C1 đều dựa trên nguyên lý đã có. Không có cơ sở chỉ từ tính chất partial-path/rank để nói cavity mới hơn C1. Phần đề xuất của E17 là operator chi tiết và factorization; mức đóng góp publication vẫn cần phản biện. Mặt khác, “thuộc một họ cũ” cũng không tự chứng minh toàn bộ thuật toán đã có; phải chỉ ra equivalence cụ thể.

Gate+.3pp một seed là quyết định đầu tư, không significance test. Đồng ý không gọi một screen âm là bác bỏ mọi kiến trúc. Nhưng so với ±SD test của một thí nghiệm khác cũng không phải kiểm định cho delta paired hiện tại: cần paired multi-seed hoặc uncertainty phù hợp khi chốt.

## 5. Quyết định tiếp theo

1. E17 tiếp tục nguyên protocol; không đổi source, loss,checkpoint selection hoặc gate giữa chừng.
2. Khi đọc E17, nên bổ sung decomposition từ saved validation predictions: missed/spurious/wrong-polarity, single/multi-aspect, mixed-polarity và minority. Đây là phân tích exploratory hậu nghiệm, không đổi primarymetric hoặc lựa lại checkpoint. Không cần inference mới cho phần này.
3. HướngA — readout riêng theo aspect — là phương án thực dụng đáng cân nhắc sau pilot. Nhưng phải đo joint aspect retention và pairF1, không chỉ rescueA. Không coi random query init hoặc CLS averaging của E9 là “bug đã chứng minh”; cần ablation để quy nguyên nhân. Hard exclusivity cũng chưa chắc phù hợp vì một cụm từ có thể hỗ trợ nhiều aspect.
4. Không cần tự đổi mục tiêu paper ngay. “Mechanistic diagnosis + targeted fix” có thể là đóng góp tốt nếu causal claims được thu hẹp và kiểm chứng; novelty của finding cũng cần literature audit riêng, không suy từ việc ARTS2020 chủ yếu đánh giá hành vi.

## Tài liệu phương pháp

- Heimersheim & Nanda, [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255),2024: nguồn tham khảo về diễn giải patching và lựa chọn metric; các phản biện code cụ thể ở trên do root đối chiếu trực tiếp.
- Xing et al., [Tasty Burgers, Soggy Fries / ARTS](https://aclanthology.org/2020.emnlp-main.292/),EMNLP2020: tiền lệ aspect robustness và biến đổi sentiment không-target; không chứng minh đã hoặc chưa có mọi causal analysis ABSA sau đó.

## Bổ sung sau vòng C4, mục13 của báo cáo độc lập

### Kết quả mới làm hướng reader đáng thử hơn, nhưng không củng cố toàn bộ chẩn đoán cũ

Mục13.1 xác nhận probe học trên composite đọc được nhiều thông tin A hơn từ CLS(A+B), và masked-CLS cứu A. Điều này yêu cầu sửa câu “CLS chỉ chứa được một ý / không attention nào cứu được” trong mục12 và bản tóm tắt. Kết luận phù hợp hơn là checkpoint/head hiện tại không sử dụng ổn định bằng chứng đa aspect trong những điều kiện đã thử. Không có cơ sở nâng các kết luận tuyệt đối cũ thành “mới, chắc”.

### Implementation hiện tại chưa kiểm chứng shared token stack

Trong `clause_readers.py`, `run_readers` tạo danh sách rows lặp lại cho mỗi reader rồi thực hiện `ids=enc['input_ids'][rows]`, `model(ids[subbatch], am[subbatch])`. Mỗi reader chạy toàn bộ encoder trên một bản sao của input. Batching các bản sao không biến chúng thành một token stack tính chung; chỉ weights được dùng chung.

`MaskedCLS` thay mask hàngCLS nhưng các content rows vẫn được đọc CLS. Vì CLS khác giữa các reader, content hidden states ở các layer sau cũng có thể khác. Do đó chuyển implementation này sang “K CLS tokens trong một forward với chung content states” là một thay đổi kiến trúc chưa được kiểm chứng tương đương. Cần chốt content→reader và reader→reader masks; thêmKtokens còn có thể đổi softmax denominators/positions. Không được lấy số65,5% của nhiều full forwards làm bằng chứng model shared-stack mới cũng sẽ đạt mức đó.

Chi phí: prototype chạy trung bình khoảng3encoder instances mỗi câu Restaurant/Hotel,5,6 ởViSFD, theo `mean_readers`. Đây là số lần tính encoder tương đối, không đo latency×3/5,6 vì batching vàGPU utilization khác nhau. “Không thêmparams” không đồng nghĩa không thêmcompute.

### Recall tăng cần đối chứng decision rule

`decode` chọn reader có1−P(absent) lớn nhất, rồi dùng argmax của reader đó với các ngưỡngpresence0,5/0,7. Nhiều cơ hội chọn một dự đoánpresent có thể nâng recall đồng thời tăngfalse positives. Chưa tách được lợi ích evidence tốt hơn khỏi việc operating point trở nên thiênpresent hơn.

Student Restaurant: P72,7→62,9/R70,0→73,0/F171,31→67,60 vớimaxpresent. Hotel: P77,1→63,2/R69,4→74,9/F173,05→68,58. Hiện đây là trade-off giảmprimarymetric, dù missed rate giảm rõ.

Đối chứng rẻ nhất trước khi đầu tưtraining lớn là baseline mộtCLS với absent-logit offset/threshold được chọn trên calibration split thuộctrain hoặc out-of-fold, rồi so ởmatchedrecall/matchedprecision vàpairF1. Dùngvalidation để vẽ PR trade-off thì ghiexploratory, không chọnthreshold rồi gọi đó là validation độc lập. Có thể thêmrandom partitions cùngsốreader/độdài để tách lợi ích linguisticpartition khỏi việc thêmviews; chỉ là đề xuất, chưa launch.

### Chưa biết vì sao precision giảm và training sẽ sửa được bao nhiêu

“Reader chưa được dạy absent cấpmệnhđề” là giảthuyết hợp lý, không nguyên nhân đã được côlập. Shiftphânphối, maxselection, cắt mất scope vàcontextleakage đều cóthể góp phần. Với sentence-level gold, không gánmọiclause cùng nhãnpositive của sentence. MIL/max-per-slot phải xácđịnh rõ loss áp vào bag và xửlý trườnghợp cùngaspect cópolaritytrái nhau. Gradient của hardmax thường đi qua winningreader; cần kiểmfalsepositives vàreaderusage thayvì mặcđịnhlossnàyđãgiảiquyếtvấnđề.

Mục13.3 nói khônghiệuchỉnhthresholdđượctrêntrain vìF1traincao cũng chưa loại calibration bằngheld-outtrain/out-of-fold. Không kếtluận “chỉcòntrainingreader” từfailure của một cáchcalibration.

### Novelty và quyết định

Đồngý không gọi C4 là phép tính hoàn toàn mới khi đã có maskedmulti-reader/pooledtoken literature. Metadata nguồn arXiv2011.09739 cần sửa tên: tác giả là **Ruifeng Yuan, Zili Wang, Wenjie Li**, không phải “Yuxiang Ruan”. [Nguồn chính thức](https://arxiv.org/abs/2011.09739) xácnhận BERT với hierarchicalgraphmask; lượt bổsung này mới kiểm abstract/metadata, khôngclaim đãaudit toànpaper.

Sáuvòng tìm ý tưởng không chứngminh literature đãphủ gầnmọiphéptổnghợp hoặc đây là bướcduynhất cònđượcbằngchứngủnghộ. Cóthể quyếtđịnhdừng một hướng vìreturn-on-effort, nhưngđó là quyếtđịnh nghiêncứu, không địnhlýkhôngcònkhả năngphátminh.

Root giữ đánhgiá: C4 là ứngviên thực dụng đángxem xét, chưa phải qualitywin hoặc shared-stackprototype đãpass. Trướctraining cần chốtimplementationmatch và baselinePRcontrol. E17 giữnguyênpilot; không tựmở C4training hoặcđổi mục tiêunovelty theo lời đề nghị của agent độclập.
