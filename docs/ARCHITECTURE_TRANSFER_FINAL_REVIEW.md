# Review hoàn tất E13/E15 và lựa chọn thí nghiệm kế tiếp

## Quyết định

**Dừng hướng centered pair-value residual E13/E15. Không mở một vòng chỉ tăng scale, đổi LR hoặc fusion precision để cứu cơ chế này.** Các phép đo mới cho thấy nhánh đã được cập nhật, còn gradient tại checkpoint, nhưng ảnh hưởng trực tiếp đến quyết định cuối nhìn chung nhỏ. Riêng cải thiện phép cộng BF16 không tạo lợi ích F1 có hướng nhất quán.

Candidate kế tiếp được chọn là **identity-gated FFN dùng lại feature và projection đã pretrained**, so sánh hai nguồn conditioning: attention update và input FFN. Khởi tạo giữ hàm cũ; gate nhận gradient trực tiếp qua W2/z pretrained. Không thêm MLM hoặc distillation trong pilot đầu. Đây là một giả thuyết kiến trúc mới cho project, chưa chứng minh novelty toàn cầu hoặc chất lượng tốt hơn BamiBERT.

Không tuyên bố audit đã tìm được nguyên nhân nhân quả duy nhất của E15. Kết quả đủ để loại bỏ hai cách giải thích quá đơn giản, đánh giá lại độ mạnh của đối chứng và thiết kế một phép thử khác có hợp đồng rõ ràng.

## 1. Những việc đã hoàn thành

| Hạng mục | Thực hiện | Trạng thái |
|---|---|---|
| Đóng góp nhánh E13/E15 | 20 best checkpoints × native/branch-off/FP32-fusion trên full validation | 60 lượt đã hoàn tất |
| Độ chính xác số | Dtype thực, intended/effective delta, tỷ lệ tọa độ bị làm tròn, norm trước/sau LayerNorm | 128 validation rows đầu mỗi checkpoint |
| Gradient và weights | CE backward trên batch16 train cố định, dropoutoff; norm U/W/P, Wo/FFN và drift | 20/20 checkpoint |
| Tính đúng số liệu | Root tính lại macro/pair F1 từ predictions; so IDs/gold với bản gốc | 60/60 khớp |
| Bảo toàn checkpoint | SHA trước/sau, kiểm tra P khác0 và các gradientgroup đầy đủ | 20/20 đạt |
| Literature | Encoder/NLU, growth/replacement, identity init, adaptation, gate và prior art | Báo cáo 19 nguồn |
| Hướng tiếp theo | Cơ chế, init, precision, đối chứng, budget và tiêu chí dừng | Đã chốt đề xuất; chưa train |

Số liệu mới được tạo bởi implementation theo [đặc tả đã chốt](ARCHITECTURE_TRANSFER_MEASUREMENT_SPEC.md). Việc đọc bằng chứng và lựa chọn hướng thuộc review này. Không chạy training, optimizer.step hoặc test mới. Gradient probe không cập nhật model.

## 2. Tắt nhánh thực sự thay đổi bao nhiêu?

Mỗi mode dùng cùng weights, cùng validation rows, cùng microbatch16, BF16/SDPA. `branch_off` đặt delta về0 tại fusion; `fp32_fusion` chỉ nâng phép cộng Wo+delta sangFP32 trước residual/LayerNorm. Native vẫn tính như source gốc.

Bảng dưới là **chênh lệch F1 tính bằng điểm phần trăm so với native trong audit**, không so trực tiếp với baseline E8/E14. Số âm ở cột tắt nhánh nghĩa là nhánh đang có ích tại checkpoint đó.

| Model | Task | Tắt nhánh − native | FP32 fusion − native |
|---|---|---:|---:|
| E13 centered | Sentiment | 0,000 | 0,000 |
| E13 centered | Topic | 0,000 | −0,065 |
| E13 centered | ViSFD | −0,123 | −0,012 |
| E13 centered | Restaurant | 0,000 | +0,035 |
| E13 centered | Hotel | −0,051 | 0,000 |
| E13 uncentered | Sentiment | 0,000 | 0,000 |
| E13 uncentered | Topic | 0,000 | −0,065 |
| E13 uncentered | ViSFD | +0,012 | 0,000 |
| E13 uncentered | Restaurant | −0,034 | 0,000 |
| E13 uncentered | Hotel | +0,051 | +0,029 |
| E15 centered | Sentiment | 0,000 | 0,000 |
| E15 centered | Topic | 0,000 | 0,000 |
| E15 centered | ViSFD | +0,025 | −0,017 |
| E15 centered | Restaurant | 0,000 | 0,000 |
| E15 centered | Hotel | 0,000 | 0,000 |
| E15 uncentered | Sentiment | 0,000 | 0,000 |
| E15 uncentered | Topic | 0,000 | 0,000 |
| E15 uncentered | ViSFD | +0,012 | 0,000 |
| E15 uncentered | Restaurant | 0,000 | 0,000 |
| E15 uncentered | Hotel | +0,056 | +0,085 |

Tắt nhánh không đổi F1 ở12/20 checkpoint; biên độ lớn nhất là0,123điểm. Riêng E15centered, không task nào giảm F1 khi tắt nhánh trong phép đo này. Tuy nhiên, F1 bằng nhau không bảo đảm predictions giống nhau: E13centered Restaurant đổi2câu, E15centered Hotel đổi1câu nhưng tổngF1 không đổi.

Các weights còn lại đã co-adapt cùng nhánh trong training. Vì vậy, kết quả không chứng minh nhánh không ảnh hưởng quá trình học trước đó. Nó chỉ cho thấy không có lợi ích trực tiếp lớn ở output checkpoint được chọn. Chênh lệch E13/E15 với baseline cũ có thể đến từ trajectory của cả model; không được gán toàn bộ cho việc nhánh đang hoạt động tại inference.

## 3. BF16 có làm mất tín hiệu không?

Có, ở phép cộng được đo. Dtype thực của Wo, delta và Wo+delta làBF16; hidden residual và inputLayerNorm làFP32. Tỷ lệ tọa độ delta khác0 mà phép cộng không làm đổi Wo dao động từ **5,71% đến83,52%**, tùy checkpoint. Đây không phải tỷ lệ “thông tin bị mất” hoặc xác suất dự đoán sai.

Với E15centered:

| Task | Tọa độ delta khác0 bị làm tròn (%) | Norm delta / norm Wo (%) |
|---|---:|---:|
| Sentiment | 72,55 | 0,110 |
| Topic | 83,52 | 0,067 |
| ViSFD | 50,61 | 0,266 |
| Restaurant | 59,75 | 0,181 |
| Hotel | 53,82 | 0,234 |

Các số dùng128rows đầu, chỉ non-padding queries. Không thay thế một phân bố độc lập trên toàn dataset. Mẫu số Wo khác mẫu số hidden trong metadata E15 trước đây; không so chúng như cùng một metric.

Chuyển riêng fusion sangFP32 không đổi F1 ở13/20 checkpoint. Các thay đổi nằm trong khoảng **−0,065 đến+0,085điểm**, không tạo gain rộng. Do đó dữ liệu không ủng hộ lời giải thích “chỉ sửa phép cộng là cứu được E15”.

Phép đo này không kiểm tra một model được train bằngFP32, cũng không nâng toàn bộ U/W/GELU/P sangFP32. Không thể từ đó loại mọi tác động precision trong quá trình học. Nhưng chưa có lý do thực nghiệm để đầu tư retrain E15 chỉ dựa vào nghi ngờ fusion.

## 4. Nhánh có học và có gradient không?

P kháczero ở cả20 checkpoint, norm nằm trong khoảng0,0624–0,7889. Các group U/W/P, Wo vàFFN đều có gradient hữu hạn, khác0 trong phép CE probe trên batch16 train cố định, evalmode/FP32. Không có optimizerstep trong audit.

Theo initializer được tái dựng từ seed42 và constructor, displacement U khoảng2,63%–35,25%, W khoảng2,52%–32,38%. Các checkpoint không lưu init snapshot riêng, nên đây là **ước lượng có điều kiện theo reconstruction**, không một audit lịch sử bitwise hoàn chỉnh. Source-encoder displacement dùng trọng số pretrained thật làm reference riêng.

Kết luận được phép: không thấy nhánh chết gradient tại checkpoint; nhánh P đã nhận update trong training; biểu diễn U/W tái dựng cho thấy có thay đổi đáng kể ở nhiều trường hợp. Không có bằng chứng đủ để khẳng định optimizer không cho nhánh học.

Kết luận không được phép: learning rate là tối ưu, không từng có gradient starvation ở giai đoạn đầu, hoặc các gradient hiện tại phản ánh trajectory suốt30epochs. Gradient lịch sử không được lưu đầy đủ. Raw gradient norm giữa các group có số chiều và scale khác nhau cũng không tự đo mức độ học “hợp lý”.

Ở bước0, Pzero khiến gradient U/W bằng0 về toán học; preflight cũ kiểm tra gradient U/W xuất hiện sau khi P được cập nhật. Đây là delay một giai đoạn khởi tạo, không phải bằng chứng về sự tắc nghẽn kéo dài. Review này rút lại việc dùng zero-init delay như lời giải thích ưu tiên cho thất bại khi chưa có trajectory.

## 5. Các đối chứng có tương đương hoàn toàn không?

Không hoàn toàn. Backbone, head, task loss và họ optimizer khớp theo từng phép so E13–E8 hoặc E15–E14; seed và initialization được thiết kế bảo toàn. Nhưng một số lượt dùng microbatch khác nhau:

| Task | E14 selected | E15 centered | E15 uncentered |
|---|---:|---:|---:|
| Sentiment | 32 | 32 | 32 |
| Topic | 16 | 32 | 16 |
| ViSFD | 16 | 32 | 32 |
| Restaurant | 32 | 16 | 32 |
| Hotel | 32 | 32 | 32 |

Effective batch đều32. Gradient accumulation trong source có trọng số theo số mẫu mỗi window, nhưng microbatch khác vẫn có thể đổi dropout/RNG consumption và kernel arithmetic. Không coi các khác biệt nhỏ là kiến trúc-only. Đồng thời, Sentiment vàHotel vẫn giảm ở các lượt microbatch khớp, nên microbatch không phải lời giải thích đủ cho toàn bộ failure.

Trong audit, native fixedmicro16 so với metric gốc dao động khoảng **−0,0565 đến+0,0437điểm**; E13 cả10lượt khớp F1 gốc. Các chênh lệch này được giữ riêng, không sửa bảng benchmark lịch sử. Tất cả phép so bật/tắt/fusion mới đều dùng cùngmicro16, nên tránh trộn hai điều kiện trong một delta.

Việc chuyển E13 lênE15 còn đổi hidden size384→768, depth10→12, head384→768 và recipeNC+LLRD→CEuniform; rank64 được giữ nguyên nên tỷ lệ rank/hidden cũng thay đổi. Nó là phép kiểm tra portability của cơ chế, không tách riêng pure-size effect. Không có cơ sở kết luận model càng lớn càng không dùng được interaction.

## 6. Literature dẫn tới lựa chọn nào?

Đã tổng hợp [báo cáo research19nguồn](PRETRAINED_ENCODER_TRANSFORMATION_RESEARCH.md), phân biệt rõ các loại bằng chứng:

- **BERT-of-Theseus:** chuyển module trong encoder với cơ chế replacement; bằng chứng gần NLP nhưng chủ yếu compression.
- **bert2BERT, LiGO, LEMON:** có quy trình tái sử dụng weights và bảo toàn/chuyển hàm, thường gắn với growth và học tiếp. Không bảo đảm một module mới sẽ tăng điểm ngay.
- **Adapters, (IA)³:** có bằng chứng dùng pretrained features cho adaptation. Chúng bác bỏ cách nói chung rằng zero/identity initialization luôn không học được.
- **GLU/FiLM:** gating và conditional modulation đã có prior art, nên proposal mới phải có control để nhận ra lợi ích riêng của nguồn context.
- **DEX, sparse upcycling, các kết quả decoder/vision:** dùng như gợi ý kỹ thuật có giới hạn, không làm bằng chứng quyết định cho BamiBERT NLU.

Nhận định sau khi có số liệu: **không cần cứu E13 bằng một giai đoạn học thêm chưa được kiểm soát**. Nên thử parameterization trực tiếp trên feature bank pretrained, với initialization và phép so đủ rõ. Candidate này được chọn theo tính kiểm chứng và khả năng tận dụng weights, không theo lời hứa novelty hay SOTA.

## 7. Thí nghiệm kế tiếp được đề xuất

### Cơ chế chốt

Tại FFN block8 của BamiBERT gốc:

`z=GELU(W1x+b1)`

`f_new = W2 [z ⊙ (1+tanh(Gc))] + b2`.

W1/W2/bias giữ từ pretrained. G là ma trận3072×768, no bias, zero init; thêm2.359.296params. c là attention update quaWo ở cùngblock. Gate ban đầu bằng1 nên giữ FFN gốc; gradient G có thể đi trực tiếp qua W2/z pretrained ngay từ bước đầu. Điều đó chưa bảo đảm gate sẽ hiệu quả hơn.

**Control:** cùngG, cùngformula và sốparams, nhưng conditioning bằng x, tức inputFFN sau khi đã trộn attention/residual/LayerNorm. x cũng có context; không gọi control là “không có ngữ cảnh”. Câu hỏi thực nghiệm là nguồn conditioning nào hữu ích hơn, và có hơn việc giữFFN gốc không.

Hai nguồn được scale bằng scalarRMS lấy từ train-onlycalibration cùngquy tắc, khóa trướcvalidation; không để khácbiệt biênđộ trở thành lýdo giả cho gainsemantic. Precision tại targetblock được chốt giống nhau cho tất cảarms; kiểm tra initparity thực trênruntime.

### Cách học chốt

Pilot đầu **không thêm MLM, KD, corpus mới hoặc stage thích nghi riêng**. Cảbaseline vàhai gate models fine-tune trực tiếp theo recipeE14, fixedLRselected,seed42,30epochs; toànencoder trainable, cùngmicro16×accumulation2 vàprecisionpolicy. Không nângLR riêng cho gate.

Lý do: gate đã có identity embedding; thêm feature-matching để khớp một hàm đã khớp ởG0 có thể chỉ khuyến khích gate tiếp tục bằng0. Nếu sau này cần học trước fine-tune, phải mở thí nghiệm architecture×adaptation có baseline nhận cùngdata/compute, không chỉ cho candidate học thêm.

### Ngân sách và tiêu chí

Đề xuất **15trials single-seed**: baseline replay và2gate arms ×5task. Baseline replay cần thiết vì chínhsáchmicrobatch/precision mới phải khớp, không dùng E14 khácđiềukiện làm đối chứng duy nhất. Không quétLR/rank/layer; vịtrí block8 chốt trước điểm.

Ghi gradient/update của gate ở các mốc đầu, gateactivation vàcorrectionnorm; có dự phòng dừng nếu phép tính sai/gradientkhôngđi hoặc runtimekhôngchấpnhận. Validation chọncheckpoint, tiêu chí đầu tư≥4/5taskgain,≥2gain≥.3pp,nodrop>.3pp và hơncontrol đủ rõ. Chỉ có tín hiệu mới xác nhận test/multiseed.

Nếu hai gate arms cùng tăng ngang nhau, đóng góp được hỗ trợ là modulation/extra capacity, chưa phải attention-update conditioning. Nếu cảhai khôngtăng, dừng candidate; không tự kéo dài corpusstage để cứu. Tênmethod vàclaimnovelty chỉ chốt sau prior-artreview/ablation tiếp theo.

Đây là **protocol đề xuất cuối**, chưa có code/training E16 được launch. [Đặc tả candidate](NEXT_ARCHITECTURE_PROTOCOL_DRAFT.md) ghi chi tiết implementation để có thể bàn giao sau khi thí nghiệm được duyệt.

## 8. Artifacts và phạm vi kiểm chứng

- [Summary phép đo](../artifacts/architecture_transfer_audit/summary.json) và [CSV](../artifacts/architecture_transfer_audit/summary.csv).
- `artifacts/architecture_transfer_audit/per_model/<e13|e15>/<variant>/<task>/seed_42/`: mode predictions/logits/metrics, telemetry, gradient probe, norms/drift vàprovenance.
- [Script implementation](../scripts/measure_architecture_transfer.py); [spec](ARCHITECTURE_TRANSFER_MEASUREMENT_SPEC.md).
- [Research có trích dẫn](PRETRAINED_ENCODER_TRANSFORMATION_RESEARCH.md).

Root đã tính lại60F1 từpredictions bằng công thức fixed-labelmacro vàpairmicro, đối chiếu IDs/gold/order với bản validation gốc, kiểm tra20gradientreports đủgroup/finite/nonzero, Pnonzero vàcheckpointSHA trước/sau. Không đọctest để chọnhướng, khôngtrain mới vàkhôngoptimizerstep. Không có trajectorygradient đầyđủ để hồi tố mọi dynamics; hạn chế này được báo rõ, không được lấp bằng suy đoán.
