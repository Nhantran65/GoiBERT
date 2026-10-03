# Deep research: có thể đổi kiến trúc encoder để đạt hai mục tiêu GoiBERT không?

Ngày 2026-09-21. Phạm vi: ≤100M **thắng BamiBERT** hoặc ≤1B **thắng cả BamiBERT và CafeBERT** trên năm task local, với một encoder có tính mới. Root đọc kết quả nội bộ và nguồn gốc paper/model card; chỉ đếm tham số từ config trên meta device, không train hoặc xem test trong lượt này.

## Kết luận thẳng

**Đã thử sửa kiến trúc nhiều lần.** E9–E17 gồm reader theo nhãn, span composition, attention bổ sung, pair-value residual, FFN/attention gate và cavity keys. Một số module chạy đúng và học được; chúng không tạo gain bền rộng năm task. Nói rằng dự án “chỉ fine-tune hoặc pretrain lại” là chưa đủ chính xác. [Lịch sử](EXPERIMENTS_SUMMARY_STANDALONE.md), [audit nhánh](ARCHITECTURE_TRANSFER_FINAL_REVIEW.md), [E16 test](E16_FFN_TEST_RESULT.md).

**Có một thay đổi operator chưa kiểm chứng đúng trong project:** cho các attention head trao đổi score/weights với nhau *bên trong các block pretrained*, thay vì thêm một nhánh nhỏ sau attention. Dùng Talking-Heads tĩnh, khởi tạo identity để giữ đầu ra cũ ở bước đầu. Đây là phép thử kiến trúc có chi phí kiểm chứng vừa phải nhất. Nó **không phải thuật toán mới**; nếu có gain, phần nghiên cứu là cách chuyển checkpoint đã pretrained và đối chứng cơ chế trên tiếng Việt. [Đặc tả research trước](ENCODER_REPLACEMENT_GAP_RESEARCH.md), [paper gốc](https://arxiv.org/html/2003.02436).

Mình **không có bằng chứng để hứa một kiến trúc chỉ học vài nghìn nhãn sẽ vượt Bami/Cafe ổn định trên cả năm task**. Nền mạnh là kết quả của pretraining dài. Thay block lớn và train lại ít bước thường phá weights đã học; những paper operator mới đạt lợi ích khi pretrain từ đầu hoặc có continuation đáng kể.

Nếu phải chọn một phép kiểm kiến trúc tiếp theo: **probe Talking-Heads trên Cafe/XLM-R-large cho mục tiêu ≤1B**; đóng nếu không hơn control chéo head. Nhánh ≤100M cần đồng thời nén thật xuống khoảng 80–90M và hồi phục chất lượng, nên là hai rủi ro nối tiếp. Không lấy Bami 99,5M làm headline kích thước.

## 1. Điều dữ liệu local giới hạn

| Kết quả | Ý nghĩa với ý tưởng kiến trúc mới |
|---|---|
| Bami103M Restaurant74,74; XLM-R-large/Cafe ~77 mean/2seed; ViSFD ~85,5 so Bami84,4 | Dư địa để vượt Cafe trên hai ABSA không lớn và validation noisy. |
| Cafe Restaurant≈XLM-R-large chưa CPT; Topic Cafe hơn ~1,1 ở một seed | Nền pretrained Large tạo phần lớn chất lượng đã thấy; đổi architecture phải so với **nền Large**, không so riêng Bami. |
| E15 pair-value/reader, E16 FFN-input gate: gain rải rác, không giữ trên test năm task | Không mở lại thêm residual/gate tương tự chỉ vì đổi công thức hoặc tăng LR. |
| Composite trên Bami control tăng Restaurant ~1,16/3seed nhưng trên Cafe −1,02/1seed | Kết quả từ backbone nhỏ không chuyển tự động lên Large. |
| B8 từ XLM-R-base cắt vocab Restaurant71,09; B12 73,54 | Bỏ depth là canh bạc xấu cho student ABSA; không dùng B8 làm bằng chứng mọi nén đều vô vọng. |
| 0% ví dụ train bị truncation ở max256 trong audit cũ | Operator tối ưu context dài ít khớp năm task ngắn hiện tại. Cần xác minh lại nếu đổi tokenizer. |

Nguồn: [Large baseline](XLMR_LARGE_BASELINE_RESULT.md), [E15](E15_RESULT.md), [E16 test](E16_FFN_TEST_RESULT.md), [vocab/student](VOCAB_TRIM_STUDENT_RESULT.md), [E22](E22_RESULT.md).

E16 test một seed: FFN-input gate −0,77 Sentiment, +0,46 Topic, −0,38 ViSFD, −0,03 Restaurant, −0,40 Hotel so matched control. Đây là quyết định ngừng cấu hình đó, không phải phủ định mọi dạng attention/FFN.

## 2. Hướng A: thay phép trộn giữa các attention head đã pretrained

### Cơ chế và lý do khác các thí nghiệm cũ

Trong attention thường, mỗi head tính score riêng rồi lấy trung bình từ các token khác. Talking-Heads đưa ma trận trộn **giữa các head** trước softmax và ma trận nữa sau softmax. Nếu hai ma trận là identity ở lúc khởi tạo, Q/K/V/Wo và đầu ra cũ được giữ (cần kiểm thực tế BF16/dropout/mask). Full matrices cho mỗi head dùng thông tin từ các head khác. E11 chỉ thêm/ghép hai attention map *trong cùng head*; E15 thêm residual vào output; cả hai không kiểm câu hỏi này. [Paper gốc](https://arxiv.org/html/2003.02436), [phân loại nội bộ](ENCODER_REPLACEMENT_GAP_RESEARCH.md).

Trong paper, BERT được **pretrain với operator này**: 12-head MNLI 82,6→83,6, SQuAD1.1 88,51→89,2. Đây là prior cho operator, **không phải bằng chứng gắn nó vào checkpoint XLM-R/Bami đã pretrained rồi chỉ fine-tune sẽ tăng điểm**. Paper cũng dùng relative positions trong thí nghiệm BERT, nên không có phép so hoàn toàn tương đương với checkpoint project.

Một phiên bản tối giản trên bốn block trên cùng:

- Bami12head: hai ma trận12×12/block, tổng **1.152** tham số thêm.
- Cafe/XLM-R-large16head: hai ma trận16×16/block, tổng **2.048** tham số thêm.
- Không cần corpus mới để làm pilot. Tăng FLOPs của attention và có thể mất fused SDPA; phải đo latency và dùng **matched eager control**.

### Pilot có khả năng bác bỏ

1. Trên **CafeBERT**, giữ nguyên tokenizer, encoder, task head và train split. Thay attention ở bốn block trên cùng; khởi tạo trộn bằng identity. Kiểm logits/gradients/mask tại init, đặc biệt tránh trộn `−inf` trước mask dẫn tới NaN.
2. Ba arms cùng implementation/precision và cùng số bước: **MHA nguyên bản**, **diagonal mixing** (chỉ chỉnh nhiệt độ/scale từng head), **full cross-head mixing**. Full phải hơn diagonal mới ủng hộ giả thuyết *trao đổi giữa head*, không chỉ thêm tham số hay đổi scale.
3. Diagnostic rẻ có thể tối ưu chỉ các ma trận trộn trong các checkpoint đã fine-tune, trên **train** và xem validation; vì checkpoint đã chọn bằng validation, nó không chứng minh gain. Nếu có tín hiệu, khởi tạo lại từ **Cafe pretrained** và fine-tune đầy đủ trong một protocol khóa trước trên Restaurant, ViSFD, Topic; sau đó mới mở Hotel/Sentiment và ≥3 seed.
4. Nếu chỉ có Restaurant +0,2 ở một seed, hoặc full≈diagonal, dừng. Nếu gain qua seed nhưng chỉ ở một task, báo task-specific; chưa đạt yêu cầu năm task.

Với XLM-R-large gốc, mục tiêu đầu của operator là hơn Cafe trên Restaurant/ViSFD/Topic. Nhưng Cafe đã hơn Topic ~1,1 một seed; operator phải tạo gain có ý nghĩa ở Topic và không mất các task khác. Bắt đầu trên Cafe giảm khoảng cách target; nếu operator chỉ có ích trên XLM-R-large thì đó là một kết quả khác cần đối chứng riêng.

**Tính mới:** Talking-Heads đã có từ2020. Có thể nghiên cứu *retrofitting có bảo toàn hàm*, lựa chọn block, và chứng minh gain vượt diagonal/compute-matched baseline. Chưa thấy bằng chứng đây là đóng góp phương pháp mới đủ publish nếu chỉ áp dụng paper cũ lên Bami/Cafe.

## 3. Hướng B: phân bổ lại tham số cho model nhỏ một cách thật sự có ý nghĩa

Đây là sửa kiến trúc theo *capacity allocation*, phục vụ nhánh ≤100M. Bami103M→99,5M là −3,4%, khó gọi là model nhỏ hơn đáng kể. Phương án cần nhắm khoảng **80–90M** trước khi bàn thắng Bami.

Từ config Bami local: encoder102,36M, head Hotel≈0,70M. Nếu giữ12lớp/hidden768/tokenizer và giảm FFN3072→2304, **ước tính encoder88,20M; cộng head≈88,90M** (công thức: bỏ768 kênh × hai ma trận FFN ×12lớp; cần đếm lại tensor export). Kế hoạch là chọn kênh FFN có cấu trúc, export trọng số hẹp thật, phục hồi bằng dữ liệu train sẵn có. So với hai controls: random channel pruning cùng budget và student bỏ lớp B8 đã đo. [CoFi](https://aclanthology.org/2022.acl-long.107/) và [BERT-of-Theseus](https://aclanthology.org/2020.emnlp-main.633/) cho tiền lệ pruning/replacement và recovery; paper chủ yếu task-specific, không chứng minh student general-purpose của project sẽ giữ đủ năm task.

**Gate:** trước khi distill dài, model ~89M phải giữ được Restaurant/ViSFD trong khoảng gần Bami ở nhiều seed. Nếu mất2–4 điểm ABSA như B8, không dựa vào một tham vọng KD bù lại để mở rộng. Nếu giữ chất lượng, sau đó mới thử knowledge transfer hoặc supervision mới để vượt Bami. Vì vậy đây là kiến trúc đáng khảo sát về **hiệu quả**, nhưng xác suất thắng cả năm task vẫn thấp.

Một lựa chọn khác đã đếm trong [nghiên cứu hai hướng](TWO_ROUTES_DEEP_RESEARCH_20260921.md): XLM-R-base 12L/FF2304/vocab35K≈98,86M cả head. Nó thấp hơn về kích thước so với B12 120M nhưng không có lý do sẵn để mạnh hơn Bami; phải đối chứng pruning-only và distillation. Không gộp hai chương trình nén vào một search vô hạn.

## 4. Hướng C: đổi tokenizer/vocabulary là một thay đổi kiến trúc đầu vào, nhưng không phải lời giải mạnh đã biết

XLM-V (Meta) giữ lõi XLM-R-base 12L/768 nhưng dùng vocab phân bổ lại cho nhiều ngôn ngữ. Checkpoint public có vocab901.629; meta-device đếm **777.903.360 encoder params**. Cắt embedding còn45K trong cùng kiến trúc sẽ là **120.012.288**, nhưng đó chỉ là đếm shape, **chưa có tokenizer/export được kiểm chứng**. Nó có thể là một probe về representation/tokenization, không là một model mới do project phát minh. [Config checkpoint](https://huggingface.co/facebook/xlm-v-base/blob/main/config.json), [paper](https://aclanthology.org/2023.emnlp-main.813/).

Điểm quan trọng khi đọc paper gốc theo **tiếng Việt**, không theo headline trung bình đa ngữ:

| Phép đo paper (tiếng Việt) | XLM-R-base đối chứng | XLM-V | Chênh |
|---|---:|---:|---:|
| XNLI zero-shot từ English | 75,6 | 76,2 | +0,6 |
| XNLI translate-train-all | 79,9 | 80,0 | +0,1 |
| XQuAD F1 | 72,9 | 73,1 | +0,2 |
| MLQA F1 | 65,2 | 67,2 | +2,0 |
| NER F1 | 67,7 | 67,1 | −0,6 |
| Độ dài token hóa FLORES | 34,3 | 32,4 | −5,5% |

Các số từ [paper XLM-V, Tables 3, 6–10](https://aclanthology.org/2023.emnlp-main.813.pdf). Chúng không so với XLM-R-large/Cafe trên UIT-ABSA. Vì vocab lớn chủ yếu làm chuỗi ngắn hơn, mà project không truncation ở256 token theo audit cũ, không có cơ sở đặt XLM-V lên trên phương án A. Cắt vocab1M→45K có thể xóa các token giúp XLM-V, nên phải đo fertility/`unk`/logit parity riêng cho tokenizer của nó trước mọi claim. Không tái sử dụng keep-set XLM-R.

## 5. Các hướng architecture khác đã khảo sát nhưng không xếp đầu

| Hướng | Prior art/bằng chứng | Đánh giá với năm task ngắn |
|---|---|---|
| Bias hình thái/span tiếng Việt | [HuTieuBERT ACL2026](https://aclanthology.org/2026.acl-long.472/) cải thiện POS/NER và sentence classification; E10 đã thử một loại span composition | Có động cơ tiếng Việt nhưng novelty trùng gần; chưa có bằng chứng ABSA gain rộng. Chỉ mở nếu audit token-boundary trên lỗi thật cho tín hiệu mạnh. |
| Softmax→entmax/sparse attention | [AdaSplash](https://arxiv.org/html/2502.12082) chuyển RoBERTa/ModernBERT với **2B token** continuation và anneal trong1B token đầu | Không phải sửa nhẹ rồi fine-tune. Lợi thế speed/sparsity chưa chuyển thành F1 năm task. |
| Sparse upcycling/MoE FFN | [Paper gốc](https://arxiv.org/html/2212.05055) cho T5/ViT có gain, nhưng continuation khoảng50% initial pretraining cost trong setting nổi bật | Có thể dưới1B params nếu ít expert, song tăng pipeline/compute; vài nghìn nhãn task không đủ là luận cứ để router giỏi hơn Cafe. Dự phòng dài hạn. |
| Mamba/SSM hoặc conv-token mixer | [Gap review cũ](ENCODER_REPLACEMENT_GAP_RESEARCH.md) | Điểm mạnh thường context dài/tốc độ. Đổi operator làm mất pretrained attention weights, trong khi task hiện tại rất ngắn. |
| RoPE/Pre-LN/SwiGLU mới | [Chuyển weights](PRETRAINED_ENCODER_TRANSFORMATION_RESEARCH.md) | Không có conversion giữ hàm đơn giản. Pretrain lại đáng kể mới là so sánh công bằng, quá nhiều confound để mở đầu. |
| Hai encoder/ensemble hoặc task-specific MoE | Có thể gộp Bami+Cafe dưới1B tùy thành phần | Có thể giúp điểm, nhưng là hệ thống hai backbone/latency cao; không chứng minh một encoder mới tốt hơn và novelty thấp. Chỉ dùng làm upper-bound/control. |

## 6. Kết luận quyết định và kế hoạch có giới hạn

**Nếu muốn kiểm architecture mà không tạo dataset mới:** thực hiện **một** pilot thay operator Talking-Heads trên nền mạnh nhất CafeBERT, với baseline/diagonal/full controls. Đây là phép thử chưa có trong GoiBERT, có initialization bảo toàn hàm và tiền lệ encoder, nhưng prior art đã tồn tại. Nếu nó âm, không tự mở lần lượt dynamic heads, entmax, MoE và growth để “cứu architecture”.

**Nếu mục tiêu model nhỏ có ý nghĩa:** thử chương trình **Bami≈89M** bằng FFN structured pruning và recovery, với gate giữ chất lượng trước distillation. Không gọi 99,5M là thành quả hiệu quả. Không trộn kết quả của pilot này với bài toán thắng Cafe.

**Nếu yêu cầu một phương pháp mới thật sự và thắng Cafe cả năm task:** hiện chưa có cơ chế architecture được local errors hoặc literature chứng minh đủ mạnh. Điều có thể bảo vệ trước hội đồng là một cơ chế được thiết kế sau khi đo bottleneck, có ablation vượt prior art, nhiều seed, external holdout và ít nhất một nhóm task ngoài năm benchmark; không chỉ đổi tên Talking-Heads hoặc thêm gate. Kết quả của pilot A/B quyết định liệu có nền tảng để làm research đó hay không. Đây là kết luận giới hạn của bằng chứng hiện có, không phải tuyên bố kiến trúc mới bất khả thi.

Các bài đọc sâu hơn đã có: [kiến trúc tiếng Việt](ENCODER_ARCHITECTURE_RESEARCH.md), [thay operator](ENCODER_REPLACEMENT_GAP_RESEARCH.md), [chuyển pretrained weights](PRETRAINED_ENCODER_TRANSFORMATION_RESEARCH.md). Bản này bổ sung số liệu BidirLM/Large/vocab từ sau12/09 và sửa ưu tiên theo hai mục tiêu mới.
