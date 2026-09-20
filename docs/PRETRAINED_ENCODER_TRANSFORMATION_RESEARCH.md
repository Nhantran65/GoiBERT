# Thay đổi kiến trúc encoder pretrained: bảo toàn hàm và khả năng học

## 1. Kết luận nghiên cứu

Những vòng E13 và E15 chưa chứng minh được cơ chế pair-value residual mang lại cải thiện rộng. Tuy nhiên, kiểm tra đầu ra bằng baseline tại initialization chưa đủ để phân biệt ba khả năng: cơ chế không phù hợp, đường tham số mới chưa được tối ưu hiệu quả, hoặc đóng góp nhỏ bị suy giảm trong phép tính số. Không nên kết luận toàn bộ kiến trúc đã chạm trần chỉ từ những phép thử thêm nhánh rồi fine-tune.

Hướng đề xuất tiếp theo là **thay cách điều khiển các feature FFN đã pretrained**, thay vì xây thêm một nhánh random với projection output khởi tạo zero. Candidate cụ thể là FFN được điều biến theo context, khởi tạo gate bằng identity và dùng lại cả hai ma trận FFN gốc. Tên mô tả là *context-conditioned identity-gated FFN*. Gating, identity initialization và conditional modulation đều có tiền lệ; đây là candidate cho project, chưa phải novelty đã được xác nhận.

Phần literature/source ban đầu được viết trước phép đo ablation và fusion FP32. Các phép đo đó nay đã hoàn tất trên20checkpoint,60lượtvalidation; kết quả và quyết định cập nhật nằm trong [review cuối](ARCHITECTURE_TRANSFER_FINAL_REVIEW.md). Chúng không ủng hộ việc chỉ sửa fusionprecision để cứu E13/E15. Bảng metadata ở mục2 vẫn được giữ để phân biệt dữ liệu cũ với phép đo mới, không dùng giả thuyết BF16 để giải thích chắc chắn các vòng thất bại.

## 2. Phạm vi bằng chứng nội bộ

E13 thêm centered/uncentered pair-value residual vào student; E15 chuyển cùng họ cơ chế sang BamiBERT gốc. Cả hai giữ đường cũ, khởi tạo projection P của nhánh mới bằng zero và cho cả encoder học downstream. Phép kiểm tra gradient ở preflight xác nhận có gradient tới P ở bước đầu, rồi tới U/W sau khi P đã cập nhật. Nó không thay thế trajectory gradient trong toàn bộ training.

E15 có các giá trị `correction_relative_norm_probe` sau:

| Task | Centered: delta/hidden (%) | Uncentered: delta/hidden (%) |
|---|---:|---:|
| Sentiment | 0,0374 | 0,1552 |
| Topic | 0,0178 | 0,1588 |
| ViSFD | 0,1088 | 0,3355 |
| Restaurant | 0,0877 | 0,3835 |
| Hotel | 0,0732 | 0,2643 |

Các số được đổi từ ratio trong metadata bằng cách nhân100. Source `scripts/run_e15.py` nạp lại best checkpoint, lấy batch đầu tiên từ validation loader rồi bật probe. `src/goibert_e15/modeling.py` chia norm delta cho norm **hidden đầu vào attention**, không phải output của Wo. Mẫu đo có padding và độ dài khác nhau; không coi bảng này là phân bố toàn validation, không xếp hạng task bằng nó.

Trong source E15, phép cộng nhánh là `self.output.dense(context) + delta`, sau đó dropout, residual và LayerNorm. Under autocast, cần đo dtype thực tại phép cộng mới biết tỷ lệ cập nhật được giữ lại. Norm nhỏ so với hidden không đồng nghĩa nhỏ so với từng tọa độ Wo hoặc không đổi decision margin.

E15 còn có khác biệt microbatch: centered Restaurant và uncentered Topic dùng16 với accumulation2, các lượt còn lại dùng32. Effective batch bằng32, nhưng dropout, kernel và thứ tự số học có thể khác. Điều này không giải thích chắc chắn việc giảm điểm, song làm yếu diễn giải chênh lệch nhỏ giữa các treatment.

Nguồn nội bộ: [E13_RESULT.md](E13_RESULT.md), [E15_RESULT.md](E15_RESULT.md), metadata từng trial và source E13/E15. Không có training hoặc inference mới trong các số trên.

## 3. Bảo toàn hàm không đồng nghĩa bảo toàn khả năng tối ưu

Xét nhánh cũ `y=f(x)+P gθ(x)`. Khi P=0, forward giữ nguyên f. Gradient của θ có nhân P nên ở bước đầu bằng0 trong số học lý tưởng, trong khi gradient P có thể khác0. Sau khi P cập nhật, θ có thể học; đây không phải nhánh chết vĩnh viễn. Nhưng tốc độ mở đường θ phụ thuộc scale, optimizer và gradient task, nên parity ban đầu không tự bảo đảm khai thác capacity mới.

Việc này cũng không chứng minh zero initialization là sai nói chung. Adapter cho NLP là một hướng thành công trên pretrained models; Houlsby et al. cung cấp bằng chứng trực tiếp trên task NLP cho việc chèn module nhỏ.[^1] ReZero dùng residual gate khởi tạo zero để hỗ trợ tối ưu mạng sâu, nhưng kết quả về trainability không phải bằng chứng một gate mới sẽ vượt BamiBERT.[^2]

Riêng E13/E15 còn có một yếu tố toán học khác. Gần0, GELU(t)=t/2+t²/√(2π)+O(t⁴). Vì vậy centered residual GELU(q+k)−GELU(q)−GELU(k) có hạng dẫn đầu √(2/π)qk: hạng tuyến tính bị loại theo đúng thiết kế. Uncentered control vẫn giữ hạng tuyến tính. Đây là khác biệt về scale và lớp hàm, không phải hai nhánh chỉ khác một nhãn tên. Khai triển này chỉ đúng cục bộ; cần phân bố q/k thực trước khi dùng nó để giải thích amplitudes đã đo. Không thể kết luận mọi tác dụng nhỏ đều do precision hoặc optimizer.

Ngược lại, nếu mọi tham số mới đều nonzero để có gradient ngay, forward có thể làm lệch biểu diễn đã học. Cần phân biệt **exact preservation**, **xấp xỉ preservation trên calibration data**, và **chuyển kiến trúc bằng học thêm**. Ba trường hợp có protocol và chi phí khác nhau.

## 4. Những gì literature thực sự hỗ trợ

### 4.1. Nhóm mở rộng model và tái sử dụng weights

Net2Net đưa ra biến đổi bảo toàn hàm để mở rộng mạng.[^3] Không thể bê nguyên thao tác copy neuron/layer vào Transformer có LayerNorm rồi giả định tương đương. Cách xử lý normalization, bias, head partition và residual đều ảnh hưởng.

bert2BERT mở rộng ý tưởng tái sử dụng parameter initialization sang BERT/GPT, kèm quá trình học tiếp; kết quả chính liên quan hiệu quả pretraining.[^4] Đây là tiền lệ cho sử dụng checkpoint cũ khi thay quy mô, không phải bằng chứng mọi model mở rộng sẽ hơn ngay sau một fine-tune ngắn.

LiGO học toán tử tăng chiều và độ sâu của weights. Đặc biệt, Appendix C.3 của bản được đọc báo cáo BERT-Small 80,38 trung bình GLUE; BERT-Base từ LiGO init không pretrain thêm 81,04; có pretraining thêm 82,57; Base từ scratch 82,25.[^5] Các con số thuộc thiết lập paper, không chuyển thành kỳ vọng tăng của GoiBERT. Chúng cho thấy cả initialization lẫn giai đoạn học tiếp đều đáng kiểm soát.

LEMON nghiên cứu mở rộng lossless, gồm các xử lý riêng cho LayerNorm.[^6] Nó củng cố yêu cầu kiểm tra hàm khi đổi architecture; không giải quyết tự động bài toán liệu capacity mới có khái quát hóa tốt hơn hay không. Cũng cần tránh nhầm với paper khác cùng tên LEMON về nén model.

### 4.2. Nhóm thay module dần

BERT-of-Theseus là bằng chứng gần bài toán encoder nhất: thay dần module BERT bằng các module nhỏ trong training, tăng xác suất thay thế, không cần thêm loss distillation riêng trong thiết kế đó.[^7] Mục tiêu là nén, nên không được diễn giải kết quả thành cam kết vượt model gốc.

Network of Theseus, bản preprint2025, nghiên cứu chuyển architecture từng phần với representational alignment; có cả ví dụ ngoài encoder NLP.[^8] Nó cung cấp quy trình thay module đáng tham khảo. Matching CKA/geometry không đồng nghĩa logit hoặc task behavior được giữ nguyên, nên nếu dùng phải kiểm tra cả hành vi downstream.

Hai hướng này có bài học chung: module đích có thể cần một giai đoạn trở thành đường tính toán hữu ích, thay vì tồn tại như nhánh phụ tùy chọn suốt quá trình. Tuy nhiên, ép sử dụng module không tự làm module tốt hơn; một thiết kế kém vẫn có thể gây giảm điểm.

### 4.3. Nhóm chuyển dense sang conditional computation

Sparse Upcycling sao chép dense FFN để khởi tạo experts rồi tiếp tục học. Paper có T5 và SuperGLUE, nhưng không phải BERT encoder-only; budget được mô tả tới khoảng50% sunk pretraining cost, không phải vài trăm step miễn phí.[^9] Vì vậy, đây là lựa chọn dài hạn có căn cứ, chưa hợp lý để mở một nhánh MoE lớn ngay trong project.

MoEfication phân tách FFN thành experts.[^10] G-MoEfication mở rộng sang các dense models không đáp ứng giả định sparsity của ReLU.[^11] Điểm này liên quan vì BamiBERT dùng GELU. Chỉ giữ top-k neuron/expert mà không đo sai lệch hàm ban đầu sẽ thêm một confound lớn.

### 4.4. Nhóm initialization và gate cho adaptation

LoRA-GA căn chỉnh initialization theo gradient full fine-tuning ở bước đầu.[^12] Nó cho thấy hình học update có thể quan trọng ngoài forward parity, nhưng không chứng minh nên thay mọi module mới bằng LoRA hoặc gradient matching.

DEX nghiên cứu thích nghi output attention của pretrained LLM và có schedule đưa cơ chế vào dần. Paper chỉ ra vấn đề zero gate không được khai thác trong thiết lập của họ.[^13] Đây là bằng chứng có điều kiện trên LLM, không chẩn đoán của E15.

GLU variants nghiên cứu gated FFN trong T5. Đối chứng paper giảm intermediate width để giữ parameter/compute gần nhau, và có pretraining trước downstream.[^14] Việc chuyển BamiBERT GELU trực tiếp sang SwiGLU mà không chuyển weights phù hợp sẽ không tái tạo protocol đó.

(IA)³ học vector scale cho activations, còn FiLM là conditional feature modulation.[^15][^16] Hai tiền lệ này giới hạn claim novelty của một FFN gate: điểm nghiên cứu phải nằm ở cách điều kiện hóa, hợp đồng khởi tạo và bằng chứng so với control, không phải từ “gate”.

Một bằng chứng encoder cần phân loại đúng là *Quantizable Transformers*: paper thử clipped softmax/gated attention trên BERT và OPT, tập trung outliers và khả năng quantization; phần BERT dùng pretraining MLM.[^17] Kết quả đó không chứng minh một gate được gắn trực tiếp lên BamiBERT sẽ tăng NLU F1. Nó có giá trị như đối chứng về cơ chế no-op và numerical behavior, không phải lý do hứa hẹn điểm benchmark.

Các preprint2026 như *Gate-Zero Growth* tiếp tục nghiên cứu hình học khởi tạo residual gate, nhưng bằng chứng Transformer được nêu thuộc continual learning với decoder.[^18] *Rational ANOVA Networks* nghiên cứu tương tác bậc thấp và activation học được, với bằng chứng chính trên controlled/vision tasks.[^19] Hai nguồn nhắc rằng bảo toàn hàm và phân rã interaction đã có prior art rộng; chúng không phải căn cứ để vượt qua thiếu hụt bằng chứng encoder/NLU trong project.

## 5. Đánh giá các hướng đối với GoiBERT

| Hướng | Giữ weights cũ | Chi phí thích nghi | Rủi ro chính | Vai trò đề xuất |
|---|---|---|---|---|
| Tiếp tục tăng scale E13 residual | Dễ | Thấp nếu chỉ fine-tune | Chỉ thành tuning amplitude, thiếu bằng chứng cơ chế | Không ưu tiên |
| Đổi hẳn sang SwiGLU/RoPE/Pre-LN | Khó giữ exact function | Có thể đáng kể | Đổi nhiều thành phần cùng lúc | Chưa dùng vòng đầu |
| Growth kiểu bert2BERT/LiGO/LEMON | Có cơ sở chuyển weights | Thường có học thêm | Confound capacity/budget | Hướng dài hạn |
| Progressive replacement | Module cũ làm đường hướng dẫn | Cần lịch thay và dữ liệu phù hợp | Matching không bảo đảm task gain | Dùng khi module không có identity embedding đơn giản |
| Sparse upcycling/MoEfication | Tận dụng FFN pretrained | Có thể lớn | Routing, sparsity, tăng parameters | Dự phòng, chưa mở ngay |
| Identity-gated FFN dùng feature pretrained | Exact ở gate identity về toán học | Có thể pilot không cần corpus mới | Novelty vừa phải; gate vẫn có thể không học | Candidate ưu tiên kiểm chứng |

Xếp hạng này là đánh giá kỹ thuật, không phải xác suất thắng. Candidate cuối được chọn vì có thể tách rõ initialization, precision, adaptation và phần kiến trúc, với ít thay đổi hơn việc chuyển toàn model.

## 6. Candidate: FFN có gate theo context với identity initialization

### 6.1. Định nghĩa

Giữ FFN cũ ở một block BamiBERT: `z_i=GELU(W1 x_i+b1)` và `f_i=W2 z_i+b2`. x_i là input FFN sau attention/residual/LayerNorm. Lấy c_i là update attention đã qua Wo ở cùng block, trước khi cộng residual; cùng shape768 và không đọc gold/category.

Thêm ma trận G có shape3072×768, no bias, khởi tạo zero. Đặt `s_i=tanh(G c_i)` và:

`f_new_i = W2 [z_i ⊙ (1+s_i)] + b2`.

Tại G=0, s=0 nên f_new=f_old. Gate nằm trong(0,2); không tạo expert/tokenizer mới. FFN features và output projection đều là weights pretrained có tín hiệu sẵn. Một G đầy đủ thêm2.359.296weights ở một block. Không dùng hai factor cùngzero vì sẽ làm mất gradient; chưa chọn low-rank factorization trong pilot đầu.

### 6.2. Gradient khác zero-output residual ở đâu?

Với L là task loss, tại G=0 có:

`∂L/∂G = [(W2ᵀ ∂L/∂f) ⊙ z_i] c_iᵀ`.

Do đó gradient G không bị nhân với projection output mới bằngzero. Nó vẫn có thể nhỏ hoặc bằngzero trên một batch cụ thể; công thức không bảo đảm gradient lớn. Điểm khác biệt là tín hiệu đi qua W2 và z đã pretrained ngay từ đầu, thay vì phải học P trước rồi mới học đường feature ngẫu nhiên.

Đây là lý do kỹ thuật để thử, không phải chẩn đoán rằng Pzero là nguyên nhân E15 thất bại. Nếu gate mới không được sử dụng hoặc không giúp downstream, phải chấp nhận kết quả âm.

### 6.3. Precision và cách triển khai có thể kiểm tra

Thay vì tính `1+tanh(...)` bằngBF16 rồi nhân, có thể tính correction tương đương:

`delta_ffn = W2 [z_i ⊙ tanh(G c_i)]`;
`f_new = f_old + delta_ffn`.

Phần gate, correction và fusion nên được kiểm tra FP32 tại targetblock để không làm tròn gate gần1 quá sớm. Cả candidate và control dùng cùng precision policy. Giữ bias b2 đúng một lần. Không thay toàn model sangFP32 hoặc tự coi sự thay đổi precision là đóng góp kiến trúc.

Exact function preservation là phát biểu số học thực. Trong runtime, cần kiểm tra eval logits với baseline, dtype, dropout và kernel. Nếu buộc dùng policy khác khiến t0logits thay đổi, phải có precision-matched baseline và báo sai lệch; không ghi “exact” chỉ từ công thức.

### 6.4. Control để phân biệt context với gate thông thường

Control chính dùng **c_i=x_i**, tức gate nhận chính input FFN thay vì update attention. Nó có cùng G, số tham số, initialization, nonlinear function và precision. Nếu hai cấu hình ngang nhau, chỉ có bằng chứng gate/extra capacity có thể hữu ích, không có bằng chứng conditioning theo attention update cần thiết.

Input FFN x_i cũng đã có ngữ cảnh từ attention và các block trước. Vì vậy, hai nhánh phải được gọi chính xác là **attention-update gate** và **FFN-input gate**; không mô tả đây là so sánh “có ngữ cảnh” với “không có ngữ cảnh”. Điều đang kiểm tra là nguồn conditioning trước hay sau phép trộn residual/LayerNorm.

Baseline E14 không gate là reference. Chỉ sau khi có gain mới thêm control constant-channel hoặc norm-only để kiểm tra model dùng nội dung context hay đơn giản rescale theo biên độ. Không mở tất cả ablation ngay lúc đầu.

Khác biệt scale giữa c_i và x_i là confound còn lại. Trước training, đo RMS trên cùng train-only calibration rows; chốt một phép chuẩn hóa scalar không học để hai nguồn có RMS cùng mức, giống nhau về quy tắc. Nếu không kiểm soát scale, không gán chênh lệch cho semantics. Không lấy validation/test để chọn hệ số làm đẹp kết quả.

### 6.5. Novelty và giới hạn

Conditional gate đã có trong GLU, FiLM và các biến thể adaptation. Candidate này không được gọi là một family kiến trúc hoàn toàn mới. Giá trị có thể bảo vệ là cách dùng pretrained feature bank, gradient ngay từ identity init, conditioning bởi attention update và protocol tách yếu tố thích nghi. Muốn claim công bố phải rà prior art sâu hơn ở công thức cụ thể và chứng minh hơn receiver-only gate.

Chưa có lý do nhận đây là cơ chế đặc thù tiếng Việt. Tiếng Việt là phạm vi đánh giá và nguồn hypothesis; nếu muốn claim ngôn ngữ học phải có phép thử tương ứng. Mục tiêu thực dụng vẫn là vượt baseline BamiBERT local trên nhiều task.

## 7. Có cần một giai đoạn thích nghi trước fine-tune?

Không mặc định có. Candidate có identity embedding nên chưa cần feature-matching chỉ để giữ output: ở initialization đã match, loss tái tạo đơn thuần có thể cho nghiệm giữ gate bằngzero. Tăng thêm một stage với loss đó mà không có tín hiệu mới không tự giúp gate học.

Nếu pilot cho thấy cần thích nghi, thiết kế một thí nghiệm **architecture × adaptation** gồm hai trục rõ ràng. Cả baseline không gate và model có gate cùng nhận dữ liệu không nhãn, masking, số token/step và objective MLM. Không chỉ pretrain model mới rồi quy gain cho architecture.

BamiBERT gốc có MLM checkpoint là lợi thế cho giai đoạn này; cần kiểm tra load đúng decoder/tied weights trước khi dùng. Không khởi tạo lại MLMhead rồi coi loss ban đầu là chất lượng encoder. Corpus phải có provenance và không có downstream validation/test trong phần bổ sung. Bộ33nghìn câu downstream của E6 không đại diện cho pretraining rộng; đồng thời thất bại E6 không cho phép giả định corpus lớn hơn chắc chắn cứu được.

Không chốt token budget pretraining khi chưa đo throughput và xác nhận corpus sẵn có. “50% pretraining cost” trong sparse upcycling và hàng trăm nghìn step trong các paper khác là chi phí thật, không thể đổi thành “vài giờ” trên GPU hiện tại bằng trực giác.

## 8. Phép đo implementation cần giao sau khi chốt đặc tả

Để phân biệt các giải thích của E15, một implementation diagnostic đủ hẹp gồm:

1. Dùng đúng bestcheckpoint, batch validation cố định, eval/no-dropout; ghi norm delta/Wo và delta/hidden trên non-pad tokens, không chỉ global batch norm.
2. Đo tỷ lệ tọa độ `BF16(Wo+delta)==Wo`, effective-correction norm và giá trị còn giữ sau LayerNorm. Equality nhiều tọa độ không đủ chứng minh không đổi hành vi.
3. So active branch, branch-off và FP32-fusion trên cùng rows; trước hết chỉ sample128, sau đó fullvalidation nếu có lý do. Không scale delta theo validation để tối ưu điểm.
4. Ghi đúng checkpoint, dtype và branchhookrestore. Tắt nhánh sau training là can thiệp inference trên model đã co-adapt, không phải thay thế đối chứng train-without-branch.
5. Nếu gradient trajectory không được lưu, ghi “không có dữ liệu”. Có thể đo gradient hiện tại ở checkpoint nhưng đó không phải gradient lịch sử của epoch đầu.

Đặc tả được cụ thể hóa và đã thực thi trong [phép đo E13/E15](ARCHITECTURE_TRANSFER_MEASUREMENT_SPEC.md). [Review cuối](ARCHITECTURE_TRANSFER_FINAL_REVIEW.md) báo các kết quả quan sát được và giới hạn: làm trònBF16 có thật, nhưng sửa riêngfusion không tạo cải thiện rộng; gradient hiện tại không chết. Công việc implementation chỉ tạo số liệu, còn diễn giải và quyết định thuộc review nghiên cứu.

## 9. Protocol pilot đề xuất, chưa launch

Chọn một block BamiBERT ở giữa-trên trước khi thấy điểm, không quét layer. Hai gate configurations context/receiver giữ cùng recipe E14, cùng microbatch, originalcheckpoint và seed42. Khởi đầu bằng kiểm tra số học, gradient và một smoke có giới hạn; không dùng smoke để chọn cấu hình theo validation.

Nếu gate vận hành đúng và chi phí chấp nhận được, chạy cả5task cho hai cấu hình. Dùng baselineE14 đã có làm reference; nếu precision/microbatch thay đổi vật chất thì cần baseline cùng điều kiện thay vì tái dùng một reference không tương đương. Không cần multi-seed trước khi có tín hiệu.

Tiêu chí giá trị đầu tư vẫn là vector năm task: ≥4/5 tăng, ít nhất hai gain≥0,3điểm, không có regression>0,3điểm, và candidate có lợi thế so với receiver-only control. Đây là heuristic, không significance. Nếu chỉ Restaurant/Hotel tăng, không gọi thắngrộng. Chỉ các cấu hình có tín hiệu mới được xác nhận test/multi-seed theo plan được duyệt.

Nếu cả gate configurations thất bại, không tự kéo dài adaptation, tăngG hoặc sửa learning rate để cứu. Khi đó kết quả sẽ làm yếu hypothesis feature modulation trong recipe này. Có thể xem growth/upcycling là chương trình khác với budget riêng; không nối chúng thành một search vô hạn.

## 10. Điều rút ra cho chương trình nghiên cứu

Bảo toàn forward là một điều kiện kỹ thuật, không phải bằng chứng architecture tốt. Tương tự, một nhánh có gradient không chứng minh nó tạo tín hiệu đủ mạnh, và một norm lớn không chứng minh tín hiệu hữu ích. Cần nối ba mức: hàm ban đầu, dynamics sử dụng module, và chất lượng task.

Các paper growth/replacement hỗ trợ việc thiết kế **cách chuyển kiến trúc**, không cho phép quy mọi kết quả âm trước đó cho thiếupretraining. E13/E15 vẫn là các kết quả âm hợp lệ trong protocol đã chạy. Điều cần sửa là mức độ rộng của kết luận, không phải xóa hoặc làm lại lịch sử cho đến khi tìm được thắng.

Candidate gate ởFFN được đề xuất vì tạo một phép thử khác về cách sử dụng kiến thức pretrained. Chưa chứng minh nó mới trên toàn literature, chưa có kết quả downstream và chưa được launch. Nền chính là BamiBERTE14; studentE8 tiếp tục là reference nhỏ để kiểm tra portability nếu có tiến bộ.

## Nguồn

[^1]: Neil Houlsby et al. [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html). ICML2019. Bằng chứng adapter trên NLP.
[^2]: Thomas Bachlechner et al. [ReZero is All You Need: Fast Convergence at Large Depth](https://arxiv.org/abs/2003.04887). 2020. Residual identity initialization và trainability.
[^3]: Tianqi Chen, Ian Goodfellow, Jonathon Shlens. [Net2Net: Accelerating Learning via Knowledge Transfer](https://arxiv.org/abs/1511.05641). 2015/ICLR2016.
[^4]: Cheng Chen et al. [bert2BERT: Towards Reusable Pretrained Language Models](https://aclanthology.org/2022.acl-long.151/). ACL2022. Chuyển weights và pretraining sau growth.
[^5]: Peihao Wang et al. [Learning to Grow Pretrained Models for Efficient Transformer Training](https://arxiv.org/html/2303.00980). 2023, bảnv1. Đặc biệt AppendixC.3 về có/không pretraining thêm.
[^6]: [LEMON: Lossless model expansion](https://arxiv.org/html/2310.07999). 2023/ICLR2024. Bảo toàn hàm và xử lý LayerNorm.
[^7]: Canwen Xu et al. [BERT-of-Theseus: Compressing BERT by Progressive Module Replacing](https://aclanthology.org/2020.emnlp-main.633/). EMNLP2020.
[^8]: Vighnesh Subramaniam et al. [Network of Theseus (like the ship)](https://arxiv.org/html/2512.04198). Preprint2025, bảnv1. Chuyển kiến trúc từng phần, không đồng nhất với BERT-of-Theseus.
[^9]: [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/html/2212.05055). 2022/ICLR2023. T5/ViT và continuation budget.
[^10]: [MoEfication: Transformer Feed-forward Layers are Mixtures of Experts](https://aclanthology.org/2022.findings-acl.71/). FindingsACL2022.
[^11]: [Breaking ReLU Barrier: Generalized MoEfication for Dense Pretrained Models](https://aclanthology.org/2024.emnlp-main.563/). EMNLP2024.
[^12]: [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://arxiv.org/abs/2407.05000). 2024. Initialization theo gradient.
[^13]: Chaerin Kong, Jiho Jang, Nojun Kwak. [Understanding Differential Transformer Unchains Pretrained Self-Attentions](https://arxiv.org/html/2505.16333v1). 2025. DEX, init và annealing; bằng chứng chính trênLLM.
[^14]: Noam Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/html/2002.05202). 2020. T5 và parameter-matched gatedFFN.
[^15]: Haokun Liu et al. [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638). NeurIPS2022. (IA)³.
[^16]: Ethan Perez et al. [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871). 2017/AAAI2018. Tiền lệ conditional feature modulation, không phải bằng chứng encoder tiếng Việt.
[^17]: Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort. [Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing](https://arxiv.org/html/2306.12929). 2023. BERT/OPT, outliers và quantization.
[^18]: Dante Lok. [Gate-Zero Growth: A Geometric Framework for Function-Preserving Continual Learning](https://arxiv.org/abs/2607.14571). Preprint2026. Không coi decoder continual-learning results là NLU evidence.
[^19]: Jusheng Zhang et al. [Rational ANOVA Networks](https://arxiv.org/abs/2602.04006). Preprint2026. Prior art interaction decomposition; kết quả chính không thuộc encoder tiếng Việt.

Nguồn nội bộ: metadata E15 từngtask/variant, `scripts/run_e15.py:474–494`, `src/goibert_e15/modeling.py:176–246`, các báo cáo E13/E14/E15. Các phân tích số học và candidate gate trong báo cáo là suy luận/đề xuất, không phải kết quả thực nghiệm mới.
