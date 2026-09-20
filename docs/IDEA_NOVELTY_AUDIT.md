# Kiểm chứng tính mới của hai cơ chế encoder

**Kết luận:** chưa có ứng viên nào đủ cơ sở để chuyển sang triển khai dưới mục tiêu phát minh kiến trúc mới. Ý tưởng lưu quan hệ xuyên layer trùng với một dòng kiến trúc đã có. Ý tưởng hợp thành toán tử theo thứ tự còn một cấu hình hẹp chưa xác định được tiền lệ hoàn toàn giống, nhưng các nguyên lý chính đều đã có; riêng việc chưa tìm thấy công thức trùng không chứng minh tính mới.

Phạm vi bằng chứng đến ngày 12/09/2026. Báo cáo phân biệt ba mức: cùng động cơ, cùng họ phép tính, và cùng kiến trúc cụ thể. Không đánh đồng các mức này. Đây là đánh giá tính mới và độ chặt của giả thuyết; không phải kết quả training của hai cơ chế.

## 1. Hai giả thuyết được kiểm tra

### 1.1. Quan hệ có trạng thái xuyên chiều sâu

Mỗi cặp token có hướng có một vector riêng, được giữ và sửa qua các layer:

\[
R_{ij}^{l+1}=(1-g_{ij}^{l})R_{ij}^{l}+g_{ij}^{l}F(h_i^{l},h_j^{l},R_{ij}^{l}).
\]

Trạng thái này tác động lên attention ở layer kế tiếp. Động cơ là giúp quan hệ giữa các thành phần tồn tại rõ ràng, thay vì chỉ được mã hóa ngầm trong token states. Ví dụ minh họa là phân biệt nhận xét về phòng với nhận xét về nhân viên trong cùng câu. Đây chưa phải bằng chứng rằng baseline đang thiếu cơ chế đó.

Đặc tả ban đầu chưa chốt F, khởi tạo R, vị trí gắn, hay cách R tác động lên logits/value. Không có phép cập nhật tam giác trong đặc tả. Đối chứng đề xuất là reset R mỗi layer, giữ phần tính toán còn lại tương đương.

### 1.2. Token tạo toán tử, mỗi token nhận hợp thành một chuỗi riêng

Với token nhận i, quét nguồn j theo thứ tự:

\[
s_{i,j}=s_{i,j-1}+\alpha_{ij}M(h_j)s_{i,j-1}.
\]

Hai chiều quét được dự kiến dùng cho encoder hai chiều. Đối chứng ban đầu:

\[
s_i^{\mathrm{sum}}=h_i+\sum_j\alpha_{ij}M(h_j)h_i.
\]

Phân tích dưới đây lấy s(i,0)=h_i; đây là giả định bổ sung cần nêu rõ vì điểm khởi đầu chưa được chốt. Các hệ số attention được tính trước phép quét từ biểu diễn layer hiện tại. Nếu chúng được tính lại từ trạng thái quét thì đó là một cơ chế khác.

## 2. Tiền lệ của trạng thái quan hệ

**Relational Transformer là đối chiếu trực tiếp nhất.** Công trình duy trì node vectors và directed edge vectors, dùng edge để điều kiện hóa Q/K/V, rồi cập nhật edge từ trạng thái cũ, edge ngược chiều và hai node đầu mút. Xem §3.1–3.2, phương trình 3–7. Bằng chứng thực nghiệm chính thuộc graph/algorithmic reasoning, không phải benchmark tiếng Việt.[^1]

**Edge Transformer** đặt vector tại mỗi cặp node và cập nhật qua triangular attention. Công trình có thực nghiệm semantic parsing và dependency parsing, nên tiền lệ không chỉ nằm trong bài toán đồ thị ngoài ngôn ngữ. Tuy nhiên cơ chế tam giác của họ không đồng nhất với F cục bộ trong đề xuất.[^2]

**RealFormer** truyền attention scores trước softmax giữa các layer. Đây là tiền lệ hẹp hơn: lưu lịch sử attention không đồng nghĩa lưu vector quan hệ tổng quát. Công trình có thực nghiệm BERT/NLU, vì thế cũng không thể lấy riêng việc đưa lịch sử attention vào encoder làm tuyên bố mới.[^3]

| Thành phần đề xuất | Đánh giá |
|---|---|
| Một vector cho mỗi cặp token | Đã có tiền lệ rõ |
| Giữ vector qua layer và cập nhật bằng hai đầu mút | Đã có tiền lệ gần trực tiếp |
| Dùng quan hệ để thay đổi attention | Đã có tiền lệ rõ |
| Gate giữ/sửa/xóa trạng thái | Cần một quy tắc có đóng góp riêng; công thức nội suy hiện tại chưa cung cấp điều đó |
| Reset mỗi layer làm đối chứng | Có ích để đo vai trò bộ nhớ, không phải bản thân phát minh |
| Áp dụng trên distilled encoder tiếng Việt | Có thể là đóng góp thực nghiệm, chưa đủ cho mục tiêu kiến trúc mới |

**Phán quyết:** bác bỏ tuyên bố mới ở cấp nguyên lý. Không khẳng định code tương lai sẽ giống hoàn toàn một paper, vì đặc tả chưa đủ đầy. Nhưng thay F hoặc thêm gate vào một cơ chế có sẵn không tự tạo ra đóng góp khoa học có ý nghĩa.

Bộ nhớ cho tensor quan hệ là O(N²r) mỗi layer được lưu, chưa tính autograd. Với N=256 và r=32, riêng một tensor BF16 có 2.097.152 phần tử, tức 4 MiB mỗi mẫu. Đây là phép tính kích thước, không phải đo VRAM training. Cần một lợi ích cụ thể để biện minh chi phí này; ví dụ ngôn ngữ trực giác hiện tại chưa đủ.

## 3. Tiền lệ của hợp thành toán tử

### 3.1. Từ và cụm từ là phép biến đổi

Yessenalina và Cardie đã biểu diễn từ bằng ma trận, hợp thành bằng phép nhân ma trận lặp lại cho sentiment vào năm 2011. Vì vậy câu chuyện “không” biến đổi “tốt”, thay vì chỉ cộng hai vector, không phải nguyên lý mới.[^4]

MV-RNN gán vector và ma trận cho từ/cụm từ; ma trận thay đổi biểu diễn của thành phần kết hợp với nó. Điểm khác là phép hợp thành theo cây, không phải chuỗi attention riêng của mỗi token nhận.[^5]

Lifted Matrix-Space Model sinh ma trận từ word embedding bằng một phép biến đổi dùng chung rồi hợp thành. Công trình đánh giá trên SNLI, MultiNLI và SST. Do đó thay bảng ma trận theo từ bằng mạng sinh M(h) cũng không tự giải quyết vấn đề tính mới.[^6]

### 3.2. Transition phụ thuộc đầu vào

Multiplicative RNN năm 2011 đã để ký tự đầu vào điều khiển ma trận chuyển hidden state. Nó có phi tuyến và các thành phần khác; không đồng nhất với phép quét tuyến tính thuần nhất đang đề xuất.[^7]

Họ linear RNN hiện đại có dạng H(t)=A(x(t))H(t−1)+B(x(t)). DeltaNet dùng transition identity trừ một outer product có gate; DeltaProduct mở rộng bằng tích các generalized Householder transformations. Các ma trận transition không chéo này cho phép tương tác kênh và phụ thuộc thứ tự. Kết quả state-tracking/language modeling của họ không tự chứng minh lợi ích trên encoder tiếng Việt.[^8]

### 3.3. Query điều khiển phép quét

DMN+ tính attention gate dựa trên câu hỏi và episodic memory rồi dùng gate này thay update gate của GRU khi quét facts. §3.3, phương trình 11 và §5 so sánh attention-based GRU với soft attention. Như vậy nguyên lý “query điều khiển một phép tổng hợp có thứ tự thay vì weighted sum” cũng có tiền lệ NLU.[^9]

DMN+ không phải chính xác phép nhân ma trận thuần nhất cho tất cả token nhận. Khác biệt đó được giữ lại trong đánh giá; không thể nói hai kiến trúc đồng nhất. Tuy nhiên cũng không thể bảo vệ tính mới chỉ bằng cách thêm từ “query-conditioned” vào ordered composition.

| Mệnh đề | Phán quyết |
|---|---|
| Token là toán tử thay đổi nghĩa | Đã có |
| Hợp thành toán tử theo thứ tự | Đã có |
| Sinh ma trận từ embedding | Đã có |
| Ma trận chuyển trạng thái phụ thuộc token | Đã có |
| Attention phụ thuộc query điều khiển recurrence | Đã có |
| Đúng tích (I+αijMj) cho mọi token nhận trong encoder checkpoint sẵn có | Chưa xác định được một nguồn trùng toàn bộ đặc tả |
| Cấu hình cuối cùng có đóng góp vượt tổ hợp thành phần cũ | Chưa có bằng chứng hoặc lập luận đủ mạnh |

## 4. Kiểm tra đại số của ý tưởng 2

Các kết quả trong phần này là suy luận trực tiếp từ công thức đề xuất, không phải kết quả thực nghiệm hoặc định lý được lấy từ các paper.

### 4.1. Chính xác là một recurrence tuyến tính có điều kiện

Đặt T(i,j)=I+α(i,j)M(j). Khi giữ i cố định:

\[
s_{i,N}=T_{i,N}\cdots T_{i,2}T_{i,1}h_i.
\]

Đây là recurrence tuyến tính theo trạng thái quét, với transition phụ thuộc query và nguồn; additive write bằng 0. Toàn bộ block vẫn có thể phi tuyến theo đầu vào vì M và α phụ thuộc hidden states. Không nên gọi toàn bộ kiến trúc là một ánh xạ tuyến tính.

Nó thuộc họ recurrence đã biết, nhưng việc thuộc một họ rộng không tự phủ nhận mọi khả năng đóng góp mới. Đóng góp phải nằm ở hạn chế cấu trúc, thuật toán, tính chất hoặc bằng chứng riêng của trường hợp này.

Nếu khởi tạo s(i,0)=0 thì mọi s(i,j)=0. Nếu định dùng zero initialization để giữ checkpoint, phải zero nhánh residual đầu ra hoặc dùng cách khác; không được vô tình đặt trạng thái thuần nhất về zero rồi chờ nó học tín hiệu.

### 4.2. Đối chứng cộng đang trộn hai yếu tố

Với hai nguồn a rồi b, đặt a=α(i,a), b=α(i,b):

\[
(I+bM_b)(I+aM_a)h_i
=h_i+aM_ah_i+bM_bh_i+abM_bM_ah_i.
\]

Đối chứng cộng bỏ hạng cuối. Hạng này vẫn có thể khác zero khi MaMb=MbMa. Ví dụ Ma=Mb=I và a=b=1/2: tích cho 2,25h, tổng cho 2h, dù hai toán tử giao hoán tuyệt đối.

Do đó thắng đối chứng cộng chỉ cho thấy lợi ích của các tương tác mà phép cộng đã bỏ, chưa chứng minh tính không giao hoán là nguyên nhân. Đó là điểm yếu trong thiết kế đối chứng ban đầu và cần sửa trước mọi training.

Khi đổi đúng thứ tự hai toán tử nhưng giữ nguyên M và hệ số đi kèm:

\[
s_{a\to b}-s_{b\to a}
=ab(M_bM_a-M_aM_b)h_i.
\]

Đây mới là hiệu ứng commutator. Đổi thứ tự từ trong câu rồi chạy lại encoder còn làm đổi positional embeddings, hidden states và attention; thao tác đó là phép đo hành vi ngôn ngữ, không phải cô lập đại số ở trên.

### 4.3. Hai chiều có thể triệt tiêu tín hiệu mong muốn

Nếu hai chiều dùng cùng toán tử, cùng hệ số và lấy trung bình, với hai nguồn ta có:

\[
\tfrac12[(I+bM_b)(I+aM_a)+(I+aM_a)(I+bM_b)]h_i.
\]

Hạng bậc hai là ab(MbMa+MaMb)h/2; phần phản đối xứng bị triệt tiêu. Không phải mọi encoder hai chiều đều mất thứ tự, nhưng cách hợp nhất đối xứng này làm mất đúng tín hiệu đổi thứ tự đang muốn khai thác trong trường hợp hai nguồn.

Vì thế câu “quét hai chiều” chưa đủ thành đặc tả. Phải chốt cách phân biệt chiều và phép hợp nhất, đồng thời kiểm tra xem phép hợp nhất có xóa đặc tính thiết kế không.

### 4.4. Giới hạn tính toán và ổn định

Với N token nhận và N token nguồn, cách thực thi trực tiếp có N² transitions. Ma trận đầy đủ d×d dẫn tới O(N²d²) công cho matrix-vector products, chưa tính sinh toán tử. Rank-r có thể giảm chi phí áp dụng xuống O(N²dr), nhưng không tự biến toàn bộ cơ chế thành linear-time theo N.

Parallel scan có thể giảm chiều sâu tuần tự; không làm biến mất tổng công tính toán. Transition riêng cho từng query cũng ngăn việc mặc định chia sẻ một prefix state cho tất cả query như nhiều recurrent attention thông thường.

Ma trận diagonal trong một cơ sở chung giao hoán. Low-rank nói chung không bắt buộc giao hoán, nhưng cần kiểm tra tham số hóa cụ thể. Nếu chọn dạng quá hạn chế để chạy nhanh, có thể vô tình bỏ mất tính chất cần đo.

Nếu α không âm, tổng bằng 1 và ||Mj||≤C thì:

\[
\|\prod_j(I+\alpha_{ij}M_j)\|
\leq\prod_j(1+\alpha_{ij}C)\leq e^C.
\]

Đây chỉ là chặn norm đầu ra tương đối với đầu vào của một phép quét. Nó không bảo đảm gradient tốt, khả năng phân biệt nghĩa, hay ổn định toàn bộ mạng. Khi scale toán tử nhỏ, tương tác bậc hai có thể quá yếu; khi tăng scale, cần kiểm soát số học và tác động lên checkpoint.

## 5. Ý nghĩa đối với GoiBERT

Hai đề xuất chưa gắn được một lợi ích kiến trúc độc lập với các vấn đề đã thấy trong thí nghiệm trước. E16 tăng test Topic nhưng giảm ở bốn task còn lại trong lượt seed42; kết luận phù hợp là nhánh đó không đạt mục tiêu thắng rộng trong protocol đã chạy. Không thể suy ra encoder đã chạm trần hoặc mọi cải tiến kiến trúc đều vô ích. Số liệu và giới hạn được lưu trong [review E16](E16_FFN_TEST_RESULT.md).

Đối với cả hai ý tưởng, ví dụ phủ định/target binding mới là động cơ. Chưa có bằng chứng rằng đúng thiếu hụt tính toán này giải thích phần lỗi có thể sửa trên cả năm task. Tăng expressive power trên một bài toán synthetic cũng chưa bảo đảm cải thiện dữ liệu thực.

Nếu sau này xây dựng một biến thể đủ khác biệt, student vẫn là điểm bắt đầu hợp lý vì baseline và pipeline đã có. Đây là lựa chọn thực thi, không phải bằng chứng mới hay giả định student chắc chắn sẽ hưởng lợi. Nếu cần adaptation trước fine-tune, baseline phải nhận cùng dữ liệu và ngân sách tương ứng; không lấy lợi ích học thêm làm lợi ích kiến trúc.

## 6. Quyết định và điều kiện mở lại

**Đóng ý tưởng 1 ở cấp đề xuất hiện tại.** Muốn mở lại phải có quy tắc suy luận/cập nhật riêng có tính chất kiểm chứng được, vượt việc lưu edge state và gắn gate. Reset-memory control chỉ đo một thành phần, chưa phân biệt với dòng relational transformer.

**Chưa chuyển ý tưởng 2 sang training.** Giữ lại như một hướng thiết kế chưa hoàn chỉnh, không đặt tên một block mới và không tuyên bố đã tìm ra gap. Phần query-specific operator product là chỗ chưa được xác định hoàn toàn, nhưng tiền lệ matrix composition, multiplicative recurrence và attention-gated scan đặt yêu cầu phản biện cao.

Một phiên bản tiếp theo chỉ đáng triển khai khi trả lời được bốn câu hỏi:

1. Phép tính cụ thể khác các tiền lệ ở đâu, và khác biệt đó giải quyết hạn chế nào?
2. Có ví dụ toán học hoặc phép đo phân biệt được đóng góp này với cộng, tích giao hoán và recurrent aggregation đã biết không?
3. Khởi tạo, hợp nhất hai chiều và tham số hóa có giữ được tính chất đó không?
4. Có cách kiểm tra tín hiệu rẻ trước khi fine-tune đủ năm task, mà không cần chọn cấu hình bằng test không?

Nếu nghiên cứu tiếp, nhiệm vụ kế tiếp là thiết kế lại một cơ chế có khác biệt rõ và chứng minh tính chất phân biệt trên giấy. Không phải chạy thêm nhiều seed cho hai công thức hiện tại. Việc sàng lọc này chưa tạo ra phát minh đạt yêu cầu; nó loại được hai tuyên bố quá rộng trước khi tiêu tốn GPU.

## Nguồn

[^1]: Cameron Diao, Ricky Loynd. *Relational Attention: Generalizing Transformers for Graph-Structured Tasks*. ICLR 2023; [bản đầy đủ, §3](https://arxiv.org/html/2210.05062v3).
[^2]: Leon Bergen, Timothy J. O’Donnell, Dzmitry Bahdanau. *Systematic Generalization with Edge Transformers*. NeurIPS 2021; [paper](https://proceedings.neurips.cc/paper/2021/file/0a4dc6dae338c9cb08947c07581f77a2-Paper.pdf).
[^3]: Ruining He, Anirudh Ravula, Bhargav Kanagal, Joshua Ainslie. *RealFormer: Transformer Likes Residual Attention*. Findings ACL 2021; [paper và metadata](https://aclanthology.org/2021.findings-acl.81/).
[^4]: Ainur Yessenalina, Claire Cardie. *Compositional Matrix-Space Models for Sentiment Analysis*. EMNLP 2011; [paper](https://aclanthology.org/D11-1016.pdf).
[^5]: Richard Socher, Brody Huval, Christopher D. Manning, Andrew Y. Ng. *Semantic Compositionality through Recursive Matrix-Vector Spaces*. EMNLP-CoNLL 2012; [paper](https://aclanthology.org/D12-1110.pdf).
[^6]: WooJin Chung, Sheng-Fu Wang, Samuel R. Bowman. *The Lifted Matrix-Space Model for Semantic Composition*. CoNLL 2018; [paper](https://aclanthology.org/K18-1049.pdf), §3–4.
[^7]: Ilya Sutskever, James Martens, Geoffrey Hinton. *Generating Text with Recurrent Neural Networks*. ICML 2011; [paper](https://icml.cc/2011/papers/524_icmlpaper.pdf).
[^8]: Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, Riccardo Grazzi. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*. 2025; [bản đầy đủ](https://arxiv.org/html/2502.10297v3), §2–3. Metadata bản đầu có tiêu đề *Increasing the Expressivity of DeltaNet Through Products of Householders*; cùng arXiv 2502.10297.
[^9]: Caiming Xiong, Stephen Merity, Richard Socher. *Dynamic Memory Networks for Visual and Textual Question Answering*. ICML 2016; [paper](https://proceedings.mlr.press/v48/xiong16.pdf), §3.3 và §5.
