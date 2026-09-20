# Kiểm chứng ứng viên phản hồi FFN vào attention

## Kết luận

**Không chuyển đặc tả hiện tại sang training.** Chưa tìm được một paper trùng toàn bộ công thức, nên không kết luận nó là bản sao chính xác. Tuy nhiên nguyên lý attention theo độ nhạy đã có tiền lệ; phép rút gọn cho thấy phần score chủ đạo thuộc một dạng compatibility score quen thuộc; và proxy hiện tại có các phản ví dụ làm yếu động cơ “chọn thông tin theo tác động lên FFN”.

Quyết định này là loại một thiết kế trước thực nghiệm, không phải kết quả âm trên benchmark. Không chạy code model, inference, gradient probe hoặc training trong lượt này. Phạm vi tra cứu đến 12/09/2026, tập trung literature gần cơ chế và phản biện toán học; không phải chứng nhận đầy đủ mọi prior art tồn tại.

Đặc tả được kiểm: [thiết kế vòng hai](ARCHITECTURE_IDEATION_ROUND2.md). Các suy luận toán học bên dưới do nghiên cứu project thực hiện, tách khỏi các kết quả được báo trong paper.

## 1. Cơ chế đang xét

Attention sơ bộ tạo x_i=LN(h_i+o_i). Lấy r hàng FFN để có z_i=W1,S x_i+b1,S. Hướng nguồn d_j là W1,S Wo^t v_j được chuẩn hóa RMS. Điểm trước chuẩn hóa hàng:

\[
R_{ij}(\epsilon)=\frac1{r\epsilon^2}
\|g(z_i+\epsilon d_j)-g(z_i)\|^2,
\quad g=\mathrm{GELU}.
\]

Sau mean/std/tanh theo hàng, cộng score với hệ số tanh(beta_t) vào logits attention gốc; tổng hợp lại V rồi dùng FFN đầy đủ. Beta=0 giữ đường gốc về toán học. Không có state quan hệ qua layer hoặc ordered matrix scan.

Phần chưa được chứng minh là: score này có đưa vào thông tin hữu ích hơn một compatibility function thông thường hoặc việc cho attention tính hai lượt không?

## 2. Tiền lệ gần và mức tương đương

### Elliptical Attention

Nielsen et al., NeurIPS 2024, xây metric Mahalanobis từ độ biến thiên theo hướng. §3.2 định nghĩa độ biến thiên bằng directional derivative; §3.4 dùng estimator dựa trên chênh lệch values giữa các layer; §3.5 dẫn đến softmax(QMKᵀ/√d)V. Paper có đánh giá language modeling, LRA và vision, không phải bằng chứng trực tiếp trên GoiBERT.[^1]

Đây là tiền lệ trực tiếp cho **dùng độ nhạy/biến thiên của phép xử lý để thay đổi geometry attention**. Khác biệt còn lại: đề xuất của project thăm dò GELU tại từng query, theo hướng source và trong cùng block; paper không dùng đúng probe FFN đó. Không được gọi hai công thức đồng nhất, nhưng broad claim về sensitivity-aware attention không còn mới.

### Universal Transformer và Feedback Transformer

Universal Transformer lặp self-attention và transition với weights chia sẻ theo bước; phương trình 4–5 mô tả refinement. Đây là tiền lệ cho việc thêm bước xử lý để dùng biểu diễn đã tinh chỉnh cho attention tiếp theo, không phải exact finite-response score.[^2]

Feedback Transformer đưa biểu diễn từ nhiều layer vào memory để các bước thời gian sau truy cập. §3 và §4.3.2 phân biệt thiết kế memory cùng ablation. Đây là feedback qua thời gian trong mô hình autoregressive; không phải preview FFN cùng block của bidirectional encoder.[^3]

Hai nguồn trên bác bỏ việc chỉ gọi “feedback” hoặc “refinement” là đóng góp mới. Chúng không bác bỏ mọi quy tắc feedback cụ thể và cũng không chứng minh candidate sẽ có ích trên short-text classification.

### FFN memory và attention

Geva et al. phân tích FFN như key–value memories: activation coefficients còn phải nhân value/output matrix để tạo output. §2 và §3 phân biệt pattern detectors với values. Việc xem feature bank pretrained như tài nguyên để tận dụng là động cơ có nguồn, không phải phát minh của project.[^4]

Sukhbaatar et al. đã kết hợp persistent memory vào attention nhằm thực hiện vai trò của FFN. Công trình đánh giá character/word language modeling; không thực hiện probe response như đề xuất.[^5]

### Claim chart

| Claim | Đánh giá |
|---|---|
| Dùng FFN pretrained như feature/memory bank | Đã có nguyên lý |
| Dùng sensitivity để hiệu chỉnh attention | Tiền lệ gần trực tiếp: Elliptical Attention |
| Dùng xử lý sơ bộ để attention tinh chỉnh lần nữa | Đã có họ iterative/recurrent refinement |
| Beta=0 giữ checkpoint | Kỹ thuật identity/residual initialization, không tự là novelty |
| Probe GELU hữu hạn theo source ở mỗi query, dùng cùng-block preview | Chưa xác định exact prior art trùng toàn bộ |
| Probe đó mang một lợi ích tính toán/ngữ nghĩa riêng | Chưa có chứng minh; có phản ví dụ ở phần 4 |

Không dùng các trang tổng hợp hoặc kết quả chỉ trùng từ khóa “feedback” làm bằng chứng đồng nhất kiến trúc. Mahalanobis attention theo khoảng cách cũng không mặc nhiên bằng quadratic score theo source ở đây.

## 3. Phép rút gọn và điều nó thực sự chứng minh

### 3.1. Giới hạn perturbation nhỏ là dot-product trên feature maps

Taylor theo từng tọa độ, với z,d cố định:

\[
R_{ij}(\epsilon)=\frac1r\sum_k
g'(z_{ik})^2d_{jk}^2
+\frac\epsilon r\sum_k g'(z_{ik})g''(z_{ik})d_{jk}^3
+O(\epsilon^2).
\]

Đặt phi_i=g'(z_i)²/r và psi_j=d_j². Khi epsilon→0:

\[
R^0_{ij}=\phi_i^\top\psi_j.
\]

Đó là một compatibility dot-product với query/key feature maps xác định bởi pretrained weights và preview. Trước row normalization, ma trận R0 của mỗi head có rank không quá r. Do có original logits cộng vào, chuẩn hóa hàng và tanh, **không suy ra toàn bộ attention có rank r**, hoặc toàn bộ model tương đương một linear-attention implementation.

Với epsilon hữu hạn, có higher-order terms. Chưa đo trên checkpoint nên không được khẳng định epsilon=0,1 đã đủ nhỏ để chúng không đáng kể. Ngược lại, không được coi những terms đó có ích chỉ vì khác zero.

### 3.2. Chuẩn hóa hàng không cho phép bỏ qua higher-order terms một cách tùy tiện

Nếu R0 gần constant theo nguồn, standard deviation của R0 nhỏ; phần hữu hạn có thể quyết định thứ tự scores sau chuẩn hóa. Floor tau giúp tránh chia zero nhưng không đảm bảo relative error nhỏ. Vì vậy phép rút gọn ở trên là giới hạn và đối chứng cần có, **không phải chứng minh toàn bộ candidate chính xác bằng attention thường tại epsilon=0,1**.

### 3.3. Dấu nguồn bị mất ở bậc dẫn đầu

R0(d)=R0(−d). Hai nguồn tác động ngược hướng có cùng squared directional sensitivity. Beta dương hoặc âm theo head không khôi phục được phân biệt hai nguồn nếu R bằng nhau. Phần epsilon có thể phân biệt dấu qua d³, nhưng chưa biết nó đủ mạnh hoặc phù hợp ngôn ngữ.

Đây không chứng minh model hoàn chỉnh không phân biệt được phủ định: original Q/K/V vẫn có thể mang tín hiệu đó. Nó cho thấy riêng feedback score không bảo đảm bổ sung được khả năng này, nên không dùng ví dụ “tốt/không tốt” làm lời giải thích chắc chắn.

## 4. Hai phản ví dụ về đại lượng đang đo

### 4.1. LayerNorm có thể xóa hoàn toàn hướng mà proxy chấm cao

Đặt y_i=h_i+o_i trước LayerNorm. Xét một perturbation source trong residual space bằng t·1, cùng một scalar trên mọi hidden coordinate. Với LayerNorm chuẩn, kể cả affine gamma/beta và epsilon số học:

\[
LN(y_i+t\mathbf1)=LN(y_i).
\]

Do đó output FFN sau LN không thay đổi dưới perturbation này. Nhưng proxy dùng W1,S(t·1) làm hướng; vector đó nói chung không bằng zero và có thể tạo GELU response khác zero. Với thêm một source có response khác, correction giữa các nguồn có thể khác dù hướng đầu không có tác động qua đường LN thật.

Đây là một **phản ví dụ cho độ trung thực của proxy**, không phải phát hiện dữ liệu thật có nhiều sources như vậy. Đặc tả đã gọi probe là heuristic thay vì causal effect; phản ví dụ cho thấy sự khác biệt ấy có thể quan trọng, không chỉ là chú thích kỹ thuật.

Đối với perturbation rất nhỏ trong đường thật, hướng đúng phải có Jacobian LayerNorm:

\[
\delta z_i=W_{1,S}J_{LN}(y_i)\,\delta o_i.
\]

Nó phụ thuộc query, không còn chỉ một d_j dùng chung cho mọi query. Thay đổi này kéo theo score và chi phí mới; chưa tự triển khai một biến thể “sửa LN” để cứu proposal.

### 4.2. Feature response có thể nằm trong phần không ảnh hưởng output

Giả sử một cột k của W2 bằng zero. Activation neuron k có thể thay đổi nhiều nhưng đóng góp output FFN luôn bằng zero. Proxy norm trên activation vẫn cộng thay đổi đó vào score. Đây là cấu hình phản ví dụ hợp lệ, không phải khẳng định checkpoint thực tế có cột W2 bằng zero.

Tổng quát hơn, delta activation có thể nằm trong nullspace của W2 hoặc có các cột triệt nhau. Đại lượng ||delta activation||² không đo ||W2·delta activation||². Việc lấy 32 feature rồi bỏ W2 khiến chưa có cơ sở gọi score là tác động lên computation hữu ích.

Nếu bổ sung W2, score trở thành delta_aᵀ(W2ᵀW2)delta_a. Nếu thêm cả LN và lấy giới hạn nhỏ, nó tiến tới pullback quadratic metric qua Jacobian của đường xử lý. Đây là hướng metric learning cần đối chiếu prior art, không phải một cách sửa miễn phí tạo novelty.

### 4.3. Ngay cả đáp ứng output chính xác cũng không phải giá trị task

Một thay đổi output có norm lớn có thể mang nhiễu; một thay đổi nhỏ có thể quyết định đúng class gần decision boundary. Không có định lý nối response norm với giảm CE hoặc tăng pair F1 trong đặc tả này. Beta học từ labels có thể thích nghi, nhưng đó là khả năng thực nghiệm, chưa giải thích vì sao inductive bias này tốt hơn một learned compatibility score khác.

## 5. Bằng chứng local không lấp được khoảng trống này

E13/E15 có nhánh inference tác động nhỏ không kéo theo rằng mọi routing theo pretrained feature sẽ tốt hơn. E16 đã dùng pretrained FFN features nhưng không giữ broad gain trên test. D1/D2 không xác định một block thiếu phản hồi. Các kết quả này đặt yêu cầu kiểm chứng cao hơn, không chứng minh tất cả feedback đều thất bại.

Chỉ thêm ít beta không đồng nghĩa ít compute: thiết kế có O(aN²r) pair responses và thêm aggregation. So với hai-pass attention control, nếu có gain vẫn phải tách công tính thêm, score phi tuyến và việc dùng đúng feature bank pretrained.

Nguồn local: [audit kiến trúc](ARCHITECTURE_TRANSFER_FINAL_REVIEW.md), [test E16](E16_FFN_TEST_RESULT.md). Không lấy các scores test cũ làm validation để chọn epsilon/layer mới.

## 6. Quyết định nghiên cứu

**Loại bản đặc tả hiện tại khỏi hàng đợi triển khai.** Lý do phối hợp gồm:

1. Broad novelty đã bị thu hẹp đáng kể bởi sensitivity-based attention và iterative refinement.
2. Phần còn riêng là một heuristic chưa chứng minh ích lợi so với kernel score và có phản ví dụ lệch khỏi tác động đường model thật.
3. Chi phí đáng kể so với lượng bằng chứng hiện có; chưa có lý do mở năm task chỉ để xem may mắn có tăng không.

Không kết luận “paper X đã làm đúng y hệt” hoặc “không thể cải thiện F1”. Nếu mục tiêu chỉ là thử biến thể có thể tăng điểm, candidate vẫn có thể được xem như một exploratory baseline. Nhưng theo mục tiêu hiện tại là phát minh có đóng góp rõ, chưa đáng ưu tiên.

Điều kiện mở lại không phải thêm gate, epsilon grid hoặc đổi tên. Cần xác định một đại lượng có tính chất cần thiết cho nhiệm vụ — ví dụ một quan hệ bảo toàn/phân biệt cụ thể mà score phải thỏa — rồi thiết kế phép tính và đối chứng chứng minh tính chất đó. Không nên mặc định norm sensitivity là đại lượng phù hợp.

Bước 1 và bước 2 của vòng này đã hoàn tất; chưa có ứng viên mới được thông qua. Kết quả có ích là hai phản ví dụ và phép rút gọn giúp tránh một lượt training không có giả thuyết đủ chặt.

## Nguồn

[^1]: Stefan K. Nielsen, Laziz U. Abdullaev, Rachel S. Y. Teo, Tan M. Nguyen. *Elliptical Attention*. NeurIPS 2024. [Paper đầy đủ](https://papers.nips.cc/paper_files/paper/2024/file/c63908a3e946af0e7978c23737229137-Paper-Conference.pdf), §3.2–3.5, §4. Đọc methods và mô tả thực nghiệm; không tái kiểm chứng toàn bộ chứng minh lý thuyết của tác giả.
[^2]: Mostafa Dehghani et al. *Universal Transformers*. ICLR 2019. [Paper](https://arxiv.org/pdf/1807.03819), §2, equations 4–5.
[^3]: Angela Fan et al. *Addressing Some Limitations of Transformers with Feedback Memory*. [Paper](https://arxiv.org/pdf/2002.09402), §3, §4.3.2 và appendix ablations. arXiv xuất bản đầu năm 2020.
[^4]: Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy. *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP 2021. [Paper](https://aclanthology.org/2021.emnlp-main.446.pdf), §2–3. Phân tích language model; không chứng minh phản hồi FFN tăng NLU tiếng Việt.
[^5]: Sainbayar Sukhbaatar et al. *Augmenting Self-attention with Persistent Memory*. 2019. [Nguồn gốc](https://arxiv.org/abs/1907.01470). Dùng abstract như tiền lệ bổ trợ về kết hợp FFN-memory/attention, không dùng để kết luận exact equivalence của candidate.
