# Thiết kế vòng hai: phản hồi từ FFN vào attention routing

**Cập nhật sau bước 2:** bản thiết kế này không được chuyển sang training. [Review novelty và toán học](FFN_FEEDBACK_NOVELTY_REVIEW.md) ghi tiền lệ gần, phép rút gọn và các phản ví dụ LayerNorm/W2. Nội dung dưới giữ nguyên như hồ sơ thiết kế ban đầu.

## Quyết định ở bước thiết kế

Giữ **một ứng viên để phản biện ở bước 2**, chưa triển khai hoặc training. Cơ chế tạm mô tả là *attention routing dựa trên đáp ứng hữu hạn của đặc trưng FFN*. Chưa đặt tên method hoặc tuyên bố phát minh. Không có bằng chứng hiện tại rằng nó thắng E8/BamiBERT.

Không cố bổ sung đủ ba ứng viên bằng các biến thể chưa rõ đóng góp. Hai hướng sơ bộ khác — cạnh tranh quyền nhận thông tin giữa token và attention theo độ nhạy Jacobian đơn thuần — chưa được giữ: hướng đầu có tiền lệ normalization/routing dày và giả định độc quyền không hợp ngôn ngữ nói chung; hướng sau dễ chỉ thành một metric attention phụ thuộc input. Đây là loại ở giai đoạn thiết kế, không phải kết luận novelty đã audit đầy đủ.

## 1. Vấn đề và giới hạn bằng chứng

Trong một block thông thường, Q/K quyết định routing, values được tổng hợp, rồi FFN xử lý biểu diễn nhận được. Các weights được học chung nên attention không thực sự “mù” với mục tiêu downstream. Nhưng forward của block không dùng trực tiếp trạng thái kích hoạt của FFN sau tổng hợp sơ bộ để điều chỉnh routing ngay tại block đó.

Giả thuyết cần thử: **phản hồi về đáp ứng của feature bank pretrained có thể giúp phân biệt hai nguồn có mức tương đồng Q/K gần nhau nhưng gây tác động khác nhau lên trạng thái hiện tại của token nhận**. Mục tiêu là thay đổi routing có phụ thuộc cả nguồn lẫn trạng thái nhận, không thêm một ngân hàng value ngẫu nhiên mới.

Bằng chứng local chỉ hỗ trợ động cơ yếu: E13/E15 cho thấy các nhánh value bổ sung có tác động inference nhỏ; E16 không cho gain rộng khi điều biến FFN; D1/D2 chưa xác định bottleneck layer. Không kết quả nào chứng minh routing thiếu phản hồi FFN là nguyên nhân. Ứng viên này cần tự kiếm bằng chứng, không được trình bày như lời giải đã được audit xác nhận.

## 2. Cơ chế ở mức khái niệm

1. Tính attention sơ bộ như checkpoint gốc.
2. Từ biểu diễn sơ bộ của token nhận, lấy một phần preactivation FFN đã có.
3. Với từng token nguồn, tính một phép thăm dò nhỏ: nếu đưa một hướng thông tin từ nguồn này vào trạng thái nhận, các feature FFN sẽ thay đổi bao nhiêu?
4. Dùng đáp ứng ấy hiệu chỉnh attention logits một lần; tổng hợp lại bằng **V/Wo gốc**.
5. Chạy FFN gốc trên biểu diễn cuối. Không lặp tới hội tụ, không thay loss, không dùng gold trong routing.

Đây là một forward với hai lần tổng hợp attention, không phải chạy cả model hai lần. Tuy nhiên có chi phí activation và phép tính thêm, phải đo thực tế.

## 3. Đặc tả toán học

Đặc tả trước tiên ở eval, chưa dropout. H có N token, hidden d, a heads, head width p=d/a; dùng quy ước vector cột. Tất cả Q/K/V/Wo/W1/b1/W2/b2 và LayerNorm kế thừa checkpoint. Hệ số mới duy nhất là beta cho từng head.

### 3.1. Lượt sơ bộ

Với head t:

\[
L_{ij}^{t}=q_i^{t\top}k_j^t/\sqrt p + M_{ij},\quad
A_{ij}^{t}=\operatorname{softmax}_j L_{ij}^t,
\]

\[
o_i^t=\sum_j A_{ij}^t v_j^t,\quad
x_i^0=\operatorname{LN}_{attn}(h_i+W_O\operatorname{concat}_t o_i^t+b_O).
\]

Chọn r hàng của W1, ký hiệu W1,S và b1,S, cùng một tập S cho cả heads. Pilot dự kiến r=32. Chọn S bằng seed cố định từ chỉ số FFN, độc lập labels/validation; đây là approximation có rủi ro bỏ sót feature hữu ích, không tuyên bố chọn được neuron ngữ nghĩa.

\[
z_i=W_{1,S}x_i^0+b_{1,S}\in\mathbb R^r.
\]

### 3.2. Hướng nguồn trong không gian feature FFN

W_O^t là phần d×p của Wo ứng với head t. Đặt:

\[
b_j^t=W_{1,S}W_O^t v_j^t\in\mathbb R^r,
\quad
d_j^t=\frac{b_j^t}{\max(\operatorname{RMS}(b_j^t),\delta)}.
\]

Delta ở đây là epsilon số học cố định, ví dụ 1e-6. Nguồn có b=0 cho d=0. Không nhân hướng này với attention weight Aij: tránh đưa sự ưu tiên hiện tại vào độ lớn perturbation rồi vô tình khuếch đại nó một cách hiển nhiên.

Quan trọng: đây là **thăm dò trong không gian preactivation**, không phải counterfactual xóa/thêm token chính xác. LayerNorm, ngữ cảnh hóa trước đó và attention renormalization khiến hai thao tác khác nhau. Không gọi R dưới đây là causal contribution hay xác suất nguồn đúng.

### 3.3. Đáp ứng hữu hạn

Với epsilon thăm dò cố định, dự kiến 0,1:

\[
R_{ij}^t=\frac{1}{r\epsilon^2}
\left\|\operatorname{GELU}(z_i+\epsilon d_j^t)-\operatorname{GELU}(z_i)\right\|_2^2.
\]

Tính bằng FP32. Không dùng autodiff để tính Jacobian trong forward. R đo độ thay đổi activation được lấy mẫu, **không đo độ thay đổi output FFN đầy đủ**, vì chưa xét W2 và feature tương tác ngoài S.

Chuẩn hóa theo nguồn hợp lệ của từng query/head, không tính pad:

\[
\widehat R_{ij}^t=\tanh\left(
\frac{R_{ij}^t-\operatorname{mean}_kR_{ik}^t}
{\max(\operatorname{std}_kR_{ik}^t,\tau)}\right).
\]

Dùng population std, tau cố định 1e-4; khi mọi R bằng nhau, correction bằng zero. Tanh giữ correction trong [-1,1], tránh một query gần constant bị khuếch đại vô hạn.

\[
\widetilde A_{ij}^t=
\operatorname{softmax}_j\left(L_{ij}^t+\tanh(\beta_t)\widehat R_{ij}^t\right).
\]

Beta có dấu, khởi tạo 0: model có thể học ưu tiên hoặc tránh nguồn gây đáp ứng lớn. Độ nhạy không mặc định đồng nghĩa hữu ích. Không được diễn giải beta dương thành bằng chứng semantic relevance.

Sau đó tổng hợp V bằng A-tilde, dùng attention output/residual/LN và FFN đầy đủ như gốc. Correction tối đa một đơn vị logit theo thiết kế hiện tại; đây là một giới hạn có thể làm cơ chế quá yếu, phải được ghi nhận nếu thử.

## 4. Đặc tính phân biệt và phản ví dụ

### Phụ thuộc trạng thái nhận

Hai query có cùng logits Q/K nhưng z khác có thể nhận correction khác trước cùng một nguồn. Hai nguồn có cùng logit nhưng perturbation hướng tới feature đang gần vùng chuyển tiếp GELU hoặc vùng âm bão hòa có thể có R khác. Đây là khả năng của phép tính, chưa phải bằng chứng các feature đó đúng cho sentiment/topic.

### Giới hạn affine

Nếu thay GELU bằng identity, với các hướng có RMS=1, R=1 cho mọi nguồn khác zero. Khi đó chuẩn hóa hàng xóa correction. Cơ chế được xây dựng để không chỉ chọn norm source dưới phép biến đổi tuyến tính. Trường hợp hướng gần zero/clamped là ngoại lệ cần kiểm tra riêng.

### Giới hạn Jacobian

Khi epsilon tiến về zero:

\[
R_{ij}^t\to\frac1r\|\operatorname{diag}(\operatorname{GELU}'(z_i))d_j^t\|_2^2.
\]

Giới hạn này là một **quadratic metric phụ thuộc query**, có thể được diễn đạt bằng feature lift/kernel attention. Nếu epsilon=0,1 thực tế gần như không khác giới hạn này, không được tuyên bố finite-response là đóng góp riêng. Đây là một đường bác bỏ quan trọng cần ưu tiên ở bước 2.

### Trường hợp không giúp hoặc có hại

- Token nhiễu có thể gây activation thay đổi lớn hơn token hữu ích.
- Attention sơ bộ đã sai có thể tạo z khiến feedback củng cố sai lầm.
- Feature được lấy mẫu có thể không liên quan task hoặc dồn vào vùng gần tuyến tính.
- Nguồn cùng hướng nhưng khác độ lớn bị normalization làm giống nhau.
- Việc làm mất độ lớn có thể loại thông tin cường độ hữu ích, ví dụ “rất tốt”.
- Giới hạn ở từng head không trực tiếp đo interaction giữa các head.
- Feature-bank feedback có thể chỉ là learned similarity phức tạp hơn, không mang lợi ích đặc thù nào từ pretrained FFN.

## 5. Khác các thử nghiệm đã chạy

| Thử nghiệm | Can thiệp | Ứng viên hiện tại |
|---|---|---|
| E9 | Label/aspect queries ở readout | Token-to-token routing bên trong encoder, không label queries |
| E13/E15 | Thêm nonlinear pair value với U/W/P mới | Giữ value bank, dùng feature response để sửa logits |
| E16 | Attention/input điều kiện hóa gate FFN | FFN preliminary state phản hồi ngược vào routing |
| Ý tưởng pair-state vừa loại | Giữ quan hệ qua nhiều layer | Không có edge memory qua layer |
| Ý tưởng ordered operators vừa loại | Chuỗi nhân toán tử | Không scan/token-operator products |

Khác các thử nghiệm local không đồng nghĩa khác literature. Hai lượt attention cũng có thể chỉ tương đương tăng chiều sâu hoặc recurrent refinement. Phải so đúng trước khi tuyên bố đóng góp.

## 6. Init, gradient, checkpoint và runtime

Pilot nếu được duyệt: student GD với E8 recipe, chỉ một block index6/layer7; đây là vị trí kiểm soát để so với E13, **không phải bottleneck đã xác định**. Chưa mở BamiBERT hoặc nhiều layer song song. Baseline E8 lịch sử chỉ làm reference; đối chứng mới cần cùng path runtime nếu path thay đổi.

Beta=0 bảo toàn logits attention về toán học. Không thêm projection ngẫu nhiên, không cần warmup corpus trong pilot. Gradient beta có thể đi ngay bước đầu nếu R phân biệt nguồn, values và task gradient cho tín hiệu; không bảo đảm beta gradient khác zero trên mọi batch. Gradient qua response path tới weights cũ bằng zero tại beta=0, nhưng weights cũ vẫn học qua đường model gốc. Không che giấu delay này.

Mọi source projections ở feedback, GELU differences, reductions và corrected logits tính FP32 dưới autocast-disabled. V và đường model chính có thể BF16. Việc tái tính attention bằng eager/FP32 so với SDPA tạo khác biệt số học; cần matched baseline, không lấy beta0 algebra làm bằng chứng bitwise parity.

Training cần quy tắc dropout cố định: feedback preview dùng A trước attention dropout và x-preview không output dropout; final path dùng dropout gốc đúng một lần. Preview là trạng thái xác định phụ trợ, không giả là đúng trạng thái train cuối. Các đối chứng hai lượt dùng cùng quy tắc. Không detach response trong candidate mặc định; nếu thử detach, đó là một ablation riêng, không thay âm thầm.

S phải được lưu buffer trong checkpoint; beta cũng được lưu; padding query correction=0, pad keys bị mask ở mọi reduction/softmax. Special tokens hợp lệ trong routing theo encoder gốc. Không chạy batch toàn pad; loader hiện có special tokens cho input hợp lệ.

Chi phí phụ principal: feature projections, preliminary aggregation và O(aN²r) pairwise feature responses. Với a6,N256,r32 có 12.582.912 phần tử trong tensor pair-feature nếu materialize toàn bộ; một tensor FP32 khoảng48 MiB/mẫu, chưa autograd. Batch32 chỉ tensor đó đã khoảng1,5 GiB. Bắt buộc query chunking/recomputation trước khi nghĩ training; không gọi phương pháp “gần miễn phí” vì chỉ thêm6beta.

Có thể tính W1,S Wo^t trước rồi nhân vj, tránh tensor d chiều mỗi edge. Dù chunking giảm peak tensor, autograd vẫn có thể giữ tổng activation qua chunks; cần kiểm tra checkpointing/custom recomputation, không mặc định loop chunk là đủ. Chưa có VRAM/runtime đo thực tế và chưa cấp ngân sách training.

## 7. Đối chứng và phép bác bỏ tối thiểu

Không đề xuất chạy tất cả đối chứng ngay. Thứ tự nghiên cứu:

**A. Trên giấy và toy algebra, trước model experiments.** Kiểm tra affine-null, beta0, query dependence, hướngzero, padding và Jacobian limit. Một ví dụ phân biệt được phương trình không đủ chứng minh utility.

**B. Khi bước 2 có lý do giữ lại, implementation probe nhỏ.** Chạy trên fixed train cohort để đo phân bố z, R, beta gradient, runtime và mức chênh giữa finite-response với Jacobian approximation. Việc này chưa được chạy trong bước thiết kế.

**C. Đối chứng phải phân biệt ba lời giải thích:**

| Control | Tách câu hỏi nào | Giới hạn |
|---|---|---|
| Matched baseline beta0 | Có hơn giữ attention gốc trong runtime mới không? | Candidate có thêm compute |
| Two-pass Q refinement: q'=Q(x-preview), same K/V, corrected logits L+tanh(beta)(L'-L) | Gain chỉ từ cho query xem context rồi attention lại? | Compute khác; cần báo FLOPs/runtime, không giả khớp tuyệt đối |
| Jacobian-limit response | Finite perturbation có thêm gì ngoài quadratic query-dependent metric? | Cần normalize giống nhau; implementation rẻ hơn |
| Fixed shuffled FFN-channel correspondence giữa z và d | Kết nối đúng với feature bank có quan trọng hay chỉ random nonlinear scores? | Perturbation đổi distribution; không phải control hoàn hảo |

Nếu candidate chỉ ngang hai-pass control, không bảo vệ được đóng góp FFN feedback. Nếu ngang Jacobian response, thu hẹp claim về nonlinear metric và đối chiếu prior art tương ứng; không vẫn giữ câu chuyện finite-response. Nếu các controls cũng không tăng, đóng candidate ở protocol đó, không tự quét layer/rank/epsilon để cứu.

Một single-seed screen sau này phải báo đủ5task và same-recipe BamiBERT reference đúng phạm vi. Chưa cần multi-seed trước khi có lợi ích rõ. Chưa mở test hoặc thay đổi protocol hiện tại.

## 8. Phản biện novelty cần làm ở bước 2

Tìm và đối chiếu equations của: iterative/recurrent attention refinement; FFN-informed attention; adaptive kernel/Mahalanobis attention; activation-sensitive routing; predictive-coding feedback; tied-weight lookahead; sensitivity-based feature selection. Không chỉ tìm đúng tên đề xuất.

Câu claim tạm thời có thể kiểm tra là: **dùng finite response của các preactivation FFN pretrained sau preliminary aggregation để hiệu chỉnh attention cùng block, với value bank gốc và identity initialization**. Mỗi phần đã có những họ tiền lệ; chỉ tổ hợp mô tả này chưa đủ mới.

Hai nguồn sơ bộ làm điểm khởi đầu, chưa chứng nhận novelty:

- Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*: [arXiv2012.14913](https://arxiv.org/abs/2012.14913). Diễn giải FFN như memory đã có; không được dùng nó như phát minh của project.
- *Attention that does not Explain Away*: [arXiv2009.14308](https://arxiv.org/abs/2009.14308). Attention normalization và cạnh tranh đã có nghiên cứu; không quay lại một cơ chế cạnh tranh chung mà gọi mới.

Chỉ đọc abstract/kết quả tìm kiếm của các lead này ở lượt thiết kế; bước 2 cần methods và ablations nguồn gốc. Nguồn local nền: [audit E13/E15](ARCHITECTURE_TRANSFER_FINAL_REVIEW.md), [test E16](E16_FFN_TEST_RESULT.md), [novelty audit trước](IDEA_NOVELTY_AUDIT.md).

## 9. Kết luận bước 1

Đã có một đặc tả có thể phản biện, gồm phép tính, giới hạn rút gọn, init, chi phí và controls. **Chưa có phát minh được xác nhận hoặc lý do chạy full training.** Ưu tiên kiểm tra xem cơ chế có quy về nonlinear kernel attention/recurrent refinement đã biết và có lý do kỳ vọng lợi ích ngoài việc tính thêm một lượt hay không. Nếu không vượt được hai câu hỏi đó, loại trước khi dùng GPU.
