# Tăng năng lực theo phần sửa lỗi mà cập nhật hiện tại không thực hiện được

## Trạng thái

Giả thuyết nghiên cứu mới ở mức thiết kế, chưa xác nhận novelty, chưa launch training. Không phải đề xuất áp dụng nguyên xi TINY hoặc GradMax rồi đổi tên. Nếu khác biệt chỉ là thay loss bằng counterfactual loss, không đủ claim mới.

## Vấn đề cần giải

E4 tăng FFN rộng đồng loạt không có lợi ích rộng; E13–E17 có nhiều nhánh tác động rất nhỏ. Các phép đo này không phân biệt được: đã thiếu năng lực biểu diễn, khó tối ưu từ checkpoint, hay can thiệp nhắm sai lỗi. Controlled Restaurant probe cung cấp một contract rõ: thay polarity FOOD không được thay SERVICE khi phần SERVICE giữ nguyên; model vẫn phải đáp ứng thay đổi của chính FOOD. Contract chỉ có giá trị trên các câu được kiểm ngữ nghĩa, không tự áp dụng mọi đoạn văn.

## Khung tính toán cần kiểm chứng

Tập contract tạo vector outputs f(theta) và desired correction d từ nhãn/đối chứng đã xác minh. Gọi J là Jacobian của outputs theo nhóm tham số được phép cập nhật. Trước khi tăng kiến trúc, giải bài toán sửa bằng tham số sẵn có trong trust region:

min_delta ||W^(1/2)(J delta-d)||² + lambda||delta||²,
subject to C delta=0 hoặc các bất đẳng thức margin-preservation đã khai báo.

Residual r=d-J delta_star là phần chưa sửa theo local linear model và constraints. Chỉ xem xét tăng rank/neurons nếu residual giữ được trên các contract held-out và một bước cập nhật finite thực sự không đạt mức sửa kỳ vọng. Không gọi local residual là chứng minh toàn kiến trúc bất khả thi: tuyến tính hóa, norm budget và parameter subset có thể tạo residual giả.

Nếu cần tăng, chọn tính năng mới có hướng tác động phù hợp residual sau khi loại phần đã có thể sửa bằng tham số cũ. Candidate cần một cách sinh/tối ưu features mới và bound finite-step rõ; phần này hiện chưa chốt. Chỉ dùng SVD của loss gradient như GradMax hoặc projection như TINY chưa phải đóng góp mới.

## Prior art gần và nguy cơ trùng

TINY đã định nghĩa expressivity bottleneck bằng chênh lệch giữa functional update mong muốn và update có thể biểu diễn với architecture hiện tại; họ dùng nó để chọn neurons tránh redundancy.[^1] Vì thế novelty không thể là 'đo residual rồi thêm neurons'. Điểm đang xem xét là contract-selective correction, ràng buộc giữ phản ứng đúng của các factor khác, và kiểm finite-step trước growth. Những phần này phải được đối chiếu với constrained optimization, continual-learning gradient constraints và network morphism; chưa có kết luận khác biệt đủ lớn.

Thay metric W bằng một ma trận cố định có thể chỉ là TINY trên dữ liệu đã biến đổi. Ràng buộc tuyến tính có thể rút về projection vào feasible subspace. Cần thử các phép rút gọn này trước khi code; không gọi một generalized eigenproblem quen thuộc là phát minh.

## Tiêu chí tiếp tục

## Audit bổ sung: inner solve không mới và residual có confound

Với ràng buộc đẳng thức C delta=0, lấy N là orthonormal basis của null(C), đặt delta=N u. Bài toán trở thành ridge regression trên JN với targets d và metric W. Nếu W dương xác định, whiten bởi W^(1/2) cho bài toán least-squares quen thuộc. Phần này không tạo operator mới. Với bất đẳng thức, vẫn là quadratic program; GEM đã dùng gradient constraints để bảo vệ các nhiệm vụ trước.[^2] Không kết luận toàn bộ phương pháp giống GEM, nhưng không dùng riêng constraint projection làm claim mới.

Quan trọng hơn, residual regularized không đo thuần expressivity. Counterexample: f_theta(1)=theta, desired correction1, lambda9. Với tham số theta, ridge sửa0.1 và residual0.9. Viết cùng function class thành theta=10psi, dùng cùng lambda trên psi thì functional correction=100/109 và residual9/109≈0.08257. Scale100 cho residual≈0.000899. Function class không đổi nhưng 'bottleneck' gần biến mất. Kiểm số ở artifacts/research_diagnostics/algebraic_counterexamples/growth_parameterization_residual.json.

Do đó không dùng residual này làm growth trigger nếu chưa tách regularization/parameterization khỏi component ngoài attainable tangent space. Gradient norm lớn hơn cũng có thể chỉ do scale, không phải thêm năng lực. GradMax đã tối ưu initialization mới bằng gradient/SVD; đổi tiêu chí dữ liệu đầu vào chưa đủ khác biệt.[^3]

Trạng thái cập nhật: chưa thông qua candidate. Cần sửa định nghĩa criterion hoặc bỏ hướng; chưa có bằng chứng để triển khai growth. Đặc tả grow operation vẫn thiếu, nên chưa thể đánh giá phương pháp hoàn chỉnh là mới/khả thi.

1. Định nghĩa được grow operation và criterion khác cụ thể với các đối chứng.
2. Bound/measurement không đánh đồng optimization difficulty với expressivity.
3. Corpus/contract generation không dùng test, và baseline có cùng data/compute.
4. Cải thiện phải được kiểm trên natural validation, gồm factual neutral mentions; toy contract repair riêng không đủ.

Đây là một hướng algorithm/architecture adaptation có thể khảo sát, không thay thế mục tiêu chất lượng/novelty bằng việc hoàn thành một diagnostic.

## Nguồn

[^1]: Manon Verbockhaven, Théo Rudkiewicz, Sylvain Chevallier, Guillaume Charpiat. *Growing Tiny Networks: Spotting Expressivity Bottlenecks and Fixing Them Optimally*. TMLR2024. [Author PDF](https://www.lri.fr/~gcharpia/Expressivity_bottlenecks_preprint.pdf), introduction và §2 về functional gradient/parameterized spaces. Đã đọc các phần này; chưa audit derivation growth và experiments đầy đủ. Kết quả được giới thiệu trên CIFAR, không phải xác nhận hiệu quả BERT/NLU.
[^2]: David Lopez-Paz, Marc'Aurelio Ranzato. *Gradient Episodic Memory for Continual Learning*. NeurIPS2017. [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf). Đọc primary excerpts về local-linear assumption và QP gradient projection, chưa audit toàn bộ dual derivation.
[^3]: Utku Evci et al. *GradMax: Growing Neural Networks using Gradient Information*. ICLR2022. [Author source](https://research.google/pubs/gradmax-growing-neural-networks-using-gradient-information/). Đọc abstract và primary PDF excerpts, chưa reproduce.
