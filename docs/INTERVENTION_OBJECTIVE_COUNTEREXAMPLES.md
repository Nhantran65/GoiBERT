# Điều kiện cần cho mục tiêu học chống nhiễu aspect

## Ứng viên: phạt mixed finite difference

Quartet polarity đã đo cho thấy student có mixed finite difference lớn. Một đề xuất tự nhiên là phạt ||f(A+,B+)-f(A+,B-)-f(A-,B+)+f(A-,B-)||² để giảm nonlinear cross-scope interactions.

Điều kiện này không đủ. Xét a,b thuộc{-1,+1}, score của aspect A là f_A(a,b)=a+2b. Mixed difference bằng0 trên toàn quartet, nhưng dấu score chỉ đúng A ở2/4trường hợp. Model vẫn chịu ảnh hưởng từ B mạnh hơn từ A. Đây là counterexample đại số chính xác; kiểm số lưu tại artifacts/research_diagnostics/algebraic_counterexamples/mixed_difference_leakage.json.

Vì vậy không chọn pure-curvature objective làm fix đầy đủ. Nó chỉ loại phần tương tác không cộng tính, không loại additive nuisance. Thêm first-order invariance và own-factor sensitivity có thể sửa thiếu hụt điều kiện này, nhưng chưa tạo novelty: phải đối chiếu causal intervention learning và invariance training.

## Không tự áp dụng bộ luật trên mọi văn bản

Ngay cả target-specific invariance cũng chỉ hợp lệ khi B không đổi nghĩa A. Một mệnh đề chứa tham chiếu, phủ định hoặc điều kiện có thể thay đổi nghĩa phần trước. Fixture đã viết dùng các mệnh đề khép kín để có contract rõ; không mở rộng loss sang mọi phần ngoài một regex clause.

## So với IIT/DAS

IIT đã dùng counterfactual outputs để giám sát interchange interventions. DAS học alignment giữa biến cấp cao và subspace của neural representation khi đóng băng model.[^1] Học một hướng intervention bằng gradient không tự chứng minh khám phá được cơ chế đúng hoặc một architecture có quality tốt hơn; phải kiểm transfer ngoài các templates dùng để học alignment.

Quyết định: loại pure mixed-difference penalty như lời giải đủ; chưa train loss hoặc alignment mới. Cần cơ chế/đối chứng mới cụ thể trước khi bàn giao execution.

## Ứng viên: áp đặt phủ định như đổi dấu vector

Một cách tạo inductive bias là quy định phi(not p)=-phi(p), rồi dùng toán tử tuyến tính Lq để ghép p với q. Nếu đòi phi(p AND q)=Lq phi(p) cho mọi mệnh đề, cấu trúc này có nguy cơ không nhất quán. Với TRUE/FALSE và luật Boolean đầy đủ: phi(q)=Lq phi(TRUE); phi(FALSE)=Lq phi(FALSE)=-Lq phi(TRUE)=-phi(q). Suy ra mọi phi(q) bằng phi(TRUE), kể cả q=FALSE, nên tất cả bằng0. Đây là lập luận dưới assumptions mạnh đã nêu, không phải định lý về mọi encoder hoặc mọi kiểu negation representation.

Ngoài mâu thuẫn đại số, sentiment tự nhiên không đồng nhất với truth-value negation: 'không chỉ ngon' không mang nghĩa 'không ngon', còn 'không gian đẹp' không có cue phủ định dù chứa chuỗi 'không'. Không ép một token lexical thành sign-flip toàn cục. Scope và loại phép biến đổi phải được xác định trước khi áp một equivariance contract.

Một preprint2026 nghiên cứu giới hạn linear propagation với relation algebra và sự bất tương thích giữa composition/conjunction cùng negation.[^2] Chỉ dùng làm prior-art lead và cảnh báo assumptions; chưa áp theorem của paper vào GoiBERT. Certificate về head post-tanh trong project không mâu thuẫn: nó không giả định một linear operator thực hiện conjunction trên tất cả input representations.

Quyết định: không chọn sign-equivariant lane đơn giản như một fix đã có cơ sở cho semantic binding. Không mở implementation cho ý tưởng này.

## Nguồn

[^1]: Atticus Geiger et al. *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*. [Paper](https://proceedings.mlr.press/v236/geiger24a/geiger24a.pdf), §3.5/Definition4 về distributed alignment search. Đọc primary excerpt về objective và frozen models; chưa reproduce.
[^2]: Hoyeon Chang, Bálint Mucsányi, Seong Joon Oh. *Dynamics Reveals Structure: Challenging the Linear Propagation Assumption*. arXiv2601.21601v2,23May2026. [Nguồn](https://arxiv.org/abs/2601.21601). Đọc abstract và metadata phiên bản; chưa audit proof đầy đủ. Lập luận TRUE/FALSE trong ghi chú này do root tự trình bày với assumptions rõ.
