# Keys loại đóng góp từ người nhận: thiết kế và phản biện

**Trạng thái sau kiểm chứng:** giữ ứng viên cho giai đoạn triển khai nhỏ; xem [review cuối](RECIPIENT_EXCLUDED_KEYS_REVIEW.md). Phép kiểm đại số và guarded evaluation đã hoàn tất trên tensors tổng hợp, chưa model training. Novelty được đề xuất ở operator/algorithm cụ thể, không phải nguyên lý non-backtracking.

## 1. Câu hỏi nghiên cứu

Có thể cho attention đọc các keys đã nhận thêm context mà tránh dùng chính message vừa gửi đi như bằng chứng mới quay về hay không? Thiết kế dưới đây đặt điều kiện **loại một đường truyền cụ thể khỏi key**, không đo “độ hữu ích” bằng một norm heuristic.

Đây là đề xuất mới của project ở cấp **operator và cách tính chính xác**, chưa có training hoặc bằng chứng cải thiện GoiBERT. Non-backtracking và cavity messages là nguyên lý đã có. Không tuyên bố phát minh nguyên lý đó hoặc model hoàn chỉnh không có backtracking.

Ứng viên đã được kiểm chứng đại số theo [spec](CAVITY_KEY_ALGEBRA_SPEC.md). Trạng thái cuối và lỗi số học phải đọc cùng kết quả kiểm chứng; một công thức đúng trên số thực không bảo đảm implementation FP32 ổn định.

## 2. Động cơ và giới hạn local

Một token nguồn B được làm giàu context từ nhiều token, trong đó có A. Khi A dùng key của B để chọn nguồn, key ấy có thể chứa chính message từ A. Thông tin quay về có thể hữu ích, nhưng cũng có thể củng cố một liên kết do chính A tạo ra. Ý tưởng là cho model lựa chọn một đường refinement có kiểm soát nguồn gốc.

GoiBERT có các ví dụ target/ontology binding sai trong E12, nhưng chưa đo được tỷ lệ lỗi do đường quay về này. Không nói đây là bottleneck đã tìm ra, không lấy lý thuyết GNN về oversquashing làm bằng chứng cho short-text BERT. Câu hỏi thực nghiệm là liệu **refinement có loại đóng góp người nhận** hơn refinement không loại, với cùng weights/recipe/compute gần khớp.

## 3. Định nghĩa operator

Dùng N tokens, H heads, hidden d, p=d/H. Vector cột. Tại một block, tính Q,K,V và attention A gốc từ input h. A^t_{jk} là xác suất token j đọc token k trong head t. Preview không dropout; final path giữ dropout gốc.

Đặt message của token k trong residual space:

\[
u_k^t=W_O^t v_k^t,
\qquad
z_j=h_j+b_O+\sum_{t,k}A_{jk}^t u_k^t.
\]

W_O^t là phần output projection tương ứng head t. Không thêm value bank mới. Preview key toàn context dùng LN(z_j).

**Quy ước bias khi gắn vào RoBERTa:** trong các công thức ở đây, v_k^t là phần Wv^t h_k chưa cộng value bias. Gộp bias chung vào b_O hiệu dụng bằng b_O,eff=b_O,original+sum_t Wo^t bv^t, vì mỗi hàng attention tổng bằng1. Như vậy deletion loại content message chứ không vô tình coi bias chung là thông tin thuộc riêng token i. Q và K vẫn giữ bias gốc. Spec đại số dùng V và bo làm inputs độc lập nên áp dụng nguyên vẹn với quy ước này; parity toàn model còn phải kiểm nếu mở implementation.

Đối với receiver i đọc source j, i≠j, định nghĩa:

\[
C_{j\setminus i}=z_j-\sum_t A_{ji}^t u_i^t
=h_j+b_O+\sum_t\sum_{k\ne i}A_{jk}^t u_k^t.
\]

**Không renormalize A sau loại message.** Operator loại một contribution trong phép tổng hợp, không mô phỏng một lần softmax mới khi token biến mất. Do đó không chia 1−Aji và không gặp singularity khi Aji→1. Với i=j, giữ C=z_j để không đồng thời áp một chính sách bỏ self-update khác.

Key phụ thuộc recipient:

\[
k^s_{j\setminus i}=W_K^s LN(C_{j\setminus i})+b_K^s,
\quad
L^{cav,s}_{ij}=q_i^{s\top}k^s_{j\setminus i}/\sqrt p.
\]

Query lấy từ h_i gốc của block, không lấy preview của i. Final values vẫn từ h_j gốc. Đây là **refinement của routing keys**, không lặp cả transformer block hoặc cập nhật toàn bộ messages trên line graph.

Để kế thừa checkpoint:

\[
L^{new,s}=(1-\eta_s)L^{orig,s}+\eta_s L^{cav,s},
\qquad \eta_s(0)=0.
\]

Eta là scalar trainable không bị ràng buộc trong bản đặc tả tối thiểu; có thể âm hoặc lớn hơn1, nên không gọi đây là convex mixture. Giữ logit mask riêng, blend chỉ finite valid logits rồi áp mask; tránh 0×(−inf). Không dùng coefficient như phần novelty. Có thể cần giới hạn eta nếu runtime probe cho thấy mất ổn định, nhưng chưa âm thầm thêm constraint.

Tính softmax cuối, AV gốc, Wo/residual/LN/FFN gốc. Không sửa loss, không thêm teacher/corpus. Eta0 giữ hàm gốc về toán học, cần matched-runtime baseline để kiểm chứng số học/dropout.

## 4. Tính chất có thể chứng minh

### 4.1. Loại chính xác đường value quay về

Giữ h,Q,A và tất cả v ngoài v_i cố định. Với i≠j, thay v_i bằng v_i+delta làm z_j tăng sum_t Aji^t Wo^t delta^t; phần trừ tăng đúng bằng nó. C và key cavity không đổi, kể cả qua LayerNorm.

\[
\left.\frac{\partial k_{j\setminus i}}{\partial v_i}\right|_{h,Q,A,\,v_{k\ne i}}=0.
\]

Đây là partial-path property tại tensor values. Không khẳng định bất biến khi sửa raw token i: h_i,Q,A có thể đổi; thông tin từ i đã tồn tại trong h_j do các layer trước; final softmax của cả hàng có diagonal entry khác; và original path còn được blend. Không gọi property này là full causal independence.

### 4.2. Không bằng việc mask diagonal

Mask Aii chỉ chặn self-loop i→i. Đường i→j→i với i≠j vẫn còn. Cavity key loại contribution trên đường hai bước ở phần dùng để score. Nó không cấm A đọc B và B đọc A; không có hard ban quan hệ tương hỗ.

### 4.3. Không bằng một penalty Aji đơn giản khi có LayerNorm

Nếu bỏ LN, score cavity bằng full-context score trừ sum_t Aji^t q_iᵀWk Wo^t v_i. Khi có LN, cả mean và variance thay đổi riêng từng cặp; không thể chỉ trừ một bias theo Aji rồi giữ nguyên scale của full context. Thuật toán dưới tính lại scale chính xác, không giữ norm cố định như attribution linearization.

### 4.4. Permutation equivariance ở interface hidden states

Hoán vị đồng bộ h,Q,V,A/masks chỉ hoán vị tương ứng output. Absolute positional embeddings trong backbone vẫn giữ vai trò thứ tự. Equivariance của operator theo token indexing không phải model bất biến với đổi thứ tự từ trong câu.

### 4.5. Không chỉ reparameterize một QK dot-product cùng head width

Một score head thông thường QKᵀ với width p có rank≤p. Sau trừ mean từng hàng, rank vẫn≤p; key bias chỉ thêm row constant nên bị loại. Cavity scores không chịu cùng giới hạn đó, dù tất cả projections vẫn tuyến tính từ một H đầu vào.

Một construction trên giấy: N5,d=p2,H1; x=(-2,-1,0,1,2), h_i=(x_i,−x_i). Chọn Wq=0,bq=(1,1),Wk=I,Wv=5I,Wo=I; các bias còn lại bằngzero. LN gamma=(1,2),beta=0. Khi đó QKᵀ=0, A uniform, tổng values bằng0, z_j=h_j. Với i≠j, C(j\\i)=(x_j−x_i,−x_j+x_i); diagonal giữh_i.

Để tránh mọi nhập nhằng: tất cả biases bằngzero **trừ bq=(1,1)**. Cavity score là −t/(sqrt(2)*sqrt(t²+epsilonLN)), với t=x_j−x_i offdiagonal và t=x_i diagonal. Khi epsilonLN→0+, đặt c=1/sqrt2, score matrix tiến tới:

```
c -c -c -c -c
c +c -c -c -c
c +c  0 -c -c
c +c +c -c -c
c +c +c +c -c
```

Lấy hiệu các hàng (1−0),(3−1),(4−3), rồi trừ mean theo source, và giữ columns1,2,3. Submatrix thu được 2c*(I3−J3/5), có eigenvalues2c,2c,4c/5 nên rank3. Mỗi phần tử score tại epsilonLN=1e-5 khác limit không quá c*epsilonLN/2 (các t kháczero có |t|≥1, t0 luôncho0). Sai lệch của submatrix row-difference/centering bị chặn spectralnorm bởi6c*epsilonLN, nhỏ hơn4c/5; vì vậy rank vẫn≥3>p2.

Đây là chứng minh tồn tại cho operator, không tuyên bố rank lớn luôn tốt, không chứng minh standard transformer nhiều layers không làm được, và không khẳng định rank64 là bottleneck trên text ngắn hiện tại. Rank expansion cũng đã có trong nhiều họ attention khác; không dùng nó làm novelty độc lập. Nó chỉ loại giải thích rằng toàn bộ cavity score là QK cùng width đổi ký hiệu.

## 5. Tính scores mà không lưu tensor N×N×d

Gọi w_j=center(z_j), e_i^t=center(u_i^t), a_{ijt}=Aji^t với diagonal exception. Đặt:

\[
\sigma_{ij}^2=\frac1d\left[\|w_j\|^2
-2\sum_t a_{ijt}\langle w_j,e_i^t\rangle
+\sum_{t,g}a_{ijt}a_{ijg}\langle e_i^t,e_i^g\rangle\right]+\epsilon_{LN}.
\]

Với affine LN gamma,beta và output scoring head s:

\[
r_i^s=\gamma\odot W_K^{s\top}q_i^s,\quad
c_i^s=q_i^{s\top}(W_K^s\beta+b_K^s),
\]

\[
L^{cav,s}_{ij}=\frac1{\sqrt p}
\left[\frac{\langle r_i^s,w_j\rangle
-\sum_t a_{ijt}\langle r_i^s,e_i^t\rangle}{\sigma_{ij}}
+c_i^s\right].
\]

Các thành phần cần có: pair dot tables kích thước N×N×H, per-token head Gram N×H×H, và vectors N×H×d. Không cần edge vectors N×N×d hoặc line-graph adjacency N²×N². Equality là trên số thực, có thể kiểm bằng reference materialize và autodiff gradients.

Chi phí chuẩn bị projection O(Nd²), các cross-token products O(N²Hd), head Gram O(NH²d), contraction O(N²H²). Storage của forward intermediates theo cách reuse là O(N²H+NHd+NH²), chưa tính autograd lifetime. Đây không phải bằng chi phí standard SDPA; H là hệ số thêm đáng kể.

So với **implementation literal** materialize edge states rồi project key per edge, đây là giảm memory và tránh projection d→p trên từng edge. Không so với mọi thuật toán NBA-GNN bằng một con số speedup chưa đo. Sparse GNN có topology khác; chi phí phụ thuộc số cạnh/degree.

Với student N256,d384,H6: tensor cavity literal N²d FP32 là96 MiB/mẫu, so với một N²H table1,5 MiB; cần nhiều tables nên không gọi tổng memory giảm64lần. Chưa đo runtime/peak VRAM hoặc whole-model training memory.

## 6. Câu hỏi tính mới và prior art

| Nguồn | Phần đã biết | Khác biệt đề xuất cần bảo vệ |
|---|---|---|
| NBA-GNN, Park et al. | Exclude reverse message, directed edge state, graph learning | Recipient-conditioned normalized key scores trong dense pretrained attention; không persistent edge states; factorized exact LN computation |
| Line-graph SLU, Duan et al. | Non-backtracking message passing trong NLP/intent-slot graphs | Không xây BIO/intent graph, không edge-to-edge GNN; shared pretrained QKV/Wo/LN operator |
| Local Interpretation via Linear Decomposition | Phân rã attention và LN để attribution | Tính lại pair-specific variance sau deletion, không đóng băng standard deviation hoặc dùng để giải thích model |
| Relational Transformer/Edge Transformer | Pair-specific relation features hoặc edge representations | Edge key ở đây được suy trực tiếp bằng deletion, không learned edge memory; novelty không nằm ở pair-specificity đơn thuần |
| SAOBP, Lee et al., Findings EMNLP2025 | One-step belief propagation tinh chỉnh attention ở BERT nhỏ | Cavity score phụ thuộc residual/value vectors và pair-specific LN, không chỉ biến đổi ma trận A |
| CoTNet, Li et al. | Contextualize keys trước attention | Context của source được loại contribution khác nhau cho từng receiver, không một contextual key dùng chung |
| Deep Implicit Attention, Bal | Mean-field/Onsager self-correction và implicit attention | Không giải fixed point hoặc học node-wise self-correction; explicit recipient-specific deletion trước LN/key projection |

**Broad claim “non-backtracking cho NLU” bị bác bỏ.** Phần ứng viên có thể còn mới là phương pháp tạo/tính chính xác cavity keys với pretrained multi-head projections và LN, không cần lưu dense edge embeddings. Chưa tìm nguồn trùng tổ hợp công thức này trong các nguồn đã đối chiếu. Không tuyên bố chưa từng có trên toàn literature.

Đây là một specialization của nguyên lý message exclusion; phải trình bày công bằng như vậy. Nếu việc specialization và thuật toán factorization không đủ đóng góp theo phản biện khoa học, không được cứu bằng cách gọi một tên BERT mới.

### Đối chiếu SAOBP bằng một phân biệt hàm cụ thể

SAOBP, §4 equations9–11/Algorithm1, lấy A và hệ sốrepulsion để tạo refined attention. Giữ A và hệ số đó cố định thì refined attention của công thức paper không đổi khi thay values. Trong operator đang đề xuất, thay value của một nguồn k khác i có thể đổi C(j\\i), variance và refined logits. Do đó cavity routing không phải chỉ một cách viết lại F(A) của SAOBP. Đây là phân biệt trên interface tensors, không chứng minh superiority hoặc khác mọi extension của BP.

Paper SAOBP có thí nghiệm BERT-Mini/Small/Medium và downstream NLU; vì vậy nó là đối thủ gần bắt buộc, không thể bỏ qua bằng cách nói literature chỉ có decoder/GNN. Tuy nhiên câu paper cho rằng phải pretrain từ đầu khi thay attention không phải định lý cấm chuyển checkpoint; candidate này vẫn cần kiểm identity và thích nghi thực nghiệm.

### Vì sao vẫn giữ để kiểm chứng thuật toán

Khác vòng FFN-response, phép tính ở đây xuất phát từ một điều kiện chính xác trên đường truyền và giữ đúng LayerNorm sau intervention. Đóng góp được đề xuất gồm cả **operator định nghĩa được** và **cách đánh giá operator ấy bằng sufficient dot-product statistics**, thay vì một score heuristic với câu chuyện semantic chưa kiểm chứng. Điều đó đủ để đầu tư phép kiểm đại số nhỏ; chưa đủ để thông qua training hay nhận định thắng benchmark.

## 7. Phản ví dụ và giới hạn

- Quan hệ hai chiều có thể hữu ích cho coreference/agreement hoặc semantic binding; giảm feedback có thể làm chất lượng kém hơn.
- Cavity chỉ bỏ explicit current-preview contribution, không xóa lịch sử context của h_j.
- A phụ thuộc inputs: rerouting effects vẫn tồn tại. Invariance theorem không áp dụng cho toàn raw-input intervention.
- Key cavity đổi miền phân phối: Wk pretrained nhận LN preview thay vì inputhidden cũ. Eta0 giúp giữ checkpoint, không bảo đảm học tốt.
- Không renormalize làm cavity có ít message mass hơn; control phải tách content-specific exclusion với giảm magnitude chung.
- Square-norm expansion có thể catastrophic cancellation khi z gần removed message. Clamp variance>=0 ngănNaN nhưng không chữa sai số hoặc gradients. Phải kiểm FP32 stress, cần stable fallback nếu tiếp tục.
- Chỉ sửa routing key nên value information vẫn có thể chứa quan hệ không mong muốn; không hứa giải quyết tất cả binding.

## 8. Đối chứng và hướng kiểm chứng khi được triển khai

Chưa mở model experiments. Thứ tự đề xuất:

1. Literal-vs-factorized forward/backward, padding, permutation, exclusion-invariance và witness; phần tensors tổng hợp này đã hoàn tất, xem review cuối.
2. Kiểm arithmetic và chi phí trên operator kích thước thực, không cần labels; nếu không ổn định hoặc quá đắt thì sửa thuật toán/loại trước training.
3. Sau review, student single-seed trên5task với cùng E8 recipe, matched baseline và full-context refinement control. Không chọn mỗi task một kiểu.

Control chính dùng Lfull=q_iᵀWk LN(z_j) cùng interpolation eta, weights và final path. Chạy cùng factorized machinery với a=0 nếu cần để khớp thực thi/compute. Cavity phải vượt control này để chứng minh lợi ích exclusion, không chỉ thêm refinement.

Control phụ nếu có tín hiệu: trừ cùng lượng nhưng direction không phải u_i, hoặc giảm đều message mass theo Aji. Mỗi control có confound về norm/direction cần báo; không gọi chúng hoàn toàn tương đương. Identity-LN limit để kiểm reciprocal-bias explanation là phép tách cơ chế, không baseline thay LN toàn model.

Nếu candidate chỉ thắng E8 nhưng không thắng full-context control, đóng góp được hỗ trợ là refinement. Nếu cả hai không tốt, không tự mở rank/layer/eta grids. Nếu candidate có broad gain và hơncontrol, mới chốt additionalseeds/test đúng authorization.

Tại eta0, eta gradient có thể có ngay; oldweights học qua đường gốc, gradient qua cavitypath có coefficientzero. Đây là một warm-start identity, không chứng minh hết gradient starvation. Phiên đầu không MLM/adaptationstage riêng; nếu cần stage mới, baseline nhận cùng data/compute.

## 9. Nguồn đối chiếu và phạm vi truy cập

1. Park et al., *Non-backtracking Graph Neural Networks*, arXiv2310.07430v2 (2024 version). [Methods §3.2, appendixE, complexity§5.4](https://arxiv.org/html/2310.07430). Đã đọc phương trình message và mô tả complexity; không tái lập graph results.
2. Duan, Yu, Tian, *A two-stage line graph reasoning framework for multi-intent spoken language understanding*, Neurocomputing676,133053,2026. [Publisher](https://www.sciencedirect.com/science/article/pii/S0925231226004509). Truy cập được indexed publisher methods/summary, chưa fullpaper. Đủ bác bỏ broad NLP novelty; chưa đủ để kết luận họ không có mọi chi tiết liên quan. Đây là giới hạn còn phải giữ rõ.
3. *Local Interpretation of Transformer Based on Linear Decomposition*, ACL2023. [Paper](https://aclanthology.org/2023.acl-long.572.pdf), §3–4, đặc biệt cách giữ attention scores và standard deviation để linearize.
4. Diao & Loynd, *Relational Attention*, ICLR2023. [Methods](https://arxiv.org/html/2210.05062v3), §3.1–3.2; có directed edge state và conditioningQKV.
5. Bergen et al., *Systematic Generalization with Edge Transformers*, NeurIPS2021. [Paper](https://proceedings.neurips.cc/paper/2021/file/0a4dc6dae338c9cb08947c07581f77a2-Paper.pdf). Triangular edge updates là tiền lệ về edge computation, không phải claim riêng của proposal.
6. Nakyung Lee et al., *Mitigating Attention Localization in Small Scale: Self-Attention Refinement via One-step Belief Propagation*. Findings EMNLP2025. [Paper](https://aclanthology.org/2025.findings-emnlp.578.pdf), §4,§6. Đã đọc message/refined-belief equations và BERT-style setup; không dựa trên thuật ngữ “BP” đơn thuần để kết luận tương đương.
7. Yehao Li, Ting Yao, Yingwei Pan, Tao Mei, *Contextual Transformer Networks for Visual Recognition*. [Paper2021](https://arxiv.org/pdf/2107.12292). Contextual key bằng convolution, không recipient-excluded normalized keys của proposal.
8. Matthias Bal, *Deep Implicit Attention: A Mean-Field Theory Perspective on Attention Mechanisms*, 2021. [Bài viết tác giả](https://mcbal.github.io/post/deep-implicit-attention-a-mean-field-theory-perspective-on-attention-mechanisms/) và [implementation/README](https://github.com/mcbal/deep-implicit-attention). Nguồn tác giả có code, không coi là peer-reviewed paper; đối chiếu self-correction/fixed-point formulation, không mượn nó làm bằng chứng NLU hiệu quả.

Các truy vấn gồm non-backtracking attention/transformer/BERT; cavity attention/softmax; recipient-specific keys; leave-one-out attention/layer normalization; reciprocal attention transpose. Các kết quả chỉ về acoustic echo, physical transformer cavities hoặc leave-one-out crossvalidation không được tính là counterevidence. Search absence không phải bằng chứng tuyệt đối về novelty.
