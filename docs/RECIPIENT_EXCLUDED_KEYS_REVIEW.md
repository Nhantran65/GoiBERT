# Review nghiên cứu: attention với key loại đóng góp từ người nhận

## Quyết định nghiên cứu

**Giữ ứng viên này sau vòng thiết kế và đối chiếu tính mới để chuẩn bị một phép thử triển khai nhỏ.** Khác các vòng vừa loại, đã có một operator xác định bằng điều kiện đường truyền, thuật toán tính chính xác trên số thực, bằng chứng phân biệt với QK cùng width và phép kiểm forward/backward độc lập.

Đóng góp được đề xuất là **recipient-conditioned cavity keys với LayerNorm được tính lại đúng sau deletion, cùng cách đánh giá scores bằng bảng tích vô hướng thay vì lưu edge embeddings d chiều**. Nguyên lý non-backtracking đã có; không tuyên bố phát minh nguyên lý đó. Novelty ở đây là nhận định nghiên cứu có phạm vi từ các nguồn đã đối chiếu, không phải bảo đảm không tồn tại công trình trùng ở bất kỳ nơi nào.

Không có kết quả model hoặc F1 mới. Việc giữ một ý tưởng không đồng nghĩa nó sẽ thắng BamiBERT, hoặc đã đủ một paper được chấp nhận. Mục tiêu hiện tại là tìm một đề xuất mới cụ thể đứng vững qua phản biện bước1–2; giai đoạn quality evaluation vẫn ở phía trước.

## 1. Ý tưởng dễ hiểu

A gửi một message cho B. B cũng nhận message từ C và các token khác. Nếu A sau đó đánh giá B có phải nguồn thông tin phù hợp không, key thông thường sau refinement chứa cả message vừa đến từ A.

Operator mới tạo riêng **key của B khi được A đọc** bằng cách bỏ contribution A→B khỏi preview của B, rồi tính lại LayerNorm và key projection. Key của B khi C đọc sẽ loại contribution C→B, nên không nhất thiết giống key dành cho A. Values cuối vẫn dùng ngân hàng pretrained gốc.

Mục tiêu là thêm context cho routing mà tách được thông tin đến từ người nhận khỏi thông tin đến từ nơi khác. Chưa có bằng chứng rằng vòng phản hồi này là lỗi chủ đạo trong GoiBERT; đó là giả thuyết cần test.

## 2. Phép tính và phần đã chứng minh

Với Aji^t là trọng số j nhận i ở preview, u_i^t là content message qua value/output projections:

\[
C_{j\setminus i}=h_j+b_{eff}+\sum_t\sum_{k\ne i}A_{jk}^t u_k^t,
\qquad
L_{ij}^{cav,s}=q_i^{s\top}[W_K^s LN(C_{j\setminus i})+b_K^s]/\sqrt p.
\]

Value bias chung được gộp vào b_eff, không bị coi là contribution sở hữu riêng của i. i=j giữ self-preview bình thường. Không renormalize attention mass sau deletion. Công thức và phép chuyển checkpoint đầy đủ ở [proposal](RECIPIENT_EXCLUDED_KEYS_PROPOSAL.md).

Ba kết quả toán học:

1. Với inputs h,Q,A cố định, key cho receiver i không phụ thuộc vào explicit value contribution v_i vừa đi qua source j. Mean và variance LN đều được tính sau deletion, nên property không dựa trên frozen-normalization approximation.
2. Mask diagonal Aii không thực hiện cùng việc này: đường i→j→i vẫn có thể tồn tại dù mọi self-edges bằngzero.
3. Có construction với d=p2,N5 làm row-centered cavity score có rank≥3. Do đó không chỉ là một QK dot-product width2 được viết lại. Không suy ra standard transformer nhiều layers không làm được, hoặc rank cao luôn có ích.

Đây không phải full independence khi sửa raw token i. A,Q,hidden states cũ có thể thay đổi; các layer trước đã trộn thông tin; original logits vẫn hiện diện khi interpolation chưa về cavity hoàn toàn. Không gọi cả encoder là non-backtracking.

## 3. Vì sao không chỉ áp lại một phương pháp có sẵn?

| Tiền lệ gần | Phần phải ghi nhận | Phần khác biệt cụ thể |
|---|---|---|
| NBA-GNN | Reverse-message exclusion và directed messages | Operator ở đây tạo key score trong dense pretrained attention, không duy trì learned edge states; có factorization riêng cho affine LN |
| SAOBP trên BERT nhỏ | Một bước BP tinh chỉnh attention, có NLU experiments | SAOBP biến đổi A; cavity keys còn phụ thuộc content values và residual geometry, nên không quy về cùng F(A) |
| CoTNet | Contextualized keys | Recipient-specific deletion thay vì một key contextualized dùng chung |
| Relational/Edge Transformer | Pair-specific relations | Pair state được xác định bằng exclusion có identity rõ, không một relation vector tự do học thêm |
| Attribution decomposition | Tách các contribution quanh attention/LN | Tính lại variance mỗi cặp thay vì giữ standard deviation để linearize |
| Deep Implicit Attention | Self-correction/feedback từ mean-field reasoning | Không fixed-point solver hoặc node-wise learned correction; explicit source contribution removal trong key |

Các nguồn chính đã đọc: [NBA-GNN](https://arxiv.org/html/2310.07430), [SAOBP](https://aclanthology.org/2025.findings-emnlp.578.pdf), [CoTNet](https://arxiv.org/pdf/2107.12292), [Relational Attention](https://arxiv.org/html/2210.05062v3), [linear decomposition](https://aclanthology.org/2023.acl-long.572.pdf), và [Deep Implicit Attention của tác giả](https://github.com/mcbal/deep-implicit-attention).

Còn một giới hạn tài liệu: [line-graph SLU2026](https://www.sciencedirect.com/science/article/pii/S0925231226004509) chỉ truy cập được publisher content được index, chưa fullpaper. Nó đủ bác bỏ claim “lần đầu non-backtracking trong NLP”, nhưng không cho phép khẳng định mọi implementation detail đều khác. Không giấu giới hạn này khi phát triển paper.

## 4. Bằng chứng phép kiểm tensor

Implementation agent gpt-5.6-luna/xhigh chỉ viết/chạy phép kiểm CPU theo spec của root; không research, không tải model/dataset hoặc optimizer step. Root đọc code, đối chiếu công thức, raw JSON và phạm vi các checks, yêu cầu sửa broadcasting/reference labels trước khi chốt.

| Phép kiểm | Kết quả |
|---|---|
| Literal vs factorized forward FP64,12cases | Max absolute error5,33e-15 |
| Backward FP64,10nhóm inputs/weights | Max absolute error1,42e-14 |
| Permutation equivariance | Pass trong case đã chốt |
| Perturb value của recipient, giữ routing | Cavity score change0; full-context change0,447692 trong witness |
| Mask toàn diagonal | Vẫn có ranking reversal giữa full-context và cavity |
| Rank witness riêng N9,p3, projections cùng input H | Row-centered cavity rank8; full-context rank2, threshold1e-8 |

Trong ranking witness do root thiết kế, full-context cho nguồn chứa message quay về score1,0690, cao hơn nguồn khác0,3922. Cavity hạ nguồn quay về xuống0, giữ nguồn kia0,3922. Đây là một ví dụ xây dựng để phân biệt computation, không một sample benchmark hoặc accuracy measurement.

Nguồn: [audit đầu](../artifacts/research_checks/cavity_key_algebra/REPORT.md), [raw results](../artifacts/research_checks/cavity_key_algebra/results.json). Định nghĩa phép kiểm trong [spec đại số](CAVITY_KEY_ALGEBRA_SPEC.md).

## 5. Số học: vấn đề đã tìm thấy và cách xử lý

Factorization tính variance bằng hiệu các squared norms. FP32 có thể mất chính xác khi tổng và contribution bị loại gần bằng nhau. Ca stress có messages được khuếch đại cho max absolute score error2,79648, relativeL2 khoảng0,109762. Đây là failure có thật trong tensors tổng hợp, đã giữ nguyên trong audit đầu.

Đặc tả bổ sung dùng tổng độ lớn các hạng của variance để đánh dấu cặp có nguy cơ cancellation. Những cặp đó dùng công thức trực tiếp theo chunk tối đa64pairs; còn lại dùng factorized path. Không đổi operator, không sửa kết quả bằng gold hoặc benchmark. Guard không phải một theorem về mọi floating-point input.

Kết quả kiểm guarded và mã nguồn được lưu riêng tại [report](../artifacts/research_checks/cavity_key_guarded/REPORT.md), [raw JSON](../artifacts/research_checks/cavity_key_guarded/results.json), [spec](CAVITY_KEY_GUARDED_SPEC.md). Root đã chốt version cuối sau khi yêu cầu sửa tổng trị tuyệt đối của từng Gram term đúng spec. Ở ca scaled-cancellation, max absolute error giảm từ2,79648 xuống1,90735e-6, relativeL2 từ0,109762 xuống1,09977e-7; chỉ1/16pairs dùng fallback. Sáu cases FP32 pass criterion absdiff≤1e-5+1e-4*abs(reference), được khai báo trước run bổ sung. Đây không phải nới hoặc xóa các strict-FP64-threshold failures trong audit cũ.

Hai cases well-conditioned và một near-cancellation FP64 đều pass forward/backward trên đủ10nhóm inputs/weights, gradients hữu hạn. Root kiểm raw fields, đọc selected-branch code và đối chiếu SHA thực: script guarded `0eb2d8a6ef53f7dd204146798a40e6cb5799b637f7a10ab3264aac33307af6d0`; source audit cũ vẫn `e2f292b575a9eb35b2e62d2554f8da3e988c7795d10e342fcd7d768a311ae4c1`. Recovery khi overflow/nonfinite vẫn **not_verified**; không còn wording khẳng định miễn nhiễm NaN gradients.

Không gọi tensor workspace ratio là GPU speedup. Không có actual model-sized kernel/VRAM profile. Nếu nhiều pairs cần fallback, runtime có thể gần literal computation; autograd cũng có thể giữ activations từ nhiều chunks. Một triển khai model phải profile thực và xử lý nonfinite fail-closed, không mặc định các cases đã test phủ mọi tình huống.

## 6. Hướng thí nghiệm đề xuất, chưa khởi chạy

Student GD là điểm bắt đầu vì có baseline E8 và frozen evaluator, không phải vì đã chứng minh student có lỗi này. Chỉ một upper-middle block được chọn trước kết quả; eta0 giữ hàm gốc, toàn encoder vẫn trainable. Không corpus/MLM hoặc task KD mới.

Thử nhỏ cần trả lời ba câu hỏi theo thứ tự:

1. Operator trên shape student có ổn định, giữ checkpoint và chi phí chấp nhận được không?
2. Cavity refinement có tốt hơn **full-context key refinement** cùng weights, interpolation và runtime machinery không?
3. Gain có trải trên đủ5task hay chỉ một domain?

Matched baseline, full-context refinement và cavity refinement là ba arms cốt lõi. Giữ E8 loss/LLRD để so student trong cùng recipe; không lấy E14/E16 khác head/runtime làm architecture-only controls. Một seed trước; chỉ hướng có tín hiệu mới mở confirmation. Chưa tự chạy test hoặc hyperparameter grid.

Nếu chỉ thắng baseline nhưng ngang full-context control, đóng góp được hỗ trợ là computation thêm. Nếu thua vì xóa feedback hữu ích, loại candidate thay vì gọi dữ liệu không phù hợp. Nếu vượt control rộng, mới đầu tư multi-seed và đối đầu original BamiBERT cùng pipeline.

## 7. Audit kết thúc giai đoạn tìm ý tưởng

| Yêu cầu nghiên cứu | Bằng chứng và trạng thái |
|---|---|
| Tự lặp thiết kế và phản biện, không đợi người dùng nhắc từng bước | Vòng3 đã đi từ condition→operator→prior art→algebra→phát hiện lỗi số học→guarded verification trong cùng công việc |
| Có một ý tưởng cụ thể, không chỉ tên block hoặc hyperparameter | Cavity-key forward, bias/mask/init/normalization policy, exact factorization và controls ở proposal |
| Đối chiếu tiền lệ, không đổi tên phương pháp cũ để claim broad novelty | Claimchart ghi rõ nền NBA/BP đã biết; claim mới giới hạn ở normalized-key operator và đánh giá factorized; SAOBP/contextual-key/attribution/TAP được phân biệt |
| Có tính chất phân biệt kiểm chứng được | Partial-path exclusion, diagonal-mask witness, rank construction; literal/factorized forward/backward checks |
| Không bỏ qua implementation impossibility hoặc numerical failure | Phát hiện FP32 cancellation, giữ rawfailure, chốt guarded evaluator; GPU/model-size và nonfinite recovery vẫn là deployment work tương lai |
| Có hướng kiểm tra utility trên student và đủ5task | Three-arm protocol outline, same-recipe controls, one-seed trước; chưa coi utility được chứng minh |

Quyết định cuối: đủ để kết thúc **giai đoạn tìm một ứng viên mới đã được phản biện** và bàn giao ý tưởng này cho thảo luận/triển khai tiếp. Chưa kết thúc mục tiêu chất lượng dài hạn của GoiBERT; chưa có test benchmark mới hoặc claim SOTA. Novelty vẫn phải được trình bày có phạm vi và có thể bị bác bỏ bởi prior art bổ sung, như mọi đề xuất nghiên cứu chưa xuất bản.
