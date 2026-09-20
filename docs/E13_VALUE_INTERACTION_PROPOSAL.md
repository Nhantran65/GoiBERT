# E13 — Residual tương tác giữa token nhận và token gửi

## Ý tưởng kiến trúc

Giữ student GD và head E8. Thêm một nhánh trong attention output để tạo **message phụ thuộc đồng thời vào token nhận i và token gửi j**, rồi cộng phần tương tác vào đường biểu diễn gốc. Trọng tâm chuyển từ thay trọng số attention sang thay nội dung message.

Trong một head self-attention thông thường, output tại i là tổng A[i,j]V[j]. V[j] được dùng chung cho các receiver ở layer đó; A[i,j] quyết định mức đóng góp. Điều này không có nghĩa Transformer không thể biểu diễn quan hệ: hidden states đã có ngữ cảnh và nhiều layer có thể hợp thành thông tin. Giả thuyết ở đây chỉ là một đường biểu diễn tương tác trực tiếp có thể giúp sample efficiency/khái quát hóa tốt hơn trong student.

Ví dụ động cơ là tách “pin máy” khỏi “sạc dự phòng được tặng”, hoặc nhận xét về người quản lý khỏi nhận xét chung về quán. Không đưa các tên category này vào nhánh ở inference; mọi token có thể là receiver/sender, nên cùng kiến trúc dùng được cho VSFC và ABSA. Diagnostic chỉ có thể cho thấy hành vi lỗi, không chứng minh chính value path là nguyên nhân.

## Cơ chế cụ thể — bản đề xuất để kiểm chứng

Chọn một block ở vùng giữa-trên, đề xuất ban đầu block7/index6 của student10blocks. Đây là lựa chọn thí nghiệm, không phải kết quả đã định vị bottleneck. Giữ Q/K/V/Wo và attention map gốc; không thêm head attention thứ hai.

Với h_i, h_j là hidden đầu vào self-attention, đặt q_i=U h_i, k_j=W h_j trong không gian r=64. U và W khác nhau, không bias, nhằm cho phép vai trò có hướng. Tạo residual tương tác:

`r_ij = GELU(q_i + k_j) − GELU(q_i) − GELU(k_j)`.

Đây là mixed finite difference quanh reference0, vì GELU(0)=0. Nếu q_i=0 hoặc k_j=0, r_ij=0; nếu thay GELU bằng hàm tuyến tính, r_ij=0. Những đồng nhất này kiểm tra được bằng unit check. Không gọi đó là “đã loại mọi main effect thống kê” hay phân rã nhân quả: reference0 không phải token đối chứng tự nhiên.

Lấy mean các attention maps gốc qua heads, dropout-off cho weighting nhánh phụ, có mask padding giống input. Với a_ij đã chuẩn hóa, tính:

`delta_i = P sum_j a_ij r_ij`.

Cộng delta vào attention output sau Wo và trước residual+LayerNorm hiện hành. P ánh xạ64→384, không bias, khởi tạo zero; U/W khởi tạo nonzero trong fork_rng. Tại bước0 model giữ nguyên hàm cũ. Các step sau có gradient đi vào P rồi U/W, phải xác minh nhánh thực sự học bằng norm/gradient chứ không chỉ pass init parity.

Đường gốc vẫn có nội dung lexical và attention nhiều head. Nhánh mới chỉ dùng mean attention cho một luồng quan hệ phụ nhỏ; cần ghi rõ hạn chế mất phân biệt giữa head trong luồng phụ. Không tuyên bố mỗi chiều tự tương ứng một vai ngữ pháp hoặc một polarity.

Ba ma trận chính có `3*384*64=73.728` trọng số. Đây là phép đếm thiết kế, không phải số đếm code đã chạy. Bởi vậy chưa cần bỏ student để thử ý tưởng. Có thể tăng r/model về sau, nhưng không đưa scaling vào vòng đầu.

## Khác với các vòng đã thử

| Vòng | Thay đổi | E13 kiểm tra thêm gì |
|---|---|---|
| E9 | Query theo nhãn đọc hidden layer cuối | Tương tác token–token bên trong encoder, head vẫn cũ |
| E10 | Hợp thành span liên tiếp, lựa chọn span | Message nhận biết cả sender/receiver, không cần raw-offset units hoặc span adjacency |
| E11 | Trộn hai attention outputs signed/convex | Giữ map gốc, thay nội dung message bằng residual tương tác phi tuyến |
| E12 | Audit ký tự/rarity | Không giả định thiếu dấu là bottleneck; dùng kết quả audit để tránh ưu tiên character branch thiếu cơ sở |

E13 không được gọi là giải pháp đã chứng minh cho lỗi binding. Khác biệt công thức là điều kiện cần để thử hypothesis khác, không là bằng chứng sẽ tăng F1.

## Prior art và mức novelty có thể nhận

[Query–Value Interaction, Wu et al. 2020](https://arxiv.org/html/2010.03766) đã điều kiện hóa values theo queries; vì vậy “query-aware value” không mới. [Dual Attention Transformer, Altabaa và Lafferty, ICML2025](https://proceedings.mlr.press/v267/altabaa25a.html) tách thông tin quan hệ và nội dung; không thể nhận “thêm relational branch” là phát minh đầu tiên. [Shaw et al. 2018](https://aclanthology.org/N18-2074/) cũng thêm relation representations vào attention. [Gated Attention, Qiu et al. 2025](https://arxiv.org/abs/2505.06708) là đối chứng gần cho điều biến output theo receiver.

Phần đề xuất cụ thể là **residual tương tác được center quanh reference0, gắn vào checkpoint bidirectional sẵn có mà giữ đường routing gốc**, và kiểm tra rằng gain đến từ tương tác chứ không chỉ thêm nonlinear capacity. Chưa tìm thấy trong lượt đọc hiện tại một công trình đúng toàn bộ cấu hình; đây không phải bảo đảm novelty toàn cầu hoặc khả năng công bố. Mixed-difference tự thân là phép toán đã biết.

Không tuyên bố đây là kỹ thuật chỉ có ích cho tiếng Việt. Động cơ xuất phát từ lỗi tiếng Việt của project; nếu muốn claim đặc thù ngôn ngữ, cần đối chứng ngôn ngữ/hiện tượng bổ sung. Mục tiêu gần là một kiến trúc mới đối với baseline project, có đóng góp cơ chế kiểm tra được.

## Đối chứng để tránh một vòng đoán mò khác

Giữ originalGD, E8loss/LR/seed/splits và head. Nếu mở pilot, tối thiểu so candidate với **uncentered pair message**:

`r_control[i,j] = GELU(q_i + k_j)`.

Hai cấu hình cùng U/W/P, r, vị trí, attention map và initialization. Khác biệt duy nhất là subtraction của hai unaryterms; do đó nếu ngang nhau thì chưa có bằng chứng cho ý tưởng centering. Một ablation unary-only `GELU(q_i)+GELU(k_j)` chỉ mở sau khi có tín hiệu để tách tương tác khỏi thêm capacity, không cần train tất cả controls ngay.

Cần kiểm tra chi phí thực. Tensor pair B×N×N×64 có khoảng128MiB ởBF16 khi B32,N256, chỉ một tensor; autograd và intermediates làm tăng thêm. Chunk theo queries để không materialize toàn bộ cùng lúc. SDPA không trực tiếp trả map; việc tính lại QK/softmax cho nhánh phụ có chi phí. Không sửa cảmodel sang eager chỉ để lấy map mọi layer. Profile trước và giữ budget thực thi của hai cấu hình tương đương, báo cả walltime/VRAM.

Vòng training chỉ nên bắt đầu khi diagnostic được review, thiết kế được duyệt và chi phí chấp nhận được. Đề xuất 2configs×5tasks seed42, không mở grid/layer sweep. Không dùng diagnostic synthetic để chọn epoch; vẫn validation benchmark. Chưa chạy training cho E13 trong tài liệu này.

## Điều kiện giữ hoặc bỏ ý tưởng

- Phải có lợi ích trên nhiều task benchmark sạch, không chỉ trên cặp tổng hợp dùng để nghĩ ra thiết kế.
- Phải hơn hoặc có dấu hiệu vượt uncentered control; nếu chỉ hai nhánh cùng tăng, chỉ kết luận thêm pairwise capacity có ích.
- Diagnostic E13A phải được báo cả trường hợp E8 đã đúng và modelmới làm sai. Không thêm câu đến khi đủ “chứng minh bottleneck”.
- Nếu correctionnorm/gradient gần0, báo nhánh chưa được khai thác trong recipe này; không tự đổiλ/LR rồi chọn kết quả đẹp.
- Không dùng một seed để claim SOTA hoặc architecture superiority ổn định. Test/multiseed chỉ dành cấu hình có tín hiệu rộng.

## Trạng thái bằng chứng

Thiết kế trên là giả thuyết của root, dựa vào review lỗi thật và prior art. E13A gồm 30 cặp được chốt ở [spec](E13A_BINDING_DIAGNOSTIC_SPEC.md), đã được đánh giá bằng checkpoint E8/E11 có sẵn, CPU FP32 cùng điều kiện cho cả ba model. Không có training hoặc test mới. Root đã đối chiếu hash script với artifact. Có 60 lượt xuất hiện nhưng chỉ 56 chuỗi riêng biệt; không tìm thấy exact-match với train/validation. Điều đó không chứng minh không có near-duplicates hoặc dữ liệu pretraining liên quan.

| Task | E8 đúng cả cặp | E11 signed đúng cả cặp | E11 convex đúng cả cặp | E8 đúng slot kỳ vọng |
|---|---:|---:|---:|---:|
| Sentiment | 4/6 | 4/6 | 4/6 | 10/12 |
| Topic | 6/6 | 6/6 | 6/6 | 12/12 |
| ViSFD | 4/6 | 4/6 | 4/6 | 15/19 |
| Restaurant | 5/6 | 5/6 | 5/6 | 16/17 |
| Hotel | 3/6 | 3/6 | 3/6 | 12/17 |

Cả ba đạt 22/30 ở metric cặp không có nghĩa predictions giống hệt: trên ViSFD, cả hai E11 đúng 13/19 slot, thấp hơn E8; Hotel signed đúng 11/17, E8/convex đúng 12/17. Do đó không có cải thiện ở metric này để quy cho differential attention. Đã lưu full logits trong `artifacts/e13a/results.json`.

Các lỗi nổi bật được giữ nguyên expectation trước inference: S4 “mong cô tiếp tục cách dạy dễ hiểu này” vẫn negative; V3 phụ kiện được tặng không được gán SER&ACC; R1B “món ăn dở nhưng nhân viên...lịch sự” bị chấm service negative; H1A/H2 lời chê nhân viên đi cùng phòng sạch vẫn dự đoán service positive. E8 xử lý đúng các cặp Topic và nhiều cặp đổi aspect/polarity khác, nên không gọi là thất bại binding toàn diện.

S3 “không hề khó hiểu” có sắc thái positive/neutral và H6 giá phòng so với giá bữa sáng có thể phụ thuộc ontology; chúng không phải bằng chứng mạnh khi đứng riêng. H1/H2 chia sẻ một input, không phải hai lỗi độc lập. Các điều này giới hạn sức mạnh suy luận nhân quả và khái quát hóa của bộ diagnostic.

**Quyết định:** đủ để hình thành một candidate cụ thể về tương tác message, chưa đủ để kết luận đây là kiến trúc phải thắng rộng. Không tiếp tục sửa gate E11 hoặc thêm character branch vì thiếu bằng chứng. Đề xuất một pilot candidate so với uncentered control theo mục trên, không triển khai trong lượt tạo ý tưởng này. Nếu muốn khẳng định novelty để công bố, cần thêm rà soát prior art và ablation; không gắn nhãn SOTA khi chưa có kết quả.
