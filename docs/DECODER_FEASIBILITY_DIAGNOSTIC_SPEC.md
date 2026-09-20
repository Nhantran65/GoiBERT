# Kiểm tra miền quyết định của decoder ABSA

## Mục đích và phạm vi

Kiểm tra một giả thuyết cụ thể: head đã học có loại trừ các tổ hợp aspect ngay cả khi được cấp một latent vector tùy ý sau tanh hay không. Đây là diagnostic hình học, không phải kiến trúc mới, không dùng để tuyên bố encoder thực tế đạt mọi tổ hợp. Không training, không forward encoder, không đọc test. Thực hiện sau khi sửa và hoàn thành phép đo reader Restaurant; dùng cùng agent Luna xhigh.

Đọc sáu checkpoint student E8/BamiBERT E14 của Restaurant, Hotel, ViSFD từ đặc tả reader. Kiểm tra implementation head để xác nhận output affine dùng chung vector sau tanh; lưu tên tensor, shape và SHA256 checkpoint. Không giả định tên khóa.

## Ma trận margin

Với head output z=Wv+b và mỗi aspect a, đặt hàng D_(a,c)=W_(a,c)-W_(a,0), beta_(a,c)=b_(a,c)-b_(a,0), c thuộc các lớp present. Khi đó các margin so với absent là Dv+beta. Bỏ slot OTHERS khỏi claim chính ViSFD và báo rõ điều này. Với head tanh, v nằm trong box mở (-1,1)^d. Dùng float64 cho phân tích CPU.

Lưu singular values, số hàng/cột, numerical rank ở tolerance tiêu chuẩn và hai mức tương đối 1e-8/1e-10. Full row rank chưa đủ chứng minh feasibility trong box.

## Certificate đủ cho mọi tổ hợp

Giải LP min t với Dv0=-beta, -t<=v0_j<=t, t>=0. Nếu có nghiệm t<1 và D full row rank, head có điểm mọi lớp hòa nhau nằm trong interior của box. Một right inverse R của D cho phép tạo v=v0+epsilon R u.

Để mã hóa một tổ hợp nhãn: với aspect absent đặt mọi u_(a,c)=-1; với aspect present lớp y đặt u_(a,y)=1 và các lớp present khác bằng0. Mỗi u có infinity norm<=1. Nếu epsilon*||R||_infinity < 1-||v0||_infinity thì v vẫn trong box và Dv+beta=epsilon u; lớp mong muốn thắng với margin ít nhất epsilon. Vì vậy một certificate đủ có thể bao phủ mọi tổ hợp mà không liệt kê chúng.

Tính R bằng pseudoinverse khi full row rank, và kiểm residual ||DR-I||_infinity. Chọn epsilon bằng một nửa bound trên; kiểm thêm residual Dv0+beta. Numerical report phải nêu tolerance và chỉ gọi là numerical certificate, không phải exact arithmetic proof. Nếu bound cực nhỏ so với residual thì kết quả inconclusive, không pass.

Nếu LP không có interior solution hoặc D thiếu hạng, **không suy ra có tổ hợp bất khả thi**: điều kiện trên chỉ đủ, không cần. Khi đó báo inconclusive; root sẽ quyết định có cần LP theo tổ hợp hay không.

## Diễn giải giới hạn

Certificate dương cho biết output layer có miền quyết định cho mọi tổ hợp trong post-tanh box. Nó không chứng minh affine trước tanh hoặc encoder/LayerNorm có thể tạo ra mọi điểm trong box. Không dùng số chiều hidden lớn như bằng chứng thay cho kiểm rank và box. Không gọi đây là novelty; prior art liên quan là Grivas et al., AAAI2024, Taming the Sigmoid Bottleneck.

## Artifacts

- scripts/research_checks/decoder_feasibility.py
- artifacts/research_diagnostics/decoder_feasibility/summary.json và FACTUAL_REPORT.md
- Fixture CPU cho ma trận full-row-rank có center nội miền; fixture rank deficient; fixture center ngoài box phải inconclusive. Không chạy thí nghiệm model mới.

Ưu tiên hoàn thành reader diagnostic trước; không mở nhiều agent hoặc chiếm thêm GPU cho phép đo này.
