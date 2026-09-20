# Định vị lỗi polarity lan sang aspect không đổi

## Cặp đã tái hiện lỗi

Student E8 Restaurant hiểu đúng standalone nhưng đọc SERVICE#GENERAL thành negative trong cả4biến thể khi FOOD negative và SERVICE positive. E14 Restaurant đúng mọi factorial input. Chỉ xét hai biến thể thứ tự BA (service đứng đầu), connector dấu chấm và ' và '.

Source: 'Nhân viên phục vụ tận tình. Món ăn ngon' hoặc bản nối bằng ' và '.
Base lỗi: 'Nhân viên phục vụ tận tình. Món ăn không ngon' hoặc bản tương ứng.

Source và base có cùng service prefix, exact token IDs và positions trước phần FOOD. Service label1 không đổi; food source1/base2. Đây là lý do chọn BA: tránh confound dịch vị trí khi thêm 'không' trước service. Không patch CLS từ source vì sẽ mang food-positive code sang và dễ cứu B bằng làm sai A.

## Phép đo

Cùng Luna xhigh; frozen E8 Restaurant checkpoint, inference-only, không test/training. Dùng FP32 cho causal patching để giảm ambiguity số học; trước patch xác nhận source/base predictions tái hiện lỗi ở FP32. Nếu FP32 không tái hiện, báo và dừng, không thay câu tìm failure mới.

Lưu source và base hidden outputs từng block. Dùng offsets và exact-prefix token-ID assertion để xác định tất cả service content tokens; không BOS/EOS, không delimiter. Patch block-output service token states từ source sang base lần lượt sau mỗi block0..9. Giữ tất cả vị trí khác, đặc biệt CLS và FOOD tokens của base. Forward phần còn lại; chấm đồng thời FOOD#QUALITY=2 và SERVICE#GENERAL=1. Báo logits/margins mỗi target, joint correctness, không chỉ service rescue.

Đối chứng: replay base service states tại cùng site phải khớp base; patch sau layer cuối phải không đổi head đọc CLS; same-mask/same-shape operations. Không suy causal importance của layer chỉ từ norm/cosine. Lưu source/base vectors ở các site nhỏ cần thiết, patch index masks, IDs, logits và code/checkpoint hashes.

Mục đích là phân biệt lỗi đi qua contextual service states với lỗi CLS aggregation trực tiếp. Rescue joint bằng token patch hỗ trợ đường thứ nhất ở case này; không rescue không chứng minh tokens 'đúng' hoặc đường đó không liên quan vì patch can thiệp chỉ một nhóm states. Không extrapolate2inputs thành lỗi chung cảdataset.

## Ngân sách và artifacts

Phần mở rộng diagnostic chỉ dùng tối đa0.25GPU-hour một GPU rảnh, thực tế khoảng vài chục forwards. Không mở grid/training khi có tín hiệu. scripts/research_checks/controlled_scope_patching.py và artifacts/research_diagnostics/controlled_scope_patching/. Root tự đọc kết quả và quyết định giả thuyết kiến trúc.
