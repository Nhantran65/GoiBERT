# GoiBERT — hướng dẫn cộng tác

## Chỉ dẫn mới nhất của người dùng — ưu tiên hơn các mục bên dưới

Theo yêu cầu ngày 2026-09-17: agent chính tự thực hiện công việc, kể cả kiểm tra, triển khai và chạy phép đo/thí nghiệm trong phạm vi được giao. **Không tạo subagent trừ khi người dùng yêu cầu rõ.** Các quy định bên dưới về giao việc cho subagent không tự tạo quyền delegation. Khi người dùng có yêu cầu delegation, giữ cấu hình model/reasoning đã quy định, trừ khi người dùng chỉ định khác.

## Model mặc định cho subagents thực thi

Theo yêu cầu trực tiếp của người dùng, mọi subagent được giao triển khai code, chạy thí nghiệm hoặc thực thi công việc trong project này phải dùng:

- `model: "gpt-5.6-luna"`
- `reasoning_effort: "xhigh"` (Extra High)

Luôn truyền rõ cả hai trường khi tạo subagent; tuyệt đối không bỏ trống để kế thừa model hoặc reasoning của phiên chính. Quy tắc áp dụng cả khi subagent giao công việc cho subagents khác. Không tiếp tục giao việc thực thi cho agent cũ dùng model kế thừa hoặc model khác bằng follow-up; tạo agent mới với cấu hình trên, trừ khi người dùng chỉ định rõ ngoại lệ.

Với công cụ `collaboration.spawn_agent`, sử dụng `fork_turns: "none"` hoặc một số lượt cụ thể được công cụ hỗ trợ, vì `fork_turns: "all"` không cho override model/reasoning. Khi không fork context, bàn giao rõ mục tiêu, file liên quan, cấu hình đã chốt, trạng thái và tiêu chí hoàn tất.

Không âm thầm chuyển sang model khác hoặc kế thừa phiên chính nếu cấu hình này không khả dụng; báo rõ giới hạn. Chỉ thay lựa chọn model khi người dùng chỉ định rõ ngoại lệ.

Trong phạm vi thí nghiệm E9 đã được người dùng ủy quyền, dùng chính agent Luna xhigh được cấu hình cho toàn bộ triển khai, training và đánh giá; agent chính lập kế hoạch/review và không cần xin lại xác nhận cho các bước cố định đã thống nhất.

## Vai trò nghiên cứu

Agent chính phụ trách kế hoạch, điều phối, đọc bằng chứng và tổng hợp kết quả. Việc triển khai và trực tiếp chạy thí nghiệm model được giao cho subagents thực thi theo cấu hình trên. Giữ nguyên các giới hạn và ngân sách thí nghiệm mà người dùng đã duyệt.

Theo chỉ dẫn cập nhật của người dùng: agent chính tự thực hiện research, đọc và đối chiếu paper, phân tích bằng chứng, diễn giải kết quả và lựa chọn giả thuyết/kiến trúc. Không giao các công việc nghiên cứu hoặc quyết định hướng đi này cho subagent. Subagent chỉ triển khai code, chạy phép đo/thí nghiệm theo đặc tả đã chốt, và trả artifacts/số liệu để agent chính review. Khi người dùng yêu cầu dừng subagent, không tự khởi động lại để tiếp tục research.
