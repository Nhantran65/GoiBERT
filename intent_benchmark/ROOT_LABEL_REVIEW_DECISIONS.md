# Quyết định rà nhãn cuối của root

Root đã đọc250cases/512turns, bao gồm507lượt có nhãn. Đây là AI-assisted review, không human gold. Nguồn rubric là bản INTENT_CLASSIFIER_PROMPT.md có trong snapshot và taxonomy8lớp; không khẳng định bản tài liệu khớp byte-for-byte với promptDB từng tạo nhãn. Bản nháp cũ được giữ trong review_policy_history.md và không còn hiệu lực.

## Ranh giới nhãn

- GREETING: chào/gọi mở hội thoại, Bắt đầu. Không lấn át một yêu cầu cụ thể chỉ vì có lời chào đầu câu.
- GRATITUDE: cảm ơn/appreciation đơn thuần, gồm Ngon quá. Hành động chính trong câu mixed intent được ưu tiên.
- CONFIRMATION: đồng ý/xác nhận rõ. Yêu cầu tạo summary không phải confirmation; intent này cũng không tương đương quyền tạo đơn thật.
- MENU_QUERY: menu/product/category/price/availability/recommendation, browsing hoặc lựa chọn món mới. Pure price/updated-total questions được giữ ở đây. Không tự map mọi draft operational edit thành MENU_QUERY.
- COMPLAINT: dissatisfaction/service problems/refund disputes. Không suy từ một từ sai/nhầm/lâu đơn lẻ. Neutral cancellation không mặc định complaint.
- ADDRESS: delivery-address details/corrections/destination; không store directions hoặc invoice company address.
- PHONE: phone-number details/corrections/requests/references trong phone flow. Không yêu cầu số phải hợp lệ; giữ placeholder.
- OTHER: reservation, neutral full-order cancellation, payment/topup/invoice, status/checkout/cart operations, storehours/location và các hành vi không thuộc7lớp trên.

## Quyết định ranh giới cần áp dụng nhất quán

- Pure total/updated-price question: MENU_QUERY, gồm003:t3 và214/216/217/219/221:t4,225:t7,226:t7,227:t8,230:t9 (indices0based).
- Pure regenerate/recap summary: OTHER, gồm215/218/220:t4,228:t7,229:t7. Mixed time/COD+summary: OTHER, gồm222:t2/t3.
-0019 invoice address ->OTHER. 0062/0063 store directions/address ->OTHER.
-0043/0045 neutral cancellation ->OTHER;0042/0044/0048 explicitfrustration ->COMPLAINT.
-0047 expensive-shipping objection ->COMPLAINT với uncertainty do mixed refusal/feedback.
-0009 và0136 giữ originalOTHER, flag order-state/selection ambiguity. Không suy opaque initialStateFixture thành context model.
-0050 giữ originalCOMPLAINT, flag câu noisy không đủ để chắc là cancellation-status query.
-0081 giữ originalCOMPLAINT nhưng flag lỗi gõ lên lâu/lên lầu; không gọi chắc chắn complaint.
-0233:t1 giữGREETING với uncertainty: chủ yếu chào/reopen, chưa chọn sản phẩm cụ thể.
-Other draft edits/variant-only answers có competinginterpretation được giữ conservative và flag, không mass-relabel chỉ dựafoodkeywords.

## Sử dụng

Root đã review ledger507rows được xuất: chấp nhận19thay đổi, bao gồm181/183/184/185/189 các lời đóng lựa chọn ->CONFIRMATION theo history và0240 self-cancellation ->OTHER. Chấp nhận82uncertainty flags ở lượt review này; những câu0037–0041,0081,0206 giữ original theo quyết định conservative đã xử lý, không còn pendingroot. Nhãn revised counts: OTHER211, MENU_QUERY144, ADDRESS38, CONFIRMATION44, COMPLAINT35, GREETING12, PHONE12, GRATITUDE11. Mọi count cuối phải được xác nhận lại bằng script/audit trước bàn giao.

Ledger phải lưu original/reviewed label, rationale và uncertainty từngsample. Mọi mẫu uncertain vẫn nằm trong507mẫu để không tự mở rộng phạm vi loại bỏ ngoài5missinglabels; có thể phân tích sensitivity sau này. Split không được dùng nhãn ở futureturn hay fixture vào model_input. README và ledger cuối là điểm truy cập cho người dùng; các draft không phải targetlabels.
