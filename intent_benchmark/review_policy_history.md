# Bản nháp đã supersede — không dùng làm rubric

# Quyết định rubric của benchmark

## Cập nhật có hiệu lực — ưu tiên prompt có trong snapshot

Agent tìm được bản tài liệu `src/modules/ai/docs/technicals/prompts_doc/production/INTENT_CLASSIFIER_PROMPT.md` trong bundle. Root đã đọc. Bản này là nguồn định nghĩa tốt hơn các suy đoán rộng bên dưới, dù không chứng minh byte-identical với prompt DB từng tạo expectedIntent. **Các quyết định rộng bên dưới là bản nháp đã bị supersede ở các điểm sau:**

- Không mass-relabel OTHER của draft edits/customizations/quantity updates thành MENU_QUERY. MENU_QUERY dành cho concrete menu/product/price/availability/recommendation/browsing/selection. Các thao tác operational không rõ là truy vấn/chọn món giữ OTHER hoặc đánh dấu policy ambiguity.
- ADDRESS chỉ delivery-address detail/correction, không mọi địa điểm. Store directions/address queries0062/0063 ->OTHER; invoice address0019 ->OTHER.
- 0009 thêm vào order đang tồn tại nhưng không có visible history: giữ originalOTHER có context/state uncertainty, không ép MENU_QUERY.
- 0222 requests summary không phải positive confirmation. Mixed checkout time/COD+summary ->OTHER với ambiguity note nếu cần; pure draft price/summary queries vốn MENU_QUERY có thể giữ theo ordering-query convention.
- Neutral cancellation0043/0045 ->OTHER; explicit complaint/frustration vẫn COMPLAINT. 'Ngon quá' appreciation và 'Bắt đầu' opening giữ nhãn gốc hợp rubric.
- PHONE requests/references trong phone flow giữ khi có căn cứ; không chỉ numeric-only. Store-phone request0015 giữPHONE nhưng nêu convention.

Per-row review ledger và final rubric phải theo cập nhật này. Không dùng đoạn lịch sử bên dưới làm mapping tự động. Các đề xuất chưa chắc giữ nguyên và flag; không tự tạo ngữ cảnh bot/state.

## Bản nháp lịch sử trước khi đọc prompt snapshot

Root đã đọc toàn bộ250cases/512target turns trong bundle. Review nhằm tạo nhãn intent nhất quán theo nội dung nhìn thấy, không sao chép mọi output của classifier cũ hoặc trạng thái backend ẩn. Đây là AI-assisted review, không phải human gold. Production prompt trong DB không có trong snapshot.

## Quy tắc chính

- MENU_QUERY: hỏi menu/món/giá/promotions; đặt hoặc thêm món; chọn/đổi/bỏ món, sửa số lượng/customization; hỏi nội dung giỏ/tổng/summary. Shipping fee hoặc điều kiện giao món không kèm địa điểm cụ thể được giữ trong phạm vi này để nhất quán quy ước broad ordering-query của bundle.
- ADDRESS: cung cấp/sửa/hỏi địa chỉ, vị trí, đường đi; địa điểm giao cụ thể ngay cả khi kèm hỏi phí. Không suy ra địa chỉ mới từ fixture.
- PHONE: cung cấp/sửa/xác nhận/tham chiếu hoặc yêu cầu số điện thoại; gồm từ chối cung cấp số trong phone flow. Không yêu cầu chuỗi phải là số hợp lệ.
- COMPLAINT: mô tả sự cố, bất mãn, giao sai/thiếu/trễ, tranh chấp thanh toán. Không gán chỉ vì có 'sai', 'nhầm', 'lâu': sửa số lượng do mình nói nhầm là MENU_QUERY; 'lên lâu' có thể là lỗi gõ 'lên lầu'.
- CONFIRMATION: xác nhận/đồng ý/chốt/đóng lựa chọn, gồm 'đủ rồi', 'xong', 'vậy thôi' khi history ordering hỗ trợ. Không đồng nghĩa có quyền tạo đơn thật. Yêu cầu 'lên summary giúp' là query, không confirmation dù có giờ giao và COD.
- GRATITUDE: cảm ơn hoặc lời tán thưởng đơn thuần ('Ngon quá!'); lời cảm ơn kèm hành động cụ thể không tự lấn át hành động đó.
- GREETING: chào/gọi mở hội thoại, 'Bắt đầu'. Greeting mở đầu câu có yêu cầu cụ thể không lấn át yêu cầu.
- OTHER: đặt/hủy bàn, hủy toàn đơn trung tính, hỏi trạng thái đơn không phàn nàn, thanh toán/top-up/hóa đơn, giờ hoạt động, tuyển dụng, fulfillment/time/payment updates thuần túy và ngoài8lớp.

Không map OTHER thành MENU_QUERY chỉ vì có tên món: hủy toàn đơn và sửa món trong giỏ là hai hành động khác nhau. Ngược lại, không giữ OTHER cho mọi draft edit chỉ vì classifier cũ đã trả OTHER.

## Các điểm neo review

- 0009: yêu cầu thêm cơm -> MENU_QUERY; không dùng initialStateFixture ẩn để ép intent.
- 0043/0045: hủy đơn/bàn trung tính -> OTHER. 0042/0044/0048 có problem/frustration -> COMPLAINT.
- 0063: hỏi địa chỉ quán -> ADDRESS, nhất quán0062 directions.
- 0136: thay đổi món đã chọn -> MENU_QUERY.
- 0222:t2/t3 theo thứ tự người đọc(0based index2/3): yêu cầu summary -> MENU_QUERY, không confirmation.
- 0081 'đem...lên lâu': cần flag ambiguous upstairs-vs-delay, không regex-label complaint.
- 0019 invoice+address, 0050 noisy cancel wording, 0139 membership-link, 0240 cancel do tự đặt nhầm: cần lưu uncertainty và quyết định cụ thể, không gọi tất cả là high-confidence.

Những dòng ambiguity cần giữ nguyên original nếu chưa có cơ sở phân xử rõ, kèm reason. Mọi thay đổi cần per-row ledger; original labels không bị ghi đè ở dữ liệu nguồn.
