# E12 — Audit lỗi bề mặt trước khi thử nhánh ký tự/dấu

## Quyết định cần hỗ trợ

Kiểm tra xem lỗi E8 có tập trung ở input phân mảnh BPE, đơn vị hiếm hoặc biến thể chính tả sau khi kiểm soát độ dài/nhãn không. Đây là audit chọn hướng kiến trúc, không chứng minh quan hệ nhân quả. E10/E11 không cải thiện rộng; không tự mở training mới chỉ vì một nhóm lỗi có tương quan.

Đúng một subagent GPT5.6Luna xhigh triển khai và thực thi; root review và kết luận. Chỉ dùng train, validation và predictions E8 seed42. Không test, inference model hoặc training encoder mới. CPU đủ; không giữ GPU không cần thiết.

## Dữ liệu và tính đúng

- 5 task: vsfc_sentiment, vsfc_topic, visfd, absa_restaurant, absa_hotel. VSFC có chung văn bản: tính feature một lần, không coi hai task là hai domain độc lập.
- Dùng loader hiện hành với include_test=False. Predictions `artifacts/e8/<task>/seed_42/validation_predictions.json`, tokenizer `models/goiturn-s-gd`. Đối chiếu IDs, gold, số mẫu và metric với best_validation_metrics trước phân tích.
- Giữ tokenizer đầu vào gốc, không normalize input để giả vờ sửa lỗi model. Feature phụ có thể NFC/lowercase nhưng phải ghi rõ; không làm mất dấu trong feature frequency chính.
- Đơn vị chính tả là chuỗi letter/mark/digit liên tiếp, lưu nguyên offsets; punctuation/emoji thống kê riêng. Không gọi tất cả là từ/âm tiết chuẩn.
- Phân mảnh: số token BPE có overlap ký tự với mỗi đơn vị; zero-length special offsets loại. Ghi overlap nhiều đơn vị, Unicode combining marks và đơn vị rỗng. Tokenize đúng maxlen256 cho mô tả input model; thống kê full length riêng nếu cần nhận diện truncation.
- Tần suất chỉ tính từ train. Report rare<=5 và unseen=0 theo đơn vị NFC/lowercase giữ dấu; tách số/foreign/URL nếu nhận diện được bằng quy tắc minh bạch.

## Feature và phân tích

Cho mỗi câu: số đơn vị, số BPE, BPE/unit trung bình, tỷ lệ đơn vị >=3pieces, tỷ lệ rare/unseen, Unicode/emoji/character repetition và thiếu dấu nghi vấn. Ngưỡng high/low lấy train (quartile với tie handling, không ép chia nhóm nếu feature hằng); khóa trước khi xem lỗi validation.

Không tự gọi chuỗi không dấu là lỗi: nhiều đơn vị hợp lệ không có dấu, từ ngoại ngữ cũng vậy. Proxy thiếu dấu chỉ dùng khi đơn vị ít/không có trong train nhưng có đối tác mang dấu trong train với cùng skeleton bỏ dấu; báo cả ambiguity số đối tác, không coi đây là gold normalization. Proxy viết tắt/lặp ký tự cũng không thay thế annotation.

VSFC: error rate theo câu và fixed-label macroF1. ABSA: báo missed/spurious/wrong-polarity, pair microF1, error burden chuẩn hóa theo sốslot/scoredgold; không so raw any-error giữaHotel34aspects vàRestaurant12. ViSFD OTHERS binary được loại khỏi pair-scored metrics như evaluator gốc.

Trình bày high/low sample counts và raw gaps, rồi đối chiếu trong strata độ dài (ngưỡng train), true class cho VSFC; aspect/polarity/gold-positive burden thích hợp choABSA. Có thể dùng matching/standardization gọn, chỉ trên common support, báo coverage/dropped counts. Không đưa vài trăm dòng ở strata cực nhỏ thành bằng chứng mạnh. Không thêm mô hình thống kê phức tạp nếu bảng stratified đã đủ; nêu confound còn lại.

Kèm mẫu đọc tay có seed cố định: khoảng60câu chia domain và cả đúng/sai, có feature/pred/gold; giữ ví dụ phổ biến và ngoại lệ, không chỉ cherry-pick lỗi xấu. Agent gán mô tả sơ bộ và mức chắc chắn; root review ngữ nghĩa. Không tự tạo hoặc đánh giá câu synthetic trong lượt này.

## Kết luận và artifacts

Tạo script isolated `scripts/analyze_e12_surface_errors.py`, output `artifacts/e12/` gồm feature/slice tables CSV, summaryJSON, sample review Markdown/JSON, provenance input hashes, report `docs/E12_DIAGNOSTIC_RESULT.md`. Không sửa frozen source E1–E11, không sửa predictions/gold.

Gate định hướng: chỉ ưu tiên nhánh ký tự khi nhóm liên quan có support đủ, adjusted association cùng chiều trên nhiều domain và ví dụ đọc tay phù hợp cơ chế. Với ví dụ ít, confounded hoặc chỉ riêng một domain, kết luận inconclusive/task-specific thay vì tự động approve10training. Không ngầm diễn giải p-value như xác suất kiến trúc thắng. Báo negative evidence và phân biệt fragmentation,rarity,orthography thay vì gộp chúng.

Gửi root kết quả thực cùng đường dẫn, không dừng ở script/plan. Root kiểm tra rồi quyết định bước sau; không tự launch character model.
