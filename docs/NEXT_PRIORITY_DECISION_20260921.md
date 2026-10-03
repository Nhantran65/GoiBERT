# Quyết định ưu tiên sau deep research — 2026-09-21

**Ưu tiên mục tiêu ≤1B vượt BamiBERT/CafeBERT trên đủ năm task.** Nhánh ≤100M giữ làm giai đoạn sau, không dùng Bami 99,5M như thành tựu thu nhỏ có ý nghĩa. Đây là quyết định phân bổ nghiên cứu, chưa phải kết quả thực nghiệm mới hoặc cam kết sẽ thắng.

## Vì sao chọn nhánh lớn trước?

- CafeBERT (560M) đã dưới trần1B và đang là nền gần mốc nhất trong các model được đo đủ5task. XLM-R-large gốc cùng cỡ bằng Cafe trên Restaurant mean2seed (77,01 vs77,06) và ViSFD một seed (85,48 vs85,47); Topic thua Cafe1,13 một seed. Nhưng XLM-R-large chưa đo Sentiment và Hotel. [Large baseline](XLMR_LARGE_BASELINE_RESULT.md).
- Trên test một seed đã mở, Cafe còn **thua Bami ở Sentiment** (82,36 vs83,03) dù hơn ở bốn task còn lại. Mục tiêu “thắng cả hai đủ5task” phải xử lý Sentiment, không chỉ Restaurant/ViSFD. [Benchmark](EXPERIMENTS_SUMMARY_STANDALONE.md).
- Nhánh nhỏ đòi hỏi hai thành công cùng lúc: giảm thật ~103M về khoảng80–90M và không mất chất lượng, sau đó tăng đủ5task. B8 Restaurant71,09; MiniLM28–39M Restaurant62–65; B12 120M Restaurant73,54. Không có candidate hiện tại gần thắng Bami đủ5task. [Student](VOCAB_TRIM_STUDENT_RESULT.md).
- Một teacher lớn tốt hơn có thể phục vụ nghiên cứu distillation sau, nhưng không giả định distillation sẽ tự chuyển mọi gain xuống student.

## Bước thực hiện đề xuất cho nhánh lớn

**Bước 0: hoàn chỉnh bản đồ baseline.** Fine-tune XLM-R-large pretrained gốc trên hai task thiếu: VSFC Sentiment và ABSA Hotel, bằng recipe E14/grid LR đã dùng trên các backbone khác. Validation-only, không test; đối chiếu Cafe/Bami trên đủ năm task, quyết định Cafe hay XLM-R-large làm nền. Cắt vocab Large348,9M hiện mới chứng minh logits bằng nhau cho Restaurant checkpoint đã fine-tune; muốn dùng bản trimmed làm encoder chung phải kiểm từng task riêng. Không lấy Restaurant đại diện cho năm task.

**Bước 1: thử supervision hiện có trên nền đã chọn, không tạo corpus mới.** Một pilot gồm hai phép chuyển nhãn có cơ sở nhất: Restaurant↔Hotel với shared encoder/head riêng; VSFC Sentiment↔Topic trên cùng văn bản với hai head. Mỗi phép phải có target-only control **cùng số optimizer steps, lượng target data, LR selection và seed**. Khi domain/ontology khác nhau, không gán nhãn không có annotation thành absent. ViSFD vẫn giữ baseline ở giai đoạn này, không claim thắng năm task từ gain ở bốn task. Nếu không có gain trên các task được nhắm, dừng dạng multitask đó.

**Bước 2 chỉ khi cần:** thử dữ liệu có nhãn mới/synthetic đúng ontology và kiểm chất lượng, ưu tiên nhóm lỗi còn headroom; hoặc pilot architecture replacement có giới hạn. [LACA](https://aclanthology.org/2025.acl-long.41/) là tiền lệ để thử augmentation nhưng setting cross-lingual của paper khác của ta. [Talking-Heads](https://arxiv.org/html/2003.02436) là phép thử operator chưa chạy, nhưng phương pháp gốc đã có và gain của paper đến từ model được pretrain với operator đó. Không gộp augmentation, multitask, operator và KD vào một arm đầu tiên.

**Xác nhận:** chỉ candidate có tín hiệu validation rõ mới chạy ít nhất3seed trên tất cả task mà mục tiêu đặt ra. Khóa recipe trước test; theo benchmark protocol hiện có, final ưu tiên5seed. Báo từng task và paired comparison; không chọn mô hình khác nhau cho từng task rồi gọi là một encoder duy nhất. Test hiện đã được quan sát nhiều lần, nên claim SOTA công khai cần thêm đối chứng split/metric và holdout ngoài vòng phát triển. [Protocol](BENCHMARK_PROTOCOL.md).

## Khi nào quay sang mục tiêu ≤100M?

- Nếu nhánh lớn tạo teacher/recipe có gain lặp lại, bắt đầu student **~80–90M**, không headline Bami99,5M. Đối chứng structured FFN pruning của Bami (giữ12lớp; FFN3072→2304 ≈88,9M gồm head Hotel) với 12-layer XLM-R student được nén/distill; tất cả phải là checkpoint export thật, không chỉ đếm config. [Nghiên cứu architecture](ARCHITECTURE_ONLY_DEEP_RESEARCH_20260921.md).
- Nếu large không cải thiện, vẫn có thể nghiên cứu small như mục tiêu efficiency riêng, nhưng xác suất vừa nhỏ hơn đáng kể vừa thắng Bami đủ năm task thấp theo dữ liệu hiện có. Không mặc định chuyển toàn bộ ngân sách sang KD.

## Ranh giới giữa điểm benchmark và đóng góp nghiên cứu

Multitask và synthetic data có thể tăng điểm mà không tạo kiến trúc encoder mới. Pruning, vocabulary trimming và Talking-Heads đều có prior art. Nếu project vẫn yêu cầu **encoder mới có tính mới**, phải có cơ chế/đối chứng giải thích phần gain và đánh giá ngoài năm task local, không chỉ đặt tên mới cho checkpoint đã fine-tune.

Quyết định này cập nhật mức ưu tiên trong [deep research hai nhánh](TWO_ROUTES_DEEP_RESEARCH_20260921.md); các báo cáo cũ giữ nguyên làm lịch sử. Chưa launch bất kỳ training/test nào trong lượt quyết định này.

## Cập nhật 2026-09-22 sau Bước 0

Đã hoàn tất XLM-R-large Sentiment và Hotel với ba LR, seed42, validation-only: **Sentiment 86,73** và **Hotel 73,66** best-of-grid. So Cafe cùng seed42, Large **−0,96 Sentiment, −1,13 Topic, +0,02 ViSFD, +0,10 Restaurant, +0,32 Hotel**; Restaurant mean2seed Large77,01 vs Cafe77,06. Hotel cùng LR2e-5 chỉ +0,23 so Cafe. Các chênh nhỏ chưa phân biệt chắc chắn. [Báo cáo và recovery](XLMR_LARGE_FIVE_TASK_EXTENSION_RESULT.md).

**Chọn CafeBERT làm nền làm việc cho pilot tiếp theo** vì nó cao hơn ở Sentiment và Topic trên validation hiện có, trong khi Large không cho lợi thế rõ trên các task khác. Đây là quyết định thiết kế pilot, không phải kết luận Cafe hơn Large có ý nghĩa thống kê trên mọi task/test. Bước 1 ở trên vẫn là đề xuất chưa launch; không dùng các ô mới để mở test hay đổi metric.

Pilot đầu được đặc tả cụ thể tại [Restaurant–Hotel joint learning trên Cafe](CAFEBERT_RESTAURANT_HOTEL_JOINT_PILOT_PLAN_20260922.md). Đây là phép thử transfer từ nhãn train sẵn có; không đủ một mình để đạt mục tiêu thắng năm task hoặc tính mới kiến trúc.

**Đổi ưu tiên theo phản hồi mới của người dùng:** pilot Restaurant–Hotel được tạm gác vì chỉ kiểm hai task ABSA. Pilot kiến trúc có cùng operator trên năm task tại [Cafe cross-head](CAFEBERT_CROSS_HEAD_GENERAL_PILOT_PLAN_20260922.md) là đề xuất kế tiếp. Nó vẫn là phép sàng lọc có prior art, chưa hứa đạt mục tiêu hoặc tính mới trước khi có kết quả nhiều seed và đối chứng diagonal.

**Cập nhật kết quả 2026-09-22:** pilot Cafe cross-head đã chạy 3 arm ×3 task, seed42, validation-only; [báo cáo](CAFEBERT_CROSS_HEAD_SCREEN_RESULT.md). Full không hơn cả MHA và diagonal ở hai nguồn bất kỳ với biên ≥0,5 điểm nên **dừng nhánh theo gate**. Việc này không làm pilot Restaurant–Hotel tự động trở thành lựa chọn tiếp theo; cần xét lại hướng nghiên cứu theo mục tiêu encoder tổng quát trước khi mở GPU mới.
