# Pilot kế tiếp trên CafeBERT: học chung Restaurant và Hotel

**Trạng thái cập nhật 2026-09-22: tạm gác, chưa chạy training/test.** Người dùng ưu tiên một phép thử có cơ chế dùng chung cho năm task hơn là một pilot chỉ nhắm Restaurant/Hotel. [Pilot kiến trúc tổng quát thay thế](CAFEBERT_CROSS_HEAD_GENERAL_PILOT_PLAN_20260922.md). Nội dung dưới đây giữ làm phương án dự phòng, không còn là lượt chạy kế tiếp.

## Câu hỏi duy nhất

Với cùng `uitnlp/CafeBERT` pretrained, việc cho **một encoder** học đồng thời review Restaurant và Hotel có cải thiện pair-micro F1 của hai domain so với train từng domain riêng, khi đã kiểm soát số bước cập nhật không?

Hai tập train có 7.028 và 7.180 review; validation lần lượt 771 và 795. Restaurant có 12 aspect, Hotel 34 aspect. Hai miền có cùng bốn trạng thái slot (absent/negative/neutral/positive) theo loader hiện tại, nhưng **không chia sẻ chỉ số hoặc head aspect**. [Loader và metric audit](E0_DATA_METRIC_AUDIT.md).

## Model và dữ liệu

- Khởi tạo từ cùng checkpoint CafeBERT và tokenizer/revision của baseline hiện có. Shared encoder; hai classification heads E14 độc lập: 48 logits Restaurant, 136 logits Hotel. Inference biết domain từ task đang đánh giá và dùng head tương ứng. Không cho validation/test tham gia huấn luyện, chọn nhãn hay cách viết prompt.
- Chỉ dùng train splits hiện có; không thêm LLM text, composite, CPT, KD, aspect-pair hoặc LLRD. Không coi một aspect Hotel không được gán nhãn trong Restaurant là absent; mỗi head nhận loss chỉ trên dữ liệu miền mình.
- Seed 42, LR 2e-5, optimizer/precision/maxlen/effective batch/head/dropout/loss E14 như Cafe reference. Freeze mọi quyết định trước validation; nếu thay microbatch hoặc attention kernel, cả controls và joint phải đổi giống nhau.

## Ba arms và vì sao cần cả hai control

1. **Target-only chuẩn:** Cafe E14 Restaurant và Hotel 30 epoch hiện có, để thấy gain thực dụng so mốc lịch sử (Restaurant 77,72; Hotel 73,34 validation seed42). Kiểm tra đúng code/checkpoint/split, không tự lấy bản cũ nếu recipe của pilot đổi.
2. **Target-only cùng tổng số optimizer steps:** một model Restaurant-only, một Hotel-only, mỗi model chạy đến số bước bằng joint. Lặp train target khi cần. Đây là đối chứng cho việc joint được **train lâu hơn**. Số lần thấy dữ liệu target của control này lớn hơn joint; ghi rõ, không gọi là đối chứng đồng thời khớp exposure.
3. **Joint:** luân phiên batch Restaurant/Hotel với tần suất 1:1 theo update, mỗi miền có khoảng 30 lượt qua train; tổng số bước xấp xỉ 30×ceil(7028/32)+30×ceil(7180/32)≈13.350. Chốt số bước chính xác sau khi dựng loader, rồi đặt arm 2 bằng đúng số đó. Head của miền kia không nhận gradient trong update hiện tại.

Chọn checkpoint theo validation trên cùng lịch đánh giá tính theo **optimizer steps** cho cả ba arms để số lần được chọn đỉnh tương đương. Joint cần một checkpoint dùng chung encoder; **không chọn một epoch cho Restaurant và epoch khác cho Hotel rồi ghép thành một model**. Có thể dùng rule đăng ký trước như tối đa hóa trung bình delta F1 chuẩn hóa theo hai baseline, với ràng buộc task nào cũng không dưới baseline quá 0,3 điểm; lưu cả đường cong để xem negative transfer. Rule này là heuristic chọn pilot, không là significance.

## Gate và bước tiếp theo

- Screening seed42: joint phải hơn ít nhất một trong hai target-only full-step controls khoảng ≥1,0 điểm validation, miền kia không giảm >0,3, mới đáng chạy seed43/44. Ngưỡng này là quyết định đầu tư theo nhiễu validation đã quan sát, không ước lượng xác suất thống kê.
- Xác nhận: ≥3 seed, paired comparison ở Restaurant và Hotel; không mở test trong pilot. Nếu gain chỉ ở một miền, báo đúng một miền; nếu joint thua control lặp target, không quy gain cho transfer.
- Pilot này chưa đụng VSFC Sentiment/Topic hoặc ViSFD. Nếu có tín hiệu transfer rõ, nghiên cứu VSFC Sentiment↔Topic bằng hai head trên cùng văn bản, rồi xử lý ViSFD riêng. Nếu không có tín hiệu, dừng joint Rest/Hotel và cân nhắc dữ liệu được kiểm nhãn hoặc operator attention đã nghiên cứu. Không cộng gain của các nhánh khác một cách cơ học.

## Tính mới và giới hạn

Shared encoder + task heads là kỹ thuật đã có; literature về multi-task learning cũng ghi nhận negative transfer và ảnh hưởng của lịch lấy mẫu. [Learning Task Sampling Policy](https://aclanthology.org/2021.findings-emnlp.375/). Pilot này là phép kiểm **có thể bác bỏ** trên dữ liệu hiện có, không phải claim phương pháp mới. Nếu project đòi encoder mới có tính mới, kết quả pilot chỉ làm nền để chọn/thiết kế cơ chế riêng sau đó.

So sánh với CafeBERT/BamiBERT paper vẫn bị giới hạn bởi provenance/evaluator của bản UIT-ABSA local; chỉ tuyên bố về benchmark local cho đến khi audit xong. [Benchmark protocol](BENCHMARK_PROTOCOL.md).
