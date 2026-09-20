# Nguồn và trạng thái bằng chứng

Snapshot tạo ngày 2026-09-07 từ working tree `/home/kitkit/Work/Code/2hearts/ai-foundary`, branch `ml-models`, HEAD `366f8121e7a0c5aa29f4b8ce2680b1e3cc4bf47d`.

`GOI_MODELS_SUMMARY_REPORT_EN.md` có thay đổi chưa commit tại thời điểm copy. Gói giữ nguyên bản người dùng đã sửa; đây không phải export thuần HEAD. Không commit/reset hoặc thay đổi source repo. Hash manifest xác định nội dung thực tế được chuyển.

## Nguồn chính

- BamiBERT paper: https://arxiv.org/html/2607.02259 (bảng 1–2; đã đối chiếu trong phiên làm việc trước khi đóng gói).
- Model public: https://huggingface.co/Qualcomm-AI-Research/BamiBERT
- `reference/ai-foundary/GOI_MODELS_SUMMARY_REPORT_EN.md`: family results và synthetic turn results lịch sử; phần turn completion ngoài phạm vi GoiBERT mới.
- `reference/ai-foundary/experiments/goibert-family-v1/REPORT.md`: bản report family.
- `reference/ai-foundary/experiments/goibert-family-v1/MODEL_SPECS.md`: spec cũ Stage A; không override Stage P report/code.
- `reference/ai-foundary/experiments/goiturn-arch-scaling/results-lumi/summary.md`: các kết quả kiến trúc cũ, khác metric/protocol và parameter counting.
- `reference/ai-foundary/DATA.md`: hướng dẫn tải lịch sử, không đảm bảo mirror hiện còn đúng hoặc tương đương dataset paper. Kiểm tra điều kiện sử dụng từ nguồn khi tải; không xem lời khẳng định license trong tài liệu cũ là xác minh mới.

## Cách đọc số liệu

`paper_targets.csv` là số liệu paper được chép lại, scale 100. `historical_family.csv` là số liệu đã báo cáo, scale 1, không phải rerun. Các dòng VLSP mean là mean ba seeds; các dòng khác ở bảng Stage P là seed 42. Bảng CSV chọn lọc không thay thế report đầy đủ.

Bản đầu chưa có weights; bản cập nhật người dùng đã bổ sung hai model trong `models/`, nay được đưa vào ZIP cùng metadata. Xem MODEL_STATUS.md. Không copy credentials, chat production, `.env` hoặc raw datasets. Truy cập model card gốc để xem điều kiện sử dụng.

Kiểm tra đóng gói chỉ chứng minh nội dung và tính toàn vẹn file/code syntax, không chứng minh metric correctness, reproducibility của run cũ hoặc khả năng đạt mục tiêu trên server.
