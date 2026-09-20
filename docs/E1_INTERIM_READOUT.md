# E1 — Nhận định ban đầu của root

2026-09-08. Đây là readout interim, không phải kết quả baseline đa seed hoặc claim vượt paper.

## Kết quả đã hoàn tất

| Task | LR | Seed | Epochs | Best epoch | Validation macro F1 | Runtime |
|---|---|---|---|---|---|---|
| VSFC sentiment | 2e-5 | 42 | 30/30 | 7 | 0.8433919945 (84.34%) | 638.06s (~10.6 phút) |

Nguồn: `artifacts/e1/vsfc_sentiment/screening/2e-05/seed_42/metadata.json`, `epoch_curves.json`, `validation_predictions.json`, `best_checkpoint.pt`. Phiên bản runner `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`, BF16/SDPA, batch32, max_length256 theo thống kê train. Chỉ train/validation, `include_test=false`.

## Diễn giải

- Pipeline đã chứng minh chạy được một full trial từ backbone gốc, học bằng nhãn thật, chọn/lưu checkpoint bằng validation và hoàn tất30epochs. Đây là mức kiểm chứng mạnh hơn smoke vài mẫu.
- Best checkpoint nằm ở epoch7. Trong phần sau của learning curve, loss train tiếp tục giảm nhưng validation không vượt best; dấu hiệu phù hợp với overfitting trên cấu hình này. Chưa đủ để kết luận về toàn bộ recipe hoặc kiến trúc.
- Chưa đánh giá test, chưa kiểm tra độ ổn định nhiều seed, chưa biết LR này có phải tốt nhất. Không lấy84.34 validation trừ83.41 test paper hoặc gọi là vượt BamiBERT.
- Dữ liệu/corpus kế thừa có giới hạn provenance đã ghi trong E0 audit; kết quả này không chứng minh backbone chưa từng tiếp xúc validation text.

## Quyết định tiếp theo

Tái dùng trial này đúng một lần trong screening. Giữ nguyên recipe, chạy bốn LR còn lại và bốn task còn lại; sau đó top2LR × seeds43/44 và xác nhận seeds45/46 theo plan. Không đổi loss, thêm class weights hoặc early stopping chỉ vì curve của một seed. Chưa mở test/E2. Orchestrator validation-only dừng khi có lỗi hoặc thiếu evidence thay vì bỏ run thất bại khỏi mean.
