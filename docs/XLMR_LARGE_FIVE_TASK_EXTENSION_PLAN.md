# XLM-R-large — hoàn tất Sentiment và Hotel

Theo yêu cầu 2026-09-21: đo hai task còn thiếu, **validation-only**, chỉ GPU **2 và 3**. Dùng checkpoint XLM-R-large pinned và recipe E14 đã dùng trong [baseline ba task](XLMR_LARGE_BASELINE_RESULT.md). Không thay đổi ba task đã đo hoặc mở test.

- Mỗi task `vsfc_sentiment`, `absa_hotel`: LR `{1e-5,2e-5,5e-5}`, seed42, 30epoch, batch hiệu dụng32, micro32, maxlen256, BF16/SDPA, CE/head CLS E14. Best epoch theo F1 validation. Nếu LR phân kỳ, ghi `diverged`, không cứu bằng thay recipe.
- Chạy một job/GPU, chỉ GPU2/3; kiểm `nvidia-smi` và từ chối GPU có compute process của người khác hoặc <20GB trống. `run_missing_tasks.sh` chỉ khởi chạy hai task mới và có thể chạy lại sau khi dừng.
- Artifacts trong `artifacts/xlmr_large_baseline/xlmr_large/<task>/lr_<lr>/seed_42/`, logs tách tại `logs_five_task_extension/`. Mỗi run phải đạt reload/prediction/ID checks của runner hiện có.
- So Bami/Cafe trên validation cùng split. Chỉ suy phạm vi một seed và LR được chọn; không gọi đây là kết quả test hay SOTA.

**Cập nhật khi đang chạy:** người dùng mở rộng quyền sử dụng sang **cả bốn GPU** và cho phép chạy chung khi còn VRAM. Hai run LR1e-5 đã hoàn tất trên GPU2/3. Job Sentiment LR2e-5 trên GPU2 được giữ; ba ô còn lại xếp lên GPU0/1/3 với kiểm tra ≥20GB VRAM trống trước launch. Không dừng hoặc sửa job của người khác. Việc này thay giới hạn GPU2/3 và một job tổng/GPU ở trên theo chỉ dẫn mới của người dùng.

**Recovery:** Hotel LR2e-5 trên GPU3 và Sentiment LR5e-5 trên GPU0 gặp CUDA illegal instruction/memory access khi đang train. Giữ nguyên artifacts/status failed. Chạy lại đúng seed/LR/recipe vào `artifacts/xlmr_large_baseline/recovery/...` trên GPU1/2 là các GPU vừa chạy benchmark ổn định; không dùng các partial checkpoints làm kết quả và không tiếp tục optimizer từ epoch lỗi. Đây là phục hồi lỗi thiết bị, không đổi cấu hình dựa trên validation.
