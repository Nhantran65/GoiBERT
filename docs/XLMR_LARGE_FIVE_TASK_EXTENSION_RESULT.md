# XLM-R-large: Sentiment và Hotel, hoàn chỉnh bảng năm task

Validation-only, seed 42; recipe E14 và checkpoint giống baseline XLM-R-large ba task. Hai task mới chạy grid ba LR riêng; ba task cũ tái dùng, không chạy lại.

| Task | XLM-R-large | BamiBERT | CafeBERT | Large−Cafe |
|---|---:|---:|---:|---:|
| Sentiment | 86.73 | 86.33 | 87.69 | -0.96 |
| Topic | 80.61 | 79.53 | 81.74 | -1.13 |
| ViSFD | 85.48 | 84.38 | 85.47 | +0.02 |
| Restaurant | 77.81 | 74.74 | 77.72 | +0.10 |
| Hotel | 73.66 | 73.24 | 73.34 | +0.32 |

Các điểm trong bảng đều là validation seed 42. Bami được chọn trong grid LR riêng; Cafe dùng LR 2e-5; XLM-R-large Sentiment/Hotel chọn trong ba LR. Restaurant XLM-R-large/Cafe mean hai seed lần lượt 77,01/77,06, nên không diễn giải chênh seed 42 là thắng ổn định.

| Task mới | 1e-5 | 2e-5 | 5e-5 | Chọn |
|---|---:|---:|---:|---|
| Sentiment | 86.69 | 86.73 | 82.35 | 2e-05, epoch 10 |
| Hotel | 72.86 | 73.57 | 73.66 | 5e-05, epoch 17 |

Mỗi ô mới hoàn thành 30 epoch, reload best checkpoint, tái tính F1 từ predictions, đối chiếu IDs/gold trong runner; script tổng hợp độc lập tính lại F1 từ các prediction files. Cả sáu ô phải đạt trước khi báo cáo được ghi.

**Sentiment LR 5e-5 bất ổn:** validation rơi xuống ~22,47 từ epoch 5 và không hồi phục; checkpoint tốt nhất 82,35 ở epoch 3 của run recovery, thấp hơn rõ hai LR còn lại. Vì vậy không coi đây là một LR cạnh tranh ổn định.

**Recovery lỗi thiết bị:** lượt Hotel LR2e-5 trên GPU3 lỗi CUDA illegal instruction; lượt Sentiment LR5e-5 trên GPU0 lỗi CUDA illegal memory access. Không dùng điểm của hai run dở. Cả hai được chạy lại từ đầu cùng seed/recipe trong thư mục `recovery/` trên GPU1/2. Người dùng đã mở rộng quyền sử dụng từ GPU2/3 sang cả GPU0–3; không dừng job khác.

GPU-giờ tính từ **sáu run hoàn tất**: 2.79; chưa gồm hai run CUDA lỗi. Không mở test, không sửa baseline cũ.

Giới hạn: một seed cho hai task mới; việc chọn best epoch và best LR trên validation làm điểm lạc quan; chưa kiểm định khác biệt nhỏ. Hotel cùng LR2e-5 so Cafe là 73,57 vs 73,34 (+0,23), nhỏ hơn chênh best-of-grid +0,32. Xem artifacts máy đọc ở `artifacts/xlmr_large_baseline/five_task_extension_summary.json`.
