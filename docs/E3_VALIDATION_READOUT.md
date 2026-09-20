# E3 validation — seed42 screening

## Bổ sung Hotel — hoàn tất 09/09/2026

Theo yêu cầu người dùng, đã chạy thêm đúng một trial Hotel seed42, 30 epochs, beta0.5 với cùng loss/recipe E3. Không thêm multi-seed hoặc mở test. Adapter riêng dùng lại runner; source/config E3 gốc không thay đổi.

Bảng validation pair micro F1 cùng seed42 trên cả ba task ABSA:

| Task | E1 CE | E2 NC | E3 | E3 − NC |
|---|---:|---:|---:|---:|
| ViSFD | 80.07 | 81.09 | 80.22 | -0.87 |
| Restaurant | 69.02 | 69.06 | 70.13 | +1.07 |
| Hotel | 72.04 | 73.58 | 70.32 | -3.26 |

Hotel chọn epoch28, pair F1 70.3196; thấp hơn E1 1.7179 điểm. Detection F1 74.4707, pair precision77.00%, pair recall64.7059%. Trial mất khoảng339giây, peak reserved1.895GiB.

Kết luận: loss E3 không phải cải tiến chung trên ABSA. Giữ Restaurant làm ứng viên experimental; dừng E3 trên Hotel/ViSFD, không chạy thêm seed hiện tại. Bước tiếp theo có thể chuyển sang thử kiến trúc như đã bàn; chưa khởi chạy thí nghiệm kiến trúc. Chi tiết lượt Hotel: [E3_HOTEL_RESULT.md](E3_HOTEL_RESULT.md).

## Lượt đầu: ViSFD và Restaurant

Ngày 09/09/2026. Đã hoàn tất đúng hai trial, mỗi trial 30 epochs; không mở test hoặc thêm seed. Giữ nguyên student/head và recipe E1; chỉ thêm 0.5 × conditional polarity CE trên gold-present slots, loại ViSFD OTHERS. Tất cả khởi tạo từ models/goiturn-s-gd.

## Kết quả cùng seed42

Pair micro F1 trên validation, thang100. E1/NC dùng checkpoint đã chốt; không so với trung bình3/5seeds.

| Task | E1 CE | E2 NC | E3 | ΔE1 | ΔNC | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| visfd | 80.072 | 81.088 | 80.223 | +0.151 | -0.865 | 17 |
| absa_restaurant | 69.017 | 69.060 | 70.130 | +1.112 | +1.070 | 30 |

ViSFD tăng 0.151 điểm so với E1, dưới ngưỡng screening 0.3 điểm, và giảm 0.865 điểm so với NC. Restaurant tăng 1.112 điểm so với E1 và 1.070 điểm so với NC, đạt ngưỡng screening.

Đây là tín hiệu theo task; chưa chứng minh phương pháp tốt chung trên cả hai hoặc vượt BamiBERT trên test. Restaurant mới một seed nên chưa gọi là ổn định hay có ý nghĩa thống kê. Checkpoint Restaurant tốt nhất ở epoch 30; không kéo dài lịch train sau khi thấy kết quả.

## Chi phí và kiểm tra

ViSFD chạy CUDA2 khoảng 340.6 giây, peak reserved 2.379 GiB; Restaurant CUDA1 khoảng 309.7 giây, peak reserved 1.109 GiB. Hai trial chạy song song, BF16/SDPA, 30 epochs, batch32, max_length256. Không đổi số tham số hoặc inference decoder.

12 checks E1/E3 pass, gồm unequal-support pooling, logits3D Restaurant, all-absent differentiable zero và OTHERS exclusion. GPU smoke forward/backward hữu hạn. Source/config/checkpoints E1/E2 giữ nguyên. Review predictions và provenance cuối xem [E3_REVIEW_FINAL.md](E3_REVIEW_FINAL.md).

## Nhận định và bước tiếp theo

Detection F1 giảm ở cả hai: ViSFD 85.954, Restaurant 81.962. Pair gain vì vậy không phải cải thiện đồng đều mọi thành phần. Phân rã lỗi trong review sẽ làm rõ polarity/detection trade-off.

Root đề xuất dừng nhánh loss này trên ViSFD, giữ NC làm mốc; chỉ xác nhận Restaurant với seeds43/44 nếu tiếp tục. Chưa khởi chạy multi-seed hoặc test. Phép thử này chưa xác định được giới hạn dung lượng của backbone.

Plan: [E3_EXECUTION_PLAN.md](E3_EXECUTION_PLAN.md). Implementation: [E3_IMPLEMENTATION.md](E3_IMPLEMENTATION.md). Artifacts: artifacts/e3/{visfd,absa_restaurant}/seed_42.

## Audit cuối — PASS

Reviewer độc lập xác nhận cả hai trial: IDs/order/gold đúng, tính lại metrics khớp predictions, checkpoint SHA256 và source hashes khớp plan, chỉ Train/Dev, không trial lỗi hoặc trùng được tính.

| Task | Missed E1 → E3 | Spurious E1 → E3 | Wrong polarity E1 → E3 |
|---|---:|---:|---:|
| ViSFD | 460 → 498 | 476 → 505 | 318 → 295 |
| Restaurant | 163 → 201 | 184 → 174 | 156 → 123 |

Wrong polarity chỉ tính khi gold và prediction đều present, loại ViSFD OTHERS; mẫu số thay đổi nên không xem số lỗi giảm riêng lẻ là bằng chứng đủ. Với Restaurant, conditional polarity accuracy tăng từ 734/890=82.472% lên 729/852=85.563%, trong khi số pair đúng giảm nhẹ từ734 xuống729. Pair F1 tăng nhờ predicted pairs giảm từ1074 xuống1026, cải thiện pair precision; detection recall giảm. Đây là trade-off có lợi cho metric tổng trên seed này, không phải model tốt hơn mọi mặt.
