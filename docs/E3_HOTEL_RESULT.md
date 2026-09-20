# E3 Hotel extension — kết quả screening

Đã chạy đúng một trial E3 trên UIT-ABSA Hotel: seed 42, 30 epochs, LR `1e-4`, batch 32, max length 256, BF16/SDPA, khởi tạo mới từ `models/goiturn-s-gd`. Kiến trúc và head giữ nguyên; loss là `CE_E1 + 0.5 × conditional polarity CE`. Chọn checkpoint theo validation end-to-end pair micro F1.

## Kết quả validation

| Model | Detection P | Detection R | Detection F1 | Pair P | Pair R | Pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| E1 CE, seed42 | 80.369 | 76.623 | 78.451 | 73.798 | 70.359 | **72.038** |
| E2 NC, seed42 | 80.509 | 77.311 | 78.878 | 75.099 | 72.116 | **73.578** |
| E3 conditional polarity CE, seed42 | 81.545 | 68.526 | 74.471 | 77.000 | 64.706 | **70.320** |

E3 thấp hơn E1 `-1.718` điểm pair F1 và thấp hơn NC `-3.258` điểm. Vì vậy Hotel không cho tín hiệu để tiếp tục nhánh loss này.

Phân rã lỗi trên 795 câu validation:

| Model | Missed aspect | Spurious aspect | Wrong polarity |
|---|---:|---:|---:|
| E1 | 306 | 245 | 82 |
| E2 NC | 297 | 245 | 68 |
| E3 | 412 | 203 | 50 |

Loss phụ thực sự tăng độ chính xác polarity khi cả gold và prediction đều present: E3 `847/897 = 94.426%`, E1 `921/1003 = 91.825%`, NC `944/1012 = 93.281%`. Tuy nhiên E3 bỏ sót nhiều aspect hơn, làm recall và pair F1 giảm mạnh. Đây là trade-off thất bại cho mục tiêu end-to-end.

## Contract và provenance

- Hotel mapping đã kiểm tra: 34 aspect, mỗi slot có 4 lớp `absent/negative/neutral/positive`, tổng 136 logits; dùng cả 34 slot cho auxiliary polarity loss.
- Checkpoint tốt nhất ở epoch 28; trial hoàn tất đủ 30 epochs trong 338.9 giây.
- Peak GPU allocated `1.70 GiB`, reserved `1.89 GiB`, chạy trên `cuda:2` (NVIDIA L40S).
- 26,597,896 tham số, giống E1; validation có 795 dòng và 795 ID duy nhất.
- Đã tái tính metrics từ predictions, kiểm tra ID/order/gold khớp split validation và không load test.
- E1 dependency hash: `456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873`.
- Original E3 source hash: `a17ec51f8386162e34ed35a7d4695feb06bb813d21c8c812b221b1084b1b2777`.
- Extension source hash: `5967bcef63779c02ab58d2fa78e4aaffa2724410002d0f6b6228fcf84a999b72`.

Artifact đầy đủ nằm tại [artifacts/e3/hotel_extension/seed_42](/home/kiettqa/GoiBERT/artifacts/e3/hotel_extension/seed_42). Launcher riêng là [scripts/run_e3_hotel_extension.py](/home/kiettqa/GoiBERT/scripts/run_e3_hotel_extension.py).

Kết luận screening: dừng E3 conditional polarity CE trên Hotel cùng với ViSFD; chỉ Restaurant còn tín hiệu seed42, nhưng chưa cần xác nhận thêm seed ở giai đoạn experimental hiện tại.
