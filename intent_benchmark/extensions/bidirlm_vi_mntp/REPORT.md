# BidirLM-Vi-MNTP trên benchmark 8 intent Joi

**Ngày chạy:** 22/09/2026  
**Checkpoint:** `artifacts/bidirlm_vi_mntp/mntp/hf_bidirlm`  
**Benchmark:** `intent_benchmark` — cùng split và tám nhãn với pilot ngày 13/09/2026.

## Kết quả

| Model | Dev Macro-F1 | Test Macro-F1 | Test accuracy | Đúng/77 | Tham số gồm head |
|---|---:|---:|---:|---:|---:|
| BamiBERT | 50,29% | 47,45% | 79,22% | 61 | 102,96M |
| Goi DAPT | 72,23% | 83,30% | 89,61% | 69 | 102,96M |
| Goi student S | 72,41% | 82,09% | 88,31% | 68 | 26,55M |
| **BidirLM-Vi-MNTP** | **88,38%** | **89,49%** | **89,61%** | **69** | **597,11M** |

BidirLM dẫn đầu về **Macro-F1**, hơn DAPT **6,19 điểm** và student S **7,40 điểm**. Accuracy của nó bằng DAPT: cả hai đúng 69/77 mẫu. Trên cùng test, BidirLM đúng riêng 3 mẫu và DAPT đúng riêng 3 mẫu. So với BamiBERT, BidirLM đúng thêm 10 mẫu nhưng sai 2 mẫu mà BamiBERT làm đúng.

| Intent | Support | BidirLM F1 | Nhận đúng |
|---|---:|---:|---:|
| GREETING | 2 | 66,67% | 1/2 |
| CONFIRMATION | 7 | 100,00% | 7/7 |
| GRATITUDE | 2 | 100,00% | 2/2 |
| MENU_QUERY | 21 | 86,96% | 20/21 |
| COMPLAINT | 5 | 80,00% | 4/5 |
| ADDRESS | 6 | 92,31% | 6/6 |
| PHONE | 2 | 100,00% | 2/2 |
| OTHER | 32 | 90,00% | 27/32 |

Điểm cải thiện đáng chú ý là **COMPLAINT recall 4/5**, so với **2/5** của BamiBERT, DAPT và student S trong pilot cũ. Nhưng support chỉ có năm mẫu; GREETING, GRATITUDE, PHONE chỉ có hai mẫu mỗi lớp.

## Cách chạy

- Giữ nguyên `train.jsonl` (353), `dev.jsonl` (77), `test.jsonl` (77), tám nhãn, định dạng lịch sử khách hàng, max length 512 và hàm Macro-F1 của benchmark cũ.
- Fine-tune toàn bộ encoder với head 8 lớp, cross-entropy, seed 42, batch 16, AdamW, LR `2e-5` và `5e-5`, tối đa 15 epochs, early stopping patience 3. Không KD/NC. Hai trial dừng ở epoch 7 và 9.
- Chọn theo dev trước khi đọc test: `2e-5` đạt 87,19%; `5e-5` đạt **88,38% ở epoch 6** và được chọn. Test chỉ chạy cho checkpoint đã chọn.
- Model BidirLM có **late pooling** mặc định: áp head cho các token hợp lệ rồi lấy trung bình logits. Ba model cũ lấy biểu diễn token đầu. Tokenizer BidirLM dùng `fix_mistral_regex=True` để xử lý cảnh báo tokenizer. Runtime mới dùng Transformers 4.57.6; pilot cũ dùng 4.49.0.
- Smoke forward/backward đạt: không thiếu backbone weights, loss/gradient hữu hạn. Test metrics đã được tính lại độc lập từ 77 saved predictions, khớp kết quả lưu.

## Kích thước và latency

| Model | Tham số | Mean forward, batch 1 |
|---|---:|---:|
| BamiBERT (pilot cũ) | 102,96M | 4,025 ms |
| Goi DAPT (pilot cũ) | 102,96M | 4,016 ms |
| Goi student S (pilot cũ) | 26,55M | 3,467 ms |
| **BidirLM-Vi-MNTP** | **597,11M** | **30,87 ms** |

BidirLM có khoảng **5,8 lần** số tham số và **7,7 lần** thời gian forward so với BamiBERT theo hai phép đo. BidirLM timing trên L40S/GPU1: batch 1, BF16 autocast, cùng 77 input, 10 warmup + 154 forward có CUDA synchronization; median 29,50 ms, p95 39,37 ms. Peak inference allocated 3,06 GiB; peak training allocated 11,47 GiB. Các phép đo cũ là lượt chạy khác trên L40S; máy đang dùng chung, nên tỷ lệ latency chỉ mang tính định hướng. Timing không gồm tokenize/network/app overhead.

## Cách diễn giải

Đây là kết quả tốt nhất **về Macro-F1 trong pilot 8 intent hiện có**, nhưng đổi lại model lớn và chậm hơn nhiều. BidirLM không hơn DAPT về số mẫu đúng. Kết quả một seed trên 77 lượt thuộc 38 cases không đủ để kết luận ưu thế ổn định trên hội thoại mới. Test này đã được nhóm xem trong pilot trước; lần chạy bổ sung là **so sánh trên benchmark đã biết**, không phải một đánh giá độc lập còn nguyên tính mù.

Nguồn corpus của checkpoint mới được metadata đánh dấu `benchmark_text=false`, nhưng lượt benchmark này không tự audit raw corpus hoặc dữ liệu pretraining ban đầu của model nền. Cần đánh giá tập hội thoại mới và nhiều seed nếu dùng kết quả để ra quyết định triển khai.

## Artifacts

Tất cả nằm ở `intent_benchmark/extensions/bidirlm_vi_mntp/`:

- `run.py`, `timing.py`: code mở rộng; không sửa runner hoặc kết quả cũ.
- `smoke/`, `lr_2e-5/`, `lr_5e-5/`: smoke, logs, summaries, checkpoint và dev predictions.
- `selection.json`: lựa chọn checkpoint theo validation và SHA256.
- `test_metrics.json`, `test_predictions.jsonl`: test results và dự đoán từng mẫu.
- `timing.json`: phép đo inference trên L40S.
