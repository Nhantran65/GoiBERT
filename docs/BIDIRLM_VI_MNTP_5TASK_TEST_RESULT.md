# P_vi trên đủ 5 task và test đã khóa

**Ngày:** 2026-09-21. **Trạng thái:** đã hoàn tất fine-tune Sentiment/Hotel cho P_vi và đối chứng O; đã mở test đúng một lần cho năm checkpoint P_vi được chọn bằng validation. Việc mở test theo yêu cầu trực tiếp của người dùng trong phiên này, thay cho phạm vi validation-only của kế hoạch chặng 1 ban đầu.

## Cấu hình và khóa test

- **P_vi:** BidirLM-0.6B-Base, 596.049.920 tham số encoder, đã luyện thêm MNTP trên gần 250M token tiếng Việt CC0. **O:** cùng model gốc trước pha MNTP tiếng Việt. Nguồn và kiểm tra MNTP: [báo cáo chặng 1](BIDIRLM_VI_MNTP_STAGE1_RESULT.md).
- Sentiment và Hotel của cả hai arm dùng đúng recipe chặng 1: LR `1e-5` đã chọn từ grid Restaurant, seed 42, 30 epoch, batch hiệu dụng 32, max length 256, BF16, head E14, best epoch theo F1 validation. Ba task cũ dùng checkpoint đã hoàn tất, không train lại. Hai run mới của mỗi arm đều có reload F1, dự đoán, ID và strict load check đạt.
- Manifest [locked_selection.json](../artifacts/bidirlm_vi_mntp/test/P_vi/locked_selection.json) được ghi lúc **2026-09-21T15:51:24Z**, trước khi truy cập test. Nó cố định đúng năm task, checkpoint SHA-256, metadata SHA-256, epoch, LR, seed và hash source. Mỗi test chỉ nạp checkpoint đã khóa, không chọn lại theo điểm test.
- Test dùng cùng tokenizer, model wrapper, max length, BF16 và evaluator trong `goibert.train` như validation; metric được tính lại từ dự đoán trước khi lưu. Audit sau run xác nhận 5/5 checkpoint hash đúng manifest, số hàng/ID đầy đủ và `test_f1` khớp file metric. Artifacts: [summary.json](../artifacts/bidirlm_vi_mntp/test/P_vi/summary.json), [test/P_vi](../artifacts/bidirlm_vi_mntp/test/P_vi), [ft summary](../artifacts/bidirlm_vi_mntp/summary.json).

## Validation: tác động của 250M token tiếng Việt

F1 %, seed 42. O và P_vi dùng cùng LR `1e-5`; ba task cũ có kết quả từ chặng 1.

| Task | O | P_vi | P_vi − O |
|---|---:|---:|---:|
| VSFC Sentiment | 85,95 | **86,52** | +0,57 |
| VSFC Topic | **80,74** | 80,69 | −0,04 |
| ViSFD | 84,85 | **85,44** | +0,58 |
| UIT-ABSA Restaurant | 75,72 | **76,15** | +0,43 |
| UIT-ABSA Hotel | **73,965** | 73,962 | −0,003 |

P_vi cao hơn ở 3/5 task, gần bằng ở Hotel và thấp hơn rất nhẹ ở Topic. Restaurant chỉ nhỉnh hơn ở **best epoch**; trung bình F1 epoch 11–30 là 75,00 so với 74,88 của O. Một seed không xác nhận cải thiện ổn định của pha MNTP tiếng Việt.

## Test: năm checkpoint P_vi đã khóa

F1 %, seed 42. BamiBERT là E14 local test, CafeBERT là checkpoint tham chiếu seed 42 trong SEA-LION benchmark; đây là so sánh kết quả hệ thống, không cùng ngân sách tuning. `Δ` tính theo điểm phần trăm F1. Metric là macro F1 cố định nhãn cho VSFC, pair micro F1 đầu cuối cho ViSFD và UIT-ABSA.

| Task | n test | P_vi | BamiBERT | Δ Bami | CafeBERT | Δ Cafe |
|---|---:|---:|---:|---:|---:|---:|
| Sentiment | 3.166 | **82,37** | 83,03 | −0,66 | 82,36 | +0,01 |
| Topic | 3.166 | **78,81** | 78,18 | +0,63 | 80,77 | −1,96 |
| ViSFD | 2.224 | **86,11** | 84,50 | +1,61 | 85,48 | +0,62 |
| Restaurant | 1.938 | **76,65** | 74,15 | +2,50 | 77,39 | −0,74 |
| Hotel | 2.030 | **74,43** | 73,94 | +0,49 | 75,17 | −0,75 |

Nguồn benchmark: `artifacts/e14/trials/<task>/lr_<selected>/seed_42/test_metrics.json` cho BamiBERT và `artifacts/sealion_benchmark/test/cafebert/<task>/test_metrics.json` cho CafeBERT. CafeBERT Restaurant **77,39** ở seed 42; con số **77,12** trong tóm tắt phiên trước là **trung bình hai seed** 42/43, nên không trộn hai cách tổng hợp này trong bảng.

## Diễn giải

- **Mục tiêu vượt cả BamiBERT và CafeBERT trên đủ 5 task chưa đạt.** P_vi cao hơn BamiBERT ở 4/5 task (trừ Sentiment); cao hơn CafeBERT ở ViSFD và về số học ở Sentiment, nhưng Sentiment chỉ hơn **0,014 điểm**, không có ý nghĩa thực tế. Topic, Restaurant và Hotel vẫn dưới CafeBERT.
- **ViSFD là kết quả nổi bật nhất:** 86,11 test, hơn CafeBERT 0,62 và BamiBERT 1,61 điểm. Restaurant 76,65 test vượt BamiBERT 2,50 nhưng dưới CafeBERT 0,74. Hotel P_vi và O gần như bằng nhau trên validation, nên không quy kết 74,43 test của P_vi là lợi ích từ 250M token tiếng Việt.
- **Chỉ một seed P_vi và một lần mở test.** Kết quả mô tả hiệu năng của năm checkpoint đã khóa, chưa chứng minh khác biệt có ý nghĩa thống kê. Không dùng các điểm test này để chọn lại LR, epoch, checkpoint hoặc tạo vòng đánh giá mới trên cùng test.
