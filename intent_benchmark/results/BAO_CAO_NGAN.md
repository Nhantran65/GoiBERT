# Benchmark intent: Goi DAPT, Goi student và BamiBERT

## Dataset và split

- 250 cases, 512 lượt khách hàng; bỏ 5 lượt thiếu nhãn, giữ **507 mẫu**. Đã rà và sửa 19 nhãn; mọi mẫu mơ hồ vẫn được dùng bình thường.
- **8 intents:** GREETING, CONFIRMATION, GRATITUDE, MENU_QUERY, COMPLAINT, ADDRESS, PHONE, OTHER.
- Input gồm câu khách hàng hiện tại và các câu khách hàng trước đó trong cùng case; không chứa nhãn, fixture hoặc lượt tương lai.
- Split cố định seed 42: **train 353 / dev 77 / test 77**, xấp xỉ 70/15/15. Chia theo nhóm để cùng case, nguồn và nhóm gần trùng đã xác định không sang split khác. Không trùng toàn bộ input giữa các split; một số câu hiện tại vẫn lặp với lịch sử khác nhau.

## Cách benchmark

Fine-tune toàn bộ encoder cùng head phân loại 8 lớp mới, CE không trọng số. Mỗi model thử LR `2e-5` và `5e-5`, seed 42, batch 16, tối đa 15 epochs, early stopping patience 3. Tổng cộng **6 lượt train**. Chọn checkpoint theo dev Macro-F1, sau đó đánh giá đúng 3 model đã chọn trên test; không chọn cấu hình bằng test.

## Kết quả test

| Model                 | LR được chọn | Best epoch |         Macro-F1 |         Accuracy |  Đúng / 77 |         Tham số |
| --------------------- | ---------------: | ---------: | ---------------: | ---------------: | -----------: | ---------------: |
| **Goi DAPT**    |             5e-5 |         11 | **83,30%** | **89,61%** | **69** |          102,96M |
| **Goi student** |             5e-5 |         13 | **82,09%** | **88,31%** | **68** | **26,55M** |
| BamiBERT              |             2e-5 |         11 |           47,45% |           79,22% |           61 |          102,96M |

DAPT tốt nhất trong pilot; student chỉ kém 1 mẫu đúng, với số tham số ít hơn khoảng **3,9 lần**. Audit độc lập tính lại metrics từ predictions và kiểm tra checkpoint, tokenizer, input, split; chưa phát hiện lỗi nghiêm trọng làm sai kết quả.

## Vì sao Goi đang tốt hơn?

1. **Chênh lệch chủ yếu ở lớp hiếm:** BamiBERT có F1 bằng 0 ở GREETING, GRATITUDE và PHONE; mỗi lớp chỉ có 2 mẫu test và 7–8 mẫu train. Ba lớp này tạo khoảng **84% chênh lệch Macro-F1 student–BamiBERT**. Đây là phân rã lỗi đã đo được, chưa giải thích nguyên nhân học thất bại.
2. **Recipe ảnh hưởng mạnh:** ở LR 2e-5, dev F1 của BamiBERT và DAPT gần ngang nhau (50,29% và 50,09%); ở LR 5e-5, DAPT đạt 72,23% nhưng BamiBERT chỉ 27,51%. Pilot chưa xác định cấu hình tốt nhất hay độ ổn định qua nhiều seed của BamiBERT.
3. **Goi đã học thêm:** theo manifest, DAPT tiếp tục 40.000 bước MLM từ BamiBERT; student distill 30.000 bước từ DAPT. Việc học thêm có thể giúp thích nghi, nhưng chưa được chứng minh là nguyên nhân chính của lợi thế hiện tại.

**Phạm vi kết luận:** Goi thắng trong pilot một seed trên 77 mẫu test này; chưa thể kết luận thắng tổng quát BamiBERT. Nhãn được rà với AI hỗ trợ, các lớp hiếm ít mẫu và corpus pretraining lịch sử chưa đủ để audit overlap. SOTA của BamiBERT trong [paper](https://arxiv.org/html/2607.02259v1) được báo cáo trên các benchmark khác, không bao gồm bộ intent này.
