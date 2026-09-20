# Nhận định của root về benchmark intent

Phạm vi: kiểm toán pilot hiện có, không chạy lại training, không sửa nhãn/split và không chọn lại checkpoint bằng test. Bản kiểm toán kỹ thuật độc lập nằm trong `AUDIT_REPORT.md`; các nhận định dưới đây phân biệt quan sát với giả thuyết nguyên nhân.

## Kết quả có nghĩa gì?

Predictions đã lưu cho thấy BamiBERT đúng 61/77, Goi DAPT đúng 69/77, student đúng 68/77. DAPT giữ đúng cả 61 mẫu BamiBERT làm đúng và sửa thêm 8 mẫu. Student sửa 9 lỗi của BamiBERT nhưng sai thêm 2 mẫu BamiBERT làm đúng. Đây là lợi thế quan sát được trong pilot, chưa phải thứ hạng ổn định ngoài dataset này.

GREETING, GRATITUDE và PHONE mỗi lớp chỉ có 2 mẫu test; BamiBERT có F1 bằng 0 ở cả ba. Student có F1 tương ứng 2/3, 1 và 2/3. Ba lớp này đóng góp `(2/3 + 1 + 2/3) / 8 = 0.291667`, tức 29,17 điểm phần trăm trong chênh lệch 34,64 điểm Macro-F1 student–BamiBERT (khoảng 84%). Macro-F1 là metric đúng cho việc quan tâm ngang nhau tới tám lớp; số mẫu nhỏ làm ước lượng từng lớp thiếu ổn định, không làm metric bị tính sai.

## Vì sao danh tiếng SOTA không mâu thuẫn với kết quả này?

[Paper BamiBERT](https://arxiv.org/html/2607.02259v1) báo cáo đứng đầu 11/15 metrics trên tám benchmark, không bao gồm bộ intent hội thoại này. Kết quả đó không bảo đảm đứng đầu mọi downstream dataset và mọi ngân sách fine-tune.

Manifest local ghi Goi DAPT học tiếp 40.000 bước MLM từ BamiBERT; student distill 30.000 bước từ DAPT. Vì vậy đây là so sánh checkpoint gốc với các checkpoint đã học thêm, không phải kiến trúc student tự nhiên vượt một teacher đã được tối ưu tương đương. Lợi ích thích nghi domain là giả thuyết hợp lý, nhưng chưa được tách nhân quả trong thí nghiệm này. Corpus lịch sử đầy đủ không có tại workspace để kiểm chứng overlap; không có bằng chứng ở đây cho phép kết luận bộ intent đã bị lộ vào pretraining.

## Dấu hiệu optimization cần lưu ý

| LR | BamiBERT dev Macro-F1 | Goi DAPT dev Macro-F1 |
|---|---:|---:|
| 2e-5 | 50,29% | 50,09% |
| 5e-5 | 27,51% | 72,23% |

Ở LR thấp hai backbone base gần như ngang nhau. LR cao giúp DAPT nhưng khiến BamiBERT học kém trong ngân sách hiện tại. BamiBERT LR5e-5 còn đạt best ở epoch cuối 15, train loss cuối 0,811; vì vậy chưa có bằng chứng rằng ngân sách hiện tại đã tìm được baseline BamiBERT tốt. Không thể từ các log này kết luận tăng epoch hoặc giảm LR chắc chắn khắc phục được.

Paper dùng batch32, LR {1e-5, 2e-5, 5e-5}, 30 epochs và chọn checkpoint theo validation. Pilot dùng batch16, hai LR {2e-5, 5e-5}, tối đa15 epochs, patience3, seed42. Đây là ngân sách chung nhất quán giữa ba model, nhưng không phải tái lập recipe paper hay so sánh các model đã được tune đầy đủ.

## Điều đã loại trừ và điều chưa loại trừ

- Checkpoint BamiBERT local có SHA256 `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`, khớp [weights chính thức](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/blob/main/model.safetensors). Không có dấu hiệu tải nhầm checkpoint.
- Ba tokenizer cho cùng IDs trên 507 inputs; loading log không thiếu backbone weights. Không thấy bất lợi riêng do tokenization hoặc bỏ sót weights BamiBERT.
- Kiểm toán hiện có không phát hiện input chứa future turn, nhãn hay fixture; các metric headline tính lại khớp. Điều này không thay cho kiểm chứng corpus tiền huấn luyện lịch sử.
- Năm nhãn test đã sửa giúp BamiBERT và DAPT cùng tăng5 mẫu đúng, student tăng4, khi giữ predictions cố định. Chúng không giải thích lợi thế của Goi; phép tính này không đo tác động của 12 nhãn train đã sửa lên quá trình học.
- Nhãn là review có AI hỗ trợ; split theo case không biến 77 turns thành 77 hội thoại độc lập. Test có38cases; hai mẫu PHONE thuộc một case. Một số current texts lặp giữa splits với histories khác nhau; không thể gọi đây là kiểm định tổng quát hóa sang template/ngôn ngữ hoàn toàn mới.

## Bước kiểm chứng tiếp theo được đề xuất, chưa thực thi

Ưu tiên kiểm tra độ ổn định của baseline BamiBERT trên dev trước khi đầu tư cải tiến kiến trúc: khóa một protocol rộng hơn gồm LR1e-5 và ngân sách epoch đủ dài, giữ nguyên preprocessing/head để đọc được tác động; sau đó áp ngân sách đối chiếu tương đương cho các backbone và kiểm tra seed mới. Không dùng test để lựa chọn recipe. Muốn claim ngoài pilot cần bộ đánh giá độc lập lớn hơn, đặc biệt các lớp hiếm.

Kết luận: các số hiện tại là kết quả hợp lệ của lượt chạy đã khóa; chúng chưa chứng minh vì sao Goi thắng theo quan hệ nhân quả, cũng chưa đo được trần năng lực của BamiBERT trên intent classification.
