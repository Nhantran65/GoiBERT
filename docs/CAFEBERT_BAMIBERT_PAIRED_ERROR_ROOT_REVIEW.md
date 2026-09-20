# Root review — CafeBERT và BamiBERT sai khác ở đâu?

Ngày review: 2026-09-19. Đã đọc toàn bộ **72 case** trong `artifacts/cafe_bami_paired_error_audit/packet.jsonl`, đối chiếu `aggregate.json`, `manifest.json` và sáu file predictions. Chỉ dùng validation seed 42; không chạy model, không mở test, không sửa gold. Đây là **diễn giải của một người đọc** dựa trên câu đứng riêng; không thay thế adjudication theo guideline của người gán nhãn.

## Kết quả kiểm tra số

Hash của sáu prediction files và script audit khớp manifest. Packet có 24 ID khác nhau cho mỗi task, đúng 8/8/4/4 bucket; 6 F1 trong báo cáo cơ học khớp evaluator và phép đếm độc lập. Các paired counts cộng đủ 771/1.583/1.112 hàng hoặc slot gold-present tương ứng. ViSFD OTHERS được tách khỏi pair F1; tên polarity Restaurant đúng là `1 positive, 2 negative, 3 neutral` theo E12 audit.

| Task | Lợi thế đo được của Cafe trên validation | Cách đọc giới hạn |
|---|---|---|
| Topic | 57 câu Cafe-only đúng, 38 Bami-only; macro F1 +2,21 điểm. Training program recall 74,5→83,2%; Others F1 55,2→61,7%. | Tập trung ở hai lớp, nhưng nhiều câu rất ngắn/thiếu chủ thể; không suy ra encoder hiểu quan hệ tốt hơn từ paired correctness đơn thuần. |
| Restaurant | TP +22, FP −47; pair F1 +2,98 điểm. Wrong-polarity 125→107, spurious 156→127, miss 132→128. | Lợi thế rải giữa nhận diện aspect và polarity; không có một loại lỗi chi phối toàn bộ chênh lệch. |
| ViSFD | TP +32, FP −42; pair F1 +1,08 điểm. Wrong-polarity 253→233, spurious 303→281, miss 237→225. | Cũng là cải thiện phân tán; Cafe vẫn sai nhiều case đa aspect/mixed sentiment. |

Không lấy tỷ lệ trong packet 72 case để ước lượng tỷ lệ nguyên nhân ở toàn validation: packet được cân bằng có chủ đích theo outcome/error type. Bảng toàn validation chỉ cho biết **loại sai theo mã gold**, chưa cho biết cơ chế neural hoặc gold có hợp lý trong từng trường hợp.

## Những gì 72 case cho thấy

### Topic

Có một số Cafe-only thắng có vẻ có căn cứ: `vsfc_topic:validation:000292` nói về nỗ lực của bản thân, nên Bami gán `lecturer` có vẻ sai; `000303` nói về việc dừng ở mức khái niệm, có thể thuộc chương trình/nội dung học. Nhưng `001443` chỉ viết “áp lực nặng” mà gold là `lecturer`, `000189` “không áp lực khi học tập” cũng không nêu chủ thể; Cafe khớp gold ở các câu này chưa chứng minh hiểu chủ đề. Chiều ngược cũng tồn tại: Bami-only `000418` nói rõ nâng cấp phần mềm (`facility`), Cafe gán `training_program`; `000946` nói đề khó và Bami gán `training_program` có lý. Câu “good” (`000296`) có gold `lecturer` nhưng thiếu antecedent; không nên xem đây là bằng chứng năng lực của Bami.

Do đó **gain Topic đo được là thật theo evaluator**, nhưng packet không chỉ ra một hiện tượng ngôn ngữ chung và sạch để thiết kế kiến trúc. Nó có thể gồm năng lực model, prior nhãn và giới hạn của câu feedback rời ngữ cảnh; chưa tách được các phần này.

### Restaurant

Cafe có những sửa lỗi đáng tin: `absa_restaurant:validation:000692` nhận ra SERVICE negative trong “kh thích sự không theo thứ tự”, nơi Bami chọn neutral; `000717` giữ positive cho FOOD#STYLE&OPTIONS khi câu khen món trực tiếp. Bami cũng có thắng lợi rõ: `000625` giữ SERVICE negative trong “Tệ từ cách phục vụ đến chất lượng” còn Cafe bỏ slot; `000336` nhận ra AMBIENCE negative từ “trang trí ... hơi chán”, Cafe neutral.

Một số khác phụ thuộc ranh giới ontology hoặc gold: `000637` gold RESTAURANT#MISC positive từ cách đóng gói nước chấm, còn Bami chọn FOOD#STYLE positive; không thể gọi sự khác biệt này là lỗi ngôn ngữ rõ của Bami. `000696` Bami thêm FOOD#STYLE positive cho “Súp ăn lạ và thơm” dù gold chỉ có FOOD#QUALITY positive; extra slot này có thể là diễn giải hợp lý. `000626` gold FOOD#STYLE positive dù câu nói nem lụi và bún thịt nướng “không ngon cho lắm”; cả hai model chọn neutral. `000720` chỉ cho giá “hơn 100k/người” nhưng gold positive và cả hai model neutral. Các ví dụ đó **cần guideline/adjudication**, không tự sửa nhãn trong audit.

Vì Cafe hơn nhờ giảm cả spurious lẫn wrong-polarity, trong khi vài spurious của Bami có thể là gold thiếu, chưa có bằng chứng một module aspect-binding hoặc chỉnh ngưỡng nào sẽ thắng rộng. E9/E20 và kiểm prior nhãn trước đây cũng chưa cho gain sạch rộng.

### ViSFD

Cafe-only có ví dụ tốt: `visfd:validation:000427` phát hiện SCREEN negative trong câu nói màn hình ngả đỏ; `000787` giữ PERFORMANCE positive từ mô tả hoạt động nhanh. Bami-only cũng có ví dụ tốt: `001018` nhận ra CAMERA negative từ camera trước “khá tệ” trong review pha khen/chê; `000802` giữ SCREEN neutral cho điểm màn hình 5/10, nơi Cafe dự đoán positive.

Case `000661` đáng cảnh giác: câu nói màn hình có “chấm”, gold SCREEN absent; Bami dự đoán SCREEN negative nên bị tính spurious, nhưng dự đoán ấy **có căn cứ từ văn bản**. `000328` GENERAL neutral trong câu vừa “ok” vừa “hết sức thất vọng” là bài toán gộp nhận xét, có thể đọc khác nhau nếu không có guideline. Cả hai model cùng sai một số câu đa aspect hoặc cảm xúc trái chiều (`000493`, `000783`, `000066`), nhưng những case này không chứng minh cùng một lỗi kiến trúc và có gold cần giải thích.

## Quyết định nghiên cứu

**Chưa có một bottleneck chung, đủ sạch và đủ rộng để khuyên train kiến trúc mới nhằm vượt CafeBERT trên nhiều task.** Topic là phân loại chủ đề với nhiều fragment thiếu ngữ cảnh; hai ABSA task có lợi thế Cafe phân tán giữa polarity, spurious và missed slots. Packet gồm cả lỗi model thực và nhãn/ontology có thể tranh cãi. Chỉ một seed và checkpoint đã chọn bằng validation còn để ngỏ dao động seed; so sánh này không cô lập pretraining, kích thước hay tokenizer.

Hướng tiếp theo có giá trị cao nhất là **adjudication mù với prediction** trên một mẫu disagreement đã khóa, dùng guideline gốc và ít nhất hai người gán nhãn độc lập, ưu tiên các ca nhãn/ontology quyết định chênh lệch. Mục tiêu là đo Cafe-only/Bami-only nào thật sự có gold hỗ trợ, không “sửa” benchmark để tăng điểm. Nếu sau đó hiện ra một lỗi rõ lặp ở cả Restaurant và ViSFD, mới thiết kế một can thiệp **một encoder pass** đúng lỗi đó và so với đối chứng có cùng compute. Nếu không có pattern ổn định, dừng vòng thử kiến trúc; cần dữ liệu/train supervision hoặc một benchmark độc lập hơn trước khi tiếp tục claim thắng rộng.

Không mở thêm training, test, sweep, hoặc sửa prediction/gold từ review này. Mọi hướng được chọn vì đã xem validation errors phải được xác nhận ở dữ liệu độc lập trước khi tuyên bố cải thiện tổng quát.
