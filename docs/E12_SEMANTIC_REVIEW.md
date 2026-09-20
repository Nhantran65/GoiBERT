# E12 — Review ngữ nghĩa 60 mẫu của root

## Kết luận nghiên cứu

Đã đọc toàn bộ 60 câu trong `artifacts/e12/sample_review.json`, gồm 15 câu/domain. Chưa có cơ sở chọn nhánh ký tự/dấu để cải thiện rộng. Có các lỗi rõ về gán nhận xét vào đúng đối tượng và hiểu đánh giá ngầm, nhưng cũng có nhiều trường hợp gold khó giải thích từ câu đứng riêng. Hai nhóm này phải tách trước khi dùng error analysis để đề xuất block mới.

Đây là review định tính của một người đọc, không phải annotation adjudication. Không đổi gold, không tính lại benchmark bằng nhãn tự sửa. Mẫu đã được chọn có điều kiện đúng/sai, chỉ 15/domain: không suy ra tỷ lệ lỗi toàn bộ dataset từ bảng này. Dấu hiệu ngôn ngữ hiện diện trong câu không chứng minh nguyên nhân neural model sai.

## Vấn đề nhãn cần giải quyết trước diễn giải polarity

`src/goibert/data.py` khai báo UIT-ABSA 1=negative,2=neutral,3=positive; metadata E8 kế thừa khai báo đó. Tuy nhiên, các câu khen rõ trong CSV/sample thường có code1, câu chê thường code2 và liệt kê không đánh giá có code3. Đã giao agent audit nguồn, không tự kết luận thay code dựa trên trực giác.

Trong review dưới đây, khi cần sẽ giữ **code số** và nêu cách đọc câu, tránh gọi code1 là negative theo metadata một cách máy móc. Các trường hợp annotation đáng nghi được phân biệt với vấn đề tên nhãn hệ thống. Nếu chỉ tên polarity sai còn train/gold/pred dùng cùng mã, điều đó không tự làm sai F1 exact-match; cần audit đường dữ liệu trước khi kết luận ảnh hưởng đến training. Diagnostic synthetic hoặc báo cáo gắn tên sentiment có thể chịu ảnh hưởng khác.

VSFC có hai outcome trên cùng câu. Trong file sample, `error` có thể là union của sentiment/topic, còn `gold`, `prediction`, `error_events` hiển thị sentiment. Vì vậy một dòng `error=1` với gold=prediction không có nghĩa evaluator mâu thuẫn. Review tách rõ hai task. ViSFD OTHERS raw4/pred1 là nhãn detection-only, không coi 4→1 là sai polarity.

## Các mẫu đáng dùng để hình thành giả thuyết

| ID | Bằng chứng câu và dự đoán | Diễn giải có thể bảo vệ |
|---|---|---|
| vsfc:04 | “buổi thực hành giáo viên không đi buổi nào”; sentiment đúng, topic lecturer→training_program | Có lỗi gán topic dù đối tượng “giáo viên” hiện diện; không phải mất dấu hoặc không hiểu phủ định toàn câu |
| visfd:05 | Phàn nàn sạc dự phòng được tặng; gold SER&ACC, model BATTERY | Nhầm đối tượng được phàn nàn với từ khóa pin/sạc; hợp với giả thuyết target binding, vẫn cần kiểm tra convention accessory |
| absa_restaurant:00 | “ấn tượng với anh quản lí”; gold SERVICE, model RESTAURANT | Nhận xét về người phục vụ bị chuyển thành nhận xét chung quán; cần biết ontology nhưng không cần sửa ký tự |
| vsfc:09 | “mong cô tiếp tục dạy như vậy nữa”; positive→negative | Nhầm mong muốn duy trì với yêu cầu khắc phục là một giả thuyết hợp lý; không biết chắc model dùng shortcut nào |
| vsfc:11 | “vừa dạy vừa chat hoặc gọi điện thoại thường xuyên”; negative→positive | Không nhận ra đánh giá tiêu cực ngầm về hành vi; proxy thiếu dấu ở từ ngoại ngữ không chứng minh lỗi dấu |
| visfd:03 | “cam gốc rộng hơi thất vọng”; CAMERA negative→neutral | Bỏ nhẹ mức đánh giá ở riêng một aspect trong câu pha trộn |
| absa_hotel:08 | Sạch sẽ và đủ thiết bị được nhận, ROOMS#GENERAL code3 bị bỏ | Có lỗi bỏ phần nhận xét tổng quát nhẹ trong câu có nhiều aspect; không phải bỏ mọi thông tin |

Phản ví dụ bắt buộc: `visfd:02` và `visfd:12` có viết tắt, typo, emoji và nhiều aspect nhưng model vẫn đúng pair labels. `visfd:09` phân biệt đúng CAMERA positive và BATTERY negative. `absa_hotel:10` xử lý được phủ định “không sử dụng được”. Do đó không được kết luận rộng rằng model không hiểu tiếng Việt không chuẩn, không biết phủ định, hoặc không xử lý được multi-aspect.

## Review từng mẫu — VSFC

“Đúng” trong bảng là khớp benchmark; không đồng nghĩa gold đã được root xác nhận độc lập. Chủ đề được đọc theo map hiện hành lecturer/training_program/facility/others.

| ID | Kết quả và nhận định |
|---|---|
| vsfc:00 | Hai task đúng. Hoạt động giao lưu được đánh giá tích cực theo gold; câu đứng riêng không nêu rõ chủ thể lecturer, nên không dùng làm bằng chứng semantic attribution hoàn hảo. |
| vsfc:01 | Sai sentiment, topic đúng. Câu nhận xét ngoại hình có “nhưng”; tiêu cực phụ thuộc thái độ/convention annotation. Không coi “nữ tính” tự thân là thuộc tính tiêu cực. Mẫu mơ hồ, không thích hợp làm diagnostic polarity chắc chắn. |
| vsfc:02 | Sentiment đúng; topic others→training_program. “điều này” thiếu antecedent, có bài thi/bài tập. Topic prediction có lý từ câu riêng; chưa thể quy thành lỗi kiến trúc. |
| vsfc:03 | Positive→negative với “thường xuyên cho bài tập về nhà”. Sự kiện có thể được khen/chê tùy ngữ cảnh; gold không hoàn toàn suy ra từ câu. |
| vsfc:04 | Sentiment đúng; topic nhầm giảng viên thành chương trình. Đây là mẫu attribution tương đối rõ. |
| vsfc:05 | Sentiment đúng dù tên mã hóa/typo; topic gold others, model lecturer trong câu khen cô giáo. Cần convention hoặc kiểm tra annotation, không lấy làm bằng chứng word rarity gây lỗi. |
| vsfc:06 | Hai task đúng; diễn đạt vui tính/liên hệ thực tiễn, không có lỗi cần giải thích. |
| vsfc:07 | Hai task đúng; lời khen trực tiếp về giảng viên và khả năng tiếp thu. |
| vsfc:08 | Hai task đúng; lời khen ngắn, rõ. |
| vsfc:09 | Sai sentiment với mong muốn cô tiếp tục. Đáng kiểm tra sự khác biệt mong duy trì/mong thay đổi; topic đúng. |
| vsfc:10 | Hai task đúng dù có từ “đề cương/đề thi”; phản ví dụ cho quy tắc model luôn nhầm mọi từ khóa chương trình. |
| vsfc:11 | Sai sentiment trong phàn nàn hành vi dạy/chat/gọi điện. Có nghĩa đánh giá ngầm; từ “chat” không tự là lỗi thiếu dấu. |
| vsfc:12 | Hai task đúng; khen dễ hiểu. |
| vsfc:13 | Hai task đúng; nhận xét đúng giờ hàm ý khen theo gold. Phản ví dụ cho kết luận model luôn thiếu đánh giá ngầm. |
| vsfc:14 | Hai task đúng; khen trực tiếp, biểu diễn ít phân mảnh. |

## Review từng mẫu — ViSFD

Map đã khai báo: 1 positive,2 negative,3 neutral; OTHERS không có polarity. Tên aspect lấy từ metric artifact, không suy đoán thứ tự vector.

| ID | Kết quả và nhận định |
|---|---|
| visfd:00 | Sai CAMERA polarity, bỏ FEATURES, thêm PERFORMANCE. Nhiều typo nhưng CAMERA vẫn được phát hiện; “liên quân không loạn” có thể giải thích PERFORMANCE prediction. Cần convention annotation, không quy toàn bộ lỗi cho noise. |
| visfd:01 | Pair đúng; có “ko”, “sài”, game name, phủ định “ko sai lầm”. Phản ví dụ noise/negation. |
| visfd:02 | Pair đúng trên câu dài nhiều emoji/viết tắt và Wifi/Pubg/3G. Phản ví dụ mạnh cho thiếu độ bền bề mặt trên mọi input. |
| visfd:03 | CAMERA negative→neutral dù “hơi thất vọng”; các aspect khác theo gold. Có dấu hiệu làm nhẹ đánh giá hoặc trộn phạm vi, chưa biết cơ chế. |
| visfd:04 | SER&ACC neutral→positive trong câu khen nhân viên nhưng chê quà tặng phải đợi. Có conflict cùng aspect; gold neutral là convention aggregation cần biết. |
| visfd:05 | Bỏ SER&ACC negative và thêm BATTERY negative. Nhầm pin/sạc phụ kiện được tặng với pin máy là lỗi target/ontology có khả năng kiểm chứng bằng cặp câu. |
| visfd:06 | Pair đúng; typo “sang trong” không cản các nhãn được chấm. |
| visfd:07 | Pair đúng với “960fps”, “siêu chậm”, typo “sử dung”. Không thể suy từ rarity đơn thuần sang lỗi model. |
| visfd:08 | FEATURES neutral→negative; PERFORMANCE neutral→positive trong câu nhiều “tạm ổn” và ngoại lệ game nặng lag. Có bài toán aggregation nhận xét trái chiều. OTHERS spurious chỉ detection, không gộp vào pair polarity. |
| visfd:09 | Pair đúng CAMERA positive/BATTERY negative dù “k thích sài” và thời gian/số phần trăm. Model có thể bind đúng hai polarity trong một câu. |
| visfd:10 | Bỏ GENERAL neutral và SER&ACC negative; thêm OTHERS. Phàn nàn máy đã kích hoạt bảo hành dù dùng bình thường đòi hỏi kiến thức giao dịch và chủ thể. Có dấu hiệu sự kiện→đánh giá ngầm. |
| visfd:11 | Pair đúng BATTERY negative/GENERAL positive; bỏ OTHERS detection trong chuyện bị trộm. Không đánh dấu đây là sai sentiment của model. |
| visfd:12 | Pair đúng nhiều aspect, có vân tay gây bất tiện và phần còn lại tích cực, nhiều viết tắt/typo. Phản ví dụ cả surface và scope. |
| visfd:13 | Pair đúng; chỉ bỏ OTHERS detection. Không biến reference iPhone thành bằng chứng lỗi kiến thức model ở pair task. |
| visfd:14 | Bỏ PRICE neutral và SER&ACC positive. Có “tầm giá” giải thích PRICE, nhưng câu không rõ service/accessory; cần convention/annotation cho slot thứ hai. Không coi cả hai đều lỗi semantic chắc chắn. |

## Review từng mẫu — Restaurant

Giữ mã số vì mapping tên UIT-ABSA đang được audit. Các mô tả khen/chê dưới đây là cách đọc văn bản, không tự sửa nhãn.

| ID | Kết quả và nhận định |
|---|---|
| absa_restaurant:00 | Bỏ SERVICE code1, thêm RESTAURANT code1 cho lời khen quản lí. Lỗi gán đối tượng, không phải khác polarity code. |
| absa_restaurant:01 | FOOD#QUALITY 1→3, thêm STYLE&OPTIONS3. Câu đối chiếu khẩu vị “ai ăn mặn...còn mình thấy vừa” cần điểm nhìn người đánh giá; thêm món không nhất thiết thêm style label. |
| absa_restaurant:02 | Khớp SERVICE1 trên lời khen rõ. Đồng thời là bằng chứng cần audit tên code1. |
| absa_restaurant:03 | Khớp SERVICE2 dù thiếu dấu cách/typo, diễn đạt vòng về gọi phục vụ không nghe. Phản ví dụ cho mọi implicit complaint đều thất bại. |
| absa_restaurant:04 | RESTAURANT3→1. Quán vắng là sự kiện nhưng người viết thích chụp ảnh thoải mái; ranh giới neutral/positive cần guideline. |
| absa_restaurant:05 | Khớp FOOD#PRICES3 ở câu chỉ giá. Code3 có vẻ neutral theo nội dung; không coi model đã đọc sentiment từ giá tuyệt đối. |
| absa_restaurant:06 | Bỏ DRINKS#STYLE&OPTIONS3 nhưng SERVICE1 đúng. Liệt kê hai biến thể trà sữa là evidence cho aspect không mang polarity rõ. |
| absa_restaurant:07 | Khớp RESTAURANT#PRICES1 với ưu đãi/coupon; token ngoại ngữ không tự gây lỗi. |
| absa_restaurant:08 | Khớp SERVICE1, khen thân thiện. |
| absa_restaurant:09 | Khớp DRINKS#QUALITY1 với “Milkfoam”; có ngoại ngữ nhưng không lỗi benchmark. |
| absa_restaurant:10 | RESTAURANT1→3. Thay đổi quán vắng→đông sau deco có hàm ý tích cực nhưng không phải đánh giá trực tiếp; khác biệt code cần xác minh. |
| absa_restaurant:11 | Khớp FOOD#QUALITY2 với “khô và ngán hơn”; thêm dấu hiệu metadata polarity không phù hợp code. |
| absa_restaurant:12 | Khớp AMBIENCE1/FOOD#QUALITY1 với câu khen sạch/thoáng/ngon. Không có lý do ngôn ngữ để gọi hai code1 là negative. |
| absa_restaurant:13 | FOOD#QUALITY1 đúng, thêm STYLE&OPTIONS1. Ranh giới chất lượng hương vị/nước chấm đặc biệt và style là vấn đề ontology; không rõ lỗi semantic tổng quát. |
| absa_restaurant:14 | AMBIENCE1 đúng, bỏ RESTAURANT2 dù “chất lượng đã giảm sút”. Lỗi bỏ nhận xét chung ở câu đối lập khá rõ; spelling “ko” chưa đủ làm nguyên nhân. |

## Review từng mẫu — Hotel

| ID | Kết quả và nhận định |
|---|---|
| absa_hotel:00 | Bỏ ROOM_AMENITIES#QUALITY code1 trong câu dột/nước chập chờn. Aspect tương đối liên quan nhưng code1 xung đột cách đọc chê nếu map thực1positive; cần audit raw annotation trước dùng polarity. |
| absa_hotel:01 | Bỏ HOTEL#GENERAL3/ROOM_AMENITIES#GENERAL3, thêm ROOMS#CLEANLINESS2; vẫn nhận giá/vị trí/service3. Nhiều aspect được phủ bởi đánh giá chung “khá ổn...không nổi bật”; phạm vi assessment là vấn đề cụ thể. |
| absa_hotel:02 | Khớp SERVICE1 với lời khen thân thiện. |
| absa_hotel:03 | Khớp SERVICE1 dù đơn vị “hiền” unseen theo feature. Phản ví dụ rarity đồng nghĩa model không hiểu. |
| absa_hotel:04 | Bỏ HOTEL#GENERAL1 và ROOM_AMENITIES#COMFORT2 trong câu “hài lòng, ngoại trừ...” về sàn trơn. Có lỗi phát hiện nhận xét chung và vấn đề cụ thể; mapping sàn→amenity cần ontology. |
| absa_hotel:05 | Khớp LOCATION1/SERVICE1; hai đối tượng tích cực rõ. |
| absa_hotel:06 | Gold FACILITIES#DESIGN&FEATURES1 và SERVICE1, model SERVICE2 trên câu “không hài lòng...tự ý đổi phòng”. Gold cần xem lại theo mapping/annotation; không gọi model không hiểu phủ định vì model đang dự đoán code phù hợp lời chê theo mapping nghi vấn. |
| absa_hotel:07 | Khớp HOTEL#GENERAL1 với câu đặt phòng thông qua Mytour, chưa thấy sentiment rõ. Khớp gold không đồng nghĩa gold hiển nhiên đúng từ câu. |
| absa_hotel:08 | Nhận sạch sẽ/thiết bị, bỏ ROOMS#GENERAL3 ở “không quá tuyệt vời”. Mẫu bỏ assessment nhẹ có ích cho follow-up. |
| absa_hotel:09 | Gold ROOMS#CLEANLINESS2, model1 dù câu “phòng ở sạch sẽ”; service đúng1. Cần audit annotation. Không dùng làm ví dụ model đảo polarity sạch/bẩn. |
| absa_hotel:10 | Khớp FACILITIES#QUALITY2 cho wifi yếu/không sử dụng được. Phản ví dụ phủ định. |
| absa_hotel:11 | Bỏ ROOM_AMENITIES#MISCELLANEOUS3 với đề nghị kiểm tra điện/pin remote. Có nghĩa góp ý và mapping ontology không hiển nhiên; không phải cứ có “pin” là aspect BATTERY như ViSFD. |
| absa_hotel:12 | Khớp sạch/tiện nghi code1 nhưng “chú ý giữ gìn” có thể là yêu cầu. Không coi là positive chắc chắn nếu thiếu ngữ cảnh. |
| absa_hotel:13 | Khớp ROOMS#CLEANLINESS1/COMFORT1/SERVICE1; “dịch vụ...tốt” khá tổng quát. Model có thể phát hiện nhiều nhãn nhưng ontology vẫn cần tôn trọng. |
| absa_hotel:14 | Khớp HOTEL#DESIGN&FEATURES1 dù tên riêng tiếng Anh dài. Không ủng hộ quy mọi fragmentation/foreign token thành lỗi. |

## Quyết định tiếp theo

1. Hoàn tất audit mapping UIT-ABSA và phân biệt ảnh hưởng tên nhãn, metric, diagnostic synthetic. Không retrain hoặc sửa raw labels khi chưa biết phạm vi lỗi.
2. Không launch nhánh ký tự/dấu. Không có pattern surface chung đủ mạnh trong review và thống kê hiện tại.
3. Failure mode đáng kiểm tra hẹp tiếp: **gán evidence vào đúng target/category và phân biệt assessment chung với aspect cụ thể**. Ví dụ rõ nhất là vsfc:04, visfd:05, restaurant:00/14. Có phản ví dụ đúng và coverage hiện chỉ một phần benchmark, chưa đủ đề xuất shared block như kết luận chắc chắn.
4. Nếu tiếp tục diagnostic, dùng các cặp từ chính những mẫu này, thay đối tượng/quan hệ nhưng giữ từ khóa càng nhiều càng tốt, và cặp control không đổi quan hệ. Root chốt kỳ vọng từng task/slot trước inference. Đây là đề xuất mới, chưa thực thi trong review này.
5. Chưa chọn kiến trúc E13. E11 convex tăng giống signed không đủ chứng minh “thêm đường attention” là cách chữa target binding; cần so cùng mẫu trước khi gán cơ chế.

Review hoàn tất 60/60; phần không chắc chắn được giữ nguyên thay vì đổi thành nhãn lỗi model. Nguồn: sample JSON seed12042, metadata/metrics E8 và `src/goibert/data.py`; không dùng test hoặc chạy model mới.
