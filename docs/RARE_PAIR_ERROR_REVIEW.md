# Rà lỗi Restaurant trong nhóm cặp nhãn hiếm

## Phạm vi

Root đọc toàn bộ16câu có lỗi trên gold-positive slots trong cohort đã khóa: validation có đúng2aspect, mỗi atomic label có>=100train examples, pair xuất hiện<=5lần trong train. Văn bản và predictions ở `artifacts/research_diagnostics/composition_frequency/rare_pair_errors_for_review.json`. Đây là đánh giá một người, không adjudication độc lập và không sửa official gold.

## Những lỗi tương đối rõ

| Row | Quan sát | Diễn giải giới hạn |
|---|---|---|
|125|Giá tiền 'trên dưới1tr' bị bỏ sót trong câu dài|Thiếu giữ mention giá không có đánh giá riêng; không phải flip polarity của aspect khác|
|368|Hai loại trà sữa được nêu rõ, model chỉ giữ SERVICE positive|Bỏ sót factual DRINKS#STYLE&OPTIONS neutral|
|392|Danh sách mì/bún/hủ tiếu bị bỏ sót, LOCATION positive được giữ|Bỏ sót factual FOOD#STYLE&OPTIONS neutral|
|431|Cách bán bún/cá riêng bị bỏ sót, RESTAURANT positive được giữ|Có bằng chứng cho offering/style, nhưng category boundary vẫn cần thận trọng|
|292|'nước chấm không quá ngon' được đọc positive|Lỗi xử lý qualifier/phủ định có cơ sở; hai gold aspects đều negative, không phải riêng mixed-polarity|
|543|'ko mắc' được đọc price negative|Lỗi phủ định giá rõ hơn phần RESTAURANT neutral-vs-negative trong cùng câu|

## Các trường hợp nhãn hoặc ontology có thể gây nhiễu diễn giải

| Row | Điểm cần thận trọng |
|---|---|
|82|FOOD#STYLE positive từ việc nhắc món ăn vặt, dù câu chỉ đánh giá cao trà sữa; polarity annotation chưa hiển nhiên|
|391|'giá cả tương đối xứng tầm' mang gold neutral; category price chung hay drinks cũng không hoàn toàn rõ|
|434|RESTGeneral neutral dựa vào quán đông; model chọn FOODStyle từ 'các thể loại bánh', là khác biệt ontology chứ không chỉ mất polarity|
|443|'chất lượng giảm sút' có thể là whole restaurant hoặc food/drinks implicit; missed sentiment có thật nhưng referent chưa tường minh|
|468|'giá ko đổi' factual neutral nhưng chất lượng giảm làm value-for-money negative hợp lý; gold và suy luận đánh giá có thể khác nhau|
|547|'giá1em này' thiếu referent trong câu, FOODPrice vs RESTPrice cần context ngoài câu|
|575|'Giá ổn' được gán neutral trong gold nhưng model positive; không gọi toàn bộ mismatch là không hiểu sentiment|
|683|Tầng lầu/WC/tối thuộc AMBIENCE hay RESTMisc/General phụ thuộc quy ước ontology|
|687|Menu mô tả món, gold RESTGeneral positive không có lời đánh giá trực tiếp; FOODStyle neutral-vs-positive có thể liên quan 'vơ đét'|
|691|Sáng/rộng và ồn cùng câu; tách AMBIENCE neutral và RESTMisc negative là quy ước cần kiểm tra|

Một câu có thể đồng thời chứa lỗi model rõ và nhãn mơ hồ. Các bảng không chia thành hai nhóm true/false gold loại trừ nhau, và không suy tỷ lệ lỗi thật cho toàn dataset.

## Hệ quả nghiên cứu

Không nên gom toàn bộ rare-pair gap thành một failure mode 'polarity interference'. Có ít nhất ba vấn đề: factual neutral mention bị bỏ sót, phủ định/qualifier, và ontology/annotation ambiguity. Probe factorial vừa làm cô lập được một lỗi riêng trong câu nhân tạo, nhưng chưa giải thích mọi trường hợp thật.

Hướng can thiệp sau này phải có subgroup test cho factual neutral mention, thay vì chỉ cứu positive/negative. E3 đã thử phân tách detection/polarity trong objective, nên cần đọc lại kết quả đó trước khi coi đây là ý tưởng mới. Chưa có architecture hoặc loss mới được phê duyệt từ review này.
