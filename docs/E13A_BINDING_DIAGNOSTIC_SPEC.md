# E13A — Cặp kiểm tra gán bằng chứng và phạm vi nhận xét

Root đã đọc và chốt các kỳ vọng dưới đây trước inference. Đây là mẫu tổng hợp phục vụ hypothesis testing, không phải benchmark độc lập hoặc ước lượng prevalence. Không đưa vào train. Không chỉnh kỳ vọng sau khi thấy model output. Nếu câu bị trùng train/validation, ghi rõ và tách khỏi nhóm mới; không đọc test để kiểm tra trùng.

Map numeric: Sentiment0negative1neutral2positive; Topic0lecturer1training_program2facility3others; cả baABSA0absent1positive2negative3neutral, ViSFDOTHERS không đánh giá ở đây. ABSA chỉ chấm slots ghi dưới, không tự gán absent cho mọi slot còn lại.

Chạy E8seed42 và E11signed/convexseed42 trên chính các input này. Lưu full logits để review; diagnostic gate metadata/status phải thể hiện root-reviewed spec. Không sử dụng expectedIDs từ loader polarity names sai. Bước này đã được user cho phép trong phạm vi tìm ý tưởng, không cần xin lại.

## VSFC Sentiment

| ID | Loại | A | B | Expected A→B |
|---|---|---|---|---|
| S1 | paraphrase | Thầy giảng bài rất dễ hiểu. | Bài giảng của thầy rất dễ hiểu. | 2→2 |
| S2 | negation | Thầy giảng bài dễ hiểu. | Thầy giảng bài không dễ hiểu. | 2→0 |
| S3 | negated_negative | Thầy dạy rất khó hiểu. | Thầy dạy không hề khó hiểu. | 0→2 |
| S4 | desire_scope | Mong cô tiếp tục cách dạy dễ hiểu này. | Mong cô thay đổi cách dạy khó hiểu này. | 2→0 |
| S5 | speaker_contrast | Có người nói cô dạy khó hiểu, nhưng tôi thấy cô dạy rất dễ hiểu. | Có người nói cô dạy dễ hiểu, nhưng tôi thấy cô dạy rất khó hiểu. | 2→0 |
| S6 | reported_disagreement_control | Tôi thấy cô dạy rất dễ hiểu, dù có người chê cô dạy khó hiểu. | Dù có người chê cô dạy khó hiểu, tôi vẫn thấy cô dạy rất dễ hiểu. | 2→2 |

S5/S6 đánh giá cảm nhận của người viết, không tổng hợp polarity của mọi người; ghi assumption này rõ. S3 là cách nói đánh giá tích cực nhẹ, có thể nhạy annotation; report riêng và không dùng một mình để chốt cơ chế.

## VSFC Topic

| ID | Loại | A | B | Expected A→B |
|---|---|---|---|---|
| T1 | paraphrase | Máy chiếu trong phòng học bị hỏng. | Phòng học có máy chiếu bị hỏng. | 2→2 |
| T2 | entity_context | Máy chiếu trong phòng học bị hỏng. | Máy chiếu trong phòng học do thầy phụ trách bị hỏng. | 2→2 |
| T3 | target_change | Giảng viên thường xuyên vắng mặt trong buổi thực hành. | Các buổi thực hành bị cắt khỏi chương trình đào tạo. | 0→1 |
| T4 | negation_preserves_topic | Giảng viên dạy nhiệt tình. | Giảng viên dạy không nhiệt tình. | 0→0 |
| T5 | target_change | Nội dung chương trình đào tạo quá nặng. | Máy chiếu của phòng học quá mờ. | 1→2 |
| T6 | event_paraphrase | Giáo viên không đến lớp trong buổi thực hành. | Buổi thực hành diễn ra nhưng giáo viên vắng mặt. | 0→0 |

Topic classification có convention riêng; T2 có nhắc lecturer nhưng đối tượng bị đánh giá là máy chiếu. Ghi nhóm này riêng để không coi ambiguity possible như lỗi neural chắc chắn.

## ViSFD

| ID | Loại | A | B | Slots expected A→B |
|---|---|---|---|---|
| V1 | swap_binding | Pin dùng rất lâu nhưng camera chụp ảnh rất xấu. | Camera chụp ảnh rất đẹp nhưng pin hết rất nhanh. | BATTERY 1→2; CAMERA 2→1 |
| V2 | clause_reorder | Pin dùng rất lâu nhưng camera chụp ảnh rất xấu. | Camera chụp ảnh rất xấu nhưng pin dùng rất lâu. | BATTERY 1→1; CAMERA 2→2 |
| V3 | accessory_target | Pin của điện thoại bị hỏng, còn sạc dự phòng được tặng thì dùng tốt. | Pin của điện thoại dùng tốt, còn sạc dự phòng được tặng thì bị hỏng. | BATTERY 2→1; SER&ACC 1→2 |
| V4 | negation | Camera chụp ảnh đẹp. | Camera chụp ảnh không đẹp. | CAMERA 1→2 |
| V5 | added_other_target | Pin máy dùng rất lâu. | Pin máy dùng rất lâu, nhưng nhân viên phục vụ thiếu nhiệt tình. | BATTERY 1→1; SER&ACC onlyB2 |
| V6 | paraphrase | Nhân viên bán hàng rất nhiệt tình. | Người bán hàng phục vụ rất tận tình. | SER&ACC 1→1 |

V3 dựa ontology SER&ACC gồm service/accessory; theo dõi riêng, không coi là gold span detection task.

## Restaurant

| ID | Loại | A | B | Slots expected A→B |
|---|---|---|---|---|
| R1 | swap_binding | Món ăn ngon nhưng nhân viên phục vụ rất bất lịch sự. | Món ăn dở nhưng nhân viên phục vụ rất lịch sự. | FOOD#QUALITY 1→2; SERVICE#GENERAL 2→1 |
| R2 | clause_reorder | Món ăn ngon nhưng nhân viên phục vụ rất bất lịch sự. | Nhân viên phục vụ rất bất lịch sự nhưng món ăn ngon. | FOOD#QUALITY 1→1; SERVICE#GENERAL 2→2 |
| R3 | role_paraphrase | Nhân viên phục vụ rất thân thiện. | Anh quản lý phục vụ khách rất thân thiện. | SERVICE#GENERAL 1→1 |
| R4 | negation | Đồ uống có hương vị ngon. | Đồ uống có hương vị không ngon. | DRINKS#QUALITY 1→2 |
| R5 | added_other_target | Không gian quán rất sạch sẽ. | Không gian quán rất sạch sẽ nhưng món ăn có mùi hôi. | AMBIENCE#GENERAL 1→1; FOOD#QUALITY onlyB2 |
| R6 | target_change | Giá món ăn quá đắt. | Giá đồ uống quá đắt. | FOOD#PRICES onlyA2; DRINKS#PRICES onlyB2 |

## Hotel

| ID | Loại | A | B | Slots expected A→B |
|---|---|---|---|---|
| H1 | swap_binding | Phòng rất sạch nhưng nhân viên phục vụ bất lịch sự. | Phòng rất bẩn nhưng nhân viên phục vụ lịch sự. | ROOMS#CLEANLINESS 1→2; SERVICE#GENERAL 2→1 |
| H2 | clause_reorder | Phòng rất sạch nhưng nhân viên phục vụ bất lịch sự. | Nhân viên phục vụ bất lịch sự nhưng phòng rất sạch. | ROOMS#CLEANLINESS 1→1; SERVICE#GENERAL 2→2 |
| H3 | paraphrase | Phòng ở được vệ sinh sạch sẽ. | Phòng ốc rất sạch sẽ. | ROOMS#CLEANLINESS 1→1 |
| H4 | negation | Nhân viên lễ tân lịch sự. | Nhân viên lễ tân không lịch sự. | SERVICE#GENERAL 1→2 |
| H5 | added_other_target | Bữa sáng rất ngon. | Bữa sáng rất ngon nhưng phòng ngủ rất bẩn. | FOOD&DRINKS#QUALITY 1→1; ROOMS#CLEANLINESS onlyB2 |
| H6 | target_change | Giá phòng quá đắt. | Giá bữa sáng quá đắt. | ROOMS#PRICES onlyA2; FOOD&DRINKS#PRICES onlyB2 |

## Metrics và kiểm tra

30pairs,60occurrences; có input được lặp giữa control và swap, phải báo unique_text_count và không coi mọi observation độc lập. Lưu expectation list riêng mỗi side; onlyA/onlyB không yêu cầu chấm counterpart absent. Target_change chỉ cho biết chấm đúng aspect explicit, không dùng làm bằng chứng loại trừ false positives ở slot không được chấm.

Report mỗi task/model: expectedslot count/correct, pairbothcorrect, tỷ lệ fail trên cặp labelpreserving có Ađúng, pairtransitioncorrect trên cặp có expectedchange. Với mixedaddedslots, đánh giá preservation ở sharedslots, addition riêng. Candidate-vsE8 regression = count(E8correct AND candidatewrong)/count(E8correct), không đếm mọi disagreement ở tử số.

Không điều chỉnh nhãn theo predictions. Báo perpair predictions để root review, không gộp thành một scalarSOTA. Diagnosticsynthetic không dùng để kết luận prevalence hoặc thắng BamiBERT. Nếu E8 đã làm tốt, hypothesis phải được hạ ưu tiên, không tự sinh thêm câu khó cho đến khi model sai.
