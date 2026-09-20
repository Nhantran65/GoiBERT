# Packet 72 trường hợp — CafeBERT vs BamiBERT (validation, seed 42)

Chọn theo quota §4 và thứ tự SHA256 cố định; **không** là mẫu ngẫu nhiên. Mã ABSA `0/1/2/3 = absent/positive/negative/neutral`. Cột `gold_supported / evidence_span / possible_failure / ambiguity_note` để root điền.

## absa_restaurant

### cafe_only (8)

**1. `absa_restaurant:validation:000637`** — event `miss` — target **RESTAURANT#MISCELLANEOUS** (slot 9)
> Tuy nhiên nước chấm được đóng gói kiểu chuyên nghiệp hơn là giống như mấy gói gia vị trong mì gói á, có răng cưa để xé ra.
- gold: `1` positive · Bami: `0` absent (miss) · Cafe: `1` positive (correct)
- gold vector: RESTAURANT#MISCELLANEOUS=positive  ·  gold-present scored: 1
- Bami vector: FOOD#STYLE&OPTIONS=positive
- Cafe vector: RESTAURANT#MISCELLANEOUS=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `absa_restaurant:validation:000473`** — event `miss` — target **RESTAURANT#GENERAL** (slot 8)
> Quán này ngày trước cứ ra Đồ Sơn là mới được ăn bây giờ mở thêm chi nhánh ở Lê Hồng Phong nên đi lại tiện lắm luôn ấy ^-^.
- gold: `1` positive · Bami: `0` absent (miss) · Cafe: `1` positive (correct)
- gold vector: LOCATION#GENERAL=neutral, RESTAURANT#GENERAL=positive  ·  gold-present scored: 2
- Bami vector: LOCATION#GENERAL=positive
- Cafe vector: LOCATION#GENERAL=positive, RESTAURANT#GENERAL=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `absa_restaurant:validation:000304`** — event `miss` — target **LOCATION#GENERAL** (slot 7)
> Rất thích hợp để họp mặt bạn bè, ăn uống tiệc tùng gia đình, không gian rộng rãi buổi tối thì nghe em mình nói cao cấp hơn, có hào ăn thoải mái mình ăn dìn ký từ hồi còn ở q1, không nhớ tên đường, đến sau này mở thêm ở Cộng Hoà, thì mình chuyển qua đó ăn luôn.
- gold: `3` neutral · Bami: `0` absent (miss) · Cafe: `3` neutral (correct)
- gold vector: AMBIENCE#GENERAL=positive, FOOD#STYLE&OPTIONS=positive, LOCATION#GENERAL=neutral  ·  gold-present scored: 3
- Bami vector: AMBIENCE#GENERAL=positive
- Cafe vector: AMBIENCE#GENERAL=positive, LOCATION#GENERAL=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `absa_restaurant:validation:000163`** — event `wrong_polarity` — target **DRINKS#QUALITY** (slot 2)
> Ly Protein Pro đậm mùi chuối, như mùi sữa bột( mình nghĩ là do có bột Whey).
- gold: `1` positive · Bami: `3` neutral (wrong_polarity) · Cafe: `1` positive (correct)
- gold vector: DRINKS#QUALITY=positive  ·  gold-present scored: 1
- Bami vector: DRINKS#QUALITY=neutral
- Cafe vector: DRINKS#QUALITY=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**5. `absa_restaurant:validation:000717`** — event `wrong_polarity` — target **FOOD#STYLE&OPTIONS** (slot 6)
> Heo giả cầy đậm đà, thơm, chấm với bánh mì ăn ngon.
- gold: `1` positive · Bami: `3` neutral (wrong_polarity) · Cafe: `1` positive (correct)
- gold vector: FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=positive  ·  gold-present scored: 2
- Bami vector: FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=neutral
- Cafe vector: FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**6. `absa_restaurant:validation:000692`** — event `wrong_polarity` — target **SERVICE#GENERAL** (slot 11)
> Quán rất đông và mình chấp nhận đợi gần 30ph nhưng mình kh thích sự không theo thứ tự.
- gold: `2` negative · Bami: `3` neutral (wrong_polarity) · Cafe: `2` negative (correct)
- gold vector: RESTAURANT#GENERAL=positive, SERVICE#GENERAL=negative  ·  gold-present scored: 2
- Bami vector: RESTAURANT#GENERAL=neutral, SERVICE#GENERAL=neutral
- Cafe vector: RESTAURANT#GENERAL=neutral, SERVICE#GENERAL=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**7. `absa_restaurant:validation:000696`** — event `spurious` — target **FOOD#STYLE&OPTIONS** (slot 6)
> Súp ăn lạ và thơm 🔴.
- gold: `0` absent · Bami: `1` positive (spurious) · Cafe: `0` absent (correct_absent)
- gold vector: FOOD#QUALITY=positive  ·  gold-present scored: 1
- Bami vector: FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=positive
- Cafe vector: FOOD#QUALITY=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**8. `absa_restaurant:validation:000375`** — event `spurious` — target **FOOD#QUALITY** (slot 5)
> Dạo này gà viên lotte cho ít quá, hơi buồn, lúc trước phần cơm cho gà nhiều hơn bây giờ, tại mình thích cái sốt của nó :(((.
- gold: `0` absent · Bami: `1` positive (spurious) · Cafe: `0` absent (correct_absent)
- gold vector: FOOD#STYLE&OPTIONS=negative  ·  gold-present scored: 1
- Bami vector: FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=negative
- Cafe vector: FOOD#STYLE&OPTIONS=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### bami_only (8)

**1. `absa_restaurant:validation:000599`** — event `miss` — target **RESTAURANT#MISCELLANEOUS** (slot 9)
> Mình khá thích trang trí của quán, những bức tranh đen trắng gợi nhớ Cố đô, bát đĩa men gắn nghệ thuật cảm giác rất gần gũi.
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `0` absent (miss)
- gold vector: AMBIENCE#GENERAL=positive, RESTAURANT#MISCELLANEOUS=positive  ·  gold-present scored: 2
- Bami vector: AMBIENCE#GENERAL=positive, RESTAURANT#MISCELLANEOUS=positive
- Cafe vector: AMBIENCE#GENERAL=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `absa_restaurant:validation:000327`** — event `miss` — target **FOOD#STYLE&OPTIONS** (slot 6)
> Mình ăn phần bò 150gr tầm 100k thì phải, chất lượng, bò hơi tái nên ăn mềm ngọt thịt.
- gold: `3` neutral · Bami: `3` neutral (correct) · Cafe: `0` absent (miss)
- gold vector: FOOD#PRICES=neutral, FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=neutral  ·  gold-present scored: 3
- Bami vector: FOOD#PRICES=neutral, FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=neutral
- Cafe vector: FOOD#PRICES=neutral, FOOD#QUALITY=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `absa_restaurant:validation:000625`** — event `miss` — target **SERVICE#GENERAL** (slot 11)
> Tệ từ cách phục vụ đến chất lượng.
- gold: `2` negative · Bami: `2` negative (correct) · Cafe: `0` absent (miss)
- gold vector: RESTAURANT#GENERAL=negative, SERVICE#GENERAL=negative  ·  gold-present scored: 2
- Bami vector: RESTAURANT#GENERAL=negative, SERVICE#GENERAL=negative
- Cafe vector: RESTAURANT#GENERAL=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `absa_restaurant:validation:000114`** — event `wrong_polarity` — target **LOCATION#GENERAL** (slot 7)
> Quán rất đẹp và sang chảnh nằm ngay gần vincom bà triệu.
- gold: `3` neutral · Bami: `3` neutral (correct) · Cafe: `1` positive (wrong_polarity)
- gold vector: AMBIENCE#GENERAL=positive, LOCATION#GENERAL=neutral  ·  gold-present scored: 2
- Bami vector: AMBIENCE#GENERAL=positive, LOCATION#GENERAL=neutral
- Cafe vector: AMBIENCE#GENERAL=positive, LOCATION#GENERAL=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**5. `absa_restaurant:validation:000336`** — event `wrong_polarity` — target **AMBIENCE#GENERAL** (slot 0)
> Trang trí thì hơi chán, dù giờ có thêm mấy bức tranh vẽ vời này nọ:v.
- gold: `2` negative · Bami: `2` negative (correct) · Cafe: `3` neutral (wrong_polarity)
- gold vector: AMBIENCE#GENERAL=negative  ·  gold-present scored: 1
- Bami vector: AMBIENCE#GENERAL=negative
- Cafe vector: AMBIENCE#GENERAL=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**6. `absa_restaurant:validation:000385`** — event `wrong_polarity` — target **FOOD#STYLE&OPTIONS** (slot 6)
> Lẩu nước ngon, rau nhút hơi già, ghẹ tươi ngon ngọt chắc thịt.
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `3` neutral (wrong_polarity)
- gold vector: FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=positive  ·  gold-present scored: 2
- Bami vector: FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=positive
- Cafe vector: FOOD#QUALITY=positive, FOOD#STYLE&OPTIONS=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**7. `absa_restaurant:validation:000507`** — event `spurious` — target **DRINKS#STYLE&OPTIONS** (slot 3)
> Trà đá và khắn lạnh tính tiền riêng 2 k / cái.
- gold: `0` absent · Bami: `0` absent (correct_absent) · Cafe: `3` neutral (spurious)
- gold vector: RESTAURANT#MISCELLANEOUS=neutral  ·  gold-present scored: 1
- Bami vector: DRINKS#PRICES=neutral, FOOD#PRICES=neutral
- Cafe vector: DRINKS#PRICES=neutral, DRINKS#STYLE&OPTIONS=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**8. `absa_restaurant:validation:000345`** — event `spurious` — target **RESTAURANT#GENERAL** (slot 8)
> Trần đời không lẽ bình thường bán chè bưởi cho khách ko bán nc cốt dừa.
- gold: `0` absent · Bami: `0` absent (correct_absent) · Cafe: `3` neutral (spurious)
- gold vector: SERVICE#GENERAL=negative  ·  gold-present scored: 1
- Bami vector: FOOD#STYLE&OPTIONS=negative
- Cafe vector: RESTAURANT#GENERAL=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### both_wrong (4)

**1. `absa_restaurant:validation:000306`** — event `gold_present` — target **DRINKS#QUALITY** (slot 2)
> Milkfoam ở đây còn ít hơn gongcha mua ở bên bình tân, tsữa k còn béo thơm như xưa mà như nc dão thêm tí đường.
- gold: `2` negative · Bami: `3` neutral (wrong_polarity) · Cafe: `3` neutral (wrong_polarity)
- gold vector: DRINKS#QUALITY=negative, DRINKS#STYLE&OPTIONS=negative  ·  gold-present scored: 2
- Bami vector: DRINKS#QUALITY=neutral, DRINKS#STYLE&OPTIONS=negative
- Cafe vector: DRINKS#QUALITY=neutral, DRINKS#STYLE&OPTIONS=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `absa_restaurant:validation:000626`** — event `gold_present` — target **FOOD#STYLE&OPTIONS** (slot 6)
> Ở đây còn có bún thịt nướng và nem lụi nữa, nhưng mình đến đây chỉ ăn bún riêu cua thôi, còn nem lụi và bún thịt nướng thì không ngon cho lắm.
- gold: `1` positive · Bami: `3` neutral (wrong_polarity) · Cafe: `3` neutral (wrong_polarity)
- gold vector: FOOD#QUALITY=neutral, FOOD#STYLE&OPTIONS=positive  ·  gold-present scored: 2
- Bami vector: FOOD#QUALITY=neutral, FOOD#STYLE&OPTIONS=neutral
- Cafe vector: FOOD#QUALITY=neutral, FOOD#STYLE&OPTIONS=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `absa_restaurant:validation:000556`** — event `gold_present` — target **RESTAURANT#GENERAL** (slot 8)
> Hôm nay đến quán khá vắng dù đã 5h chiều rồi nên chụp ảnh thoải mái mà không bị vướng người gì cả.
- gold: `3` neutral · Bami: `2` negative (wrong_polarity) · Cafe: `1` positive (wrong_polarity)
- gold vector: RESTAURANT#GENERAL=neutral  ·  gold-present scored: 1
- Bami vector: RESTAURANT#GENERAL=negative
- Cafe vector: RESTAURANT#GENERAL=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `absa_restaurant:validation:000720`** — event `gold_present` — target **RESTAURANT#PRICES** (slot 10)
> ▪️Giá hơn 100 k / ng thui.
- gold: `1` positive · Bami: `3` neutral (wrong_polarity) · Cafe: `3` neutral (wrong_polarity)
- gold vector: RESTAURANT#PRICES=positive  ·  gold-present scored: 1
- Bami vector: RESTAURANT#PRICES=neutral
- Cafe vector: RESTAURANT#PRICES=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### both_correct (4)

**1. `absa_restaurant:validation:000346`** — event `correct` — target **FOOD#STYLE&OPTIONS** (slot 6)
> Rau đủ loại rau rau cải rau muốn các thứ và các món ăn kèm như sushi lém.
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `1` positive (correct)
- gold vector: FOOD#STYLE&OPTIONS=positive  ·  gold-present scored: 1
- Bami vector: FOOD#STYLE&OPTIONS=positive
- Cafe vector: FOOD#STYLE&OPTIONS=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `absa_restaurant:validation:000394`** — event `correct` — target **DRINKS#QUALITY** (slot 2)
> Theo mình thì trà sữa ở đây cũng không có gì đặc sắc, không đậm đà, rõ vị, lớp sủi bọt milkfoam bên trên khá nhạt và loãng.
- gold: `2` negative · Bami: `2` negative (correct) · Cafe: `2` negative (correct)
- gold vector: DRINKS#QUALITY=negative  ·  gold-present scored: 1
- Bami vector: DRINKS#QUALITY=negative
- Cafe vector: DRINKS#QUALITY=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `absa_restaurant:validation:000044`** — event `correct` — target **DRINKS#QUALITY** (slot 2)
> Giá cả, chất lượng:trà sữa ô long 35k siêu ngon, hồi trước mình uống đậm vị trà quá chớ bây giờ ngon hơn hẳn rồi ý, thơm nữa, nói chung bị ghiền.
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `1` positive (correct)
- gold vector: DRINKS#PRICES=neutral, DRINKS#QUALITY=positive  ·  gold-present scored: 2
- Bami vector: DRINKS#PRICES=neutral, DRINKS#QUALITY=positive
- Cafe vector: DRINKS#PRICES=neutral, DRINKS#QUALITY=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `absa_restaurant:validation:000679`** — event `correct` — target **RESTAURANT#MISCELLANEOUS** (slot 9)
> Nhân dịp 8/3, cả team mình kéo nhau đi ăn đặt bàn qua tablenow vì đang có mức ưu đãi lẩu 4 người chỉ có 250k mà thực tế giá của nó là 420k.
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `1` positive (correct)
- gold vector: RESTAURANT#MISCELLANEOUS=positive, RESTAURANT#PRICES=neutral  ·  gold-present scored: 2
- Bami vector: RESTAURANT#MISCELLANEOUS=positive, RESTAURANT#PRICES=neutral
- Cafe vector: RESTAURANT#MISCELLANEOUS=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

## vsfc_topic

### cafe_only (8)

**1. `vsfc_topic:validation:001443`** — event `wrong_class`
> áp lực nặng .
- gold: `0` lecturer · Bami: `3` others (wrong_class) · Cafe: `0` lecturer (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `vsfc_topic:validation:001333`** — event `wrong_class`
> điểm danh quá nhiều colonsurprise , silde bài giảng cần phải đầu tư nhiều hơn .
- gold: `1` training_program · Bami: `0` lecturer (wrong_class) · Cafe: `1` training_program (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `vsfc_topic:validation:001199`** — event `wrong_class`
> đa số sinh viên tự học và không liên hệ đến giảng viên thực hành .
- gold: `3` others · Bami: `0` lecturer (wrong_class) · Cafe: `3` others (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `vsfc_topic:validation:000028`** — event `wrong_class`
> ôn tập kỹ trước khi thi hoặc trong lúc giảng dạy cập nhật đến những vấn đề có thể cho ra trong bài thi .
- gold: `0` lecturer · Bami: `1` training_program (wrong_class) · Cafe: `0` lecturer (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**5. `vsfc_topic:validation:000303`** — event `wrong_class`
> không nên dừng lại chỉ ở mức khái niệm .
- gold: `1` training_program · Bami: `0` lecturer (wrong_class) · Cafe: `1` training_program (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**6. `vsfc_topic:validation:000292`** — event `wrong_class`
> không hài lòng với sự cố gắng của bản thân mình .
- gold: `3` others · Bami: `0` lecturer (wrong_class) · Cafe: `3` others (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**7. `vsfc_topic:validation:000189`** — event `wrong_class`
> không áp lực khi học tập .
- gold: `0` lecturer · Bami: `1` training_program (wrong_class) · Cafe: `0` lecturer (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**8. `vsfc_topic:validation:000523`** — event `wrong_class`
> nên mất tiết khá nhiều .
- gold: `1` training_program · Bami: `3` others (wrong_class) · Cafe: `1` training_program (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### bami_only (8)

**1. `vsfc_topic:validation:000296`** — event `wrong_class`
> good .
- gold: `0` lecturer · Bami: `0` lecturer (correct) · Cafe: `3` others (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `vsfc_topic:validation:000946`** — event `wrong_class`
> nhưng lại ra đề rất khó .
- gold: `1` training_program · Bami: `1` training_program (correct) · Cafe: `0` lecturer (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `vsfc_topic:validation:000327`** — event `wrong_class`
> mini test trong cả khóa học nhiều hơn .
- gold: `2` facility · Bami: `2` facility (correct) · Cafe: `1` training_program (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `vsfc_topic:validation:000584`** — event `wrong_class`
> buồn ngủ .
- gold: `3` others · Bami: `3` others (correct) · Cafe: `0` lecturer (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**5. `vsfc_topic:validation:000117`** — event `wrong_class`
> sự nhiệt tình .
- gold: `0` lecturer · Bami: `0` lecturer (correct) · Cafe: `3` others (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**6. `vsfc_topic:validation:001367`** — event `wrong_class`
> áp dụng thực tiễn .
- gold: `1` training_program · Bami: `1` training_program (correct) · Cafe: `3` others (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**7. `vsfc_topic:validation:000418`** — event `wrong_class`
> nâng cấp phiên bản của các phần mềm .
- gold: `2` facility · Bami: `2` facility (correct) · Cafe: `1` training_program (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**8. `vsfc_topic:validation:000742`** — event `wrong_class`
> được tiếp xúc môi trường thực tế ở các công ty .
- gold: `3` others · Bami: `3` others (correct) · Cafe: `1` training_program (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### both_wrong (4)

**1. `vsfc_topic:validation:000549`** — event `wrong_class`
> thiếu kinh nghiệm .
- gold: `0` lecturer · Bami: `1` training_program (wrong_class) · Cafe: `2` facility (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `vsfc_topic:validation:000925`** — event `wrong_class`
> không khí thoải mái , không bị gò bó về vấn đề bài tập , không bị áp lực .
- gold: `1` training_program · Bami: `0` lecturer (wrong_class) · Cafe: `0` lecturer (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `vsfc_topic:validation:000053`** — event `wrong_class`
> mỗi khi cần giảng giải bài tập , kiến thức , vấn đề gì mà sử dụng đến mạng wifi thì rất tệ làm tốn thời gian của thầy và sinh viên .
- gold: `2` facility · Bami: `0` lecturer (wrong_class) · Cafe: `0` lecturer (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `vsfc_topic:validation:000759`** — event `wrong_class`
> không có đề cương trên khoá học .
- gold: `3` others · Bami: `1` training_program (wrong_class) · Cafe: `1` training_program (wrong_class)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### both_correct (4)

**1. `vsfc_topic:validation:001291`** — event `correct`
> thầy nên có đề mẫu , bài mẫu .
- gold: `0` lecturer · Bami: `0` lecturer (correct) · Cafe: `0` lecturer (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `vsfc_topic:validation:001581`** — event `correct`
> em muốn đổi tên môn học vì tên môn là lập trình c fraction cplusplus nhưng lại học thiên về csharp .
- gold: `1` training_program · Bami: `1` training_program (correct) · Cafe: `1` training_program (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `vsfc_topic:validation:000491`** — event `correct`
> nhiều trang thiết bị thực hành trong phòng thực hành điện tử cơ bản đa số hư hỏng nhiều cần phải thay mới .
- gold: `2` facility · Bami: `2` facility (correct) · Cafe: `2` facility (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `vsfc_topic:validation:000248`** — event `correct`
> khả năng tiếng anh .
- gold: `3` others · Bami: `3` others (correct) · Cafe: `3` others (correct)
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

## visfd

### cafe_only (8)

**1. `visfd:validation:000427`** — event `miss` — target **SCREEN** (slot 8)
> Camera chụp chất lượng kém hình ảnh toàn bị đỏ. Ngoài màn hình cũng thiên về màu đỏ
- gold: `2` negative · Bami: `0` absent (miss) · Cafe: `2` negative (correct)
- gold vector: CAMERA=negative, SCREEN=negative  ·  gold-present scored: 2
- Bami vector: CAMERA=negative
- Cafe vector: CAMERA=negative, SCREEN=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `visfd:validation:000279`** — event `miss` — target **GENERAL** (slot 4)
> Mới xài 2 ngày Ok coi YouTube rất mượt mà , lượt Facebook và zalo nhanh, chụp hình qua xuất sắc
- gold: `1` positive · Bami: `0` absent (miss) · Cafe: `1` positive (correct)
- gold vector: CAMERA=positive, GENERAL=positive, PERFORMANCE=positive  ·  gold-present scored: 3
- Bami vector: CAMERA=positive, PERFORMANCE=positive
- Cafe vector: CAMERA=positive, GENERAL=positive, PERFORMANCE=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `visfd:validation:000787`** — event `miss` — target **PERFORMANCE** (slot 6)
> Mình dùng thấy sp oki , pin bền cam nét , đa nghiệm nhanh ,  nv thì nhiệt tình khỏi bàn luôn , hài lòng
- gold: `1` positive · Bami: `0` absent (miss) · Cafe: `1` positive (correct)
- gold vector: BATTERY=positive, CAMERA=positive, GENERAL=positive, PERFORMANCE=positive, SER&ACC=positive  ·  gold-present scored: 5
- Bami vector: BATTERY=positive, CAMERA=positive, GENERAL=positive, SER&ACC=positive
- Cafe vector: BATTERY=positive, CAMERA=positive, GENERAL=positive, PERFORMANCE=positive, SER&ACC=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `visfd:validation:000328`** — event `wrong_polarity` — target **GENERAL** (slot 4)
> Nói chung xài cũng ok.Nhưng không có kết nối NFC.thua cái Sony xa ultra mình mua cách đây 3 năm. Hết sức thất vọng mà cũng bằng giá.
- gold: `3` neutral · Bami: `1` positive (wrong_polarity) · Cafe: `3` neutral (correct)
- gold vector: FEATURES=negative, GENERAL=neutral  ·  gold-present scored: 2
- Bami vector: FEATURES=negative, GENERAL=positive, PERFORMANCE=negative, PRICE=negative
- Cafe vector: FEATURES=negative, GENERAL=neutral, PRICE=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**5. `visfd:validation:000276`** — event `wrong_polarity` — target **PERFORMANCE** (slot 6)
> Nhìn chung với nhu cầu không game thì cho đến giờ không có gì phải phàn nàn ! Cảm biến vân tay OK. Màn đẹp. Cấu hình thấp nhưng quá đủ cho người không game !!!
- gold: `3` neutral · Bami: `1` positive (wrong_polarity) · Cafe: `3` neutral (correct)
- gold vector: FEATURES=positive, OTHERS=OTHERS, PERFORMANCE=neutral, SCREEN=positive  ·  gold-present scored: 3
- Bami vector: FEATURES=positive, PERFORMANCE=positive, SCREEN=positive
- Cafe vector: FEATURES=positive, GENERAL=positive, PERFORMANCE=neutral, SCREEN=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**6. `visfd:validation:001071`** — event `wrong_polarity` — target **CAMERA** (slot 1)
> Máy ngon giá rẻ anh em nên mua. Chip 720g tốt cam đủ sài màn hình ok nói chung là tuyệt
- gold: `3` neutral · Bami: `1` positive (wrong_polarity) · Cafe: `3` neutral (correct)
- gold vector: CAMERA=neutral, GENERAL=positive, PERFORMANCE=positive, PRICE=positive, SCREEN=positive  ·  gold-present scored: 5
- Bami vector: CAMERA=positive, GENERAL=positive, PERFORMANCE=positive, PRICE=positive, SCREEN=positive
- Cafe vector: CAMERA=neutral, GENERAL=positive, PERFORMANCE=positive, PRICE=positive, SCREEN=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**7. `visfd:validation:000661`** — event `spurious` — target **SCREEN** (slot 8)
> Máy mình mới máy gần , xài kỹ mà bị chấm 1 miếng trên mạng hình , gần cây sóng,,,, pin thì mau tuột
- gold: `0` absent · Bami: `2` negative (spurious) · Cafe: `0` absent (correct_absent)
- gold vector: BATTERY=negative, DESIGN=negative  ·  gold-present scored: 2
- Bami vector: BATTERY=negative, FEATURES=negative, SCREEN=negative
- Cafe vector: BATTERY=negative, FEATURES=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**8. `visfd:validation:000105`** — event `spurious` — target **DESIGN** (slot 2)
> Mua gần 1 tháng. Máy cũng ổn, chụp hình đẹp. Mỗi tội lúc chụp hình lưu hình hơi chậm, lệch tay đi 1 chút là hư luôn bức hình. Với 1 điểm nữa là trời hơi tối chụp nhá flash cũng lưu hình chậm. Còn lại là ổn =)))
- gold: `0` absent · Bami: `2` negative (spurious) · Cafe: `0` absent (correct_absent)
- gold vector: CAMERA=positive, GENERAL=neutral, OTHERS=OTHERS, PERFORMANCE=negative  ·  gold-present scored: 3
- Bami vector: CAMERA=negative, DESIGN=negative, GENERAL=positive
- Cafe vector: CAMERA=positive, GENERAL=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### bami_only (8)

**1. `visfd:validation:000528`** — event `miss` — target **FEATURES** (slot 3)
> Sp khá ổn, pin tạm, thương hiệu dâu đen với cơ chế bảo mật trứ danh nhưng mà snap450 với tầm giá đó thì next
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `0` absent (miss)
- gold vector: BATTERY=neutral, FEATURES=positive, GENERAL=positive, PRICE=negative, SER&ACC=positive  ·  gold-present scored: 5
- Bami vector: BATTERY=neutral, FEATURES=positive, GENERAL=positive, OTHERS=OTHERS, SER&ACC=positive
- Cafe vector: BATTERY=neutral, GENERAL=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `visfd:validation:000515`** — event `miss` — target **DESIGN** (slot 2)
> Mua hồi tháng 12 trúng dịp giảm 1tr quá ngon luôn, xài đến nay máy cũng ổn hiệu năng dư so với mục đích. Chỉ mỗi tội man o hơn dự kiến nên hơi bị cấn, sau khi update thì khi thiết lập vân tay dễ bị lỗi "ko thể thiết lập" ( hễ đến 43% là bị) mà giờ hết r... tgdđ ko phát hành bản up đó nên vẫn cho 5*
- gold: `2` negative · Bami: `2` negative (correct) · Cafe: `0` absent (miss)
- gold vector: DESIGN=negative, GENERAL=neutral, PERFORMANCE=positive, PRICE=positive  ·  gold-present scored: 4
- Bami vector: DESIGN=negative, FEATURES=negative, GENERAL=positive, OTHERS=OTHERS, PERFORMANCE=positive, PRICE=positive
- Cafe vector: FEATURES=negative, GENERAL=positive, OTHERS=OTHERS, PERFORMANCE=neutral, PRICE=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `visfd:validation:000457`** — event `miss` — target **GENERAL** (slot 4)
> Sáng nay mưa máy xong phục vụ nhiệt tình
Chs game tốt mưọt ko giậg lag ,.cam cũng bình thg
Pin châu
Nên mua
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `0` absent (miss)
- gold vector: BATTERY=positive, CAMERA=neutral, GENERAL=positive, PERFORMANCE=positive, SER&ACC=positive  ·  gold-present scored: 5
- Bami vector: BATTERY=positive, CAMERA=neutral, GENERAL=positive, PERFORMANCE=positive, SER&ACC=positive
- Cafe vector: BATTERY=positive, CAMERA=neutral, PERFORMANCE=positive, SER&ACC=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `visfd:validation:000629`** — event `wrong_polarity` — target **FEATURES** (slot 3)
> Sạc pin nóng máy quá .còn cái khác ok tạm dc. Cam không tốt lắm. 4g tạm. Wi-fi tạm
- gold: `3` neutral · Bami: `3` neutral (correct) · Cafe: `2` negative (wrong_polarity)
- gold vector: BATTERY=negative, CAMERA=negative, FEATURES=neutral, GENERAL=positive  ·  gold-present scored: 4
- Bami vector: BATTERY=neutral, CAMERA=negative, FEATURES=neutral, GENERAL=positive
- Cafe vector: BATTERY=negative, CAMERA=negative, FEATURES=negative, GENERAL=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**5. `visfd:validation:000802`** — event `wrong_polarity` — target **SCREEN** (slot 8)
> Thang điểm 10 :
Màn hình hiển thị 5
Chụp ảnh 4
Tốc độ máy 7
Loa  9.5
Pin 6
Nói chung vẫn ổn hơn mấy bạn tàu
- gold: `3` neutral · Bami: `3` neutral (correct) · Cafe: `1` positive (wrong_polarity)
- gold vector: CAMERA=negative, FEATURES=positive, GENERAL=positive, PERFORMANCE=positive, SCREEN=neutral  ·  gold-present scored: 5
- Bami vector: BATTERY=neutral, CAMERA=neutral, FEATURES=positive, GENERAL=positive, PERFORMANCE=neutral, SCREEN=neutral
- Cafe vector: BATTERY=neutral, CAMERA=positive, FEATURES=positive, GENERAL=positive, PERFORMANCE=positive, SCREEN=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**6. `visfd:validation:001018`** — event `wrong_polarity` — target **CAMERA** (slot 1)
> Cá nhân mjh mua con này dùng dc 1 ngày cảm thấy như sau
Ưu điểm trong tầm giá  là pin trâu và sạc cũng nhanh . Màn hình nét đẹp, chơi game ổn định, lướt wep mượt, camers sau chụp đẹp hơn live đời đầu
Nhược điểm là camera trước khá tệ...loa lúc mới mua hơi kém nhưng cập nhật xog loa ổn.. Máy chơi game nhanh nóng ở phía lưng . Ứng dụng nhìu lúc tự thoát ra. Mong hãng sớm có bản vá lỗi. ...
Nói chung là giá 4tr ăn đứt mấy hãng khác và được   vsmast biết quan tâm lắng nghe người dùng.
- gold: `2` negative · Bami: `2` negative (correct) · Cafe: `3` neutral (wrong_polarity)
- gold vector: BATTERY=positive, CAMERA=negative, FEATURES=neutral, PERFORMANCE=neutral, PRICE=neutral, SCREEN=positive, SER&ACC=positive  ·  gold-present scored: 7
- Bami vector: BATTERY=positive, CAMERA=negative, FEATURES=negative, OTHERS=OTHERS, PERFORMANCE=neutral, PRICE=neutral, SCREEN=positive, SER&ACC=positive
- Cafe vector: BATTERY=positive, CAMERA=neutral, FEATURES=negative, PERFORMANCE=neutral, PRICE=neutral, SCREEN=positive, SER&ACC=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**7. `visfd:validation:000939`** — event `spurious` — target **SER&ACC** (slot 9)
> Mới mua 1 ngày mà vân tay k nhạy
Cho mình hỏi thêm tgdd có hỗ trợ nâng lên sum 4g k ạ
- gold: `0` absent · Bami: `0` absent (correct_absent) · Cafe: `3` neutral (spurious)
- gold vector: FEATURES=negative, OTHERS=OTHERS  ·  gold-present scored: 1
- Bami vector: FEATURES=negative, OTHERS=OTHERS
- Cafe vector: FEATURES=negative, OTHERS=OTHERS, SER&ACC=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**8. `visfd:validation:000626`** — event `spurious` — target **GENERAL** (slot 4)
> Máy thì mụot . Nhung nếu ai xài amoled r thì chác k thik màn hình này dâu . Có 1 diều là loa rất rất tệ lun . Kể cả dút tai phone vào
- gold: `0` absent · Bami: `0` absent (correct_absent) · Cafe: `1` positive (spurious)
- gold vector: FEATURES=negative, OTHERS=OTHERS, PERFORMANCE=positive, SCREEN=negative  ·  gold-present scored: 3
- Bami vector: FEATURES=negative, OTHERS=OTHERS, PERFORMANCE=positive, SCREEN=negative
- Cafe vector: FEATURES=negative, GENERAL=positive, SCREEN=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### both_wrong (4)

**1. `visfd:validation:000493`** — event `gold_present` — target **PERFORMANCE** (slot 6)
> Tui đã xài con thứ 2 và cảm thấy rất tốt..
Máy đầu xài ngon nhưng pin hoi tụt tí và con này pin trâu hơn....ko có gì ngon hơn chỉ có cảm mờ , đây là dt để chơi game thôi nhé...ai có chơi game như lq bupg các loại thì cứ mua về mà trải nghiệm....realms 5 cx chất lượng ngang con này nhưng xài tạm tháng là có hơi giật lag nhẹ
...realms 5 và vivo u10 cho ae lựa chọn chơi game
- gold: `1` positive · Bami: `3` neutral (wrong_polarity) · Cafe: `3` neutral (wrong_polarity)
- gold vector: BATTERY=neutral, CAMERA=negative, GENERAL=positive, PERFORMANCE=positive  ·  gold-present scored: 4
- Bami vector: BATTERY=neutral, CAMERA=negative, GENERAL=positive, OTHERS=OTHERS, PERFORMANCE=neutral
- Cafe vector: BATTERY=neutral, FEATURES=negative, GENERAL=positive, OTHERS=OTHERS, PERFORMANCE=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `visfd:validation:000783`** — event `gold_present` — target **PRICE** (slot 7)
> Sau khi sử dụng được 2 tháng mình nhận thấy Máy màn hình đẹp pin ổn, hiệu năng chơi game khá, camera ổn sạc nhanh wiffi mạnh. Nhược điểm là yếu sóng ra ngoài trời phải bật sáng màn hình lên mới xem được, tốc độ phản hồi cảm ứng đôi khi ko nhạy. Nói chung tạm ổn trong tầm giá. Máy mình mua lúc tgdd giảm 1tr
- gold: `1` positive · Bami: `3` neutral (wrong_polarity) · Cafe: `3` neutral (wrong_polarity)
- gold vector: BATTERY=positive, CAMERA=positive, FEATURES=neutral, GENERAL=neutral, PERFORMANCE=positive, PRICE=positive, SCREEN=neutral  ·  gold-present scored: 7
- Bami vector: BATTERY=positive, CAMERA=positive, FEATURES=neutral, GENERAL=neutral, PERFORMANCE=positive, PRICE=neutral, SCREEN=positive
- Cafe vector: BATTERY=positive, CAMERA=positive, FEATURES=negative, PERFORMANCE=positive, PRICE=neutral, SCREEN=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `visfd:validation:000074`** — event `gold_present` — target **GENERAL** (slot 4)
> Sài được 1 tháng.. máy mạnh pin trâu.. ko thấy lỗi gì. Nhưng có 1 bất tiện là máy không có ghi âm cuộc gọi.
- gold: `1` positive · Bami: `0` absent (miss) · Cafe: `0` absent (miss)
- gold vector: BATTERY=positive, FEATURES=negative, GENERAL=positive, PERFORMANCE=positive  ·  gold-present scored: 4
- Bami vector: BATTERY=positive, FEATURES=negative, OTHERS=OTHERS, PERFORMANCE=positive
- Cafe vector: BATTERY=positive, FEATURES=negative, PERFORMANCE=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `visfd:validation:000066`** — event `gold_present` — target **CAMERA** (slot 1)
> Mới đầu sử dụng máy ảnh khá ok nhứng sau 2 tháng mỗi khi phóng to lên x4 là nó thoát ra khỏi máy ảnh
- gold: `3` neutral · Bami: `1` positive (wrong_polarity) · Cafe: `1` positive (wrong_polarity)
- gold vector: CAMERA=neutral  ·  gold-present scored: 1
- Bami vector: CAMERA=positive
- Cafe vector: CAMERA=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

### both_correct (4)

**1. `visfd:validation:000501`** — event `correct` — target **CAMERA** (slot 1)
> Cam chụp khẩu độ không tốt,chụp auto xấu ,mờ trong điều kiện ban đêm
Các chức năng khác không có gì nổi bật để tương xứng với giá tiền.
Theo mình .con này cỡ 5 triệu thì ok
- gold: `2` negative · Bami: `2` negative (correct) · Cafe: `2` negative (correct)
- gold vector: CAMERA=negative, GENERAL=negative, PRICE=negative  ·  gold-present scored: 3
- Bami vector: CAMERA=negative, GENERAL=neutral, PRICE=neutral
- Cafe vector: CAMERA=negative, GENERAL=neutral, PRICE=neutral
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**2. `visfd:validation:000646`** — event `correct` — target **FEATURES** (slot 3)
> Mới mua bản iphone 7 plus 128GB xài đựoc, bấm liên tục thấy pin cũng lâu hết, vân tay rất nhạy.
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `1` positive (correct)
- gold vector: BATTERY=positive, FEATURES=positive, GENERAL=neutral  ·  gold-present scored: 3
- Bami vector: BATTERY=positive, FEATURES=positive, STORAGE=positive
- Cafe vector: BATTERY=positive, FEATURES=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**3. `visfd:validation:000148`** — event `correct` — target **PERFORMANCE** (slot 6)
> Máy em mua hồi thứ 7 ,  mọi thứ đều rất tốt chỉ vô album ảnh  tải rất chậm chừng nữa phút đến một phút hơn luôn mới tải được vài tấm .
- gold: `2` negative · Bami: `2` negative (correct) · Cafe: `2` negative (correct)
- gold vector: GENERAL=positive, PERFORMANCE=negative  ·  gold-present scored: 2
- Bami vector: GENERAL=positive, OTHERS=OTHERS, PERFORMANCE=negative
- Cafe vector: GENERAL=positive, PERFORMANCE=negative
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

**4. `visfd:validation:000435`** — event `correct` — target **SER&ACC** (slot 9)
> Máy dùng tốt,đẹp và sang trọng cấu hình tốt và nhân viên viễn thông-mạo khê-quảng ninh tư vấn nhiệt tình,vui tính
- gold: `1` positive · Bami: `1` positive (correct) · Cafe: `1` positive (correct)
- gold vector: DESIGN=positive, GENERAL=positive, PERFORMANCE=positive, SER&ACC=positive  ·  gold-present scored: 4
- Bami vector: DESIGN=positive, GENERAL=positive, PERFORMANCE=positive, SER&ACC=positive
- Cafe vector: DESIGN=positive, GENERAL=positive, PERFORMANCE=positive, SER&ACC=positive
- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____

