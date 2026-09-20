# Mẫu adjudication 0c — ViSFD validation, E8 seed42, lỗi polarity của aspect thiểu số

## 1. visfd:validation:000050 | aspect BATTERY | gold neu → pred neg | majority pos
Text: Máy đẹp pin trâu sạc hơi lâu và nhanh nóng gần camera nhân viên tư vấn nhiệt tình

Gold: {'BATTERY': 'neu', 'DESIGN': 'pos', 'SER&ACC': 'pos'}

Pred: {'BATTERY': 'neg', 'DESIGN': 'pos', 'PERFORMANCE': 'neg', 'SER&ACC': 'pos'}

## 2. visfd:validation:000725 | aspect PERFORMANCE | gold neu → pred pos | majority pos
Text: Đánh giá sau gần 20 ngày sử dụng .
Mọi thứ đều ổn, chụp hình đẹp , camera sắc nét chân thật , không biết co hao pin không chứ 1 ngày sạc 3 lần bằng củ sạc siêu tốc 45w chắc từ 20% lên 80 tầm hơn nửa tiếng game chiến mượt  mọi thứ vẫn ổn nhưng có 1 điểm trừ không biết là do máy hay do phần mềm chứ mình sài fb nhiều mỗi lần vào fb trả lời cmt thì toàn là không thành công hầu như ngày nào củng bị, đã đem máy đi đổi  chưa sài nên chưa biết , được cái nhân viên tư vấn nhiệt tình.

Gold: {'BATTERY': 'neu', 'CAMERA': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'neu', 'SER&ACC': 'pos'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'SER&ACC': 'pos'}

## 3. visfd:validation:000447 | aspect BATTERY | gold neu → pred neg | majority pos
Text: Giá hạt dẻ mà nhận được nhiều tiện ích. Điện thoại sẽ càng ít hao pin hơn nếu bạn bật chế độ hao pin thấp. Phải nói là trâu bò nhất ở phân khúc 3tr. Loa to, phim ảnh facebook trải nghiệm - đa nhiệm tốt lắm. Đáng mua

Gold: {'BATTERY': 'neu', 'FEATURES': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos'}

Pred: {'BATTERY': 'neg', 'FEATURES': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos'}

## 4. visfd:validation:000783 | aspect SCREEN | gold neu → pred pos | majority pos
Text: Sau khi sử dụng được 2 tháng mình nhận thấy Máy màn hình đẹp pin ổn, hiệu năng chơi game khá, camera ổn sạc nhanh wiffi mạnh. Nhược điểm là yếu sóng ra ngoài trời phải bật sáng màn hình lên mới xem được, tốc độ phản hồi cảm ứng đôi khi ko nhạy. Nói chung tạm ổn trong tầm giá. Máy mình mua lúc tgdd giảm 1tr

Gold: {'BATTERY': 'pos', 'CAMERA': 'pos', 'FEATURES': 'neu', 'GENERAL': 'neu', 'PERFORMANCE': 'pos', 'PRICE': 'pos', 'SCREEN': 'neu'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'FEATURES': 'neg', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu', 'SCREEN': 'pos'}

## 5. visfd:validation:001095 | aspect BATTERY | gold neu → pred pos | majority pos
Text: Mới mua 2 chiếc tại tgdd. Máy dùng ngon, nhân viên phục vụ ngọt. Máy khoẻ pin trâu dùng lâu hơi nóng :))

Gold: {'BATTERY': 'neu', 'GENERAL': 'pos', 'SER&ACC': 'pos'}

Pred: {'BATTERY': 'pos', 'GENERAL': 'pos', 'OTHERS': 'pos', 'PERFORMANCE': 'neu', 'SER&ACC': 'pos'}

## 6. visfd:validation:000663 | aspect PERFORMANCE | gold neu → pred neg | majority pos
Text: Máy qúa tốt.trong phân khúc.mình không chơi game,chủ yếu nghe gọi và cv,nên pin phải nói rất trâu.mọi người nên mua,còn nếu ai chơi game nặng thì nên mua những máy có cấu hình cao hơn.nói chung nokia mình rất hài lòng.

Gold: {'BATTERY': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'neu', 'PRICE': 'neu', 'SER&ACC': 'pos'}

Pred: {'BATTERY': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'neg', 'SER&ACC': 'pos'}

## 7. visfd:validation:001024 | aspect CAMERA | gold neu → pred pos | majority pos
Text: Test được 1 tuần xong đi du lịch . Máy pin cực trâu đối với người làm việc với điện thoại nhiều như mình . Màng hình cài chế độ đêm rất đẹp camara thì bít chách chụp mới ra nhu8ng tam hình đẹp và sắc nét . Mang hình chơi liên quân không loạn gì  sạc thì tầm 1.5h nói chung là rất ok  theo cảm nhận của mình mình đã trải nhiệm sau kỳ du lịch . Đừng nghe nhưng người chưa sài mà la la lên là samsung thua đồ tàu.....

Gold: {'BATTERY': 'pos', 'CAMERA': 'neu', 'FEATURES': 'neu', 'GENERAL': 'pos', 'OTHERS': 'OTHERS', 'SCREEN': 'pos'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'GENERAL': 'pos', 'OTHERS': 'pos', 'PERFORMANCE': 'pos', 'SCREEN': 'pos'}

## 8. visfd:validation:000235 | aspect SER&ACC | gold neu → pred neg | majority pos
Text: ĐT đẹp xịn xịn . Pin trâu . game mượt không lag . đáng giá tiền. Tiếc vì k có tặng kèm ốp dẻo thôi kkk

Gold: {'BATTERY': 'pos', 'DESIGN': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos', 'SER&ACC': 'neu'}

Pred: {'BATTERY': 'pos', 'DESIGN': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos', 'SER&ACC': 'neg'}

## 9. visfd:validation:000227 | aspect FEATURES | gold neu → pred neg | majority pos
Text: Sau thời gian sử dụng. Thì mình thấy như sau.
 -Ưu điểm: máy thiết kế đẹp, chụp hình đẹp. Màn hình to, sáng và đẹp. Sạc nhanh, vân tay dưới màn hình nhạy ( nhớ đặt đúng vị trí), nghe nhạc bằng headphone hay.
- Nhược điểm: pin tụt nhanh hơn Oppo F11 pro, nhưng vẫn đủ dùng- chỉ dùng lướt Facebook , xem Youtube nhiều giờ thì có thể dùng được gần 2 ngày (chơi game nặng và nhiều giờ thì dù có cho Ipad cũng chỉ được 1 ngày). Chip yếu so với Xiaomi cùng giá tiền, bắt wifi kém.
Máy phù hợp với người thích chụp ảnh và thời trang. Không phù hợp với game thủ.

Gold: {'BATTERY': 'neu', 'CAMERA': 'pos', 'DESIGN': 'pos', 'FEATURES': 'neu', 'OTHERS': 'OTHERS', 'PERFORMANCE': 'neg', 'SCREEN': 'pos'}

Pred: {'BATTERY': 'neu', 'CAMERA': 'pos', 'DESIGN': 'pos', 'FEATURES': 'neg', 'PERFORMANCE': 'neg', 'SCREEN': 'pos'}

## 10. visfd:validation:000173 | aspect CAMERA | gold neu → pred pos | majority pos
Text: Máy cầm nặng cảm giác chắc tay, màu xanh lục bảo rất đẹp, máy cho cảm giác như được viền kim loại sang trọng, máy khá mướt xài ít nóng, tuy nhiên cảm giác máy nhanh tuột pin hơn so với con joy 3 mua tháng trước, chụp hình rõ tuy nhiên màu sắc không ưng ý cho lắm

Gold: {'BATTERY': 'neg', 'CAMERA': 'neu', 'DESIGN': 'pos', 'PERFORMANCE': 'pos'}

Pred: {'BATTERY': 'neg', 'CAMERA': 'pos', 'DESIGN': 'pos', 'PERFORMANCE': 'pos'}

## 11. visfd:validation:000001 | aspect PERFORMANCE | gold neu → pred pos | majority pos
Text: Mới sài được 3 ngày sản phẩm thì khá là ok , lỗi thì chưa thấy nhưng chỉ có cái nghe gọi thì loa hơi nhỏ phải bật loa ngoài mới nghe được , chụp hình thì đẹp rõ chơi game thì đối với liên quân nha , lúc không giao tranh thì mạng hơi cao nhưng lúc giao tranh chỉ 8ms hoặc ít hơn , còn bubg thì mượt bắn thì khá là đc mạng thì không cao lắm mua tầm giá này chắc là ok pin sài thì 2 ngày sạc 1 lần có chơi game đấy , 1 trận liên quân mấy chục phút mà tuột có 4-5℅ pin thì khá là ok 👌

Gold: {'BATTERY': 'pos', 'CAMERA': 'pos', 'FEATURES': 'neg', 'GENERAL': 'pos', 'PERFORMANCE': 'neu'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'FEATURES': 'neg', 'GENERAL': 'pos', 'PERFORMANCE': 'pos'}

## 12. visfd:validation:000438 | aspect FEATURES | gold neu → pred neg | majority pos
Text: Đt tốt nhân viên tgdd nhiệt tình ns chung là ok tuy ko có ghi âm cuộc gọi nhưg vẫn tốt

Gold: {'FEATURES': 'neu', 'GENERAL': 'pos', 'SER&ACC': 'pos'}

Pred: {'FEATURES': 'neg', 'GENERAL': 'pos', 'SER&ACC': 'pos'}

## 13. visfd:validation:000656 | aspect BATTERY | gold neu → pred pos | majority pos
Text: Trước khi mua thấy đánh giá cũng ngại. Nhưng qd mua, sau 2 tuần sd thấy sp OK.
Màn hình sáng đẹp, pin sài dc 2 ngày (xem fb hoặc youtube 1h tầm 11%) khi sạt pin các bạn nên tắt hết các ứng dụng chạy ngầm khoảng hơn 2h là đầy pin.
Chưa thử các game nặng nên ko biết. Dt hơi dày nhưng mình là nam sài vừa tay (thích).
Đáng mua☆☆☆☆☆

Gold: {'BATTERY': 'neu', 'DESIGN': 'pos', 'GENERAL': 'pos', 'SCREEN': 'pos'}

Pred: {'BATTERY': 'pos', 'DESIGN': 'pos', 'GENERAL': 'pos', 'SCREEN': 'pos'}

## 14. visfd:validation:001057 | aspect PRICE | gold neu → pred pos | majority pos
Text: Nói về mức giá tầm trung thì con này là ngon nhất rồi, ram 6GB , ROM 128 GB thì quá tuyệt

Gold: {'GENERAL': 'pos', 'PRICE': 'neu', 'STORAGE': 'pos'}

Pred: {'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos', 'STORAGE': 'pos'}

## 15. visfd:validation:000496 | aspect CAMERA | gold neu → pred pos | majority pos
Text: Mình thích nhất là cảm biến khuôn mặt của máy, chỉ cần cầm điện thoại lên là máy tự mở khóa. 
Về pin thì do nhu cầu sử dụng của mình chỉ là nghe gọi, lướt web,.. không chơi game.. nên pin dùng được trung bình là 2 ngày.
Máy chạy mượt, bắt sóng wifi tốt. Camera sau chụp rõ, khá đẹp; camera trước đẹp kiểu hơi ảo diệu. Máy làm việc đa nhiệm rất ổn.

Gold: {'BATTERY': 'pos', 'CAMERA': 'neu', 'FEATURES': 'pos', 'PERFORMANCE': 'pos'}

Pred: {'BATTERY': 'neu', 'CAMERA': 'pos', 'FEATURES': 'pos', 'PERFORMANCE': 'pos'}

## 16. visfd:validation:000429 | aspect CAMERA | gold neu → pred pos | majority pos
Text: Mới mua được 1 tháng nhìn chung máy xài rất ok cấu hình ổn màng hình đẹp, chụp hình ở mức khá tuy nhiên chế độ quay phim chống rung thì máy làm không được tốt lắm hy vọng đời sau phát triển hơn nữa

Gold: {'CAMERA': 'neu', 'GENERAL': 'pos', 'OTHERS': 'OTHERS', 'PERFORMANCE': 'pos', 'SCREEN': 'pos'}

Pred: {'CAMERA': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'SCREEN': 'pos'}

## 17. visfd:validation:001071 | aspect CAMERA | gold neu → pred pos | majority pos
Text: Máy ngon giá rẻ anh em nên mua. Chip 720g tốt cam đủ sài màn hình ok nói chung là tuyệt

Gold: {'CAMERA': 'neu', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos', 'SCREEN': 'pos'}

Pred: {'CAMERA': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos', 'SCREEN': 'pos'}

## 18. visfd:validation:001110 | aspect PRICE | gold neu → pred pos | majority pos
Text: Máy có mẫu mã đẹp, pin sử dụng được 2 ngày, giá phù hợp với sinh viên.
Tôi mua máy màu tím ở Thế giới di động Đồng nai, tôi rất ưng ý.

Gold: {'BATTERY': 'pos', 'DESIGN': 'pos', 'GENERAL': 'pos', 'PRICE': 'neu'}

Pred: {'BATTERY': 'pos', 'DESIGN': 'pos', 'GENERAL': 'pos', 'OTHERS': 'pos', 'PRICE': 'pos'}

## 19. visfd:validation:000422 | aspect PERFORMANCE | gold neu → pred pos | majority pos
Text: Giá tốt trong tầm pin trâu cấu hình ok có một số lỗi vặt . Sau 15 day sử dụng thấy ổn .Mong nhà Xiaomi khắc phục để máy ổn hơn

Gold: {'BATTERY': 'pos', 'GENERAL': 'pos', 'OTHERS': 'OTHERS', 'PERFORMANCE': 'neu', 'PRICE': 'pos'}

Pred: {'BATTERY': 'pos', 'GENERAL': 'pos', 'OTHERS': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos'}

## 20. visfd:validation:000306 | aspect FEATURES | gold neu → pred neg | majority pos
Text: Máy ngon ,game mượt,pin trâu ,camera tuyệt ms lúc đầu khó sử dụng ,chưa nhạy màn hình nhưg khi cập nhật r thì máy mượt hơn ,độ chưa nhạy đã fix 1 chút r .Quá ngon trog tầm 3tr này

Gold: {'BATTERY': 'pos', 'CAMERA': 'pos', 'FEATURES': 'neu', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'FEATURES': 'neg', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu'}

## 21. visfd:validation:000756 | aspect PERFORMANCE | gold neu → pred pos | majority pos
Text: Thiết kế đẹp, sạc khá nhanh, pin khá, chơi game không quá nóng chỉ hơi ấm, chơi game mượt mà,

Gold: {'BATTERY': 'pos', 'DESIGN': 'pos', 'PERFORMANCE': 'neu'}

Pred: {'BATTERY': 'pos', 'DESIGN': 'pos', 'PERFORMANCE': 'pos'}

## 22. visfd:validation:001022 | aspect DESIGN | gold neu → pred pos | majority pos
Text: Mua được 1 ngày, màn khá trong và đẹp, pin trâu, mượt, ko nóng, cam đẹp, có lẽ đã cập nhật phần mềm đầy đủ nên ko thấy lỗi gì như mọi người phản hồi, nhân viên lễ phép, shop thì vắng vẻ chắc do dịch. Máy cầm trên tay cũng đâu có quá nặng và dày, rất vừa vặn. Chỉ có cam quá lồi

Gold: {'BATTERY': 'pos', 'CAMERA': 'pos', 'DESIGN': 'neu', 'PERFORMANCE': 'pos', 'SCREEN': 'pos', 'SER&ACC': 'pos'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'DESIGN': 'pos', 'OTHERS': 'pos', 'PERFORMANCE': 'pos', 'SCREEN': 'pos'}

## 23. visfd:validation:000087 | aspect PERFORMANCE | gold neu → pred neg | majority pos
Text: Chơi game ít Lác 
Pin trâu camera chụp hình đẹp 
NV tư vấn tận tình sẽ còn ủng hộ tgdd
Nhiều

Gold: {'BATTERY': 'pos', 'CAMERA': 'pos', 'PERFORMANCE': 'neu', 'SER&ACC': 'pos'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'PERFORMANCE': 'neg', 'SER&ACC': 'pos'}

## 24. visfd:validation:000971 | aspect PRICE | gold neu → pred pos | majority pos
Text: với tầm giá này thì sản phẩm như vậy chấp nhận được.
trải nghiệm cũng ok, không quá chậm và lag. phù hượp túi tiền

Gold: {'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu'}

Pred: {'GENERAL': 'pos', 'PERFORMANCE': 'neu', 'PRICE': 'pos'}

## 25. visfd:validation:000793 | aspect PERFORMANCE | gold neu → pred pos | majority pos
Text: Chip 855. Mát. Màn to. Quá ngon cho 1 cuộc tình. Chỉ mỗi tội là lưng nhựa. Nên ko sang trọng.

Gold: {'DESIGN': 'neg', 'GENERAL': 'pos', 'PERFORMANCE': 'neu', 'SCREEN': 'pos'}

Pred: {'DESIGN': 'neg', 'PERFORMANCE': 'pos', 'SCREEN': 'pos'}

## 26. visfd:validation:000535 | aspect SER&ACC | gold neu → pred pos | majority pos
Text: Đt mua về xoài rất ngon pin trâu cammera chụp hình đẹp không hề bị như mấy bn khác cmt rất đáng tiền nhân viên phục vụ cũng tạm dk

Gold: {'BATTERY': 'pos', 'CAMERA': 'pos', 'GENERAL': 'pos', 'PRICE': 'pos', 'SER&ACC': 'neu'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'GENERAL': 'pos', 'PRICE': 'pos', 'SER&ACC': 'pos'}

## 27. visfd:validation:000811 | aspect SER&ACC | gold neu → pred pos | majority pos
Text: máy rất mượt, chơi game đã, camera đẹp, thái độ nhân viên vui vẻ nhiệt tình, có điều mình mua máy mới thì muốn tự tay bóc seal hơn. chị nhân viên tự bóc hộp, tự bóc dt mà mình xót hết cả ruột @@

Gold: {'CAMERA': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'SER&ACC': 'neu'}

Pred: {'CAMERA': 'pos', 'PERFORMANCE': 'pos', 'SER&ACC': 'pos'}

## 28. visfd:validation:000813 | aspect FEATURES | gold neu → pred pos | majority pos
Text: Mua ngày 5/3/2020. Máy quá đẹp, 4camera, vân tay ẩn, cường lực tốt, màu trắng đẹp tuyệt vời. Ngon, bổ, rẻ. Đáng để nâng cấp

Gold: {'CAMERA': 'neu', 'DESIGN': 'pos', 'FEATURES': 'neu', 'GENERAL': 'pos', 'OTHERS': 'OTHERS', 'PRICE': 'pos'}

Pred: {'DESIGN': 'pos', 'FEATURES': 'pos', 'GENERAL': 'pos', 'OTHERS': 'pos', 'PRICE': 'pos'}

## 29. visfd:validation:000417 | aspect PERFORMANCE | gold neu → pred neg | majority pos
Text: Máy mượt, vào mạng nhanh và chụp hình đẹp. Nhưng máy rất nhanh nóng khi dùng 4g, chỉ 15 phút dùng 4g là máy lên 45 độ.

Gold: {'CAMERA': 'pos', 'FEATURES': 'pos', 'PERFORMANCE': 'neu'}

Pred: {'CAMERA': 'pos', 'FEATURES': 'pos', 'PERFORMANCE': 'neg'}

## 30. visfd:validation:001079 | aspect PRICE | gold neu → pred pos | majority pos
Text: Mua dc 10 ngày xài rất tốt. Xài lâu máy ko nóng. Pin trâu.nói chung đáng mua trên giá tiền

Gold: {'BATTERY': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu'}

Pred: {'BATTERY': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'pos'}

## 31. visfd:validation:000176 | aspect BATTERY | gold neu → pred neg | majority neg
Text: Sau một ngày sử dụng thật sự rât thât vọng, màn hình full hd+ mà nhìn đục và tối, du đã chỉnh sáng lên 70% . Thân máy tkj dày cộp như những con oppo từ đời đầu,, cầm một lúc tkj mỏi tay,, camera thấy ghi thông số la 48mp ma chup hình còn thua cả 16mp trên oppo nữa.. cấu hình tkj chơi game 
Khôg co j nổi bật ram 4g bô nhơ trống khoảng 42gb,snapdragon 665 .. co mỗi điểm la pin 4000mah. Chưa tes thử pin nên chưa bjêt tụt nhah khôg

Gold: {'BATTERY': 'neu', 'CAMERA': 'neg', 'DESIGN': 'neg', 'GENERAL': 'neg', 'PERFORMANCE': 'neg', 'SCREEN': 'neg'}

Pred: {'BATTERY': 'neg', 'CAMERA': 'neg', 'DESIGN': 'neg', 'OTHERS': 'pos', 'PERFORMANCE': 'neg', 'SCREEN': 'neg'}

## 32. visfd:validation:000629 | aspect FEATURES | gold neu → pred neg | majority neg
Text: Sạc pin nóng máy quá .còn cái khác ok tạm dc. Cam không tốt lắm. 4g tạm. Wi-fi tạm

Gold: {'BATTERY': 'neg', 'CAMERA': 'neg', 'FEATURES': 'neu', 'GENERAL': 'pos'}

Pred: {'BATTERY': 'neu', 'CAMERA': 'neg', 'FEATURES': 'neg', 'GENERAL': 'pos'}

## 33. visfd:validation:001043 | aspect PRICE | gold neu → pred neg | majority neg
Text: Quá thất vọng khi bỏ tiền ra mua 1 chiếc điện thoại samsung với giá hơn 10 triệu đồng. Máy đời cao mà dùng chả khác gì cục gạch, bàn phím lỗi nhảy kí tự lung tung, thi thoảng mất loa phải reset lại máy, máy mua đc tầm hơn 4 tháng hỏng chân sạc, mang đi bảo hành thì cty kêu lỗi chân sạc bị rỉ sét do ngưòi sử dụng không bảo hành, thay thế mất 750k. Quá thất vọng vì bỏ hơn 10tr ra mua 1 cục gạch

Gold: {'DESIGN': 'neg', 'FEATURES': 'neg', 'GENERAL': 'neg', 'PRICE': 'neu'}

Pred: {'FEATURES': 'neg', 'GENERAL': 'neg', 'PRICE': 'neg', 'SER&ACC': 'neg'}

## 34. visfd:validation:000137 | aspect PERFORMANCE | gold neu → pred neg | majority neg
Text: Hiệu suất máy sam sung không bàn tới. Máy bị các điểm trừ như sau:
- Thiết kế dở tệ: nút Volume và nút nguồn đều ở cùng bên phải, muốn chụp màn hình thì cần 2 tay. Muốn thao tác một tay hả, quên đi nhé.
- Camera ngoài mở rộng góc ra chả có gì thêm. Thậm chí không có cả emeoj

Gold: {'CAMERA': 'neg', 'DESIGN': 'neg', 'PERFORMANCE': 'neu'}

Pred: {'CAMERA': 'neg', 'DESIGN': 'neg', 'PERFORMANCE': 'neg'}

## 35. visfd:validation:000346 | aspect BATTERY | gold neu → pred pos | majority neg
Text: Camera trước rất tệ. Màu nhợt nhạt. Ram 8gb nhưng cảm giác k nhạy. Dc cái pin trâu mà k có sạc nhanh quá chán. Lỡ mua chứ k mình đổi qua samsung a71 khỏe hơn.

Gold: {'BATTERY': 'neu', 'CAMERA': 'neg', 'GENERAL': 'neg', 'OTHERS': 'OTHERS', 'PERFORMANCE': 'neg', 'SCREEN': 'neg'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'neg', 'FEATURES': 'neg', 'GENERAL': 'neg', 'PERFORMANCE': 'neg', 'SER&ACC': 'neg'}

## 36. visfd:validation:000954 | aspect BATTERY | gold neu → pred pos | majority neg
Text: Mua xài  được 2  tháng mạng  đơ quá  trời mặc  dù  chạy  bằng  wf cứ  xoay vòng vòng.. camara ổn.  Pin xài  đc 1 ngày. Rất thất  vọng

Gold: {'BATTERY': 'neu', 'CAMERA': 'pos', 'FEATURES': 'neg', 'GENERAL': 'neg'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'FEATURES': 'neg', 'GENERAL': 'neg', 'PERFORMANCE': 'neg'}

## 37. visfd:validation:000522 | aspect PERFORMANCE | gold neu → pred neg | majority neg
Text: Máy mới mua đc 1 ngày mà lỗi quá, vừa bắt wifi yếu vừa hơi lag máy. Đc mỗi cấu hình với cam đẹp

Gold: {'CAMERA': 'pos', 'FEATURES': 'neg', 'GENERAL': 'neg', 'PERFORMANCE': 'neu'}

Pred: {'CAMERA': 'pos', 'FEATURES': 'neg', 'PERFORMANCE': 'neg'}

## 38. visfd:validation:000611 | aspect FEATURES | gold neu → pred neg | majority neg
Text: Mờ khóa vân tay ko đc nhanh, chơi game thì có nhiều hiện tượng lag rất khó chịu, wifi 4g thì cũng ổn định, nghe gọi cũng oke, lướt wed cũng lag. Nói chung là ss a50s ko đánh với cái giá hơn 6tr

Gold: {'FEATURES': 'neu', 'GENERAL': 'neg', 'PERFORMANCE': 'neg', 'PRICE': 'neg'}

Pred: {'FEATURES': 'neg', 'PERFORMANCE': 'neg', 'PRICE': 'neu'}

## 39. visfd:validation:000163 | aspect SER&ACC | gold neu → pred neg | majority neg
Text: Điện thoại mất sóng 4G liên tục.đem bảo hành tại xiaomi báo không bị lỗi về vài tiếng là bị lại.(có hình chụp video lưu lại rõ ràng)mọi người không nên mua quá thất vọng về xiaomi.bù lại nhân viên tại thế giới di động phục vụ tốt nhiệt tình

Gold: {'FEATURES': 'neg', 'GENERAL': 'neg', 'OTHERS': 'OTHERS', 'SER&ACC': 'neu'}

Pred: {'FEATURES': 'neg', 'SER&ACC': 'neg'}

## 40. visfd:validation:001021 | aspect GENERAL | gold neu → pred pos | majority neg
Text: tổng quan tương đối ổn máy nhìn đẹp pin tụt hơinhanh chắc ko dc 4000 nhu quảng cáo.   máy bắt wifi quá tệ hay bị lỗi phần wifi

Gold: {'BATTERY': 'neg', 'DESIGN': 'pos', 'FEATURES': 'neg', 'GENERAL': 'neu'}

Pred: {'BATTERY': 'neg', 'DESIGN': 'pos', 'FEATURES': 'neg', 'GENERAL': 'pos'}

## 41. visfd:validation:000288 | aspect PRICE | gold neu → pred neg | majority neg
Text: Cằm dày rất nhiều so với a10. Còn về hiệu năng thì hơi thấp so với tầm giá. Pin như mog đợi. Camera xấu.bền hay ko còn chờ 10 năm sau đáh giá lại

Gold: {'BATTERY': 'pos', 'CAMERA': 'neg', 'DESIGN': 'neg', 'PERFORMANCE': 'neg', 'PRICE': 'neu'}

Pred: {'BATTERY': 'neg', 'CAMERA': 'neg', 'DESIGN': 'neg', 'PERFORMANCE': 'neu', 'PRICE': 'neg'}

## 42. visfd:validation:000158 | aspect BATTERY | gold neu → pred neg | majority neg
Text: Mới mua xong, theo mình đánh giá. Dt hơi nhỏ, thao tác bị chậm,  Mặt sau mong manh , camera trước cực mờ, có thể gọi là trẻ em như búp trên cành, khá mong manh , . So với một e Motorola cùng phân khúc thì thua. 
Và sau tất cả chỉ cần tỉ lệ nội địa cao thì thua nữa mình vẫn thấy vui khi dùng. Còn về pin thì mới mua nên chưa kiểm được nên để sau

Gold: {'BATTERY': 'neu', 'CAMERA': 'neg', 'DESIGN': 'neg', 'GENERAL': 'neg', 'PERFORMANCE': 'neg', 'SER&ACC': 'neg'}

Pred: {'BATTERY': 'neg', 'CAMERA': 'neg', 'OTHERS': 'pos', 'PERFORMANCE': 'neg'}

## 43. visfd:validation:000970 | aspect GENERAL | gold neu → pred neg | majority neg
Text: Chụp hình quá tệ.không đáng với giá trị bỏ ra.mọi ng cân nhẵc trk khi mua.quá tệ với số tiền trên 13tr

Gold: {'CAMERA': 'neg', 'GENERAL': 'neu', 'PRICE': 'neg'}

Pred: {'CAMERA': 'neg', 'GENERAL': 'neg', 'PRICE': 'neg'}

## 44. visfd:validation:000847 | aspect BATTERY | gold neu → pred neg | majority neg
Text: Sóng dt và wifi hơi yếu,cảm ứng k dc nhạy lắm,pin như 4000mah,came chụp thiếu sáng tệ,cần cho fần mềm cải thiện gấp...

Gold: {'BATTERY': 'neu', 'CAMERA': 'neg', 'FEATURES': 'neg'}

Pred: {'BATTERY': 'neg', 'CAMERA': 'neg', 'FEATURES': 'neg'}

## 45. visfd:validation:000406 | aspect FEATURES | gold neu → pred neg | majority neg
Text: Cảm ứng chán mua rồi sài tg thấy ko đáng. Mua con lấy 10tr mà cảm ứng ntn thì toang sài thua máy hàng 6 7tr những loại máy ntn hy vọng GTDD KO nên nhập về lại dị mất uy tín khi người dùng sài k hài lòng

Gold: {'FEATURES': 'neu', 'GENERAL': 'neg', 'PRICE': 'neg', 'SER&ACC': 'neg'}

Pred: {'FEATURES': 'neg', 'GENERAL': 'neg', 'PRICE': 'neg'}

## 46. visfd:validation:000760 | aspect BATTERY | gold neg → pred neu | majority pos
Text: Tui đã kua về và sử dụng thấy sự thì khá tốt chỉ có cái k có sạc nhanh camera thì tạm ổn nhưng thích nhất camera chính làm ải ảo k cần chỉnh sủa có thể đăng facebook bình thường game thì siêu mượt

Gold: {'BATTERY': 'neg', 'CAMERA': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos'}

Pred: {'BATTERY': 'neu', 'CAMERA': 'neu', 'GENERAL': 'pos', 'PERFORMANCE': 'pos'}

## 47. visfd:validation:000802 | aspect CAMERA | gold neg → pred neu | majority pos
Text: Thang điểm 10 :
Màn hình hiển thị 5
Chụp ảnh 4
Tốc độ máy 7
Loa  9.5
Pin 6
Nói chung vẫn ổn hơn mấy bạn tàu

Gold: {'CAMERA': 'neg', 'FEATURES': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'SCREEN': 'neu'}

Pred: {'BATTERY': 'neu', 'CAMERA': 'neu', 'FEATURES': 'neu', 'GENERAL': 'neu', 'SCREEN': 'neu'}

## 48. visfd:validation:000431 | aspect BATTERY | gold neg → pred pos | majority pos
Text: Em mua A51 đến nay là 3 tháng! E có review như sau:
- Camera của máy chụp ảnh tốt, ko bị nhòe. 
- Game như FF, Liên Quân e đều bật cấu hình cao và ko bị giật lag! 
- Máy chạy mượt, ổn! 
Tuy nhiên cốc sạc nhanh chỉ có 15W nên sạc khá lâu! Tầm 1h30-2h mới đầy!

Gold: {'BATTERY': 'neg', 'CAMERA': 'pos', 'PERFORMANCE': 'pos'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'PERFORMANCE': 'pos'}

## 49. visfd:validation:000449 | aspect GENERAL | gold neg → pred pos | majority pos
Text: Mong đợi là sẽ có 5g.hơi thất vọng.nhưng về cấu hình và mức giá này thì là quá ok

Gold: {'GENERAL': 'neg', 'OTHERS': 'OTHERS', 'PERFORMANCE': 'pos', 'PRICE': 'pos'}

Pred: {'FEATURES': 'neg', 'GENERAL': 'pos', 'PRICE': 'neu'}

## 50. visfd:validation:001051 | aspect FEATURES | gold neg → pred neu | majority pos
Text: ngon lành lắm nha mọi người, vân tay thì chắc do mọi người dùng không quen nên cảm thấy nó không nhậy, màn hình cực kỳ đẹp nha

Gold: {'FEATURES': 'neg', 'GENERAL': 'pos', 'SCREEN': 'pos'}

Pred: {'FEATURES': 'neu', 'GENERAL': 'pos', 'SCREEN': 'pos'}

## 51. visfd:validation:000440 | aspect BATTERY | gold neg → pred neu | majority pos
Text: Máy Oppo Reno 4 Pro mình thấy nó hơi hút pin một tí nếu cải thiện pin một tí nữa thì ok . Cảm biến vân tay dùng tốt . Mình chưa chụp hình nên cũng ko biết camera của nó ra sao . Máy nhẹ cầm nắm thoải mái . Mình mới dùng máy lướt FB với nghe nhạc thôi chứ chưa tìm hiểu sâu .

Gold: {'BATTERY': 'neg', 'CAMERA': 'neu', 'DESIGN': 'pos', 'FEATURES': 'pos', 'OTHERS': 'OTHERS'}

Pred: {'BATTERY': 'neu', 'DESIGN': 'pos', 'FEATURES': 'pos', 'OTHERS': 'pos'}

## 52. visfd:validation:000638 | aspect BATTERY | gold neg → pred pos | majority pos
Text: mua đc 5 ngày Nói chung giá  máy này Sài đc tất cả ổn pin trâu nhung có gam 4G chơi game chạy  hơi yếu cho 5* lùn cô bé tư vấn Huỳnh Thị Thu Sang  Mỹ Long rất dễ thương nhiệt tình

Gold: {'BATTERY': 'neg', 'GENERAL': 'pos', 'PRICE': 'neu', 'SER&ACC': 'pos'}

Pred: {'BATTERY': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'neg', 'SER&ACC': 'pos'}

## 53. visfd:validation:001026 | aspect PRICE | gold neg → pred pos | majority pos
Text: Đặt mua mua 25/8 250k , sáng 26/8 online thấy còn có 200k 😪😪😪.máy to rỏ , ổn .

Gold: {'DESIGN': 'pos', 'GENERAL': 'pos', 'PRICE': 'neg'}

Pred: {'GENERAL': 'pos', 'OTHERS': 'pos', 'PRICE': 'pos'}

## 54. visfd:validation:000781 | aspect CAMERA | gold neg → pred neu | majority pos
Text: ổn, đẹp, đáp ứng tốt nhu cầu cơ bản, cảm biến vân tay nhạy, camera cảm nhận là chưa đẹp, sạc khoảng 3h30p.

Gold: {'BATTERY': 'neu', 'CAMERA': 'neg', 'DESIGN': 'pos', 'FEATURES': 'pos', 'GENERAL': 'pos'}

Pred: {'BATTERY': 'neu', 'CAMERA': 'neu', 'DESIGN': 'pos', 'FEATURES': 'pos', 'GENERAL': 'pos'}

## 55. visfd:validation:000922 | aspect PERFORMANCE | gold neg → pred neu | majority pos
Text: Sài dc 1tuan minh thay ổn, không nóng máy, mọi tác vụ đều tốt, vân tay nhạy, camera k đẹp lắm,,choi game liên quân cấu hinh cao hơi lag, pin tuot hoi nhanh đủ sài 1 ngay, nhờ sạc nhanh nên củng ổn, nhân viên tgdd nhiệt tình vui vẻ,ủng hộ tiep

Gold: {'BATTERY': 'neu', 'CAMERA': 'neg', 'FEATURES': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'neg', 'SER&ACC': 'pos'}

Pred: {'BATTERY': 'neu', 'CAMERA': 'neg', 'FEATURES': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'neu', 'SER&ACC': 'pos'}

## 56. visfd:validation:000298 | aspect SCREEN | gold neg → pred pos | majority pos
Text: Pin có cải thiện hơn so với các dòng cũ,mình xài từ 7h sáng đến 10h tối là còn 20% pin.Có kết hợp xem video,nghe nhạc,lướt fb...lướt khá mượt,màn hình thì cảm nhận k bằng con XSMax. Nhưng cũng xứng với giá tiền.

Gold: {'BATTERY': 'pos', 'FEATURES': 'pos', 'GENERAL': 'pos', 'PRICE': 'neu', 'SCREEN': 'neg'}

Pred: {'BATTERY': 'pos', 'PRICE': 'pos', 'SCREEN': 'pos'}

## 57. visfd:validation:001046 | aspect PRICE | gold neg → pred pos | majority pos
Text: Sản phẩm ok trong tầm giá. Rẻ, khoẻ. Ủng hộ hàng việt nam. Hơi tệ là loa ngoài nghe không êm lắm, âm treble hơi cao nghe ko nhói

Gold: {'FEATURES': 'neg', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neg', 'SER&ACC': 'pos'}

Pred: {'FEATURES': 'neg', 'GENERAL': 'pos', 'PRICE': 'pos', 'SER&ACC': 'pos'}

## 58. visfd:validation:000022 | aspect BATTERY | gold neg → pred pos | majority pos
Text: Sau 2 ngày trải nghiệm thấy máy tốt trong tầm giá,mượt,pin 6p tụt 1% chỉ dùng các tác vụ bình thường kh chơi game có dc kh mọi người(kh hài lòng lắm),camera nét,bắt wifi khỏe.

Gold: {'BATTERY': 'neg', 'CAMERA': 'pos', 'FEATURES': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'pos', 'FEATURES': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu'}

## 59. visfd:validation:000903 | aspect CAMERA | gold neg → pred neu | majority pos
Text: Pin trâu sao ai cũng chê nhỉ. Sài dc 2 ngày cảm thấy mượt, camera trước hơi tệ. Camea sau thì tạm được. Thiết kế đẹp , game liên quân, pubg mượt. Nói chung trong tầm giá thì note 7 vô địch

Gold: {'BATTERY': 'pos', 'CAMERA': 'neg', 'DESIGN': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu'}

Pred: {'BATTERY': 'pos', 'CAMERA': 'neu', 'DESIGN': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu'}

## 60. visfd:validation:000682 | aspect PERFORMANCE | gold neg → pred pos | majority pos
Text: Mình mua máy 3 tuần r, máy chạy khá ổn chỉ hôm bữa bị treo logo Samsung khởi động lại thì bình thường, với giá thế này mà đòi hỏi quá cao thì nên xài hàng china thôi... Chụp hình rất tốt...

Gold: {'CAMERA': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'neg', 'PRICE': 'neu'}

Pred: {'CAMERA': 'pos', 'GENERAL': 'pos', 'PERFORMANCE': 'pos', 'PRICE': 'neu', 'SER&ACC': 'neg'}

