# Kết quả dựng corpus v2 cho general distillation

Plan khóa: `docs/GD_RECIPE_ABLATION_PLAN.md` §2–3. Dựng bằng CPU, seed 42. Tổng token được đếm lại bằng tokenizer Bami trong `models/goiturn-s-gd`.

## Thành phần

| Nguồn | v1 token Bami | v2 token Bami | v2 doc | Tỷ lệ v2 |
|---|---:|---:|---:|---:|
| web | 307699789 | 232889660 | 284991 | 0.708658819 |
| wiki | 0 | 49353825 | 41901 | 0.150178515 |
| ugc | 0 | 25456887 | 70952 | 0.077462638 |
| domain | 20934020 | 20934020 | 511885 | 0.063700028 |
| Tổng | 328633809 | 328634392 | 909729 | 1 |

Domain v1 được đo từ file v1 bằng `sort | uniq -c`, chọn dòng vật lý có tần suất ≥20 và chia hết cho 20: 735000 lượt dòng, 20934020 token Bami, tỷ lệ 0.06370014108925719. Bộ CSV lịch sử của script v1 không còn tại đường dẫn cũ; phép đo này có thể tính cả dòng CC100 tình cờ lặp bội số 20. Domain v2 lặp đầy đủ 9 lần và thêm 24706 doc ở lượt thứ 10 để đạt tỷ lệ; không doc nào vượt 20 lần.

General web được lấy ngẫu nhiên từ đúng manifest v2 `general_docs.jsonl` gồm 1163152 doc, đã qua bin 7–9, lọc spun/MT, trừ register, host cap 2% và near-dedup. Nguồn thô HPLT đối chiếu SHA từ `sha256.txt`; Wikipedia lấy text gốc từ parquet. UGC là doc thuộc `ugc_refined` nhưng đoạn được khôi phục bằng bộ lọc v2 trên raw HPLT.

## Các bước lọc

| Nguồn | Chỉ số | Giá trị |
|---|---|---:|
| domain | docs_after_filter | 54131 |
| domain | docs_decontam_dropped | 1586 |
| domain | docs_empty_after_filter | 72 |
| domain | docs_input | 55789 |
| domain | paragraphs_after_filter | 84436 |
| domain | paragraphs_decontam_dropped | 115 |
| domain | paragraphs_exact_dedup_dropped | 483 |
| domain | paragraphs_input | 87235 |
| ugc | docs_after_filter | 70952 |
| ugc | docs_decontam_dropped | 30 |
| ugc | docs_empty_after_filter | 3334 |
| ugc | docs_input | 78886 |
| ugc | paragraphs_after_filter | 485811 |
| ugc | paragraphs_decontam_dropped | 16 |
| ugc | paragraphs_exact_dedup_dropped | 34950 |
| ugc | paragraphs_input | 520858 |
| web | docs_after_filter | 591258 |
| web | docs_decontam_dropped | 347 |
| web | docs_empty_after_filter | 1 |
| web | docs_input | 553528 |
| web | paragraphs_after_filter | 8931290 |
| web | paragraphs_decontam_dropped | 22848 |
| web | paragraphs_exact_dedup_dropped | 627624 |
| web | paragraphs_input | 9586754 |
| web | paragraphs_over_2048_dropped | 206 |
| wiki | docs_after_filter | 41901 |
| wiki | docs_decontam_dropped | 667 |
| wiki | docs_input | 29781 |
| wiki | paragraphs_after_filter | 1011934 |
| wiki | paragraphs_decontam_dropped | 7355 |
| wiki | paragraphs_exact_dedup_dropped | 311649 |
| wiki | paragraphs_input | 1408592 |
| wiki | paragraphs_over_2048_dropped | 2 |

Đoạn exact dedup dùng NFC + casefold + gộp khoảng trắng trên toàn bộ nguồn; các bản lặp domain được thêm sau dedup. Web boilerplate tần suất ≥5 vì vậy không thể còn trong phần web. Doc được cắt tại ranh giới đoạn, ngưỡng 2.048 token Bami.

## Khử nhiễm

Text validation/test chỉ dùng làm khóa; nhãn test không được đọc. Tám task có bảy nhóm text vì VSFC Sentiment và Topic dùng chung câu, được nạp một lần dưới tên VSFC. Text từ 13 âm tiết trở lên tạo tất cả 13-gram; text ngắn hơn đòi trùng hoàn toàn sau chuẩn hóa. Doc mất >20% đoạn bị loại toàn bộ. Bảng dưới ghi các lần dính theo task/split; một đoạn hoặc doc có thể thuộc nhiều hàng.

| Nguồn | Task/split | Đoạn loại | Doc loại |
|---|---|---:|---:|
| web | hotel/test | 166 | 0 |
| web | hotel/validation | 3 | 0 |
| web | phoner/test | 600 | 55 |
| web | phoner/validation | 753 | 67 |
| web | restaurant/test | 39 | 0 |
| web | restaurant/validation | 1 | 0 |
| web | vinli/test | 645 | 47 |
| web | vinli/validation | 228 | 14 |
| web | visfd/test | 0 | 0 |
| web | visfd/validation | 0 | 0 |
| web | vispam/test | 18276 | 50 |
| web | vispam/validation | 17552 | 49 |
| web | vsfc/test | 3367 | 185 |
| web | vsfc/validation | 1189 | 18 |
| wiki | hotel/test | 0 | 0 |
| wiki | hotel/validation | 0 | 0 |
| wiki | phoner/test | 0 | 0 |
| wiki | phoner/validation | 1 | 0 |
| wiki | restaurant/test | 0 | 0 |
| wiki | restaurant/validation | 0 | 0 |
| wiki | vinli/test | 38 | 0 |
| wiki | vinli/validation | 8 | 0 |
| wiki | visfd/test | 0 | 0 |
| wiki | visfd/validation | 0 | 0 |
| wiki | vispam/test | 38251 | 667 |
| wiki | vispam/validation | 37731 | 662 |
| wiki | vsfc/test | 962 | 8 |
| wiki | vsfc/validation | 280 | 7 |
| ugc | hotel/test | 0 | 0 |
| ugc | hotel/validation | 0 | 0 |
| ugc | phoner/test | 15 | 8 |
| ugc | phoner/validation | 12 | 6 |
| ugc | restaurant/test | 0 | 0 |
| ugc | restaurant/validation | 0 | 0 |
| ugc | vinli/test | 15 | 12 |
| ugc | vinli/validation | 8 | 4 |
| ugc | visfd/test | 0 | 0 |
| ugc | visfd/validation | 0 | 0 |
| ugc | vispam/test | 5 | 2 |
| ugc | vispam/validation | 9 | 4 |
| ugc | vsfc/test | 0 | 0 |
| ugc | vsfc/validation | 0 | 0 |
| domain | hotel/test | 171 | 171 |
| domain | hotel/validation | 50 | 50 |
| domain | phoner/test | 471 | 465 |
| domain | phoner/validation | 384 | 380 |
| domain | restaurant/test | 4 | 4 |
| domain | restaurant/validation | 2 | 2 |
| domain | vinli/test | 43 | 11 |
| domain | vinli/validation | 18 | 1 |
| domain | visfd/test | 35 | 34 |
| domain | visfd/validation | 10 | 9 |
| domain | vispam/test | 664 | 541 |
| domain | vispam/validation | 340 | 280 |
| domain | vsfc/test | 103 | 100 |
| domain | vsfc/validation | 66 | 62 |

## Kiểm cuối trên file đã ghi

| Kiểm | Trạng thái |
|---|---|
| exact_matches_zero | PASS |
| token_total_within_5pct | PASS |
| packed_first_100_full_readable_vi | PASS |
| web_duplicate_ratio_below_1pct | PASS |
| sha_matches_manifest | PASS |
| stats_token_count_matches_reread | PASS |
| doc_count_matches_manifest | PASS |
| ugc_le_10pct | PASS |
| domain_repeat_le_20 | PASS |

SHA256 `mlm_mix.txt`: `3ba6537ced296669373f6d3e652a8d5a75e152e97865ff7333f75c43896295c1`. Trùng đoạn web: 0/4302033 = 0.0. `READY`: đã ghi.

| Task/split | Text unique | Khớp |
|---|---:|---:|
| vsfc/validation | 1583 | 0 |
| visfd/validation | 1111 | 0 |
| restaurant/validation | 771 | 0 |
| hotel/validation | 795 | 0 |
| vsfc/test | 3166 | 0 |
| visfd/test | 2224 | 0 |
| restaurant/test | 1937 | 0 |
| hotel/test | 2030 | 0 |
| vispam/validation | 1589 | 0 |
| vispam/test | 3971 | 0 |
| vinli/validation | 3387 | 0 |
| vinli/test | 3367 | 0 |
| phoner/validation | 2000 | 0 |
| phoner/test | 2999 | 0 |

100 chuỗi packed đầu dùng đúng `iter_packed_sequences`; ví dụ decode nằm trong `data/corpus_v2/final_check.json`.

## Audit 20 doc ngẫu nhiên

| # | Nguồn | Trích đoạn (≤400 ký tự) |
|---:|---|---|
| 1 | web | Thứ tư, 30/10/2024, 14:35 GMT+7 Bạn đang mang thai và băn khoăn về việc sanh non có ảnh hưởng như thế nào? Câu hỏi này đã được giải đáp tại Câu lạc bộ Tiền sản tổ chức vào 30/10/2024. Cùng nhìn lại buổi gặp mặt đầu tiên đầy ý nghĩa của Câu lạc bộ Tiền sản tại Bệnh viện Đa khoa Tâm Trí Cao Lãnh, nơi các mẹ bầu đã có cơ hội lắng nghe những chia sẻ quý báu từ chuyên gia sản khoa. Việc sanh non sẽ cản |
| 2 | domain | Cơm gà sốt đậu là món mình thích nhất, ăn rất vừa miệng. |
| 3 | web | mà nhiều người dân cũng như cán bộ của ngành bưu chính viễn thông dành cho bệnh viện Bưu điện danh xưng: "Bệnh viện không phong bì". Suốt nhiều năm liền, bệnh viện đã dành được niềm tin từ người dân và trở thành điểm sáng trong ngành y tế, thế nhưng "scandal" của nguyên Giám đốc Nguyễn Văn Oai đang làm sụp đổ niềm tin đó. Cũng theo nguồn tin riêng của PV, vẫn còn những "nhân tố khác" đang bị nghi  |
| 4 | domain | Giao hàng nhanh. Máy có hình thức đẹp, chi tiết. Lực matxa hơi nhẹ nếu so với các máy khác nhưng chấp nhận đc |
| 5 | ugc | Huawei P30 và P30 Pro lộ nhiều ảnh render sắc nét: Màn hình cong 3D, 3 camera Bộ đôi flagship Galaxy S10, Galaxy S10+ đã trình làng và chuẩn bị lên kệ, vì vậy, nhiều tín đồ công nghệ đang chuyển hướng sự chú ý tới bộ đôi siêu phẩm khác, đó là Huawei P30 và P30 Pro. Sau khi xuất hiện hình ảnh thực tế tại MWC 2019 thì mới đây, Huawei P30 và P30 Pro đã lộ nhiều hình ảnh render sắc nét, nhờ vậy mà nhi |
| 6 | domain | sản phẩm cũng ổn mọi người ... y hình |
| 7 | wiki | Toàn bộ Đoàn chủ tịch, bao gồm Tổng Bí thư Miloš Jakeš, đã từ chức, và Karel Urbánek, một người Cộng sản ôn hòa hơn, được đưa lên vị trí Tổng Bí thư. Truyền hình Liên bang đã cho thấy hình ảnh từ ngày 17 tháng 11 lần đầu tiên và truyền trực tiếp truyền hình diễn văn đầu tiên của Václav Havel, chủ yếu liên quan đến cuộc tổng đình công theo kế hoạch. Đài phát thanh và truyền hình Tiệp Khắc tuyên bố  |
| 8 | domain | Các nhà nghiên cứu phát hiện cầu dẫn nước Valens dài hơn 500 km không có cặn lắng nhờ thiết kế cho phép vệ sinh kênh mà không gián đoạn dòng chảy. Thiết kế của cầu dẫn nước Valens vừa làm cho dòng nước chảy mượt mà vừa vệ sinh được kênh chảy. Cầu dẫn nước Valens không có một hạt cặn nào nhờ thiết kế cho phép vệ sinh mà không làm tắc nghẽn dòng chảy. Valens dài chỉ khoảng quãng đường đi được tối đa |
| 9 | web | Thực phẩm giàu chất xơ, chứa nhiều citrate... có lợi cho sức khỏe người bệnh. Những năm gần đây, nhờ sự phát triển của ngành Y - Dược, điều trị sỏi thận tiết niệu đã có những tiến bộ vượt bậc. Bên cạnh đó, dinh dưỡng hợp lý cũng đóng vai trò quan trọng trong quá trình chữa sỏi thận. Theo nhiều tài liệu y học, thực phẩm giàu chất xơ có lợi trong việc điều trị sỏi thận. Chúng có tác dụng làm sạch cơ |
| 10 | ugc | Ngày hôm qua, bộ đôi iPhone 7 và iPhone 7 Plus đã chính thức được bán ra tại một số quốc gia trên thế giới và theo lệ thường, các chuyên gia từ website chuyên mổ xẻ thiết bị di động iFixit lại tiếp tục xếp hàng tại Úc (nơi chiếc iPhone ra mắt sớm nhất tính theo múi giờ) để có thể đưa siêu phẩm nhà Táo lên bàn mổ càng sớm càng tốt. Hình ảnh từ GeekBar đã xác nhận dung lượng pin 1960mAh trên iPhone  |
| 11 | wiki | Tự chủ là thời kỳ đầu khôi phục lại nền độc lập của Việt Nam đầu thế kỷ 10 sau hơn 1000 năm Bắc thuộc. Thời kỳ này, người Việt đã tự cai trị lãnh thổ nhưng chưa xưng làm vua, và chưa đặt quốc hiệu. Bề ngoài, Việt Nam vẫn là một phần lãnh thổ của "thiên triều" phương Bắc ở Trung nguyên với tên gọi "Tĩnh Hải quân" và người đứng đầu chỉ nối tiếp nhau làm chức Tiết độ sứ như một quan cai trị của Trung |
| 12 | web | “Phương Uyên, một cô gái trẻ đến từ Hà Nội, đã có một cuộc sống ổn định với công việc và gia đình hạnh phúc. Tuy nhiên, dưới vẻ ngoài hạnh phúc đó, tâm hồn cô ấy luôn nặng trĩu bởi những biến cố và áp lực trong cuộc sống… Bước ra từ văn phòng lúc chiều muộn. Cô thấy lòng mình nặng trĩu. Dự án hôm nay Lâm cố lắm, dậy sớm thức khuya, nhưng cuối cùng cũng chẳng hiệu quả. Ôm chặt mấy cuốn sách marketi |
| 13 | web | Chiến thắng của U22 Việt Nam khiến cho truyền thông Thái Lan cảm thấy lo lắng cho số phận của đội nhà trước cuộc đại chiến giữa 2 đội vào ngày 5/12 sắp tới. VIDEO: Trận thắng đầy cảm xúc của U22 Việt Nam trước U22 Indonesia (Nguồn: VTV) Vào tối qua, U22 Việt Nam đã có màn lội ngược dòng ngoạn mục trước U22 Indonesia ở môn bóng đá nam SEA Games 30. Bị dẫn trước ngay ở phút thứ 22 sau sai lầm của th |
| 14 | web | [Báo chí] Diễn giả Trần Quốc Phúc và Nhân tâm thu phục lòng người Diễn giả Trần Quốc Phúc (giám đốc điều hành Hội thảo Việt) đã có rất nhiều chia sẻ thú vị xoay quanh đề tài “Nhân tâm thu phục lòng người”. Báo tuổi trẻ giới thiệu bạn đọc bài viết. Anh Trần Quốc Phúc cùng bạn trẻ thảo luận các vấn đề tại hội thảo “Nhân tâm thu phục lòng người” 1. Gieo ân đức Nghe như triết lý nhà Phật, tuy nhiên dù |
| 15 | domain | Rất hài lòng về sản phẩm. Tiki giao hàng nhanh (mình ở Vĩnh Phúc-Vĩnh Yên). Hình thức sản phẩm thì túi có 2 lớp, lớp ngoài dầy, lớp trong ni-lon thường để ngăn cách. Mèo nhà mình ăn vị này thấy khá hợp. |
| 16 | domain | giảng viên nên hạn chế dùng slide , hướng dẫn cụ thể hơn cho sinh viên . |
| 17 | ugc | Mặc dù Apple sẽ không bao giờ để lộ ý tưởng của mình nhưng những nhận xét của Cirkovic ở trên vẫn đem lại cho chúng ta không ít tò mò và hình dung về chiếc Iphone 7 trong tương lai. Mặc dù Apple mới phát hành phiên bản iPhone 6 và iPhone 6 Plus vào tháng trước, nhưng đã có những suy đoán về hình dáng của chiếc iPhone tiếp theo-chiếc iPhone 7và sẽ là bất ngờ hơn nếu nó ra mắt vào năm 2016. Những lý |
| 18 | domain | quá trình giảng dạy luôn lấy ví dụ từ thực tiễn , hài hước tạo nên hứng thú cho sinh viên ! |
| 19 | domain | thầy nên cung cấp đầy đủ slide để sinh viên có thể học tập dễ dàng hơn . |
| 20 | domain | Dùng được hơn 20 ngày. Pin dùng dư 1 ngày. Cam vừa đủ dùng. Sản phẩm tốt cho người dùng phổ thông. |

## Thời gian và giới hạn

Thời gian dựng: 4384.18963265419 giây (không gồm lượt đo v1, sort và kiểm cuối). Giới hạn: UGC raw paragraphs selected by v2 cues/refine rules using original text; v2 intermediate paragraph S1 decisions cannot be reproduced from refined manifest alone; A paragraph exceeding 2048 Bami tokens cannot be split at a paragraph boundary and is dropped; Historical v1 CSV inputs are absent; multiple-of-20 frequency can include incidental CC100 boilerplate and miss domain lines with added CC100 occurrences. Nguồn và SHA đầy đủ nằm trong `data/corpus_v2/manifest.json`.
