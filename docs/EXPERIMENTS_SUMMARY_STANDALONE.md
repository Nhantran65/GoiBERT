# GOIBERT — TỔNG KẾT CÁC THÍ NGHIỆM

**Cập nhật ngày 18/09/2026 · Bản độc lập để chia sẻ**

## 1. Bối cảnh và cách đọc

GoiBERT nghiên cứu encoder tiếng Việt, với mục tiêu đạt chất lượng cao hơn BamiBERT và CafeBERT trên năm tác vụ. Giới hạn kích thước hiện tại khoảng **100–600 triệu tham số**. Giai đoạn đầu tập trung vào student nhỏ; các giai đoạn sau mở rộng sang backbone lớn hơn, kiến trúc, dữ liệu huấn luyện và distillation.

Hai hướng chính đã triển khai:

1. **Post-training:** fine-tune cho từng tác vụ, thay head/kiến trúc/loss, điều chỉnh cách tối ưu, tăng cường dữ liệu và task distillation.
2. **Pretraining:** tiếp tục học ngôn ngữ trên corpus, thay biểu diễn embedding và general distillation để tạo backbone nhỏ.

### Bộ đánh giá chính

| Tác vụ | Nội dung | Train / Validation / Test | Metric chính |
|---|---|---:|---|
| VSFC Sentiment | Cảm xúc trong phản hồi sinh viên | 11.426 / 1.583 / 3.166 | Macro F1 |
| VSFC Topic | Chủ đề của phản hồi sinh viên | 11.426 / 1.583 / 3.166 | Macro F1 |
| ViSFD | Aspect và cảm xúc trong review điện thoại | 7.786 / 1.112 / 2.224 | Pair micro F1 |
| UIT-ABSA Restaurant | Aspect và cảm xúc trong review nhà hàng | 7.028 / 771 / 1.938 | Pair micro F1 |
| UIT-ABSA Hotel | Aspect và cảm xúc trong review khách sạn | 7.180 / 795 / 2.030 | Pair micro F1 |

Hai tác vụ VSFC dùng chung văn bản: năm tác vụ tương ứng bốn miền dữ liệu. **Pair micro F1** chấm đúng đồng thời aspect và polarity; không chỉ chấm cảm xúc khi đã biết aspect đúng.

### Model và thuật ngữ

- **BamiBERT gốc:** khoảng 103M tham số; baseline encoder tiếng Việt.
- **GoiBERT-B / BamiBERT-DAPT:** cùng cỡ 103M, được học tiếp từ BamiBERT gốc; không phải một model huấn luyện từ đầu.
- **GoiTurn-S / XS:** student khoảng 26,5M / 10,6M, tạo bằng distillation.
- **CafeBERT:** khoảng 560M; kế thừa XLM-R-large đã được tác giả học tiếp bằng tiếng Việt.
- **SEA-LION-ModernBERT-300M / mDeBERTa-v3-base:** model có sẵn, khoảng 312M / 279M khi gắn head trong các lượt benchmark của project.
- **CE:** loss học từ nhãn đúng. **KD:** học thêm phân bố dự đoán của teacher. **NC:** yêu cầu dự đoán nhất quán giữa văn bản gốc và bản thêm nhiễu. **LLRD:** cho các tầng encoder dùng learning rate khác nhau.
- **MLM:** che token rồi học dự đoán token gốc. **CPT/DAPT:** tiếp tục pretraining, trong đó DAPT nhắm thích nghi domain. **General distillation (GD):** học biểu diễn bên trong của teacher trên corpus; khác task KD chỉ học đầu ra phân loại.
- **Composite:** ghép các câu train có nhãn tương thích để tạo ví dụ nhiều aspect.

**Quy ước số liệu:** F1 hiển thị trên thang 0–100; Δ là chênh lệch điểm F1 tuyệt đối. `Val` là validation dùng để chọn cấu hình/checkpoint; `test` là đánh giá checkpoint đã chọn. Một seed là một lần khởi tạo/ngẫu nhiên huấn luyện. Các số ± là độ lệch chuẩn mẫu nếu không ghi khác, không phải khoảng tin cậy.

Báo cáo gộp theo chương trình thí nghiệm, không đếm các lượt smoke, khôi phục do lỗi hay kiểm correctness thành phương pháp mới. Mã E1–E22 được giữ để phân biệt các lượt; chúng không tương đương 22 giả thuyết độc lập đều thất bại.

### Mức bằng chứng

- **Đã lặp lại nhiều seed:** đã chạy nhiều seed trong phạm vi được ghi ở từng dòng. Điều này không tự có nghĩa gain dương, ổn định hoặc có ý nghĩa thống kê.
- **Screening một seed:** kết quả sơ bộ của một seed; kể cả có test cũng chưa xác nhận độ ổn định qua seed.
- **Chỉ diagnostic:** phép chẩn đoán, kiểm lỗi hoặc phân tích trên checkpoint có sẵn; không được coi là bằng chứng một phương pháp mới cải thiện benchmark.
- **Hỗn hợp:** các phần trong cùng chương trình có mức xác nhận khác nhau; không mở rộng mức bằng chứng mạnh nhất sang toàn bộ chương trình.


## 2. Kết luận chính

- **Có cải thiện lặp lại:** NC + LLRD trên student 26M; ghép câu composite trên Restaurant với backbone DAPT 103M.
- **Chưa có kiến trúc mới thắng rộng:** các nhánh attention, FFN, reader/routing và invariance chưa đạt mục tiêu trên toàn bộ tác vụ.
- **DAPT không giúp đồng đều:** BamiBERT-DAPT cũ có lợi thế rõ về số điểm ở Topic; CafeBERT-DAPT social chưa giúp Restaurant.
- **Task KD chưa giúp Restaurant:** kết quả âm xuất hiện với cả teacher BamiBERT-DAPT, CafeBERT và CafeBERT-DAPT. Điều này không phủ nhận lợi ích nén model của general distillation.
- **Đổi sang backbone mới chưa thành công:** SEA-LION thua cả hai baseline trên năm tác vụ; mDeBERTa chỉ hơn BamiBERT ở Topic validation.
- Trong các baseline đã đo, **CafeBERT cao nhất ở 4/5 tác vụ test; BamiBERT cao hơn ở Sentiment**. Chưa có hệ thống mới vượt cả hai trên toàn bộ năm tác vụ.

### Ưu tiên hiện tại và lý do — đề xuất chưa kiểm chứng

Sau các pilot backbone mới không đạt mục tiêu, ưu tiên đề xuất hiện tại là **giữ CafeBERT làm nền và xem xét cách đưa thông tin aspect vào đầu vào encoder**, thay vì tiếp tục đổi backbone hoặc lặp lại plain MLM/task KD cùng kiểu đã thử. Cụ thể, model đọc cặp **review + mô tả aspect** trước khi dự đoán; cách này chưa được chạy trong project và khác query đọc token states ở head của E9.

Lý do: CafeBERT đang là baseline mạnh nhất ở 4/5 tác vụ test, còn các lượt DAPT/KD gần đây chưa cải thiện Restaurant. Hướng trên nhắm trực tiếp bài toán ABSA mà giữ kích thước model, nhưng có thể tăng đáng kể số lượt encoder khi xét từng aspect. **Chưa có kết quả cho thấy nó sẽ giúp, và nó không tự giải quyết cả hai tác vụ VSFC.**

Đây là ưu tiên để người nhận hiểu lập luận hiện tại và phản biện, **không phải kế hoạch đã chốt hay yêu cầu phải tiếp tục theo hướng này**. Người nhận tự lựa chọn hướng nghiên cứu từ bằng chứng trong báo cáo.

## 3. Hướng post-training

E1–E13 và E17–E18 chủ yếu dùng student 26M. E14–E16 dùng BamiBERT gốc 103M; E19–E22 dùng BamiBERT-DAPT 103M. E6 là MLM continuation, được trình bày trong phần pretraining. E0/E12 và E13A là chẩn đoán, không phải lượt huấn luyện model mới.

| Thí nghiệm | Mức bằng chứng | Cách làm | Kết quả ngắn gọn |
| --- | --- | --- | --- |
| **E1 — Student CE** | **Đã lặp lại nhiều seed** — 5 seed, baseline. | Fine-tune toàn bộ student GD 26M bằng nhãn đúng | Test trung bình 5 seed: Sentiment **80,60**; Topic **78,62**; ViSFD **79,83**; Restaurant **69,08**; Hotel **71,78**. Baseline ban đầu. |
| **E2 — Task KD và NC** | **Đã lặp lại nhiều seed** — 3 seed; hướng thay đổi tùy tác vụ. | Thử riêng học logits teacher DAPT và consistency với văn bản nhiễu | Val 3 seed so student CE: KD giúp Topic **+0,96**, nhưng Restaurant **−0,59**, Hotel **−0,43**, giảm cả 3 seed. NC giúp Sentiment **+0,44**, ViSFD **+0,99**; Restaurant/Hotel kém ổn định hơn. |
| **E3 — Loss polarity phụ** | **Screening một seed.** | Thêm loss cảm xúc trên các aspect có mặt | Val 1 seed so E1: Restaurant **+1,11**, ViSFD **+0,15**, Hotel **−1,72**. Cảm xúc chính xác hơn ở một số trường hợp nhưng bỏ sót aspect nhiều hơn; không thắng rộng. |
| **E4 — Tăng FFN** | **Screening một seed.** | Mở FFN 1536→3072, tăng student từ 26,5M lên 38,4M | Không tác vụ nào tăng ≥0,3 điểm val; Restaurant **−1,17**. Không đạt. |
| **E5 — Mean pooling** | **Screening một seed.** | Dùng trung bình token thay vector CLS | Val 1 seed: Sentiment **+0,32**, Restaurant **−1,53** so E1; không cải thiện chung. Chỉ kiểm trên student cũ. |
| **E7 — LLRD** | **Screening một seed.** | Giảm learning rate ở các tầng dưới | Val 1 seed so E1: ViSFD **+1,42**, Restaurant **+2,50**, Hotel **+1,25**; Sentiment **−0,44**. Có tín hiệu trên ABSA. |
| **E8 — NC + LLRD** | **Hỗn hợp:** đã lặp lại 3 seed trên ViSFD/Restaurant; các tác vụ còn lại screening 1 seed. | Kết hợp consistency và LR theo tầng | **Test 3 seed so E2 NC:** ViSFD **80,62→81,81 (+1,19)**; Restaurant **69,86→71,88 (+2,02)**; tăng cả 3 seed. Chưa thắng BamiBERT rộng. |
| **E9 — Label-aware attention** | **Screening một seed.** | Mỗi nhãn/aspect có query đọc token states sau encoder | Val 1 seed so E8: Restaurant **+1,03**, Hotel **−0,62**, ViSFD **−0,34**. Không đạt mục tiêu chung. |
| **E10 — Span residual** | **Screening một seed.** | Nhánh gom span theo nội dung, có control convolution | Val 1 seed: gần như không tăng, không có lợi thế rõ so control. |
| **E11 — Signed attention** | **Screening một seed.** | Cho phép kết hợp attention có trọng số âm; so với convex control | Val 1 seed so E8: Restaurant **+1,17**, Hotel **+0,43**, Sentiment **−1,02**. Control gần tương đương; chưa quy được gain cho trọng số âm. |
| **E13 — Pair-value residual** | **Screening một seed.** | Tương tác phi tuyến giữa cặp token; centered vs uncentered | Restaurant tăng **+0,28 val / +0,50 test** so E8 ở 1 seed; các tác vụ khác gần ngang hoặc giảm. Không thắng rộng. |
| **E14 — BamiBERT gốc** | **Screening một seed.** | CE, chọn trong 3 LR, 30 epochs, seed 42 | Thiết lập baseline local đủ val/test. Tốt hơn student E8 ở 4/5 tác vụ test, Topic là ngoại lệ. |
| **E15 — Pair-value trên BamiBERT** | **Screening một seed.** | Chuyển cơ chế E13 sang model 103M | Val 1 seed: chỉ Topic **+0,55** so E14; bốn tác vụ còn lại giảm **0,50–1,51** điểm. |
| **E16 — Attention/FFN gates** | **Screening một seed.** | Gate ở attention update hoặc đầu vào FFN | Attention gate không có gain rõ. FFN gate có tín hiệu val nhưng test 1 seed chỉ Topic **+0,46**; Sentiment **−0,77**, ViSFD **−0,38**, Restaurant **−0,03**, Hotel **−0,40** so baseline cùng lượt. |
| **E17 — Cavity keys** | **Screening một seed.** | Loại đóng góp của token nhận khỏi contextual key; so baseline và full-context | 15 lượt hoàn tất. Cavity−baseline val: Sentiment **0**; Topic **0**; ViSFD **−0,01**; Restaurant **−0,14**; Hotel **−0,19**. Không có gain. |
| **E18 — Fork consistency** | **Screening một seed.** | Huấn luyện đọc theo đoạn, so control che token ngẫu nhiên | Val 1 seed: Restaurant **−0,89**, Hotel **+0,50** so baseline; control ngang hoặc hơn. Cơ chế chẩn đoán thay đổi nhưng chất lượng chung không tăng. |
| **E19 — NC + LLRD trên DAPT 103M** | **Screening một seed.** | Áp recipe E8 lên teacher lớn | Val 1 seed so cùng encoder CE: ViSFD **+0,72**, Hotel **+0,41**; Sentiment **−0,21**, Topic **−0,40**, Restaurant **−0,11**. Không thắng rộng. |
| **E20 — CTI và composite** | **Screening một seed.** | Ghép câu, thêm loss giữ token states bất biến khi ghép | CTI giảm chỉ số nhiễm token Hotel **16,5→2,0%**, nhưng không tạo gain F1 rõ. Composite riêng giúp Restaurant **+0,95 val**, seed 42. |
| **E21 — Xác nhận composite** | **Đã lặp lại nhiều seed** — Restaurant, 3 seed. | Thêm hai seed cho Restaurant | Val trung bình 3 seed **74,10→75,26**; chênh lệch ghép cặp **+1,16±0,18**, tăng cả 3 seed. |
| **E22 — Mở rộng composite** | **Đã lặp lại nhiều seed** — 3 seed; không phải mọi tác vụ đều có gain. | Thử Hotel/ViSFD và lấy cặp đều thay vì ưu tiên cặp hiếm | Val 3 seed: Hotel **+0,40±0,72**, ViSFD **−0,13**. Restaurant lấy cặp đều **75,74**, ưu tiên cặp hiếm **75,26**: chưa chứng minh lợi ích của ưu tiên cặp hiếm. |
| **CafeBERT→BamiBERT KD, M1** | **Screening một seed.** | Teacher Cafe CE; student học 0,5 CE + 0,5 KD, temperature 2 | Restaurant val 1 seed: teacher **77,72**; Bami tự học **74,74**; student KD **73,94 (−0,80)**. Kiểm correctness pass, dừng. |
| **CafeBERT-DAPT→BamiBERT KD, D1** | **Screening một seed.** | Giữ recipe M1, thay teacher đã DAPT social | Student Restaurant **73,34**: **−1,41** so Bami, **−0,61** so KD cũ. Không giúp. Đây là bước cuối cùng của chuỗi DAPT D1, không phải chương trình độc lập thứ hai. |
| **mmBERT-base probe** | **Screening một seed.** | Fine-tune CE, 20 epochs, một LR | Val 1 seed: Restaurant **74,38**, Hotel **73,66**. Gần Bami, chưa vượt Cafe ở Restaurant. Không có CPT mmBERT trong lượt này. |
| **CafeBERT baseline** | **Screening một seed.** | Fine-tune checkpoint có sẵn bằng CE; từ probe 20 epochs đến baseline 30 epochs | Mạnh nhất trong các baseline đã đo ở 4/5 tác vụ test. Không phải project tự pretrain CafeBERT từ đầu. |
| **SEA-LION-ModernBERT-300M** | **Screening một seed.** | 5 tác vụ × 2 LR, 30 epochs, seed 42; first-token head | Thua Bami/Cafe cả 5 tác vụ val và test. Restaurant val **66,44** so Bami **74,74**, Cafe **77,72**. Nhánh đã đóng; chưa thử native mean pooling. |
| **mDeBERTa-v3-base** | **Screening một seed.** | 5 tác vụ × 1 LR 2e-5, 30 epochs, seed 42; native pooler/head | Chỉ Topic hơn Bami **+0,55 val**; bốn tác vụ khác thấp hơn. Thua Cafe cả 5 tác vụ val. Chưa test hoặc chạy thêm LR/seed. |

## 4. Hướng pretraining

| Chương trình | Mức bằng chứng | Cách làm | Kết quả ngắn gọn |
| --- | --- | --- | --- |
| **GoiBERT-B DAPT lịch sử** | **Hỗn hợp:** DAPT 1 seed; downstream VLSP 3 seed, các dataset khác 1 seed. | BamiBERT 103M + MLM 40.000 bước; corpus 1,67 GB gồm CC100 và văn bản benchmark tăng trọng số ×20 | Test lịch sử: Sentiment **82,74→80,48**; ViSFD macro F1 **75,48→74,83**; PhoATIS **56,29→54,68**. VLSP2018 Restaurant trung bình 3 seed **43,37→45,94 (+2,57)**, nhưng biến thiên lớn. Không thắng rộng. |
| **Đánh giá lại backbone DAPT cũ** | **Screening một seed.** | Fine-tune CE trên năm tác vụ hiện tại; so BamiBERT gốc | Val 1 seed, DAPT−gốc: Sentiment **−0,40**; Topic **+2,64**; ViSFD **−0,03**; Restaurant **−0,12**; Hotel **−0,02**. Đây không phải run DAPT mới. |
| **General distillation S/XS** | **Screening một seed** cho so sánh VSFC được nêu. | Teacher DAPT→student 26,5M/10,6M; PCA embedding, transformer khởi tạo ngẫu nhiên; khớp hidden states và attention trong 30.000 bước, rồi học task | VSFC test: student S từ task-distill-only **75,34→79,16** khi thêm gói DAPT+GD; XS **74,63→78,65**. Nén model hữu ích, chưa tách riêng tác dụng DAPT/GD và chưa thắng rộng. |
| **E6 — MLM trên student** | **Screening một seed.** | Warmup MLM head 500 bước + joint MLM 2.000 bước trên khoảng 33.000 văn bản train; rồi fine-tune | Val 1 seed: cả 5 tác vụ giảm; Restaurant **−1,09**, Topic **−0,82** so E1. Loss MLM giảm không tạo gain downstream. |
| **Pilot B — plain CPT** | **Hỗn hợp:** CPT 1 seed; fine-tune ABSA/ViSFD 2 seed, VSFC 1 seed. | CafeBERT + 300M token web tiếng Việt từ FineWeb2; embedding gốc | Không có gain sạch rộng. Trung bình mô tả năm metric **81,00→80,96**; không dùng trung bình này thay kết quả từng tác vụ. |
| **Pilot B — dấu/embedding** | **Hỗn hợp:** mỗi arm CPT 1 seed; fine-tune ABSA/ViSFD 2 seed, VSFC 1 seed. | Bốn biến thể cùng 300M token: dropout dấu; cộng embedding khung chữ/dấu; tăng LR nhánh cộng ×40; embedding factorized với residual rank 64 | Chưa có gain sạch ổn định. Bản factorized giúp mạnh khi bỏ dấu toàn bộ input: trung bình mô tả **59,1** so control dropout **43,8**; Hotel **48,9** so **19,5**. Đây là robustness trên nhiễu nhân tạo. |
| **Pilot B — kiểm social/teencode** | **Hỗn hợp:** ViHSD screening 1 seed; chuẩn hóa teencode chỉ diagnostic. | Fine-tune/eval ViHSD; chuẩn hóa teencode bằng từ điển 384 biến thể khi đánh giá | ViHSD sạch **70,9→71,1**, chưa có gain rõ. Chuẩn hóa: ViHSD **+0,5**; ViSpam **0**; ViSFD **+0,1**; Restaurant **−1,9**. Chưa train embedding teencode mở rộng; phép chuẩn hóa không phải cận trên của mọi phương pháp học. |
| **CafeBERT-DAPT social, D1** | **Screening một seed.** | 300M token, khoảng 1,80 lượt qua corpus ViSoBERT đã lọc; MLM rồi fine-tune Restaurant và task KD | MLM holdout **3,665→2,426**; teacher Restaurant val **77,72→77,58**; student KD **73,34**, dưới baseline **74,74**. Corpus social rộng, chưa sát review nhà hàng. Dừng sau pilot. |

**Hai lưu ý về lịch sử:** corpus DAPT/GD cũ có tiếp xúc validation text không nhãn và chưa được kiểm toán độc lập đầy đủ. VLSP2018 Restaurant là dataset khác UIT-ABSA Restaurant; metric lịch sử cũng khác. Không so trực tiếp các điểm tuyệt đối giữa hai bộ.

## 5. Bảng đối chiếu backbone hiện tại

Các model trong bảng đều được fine-tune riêng cho từng tác vụ. Đây không phải một model multi-task duy nhất. **Mức bằng chứng của cả hai bảng dưới: screening một seed (seed 42), kể cả bảng test.**

### Validation — seed 42, F1 thang 0–100

| Model | Sentiment | Topic | ViSFD | Restaurant | Hotel |
|---|---:|---:|---:|---:|---:|
| BamiBERT gốc, CE | 86,33 | 79,53 | 84,38 | 74,74 | 73,24 |
| BamiBERT-DAPT cũ, CE | 85,92 | 82,17 | 84,36 | 74,62 | 73,22 |
| CafeBERT, CE | 87,69 | 81,74 | 85,47 | 77,72 | 73,34 |
| SEA-LION-ModernBERT-300M, CE | 82,26 | 77,59 | 79,43 | 66,44 | 71,35 |
| mDeBERTa-v3-base, CE | 85,92 | 80,08 | 79,85 | 70,56 | 70,82 |

### Test — seed 42, chỉ các baseline có kết quả tương ứng

| Model | Sentiment | Topic | ViSFD | Restaurant | Hotel |
|---|---:|---:|---:|---:|---:|
| BamiBERT gốc | **83,03** | 78,18 | 84,50 | 74,15 | 73,94 |
| CafeBERT | 82,36 | **80,77** | **85,48** | **77,39** | **75,17** |
| SEA-LION-ModernBERT-300M | 77,90 | 77,29 | 79,42 | 65,06 | 71,21 |

BamiBERT được chọn trong 3 LR, SEA-LION trong 2 LR; CafeBERT và mDeBERTa dùng một LR trong các baseline này. Vì vậy đây là so sánh hệ thống thực dụng, chưa khớp hoàn toàn ngân sách tuning. Một seed không đủ để tuyên bố khác biệt có ý nghĩa thống kê. Không lấy validation thay test ở các model chưa có test.

## 6. Các phép chẩn đoán đã thực hiện

| Nhóm | Mức bằng chứng | Cách làm và kết luận |
| --- | --- | --- |
| **E0/E12 — Data và nhãn** | **Chỉ diagnostic.** | Kiểm loader, split, metric và tên polarity. Mapping ABSA đúng: 0 absent, 1 positive, 2 negative, 3 neutral. Một số tên nhãn cũ bị sai; F1 số không tự sai vì lỗi tên, nhưng diagnostic synthetic tạo bằng mapping sai đã bị loại khỏi diễn giải. |
| **D1/D2 — Layer probes và CKA** | **Chỉ diagnostic.** | Đo khả năng đọc nhãn và độ tương đồng biểu diễn trước–sau fine-tuning. Chưa xác định một layer hỏng chung; teacher thích nghi tốt hơn ở nhiều tác vụ. Không chứng minh catastrophic forgetting hay trần năng lực student. |
| **E12 — Surface/error review** | **Chỉ diagnostic.** | Phân tầng từ hiếm, noise và đọc 60 mẫu. Có lỗi gán đánh giá vào đúng đối tượng, đánh giá ngầm và nhãn mơ hồ. Mẫu nhỏ, có chọn lọc; không suy tỷ lệ lỗi nhãn toàn dataset. |
| **E13A — Binding diagnostic** | **Chỉ diagnostic.** | 30 cặp câu đã review; signed/convex attention không sửa các lỗi của baseline trong mẫu này. Diagnostic nhân tạo, không thay đánh giá tự nhiên. |
| **Reader/composition battery** | **Chỉ diagnostic.** | Thử reader, fork, competitive slots, patching, shadow reader và oracle. Có cải thiện trên câu ghép nhưng chưa chuyển rõ sang validation tự nhiên; oracle cần đối chứng thiên lệch do chọn đáp án. |
| **Debias prior nhãn** | **Chỉ diagnostic.** | Điều chỉnh theo prior đồng xuất hiện nhãn, kể cả oracle biết các aspect còn lại. Chưa có cải thiện hữu ích ổn định; điều chỉnh mạnh làm giảm F1. |
| **Scope-difference readout** | **Chỉ diagnostic.** | Đọc cảm xúc bằng chênh logits khi bỏ phần câu liên quan. Chưa chứng minh lợi ích hơn đọc phần nguồn riêng trong diagnostic; không mở training nhánh mới. |

**Phân biệt tên:** D1/D2 trong nhóm chẩn đoán layer khác với lượt CafeBERT-DAPT mang mã D1 ở chuỗi KD.

## 7. Thí nghiệm lịch sử và phụ trợ ngoài bộ năm tác vụ chính

| Chương trình | Mức bằng chứng | Cách làm và kết quả |
| --- | --- | --- |
| **Architecture scaling trên LUMI** | **Screening một seed** cho các kết quả được nêu. | Trước pipeline GD hiện tại, thử student 8 tầng × 512 và 10 tầng × 384, scratch hoặc task distillation. Nhỏ/nhanh hơn nhưng chưa thắng BamiBERT trên ba dataset được đo. Protocol và metric khác chương trình hiện tại. |
| **NC robustness của model family** | **Screening một seed** cho các kết quả được nêu. | Thêm consistency cho BamiBERT, GoiBERT-B, S và XS. Cải thiện mạnh trên văn bản bị bỏ dấu nhân tạo, thường hàng chục điểm; chất lượng sạch không tăng nhất quán. |
| **GoiTurn v0/v0.5/v1** | **Screening một seed** cho các kết quả được nêu. | Turn-completion trên dữ liệu synthetic: scratch **93,31→95,77** khi khởi tạo GD, thêm NC đạt **95,82** macro F1; BamiBERT **96,58**. Kết quả nén/ứng dụng, chưa phải benchmark production. |
| **Intent hội thoại** | **Screening một seed** cho các kết quả được nêu. | Fine-tune BamiBERT/Goi DAPT/student 26M, hai LR, một seed. Test 77 mẫu: macro F1 **47,45 / 83,30 / 82,09**; số đúng **61 / 69 / 68**. Chênh lệch chịu ảnh hưởng lớn của lớp hiếm và độ nhạy LR; chưa suy thắng tổng quát. |

## 8. Các hướng mới chỉ được đề xuất, chưa chạy

- CafeBERT + composite; KD trên composite hoặc nguồn văn bản mới.
- Phân loại bằng cặp review và mô tả aspect đưa trực tiếp vào encoder.
- SEA-LION với native mean pooling.
- Pilot LFM2.5-Encoder, mGTE hoặc EuroBERT.
- Chương trình pretraining mới có supervision sentiment theo hướng SKEP/SCAPT.
- Embedding teencode mở rộng hoặc tăng các pilot CPT hiện tại lên 1B/5B token.

## 9. Giới hạn và kết luận cuối

Phần lớn phép thử kiến trúc là screening một seed; “không đạt” là quyết định ngừng đầu tư trong cấu hình đó. Một số gain đã lặp lại qua nhiều seed, nhưng không tự tương đương kiểm định significance. Bộ test hiện tại đã được quan sát qua nhiều vòng nghiên cứu; không còn là tập hoàn toàn chưa tiếp xúc. Nguồn mirror UIT-ABSA và sự tương đương metric với các paper chưa được xác minh hoàn toàn, nên các so sánh local không tự tạo claim SOTA công khai.

**Tổng kết:** project đã đạt lợi ích về nén model, robustness trước mất dấu, recipe fine-tuning và tăng cường dữ liệu theo tác vụ. Tuy nhiên, chưa có phương pháp mới thắng rộng BamiBERT/CafeBERT. Bằng chứng không hỗ trợ cách nói “mọi thí nghiệm đều thất bại”, cũng chưa chứng minh đã chạm trần post-training, pretraining hoặc tokenizer.
