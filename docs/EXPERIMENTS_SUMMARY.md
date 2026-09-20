# GoiBERT — tổng kết các thí nghiệm đã chạy

**Cập nhật: 18/09/2026.** Mục tiêu hiện tại: vượt BamiBERT/CafeBERT, model khoảng100–600M tham số. Bản này tổng hợp theo **chương trình thí nghiệm**, gộp các lượt smoke, chạy lại do lỗi và xác nhận seed; không coi E1–E22 là22 lần thử độc lập đều thất bại.

**Cách đọc:** `val` = validation; `test` = test; `Δ` = điểm F1 tuyệt đối so với đối chứng được nêu. Đa số screening chỉ1seed. “Không đạt” nghĩa không đạt mục tiêu trong cấu hình đã thử, không chứng minh mọi biến thể đều vô ích. VSFC dùng macroF1; ABSA hiện tại dùng end-to-end pair microF1. Kết quả lịch sử khác dataset/metric được ghi riêng.

## 1. Kết luận nhanh

- **Có cải thiện thật:** E8 NC+LLRD trên student26M; ghép câu composite trên Restaurant ở backbone103M.
- **Chưa có cơ chế kiến trúc mới thắng rộng:** các nhánh attention, FFN, reader/routing và invariance đã thử không đạt mục tiêu chung.
- **DAPT không thắng rộng:** BamiBERT-DAPT cũ có lợi thế Topic; CafeBERT-DAPT300M social không giúp Restaurant trong pilot.
- **Task-logit KD không giúp Restaurant:** đã giảm ở E2, CafeBERT→BamiBERT M1 và CafeBERT-DAPT→BamiBERT D1. Không đồng nhất kết quả này với general distillation.
- **Backbone mới chưa thắng:** SEA-LION300M thua cả5task; mDeBERTa chỉ hơn Bami ở Topic val, thua Cafe cả5task val.
- **CafeBERT là reference mạnh nhất hiện có trên4/5task test**; BamiBERT vẫn cao hơn ở Sentiment test. Chưa có hệ thống mới vượt cả hai trên toàn bộ5task.

## 2. Hướng post-training — fine-tuning, kiến trúc, dữ liệu và task KD

Các E1–E13/E17–E18 chủ yếu dùng student26M; E14–E16 dùng BamiBERT gốc~103M; E19–E22 dùng BamiBERT-DAPT~103M. E6 là MLM continuation, nằm ở mục3.

| Thí nghiệm | Cách làm ngắn gọn | Kết quả chính |
|---|---|---|
| [E1 — baseline student](E1_TEST_BENCHMARK.md) | Fine-tune toàn bộ student GD26M bằng CE | Test5seed: Sentiment80,60; Topic78,62; ViSFD79,83; Restaurant69,08; Hotel71,78. Là mốc so sánh, chưa thắng rộng. |
| [E2 — KD / NC](E2_VALIDATION_READOUT.md) | Hai nhánh: học logits teacher DAPT hoặc giữ dự đoán nhất quán khi thêm nhiễu | Val3seed: KD giúp Topic+0,96; giảm Restaurant−0,59 và Hotel−0,43 ở cả3seed. NC giúp Sentiment+0,44, ViSFD+0,99; lợi ích Restaurant/Hotel ít ổn định hơn. |
| [E3 — loss polarity](E3_REVIEW_FINAL.md) + [Hotel](E3_HOTEL_RESULT.md) | CE chung + CE polarity trên aspect có mặt | Val1seed so E1: Restaurant+1,11, ViSFD+0,15, Hotel−1,72. Polarity tốt hơn nhưng bỏ sót aspect nhiều hơn; không thắng rộng. |
| [E4 — tăng FFN](E4_RESULT.md) | Mở FFN1536→3072, student26,5M→38,4M | Val1seed: không task nào tăng≥0,3; Restaurant−1,17. Không đạt. |
| [E5 — mean pooling](E5_RESULT.md) | Thay CLS bằng trung bình token content | Val1seed: Sentiment+0,32, Restaurant−1,53; không cải thiện chung. Chỉ kiểm trên student cũ. |
| [E7 — LLRD](E7_RESULT.md) | Tầng dưới dùng LR nhỏ hơn, gamma0,8 | Val1seed so E1: ViSFD+1,42; Restaurant+2,50; Hotel+1,25; Sentiment−0,44. Có tín hiệu ABSA. |
| [E8 — NC + LLRD](E8_CONFIRMATION_RESULT.md) | Kết hợp noise consistency và LR theo tầng | **Test3seed so E2 NC:** ViSFD80,62→81,81 (**+1,19**); Restaurant69,86→71,88 (**+2,02**), tăng3/3seed. Lợi ích đã lặp lại; chưa thắng Bami rộng. |
| [E9 — label-aware attention](E9_RESULT.md) | Mỗi nhãn/aspect có query đọc token states | Val1seed so E8: Restaurant+1,03 nhưng Hotel−0,62, ViSFD−0,34. Không đạt gate thắng rộng. |
| [E10 — span / convolution](E10_RESULT.md) | Nhánh residual gom span theo nội dung, so control convolution | Val1seed: span gần như không giúp; không hơn control. Không đạt. |
| [E11 — signed attention](E11_RESULT.md) | Cho attention kết hợp trọng số âm, so convex control | Val1seed: Restaurant+1,17, Hotel+0,43 so E8 nhưng Sentiment−1,02; control convex gần tương đương. Chưa thấy lợi ích riêng của signed mechanism. |
| [E13 — pair-value residual](E13_RESULT.md) | Tương tác phi tuyến theo cặp token, centered vs uncentered | Không thắng rộng. Restaurant centered tăng+0,28val/+0,50test so E8 ở1seed; các task khác gần ngang hoặc giảm. |
| [E14 — BamiBERT gốc](E14_BAMIBERT_BASELINE_RESULT.md) | CE,3LR,30epochs,seed42 trên5task | Có baseline local val/test chính thức; tốt hơn student E8 ở4/5task test, Topic là ngoại lệ. Xem bảng mục4. |
| [E15 — pair-value trên Bami](E15_RESULT.md) | Chuyển E13 sang BamiBERT gốc103M | Val1seed: chỉ Topic+0,55; bốn task còn lại giảm0,50–1,51. Không đạt. |
| [E16 — attention/FFN gates](../artifacts/e16/E16_RESULT.md) | Gate identity-preserving ở attention update hoặc input FFN | Attention gate không có gain rõ. FFN gate có tín hiệu val nhưng [test1seed](E16_FFN_TEST_RESULT.md) chỉ Topic+0,46; Sentiment−0,77, ViSFD−0,38, Restaurant−0,03, Hotel−0,40 so baseline cùngrun. |
| [E17 — cavity keys](E17_RECOVERY.md) | Loại đóng góp của token nhận khỏi contextual key; baseline/full-context/cavity |15trials hoàn tất. Cavity−baseline val1seed: Sentiment0; Topic0; ViSFD−0,01; Restaurant−0,14; Hotel−0,19. Không có gain. |
| [E18 — fork consistency](E18_RESULT.md) | Huấn luyện đọc theo đoạn, so random-mask control | Val1seed: Restaurant−0,89, Hotel+0,50 so baseline; random control ngang hoặc hơn. Thay đổi được diagnostic cơ chế nhưng không tạo lợi ích chất lượng chung. |
| [E19 — E8 trên DAPT103M](E19_RESULT.md) | NC+LLRD trên teacher BamiBERT-DAPT | So cùng encoder CE: ViSFD+0,72, Hotel+0,41; Sentiment−0,21, Topic−0,40, Restaurant−0,11. Không chuyển gain student26M thành gain rộng ở103M. |
| [E20 — CTI + composite](E20_RESULT.md) | Ghép câu có nhãn hợp; thêm loss bất biến token khi ghép | CTI giảm diagnostic nhiễm token Hotel16,5→2,0%, nhưng không sửa lỗi/F1 như kỳ vọng và không thắng composite control. Composite riêng Restaurant+0,95val ởseed42. |
| [E21 — xác nhận composite](E21_MULTISEED_D1.md) | Thêm seed43/44 cho E19 vs composite Restaurant | **Val3seed:**74,10→75,26; **+1,16±0,18 paired SD**, tăng3/3seed. |
| [E22 — mở rộng/ablation composite](E22_RESULT.md) | Thử Hotel/ViSFD; Restaurant chọn cặp đều vs ưu tiên cặp hiếm | Val3seed: Hotel+0,40±0,72; ViSFD−0,13. Restaurant uniform75,74 vs rare-priority75,26: ưu tiên cặp hiếm không chứng minh lợi ích thêm. |
| [CDBK M1 — Cafe→Bami KD](CDBK_PILOT_RESULT.md) | Teacher Cafe CE; student Bami học0,5CE+0,5KD,T=2 | Restaurant val1seed: T77,72; B74,74; K73,94 (**−0,80 so B**). Correctness pass; đóng cấu hình. |
| [CDBK D1 — teacher đã DAPT→KD](CDBK_DAPT_PILOT_RESULT.md) | Giữ recipe K, đổi sang teacher CafeBERT-DAPT social | Restaurant student73,34: **−1,41 so B**, −0,61 so K. Không giúp; cùng chuỗi DAPT ở mục3, không phải hai chương trình độc lập. |
| [mmBERT-base probe](NEW_DIRECTIONS_VI_ENCODER.md) | Fine-tune checkpoint đa ngữ hiện đại, CE,20epochs,1LR | Val1seed: Restaurant74,38; Hotel73,66. Ngang cỡ Bami, chưa vượt Cafe Restaurant. Không CPT mmBERT trong lượt này. |
| [CafeBERT probe/baseline](CAFEBERT_PROBE.md) | Fine-tune CafeBERT pretrained sẵn; sau đó hoàn thiện baseline30epochs | Probe20epoch cho tín hiệu mạnh; kết quả30epoch/test mới nhất ở mục4. Đây là fine-tuning model public, không phải pretraining CafeBERT từ đầu trong project. |
| [SEA-LION-ModernBERT300M](SEALION_300M_BENCHMARK_RESULT.md) |5task×2LR,30epochs,seed42; first-token head; thêm4Cafe baselines | **Thua Bami/Cafe cả5task val và test.** Restaurantval66,44 vs74,74/77,72. Đã đóng nhánh; native mean pooling chưa thử. |
| [mDeBERTa-v3-base](MDEBERTA_PILOT_RESULT.md) |5task×1LR2e-5,30epochs,seed42; native head/pooler | Val: chỉ Topic+0,55 so Bami; bốn task khác thua. Thua Cafe cả5task; chưa test, không mở thêm lượt. |

## 3. Hướng pretraining — DAPT/CPT, general distillation và embedding

**DAPT/CPT** là học tiếp checkpoint; **general distillation** học biểu diễn từ teacher trên corpus. Các checkpoint public dùng trong mục2 không được tính là chúng ta đã pretrain chúng.

| Chương trình | Cách làm | Kết quả chính |
|---|---|---|
| [GoiBERT-B DAPT lịch sử](../tmp/GOI_MODELS_SUMMARY_REPORT_EN.md) | BamiBERT103M + MLM40Ksteps; corpus1,67GB gồm CC100 và text benchmark tăng trọng số×20 | Test lịch sử: Sentiment82,74→80,48; ViSFD macro75,48→74,83; PhoATIS56,29→54,68. VLSP2018 Restaurant mean3seed43,37→45,94 (+2,57), biến thiên lớn. **Không thắng rộng.** |
| [DAPT backbone cũ trên5task hiện tại](E19_RESULT.md) | Fine-tune CE teacher DAPT ở E2, đối chiếu Bami gốc E14 | Val1seed DAPT−gốc: Sentiment−0,40; **Topic+2,64**; ViSFD−0,03; Restaurant−0,12; Hotel−0,02. Đây là đánh giá lại backbone cũ, không DAPT run mới. |
| [General distillation S/XS](../tmp/GOI_MODELS_SUMMARY_REPORT_EN.md) | Teacher DAPT→student26,5M/10,6M; PCA embedding, transformer random-init, khớp hidden+attention30Ksteps; rồi task stage | So StageA task-distill-only→StageP DAPT+GD: VSFC student S75,34→79,16; XS74,63→78,65 test. **Nén model có ích**, chưa chứng minh student thắng rộng hay tách riêng tác dụng DAPT. |
| [E6 — MLM tiếp trên student](E6_RESULT.md) | Warmup MLM head500steps + jointMLM2000steps trên~33Ktraintexts, rồi fine-tune | Val1seed: cả5task giảm; Restaurant−1,09, Topic−0,82. MLM loss giảm không chuyển thành gain downstream. |
| [Pilot B — a: plain CPT](PILOT_B_RESULT.md) | CafeBERT +300M token FineWeb2 tiếng Việt, embedding gốc | Không có gain sạch rộng; mean5task81,00→80,96 (mean chỉ mô tả, trộn metric). |
| [Pilot B — c/b2/b2_hlr/b](PILOT_B_RESULT.md) | Cùng300M tokens: dropout dấu; additive embedding dấu; additive LR×40; factorized embedding có residual rank64 | Các arm chưa cho gain sạch ổn định. **Arm b** thắng control dropout rõ trên input bị bỏ dấu hoàn toàn: mean stripped59,1 vs43,8; Hotel48,9 vs19,5. Đây là robustness synthetic, chưa phải clean SOTA. |
| [Pilot B — social/teencode follow-up](PILOT_B_RESULT.md) | Fine-tune/eval ViHSD; chuẩn hóa teencode bằng từ điển384biến thể khi đánh giá | ViHSD clean70,9→71,1: chưa có gain rõ. Normalize: ViHSD+0,5; ViSpam0; ViSFD+0,1; Restaurant−1,9. **Chưa train embedding teencode mở rộng**; phép normalize không phải cận trên toán học cho mọi phương pháp. |
| [CDBK D1 — CafeBERT-DAPT social](CDBK_DAPT_PILOT_RESULT.md) |300M tokens (~1,80passes) trên corpus ViSoBERT đã lọc; MLM, rồi Restaurant CE và taskKD | MLM holdout3,665→2,426; teacher Restaurantval77,72→77,58; studentKD73,34<baseline74,74. Corpus social rộng, không sát review nhà hàng. Không tăng token/E8/seed sau fail. |

**Giới hạn lịch sử:** corpus DAPT/GD cũ có tiếp xúc validation text không nhãn; chưa audit đầy đủ raw corpus gốc. VLSP2018 Restaurant, PhoATIS và macroF1 lịch sử không được trộn với UIT-ABSA/pairmicroF1 hiện tại. Không lấy “student gần teacher” làm bằng chứng DAPT teacher hoặc taskKD tự nó hiệu quả.

## 4. Bảng backbone hiện tại — mốc để tránh nhớ nhầm

### Validation, seed42 — thang F1×100

| Model | Sentiment | Topic | ViSFD | Restaurant | Hotel |
|---|---:|---:|---:|---:|---:|
| BamiBERT gốc, CE |86,33|79,53|84,38|74,74|73,24|
| BamiBERT-DAPT cũ, CE |85,92|82,17|84,36|74,62|73,22|
| CafeBERT, CE30epochs |87,69|81,74|85,47|77,72|73,34|
| SEA-LION300M, CE |82,26|77,59|79,43|66,44|71,35|
| mDeBERTa, CE |85,92|80,08|79,85|70,56|70,82|

### Test, seed42 — chỉ các model có kết quả tương ứng

| Model | Sentiment | Topic | ViSFD | Restaurant | Hotel |
|---|---:|---:|---:|---:|---:|
| BamiBERT gốc E14 |**83,03**|78,18|84,50|74,15|73,94|
| CafeBERT |82,36|**80,77**|**85,48**|**77,39**|**75,17**|
| SEA-LION300M |77,90|77,29|79,42|65,06|71,21|

Nguồn: [E14](E14_BAMIBERT_BASELINE_RESULT.md), [E19/E2 teacher](E19_RESULT.md), [SEA/Cafe benchmark](SEALION_300M_BENCHMARK_RESULT.md), [mDeBERTa](MDEBERTA_PILOT_RESULT.md). Bami chọn3LR, SEA2LR, Cafe/mDeBERTa1LR; không phải so sánh cùng ngân sách tuning hoàn toàn. Không điền test bằng validation. Cafe Hotel75,17 cũng không ủng hộ khẳng định “trần nhãn74”.

## 5. Các phép chẩn đoán đã chạy — không phải model mới

| Nhóm | Cách làm và kết luận ngắn |
|---|---|
| E0 / E12 mapping audit | Kiểm data/split/evaluator; sửa diễn giải ABSA đúng là0absent/1positive/2negative/3neutral. Tên nhãn sai trong tài liệu cũ không tự làm sai F1 số, nhưng diagnostic synthetic dùng sai mapping phải loại. |
| [D1/D2 layer probes](D2_READOUT.md) | Linear probes/CKA trước–sau fine-tune teacher/student: chưa tìm được một layer hỏng chung; teacher thích nghi tốt hơn ở nhiều task. Không chứng minh forgetting hay trần student. |
| [E12 surface/error review](E12_SEMANTIC_REVIEW.md) | Phân tầng rarity/noise và đọc60mẫu: có target-binding, đánh giá ngầm và nhãn mơ hồ; không chứng minh tokenizer là nút thắt chính hoặc Hotel có trần nhãn. |
| [E13A binding diagnostic](E13A_RESULT.md) |30cặp đã review; E11 signed/convex không sửa các lỗi E8 trong mẫu này. Synthetic, không thay benchmark tự nhiên. |
| [Reader/composition battery](RESEARCH_LOOP_STATUS.md) | Reader, fork, competitive slots, oracle/patching/shadow-reader: có thể cải thiện trên câu ghép nhưng không chuyển rõ sang validation tự nhiên; oracle cần đối chứng selection bias. E18/E20 là các can thiệp train nối tiếp. |
| [Debias prior nhãn](DECISION_LEVEL_DEBIAS_CHECK.md) | Trừ prior đồng xuất hiện, kể cả oracle: không tạo cải thiện hữu ích ổn định; chỉnh mạnh làm giảm F1. Không loại mọi phương pháp debias. |
| [Scope-difference readout](ORACLE_SCOPE_DIFFERENCE_READOUT.md) | Đọc bằng chênh logits khi bỏ scope: chưa chứng minh hơn source-only trong diagnostic; không mở training nhánh mới. |

## 6. Thí nghiệm lịch sử/phụ trợ ngoài bộ5task chính

| Chương trình | Cách làm + kết quả |
|---|---|
| [Architecture scaling LUMI](../reference/ai-foundary/experiments/goiturn-arch-scaling/results-lumi/summary.md) | Student8×512 và10×384, scratch/task-distill, trước pipelineGD hiện tại; nhanh/nhỏ hơn nhưng không thắng Bami trên3dataset đã đo. Protocol/metric khác, không nhập chung leaderboard mới. |
| [NC robustness family](../tmp/GOI_MODELS_SUMMARY_REPORT_EN.md) | Thêm noise consistency cho Bami/Goi/S/XS; tăng mạnh trên input bỏ dấu nhân tạo (thường hàng chục điểm), clean không tăng nhất quán. Đây là robustness, không thắng clean rộng. |
| [GoiTurn v0/v0.5/v1](../tmp/GOI_MODELS_SUMMARY_REPORT_EN.md) | Turn-completion synthetic: scratch93,31→GD-init95,77→NC95,82 macroF1; Bami96,58. Mục tiêu nén/ứng dụng, khác5task encoder; chưa phải kết quả production. |
| [Intent pilot](../intent_benchmark/results/BAO_CAO_NGAN.md) | Fine-tune Bami/GoiDAPT/student26M,2LR,seed42. Test77mẫu: macroF1 Bami47,45/DAPT83,30/student82,09; đúng61/69/68. Lớp hiếm và baseline nhạyLR, chưa suy thắng tổng quát. |

## 7. Chưa chạy — không tính thành kết quả

- CafeBERT + composite; KD trên composite/nguồn text mới.
- CafeBERT/aspect-conditioned sentence-pair classifier.
- SEA-LION native mean-pooling follow-up.
- LFM2.5-Encoder, mGTE, EuroBERT pilots; pretraining sentiment-aware SKEP/SCAPT mới.
- Embedding teencode mở rộng; pilot1B/5B từ các kế hoạch chưa thực thi.

**Chốt:** đã có lợi ích về nén model, robustness, recipe fine-tuning và composite theo task. Chưa có evidence cho một phương pháp mới thắng rộng BamiBERT/CafeBERT. Không tổng kết thành “tất cả thất bại”, cũng không kết luận đã chạm trần post-training/pretraining/tokenizer.
