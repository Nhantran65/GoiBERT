# BENCH8 — bổ sung ViSpamReviews, ViNLI, PhoNER_COVID19

## Phạm vi và protocol

Kết quả dưới đây chỉ dùng tập **validation**. Không nạp, đếm dòng hay đánh giá trên file test. Dữ liệu train/dev được kiểm SHA256 trước khi dùng; manifest ở `artifacts/bench8/data_manifest.json`. Hai checksum test ViNLI/PhoNER trong manifest là giá trị do người dùng cung cấp, **chưa được xác minh**, vì ràng buộc không mở test. Checksum test ViSpam không thể tính nếu vẫn giữ ràng buộc này.

BamiBERT dùng encoder gốc và head E14, AdamW E14. Student S dùng encoder `goiturn-s-gd`, head E1 và LLRD γ = 0,8. Cả hai dùng CE thuần, batch hiệu dụng 32, tối đa 30 epoch, warmup tuyến tính 10%, clip 1,0, bf16 và SDPA. LR quét seed 42 riêng theo model × task, chọn theo validation F1, hòa chọn LR thấp hơn; seed 43/44 chạy ở LR đã chọn. `vispam_binary` dùng LR của `vispam` cùng model. Không dùng KD cho ba dataset mới.

ViNLI được mã hóa bằng pair encoding `(premise, hypothesis)`, tối đa 256 token; smoke test in ba ví dụ cho mỗi tokenizer, với hai separator liền nhau giữa hai câu. PhoNER dùng nhãn BIO trên âm tiết, chỉ gán nhãn cho subword đầu, các subword còn lại là `-100`, tối đa 512 token. Dự đoán cho âm tiết bị cắt sẽ là `O`. Bộ chấm thực thể dùng micro precision/recall/F1, đúng khi loại và cả hai biên đều đúng; `I-` mở đầu được hiểu là bắt đầu một thực thể. Bốn unit test bao gồm `I-` mở đầu, thực thể liền nhau, sai biên/loại và chuyển loại từ `I-` đều đạt.

## Kiểm tái tạo bắt buộc

`vsfc_topic` seed 42, LR `1e-5`: Bami và Student đều khớp F1 ở từng epoch 1–5 với run cũ, sai lệch mỗi epoch **0,0 điểm**. Best F1 Bami cũ/mới đều `0.7953280821882547`; Student cũ/mới đều `0.7995390397547542`, sai lệch **0,0 điểm**. Hai run đủ 30 epoch và bốn postrun check đều PASS. `artifacts/bench8/reproduction_check.json` ghi các phép so theo ngưỡng plan (≤ 0,05 điểm mỗi epoch 1–5 và ≤ 1,0 điểm best).

## Diễn giải 15 chỉ số

Plan ấn định tổng 15 chỉ số nhưng không liệt kê chi tiết từng cột của năm dataset cũ. Để bám evaluator E14/E7 hiện có, bảng dùng accuracy và macro F1 cho hai task VSFC và hai task phân loại câu mới, detection micro F1 và pair micro F1 cho ViSFD/ABSA Restaurant/ABSA Hotel, và entity micro F1 cho PhoNER. Tổng: 4 + 6 + 4 + 1 = 15. `vispam_binary` là phân tích phụ, không nằm trong 15 cột chính.


## Kết quả và kiểm toán

Toàn bộ 36 run mới và 2 run tái tạo hoàn tất. `summarize.py` đã tính lại tất cả 36 run mới, kể cả các ô LR seed 42 không được chọn, từ `validation_predictions.json`; 30 run của năm dataset cũ cũng được tính lại. Mọi chỉ số khớp file metrics đã lưu; 36/36 run mới có bốn postrun checks PASS. Không có run lỗi, check FAIL hay thư mục aborted.

### LR được chọn

| Dataset | Bami | Student S |
|---|---:|---:|
| vispam | `2e-05` | `5e-05` |
| vinli | `1e-05` | `0.0001` |
| phoner | `5e-05` | `0.0001` |

`vispam_binary` dùng cùng LR với `vispam` theo từng model; không quét LR riêng.

### Ba dataset mới (validation; phần trăm, trung bình và range của seed 42/43/44)

| Dataset | Chỉ số | Bami TB (range) | Student S TB (range) | Student − Bami |
|---|---|---:|---:|---:|
| vispam | accuracy | 90.104821802935 (90.0–90.18867924528303) | 89.70649895178198 (89.43396226415095–89.87421383647799) | -0.3983228511530257 |
| vispam | macro_f1 | 77.37953460125227 (76.12649960236213–78.05437251819183) | 75.07995170121185 (74.20763760049473–75.66217280692965) | -2.29958290004042 |
| vinli | accuracy | 80.94605073667886 (80.259222333001–81.48886673313393) | 69.19242273180458 (68.6939182452642–69.69092721834497) | -11.753628004874273 |
| vinli | macro_f1 | 80.93517219321599 (80.23325351443317–81.48300667157972) | 69.32069117761473 (68.87172178421949–69.8045234899289) | -11.614481015601257 |
| phoner | f1 | 95.00436339092542 (94.94101179764047–95.12015529821272) | 92.73755005563906 (92.58349379191289–92.85998672859986) | -2.2668133352863578 |

### ViSpam nhị phân (phân tích phụ)

| Chỉ số | Bami TB (range) | Student S TB (range) |
|---|---:|---:|
| accuracy | 90.62893081761005 (90.50314465408805–90.88050314465409) | 90.58700209643607 (90.25157232704403–90.88050314465409) |
| macro_f1 | 87.90710178535731 (87.83168404647567–88.0579372631206) | 87.81538542243678 (87.37461059549113–88.13356494885157) |

### Bảng 8 dataset / 15 chỉ số

Đơn vị chỉ số là phần trăm; chênh lệch là Student trừ Bami theo điểm phần trăm. Tham số là số tham số thực của model và head theo từng dataset.

| Dataset | Chỉ số | Bami TB (range) | Student S TB (range) | Student − Bami | Tham số Bami / Student |
|---|---|---:|---:|---:|---:|
| vsfc_sentiment | accuracy | 94.69361970941252 (94.63044851547694–94.75679090334808) | 94.08296483470204 (93.8092229943146–94.31459254579912) | -0.6106548747104767 | 102953475 / 26546691 |
| vsfc_sentiment | macro_f1 | 86.82650498038737 (86.11572017056693–88.03856639708009) | 84.50367522501115 (84.02032535756608–85.426911313038) | -2.322829755376219 | 102953475 / 26546691 |
| vsfc_topic | accuracy | 88.81869867340492 (88.6923562855338–88.88186986734048) | 89.49252474205096 (89.38723941882502–89.70309538850285) | 0.6738260686460364 | 102954244 / 26547076 |
| vsfc_topic | macro_f1 | 79.53172479547898 (79.44195162727668–79.62041454033478) | 80.24676945801555 (79.95390397547541–80.64284587060044) | 0.7150446625365703 | 102954244 / 26547076 |
| visfd | detection_micro_f1 | 89.1542915558615 (88.94736842105263–89.33981931897151) | 89.17912149675311 (88.86716021790753–89.3909626719057) | 0.02482994089160684 | 102983466 / 26561706 |
| visfd | pair_micro_f1 | 83.98660341780912 (83.69581520923954–84.38339802926247) | 83.09514623067699 (82.68625393494229–83.32088995072421) | -0.8914571871321328 | 102983466 / 26561706 |
| absa_restaurant | detection_micro_f1 | 85.56649684716096 (84.60093896713616–86.47887323943661) | 85.7555338317699 (85.70066730219257–85.82412301777993) | 0.1890369846089328 | 102988080 / 26564016 |
| absa_restaurant | pair_micro_f1 | 73.3609012426287 (71.83098591549296–74.74178403755869) | 73.44540743530266 (73.40324118207818–73.50674373795762) | 0.0845061926739561 | 102988080 / 26564016 |
| absa_hotel | detection_micro_f1 | 79.46142711085193 (79.14110429447852–79.64737447297814) | 77.87531243287529 (77.54937525191455–78.1766841468334) | -1.5861146779766386 | 103055752 / 26597896 |
| absa_hotel | pair_micro_f1 | 73.68595969124232 (73.2361963190184–74.07695297318305) | 73.04859383161946 (72.8738411930673–73.17466720451795) | -0.6373658596228609 | 103055752 / 26597896 |
| vispam | accuracy | 90.104821802935 (90.0–90.18867924528303) | 89.70649895178198 (89.43396226415095–89.87421383647799) | -0.3983228511530257 | 102954244 / 26547076 |
| vispam | macro_f1 | 77.37953460125227 (76.12649960236213–78.05437251819183) | 75.07995170121185 (74.20763760049473–75.66217280692965) | -2.29958290004042 | 102954244 / 26547076 |
| vinli | accuracy | 80.94605073667886 (80.259222333001–81.48886673313393) | 69.19242273180458 (68.6939182452642–69.69092721834497) | -11.753628004874273 | 102954244 / 26547076 |
| vinli | macro_f1 | 80.93517219321599 (80.23325351443317–81.48300667157972) | 69.32069117761473 (68.87172178421949–69.8045234899289) | -11.614481015601257 | 102954244 / 26547076 |
| phoner | f1 | 95.00436339092542 (94.94101179764047–95.12015529821272) | 92.73755005563906 (92.58349379191289–92.85998672859986) | -2.2668133352863578 | 102376725 / 26405781 |

### Cắt chuỗi và ngân sách

Hai tokenizer cho cùng thống kê cắt ở các task này. Tỷ lệ là số ví dụ dài hơn max length chia cho tổng số ví dụ của split; các con số không làm tròn.

| Dataset | Max length | Train bị cắt | Dev bị cắt | Âm tiết bị cắt train; dev |
|---|---:|---:|---:|---:|
| vispam | 256 | 42/14306 = 0.0029358311198098698 | 8/1590 = 0.005031446540880503 | — |
| vinli | 256 | 0/24376 = 0.0 | 0/3009 = 0.0 | — |
| phoner | 512 | 0/5027 = 0.0 | 0/2000 = 0.0 | 0/167541; 0/71325 |

Tổng thời gian GPU của 36 run mới, 2 run tái tạo và 6 smoke test: **46398.076812028885 giây = 12.888354670008024 GPU-giờ**, dưới trần 16 GPU-giờ. Thời gian được cộng từ `elapsed_seconds` từng run; không làm tròn.

### Kiểm định, sự cố và giới hạn

- Check FAIL: **0**. Run failed: **0**. Run aborted: **0**. Cả 36 run mới có checkpoint reload khớp best, file dự đoán khớp best, dự đoán sau reload giống hệt và ID validation ổn định, đủ.
- Smoke: 6/6 run một epoch với 64 mẫu train hoàn tất, toàn bộ postrun checks PASS; unit test evaluator NER 4/4 PASS. Hai log ViNLI in đúng ba cặp đã mã hóa cho mỗi tokenizer.
- Launcher được tạm dừng và khởi động lại bằng process do lượt này tạo để dành GPU 3 cho smoke ViNLI/PhoNER; bốn worker đang chạy tiếp tục và được launcher nhận lại. Một lỗi trong code launcher mới về giải phóng slot sau khi worker hoàn tất đã được sửa trước khi gây chồng job; không có worker phải chạy lại. Lỗi cú pháp trong guard SHA256 mới được bắt bằng `py_compile` và sửa trước khi worker dùng phiên bản đó.
- ViNLI là mirror chưa xác minh nguồn gốc; kết quả này chỉ áp dụng cho đúng dữ liệu mirror đã ghim. Bộ dữ liệu chỉ dùng cho nghiên cứu. Test chưa mở, nên không có kết quả test hoặc xác minh SHA256 test. Hai hash test ViNLI/PhoNER trong manifest là giá trị được cung cấp; hash test ViSpam để trống nhằm tuân thủ lệnh không mở test.
- Plan không quy định chi tiết cách xếp 15 cột của năm dataset cũ; bảng dùng đúng metric E14/E7 hiện có như đã nêu ở mục “Diễn giải 15 chỉ số”.

### File tạo

- Code: `scripts/bench8/data.py`, `ner.py`, `test_ner.py`, `train.py`, `launcher.py`, `summarize.py`, `check_reproduction.py`.
- Tổng hợp và manifest: `artifacts/bench8/data_manifest.json`, `smoke_manifest.json`, `reproduction_check.json`, `selected_lrs.json`, `summary.json`, `table_8datasets.md`, `launcher_state.json`, `launcher.log`, `summarize.log`.
- Run artifacts: `artifacts/bench8/reproduction/{bami,student}/` và `artifacts/bench8/runs/<task>/<model>/lr_<LR>/seed_<seed>/`, mỗi run có `resolved_config.json`, `epoch_curves.json`, `best_validation_metrics.json`, `validation_predictions.json`, `best_checkpoint.pt`, `metadata.json`, `status.json`. Log worker nằm trong `artifacts/bench8/logs/`.
- Smoke artifacts và log: `/tmp/bench8-smoke-qWnXO6/`; báo cáo: `docs/BENCH8_NEW_DATASETS_RESULT.md`. Danh sách đầy đủ 369 file nằm ở `artifacts/bench8/files_created.txt`.
