# BENCH8 E7 + Cafe OOF KD — kết quả validation

## Phạm vi và recipe

Chỉ dùng train và validation; không đọc hoặc nạp test. Student dùng E7 LLRD γ = 0,8, KD từ logits Cafe OOF 5 fold với α = 0,5 và T = 2. LR student: ViSpam 5e-5, ViNLI 1e-4, PhoNER 1e-4; seed 42/43/44. Teacher Cafe dùng LR 2e-5, batch 32, 30 epoch, warmup 10%, clip 1, bf16; chọn best epoch theo validation. ViNLI là mirror chưa xác minh; các dataset chỉ dùng cho nghiên cứu. α và T không được tinh chỉnh. `vispam_binary` là chỉ số phụ, không huấn luyện KD.

Điểm plan chưa nói rõ: seed cho từng teacher fold được đặt 42, theo `cafe_oof_kd` hiện có; mỗi fold giữ nguyên thứ tự train từ `bench8/data.py`. AdamW của teacher dùng `goibert.train.make_optimizer`, như teacher Cafe hiện có. Student giữ batch 32, max length 256/512, 30 epoch, warmup 10%, clip 1, bf16 và chọn best validation F1 theo `bench8/train.py`.

## PhoNER: căn nhãn và sáu kiểm tra

21 nhãn BIO lấy đúng thứ tự từ `bench8/data.py` và được assert trong code mới. Cả train (5.027 câu, 167.541 âm tiết) và dev (2.000 câu, 71.325 âm tiết) có **0 âm tiết không sinh subword, 0 âm tiết bị cắt và 0 câu bị cắt** với cả tokenizer student 20.481 token và tokenizer Cafe 250.002 token. Kiểm chéo 20 câu ngẫu nhiên: 20/20 khớp số âm tiết giữa hai tokenizer và dữ liệu gốc. Chi tiết ở `artifacts/bench8_kd/phoner/alignment_check.json`.

1. Gom vị trí trên batch hai câu, có padding, subword phụ và âm tiết bị cắt: PASS.
2. Tra nhãn mềm theo (ID câu, chỉ số âm tiết) sau xáo thứ tự batch: PASS.
3. KD loss với teacher one-hot và logits đồng nhất, gradient hữu hạn: PASS.
4. Smoke teacher một epoch với 64 câu train, OOF fold 0 có ID, offset, shape `[n_words,21]`, `has_logit` và giá trị hữu hạn: PASS.
5. Smoke student một epoch với nhãn mềm one-hot gold, loss hữu hạn và postrun checks: PASS.
6. Kiểm chéo số âm tiết của 20 câu ngẫu nhiên: 20/20 PASS.

## Chất lượng teacher

| Task | F1 validation từng fold | Mean | Range | F1 OOF trên train | Bất đồng | Mean max probability | Tỷ lệ max probability ≥0,99 |
|---|---|---:|---:|---:|---:|---:|---:|
| vispam | 0.7799401940188558, 0.759527604686578, 0.754770538156418, 0.772004037956471, 0.7544206808967168 | 0.764132611143008 | 0.7544206808967168–0.7799401940188558 | 0.7530016431361506 | 0.09366699287012442 | 0.97996586561203 | 0.7747099119250664 |
| vinli | 0.8513177963672013, 0.8562036338451319, 0.8482027008107152, 0.8510697813037659, 0.8526637859507251 | 0.851891539655508 | 0.8482027008107152–0.8562036338451319 | 0.8558778867617673 | 0.14432228421398097 | 0.9940444827079773 | 0.960206760748277 |
| phoner | 0.9579764987054372, 0.9563655713715578, 0.9561853951317105, 0.956701581159517, 0.9594351561979617 | 0.9573328405132369 | 0.9561853951317105–0.9594351561979617 | 0.9660664819944599 | 0.011668785550999457 | 0.9990220475780714 | 0.9928554801511272 |

Mỗi task có độ phủ OOF 100%, ID train không trùng. Best epoch từng fold:

- vispam: [4, 23, 21, 9, 7]
- vinli: [28, 14, 12, 26, 29]
- phoner: [22, 29, 18, 20, 29]

F1 OOF ViSpam theo lớp:

- Lớp 0: 0.9436685724997657
- Lớp 1: 0.5435356200527705
- Lớp 2: 0.6332869726471951
- Lớp 3: 0.8915154073448712

PhoNER: accuracy theo âm tiết 0.9883312144490005; bất đồng trên âm tiết gold là thực thể 0.029846444106703004; âm tiết có logits 167541; thiếu logits 0. F1 theo loại thực thể:

| Loại | Gold | Dự đoán | TP | F1 |
|---|---:|---:|---:|---:|
| AGE | 682 | 693 | 666 | 0.9687272727272728 |
| DATE | 2549 | 2567 | 2537 | 0.991790461297889 |
| GENDER | 542 | 559 | 525 | 0.9536784741144414 |
| JOB | 205 | 230 | 184 | 0.8459770114942529 |
| LOCATION | 5398 | 5458 | 5255 | 0.9681282240235815 |
| NAME | 349 | 356 | 334 | 0.9475177304964539 |
| ORGANIZATION | 1137 | 1144 | 1064 | 0.9329241560718983 |
| PATIENT_ID | 3240 | 3253 | 3205 | 0.9872170029262283 |
| SYMPTOM_AND_DISEASE | 1439 | 1509 | 1350 | 0.9158751696065129 |
| TRANSPORTATION | 226 | 232 | 225 | 0.982532751091703 |

vispam: tỷ lệ cắt teacher train từng fold [8.738203425375742e-05, 0.00017474879860200962, 0.00017474879860200962, 0.00017474879860200962, 8.737439930100481e-05]; validation từng fold [0.0, 0.0, 0.0, 0.0, 0.0].

vinli: tỷ lệ cắt teacher train từng fold [0.0, 0.0, 0.0, 0.0, 0.0]; validation từng fold [0.0, 0.0, 0.0, 0.0, 0.0].

## KD − CE, 3 task mới

Các số dưới đây là phần trăm của chỉ số validation, tính lại từ file dự đoán từng seed.

| Task | Chỉ số | CE mean (range) | KD mean (range) | KD − CE mean (range) | Seed dương |
|---|---|---:|---:|---:|---:|
| vispam | macro_f1 | 75.07995170121185 (74.20763760049473–75.66217280692965) | 75.436089241921 (75.36398677404834–75.53864772250982) | 0.3561375407091371 (-0.2981860328813042–1.1979956287100748) | 2/3 |
| vinli | macro_f1 | 69.32069117761473 (68.87172178421949–69.8045234899289) | 71.06698446147803 (70.64210009668068–71.35713191895785) | 1.7462932838632999 (1.3562718379848633–2.329999584576086) | 3/3 |
| phoner | f1 | 92.73755005563906 (92.58349379191289–92.85998672859986) | 93.20302623891332 (93.0950016550811–93.32188387608164) | 0.4654761832742622 (0.32583200867665596–0.7383900841687563) | 3/3 |

KD − CE theo seed:

- vispam: {'42': -0.2981860328813042, '43': 0.16860302629864066, '44': 1.1979956287100748}
- vinli: {'42': 2.329999584576086, '43': 1.3562718379848633, '44': 1.5526084290289504}
- phoner: {'42': 0.7383900841687563, '43': 0.3322064569773744, '44': 0.32583200867665596}

## 8 dataset / 15 chỉ số

Trung bình 3 seed trên validation. Tất cả student ở đây dùng E7 + Cafe OOF KD; năm task cũ lấy nhánh KD, gồm cả Sentiment và Topic.

| Dataset | Chỉ số | Bami mean (range) | Student E7 + KD mean (range) | Student − Bami | Tham số Bami / Student |
|---|---|---:|---:|---:|---:|
| vsfc_sentiment | accuracy | 94.69361970941252 (94.63044851547694–94.75679090334808) | 94.29353548115392 (94.25142135186356–94.31459254579912) | -0.40008422825859213 | 102953475 / 26546691 |
| vsfc_sentiment | macro_f1 | 86.82650498038737 (86.11572017056693–88.03856639708009) | 84.34329986928745 (84.07607338121105–84.49202979103964) | -2.4832051110999203 | 102953475 / 26546691 |
| vsfc_topic | accuracy | 88.81869867340492 (88.6923562855338–88.88186986734048) | 89.66098125921246 (89.32406822488946–89.9557801642451) | 0.8422825858075385 | 102954244 / 26547076 |
| vsfc_topic | macro_f1 | 79.53172479547898 (79.44195162727668–79.62041454033478) | 79.91593400765736 (79.58543930118331–80.17222295149298) | 0.38420921217837645 | 102954244 / 26547076 |
| visfd | detection_micro_f1 | 89.1542915558615 (88.94736842105263–89.33981931897151) | 89.17912149675311 (88.86716021790753–89.3909626719057) | 0.02482994089160684 | 102983466 / 26561706 |
| visfd | pair_micro_f1 | 83.98660341780912 (83.69581520923954–84.38339802926247) | 83.09514623067699 (82.68625393494229–83.32088995072421) | -0.8914571871321328 | 102983466 / 26561706 |
| absa_restaurant | detection_micro_f1 | 85.56649684716096 (84.60093896713616–86.47887323943661) | 85.7555338317699 (85.70066730219257–85.82412301777993) | 0.1890369846089328 | 102988080 / 26564016 |
| absa_restaurant | pair_micro_f1 | 73.3609012426287 (71.83098591549296–74.74178403755869) | 73.44540743530266 (73.40324118207818–73.50674373795762) | 0.0845061926739561 | 102988080 / 26564016 |
| absa_hotel | detection_micro_f1 | 79.46142711085193 (79.14110429447852–79.64737447297814) | 77.87531243287529 (77.54937525191455–78.1766841468334) | -1.5861146779766386 | 103055752 / 26597896 |
| absa_hotel | pair_micro_f1 | 73.68595969124232 (73.2361963190184–74.07695297318305) | 73.04859383161946 (72.8738411930673–73.17466720451795) | -0.6373658596228609 | 103055752 / 26597896 |
| vispam | accuracy | 90.104821802935 (90.0–90.18867924528303) | 90.20964360587003 (90.0–90.56603773584906) | 0.10482180293502097 | 102954244 / 26547076 |
| vispam | macro_f1 | 77.37953460125227 (76.12649960236213–78.05437251819183) | 75.436089241921 (75.36398677404834–75.53864772250982) | -1.9434453593312782 | 102954244 / 26547076 |
| vinli | accuracy | 80.94605073667886 (80.259222333001–81.48886673313393) | 70.99811676082862 (70.62146892655367–71.21967431040213) | -9.947933975850233 | 102954244 / 26547076 |
| vinli | macro_f1 | 80.93517219321599 (80.23325351443317–81.48300667157972) | 71.06698446147803 (70.64210009668068–71.35713191895785) | -9.868187731737962 | 102954244 / 26547076 |
| phoner | f1 | 95.00436339092542 (94.94101179764047–95.12015529821272) | 93.20302623891332 (93.0950016550811–93.32188387608164) | -1.8013371520120955 | 102376725 / 26405781 |

## GPU-giờ, kiểm tra và sự cố

Tổng thời gian GPU: 48484.59940505028 giây = 13.467944279180633 GPU-giờ, gồm 15 teacher fold, 9 student KD và hai smoke PhoNER. Trần: 18 GPU-giờ.
15/15 teacher fold và 9/9 student KD có trạng thái `completed`, postrun checks PASS. Các chỉ số trong bảng được tính lại từ file dự đoán và đối chiếu metric đã lưu. Mỗi GPU 0–3 chỉ có một job của đợt chạy này tại một thời điểm; ngưỡng bộ nhớ trống khi launcher cấp job thấp nhất ghi trong log là 21.422 MiB. Không có check FAIL hoặc sự cố cần chạy lại.

## File tạo

Danh sách đầy đủ ở `artifacts/bench8_kd/files_created.txt`. Các đầu ra chính:
- `scripts/bench8_kd/`: code trainer, launcher, kiểm PhoNER, ghép OOF, tổng hợp, unit test và render báo cáo.
- `artifacts/bench8_kd/teacher/{task}/fold_{0..4}/`: checkpoint, logits OOF, prediction validation, config, metric, curve, metadata và status.
- `artifacts/bench8_kd/soft/{task}.npz` và `artifacts/bench8_kd/{task}/teacher_stats.json`: nhãn mềm và chất lượng teacher.
- `artifacts/bench8_kd/student/{task}/seed_{42,43,44}/`: checkpoint, prediction validation, metric, curve, config, metadata và status.
- `artifacts/bench8_kd/phoner/alignment_check.json` và `smoke_checks.json`.
- `artifacts/bench8_kd/summary.json` và `table_8datasets_kd.md`.
- `docs/BENCH8_KD_RESULT.md`.
