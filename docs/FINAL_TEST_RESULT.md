# Kết quả final test: Student E7 + KD và BamiBERT gốc

Khóa manifest: `2026-09-27T03:51:52.854198+00:00`. Validation: 48/48 checkpoint PASS; sai số chỉ số tối đa `0.0`; dự đoán validation khớp hoàn toàn 48/48. Test: 48/48 bộ dự đoán hoàn chỉnh, mỗi checkpoint đúng một lượt inference; một lượt chuẩn bị của seed 43 lỗi trước inference và được ghi log. Chỉ số trong bảng là phần trăm (giá trị evaluator nhân 100).

## Test 8 dataset / 15 chỉ số

| Dataset | Chỉ số | Bami test mean (range) | Student test mean (range) | Student − Bami | Bami validation mean | Student validation mean | Paper BamiBERT test | Tham số Bami / Student |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vsfc_sentiment | accuracy | 93.50389555695936 (93.2406822488945–93.74605180037902) | 92.45104232469994 (92.38787113076438–92.48262792166771) | -1.052853232259423 | 94.69361970941252 | 94.29353548115392 | 93.86 | 102953475 / 26546691 |
| vsfc_sentiment | macro_f1 | 82.35541605059437 (81.61328179035995–83.03297211171513) | 80.53736474813714 (79.72745289321686–81.33450334094049) | -1.818051302457235 | 86.82650498038737 | 84.34329986928745 | 83.41 | 102953475 / 26546691 |
| vsfc_topic | accuracy | 88.81869867340492 (88.5028427037271–89.35565382185723) | 89.0608549168246 (88.81869867340492–89.2293114339861) | 0.24215624341967157 | 88.81869867340492 | 89.66098125921246 | 89.34 | 102954244 / 26547076 |
| vsfc_topic | macro_f1 | 78.8788403933251 (78.1765763472828–79.79810031544696) | 78.70761279480165 (78.60269981828473–78.84019858821661) | -0.17122759852345837 | 79.53172479547898 | 79.91593400765736 | 79.9 | 102954244 / 26547076 |
| visfd | detection_micro_f1 | 89.25639960806906 (89.03367395926205–89.39507604173771) | 88.95547372414364 (88.93490715814178–88.99640586121095) | -0.30092588392541586 | 89.1542915558615 | 89.17912149675311 | 89.14 | 102983466 / 26561706 |
| visfd | pair_micro_f1 | 84.41477048152257 (84.10635155096013–84.63627688468031) | 83.2380772640192 (83.18178457899018–83.32227872356106) | -1.1766932175033702 | 83.98660341780912 | 83.09514623067697 | 84.24 | 102983466 / 26561706 |
| absa_restaurant | detection_micro_f1 | 87.03105700158335 (86.45500657523952–87.51652502360717) | 86.77639904049137 (86.7595818815331–86.79753656658968) | -0.25465796109197925 | 85.56649684716096 | 85.75553383176991 | 88.01 | 102988080 / 26564016 |
| absa_restaurant | pair_micro_f1 | 73.5281203373733 (72.36520758970505–74.14927617973304) | 73.69420471875858 (73.55787843592722–73.8526802930968) | 0.16608438138528925 | 73.3609012426287 | 73.44540743530266 | 74.89 | 102988080 / 26564016 |
| absa_hotel | detection_micro_f1 | 80.15254284305225 (80.11677934849416–80.18532818532819) | 80.37626349061205 (80.30104712041884–80.42037186742118) | 0.22372064755980148 | 79.46142711085191 | 77.87531243287528 | 79.99 | 103055752 / 26597896 |
| absa_hotel | pair_micro_f1 | 73.94041941397462 (73.82239382239382–74.05909797822706) | 74.86750468924039 (74.54188481675392–75.05254648342765) | 0.9270852752657675 | 73.68595969124232 | 73.04859383161944 | 72.65 | 103055752 / 26597896 |
| vispam | accuracy | 90.18621036738801 (89.98490186210367–90.41268243583292) | 89.93457473578258 (89.78359335681932–90.01006542526422) | -0.25163563160543845 | 90.10482180293502 | 90.20964360587003 | 90.76 | 102954244 / 26547076 |
| vispam | macro_f1 | 77.55027950755748 (75.58476450795648–78.62545946218769) | 76.40267717472103 (75.98065842447805–76.94608182329775) | -1.147602332836442 | 77.37953460125227 | 75.43608924192098 | 78.2 | 102954244 / 26547076 |
| vinli | accuracy | 80.07355399531929 (79.93981945837513–80.2407221664995) | 69.22991195809651 (68.80641925777333–69.70912738214643) | -10.843642037222779 | 80.94605073667886 | 70.99811676082862 | 81.01 | 102954244 / 26547076 |
| vinli | macro_f1 | 80.18251810969853 (80.04418193938642–80.3352662216309) | 69.46471198457472 (68.9426127194247–70.01766426227016) | -10.717806125123815 | 80.93517219321599 | 71.06698446147804 | 81.15 | 102954244 / 26547076 |
| phoner | f1 | 94.32204948374392 (94.23990498812351–94.38870940316274) | 92.1770076041068 (92.04788909892879–92.26489061248577) | -2.1450418796371196 | 95.00436339092543 | 93.20302623891332 | 94.9 | 102376725 / 26405781 |

## Tổng hợp

| Phạm vi | Student ≥ Bami | Bami mean | Student mean | Student − Bami |
|---|---:|---:|---:|---:|
| 15 chỉ số | 4/15 | 83.61298478823774 | 81.72491207793402 | -1.8880727103037296 |
| Bỏ ViNLI (13 chỉ số) | 4/13 | 84.14913074758064 | 83.62915824817992 | -0.5199724994007193 |

## Tốc độ và bộ nhớ

Head phân loại câu lấy từ checkpoint `vsfc_sentiment` seed 42. Dung lượng checkpoint dưới đây là `state_dict` encoder + head lưu riêng vào `/tmp`, không bao gồm optimizer; file checkpoint gốc có thể lớn hơn. GPU bf16; CPU fp32 với 8 luồng. Median/p90 tính trên thời gian mỗi lần forward, có đồng bộ CUDA.

| Model | Tham số | Checkpoint fp32 (byte) | Checkpoint bf16 (byte) | Checkpoint gốc (byte) |
|---|---:|---:|---:|---:|
| bami | 102953475 | 411879674 | 205972730 | 1235724148 |
| student | 26546691 | 106242497 | 53149121 | 106257800 |

| Thiết bị | Batch | Length | Bami median ms | Bami p90 ms | Student median ms | Student p90 ms | Tăng tốc median | Bami mẫu/s | Student mẫu/s | Bami peak VRAM byte | Student peak VRAM byte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU bf16 | 1 | 64 | 8.957252837717533 | 9.783755987882614 | 7.82934483140707 | 8.87492299079895 | 1.1440616080397825× | 111.64137242940858 | 127.72460806535743 | 641325056 | 179716096 |
| GPU bf16 | 1 | 128 | 9.022491052746773 | 9.969854727387428 | 7.845081388950348 | 8.815368637442589 | 1.1500825301130444× | 110.83413595578615 | 127.46840350292369 | 642103808 | 180257792 |
| GPU bf16 | 1 | 256 | 9.300035424530506 | 11.254707351326942 | 8.015718311071396 | 8.99219885468483 | 1.1602248312150887× | 107.52647214249542 | 124.75488299268068 | 644794880 | 181341184 |
| GPU bf16 | 32 | 64 | 9.43074282258749 | 10.651679709553719 | 8.095135912299156 | 9.21802781522274 | 1.1649888185643815× | 3393.1579518165927 | 3952.9910734891487 | 691499520 | 208321536 |
| GPU bf16 | 32 | 128 | 9.554595686495304 | 10.798443108797073 | 7.9406918957829475 | 8.956382051110268 | 1.2032447313022496× | 3349.173638527643 | 4029.8755347747715 | 735719936 | 227638272 |
| GPU bf16 | 32 | 256 | 14.396632090210915 | 17.454907298088074 | 8.192049339413643 | 9.624576196074486 | 1.7573907936498554× | 2222.742082973601 | 3906.2264732759095 | 826651136 | 273480704 |
| CPU fp32 | 1 | 64 | 34.15943682193756 | — | 9.0867904946208 | — | 3.7592411580479674× | — | — | — | — |
| CPU fp32 | 1 | 128 | 46.9826515763998 | — | 13.6313047260046 | — | 3.44667311902803× | — | — | — | — |

GPU đo: `NVIDIA L40S`; snapshot VRAM/tải và process khác trước/sau phép đo: `artifacts/final_test/efficiency.json`. Tổng thời gian dùng một GPU ≤ `0.2754986008670595` GPU-giờ, tính bảo thủ từ lúc tạo thư mục final_test trước lượt kiểm đầu tiên đến khi ghi xong efficiency, gồm cả thời gian viết code không chạy GPU. Riêng cửa sổ test và efficiency là `0.20157532897260455` GPU-giờ.

## Kiểm tra và giới hạn

- Lượt kiểm validation ban đầu dừng ở Bami ViSFD seed 42 do script sử dụng micro batch 32 trong khi checkpoint lưu micro batch 16. Sau khi sửa script đọc batch size từ metadata, toàn bộ validation đã được kiểm lại trước khóa manifest. Không truy cập test trong lượt dừng đó.
- Student ViNLI seed 42 đã lưu đủ 2991 dự đoán trước khi process bị dừng do bước tính metric kéo dài; metric được tính từ file đã lưu, không chạy inference lần hai. Student ViNLI seed 43 lượt đầu lỗi import trước inference; chỉ checkpoint đó chạy lại theo log.
- SHA256 của hai file ViNLI/PhoNER khớp giá trị được chỉ định trước khi đọc test; SHA256 của 5 dataset cũ được đối chiếu sau test với manifest dữ liệu gốc. SHA256 ViSpam được ghi nhận vì manifest trước đó chưa có giá trị tham chiếu.
- Các số Paper BamiBERT chỉ là tham khảo; số local dùng checkpoint, tokenizer, split và evaluator của repo.
- Không train, không sửa checkpoint, không chọn lại cấu hình sau khi xem test.

## File tạo

- `scripts/final_test/run.py`, `scripts/final_test/resume.py`, `scripts/final_test/efficiency.py`, `scripts/final_test/report.py`
- `artifacts/final_test/validation_checks.json`, `locked_manifest.json`, `test_access_log.json`, `test_results.json`, `efficiency.json`
- `artifacts/final_test/predictions/`: 48 file; `artifacts/final_test/metrics/`: 48 file
- `docs/FINAL_TEST_RESULT.md`
