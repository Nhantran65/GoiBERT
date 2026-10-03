# Chất lượng pool HPLT so với corpus v2 và v1

Đo bằng CPU trên text trước bước làm sạch của v2. U là 20.000 chunk web được chọn đều từ 284.991 chunk web *đã xuất* của v2, nối về doc raw qua đoạn đầu đã chuẩn hóa; R là 20.000 doc chọn đều từ 609.624 doc manifest HPLT chưa vào lượt chọn seed 42; C là 20.000 khối 30 dòng CC100 liên tiếp sau khi bỏ dòng domain của v1. Seed U/R/C = 1234. Một doc raw có thể tạo nhiều chunk U; vì vậy U có trọng số theo số chunk đầu ra. Các tỷ lệ đoạn có mẫu số là toàn bộ đoạn trong mẫu trừ khi ghi khác.

V1 có 34763 dòng domain phân biệt theo heuristic tần suất bội số 20, tương ứng 735000 lượt dòng bị bỏ; còn 296430 khối CC100 hoàn chỉnh. C được lấy bằng reservoir sampling seed 1234. `split_paragraphs` của v2 còn tách các ký tự xuống dòng Unicode bên trong vài dòng v1, nên số đoạn đo của C có thể lớn hơn 30 lần số khối.

## Truy vết v2

`select_general()` với seed 42 chọn 553528 doc đầu vào, 540002322 token mmBERT từ 1163152 doc manifest. Danh sách ID/pool ordinal nằm tại `artifacts/pool_quality/v2_selected_input_ids.jsonl`. Sau lọc, cắt đoạn và lựa chọn ngẫu nhiên lần hai, `doc_sources.jsonl` ghi 284991 chunk web với tổng 232889660 token Bami, khớp 232889660. Đây là đối chiếu metadata đầu ra, không phải chạy lại toàn bộ bộ lọc v2. Lượt nối raw của mẫu U tìm 20000/20000 chunk, ứng với 19930 ID raw khác nhau; số khóa đoạn đầu có nhiều doc ứng viên khớp là 161.

## Chỉ số trước làm sạch

| Chỉ số | U | R | C |
|---|---:|---:|---:|
| Số đơn vị mẫu | 20000 | 20000 | 20000 |
| Tổng token Bami | 22578825 | 17647934 | 20741900 |
| Token/doc trung bình | 1128.94125 | 882.3967 | 1037.095 |
| Token/doc trung vị | 717.0 | 666.0 | 1024.0 |
| Token/doc p90 | 2199.0 | 1641.1000000000022 | 1286.0 |
| Đoạn/doc trung bình | 21.52155 | 17.17335 | 30.0007 |
| Đoạn/doc trung vị | 13.0 | 12.0 | 30.0 |
| Token/đoạn trung bình | 52.45631704036187 | 51.38174555343017 | 34.5690267227098 |
| Token/đoạn trung vị | 37.0 | 38.0 | 21.0 |
| Token/đoạn p90 | 115.0 | 111.0 | 80.0 |
| Token thuộc đoạn vietnameseish (%) | 99.46348403869555 | 99.45814053928352 | 99.99996625188628 |
| Đoạn trùng exact, lượt lặp (%) | 4.926225109250961 | 3.537166598246702 | 9.00612319045889 |
| Đoạn có tần suất ≥5 (%) | 2.3021111397645613 | 2.5367793703616357 | 6.378351171805992 |
| Đoạn <5 âm tiết (%) | 7.811937337227105 | 7.903233789563481 | 1.8839560410257095 |
| Đoạn có URL/email/điện thoại (%) | 1.3546886725166172 | 1.4132362060984025 | 1.6804607892482508 |
| Đoạn trúng decontam (%) | 0.29063891773594375 | 0.24660302154209868 | 0.008166476115557304 |
| Doc vượt ngưỡng decontam 0,2 (%) | 0.0 | 0.03 | 0.0 |

Đoạn được tách bằng `split_paragraphs`; độ dài tính bằng Bami. Trùng exact dùng NFC + casefold + gộp khoảng trắng, đếm các lần xuất hiện vượt lần đầu trong mẫu. Boilerplate đếm mọi lần xuất hiện của đoạn có tần suất ≥5 trong mẫu. `vietnameseish` là hàm nguyên bản của v1, tỷ lệ được cân theo token Bami của đoạn. Decontam dùng nguyên `Decontam.hit` và ngưỡng doc >0,2 của `build.py`; mã v2 hiện có 7 khóa task với validation/test, dù mô tả cũ gọi là 8 task. Chỉ text val/test được dùng làm khóa; không đọc nhãn test.

## Metadata HPLT

| Chỉ số | U | R |
|---|---:|---:|
| Token mmBERT trong mẫu | 24787789 | 19443627 |
| Bami/mmBERT | 0.9108849926066419 | 0.9076461917316146 |
| Token của top-20 host (%) | 7.915044294820479 | 9.289778622245528 |
| s1_mean n | 20000 | 20000 |
| s1_mean mean | -5.46957214434436 | -5.421103242976043 |
| s1_mean median | -5.476701500903372 | -5.43443020149695 |
| s1_mean p90 | -4.494211944046928 | -4.451893157784126 |
| s2_prob n | 20000 | 20000 |
| s2_prob mean | 0.18007931058974952 | 0.17801008716283867 |
| s2_prob median | 0.18421864928463566 | 0.1816185970987379 |
| s2_prob p90 | 0.2412856392618453 | 0.2397630818883637 |
| mt n | 20000 | 20000 |
| mt mean | 0.07370305 | 0.07278 |
| mt median | 0.067 | 0.067 |
| mt p90 | 0.103 | 0.102 |
| op n | 20000 | 20000 |
| op mean | 0.15760595 | 0.15042035 |
| op median | 0.096 | 0.094 |
| op p90 | 0.357 | 0.33 |
| re n | 20000 | 20000 |
| re mean | 0.0626954 | 0.0630574 |
| re median | 0.063 | 0.063 |
| re p90 | 0.076 | 0.076 |
| pii_n n | 20000 | 20000 |
| pii_n mean | 0.01525 | 0.0134 |
| pii_n median | 0.0 | 0.0 |
| pii_n p90 | 0.0 | 0.0 |
| pii_n positive_n | 268 | 251 |
| Bin, số doc | {"7": 5722, "8": 6457, "9": 7821} | {"7": 6113, "8": 6741, "9": 7146} |
| Bin 7 (%) | 28.610000000000003 | 30.564999999999998 |
| Bin 8 (%) | 32.285000000000004 | 33.705 |
| Bin 9 (%) | 39.105000000000004 | 35.730000000000004 |

`s1_mean` và `s2_prob` lấy từ `data/mmbert_vi_cpt_pilot/v2/spun_scores/`; đây là tên trường tương ứng với điểm S1 doc và S2 doc của pipeline. `op`, `re`, `mt`, `pii_n` lấy từ `pool.jsonl`.

## Đọc mù loại nội dung

Seed 7 chọn 60 mục mỗi mẫu, lấy 600 ký tự đầu, trộn 180 trích đoạn và ẩn nguồn khi gán nhãn. Sau khi gán nhãn mới ghép lại U/R/C. Mỗi ô là số mục trên 60 và tỷ lệ phần trăm.

| Loại | U | R | C |
|---|---:|---:|---:|
| tin tức/báo chí | 37/60 (61.66666666666667%) | 31/60 (51.66666666666667%) | 0/60 (0.0%) |
| bài giới thiệu/kiến thức | 9/60 (15.0%) | 11/60 (18.333333333333332%) | 1/60 (1.6666666666666667%) |
| diễn đàn/UGC | 2/60 (3.3333333333333335%) | 0/60 (0.0%) | 0/60 (0.0%) |
| thương mại/quảng cáo/SEO | 6/60 (10.0%) | 13/60 (21.666666666666668%) | 0/60 (0.0%) |
| truyện/văn học | 1/60 (1.6666666666666667%) | 1/60 (1.6666666666666667%) | 1/60 (1.6666666666666667%) |
| văn bản pháp luật/hành chính | 2/60 (3.3333333333333335%) | 2/60 (3.3333333333333335%) | 0/60 (0.0%) |
| rác/lỗi/máy dịch | 3/60 (5.0%) | 2/60 (3.3333333333333335%) | 58/60 (96.66666666666667%) |

## Ước lượng dung lượng và thành phần

Pool manifest có 1134178900 token mmBERT. Hệ số R = 0.9076461917316146 Bami/mmBERT cho 1029433159.3273518 token Bami thô khi ngoại suy toàn pool. Trong mẫu R, decontam (gồm doc và đoạn) loại 0.05829577558483616% token, tương ứng 600116.0443573619 token ngoại suy; exact dedup sau decontam loại 0.432781536920979% token, tương ứng 4455196.648511103 token, trong đó các đoạn boilerplate ≥5 lần đóng góp 0.19759253406092747% token, tương ứng 2034083.0659783795 token. Tỷ lệ token giữ = 0.9950892268749418; suy ra 1024377846.6344833 token web. Boilerplate là tập con của exact dedup, không trừ lần thứ hai. Dedup trên toàn pool và giữa các nguồn có thể loại nhiều hơn trong mẫu R; ước lượng web này có xu hướng lạc quan. Cắt đoạn ở 2048 token chưa được ngoại suy.

Wiki còn 6770 doc, 12630675 token mmBERT chưa chọn; dùng tỷ lệ Bami/mmBERT của wiki v2 ước toàn wiki 60427333.747554936 token Bami. UGC refined có 30366956 token mmBERT và v2 đã đưa cả 78886 doc vào lượt lọc, nên không còn doc refined chưa dùng; phần sạch hiện có 25456887 token Bami. Domain base sạch = 2213077 token, lặp tối đa 20 = 44261540 token.

| Nguồn | v1 token Bami | v1 % | v2 token Bami | v2 % | v2-large token Bami ước | v2-large % ước |
|---|---:|---:|---:|---:|---:|---:|
| web | 307699789 | 93.62998589107428 | 232889660 | 70.86588186424505 | 869854239.2524451 | 86.9854239252445 |
| wiki | 0 | 0.0 | 49353825 | 15.017851509588809 | 60427333.747554936 | 6.042733374755493 |
| ugc | 0 | 0.0 | 25456887 | 7.746263817695623 | 25456887 | 2.5456887 |
| domain | 20934020 | 6.3700141089257185 | 20934020 | 6.370002808470515 | 44261540 | 4.426154 |
| Tổng | 328633809 | 100 | 328634392 | 100 | 1000000000.0 | 100 |

Web của v1 là CC100; web của v2 và v2-large là HPLT. Để đủ 1.000.000.000 token với wiki/UGC/domain như trên cần web 869854239.2524451 token. Ngoại suy R cho dung lượng web sạch dư 154523607.38203824 token so với nhu cầu; nếu tổn thất bổ sung trên toàn pool vượt 15.010552747592321% token Bami thô thì kịch bản 1B bắt đầu thiếu web (giữ nguyên các giả định khác). Tỷ lệ UGC vẫn dưới 10%; domain không vượt 20 lượt. Wiki không thể giữ mức 15% như v2 vì toàn bộ wiki đủ điều kiện ước chỉ có 60427333.747554936 token Bami.

## Kết luận và giới hạn

R gần U về tỷ lệ token vietnameseish (0.9945814053928352 so với 0.9946348403869555), độ dài đoạn và điểm spun/MT. R có tỷ lệ đoạn trùng exact thấp hơn (0.03537166598246702 so với 0.049262251092509604), nhưng trong 60 trích đoạn đọc mù có nhiều mục thương mại/SEO hơn (13 so với 6); đây là tín hiệu nhỏ, chưa đủ để kết luận R kém chất lượng rõ rệt. Hai mẫu chịu thiên lệch khác nhau: U là mẫu chunk đã qua lựa chọn sau làm sạch, còn R là mẫu doc đầu vào chưa dùng. v2-large dự kiến tăng web từ 0.7086588186424505 lên 0.8698542392524451; tỷ trọng tương đối wiki, UGC và domain giảm. Đạt đúng 1B còn phụ thuộc dedup toàn pool.

Giới hạn: không chạy lại toàn bộ v2 để truy vết chính xác mọi chunk về raw ID; đối chiếu 232889660 dựa trên `doc_sources.jsonl`. Khi nhiều doc raw cùng chứa đoạn đầu của một chunk, lượt nối chọn doc xuất hiện trước trong thứ tự quét như `build.py`; một doc trước đó bị loại ở ngưỡng decontam có thể khiến phép nối sai. Trích đoạn đọc mù chỉ 600 ký tự đầu, mỗi nhóm 60 mục. C là khối 30 dòng CC100, không phải doc thật; nhãn rác/lỗi của C chủ yếu phản ánh phép ghép dòng khác chủ đề. Heuristic dòng domain bội số 20 có thể bỏ sót hoặc gộp nhầm. Tỷ lệ lặp/boilerplate trong 20.000 mục thấp hơn khi đo trên toàn pool; các khoản loại trừ có giao nhau. Wiki mở rộng và tương tác dedup liên nguồn được ước bằng hệ số cũ, chưa dựng corpus mới. Các chỉ số metadata chỉ là phân bố mẫu.

## Lệnh tái lập

```bash
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
.venv-e0-cuda/bin/python scripts/pool_quality/measure.py prepare
.venv-e0-cuda/bin/python scripts/pool_quality/measure.py extract
.venv-e0-cuda/bin/python scripts/pool_quality/measure.py v1
.venv-e0-cuda/bin/python scripts/pool_quality/summarize.py --samples C
.venv-e0-cuda/bin/python scripts/pool_quality/summarize.py --samples U R
.venv-e0-cuda/bin/python scripts/pool_quality/merge.py
.venv-e0-cuda/bin/python scripts/pool_quality/project.py
.venv-e0-cuda/bin/python scripts/pool_quality/audit.py prepare
# Đọc mù audit_reading.txt rồi ghi blind_id và mã nhãn 0..6 vào /tmp/goibert_pool_quality_20260928/audit_labels.txt
.venv-e0-cuda/bin/python scripts/pool_quality/audit.py finish
.venv-e0-cuda/bin/python scripts/pool_quality/report.py
```
