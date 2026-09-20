# E20 Result -- CTI (Compositional Token-state Invariance) screening

Generated 2026-09-14T01:26:11Z. See `docs/E20_PLAN.md` (locked spec) and `docs/CTI_PROPOSAL.md` (motivation).


## absa_restaurant


### Validation pair F1 (best epoch)

| arm | pair F1 | detection micro F1 | best epoch | elapsed (s) | status |
|---|---|---|---|---|---|
| e19 | 74.51 | 87.28 | 22 | 1330.9810473918915 | ok |
| d1_composite | 75.46 | 87.66 | 21 | 1818.1455504894257 | ok |
| cti | 75.12 | 87.56 | 29 | 2152.8490772247314 | ok |

#### e19 -- error decomposition groups

| group | rows | gold | missed% | wrong% | spurious/row |
|---|---|---|---|---|---|
| k=1 | 521 | 521 | 10.4 | 12.7 | 0.173 |
| k=2 | 219 | 438 | 14.6 | 12.1 | 0.1 |
| k=3 | 31 | 94 | 27.7 | 14.9 | 0.29 |
| k>=2|same | 141 | 295 | 15.3 | 7.1 | 0.106 |
| k>=2|mixed | 109 | 237 | 19.0 | 19.4 | 0.147 |
| k>=2|freq_pair | 241 | 508 | 15.6 | 12.6 | 0.112 |
| k>=2|rare_pair(<=5) | 9 | 24 | 45.8 | 12.5 | 0.444 |


#### d1_composite -- error decomposition groups

| group | rows | gold | missed% | wrong% | spurious/row |
|---|---|---|---|---|---|
| k=1 | 521 | 521 | 9.6 | 12.5 | 0.2 |
| k=2 | 219 | 438 | 13.0 | 11.4 | 0.096 |
| k=3 | 31 | 94 | 20.2 | 14.9 | 0.323 |
| k>=2|same | 141 | 295 | 12.9 | 8.8 | 0.106 |
| k>=2|mixed | 109 | 237 | 16.0 | 16.0 | 0.147 |
| k>=2|freq_pair | 241 | 508 | 14.0 | 11.8 | 0.12 |
| k>=2|rare_pair(<=5) | 9 | 24 | 20.8 | 16.7 | 0.222 |


#### cti -- error decomposition groups

| group | rows | gold | missed% | wrong% | spurious/row |
|---|---|---|---|---|---|
| k=1 | 521 | 521 | 9.8 | 12.5 | 0.2 |
| k=2 | 219 | 438 | 13.0 | 11.6 | 0.096 |
| k=3 | 31 | 94 | 24.5 | 16.0 | 0.194 |
| k>=2|same | 141 | 295 | 13.9 | 6.1 | 0.099 |
| k>=2|mixed | 109 | 237 | 16.5 | 20.3 | 0.119 |
| k>=2|freq_pair | 241 | 508 | 14.4 | 12.4 | 0.108 |
| k>=2|rare_pair(<=5) | 9 | 24 | 29.2 | 12.5 | 0.111 |


### Fork contamination (masked-CLS, part (ii)) and composite battery

| arm | A_correct (mask A) | B_present (mask A, contamination) | A_correct (A alone) | battery A_correct | battery B_correct | battery both_correct |
|---|---|---|---|---|---|---|
| e19 | 67.5 | 10.5 | 78.5 | 48.0 | 44.5 | 11.5 |
| d1_composite | 77.5 | 16.0 | 80.0 | 69.5 | 64.0 | 42.5 |
| cti | 77.0 | 7.5 | 80.5 | 60.5 | 55.0 | 29.0 |

### Gate verdict (absa_restaurant)

FAIL -- cti vs d1 rare-pair missed improvement = -8.4 pts (need >=5); B_present cti=7.5 vs e19=10.5 (need reduced); F1 cti=75.12 vs d1=75.46 (need not down >0.3 pts)


## absa_hotel


### Validation pair F1 (best epoch)

| arm | pair F1 | detection micro F1 | best epoch | elapsed (s) | status |
|---|---|---|---|---|---|
| e19 | 73.63 | 79.84 | 29 | 1888.6563851833344 | ok |
| d1_composite | 73.30 | 78.57 | 13 | 2062.7661056518555 | ok |
| cti | 73.73 | 79.33 | 18 | 2490.930107116699 | ok |

#### e19 -- error decomposition groups

| group | rows | gold | missed% | wrong% | spurious/row |
|---|---|---|---|---|---|
| k=1 | 463 | 463 | 16.6 | 6.0 | 0.238 |
| k=2 | 210 | 420 | 22.9 | 9.3 | 0.305 |
| k=3 | 122 | 426 | 25.4 | 3.1 | 0.525 |
| k>=2|same | 274 | 692 | 21.8 | 3.8 | 0.369 |
| k>=2|mixed | 58 | 154 | 34.4 | 16.9 | 0.466 |
| k>=2|freq_pair | 290 | 717 | 19.8 | 6.1 | 0.334 |
| k>=2|rare_pair(<=5) | 42 | 129 | 48.1 | 6.2 | 0.738 |


#### d1_composite -- error decomposition groups

| group | rows | gold | missed% | wrong% | spurious/row |
|---|---|---|---|---|---|
| k=1 | 463 | 463 | 16.8 | 5.4 | 0.276 |
| k=2 | 210 | 420 | 25.0 | 7.6 | 0.31 |
| k=3 | 122 | 426 | 26.3 | 2.6 | 0.533 |
| k>=2|same | 274 | 692 | 23.8 | 2.6 | 0.376 |
| k>=2|mixed | 58 | 154 | 33.8 | 16.2 | 0.466 |
| k>=2|freq_pair | 290 | 717 | 21.9 | 5.0 | 0.338 |
| k>=2|rare_pair(<=5) | 42 | 129 | 46.5 | 5.4 | 0.762 |


#### cti -- error decomposition groups

| group | rows | gold | missed% | wrong% | spurious/row |
|---|---|---|---|---|---|
| k=1 | 463 | 463 | 16.4 | 5.6 | 0.255 |
| k=2 | 210 | 420 | 25.2 | 7.9 | 0.271 |
| k=3 | 122 | 426 | 25.4 | 3.1 | 0.541 |
| k>=2|same | 274 | 692 | 23.8 | 3.0 | 0.354 |
| k>=2|mixed | 58 | 154 | 31.8 | 16.2 | 0.448 |
| k>=2|freq_pair | 290 | 717 | 21.3 | 5.9 | 0.31 |
| k>=2|rare_pair(<=5) | 42 | 129 | 47.3 | 3.1 | 0.786 |


### Fork contamination (masked-CLS, part (ii)) and composite battery

| arm | A_correct (mask A) | B_present (mask A, contamination) | A_correct (A alone) | battery A_correct | battery B_correct | battery both_correct |
|---|---|---|---|---|---|---|
| e19 | 59.0 | 16.5 | 66.5 | 45.0 | 58.0 | 24.0 |
| d1_composite | 68.0 | 7.5 | 68.5 | 59.0 | 66.5 | 38.0 |
| cti | 70.5 | 2.0 | 68.0 | 58.0 | 65.5 | 34.5 |

### Gate verdict (absa_hotel)

FAIL -- cti vs d1 rare-pair missed improvement = -0.8 pts (need >=5); B_present cti=2.0 vs e19=16.5 (need reduced); F1 cti=73.73 vs d1=73.30 (need not down >0.3 pts)


## Preflight (from artifacts/e20/preflight.json)

- status: **passed**, elapsed 10.4s

- checks: `{"composite_assert_all_pass": true, "cti_sanity_all_pass": true, "eval_parity_all_pass": true, "encoder_no_mutation": true}`

- composites/absa_restaurant: built 2343/2343, valid pairs 66, rare pairs(<=5) 15 (covered 15), distinct pairs used 65

- composites/absa_hotel: built 2393/2393, valid pairs 528, rare pairs(<=5) 361 (covered 346), distinct pairs used 429

- cti/d1 step-time ratio (absa_restaurant): 1.21x

- cti/d1 step-time ratio (absa_hotel): 1.17x


## E19 summary (5 tasks vs E2 teacher CE / E14 BamiBERT CE references)

| task | status | E19 selection F1 | E2 teacher CE % | E14 BamiBERT CE % | delta vs E2 teacher (pts) |
|---|---|---|---|---|---|
| vsfc_sentiment | completed | 85.71 | 85.92 | 86.33 | -0.21 |
| vsfc_topic | completed | 81.77 | 82.17 | 79.53 | -0.4 |
| visfd | completed | 85.08 | 84.36 | 84.38 | 0.72 |
| absa_restaurant | completed | 74.51 | 74.62 | 74.74 | -0.11 |
| absa_hotel | completed | 73.63 | 73.22 | 73.24 | 0.41 |

## Root review (2026-09-14, sau khi chạy lại eval trên thư mục `seed_42__rerun1`)

Vận hành: E2 loop ghi artifact vào `seed_42__rerun1` (do runner đã tạo sẵn `seed_42/` chứa composites_manifest); hai trial Hotel được root chạy tay trên GPU0/GPU2 để dùng GPU trống, nên controller của watcher lỗi FileExistsError khi tới Hotel và eval tự động lần đầu (18:57 UTC) đọc thiếu — đã sửa đường dẫn trong `scripts/eval_e20.py` và chạy lại (01:26 UTC). Cả 4 trial hoàn tất 30 epochs; F1 trong bảng lấy từ `best_validation_metrics.json`.

Đọc kết quả:

1. **CTI làm đúng việc nó được thiết kế** — nhiễm fork (B_present khi CLS chỉ đọc A) giảm mạnh: Hotel 16,5 → 2,0; Restaurant 10,5 → 7,5 (d1 chỉ ghép câu: 7,5 / tăng lên 16,0). **Nhưng không đổi lỗi mục tiêu**: Hotel cặp hiếm missed 48,1 → 47,3 (d1 46,5); trái polarity wrong 16,9 → 16,2 (= d1). F1 Hotel 73,73 vs e19 73,63 vs d1 73,30 — trong nhiễu. ⇒ Giả thuyết "nhiễm token giữa các đoạn gây bias tập nhãn" **bị bác bỏ** với đối chứng sạch. Gate FAIL ở cả hai task.
2. **Ghép câu nhãn hợp (d1, ưu tiên cặp hiếm)** cho Restaurant **75,46** (+0,95 so e19; cao nhất project trên validation Restaurant; E14 BamiBERT 74,74) và cặp hiếm 45,8 → 20,8 (nhưng chỉ 9 câu/24 slot), trái polarity missed 19,0 → 16,0 và wrong 19,4 → 16,0. Hotel thì −0,3 và cặp hiếm không đổi dù composite phủ 346/361 cặp hiếm. Một seed, nhiễu ≈ 0,5–1. Tiền lệ: C3DA/Chai 2023.
3. Hệ quả: lỗi cặp hiếm ở Hotel không sửa được bằng (a) đổi cách đọc (D2), (b) bỏ nhiễm token (CTI), (c) cho model thấy cặp hiếm qua ghép câu (d1). Phần còn lại hợp lý nhất: **thống kê nhãn/ontology** (ROOMS# vs ROOM_AMENITIES#, GENERAL vs MISCELLANEOUS) và độ khó nội tại của câu dài ≥3 aspect — cần adjudication mẫu và/hoặc debias ở quy tắc quyết định (counterfactual theo prior đồng xuất hiện), không phải ở encoder.
