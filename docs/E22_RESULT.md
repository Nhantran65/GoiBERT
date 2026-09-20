# E22 Result -- ViSFD/Hotel d1_composite multi-seed + Restaurant d1_uniform ablation

Generated 2026-09-14T04:40:28Z. Produced by `scripts/eval_e22.py`, reusing `scripts/eval_e20.py`'s `train_pair_frequency` / `error_decomposition_groups` (scored-aspects-aware; ViSFD's binary OTHERS slot excluded) and seed-aware path helpers. Pair F1 = `selection_f1` at the best validation epoch (never test). A `-` cell means that trial has not completed yet (`scripts/e22_launch.sh` still running or not yet launched).


## 1. e19 vs d1_composite, 3 seeds (42/43/44), per task


### absa_hotel

| seed | e19 F1 | e19 epoch | d1_composite F1 | d1_composite epoch | paired diff (d1_composite - e19) |
|---|---|---|---|---|---|
| 42 | 73.63 (ok) | 29 | 73.3 (ok) | 13 | -0.33 |
| 43 | 72.99 (ok) | 23 | 73.41 (ok) | 18 | 0.42 |
| 44 | 72.5 (ok) | 19 | 73.6 (ok) | 29 | 1.1 |

**Mean +/- sd:** e19 = 73.04 +/- 0.57 (n=3); d1_composite = 73.44 +/- 0.15 (n=3)

**Mean paired diff (d1_composite - e19):** 0.4 pts (sd=0.72, n=3/3 seeds present in both arms)


#### Group rows (rare-pair / mixed), per seed

| seed | arm | rare_pair(<=5) missed% | rare_pair(<=5) wrong% | mixed missed% | mixed wrong% |
|---|---|---|---|---|---|
| 42 | e19 | 48.1 | 6.2 | 34.4 | 16.9 |
| 42 | d1_composite | 46.5 | 5.4 | 33.8 | 16.2 |
| 43 | e19 | 51.2 | 3.9 | 38.3 | 14.3 |
| 43 | d1_composite | 44.2 | 3.9 | 33.1 | 15.6 |
| 44 | e19 | 53.5 | 4.7 | 38.3 | 14.3 |
| 44 | d1_composite | 42.6 | 5.4 | 33.1 | 16.2 |


### visfd

| seed | e19 F1 | e19 epoch | d1_composite F1 | d1_composite epoch | paired diff (d1_composite - e19) |
|---|---|---|---|---|---|
| 42 | 85.08 (ok) | 26 | 84.9 (ok) | 30 | -0.18 |
| 43 | 85.14 (ok) | 30 | 85.21 (ok) | 25 | 0.07 |
| 44 | 84.97 (ok) | 27 | 84.69 (ok) | 28 | -0.28 |

**Mean +/- sd:** e19 = 85.06 +/- 0.09 (n=3); d1_composite = 84.93 +/- 0.26 (n=3)

**Mean paired diff (d1_composite - e19):** -0.13 pts (sd=0.18, n=3/3 seeds present in both arms)


#### Group rows (rare-pair / mixed), per seed

| seed | arm | rare_pair(<=5) missed% | rare_pair(<=5) wrong% | mixed missed% | mixed wrong% |
|---|---|---|---|---|---|
| 42 | e19 | - | - | 6.6 | 10.8 |
| 42 | d1_composite | - | - | 6.3 | 11.0 |
| 43 | e19 | - | - | 6.0 | 11.3 |
| 43 | d1_composite | - | - | 6.2 | 10.8 |
| 44 | e19 | - | - | 6.4 | 11.5 |
| 44 | d1_composite | - | - | 6.7 | 11.5 |


### absa_restaurant

| seed | e19 F1 | e19 epoch | d1_composite F1 | d1_composite epoch | paired diff (d1_composite - e19) |
|---|---|---|---|---|---|
| 42 | 74.51 (ok) | 22 | 75.46 (ok) | 21 | 0.95 |
| 43 | 74.07 (ok) | 16 | 75.31 (ok) | 14 | 1.24 |
| 44 | 73.72 (ok) | 26 | 75.0 (ok) | 18 | 1.28 |

**Mean +/- sd:** e19 = 74.1 +/- 0.4 (n=3); d1_composite = 75.26 +/- 0.23 (n=3)

**Mean paired diff (d1_composite - e19):** 1.16 pts (sd=0.18, n=3/3 seeds present in both arms)


#### Group rows (rare-pair / mixed), per seed

| seed | arm | rare_pair(<=5) missed% | rare_pair(<=5) wrong% | mixed missed% | mixed wrong% |
|---|---|---|---|---|---|
| 42 | e19 | 45.8 | 12.5 | 19.0 | 19.4 |
| 42 | d1_composite | 20.8 | 16.7 | 16.0 | 16.0 |
| 43 | e19 | 41.7 | 20.8 | 20.3 | 19.0 |
| 43 | d1_composite | 33.3 | 16.7 | 16.5 | 18.1 |
| 44 | e19 | 45.8 | 12.5 | 20.7 | 19.8 |
| 44 | d1_composite | 37.5 | 16.7 | 18.6 | 17.3 |


## 2. Restaurant: d1_composite vs d1_uniform (sampling ablation), seeds 42-44

d1_uniform = same composite count/pool/boundary-assert recipe as d1_composite, but aspect pairs are drawn uniformly over valid (a,b) pairs instead of ∝ 1/(1+pair_freq). See `scripts/e22_preflight.json` for the pair-use-count uniformity check (stdev comparison) that validated the sampler before launch.

| seed | d1_composite F1 | d1_composite epoch | d1_uniform F1 | d1_uniform epoch | paired diff (d1_uniform - d1_composite) |
|---|---|---|---|---|---|
| 42 | 75.46 (ok) | 21 | 76.44 (ok) | 29 | 0.98 |
| 43 | 75.31 (ok) | 14 | 75.18 (ok) | 28 | -0.13 |
| 44 | 75.0 (ok) | 18 | 75.6 (ok) | 11 | 0.6 |

**Mean +/- sd:** d1_composite = 75.26 +/- 0.23 (n=3); d1_uniform = 75.74 +/- 0.64 (n=3)

**Mean paired diff (d1_uniform - d1_composite):** 0.48 pts (sd=0.56, n=3/3 seeds present in both arms)


### Group rows (rare-pair / mixed), per seed

| seed | arm | rare_pair(<=5) missed% | rare_pair(<=5) wrong% | mixed missed% | mixed wrong% |
|---|---|---|---|---|---|
| 42 | d1_composite | 20.8 | 16.7 | 16.0 | 16.0 |
| 42 | d1_uniform | 29.2 | 16.7 | 16.5 | 16.0 |
| 43 | d1_composite | 33.3 | 16.7 | 16.5 | 18.1 |
| 43 | d1_uniform | 29.2 | 16.7 | 17.3 | 16.5 |
| 44 | d1_composite | 37.5 | 16.7 | 18.6 | 17.3 |
| 44 | d1_uniform | 37.5 | 16.7 | 16.9 | 15.2 |


## Sources

- e19: `artifacts/e19/<task>/seed_<seed>`.
- d1_composite / d1_uniform: `artifacts/e20/<arm>/<task>/seed_<seed>` (or `seed_<seed>__rerun1` for the pre-fix seed-42 trials -- resolved automatically by `eval_e20._artifact_dir`).
- Launch queue / logs: `scripts/e22_launch.sh`, `artifacts/e22/logs/`.
- Preflight: `artifacts/e22/preflight.json`.


## Root review (2026-09-14)

Vận hành: 12 trial, 4 GPU; hai trial d1_uniform seed 43/44 được root khởi chạy sớm làm worker thứ hai trên GPU2/3 (GPU chỉ dùng ~35%/trial), launcher gốc báo fail ở hai job này vì thư mục đã tồn tại; eval chạy lại sau khi tất cả hoàn tất. e19 seed 42/43/44 Hotel: 73,63/72,99/72,50; ViSFD: 85,08/85,14/84,97.

Tóm tắt 3 seed (validation pair F1, mean ± sd; chênh cặp mean ± sd):

| Task | e19 | d1_composite (ưu tiên cặp hiếm) | Δ d1 − e19 | d1_uniform | Δ uniform − d1 |
|---|---:|---:|---:|---:|---:|
| Restaurant | 74,10 ± 0,40 | 75,26 ± 0,23 | **+1,16 ± 0,18** | **75,74 ± 0,64** | +0,48 ± 0,57 |
| Hotel | 73,04 ± 0,57 | 73,44 ± 0,15 | +0,40 ± 0,72 | – | – |
| ViSFD | 85,06 ± 0,09 | 84,93 ± 0,26 | −0,13 ± 0,18 | – | – |

Đọc:

1. **Ghép câu nhãn hợp có tác dụng thật ở Restaurant** (+1,2, gấp 6 lần nhiễu), **yếu và nhiễu ở Hotel** (+0,4 ± 0,7, nhưng cặp hiếm missed giảm ở cả 3 seed: 48→46, 51→44, 54→43), **không tác dụng ở ViSFD** (đã nhiều câu đa aspect tự nhiên). Tác dụng tỉ lệ nghịch với mật độ câu đa aspect sẵn có trong train — đúng với chẩn đoán "lỗi nằm ở phân bố tổ hợp nhãn".
2. **Ưu tiên cặp hiếm không phải phần tạo gain**: chọn cặp đều (d1_uniform) bằng hoặc hơn (+0,5 ± 0,6). Phần "twist" định claim là mới bị bác bỏ; phương pháp còn lại là ghép câu nhãn hợp thuần — có tiền lệ (C3DA 2022, Chai 2023).
3. Hệ quả cho công bố: đóng góp phương pháp không đủ mới; đóng góp là **chẩn đoán có đối chứng + kết luận "chỉ dữ liệu tổ hợp mới sửa được"** + baseline mạnh. Hệ quả cho mục tiêu thắng BamiBERT: hệ thống chọn theo validation từng task (Restaurant d1_uniform 75,74; Hotel d1 73,44; ViSFD e19 85,06; Topic/Sentiment E2 teacher CE 82,17/85,92) đủ điều kiện mở test một lần với 3 seed ở các task có đủ seed.
