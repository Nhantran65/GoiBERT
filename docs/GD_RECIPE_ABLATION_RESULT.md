# Kết quả ablation công thức GD cho student S

Chạy ngày 2026-09-27–28 (Asia/Ho_Chi_Minh). **80/80 job hoàn tất, 0 lỗi**, toàn bộ đánh giá chỉ dùng validation. Không đọc text hoặc nhãn test. Plan khoa học và ngưỡng quyết định giữ nguyên trong `docs/GD_RECIPE_ABLATION_PLAN.md`.

## Kiểm tra trước khi chạy thật

| Kiểm tra | Số liệu | Trạng thái |
|---|---|---|
| Packing v1 | 50/50 chuỗi đầu khớp từng token với `iter_packed_sequences` gốc | PASS |
| Config export smoke A1 | Không có trường khác `models/goiturn-s-gd/config.json`; `goibert.modeling.build_classifier` nạp được, forward `(1, 3)` hữu hạn | PASS |
| Smoke GD A1, 200 bước | Loss 0.6128892302513123 → 0.28610315918922424; 0.15534950335510075 giây/bước; peak VRAM 6131448320 byte | PASS |
| Smoke GD A5, 200 bước | Loss 0.6226379871368408 → 0.28032392263412476; 0.14508776824921368 giây/bước; peak VRAM 6131448320 byte | PASS |
| Resume từ checkpoint bước 100 | Sai khác loss lớn nhất bước 101–110: A1 = 0.0; A5 = 0.0 | PASS |
| Dự báo 330000 bước GD | 14.240371140884236 GPU-giờ | PASS |
| Dự báo tổng gồm 15 GPU-giờ fine-tune | 29.240371140884236 / 50 GPU-giờ | PASS |
| Wrapper Restaurant seed 42 | ΔF1 epoch 1–5: 0.0, 0.0, 0.0, 0.0, 0.0 điểm; best lệch 0.0 điểm | **PASS** |
| Wrapper ViSpam seed 42 | ΔF1 epoch 1–5: 0.0, 0.03070880626245831, 0.0, 0.0, 0.2559719338504407 điểm; best lệch 0.0792121394088352 điểm | **FAIL thô; root chấp nhận** |

**Quyết định parity của root:** `root_accepted_parity: vispam` trong `preflight.json`. Root xác nhận sai khác ViSpam là nhiễu GPU không tất định: train loss khớp khoảng 1e-4 tương đối ở cả 30 epoch, sai khác cỡ 1e-5 xuất hiện từ epoch 1; F1 epoch 1, 3, 4, 6 và 7 giống hệt; lớp 1 chỉ có 26 mẫu validation nên một dự đoán đổi có thể làm macro-F1 biến động vài phần mười điểm. Best lệch 0.0792121394088352 điểm, dưới ngưỡng 1.0; Restaurant khớp tuyệt đối. **Ngưỡng epoch ≤ 0.05 điểm không được đổi**; ViSpam vẫn được ghi FAIL thô và root cho phép chạy thật.

Corpus v1 SHA-256: `fb0b4dd4f2a8bc6912bf13bb6a6d4f2af9d3271a6cfe49941eed0650f178c63f`. Corpus v2 chỉ được đọc sau `READY`; SHA-256 `mlm_mix.txt` đã đối chiếu và khớp nội dung `READY`: `3ba6537ced296669373f6d3e652a8d5a75e152e97865ff7333f75c43896295c1`.

## General distillation

Giây/bước lấy từ timer train trong manifest; GPU-giờ tính từ thời gian job trong launcher, gồm khởi tạo và export. Loss cuối là bước cuối, không làm tròn.

| Arm | Bước | Giây/bước | GPU-giờ GD | Loss cuối | Corpus SHA-256 | Teacher | GPU |
|---|---:|---:|---:|---:|---|---|---:|
| A1 | 30000 | 0.14024955748369297 | 1.179176624417305 | 0.08215554058551788 | `fb0b4dd4f2a8bc6912bf13bb6a6d4f2af9d3271a6cfe49941eed0650f178c63f` | `goibert-b-dapt` | 1 |
| A2 | 30000 | 0.12624548816401512 | 1.05674361579948 | 0.08113518357276917 | `3ba6537ced296669373f6d3e652a8d5a75e152e97865ff7333f75c43896295c1` | `goibert-b-dapt` | 3 |
| A3 | 120000 | 0.10795157487692 | 3.6098319767581093 | 0.0734415352344513 | `fb0b4dd4f2a8bc6912bf13bb6a6d4f2af9d3271a6cfe49941eed0650f178c63f` | `goibert-b-dapt` | 0 |
| A4 | 120000 | 0.14791070796448427 | 4.933572316037284 | 0.07188256829977036 | `3ba6537ced296669373f6d3e652a8d5a75e152e97865ff7333f75c43896295c1` | `goibert-b-dapt` | 2 |
| A5 | 30000 | 0.15065849966562042 | 1.2681449605358972 | 0.08137308806180954 | `fb0b4dd4f2a8bc6912bf13bb6a6d4f2af9d3271a6cfe49941eed0650f178c63f` | `bamibert-base-original` | 2 |

Đường cong loss total/hidden/attention cùng giây/bước và peak VRAM được ghi mỗi 100 bước ở `artifacts/gd_ablation/backbones/<arm>/loss_curve.jsonl`, đồng thời nằm trong `summary.json` → `gd_loss_curves`. Mỗi arm giữ đúng hai checkpoint gần nhất cộng backbone `final/`.

## Fine-tune validation: từng task và seed

Đơn vị là **điểm F1**. Sentiment, ViSpam và ViNLI dùng macro F1; PhoNER dùng entity F1; Restaurant dùng pair micro F1. Mean là trung bình ba seed; range là min–max. Các metrics dưới đây được tính lại từ `validation_predictions.json` và khớp `best_validation_metrics.json`/`metadata.json` với sai số dưới 1e-9.

| Arm | Task | Seed 42 | Seed 43 | Seed 44 | Mean | Range |
|---|---|---:|---:|---:|---:|---|
| R0 | Sentiment | 84.49202979103964 | 84.46179643561166 | 84.07607338121105 | 84.34329986928745 | 84.07607338121105–84.49202979103964 |
| R0 | Restaurant | 73.50674373795762 | 73.40324118207818 | 73.42623738587217 | 73.44540743530266 | 73.40324118207818–73.50674373795762 |
| R0 | ViSpam | 75.36398677404834 | 75.53864772250982 | 75.40563322920481 | 75.436089241921 | 75.36398677404834–75.53864772250982 |
| R0 | ViNLI | 71.20172136879557 | 70.64210009668068 | 71.35713191895785 | 71.06698446147803 | 70.64210009668068–71.35713191895785 |
| R0 | PhoNER | 93.32188387608164 | 93.19219318557724 | 93.0950016550811 | 93.20302623891332 | 93.0950016550811–93.32188387608164 |
| A1 | Sentiment | 84.14660486800526 | 84.22090280319937 | 84.6959025099223 | 84.35447006037565 | 84.14660486800526–84.6959025099223 |
| A1 | Restaurant | 73.18944844124701 | 73.47931873479318 | 73.14725697786332 | 73.27200805130117 | 73.14725697786332–73.47931873479318 |
| A1 | ViSpam | 78.19203413805805 | 75.88793768103032 | 75.9563837830787 | 76.67878520072236 | 75.88793768103032–78.19203413805805 |
| A1 | ViNLI | 70.02326612220789 | 70.5399767214564 | 69.97030772762426 | 70.17785019042951 | 69.97030772762426–70.5399767214564 |
| A1 | PhoNER | 93.18392786102639 | 92.99025134292725 | 93.04578633045786 | 93.07332184480383 | 92.99025134292725–93.18392786102639 |
| A2 | Sentiment | 84.55423016255702 | 84.52025545783573 | 83.53876726519563 | 84.20441762852946 | 83.53876726519563–84.55423016255702 |
| A2 | Restaurant | 73.33333333333333 | 72.7536231884058 | 73.24350336862368 | 73.1101532967876 | 72.7536231884058–73.33333333333333 |
| A2 | ViSpam | 75.33034076267901 | 77.00941890419887 | 75.29927751267668 | 75.87967905985153 | 75.29927751267668–77.00941890419887 |
| A2 | ViNLI | 72.20654248168114 | 72.3955904585112 | 71.9174648452236 | 72.17319926180531 | 71.9174648452236–72.3955904585112 |
| A2 | PhoNER | 93.10734463276836 | 93.24396071144146 | 93.2227016262861 | 93.19133565683197 | 93.10734463276836–93.24396071144146 |
| A3 | Sentiment | 85.48803433968581 | 85.87781380630543 | 85.04699601473794 | 85.4709480535764 | 85.04699601473794–85.87781380630543 |
| A3 | Restaurant | 74.92795389048992 | 73.69942196531792 | 75.01194457716197 | 74.54644014432327 | 73.69942196531792–75.01194457716197 |
| A3 | ViSpam | 76.74674064492906 | 76.54206980882027 | 79.35258648564354 | 77.54713231313096 | 76.54206980882027–79.35258648564354 |
| A3 | ViNLI | 74.26054936944747 | 74.53909564053292 | 74.33541069524178 | 74.37835190174073 | 74.26054936944747–74.53909564053292 |
| A3 | PhoNER | 93.94663480684986 | 94.16771962492518 | 94.26387504154205 | 94.12607649110569 | 93.94663480684986–94.26387504154205 |
| A4 | Sentiment | 84.24688052999323 | 84.83731793070146 | 85.23144252979081 | 84.77188033016184 | 84.24688052999323–85.23144252979081 |
| A4 | Restaurant | 74.33206106870229 | 74.34304825609173 | 74.6311280342694 | 74.43541245302114 | 74.33206106870229–74.6311280342694 |
| A4 | ViSpam | 79.06882909276986 | 78.00855190355206 | 78.85585046916165 | 78.64441048849453 | 78.00855190355206–79.06882909276986 |
| A4 | ViNLI | 75.0073438563665 | 75.96116775104821 | 75.83909827059543 | 75.60253662600338 | 75.0073438563665–75.96116775104821 |
| A4 | PhoNER | 94.00278940027894 | 94.06356186288227 | 94.18326693227091 | 94.08320606514404 | 94.00278940027894–94.18326693227091 |
| A5 | Sentiment | 83.37456820511129 | 83.83627828533483 | 82.72288564548627 | 83.3112440453108 | 82.72288564548627–83.83627828533483 |
| A5 | Restaurant | 73.88167388167388 | 73.10478029937228 | 73.33012974531475 | 73.43886130878697 | 73.10478029937228–73.88167388167388 |
| A5 | ViSpam | 77.24434092686714 | 76.00148624362582 | 76.47431346717518 | 76.57338021255605 | 76.00148624362582–77.24434092686714 |
| A5 | ViNLI | 71.33151437539416 | 71.76053228199926 | 71.53456953569219 | 71.5422053976952 | 71.33151437539416–71.76053228199926 |
| A5 | PhoNER | 93.47348812454261 | 93.35106382978722 | 93.25984461119596 | 93.36146552184194 | 93.25984461119596–93.47348812454261 |

## Điểm P và quy tắc quyết định đã khóa

| Arm | P | P bỏ ViNLI |
|---|---:|---:|
| R0 | 79.49896144938049 | 81.60695569635611 |
| A1 | 79.5112870695265 | 81.84464628930075 |
| A2 | 79.71175698076117 | 81.59639641050013 |
| A3 | 81.21378978077541 | 82.92264925053408 |
| A4 | 81.50748919256499 | 82.98372733420538 |
| A5 | 79.64543129723819 | 81.67123777212393 |

| So sánh | ΔP, điểm |
|---|---:|
| A2-A1 | 0.20046991123466285 |
| A3-A1 | 1.7025027112489113 |
| A4-A1 | 1.9962021230384863 |
| A5-A1 | 0.13414422771168688 |
| A1-R0 | 0.012325620146015126 |

Đầu vào của từng điều kiện; **chỉ ghi số và PASS/FAIL, không chọn công thức cuối**:

| Điều kiện | Số liệu | Kết quả |
|---|---|---|
| Kiểm port \|P(A1)−P(R0)\| ≤ 1,0 | 0.012325620146015126 | PASS |
| A2: ΔP ≥ +0,5 | 0.20046991123466285 | FAIL |
| A2: không task nào giảm quá 1,0 | Sentiment -0.15005243184619133, Restaurant -0.1618547545135698, ViSpam -0.7991061408708333, ViNLI 1.9953490713757986, PhoNER 0.11801381202813843 | PASS |
| A2: đồng thời hai điều kiện trên | ΔP 0.20046991123466285 | FAIL |
| A3: ΔP ≥ +0,5 | 1.7025027112489113 | PASS |
| A3: không task nào giảm quá 1,0 | Sentiment 1.1164779932007463, Restaurant 1.2744320930220994, ViSpam 0.8683471124085997, ViNLI 4.200501711311219, PhoNER 1.0527546463018638 | PASS |
| A3: đồng thời hai điều kiện trên | ΔP 1.7025027112489113 | PASS |
| A4: ΔP ≥ +0,5 | 1.9962021230384863 | PASS |
| A4: không task nào giảm quá 1,0 | Sentiment 0.41741026978618834, Restaurant 1.1634044017199727, ViSpam 1.9656252877721698, ViNLI 5.424686435573875, PhoNER 1.0098842203402114 | PASS |
| A4: đồng thời hai điều kiện trên | ΔP 1.9962021230384863 | PASS |
| A5: ΔP ≥ +0,5 | 0.13414422771168688 | FAIL |
| A5: không task nào giảm quá 1,0 | Sentiment -1.043226015064846, Restaurant 0.1668532574857977, ViSpam -0.10540498816631327, ViNLI 1.3643552072656888, PhoNER 0.28814367703810717 | FAIL |
| A5: đồng thời hai điều kiện trên | ΔP 0.13414422771168688 | FAIL |
| DAPT: P(A5) ≥ P(A1)−0,3 | A5−A1 = 0.13414422771168688 | PASS |

## Ngân sách, GPU và sự cố

Loạt chính: **28.816040821870168 GPU-giờ** = tổng 80 job trong `launcher_jobs.json`, dưới trần 50. Kiểm tra trước chạy ghi nhận thêm 0.2910175677150902 GPU-giờ theo timer của smoke/resume/parity; tổng cộng theo các timer ghi nhận là **29.10705838958526 GPU-giờ**. Timer GD không gồm toàn bộ thời gian nạp model trước khi bắt đầu train; số chính của launcher bao gồm thời gian job.

| GPU | GPU-giờ loạt chính |
|---:|---:|
| 0 | 7.319465146925714 |
| 1 | 7.55849262005753 |
| 2 | 7.625584522353279 |
| 3 | 6.312498532533646 |

| Arm | GPU-giờ fine-tune |
|---|---:|
| A1 | 3.654107520911429 |
| A2 | 3.3037349713510933 |
| A3 | 3.275974493026733 |
| A4 | 3.2481175834602785 |
| A5 | 3.2866367595725587 |

Artifact `artifacts/gd_ablation/` chiếm khoảng 7,0 GB khi kết thúc, dưới giới hạn 60 GB. Không có job chính thất bại; `launcher_summary.json` có `failed: []`, `pending: []`. Không kill, dừng hoặc renice process của người khác.

Sự cố tiền chạy: lần gọi smoke đầu tiên gặp lỗi `torch.cuda.reset_peak_memory_stats` trước khi khởi tạo thiết bị; sửa trong mã mới và chạy lại smoke thành công. Sai khác ViSpam parity vượt ngưỡng epoch thô; root review và chấp nhận như ghi ở trên. Các arm thật dùng bf16 autocast teacher/student, loss fp32, attention eager thay cho fp32 trên MI250X của S gốc theo plan; A1−R0 đo độ lệch port này.

Giới hạn: mỗi arm chỉ có **một GD seed**; không quét lại LR fine-tune; kết quả **validation-only**. Corpus v2 thay đổi nhiều thành phần như một gói, nên không tách đóng góp riêng. Không suy ra kết quả test.

## Lệnh tái lập

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/gd_ablation/gd.py --arm A1 --corpus data/corpus_v1/mlm_mix.txt --teacher models/goibert-b-dapt --steps 200 --smoke-steps 200 --out artifacts/gd_ablation/smoke/A1
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/gd_ablation/gd.py --arm A5 --corpus data/corpus_v1/mlm_mix.txt --teacher models/bamibert-base-original --steps 200 --smoke-steps 200 --out artifacts/gd_ablation/smoke/A5
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/gd_ablation/finetune.py --backbone models/goiturn-s-gd --task absa_restaurant --seed 42 --out artifacts/gd_ablation/parity/absa_restaurant/seed_42
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/gd_ablation/finetune.py --backbone models/goiturn-s-gd --task vispam --seed 42 --out artifacts/gd_ablation/parity/vispam/seed_42
setsid nohup env PYTHONPATH=src HF_HUB_OFFLINE=1 GDA_GPUS=0,1,2,3 .venv-e0-cuda/bin/python scripts/gd_ablation/launcher.py > artifacts/gd_ablation/launcher.log 2>&1 < /dev/null &
PYTHONPATH=src .venv-e0-cuda/bin/python scripts/gd_ablation/summarize.py
```

Để chạy lại từ đầu, dùng một thư mục output mới hoặc chuyển run dở dang sang `artifacts/gd_ablation/aborted/`; GD hỗ trợ `--resume` với đúng vị trí chuỗi packed và trạng thái RNG/optimizer/scheduler. Không sửa artifacts cũ hoặc `data/corpus_v2/`.

## Root review (2026-09-28)

- **Tính lại độc lập** từ `validation_predictions.json` của 90 run (R0 + A1–A5, 5 task × 3 seed) bằng evaluator riêng của root: khớp mọi mean và P tới 3 chữ số thập phân.
- **Manifest backbone:** A1–A5 có số bước, teacher và sha corpus (v1 `fb0b4dd4…`, v2 `3ba6537c…`) đúng thiết kế.
- **Lỗi nhãn, không ảnh hưởng số:** `metadata.json` và `resolved_config.json` của các run ViSpam/ViNLI/PhoNER ghi `"backbone": "goiturn-s-gd"`. Chuỗi này bị hard-code từ trainer cũ. Model thật được nạp qua `GDA_BACKBONE` (`bench_student.py:21,94`), và kết quả khác nhau giữa các arm xác nhận điều đó.
- **Parity ViSpam** được root chấp nhận là nhiễu GPU (xem lượt review trước).
- **Quyết định theo quy tắc đã khóa:**
  - GD dài giúp: A3 +1,70, A4 +2,00.
  - Corpus v2 ở 30K không đạt: A2 +0,20.
  - Corpus v2 ở 120K: A4 − A3 = +0,29, nằm trong mức nhiễu.
  - DAPT không cần: A5 − A1 = +0,13.
  - Arm P cao nhất trong nhóm đạt là A4.
