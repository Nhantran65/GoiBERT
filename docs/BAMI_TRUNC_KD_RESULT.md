# BamiBERT cắt lớp + KD từ CafeBERT — kết quả pilot

Ngày chạy: 2026-09-26. Đặc tả: [BAMI_TRUNC_KD_PLAN.md](BAMI_TRUNC_KD_PLAN.md). Chỉ dùng train và validation; mọi lời gọi nạp task đặt `include_test=False`. Không mở test.

## Trạng thái và quyết định

**Hoàn tất 32/32 run.** Lượt launcher đầu dừng sau hai run tái tạo L12-CE seed 42: Sentiment lệch E14 **+0,3233172330774581 điểm F1**, vượt ngưỡng 0,3 điểm đã khóa; Restaurant khớp chính xác. Root đã xem xét đường cong và config, **chấp nhận đây là tái tạo đúng E14 và cho phép chạy tiếp 30 run**. Lượt launcher tiếp theo dùng cờ `--root-accepted-reproduction`; cờ này chỉ bỏ chặn gate tái tạo đã được root xét, không sửa hai run cũ hoặc recipe train.

**Gate pilot FAIL.** Không có độ sâu k ∈ {10, 8} nào đạt mean Lk-KD ≥ mean Bami − 0,005 ở cả hai task. Có 32/32 run hoàn tất, 0 run lỗi và 0 postrun check FAIL. Chi tiết số liệu trong `artifacts/bami_trunc_kd/summary.json` và `gate.json`.

## Số tham số

Đếm trực tiếp từ model dựng bằng `build_bamibert_classifier` rồi giữ các lớp encoder dưới cùng. Forward L8/L10 đã chạy được và `config.num_hidden_layers` khớp số lớp thực tế.

| Độ sâu | Encoder | Head Sentiment | Tổng Sentiment | Head Restaurant | Tổng Restaurant |
|---|---:|---:|---:|---:|---:|
| L12 | 102.360.576 | 592.899 | 102.953.475 | 627.504 | 102.988.080 |
| L10 | 88.184.832 | 592.899 | 88.777.731 | 627.504 | 88.812.336 |
| L8 | 74.009.088 | 592.899 | 74.601.987 | 627.504 | 74.636.592 |

## Bảng validation F1

Giá trị là tỉ lệ F1, không phải phần trăm. Bami là ba mốc có sẵn trong plan. Mọi F1 hiển thị dưới đây được tính lại từ `validation_predictions.json` và kiểm khớp metadata.

| Arm | Task | Seed 42 | Seed 43 | Seed 44 | Mean ba seed | Range ba seed |
|---|---|---:|---:|---:|---:|---:|
| Bami L12-CE, mốc cũ | Sentiment | 0,8632522837351507 | 0,8611572017056693 | 0,8803856639708009 | 0,8682650498038736 | 0,8611572017056693–0,8803856639708009 |
| Bami L12-CE, mốc cũ | Restaurant | 0,7474178403755869 | 0,7183098591549296 | 0,7350993377483444 | 0,733609012426287 | 0,7183098591549296–0,7474178403755869 |
| L12-CE, tái tạo | Sentiment | 0,8664854560659253 | — | — | — | — |
| L12-CE, tái tạo | Restaurant | 0,7474178403755869 | — | — | — | — |
| L12-KD | Sentiment | 0,8561125213959446 | 0,8462296906549471 | 0,8505710935048812 | 0,8509711018519243 | 0,8462296906549471–0,8561125213959446 |
| L12-KD | Restaurant | 0,759110269758637 | 0,7431018078020932 | 0,7503526093088858 | 0,7508548956232053 | 0,7431018078020932–0,759110269758637 |
| L10-CE | Sentiment | 0,8596185737619008 | 0,8463208502528028 | 0,8562036691694744 | 0,8540476977280593 | 0,8463208502528028–0,8596185737619008 |
| L10-CE | Restaurant | 0,7220657276995305 | 0,7350023618327822 | 0,7295774647887324 | 0,7288818514403483 | 0,7220657276995305–0,7350023618327822 |
| L10-KD | Sentiment | 0,8496174542563999 | 0,8521853904041982 | 0,8449675751657016 | 0,8489234732754332 | 0,8449675751657016–0,8521853904041982 |
| L10-KD | Restaurant | 0,7416267942583732 | 0,7501185395922239 | 0,7505938242280285 | 0,7474463860262085 | 0,7416267942583732–0,7505938242280285 |
| L8-CE | Sentiment | 0,838721356291532 | 0,8438772666642217 | 0,8375926090069482 | 0,8400637439875673 | 0,8375926090069482–0,8438772666642217 |
| L8-CE | Restaurant | 0,7053903345724907 | 0,7110070257611241 | 0,6977629700142789 | 0,7047201101159645 | 0,6977629700142789–0,7110070257611241 |
| L8-KD | Sentiment | 0,8404771151712053 | 0,8517036432485862 | 0,8491978308243554 | 0,8471261964147155 | 0,8404771151712053–0,8517036432485862 |
| L8-KD | Restaurant | 0,7252017085904129 | 0,7256038647342995 | 0,7070515854235684 | 0,7192857195827602 | 0,7070515854235684–0,7256038647342995 |

Tái tạo Sentiment chọn best epoch 16, E14 seed 42 chọn epoch 8. F1 theo epoch và train loss trùng chính xác từ epoch 1 đến 10; đường cong bắt đầu khác ở epoch 11. Tái tạo Restaurant chọn epoch 29 và toàn bộ 30 epoch trùng E14. Root quy sai khác Sentiment về backward SDPA memory-efficient không tất định; PyTorch có UserWarning tương ứng. Config khớp micro 32, accum 1, LR 2e-5, warmup 1074 và bf16. `metadata.attention_implementation` của hai run tái tạo chưa có trường này, nhưng `train.py` thực tế gọi `build_bamibert_classifier(..., attention_implementation="sdpa")`. Trường này đã được thêm vào metadata của các run mới, không đổi hành vi train. Cả hai run tái tạo có `reload_matches_best`, `prediction_file_matches_best`, `reload_predictions_identical`, `ids_stable_and_complete`, `finite_train_loss` và `teacher_train_coverage` đều `true`.

Đối chiếu đường cong Sentiment, số gốc từ `epoch_curves.json`:

| Epoch | E14 train loss | Mới train loss | E14 F1 | Mới F1 |
|---:|---:|---:|---:|---:|
| 1 | 0,7171357787679963 | 0,7171357787679963 | 0,6256842985279083 | 0,6256842985279083 |
| 2 | 0,28268993503885054 | 0,28268993503885054 | 0,7847162102007962 | 0,7847162102007962 |
| 3 | 0,22122168077722273 | 0,22122168077722273 | 0,8108958296608505 | 0,8108958296608505 |
| 4 | 0,18275142905383693 | 0,18275142905383693 | 0,828160826481792 | 0,828160826481792 |
| 5 | 0,14447411834807608 | 0,14447411834807608 | 0,8376178041821518 | 0,8376178041821518 |
| 6 | 0,1204764260816716 | 0,1204764260816716 | 0,8387348762852888 | 0,8387348762852888 |
| 7 | 0,09673194311012763 | 0,09673194311012763 | 0,8322316237244118 | 0,8322316237244118 |
| 8 | 0,07816866636484908 | 0,07816866636484908 | 0,8632522837351507 | 0,8632522837351507 |
| 9 | 0,07258254000628535 | 0,07258254000628535 | 0,8421378958454744 | 0,8421378958454744 |
| 10 | 0,05745406144975301 | 0,05745406144975301 | 0,8531819528765224 | 0,8531819528765224 |
| 11 | 0,053331019726897966 | 0,052762114352735895 | 0,8482769166384986 | 0,8380368025308066 |
| 12 | 0,04427758712140896 | 0,04060257384682336 | 0,8465079566567532 | 0,8395823916481642 |
| 13 | 0,02922164640555172 | 0,030347337827944576 | 0,8524066502158826 | 0,860722386130509 |

## Delta và gate

Tái tạo seed 42 trừ E14 seed 42: Sentiment **+0,003233172330774581 F1**, tương đương **+0,3233172330774581 điểm**; Restaurant **0**. Các delta dưới đây là **chênh lệch mean ba seed theo tỉ lệ F1**; nhân 100 để đổi sang điểm F1.

| So sánh | Sentiment | Restaurant |
|---|---:|---:|
| L12-KD − Bami | −0,017293947951949384 | +0,017245883196918332 |
| L10-KD − L10-CE | −0,005124224452626169 | +0,01856453458586016 |
| L10-CE − Bami | −0,014217352075814316 | −0,0047271609859386166 |
| L10-KD − Bami | −0,019341576528440485 | +0,013837373599921543 |
| L8-KD − L8-CE | +0,0070624524271482025 | +0,014565609466795748 |
| L8-CE − Bami | −0,02820130581630631 | −0,028888902310322484 |
| L8-KD − Bami | −0,021138853389158108 | −0,014323292843526736 |

| Độ sâu | Task | Mean KD | Ngưỡng Bami − 0,005 | Đạt task? | Đạt cả hai task? |
|---|---|---:|---:|---|---|
| L10 | Sentiment | 0,8489234732754332 | 0,8632650498038736 | Không | **Không** |
| L10 | Restaurant | 0,7474463860262085 | 0,728609012426287 | Có | **Không** |
| L8 | Sentiment | 0,8471261964147155 | 0,8632650498038736 | Không | **Không** |
| L8 | Restaurant | 0,7192857195827602 | 0,728609012426287 | Không | **Không** |

KD giúp Restaurant ở L10 và L8, và giúp Sentiment ở L8, nhưng làm mean Sentiment L10 giảm. L12-KD cũng cải thiện Restaurant nhưng giảm Sentiment so với Bami ba seed. Pilot không đề xuất mở rộng 5 task theo gate đã chốt.

## Smoke test và nhãn mềm

Nhãn mềm Sentiment phủ **11.426/11.426** ID train, logits rộng **3**; Restaurant phủ **7.028/7.028**, logits rộng **48** với 12 slot × 4 lớp. L8 và L10 có số lớp thực tế đúng 8 và 10, forward thành công. KD 1 epoch, 2 batch trên Sentiment L8 và Restaurant L10 ở hai thư mục `/tmp/btk_smoke_*_20260926`: loss hữu hạn và mọi postrun check PASS. F1 smoke không dùng cho gate.

## GPU, sự cố và giới hạn

Launcher ghi **5,49460031138526 GPU-giờ** cộng dồn cả lượt tái tạo và 30 run sau, tính từ lúc giao tới khi thu worker (bao gồm thời gian chờ vòng kiểm tra). GPU thực sự dùng: **0, 1, 2, 3**; số run tương ứng **10, 9, 6, 7**. Ngân sách 8 GPU-giờ không bị vượt. Không có worker lỗi hay postrun check FAIL. Sự cố duy nhất là **gate số học tái tạo Sentiment FAIL** trong lượt đầu; root xét bằng chứng curve/config và cho phép chạy tiếp. Hai run tái tạo cũ được giữ nguyên. Không tác động tới process của người khác.

Giới hạn của thiết kế: chỉ có validation, chưa đánh giá test; LR của bản cắt dùng LR Bami gốc, không quét; KD chỉ thử một cấu hình α = 0,5 / T = 2. Chênh lệch giữa các seed cho thấy kết quả trung bình ba seed vẫn có độ biến thiên và không đại diện cho mọi cách cắt lớp hoặc mọi cấu hình KD.

## Lệnh tái lập

Chạy trong root của repo, trên máy có CUDA và checkpoint/dữ liệu cục bộ. Hai smoke test mẫu:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python -B scripts/bami_trunc_kd/train.py --task vsfc_sentiment --layers 8 --arm kd --seed 42 --epochs 1 --max-train-batches 2 --out /tmp/btk_smoke_sentiment_20260926
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python -B scripts/bami_trunc_kd/train.py --task absa_restaurant --layers 10 --arm kd --seed 42 --epochs 1 --max-train-batches 2 --out /tmp/btk_smoke_restaurant_20260926
```

Launcher chuẩn (chỉ giao job mới trên GPU có ít nhất 12.000 MiB trống, một job của mình mỗi GPU, tự áp gate tái tạo và ngân sách):

```bash
setsid nohup env PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 BTK_GPUS=0,1,2,3 .venv-e0-cuda/bin/python -B scripts/bami_trunc_kd/launcher.py --root-accepted-reproduction >> artifacts/bami_trunc_kd/launcher.log 2>&1 < /dev/null &
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv-e0-cuda/bin/python -B scripts/bami_trunc_kd/summarize.py
```

Run hoàn tất được bỏ qua khi chạy lại. Thư mục run dở dang cần được lưu sang `artifacts/bami_trunc_kd/aborted/` trước khi chạy lại; lần chạy này không có thư mục dở dang. Cờ `--root-accepted-reproduction` phản ánh đúng quyết định review của root trong lần chạy này; nếu tái lập từ đầu, cần xem lại gate tái tạo và quyết định tương ứng.
