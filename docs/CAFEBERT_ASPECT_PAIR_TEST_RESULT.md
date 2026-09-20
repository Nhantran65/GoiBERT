# CafeBERT aspect-pair A — Restaurant test benchmark

Đánh giá theo yêu cầu user trên các checkpoint đã chọn bằng validation. Không training, không đổi mapping/recipe/epoch hoặc chọn lại theo test. A dùng mã “mục 01”…“mục 12”, giữ toàn câu, shared head4lớp.

## Kết quả test (pair micro F1, thang100)

| Seed | BamiBERT gốc | CafeBERT T | CafeBERT aspect-pair A | A−T (pp) |
|---|---:|---:|---:|---:|
|42|74.1493|77.3902|77.2881|-0.1020|
|43|—|76.8534|77.6404|+0.7870|
|Mean42/43|—|77.1218|77.4643|+0.3425|

Bami chỉ có baseline seed42 được dùng ở đây; không so mean hai seed A với một seed Bami như matched comparison. Ở seed42 A hơn Bami 3.1389pp, nhưng CafeBERT thường vốn đã hơn Bami 3.2409pp.

## Validation so với test

| Seed | A−T validation (pp) | A−T test (pp) |
|---|---:|---:|
|42|+0.7338|-0.1020|
|43|+1.6481|+0.7870|
|Mean|+1.1910|+0.3425|

Lợi ích validation không chuyển đầy đủ sang test: seed42 gần ngang/hơi giảm, seed43 tăng. Hai seed chưa chứng minh statistical significance hoặc tương đương; không tuyên bốSOTA.

## Thực thi và kiểm tra

- Manifest khóa SHA256, seed và epoch trước khi mở test: `artifacts/cafebert_aspect_pair_test/locked_manifest.json`.
- Inference mới: A42epoch23, A43epoch20, T43epoch23; GPU3, FP32masterweights/BF16autocast,SDPA,eval/inference-only. BatchA64pairs;T32reviews,maxlen256.
- Reuse T42epoch21 từ test SEA/Cafe benchmark và Bami42epoch29 từE14; metrics được tính lại từ predictions, không chạy lại hai checkpoint.
- Restaurant test1938reviews; A tạo23256pairs, đủ12slots rồi regroup.
- IDs/order/gold khớp test loader ở cả5bộpredictions. Root đếmTP/predicted/gold pairs độc lập và F1 khớp đến1e-12: `count_audit.json`.
- Code mới `scripts/evaluate_aspect_pair_test.py`; giữ nguyên code/checkpoints training. Test access log và predictions/metrics lưu riêng trong `artifacts/cafebert_aspect_pair_test/`.
- Bộ test đã được dùng trong nghiên cứu trước; không gọi virgin holdout. Runtime evaluator gồm load/checks, không phải benchmark latencyproduction.

## Quyết định theo mục tiêu production

A có mean testgain nhỏ (+0.34pp) và không dương cảhai seed, trong khi cần12encoderpasses/review thay vì1. Chưa đo latencyinference khớpđiềukiện, nhưng với ưu tiên tốc độ của user, kết quả này chưa đủ lý do chọn A thay CafeBERT thường. Giữ kết quả nghiên cứu, không mở thêm seed/wording/task để cứu nhánh.
