# CafeBERT aspect-pair — pilot Restaurant

## 1. Mục tiêu và scope

Kiểm tra việc đưa aspect vào đầu vào encoder có giúp CafeBERT phân loại Restaurant tốt hơn hay không. **Giữ nguyên toàn câu và numeric gold**, không chia mệnh đề, không gán nhãn lại.

Hai lượt fine-tune mới, một task `absa_restaurant`, seed42, một LR. Không DAPT, KD, NC, LLRD, composite, quét prompt/LR, thêm seed hay test. Bản này chỉ là kế hoạch; chưa triển khai hoặc launch training.

## 2. Hai arm và đối chứng đã có

| Arm | Cặp input | Head |
|---|---|---|
| A — aspect ID | `(review nguyên văn, "mục 01")`, tương tự cho12mã | Shared classifier4lớp |
| B — aspect description | `(review nguyên văn, "không gian nhà hàng")`, tương tự cho12aspect | Shared classifier4lớp |
| Reference đã có | CafeBERT đọc review, xuất48logits cho12slots | Không chạy lại |

Reference: `artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42/`, validation pair microF1 **77.7197149644**, LR2e-5,30epochs,seed42. BamiBERT CE reference **74.7417840376**. Chỉ dùng baseline validation; không chọn theo test.

A/B phải cùng sourceencoder, head-init, review order, số lượt cập nhật, loss, optimizer và số pair được học. B−A kiểm lợi ích của mô tả có ngữ nghĩa; A/B−reference đo lợi ích của cả cách đặt bài toán. **So với reference không cô lập compute/head**: hai arm mới đọc review12lần và dùng head chung4lớp. Không nhận claim chỉ tênaspect tạo gain nếuA cũng tăng tương tự.

## 3. Bảng mapping cố định trước training

Thứ tự phải khớp `RESTAURANT_ASPECT_NAMES` của loader:

| Slot (0-based) | Aspect gốc | Arm A | Arm B |
|---:|---|---|---|
|0|AMBIENCE#GENERAL|mục 01|không gian nhà hàng|
|1|DRINKS#PRICES|mục 02|giá đồ uống|
|2|DRINKS#QUALITY|mục 03|chất lượng đồ uống|
|3|DRINKS#STYLE&OPTIONS|mục 04|cách trình bày và sự lựa chọn đồ uống|
|4|FOOD#PRICES|mục 05|giá món ăn|
|5|FOOD#QUALITY|mục 06|chất lượng món ăn|
|6|FOOD#STYLE&OPTIONS|mục 07|cách trình bày và sự lựa chọn món ăn|
|7|LOCATION#GENERAL|mục 08|vị trí nhà hàng|
|8|RESTAURANT#GENERAL|mục 09|đánh giá chung về nhà hàng|
|9|RESTAURANT#MISCELLANEOUS|mục 10|các khía cạnh khác của nhà hàng|
|10|RESTAURANT#PRICES|mục 11|mức giá chung của nhà hàng|
|11|SERVICE#GENERAL|mục 12|dịch vụ phục vụ|

Đây là cách diễn đạt đề xuất, chưa được chứng minh tối ưu. Không thêm từ gợi polarity hoặc sửa wording sau khi xem validation. Không thêm token mới vào vocabulary. Mã aspect A là định danh có thể học, không phải đối chứng hoàn toàn không có thông tin. Khác biệt độ dài prompt giữa A/B cần ghi nhận.

## 4. Chuyển dữ liệu và gộp dự đoán

- Train7028reviews → **84336pairs**; validation771reviews → **9252pairs**. Không load test.
- Mỗi review luôn tạo đủ12pairs, kể cả aspect absent. Pairkey=`(original_id, slot_index)`; target là đúng phần tử `gold[slot_index]`.
- Mapping labels: **0absent,1positive,2negative,3neutral**. Không dùng tên polarity sai trong metadata lịch sử để recode.
- Dùng tokenizer pair API chính thức của XLM-R: `tokenizer(review, aspect_text, ...)`; không tự ghép separator bằng chuỗi. Tổng maxlen256, `truncation="only_first"` nếu cần. Đếm truncation; nếu review nào bị cắt trong hai arm, báo trước fullrun vì yêu cầu giữ nguyên câu cần được xem xét lại.
- Khi validation, argmax4classes cho từngpair; gộp về đúng12slots theo ID vàslot, chấm cùng evaluator pairmicroF1 hiện tại. Không chỉ chấm polarity trên gold-present aspects, không dùng gold chọn aspect ở inference.
- Preflight phải chứng minh: expand→regroup gold bằng chính xác vector gốc, IDs/order/counts đúng, không thiếu hoặc trùngslot. Đây là bảo đảm chuyển đổi dữ liệu; **không bảo đảm model dự đoán đúng**.

## 5. Recipe huấn luyện

- Khởi tạo cảA/B từ **CafeBERT pretrained gốc**, không từ checkpoint T đã fine-tune.
- Model `uitnlp/CafeBERT@af76fcf2a04096b2b54b348a3e4eb48253c93c5d`; tokenizer tương thích của baseline `FacebookAI/xlm-roberta-large@c23d21b0620b635a76227c604d44e43a9f0ee389`. Tái dùng localcache và xác minh source.
- Encoder24layers/hidden1024; fresh head CLS→dropout.1→dense1024→tanh→dropout.1→4logits. Tất cả weights trainable, khôngMLMhead.
- Seed42, LR2e-5,30epochs, AdamW betas(.9,.999),eps1e-8,wd.01 (bias/norm khôngdecay), warmup10%+linear decay,clip1, FP32masterweights+BF16autocast,SDPA.
- **Effective batch =32reviews =384pairs**, không phải32pairs. Một epoch là một lượt qua7028reviews và toàn bộ12aspect mỗi review.
- Shuffle review IDs như baseline; mỗi group≤32reviews expand đủ12slots, chia thành microbatch16pairs và accumulate trong group; mỗi microbatch contribution là `CE_mean * n_pairs_micro / n_pairs_group`. Chỉ optimizerstep/clip/scheduler một lần sau toàn group. Groupcuối có20reviews=240pairs, chuẩn hóa theo240.
- Như vậy mỗi epoch220optimizersteps, tổng6600, warmup660; mỗi review-slot xuất hiện đúng1lần/epoch. Không vô tình tạo sốsteps gấp12 rồi gọi cùngrecipe.
- Dùng cùng microbatch A/B; nếu cần đổi theo profile, ghi lại trước launch. Không cần giữ toàn bộ384pair activations trongVRAM cùnglúc.
- A/B dùng cùng review-order và head-init; ghi hashes. Không hứa dropout masks hay kernel bitwise đồng nhất khi chiều dàiinput khác.
- Validate sau mỗi epoch; bestcheckpoint chọn bằng fullprecision validation pairF1; tie chọn epoch sớm hơn. Train đủ30epochs trừ lỗi vận hành/numerics.

## 6. Triển khai tối thiểu và lưu kết quả

- Workspace `/home/kiettqa/GoiBERT`, môi trường `.venv-e0-cuda`. Đọc `AGENTS.md`; không tự tạo thêm agent.
- Tái dùng loader/evaluator `src/goibert/train.py`, head/source builder của `scripts/cdbk_pilot/common.py` và `src/goibert_e14/modeling.py`. Builder mới phải xuất4logits, không giữ head48logits rồi chọn output theo gold.
- Source mới `scripts/cafebert_aspect_pair/`, artifacts `artifacts/cafebert_aspect_pair/`; không sửa code/artifacts/gold cũ.
- Tiny preflight: mapping/regroup, mask/pairencoding, finiteforward/backward, gradient accumulation weighting, sourceencoder/headinit, checkpointreload. Profile một optimizerstep384pairs trênGPU đểước lượng runtime/memory trước fullrun.
- Chi phí encoder dự kiến cỡ12lần baseline mộtpass, còn phụ thuộcpadding vàđộdàiprompt; **hai lượt train không đồng nghĩa hai lượt rẻ như baseline**. Báo ETA thực từprofile, không cam kết thời gian trước.
- Chọn GPU theo freeVRAM thực; có thể chạyA/B song song nếu đủbộnhớ, không dừng process khác. Hai taskworkers là đủ; không mở thêmarm để lấpGPU.
- Save mỗiarm: resolvedconfig+mapping, logs/epochcurves, bestcheckpoint, validationpredictions đãgộp, metrics, runtime/status. Ghi model-only hayresumable rõ; không tự restart trial đã hoàn tất.
- Sau train reload bestcheckpoint, tính lại F1 và kiểm alignment. Báo `docs/CAFEBERT_ASPECT_PAIR_RESULT.md` với reference/A/B, Δtừngarm−reference vàB−A, bestepoch/runtime/parametercount.

## 7. Điều kiện đọc kết quả và dừng

- Nếu max(A,B) hơn reference **≥0,5pp**: tín hiệu đáng xác nhận seed ở lượt sau; chưa tự chạy xác nhận.
- Gain0,2–<0,5pp: chưa rõ, báo đúng độ chắc chắn.
- Gain<0,2pp hoặc giảm: chưa có tín hiệu để đầu tư tiếp trong cấu hình này.
- NếuA/B cùng tăng tương tự, gain có thể từ phân loại riêng từngaspect/head/compute; không quy cho semantics mô tả. NếuB hơnA, đó mới là tín hiệu đóng góp thêm của cáchdiễnđạt đãchọn, vẫn cần xácnhận.
- Đây là heuristic đầu tư, không significance. Không suy thành thắngcả5tasks haySOTA.
- **Kết thúc sau2runs vàreport.** Không tự thêmLR/prompt/seed/test, mởHotel/ViSFD, hoặc trộnaugmentation để cứu kết quả.
