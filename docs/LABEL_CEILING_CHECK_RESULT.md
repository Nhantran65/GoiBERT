# Kiểm tra trần nhãn gốc — validation only

## Trạng thái và protocol

**Phần A đã đủ ba seed.** Phần B và C đã hoàn tất; số liệu validation only.

Mục tiêu là đo nhiễu seed CafeBERT và phân rã lỗi còn lại bằng pool các họ model khác nhau. Đơn vị là câu cho VSFC, slot aspect cho ABSA/ViSFD. CE: gold khác nhãn được ít nhất N−1 phiếu; mức chặt là N phiếu. LE gốc: Cafe sai và ít nhất một model khác đúng; HE: Cafe sai, không model đúng, không thuộc CE.

**Giao thoa định nghĩa:** CE và LE gốc có thể trùng nhau khi N−1 model cùng sai và model còn lại đúng. Bảng phân rã dùng CE ưu tiên, LE = LE gốc trừ CE, rồi HE để tạo phân hoạch. Cột “giao thoa” và LE gốc được lưu trong `consensus_analysis.json`; H_max sửa toàn bộ LE gốc.

## A. Recipe và nhiễu seed

Dry-run seed 43/44 cho cả năm task khớp core recipe seed 42: LR 2e-5, 30 epoch, batch hiệu dụng 32, max length 256, bf16, first-token → dropout 0,1 → dense → tanh → dropout 0,1 → logits, CE thường, best validation. Restaurant T42 và seed 43 cũ khớp hyperparameter, model/tokenizer và evaluator; cách diễn đạt protocol/head/selection, adapter, hash nguồn và runtime metadata khác. Seed 43 cũ được tái dùng qua liên kết tới artifact gốc. Chi tiết tại `recipe_diff.json`.

| Task | Cafe 42 (%) | Cafe 43 (%) | Cafe 44 (%) | Mean (%) | SD mẫu (%) | R (%) |
|---|---|---|---|---|---|---|
| Sentiment | 87.69241120832682 | 88.3227335863605 | 87.56091232936947 | 87.85868570801894 | 0.4072202276861304 | 0.7618212569910243 |
| Topic | 81.74458365092477 | 81.49534089307537 | 82.3715746082198 | 81.87049971740664 | 0.45148369044480136 | 0.8762337151444366 |
| ViSFD | 85.46650717703349 | 85.51498961116059 | 85.15937731653077 | 85.38029136824161 | 0.192846831156752 | 0.3556122946298257 |
| Restaurant | 77.71971496437054 | 76.39605818864383 | 77.61476573592049 | 77.24351296297829 | 0.735790922099996 | 1.3236567757267115 |
| Hotel | 73.34384858044164 | 73.20666925164792 | 74.61300309597523 | 73.72117364268827 | 0.7753865954313796 | 1.406333844327301 |


Restaurant seed 42 → seed 43 đã thấy: 77,71971496437055 → 76,39605818864383, chênh 1,32365677572672 điểm. Đây mới là chênh hai seed, chưa phải R ba seed.

### Chênh lệch đã thấy với Cafe seed 42

| Task | P_vi − Cafe (điểm) | BidirLM-1B − Cafe (điểm) | Cafe + P_vi − Cafe, artifact (điểm) |
|---|---|---|---|
| Sentiment | -1.1738581146078508 |  |  |
| Topic | -1.050140921356646 | -0.22890671576026866 | -0.34498884254720963 |
| ViSFD | -0.029613973150000028 | 0.2905245405009249 | 0.8672248803827785 |
| Restaurant | -1.570706739890626 | -0.6333666768645108 | 0.06341756574994406 |
| Hotel | 0.617812761411396 |  |  |


R ba seed đã có trong bảng trên; đối chiếu trực tiếp với các chênh đã thấy. “PAIR” trong artifact ensemble là Cafe + P_vi.

## B. Pool và phân rã

Pool có N=6 cho Restaurant/ViSFD/Topic, N=5 cho Sentiment/Hotel. LR của mmBERT lấy từ `summary.json`; Bami từ locked selection E14; các run khác theo đường dẫn cố định của spec. Đã tính lại F1 bằng `goibert.train.evaluate_predictions` và `selection_score`; ID và gold trùng tuyệt đối với validation bundle và giữa mọi model. XLM-R-large gốc cùng họ Cafe nên chỉ là tham khảo ngoài pool.

| Task | Cafe | mmBERT | Bami | mDeBERTa | P_vi | BidirLM-1B |
|---|---|---|---|---|---|---|
| Sentiment | 87.69241120832682 | 86.69631606877476 | 86.32522837351507 | 85.9183396137906 | 86.51855309371896 |  |
| Topic | 81.74458365092477 | 81.18084870447339 | 79.53280821882547 | 80.07961846549284 | 80.69444272956812 | 81.51567693516449 |
| ViSFD | 85.46650717703349 | 83.6390977443609 | 84.38339802926247 | 79.84848484848484 | 85.43689320388349 | 85.75703171753442 |
| Restaurant | 77.71971496437054 | 75.626204238921 | 74.74178403755869 | 70.55984555984556 | 76.14900822447991 | 77.08634828750604 |
| Hotel | 73.34384858044164 | 73.69696969696969 | 73.2361963190184 | 70.82494969818913 | 73.96166134185303 |  |


XLM-R-large seed 42 tham khảo: Restaurant 77.81472684085512% (LR 2e-05), Topic 80.6123331161886% (LR 2e-05), ViSFD 85.48459133541759% (LR 2e-05); không vào pool.

| Task | Seed | Đơn vị | Lỗi Cafe | CE | LE riêng | HE | CE/lỗi | CE∩LE gốc | CE cả N |
|---|---|---|---|---|---|---|---|---|---|
| Sentiment | 42 | 1583 | 77 | 51 | 24 | 2 | 66.23376623376623 | 14 | 33 |
| Sentiment | 43 | 1583 | 76 | 44 | 28 | 4 | 57.89473684210527 | 10 | 31 |
| Sentiment | 44 | 1583 | 79 | 46 | 30 | 3 | 58.22784810126582 | 11 | 32 |
| Topic | 42 | 1583 | 157 | 94 | 57 | 6 | 59.87261146496815 | 26 | 60 |
| Topic | 43 | 1583 | 159 | 94 | 60 | 5 | 59.11949685534591 | 24 | 61 |
| Topic | 44 | 1583 | 153 | 96 | 54 | 3 | 62.745098039215684 | 23 | 64 |
| ViSFD | 42 | 12232 | 973 | 508 | 430 | 35 | 52.20966084275437 | 151 | 320 |
| ViSFD | 43 | 12232 | 960 | 503 | 416 | 41 | 52.39583333333333 | 145 | 322 |
| ViSFD | 44 | 12232 | 971 | 500 | 431 | 40 | 51.493305870236874 | 145 | 317 |
| Restaurant | 42 | 9252 | 362 | 168 | 163 | 31 | 46.408839779005525 | 54 | 99 |
| Restaurant | 43 | 9252 | 384 | 172 | 181 | 31 | 44.79166666666667 | 54 | 100 |
| Restaurant | 44 | 9252 | 364 | 172 | 165 | 27 | 47.25274725274725 | 51 | 100 |
| Hotel | 42 | 27030 | 612 | 423 | 176 | 13 | 69.11764705882352 | 79 | 316 |
| Hotel | 43 | 27030 | 613 | 407 | 192 | 14 | 66.394779771615 | 64 | 317 |
| Hotel | 44 | 27030 | 597 | 408 | 176 | 13 | 68.34170854271356 | 70 | 311 |


Tỷ lệ LE/HE và LE gốc đầy đủ nằm trong `consensus_analysis.json`. ViSFD OTHERS là slot detection-only: tham gia đếm lỗi/đồng thuận, nhưng bị loại đúng như evaluator khi tính pair micro F1.

| Task | Seed | F1 gốc (%) | L_max (điểm) | H_max (điểm) | Oracle pool (%) |
|---|---|---|---|---|---|
| Sentiment | 42 | 87.69241120832682 | 8.493310522722208 | 5.937158681861698 | 93.62956989018852 |
| Sentiment | 43 | 88.3227335863605 | 6.542612150126914 | 5.7259261351770725 |  |
| Sentiment | 44 | 87.56091232936947 | 6.8511085468784705 | 6.294599256771405 |  |
| Topic | 42 | 81.74458365092477 | 10.970126357128851 | 8.584965713908232 | 90.329549364833 |
| Topic | 43 | 81.49534089307537 | 11.460345271911043 | 8.987516541160701 |  |
| Topic | 44 | 82.3715746082198 | 11.765911812278418 | 8.12758023258665 |  |
| ViSFD | 42 | 85.46650717703349 | 7.432456942468091 | 8.451498529529056 | 93.91800570656255 |
| ViSFD | 43 | 85.51498961116059 | 7.3133986399203765 | 8.228446247804156 |  |
| ViSFD | 44 | 85.15937731653077 | 7.162077879533058 | 8.693696146737606 |  |
| Restaurant | 42 | 77.71971496437054 | 10.224210269274314 | 13.486435924625129 | 91.20615088899568 |
| Restaurant | 43 | 76.39605818864383 | 10.372037708082772 | 14.582828567594175 |  |
| Restaurant | 44 | 77.61476573592049 | 10.29221100826556 | 13.407797874257144 |  |
| Hotel | 42 | 73.34384858044164 | 18.83006446303662 | 10.568327068260952 | 83.91217564870259 |
| Hotel | 43 | 73.20666925164792 | 18.014847371437614 | 10.636125508177397 |  |
| Hotel | 44 | 74.61300309597523 | 17.597209352776165 | 9.752076269104137 |  |


L_max là thay dự đoán Cafe ở mọi CE mà Cafe sai bằng gold rồi chạy lại evaluator; H_max tương tự với mọi LE gốc. Oracle đúng nếu bất kỳ model nào đúng từng đơn vị: đây là **cận trên lạc quan**, không phải hệ thống khả thi.

### Bỏ CE khỏi chấm điểm

Với ABSA/ViSFD, TP, số dự đoán dương và số gold dương được cộng thủ công trên slot còn lại theo pair micro F1 gốc; kiểm bộ đếm không bỏ slot khớp evaluator. Với VSFC, evaluator macro F1 chạy lại trên các câu còn lại.

| Task | F1 khi bỏ CE (%) theo thứ tự Cafe / mmBERT / Bami / mDeBERTa / P_vi / B1B | Thứ hạng đổi? |
|---|---|---|
| Sentiment | 95.06619423436234 / 93.18790805936638 / 92.92227153773305 / 92.47199731519225 / 92.24926186003705 /  | có |
| Topic | 90.79567746042828 / 90.8956058950434 / 88.54482744914051 / 89.35125333274807 / 90.00546246861106 / 91.34378843093992 | có |
| ViSFD | 92.32516631510627 / 90.5469387755102 / 91.01214574898785 / 85.93493263227079 / 92.28775113415425 / 92.72816007808687 | không |
| Restaurant | 86.29117959617429 / 83.97193739881274 / 82.69331930562862 / 78.27493261455525 / 84.9240780911063 / 85.20518358531316 | không |
| Hotel | 89.84796468857283 / 91.34808853118712 / 89.11079410366143 / 86.95652173913044 / 91.52542372881356 /  | không |


Thứ hạng cụ thể trước/sau nằm trong `consensus_analysis.json`.

### Đối chiếu đọc mù 14b

UID được nối qua `*_uid_mapping.json` → ID validation; slot theo `aspect_names` của bundle gốc và vector nhãn. Với Topic nối câu. Reader của đợt cũ gán toàn review; phép đếm dưới đây chỉ giữ các slot/câu CE đã được đọc.

| Task | CE đã đọc | Reader = gold | Reader = nhãn CE | Reader khác cả hai |
|---|---|---|---|---|
| Restaurant | 96 | 37 | 47 | 12 |
| ViSFD | 274 | 78 | 177 | 19 |
| Topic | 46 | 12 | 32 | 2 |


## C. Gói phân xử mù

Mỗi task có 24 CE, 8 đối chứng toàn pool đúng, 8 đơn vị pool chia rẽ; tổng 200 mục. `items.jsonl` và `items.csv` chỉ có UID ngẫu nhiên, task, text, aspect, nhãn hợp lệ, ô trống; đã xáo seed 42. `key.jsonl` chứa đáp án và nhóm, không mở trước khi phân xử. `score_adjudication.py` chỉ được chuẩn bị, chưa chạy.

## Quy tắc diễn giải đã chốt

Đơn vị R/L/H ở bảng sau là điểm F1; C là tỷ lệ lỗi. Điều kiện trần nhãn của task: C ≥ 0,5 và L ≥ 2R. Kết luận chung cần ≥3/5 task đạt. H < 2R là phép kiểm dư địa học được.

| Task | R | C | L | H | C ≥ 0,5 | L ≥ 2R | H < 2R | Đạt cả hai |
|---|---|---|---|---|---|---|---|---|
| Sentiment | 0.007618212569910243 | 0.6623376623376623 | 0.08493310522722208 | 0.059371586818616984 | True | True | False | True |
| Topic | 0.008762337151444366 | 0.5987261146496815 | 0.10970126357128851 | 0.08584965713908232 | True | True | False | True |
| ViSFD | 0.003556122946298257 | 0.5220966084275437 | 0.07432456942468091 | 0.08451498529529056 | True | True | False | True |
| Restaurant | 0.013236567757267115 | 0.46408839779005523 | 0.10224210269274314 | 0.1348643592462513 | False | True | False | False |
| Hotel | 0.01406333844327301 | 0.6911764705882353 | 0.18830064463036622 | 0.10568327068260952 | True | True | False | True |


Điều kiện chung ≥3/5 task: True

## GPU, kiểm tra và giới hạn

- GPU-giờ dùng cho run mới: **4.271174383362134**; GPU đã dùng: **1, 2**. Cafe Restaurant seed 43 là artifact có sẵn, không tính vào ngân sách phiên này. Budget launcher: 15 GPU-giờ, ước tính lại sau hai run mới đầu và dừng trước đợt kế nếu vượt trần.
- Check CPU: F1 prediction = F1 best metric; IDs/gold đồng nhất; evaluator thủ công không bỏ CE = F1 gốc; recipe core PASS. Không có check FAIL ở phần đã chạy. Postrun checks của seed 43 cũ PASS theo metadata gốc.
- CE chỉ là ứng viên nhãn sai, chưa phải nhãn sai đã xác nhận. Pool 5–6 model vẫn có thể sai chung vì nguyên nhân khác. Đọc mù cũ chỉ có một người đọc LLM trên mỗi task. Mọi số chỉ thuộc validation.

## Lệnh tái lập và hoàn tất GPU

Chạy tại root repo. Launcher tự chọn GPU 1/2 khi VRAM trống ≥20 GiB, tối đa một job của mình trên mỗi GPU, không dùng GPU 0/3. Mỗi lượt kiểm sau ≥5 phút. Lệnh này dùng venv của run Cafe gốc, `PYTHONPATH=src`, `HF_HUB_OFFLINE=1`:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/recipe_diff.py
nohup env PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/launcher.py > artifacts/label_ceiling_check/logs/launcher.log 2>&1 &
```

Sau khi launcher hoàn tất hoặc dừng vì budget, chạy lại toàn bộ phân tích CPU và báo cáo bằng một lệnh:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/analyze.py && PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/summarize.py
```

Dry-run một cấu hình riêng, không nạp model lên GPU:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/label_ceiling_check/seed_train.py --task visfd --seed 43 --dry-run-config
```


## Ghi chú của root về kiểm tra recipe (2026-09-24)

Launcher dừng hai lần sau khi run **đã train xong và qua postrun**. Cả hai lần đều do bộ lọc so config trong `seed_train.py` quá chặt, không phải do run sai:

1. **Sentiment/Topic seed 43.**
   - Bộ lọc so khớp chính xác tên trường. Vì vậy các trường con `gpu_snapshot_at_start.<i>.*` không được nhận là đã cho phép.
   - Thêm vào đó, thư mục snapshot của tokenizer XLM-R-large có thêm file `model.safetensors`, tải về sau này cho `xlmr_large_baseline`. File này không phải file tokenizer.
   - Trọng số CafeBERT giống hệt run gốc (`model.weight_file_sha256 = a7306a…`); các file tokenizer không đổi.
2. **Restaurant seed 44.** Run seed 42 gốc được tạo bằng `cdbk_pilot`, ghi `arm=T, init=cafebert, kd={enabled: False}`; runner `sealion_benchmark` ghi `model_kind=cafebert`. Đây là cùng một model, CafeBERT thường, không KD.

Cách sửa:

- Root sửa `blocked_differences` trong `seed_train.py`: trường được cho phép khớp theo tiền tố, và chỉ chấp nhận đúng các giá trị tương đương nêu trên.
- Thêm `scripts/label_ceiling_check/recheck_recipe.py` để kiểm lại các run bị chặn. Bản kết quả cũ được giữ ở `recipe_check.pre_fix.json`.
- Sau khi sửa, cả 10 ô đều `finished = True`. Không run nào phải train lại.

Tổng GPU của 9 run mới khoảng **4,27 GPU-giờ**, chạy trên GPU 1 và GPU 2.
