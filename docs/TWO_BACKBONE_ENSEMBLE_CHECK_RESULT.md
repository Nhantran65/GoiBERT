# Kiểm tra ghép hai backbone bằng trung bình xác suất — validation only

## Bối cảnh và câu hỏi

Các backbone đơn lẻ BidirLM-1B và P_vi chưa vượt CafeBERT ổn định. Phân tích lỗi trước đó gợi ý các model khác họ có lỗi bổ sung cho nhau. Phép đo này kiểm tra CafeBERT cắt vocab + P_vi, với trọng số 0,5/0,5, tổng tham số tối đa 1 tỷ. Hai đối chứng gồm Cafe + XLM-R-large gốc và Cafe seed 42 + Cafe seed 43 (Restaurant). Tất cả số liệu chỉ dùng validation; không train, không mở test.

## F1 của các model và tổ hợp

F1 dưới đây là `selection_f1 × 100`, tính lại từ logits và metric của repo. Chữ số giữ nguyên ở độ chính xác báo cáo, không làm tròn cho đẹp.

| Model/tổ hợp | Restaurant | ViSFD | Topic |
|---|---:|---:|---:|
| Cafe | 77.71971496437054 | 85.46650717703349 | 81.74458365092477 |
| Cafe seed 43 | 76.39605818864383 | — | — |
| XLM-R-large | 76.3445978105664 | 85.48459133541759 | 80.6123331161886 |
| P_vi | 76.14900822447991 | 85.43689320388349 | 80.69444272956812 |
| O | 75.71976967370442 | 84.85393258426967 | 80.73548845647585 |
| B1B | 77.08634828750604 | 85.75703171753442 | 81.51567693516449 |
| PAIR Cafe + P_vi | 77.78313253012048 | 86.33373205741627 | 81.39959480837756 |
| CTRL_FAM Cafe + XLM-R-large | 77.2400574988021 | 85.73985680190931 | 80.14903738360996 |
| CTRL_SEED Cafe + Cafe seed 43 | 76.61596958174906 | — | — |
| Cafe + O | 77.21822541966426 | 86.05350165314096 | 81.97154638038117 |
| Cafe + B1B | 77.47834456207893 | 86.15984405458089 | 81.41939601072954 |
| P_vi + B1B | 76.76570871894788 | 86.15153788447113 | 80.44604778625066 |
| Cafe + P_vi + B1B | 78.68217054263566 | 86.62382304588253 | 82.2924293453649 |

| Tổ hợp | Task | Δ vs Cafe | Δ vs CTRL_FAM |
|---|---|---:|---:|
| PAIR Cafe + P_vi | Restaurant | 0.06341756574994406 | 0.5430750313183808 |
| PAIR Cafe + P_vi | ViSFD | 0.8672248803827785 | 0.5938752555069584 |
| PAIR Cafe + P_vi | Topic | -0.34498884254720963 | 1.2505574247675924 |
| CTRL_FAM Cafe + XLM-R-large | Restaurant | -0.4796574655684367 | 0.0 |
| CTRL_FAM Cafe + XLM-R-large | ViSFD | 0.27334962487582004 | 0.0 |
| CTRL_FAM Cafe + XLM-R-large | Topic | -1.595546267314802 | 0.0 |
| CTRL_SEED Cafe + Cafe seed 43 | Restaurant | -1.103745382621483 | -0.6240879170530462 |
| Cafe + O | Restaurant | -0.5014895447062742 | -0.021832079137837468 |
| Cafe + O | ViSFD | 0.5869944761074635 | 0.31364485123164343 |
| Cafe + O | Topic | 0.22696272945640317 | 1.8225089967712051 |
| Cafe + B1B | Restaurant | -0.2413704022916079 | 0.2382870632768288 |
| Cafe + B1B | ViSFD | 0.6933368775473951 | 0.4199872526715751 |
| Cafe + B1B | Topic | -0.3251876401952245 | 1.2703586271195775 |
| P_vi + B1B | Restaurant | -0.9540062454226614 | -0.47434877985422474 |
| P_vi + B1B | ViSFD | 0.6850307074376332 | 0.4116810825618131 |
| P_vi + B1B | Topic | -1.2985358646741076 | 0.29701040264069434 |
| Cafe + P_vi + B1B | Restaurant | 0.9624555782651214 | 1.442113043833558 |
| Cafe + P_vi + B1B | ViSFD | 1.1573158688490395 | 0.8839662439732194 |
| Cafe + P_vi + B1B | Topic | 0.5478456944401273 | 2.1433919617549293 |

## Chẩn đoán validation

Paired bootstrap lấy lại ví dụ (1.000 lần, seed 42); khoảng tin cậy là percentile 2,5–97,5%. Tỷ lệ `>0` là tỷ lệ replicate có chênh lệch dương.

| Task | Chênh lệch | Điểm | CI 95% | Tỷ lệ >0 |
|---|---|---:|---|---:|
| Restaurant | PAIR_minus_cafe | 0.06341756574994406 | [-1.3343244093566806, 1.4865830280351688] | 0.57 |
| Restaurant | PAIR_minus_CTRL_FAM | 0.5430750313183808 | [-0.8160974805279778, 1.830758258484508] | 0.813 |
| ViSFD | PAIR_minus_cafe | 0.8672248803827785 | [0.1512514050087962, 1.5700948171823366] | 0.993 |
| ViSFD | PAIR_minus_CTRL_FAM | 0.5938752555069584 | [0.011746591415576563, 1.192541175489399] | 0.975 |
| Topic | PAIR_minus_cafe | -0.34498884254720963 | [-1.984109982520306, 1.2225658463759626] | 0.323 |
| Topic | PAIR_minus_CTRL_FAM | 1.2505574247675924 | [-0.5120554498915851, 3.1133156259560932] | 0.902 |

Quét trọng số Cafe dưới đây chọn trên validation nên kết quả tốt nhất là lạc quan; không dùng cho gate.

| w_Cafe | Restaurant | ViSFD | Topic |
|---:|---:|---:|---:|
| 0.3 | 76.86350435624395 | 85.84229390681004 | 80.48631795945987 |
| 0.4 | 77.54707870593917 | 86.0638615338705 | 80.77756272576366 |
| 0.5 | 77.78313253012048 | 86.33373205741627 | 81.39959480837756 |
| 0.6 | 77.75131014768938 | 86.0381861575179 | 81.56343474570696 |
| 0.7 | 77.52380952380953 | 85.80808834502312 | 81.61085268247398 |

| Task | Slot sửa đúng | Slot làm sai | Ví dụ có sửa đúng | Ví dụ có làm sai | Ví dụ có cả hai |
|---|---:|---:|---:|---:|---:|
| Restaurant | 55 | 46 | 52 | 45 | 1 |
| ViSFD | 165 | 111 | 143 | 107 | 9 |
| Topic | 18 | 22 | 18 | 22 | 0 |

## Cắt vocab và số tham số

Cafe dùng cùng SentencePiece với XLM-R-large, vocab gốc 250.002. Keep-set và tokenizer cắt từ `artifacts/vocab_trim_student/` được áp vào checkpoint Cafe trong bộ nhớ; logits được so với checkpoint gốc cùng precision BF16.

| Task | Chuỗi token ID khớp | max abs(Δ logits) | Dự đoán giống hệt | F1 gốc | F1 cắt | Cafe cắt + head | P_vi + head | Tổng |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Restaurant | 771/771 (1.0) | 0.0 | True | 77.71971496437054 | 77.71971496437054 | 350017584 | 597148720 | 947166304 |
| ViSFD | 1112/1112 (1.0) | 0.0 | True | 85.46650717703349 | 85.46650717703349 | 350011434 | 597142570 | 947154004 |
| Topic | 1583/1583 (1.0) | 0.0 | True | 81.74458365092477 | 81.74458365092477 | 349972484 | 597103620 | 947076104 |

Tổng lớn nhất khi dùng head lớn nhất: **947166304** tham số. Điều kiện ≤1.000.000.000: **True**.

## Gate cố định

PASS chỉ khi cả ba task có PAIR − Cafe ≥0,8 điểm và PAIR > CTRL_FAM, đồng thời Cafe cắt cho dự đoán giống hệt và tổng ≤1B.

| Task | Δ PAIR − Cafe | ≥0,8 | Δ PAIR − CTRL_FAM | >0 |
|---|---:|---|---:|---|
| Restaurant | 0.06341756574994406 | False | 0.5430750313183808 | True |
| ViSFD | 0.8672248803827785 | True | 0.5938752555069584 | True |
| Topic | -0.34498884254720963 | False | 1.2505574247675924 | True |

Trim dự đoán giống hệt: **True**. Ngân sách tham số: **True**. Quyết định: **CLOSE**.

## Tái tạo và giới hạn

Đã strict load và tái tạo dự đoán/F1 cho 16 checkpoint; mọi check trong `reproduction_checks.json` đều PASS. SHA-256 của 16 checkpoint trong manifest trùng metadata gốc. Chỉ dùng GPU vật lý 1 (`CUDA_VISIBLE_DEVICES=1`, trong process là `cuda:0`). Tổng GPU giờ theo thời gian model nằm trên GPU: **0.014886567203017573**.

Cảnh báo từ Transformers về regex tokenizer xuất hiện khi nạp BidirLM-0.6B/1B. Đã giữ nguyên tokenizer của các run gốc; dự đoán tái tạo trùng tuyệt đối. Khi dựng tokenizer cắt ở lượt thử đầu, thiếu khai báo pad token làm loader dừng trước phép kiểm trim; sau đó dùng đúng năm special token của `trim.py` và cả ba phép kiểm trim PASS. Không cần cài hoặc nâng cấp gói. Không có sai lệch phương pháp so với spec.

Chỉ có một seed chung; các thành viên đã chọn epoch theo validation nên ensemble trên validation lạc quan. Các đối chứng chịu cùng thiên lệch chọn epoch. Trung bình xác suất khác bầu chọn đa số trước đó. Chưa có kết quả test.

## Lệnh tái lập

```bash
bash scripts/two_backbone_ensemble/run_all.sh
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/two_backbone_ensemble/ensemble_metrics.py
PYTHONPATH=src .venv-e0-cuda/bin/python scripts/two_backbone_ensemble/summarize.py
```
