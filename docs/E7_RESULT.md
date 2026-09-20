# E7 result — layer-wise learning-rate decay

E7 ran five validation-only trials with seed 42. The model, CLS head, CE
loss, data split, base learning rates, and all E1 training settings were
held fixed. The only change was AdamW learning rates by encoder depth with
`gamma=0.8`: embeddings `base * gamma^10`, blocks 0–9 `base *
gamma^(9-layer_index)`, and the last block plus classification head at the
base rate. All parameters remained trainable.

The CUDA preflight passed all checks: 24 groups covered all 169 trainable
parameter tensors exactly once, bias/LayerNorm parameters retained zero
weight decay, the scheduler preserved group ratios, and the gamma=1 setting
matched E1's parameterwise learning-rate/weight-decay settings. All five
workers completed with return code 0 on GPUs 2, 1, and 3. Test data was not
loaded.

## Validation readout

| Task | E1 CE | E7 LLRD | Delta | Best epoch | Existing practical exploratory reference |
|---|---:|---:|---:|---:|---:|
| VSFC Sentiment | 84.51 | 84.06 | −0.44 | 13 | E2 NC 84.77 |
| VSFC Topic | 79.77 | 79.95 | +0.18 | 30 | E2 KD 80.78 |
| ViSFD pair F1 | 80.07 | **81.49** | **+1.42** | 26 | E2 NC 81.09 |
| Restaurant pair F1 | 69.02 | **71.52** | **+2.50** | 28 | E3 polarity loss 70.13 |
| Hotel pair F1 | 72.04 | **73.29** | **+1.25** | 26 | E2 NC 73.58 |

The Sentiment row also has the small E5 mean-pooling candidate at 84.83; it
is shown separately because the table reference column keeps the previously
selected practical method for each task.

The three ABSA tasks all improved over the matched E1 seed-42 trial, with
the largest gains on Restaurant and ViSFD. This is a promising screening
signal, not a confirmed multi-seed result. E7 does not win every task: E2 NC
remains higher on Hotel and Sentiment is below E1. No test evaluation or
additional seed was opened.

## Drift check

The CPU-only readout compares each selected checkpoint with the original
`goiturn-s-gd` encoder weights. It reports aggregate
`sqrt(sum(||after-before||^2) / sum(||before||^2))`; it does not run an
encoder forward pass. E7 substantially reduced lower-layer drift while
allowing more change in the upper blocks:

| Task | Region | E1 drift | E7 drift |
|---|---|---:|---:|
| VSFC Sentiment | embeddings / early / mid / late | 0.0061 / 0.0111 / 0.0122 / 0.0137 | 0.0016 / 0.0050 / 0.0113 / 0.0235 |
| VSFC Topic | embeddings / early / mid / late | 0.0092 / 0.0142 / 0.0152 / 0.0170 | 0.0025 / 0.0062 / 0.0140 / 0.0307 |
| ViSFD | embeddings / early / mid / late | 0.1417 / 0.1137 / 0.1070 / 0.1168 | 0.0267 / 0.0327 / 0.0711 / 0.1491 |
| Restaurant | embeddings / early / mid / late | 0.1157 / 0.1156 / 0.1102 / 0.1242 | 0.0200 / 0.0298 / 0.0650 / 0.1411 |
| Hotel | embeddings / early / mid / late | 0.1093 / 0.1312 / 0.1213 / 0.1383 | 0.0179 / 0.0318 / 0.0678 / 0.1452 |

This supports the intended mechanical effect of LLRD, but the checkpoint
drift is measured at each method's selected epoch (E1 epochs 5/7/14/22/27;
E7 epochs 13/30/26/28/26). Therefore it does not isolate LR from the amount
of training. The full machine-readable records are in
`artifacts/e7/summary.json` and `artifacts/e7/drift_readout.json`.

## Root research decision

Giữ E7 làm ứng viên mới cho ViSFD và Restaurant: vượt E2 NC cùngseed42 khoảng0.40 điểm trên ViSFD và vượt E3 khoảng1.39 điểm trên Restaurant. Hotel tăng soE1 nhưng vẫn dưới NC khoảng0.29 điểm. Sentiment giảm; Topic tăng nhỏ và vẫn dưới KD. Mean-pooling E5 trên Sentiment có84.8285, cao hơn mốc NC84.7722 một chút; E7 không vượt cảhai.

Vòng này ủng hộ giả thuyết phân bố mức cập nhật theo layer trên các task ABSA: drift tầng dưới giảm và F1 cùng tăng. Không coi đây là bằng chứng một block đã hỏng hoặc khẳng định cơ chế forgetting. Chỉ mộtseed, checkpoint được chọn bằngvalidation và gamma mới thử mộtgiá trị.

Không thay cấu hình mặc định chung cho5task, không tự chạy thêm seed/test/NC-combination. Tất cả5trial đã hoàn tất30epochs; prediction audit khớp ID/gold và recomputedmetrics. Không tăng kiến trúc hoặc inferenceparametercount.
