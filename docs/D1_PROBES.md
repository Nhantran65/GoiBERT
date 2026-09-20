# D1 layerwise linear probes

Ngày 09/09/2026. D1 là vòng chẩn đoán biểu diễn, không phải benchmark model mới. Encoder không được train lại và test split không được mở.

## Protocol

- Dùng 2.000 dòng train được chọn cố định bằng `sorted(np.random.default_rng(42).choice(...))`; validation dùng toàn bộ. Cache có cả CLS và mean, nhưng probe chính chỉ dùng CLS. Mean cache được giữ cho CKA.
- Student có 11 hidden states (embedding + 10 transformer layers, hidden 384); teacher DAPT có 13 hidden states (embedding + 12 layers, hidden 768). Teacher local là `goibert-b-dapt`, derivative DAPT của BamiBERT, không phải checkpoint BamiBERT gốc.
- Mỗi state dùng một linear head, khởi tạo zero, fit đúng 200 full-batch AdamW steps, learning rate `0.01`, weight decay `0.01`, seed `42`. Không chọn epoch hoặc learning rate bằng validation.
- Chuẩn hóa theo từng feature bằng mean/std của train, `std = sqrt(train_variance + 1e-6)`, sau đó chia `sqrt(hidden_size)`. Cùng công thức được dùng trong sensitivity 1.000 bước.
- VSFC dùng CE thông thường. ABSA dùng trung bình CE giữa các slot; ViSFD giữ 10 slot 4 lớp và slot OTHERS nhị phân, rồi dùng evaluator cố định để tính pair micro F1. VSFC báo fixed-label macro F1.

Artifacts chính nằm ở [`artifacts/d1/probes`](../artifacts/d1/probes). Mỗi task có `probe_results.json`, validation predictions và hash/counts. Đường cong tổng hợp ở [`layerwise_probe_f1.svg`](../artifacts/d1/probes/plots/layerwise_probe_f1.svg) và [`layerwise_probe_f1.png`](../artifacts/d1/probes/plots/layerwise_probe_f1.png). Runner là [`d1_linear_probes.py`](../scripts/d1_linear_probes.py).

## Fixed 200-step readout

Các số dưới đây là validation selection F1 (%). `best nonzero` loại state 0 (embedding); đây là mô tả trong các state đã đo, chưa phải bằng chứng layer đó là bottleneck.

| Task | Student best nonzero | Student final | Teacher best nonzero | Teacher final |
|---|---:|---:|---:|---:|
| VSFC Sentiment | 71.07 (L10) | 71.07 | 68.01 (L8) | 66.14 (L12) |
| VSFC Topic | 70.92 (L10) | 70.92 | 70.29 (L8) | 66.79 (L12) |
| ViSFD | 63.44 (L9) | 63.38 (L10) | 63.06 (L7) | 62.25 (L12) |
| UIT-ABSA Restaurant | 50.11 (L10) | 50.11 | 49.32 (L12) | 49.32 |
| UIT-ABSA Hotel | 62.68 (L10) | 62.68 | 61.98 (L3) | 59.45 (L12) |

The final-layer label-shuffle control was run once per model/task with the same train-row permutation seed `314159`; majority predictors and state-0 constant probes are recorded in each task JSON. The control is a sanity check, not a requirement that the score equal a theoretical chance value after only 200 optimizer steps.

## 1.000-step sensitivity

Because several 200-step probes were still learning at the tail, a bounded sensitivity was run only on the final state and the best nonzero state selected by the fixed 200-step result. It uses the same features, normalization and optimizer; it does not reselect a layer using the 1.000-step validation score. Outputs are in [`artifacts/d1/probe_sensitivity`](../artifacts/d1/probe_sensitivity), runner [`d1_probe_sensitivity.py`](../scripts/d1_probe_sensitivity.py).

The table reports `F1(1000) - F1(200)` in percentage points:

| Task | Student selected state(s) | Teacher selected state(s) |
|---|---:|---:|
| VSFC Sentiment | +1.51 (L10) | +1.74 (L8), +3.13 (L12) |
| VSFC Topic | −1.62 (L10) | −2.42 (L8), −0.47 (L12) |
| ViSFD | −2.09 (L9), −1.60 (L10) | −2.71 (L7), −1.86 (L12) |
| UIT-ABSA Restaurant | −0.42 (L10) | +0.12 (L12) |
| UIT-ABSA Hotel | −0.94 (L10) | −2.40 (L3), −1.25 (L12) |

Several 1.000-step fits still have a positive last-20-step improvement flag, especially teacher Sentiment/Topic and ABSA. Therefore the curves are budget-sensitive and do not justify calling a low probe score an encoder bottleneck. Conversely, the 1.000-step run often lowers train CE while validation F1 falls, which is consistent with probe overfitting on the 2.000-row train sample.

The probe compares frozen representation readout under different native widths and head parameter counts: a scalar three-class probe has 1,155 parameters on student features and 2,307 on teacher features before task-specific slot multiplication. It does not measure the teacher's fully fine-tuned task performance, and a lower teacher probe score alone cannot establish that the teacher representation is worse.

Layer indices are local to each encoder. State 8 in a 13-state teacher is not assumed to correspond exactly to state 8 in the 11-state student; interpretation should use relative depth and the CKA readout. No causal bottleneck or architectural intervention is selected by D1 alone.
