# D2 — measurements before và sau fine-tune

Vòng D2 đã hoàn tất ngày 10/09/2026. Phân tích chỉ đọc representation cache
trên 2.000 dòng train cố định và toàn bộ validation của mỗi task. Không đọc
test, không cập nhật encoder và không chạy thêm seed.

## Contract

- BEFORE là cache D1 của student GD và teacher DAPT trước fine-tune.
- AFTER là checkpoint E1 CE student seed 42 và teacher E2 stage-0 seed 42,
  theo từng task. Teacher là derivative DAPT của BamiBERT, không phải
  checkpoint BamiBERT gốc.
- AFTER dùng cùng tokenizer, max length 256, cohort, labels và feature-cache
  schema với BEFORE. Kiểm tra IDs, source indices, labels và input fingerprints
  đều khớp trên cả 5 task.
- Probe dùng CLS, chuẩn hóa theo train, linear head zero-init, AdamW,
  200 full-batch steps, learning rate 0.01, weight decay 0.01, seed 42.
  VSFC chọn fixed-label macro-F1; ViSFD/ABSA chọn end-to-end pair micro-F1.

## Probe layerwise

Điểm dưới đây là probe validation ở layer cuối sau 200 bước. Delta chỉ mô tả
độ tách tuyến tính của representation trước và sau task fine-tune.

| Task | Model | BEFORE | AFTER | Delta |
|---|---|---:|---:|---:|
| VSFC Sentiment | Student | 71.07 | 83.59 | +12.52 |
| VSFC Sentiment | Teacher | 66.14 | 85.16 | +19.02 |
| VSFC Topic | Student | 70.91 | 78.67 | +7.76 |
| VSFC Topic | Teacher | 66.79 | 81.53 | +14.74 |
| ViSFD | Student | 63.38 | 78.41 | +15.02 |
| ViSFD | Teacher | 62.25 | 82.63 | +20.39 |
| Restaurant | Student | 50.11 | 64.49 | +14.38 |
| Restaurant | Teacher | 49.32 | 71.08 | +21.76 |
| Hotel | Student | 62.68 | 65.80 | +3.13 |
| Hotel | Teacher | 59.45 | 66.47 | +7.03 |

Probe final-state sensitivity dùng 1.000 bước để kiểm tra ảnh hưởng ngân sách
fit. Layer cuối student tăng 7.65–16.87 điểm so với BEFORE; teacher tăng
10.84–22.36 điểm. Nhiều head AFTER vẫn có cờ `underconverged`, đặc biệt ở
ViSFD/ABSA, nên không dùng các con số này để khẳng định đã đạt trần probe.

Một kiểm tra boundary được mở thêm sau khi thấy đường cong 200 bước: chỉ bốn
head AFTER ở student layer 7 và teacher layer 8 của ViSFD/Restaurant. Đây là
kiểm tra post-hoc, không phải quét layer 1.000 bước:

| Task | Model/state | 200 steps | 1,000 steps | Delta |
|---|---|---:|---:|---:|
| ViSFD | Student/7 | 77.83 | 77.50 | −0.33 |
| ViSFD | Teacher/8 | 77.55 | 78.24 | +0.69 |
| Restaurant | Student/7 | 64.09 | 66.64 | +2.54 |
| Restaurant | Teacher/8 | 69.89 | 70.72 | +0.83 |

## CKA

CKA dùng 1.024 dòng train cố định, seed 123, với CLS là view chính và
content-mean là view phụ. Embedding CLS hằng nên cặp layer 0 undefined/NaN;
đó là kết quả của mẫu số CKA, không phải collapse.

| Task | Student same-layer CKA mean | Teacher same-layer CKA mean | AFTER S–T mapped | BEFORE S–T mapped |
|---|---:|---:|---:|---:|
| VSFC Sentiment | 0.529 | 0.595 | 0.768 | 0.791 |
| VSFC Topic | 0.387 | 0.682 | 0.606 | 0.791 |
| ViSFD | 0.244 | 0.486 | 0.419 | 0.497 |
| Restaurant | 0.208 | 0.473 | 0.545 | 0.694 |
| Hotel | 0.237 | 0.389 | 0.570 | 0.663 |

Representation thay đổi khá rõ sau fine-tune, và CKA student–teacher giảm ở
cả 5 task. Tuy nhiên probe F1 cũng tăng mạnh ở hầu hết task; vì vậy không thể
gọi drift này là catastrophic forgetting hoặc suy ra một layer bottleneck.
CKA thấp chỉ mô tả hình học khác đi, không đo trực tiếp chất lượng task.

## Weight drift

Các nhóm dưới đây được tổng hợp lại từ per-parameter checkpoint deltas với
phân nhóm đúng: attention loại LayerNorm, FFN gồm intermediate/output dense,
LayerNorm tách riêng. Relative L2 là `||after-before|| / ||before||`.

| Task | Model | Attention | FFN | LayerNorm | Embeddings |
|---|---|---:|---:|---:|---:|
| Sentiment | Student | 0.014 | 0.015 | 0.001 | 0.006 |
| Sentiment | Teacher | 0.019 | 0.022 | 0.002 | 0.005 |
| Topic | Student | 0.018 | 0.019 | 0.001 | 0.009 |
| Topic | Teacher | 0.006 | 0.008 | 0.001 | 0.001 |
| ViSFD | Student | 0.149 | 0.124 | 0.012 | 0.142 |
| ViSFD | Teacher | 0.056 | 0.065 | 0.020 | 0.026 |
| Restaurant | Student | 0.147 | 0.134 | 0.015 | 0.116 |
| Restaurant | Teacher | 0.028 | 0.035 | 0.013 | 0.011 |
| Hotel | Student | 0.168 | 0.147 | 0.017 | 0.109 |
| Hotel | Teacher | 0.057 | 0.068 | 0.020 | 0.019 |

Student cập nhật tương đối lớn hơn teacher ở các task ABSA, nhưng đây vẫn là
mức thay đổi trọng số chứ không phải thước đo tầm quan trọng. Không dùng bảng
này để quyết định đóng băng hay sửa một block.

## Artifacts

- Probe comparison: `artifacts/d2/probe_comparison.json` và `.csv`.
- Final-layer sensitivity: `artifacts/d2/probe_sensitivity/final_layer_sensitivity.json`.
- Boundary follow-up: `artifacts/d2/probe_sensitivity/boundary_sensitivity.json`.
- CKA matrices and summaries: `artifacts/d2/cka/cka_comparison.json` và từng
  task dưới `artifacts/d2/cka/<task>/result.json`.
- Corrected weight-drift summary: `artifacts/d2/weight_drift_comparison.json`.
- Feature caches: `artifacts/d2/cache/`; all manifests certify `test_opened: false`.

Kết luận của D2 là fine-tune tạo ra task signal tuyến tính rõ rệt trong cả
student và teacher, trong khi không có vùng layer chung được CKA, probe và
weight drift cùng xác nhận là bottleneck nhân quả. Chưa có cơ sở chọn một
block cụ thể để can thiệp tiếp theo.
