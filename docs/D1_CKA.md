# D1 CKA layer diagnostic

Ngày chạy: 2026-09-09T09:34:06.315832+00:00

Phân tích này chỉ đọc feature cache; không load test, không khởi tạo model và không chạy encoder inference.

## Contract và phương pháp

- Cache: `/home/kiettqa/GoiBERT/artifacts/d1/cache`; mỗi mảng có dạng `[layer, row, hidden]`.
- CKA chính dùng CLS; mean pooling là view phụ để kiểm tra độ nhạy của kết luận.
- CKA và within-model matrices chỉ dùng tối đa 1024 dòng train, chọn cố định bằng seed 123.
- Validation IDs vẫn được kiểm tra để xác nhận cùng hàng giữa student và teacher, nhưng không dùng để tính CKA trong pass này.
- Student có 11 layer (embedding + 10 transformer), teacher có 13 layer (embedding + 12 transformer).
- Mapping GD được khai báo trước: `[0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]`. Các cặp này chỉ được báo cáo, không được xem là mapping tối ưu do CKA.
- Reference control: cùng các dòng nhưng teacher bị permute bằng seed 314159; đây là reference hữu ích cho hiệu ứng finite-sample, không phải p-value hay permutation-test grid.
- CKA là similarity bất biến với phép biến đổi trực giao và scale; similarity thấp không tự chứng minh bottleneck có thể sửa.

## Sanity checks

`{"identity": 1.0, "orthogonal_rotation": 1.0000000000000002, "scaling": 1.0, "constant_undefined": null, "pass": true}`

Identity, orthogonal rotation và scaling phải cho CKA gần 1; biểu diễn hằng phải cho undefined/NaN do mẫu số bằng 0. NaN ở layer embedding CLS vì vậy là cờ zero-denominator, không bị đổi thành 0.

## CKA theo cặp GD (CLS, primary; train sample)

| Task | Matched mean | Row-permuted mean | Matched range | Matched minus permuted | Undefined pairs |
|---|---:|---:|---:|---:|---:|
| vsfc_sentiment | 0.7911 | 0.0067 | 0.7495–0.8423 | 0.7844 | 23 |
| vsfc_topic | 0.7911 | 0.0067 | 0.7495–0.8423 | 0.7844 | 23 |
| visfd | 0.4971 | 0.0108 | 0.3912–0.6725 | 0.4864 | 23 |
| absa_restaurant | 0.6941 | 0.0076 | 0.6155–0.8221 | 0.6866 | 23 |
| absa_hotel | 0.6631 | 0.0084 | 0.5631–0.8044 | 0.6547 | 23 |

## Reading the evidence

The CLS cross-CKA matrices are primary. Mean-pooling matrices are a secondary coverage view and must not be used to select a pooling strategy.

Within-model CKA is reported only on the fixed train sample. It describes redundancy/similarity among layer representations, not task quality. The row-wise best teacher layer and gap are retained in each JSON artifact for audit, but this report does not turn them into a recommended GD mapping.

Finite-sample caveat: student and teacher widths differ (384 vs 768), and CKA is computed on the same rows but does not directly measure functional equivalence or task accuracy. Any intervention should require agreement with the layerwise probe curves before being proposed.

## Per-task artifact index

- `vsfc_sentiment`: `artifacts/d1/cka/vsfc_sentiment/result.json`, train-only CLS/mean cross and within-model heatmaps (`.png` and `.svg`).
- `vsfc_topic`: `artifacts/d1/cka/vsfc_topic/result.json`, train-only CLS/mean cross and within-model heatmaps (`.png` and `.svg`).
- `visfd`: `artifacts/d1/cka/visfd/result.json`, train-only CLS/mean cross and within-model heatmaps (`.png` and `.svg`).
- `absa_restaurant`: `artifacts/d1/cka/absa_restaurant/result.json`, train-only CLS/mean cross and within-model heatmaps (`.png` and `.svg`).
- `absa_hotel`: `artifacts/d1/cka/absa_hotel/result.json`, train-only CLS/mean cross and within-model heatmaps (`.png` and `.svg`).
