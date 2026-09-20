# E0 dữ liệu và metric audit

Ngày audit: 2026-09-08. Audit này chỉ kiểm tra cấu trúc, split, nhãn, text, hash và evaluator fixture. Không dùng test để chọn preprocessing, hyperparameter hay checkpoint.

## Kết luận để giao runner

| Task ID | Training-ready | Paper-comparison-ready | Lý do chính |
|---|---|---|---|
| `vsfc_sentiment` | READY | BLOCKED cho F1; Accuracy có thể đối chiếu | Nguồn và label map xác minh được; paper không mô tả averaging của F1 |
| `vsfc_topic` | READY | BLOCKED cho F1; Accuracy có thể đối chiếu | Nguồn và label map xác minh được; paper không mô tả averaging của F1 |
| `visfd` | READY cho internal validation | BLOCKED | Nguồn raw xác minh được; OTHERS không có polarity và evaluator paper không công bố đủ |
| `absa_restaurant` | READY cho internal validation | BLOCKED | Bản CSV public có kích thước/header/label distribution phù hợp nhưng provenance chưa xác nhận với release gốc; evaluator paper không công bố đủ |
| `absa_hotel` | READY cho internal validation | BLOCKED | Cùng giới hạn provenance/evaluator; có duplicate giữa split cần ghi nhận |

`BLOCKED` ở cột cuối không chặn screening validation. Nó chỉ cấm gọi điểm nội bộ là cùng metric với mốc F1 của BamiBERT cho tới khi evaluator và lineage được xác minh.

## Loader contract

`src/goibert/data.py` cung cấp `load_task(task_id, repo_root, include_test=False)`. Mặc định chỉ đọc train/validation để screening không chạm test. Chỉ sau selection gate mới gọi `include_test=True`. Loader giữ thứ tự hàng, text nguyên trạng (VSFC bỏ line terminator tự nhiên của file text), split gốc và tạo ID ổn định dạng `<task>:<split>:<row-zero-based>`.

Task IDs là `vsfc_sentiment`, `vsfc_topic`, `visfd`, `absa_restaurant`, `absa_hotel`. `TaskDataset` có các property `train`, `validation`, `test`, `num_classes`, `num_aspects`, `aspect_names`, `slot_label_names`; test property báo lỗi rõ nếu chưa được load.

## Label map đã khóa

- VSFC lấy trực tiếp từ loader nguồn Hugging Face: sentiment `0=negative, 1=neutral, 2=positive`; topic `0=lecturer, 1=training_program, 2=facility, 3=others`.
- UIT-ABSA candidate dùng encoding trong ma trận: `0=absent, 1=negative, 2=neutral, 3=positive`. Mapping này khớp mô tả bên ngoài và phân bố non-zero của hai CSV, nhưng chưa có evaluator/release chính thức để chứng minh tương đương bản tác giả dùng cho BamiBERT.
- ViSFD giữ polarity raw: `0=absent, 1=Positive, 2=Negative, 3=Neutral`. `{OTHERS}` là tag không có polarity, được giữ riêng bằng code `4`.

### Policy ViSFD OTHERS

Policy nội bộ đã được root chấp thuận cho E1 validation; paper-comparison vẫn bị khóa:

1. Giữ OTHERS là một aspect slot binary: raw `0` là absent, raw `4` là present.
2. `visfd_to_model_labels` chuyển raw `4 -> model label 1` ở đúng slot OTHERS; `visfd_from_model_labels` chuyển ngược `1 -> raw 4`. Mười slot polarity còn lại giữ 4 lớp `0/1/2/3`.
3. Aspect detection tính OTHERS như mọi aspect present.
4. Aspect-polarity pair classification loại OTHERS khỏi denominator và không bao giờ coi OTHERS là positive.

Đây là policy semantically faithful cho dữ liệu raw và phù hợp với packet cũ ghi “10 aspect 4-class + binary OTHERS”. Nó chưa được gắn với evaluator chính thức của paper; vì vậy ViSFD chỉ được báo internal validation.

## Metric contract

`src/goibert/metrics.py` chỉ dùng Python chuẩn:

- `classification_metrics`: accuracy và macro precision/recall/F1 trên fixed label IDs được truyền vào; không suy class set từ test hoặc prediction.
- `absa_metrics`: `detection_micro_*`, `detection_macro_*`, `pair_micro_*`, `pair_macro_*` và per-aspect breakdown. Detection là presence (`code > 0`); pair là end-to-end exact aspect-polarity code trên các aspect được mask. ViSFD OTHERS nằm trong detection nhưng không nằm trong pair.
- Không có key chung `macro_f1` cho ABSA và không flatten ma trận 0..3 để gắn nhãn paper F1. `classification_pair_*` chỉ là tên dài minh bạch cho pair metric.

Fixture trong `tests/test_data_metrics.py` tính tay các trường hợp đúng, dư/thiếu aspect, sai polarity, absent và OTHERS; test cũng kiểm tra fixed-label rejection và screening loader không load test.

## Structural checks

- Số dòng khớp protocol: VSFC `11426/1583/3166`, ViSFD `7786/1112/2224`, Restaurant `7028/771/1938`, Hotel `7180/795/2030`.
- VSFC có 1 duplicate text trong train với sentiment conflict (`0` và `2`), giữ nguyên cả hai rows; topic của hai rows là `0`.
- ViSFD không có duplicate exact text trong hoặc giữa split.
- Restaurant không có duplicate exact text trong hoặc giữa split.
- Hotel có 4 duplicate rows trong train (cùng nhãn), 1 overlap train/validation (cùng nhãn), và 2 overlaps train/test; một overlap train/test có label vector conflict. Không xóa hoặc sửa các rows.
- Không có text trống; mọi label nằm trong fixed map. Hash từng source file và thống kê label nằm trong `artifacts/e0/data_audit.json`.

## Evidence và giới hạn

- VSFC source loader/revision: [UIT-NLP dataset repository](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) và [source script snapshot](../data/raw/uit-vsfc/source/vietnamese_students_feedback.py).
- ViSFD raw release/README: [LuongPhan/UIT-ViSFD](https://github.com/LuongPhan/UIT-ViSFD).
- UIT-ABSA paper/dataset description: [MMLab UIT dataset page](https://mmlab.uit.edu.vn/dataset), [DOI 10.1145/3446678](https://doi.org/10.1145/3446678).
- Public candidate bytes: [candidate repository at fixed commit](https://github.com/tienquocbao/ABSA-RAG-Vietnamese-Restaurant-Hotel-Bot/tree/2ff3fb7ac67ec404f5323cdc30f46e4853969998).
- Secondary label-map cross-check for UIT-ABSA: [PBA-Net dataset description](https://www.researchgate.net/publication/395747763_PBA-Net_A_Dual-Branch_Architecture_with_Positional_Bias_Attention_and_Multi-Scale_CNN_for_Low-resource_Language_Aspect-Based_Sentiment_Analysis). This is not treated as proof of the original release lineage.
- Paper targets/protocol: [BamiBERT paper](https://arxiv.org/html/2607.02259). It reports F1 and selection by validation F1, but does not publish enough evaluator detail here to establish averaging, OTHERS handling, or end-to-end versus conditional classification equivalence.

The audit therefore separates `training_ready` from `paper_comparison_ready` and keeps all claims about BamiBERT provisional until the missing evaluator evidence is recovered.

## Inherited validation-text exposure

`models/goiturn-s-gd/distill_manifest.json` identifies the historical corpus only by an unavailable server path and `corpus_stats_sha256=1fa34627adddf9206f850e4595ecb0f88fdb47477eebb5beadb191ad20b26c80`; the corpus bytes are not present in this workspace. The historical handoff also records that validation text was present in an earlier corpus mix. Therefore exact overlap against the current validation/test rows cannot be computed here, and the encoder's prior exposure to validation text is **unknown/possible**, not certified disjoint. E1 still avoids using current test rows for any new selection or training step; reports must carry this inherited-exposure caveat.
