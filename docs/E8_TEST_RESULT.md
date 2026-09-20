# E8 test result

The locked E8 NC+LLRD seed-42 checkpoint was evaluated once on each of the five test splits with the original CLS classifier, strict model-state loading, CUDA BF16, SDPA, max length 256, batch size 32, and dropout disabled by `model.eval()`. No teacher, noise, retraining, or test-based reselection was used.

| Task | Metric | E8 test | BamiBERT paper | Δ |
|---|---|---:|---:|---:|
| VSFC Sentiment | Accuracy | 92.67 | 93.86 | -1.19 |
| VSFC Sentiment | Macro F1 | 81.21 | 83.41 | -2.20 |
| VSFC Topic | Accuracy | 89.17 | 89.34 | -0.17 |
| VSFC Topic | Macro F1 | 79.38 | 79.90 | -0.52 |
| ViSFD | Detection micro F1 | 87.97 | 89.14 | -1.17 |
| ViSFD | Aspect-polarity pair micro F1 | 81.75 | 84.24 | -2.49 |
| UIT-ABSA Restaurant | Detection micro F1 | 85.58 | 88.01 | -2.43 |
| UIT-ABSA Restaurant | Aspect-polarity pair micro F1 | 71.52 | 74.89 | -3.37 |
| UIT-ABSA Hotel | Detection micro F1 | 78.88 | 79.99 | -1.11 |
| UIT-ABSA Hotel | Aspect-polarity pair micro F1 | 72.62 | 72.65 | -0.03 |

The single-seed differences are point comparisons and do not establish statistical significance. Local VSFC macro-F1 and ABSA pair-micro-F1 evaluator/averaging equivalence with the paper remains unresolved; the reported rows are retained as the project’s explicit local metrics. ViSFD OTHERS is included in detection and excluded from polarity pairs.

Manifest: `/home/kiettqa/GoiBERT/artifacts/e8/test_selection_manifest.json` (SHA-256 `87223fca670a75215188274ae5e51b2fb6c25a8bc09eba95ea088d1dcd0afae2`). Paper targets: `/home/kiettqa/GoiBERT/benchmarks/paper_targets.csv`; [BamiBERT paper](https://arxiv.org/html/2607.02259).

## Root interpretation and paired historical comparison

E8 có cùng recipe NC+LLRD trên cả5task, không ghép các cấu hình cũ. Không metric nào trong10rows cao hơn mốc paper; Hotel pair F1 gần nhất, thấp hơn khoảng0.0336điểm. Đây là mộtseed, không phải kết luận significance.

Bảng dưới dùng E2 và E8 cùngseed42, không lấy trung bình3seed làm đối chứng. E2 dùng NC cho Sentiment/ViSFD/Restaurant/Hotel và KD cho Topic, theo lựa chọn validation cũ.

| Task | E2 test seed42 | E8 test seed42 | Delta |
|---|---:|---:|---:|
| Sentiment | 80.88 | 81.21 | +0.34 |
| Topic | 79.83 | 79.38 | -0.46 |
| ViSFD | 80.33 | 81.75 | +1.42 |
| Restaurant | 69.25 | 71.52 | +2.27 |
| Hotel | 72.70 | 72.62 | -0.09 |

ViSFD và Restaurant cải thiện rõ về điểm đo so với NC riêng trong lần chạy này. E8 chưa vượt BamiBERT paper. Không chọn lại checkpoint hoặc mở thêm training/test sau khi thấy kết quả. Chỉ các artifact E8 mới được đánh giá trong lượt này; E7 Restaurant vẫn chưa được chấm test.
