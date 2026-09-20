# E8 confirmation result

The frozen E8 NC+LLRD recipe was confirmed on ViSFD and UIT-ABSA Restaurant with fresh students and seeds 43 and 44. Each best checkpoint was selected by validation only, locked in a four-row manifest, and evaluated once on the test split with CUDA BF16, SDPA, max length 256, batch size 32, strict model loading, and dropout disabled.

| Task | Seed | Best epoch | Pair F1 validation | Pair F1 test | Detection F1 test |
|---|---:|---:|---:|---:|---:|
| ViSFD | 43 | 30 | 0.8217 | 0.8195 | 0.8803 |
| UIT-ABSA Restaurant | 43 | 21 | 0.7131 | 0.7224 | 0.8609 |
| ViSFD | 44 | 24 | 0.8213 | 0.8172 | 0.8816 |
| UIT-ABSA Restaurant | 44 | 27 | 0.7318 | 0.7188 | 0.8573 |

Aggregate validation and test means use seeds 42, 43, and 44 with sample standard deviation. Paired per-seed E8 deltas against the already completed E1 CE and E2 NC runs are recorded in `summary.json`; the E8-versus-E2 NC positive-count is descriptive with three seeds and is not a significance test.

Paper-target comparisons are retained as secondary context: ViSFD pair F1 84.24 and detection F1 89.14; UIT-ABSA Restaurant pair F1 74.89 and detection F1 88.01. The local evaluator's averaging definitions may not be equivalent to the paper's reported values.

E7 LLRD has only seed 42 in the existing evidence, so this confirmation makes no multi-seed superiority claim over E7.

Manifest: `/home/kiettqa/GoiBERT/artifacts/e8/confirmation/manifest.json` (SHA-256 `48b091346b1ca2e0140845334bbb8306878a4c9e083b19f415b06ae86bd720d1`). CSV: `/home/kiettqa/GoiBERT/artifacts/e8/confirmation/summary.csv`.

The test gate resumed after a representation-only compatibility failure in the original evaluator: the runtime `ModelSpec.to_dict()` returned `aspect_class_counts` as a tuple while the locked JSON manifest stored the same values as a list. The test loader also materialized ABSA gold tuples as lists; the recovery wrapper restored the split container type before the original exact ID/gold check, preserving values and order. The original evaluator SHA-256 is `0f88e854745b4591b38b25a8159a48ad9c585ffb26347f95aa5bfb847fff14f8`; the recovery wrapper SHA-256 is `a51afe3e13c9950e150cd1d909c853942c95a404b169011f70779e306dca04a1`; recovery provenance is recorded at `/home/kiettqa/GoiBERT/artifacts/e8/confirmation/test_recovery_provenance.json`.

## Root readout — confirmation complete

Tất cả4trial mới đã train đủ30epochs;4test passes mới hoàn tất, seed42 và baseline không chạy lại. So sánh cùngseeds42/43/44, thangF1 0–100, sampleSD:

| Task | E2 NC test | E8 test | Delta | Seeds E8 > NC | BamiBERT paper |
|---|---:|---:|---:|---:|---:|
| ViSFD | 80.62 ± 0.26 | 81.81 ± 0.12 | +1.19 | 3/3 | 84.24 |
| Restaurant | 69.86 ± 0.63 | 71.88 ± 0.36 | +2.02 | 3/3 | 74.89 |

E8 đạt tiêu chí xác nhận thực dụng đã chốt (mean gain≥0.3điểm và ít nhất2/3seed dương), thực tế tăng3/3seed trên cảhai task. Giữ E8 làm baseline nghiên cứu có tín hiệu lặp lại so với NC. Chưa chứng minh significance hoặc vượt E7 đa seed (E7 chỉ cóseed42); vẫn thấp hơn mốc BamiBERT khoảng2.43điểm ViSFD và3.01điểm Restaurant, kèm giới hạn evaluator đã ghi.

Phân tích validation của ba seed cho thấy E8 giảm spurious-aspect và wrong-polarity errors nhất quán, nhưng không giảm missed-aspect errors nhất quán. So với teacher seed42, phần tăng pair FN của E8 chủ yếu là missed aspects:121/129 trên ViSFD và40/51 trên Restaurant. Đây là phân rã số đếm lỗi, không phải bằng chứng nhân quả về kiến trúc.

Các aspect đóng góp lớn vào khoảng cách đa phần có support train vừa/lớn, không chỉ lớp hiếm. Vì vậy hướng cần khảo sát tiếp là khả năng thu hồi aspect bị bỏ sót: kiểm tra decision margins/precision–recall trên validation để phân biệt ngưỡng quá bảo thủ với ranking/biểu diễn chưa đủ. Chưa chạy calibration, logits extraction hoặc một phương pháp mới trong vòng xác nhận này.

Chi tiết lỗi: [E8_ERROR_READOUT.md](E8_ERROR_READOUT.md). Kế hoạch: [E8_CONFIRMATION_PLAN.md](E8_CONFIRMATION_PLAN.md). Hai implementator đều được chỉ định rõ gpt-5.6-luna/xhigh theo AGENTS.md.
