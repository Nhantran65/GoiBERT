# E6 validation result

The E6 shared backbone is the encoder-only export
`models/goiturn-s-gd-mlm-e6`, loaded after the fixed 500-step head warmup and
2,000-step joint MLM continuation. Its exported `model.safetensors` hash is
`96163c803854bdfe67f76a58158a450df45a14742312e0b529aa96b230f7406b`.
The MLM source hash is recorded separately in the per-task metadata.

All five downstream trials used seed 42, the E1 CLS head, CE loss, 30 epochs,
the locked task learning rates, BF16/SDPA, and validation only. Metrics below
are the saved prediction metrics recomputed once after all workers completed.

| Task | E1 F1 | E6 F1 | E6 − E1 |
|---|---:|---:|---:|
| VSFC sentiment | 84.508 | 84.355 | −0.153 |
| VSFC topic | 79.771 | 78.949 | −0.822 |
| ViSFD pair | 80.072 | 79.476 | −0.596 |
| Restaurant pair | 69.017 | 67.928 | −1.089 |
| Hotel pair | 72.038 | 71.152 | −0.885 |

The MLM objective improved on its fixed holdout during continuation, but this
short supervised screen did not transfer that gain to any benchmark
validation task. No test evaluation, extra seed, or follow-up combination was
run. The controller's five workers completed successfully; the first
controller process failed only while writing the final summary, so the summary
was recomputed from the completed artifacts without rerunning training.

Machine-readable outputs are in
`artifacts/e6/finetune/summary.json` and `summary.csv`.


## Root research decision

Dừng cấu hình pilot E6 này; giữ models/goiturn-s-gd làm backbone mặc định. Không task nào tăng so với E1 cùng seed42, và không mở thêm seed, test, số bước MLM hay kết hợp NC để tìm kết quả thuận lợi.

Phép thử cho thấy MLM trên khoảng33k text train với ngân sách đã chốt giảm loss dự đoán token nhưng không tăng F1 downstream. Chưa chứng minh catastrophic forgetting hoặc pretraining nói chung không hiệu quả: corpus, mục tiêu, head khởi tạo, thời lượng và learning rate đều giới hạn kết luận. Cũng không dùng MLM holdout loss làm thay thế cho benchmark.

Một shared encoder được dùng cho cả5task; kiến trúc, tokenizer và inference parameter count giữ nguyên. MLM head chỉ dùng trong pretraining. Encoder E6 được giữ làm artifact experimental, không ghi đè backbone GD gốc.

Kế hoạch: [E6_PLAN.md](E6_PLAN.md). Corpus/MLM và mask holdout: [E6_MLM.md](E6_MLM.md).
