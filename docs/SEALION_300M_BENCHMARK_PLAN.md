# SEA-LION-ModernBERT-300M: benchmark 5 task

Ngày 2026-09-17. User yêu cầu fine-tune trên 5 task và benchmark với BamiBERT/CafeBERT; user cho phép tạo subagent cho lượt này. Executor `gpt-5.6-luna`, `xhigh`; root review. Không DAPT/KD/NC/LLRD/composite.

## Scope

- Tasks: VSFC Sentiment, VSFC Topic, ViSFD, UIT-ABSA Restaurant, UIT-ABSA Hotel. Năm task, bốn miền văn bản vì hai task VSFC chung text.
- SEA-LION pretrained encoder chính thức `aisingapore/SEA-LION-ModernBERT-300M`, revision `e3d70b810769bf86ae9041d1943e0ecd10b62b31`, local `models/sea-lion-modernbert-300m`.
- Seed42, CE, 30epochs, effectivebatch32, maxlen256; LR {2e-5,5e-5}: 10trials.
- AdamW betas(.9,.999),eps1e-8,wd.01,bias/norm không decay,warmup10%+linear decay,clip1,BF16/SDPA. Microbatch/accumulation theo profile; ghi rõ thực tế.
- Head first-token→dropout.1→dense H→tanh→dropout.1→logits, toàn encoder trainable. SEA dùng BOS2 theo tokenizer thật; không force legacy CLS50281 trong config. Không có word segmentation ngoài tokenizer.
- Model card có native mean-pooling classifier; lượt này dùng shared first-token head để so pipeline. Không claim đã tìm recipe tối ưu SEA-LION.

## Baselines và tái sử dụng

- BamiBERT E14: đủ5tasks, 3LR {1e-5,2e-5,5e-5},30epochs,seed42; reuse validation-selected checkpoints và saved test metrics/predictions.
- CafeBERT Restaurant: reuse pilot T trong `artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42`,LR2e-5,30epochs.
- CafeBERT4tasks còn lại: fine-tune CE LR2e-5,30epochs,seed42 theo cùng họhead. Bốn trials mới.
- Tổng14new FT trials, không rerun baseline đã có. Root inventory `artifacts/sealion_benchmark/root_baseline_inventory.json`.
- Ngân sách tuning **Bami3LR / SEA2LR / Cafe1LR** không đồng nhất, phải ghi bên bảng. Benchmark thực dụng các hệ thống trong ngân sách này, không claim cô lập architecture hoặc best attainable performance.

## Correctness và đánh giá

- Frozen downstream loader/evaluator; ABSA0absent1positive2negative3neutral; ViSFD42logits widths(4,4,4,4,4,2,4,4,4,4,4),OTHERSindex5 excludedpair.
- VSFC fixed-label macroF1; ABSA end-to-end pairmicroF1. Fullprecision lưu[0,1], bảng thang100.
- Pin model/tokenizer/sourcehashes; kiểm shape/finitegrad cả5schemas, checkpoint reload, ID/gold alignment và recompute F1.
- Bestepoch chọn validation; SEA chọn LR bằngvalidation, tie ưu tiên LRnhỏ/epochsớm. Freeze selectedcheckpoint manifest trước test; không chọn lại bằngtest.
- Test đúng SEA5/Cafe5selected checkpoints, Bami reuse E14. Save predictions và accesslog, tách val/test trong bảng. Test bộ hiện có đã dùng nghiên cứu trước, không virginholdout.
- Báo từngtask/delta so Bami vàCafe; một seed không claim statisticalsignificance hoặc VietnameseNLU SOTA. Exact downstream params đếm từ model, không nhầm số nonembedding.

## Vận hành/output

- ProfileGPU và dùng GPU2/3 khi khả dụng, mộtworker/GPU; không ảnh hưởng jobsGPU0/1. Sandbox không thấydriver; host exec đã kiểm CUDA khả dụng.
- Source mới `scripts/sealion_benchmark/`; không sửa frozen code/artifacts. Reuse helper/loop CDBK qua adapter có metadatađúng, tránh Restaurant/task/model hardcodes.
- Durable logs/status/heartbeat; kiểm process trước restart, không doublelaunch. Không tự thêmseed/grid.
- Output `artifacts/sealion_benchmark/`; báo cáo `docs/SEALION_300M_BENCHMARK_RESULT.md` gồm bảngval/test, LR/epoch/params/runtime và limitations. Chưa có kết quả mới tại thời điểm viết plan.
