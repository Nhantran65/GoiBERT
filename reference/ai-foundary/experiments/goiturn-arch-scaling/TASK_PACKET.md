GOAL: Implement a self-contained architecture-scaling experiment: fine-tune BamiBERT (baseline) and train 2 smaller RoBERTa variants on 3 Vietnamese classification datasets, producing per-run metric JSONs, a latency benchmark, and a summary comparison table.

BUSINESS REASON: Feasibility evidence for the GoiTurn-S production model size decision (proposal targets a ~21M-param Vietnamese turn-completion classifier distilled from a BamiBERT-family backbone). This experiment quantifies the performance-vs-size trade-off before the team commits to an architecture.

SUCCESS CRITERIA:
1. `python src/run_all.py --smoke` completes end-to-end on CPU or GPU without error.
2. Parameter counts printed at model build time match (±0.1%): baseline 102,360,576; variant_a 35,970,048; variant_b 25,807,872 (encoder+embeddings, excluding classification head — print head params separately).
3. Every run writes `results/<dataset>__<model>__<mode>.json` with config, seed, per-epoch validation metrics, best-epoch test metrics.
4. `src/latency.py` writes `results/latency.json` (3 architectures × seq {32,128} × threads {1,4}, p50/p95 ms, batch 1, CPU).
5. `src/run_all.py` (full mode) executes runs in this order per dataset: baseline finetune FIRST (saves checkpoint + train-set logits), then variant runs; writes `results/summary.md` with the comparison table.
6. All randomness seeded (seed 42): python, numpy, torch, CUDA.

FILES (create; base dir = experiments/goiturn-arch-scaling/):
- configs/variants.json — the 3 architecture configs below
- src/data.py — dataset loading/parsing
- src/modeling.py — model construction, PCA embedding init, param counting
- src/train.py — single training run CLI
- src/latency.py — CPU latency benchmark CLI
- src/run_all.py — orchestrator (supports --smoke and --only <dataset>)
- README.md — brief usage
Do NOT modify anything outside experiments/goiturn-arch-scaling/. Read-only inputs: data/raw/*.csv, models/bamibert/*.

ENVIRONMENT:
- KNOWN CONSTRAINT: Windows Smart App Control blocks torch DLLs on this machine, so you CANNOT run torch locally. Do NOT try to import torch, run the smoke test, or fix the environment. The supervisor executes the code inside a Linux container after you deliver. Your executable verification is limited to `py_compile` with the system Python (`python`, 3.13 — works, only stdlib needed).
- Write the code for a standard Linux Python 3.11+ environment with: torch 2.x, transformers, pandas, scikit-learn, safetensors, numpy. Use pathlib relative paths anchored to the repo root (resolve repo root as parents[2] of src/ files, or a --repo-root CLI arg defaulting to that) so the code runs both on Windows paths and inside a container where the repo is mounted at /workspace.
- GPU: use CUDA if available, fp16 autocast for training, CPU fallback must work. Latency benchmark always CPU fp32.
- BamiBERT checkpoint downloaded at models/bamibert/ (config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json, model.safetensors). HF architecture RobertaForMaskedLM; load with transformers `AutoModel`/`RobertaModel` + `AutoTokenizer` from that local path (local_files_only=True).

ARCHITECTURES (build with transformers `RobertaConfig` + `RobertaModel`, vocab_size=20481, type_vocab_size=1, pad_token_id=1, bos=0, eos=2, position_embedding_type="absolute"):
- baseline: num_hidden_layers=12, hidden_size=768, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=2050 — weights loaded from models/bamibert.
- variant_a: layers=8, hidden=512, heads=8, intermediate=2048, max_position_embeddings=514.
- variant_b: layers=10, hidden=384, heads=6, intermediate=1536, max_position_embeddings=514.
All use the SAME BamiBERT tokenizer. Variants are NEVER loaded from the checkpoint (shapes incompatible) — random init, except the distill mode embedding init below.

CLASSIFICATION HEADS (shared implementation):
- Pool: take <s> (first token) last hidden state, then dense(H→H)+tanh+dropout(0.1)+linear(H→out).
- VSFC: out=3, softmax, cross-entropy.
- ABSA datasets: one 4-way softmax PER aspect slot (out = n_aspects×4, reshaped to [n_aspects,4]; class 0 = aspect absent). Sum of per-aspect CE losses.

DATA (all in data/raw/):
1. UIT-VSFC_{train,valid,test}.csv — columns text,label; label in {positive,negative,neutral}. Map to ints. 11426/1583/3166 rows.
2. VLSP2018-ABSA-Restaurant.csv — columns: Review, 12 aspect columns (values 0..3), type (train/dev/test), dataset. Use type for splits. n_aspects=12.
3. ViSFD.csv — columns: dataset,type,index,comment,n_star,date_time,label. Use type for splits. Parse label strings like `{CAMERA#Positive};{BATTERY#Negative};{OTHERS};` → 10 aspects (CAMERA, FEATURES, BATTERY, PRICE, GENERAL, SER&ACC, PERFORMANCE, SCREEN, DESIGN, STORAGE) with polarity {Positive=1, Negative=2, Neutral=3}, plus OTHERS as an 11th binary-only slot (present=1). DISCOVER the actual aspect vocabulary by scanning the file — do not trust this list blindly; print the discovered set and use it, sorted, as the canonical slot order saved into every results JSON.
CSV parsing: pandas, utf-8, keep emoji/diacritics as-is. No text normalization beyond str strip. Tokenize with BamiBERT tokenizer, max_length=256, truncation.

TRAINING (src/train.py CLI: --dataset {vsfc,vlsp_restaurant,visfd} --model {baseline,variant_a,variant_b} --mode {finetune,scratch,distill} --seed 42 --smoke):
- Common: AdamW, weight_decay 0.01, linear warmup 6% then linear decay, batch 32, 10 epochs, max_length 256, select best epoch by validation macro F1 (VSFC) / validation detection micro-F1 + classification micro-F1 average (ABSA), evaluate that checkpoint on test.
- finetune (baseline only): lr 2e-5. After training, save best model state_dict to results/checkpoints/<dataset>__baseline.pt AND cache teacher logits over the train set (best checkpoint, eval mode, no dropout) to results/teacher_logits/<dataset>.pt.
- scratch (variants): lr 5e-5, random init (transformers default initializer, seed-controlled).
- distill (variants): lr 5e-5. Word-embedding init: load BamiBERT word_embeddings [20481×768], mean-center, PCA (torch.pca_lowrank or SVD) to student hidden size, scale columns to preserve per-dimension std of a default random init, copy into student word embeddings. Position/type embeddings and all other weights random. Loss = 0.5·CE + 0.5·T²·KL(student/T || teacher/T), T=2, teacher logits from results/teacher_logits/<dataset>.pt (error out with a clear message if missing). For ABSA apply KL per aspect slot and average.
- --smoke: cap train/valid/test at 200/100/100 rows, 1 epoch, batch 8 — must exercise all three modes and both task types.

METRICS (implement in train.py or a small shared module; sklearn allowed):
- VSFC: accuracy, macro F1, weighted F1.
- ABSA: detection micro-F1 (binary aspect-present over all [example×aspect] cells, present = argmax>0); classification micro-F1 (over the same cells, TP only when both present AND polarity class matches; standard micro P/R/F1 where predicted-present counts toward precision denominator, gold-present toward recall denominator). Also report macro variants.
- Per-run JSON must include: dataset, model, mode, seed, full arch config, lr, epochs run, best epoch, per-epoch val metrics, test metrics, param counts (encoder+emb and head separately), wall-clock train seconds, torch/transformers versions, device name.

LATENCY (src/latency.py): for each of the 3 architectures (random weights fine — latency is shape-determined): model.eval(), CPU, fp32, batch 1, dummy input_ids of seq len 32 and 128, torch.set_num_threads(1) then (4), 30 warmup + 300 timed iters, report p50/p95 ms per (arch, seq, threads) into results/latency.json. Also record fp32 state_dict size on disk (save to temp, measure, delete).

RUN_ALL (src/run_all.py): for each dataset in [vsfc, vlsp_restaurant, visfd]: baseline finetune → variant_a scratch → variant_a distill → variant_b scratch → variant_b distill. Then latency. Then write results/summary.md: one row per (dataset, model, mode): Layers, Hidden, Heads, FFN, Params, F1 (macro or detection/classification), Accuracy (VSFC only), plus a latency table. --smoke passes the flag through everywhere. --only <dataset> runs one dataset.

PATTERNS: plain PyTorch training loop (no Trainer API), type hints, pathlib, no global state, print progress per epoch with metrics on one line. Constants (paths, aspect lists) at module top. Keep it simple — no config frameworks, no logging frameworks.

CONSTRAINTS: no network access at runtime (local_files_only=True everywhere); no new pip installs; no files written outside experiments/goiturn-arch-scaling/results and the source tree above; deterministic seeding; total source under ~900 lines.

NON-GOALS: no ONNX/INT8 export, no turn-completion dataset, no MLM pretraining, no HF Trainer, no hyperparameter search, no production integration, no UIT-ABSA download attempts.

DONE WHEN: all source files written per spec; `python -m py_compile` passes on every source file; code is complete and runnable (no stubs/TODOs). Success criteria 1–5 are EXECUTED BY THE SUPERVISOR in the container afterward — you make them satisfiable, you do not run them.

VALIDATE (run these, show output):
1. `python -m py_compile` on each file under experiments/goiturn-arch-scaling/src/ (system Python 3.13; name each file explicitly).
2. Show a static walkthrough: for each architecture, the parameter-count formula result your modeling code will produce (arithmetic, no torch import) — must match the targets in success criterion 2.

STOP RULES: Missing prerequisite (checkpoint/data file absent, CSV schema differs from spec) — report, do not guess. Do NOT stop over torch being unloadable on Windows — that is expected and handled by the supervisor. ViSFD aspect vocabulary differs wildly from the list above — print what you found and proceed with the discovered set, noting it. Ambiguous requirement — stop and report the gap. If smoke run cannot pass after fixes — report the traceback and stop.

ROLLBACK: delete experiments/goiturn-arch-scaling/{src,configs,results,README.md}; nothing else is touched.
