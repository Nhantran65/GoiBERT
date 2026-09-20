# E14 — BamiBERT gốc trên pipeline GoiBERT

Protocol: `E14-BamiBERT-original-local-baseline-v1`; seed: `42`; HF commit: `57bc1340debbe4e348ec549047a763caebe4a977`.

Đây là local baseline dùng đúng checkpoint BamiBERT gốc và evaluator/split local. Nó không phải exact paper reproduction và không cô lập pure-size effect: BamiBERT khác pretraining source và recipe, đồng thời dùng head 768 so với student head 384; token ID samples của tokenizer gốc khớp GD trên các mẫu đã kiểm tra.

## Preflight và nguồn

- Encoder: RobertaModel 12 layers × 768 hidden, 12 heads, FFN 3072; pooling/MLM head bỏ khi downstream.
- Head: CLS → dropout 0.1 → dense 768 → tanh → dropout 0.1 → task logits.
- Recipe: AdamW, CE plain, weight decay 0.01, warmup 10% linear, max length 256, BF16 SDPA, effective batch 32; LR grid `(1e-05, 2e-05, 5e-05)`; 30 epochs; no NC/KD/LLRD.
- Weights SHA256: `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`.
- Tokenizer comparison with GD is recorded in `artifacts/e14/preflight.json`: serialized tokenizer files/config metadata have different hashes, while sampled token ID sequences matched exactly on 400 train/validation rows per task. E14 uses the original BamiBERT tokenizer.

## Validation grid

All scores are fractions in [0, 1]. LR and epoch were selected using validation only; ties prefer the smaller LR and the earliest best epoch.

| Task | LR | Best epoch | Validation selection F1 | Metric |
|---|---:|---:|---:|---|
| vsfc_sentiment | 1e-05 | 10 | 0.858823 | fixed_label_macro_f1 |
| vsfc_sentiment | 2e-05 | 8 | 0.863252 | fixed_label_macro_f1 |
| vsfc_sentiment | 5e-05 | 5 | 0.857663 | fixed_label_macro_f1 |
| vsfc_topic | 1e-05 | 13 | 0.795328 | fixed_label_macro_f1 |
| vsfc_topic | 2e-05 | 20 | 0.791482 | fixed_label_macro_f1 |
| vsfc_topic | 5e-05 | 16 | 0.791763 | fixed_label_macro_f1 |
| visfd | 1e-05 | 30 | 0.807855 | end_to_end_pair_micro_f1 |
| visfd | 2e-05 | 24 | 0.833906 | end_to_end_pair_micro_f1 |
| visfd | 5e-05 | 23 | 0.843834 | end_to_end_pair_micro_f1 |
| absa_restaurant | 1e-05 | 28 | 0.709082 | end_to_end_pair_micro_f1 |
| absa_restaurant | 2e-05 | 22 | 0.734249 | end_to_end_pair_micro_f1 |
| absa_restaurant | 5e-05 | 29 | 0.747418 | end_to_end_pair_micro_f1 |
| absa_hotel | 1e-05 | 30 | 0.684569 | end_to_end_pair_micro_f1 |
| absa_hotel | 2e-05 | 25 | 0.713888 | end_to_end_pair_micro_f1 |
| absa_hotel | 5e-05 | 24 | 0.732362 | end_to_end_pair_micro_f1 |

## Locked test results

| Task | Selected LR | Epoch | Test selection F1 | Metric | Accuracy | Macro F1 | Detection micro F1 | Pair micro F1 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| vsfc_sentiment | 2e-05 | 8 | 0.830330 | fixed_label_macro_f1 | 0.9352495262160455 | 0.8303297211171513 |  |  |
| vsfc_topic | 1e-05 | 13 | 0.781766 | fixed_label_macro_f1 | 0.8859759949463045 | 0.7817657634728281 |  |  |
| visfd | 5e-05 | 23 | 0.845017 | end_to_end_pair_micro_f1 |  |  | 0.8934044882320744 | 0.8450168300892726 |
| absa_restaurant | 5e-05 | 29 | 0.741493 | end_to_end_pair_micro_f1 |  |  | 0.8712163940590336 | 0.7414927617973304 |
| absa_hotel | 5e-05 | 24 | 0.739398 | end_to_end_pair_micro_f1 |  |  | 0.8011677934849416 | 0.7393976644130301 |

## Student E8 and paper context

The table keeps the E8 seed-42 student test readout (`artifacts/e8/test_readout.json`), the selected E14 BamiBERT local result, and the paper target (`benchmarks/paper_targets.csv`) in the same fraction scale. E8 uses the student head (384 hidden) with its NC+LLRD recipe; these are reference systems, not architecture-only controls.

| Task / metric | Student E8 seed42 | BamiBERT local | Paper target | Δ local−E8 (pp) |
|---|---:|---:|---:|---:|
| VSFC sentiment accuracy | 0.926721 | 0.935250 | 0.938600 | +0.853 |
| VSFC sentiment macro F1 | 0.812141 | 0.830330 | 0.834100 | +1.819 |
| VSFC topic accuracy | 0.891661 | 0.885976 | 0.893400 | −0.569 |
| VSFC topic macro F1 | 0.793788 | 0.781766 | 0.799000 | −1.202 |
| ViSFD detection micro F1 | 0.879698 | 0.893404 | 0.891400 | +1.371 |
| ViSFD pair micro F1 | 0.817502 | 0.845017 | 0.842400 | +2.751 |
| ABSA restaurant detection micro F1 | 0.855752 | 0.871216 | 0.880100 | +1.546 |
| ABSA restaurant pair micro F1 | 0.715203 | 0.741493 | 0.748900 | +2.629 |
| ABSA hotel detection micro F1 | 0.788764 | 0.801168 | 0.799900 | +1.240 |
| ABSA hotel pair micro F1 | 0.726164 | 0.739398 | 0.726500 | +1.323 |

Paper ABSA metrics and F1 averaging remain unresolved against this local evaluator; paper values are contextual targets only.

The 15 trials and locked test evaluations retain the frozen training source hash `bba42b8eb93dc2945fd49495bc48490daa5474e855e31b3cb261d96dfc089645`. After validation, the controller’s CSV bookkeeping header was corrected; `artifacts/e14/postrun_validation.json` records this report-only source-hash change and confirms all checkpoint, ID, gold, count, and metric checks.

## Scope and interpretation

ABSA labels in this E14 readout use the requested name overlay 0=absent, 1=positive, 2=negative, 3=neutral while preserving the frozen integer gold codes. ViSFD OTHERS raw code 4 remains detection-only and pair-excluded.

Paper values are retained only as contextual references in `benchmarks/paper_targets.csv`; this report does not relabel local metrics as paper reproduction. A single seed has no stable confidence interval and is not a SOTA claim.

E8 summary artifact available for clearly labeled student reference: `artifacts/e8/summary.json`; E14 does not rewrite or depend on E8 checkpoints. E8 summary keys present: `['baselines', 'created_at', 'e1_dependency_source_hash', 'e2_dependency_source_hash', 'e7_dependency_source_hash', 'e8_source_hash', 'metric_recomputation', 'protocol', 'rows', 'seed', 'status', 'tasks']`.
