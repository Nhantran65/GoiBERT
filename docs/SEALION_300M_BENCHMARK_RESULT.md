# SEA-LION ModernBERT 300M benchmark result

Protocol: `SEA-LION-ModernBERT-300M-five-task-v1`; seed `42`; validation-only model selection; five frozen GoiBERT tasks. The benchmark ran the requested two SEA-LION learning rates (`2e-5`, `5e-5`) for 30 epochs and reused the completed CafeBERT restaurant pilot while adding four fixed-`2e-5` CafeBERT runs.

The result is a local benchmark on the audited loader and evaluator. It is not an exact paper reproduction, a significance test, or a global SOTA claim.

## Scope and frozen recipe

| Item | Value |
|---|---|
| Tasks | `vsfc_sentiment`, `vsfc_topic`, `visfd`, `absa_restaurant`, `absa_hotel` |
| Seed | 42 |
| Epochs | 30 |
| Effective batch | 32; micro batch 32 |
| Maximum length | 256 |
| Loss | Plain cross entropy; absent class included |
| Optimizer | AdamW, weight decay 0.01, betas 0.9/0.999, epsilon 1e-8 |
| Schedule | 10% linear warmup, linear decay |
| Gradient clipping | 1.0 |
| Precision/attention | BF16 autocast, SDPA |
| Head | first token → dropout 0.1 → dense hidden → tanh → dropout 0.1 → task logits |
| Selection | Maximum validation `selection_f1`; ties use smaller LR then earlier epoch |

SEA-LION uses the local snapshot of `aisingapore/SEA-LION-ModernBERT-300M` at commit `e3d70b810769bf86ae9041d1943e0ecd10b62b31`. Its tokenizer emits `<bos>` ID 2 as the first token, `<eos>` ID 1, and `<pad>` ID 0; it emits no `token_type_ids`. The adapter therefore uses a first-token/BOS head. The checkpoint config reports native `classifier_pooling: mean`, but that native masked-LM classifier is not the downstream head used here. This is a shared first-token comparison and does not test native mean pooling.

The ViSFD head uses widths `(4,4,4,4,4,2,4,4,4,4,4)`. OTHERS is aspect index 5 and contributes to detection but is excluded from pair scoring. ABSA uses the requested report overlay `0=absent, 1=positive, 2=negative, 3=neutral`; the frozen source integer labels and evaluator were preserved without recoding.

Preflight passed for all five SEA-LION task shapes, finite logits/loss/gradients, exact validation IDs, and a checkpoint/ID serialization round trip. See [`artifacts/sealion_benchmark/preflight.json`](../artifacts/sealion_benchmark/preflight.json).

## Downstream parameter counts

Counts include the encoder and task head, so the small task-dependent differences are real output-head differences. BamiBERT is the original E14 local baseline; CafeBERT is the XLM-R-large CafeBERT encoder with the 1024-wide shared head.

| Task | SEA-LION parameters | CafeBERT parameters | BamiBERT parameters |
|---|---:|---:|---:|
| VSFC sentiment | 312,251,139 | 559,893,507 | 102,953,475 |
| VSFC topic | 312,251,908 | 559,894,532 | 102,954,244 |
| ViSFD | 312,281,130 | 559,933,482 | 102,983,466 |
| ABSA restaurant | 312,285,744 | 559,939,632 | 102,988,080 |
| ABSA hotel | 312,353,416 | 560,029,832 | 103,055,752 |

## Validation selection

The table labels the LR search explicitly: BamiBERT is tuned on the E14 3-LR grid, SEA-LION is tuned on the requested 2-LR grid, and CafeBERT is fixed at `2e-5` (restaurant is the completed CafeBERT pilot).

| Task / selection metric | BamiBERT tuned 3-LR | CafeBERT fixed 2e-5 | SEA-LION tuned 2-LR |
|---|---:|---:|---:|
| VSFC sentiment / fixed-label macro F1 | 0.863252 @ 2e-5, ep 8 | 0.876924 @ 2e-5, ep 27 | 0.822646 @ 5e-5, ep 4 |
| VSFC topic / fixed-label macro F1 | 0.795328 @ 1e-5, ep 13 | 0.817446 @ 2e-5, ep 23 | 0.775946 @ 5e-5, ep 9 |
| ViSFD / end-to-end pair micro F1 | 0.843834 @ 5e-5, ep 23 | 0.854665 @ 2e-5, ep 19 | 0.794274 @ 5e-5, ep 14 |
| ABSA restaurant / end-to-end pair micro F1 | 0.747418 @ 5e-5, ep 29 | 0.777197 @ 2e-5, ep 21* | 0.664442 @ 5e-5, ep 7 |
| ABSA hotel / end-to-end pair micro F1 | 0.732362 @ 5e-5, ep 24 | 0.733438 @ 2e-5, ep 15 | 0.713469 @ 5e-5, ep 12 |

`*` CafeBERT restaurant is reused from `artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42`; it was not rerun.

All 10 SEA-LION runs and all four new CafeBERT runs finished with successful checkpoint reload, validation metric, prediction-ID, and prediction-file checks. Machine-readable per-run curves, metadata, checkpoints, and predictions are under [`artifacts/sealion_benchmark/runs`](../artifacts/sealion_benchmark/runs) and [`artifacts/sealion_benchmark/cafe_runs`](../artifacts/sealion_benchmark/cafe_runs). The frozen LR/epoch/checkpoint manifest is [`artifacts/sealion_benchmark/selection_manifest.json`](../artifacts/sealion_benchmark/selection_manifest.json).

## Locked test evaluation

After the validation manifest was written, the selected checkpoint for each model/task was evaluated once on test. Test data was not used for LR or epoch selection. The same local evaluator was used for both families.

| Task | SEA-LION test | CafeBERT test |
|---|---:|---:|
| VSFC sentiment / accuracy; macro F1 | 0.916930; 0.778983 | 0.936829; 0.823567 |
| VSFC topic / accuracy; macro F1 | 0.874921; 0.772947 | 0.894504; 0.807678 |
| ViSFD / detection micro F1; pair micro F1 | 0.861599; 0.794204 | 0.898346; 0.854828 |
| ABSA restaurant / detection micro F1; pair micro F1 | 0.815376; 0.650634 | 0.884028; 0.773902 |
| ABSA hotel / detection micro F1; pair micro F1 | 0.779012; 0.712055 | 0.804307; 0.751742 |

For context, the locked E14 BamiBERT test readout reports sentiment macro F1 `0.830330`, topic macro F1 `0.781766`, ViSFD pair micro F1 `0.845017`, restaurant pair micro F1 `0.741493`, and hotel pair micro F1 `0.739398`. These are reused historical E14 outputs, not rerun in this benchmark.

Full test metrics and row-level predictions are under [`artifacts/sealion_benchmark/test`](../artifacts/sealion_benchmark/test), with the final test status recorded in [`artifacts/sealion_benchmark/test/selection_manifest.json`](../artifacts/sealion_benchmark/test/selection_manifest.json). The test pass took 71.18 seconds on one GPU and verified exact test IDs for every task.

## Compute and provenance

- SEA-LION: 10 fine-tunes, summed recorded run elapsed time **3.915 GPU-hours**.
- New CafeBERT runs: 4 fine-tunes, summed recorded run elapsed time **1.777 GPU-hours**.
- Reused CafeBERT restaurant pilot: historical recorded elapsed time **0.389 GPU-hours**, not rerun here.
- New fine-tune total: **5.691 GPU-hours**; locked test pass: **0.020 GPU-hours**.
- The durable controller used six authorized worker slots at peak: GPUs 0/1 one each and GPUs 2/3 two each, with memory guards for shared SEA-LION and solo CafeBERT placement. Existing unrelated jobs on GPUs 0/1 were left intact.

The SEA-LION source digest in each run manifest covers the benchmark adapter and runner (`scripts/sealion_benchmark/*.py`) plus the frozen E1 loader, evaluator, optimizer, and E14 head source. Model and tokenizer file hashes, HF commits, runtime, GPU snapshots, checkpoint hashes, and per-run source digests are recorded in each `resolved_config.json`/`metadata.json`.

## Interpretation

On this seed and recipe, CafeBERT is higher on all five validation and locked-test task selections. SEA-LION is below the tuned BamiBERT validation values on all five tasks in this local setup, while using roughly three times the BamiBERT downstream parameter count and a different pretraining/model family. The results support this benchmark’s concrete comparison only; they do not isolate pretraining, pooling, tokenizer, or model-size effects and do not establish statistical significance.
