# E3 — Polarity auxiliary supervision, seed42 screening

Approved by user on 2026-09-09: triển khai proposed loss-only experiment. Root coordinates and interprets; e3_implement_run implements/runs; e3_review independently reviews before full training and audits outputs afterward.

## Fixed hypothesis and scope

Giving explicit additional supervision to polarity on gold-present aspects improves student end-to-end pair F1. Exactly two full trials: ViSFD and UIT-ABSA Restaurant, seed42. No architecture change. No corpus, teacher, KD, NC, test access, additional coefficients or seeds in this batch.

Loss = original E1 CE + 0.5 × auxiliary CE. Auxiliary CE pools all gold-present, polarity-bearing cells in the batch. Each contributes CE over logits1:4 with label gold-1. Exclude binary ViSFD OTHERS (slot5); absent cells excluded only from auxiliary term. With no valid cells return differentiable zero. Do not average per-slot means or include absent logit in conditional softmax. Dataset polarity semantics differ but native index ordering remains intact.

## Controlled training contract

- Fresh models/goiturn-s-gd encoder and E1 head per trial; no warm start from fine-tuned checkpoint.
- LR1e-4,30epochs,effective batch32,microbatch32,max_length256,BF16,SDPA.
- Original E1 AdamW parameter grouping/betas/epsilon/weight decay,10%linear warmup and decay,clip1,seed initialization,shuffle and evaluation policy.
- Original dataset splits/rows/IDs. Loader include_test=False. Select checkpoint by validation pair micro F1; preserve E1 tie-break rule.
- Original output classes and argmax decoding. Auxiliary loss changes training only.
- Frozen E1/E2 source/configs and existing model artifacts must remain unchanged. Isolate E3 code/config and record hashes.

## Execution and gates

1. Worker implements minimal isolated runner and mathematically meaningful loss checks (pooling, absence, OTHERS,gradient,unchanged CE).
2. Reviewer compares training settings and loss implementation with approved contract. Worker may smoke/profile before review but starts full training only after explicit reviewer/root PASS.
3. Worker uses host CUDA; prefer GPU2 with >=5GiB free. GPU0 may run the other task if available. Do not kill other workloads. A sandbox CUDA/spawn failure is not evidence of host failure.
4. Durable detached execution with file logs/status/checkpoints; inspect actual PID and epoch progress, not stale status alone. Retain failed attempts and do not silently retry different seeds/settings. Exactly two successful full trials.
5. Worker returns selected checkpoints, epochs, validation metrics/predictions,losscomponents,precision/device/memory and provenance. Reviewer independently verifies metrics from predictions and source hashes.

## Readout and decision

Compare each E3 seed42 to E1 and NC seed42 (ViSFD lambda0.5; Restaurant lambda0.1), not previous three/five-seed means. Report pair F1, detection F1/P/R,missed/spurious aspects,wrong polarity; exclude OTHERS from conditional polarity denominator. Original diagnostic scripts/artifacts identify locked baseline paths.

Predeclared screening signal per task: >=0.3 percentage points pair F1 over E1 seed42. Also report NC delta; matching an inferior baseline alone is not sufficient practical gain. Both tasks must improve to describe a common positive signal. This is screening, not statistical significance. No automatic multi-seed launch; root decides next step from evidence. If both fail, stop this loss branch and reconsider architecture or another mechanism.

Final deliverables: isolated code/config/tests, artifacts/e3 outputs and comparison, docs/E3_IMPLEMENTATION.md, docs/E3_REVIEW.md, root docs/E3_VALIDATION_READOUT.md. User receives completed result or precise blocker; merely launching jobs is not completion.
