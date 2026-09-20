# Guarded cavity key evaluation

This is a CPU synthetic numerical evaluation of the same cavity-key operator. It imports the prior audit's reference implementation and deterministic input constructors. No model, dataset, checkpoint, optimizer, training, or GPU path is used.

Seed: `20260912`. Torch: `2.6.0+cu124`. Source audit unchanged: **True**.

## Guard

Pairs are flagged when `vnorm <= 1e-3*E + 1e-7` or variance/fast score is nonfinite. Flagged pairs use literal cavity construction and `layer_norm` in chunks of at most 64 pairs; unflagged pairs use the factorized fast path. Selected rows are assembled explicitly. Finite gradients were verified on the tested cases; nonfinite overflow recovery and NaN immunity are not verified or claimed.

## Results

- Six prior FP32 diagnostics: guarded elementwise criterion **pass**; raw fast and guarded errors remain in `results.json`.
- Float64 forward parity: **pass** on two well-conditioned and one near-cancellation case.
- Float64 backward parity: **pass** for H0, Q, V, Wo, bo, Wk, bk, gamma, betaLN, and A-logits.

| case | flagged valid/total | fast max abs | fast relative L2 | guarded max abs | guarded relative L2 | guarded elementwise |
|---|---:|---:|---:|---:|---:|:---:|
| random_scale_0.001 | 0/25 | 6.82121e-13 | 1.69088e-07 | 6.82121e-13 | 1.69088e-07 | True |
| random_scale_1 | 0/25 | 9.53674e-07 | 6.90708e-08 | 9.53674e-07 | 6.90708e-08 | True |
| random_scale_1000 | 0/25 | 768 | 1.06758e-07 | 768 | 1.06758e-07 | True |
| near_removed_message | 1/16 | 0.000106812 | 3.89569e-06 | 2.86102e-06 | 1.66995e-07 | True |
| scaled_near_removed_message | 1/16 | 2.79648 | 0.109762 | 1.90735e-06 | 1.09977e-07 | True |
| constant_centered_rows | 16/16 | 0.000259876 | 4.00732e-05 | 0.000173092 | 2.54343e-05 | True |

The original strict FP32 failures are preserved in the fast columns. Guarded columns report the same operator with the bounded-workspace fallback; this does not claim universal numerical stability or guaranteed fast runtime.

Zero-gradient status and per-parameter Float64 errors are recorded in `results.json`.
