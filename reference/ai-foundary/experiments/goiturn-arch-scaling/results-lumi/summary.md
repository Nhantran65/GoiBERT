# GoiTurn Architecture Scaling Results (LUMI run, 2026-08-30, job 21625380)

Source: /projappl/project_462001285/experiments/goiturn-arch-scaling/results/ on LUMI.
Copied from terminal output; full per-run JSONs remain on LUMI (289 days retention).
Run conditions: 1× MI250X GCD, torch 2.5.1+rocm6.2, fp32 (AMP off), eager attention,
seed 42, batch 32, 10 epochs, lr 2e-5 (baseline) / 5e-5 (variants), dynamic padding,
max_length 256. loss=nan count in job log: 0.

| Dataset | Model | Mode | Layers | Hidden | Heads | FFN | Params | F1 | Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| vsfc | baseline | finetune | 12 | 768 | 12 | 3072 | 102,360,576 | macro 0.8274 | 0.9337 |
| vsfc | variant_a | scratch | 8 | 512 | 8 | 2048 | 35,970,048 | macro 0.7291 | 0.8932 |
| vsfc | variant_a | distill | 8 | 512 | 8 | 2048 | 35,970,048 | macro 0.7718 | 0.8970 |
| vsfc | variant_b | scratch | 10 | 384 | 6 | 1536 | 25,807,872 | macro 0.7554 | 0.8891 |
| vsfc | variant_b | distill | 10 | 384 | 6 | 1536 | 25,807,872 | macro 0.7479 | 0.8945 |
| vlsp_restaurant | baseline | finetune | 12 | 768 | 12 | 3072 | 102,360,576 | det 0.7707 / cls 0.6070 | — |
| vlsp_restaurant | variant_a | scratch | 8 | 512 | 8 | 2048 | 35,970,048 | det 0.7236 / cls 0.5772 | — |
| vlsp_restaurant | variant_a | distill | 8 | 512 | 8 | 2048 | 35,970,048 | det 0.7309 / cls 0.5799 | — |
| vlsp_restaurant | variant_b | scratch | 10 | 384 | 6 | 1536 | 25,807,872 | det 0.6952 / cls 0.5421 | — |
| vlsp_restaurant | variant_b | distill | 10 | 384 | 6 | 1536 | 25,807,872 | det 0.7102 / cls 0.5609 | — |
| visfd | baseline | finetune | 12 | 768 | 12 | 3072 | 102,360,576 | det 0.8832 / cls 0.7887 | — |
| visfd | variant_a | scratch | 8 | 512 | 8 | 2048 | 35,970,048 | det 0.8231 / cls 0.6952 | — |
| visfd | variant_a | distill | 8 | 512 | 8 | 2048 | 35,970,048 | det 0.8329 / cls 0.6996 | — |
| visfd | variant_b | scratch | 10 | 384 | 6 | 1536 | 25,807,872 | det 0.8153 / cls 0.6893 | — |
| visfd | variant_b | distill | 10 | 384 | 6 | 1536 | 25,807,872 | det 0.8248 / cls 0.6963 | — |

F1 for ABSA rows is micro detection / micro classification; VSFC is macro F1.
Best epochs: baseline 8/8/10; variants 5-10 (all selected on validation).

## CPU latency (LUMI login-class AMD EPYC, batch 1, fp32, eager, 30 warmup + 300 iters)

| Model | Seq | Threads | p50 ms | p95 ms | fp32 MiB |
|---|---:|---:|---:|---:|---:|
| baseline | 32 | 1 | 92.876 | 95.650 | 392.82 |
| baseline | 32 | 4 | 38.295 | 47.255 | 392.82 |
| baseline | 128 | 1 | 281.003 | 303.829 | 392.82 |
| baseline | 128 | 4 | 103.504 | 104.326 | 392.82 |
| variant_a | 32 | 1 | 30.111 | 30.392 | 138.28 |
| variant_a | 32 | 4 | 13.535 | 13.851 | 138.28 |
| variant_a | 128 | 1 | 88.086 | 88.572 | 138.28 |
| variant_a | 128 | 4 | 34.681 | 35.755 | 138.28 |
| variant_b | 32 | 1 | 23.359 | 25.026 | 99.08 |
| variant_b | 32 | 4 | 11.244 | 13.413 | 99.08 |
| variant_b | 128 | 1 | 65.650 | 70.888 | 99.08 |
| variant_b | 128 | 4 | 26.600 | 27.966 | 99.08 |

## Sanity anchors (protocol差: ours 10 epochs × 1 lr; paper 30 epochs × 3-lr grid)

- UIT-VSFC sentiment: measured baseline macro F1 82.74 / acc 93.37 vs BamiBERT paper 83.41 / 93.86.
- UIT-ViSFD: measured baseline detection F1 88.32 vs BamiBERT paper 89.14.

## Wall-clock train seconds (sum ≈ 24 min GPU)

baseline: vsfc 255, vlsp 216, visfd 316. variants: 81-132 each.
