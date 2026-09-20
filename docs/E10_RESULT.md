# E10 Result — Span residual versus multi-kernel convolution

E10 ran the fixed 10-trial screening: span and convolution variants on five
tasks, seed 42, 30 epochs, E8 NC + E7 LLRD recipe. The source backbone was
`models/goiturn-s-gd`; the branch was inserted after transformer block 2
(index 1). Validation predictions were recomputed from the saved rows and
checked against the frozen validation IDs and gold labels. The test split was
not accessed.

The baseline values below come from each E8
`best_validation_metrics.json` file on the 0–1 scale. Display values are F1 ×
100 and deltas are candidate minus E8 in percentage points.

| Variant | Task | Best epoch | F1 (%) | E8 (%) | Δ E8 (pp) | Parameters | Time (min) | Peak VRAM (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| span | VSFC sentiment | 13 | 85.1454 | 85.1604 | -0.0151 | 27,137,428 | 126.3 | 5.282 |
| span | VSFC topic | 25 | 80.3701 | 80.3678 | +0.0024 | 27,137,813 | 129.1 | 5.113 |
| span | ViSFD | 28 | 82.4028 | 82.4421 | -0.0393 | 27,152,443 | 122.1 | 17.236 |
| span | UIT-ABSA Restaurant | 25 | 71.1176 | 71.2766 | -0.1590 | 27,154,753 | 76.8 | 4.013 |
| span | UIT-ABSA Hotel | 22 | 72.8595 | 73.0491 | -0.1896 | 27,188,633 | 74.4 | 6.128 |
| conv | VSFC sentiment | 13 | 85.1454 | 85.1604 | -0.0151 | 27,138,995 | 94.2 | 5.283 |
| conv | VSFC topic | 25 | 80.4173 | 80.3678 | +0.0495 | 27,139,380 | 99.5 | 5.114 |
| conv | ViSFD | 28 | 82.4955 | 82.4421 | +0.0534 | 27,154,010 | 74.7 | 17.226 |
| conv | UIT-ABSA Restaurant | 30 | 71.3320 | 71.2766 | +0.0555 | 27,156,320 | 62.4 | 4.014 |
| conv | UIT-ABSA Hotel | 30 | 72.7711 | 73.0491 | -0.2780 | 27,190,200 | 65.0 | 6.126 |

The span screening heuristic fails. Span improves over E8 on 1/5 tasks, has
zero gains of at least 0.3 pp, and stays within the -0.3 pp guardrail. It beats
the convolution control on 1/5 tasks and never loses by more than 0.3 pp. This
is a negative screening result for prioritizing content-dependent span routing
in this recipe; it is not evidence that every residual composition design
fails.

The convolution branch uses rank 140 and has 1,567 more parameters than the
rank-128 span branch for each task. Both branches have 24 validated E7 groups;
their added parameters are assigned to `block_01` with learning-rate scale
`0.8**8`, not the head group. The output projection starts at zero, and the
preflight confirmed exact baseline logits, clean/noisy offset recomputation,
gradient flow, and strict checkpoint roundtrip.

Source provenance: E10 adapter hash
`2042bd9744af9b9ed3bb7accd92ffb5ce6f50f9e1e377fc6b3f27b6d22391b87`, E7
dependency hash
`4f586eba100e66574081aac2647018b237f4754642723bf69b35eadae9e930f5`, and
frozen E1/E2 hashes are recorded in `artifacts/e10/source_manifest.json` and
`artifacts/e10/summary.json`. Two controlled pre-training restarts preserved
partial artifacts while improving clean alignment caching; those partial
attempts are excluded from the summary. No test evaluation or follow-up grid
was run.
