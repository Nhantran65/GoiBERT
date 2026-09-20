# E1 test evaluation and BamiBERT comparison

Generated: `2026-09-08T07:38:19.644036+00:00`. The locked E1 test gate evaluated **25/25 checkpoints** exactly once: five tasks × seeds 42–46. All runs used inference only with BF16/SDPA; no checkpoint, learning rate, or preprocessing choice was changed after test access.

The table reports GoiBERT mean ± sample standard deviation over five selected seeds, in percentage points. BamiBERT values are the paper targets recorded in `benchmarks/paper_targets.csv`. Arithmetic deltas are shown for orientation; F1 rows are not treated as proof of a win because the paper does not publish enough evaluator detail to establish metric equivalence.

| Task | Metric | BamiBERT paper | GoiBERT test mean ± SD | Arithmetic Δ | Comparability |
|---|---|---:|---:|---:|---|
| VSFC sentiment | Accuracy | 93.86 | 92.21 ± 0.29 | -1.65 | resolved for Accuracy |
| VSFC sentiment | Macro F1 | 83.41 | 80.60 ± 1.04 | -2.81 | unresolved |
| VSFC topic | Accuracy | 89.34 | 88.49 ± 0.25 | -0.85 | resolved for Accuracy |
| VSFC topic | Macro F1 | 79.90 | 78.62 ± 0.82 | -1.28 | unresolved |
| ViSFD | Aspect detection micro F1 | 89.14 | 86.86 ± 0.20 | -2.28 | unresolved |
| ViSFD | End-to-end aspect-polarity pair micro F1 | 84.24 | 79.83 ± 0.50 | -4.41 | unresolved |
| UIT-ABSA Restaurant | Aspect detection micro F1 | 88.01 | 84.32 ± 0.30 | -3.69 | unresolved |
| UIT-ABSA Restaurant | End-to-end aspect-polarity pair micro F1 | 74.89 | 69.08 ± 0.60 | -5.81 | unresolved |
| UIT-ABSA Hotel | Aspect detection micro F1 | 79.99 | 78.64 ± 0.25 | -1.35 | unresolved |
| UIT-ABSA Hotel | End-to-end aspect-polarity pair micro F1 | 72.65 | 71.78 ± 0.32 | -0.87 | unresolved |

## Interpretation

On the explicitly comparable VSFC Accuracy rows, the supervised GoiBERT baseline is below the paper values: 92.21 ± 0.29 versus 93.86 for sentiment, and 88.49 ± 0.25 versus 89.34 for topic. The internal F1 values and all ABSA values are useful measurements of this pipeline, but they remain provisional for a direct BamiBERT ranking until the paper evaluator and the UIT-ABSA mirror lineage are resolved.

The selected checkpoints are the best-validation checkpoints from the frozen supervised E1 runs. Test rows were loaded only in the locked evaluator. The backbone inherits the prior corpus exposure caveat recorded in the E0 audit; the current test run itself did not train on test rows.

## Provenance

- Selection manifest: `/home/kiettqa/GoiBERT/artifacts/e1/test_selection_manifest.json` (SHA-256 `bd8e8855fc3507c3c88ed8797671ca7ed9f1edc246e0363ab17e398ea98c82c4`; filesystem mtime `2026-09-08T07:34:03.849315+00:00`).
- The manifest's declared creation field is retained as provenance; the filesystem mtime above is the observed seal time used for this audit.
- Test plan index (dry run, no inference): `/home/kiettqa/GoiBERT/artifacts/e1/test_index_20260908T073435Z.json` (SHA-256 `da7ab446cfdf4215631ed1f774999c31ce2b677621840b79fa6341b6957ef2f3`) if present.
- Executed test index: `/home/kiettqa/GoiBERT/artifacts/e1/test_index_20260908T073518Z.json` (SHA-256 `fcca59e8426d9b9e6e642f6796f5fbd82285bccac64610ee9e045d013e7c0847`).
- Access log: `/home/kiettqa/GoiBERT/artifacts/e1/test_access_log.json` (SHA-256 `b3a97d6c4096696a4e14517b2e41de067a9da24e8d6dbfbf1f6f09a67ed9891b`).
- Source/evaluator/backbone/tokenizer hashes are retained per seed in `test_readout.json` and each checkpoint's `test_evaluation_metadata.json`.
- Per-checkpoint predictions and metrics remain beside the locked checkpoint under `artifacts/e1/.orchestrator/attempts/`.
