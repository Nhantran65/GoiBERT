# D2 after-fine-tuning representation cache

This cache contains the layer-wise representations of the **selected task
checkpoints after fine-tuning**. It is the input for the D2 before/after
analysis and does not contain any test representations.

The extractor is [scripts/d2_extract_after_features.py](/home/kiettqa/GoiBERT/scripts/d2_extract_after_features.py). It reuses the frozen D1 preparation and extraction helpers. Before writing a cache it reconstructs the D1 cohort and requires matching source indices, IDs, text hashes, labels, tokenized input hashes, tokenizer metadata, and `include_test: false`.

Each task/model directory contains four `float32` arrays with the D1 layout:

```text
artifacts/d2/cache/<task>/<student|teacher>/
  train_cls.npy          [layers, 2000, hidden]
  validation_cls.npy     [layers, N_validation, hidden]
  train_mean.npy         [layers, 2000, hidden]
  validation_mean.npy    [layers, N_validation, hidden]
  weight_drift.json
  manifest.json
```

The student has 11 states (`embedding + 10` transformer layers, width 384);
the teacher has 13 states (`embedding + 12` transformer layers, width 768).
CLS is the first token. Mean pooling uses attention-valid non-special tokens
with CLS fallback for an empty row. The D1 tokenizer, max length 256,
dynamic padding, BF16 and SDPA settings are retained.

The student source is the selected E1 CE seed-42 checkpoint for each task.
The teacher source is the locked E2 stage0 teacher seed-42 checkpoint for each
task. The loader reads `checkpoint["model"]`, removes only the `encoder.`
prefix, and strict-loads those weights into the matching original
`RobertaModel`; head and optimizer entries are discarded. Every manifest
records the exact fine-tuned checkpoint path/SHA, metadata path/SHA, learning
rate, best epoch, base-backbone hashes, and strict-load result.

`weight_drift.json` is a CPU-only audit made while both states are loaded. It
reports per-parameter and grouped relative L2 changes for embeddings and each
transformer layer's attention, FFN, LayerNorm and other parameters. It is a
diagnostic signal, not a causal claim about a bottleneck.

The first extraction grouped `attention.output.dense` under FFN because that
name also contains `output.dense`. The per-parameter norms and representations
were correct; the grouped summaries were repaired from those saved norms using
`scripts/d2_repair_weight_drift.py`. The original grouped files remain as
`weight_drift.json.pre_group_fix_20260910.json`, and manifests record both the
backup hash and repair-script hash. Current files use
`d2-weight-drift-v2`: attention is classified by the `attention.` prefix before
the FFN `output.dense` prefix. No checkpoint was reloaded and no array was
re-extracted during this repair.

The ten completed caches are summarized in
`artifacts/d2/cache/d2_extraction_summary.json`. The D1 dataset metadata is
copied unchanged into each `artifacts/d2/cache/<task>/dataset.json` so D1
analysis readers can reuse it. The D2 result is conditional on the selected
task-fine-tuned checkpoints and the D1 train/validation cohort; it is not an
independent generalization estimate.
