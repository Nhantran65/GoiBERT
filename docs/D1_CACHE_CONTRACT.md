# D1 diagnostic cache contract

This contract is shared by the feature extractor, frozen linear probes, and
CKA analysis.

## Scope and inputs

- Tasks: `vsfc_sentiment`, `vsfc_topic`, `visfd`, `absa_restaurant`, and
  `absa_hotel`.
- Only the loader's `train` and `validation` splits are opened. Test is never
  loaded.
- For each task, exactly `min(2000, N_train)` train rows are selected with
  `np.random.default_rng(42).choice(N_train, size=..., replace=False)` and
  sorted back into source order. Validation uses every row in source order.
  The two VSFC tasks therefore share the same selected indices and texts.
- One tokenized input is made with the local student tokenizer (the student
  and DAPT teacher tokenizer files are checked to have identical hashes) and
  passed unchanged to both encoders. Max length is 256, truncation is enabled,
  and batch padding is dynamic.

## Arrays

Each model directory contains four atomic NumPy files, all `float32`:

```
artifacts/d1/cache/<task>/<model>/train_cls.npy
artifacts/d1/cache/<task>/<model>/validation_cls.npy
artifacts/d1/cache/<task>/<model>/train_mean.npy
artifacts/d1/cache/<task>/<model>/validation_mean.npy
```

Shapes are `[L, N, H]`, with layer 0 the embedding output and layers 1..L-1
the transformer outputs. Student is `[11, N, 384]`; teacher is
`[13, N, 768]`. `*_cls` is the first (`<s>`) token. `*_mean` is the mean of
attention-valid, non-special tokens; rows with no such token fall back to CLS.

## Metadata

`artifacts/d1/cache/<task>/dataset.json` records the selected source indices,
stable row IDs, raw labels, model-space labels, class/aspect schema, support
counts, source file hashes, and the tokenized input fingerprint for train and
validation. It also records `include_test: false`.

Each model's `manifest.json` is written only after all four arrays exist and
records `status: "completed"`, model/checkpoint and tokenizer hashes, source
hashes, array shapes/dtypes/hashes, precision/backend, batch size, and mean
pooling empty-row counts. A missing or non-completed manifest means that model
cache is not consumable.

The local teacher is `models/goibert-b-dapt`, a DAPT derivative of BamiBERT;
it must not be described as the original BamiBERT checkpoint.
