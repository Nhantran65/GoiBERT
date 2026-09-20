# E6 — student MLM continuation

E6 continued masked-language-model training from the original
`models/goiturn-s-gd` encoder. The encoder architecture remained 10 layers,
hidden size 384, six attention heads, and FFN size 1536. A standard
`RobertaLMHead` was attached only for MLM and its decoder weight was tied to
the student word embedding. The MLM head is discarded from the exported
backbone.

The corpus contains only train text from VSFC sentiment, ViSFD, UIT-ABSA
Restaurant, and UIT-ABSA Hotel. VSFC topic was excluded because it repeats the
VSFC sentence text. Texts were stripped at the outer boundary and globally
deduplicated case-sensitively, while every retained source membership ID is
stored in `artifacts/e6/mlm/corpus.jsonl`. There are 33,414 unique texts:
31,716 MLM-train rows and 1,698 deterministic train-only holdout rows. No
labels are used. The source manifest and hashes are in
`artifacts/e6/mlm/corpus_manifest.json`.

The fixed protocol uses seed 42, maximum length 256, dynamic padding, 15%
masking with 80/10/10 replacement, and protection for every original special
token. The fixed holdout uses a separate seed 1729 and is evaluated by total
masked-token loss. Because a very short row can otherwise have no selected
token, 148 holdout rows received one forced eligible mask; the resulting rate
is 6,618 / 42,890 = 15.43%. This holdout is a diagnostic only, not an
official validation split.

Stage A trained only the new MLM head for 500 optimizer steps at 5e-4 while
the encoder stayed in eval mode and frozen. The encoder hash before and after
Stage A is identical. Stage B trained encoder and head jointly for exactly
2,000 optimizer steps at 2e-5, effective batch size 32, BF16/SDPA, linear
10%-warmup/decay, and gradient clipping at 1.0.

Fixed holdout MLM loss:

| Checkpoint | Loss |
|---|---:|
| Initial new head | 10.2318 |
| After head warmup (500) | 6.8234 |
| Joint step 500 | 5.3088 |
| Joint step 1,000 | 4.8243 |
| Joint step 1,500 | 4.6090 |
| Joint step 2,000 | 4.5422 |

The encoder-only export is `models/goiturn-s-gd-mlm-e6`; the handoff marker is
`artifacts/e6/mlm/READY_FOR_FINETUNE.json`, and the complete runtime and
provenance record is `artifacts/e6/mlm/mlm_metadata.json`. The MLM loss is a
training diagnostic; task-level improvement still requires the planned
five-task seed-42 fine-tuning pilot.

The export initially serialized `tokenizer.json` differently even though its
vocabulary was unchanged. The serialized copy is preserved under
`artifacts/e6/mlm/tokenizer_serialized_by_export.json`; the export now carries
byte-exact copies of the three original tokenizer files, with their hashes
recorded in the handoff metadata.
