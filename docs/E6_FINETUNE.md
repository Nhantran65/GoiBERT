# E6 fine-tune protocol

E6 tests one shared MLM-backbone hypothesis. The student architecture and
supervised recipe remain E1: the exported `models/goiturn-s-gd-mlm-e6`
encoder is loaded, a fresh CLS head is attached with the same RNG sequence,
and the full model is fine-tuned with cross entropy.

The screening runs seed 42 for all five tasks, using learning rate `1e-5` for
VSFC sentiment/topic and `1e-4` for ViSFD, Restaurant, and Hotel. Each trial
uses 30 epochs, effective and micro batch size 32, max length selected by the
E1 train-only audit, BF16, SDPA, and no test split. GPU workers use devices 2,
1, and 3 when at least 5 GiB is free; workers do not terminate unrelated
processes.

The controller refuses to launch until the final MLM export exists and its
configuration matches `goiturn-s-gd` on hidden size, layer count, head count,
FFN width, vocabulary size, and maximum position embeddings. Every trial
records the E6 directory hashes and the E6 source/config hashes. E1–E5
artifacts and source files are preserved.

The downstream adapter hash is recorded separately from the MLM worker hash;
the latter is read from the export manifest and stored as `mlm_source_hash`.

The export contract uses `models/goiturn-s-gd-mlm-e6/e6_mlm_manifest.json`
when the MLM worker publishes its final encoder. The current worker publishes
the equivalent completed manifest at `artifacts/e6/mlm/mlm_metadata.json`,
which the validator accepts because it is written after the encoder export.
The validator also accepts the existing fallback manifest names used by
earlier local runners.
That manifest should expose the MLM worker source hash under
`mlm_source_hash` (the validator also recognizes the legacy `source_hash` or
`mlm_code_hash` keys).

The result is an exploratory validation readout. Multi-seed confirmation,
test evaluation, corpus expansion, and combinations with KD/NC are outside
this screening protocol.
