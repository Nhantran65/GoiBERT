# Intent benchmark preparation

This directory contains the offline extraction, AI-assisted label audit, duplicate evidence, and grouped split for the standalone `prompt-bench-v1` intent benchmark. It does not modify the source bundle or the GoiBERT research artifacts. No model inference, API call, training, or fixture execution is used.

The source is `tmp/prompt-bench-v1-250-standalone/scripts/ai-verify/packs/prompt-bench-v1/cases.jsonl`, bundle version `2.0.5-g6.8`. The source has 250 cases and 512 target turns: 507 labeled turns and 5 unlabeled turns. The five unlabeled turns are listed in `missing_exclusion_manifest.json`; they are not targets, but the extraction logic permits any preceding source turn to remain history. The source file SHA-256 and the bundle manifest SHA-256 are recorded in `provenance.json`.

The eight labels and their source order are frozen in `label_map.json`. The checked-in classifier taxonomy and prompt excerpt were read from `src/modules/ai/ai.schema.ts` and `src/modules/ai/docs/technicals/prompts_doc/production/INTENT_CLASSIFIER_PROMPT.md`. The live prompt is DB-managed and its production row is absent from this snapshot, so the review is an AI-assisted audit against the checked-in taxonomy and documented rubric, not a human gold set or a byte-level reconstruction of the label-generation prompt. The final review decisions are in `ROOT_LABEL_REVIEW_DECISIONS.md`.

Each `corrected_dataset.jsonl` row has a stable `sample_id` (`case_id:t<zero-based-turn-index>`), source provenance, a `model_input`, and a reviewed target label. `model_input.messages` contains only the current customer message plus preceding source customer turns from the same case. No assistant messages are invented. `expectedIntent`, `assertions`, fixture state, `primaryIntent`, source IDs, and future turns are outside the model input. `review_ledger.jsonl` keeps the original label, proposed/reviewed label, confidence, rationale, uncertainty reason, source text, and preceding history. `original_turns.jsonl` is the unmodified extraction view, including the five excluded turns.

`review_changes.csv` is the compact 19-row label-change sheet. `review_uncertain.csv` is the 82-row uncertainty audit export with current text and serialized preceding history; these rows remain in the data and have already been reviewed. `label_map.json` also provides stable integer IDs in source enum order for downstream consumers.

The audit reviewed all 507 labeled turns. It changes 19 reviewed labels and flags 82 turns as uncertain; uncertain turns remain in the benchmark. Original and reviewed counts and the full cross-tab are in `provenance.json` and `label_crosstab.json`. This is label review, not model evaluation.

The anonymized `[PHONE_n]` and `[ADDRESS_n]` placeholders are intentionally preserved. They make PHONE/ADDRESS shortcuts possible and do not support claims about real-world phone/address generalization. The source snapshot has no assistant replies or top-level context, so the model inputs contain only the available preceding customer turns. `buffered_burst` cases can merge or omit runtime classifier observations; this offline benchmark preserves source turn boundaries and labels rather than claiming live observation alignment.

Duplicate analysis applies NFC normalization, case folding, and whitespace collapsing. It reports 22 repeated current-text families, four exact full-input families, and 52 fuzzy candidates with similarity and token-Jaccard evidence in `duplicate_analysis.json`. Repeated generic acknowledgments and the 26 `giao [ADDRESS_1]` turns are reported separately. Root reviewed all fuzzy candidates and confirmed nine substantive whole-case template families; those families are recorded with rationales and joined before splitting. The remaining candidates are marked `report_only_root_reviewed` with reasons. Current-text overlap across splits is intentionally reported and is not described as text-unique or leak-free.

`train.jsonl`, `dev.jsonl`, and `test.jsonl` use seed 42 and target approximately 70/15/15 by sample count. Case, source group, exact full-input duplicate, and root-confirmed template-family boundaries are indivisible. The final sizes are 353/77/77; all eight classes occur in every split. `split_report.json`, `split_audit.json`, `split_assignments.jsonl`, and `group_manifest.json` record class counts, grouping, overlap, and assertions. The largest connected group has 16 targets, so no giant-group infeasibility was encountered. The split still has repeated current-text overlap by design; exact full-input duplicate overlap across splits is zero.

Recreate the artifacts with the standard library only:

```bash
python3 intent_benchmark/prepare_intent_benchmark.py
python3 intent_benchmark/finalize_intent_benchmark.py --seed 42
```

The scripts assert the 250/512/507/5 counts, stable IDs, model-input field exclusions, history shape, case/source disjointness, grouped duplicate isolation, and all-eight-class split support. A separate output directory can be supplied to both scripts (`prepare_intent_benchmark.py --output /tmp/intent_benchmark_repro` and `finalize_intent_benchmark.py --output /tmp/intent_benchmark_repro`) for reproducibility checks.
