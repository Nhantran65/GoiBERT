# E12 label-mapping audit

## Determination

The Restaurant/Hotel polarity names in the GoiBERT loader are reversed relative to the numeric labels in the raw ABSA CSVs and the VLSP preprocessing source:

| Numeric code | Correct meaning | Current `src/goibert/data.py` name |
|---:|---|---|
| 0 | absent | absent |
| 1 | positive | negative |
| 2 | negative | neutral |
| 3 | neutral | positive |

This is a metadata and semantic-label mapping error. The raw integer values were not remapped by the loader, model heads, saved predictions, or metric formulas. Exact numeric pair/detection metrics therefore remain numerically comparable when gold and predictions use the same raw codes. Any human-readable polarity names, polarity-specific support/F1 tables, and diagnostics that derive IDs from `slot_label_names` are mislabeled.

## Evidence

The strongest mapping reference is the public VLSP preprocessing implementation used for these aspect-category CSVs. Its README states that `1` represents Positive, `2` Negative, `3` Neutral, and `0` means the aspect is absent ([mapping statement](https://github.com/ds4v/absa-vlsp-2018#ii-the-vlsp-2018-aspect-based-sentiment-analysis-dataset)). The official VLSP task page independently defines the three polarities and shows the restaurant and hotel annotation semantics ([restaurant examples](https://vlsp.org.vn/vlsp2018/eval/sa), [hotel examples](https://vlsp.org.vn/vlsp2018/eval/sa)).

The checked-in raw copy gives direct sanity checks:

- Restaurant `Train.csv:3` labels “hơi đắt” (`RESTAURANT#PRICES`) as `2`; `Train.csv:4` labels “rất hôi mùi dầu” (`FOOD#QUALITY`) as `2`; these are negative examples.
- Restaurant `Train.csv:5` labels “không ... mùi hôi hải sản” (`FOOD#QUALITY`) as `1`; `Dev.csv:2` labels “cảm giác thân thiện hơn” (`SERVICE#GENERAL`) as `1`; these are positive examples.
- Restaurant `Train.csv:2` marks “Giá 53k size vừa” price/style cells as `3`, and `Dev.csv:5` uses `3` for the mixed/neutral “vị kì kì” description.
- Hotel `Dev.csv:2` labels “thiết kế đẹp, vệ sinh đảm bảo” room-cleanliness and room-design cells as `1`; `Dev.csv:3` labels “số lượng món ăn nhiều hơn thì tốt biết mấy” as `2`; `Test.csv:3` labels a room layout statement as `3`.

The source manifest explicitly describes this checkout as a third-party copy whose numeric provenance was previously unverified (`data/uit_absa_candidate_manifest.json:2-5`). That caveat is why the audit records both raw-file evidence and the external VLSP preprocessing reference rather than relying on the local README alone.

## Code and artifact impact

The incorrect declaration is at [data.py:56](/home/kiettqa/GoiBERT/src/goibert/data.py:56), with the same wrong names propagated into `slot_label_names` and `label_policy` at [data.py:629](/home/kiettqa/GoiBERT/src/goibert/data.py:629). The loader accepts the raw codes unchanged at [data.py:605](/home/kiettqa/GoiBERT/src/goibert/data.py:605); there is no polarity permutation in ingestion. `metrics.py` compares exact integer codes and uses only `>0` for presence, so numeric E8 selection F1, pair F1, missed/spurious counts, and wrong-code counts are unaffected by the names.

The following semantic outputs are affected:

- E8 metadata and any polarity-named support/F1 summaries generated from the loader contract.
- `scripts/analyze_e8_validation_errors.py:212-216` and `:383-387`, which manually assign the same wrong Restaurant/Hotel names.
- `scripts/diagnose_validation_errors.py:372-377`, which emits wrong ABSA class names.
- E11 metadata and synthetic expectations generated from `slot_label_names`.

E12 was rerun after correcting its own human-readable ABSA slot breakdown; its numeric metrics, error events, and raw predictions were unchanged.

## Numeric label path across E1–E11

The audited path is consistent in numeric source-ID space:

| Stage | Evidence | Consequence of wrong names |
|---|---|---|
| Loader/model labels | `TaskDataset.labels_for_model` returns raw ABSA tuples unchanged; only ViSFD OTHERS is adapted (`src/goibert/data.py:320-331`). | No Restaurant/Hotel integer permutation occurs. |
| E1 CE | Training loaders use `model_labels` and preserve raw validation gold (`src/goibert/train.py:521-525`, `:789-800`); CE consumes integer labels directly (`src/goibert/train.py:466-492`). | Loss and learned numeric class IDs are unaffected. |
| E2 KD/NC | E2 uses the same model-label path (`src/goibert_e2/train.py:263-293`) and combines CE/KD/consistency on logits (`:559-590`). | Distillation and consistency are label-name agnostic. |
| E3 | The auxiliary loss explicitly treats source IDs 1–3 as polarity and targets `gold - 1` (`src/goibert_e3/losses.py:1-10`, `:104-135`). | The numeric auxiliary target is correct despite wrong names. |
| E4–E8 | These retain E1/E2 heads/losses; E6 MLM does not consume supervised polarity labels. | No numeric F1 change attributable to metadata names. |
| E9–E11 | E9 uses E1 CE; E10/E11 reuse E2 loaders and CE (`scripts/run_e10_span_residual.py:733-765`, `scripts/run_e11_signed_attention.py:459-467`). | Numeric supervised paths remain consistent; E11 synthetic labels are a separate defect. |
| Decode/evaluator | `metrics.evaluate_task` compares exact integer codes and only maps ViSFD OTHERS (`src/goibert/metrics.py:281-374`). | Exact pair/detection F1 and numeric error decompositions remain valid. |

E2 teacher caches are ordered train logits keyed by source IDs/text hash and slot schema; labels are explicitly forbidden from the cache payload (`src/goibert_e2/cache.py:1-8`, `:104-141`). This means the mapping error does not scramble cache rows or logits. A future metadata correction will, however, change `data_provenance` and source-code hashes used by cache validation (`src/goibert_e2/train.py:359-386`), so existing caches would need a deliberate rebuild or contract migration before rerunning E2-family experiments.

The artifact metadata remains semantically wrong across inherited E1–E11 records, but this is distinct from a numeric training corruption. Do not relabel historical predictions or recompute benchmark F1 by swapping integer codes; repair metadata and any semantic diagnostics, then rerun only diagnostics whose expectations depended on names.

## E11 synthetic diagnostic consequence

The Restaurant/Hotel expectations in [build_e11_diagnostic.py:40](/home/kiettqa/GoiBERT/scripts/build_e11_diagnostic.py:40)–[49](/home/kiettqa/GoiBERT/scripts/build_e11_diagnostic.py:49) hard-code positive as `3` and negative as `1`. The builder passes because it validates against the loader's wrong metadata. The saved E11 inference artifact therefore reports invalid ABSA semantic correctness: its original Restaurant and Hotel ABSA counts are `0/11` for the baseline and candidate observations because the expected IDs are reversed.

Using only the saved prediction fields in `artifacts/e11/diagnostic_results.json`, a read-only correction recipe gives:

| Task / saved slot scope | Corrected baseline | Corrected candidate | Interpretation |
|---|---:|---:|---|
| Restaurant, all 11 expected slot observations | 11/11 | 11/11 | Polarity-ID reversal explains the apparent 0/11 result. |
| Hotel, pairs e11diag-023–025 only (6 valid slot observations) | 6/6 | 6/6 | Polarity-ID reversal explains these observations. |
| Hotel, e11diag-022 and e11diag-026 | not recomputable | not recomputable | These pairs query `FACILITIES#CLEANLINESS` for “phòng sạch sẽ”; the intended slot is `ROOMS#CLEANLINESS`. Saved inference contains the wrong slot, so no new semantic score can be recovered without inference. |

Because of both defects, the E11 ABSA synthetic semantic interpretation and margin claims are quarantined. No new model inference was run. Corrective recipe for a future authorized diagnostic: use `0/1/2/3 = absent/positive/negative/neutral`, change the two cleanliness expectations to `ROOMS#CLEANLINESS`, rebuild the pair artifact, then rerun the diagnostic on the same saved checkpoints.

## VSFC shared-text sample flag

The E12 sample selector intentionally sets the shared VSFC `error` flag to `sentiment_error OR topic_error`. The displayed `gold`/`prediction` columns are the sentiment pair, while `vsfc_topic_gold`, `vsfc_topic_prediction`, and `vsfc_topic_error` carry the topic result. Among the 15 selected VSFC rows, 3 are topic-only errors (`error=1` while sentiment gold equals prediction). The refreshed `sample_review.md` now prints both task outcomes per shared text; this is a display clarification, not a metric change.

## Additional integrity findings

- `evaluate_e11_diagnostics.py` increments `changed_count` for every baseline/candidate prediction disagreement, then divides by `base_correct_count`. Its `flip_rate_conditional_base_correct` is therefore mislabeled and can include changes on baseline-incorrect rows. The saved observations allow the correct numerator to be recomputed as `base_correct AND base_prediction != candidate_prediction`; for the valid VSFC/ViSFD task maps the corrected rates are Sentiment 0/11 and Topic 1/10 for both variants, and ViSFD 0/8. This is a model-disagreement diagnostic, not robustness to the base/variant text pair.
- E12 ABSA normalization now treats burden as undefined when a slice has zero scored gold-positive slots and reports spurious-per-row separately. The prior `safe_rate(..., 0) -> 0` behavior is preserved only in the pre-fix snapshot [artifacts/e12_pre_fix_20260912](../artifacts/e12_pre_fix_20260912/SHA256SUMS.txt). In the current validation features, ViSFD has 18 zero-gold rows with 15 spurious events across 12 rows; Restaurant and Hotel have none. ABSA common-support adjustment controls length and gold-positive burden only, without joint aspect/polarity adjustment.
- The read-only source spot check did open the first few lines of raw Hotel `Test.csv` to verify the numeric mapping example. No test predictions, test metrics, model inference, or test-derived analysis was run. The E12 analysis script itself uses `include_test=False`; the test-file spot check is disclosed here so “no test evaluation” is not confused with “no test bytes read”.

## Issue ledger

| Severity | Issue | Evidence | Numeric benchmark F1 affected? | Repair proposal |
|---|---|---|---|---|
| High | Restaurant/Hotel polarity names are reversed in loader metadata and inherited reports. | `src/goibert/data.py:56-58,629-634`; raw CSV examples; VLSP preprocessing mapping. | No, raw integer path is unchanged. | Correct metadata in a future contract revision; regenerate semantic reports only. |
| High | E11 Restaurant/Hotel synthetic expectations use positive `3`/negative `1`. | `scripts/build_e11_diagnostic.py:40-49`; saved E11 observations. | No benchmark F1; E11 ABSA semantic claims invalid. | Rebuild reviewed pairs with `1/2/3 = positive/negative/neutral`; rerun only if separately authorized. |
| High | E11 Hotel cleanliness expectations use `FACILITIES#CLEANLINESS` for room-cleanliness text. | `scripts/build_e11_diagnostic.py:45,49`; raw Hotel aspect columns/examples. | No benchmark F1; two synthetic pairs not recoverable from saved wrong-slot outputs. | Use `ROOMS#CLEANLINESS` in a future diagnostic spec. |
| Medium | E11 `flip_rate_conditional_base_correct` counts all changes in numerator, not only changes where baseline was correct. | `scripts/evaluate_e11_diagnostics.py:122-125,150`; corrected artifact `artifacts/e12/e11_flip_reanalysis.json`. | No benchmark F1; diagnostic rate definition is wrong. | Recompute from saved observations or fix evaluator for future diagnostics. |
| Medium | E12 zero-gold ABSA strata previously displayed normalized burden as zero. | Old `safe_rate` behavior; 18 ViSFD zero-gold validation rows, 15 spurious events across 12. | No pair F1; slice burden/adjusted gaps could be biased. | Current E12 uses `NA` burden, separate spurious-per-row, and positive-support adjustment; pre-fix tree is preserved. |
| Medium | E2 cache metadata binds the wrong data provenance string. | `src/goibert_e2/cache.py:104-118`; `src/goibert_e2/train.py:377-386`. | No current F1; metadata-only repair would invalidate cache contracts. | Rebuild or explicitly migrate caches after a frozen metadata revision. |
| Low/unknown | Third-party ABSA copy is not proven byte-equivalent to the original release. | `data/uit_absa_candidate_manifest.json:2-5`; source notes. | Unknown for paper equivalence, no evidence of current numeric loader corruption. | Treat paper/test comparability as unverified; do not silently substitute another dataset. |

## Scope and preservation

This audit did not alter raw CSVs, gold labels, E8 predictions, E1–E11 source code, checkpoints, or test evaluation. It used local raw files, saved E11 observations, and the loader/evaluator source for read-only diagnosis. The canonical metadata correction is recorded here for root review; applying it to frozen source is a separate decision.
