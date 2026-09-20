# Independent intent benchmark audit

This report was generated from an independent read-only pass over the raw prompt-bench bundle, corrected/split JSONL, training code, local model files, run summaries, checkpoints, and saved predictions. It does not rerun training or alter original artifacts. Raw evidence is in `evidence.json`.

## Findings by severity

### High: external validity is limited by label provenance

No critical implementation error was detected in extraction, input construction, split isolation, checkpoint loading, metric recomputation, dev selection, or test timing from the artifacts inspected. The material limitation is external validity: the 507 target labels originate in the source bundle's `expectedIntent` fields and 19 rows were changed by a deterministic review override map. The checked-in README and review policy explicitly say the production prompt is DB-managed and absent; therefore label provenance is limited to the source snapshot plus an AI-assisted semantic review rather than an independently adjudicated human gold set. This is a validity limitation, not evidence of intentional model favoritism. `intent_benchmark/prepare_intent_benchmark.py:60-111,252-280`; `intent_benchmark/README.md`.
The independent counts are {'cases': 250, 'all_target_turns': 512, 'labeled': 507, 'missing': 5}; the 19 changes distribute as {'train': 12, 'dev': 2, 'test': 5} with transitions {'ADDRESS->OTHER': 2, 'COMPLAINT->OTHER': 3, 'CONFIRMATION->OTHER': 2, 'MENU_QUERY->OTHER': 6, 'OTHER->COMPLAINT': 1, 'OTHER->CONFIRMATION': 5}.

The independent row-by-row review found 12 transitions directly rubric/context-supported and 7 supported but explicitly ambiguous/context-dependent (invoice/store address, shipping-price complaint, short closure/cancellation turns). The complete 19-row evidence table is below; no label is changed by this audit.

| sample | source → reviewed | assessment |
|---|---|---|
| pb-v1-0019:t0 | ADDRESS → OTHER | rubric_supported_but_ambiguous — Invoice-company address is distinct from delivery ADDRESS under the written rubric; address token creates a plausible competing reading. |
| pb-v1-0043:t0 | COMPLAINT → OTHER | rubric_supported — Neutral cancellation has no dissatisfaction/service-problem language. |
| pb-v1-0045:t0 | COMPLAINT → OTHER | rubric_supported — Personal cancellation reason plus thanks is operational and not a complaint. |
| pb-v1-0047:t0 | OTHER → COMPLAINT | rubric_supported_but_ambiguous — Explicit shipping-cost dissatisfaction supports COMPLAINT, but refusal/thanks is a mixed short act. |
| pb-v1-0062:t0 | ADDRESS → OTHER | rubric_supported_but_ambiguous — Restaurant directions/location is outside delivery-address semantics, though ordinary readers may call it an address request. |
| pb-v1-0063:t0 | MENU_QUERY → OTHER | rubric_supported_but_ambiguous — Restaurant address is outside the written delivery ADDRESS definition, but the word address is lexically strong. |
| pb-v1-0181:t3 | OTHER → CONFIRMATION | context_supported — Closure phrase after a concrete order selection is a contextual confirmation; it does not imply real-order authorization. |
| pb-v1-0183:t3 | OTHER → CONFIRMATION | context_supported — Closure phrase after a concrete order selection is a contextual confirmation; it does not imply real-order authorization. |
| pb-v1-0184:t3 | OTHER → CONFIRMATION | context_supported — Closure phrase after a concrete order selection is a contextual confirmation; it does not imply real-order authorization. |
| pb-v1-0185:t3 | OTHER → CONFIRMATION | context_supported — Closure phrase after a concrete order selection is a contextual confirmation; it does not imply real-order authorization. |
| pb-v1-0189:t3 | OTHER → CONFIRMATION | context_supported_but_ambiguous — Closure phrase is consistent with confirmation in ordering context, but it is short and context-dependent. |
| pb-v1-0215:t4 | MENU_QUERY → OTHER | rubric_supported — Summary/recap is checkout/order state, not a menu query. |
| pb-v1-0218:t4 | MENU_QUERY → OTHER | rubric_supported — Summary request is operational checkout state, not a menu query. |
| pb-v1-0220:t4 | MENU_QUERY → OTHER | rubric_supported — Summary request after replacement is operational checkout state, not a menu query. |
| pb-v1-0222:t2 | CONFIRMATION → OTHER | rubric_supported — Time/payment plus summary request is operational checkout state, not a clear confirmation. |
| pb-v1-0222:t3 | CONFIRMATION → OTHER | rubric_supported — Time/payment plus summary request is operational checkout state, not a clear confirmation. |
| pb-v1-0228:t7 | MENU_QUERY → OTHER | rubric_supported — Summary request is operational checkout state, not a menu query. |
| pb-v1-0229:t7 | MENU_QUERY → OTHER | rubric_supported_but_ambiguous — Summary request is operational state; the source MENU_QUERY assignment is not supported by the written boundary. |
| pb-v1-0240:t1 | COMPLAINT → OTHER | context_supported_but_ambiguous — Accidental prior-order cancellation is operational and explicitly retains the current order; no dissatisfaction is stated. |

The five changed test rows all score as reviewed labels for BamiBERT and DAPT (5/5 each) and 4/5 for the student; under the raw source labels, none of the three models is correct on those five rows. This makes the reviewed labels consequential for the absolute test scores, but it does not show that the review was model-driven; the five test changes help both families equally. The 12 changed training rows can affect learned representations and remain a provenance sensitivity that saved predictions alone cannot identify.

### Medium: repeated current text crosses splits by design

Case and source groups are split-disjoint; the independent group manifest check passes for all 235 groups and all confirmed template families are one-split. Exact full-input duplicate overlap is 0; however 10 normalized current-text families cross partitions. The largest repeated family is the placeholder delivery phrase. This is documented and is not exact full-input leakage, but it allows lexical memorization for generic/placeholder turns and limits a text-generalization interpretation. `intent_benchmark/finalize_intent_benchmark.py:357-417`; `intent_benchmark/group_manifest.json`.

### Medium: the label review and single split support are fragile for rare classes

Per-split sizes/classes recomputed independently: {"dev": {"case_count": 37, "class_counts": {"ADDRESS": 6, "COMPLAINT": 5, "CONFIRMATION": 7, "GRATITUDE": 2, "GREETING": 2, "MENU_QUERY": 22, "OTHER": 31, "PHONE": 2}, "rows": 77, "source_count": 37}, "test": {"case_count": 38, "class_counts": {"ADDRESS": 6, "COMPLAINT": 5, "CONFIRMATION": 7, "GRATITUDE": 2, "GREETING": 2, "MENU_QUERY": 21, "OTHER": 32, "PHONE": 2}, "rows": 77, "source_count": 38}, "train": {"case_count": 175, "class_counts": {"ADDRESS": 26, "COMPLAINT": 25, "CONFIRMATION": 30, "GRATITUDE": 7, "GREETING": 8, "MENU_QUERY": 101, "OTHER": 148, "PHONE": 8}, "rows": 353, "source_count": 175}}. GREETING, GRATITUDE and PHONE have only 2 test examples each; COMPLAINT has 5. Macro-F1 is therefore highly sensitive to a few decisions. The benchmark is suitable as a small pilot, not a stable estimate of population performance.

### Low/medium: protocol is internally consistent but narrower than a reproduction study

The locked config uses seed 42, batch 16, max 15 epochs, patience 3, learning rates [2e-05, 5e-05], and one winner per backbone selected by dev macro-F1. Hash checks give runner=True and config=True. This is a fair common pilot budget, but two LRs/single seed/15-epoch cap do not establish a fully tuned comparison.

## Independently recomputed results

The saved test predictions contain 77 unique IDs per model and metrics recompute exactly from their class IDs. The requested headline values are:

| model | accuracy | macro-F1 |
|---|---:|---:|
| bamibert | 0.792207792 | 0.474469127 |
| goi_dapt | 0.896103896 | 0.832981273 |
| goi_student | 0.883116883 | 0.820868870 |

Independent integrity checks also confirm every saved test gold ID/label matches the frozen `test.jsonl` row, every saved dev gold ID/label matches its frozen `dev.jsonl` row, and every probability vector is length 8, sums to 1, and has `argmax == pred_id`. All six saved best-epoch dev prediction files pass these checks.

For sensitivity only, holding saved predictions fixed and replacing reviewed test labels with the raw source labels gives:

| model | reviewed-label macro-F1 | raw-source-label macro-F1 |
|---|---:|---:|
| bamibert | 0.474469 | 0.424147 |
| goi_dapt | 0.832981 | 0.780365 |
| goi_student | 0.820869 | 0.772040 |

Paired correctness counts on the same 77 test rows:

| pair | both | first only | second only | neither |
|---|---:|---:|---:|---:|
| bamibert_vs_goi_dapt | 61 | 0 | 8 | 8 |
| bamibert_vs_goi_student | 59 | 2 | 9 | 7 |
| goi_dapt_vs_goi_student | 67 | 2 | 1 | 7 |

### What explains BamiBERT's lower macro-F1 in the saved test run?

The primary verified pattern is per-class error, not a single global collapse:

| class | support | BamiBERT F1 | GoiBERT-DAPT F1 | DAPT minus Bami contribution to macro-F1 |
|---|---:|---:|---:|---:|
| GREETING | 2 | 0.000 | 0.667 | 0.083 |
| CONFIRMATION | 7 | 0.750 | 0.933 | 0.023 |
| GRATITUDE | 2 | 0.000 | 1.000 | 0.125 |
| MENU_QUERY | 21 | 0.884 | 0.930 | 0.006 |
| COMPLAINT | 5 | 0.400 | 0.571 | 0.021 |
| ADDRESS | 6 | 0.909 | 1.000 | 0.011 |
| PHONE | 2 | 0.000 | 0.667 | 0.083 |
| OTHER | 32 | 0.853 | 0.896 | 0.005 |

BamiBERT predicts zero examples of GREETING, GRATITUDE, and PHONE in the selected test checkpoint (their recalls are all zero), while DAPT and the student get at least one of each. The selected BamiBERT checkpoint still performs well on MENU_QUERY, ADDRESS and OTHER; the gap is driven mainly by the three rare classes plus weaker COMPLAINT. This is a verified error decomposition, not proof of why the representation failed.
The three rare classes contribute 0.291667 of the DAPT–BamiBERT macro-F1 gap (81.4%) and 0.291667 of the student–BamiBERT gap (84.2%). Because each has support 2, these contributions are high-variance.

The paired table and per-class table separate the accuracy/F1 gap from class support effects. The BamiBERT dev trajectories also show strong learning-rate sensitivity:

- `bamibert-lr2em5`: best dev macro-F1 0.502859 at epoch 11; 14 epochs; final train loss 0.365727; final dev macro-F1 0.488034.
- `bamibert-lr5em5`: best dev macro-F1 0.275081 at epoch 15; 15 epochs; final train loss 0.810899; final dev macro-F1 0.275081.
The 5e-5 BamiBERT run is still improving at the 15-epoch cap but ends at only 0.275 dev macro-F1, whereas 2e-5 reaches 0.503. GoiBERT-DAPT also shows the same recipe interaction (0.501 at 2e-5 versus 0.722 at 5e-5). Consequently, the saved gap is consistent with a combination of pretrained/domain representation and optimization interaction; the artifacts do not identify a causal factor.

## Provenance and architecture checks

- Local model files were hashed independently. BamiBERT config reports a 12-layer, 768-hidden, 12-head RoBERTa with 20,481 vocabulary entries and 2,050 max positions; GoiBERT-DAPT matches the base dimensions; GoiTurn student is 10-layer/384-hidden/6-head. All training inputs use max length 512, within the model position limit.
- DAPT loading artifacts show no missing keys and only MLM-head unexpected keys; BamiBERT shows no missing keys and only MLM-head unexpected keys; student has no missing/unexpected keys. This supports checkpoint identity/loading correctness for the saved runs, subject to the manifest provenance limits. The local BamiBERT weights hash is `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`, matching the official file hash reported for [Qualcomm-AI-Research/BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/blob/main/model.safetensors).
- Tokenizer verdict: all 507 serialized inputs produce identical token ID sequences across all three local tokenizers; vocabularies and special-token IDs also match. All three load as `XLMRobertaTokenizerFast` with BOS/CLS=0, PAD=1, EOS/SEP=2, MASK=20480 and empty wrapped input `[0, 2]`.
- The training code disables the pretrained pooler (`add_pooling_layer=False`) and consumes the first token hidden state. It creates a new `Linear(hidden, hidden) -> Tanh -> Linear(hidden, 8)` head with PyTorch default initialization, followed by dropout. This is dimensionally valid and shared across models, but the head initialization is not explicitly the HF `initializer_range=0.02` recipe; it is a protocol detail that can add variance and is not evidence of a Bami-specific fault.
- Source model manifests report DAPT 40k MLM steps from a Bami checkpoint and student 30k hidden/attention distillation from DAPT on the same corpus-stat hash. Those manifests establish the claimed local lineage, while the corpus files themselves are absent from this workspace, so corpus overlap/domain contamination cannot be independently ruled out.

## Verified implementation behavior
- Input history: no independently detected mismatches; source target history is customer-only, with unlabeled preceding turns retained as text context where present. No future labels, fixtures, assertions, source IDs or other metadata are in `model_input` according to independent reconstruction and the code assertions (`prepare_intent_benchmark.py:227-242,299-317`; `train_intent.py:114-170`).
- Train/eval mode, dropout behavior, gradient updates, and metric computation are explicit (`train_intent.py:546-585`, `408-436`). Selection uses dev only and the test evaluator checks the locked selection before loading checkpoints (`select_dev_winners.py:50-78`, `evaluate_selected.py:48-88`).
- Saved test predictions and dev winner predictions match independently recomputed IDs/metrics and frozen gold rows; no test score is needed to explain the published values. Timing artifact checks pass for selection/checkpoint/test hashes, 10 warmups, two 77-example passes, batch size 1, and forward-only scope. Timing is therefore a forward-only GPU measurement and should not be interpreted as end-to-end serving latency.
- `intent_benchmark/README.md` contains an introductory sentence saying the directory uses no training/inference; that sentence describes preparation scope and is stale after the training artifacts were added, but it does not alter the executed protocol.

## Remaining uncertainty and recommended follow-ups

- Obtain the original expected-intent observation/run provenance and, if a gold claim is needed, commission an independent human double annotation with adjudication, reporting label sensitivity for the 19 changes and 82 uncertainty rows. Keep all reviewed uncertainty rows in the benchmark as specified; do not filter them post hoc.
- For a stronger model ranking claim, pre-register multiple seeds and a broader LR/epoch budget, or reproduce the BamiBERT paper's stated fine-tuning recipe separately. Keep the 77-row test frozen and untouched.
- Treat DAPT/student superiority as a result on this frozen, reviewed pilot; do not claim that DAPT alone caused the improvement over BamiBERT because LR sensitivity and the unverified historical corpus remain confounders.

Generated evidence files: `intent_benchmark/audit_independent/evidence.json` and this report.
