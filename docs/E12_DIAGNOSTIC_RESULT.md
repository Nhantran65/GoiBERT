# E12 Surface Error Diagnostic Result

This validation-only audit uses the E8 seed-42 saved predictions, the frozen train and validation loaders with `include_test=False`, and the original `models/goiturn-s-gd` fast tokenizer. It does not train, load a neural checkpoint, run inference, or inspect test rows.

## Scope and correctness

Metric values were recomputed from each saved prediction file after exact ID and gold alignment against the validation loader. VSFC sentiment and topic share one text domain and are represented once in the four-domain sample/audit view. ViSFD OTHERS is included for detection but excluded from pair-scored ABSA error counts.

| Task | n | recomputed selection F1 | saved selection F1 | delta |
|---|---:|---:|---:|---:|
| vsfc_sentiment | 1583 | 0.851604 | 0.851604 | 0 |
| vsfc_topic | 1583 | 0.803678 | 0.803678 | 0 |
| visfd | 1112 | 0.824421 | 0.824421 | 0 |
| absa_restaurant | 771 | 0.712766 | 0.712766 | 0 |
| absa_hotel | 795 | 0.730491 | 0.730491 | 0 |

## Domain error decomposition

The ABSA rows below are reported with missed/spurious/wrong-polarity events and burden per scored gold-positive slot. Raw any-error row rates are retained in feature CSVs for within-task review and are not compared across domains.

| Domain | tasks/text scope | validation rows | error events | gold-positive slots | normalized burden |
|---|---|---:|---:|---:|---:|
| vsfc | shared text; sentiment+topic outcomes | 1583 | 254 | 3166 | 0.0802 |
| visfd | visfd | 1112 | 887 | 3316 | 0.2675 |
| absa_restaurant | absa_restaurant | 771 | 454 | 1053 | 0.4311 |
| absa_hotel | absa_hotel | 795 | 607 | 1309 | 0.4637 |

## Interpretation gate

The slice tables report raw high/low contrasts and common-support standardization by train-derived length bins plus true class (VSFC) or gold-positive burden (ABSA). High/low groups use train quartiles; ties are retained and slices are disabled when the train feature is constant or Q1 equals Q3. Coverage and dropped rows are reported so small or non-overlapping groups are not treated as strong evidence. ABSA adjustment controls only length and gold-positive burden; it does not jointly control aspect, polarity, rarity, and fragmentation.

A character/diacritic branch is worth prioritizing only if fragmentation, rarity, or the orthography proxy has adequate support, the adjusted direction is consistent across the relevant domains, and the sampled examples support the proposed mechanism. The audit itself is descriptive and cannot establish causality or predict a new architecture's gain.

## Architecture evidence

BPE fragmentation does not show a broad cross-domain direction after adjustment: `bpe_per_unit_mean` is +1.38 percentage points for VSFC sentiment, −8.05 pp for VSFC topic, +1.69 pp for ViSFD, −1.55 pp for Restaurant, and +2.17 pp for Hotel (high minus low error burden). The >=3-piece unit slice has usable train support only for ViSFD, where the adjusted gap is +2.16 pp. This is inconsistent and task-specific.

Rarity is the clearest task-specific signal: adjusted rare-unit burden is +6.36 pp for ViSFD and +13.20 pp for Restaurant, with near-complete positive-support coverage. The corresponding rarity slices are disabled for VSFC and Hotel because train Q1 equals Q3 at zero; forcing a split would turn ties into an artificial result. Rarity remains confounded with fragmentation and is not evidence that a character encoder will improve every task.

The accent-absence proxy has sparse validation support (22/1,583 VSFC rows, 92/1,112 ViSFD, 54/771 Restaurant, and 19/795 Hotel) and its train quartile slices are disabled by Q1=Q3=0. It therefore supplies no adjusted cross-domain evidence. Units without marks are not labeled errors; the proxy requires a rare/unseen mark-free unit and a marked train counterpart, while ambiguity is retained in `validation_units.csv`.

Gate result: broad character/diacritic architecture prioritization is inconclusive. A narrowly scoped rarity/fragmentation follow-up could be considered for ViSFD and Restaurant after root review of the seeded examples, but this audit does not approve a shared five-task character branch.

Adjusted rows with the largest absolute gaps (screening aid; inspect coverage before interpretation):

| Task | Feature | raw high-low burden | adjusted high-low burden | common strata | coverage low | coverage high |
|---|---|---:|---:|---:|---:|---:|
| absa_hotel | unit_count | 0.1304 | -0.2813 | 2 | 0.054 | 0.065 |
| visfd | unit_count | 0.1206 | 0.1995 | 4 | 0.190 | 0.992 |
| visfd | combining_mark_count | 0.1020 | 0.1717 | 4 | 0.303 | 0.992 |
| absa_hotel | combining_mark_count | 0.1292 | -0.1713 | 2 | 0.177 | 0.137 |
| absa_restaurant | rare_unit_rate | 0.1007 | 0.1320 | 6 | 1.000 | 1.000 |
| absa_restaurant | unicode_ratio | 0.1429 | 0.1317 | 6 | 1.000 | 1.000 |
| vsfc_topic | unit_count | 0.0286 | 0.0982 | 3 | 0.111 | 0.055 |
| vsfc_topic | bpe_per_unit_mean | -0.1026 | -0.0805 | 12 | 1.000 | 1.000 |
| vsfc_sentiment | unit_count | 0.0005 | 0.0786 | 2 | 0.116 | 0.052 |
| absa_restaurant | punctuation_count | -0.0560 | -0.0784 | 5 | 0.964 | 1.000 |
| visfd | rare_unit_rate | 0.1036 | 0.0636 | 6 | 0.990 | 0.972 |
| absa_hotel | unicode_ratio | 0.0277 | 0.0612 | 6 | 1.000 | 1.000 |

The seeded manual-review file contains 60 rows across all four domains, with both correct and incorrect outcomes when available. Notes in the file are mechanical preliminary labels; semantic interpretation remains for root review.

## Limitations

Features are descriptive proxies. A no-diacritic unit is not automatically an error: the proxy only marks rare/unseen units with a train counterpart sharing a mark-stripped skeleton, and reports counterpart ambiguity. Rarity and fragmentation are correlated; the ABSA stratification does not jointly adjust them. Full-text BPE length and model-truncated length are reported separately. In zero-gold ABSA strata, normalized burden is undefined and is shown as blank/NA; spurious predictions are reported separately. In this validation set, ViSFD has 18 zero-gold rows with 15 spurious events across 12 rows; Restaurant and Hotel have none. No synthetic counterfactuals, extra datasets, p-values, or model retraining were used.

Provenance is recorded in `artifacts/e12/provenance.json`; sample selection seed is 12042.
