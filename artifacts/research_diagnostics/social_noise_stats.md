# Social-media noise stats vs. existing task train text

Diacritic definition: a word is diacritic-bearing if its NFD form contains any of the tone marks U+0300/U+0301/U+0303/U+0309/U+0323, or the vowel-quality marks U+0302/U+0306/U+031B, or the letter đ/Đ.

"No-diacritic" row: >=3 alphabetic words, none diacritic-bearing. "Mixed" row: 0% < share of diacritic-bearing alphabetic words < 60%. "Teencode" row: contains >=1 whole-word token from a fixed teencode/abbreviation list (case-insensitive).

| dataset | rows | words/row | %no-diacritic rows | %mixed | %teencode rows |
|---|---|---|---|---|---|
| vihsd | 24048 | 11.51 | 3.99 | 18.2 | 25.93 |
| vispamdetection | 14306 | 26.07 | 2.29 | 8.14 | 29.46 |
| ViCTSD | 7000 | 29.22 | 0.17 | 5.37 | 17.71 |
| uit-vsmec | 5548 | 14.01 | 0.07 | 5.66 | 4.27 |
| ViLexNorm(original) | 8372 | 12.16 | 0.39 | 16.82 | 51.83 |
| vsfc_sentiment | 11426 | 14.31 | 0.04 | 1.05 | 1.3 |
| visfd | 7786 | 36.32 | 0.06 | 6.27 | 47.08 |
| absa_restaurant | 7028 | 16.47 | 0.06 | 3.71 | 18.67 |
| absa_hotel | 7180 | 18.23 | 0.0 | 0.56 | 4.9 |

## ViLexNorm original->normalized token-diff alignment (train split)

Whitespace-aligned original/normalized row pairs where token counts match; among differing token pairs, share that is diacritics-only (`strip_diacritics(normalized_token) == original_token`) vs. abbreviation/teencode (everything else).

- **row_pairs**: 8372
- **row_pairs_used_matching_token_count**: 6652
- **coverage_pct**: 79.46
- **differing_token_pairs**: 12297
- **pct_diacritics_only**: 3.2
- **pct_abbreviation_other**: 96.8
