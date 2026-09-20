# C4 reader operating-point extraction

Validation-only factual export; no threshold is selected for benchmark use.

| Model | Task | n | Baseline F1@δ0 | C4 F1@δ0 | Baseline recall | C4 recall | Reference check |
|---|---|---:|---:|---:|---:|---:|---|
| student_e8 | absa_restaurant | 771 | 71.346% | 67.604% | 69.991% | 73.029% | True |
| student_e8 | absa_hotel | 795 | 73.049% | 68.577% | 69.366% | 74.943% | True |
| student_e8 | visfd | 1112 | 82.394% | 78.910% | 81.996% | 84.288% | True |
| bami_e14 | absa_restaurant | 771 | 74.683% | 73.447% | 75.499% | 75.783% | True |
| bami_e14 | absa_hotel | 795 | 73.313% | 72.166% | 73.033% | 75.859% | True |
| bami_e14 | visfd | 1112 | 84.438% | 83.467% | 85.253% | 86.399% | True |
