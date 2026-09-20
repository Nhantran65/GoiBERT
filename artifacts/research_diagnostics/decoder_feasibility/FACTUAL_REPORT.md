# Decoder feasibility diagnostic

CPU-only checkpoint-head analysis; no encoder forward, training, or test access.

| Model | Task | Margin rows | Rank | Rows | LP t | Certificate status |
|---|---|---:|---:|---:|---:|---|
| student_e8 | absa_restaurant | 36 | 36 | 36 | 0.018198552451354974 | numerical_certificate |
| student_e8 | absa_hotel | 102 | 102 | 102 | 0.042732987512401184 | numerical_certificate |
| student_e8 | visfd | 30 | 30 | 30 | 0.013096240927788593 | numerical_certificate |
| bami_e14 | absa_restaurant | 36 | 36 | 36 | 0.008966592977438985 | numerical_certificate |
| bami_e14 | absa_hotel | 102 | 102 | 102 | 0.016293807896714876 | numerical_certificate |
| bami_e14 | visfd | 30 | 30 | 30 | 0.004515816292579048 | numerical_certificate |
