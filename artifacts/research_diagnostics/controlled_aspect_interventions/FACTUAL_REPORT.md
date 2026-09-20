# Controlled aspect interventions

Root-authored validation-only fixture; metrics score explicitly annotated target slots only.

| Model | Task | Inputs | Standalone target acc | Factorial joint acc | A changes after B flip | A correct→wrong |
|---|---|---:|---:|---:|---:|---:|
| student_e8 | absa_hotel | 56 | 0.875 | 0.438 | 0/16 | 0 |
| student_e8 | absa_restaurant | 56 | 1.000 | 0.719 | 2/16 | 2 |
| student_e8 | visfd | 56 | 1.000 | 0.844 | 3/16 | 0 |
| bami_e14 | absa_hotel | 56 | 0.750 | 0.438 | 4/16 | 0 |
| bami_e14 | absa_restaurant | 56 | 1.000 | 1.000 | 0/16 | 0 |
| bami_e14 | visfd | 56 | 1.000 | 0.938 | 2/16 | 2 |
