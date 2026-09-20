# Reader learnability diagnostic

Exploratory 3-fold validation cross-fit; frozen encoder and no test access.

| Task | Arm | Pair F1 | Detection F1 | Missed | Spurious | Wrong polarity |
|---|---|---:|---:|---:|---:|---:|
| absa_restaurant | full_only | 0.679 | 0.832 | 220 | 117 | 153 |
| absa_restaurant | clause | 0.672 | 0.829 | 209 | 140 | 160 |
| absa_restaurant | random | 0.651 | 0.817 | 225 | 147 | 168 |
| absa_hotel | full_only | 0.686 | 0.753 | 402 | 193 | 81 |
| absa_hotel | clause | 0.683 | 0.752 | 388 | 219 | 85 |
| absa_hotel | random | 0.681 | 0.750 | 395 | 213 | 84 |
