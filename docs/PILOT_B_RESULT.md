# Pilot B result — diacritic-factorized embedding on CafeBERT (2026-09-15)

Design: [PILOT_B_DIACRITIC_FACTORIZED.md](PILOT_B_DIACRITIC_FACTORIZED.md). Backbone `uitnlp/CafeBERT` (XLM-R-large,
Findings NAACL 2024 = published SOTA on our suite, see CAFEBERT_PROBE.md). All arms: 300 M FineWeb-2 vie_Latn tokens,
seq 512, global batch 256, lr 5e-5, 4×L40S DDP (~1–1.8 h each). Evaluation: plain probe recipe (CE, lr 2e-5, 20 ep,
micro 8), validation only; seeds 42+43 on ABSA/ViSFD, seed 42 on VSFC; "stripped" = all diacritics removed from the
validation text (`phòng sạch` → `phong sach`), scored at the clean-best epoch. Code: `scripts/pilot_b/`,
`src/goibert_pilot_b/`; results `artifacts/pilot_b/{cpt,probe,summary_table.json}`.

| Arm | Embedding block | Diacritic dropout p | MLM val loss (end) |
|---|---|---|---|
| a | original | – | 1.083 |
| c | original | 0.10 | 1.088 |
| b2 | additive channels E_tok + E_skel + E_pat (exact init) | 0.10 | 1.087 |
| b2_hlr | b2 with 40× lr on E_skel/E_pat | 0.10 | 1.087 |
| **b** | **constrained: E_skel + E_pat + rank-64 residual** (init loss 3.0 → recovers) | 0.10 | 1.183 |

## Clean validation F1 (mean over seeds)

| Arm | Sentiment | Topic | ViSFD | Restaurant | Hotel | mean |
|---|---|---|---|---|---|---|
| CafeBERT, 0 tokens | 87.24 | 81.75 | 85.14 | 77.14 | 73.74 | 81.00 |
| a plain CPT | 87.64 | 81.27 | 85.21 | 77.57 | 73.11 | 80.96 |
| c +dropout | 88.33 | 81.29 | 85.37 | 77.57 | 73.76 | 81.26 |
| b2 additive | 88.97 | 81.92 | 85.11 | 77.28 | 73.91 | 81.44 |
| b2 additive lr×40 | 85.29 | 81.25 | 85.46 | 77.30 | 73.51 | 80.56 |
| **b constrained** | 86.42 | 81.23 | 85.04 | 77.73 | 73.43 | 80.77 |

Everything is within seed noise (≈0.5–1; VSFC single seed) of CafeBERT itself. **No arm moves clean scores**, as the
no-train check predicted (rare diacritized tokens are ≈1 % of task tokens).

## Diacritic-stripped validation F1

| Arm | Sentiment | Topic | ViSFD | Restaurant | Hotel | mean |
|---|---|---|---|---|---|---|
| CafeBERT, 0 tokens | – | – | – | 31.2 | 10.8 | – |
| a plain CPT | 42.6 | 23.3 | 59.5 | 36.9 | 13.9 | 35.2 |
| c +dropout | 59.9 | 29.5 | 68.3 | 41.8 | 19.5 | 43.8 |
| b2 additive | 52.2 | 32.5 | 66.7 | 41.8 | 23.4 | 43.3 |
| b2 additive lr×40 | 61.3 | 27.2 | 67.4 | 43.7 | 25.7 | 45.0 |
| **b constrained** | **67.6** | **53.2** | **71.1** | **54.7** | **48.9** | **59.1** |

## Reading

1. **The constrained block is the mechanism, not the augmentation.** Same dropout, same data: b beats c by +15 mean
   on stripped input (Hotel 19.5 → 48.9, Topic 29.5 → 53.2, Restaurant 41.8 → 54.7) with clean scores unchanged
   (mean −0.2 vs a, within noise; Sentiment −1.2 is single-seed). Gate 1 of the design (no clean loss AND ≥ +3 over c
   on stripped) passes by a wide margin; gate 2 (clean gain ≥ +0.5) fails.
2. **Additive channels are inert** even at 40× lr: with full per-token identity kept, the shared channels stay at
   0.3–0.8 % of the embedding norm (measured on b2's checkpoint) and b2 ≈ c everywhere. A full-rank residual makes
   the factorization a vacuous re-parametrization; the low-rank constraint is what forces diacritics into a shared
   channel. This is the ablation a reviewer will ask for and it is clean.
3. b's MLM loss is still 0.10 nats worse than a after 300 M tokens (init damage not fully recovered) yet downstream
   clean F1 is equal — the block trades MLM fit for a representation where `phong` and `phòng` share most of their
   vector. More tokens / higher rank should close the MLM gap; stripped scores are still 15–30 below clean, so there is
   headroom (dropout was only 0.10).
4. For the "beat SOTA on clean benchmarks" goal this block does nothing; for **robustness to missing diacritics**
   (chat / social / ASR text) it is the first thing in this project that moves a number by tens of points with a
   controlled ablation, and CafeBERT / BamiBERT / mmBERT are all near-blind on such input (10–31 pair F1).

## Cheap next steps (if pursued)

- b with p = 0.3 and rank 128; 1 B tokens; partial stripping (30 % of syllables) and a real diacritic-less subset as
  extra eval; seeds 43/44 on VSFC to settle the −1.2 Sentiment reading.
- Repeat on mmBERT-base (backbone-agnostic claim); measure the "restoration" ability directly (MLM top-1 on stripped
  tokens).

## Addendum 2026-09-15: real social-media data and the teencode extension (closed)

**Real social benchmarks** (ViHSD dev macro-F1, 10 ep): clean CafeBERT 70.9 / a 70.7 / c 70.6 / b 71.1 (tie);
stripped 55.9 / 55.0 / 58.3 / **61.9**. Noise audit (`scripts/research_checks/social_noise_stats.py`,
`artifacts/research_diagnostics/social_noise_stats.md`): fully diacritic-less rows are only 0.1–4 % even on
ViHSD/ViSpam/VSMEC; mixed rows 5–18 %; ViLexNorm's human corrections are 96.8 % abbreviation/teencode and 3.2 %
diacritics. The diacritic block therefore has no lever on real benchmarks; its effect appears only under synthetic
full stripping.

**Teencode extension** — dictionary of 384 variants → canonical from ViLexNorm (`scripts/pilot_b/teencode_map.py`,
`data/cpt/teencode_map.json`); variants are 3–5 % of words on ViHSD/ViSpam/ViSFD and occur in 31–59 % of rows. Oracle
upper bound (CafeBERT fine-tuned on raw text, dev evaluated after dictionary normalisation): ViHSD +0.5, ViSpam 0,
ViSFD +0.1, Restaurant −1.9 (`artifacts/pilot_b/teencode_oracle/`). Even a perfect teencode-aware block cannot beat
this bound → not run (extended map `data/cpt/xlmr_diacritic_teencode_map.npz` and `cpt_train.py --dmap/--variant-p`
are in place if ever needed). Frequent teencode is already understood by the encoder; dictionary-level intervention
can hurt.

**Verdict for the embedding-block direction:** closed. Clean-benchmark gains must come from pretraining (domain data,
architecture, scale).
