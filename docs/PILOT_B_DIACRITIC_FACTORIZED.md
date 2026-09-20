# Pilot B — diacritic-factorized embedding block for Vietnamese encoders (design, 2026-09-14)

Goal: verify, cheaply, whether replacing the *input embedding block* of a published SOTA encoder (CafeBERT, XLM-R-large
CPT, Findings NAACL 2024) with a **diacritic-factorized embedding** and continuing pretraining briefly gives a better
Vietnamese encoder than plain continued pretraining. Backbone-agnostic by design (to be repeated on mmBERT-base later).

## Block

For every vocab token t: skeleton g(t) = t with all Vietnamese diacritics removed (tone marks ` ´ ˜ ̉ ̣ and vowel
marks ^ ˘ ˛, đ→d), diacritic pattern d(t) = the multiset of removed marks (69 distinct patterns in the XLM-R vocab;
"none" for undiacritized tokens).

    E(t) = E_skel[g(t)] + E_pat[d(t)] + U[t]·W        (U: V×r, W: r×H, r = 64 — low-rank per-token residual)

Initialised from the original matrix so that undiacritized tokens with no siblings are unchanged exactly and diacritized
tokens are reconstructed by group mean + pattern mean + rank-64 residual. Tied MLM head uses the materialised matrix.
Without the low-rank constraint the factorisation is a vacuous re-parametrisation; the constraint is what forces tone
information into a shared channel.

Diacritic dropout (arms b, c): during CPT, each non-masked diacritized input token is replaced by its ASCII-skeleton
token id with p = 0.10 (skeleton token exists in vocab for 72 % of diacritized occurrences in task text). Targets are
untouched → the model learns "stripped ≈ original" and implicitly restores diacritics.

## No-train facts (scripts/research_checks/tone_vocab_stats.py + ASCII variant)

- 60–75 % of token occurrences in the ABSA/ViSFD train text are diacritized (XLM-R and mmBERT alike).
- Rare diacritized token types (train count ≤ 5) are only ≈1 % of occurrences; 32–38 % of them have a frequent
  (≥ 50) ASCII sibling. ⇒ the "rare-token rescue" mechanism is real but **small on clean benchmarks**; the expected
  large effect is **robustness to missing diacritics** (real-world social/chat/ASR text), which is only 2–8 % of
  these benchmarks (E12). Clean-score gain, if any, will come from regularisation — this is the risk the pilot tests.

## Arms (CafeBERT, same 300 M FineWeb-2 vie_Latn tokens, same hyper-parameters, 4 × L40S DDP)

| Arm | Embedding | Diacritic dropout | Tests |
|---|---|---|---|
| a | original | no | plain CPT control (also = the strongest "published SOTA + our data" baseline) |
| c | original | yes | augmentation alone |
| b | factorized (r = 64) | yes | the block |

## Evaluation (validation only, plain probe recipe)

5 tasks; seeds 42+43 on Restaurant/Hotel/ViSFD, seed 42 on VSFC. Each run also reports F1 on the **fully
diacritic-stripped validation set** at the best clean epoch, and (Restaurant/ViSFD) the error rate on rows containing a
rare diacritized token.

Gates: (1) b ≥ a − 0.3 on clean (no loss) **and** b ≥ c + 3 on stripped (block > augmentation) → continue (5 B
tokens, domain annealing, repeat on mmBERT-base); (2) b > a + 0.5 on clean Restaurant/ViSFD → strong signal;
(3) b < a − 0.5 on clean → the low-rank constraint hurts identity; try r = 256 once, then stop.

Estimated cost: data 30 min, code 1–2 h, CPT 3 × ~1.3 h, eval ~1 h.
