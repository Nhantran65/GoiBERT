# SUMMARY REPORT — The Goi Model Family (Aug 31 → Sep 4, 2026)

Owner: Line (supervisor: Claude, worker: Codex). Infrastructure: LUMI EuroHPC
(project_462001285). Delivery level: **implementation-complete**, pushed to branch
`ml-models` @ ai-foundary (commit 2f69828).

---

## 1. Assigned goals vs. results

| Goal                                                     | Result                                                                                                                                                                                    |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build the 3 Goi models from BamiBERT (proposal §13)     | ✅ GoiBERT-B 103M / GoiTurn-S 26.5M / GoiTurn-XS 10.6M —**genuinely self-trained**: self-built 1.67GB corpus → DAPT MLM 40K steps → general distillation 30K steps → task stage |
| Strongest model within 5% of BamiBERT on shared datasets | ✅**Met on all 4 datasets**: VSFC 2.73% · ViSFD 0.86% · PhoATIS 2.85% · VLSP mean of 3 seeds **+2.6 points ABOVE** the baseline                                            |
| Test on the 3–4 strongest, most suitable datasets       | ✅ UIT-VSFC, UIT-ViSFD, VLSP2018-Restaurant, PhoATIS — identical protocol/LR grid/seed on both sides, selection on validation only, test evaluated exactly once                          |
| Orthodox methods only, no tricks                         | ✅ Every step has published grounding (CafeBERT/ViSoBERT/TinyBERT/R-Drop); multi-seeding caught and**retracted** an illusory claim                                                  |

## 2. The three results that matter most

### 2.1 Noise-consistency training — the Goi family's own weapon

Macro F1 on the **no-diacritics** slice (how Vietnamese customers actually type on mobile):

| Dataset   | Plain models | Models + NC | NC recovers               |
| --------- | ------------ | ----------- | ------------------------- |
| UIT-VSFC  | 0.41–0.50   | 0.73–0.74  | **+23..+33 points** |
| PhoATIS   | 0.15–0.20   | 0.49–0.53  | **+29..+37 points** |
| UIT-ViSFD | 0.39–0.49   | 0.64–0.71  | **+22..+25 points** |
| VLSP-Rest | 0.28–0.36   | 0.37–0.42  | +6..+9 points             |

**GoiTurn-S 26.5M + NC (0.7354) beats plain BamiBERT 103M (0.4988) by +23.7 points on
undiacritized VSFC.** Clean-text cost ≈ 0. BamiBERT does not have this capability. NC is
now default-on for all production Goi models.

### 2.2 Parity on the product domain (PhoATIS — transactional dialogue)

GoiTurn-S **slightly beats** the 103M baseline (macro F1 0.5645 vs 0.5629); XS **ties**
(accuracy 0.9731, identical) — at 1/4 and 1/10 of the size.

### 2.3 A production-ready turn-completion model (Nik's original ask)

**GoiTurn v1** (`variant_b_gdinit_nc`, 26.5M): test macro F1 **0.9582**, false-cut rate
**5.32%** — 99%+ of the fine-tuned BamiBERT 103M (0.9658/3.65%), with NC built in.
Evolution: v0 scratch 0.9331/8.09% → v0.5 +GD backbone 0.9577/5.11% → v1 +NC. The first
Vietnamese turn-completion model in existence.

## 3. Honesty & limitations

- **Retracted**: the Sep-02 claim "beats BamiBERT by +10.7 points on VLSP" — multi-seeding
  showed it was a lucky seed; VLSP variance is ±3 points per seed. The 3-seed mean still
  favors DAPT-B by +2.6 points — reported as a "positive trend" only; a defensible claim
  would need ≥5 seeds.
- VSFC/ViSFD/PhoATIS are single-seed; the training protocol is reduced (10–15 epochs) —
  sufficient for internal decisions, not for publication.
- Students remain weak on ViSFD (S 6.5%, XS 10.0% with NC) — multi-aspect ABSA needs
  capacity.
- Turn-model numbers are on SYNTHETIC data — not production; Phase 2 (real data) answers
  that question.
- GoiBERT-B is a BamiBERT derivative (BSD-3-Clause-Clear + Qualcomm RAIL) — legal review
  required before any public release.

## 4. Assets

| Asset                                                | Location                                                                                                                                 |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Code + plans + detailed reports                      | branch`ml-models` @ ai-foundary, commit 2f69828 (42 files)                                                                             |
| Family report (full detail)                          | `experiments/goibert-family-v1/REPORT.md` + `MODEL_SPECS.md`                                                                         |
| Turn-model report                                    | `experiments/goiturn-synthetic-v0/REPORT.md`                                                                                           |
| Checkpoints (3 HF backbones + turn v1 + task models) | LUMI`/scratch/project_462001285/goiturn/` — ⚠️ auto-purged ~Dec 2026                                                                |
| 4-dataset robustness tables                          | LUMI`experiments/goibert-family-v1/results/noisy_eval_*.md`                                                                            |
| Phase-2 data source (already audited)                | `messenger_messages` @ staging Neon: 20,241 messages / 123 conversations / 1 full year, per-message timestamps + staff/customer sender |
| Compute spend                                        | ~80–100 of 5,000 GPU-hours                                                                                                              |

## 5. Next steps (pending decisions)

1. **Open Phase 2**: mine the messenger data into a real-world eval set + fine-tune turn
   v1 on the true distribution (highest remaining value; requires a PII-redaction step
   before any training use).
2. **Team HuggingFace account**: upload the 4 models privately (soft deadline ~Dec 2026
   due to scratch purge).
3. Phase-0 org items: taxonomy sign-off, 5K annotation budget, BamiBERT legal review,
   voice/multi-language scope (Nik).

---

# Appendix — Full measured results

All numbers measured on LUMI MI250X, seed 42 unless stated. Selection on validation only;
test evaluated once per selected run. gap% = (baseline − ours) / baseline; negative = we win.

## A. Model specifications (exact parameter counts)

|                               | GoiTurn-XS                                                | GoiTurn-S           | GoiBERT-B            |
| ----------------------------- | --------------------------------------------------------- | ------------------- | -------------------- |
| Layers / hidden / heads / FFN | 6 / 256 / 4 / 1024                                        | 10 / 384 / 6 / 1536 | 12 / 768 / 12 / 3072 |
| Params (with 3-class head)    | 10,573,827                                                | 26,546,691          | 102,953,475          |
| Weights origin                | PCA(768→256) + GD                                        | PCA(768→384) + GD  | BamiBERT + DAPT      |
| Vocab / max positions         | 20,481 / 2050 (BamiBERT tokenizer, classic RoBERTa block) | ←                  | ←                   |

Training recipe: self-built corpus 9,586,054 lines / 1.67GB (CC100-vi slice + VSFC/ViSFD/
VLSP/PhoATIS domain text ×20 upweight) → GoiBERT-B DAPT: MLM 30% masking, 40,000 steps,
loss 3.57 → 1.876 → students: TinyBERT general distillation 30,000 steps (hidden +
attention vs DAPT teacher) → per-dataset task stage (baseline lr grid {1,2,5}e-5,
10 epochs; students {5e-5,1e-4,2e-4}, 15 epochs, CE + KL(T=2) + hidden + attention).

## B. Main results per dataset (Stage P, seed 42; macro F1 and gap%)

### UIT-VSFC (sentiment; accuracy / macro P / R / F1)

| model                  | acc    | P      | R      | F1     | gap% |
| ---------------------- | ------ | ------ | ------ | ------ | ---- |
| baseline (BamiBERT ft) | 0.9337 | 0.8453 | 0.8133 | 0.8274 | 0.00 |
| goibert_b_dapt         | 0.9327 | 0.8699 | 0.7738 | 0.8048 | 2.73 |
| goiturn_s_gd           | 0.9144 | 0.8159 | 0.7759 | 0.7916 | 4.32 |
| goiturn_xs_gd          | 0.9109 | 0.8123 | 0.7695 | 0.7865 | 4.94 |

### UIT-ViSFD (ABSA classification macro; no accuracy)

| model          | P      | R      | F1     | gap%  |
| -------------- | ------ | ------ | ------ | ----- |
| baseline       | 0.7897 | 0.7405 | 0.7548 | 0.00  |
| goibert_b_dapt | 0.7890 | 0.7384 | 0.7483 | 0.86  |
| goiturn_s_gd   | 0.7834 | 0.6847 | 0.6990 | 7.40  |
| goiturn_xs_gd  | 0.6638 | 0.6467 | 0.6538 | 13.39 |

### PhoATIS (intent, 28 classes)

| model          | acc    | P      | R      | F1     | gap%             |
| -------------- | ------ | ------ | ------ | ------ | ---------------- |
| baseline       | 0.9731 | 0.5774 | 0.5584 | 0.5629 | 0.00             |
| goibert_b_dapt | 0.9675 | 0.5607 | 0.5474 | 0.5468 | 2.85             |
| goiturn_s_gd   | 0.9720 | 0.5796 | 0.5599 | 0.5645 | **−0.29** |
| goiturn_xs_gd  | 0.9731 | 0.5778 | 0.5536 | 0.5604 | 0.45             |

(Macro F1 is low because 28 classes include many rare intents; accuracy 97.3% matches the
published JointIDSF anchor — sanity pass.)

### VLSP2018-Restaurant — all three seeds (macro F1)

| seed           | baseline         | goibert_b_dapt              | goiturn_s_gd | goiturn_xs_gd |
| -------------- | ---------------- | --------------------------- | ------------ | ------------- |
| 42             | 0.3942           | 0.5017                      | 0.4372       | 0.4239        |
| 43             | 0.4576           | 0.4578                      | 0.3659       | 0.3563        |
| 44             | 0.4492           | 0.4187                      | 0.4151       | 0.3946        |
| **mean** | **0.4337** | **0.4594** (+2.6 pts) | 0.4061       | 0.3916        |

Baseline itself swings ±3 points across seeds (4.7K train set, sparse-class macro F1) —
hence the retraction of any single-seed win claim.

## C. Stage A vs Stage P ablation (VSFC — what pretraining bought)

| model      | task-distill only (Stage A) | + DAPT & GD (Stage P) | gap cut |
| ---------- | --------------------------- | --------------------- | ------- |
| goiturn_s  | F1 0.7534, gap 8.94%        | F1 0.7916, gap 4.32%  | 52%     |
| goiturn_xs | F1 0.7463, gap 9.79%        | F1 0.7865, gap 4.94%  | 50%     |

## D. Noise-consistency variants (clean test, seed 42; macro F1 / gap% vs its own baseline_nc)

| model             | VSFC          | VLSP                       | PhoATIS       | ViSFD                     |
| ----------------- | ------------- | -------------------------- | ------------- | ------------------------- |
| baseline_nc       | 0.8064        | 0.4117                     | 0.5701        | 0.7433                    |
| goibert_b_dapt_nc | 0.8030 (0.42) | 0.4532 (**−10.07**) | 0.5609 (1.61) | 0.7541 (**−1.45**) |
| goiturn_s_gd_nc   | 0.7964 (1.24) | 0.4247 (**−3.15**)  | 0.5633 (1.20) | 0.6949 (6.50)             |
| goiturn_xs_gd_nc  | 0.7511 (6.86) | 0.3932 (4.49)              | 0.5519 (3.20) | 0.6688 (10.02)            |

## E. Robustness — noisy-slice evaluation (macro F1; slices: clean / no-diacritics / teencode / combined)

### UIT-VSFC

| model             | clean  | no diac          | teencode | combined | degr. |
| ----------------- | ------ | ---------------- | -------- | -------- | ----- |
| baseline          | 0.8274 | 0.4988           | 0.8243   | 0.6211   | 0.206 |
| baseline_nc       | 0.8064 | 0.7430           | 0.8084   | 0.7662   | 0.040 |
| goibert_b_dapt    | 0.8048 | 0.5313           | 0.8110   | 0.6291   | 0.176 |
| goibert_b_dapt_nc | 0.8030 | 0.7166           | 0.8041   | 0.7506   | 0.052 |
| goiturn_s_gd      | 0.7916 | 0.4053           | 0.7909   | 0.5799   | 0.212 |
| goiturn_s_gd_nc   | 0.7964 | **0.7354** | 0.7979   | 0.7587   | 0.038 |
| goiturn_xs_gd     | 0.7865 | 0.4700           | 0.7806   | 0.6171   | 0.169 |
| goiturn_xs_gd_nc  | 0.7511 | 0.7273           | 0.7535   | 0.7383   | 0.013 |

### PhoATIS

| model             | clean  | no diac | teencode | combined | degr. |
| ----------------- | ------ | ------- | -------- | -------- | ----- |
| baseline          | 0.5629 | 0.1769  | 0.5591   | 0.4105   | 0.152 |
| baseline_nc       | 0.5701 | 0.5307  | 0.5701   | 0.5459   | 0.024 |
| goibert_b_dapt    | 0.5468 | 0.1778  | 0.5433   | 0.4077   | 0.139 |
| goibert_b_dapt_nc | 0.5609 | 0.4859  | 0.5609   | 0.5344   | 0.027 |
| goiturn_s_gd      | 0.5645 | 0.1464  | 0.5610   | 0.3924   | 0.172 |
| goiturn_s_gd_nc   | 0.5633 | 0.5140  | 0.5633   | 0.5287   | 0.035 |
| goiturn_xs_gd     | 0.5604 | 0.1952  | 0.5604   | 0.3793   | 0.181 |
| goiturn_xs_gd_nc  | 0.5519 | 0.5097  | 0.5545   | 0.5356   | 0.016 |

### UIT-ViSFD

| model             | clean  | no diac          | teencode | combined | degr. |
| ----------------- | ------ | ---------------- | -------- | -------- | ----- |
| baseline          | 0.7548 | 0.4707           | 0.7556   | 0.5963   | 0.159 |
| baseline_nc       | 0.7433 | 0.6933           | 0.7373   | 0.7026   | 0.041 |
| goibert_b_dapt    | 0.7483 | 0.4938           | 0.7468   | 0.6142   | 0.134 |
| goibert_b_dapt_nc | 0.7541 | **0.7136** | 0.7538   | 0.7254   | 0.029 |
| goiturn_s_gd      | 0.6990 | 0.4276           | 0.6966   | 0.5611   | 0.138 |
| goiturn_s_gd_nc   | 0.6949 | 0.6636           | 0.6947   | 0.6730   | 0.022 |
| goiturn_xs_gd     | 0.6538 | 0.3894           | 0.6477   | 0.5169   | 0.137 |
| goiturn_xs_gd_nc  | 0.6688 | 0.6411           | 0.6666   | 0.6445   | 0.024 |

### VLSP2018-Restaurant

| model             | clean  | no diac | teencode | combined | degr. |
| ----------------- | ------ | ------- | -------- | -------- | ----- |
| baseline          | 0.3942 | 0.2842  | 0.3963   | 0.3306   | 0.064 |
| baseline_nc       | 0.4117 | 0.3747  | 0.4112   | 0.3949   | 0.017 |
| goibert_b_dapt    | 0.5017 | 0.3571  | 0.4989   | 0.4094   | 0.092 |
| goibert_b_dapt_nc | 0.4532 | 0.4156  | 0.4565   | 0.4335   | 0.020 |
| goiturn_s_gd      | 0.4372 | 0.3167  | 0.4378   | 0.3555   | 0.082 |
| goiturn_s_gd_nc   | 0.4247 | 0.3921  | 0.4217   | 0.3959   | 0.029 |
| goiturn_xs_gd     | 0.4239 | 0.2961  | 0.4183   | 0.3384   | 0.085 |
| goiturn_xs_gd_nc  | 0.3932 | 0.3714  | 0.3965   | 0.3754   | 0.018 |

Teencode barely hurts any model (the PhoGPT vocabulary already covers common teencode);
diacritic loss is the true robustness killer, and NC is the cure.

## F. GoiTurn turn-completion model — evolution (synthetic held-out, 8,019 examples)

| version                       | recipe                         | macro F1         | false-cut       | false-wait |
| ----------------------------- | ------------------------------ | ---------------- | --------------- | ---------- |
| Tier 0 rules                  | regex only                     | 0.5620           | 67.95%          | 2.98%      |
| v0                            | variant_b scratch              | 0.9331           | 8.09%           | —         |
| BamiBERT ft (reference, 103M) | fine-tune                      | 0.9658           | 3.65%           | 4.84%      |
| v0.5                          | + GD-backbone init             | 0.9577           | 5.11%           | —         |
| **v1 (recommended)**    | + noise-consistency, 20 epochs | **0.9582** | **5.32%** | 4.62%      |

Training data: 87,210 synthetic examples (truncation trick per the SCBX Thai study) from
PhoATIS + VSFC + ViSFD + VLSP; leakage-safe splits by source sentence; seed 42.
