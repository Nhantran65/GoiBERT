# CafeBERT probe — published SOTA (Findings NAACL 2024) on the BamiBERT 5-task suite

Date 2026-09-14. Question: is BamiBERT (preprint) really the model to beat, or is the *published* SOTA CafeBERT
(XLM-R-large 550M, continued-pretrained on 18 GB Vietnamese; Nguyen et al., Findings of NAACL 2024) already ahead on
our tasks? BamiBERT never compared against it.

Protocol: `scripts/research_checks/probe_mmbert_finetune.py --model uitnlp/CafeBERT --tokenizer FacebookAI/xlm-roberta-large`
(the CafeBERT repo ships only the sentencepiece file; sha256 identical to XLM-R-large's, so the XLM-R fast tokenizer is
equivalent). Plain recipe: CE, AdamW, warmup 10 %, first-token head, batch 32, 20 epochs, seed 42, max_len 256,
**validation-only** best-epoch selection F1. lr 2e-5 on all tasks; 1e-5 also on Restaurant/Hotel. ~25–40 min/task on
one L40S. Results: `artifacts/research_diagnostics/mmbert_probe/*CafeBERT*.json`.

## Validation, seed 42 (selection F1: macro-F1 for VSFC, pair micro-F1 for ABSA/ViSFD)

| Task | BamiBERT E14 (110M, best lr) | E19 goibert-b-dapt +NC+LLRD (103M) | Our best so far | mmBERT-base (110M+emb) | **CafeBERT (550M)** | Δ vs BamiBERT | Δ vs our best |
|---|---|---|---|---|---|---|---|
| VSFC Sentiment | 86.33 | 85.71 | 86.33 (E14) | – | **87.24** (@13) | +0.9 | +0.9 |
| VSFC Topic | 79.53 | 81.77 | 82.17 (E2 teacher CE) | – | **81.75** (@5) | +2.2 | −0.4 |
| ViSFD | 84.38 | 85.08 | 85.08 (E19) | – | **85.14** (@20) | +0.8 | +0.1 |
| ABSA Restaurant | 74.74 | 74.51 | 75.74 (E22 d1_uniform) | 74.38 | **77.14** (@19; 76.78 at lr 1e-5) | +2.4 | +1.4 |
| ABSA Hotel | 73.24 | 73.63 | 73.63 (E19) | 73.66 | **73.60** (@17; 72.17 at lr 1e-5) | +0.4 | ±0 |

Seed noise on these tasks ≈ 0.5–1 F1 (E21/E22 multi-seed), so Sentiment/ViSFD/Hotel are within noise; Topic and
Restaurant are clear wins for CafeBERT over BamiBERT. Curves are still rising at epoch 20 on Restaurant (lr 1e-5) and
ViSFD, so these CafeBERT numbers are a *lower* bound.

## What this changes

1. **The published SOTA on this suite is CafeBERT, not BamiBERT.** With a plain recipe, one seed and two lrs, CafeBERT
   ≥ BamiBERT on all 5 tasks and ≥ everything we built in E1–E22 except Topic (tie). BamiBERT's "SOTA" holds only
   inside the from-scratch / base-size bracket.
2. Backbone pretraining scale/quality is again the dominant lever (largest gains on Restaurant +2.4 and Topic +2.2),
   consistent with DAPT (+2.6 Topic) and with all fine-tuning-time mechanisms being capped.
3. Paper target must be restated: beat **CafeBERT-large (550M)** with a base-size model (ViModernBERT, ~110M
   non-embedding, 3× faster, 8k ctx) — an efficiency-and-quality claim — *and* report CafeBERT as a baseline on
   both this suite and VLUE. Simply beating BamiBERT is beating a preprint that a published model already beats.
4. Week-1 gate for the CPT plan is now: after ≈1 B Vietnamese tokens of CPT, mmBERT-base must close the Restaurant gap
   to CafeBERT (74.4 → ≥ 76.5) while holding Hotel/ViSFD.

Caveats: validation-only, single seed, no LLRD/NC/composite on CafeBERT (those would likely add ~+1 on Restaurant as
they did for us), no test split touched.
