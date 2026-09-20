# GoiTurn Architecture Scaling Experiment — Plan

Date: 2026-08-30. Owner: Line. Type: feasibility / architecture-scaling experiment (NOT production GoiTurn).
Proposal: `GoiTurn_Vietnamese_Proposal.md` (C:\Users\ADMIN\Downloads).

## Measured facts (inspected 2026-08-30, not assumed)

BamiBERT checkpoint `Qualcomm-AI-Research/BamiBERT` (HF, sha 57bc1340, 2026-07-03):

| Field | Value |
|---|---|
| Architecture | `RobertaForMaskedLM` — plain RoBERTa/BERT block |
| Layers / hidden / heads / FFN | 12 / 768 / 12 / 3072 |
| Vocab | 20,481 (PhoGPT BPE, `tokenizer.json`) |
| Max positions | 2050 (absolute learned positions) |
| Act / norm | GELU / LayerNorm (post-norm BERT style), biases present |
| Params (checkpoint, F32) | 102,993,666 total; encoder+embeddings 102,360,576 |
| License | BSD-3-Clause-Clear + Qualcomm Responsible AI License; not gated |

Mismatch vs proposal §13: proposal's GoiTurn-S (21.1M) assumes RoPE, GeGLU, RMSNorm, factorized
embeddings, custom 16K vocab. BamiBERT has NONE of those. This experiment scales the REAL
BamiBERT architecture (dense embeddings, 20,481 vocab, absolute positions), so parameter counts
differ from proposal §13.2. Flagged to Line in the final report.

Dataset mismatch: UIT-ABSA (Restaurant) — the proposal's headline benchmark — is not publicly
downloadable (author-request only, no HF/GitHub mirror found). Substitute: VLSP2018-ABSA-Restaurant
(same domain, public). Flagged to Line.

## Datasets (downloaded to `ml-models\data\raw\`)

| Dataset | Task | Rows | Role |
|---|---|---|---|
| VLSP2018-ABSA-Restaurant | 12-aspect ABSA (0=absent, 1–3=polarity) | 4,752 | F&B primary (UIT-ABSA proxy) |
| UIT-ViSFD | 10-aspect+OTHERS ABSA, `{ASPECT#Polarity}` labels | 13,718 | Commerce, in BamiBERT paper |
| UIT-VSFC sentiment | 3-class single-label | 11,426/1,583/3,166 | Sanity anchor: BamiBERT paper F1 83.41 |

## Architectures (exact param counts, encoder + embeddings, biases included)

| Model | L | H | A | FFN | Max pos | Params |
|---|--:|--:|--:|--:|--:|--:|
| Baseline BamiBERT | 12 | 768 | 12 | 3072 | 2050 | 102,360,576 |
| Variant A "wide" | 8 | 512 | 8 | 2048 | 514 | 35,970,048 |
| Variant B "deep-narrow" (GoiTurn-S geometry) | 10 | 384 | 6 | 1536 | 514 | 25,807,872 |

Variant B follows proposal §13.2 GoiTurn-S depth/width (10×384). It lands at 25.8M, not 21.1M,
because the real vocab is 20,481 dense (7.9M embed) vs the proposal's 16K factorized assumption.
Counts verified by instantiation in code before training.

## Training modes

- `finetune` (baseline only): fine-tune the real checkpoint. lr 2e-5, batch 32, 10 epochs,
  best epoch by validation macro F1, seq len 256, AdamW, seed 42.
- `scratch` (variants): random init, task data only. lr 5e-5, otherwise same. Lower bound —
  shows what the small architecture does with NO pretrained knowledge.
- `distill` (variants): word embeddings initialized by PCA projection of BamiBERT embeddings
  to student dim; loss = 0.5·CE + 0.5·KL(teacher logits, T=2) from the fine-tuned baseline of
  the same dataset. Realistic mid-point below full pretraining/distillation pipeline.

Deviation from BamiBERT paper protocol (30 epochs × lr grid {1e-5,2e-5,5e-5}) is deliberate —
feasibility budget; noted in report. 15 runs total = 3 datasets × (1 baseline + 2 variants × 2 modes).

## Metrics

- VSFC: accuracy, macro F1, weighted F1 (anchor vs paper's 83.41).
- ABSA (both): per-aspect 4-way head (none/pol1/pol2/pol3). Detection micro-F1 (aspect present),
  classification micro-F1 (aspect+polarity), matching the detection/classification split in the
  BamiBERT paper tables. Internal comparison only — protocol differs from published numbers.
- Latency: CPU, batch 1, seq {32, 128}, torch threads {1, 4}, 300 iters after 30 warmup,
  p50/p95 ms. Plus param count and fp32 size on disk. PyTorch eager (no ONNX/INT8 — out of scope).

## Success criteria

Report table (params / F1 / accuracy / latency per model per dataset) + conclusion on which
variant is worth carrying toward GoiTurn-S, and how much performance is lost per size step.
All results labeled measured; proposal numbers labeled target/hypothesis.

## Hardware

RTX 4060 Laptop 8GB, Python 3.13 venv at `ml-models\.venv` (torch cu126, transformers).
