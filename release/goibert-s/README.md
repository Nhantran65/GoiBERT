---
language:
- vi
---

# GoiBERT-S: a small, fast Vietnamese encoder distilled from BamiBERT

GoiBERT-S is a **26.4M-parameter** Vietnamese RoBERTa encoder, about **¼ the size of BamiBERT**. It is trained by general distillation from BamiBERT on a cleaned 329M-token Vietnamese corpus. It uses BamiBERT's tokenizer (20,481 tokens) and works on raw text; no word segmentation is needed.

After task fine-tuning (see *Evaluation*), GoiBERT-S scores within **0.07 points** of BamiBERT on average across 13 metrics on 7 Vietnamese benchmarks. It runs **2–3.5× faster** and uses about **⅓ of the GPU memory**.

> This is the **backbone encoder only**, with no task head. Fine-tune it for your task.

## Model details

| | |
|---|---|
| Architecture | RoBERTa encoder: 10 layers, hidden size 384, 6 attention heads, FFN size 1,536 |
| Parameters | 26,397,696 (encoder, no pooler) |
| Tokenizer | BamiBERT tokenizer, vocab 20,481; raw (unsegmented) Vietnamese text |
| Recommended max length | **256 tokens**. Distillation used 256-token sequences, so position embeddings beyond 256 are untrained. |
| Teacher | [BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT) (103M), used unchanged, with no further pretraining |
| Family | GoiBERT-S (26M), GoiBERT-M (50M) |

## Usage

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("2HeartsVN/GoiBERT-S")
model = AutoModel.from_pretrained("2HeartsVN/GoiBERT-S")

inputs = tokenizer("Sản phẩm dùng tốt, giao hàng nhanh.", return_tensors="pt",
                   truncation=True, max_length=256)
hidden = model(**inputs).last_hidden_state   # (1, seq_len, 384)
```

## Training

**Initialization**

- Word embeddings: PCA projection (768 → 384) of BamiBERT's word embeddings.
- Transformer layers: random initialization.

**General distillation** (TinyBERT-style, no labels). The student learns to match two things from BamiBERT:

- hidden states, through learned linear projections;
- attention maps, averaged over heads.

Student layers map to teacher layers [0,1,2,4,5,6,7,8,10,11,12].

| Setting | Value |
|---|---|
| Steps | 120,000 |
| Batch | 32 sequences × 256 tokens (about 0.98B tokens seen, about 3 passes over the corpus) |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01, 1,000 warmup steps, linear decay |
| Precision | bf16 autocast |
| Hardware | 1× NVIDIA L40S |

**Distillation corpus (corpus v2)**

The corpus has 328.6M BamiBERT tokens. Documents are shuffled as whole units, so paragraphs stay in order within each document.

| Source | Tokens | Share |
|---|---:|---:|
| HPLT 3.0 Vietnamese web: bins 7–9, spun/machine-translated text filtered out, at most 2% per host, near- and exact-deduplicated | 232.9M | 71% |
| Vietnamese Wikipedia (articles ≥ 300 words) | 49.4M | 15% |
| User-generated and forum text (from HPLT) | 25.5M | 8% |
| Unlabeled **training-split** text of 8 Vietnamese benchmarks, repeated up to 10× | 20.9M | 6% |

**Decontamination**

- Validation and test text from the 8 benchmarks was removed from the corpus: 30,334 paragraphs and 2,630 documents.
- A paragraph was removed if it shared any 13-gram with a validation or test example, or, for short examples, matched one exactly.
- A final scan found 0 exact matches with any validation or test example.

## Evaluation

GoiBERT-S was fine-tuned on 7 Vietnamese benchmarks (13 metrics).

**Protocol**

- 3 seeds per task.
- Epoch selected on validation.
- Test set evaluated **once**, from a manifest locked before test access.

**Fine-tuning recipe**

- 30 epochs, batch size 32, layer-wise learning-rate decay (γ = 0.8).
- Loss: cross-entropy plus knowledge distillation from out-of-fold CafeBERT soft labels (α = 0.5, T = 2).

The BamiBERT baseline was fine-tuned with plain cross-entropy, with the learning rate selected per task on validation (our local reproduction).

**Test results** (mean of 3 seeds, %)

| Dataset | Metric | BamiBERT (103M) | **GoiBERT-S (26M)** | GoiBERT-M (50M) |
|---|---|---:|---:|---:|
| UIT-VSFC sentiment | Acc / macro-F1 | 93.50 / 82.36 | 92.70 / 81.50 | 93.56 / 82.03 |
| UIT-VSFC topic | Acc / macro-F1 | 88.82 / 78.88 | **89.36 / 80.17** | 89.73 / 80.42 |
| UIT-ViSFD | aspect detection F1 / aspect-sentiment F1 | 89.26 / 84.41 | **89.30** / 83.94 | 89.81 / 84.78 |
| UIT-ABSA Restaurant | detection F1 / aspect-sentiment F1 | 87.03 / 73.53 | 86.91 / **74.15** | 88.14 / 76.00 |
| UIT-ABSA Hotel | detection F1 / aspect-sentiment F1 | 80.15 / 73.94 | **80.66 / 75.01** | 80.96 / 75.67 |
| ViSpamReviews (4-class) | Acc / macro-F1 | 90.19 / 77.55 | 89.88 / 76.52 | 90.75 / 78.13 |
| PhoNER_COVID19 (syllable) | entity F1 (strict) | 94.32 | 92.97 | 93.47 |
| **Mean, 13 metrics** | | 84.15 | 84.08 | 84.88 |

Bold marks metrics where GoiBERT-S ≥ BamiBERT (6 of 13).

**Inference efficiency**

Encoder only. GPU: NVIDIA L40S, bf16. CPU: fp32, 8 threads.

| | BamiBERT | GoiBERT-S |
|---|---:|---:|
| GPU latency, batch 1 × 128 | 2.93 ms | 2.52 ms |
| GPU throughput, batch 32 × 256 | 2,397 seq/s | **8,290 seq/s (3.5×)** |
| CPU latency, batch 1 × 128 | 31.0 ms | **14.2 ms (2.2×)** |
| Peak GPU memory, batch 32 × 256 | 801 MB | **248 MB** |
| Weights (bf16) | 206 MB | 53 MB |
