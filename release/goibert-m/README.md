---
language:
- vi
---

# GoiBERT-M: a half-size Vietnamese encoder that outperforms BamiBERT

GoiBERT-M is a **49.4M-parameter** Vietnamese RoBERTa encoder, about **½ the size of BamiBERT**. It is trained by general distillation from BamiBERT on a cleaned Vietnamese corpus of 329M tokens. It uses BamiBERT's tokenizer (20,481 tokens) and takes raw text, with no word segmentation needed.

After task fine-tuning (see *Evaluation*), GoiBERT-M **outperforms BamiBERT on 11 of 13 metrics** across 7 Vietnamese benchmarks, **+0.73 points on average**. It has about **2× the batched GPU throughput** and uses about **½ the GPU memory**.

> This is the **backbone encoder only**. It has no task head; fine-tune it for your task.

## Model details

| | |
|---|---|
| Architecture | RoBERTa encoder: 12 layers, hidden size 512, 8 attention heads, FFN size 2,048 |
| Parameters | 49,366,016 (encoder, no pooler) |
| Tokenizer | BamiBERT tokenizer, vocabulary 20,481; takes raw (unsegmented) Vietnamese text |
| Recommended max length | **256 tokens**. Distillation used 256-token sequences, so position embeddings beyond 256 are untrained. |
| Teacher | [BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT) (103M), used unchanged, with no further pretraining |
| Family | GoiBERT-S (26M), GoiBERT-M (50M) |

## Usage

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("2HeartsVN/GoiBERT-M")
model = AutoModel.from_pretrained("2HeartsVN/GoiBERT-M")

inputs = tokenizer("Sản phẩm dùng tốt, giao hàng nhanh.", return_tensors="pt",
                   truncation=True, max_length=256)
hidden = model(**inputs).last_hidden_state   # (1, seq_len, 512)
```

## Training

**Initialization**

- Word embeddings: PCA projection (768 → 512) of BamiBERT's word embeddings.
- Transformer layers: random initialization.

**General distillation** (TinyBERT-style, no labels)

The student learns to match two signals from BamiBERT, one student layer to one teacher layer (mapping [0,1,…,12]):

- hidden states, through learned linear projections;
- attention maps, averaged over heads.

| Setting | Value |
|---|---|
| Steps | 120,000 |
| Batch | 32 sequences × 256 tokens (about 0.98B tokens seen, about 3 passes over the corpus) |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01, 1,000 warmup steps, linear decay |
| Precision | bf16 autocast |
| Hardware | 1× NVIDIA L40S |

**Shape selection.** Two 50–60M shapes were distilled with the same recipe and compared on validation (5 tasks, seed 42):

| Shape | Validation score |
|---|---:|
| **12 × 512** (this model) | **82.86** |
| 6 × 768 (initialized from BamiBERT layers 2, 4, …, 12) | 82.50 |

The 6 × 768 variant has about 2× lower single-sequence GPU latency.

**Distillation corpus (corpus v2).** 328.6M BamiBERT tokens. Documents are shuffled as whole documents, so paragraphs within each document stay in order.

| Source | Tokens | Share |
|---|---:|---:|
| HPLT 3.0 Vietnamese web: bins 7–9, spun and machine-translated text filtered out, at most 2% per host, near- and exact-deduplicated | 232.9M | 71% |
| Vietnamese Wikipedia (articles of at least 300 words) | 49.4M | 15% |
| User-generated and forum text (from HPLT) | 25.5M | 8% |
| Unlabeled **train-split** text of 8 Vietnamese benchmarks, repeated up to 10× | 20.9M | 6% |

**Decontamination.** Any paragraph that shares a 13-gram with the **validation or test** text of the 8 benchmarks, or matches it exactly (for short texts), was removed: 30,334 paragraphs and 2,630 documents in total. A final scan found 0 exact matches with any validation or test example.

## Evaluation

GoiBERT-M was fine-tuned on 7 Vietnamese benchmarks, covering 13 metrics.

**Protocol**

- 3 seeds per task.
- Best epoch and learning rate chosen on validation. The learning rate for each task was picked from two candidates.
- Test set evaluated **once**, from a manifest locked before test access.

**Fine-tuning recipe**

- 30 epochs, batch size 32, layer-wise learning-rate decay (γ = 0.8).
- Loss: cross-entropy plus knowledge distillation from out-of-fold CafeBERT soft labels (α = 0.5, T = 2).

**BamiBERT baseline.** Fine-tuned with plain cross-entropy (no distillation), with the learning rate selected per task on validation. These are our local reproduction runs.

**Test results** (mean of 3 seeds, %). Bold: GoiBERT-M ≥ BamiBERT (11 of 13).

| Dataset | Metric | BamiBERT (103M) | GoiBERT-S (26M) | **GoiBERT-M (50M)** |
|---|---|---:|---:|---:|
| UIT-VSFC sentiment | Acc / macro-F1 | 93.50 / 82.36 | 92.70 / 81.50 | **93.56** / 82.03 |
| UIT-VSFC topic | Acc / macro-F1 | 88.82 / 78.88 | 89.36 / 80.17 | **89.73 / 80.42** |
| UIT-ViSFD | aspect detection F1 / aspect-sentiment F1 | 89.26 / 84.41 | 89.30 / 83.94 | **89.81 / 84.78** |
| UIT-ABSA Restaurant | detection F1 / aspect-sentiment F1 | 87.03 / 73.53 | 86.91 / 74.15 | **88.14 / 76.00** |
| UIT-ABSA Hotel | detection F1 / aspect-sentiment F1 | 80.15 / 73.94 | 80.66 / 75.01 | **80.96 / 75.67** |
| ViSpamReviews (4-class) | Acc / macro-F1 | 90.19 / 77.55 | 89.88 / 76.52 | **90.75 / 78.13** |
| PhoNER_COVID19 (syllable) | entity F1 (strict) | 94.32 | 92.97 | 93.47 |
| **Mean, 13 metrics** | | 84.15 | 84.08 | **84.88** |

**Inference efficiency.** Encoder only. GPU: NVIDIA L40S, bf16. CPU: fp32, 8 threads.

| | BamiBERT | GoiBERT-M |
|---|---:|---:|
| GPU latency, batch 1 × 128 | 2.93 ms | 3.26 ms |
| GPU latency, batch 32 × 128 | 6.09 ms | **3.28 ms (1.9×)** |
| GPU throughput, batch 32 × 256 | 2,397 seq/s | **4,827 seq/s (2.0×)** |
| CPU latency, batch 1 × 128 | 31.0 ms | **19.4 ms (1.6×)** |
| Peak GPU memory, batch 32 × 256 | 801 MB | **410 MB** |
| Weights (bf16) | 206 MB | 99 MB |

GoiBERT-M has the same depth as BamiBERT (12 layers). For single sequences on GPU, latency is bound by per-layer overhead, so it matches BamiBERT. The speed-up appears with batched inference and on CPU.
