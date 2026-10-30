---
license: other
license_name: goibert-community-model-license-1.0
license_link: https://huggingface.co/TwoheartsTechnologies/GoiBERT-S/blob/main/LICENSE
language:
- vi
---

# GoiBERT-S (GoiBERT-base): a small, fast Vietnamese encoder

GoiBERT-S is the Hugging Face release of **GoiBERT-base** from the paper [GoiBERT: Compact Vietnamese Language Models via Two-Stage Knowledge Distillation](https://arxiv.org/abs/2610.30067). It is a **26.4M-parameter** Vietnamese RoBERTa encoder, about **¼ the size of BamiBERT**, distilled from BamiBERT on a cleaned 329M-token Vietnamese corpus. It uses BamiBERT's tokenizer (20,481 tokens) and takes raw text, with no word segmentation needed.

On 8 Vietnamese benchmarks, GoiBERT-base averages **80.02**, about **99% of BamiBERT's 80.65**, at a quarter of the size. It runs at **3.5× BamiBERT's batched GPU throughput** and uses about **⅓ of the GPU memory**.

> This is the **backbone encoder only**, with no task head. Fine-tune it for your task.

## Model details

| | |
|---|---|
| Paper name | GoiBERT-base |
| Architecture | RoBERTa encoder: 10 layers, hidden size 384, 6 attention heads, FFN size 1,536 |
| Parameters | 26,397,696 (encoder, no pooler) |
| Tokenizer | BamiBERT tokenizer, vocabulary 20,481; takes raw (unsegmented) Vietnamese text |
| Recommended max length | **256 tokens**. Distillation used 256-token sequences, so position embeddings beyond 256 are untrained. |
| Teachers | Stage 1: [BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT) (102.4M), used unchanged. Stage 2 (fine-tuning only): [CafeBERT](https://huggingface.co/uitnlp/CafeBERT) out-of-fold soft labels |
| Family | [GoiBERT-S](https://huggingface.co/TwoheartsTechnologies/GoiBERT-S) (GoiBERT-base, 26.4M), [GoiBERT-M](https://huggingface.co/TwoheartsTechnologies/GoiBERT-M) (GoiBERT-large, 49.4M) |
| Paper | [arXiv:2610.30067](https://arxiv.org/abs/2610.30067) |

## Usage

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("TwoheartsTechnologies/GoiBERT-S")
model = AutoModel.from_pretrained("TwoheartsTechnologies/GoiBERT-S")

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

Results are from Table 1 of the [paper](https://arxiv.org/abs/2610.30067): test sets, mean of 3 fine-tuning seeds, one metric per task (%).

**Protocol**

- 3 seeds per task; epoch and learning rate chosen on validation.
- Test set evaluated **once** per model, from a manifest locked before test access.

**Fine-tuning recipe**

- 30 epochs, batch size 32, layer-wise learning-rate decay (γ = 0.8).
- Loss: cross-entropy plus knowledge distillation from out-of-fold [CafeBERT](https://huggingface.co/uitnlp/CafeBERT) soft labels (α = 0.5, T = 2). CafeBERT is used only as a task teacher during fine-tuning; it is not part of the released weights.

**BamiBERT baseline.** Re-run under the same downstream protocol, with its learning rate tuned on validation and standard cross-entropy fine-tuning.

| Task | Metric | BamiBERT (102.4M) | GoiBERT-base (26.4M) | GoiBERT-large (49.4M) |
|---|---|---:|---:|---:|
| ViNLI | macro-F1 | 80.18 | 75.86 | 78.52 |
| ViSpamReviews | macro-F1 | 77.55 | 76.52 | 78.13 |
| PhoNER_COVID19 (syllable) | entity F1 | 94.32 | 92.97 | 93.47 |
| UIT-ABSA Restaurant | aspect-sentiment pair F1 | 73.53 | 74.15 | 76.00 |
| UIT-ABSA Hotel | aspect-sentiment pair F1 | 73.94 | 75.01 | 75.67 |
| UIT-VSFC topic | macro-F1 | 78.88 | 80.17 | 80.42 |
| UIT-VSFC sentiment | macro-F1 | 82.36 | 81.50 | 82.03 |
| UIT-ViSFD | aspect-sentiment pair F1 | 84.41 | 83.94 | 84.78 |
| **Average (8 tasks)** | | **80.65** | **80.02** | **81.13** |
| Parameters | | 102.4M | 26.4M | 49.4M |
| GFLOPs per 128-token input | | 22.4 | 4.8 | 10.1 |

The paper's Table 1 also reports CafeBERT (task teacher), XLM-R, ViSoBERT and PhoBERT-v2 on the same tasks.

**Inference efficiency**

Encoder only. GPU: NVIDIA L40S, bf16. CPU: fp32, 8 threads.

| | BamiBERT | GoiBERT-S |
|---|---:|---:|
| GPU latency, batch 1 × 128 | 2.93 ms | 2.52 ms |
| GPU throughput, batch 32 × 256 | 2,397 seq/s | **8,285 seq/s (3.5×)** |
| CPU latency, batch 1 × 128 | 31.0 ms | **14.2 ms (2.2×)** |
| Peak GPU memory, batch 32 × 256 | 801 MB | **248 MB** |
| Weights (bf16) | 206 MB | 53 MB |

## Limitations

As stated in the paper:

- GoiBERT trails BamiBERT on ViNLI and PhoNER_COVID19; compression is costlier for inference and entity recognition than for sentiment and classification.
- The comparison with BamiBERT is not a pure architecture comparison: GoiBERT also uses CafeBERT distillation, layer-wise learning-rate decay and benchmark training text in its distillation corpus.
- General distillation was run once per model, so seed ranges reflect fine-tuning only.
- Inputs are limited to 256 tokens; dialects and unseen domains have not been evaluated.
- Efficiency figures exclude tokenization, task heads and the cost of training the teachers.

## Citation

```bibtex
@article{goibert2026,
  title   = {GoiBERT: Compact Vietnamese Language Models via Two-Stage Knowledge Distillation},
  author  = {Tran, Quang Anh Kiet and Tran, Nam Anh and Trinh, Manh Hung and Le, Thanh Danh and Vo, Khang Nhat Hoang and Tran, Huynh Dai Nhan},
  journal = {arXiv preprint arXiv:2610.30067},
  year    = {2026}
}
```

## License and commercial use

This release is offered under the [GoiBERT Community Model License](LICENSE)
for Twohearts' licensable contributions. Research and evaluation are
royalty-free, subject to the license and applicable upstream terms.
Commercial use is royalty-free for an organization group with
**fewer than 100 personnel AND fewer than 1,000
customers**, as defined in the license.

An organization group meeting **either** threshold needs a separate
written commercial agreement before
commercial use. Contact **[nhan@twohearts.vn](mailto:nhan@twohearts.vn)**.

GoiBERT was developed using BamiBERT. Applicable **BSD-3-Clause-Clear and
Qualcomm Responsible AI License** obligations remain in effect; our license
does not replace them or authorize exceptions. See [NOTICE](NOTICE),
[BSD-3-Clause-Clear](licenses/BSD-3-Clause-Clear.txt) and
[Qualcomm RAIL](licenses/Qualcomm-RAIL.txt).

This release contains the backbone encoder. Task heads, benchmark datasets
and the training corpus are not licensed or distributed by this model
license. This community license has commercial conditions and is not an
OSI-approved open-source license. The license does not waive third-party
rights or replace permissions required for training data or upstream
materials. It does not retroactively amend an earlier model release.
