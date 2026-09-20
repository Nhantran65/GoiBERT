---
language:
- en
- de
- es
- fr
- it
- nl
- pl
- pt
- ar
- hi
- ja
- ru
- tr
- vi
- zh
tags:
- liquid
- lfm2
- lfm2.5
- bidirectional
- masked-lm
- encoder
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: fill-mask
base_model: LiquidAI/LFM2.5-350M-Base
---

<div align="center">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" 
    alt="Liquid AI" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • 
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> • 
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> • 
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

# LFM2.5-Encoder-350M

**LFM2.5-Encoder** is a family of multilingual bidirectional encoders built on the LFM2 architecture, available in two sizes:

- [**LFM2.5-Encoder-230M**](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M) — a lightweight encoder for tight latency and memory budgets, punching above its size class.
- **LFM2.5-Encoder-350M** *(this model)* — a larger sibling for maximum downstream quality.

Both are masked language models with full bidirectional attention, designed to be fine-tuned into task-specific models (classification, token classification, retrieval, reranking, and semantic similarity) across 15 languages, and to run efficiently on-device.

Find more details about our encoders in our [blog post](https://www.liquid.ai/blog/lfm2-5-encoders).

**Key highlights:**

- **Top quality for its size.** Ahead of every model its size or smaller, and ~5 points above our own retrieval siblings.
- **General-purpose.** 8k context, strong across NLI, paraphrase, sentiment, and multilingual tasks.
- **Fast and on-device.** Matches or beats ModernBERT throughput, with a long-context edge on CPU.

> [!NOTE]
> 💻 **Demos**: We built the demos below from fine-tuned LFM2.5-Encoders. Each one runs in a CPU-only Hugging Face space:
> - **[Zero-shot prompt routing](https://huggingface.co/spaces/LiquidAI/prompt-routing)** — define your own routing lanes as free text. The model scores the whole prompt against every lane in one pass.
> - **[Zero-shot policy linting](https://huggingface.co/spaces/LiquidAI/policy-linting)** — check text against your company's rules, written as free text. It scores every token against every rule in one pass.
> - **[Spell checking](https://huggingface.co/spaces/LiquidAI/spellchecker)** — correct misspellings token by token.
> - **[PII detection](https://huggingface.co/spaces/LiquidAI/pii-detection)** — spot and remove 40 kinds of personal information across 16 languages.
> - **[Masked-diffusion text generation](https://huggingface.co/spaces/LiquidAI/masked-diffusion)** — bonus: run the encoder as a chatbot that generates text by iteratively unmasking instead of left to right.

## 📄 Model details

| Property | LFM2.5-Encoder-230M | LFM2.5-Encoder-350M |
|---|---|---|
| Type | Bidirectional encoder (masked language model) | Bidirectional encoder (masked language model) |
| Backbone | LFM2 | LFM2 |
| Total parameters | ~229.7M | ~354.5M |
| Hidden size | 1024 | 1024 |
| Vocabulary size | 65,536 | 65,536 |
| Context length | 8,192 tokens | 8,192 tokens |
| License | LFM Open License v1.0 | LFM Open License v1.0 |

**Supported languages:** English, German, Spanish, French, Italian, Dutch, Polish, Portuguese, Arabic, Hindi, Japanese, Russian, Turkish, Vietnamese, Chinese (15).

**Architecture.** LFM2.5-Encoder is built on the LFM2 hybrid backbone, which interleaves gated short-convolution blocks with grouped-query attention. For encoder use, the causal mask is replaced with full bidirectional (non-causal) attention and the model is trained with a masked language modeling head. The encoder body is exposed as `Lfm2BidirectionalModel`; masked-LM loading uses `Lfm2BidirectionalForMaskedLM`. Both are wired through `auto_map` and require `trust_remote_code=True`.

```
Lfm2BidirectionalForMaskedLM(
  (lfm2): Lfm2BidirectionalModel
  (lm_head): Linear(in_features=1024, out_features=65536, bias=False)
)
```

**Training.** LFM2.5-Encoder-350M is adapted from the LFM2 base and trained with a masked language modeling objective on a large multilingual corpus. Pre-training uses a two-stage schedule that extends the context window to up to 8,192 tokens.

We recommend fine-tuning LFM2.5-Encoder-350M for a range of downstream tasks, such as:

- **Text classification**: sentiment, topic, intent/routing, moderation, and business-text linting.
- **Token classification**: named-entity recognition, span extraction, and sequence labeling.
- **Retrieval and reranking**: a backbone for dense embedding or late-interaction (ColBERT-style) retrievers.
- **Semantic similarity**: STS, paraphrase, and duplicate detection.
- **Natural language inference and extractive QA**: sentence-pair reasoning and answer-span extraction.

## 🏃 How to run

Install the latest version of `transformers`:

```bash
pip install -U transformers
```

Run masked-token prediction:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-Encoder-350M", trust_remote_code=True)
mlm = AutoModelForMaskedLM.from_pretrained("LiquidAI/LFM2.5-Encoder-350M", trust_remote_code=True)

text = f"The capital of France is {tok.mask_token}."
enc = tok(text, return_tensors="pt")
with torch.no_grad():
    logits = mlm(**enc).logits
pos = (enc["input_ids"][0] == tok.mask_token_id).nonzero()[0].item()
print([tok.decode([t]).strip() for t in logits[0, pos].topk(5).indices.tolist()])
# -> ['Paris', 'Strasbourg', 'Paris', 'Lyon', 'Versailles']
```

For downstream tasks, load the encoder body and attach your own head (classification, token classification, regression, retrieval):

```python
from transformers import AutoModel
body = AutoModel.from_pretrained("LiquidAI/LFM2.5-Encoder-350M", trust_remote_code=True)
```

If your GPU supports it, we recommend using LFM2.5-Encoder-350M with Flash Attention 2 to reach the highest efficiency. To do so, install Flash Attention as follows, then use the model as normal:

```bash
pip install flash-attn
```

## 📊 Performance

For each benchmark task, we run a full supervised fine-tune and report that fine-tuned model's score. 
The results below span 14 models across 17 tasks from GLUE, SuperGLUE, and multilingual classification tasks. 
The full evaluation harness is open-sourced in the
[`eurobert-repro`](https://github.com/Liquid4All/encoder_eval) repository.

![benchmark_ranking](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/NQBokzgmbP3aThxAuQAt3.png)

### 17-task results (avg@5 fresh seeds ± std)

| Rank | Model | Params | 17-task mean | ± std |
|---:|---|---:|---:|---:|
| 1 | XLM-R XL (3.5B) | 3.5B | 83.06 | ±1.16 |
| 2 | ModernBERT-large (395M) | 395M | 81.68 | ±2.49 |
| 3 | XLM-R large (560M) | 560M | 81.34 | ±1.66 |
| **4** | **LFM2.5-Encoder-350M (ours)** | 350M | **81.02** | ±1.00 |
| 5 | mDeBERTa-v3 (280M) | 280M | 80.37 | ±1.06 |
| 6 | LFM2.5-Encoder-230M (ours) | 230M | 79.29 | ±1.02 |
| 7 | ModernBERT-base (149M) | 149M | 78.19 | ±1.39 |
| 8 | XLM-R base (280M) | 280M | 77.46 | ±1.63 |
| 9 | EuroBERT-210M | 210M | 76.87 | ±2.00 |
| 10 | mGTE-MLM (305M) | 305M | 76.53 | ±1.85 |
| 11 | LFM2.5-ColBERT-350M | 350M | 76.18 | ±1.25 |
| 12 | EuroBERT-610M | 610M | 75.87 | ±2.03 |
| 13 | LFM2.5-Embedding-350M | 350M | 75.68 | ±0.83 |
| 14 | EuroBERT-2.1B | 2.1B | 72.19 | ±5.59 |

<details>
<summary>Click to expand per-task results — all 17 tasks (avg@5 fresh seeds ± std)</summary>
### Per-task results — all 17 tasks (avg@5 fresh seeds ± std)

| Model | XNLI | PAWS-X | Amazon | MASSIVE | SeaHorse | CoLA* | SST-2* | MRPC* | STS-B* | QQP* | MNLI* | QNLI* | RTE* | BoolQ* | CB* | WiC* | WSC* | ALL |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| XLM-R XL (3.5B) | 87.12±0.45 | 93.30±0.41 | 62.29±0.05 | 88.21±0.32 | 59.51±3.52 | 84.58±1.18 | 95.69±0.30 | 87.65±1.69 | 90.20±0.93 | 91.72±0.07 | 90.09±0.11 | 94.47±0.28 | 82.38±3.51 | 83.70±0.24 | 89.88±3.72 | 66.55±1.37 | 64.62±1.58 | **83.06** |
| ModernBERT-large (395M) | 81.76±0.38 | 92.46±0.18 | 60.42±0.15 | 85.65±0.94 | 40.20±17.18 | 83.37±0.20 | 96.10±0.53 | 88.14±1.79 | 92.16±0.24 | 91.81±0.13 | 90.65±0.18 | 94.36±0.10 | 81.59±4.92 | 81.68±2.34 | 88.21±3.24 | 70.16±2.57 | 69.81±7.18 | **81.68** |
| XLM-R large (560M) | 84.69±0.59 | 93.23±0.78 | 61.58±0.13 | 88.50±0.17 | 56.12±2.31 | 83.34±1.75 | 93.83±1.04 | 88.77±2.15 | 91.35±0.23 | 90.48±0.23 | 88.29±0.07 | 93.08±0.23 | 80.79±3.14 | 80.54±0.79 | 78.21±8.69 | 66.24±5.41 | 63.65±0.43 | **81.34** |
| **LFM2.5-Encoder-350M (ours)** | 79.82±0.29 | 91.53±0.73 | 60.57±0.09 | 85.70±0.13 | 54.96±0.41 | 84.43±0.81 | 95.11±0.26 | 87.21±1.86 | 91.59±0.05 | 92.08±0.10 | 89.03±0.17 | 93.97±0.23 | 75.23±3.84 | 81.52±0.69 | 83.21±2.04 | 69.66±2.23 | 61.73±3.15 | **81.02** |
| mDeBERTa-v3 (280M) | 83.01±0.47 | 92.59±0.39 | 60.62±0.31 | 87.64±0.51 | 54.97±2.04 | 83.91±1.03 | 92.41±0.96 | 85.39±2.59 | 89.87±0.22 | 90.23±0.15 | 86.30±0.18 | 91.99±0.35 | 69.75±2.03 | 78.29±1.39 | 88.21±2.40 | 67.71±3.05 | 63.46±0.00 | **80.37** |
| LFM2.5-Encoder-230M (ours) | 77.63±0.31 | 90.86±0.24 | 59.97±0.21 | 85.52±0.69 | 54.61±0.62 | 81.42±1.56 | 94.08±0.37 | 80.20±2.66 | 90.99±0.12 | 91.71±0.07 | 87.98±0.22 | 92.96±0.37 | 67.29±1.97 | 76.54±1.01 | 83.21±4.48 | 70.31±1.34 | 62.69±1.05 | **79.29** |
| ModernBERT-base (149M) | 76.64±0.31 | 92.17±0.15 | 58.98±0.11 | 85.32±0.21 | 45.19±1.72 | 83.07±1.93 | 94.79±0.52 | 84.46±2.70 | 90.75±0.14 | 91.23±0.11 | 88.68±0.17 | 93.04±0.36 | 58.70±1.94 | 74.78±4.10 | 81.07±6.75 | 66.90±2.39 | 63.46±0.00 | **78.19** |
| XLM-R base (280M) | 78.20±0.80 | 91.36±0.41 | 60.01±0.12 | 87.47±0.49 | 51.08±3.75 | 81.17±1.07 | 91.97±0.18 | 86.47±0.76 | 88.27±0.36 | 89.21±0.04 | 83.07±0.21 | 90.17±0.32 | 62.60±7.03 | 71.43±1.64 | 79.64±7.53 | 61.25±3.00 | 63.46±0.00 | **77.46** |
| EuroBERT-210M | 80.83±0.35 | 91.94±0.31 | 59.94±0.16 | 86.36±0.67 | 45.16±16.60 | 72.75±1.30 | 90.64±0.92 | 80.74±2.99 | 89.29±0.23 | 90.75±0.09 | 85.63±0.28 | 91.49±0.27 | 54.95±2.56 | 71.68±2.35 | 86.79±2.40 | 64.64±2.01 | 63.27±0.43 | **76.87** |
| mGTE-MLM (305M) | 80.32±0.20 | 91.73±0.26 | 60.26±0.10 | 87.79±0.20 | 51.58±1.31 | 75.44±4.66 | 91.19±1.00 | 86.32±1.48 | 87.77±0.64 | 89.82±0.09 | 84.14±0.15 | 90.94±0.40 | 58.34±3.09 | 69.32±3.72 | 73.21±6.80 | 59.34±7.41 | 63.46±0.00 | **76.53** |
| LFM2.5-ColBERT-350M | 78.77±0.47 | 89.74±0.44 | 59.92±0.13 | 86.65±0.17 | 47.95±1.13 | 71.06±1.06 | 90.94±0.78 | 73.43±7.22 | 89.38±0.28 | 91.11±0.14 | 84.70±0.23 | 90.66±0.18 | 59.13±2.30 | 74.25±1.61 | 81.79±2.93 | 62.04±2.12 | 63.46±0.00 | **76.18** |
| EuroBERT-610M | 84.61±0.34 | 91.84±0.94 | 60.64±0.08 | 86.03±0.99 | 12.91±8.05 | 70.60±2.12 | 92.52±0.66 | 85.20±1.34 | 89.82±0.22 | 91.13±0.09 | 87.95±0.19 | 92.57±0.36 | 59.28±7.93 | 76.86±1.55 | 85.71±4.37 | 58.71±5.30 | 63.46±0.00 | **75.87** |
| LFM2.5-Embedding-350M | 78.59±0.11 | 89.13±0.63 | 60.47±0.13 | 87.03±0.21 | 50.19±0.90 | 72.54±0.68 | 91.70±0.69 | 77.45±1.31 | 89.38±0.09 | 91.14±0.13 | 84.70±0.12 | 90.62±0.50 | 55.38±1.74 | 70.17±1.99 | 71.43±3.57 | 63.10±1.28 | 63.46±0.00 | **75.68** |
| EuroBERT-2.1B | 70.52±14.22 | 92.34±0.19 | 60.45±0.70 | 85.40±1.36 | 6.84±6.81 | 68.99±0.67 | 92.50±1.03 | 82.94±3.10 | 66.44±32.36 | 91.03±0.29 | 81.56±16.62 | 93.56±0.25 | 53.29±0.79 | 77.23±6.77 | 82.86±5.14 | 57.90±4.75 | 63.46±0.00 | **72.19** |

`*` = dev split (GLUE/SuperGLUE test labels hidden). The 5 multilingual columns are labeled test.
SeaHorse & STS-B are Spearman×100. All other tasks are accuracy.
</details>

### Inference speed

The LFM2 backbone was built for fast inference, and the encoders inherit it. While ModernBERT-base is faster at short sequences in Apple GPU inputs, LFM2.5-Encoders overtake it as inputs grow. At long input sequences of 8k on CPU, the encoders run 3.3× faster than ModernBERT-base.

![inference_perf_cpu](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/0c0C31mmnpSqOf-z1sWxb.png)

![inference_perf_gpu](https://cdn-uploads.huggingface.co/production/uploads/644249b08443bce4c9890a0f/7wc8NKEuNN0IP0_lPhZ-m.png)

## 🔧 Fine-tuning

LFM2.5-Encoder-350M follows standard BERT-style fine-tuning. Attach a task head to the encoder body and train end-to-end. Suggested starting points (tune per task):

| Hyperparameter | Suggested range |
|---|---|
| Learning rate | 1e-5 – 5e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.1 |
| Epochs | 3 – 20 (early stopping, patience 3) |
| Precision | bf16 autocast (fp32 master weights) |

## 📬 Contact

- Got questions or want to connect? [Join our Discord community](https://discord.com/invite/liquid-ai)
- If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).

## Citation

```bibtex
@article{liquidAI2026Encoders,
  author = {Liquid AI},
  title = {LFM2.5-Encoders: Fast at Long Context, Even on CPU},
  journal = {Liquid AI Blog},
  year = {2026},
  note = {www.liquid.ai/blog/lfm2-5-encoders},
}
```
