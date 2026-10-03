---
language:
- vi
---

# Goi Corpus v2: a clean Vietnamese corpus for encoder distillation

Goi Corpus v2 is a **328.6M-token** Vietnamese text corpus (counted with the BamiBERT tokenizer). It was built to distill small Vietnamese encoders, and it is the corpus used to train [GoiBERT-S](https://huggingface.co/2HeartsVN/GoiBERT-S) and [GoiBERT-M](https://huggingface.co/2HeartsVN/GoiBERT-M).

Key properties:

- **Deduplicated:** 0% duplicate web paragraphs.
- **Document-coherent:** paragraphs of a document stay in order; only whole documents are shuffled.
- **More diverse:** web, Wikipedia and user-generated text.
- **Decontaminated** against the validation and test sets of 8 Vietnamese benchmarks.

## Contents

| Source | Documents | Tokens | Share |
|---|---:|---:|---:|
| HPLT 3.0 Vietnamese web | 284,991 | 232.9M | 71% |
| Vietnamese Wikipedia | 41,901 | 49.4M | 15% |
| User-generated and forum text (from HPLT) | 70,952 | 25.5M | 8% |
| Unlabeled train-split text of 8 benchmarks (repeated) | 511,885 | 20.9M | 6% |
| **Total** | **909,729** | **328.6M** | 100% |

- **Size:** 1.70 GB of UTF-8 text.
- **File:** `mlm_mix.txt`, sha256 `3ba6537ced296669373f6d3e652a8d5a75e152e97865ff7333f75c43896295c1`.

## Format

- Plain text.
- **One paragraph per line.**
- **A blank line separates documents.**
- Documents are in random order (seed 42), and paragraphs within a document keep their original order.

This layout works with sequence packing: read the file in order and concatenate lines with `</s>`, and each packed sequence contains consecutive paragraphs of the same document.

```python
from datasets import load_dataset

ds = load_dataset("2HeartsVN/goi-corpus-v2", data_files="mlm_mix.txt", split="train")
# one row per line; empty rows mark document boundaries
```

## How it was built

**1. Web (HPLT 3.0, `vie_Latn`)**

- Kept documents from quality bins 7–9.
- Removed spun and machine-translated text with a dedicated classifier.
- Capped each host at 2% of tokens.
- Removed near-duplicates (8-word shingles).
- Sampled 284,991 documents at random from a pool of 1.16M documents that passed these filters.

**2. Wikipedia:** Vietnamese articles with at least 300 words; stub-like pages removed.

**3. User-generated text:** paragraphs from forum and review platforms found in HPLT, extracted at paragraph level.

**4. Benchmark train text**

- Unlabeled training-split text of UIT-VSFC, UIT-ViSFD, UIT-ABSA (Restaurant, Hotel), ViSpamReviews, ViNLI and PhoNER_COVID19. Validation and test splits were never used.
- Repeated 9–10× so that it makes up about 6% of all tokens.
- Each ViNLI premise is kept together with its hypotheses in one document.

**5. Cleaning (all sources)**

- Exact paragraph deduplication (NFC, casefold, whitespace-normalized).
- Removed boilerplate paragraphs that occur ≥ 5 times in the web part.
- Documents longer than 2,048 tokens were cut at a paragraph boundary.

**6. Decontamination**

- **Keys:** the validation and test text of all 8 benchmarks.
- **Rules:**
  - For key texts of at least 13 syllables: drop any paragraph that shares a 13-gram with the key (after lowercasing and removing punctuation).
  - For shorter key texts: drop paragraphs that match exactly.
  - Drop the whole document if more than 20% of its paragraphs are affected.
- **Removed:** 30,334 paragraphs and 2,630 documents.
- **Final check:** a rescan of the released file found **0** exact matches with any validation or test example.

## Quality checks

| Check | Result |
|---|---|
| Vietnamese character ratio (web sample, 20K documents) | 99.5% |
| Duplicate paragraphs in the web part (after cleaning) | 0 / 4,302,033 |
| Exact matches with benchmark validation or test text | 0 |
| Packed 256-token sequences, first 100 | 100% full length and readable Vietnamese |
