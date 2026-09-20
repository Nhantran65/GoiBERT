---
language:
- vi
metrics:
- f1
pipeline_tag: token-classification
tags:
- transformer
- vietnamese
- nlp
- bert
- deberta
- deberta-v3
---

# ViDeBERTa: A powerful pre-trained language model for Vietnamese


ViDeBERTa, a new pre-trained monolingual language model for Vietnamese, 
with three versions - ViDeBERTa_xsmall, ViDeBERTa_base, and ViDeBERTa_large, 
which are pre-trained on 138GB of Vietnamese text of high-quality and diverse Vietnamese text using DeBERTaV3 architecture. 

Please check the [official repository][github] for more implementation details and updates

The DeBERTa V3 xsmall model comes with 12 layers and a hidden size 
of 384. It has only 22M backbone parameters with a vocabulary 
containing 128K tokens which introduces 48M parameters in the 
Embedding layer. This model was trained using CC100 dataset, which consists of 138 GB of Vietnamese text.

## Fine-tuning on NLU tasks
We present the dev results on VLSP POS, PhoNER, ViQuAD dataset.

| Model|#Params(M)| POS | NER | MRC |
|-----------|-------|---------|-----|----------|
| XLM-R-base      | 125M  | 96.2 | - | 82.0 |
| XLM-R-large  | 355M | 96.3 | 93.8 | 87.0 |
| PhoBERT-base | 135M | 96.7 | 80.1 |
| PhoBERT-large | 370M | 96.8 | 83.5 |
| ViT5-base | 310M | - | 94.5 | - |
| ViT5-large | 866M | - | 93.8 | - |
| **ViDeBERTa-xsmall** | **22M** | **96.4** | **93.6** | **81.3** |
| ViDeBERTa-base | 86M | 96.8 | 94.5 | 85.7 |
| ViDeBERTa-large | 304M | 97.2 | 95.3 | 89.9 |

## Citation
If you find ViDeBERTa useful for your work, please cite the following papers:
```latex
@article{dao2023videberta,
  title={ViDeBERTa: A powerful pre-trained language model for Vietnamese},
  author={Dao Tran, Cong and Pham, Nhut Huy and Nguyen, Anh and Son Hy, Truong and Vu, Tu},
  journal={arXiv e-prints},
  pages={arXiv--2301},
  year={2023}
}
```

[github]: https://github.com/HySonLab/ViDeBERTa