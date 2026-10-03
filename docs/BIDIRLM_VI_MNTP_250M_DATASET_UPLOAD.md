# BidirLM-0.6B Vietnamese MNTP dataset upload

Date: 2026-09-24. Uploaded to private Hugging Face dataset repository [`2HeartsVN/goibert-bidirlm-vi-mntp-250m`](https://huggingface.co/datasets/2HeartsVN/goibert-bidirlm-vi-mntp-250m) at revision `6c07c9d1945cd94d8ffbe683d050692d330c0b73`.

| Split | Sequences | 512-token IDs | SHA256 |
| --- | ---: | ---: | --- |
| train | 488,256 | 249,987,072 | `401dfdc63e6f74f0ad3f8457d23aee4d197e8ed9a6f2e3ef4726f79c3007e460` |
| validation (held-out) | 2,000 | 1,024,000 | `f8a961a139e48db56afa838cdccaf4d097f271ae02510363fce89db47365b125` |

The train split contains the exact `order[:7629*64]` source sequences consumed by the optimizer, not the 488,281 sequences reserved as the nominal 250M-token region. The 25 reserved but unused sequences were excluded. The held-out split is `order[488281:490281]`, disjoint from train. No original web text or benchmark examples were added.

Prepared with `scripts/bidirlm_vi_mntp/export_hf_dataset.py`; local stage `releases/bidirlm_vi_mntp_250m_dataset/` has README, manifest, and both Parquet files. Before export, the script checked SHA256 of `tokens.npy`, `order.npy`, and `heldout_indices.npy` against the training data manifest. After writing, it compared every Parquet row with the original token matrix and index. After upload, root confirmed `private=True` via Hugging Face API, listed the files, downloaded both Parquet files from the pinned remote revision, and matched their SHA256 values to the local manifest.

The dataset card credits HPLT 3.0 and states that HPLT's CC0 license covers packaging while rights to underlying web-page text are not owned by HPLT. The repository metadata uses `license: other` to avoid describing the text itself as CC0. The repository is private as requested; access requires permission from `2HeartsVN`.
