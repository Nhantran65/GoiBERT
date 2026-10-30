# Data

The benchmark datasets are not redistributed in this repository. Obtain each
dataset from its authors under its own terms, then place the files as below
(or point the `GOIBERT_DATA` environment variable at another directory with
the same layout). `checksums.json` lists the SHA-256 of every file we used.

```
data/raw/
├── uit-vsfc/{train,dev,test}/{sents,sentiments,topics}.txt
├── uit-visfd/{Train,Dev,Test}.csv
├── uit-absa/UIT-ABSA-Restaurant/{Train,Dev,Test}.csv
├── uit-absa/UIT-ABSA-Hotel/{Train,Dev,Test}.csv
├── vispamreviews/{train,dev,test}.csv
├── vinli/{train,dev,test}.parquet
└── phoner_covid19/{train,dev,test}_syllable.json
```

| Dataset | Source | Terms |
|---|---|---|
| UIT-VSFC | [UIT NLP group](https://nlp.uit.edu.vn/datasets/) | See source |
| UIT-ViSFD | [LuongPhan/UIT-ViSFD](https://github.com/LuongPhan/UIT-ViSFD) | Research use; contact the authors for commercial use |
| UIT-ABSA (Restaurant, Hotel) | [UIT NLP group](https://nlp.uit.edu.vn/datasets/) | User agreement with the authors |
| ViSpamReviews | [sonlam1102/vispamdetection](https://github.com/sonlam1102/vispamdetection) (v1, `SpamLabel` column) | Research use |
| ViNLI | Hugging Face mirror `presencesw/vinli_4_label` | See source |
| PhoNER_COVID19 | [VinAIResearch/PhoNER_COVID19](https://github.com/VinAIResearch/PhoNER_COVID19) (syllable level) | Research use; no redistribution |

The splits are the official ones. Validation and test text is used only for
evaluation and for decontaminating the distillation corpus.

`corpus_sources/` lists the documents selected for the distillation corpus;
see [corpus_sources/README.md](corpus_sources/README.md).
