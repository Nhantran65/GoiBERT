# Distillation corpus sources

The corpus text itself is not redistributed. These files identify the source
documents so the corpus can be assembled with `scripts/build_corpus.py`.

| File | Rows | Content |
|---|---:|---|
| `hplt_web_candidates.tsv.gz` | 553,528 | HPLT 3.0 `vie_Latn` document IDs and shards: the seeded random sample (540M tokens) drawn from the 1.16M web documents that passed the quality filters. `scripts/build_corpus.py` draws the web share from these candidates. |
| `hplt_ugc_ids.tsv.gz` | 78,886 | HPLT 3.0 document IDs and shards of the forum and review pages used as user-generated text. |
| `wikipedia_ids.txt.gz` | 29,781 | Vietnamese Wikipedia article IDs (`wikipedia_vi`, 2023-11-01 dump). |

Upstream filtering is described in
[release/goi-corpus-v2/README.md](../../release/goi-corpus-v2/README.md) and is
not included as code:

- HPLT quality bins 7–9;
- the spun and machine-translated text classifier;
- the 2% per-host cap;
- near-deduplication with 8-word shingles;
- paragraph extraction from forum and review pages.

`build_corpus.py` applies the final stage as code:

- decontamination against validation and test text;
- exact paragraph deduplication;
- the 2,048-token document cut;
- mixing to the target token counts;
- the document-level shuffle.

The corpus used for the paper has SHA-256
`3ba6537ced296669373f6d3e652a8d5a75e152e97865ff7333f75c43896295c1`. Paragraph
extraction for user-generated pages and document order within each HPLT shard
both affect deduplication, so a rebuild matches the published statistics but
is not guaranteed to be byte-identical.
