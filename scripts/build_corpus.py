#!/usr/bin/env python3
"""Assemble the distillation corpus from prepared source documents.

Inputs (see ``data/corpus_sources/README.md``):

* ``--hplt-docs``: JSONL with ``{"source": "web" | "ugc", "id": ..., "text": ...}``
  for the HPLT 3.0 documents listed in ``data/corpus_sources``, in HPLT shard
  order (web text as published; UGC text after paragraph extraction);
* ``--wikipedia``: the Vietnamese Wikipedia parquet (``id``, ``text``) and
  ``--wikipedia-ids``, the selected article IDs;
* the benchmark files under ``data/raw`` (training text is added as unlabeled
  text; validation and test text is used only for decontamination).

The output is ``mlm_mix.txt`` (one paragraph per line, a blank line after each
document) and ``stats.json``.  Token counts use the BamiBERT tokenizer.
"""

import argparse
import collections
import gzip
import json
from pathlib import Path

from transformers import AutoTokenizer

from goibert import corpus as C
from goibert.data import extra, uit
from goibert.training import sha256_file

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "paper.json").read_text())["corpus"]


def benchmark_text():
    """Training text (unlabeled) and validation/test keys for decontamination."""

    train, keys = {}, {}
    for name, task in (("vsfc", "vsfc_sentiment"), ("visfd", "visfd"), ("restaurant", "absa_restaurant"), ("hotel", "absa_hotel")):
        bundle = uit.load_task(task, include_test=True)
        train[name] = list(bundle.splits["train"].texts)
        for split in ("validation", "test"):
            keys[f"{name}/{split}"] = list(bundle.splits[split].texts)
    vispam = extra.load_task("vispam", include_test=True)
    train["vispam"] = list(vispam.train.texts)
    keys["vispam/validation"], keys["vispam/test"] = list(vispam.validation.texts), list(vispam.test.texts)
    vinli = extra.load_task("vinli", include_test=True)
    grouped = collections.OrderedDict()
    for premise, hypothesis in vinli.train.pairs:  # one document per premise with its hypotheses
        grouped.setdefault(premise, []).append(hypothesis)
    train["vinli"] = ["\n".join([p, *hs]) for p, hs in grouped.items()]
    keys["vinli/validation"] = [s for pair in vinli.validation.pairs for s in pair]
    keys["vinli/test"] = [s for pair in vinli.test.pairs for s in pair]
    phoner = extra.load_task("phoner", include_test=True)
    train["phoner"] = [" ".join(words) for words in phoner.train.words]
    keys["phoner/validation"] = [" ".join(words) for words in phoner.validation.words]
    keys["phoner/test"] = [" ".join(words) for words in phoner.test.words]
    return train, keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hplt-docs", type=Path, required=True)
    parser.add_argument("--wikipedia", type=Path, required=True)
    parser.add_argument("--wikipedia-ids", type=Path, default=Path("data/corpus_sources/wikipedia_ids.txt.gz"))
    parser.add_argument("--tokenizer", default="Qualcomm-AI-Research/BamiBERT")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.model_max_length = 10**9

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    train, keys = benchmark_text()
    clean = C.Cleaner(C.Decontaminator(keys), count)

    domain = [("domain", pars, n) for texts in train.values() for text in texts for pars, n in clean(text, "domain")]
    domain = C.repeat_domain(domain, CONFIG["benchmark_text_tokens"])

    with gzip.open(args.wikipedia_ids, "rt", encoding="utf-8") as handle:
        wiki_ids = {line.strip() for line in handle if line.strip()}
    wiki = []
    for batch in pq.ParquetFile(args.wikipedia).iter_batches(batch_size=2048, columns=["id", "text"]):
        rows = batch.to_pydict()
        for doc_id, text in zip(rows["id"], rows["text"]):
            if str(doc_id) in wiki_ids:
                wiki.extend(("wiki", pars, n) for pars, n in clean(text, "wiki"))

    web, ugc = [], []
    with args.hplt_docs.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            target = ugc if row["source"] == "ugc" else web
            target.extend((row["source"], pars, n) for pars, n in clean(row["text"], row["source"]))

    docs = C.mix(domain, wiki, ugc, web, CONFIG["total_tokens"])
    args.out.mkdir(parents=True, exist_ok=True)
    C.write_corpus(docs, args.out / "mlm_mix.txt")
    by_source = {s: {"documents": sum(d[0] == s for d in docs), "tokens": sum(d[2] for d in docs if d[0] == s)} for s in ("web", "wiki", "ugc", "domain")}
    stats = {"total_tokens": sum(d[2] for d in docs), "sources": by_source, "filter_counts": {k: dict(v) for k, v in clean.counts.items()},
             "mlm_mix_sha256": sha256_file(args.out / "mlm_mix.txt")}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["sources"]))


if __name__ == "__main__":
    main()
