"""Final assembly of the distillation corpus.

Input documents come from four sources: HPLT 3.0 web text, Vietnamese
Wikipedia, user-generated text found in HPLT, and the unlabeled training text
of the eight benchmarks.  This module applies the last stage of the pipeline
described in ``release/goi-corpus-v2/README.md``:

1. **Decontamination.**  A paragraph is dropped if it shares any 13-syllable
   n-gram with a validation or test example, or (for examples shorter than 13
   syllables) matches one exactly, after NFC normalisation, lower-casing and
   punctuation removal.  A document is dropped when more than 20% of its
   paragraphs are hit.
2. **Exact deduplication** of paragraphs across all sources (NFC, casefold,
   whitespace collapsed).
3. **Length cut.**  Documents are split at paragraph boundaries into chunks of
   at most 2,048 BamiBERT tokens.
4. **Mixing.**  Benchmark text is repeated (at most 20 times) to reach its
   target, web documents are sampled to fill the remaining token budget, and
   the documents are shuffled as whole units (seed 42).

Upstream selection of the web and user-generated documents (quality bins,
spun/machine-translated text classifier, 2% host cap, near-deduplication) is
described in the corpus card; ``data/corpus_sources`` lists the selected IDs.
"""

from __future__ import annotations

import collections
import random
import re
import unicodedata
from typing import Iterable

PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
SPACE = re.compile(r"\s+")
NGRAM = 13
MAX_DOC_TOKENS = 2048
DOC_DROP_THRESHOLD = 0.2
MAX_DOMAIN_REPEAT = 20
SEED = 42


def normalize(text: str) -> str:
    return SPACE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def syllables(text: str) -> list[str]:
    return PUNCT.sub(" ", unicodedata.normalize("NFC", text).lower()).split()


def split_paragraphs(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class Decontaminator:
    """Flags paragraphs that overlap validation or test text."""

    def __init__(self, keys: dict[str, Iterable[str]]):
        self.names = list(keys)
        self.long: dict[int, int] = {}
        self.short: dict[str, int] = {}
        for bit, name in enumerate(self.names):
            mask = 1 << bit
            for text in {normalize(x) for x in keys[name] if normalize(x)}:
                tokens = syllables(text)
                if len(tokens) < NGRAM:
                    key = " ".join(tokens)
                    self.short[key] = self.short.get(key, 0) | mask
                else:
                    for i in range(len(tokens) - NGRAM + 1):
                        h = hash(tuple(tokens[i : i + NGRAM]))
                        self.long[h] = self.long.get(h, 0) | mask

    def hit(self, text: str) -> int:
        tokens = syllables(text)
        mask = self.short.get(" ".join(tokens), 0)
        for i in range(len(tokens) - NGRAM + 1):
            mask |= self.long.get(hash(tuple(tokens[i : i + NGRAM])), 0)
        return mask


class Cleaner:
    """Decontaminate, deduplicate and cut documents, keeping counts per source."""

    def __init__(self, decontaminator: Decontaminator, count_tokens):
        self.decontaminator = decontaminator
        self.count_tokens = count_tokens
        self.seen: set[str] = set()
        self.counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    def __call__(self, text: str, source: str) -> list[tuple[list[str], int]]:
        counts = self.counts[source]
        paragraphs = split_paragraphs(text)
        counts["docs_input"] += 1
        counts["paragraphs_input"] += len(paragraphs)
        if not paragraphs:
            return []
        hits = [self.decontaminator.hit(p) for p in paragraphs]
        if sum(bool(h) for h in hits) / len(paragraphs) > DOC_DROP_THRESHOLD:
            counts["docs_decontam_dropped"] += 1
            return []
        kept = []
        for paragraph, hit in zip(paragraphs, hits):
            if hit:
                counts["paragraphs_decontam_dropped"] += 1
                continue
            key = normalize(paragraph)
            if key in self.seen:
                counts["paragraphs_exact_dedup_dropped"] += 1
                continue
            self.seen.add(key)
            kept.append(paragraph)
        chunks, current, size = [], [], 0
        for paragraph in kept:
            n = self.count_tokens(paragraph)
            if n > MAX_DOC_TOKENS:
                counts["paragraphs_over_2048_dropped"] += 1
                continue
            if current and size + n > MAX_DOC_TOKENS:
                chunks.append((current, size))
                current, size = [], 0
            current.append(paragraph)
            size += n
        if current:
            chunks.append((current, size))
        counts["docs_after_filter"] += len(chunks)
        return chunks


def repeat_domain(domain_docs: list[tuple[str, list[str], int]], target_tokens: int) -> list[tuple[str, list[str], int]]:
    """Repeat benchmark training text up to ``target_tokens`` (at most 20 times)."""

    base = sum(n for _, _, n in domain_docs)
    repeat = min(MAX_DOMAIN_REPEAT, max(1, target_tokens // max(1, base)))
    out = [("domain", pars, n) for _ in range(repeat) for _, pars, n in domain_docs]
    total = base * repeat
    if repeat < MAX_DOMAIN_REPEAT:
        partial = list(domain_docs)
        random.Random(SEED).shuffle(partial)
        for _, pars, n in partial:
            if total + n > target_tokens:
                continue
            out.append(("domain", pars, n))
            total += n
    return out


def mix(domain, wiki, ugc, web, total_tokens: int) -> list[tuple[str, list[str], int]]:
    """Fill the remaining budget with shuffled web documents, then shuffle everything."""

    remaining = total_tokens - sum(d[2] for d in domain) - sum(d[2] for d in wiki) - sum(d[2] for d in ugc)
    web = list(web)
    random.Random(SEED).shuffle(web)
    selected, tokens = [], 0
    for doc in web:
        selected.append(doc)
        tokens += doc[2]
        if tokens >= remaining:
            break
    if tokens < remaining:
        raise RuntimeError(f"Not enough web text: short by {remaining - tokens} tokens")
    docs = list(domain) + list(wiki) + list(ugc) + selected
    random.Random(SEED).shuffle(docs)
    return docs


def write_corpus(docs, path) -> None:
    """One paragraph per line, a blank line after each document."""

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for _, paragraphs, _ in docs:
            handle.write("\n".join(paragraphs) + "\n\n")
