"""Shared text sources + tokenizer for the Vietnamese vocabulary-trim step (docs/VOCAB_TRIM_STUDENT_PLAN.md §1.1).

Counting sources: general pool v3 (sampled, seed 42), Wikipedia v2 filtered.  Mandatory-token sources:
train+dev of the five benchmarks and of the social datasets (train+dev only — test text is never used to
choose the keep set, only to measure `<unk>` rates).
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts" / "mmbert_vi_cpt_pilot", ROOT / "scripts" / "mmbert_vi_cpt_pilot" / "v2",
           ROOT / "scripts" / "mmbert_vi_cpt_pilot" / "v3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

XLM_R_LARGE = ("FacebookAI/xlm-roberta-large", "c23d21b0620b635a76227c604d44e43a9f0ee389")
XLM_R_BASE = ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089")
MINILM12 = ("nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large", "d828558d1a570cbbb5e62a8dbf85c8f18bf7982a")
MINILM6 = ("nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large", "160deb78aca30f63754e512a93337ce8013a32ca")
BENCHMARKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
SOCIAL = {"ViCTSD": ("Comment", "Title"), "vihsd": ("free_text",), "uit-vsmec": ("Sentence",), "vispamdetection": ("Comment",)}
ART = ROOT / "artifacts" / "vocab_trim_student"
DATA = ROOT / "data" / "vocab_trim_student"
POOL_CACHE = Path("/tmp/vocab_trim_general_v3_docs.json")
VOCAB_SIZE = 250002
SPECIAL_IDS = (0, 1, 2, 3)          # <s>, <pad>, </s>, <unk>
MASK_ID = 250001


def snapshot(repo: str, commit: str) -> Path:
    home = Path(__import__("os").environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def xlmr_tokenizer(repo: str = XLM_R_LARGE[0], commit: str = XLM_R_LARGE[1]):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot(repo, commit)), local_files_only=True, use_fast=True)


def pool_docs() -> list[dict]:
    """Metadata of the general pool v3 documents (read-only; cached outside the repo)."""
    if POOL_CACHE.exists():
        return json.loads(POOL_CACHE.read_text())
    import audit_v3 as A3
    import common_v3 as C3

    bucket = [json.loads(line) for line in (C3.DATA_V3 / "informal_bucket" / "informal_docs.jsonl").open(encoding="utf-8")]
    docs = A3.build_general_v3({row["pool_index"] for row in bucket})["docs"]
    POOL_CACHE.write_text(json.dumps(docs))
    return docs


def part_texts(part: str, wanted_kh64: set[int] | None = None) -> dict[int, str]:
    import common_v2 as C2

    return C2.part_texts(part, wanted_kh64)


def wiki_docs(limit: int = 0) -> Iterator[str]:
    """Texts of the Wikipedia v2 filtered documents, read from the pinned parquet (CC BY-SA 4.0, text not redistributed)."""
    import pandas as pd

    filtered = [json.loads(line) for line in
                (ROOT / "artifacts" / "mmbert_vi_cpt_pilot" / "v2" / "wiki_filtered.jsonl").open(encoding="utf-8")]
    frame = pd.read_parquet(ROOT / "data" / "raw" / "wikipedia_vi" / "wiki_vi_20231101_train_0000.parquet")
    by_id = {str(row["id"]): row for _, row in frame.iterrows()}
    for i, row in enumerate(filtered):
        if limit and i >= limit:
            break
        record = by_id.get(str(row["id"]))
        if record is None:
            continue
        title = record.get("title")
        text = record.get("text") or ""
        yield (f"{title}\n\n{text}" if title else text)


def benchmark_texts(tasks: Iterable[str] = BENCHMARKS, splits: Iterable[str] = ("train", "validation")) -> Iterator[str]:
    from goibert import train as e1_train

    for task in tasks:
        bundle = e1_train.load_task_bundle(task, ROOT, include_test=False)
        for split in splits:
            if split == "test":
                continue
            for text in e1_train.split_texts(e1_train.split_field(bundle, split)):
                yield str(text)


def benchmark_test_texts(tasks: Iterable[str] = BENCHMARKS) -> Iterator[str]:
    """Test text only (no labels) — used exclusively to measure `<unk>` rates (plan §1.2)."""
    from goibert import train as e1_train

    for task in tasks:
        try:
            bundle = e1_train.load_task_bundle(task, ROOT, include_test=True)
        except Exception:  # noqa: BLE001 - some bundled splits may not ship a public test text
            continue
        for text in e1_train.split_texts(e1_train.split_field(bundle, "test")):
            yield str(text)


def social_texts(splits: Iterable[str] = ("train", "dev", "valid")) -> Iterator[str]:
    for name, columns in SOCIAL.items():
        directory = ROOT / "data" / "raw" / "social" / name
        for path in sorted(directory.glob("*.csv")):
            if not any(key in path.stem.lower() for key in splits):
                continue
            with path.open(encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    for column in columns:
                        value = (row.get(column) or "").strip()
                        if value:
                            yield value


def social_test_texts() -> Iterator[str]:
    for name, columns in SOCIAL.items():
        directory = ROOT / "data" / "raw" / "social" / name
        for path in sorted(directory.glob("*.csv")):
            if "test" not in path.stem.lower():
                continue
            with path.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    for column in columns:
                        value = (row.get(column) or "").strip()
                        if value:
                            yield value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
