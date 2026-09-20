#!/usr/bin/env python3
"""Build the train-only, deduplicated E6 MLM corpus.

The source readers deliberately open only the four train files selected for
E6.  They read text columns and retain source IDs, but never load labels,
validation rows, or test rows.  ``vsfc_topic`` is omitted because its train
texts are the same VSFC sentences already represented by
``vsfc_sentiment``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


SOURCE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "vsfc_sentiment",
        "path": "data/raw/uit-vsfc/train/sents.txt",
        "format": "lines",
        "text_column": None,
    },
    {
        "task_id": "visfd",
        "path": "data/raw/uit-visfd/Train.csv",
        "format": "csv",
        "text_column": "comment",
        "id_column": "index",
    },
    {
        "task_id": "absa_restaurant",
        "path": "data/raw/uit-absa-public-candidate/dataset/UIT-ABSA-Restaurant/Train.csv",
        "format": "csv",
        "text_column": "Review",
        "id_column": None,
    },
    {
        "task_id": "absa_hotel",
        "path": "data/raw/uit-absa-public-candidate/dataset/UIT-ABSA-Hotel/Train.csv",
        "format": "csv",
        "text_column": "Review",
        "id_column": None,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def read_source_rows(repo_root: Path, spec: dict[str, Any]) -> Iterable[dict[str, Any]]:
    path = repo_root / str(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if spec["format"] == "lines":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row_number, raw in enumerate(handle):
                yield {
                    "task_id": spec["task_id"],
                    "source_id": f"{spec['task_id']}:train:{row_number:06d}",
                    "source_index": str(row_number),
                    "text": raw.rstrip("\r\n"),
                    "path": str(path.relative_to(repo_root)),
                }
        return
    if spec["format"] != "csv":
        raise ValueError(f"Unsupported source format {spec['format']!r}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        text_column = str(spec["text_column"])
        if text_column not in reader.fieldnames:
            raise ValueError(f"CSV {path} has no text column {text_column!r}")
        id_column = spec.get("id_column")
        for row_number, row in enumerate(reader):
            text = row.get(text_column)
            if text is None:
                raise ValueError(f"Missing text at {path}:{row_number + 2}")
            source_index = str(row.get(id_column) if id_column else row_number)
            yield {
                "task_id": spec["task_id"],
                "source_id": f"{spec['task_id']}:train:{row_number:06d}",
                "source_index": source_index,
                "text": text,
                "path": str(path.relative_to(repo_root)),
            }


def non_special_count(
    input_ids: list[int], tokenizer: Any, *, max_length: int
) -> tuple[int, int]:
    special = set(int(value) for value in tokenizer.all_special_ids)
    content = sum(int(token not in special) for token in input_ids)
    return len(input_ids), content


def holdout_for_text(text: str, seed: int, fraction: float) -> bool:
    # The concatenation is intentionally literal and recorded in the manifest
    # so another runner cannot silently choose a different split function.
    digest = hashlib.sha256((str(seed) + text).encode("utf-8")).digest()
    threshold = int(fraction * (1 << 256))
    return int.from_bytes(digest, "big") < threshold


def build(repo_root: Path, output_dir: Path, *, seed: int, holdout_fraction: float) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"E6 corpus output directory is non-empty: {output_dir}; refusing overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = repo_root / "models" / "goiturn-s-gd"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)

    unique: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    source_counts: dict[str, int] = {}
    source_hashes: dict[str, str] = {}
    for spec in SOURCE_SPECS:
        path = repo_root / str(spec["path"])
        source_hashes[str(spec["path"])] = sha256_file(path)
        count = 0
        for row in read_source_rows(repo_root, spec):
            count += 1
            text = str(row["text"]).strip()
            if not text:
                continue
            uid = sha256_text(text)
            membership = {
                "task_id": row["task_id"],
                "source_id": row["source_id"],
                "source_index": row["source_index"],
                "source_file": row["path"],
            }
            if uid not in unique:
                unique[uid] = {
                    "uid": uid,
                    "text": text,
                    "memberships": [membership],
                }
            else:
                unique[uid]["memberships"].append(membership)
        source_counts[str(spec["task_id"])] = count

    records = list(unique.values())
    encoded = tokenizer(
        [record["text"] for record in records],
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    dropped_nonmaskable = 0
    kept: list[dict[str, Any]] = []
    for record, ids in zip(records, encoded):
        token_length, content_count = non_special_count(list(ids), tokenizer, max_length=256)
        if content_count <= 0:
            dropped_nonmaskable += 1
            continue
        record["token_length_256"] = token_length
        record["content_token_count_256"] = content_count
        record["split"] = (
            "mlm_holdout"
            if holdout_for_text(str(record["text"]), seed, holdout_fraction)
            else "mlm_train"
        )
        kept.append(record)

    train_records = [record for record in kept if record["split"] == "mlm_train"]
    holdout_records = [record for record in kept if record["split"] == "mlm_holdout"]
    if not train_records or not holdout_records:
        raise RuntimeError("Deterministic MLM split produced an empty train or holdout")

    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("x", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    tokenizer_hashes = {
        name: sha256_file(tokenizer_dir / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    }
    lengths = [int(record["token_length_256"]) for record in kept]
    manifest = {
        "protocol": "E6-train-only-MLM-corpus-v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_test": False,
        "input_scope": "train texts only; no validation or test rows used",
        "loader_compatibility": "equivalent to include_test=False, with direct train-file readers to avoid materialising dev rows",
        "source_tasks": [spec["task_id"] for spec in SOURCE_SPECS],
        "excluded_tasks": {
            "vsfc_topic": "same VSFC train sentence texts as vsfc_sentiment; excluded before reading to avoid duplicate text",
        },
        "source_split": "train",
        "source_row_counts": source_counts,
        "source_file_hashes": source_hashes,
        "unique_text_count_before_token_filter": len(records),
        "unique_text_count_after_token_filter": len(kept),
        "duplicate_text_count": len(records) - sum(1 for record in records if len(record["memberships"]) == 1),
        "duplicate_membership_count": sum(len(record["memberships"]) - 1 for record in records),
        "dropped_empty_text_count": 0,
        "dropped_nonmaskable_count": dropped_nonmaskable,
        "split_rule": {
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "hash": "sha256(str(seed)+text)",
            "threshold": "int(fraction * 2**256)",
            "holdout_name": "mlm_holdout; downstream train-only diagnostic, not official validation",
        },
        "mlm_train_count": len(train_records),
        "mlm_holdout_count": len(holdout_records),
        "membership_count": sum(len(record["memberships"]) for record in kept),
        "token_length_256": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.mean(lengths),
            "median": statistics.median(lengths),
        },
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_hashes": tokenizer_hashes,
        "max_length": 256,
        "corpus_jsonl": str(corpus_path),
        "corpus_jsonl_sha256": sha256_file(corpus_path),
    }
    write_json(output_dir / "corpus_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.repo_root / "artifacts" / "e6" / "mlm")
    build(
        args.repo_root.resolve(),
        output_dir.resolve(),
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
    )


if __name__ == "__main__":
    main()
