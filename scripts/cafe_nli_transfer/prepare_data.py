"""Prepare the ViANLI Phase-D data gate for the CafeBERT NLI pilot.

This module is deliberately CPU-only and reads only ViANLI train/dev plus the
five benchmark train/dev splits.  It never asks the benchmark loader for test
data and it does not infer or edit any labels.

The first operation in ``main`` is the fixed 300-row audit sample.  This lets
the scientific reviewer begin the blind semantic audit before the more
expensive near-duplicate and decontamination pass completes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "cafe_nli_transfer"
RAW_ROOT = DATA_ROOT / "raw"
ARTIFACT_ROOT = ROOT / "artifacts" / "cafe_nli_transfer"
RAW_TRAIN = RAW_ROOT / "vianli_train.jsonl"
RAW_DEV = RAW_ROOT / "vianli_dev.jsonl"
RAW_README = RAW_ROOT / "README.md"
RAW_API = RAW_ROOT / "dataset_api.json"

EXPECTED_KEYS = ("uid", "premise", "hypothesis", "label")
LABEL_NAMES = ("entailment", "neutral", "contradiction")
LABEL_TO_ID = {name: index for index, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {index: name for name, index in LABEL_TO_ID.items()}
SAMPLE_PER_CLASS = 100
SAMPLE_SEED = 42
GROUP_SIMILARITY = 0.8
BENCHMARK_TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def normalize_text(text: str) -> str:
    """The exact normalization specified by the pilot plan."""

    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def words(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    return tuple(re.findall(r"\S+", normalized))


def shingles(text: str, width: int = 5) -> frozenset[tuple[str, ...]]:
    tokens = words(text)
    if len(tokens) < width:
        return frozenset()
    return frozenset(tokens[index : index + width] for index in range(len(tokens) - width + 1))


def jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line in {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON line in {path}:{line_number}")
            rows.append(value)
    return rows


def validate_source_rows(rows: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    key_counts = Counter(tuple(sorted(row.keys())) for row in rows)
    nulls = Counter()
    blank_texts = Counter()
    type_errors: list[dict[str, Any]] = []
    bad_labels: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        for key in EXPECTED_KEYS:
            value = row.get(key)
            if value is None:
                nulls[key] += 1
            if key in ("premise", "hypothesis") and isinstance(value, str) and not value.strip():
                blank_texts[key] += 1
            if key in ("uid", "premise", "hypothesis", "label") and not isinstance(value, str):
                type_errors.append({"row": index, "field": key, "type": type(value).__name__})
        if row.get("label") not in LABEL_TO_ID:
            bad_labels.append({"row": index, "label": row.get("label")})
    uid_counts = Counter(row.get("uid") for row in rows)
    pair_counts = Counter((row.get("premise"), row.get("hypothesis")) for row in rows)
    normalized_pair_counts = Counter(
        (normalize_text(str(row.get("premise", ""))), normalize_text(str(row.get("hypothesis", ""))))
        for row in rows
    )
    label_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    label_by_normalized_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        pair = (row.get("premise"), row.get("hypothesis"))
        normalized_pair = (normalize_text(str(pair[0] or "")), normalize_text(str(pair[1] or "")))
        label_by_pair[pair].add(str(row.get("label")))
        label_by_normalized_pair[normalized_pair].add(str(row.get("label")))
    return {
        "split": split,
        "rows": len(rows),
        "key_variants": {"|".join(keys): count for keys, count in key_counts.items()},
        "expected_keys": list(EXPECTED_KEYS),
        "schema_exact": len(key_counts) == 1 and key_counts.get(tuple(sorted(EXPECTED_KEYS))) == len(rows),
        "null_counts": dict(nulls),
        "blank_text_counts": dict(blank_texts),
        "type_error_count": len(type_errors),
        "type_errors_head": type_errors[:20],
        "invalid_label_count": len(bad_labels),
        "invalid_labels_head": bad_labels[:20],
        "uid_unique": len(uid_counts) == len(rows),
        "duplicate_uid_count": sum(count - 1 for count in uid_counts.values() if count > 1),
        "duplicate_uid_values_head": [uid for uid, count in uid_counts.items() if count > 1][:20],
        "exact_duplicate_pair_count": sum(count - 1 for count in pair_counts.values() if count > 1),
        "normalized_duplicate_pair_count": sum(count - 1 for count in normalized_pair_counts.values() if count > 1),
        "exact_conflicting_gold_pair_count": sum(
            1 for labels in label_by_pair.values() if len(labels) > 1
        ),
        "normalized_conflicting_gold_pair_count": sum(
            1 for labels in label_by_normalized_pair.values() if len(labels) > 1
        ),
        "label_counts": dict(Counter(str(row.get("label")) for row in rows)),
    }


def build_sample(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[str(row["label"])].append((index, row))
    generator = random.Random(SAMPLE_SEED)
    selected: list[dict[str, Any]] = []
    for label in LABEL_NAMES:
        candidates = by_label[label]
        if len(candidates) < SAMPLE_PER_CLASS:
            raise RuntimeError(f"class {label} has only {len(candidates)} rows")
        for source_index, row in generator.sample(candidates, SAMPLE_PER_CLASS):
            selected.append(
                {
                    "sample_index": len(selected),
                    "source_split": "train",
                    "source_row": source_index,
                    "uid": row["uid"],
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": row["label"],
                    "label_id": LABEL_TO_ID[row["label"]],
                }
            )
    if len(selected) != 3 * SAMPLE_PER_CLASS:
        raise AssertionError(len(selected))
    return selected


def write_sample(sample: Sequence[dict[str, Any]]) -> None:
    write_jsonl(ARTIFACT_ROOT / "audit_sample.jsonl", sample)
    path = ARTIFACT_ROOT / "audit_sample.txt"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("sample_index\tlabel\tpremise\thypothesis\n")
        for row in sample:
            premise = str(row["premise"]).replace("\t", " ").replace("\r", " ").replace("\n", " ")
            hypothesis = str(row["hypothesis"]).replace("\t", " ").replace("\r", " ").replace("\n", " ")
            handle.write(f"{row['sample_index']}\t{row['label']}\t{premise}\t{hypothesis}\n")
    temporary.replace(path)
    write_json(
        ARTIFACT_ROOT / "audit_sample_manifest.json",
        {
            "protocol": "cafebert-nli-transfer-pilot-v1",
            "seed": SAMPLE_SEED,
            "sampling": "random.Random(42), exactly 100 rows per official ViANLI train class",
            "rows": len(sample),
            "class_counts": dict(Counter(row["label"] for row in sample)),
            "prediction_blind": True,
            "review_fields": ["sample_index", "uid", "premise", "hypothesis", "label"],
            "note": "Root owns semantic review; these notes must not be converted into training labels.",
        },
    )


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def normalized_exact_groups(values: Sequence[str]) -> tuple[dict[str, list[int]], list[str]]:
    normalized = [normalize_text(value) for value in values]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(normalized):
        groups[value].append(index)
    return groups, normalized


def group_train_components(rows: Sequence[dict[str, Any]]) -> tuple[list[int], dict[str, Any]]:
    premises = [str(row["premise"]) for row in rows]
    hypotheses = [str(row["hypothesis"]) for row in rows]
    normalized_premises = [normalize_text(value) for value in premises]
    normalized_hypotheses = [normalize_text(value) for value in hypotheses]
    premise_shingles = [shingles(value) for value in premises]
    dsu = DisjointSet(len(rows))
    premise_exact, _ = normalized_exact_groups(premises)
    exact_premise_unions = 0
    for indices in premise_exact.values():
        first = indices[0]
        for index in indices[1:]:
            dsu.union(first, index)
            exact_premise_unions += 1

    inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, value in enumerate(premise_shingles):
        for shingle in value:
            inverted[shingle].append(index)
    near_pairs = 0
    for index, value in enumerate(premise_shingles):
        if len(words(premises[index])) < 5:
            continue
        candidates: set[int] = set()
        for shingle in value:
            candidates.update(inverted[shingle])
        for candidate in sorted(candidates):
            if candidate <= index:
                continue
            candidate_words = words(premises[candidate])
            if len(candidate_words) < 5:
                continue
            if jaccard(value, premise_shingles[candidate]) >= GROUP_SIMILARITY:
                dsu.union(index, candidate)
                near_pairs += 1

    hypothesis_exact, _ = normalized_exact_groups(hypotheses)
    exact_hypothesis_unions = 0
    for indices in hypothesis_exact.values():
        first = indices[0]
        for index in indices[1:]:
            dsu.union(first, index)
            exact_hypothesis_unions += 1

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[dsu.find(index)].append(index)
    component_ids: dict[int, str] = {}
    component_sizes: Counter[str] = Counter()
    for root, indices in components.items():
        component_id = min(normalized_premises[index] for index in indices)
        component_ids[root] = component_id
        component_sizes[component_id] = len(indices)
    assigned = [component_ids[dsu.find(index)] for index in range(len(rows))]
    report = {
        "method": {
            "text_normalization": "NFC, casefold, whitespace collapse",
            "premise_connection": "exact normalized premise OR word-5-shingle Jaccard >= 0.8",
            "short_premise_rule": "premise with fewer than 5 words uses exact normalized premise only",
            "additional_connection": "exact normalized hypothesis",
            "component_id": "lexicographically smallest normalized premise in component",
        },
        "rows": len(rows),
        "components": len(component_sizes),
        "component_size_counts": dict(sorted(Counter(component_sizes.values()).items())),
        "largest_component_sizes": sorted(component_sizes.values(), reverse=True)[:20],
        "exact_premise_union_operations": exact_premise_unions,
        "near_duplicate_premise_union_operations": near_pairs,
        "exact_hypothesis_union_operations": exact_hypothesis_unions,
        "premise_exact_group_count": len(premise_exact),
        "hypothesis_exact_group_count": len(hypothesis_exact),
        "premise_unique_normalized": len(set(normalized_premises)),
        "hypothesis_unique_normalized": len(set(normalized_hypotheses)),
        "component_has_multiple_labels": sum(
            len({rows[index]["label"] for index in indices}) > 1 for indices in components.values()
        ),
    }
    return assigned, report


def load_benchmark_texts() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load only benchmark train/dev text through the audited no-test loader."""

    sys.path.insert(0, str(ROOT))
    from src.goibert.data import load_task  # pylint: disable=import-outside-toplevel

    records: list[dict[str, str]] = []
    task_counts: dict[str, dict[str, int]] = {}
    source_files: dict[str, list[str]] = {}
    for task in BENCHMARK_TASKS:
        dataset = load_task(task, ROOT, include_test=False)
        task_counts[task] = {split: len(dataset.splits[split]) for split in ("train", "validation")}
        source_files[task] = list(dataset.source_files)
        for split_name in ("train", "validation"):
            for example in dataset.splits[split_name]:
                records.append(
                    {
                        "task": task,
                        "split": split_name,
                        "id": example.id,
                        "text": example.text,
                    }
                )
    return records, {"task_counts": task_counts, "source_files": source_files, "rows": len(records)}


@dataclass
class DecontamResult:
    component_contaminated: set[str]
    row_matches: list[dict[str, Any]]
    report: dict[str, Any]


def decontaminate(
    rows: Sequence[dict[str, Any]],
    component_ids: Sequence[str],
    benchmark: Sequence[dict[str, str]],
) -> DecontamResult:
    """Find benchmark train/dev text matches and mark whole components."""

    benchmark_norm = [normalize_text(record["text"]) for record in benchmark]
    benchmark_sets = [shingles(record["text"]) for record in benchmark]
    exact_index: dict[str, list[int]] = defaultdict(list)
    shingle_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, normalized in enumerate(benchmark_norm):
        exact_index[normalized].append(index)
        for shingle in benchmark_sets[index]:
            shingle_index[shingle].append(index)

    matches: list[dict[str, Any]] = []
    matched_components: set[str] = set()
    seen_row_side: set[tuple[int, str]] = set()
    for row_index, row in enumerate(rows):
        for side in ("premise", "hypothesis"):
            text = str(row[side])
            normalized = normalize_text(text)
            candidates: set[int] = set(exact_index.get(normalized, ()))
            current_shingles = shingles(text)
            for shingle in current_shingles:
                candidates.update(shingle_index.get(shingle, ()))
            best: tuple[float, int, str] | None = None
            exact = bool(exact_index.get(normalized))
            for candidate in sorted(candidates):
                similarity = 1.0 if benchmark_norm[candidate] == normalized else jaccard(current_shingles, benchmark_sets[candidate])
                if similarity < GROUP_SIMILARITY:
                    continue
                kind = "exact_normalized" if similarity == 1.0 else "word5_jaccard"
                option = (similarity, candidate, kind)
                if best is None or option[0] > best[0] or (option[0] == best[0] and option[1] < best[1]):
                    best = option
            if best is None:
                continue
            key = (row_index, side)
            if key in seen_row_side:
                continue
            seen_row_side.add(key)
            similarity, benchmark_index, kind = best
            matched_components.add(component_ids[row_index])
            matches.append(
                {
                    "vianli_row": row_index,
                    "vianli_uid": row["uid"],
                    "side": side,
                    "component_id": component_ids[row_index],
                    "match_kind": kind,
                    "jaccard": similarity,
                    "benchmark_task": benchmark[benchmark_index]["task"],
                    "benchmark_split": benchmark[benchmark_index]["split"],
                    "benchmark_id": benchmark[benchmark_index]["id"],
                }
            )
    rows_by_component: Counter[str] = Counter(component_ids)
    contaminated_rows = sum(rows_by_component[component] for component in matched_components)
    report = {
        "method": {
            "normalization": "NFC, casefold, whitespace collapse",
            "match": "exact normalized text OR word-5-shingle Jaccard >= 0.8",
            "sides": ["premise", "hypothesis"],
            "action": "remove every row in any matched connected component before split counts",
            "benchmark_scope": "five benchmark train and validation splits only",
        },
        "benchmark_rows": len(benchmark),
        "benchmark_unique_normalized_text": len(set(benchmark_norm)),
        "vianli_rows_with_direct_text_match": len({row["vianli_row"] for row in matches}),
        "vianli_side_matches": len(matches),
        "matched_component_count": len(matched_components),
        "contaminated_rows_removed": contaminated_rows,
        "match_kind_counts": dict(Counter(row["match_kind"] for row in matches)),
        "match_by_side": dict(Counter(row["side"] for row in matches)),
        "match_by_benchmark_task": dict(Counter(row["benchmark_task"] for row in matches)),
        "matches_path": str(ARTIFACT_ROOT / "decontam_matches.jsonl"),
    }
    return DecontamResult(matched_components, matches, report)


def assign_splits(rows: Sequence[dict[str, Any]], component_ids: Sequence[str], removed: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, (row, component_id) in enumerate(zip(rows, component_ids)):
        bucket = int(hashlib.sha256(component_id.encode("utf-8")).hexdigest()[:8], 16) % 10
        split = ("fit", "select", "audit")[0] if bucket <= 7 else ("select" if bucket == 8 else "audit")
        status = "removed_decontamination" if component_id in removed else split
        output.append(
            {
                "source_row": index,
                "uid": row["uid"],
                "label": row["label"],
                "label_id": LABEL_TO_ID[row["label"]],
                "component_id": component_id,
                "bucket": bucket,
                "assigned_split": split,
                "status": status,
            }
        )
    active = [row for row in output if row["status"] in {"fit", "select", "audit"}]
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("fit", "select", "audit"):
        split_rows = [row for row in active if row["status"] == split]
        split_counts[split] = {
            "rows": len(split_rows),
            "components": len({row["component_id"] for row in split_rows}),
            "labels": dict(Counter(row["label"] for row in split_rows)),
        }
    gate = {
        "fit_rows_minimum": 4000,
        "select_rows_minimum": 400,
        "audit_rows_minimum": 400,
        "each_class_minimum": 80,
        "audit_premise_groups_with_entailment_and_contradiction_minimum": 50,
    }
    audit_groups: dict[str, set[str]] = defaultdict(set)
    for row, assignment in zip(rows, output):
        if assignment["status"] == "audit":
            audit_groups[assignment["component_id"]].add(row["label"])
    audit_mixed_groups = sum(
        {"entailment", "contradiction"}.issubset(labels) for labels in audit_groups.values()
    )
    gate_result = {
        "fit_rows": split_counts["fit"]["rows"] >= gate["fit_rows_minimum"],
        "select_rows": split_counts["select"]["rows"] >= gate["select_rows_minimum"],
        "audit_rows": split_counts["audit"]["rows"] >= gate["audit_rows_minimum"],
        "each_class": all(
            split_counts[split]["labels"].get(label, 0) >= gate["each_class_minimum"]
            for split in ("select", "audit")
            for label in LABEL_NAMES
        ),
        "audit_mixed_premise_groups": audit_mixed_groups >= gate["audit_premise_groups_with_entailment_and_contradiction_minimum"],
    }
    return output, {
        "method": {
            "source": "official ViANLI train only",
            "bucket": "int(SHA256(component_id)[:8], 16) % 10",
            "bucket_mapping": {"0..7": "fit", "8": "select", "9": "audit"},
            "decontamination": "remove the entire component before counts",
            "label_or_model_output_used_for_split": False,
        },
        "rows": len(rows),
        "removed_rows": len(rows) - len(active),
        "removed_components": len(removed),
        "split_counts": split_counts,
        "audit_premise_groups": len(audit_groups),
        "audit_groups_with_entailment_and_contradiction": audit_mixed_groups,
        "gate_thresholds": gate,
        "gate_result": gate_result,
        "gate_pass": all(gate_result.values()),
    }


def benchmark_overlap_report(train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]]) -> dict[str, Any]:
    train_premises = {normalize_text(row["premise"]) for row in train}
    dev_premises = {normalize_text(row["premise"]) for row in dev}
    train_pairs = {
        (normalize_text(row["premise"]), normalize_text(row["hypothesis"])) for row in train
    }
    dev_pairs = {
        (normalize_text(row["premise"]), normalize_text(row["hypothesis"])) for row in dev
    }
    return {
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_unique_normalized_premises": len(train_premises),
        "dev_unique_normalized_premises": len(dev_premises),
        "dev_unique_premises_also_in_train": len(dev_premises & train_premises),
        "cross_split_exact_normalized_pair_overlap": len(train_pairs & dev_pairs),
        "cross_split_exact_raw_pair_overlap": len(
            {(row["premise"], row["hypothesis"]) for row in train}
            & {(row["premise"], row["hypothesis"]) for row in dev}
        ),
    }


def make_provenance(train: Sequence[dict[str, Any]], dev: Sequence[dict[str, Any]], benchmark_info: dict[str, Any]) -> dict[str, Any]:
    api: dict[str, Any] = {}
    if RAW_API.exists():
        api = json.loads(RAW_API.read_text(encoding="utf-8"))
    return {
        "protocol": "cafebert-nli-transfer-pilot-v1",
        "stage": "D-data-gate-preparation",
        "created_utc": "2026-09-22",
        "dataset": {
            "repo": "uitnlp/ViANLI",
            "revision": api.get("sha"),
            "revision_source": "Hugging Face dataset API response saved before raw file downloads",
            "raw_files_downloaded": ["README.md", "vianli_train.jsonl", "vianli_dev.jsonl"],
            "test_downloaded": False,
            "test_opened": False,
            "license_card": "README frontmatter says MIT; body says CC BY-NC-SA 4.0; preserve this inconsistency for root decision",
            "paper": "https://arxiv.org/abs/2406.17716",
            "raw_sha256": {
                name: sha256_file(RAW_ROOT / name)
                for name in ("README.md", "vianli_train.jsonl", "vianli_dev.jsonl", "dataset_api.json")
            },
            "api_response_sha256": sha256_file(RAW_API) if RAW_API.exists() else None,
            "api_file_listing": [item.get("rfilename") for item in api.get("siblings", []) if isinstance(item, dict)],
        },
        "counts": {"train": len(train), "dev": len(dev)},
        "benchmark_scope": "train and validation only for five tasks; loader invoked with include_test=False",
        "benchmark": benchmark_info,
        "label_map": LABEL_TO_ID,
    }


def run_full(train: list[dict[str, Any]], dev: list[dict[str, Any]]) -> None:
    schema = {
        "train": validate_source_rows(train, "train"),
        "dev": validate_source_rows(dev, "dev"),
        "cross_split_uid_overlap": len({row["uid"] for row in train} & {row["uid"] for row in dev}),
        "cross_split_exact_pair_overlap": len(
            {(row["premise"], row["hypothesis"]) for row in train}
            & {(row["premise"], row["hypothesis"]) for row in dev}
        ),
        "cross_split_normalized_pair_overlap": len(
            {
                (normalize_text(row["premise"]), normalize_text(row["hypothesis"])) for row in train
            }
            & {
                (normalize_text(row["premise"]), normalize_text(row["hypothesis"])) for row in dev
            }
        ),
    }
    write_json(ARTIFACT_ROOT / "schema_audit.json", schema)
    write_json(ARTIFACT_ROOT / "split_overlap.json", benchmark_overlap_report(train, dev))

    component_ids, grouping = group_train_components(train)
    write_json(ARTIFACT_ROOT / "grouping_report.json", grouping)
    write_jsonl(
        DATA_ROOT / "train_components.jsonl",
        (
            {
                "source_row": index,
                "uid": row["uid"],
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": row["label"],
                "label_id": LABEL_TO_ID[row["label"]],
                "component_id": component_ids[index],
            }
            for index, row in enumerate(train)
        ),
    )

    benchmark, benchmark_info = load_benchmark_texts()
    write_json(ARTIFACT_ROOT / "benchmark_text_scope.json", benchmark_info)
    decontam = decontaminate(train, component_ids, benchmark)
    write_jsonl(ARTIFACT_ROOT / "decontam_matches.jsonl", decontam.row_matches)
    write_json(ARTIFACT_ROOT / "decontam_report.json", decontam.report)
    assignments, split_report = assign_splits(train, component_ids, decontam.component_contaminated)
    write_jsonl(DATA_ROOT / "train_split_map.jsonl", assignments)
    write_json(ARTIFACT_ROOT / "split_coverage.json", split_report)

    # These are train-only fit/select/audit views.  Text is retained exactly as
    # published; the split map is the single source of component assignment.
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, assignment in zip(train, assignments):
        if assignment["status"] in {"fit", "select", "audit"}:
            by_status[assignment["status"]].append(
                {
                    "source_row": assignment["source_row"],
                    "uid": row["uid"],
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": row["label"],
                    "label_id": assignment["label_id"],
                    "component_id": assignment["component_id"],
                    "bucket": assignment["bucket"],
                }
            )
    for split in ("fit", "select", "audit"):
        write_jsonl(DATA_ROOT / f"{split}.jsonl", by_status[split])

    manifest = make_provenance(train, dev, benchmark_info)
    revision = manifest["dataset"].get("revision")
    manifest["download"] = {
        "api_endpoint": "https://huggingface.co/api/datasets/uitnlp/ViANLI",
        "api_revision_read_before_raw_download": revision,
        "requested_raw_urls": [
            f"https://huggingface.co/datasets/uitnlp/ViANLI/resolve/{revision}/README.md",
            f"https://huggingface.co/datasets/uitnlp/ViANLI/resolve/{revision}/vianli_train.jsonl",
            f"https://huggingface.co/datasets/uitnlp/ViANLI/resolve/{revision}/vianli_dev.jsonl",
        ],
        "requested_test_url": None,
        "downloaded_paths": [
            str(RAW_README.relative_to(ROOT)),
            str(RAW_TRAIN.relative_to(ROOT)),
            str(RAW_DEV.relative_to(ROOT)),
        ],
        "test_file_present_under_data_root": any(
            path.name.lower() in {"vianli_test.jsonl", "test.jsonl"}
            for path in RAW_ROOT.rglob("*")
            if path.is_file()
        ),
    }
    write_json(ARTIFACT_ROOT / "download_manifest.json", manifest["download"])
    manifest["artifacts"] = {
        "schema": str(ARTIFACT_ROOT / "schema_audit.json"),
        "grouping": str(ARTIFACT_ROOT / "grouping_report.json"),
        "decontamination": str(ARTIFACT_ROOT / "decontam_report.json"),
        "split_coverage": str(ARTIFACT_ROOT / "split_coverage.json"),
        "audit_sample": str(ARTIFACT_ROOT / "audit_sample.jsonl"),
        "audit_sample_compact": str(ARTIFACT_ROOT / "audit_sample.txt"),
    }
    write_json(ARTIFACT_ROOT / "provenance_manifest.json", manifest)
    write_hash_manifest()


def write_hash_manifest() -> None:
    paths = sorted(
        path
        for base in (DATA_ROOT, ARTIFACT_ROOT)
        for path in base.rglob("*")
        if path.is_file() and not path.name.endswith(".tmp") and path.name != "sha256_manifest.json"
    )
    rows = []
    for path in paths:
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    write_json(ARTIFACT_ROOT / "sha256_manifest.json", {"files": rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-only", action="store_true", help="write the fixed 300-row sample and stop")
    args = parser.parse_args()
    for path in (RAW_TRAIN, RAW_DEV, RAW_README, RAW_API):
        if not path.exists():
            raise FileNotFoundError(path)
    train = read_jsonl(RAW_TRAIN)
    dev = read_jsonl(RAW_DEV)
    sample = build_sample(train)
    write_sample(sample)
    print(json.dumps({"sample": str(ARTIFACT_ROOT / "audit_sample.jsonl"), "rows": len(sample), "class_counts": dict(Counter(row["label"] for row in sample))}, ensure_ascii=False))
    if args.sample_only:
        return
    run_full(train, dev)
    print(json.dumps({"status": "completed", "artifact_root": str(ARTIFACT_ROOT), "data_root": str(DATA_ROOT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
