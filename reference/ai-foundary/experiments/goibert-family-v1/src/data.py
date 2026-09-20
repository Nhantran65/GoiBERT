"""Load the four local Vietnamese datasets without text normalization."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


DATASET_NAMES = ("vsfc", "vlsp_restaurant", "visfd", "phoatis")
VSFC_LABELS = {"negative": 0, "neutral": 1, "positive": 2}
VISFD_POLARITIES = {"Positive": 1, "Negative": 2, "Neutral": 3}
VISFD_LABEL_RE = re.compile(r"\{([^{}#;]+)(?:#([^{};]+))?\}")


@dataclass(frozen=True)
class DatasetSplits:
    """Texts, labels, and canonical output-slot order for one dataset."""

    train_texts: list[str]
    train_labels: list[int] | list[list[int]]
    valid_texts: list[str]
    valid_labels: list[int] | list[list[int]]
    test_texts: list[str]
    test_labels: list[int] | list[list[int]]
    aspect_names: list[str]
    class_names: list[str]

    @property
    def is_absa(self) -> bool:
        return bool(self.aspect_names)

    @property
    def output_size(self) -> int:
        return len(self.aspect_names) * 4 if self.is_absa else len(self.class_names)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header: {path}")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def _require_columns(fieldnames: list[str], required: set[str], path: Path) -> None:
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _cap(
    texts: list[str], labels: list[int] | list[list[int]], limit: int | None
) -> tuple[list[str], list[int] | list[list[int]]]:
    if limit is None:
        return texts, labels
    return texts[:limit], labels[:limit]


def _load_vsfc(raw_dir: Path) -> DatasetSplits:
    paths = {
        "train": raw_dir / "UIT-VSFC_train.csv",
        "valid": raw_dir / "UIT-VSFC_valid.csv",
        "test": raw_dir / "UIT-VSFC_test.csv",
    }
    loaded: dict[str, tuple[list[str], list[int]]] = {}
    for split, path in paths.items():
        fieldnames, rows = _read_csv(path)
        _require_columns(fieldnames, {"text", "label"}, path)
        clean_rows = [row for row in rows if row["label"].strip()]
        dropped = len(rows) - len(clean_rows)
        if dropped:
            print(f"Dropping {dropped} empty-label row(s) from {path.name}", flush=True)
        unknown = sorted({row["label"] for row in clean_rows}.difference(VSFC_LABELS))
        if unknown:
            raise ValueError(f"Unknown VSFC labels in {path}: {unknown}")
        loaded[split] = (
            [row["text"].strip() for row in clean_rows],
            [VSFC_LABELS[row["label"]] for row in clean_rows],
        )
    return DatasetSplits(
        *loaded["train"],
        *loaded["valid"],
        *loaded["test"],
        aspect_names=[],
        class_names=list(VSFC_LABELS),
    )


def _load_vlsp(raw_dir: Path) -> DatasetSplits:
    path = raw_dir / "VLSP2018-ABSA-Restaurant.csv"
    fieldnames, rows = _read_csv(path)
    _require_columns(fieldnames, {"Review", "type", "dataset"}, path)
    aspects = [name for name in fieldnames if name not in {"Review", "type", "dataset"}]
    if len(aspects) != 12:
        raise ValueError(f"Expected 12 VLSP aspect columns, found {len(aspects)}: {aspects}")
    loaded: dict[str, tuple[list[str], list[list[int]]]] = {}
    for output_split, source_split in {"train": "train", "valid": "dev", "test": "test"}.items():
        part = [row for row in rows if row["type"] == source_split]
        if not part:
            raise ValueError(f"No VLSP rows found for type={source_split!r}")
        labels = [[int(row[aspect]) for aspect in aspects] for row in part]
        if any(value not in range(4) for label in labels for value in label):
            raise ValueError(f"VLSP split {source_split!r} has a label outside 0..3")
        loaded[output_split] = ([row["Review"].strip() for row in part], labels)
    return DatasetSplits(
        *loaded["train"],
        *loaded["valid"],
        *loaded["test"],
        aspect_names=aspects,
        class_names=[],
    )


def _parse_visfd_label(label: str, aspect_to_index: dict[str, int]) -> list[int]:
    output = [0] * len(aspect_to_index)
    matches = VISFD_LABEL_RE.findall(label)
    if not matches and label.strip():
        raise ValueError(f"Could not parse ViSFD label: {label!r}")
    for aspect, polarity in matches:
        if aspect not in aspect_to_index:
            raise ValueError(f"Unknown ViSFD aspect {aspect!r}")
        if aspect == "OTHERS" and not polarity:
            output[aspect_to_index[aspect]] = 1
        elif polarity in VISFD_POLARITIES:
            output[aspect_to_index[aspect]] = VISFD_POLARITIES[polarity]
        else:
            raise ValueError(f"Unknown ViSFD polarity {polarity!r} for {aspect!r}")
    return output


def _load_visfd(raw_dir: Path) -> DatasetSplits:
    path = raw_dir / "ViSFD.csv"
    fieldnames, rows = _read_csv(path)
    _require_columns(
        fieldnames,
        {"dataset", "type", "index", "comment", "n_star", "date_time", "label"},
        path,
    )
    aspects = sorted(
        {aspect for row in rows for aspect, _ in VISFD_LABEL_RE.findall(row["label"])}
    )
    if not aspects:
        raise ValueError("No ViSFD aspects were discovered")
    print(f"Discovered ViSFD aspects ({len(aspects)}): {', '.join(aspects)}", flush=True)
    aspect_to_index = {aspect: index for index, aspect in enumerate(aspects)}
    loaded: dict[str, tuple[list[str], list[list[int]]]] = {}
    for output_split, source_split in {"train": "train", "valid": "dev", "test": "test"}.items():
        part = [row for row in rows if row["type"] == source_split]
        if not part:
            raise ValueError(f"No ViSFD rows found for type={source_split!r}")
        loaded[output_split] = (
            [row["comment"].strip() for row in part],
            [_parse_visfd_label(row["label"], aspect_to_index) for row in part],
        )
    return DatasetSplits(
        *loaded["train"],
        *loaded["valid"],
        *loaded["test"],
        aspect_names=aspects,
        class_names=[],
    )


def _read_phoatis_split(split_dir: Path) -> tuple[list[str], list[str]]:
    text_path = split_dir / "seq.in"
    label_path = split_dir / "label"
    for path in (text_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset file is missing: {path}")
    texts = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines()]
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()]
    if len(texts) != len(labels):
        raise ValueError(
            f"PhoATIS text/label counts differ in {split_dir}: {len(texts)} != {len(labels)}"
        )
    if not texts or any(not text for text in texts) or any(not label for label in labels):
        raise ValueError(f"PhoATIS split contains an empty row: {split_dir}")
    return texts, labels


def _load_phoatis(raw_dir: Path) -> DatasetSplits:
    raw_splits = {
        output_split: _read_phoatis_split(raw_dir / source_split)
        for output_split, source_split in {
            "train": "train",
            "valid": "dev",
            "test": "test",
        }.items()
    }
    class_names = sorted(
        {label for _, labels in raw_splits.values() for label in labels}
    )
    if len(class_names) != 28:
        raise ValueError(f"Expected 28 PhoATIS intents, found {len(class_names)}")
    label_to_index = {label: index for index, label in enumerate(class_names)}
    loaded: dict[str, tuple[list[str], list[int]]] = {}
    for split, (texts, labels) in raw_splits.items():
        unknown = sorted(set(labels).difference(label_to_index))
        if unknown:
            raise ValueError(f"Unknown PhoATIS labels in {split}: {unknown}")
        loaded[split] = (texts, [label_to_index[label] for label in labels])
    return DatasetSplits(
        *loaded["train"],
        *loaded["valid"],
        *loaded["test"],
        aspect_names=[],
        class_names=class_names,
    )


def load_dataset(dataset: str, repo_root: Path, smoke: bool = False) -> DatasetSplits:
    """Load one dataset and optionally apply deterministic CPU-smoke caps."""
    loaders = {
        "vsfc": _load_vsfc,
        "vlsp_restaurant": _load_vlsp,
        "visfd": _load_visfd,
        "phoatis": lambda raw_dir: _load_phoatis(raw_dir / "phoatis"),
    }
    if dataset not in loaders:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {sorted(loaders)}")
    splits = loaders[dataset](repo_root / "data" / "raw")
    limits = (24, 12, 12) if smoke else (None, None, None)
    train = _cap(splits.train_texts, splits.train_labels, limits[0])
    valid = _cap(splits.valid_texts, splits.valid_labels, limits[1])
    test = _cap(splits.test_texts, splits.test_labels, limits[2])
    return DatasetSplits(
        *train,
        *valid,
        *test,
        aspect_names=splits.aspect_names,
        class_names=splits.class_names,
    )
