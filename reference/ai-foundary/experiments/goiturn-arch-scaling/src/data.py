"""Load the three local Vietnamese classification datasets without text normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATASET_NAMES = ("vsfc", "vlsp_restaurant", "visfd")
VSFC_LABELS = {"negative": 0, "neutral": 1, "positive": 2}
VISFD_POLARITIES = {"Positive": 1, "Negative": 2, "Neutral": 3}
VISFD_LABEL_RE = re.compile(r"\{([^{}#;]+)(?:#([^{};]+))?\}")


@dataclass(frozen=True)
class DatasetSplits:
    """Texts, integer labels, and the canonical output-slot order for one task."""

    train_texts: list[str]
    train_labels: list[int] | list[list[int]]
    valid_texts: list[str]
    valid_labels: list[int] | list[list[int]]
    test_texts: list[str]
    test_labels: list[int] | list[list[int]]
    aspect_names: list[str]

    @property
    def is_absa(self) -> bool:
        return bool(self.aspect_names)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    return pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def _clean_texts(values: pd.Series) -> list[str]:
    return [str(value).strip() for value in values]


def _cap(
    texts: list[str], labels: list[int] | list[list[int]], limit: int | None
) -> tuple[list[str], list[int] | list[list[int]]]:
    if limit is None:
        return texts, labels
    return texts[:limit], labels[:limit]


def _load_vsfc(raw_dir: Path) -> DatasetSplits:
    split_files = {
        "train": raw_dir / "UIT-VSFC_train.csv",
        "valid": raw_dir / "UIT-VSFC_valid.csv",
        "test": raw_dir / "UIT-VSFC_test.csv",
    }
    loaded: dict[str, tuple[list[str], list[int]]] = {}
    for split, path in split_files.items():
        frame = _read_csv(path)
        _require_columns(frame, {"text", "label"}, path)
        junk = frame["label"].astype(str).str.strip() == ""
        if junk.any():
            print(f"Dropping {int(junk.sum())} empty-label row(s) from {path.name}", flush=True)
            frame = frame[~junk]
        unknown = sorted(set(frame["label"]).difference(VSFC_LABELS))
        if unknown:
            raise ValueError(f"Unknown VSFC labels in {path}: {unknown}")
        loaded[split] = (
            _clean_texts(frame["text"]),
            [VSFC_LABELS[value] for value in frame["label"]],
        )
    return DatasetSplits(
        *loaded["train"], *loaded["valid"], *loaded["test"], aspect_names=[]
    )


def _load_vlsp(raw_dir: Path) -> DatasetSplits:
    path = raw_dir / "VLSP2018-ABSA-Restaurant.csv"
    frame = _read_csv(path)
    _require_columns(frame, {"Review", "type", "dataset"}, path)
    aspects = [
        column
        for column in frame.columns
        if column not in {"Review", "type", "dataset"}
    ]
    if len(aspects) != 12:
        raise ValueError(f"Expected 12 VLSP aspect columns, found {len(aspects)}: {aspects}")
    split_names = {"train": "train", "valid": "dev", "test": "test"}
    loaded: dict[str, tuple[list[str], list[list[int]]]] = {}
    for output_split, source_split in split_names.items():
        part = frame.loc[frame["type"] == source_split]
        if part.empty:
            raise ValueError(f"No VLSP rows found for type={source_split!r}")
        labels = part[aspects].astype(int)
        if not labels.isin(range(4)).all().all():
            raise ValueError(f"VLSP split {source_split!r} has a label outside 0..3")
        loaded[output_split] = (_clean_texts(part["Review"]), labels.values.tolist())
    return DatasetSplits(
        *loaded["train"], *loaded["valid"], *loaded["test"], aspect_names=aspects
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
    frame = _read_csv(path)
    _require_columns(
        frame, {"dataset", "type", "index", "comment", "n_star", "date_time", "label"}, path
    )
    aspects = sorted(
        {aspect for label in frame["label"] for aspect, _ in VISFD_LABEL_RE.findall(label)}
    )
    if not aspects:
        raise ValueError("No ViSFD aspects were discovered")
    print(f"Discovered ViSFD aspects ({len(aspects)}): {', '.join(aspects)}", flush=True)
    aspect_to_index = {aspect: index for index, aspect in enumerate(aspects)}
    split_names = {"train": "train", "valid": "dev", "test": "test"}
    loaded: dict[str, tuple[list[str], list[list[int]]]] = {}
    for output_split, source_split in split_names.items():
        part = frame.loc[frame["type"] == source_split]
        if part.empty:
            raise ValueError(f"No ViSFD rows found for type={source_split!r}")
        loaded[output_split] = (
            _clean_texts(part["comment"]),
            [_parse_visfd_label(label, aspect_to_index) for label in part["label"]],
        )
    return DatasetSplits(
        *loaded["train"], *loaded["valid"], *loaded["test"], aspect_names=aspects
    )


def load_dataset(dataset: str, repo_root: Path, smoke: bool = False) -> DatasetSplits:
    """Load one dataset and optionally apply deterministic smoke-test row caps."""
    raw_dir = repo_root / "data" / "raw"
    loaders = {"vsfc": _load_vsfc, "vlsp_restaurant": _load_vlsp, "visfd": _load_visfd}
    if dataset not in loaders:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {sorted(loaders)}")
    splits = loaders[dataset](raw_dir)
    limits = (200, 100, 100) if smoke else (None, None, None)
    train = _cap(splits.train_texts, splits.train_labels, limits[0])
    valid = _cap(splits.valid_texts, splits.valid_labels, limits[1])
    test = _cap(splits.test_texts, splits.test_labels, limits[2])
    return DatasetSplits(*train, *valid, *test, aspect_names=splits.aspect_names)
