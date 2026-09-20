"""Load the frozen GoiBERT benchmark datasets.

The loaders in this module deliberately do very little.  They read the files
as published, keep row order and text bytes (apart from removing the line
terminator used by line-oriented VSFC files), and apply label maps that are
defined in code.  In particular, label vocabularies are never inferred from
the test split.

The five task IDs are the task granularity used by the E0/E1 runner:

``vsfc_sentiment`` and ``vsfc_topic`` use the same VSFC text rows;
``visfd`` is smartphone aspect feedback; and ``absa_restaurant`` and
``absa_hotel`` use the public UIT-ABSA CSV candidate.

ViSFD has an ``{OTHERS}`` tag with no polarity.  It is represented by code 4
in that task only.  Code 4 is a detection-positive, classification-unscored
slot.  The metric module receives the corresponding ``pair_scored_aspects``
mask so it cannot accidentally score OTHERS as a positive sentiment.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TASK_IDS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
SPLIT_NAMES = ("train", "validation", "test")

# These maps are fixed to the published/source conventions.  Keep the integer
# IDs stable: they are part of the serialized experiment contract.
VSFC_SENTIMENT_LABEL_NAMES = (
    "negative",
    "neutral",
    "positive",
)
VSFC_TOPIC_LABEL_NAMES = (
    "lecturer",
    "training_program",
    "facility",
    "others",
)
VSFC_SENTIMENT_LABELS = {name: index for index, name in enumerate(VSFC_SENTIMENT_LABEL_NAMES)}
VSFC_TOPIC_LABELS = {name: index for index, name in enumerate(VSFC_TOPIC_LABEL_NAMES)}

# UIT-ABSA source matrices use 0=absent, 1=negative, 2=neutral, 3=positive.
ABSA_LABEL_NAMES = ("absent", "negative", "neutral", "positive")
ABSA_LABEL_IDS = tuple(range(len(ABSA_LABEL_NAMES)))

# ViSFD source labels use 0=absent, Positive/Negative/Neutral for polarity
# slots, and a polarity-less {OTHERS} tag.  We retain source polarity IDs for
# the first three values and reserve 4 for OTHERS so no semantic information is
# lost by pretending OTHERS is a positive label.
VISFD_LABEL_NAMES = ("absent", "positive", "negative", "neutral", "others")
VISFD_POLARITIES = {"Positive": 1, "Negative": 2, "Neutral": 3}
VISFD_OTHERS_CODE = 4
VISFD_OTHERS_ASPECT_INDEX = 5
VISFD_ASPECT_NAMES = (
    "BATTERY",
    "CAMERA",
    "DESIGN",
    "FEATURES",
    "GENERAL",
    "OTHERS",
    "PERFORMANCE",
    "PRICE",
    "SCREEN",
    "SER&ACC",
    "STORAGE",
)
VISFD_LABEL_RE = re.compile(r"\{([^{}#;]+)(?:#([^{};]+))?\}")

RESTAURANT_ASPECT_NAMES = (
    "AMBIENCE#GENERAL",
    "DRINKS#PRICES",
    "DRINKS#QUALITY",
    "DRINKS#STYLE&OPTIONS",
    "FOOD#PRICES",
    "FOOD#QUALITY",
    "FOOD#STYLE&OPTIONS",
    "LOCATION#GENERAL",
    "RESTAURANT#GENERAL",
    "RESTAURANT#MISCELLANEOUS",
    "RESTAURANT#PRICES",
    "SERVICE#GENERAL",
)
HOTEL_ASPECT_NAMES = (
    "FACILITIES#CLEANLINESS",
    "FACILITIES#COMFORT",
    "FACILITIES#DESIGN&FEATURES",
    "FACILITIES#GENERAL",
    "FACILITIES#MISCELLANEOUS",
    "FACILITIES#PRICES",
    "FACILITIES#QUALITY",
    "FOOD&DRINKS#MISCELLANEOUS",
    "FOOD&DRINKS#PRICES",
    "FOOD&DRINKS#QUALITY",
    "FOOD&DRINKS#STYLE&OPTIONS",
    "HOTEL#CLEANLINESS",
    "HOTEL#COMFORT",
    "HOTEL#DESIGN&FEATURES",
    "HOTEL#GENERAL",
    "HOTEL#MISCELLANEOUS",
    "HOTEL#PRICES",
    "HOTEL#QUALITY",
    "LOCATION#GENERAL",
    "ROOMS#CLEANLINESS",
    "ROOMS#COMFORT",
    "ROOMS#DESIGN&FEATURES",
    "ROOMS#GENERAL",
    "ROOMS#MISCELLANEOUS",
    "ROOMS#PRICES",
    "ROOMS#QUALITY",
    "ROOM_AMENITIES#CLEANLINESS",
    "ROOM_AMENITIES#COMFORT",
    "ROOM_AMENITIES#DESIGN&FEATURES",
    "ROOM_AMENITIES#GENERAL",
    "ROOM_AMENITIES#MISCELLANEOUS",
    "ROOM_AMENITIES#PRICES",
    "ROOM_AMENITIES#QUALITY",
    "SERVICE#GENERAL",
)

EXPECTED_COUNTS = {
    "vsfc_sentiment": {"train": 11_426, "validation": 1_583, "test": 3_166},
    "vsfc_topic": {"train": 11_426, "validation": 1_583, "test": 3_166},
    "visfd": {"train": 7_786, "validation": 1_112, "test": 2_224},
    "absa_restaurant": {"train": 7_028, "validation": 771, "test": 1_938},
    "absa_hotel": {"train": 7_180, "validation": 795, "test": 2_030},
}


@dataclass(frozen=True)
class Example:
    """One source row with a stable experiment ID.

    ``label`` is an int for VSFC tasks and a tuple of per-aspect integer labels
    for ABSA tasks.  ``pair_scored`` is a per-aspect mask; it is only set for
    ViSFD, where OTHERS is detection-only.
    """

    id: str
    text: str
    label: int | tuple[int, ...]
    raw_label: str | None = None
    source_index: str | None = None
    pair_scored: tuple[bool, ...] | None = None

    @property
    def labels(self) -> int | tuple[int, ...]:
        """Alias used by training code."""

        return self.label


@dataclass(frozen=True)
class TaskSplit:
    """An ordered split of examples."""

    name: str
    examples: tuple[Example, ...]

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    @property
    def ids(self) -> list[str]:
        return [example.id for example in self.examples]

    @property
    def texts(self) -> list[str]:
        return [example.text for example in self.examples]

    @property
    def labels(self) -> list[int] | list[tuple[int, ...]]:
        return [example.label for example in self.examples]

    @property
    def pair_scored(self) -> tuple[bool, ...] | None:
        masks = {example.pair_scored for example in self.examples}
        if not masks:
            return None
        if len(masks) != 1:
            raise ValueError(f"Split {self.name!r} has inconsistent pair-scored masks")
        return next(iter(masks))


@dataclass(frozen=True)
class TaskDataset:
    """Loaded task and metadata consumed by the E1 runner."""

    task_id: str
    splits: Mapping[str, TaskSplit]
    label_names: tuple[str, ...]
    aspect_names: tuple[str, ...] = ()
    slot_label_names: tuple[tuple[str, ...], ...] = ()
    pair_scored_aspects: tuple[bool, ...] | None = None
    source_files: tuple[str, ...] = ()
    label_policy: str = "fixed_source_map"

    def __post_init__(self) -> None:
        if self.task_id not in TASK_IDS:
            raise ValueError(f"Unsupported task ID: {self.task_id!r}")
        # Screening loaders intentionally omit test.  Requiring train and
        # validation while allowing test to be absent makes it impossible for
        # a runner to accidentally touch test just to construct a train/dev
        # dataloader.
        missing = {"train", "validation"}.difference(self.splits)
        if missing:
            raise ValueError(f"Task {self.task_id} is missing splits: {sorted(missing)}")
        if self.aspect_names and len(self.slot_label_names) != len(self.aspect_names):
            raise ValueError("slot_label_names must have one entry per aspect")
        if self.pair_scored_aspects is not None and len(self.pair_scored_aspects) != len(
            self.aspect_names
        ):
            raise ValueError("pair_scored_aspects must have one entry per aspect")

    @property
    def is_absa(self) -> bool:
        return bool(self.aspect_names)

    @property
    def train(self) -> TaskSplit:
        return self.splits["train"]

    @property
    def validation(self) -> TaskSplit:
        return self.splits["validation"]

    @property
    def valid(self) -> TaskSplit:
        """Compatibility alias for code using the historical ``valid`` name."""

        return self.validation

    @property
    def test(self) -> TaskSplit:
        if "test" not in self.splits:
            raise RuntimeError(
                "Test split was not loaded. Call load_task(..., include_test=True) "
                "only after the selection gate."
            )
        return self.splits["test"]

    @property
    def has_test(self) -> bool:
        return "test" in self.splits

    @property
    def num_classes(self) -> int:
        """Compatibility name for the scalar head width."""

        return self.num_labels

    @property
    def class_names(self) -> tuple[str, ...]:
        """Compatibility alias consumed by the task runner."""

        return self.label_names

    @property
    def slot_class_counts(self) -> tuple[int, ...]:
        """Compatibility alias for runner code using the older name."""

        return self.aspect_class_counts

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_files": list(self.source_files),
            "label_policy": self.label_policy,
            "include_test": self.has_test,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return {"source_files": list(self.source_files), "label_policy": self.label_policy}

    @property
    def num_aspects(self) -> int:
        return len(self.aspect_names)

    @property
    def num_labels(self) -> int:
        """Total classifier output width expected by a flat ABSA head."""

        if not self.is_absa:
            return len(self.label_names)
        return sum(len(labels) for labels in self.slot_label_names)

    @property
    def aspect_class_counts(self) -> tuple[int, ...]:
        """Per-aspect head widths used to split a flat ABSA logits tensor."""

        return tuple(len(labels) for labels in self.slot_label_names)

    @property
    def output_size(self) -> int:
        """Flat head size used by the historical reference harness."""

        return self.num_labels

    def labels_for_model(self, split: str) -> list[int] | list[tuple[int, ...]]:
        """Return training-space labels for a split.

        All tasks return their frozen source IDs except ViSFD, whose OTHERS
        raw code 4 must become binary model code 1 in the OTHERS slot.  The
        raw labels remain available as ``splits[split].labels`` for evaluation.
        """

        values = self.splits[split].labels
        if self.task_id == "visfd":
            return visfd_to_model_labels(values)  # type: ignore[arg-type]
        return values


def _repo_raw_dir(repo_root: Path | str | None) -> Path:
    if repo_root is None:
        return Path(__file__).resolve().parents[2] / "data" / "raw"
    root = Path(repo_root).expanduser().resolve()
    if root.name == "raw":
        return root
    if (root / "data" / "raw").is_dir():
        return root / "data" / "raw"
    # This branch makes the API convenient in tests that pass a temporary raw
    # directory without changing its name.
    return root


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    # splitlines removes only line separators.  Do not strip text content.
    return path.read_text(encoding="utf-8").splitlines()


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header: {path}")
        fieldnames = tuple(reader.fieldnames)
        rows = [dict(row) for row in reader]
    if any(value is None for row in rows for value in row.values()):
        raise ValueError(f"Dataset has a row with missing CSV fields: {path}")
    return fieldnames, rows


def _require_exact_header(fieldnames: Sequence[str], expected: Sequence[str], path: Path) -> None:
    if tuple(fieldnames) != tuple(expected):
        raise ValueError(
            f"Unexpected columns in {path}: expected {list(expected)!r}, got {list(fieldnames)!r}"
        )


def _ensure_nonempty_text(text: str, source: str) -> None:
    if not text or not text.strip():
        raise ValueError(f"Empty text row in {source}")


def _build_line_task(
    task_id: str,
    raw_dir: Path,
    label_filename: str,
    label_names: tuple[str, ...],
    label_map: Mapping[str, int],
    include_test: bool,
) -> TaskDataset:
    loaded: dict[str, TaskSplit] = {}
    source_files: list[str] = []
    split_plan = [("train", "train"), ("validation", "dev")]
    if include_test:
        split_plan.append(("test", "test"))
    for split, directory in split_plan:
        split_dir = raw_dir / "uit-vsfc" / directory
        text_path = split_dir / "sents.txt"
        label_path = split_dir / label_filename
        texts = _read_lines(text_path)
        raw_labels = _read_lines(label_path)
        source_files.extend([str(text_path), str(label_path)])
        if len(texts) != len(raw_labels):
            raise ValueError(
                f"VSFC {split} text/label count mismatch: {len(texts)} != {len(raw_labels)}"
            )
        examples: list[Example] = []
        for row, (text, raw_label) in enumerate(zip(texts, raw_labels)):
            _ensure_nonempty_text(text, f"{text_path}:{row + 1}")
            token = raw_label.strip()
            if token not in label_map:
                # The source stores integer IDs.  Resolve them through the
                # fixed name map without ever discovering IDs from a split.
                try:
                    value = int(token)
                except ValueError as error:
                    raise ValueError(f"Unknown VSFC label {raw_label!r} in {label_path}") from error
                if value not in range(len(label_names)):
                    raise ValueError(f"VSFC label {value} outside fixed map in {label_path}")
            else:
                value = label_map[token]
            examples.append(
                Example(
                    id=f"{task_id}:{split}:{row:06d}",
                    text=text,
                    label=value,
                    raw_label=raw_label,
                    source_index=str(row),
                )
            )
        loaded[split] = TaskSplit(split, tuple(examples))
    return TaskDataset(
        task_id=task_id,
        splits=loaded,
        label_names=label_names,
        source_files=tuple(source_files),
    )


def _load_vsfc_sentiment(raw_dir: Path, include_test: bool) -> TaskDataset:
    return _build_line_task(
        "vsfc_sentiment",
        raw_dir,
        "sentiments.txt",
        VSFC_SENTIMENT_LABEL_NAMES,
        {str(i): i for i in range(len(VSFC_SENTIMENT_LABEL_NAMES))},
        include_test,
    )


def _load_vsfc_topic(raw_dir: Path, include_test: bool) -> TaskDataset:
    return _build_line_task(
        "vsfc_topic",
        raw_dir,
        "topics.txt",
        VSFC_TOPIC_LABEL_NAMES,
        {str(i): i for i in range(len(VSFC_TOPIC_LABEL_NAMES))},
        include_test,
    )


def _parse_visfd_label(label: str) -> tuple[int, ...]:
    values = [0] * len(VISFD_ASPECT_NAMES)
    aspect_to_index = {aspect: index for index, aspect in enumerate(VISFD_ASPECT_NAMES)}
    matches = VISFD_LABEL_RE.findall(label)
    if not matches:
        raise ValueError(f"Could not parse non-empty ViSFD label: {label!r}")
    seen: set[str] = set()
    for aspect, polarity in matches:
        if aspect not in aspect_to_index:
            raise ValueError(f"Unknown ViSFD aspect {aspect!r} in {label!r}")
        if aspect in seen:
            raise ValueError(f"Duplicate ViSFD aspect {aspect!r} in {label!r}")
        seen.add(aspect)
        index = aspect_to_index[aspect]
        if aspect == "OTHERS":
            if polarity:
                raise ValueError(f"ViSFD OTHERS unexpectedly has polarity {polarity!r}")
            values[index] = VISFD_OTHERS_CODE
        elif polarity in VISFD_POLARITIES:
            values[index] = VISFD_POLARITIES[polarity]
        else:
            raise ValueError(f"Unknown ViSFD polarity {polarity!r} for {aspect!r}")
    return tuple(values)


def visfd_to_model_labels(rows: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    """Map frozen ViSFD labels to per-slot training labels.

    The ten polarity-bearing slots keep ``0/1/2/3``.  The OTHERS slot is a
    binary ``0/1`` head, so raw code 4 becomes model label 1.  This explicit
    adapter prevents a generic four-class head from treating OTHERS as
    positive sentiment.
    """

    mapped: list[tuple[int, ...]] = []
    for row_index, row in enumerate(rows):
        if len(row) != len(VISFD_ASPECT_NAMES):
            raise ValueError(f"ViSFD row {row_index} has the wrong number of slots")
        values = list(row)
        others = values[VISFD_OTHERS_ASPECT_INDEX]
        if others not in (0, VISFD_OTHERS_CODE):
            raise ValueError(f"ViSFD OTHERS raw code must be 0 or 4, got {others}")
        for index, value in enumerate(values):
            if index == VISFD_OTHERS_ASPECT_INDEX:
                continue
            if value not in (0, 1, 2, 3):
                raise ValueError(f"ViSFD polarity slot {index} has invalid code {value}")
        values[VISFD_OTHERS_ASPECT_INDEX] = int(others == VISFD_OTHERS_CODE)
        mapped.append(tuple(values))
    return mapped


def visfd_from_model_labels(rows: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    """Map model predictions back to the frozen ViSFD evaluation encoding."""

    mapped: list[tuple[int, ...]] = []
    for row_index, row in enumerate(rows):
        if len(row) != len(VISFD_ASPECT_NAMES):
            raise ValueError(f"ViSFD row {row_index} has the wrong number of slots")
        values = list(row)
        others = values[VISFD_OTHERS_ASPECT_INDEX]
        if others not in (0, 1):
            raise ValueError(f"ViSFD OTHERS model code must be binary, got {others}")
        for index, value in enumerate(values):
            if index == VISFD_OTHERS_ASPECT_INDEX:
                continue
            if value not in (0, 1, 2, 3):
                raise ValueError(f"ViSFD polarity slot {index} has invalid code {value}")
        values[VISFD_OTHERS_ASPECT_INDEX] = VISFD_OTHERS_CODE if others else 0
        mapped.append(tuple(values))
    return mapped


def _load_visfd(raw_dir: Path, include_test: bool) -> TaskDataset:
    loaded: dict[str, TaskSplit] = {}
    source_files: list[str] = []
    expected_header = ("index", "comment", "n_star", "date_time", "label")
    split_plan = [("train", "Train.csv"), ("validation", "Dev.csv")]
    if include_test:
        split_plan.append(("test", "Test.csv"))
    for split, filename in split_plan:
        path = raw_dir / "uit-visfd" / filename
        fieldnames, rows = _read_csv(path)
        _require_exact_header(fieldnames, expected_header, path)
        source_files.append(str(path))
        examples: list[Example] = []
        for row_number, row in enumerate(rows):
            text = row["comment"]
            _ensure_nonempty_text(text, f"{path}:{row_number + 2}")
            source_index = row["index"]
            try:
                int(source_index)
            except ValueError as error:
                raise ValueError(f"Non-integer ViSFD source index {source_index!r}") from error
            label = _parse_visfd_label(row["label"])
            examples.append(
                Example(
                    id=f"visfd:{split}:{row_number:06d}",
                    text=text,
                    label=label,
                    raw_label=row["label"],
                    source_index=source_index,
                    pair_scored=tuple(aspect != "OTHERS" for aspect in VISFD_ASPECT_NAMES),
                )
            )
        loaded[split] = TaskSplit(split, tuple(examples))
    return TaskDataset(
        task_id="visfd",
        splits=loaded,
        label_names=VISFD_LABEL_NAMES,
        aspect_names=VISFD_ASPECT_NAMES,
        slot_label_names=tuple(
            ("absent", "others") if aspect == "OTHERS" else ("absent", "positive", "negative", "neutral")
            for aspect in VISFD_ASPECT_NAMES
        ),
        pair_scored_aspects=tuple(aspect != "OTHERS" for aspect in VISFD_ASPECT_NAMES),
        source_files=tuple(source_files),
        label_policy=(
            "ViSFD OTHERS=code4; include in detection, exclude from polarity pair "
            "classification because source has no polarity"
        ),
    )


def _load_absa(
    raw_dir: Path,
    task_id: str,
    directory_name: str,
    aspect_names: tuple[str, ...],
    include_test: bool,
) -> TaskDataset:
    loaded: dict[str, TaskSplit] = {}
    source_files: list[str] = []
    expected_header = ("Review", *aspect_names)
    split_plan = [("train", "Train.csv"), ("validation", "Dev.csv")]
    if include_test:
        split_plan.append(("test", "Test.csv"))
    for split, filename in split_plan:
        path = raw_dir / "uit-absa-public-candidate" / "dataset" / directory_name / filename
        fieldnames, rows = _read_csv(path)
        _require_exact_header(fieldnames, expected_header, path)
        source_files.append(str(path))
        examples: list[Example] = []
        for row_number, row in enumerate(rows):
            text = row["Review"]
            _ensure_nonempty_text(text, f"{path}:{row_number + 2}")
            values: list[int] = []
            for aspect in aspect_names:
                raw_value = row[aspect]
                try:
                    value = int(raw_value)
                except ValueError as error:
                    raise ValueError(f"Non-integer {task_id} label {raw_value!r} in {path}") from error
                if value not in ABSA_LABEL_IDS:
                    raise ValueError(f"{task_id} label {value} outside fixed 0..3 map in {path}")
                values.append(value)
            examples.append(
                Example(
                    id=f"{task_id}:{split}:{row_number:06d}",
                    text=text,
                    label=tuple(values),
                    raw_label=None,
                    source_index=str(row_number),
                    pair_scored=tuple(True for _ in aspect_names),
                )
            )
        loaded[split] = TaskSplit(split, tuple(examples))
    return TaskDataset(
        task_id=task_id,
        splits=loaded,
        label_names=ABSA_LABEL_NAMES,
        aspect_names=aspect_names,
        slot_label_names=tuple(ABSA_LABEL_NAMES for _ in aspect_names),
        pair_scored_aspects=tuple(True for _ in aspect_names),
        source_files=tuple(source_files),
        label_policy="UIT-ABSA source map: 0=absent, 1=negative, 2=neutral, 3=positive",
    )


def _cap_split(task: TaskDataset, limit: int) -> TaskDataset:
    if limit <= 0:
        raise ValueError("Smoke limit must be positive")
    capped = {name: TaskSplit(name, split.examples[:limit]) for name, split in task.splits.items()}
    return TaskDataset(
        task_id=task.task_id,
        splits=capped,
        label_names=task.label_names,
        aspect_names=task.aspect_names,
        slot_label_names=task.slot_label_names,
        pair_scored_aspects=task.pair_scored_aspects,
        source_files=task.source_files,
        label_policy=task.label_policy,
    )


def load_task(
    task_id: str,
    repo_root: Path | str | None = None,
    *,
    include_test: bool = False,
    smoke: bool = False,
    smoke_limit: int = 24,
) -> TaskDataset:
    """Load one of the five frozen task IDs.

    ``include_test`` is false by default so screening cannot open test by
    accident.  ``smoke=True`` takes the first ``smoke_limit`` rows of each
    loaded split only for local pipeline checks.  Full loading never caps or
    shuffles rows.
    """

    if task_id not in TASK_IDS:
        raise ValueError(f"Unsupported task ID {task_id!r}; choose from {list(TASK_IDS)!r}")
    raw_dir = _repo_raw_dir(repo_root)
    loaders = {
        "vsfc_sentiment": lambda directory: _load_vsfc_sentiment(directory, include_test),
        "vsfc_topic": lambda directory: _load_vsfc_topic(directory, include_test),
        "visfd": lambda directory: _load_visfd(directory, include_test),
        "absa_restaurant": lambda directory: _load_absa(
            directory,
            "absa_restaurant",
            "UIT-ABSA-Restaurant",
            RESTAURANT_ASPECT_NAMES,
            include_test,
        ),
        "absa_hotel": lambda directory: _load_absa(
            directory,
            "absa_hotel",
            "UIT-ABSA-Hotel",
            HOTEL_ASPECT_NAMES,
            include_test,
        ),
    }
    task = loaders[task_id](raw_dir)
    expected = EXPECTED_COUNTS[task_id]
    actual = {name: len(split) for name, split in task.splits.items()}
    expected_loaded = {
        name: count for name, count in expected.items() if name in actual
    }
    if not smoke and actual != expected_loaded:
        raise ValueError(
            f"{task_id} split counts differ from frozen protocol: expected {expected}, got {actual}"
        )
    return _cap_split(task, smoke_limit) if smoke else task


def load_dataset(
    dataset: str,
    repo_root: Path | str | None = None,
    *,
    include_test: bool = False,
    smoke: bool = False,
) -> TaskDataset:
    """Compatibility alias used by the historical training scripts."""

    aliases = {
        "vsfc": "vsfc_sentiment",
        "vsfc_sentiment": "vsfc_sentiment",
        "vsfc_topic": "vsfc_topic",
        "visfd": "visfd",
        "absa_restaurant": "absa_restaurant",
        "absa_hotel": "absa_hotel",
    }
    if dataset not in aliases:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {sorted(aliases)}")
    return load_task(aliases[dataset], repo_root, include_test=include_test, smoke=smoke)


def sha256_file(path: Path | str) -> str:
    """Hash a source file for audit artifacts without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def example_label_key(example: Example) -> tuple[int, ...] | tuple[int]:
    """Canonical hashable label representation for duplicate audits."""

    if isinstance(example.label, tuple):
        return example.label
    return (example.label,)
