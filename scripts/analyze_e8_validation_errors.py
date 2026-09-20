#!/usr/bin/env python3
"""Read-only E8 validation error analysis for ViSFD and Restaurant.

The analysis consumes saved validation predictions from E8, the selected E2
noise-consistency student, and the validation-selected E2 teacher.  It loads
the train and validation contracts with ``include_test=False`` and never
opens a checkpoint or a test artifact.  ViSFD OTHERS is mapped from model
0/1 to semantic raw 0/4 for detection, then excluded from all polarity and
pair counts.

Seed 42 is required.  If confirmation artifacts for E8 seeds 43/44 are
present with completed metadata and all 30 epoch records, they are included
automatically in the E8-vs-NC consistency section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goibert.data import load_task  # noqa: E402
from goibert.metrics import evaluate_task  # noqa: E402


TASKS = ("visfd", "absa_restaurant")
E2_NC_COEFFICIENT = {"visfd": 0.5, "absa_restaurant": 0.1}
SEEDS = (42, 43, 44)
POLARITY_LABELS = (1, 2, 3)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def as_int_row(value: Any, *, task_id: str, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{task_id}: {field} must be a list")
    row = tuple(int(item) for item in value)
    if task_id == "visfd":
        if len(row) != 11:
            raise ValueError(f"{task_id}: {field} width {len(row)} != 11")
        if any(item not in (0, 1, 2, 3) for index, item in enumerate(row) if index != 5):
            raise ValueError(f"{task_id}: invalid polarity code in {field}")
        if row[5] not in (0, 1, 4):
            raise ValueError(f"{task_id}: invalid OTHERS code {row[5]} in {field}")
    return row


def semantic_row(task_id: str, value: Any, *, field: str) -> tuple[int, ...]:
    """Return source-space semantics; ViSFD OTHERS is presence code 4."""

    row = list(as_int_row(value, task_id=task_id, field=field))
    if task_id == "visfd":
        row[5] = 4 if row[5] > 0 else 0
    return tuple(row)


def empty_counts() -> dict[str, int]:
    return {
        "cells": 0,
        "gold_present": 0,
        "predicted_present": 0,
        "detection_tp": 0,
        "missed_aspect": 0,
        "spurious_aspect": 0,
        "both_present": 0,
        "absent_correct": 0,
        "pair_both_present": 0,
        "polarity_correct": 0,
        "polarity_error": 0,
        "pair_tp": 0,
        "pair_gold": 0,
        "pair_predicted": 0,
        "pair_fp": 0,
        "pair_fn": 0,
    }


def add_counts(counts: dict[str, int], gold: int, prediction: int, pair_scored: bool) -> None:
    gold_present = gold > 0
    predicted_present = prediction > 0
    counts["cells"] += 1
    counts["gold_present"] += int(gold_present)
    counts["predicted_present"] += int(predicted_present)
    if gold_present and predicted_present:
        counts["detection_tp"] += 1
        counts["both_present"] += 1
        if pair_scored:
            counts["pair_both_present"] += 1
            counts["pair_gold"] += 1
            counts["pair_predicted"] += 1
            if gold == prediction:
                counts["polarity_correct"] += 1
                counts["pair_tp"] += 1
            else:
                counts["polarity_error"] += 1
    elif gold_present:
        counts["missed_aspect"] += 1
        if pair_scored:
            counts["pair_gold"] += 1
    elif predicted_present:
        counts["spurious_aspect"] += 1
        if pair_scored:
            counts["pair_predicted"] += 1
    else:
        counts["absent_correct"] += 1
    if pair_scored:
        counts["pair_fp"] = counts["pair_predicted"] - counts["pair_tp"]
        counts["pair_fn"] = counts["pair_gold"] - counts["pair_tp"]


def merge_counts(items: Sequence[Mapping[str, int]]) -> dict[str, int]:
    result = empty_counts()
    for item in items:
        for key in result:
            result[key] += int(item.get(key, 0))
    return result


def prf(tp: int, predicted: int, support: int) -> dict[str, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / support if support else 0.0
    f1 = 2.0 * tp / (predicted + support) if predicted + support else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def with_rates(counts: Mapping[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {key: int(value) for key, value in counts.items()}
    result.update(
        {
            "detection_precision": prf(
                counts["detection_tp"], counts["predicted_present"], counts["gold_present"]
            )["precision"],
            "detection_recall": prf(
                counts["detection_tp"], counts["predicted_present"], counts["gold_present"]
            )["recall"],
            "detection_micro_f1": prf(
                counts["detection_tp"], counts["predicted_present"], counts["gold_present"]
            )["f1"],
            "pair_precision": prf(
                counts["pair_tp"], counts["pair_predicted"], counts["pair_gold"]
            )["precision"],
            "pair_recall": prf(
                counts["pair_tp"], counts["pair_predicted"], counts["pair_gold"]
            )["recall"],
            "pair_micro_f1": prf(
                counts["pair_tp"], counts["pair_predicted"], counts["pair_gold"]
            )["f1"],
            "polarity_accuracy_when_both_present": (
                counts["polarity_correct"] / counts["pair_both_present"]
                if counts["pair_both_present"]
                else 0.0
            ),
        }
    )
    return result


def class_metrics(gold: Sequence[int], prediction: Sequence[int], label: int) -> dict[str, Any]:
    support = sum(value == label for value in gold)
    predicted = sum(value == label for value in prediction)
    tp = sum(left == label and right == label for left, right in zip(gold, prediction))
    return {
        "support": support,
        "predicted": predicted,
        "tp": tp,
        "fp": predicted - tp,
        "fn": support - tp,
        **prf(tp, predicted, support),
    }


def train_frequency(task_id: str, dataset: Any) -> dict[str, Any]:
    """Compute all rarity/support fields from the complete train split only."""

    names = list(dataset.aspect_names)
    rows = [semantic_row(task_id, row, field="train label") for row in dataset.labels_for_model("train")]
    per_aspect: dict[str, Any] = {}
    presence_supports: list[int] = []
    task_polarity_support = Counter()
    polarity_names = (
        {1: "positive", 2: "negative", 3: "neutral"}
        if task_id == "visfd"
        else {1: "negative", 2: "neutral", 3: "positive"}
    )
    for index, name in enumerate(names):
        support = Counter(row[index] for row in rows)
        presence_supports.append(len(rows) - support[0])
        if name != "OTHERS":
            task_polarity_support.update({label: support[label] for label in POLARITY_LABELS})
        per_aspect[name] = {
            "train_rows": len(rows),
            "presence_support": len(rows) - support[0],
            "absent_support": support[0],
            "presence_fraction": (len(rows) - support[0]) / len(rows),
            "class_support": {str(label): int(support[label]) for label in sorted(support)},
            "polarity_support": (
                {str(label): int(support[label]) for label in POLARITY_LABELS}
                if name != "OTHERS"
                else None
            ),
        }
    task_polarity_values = [
        (int(value), polarity_names[label])
        for label, value in sorted(task_polarity_support.items())
    ]
    min_presence = min(presence_supports) if presence_supports else 0
    for name, value in per_aspect.items():
        if value["polarity_support"] is not None:
            smallest = min(value["polarity_support"].values())
            value["rarest_polarity_classes_in_slot"] = [
                label for label, count in value["polarity_support"].items() if count == smallest
            ]
            value["rarest_polarity_support_in_slot"] = smallest
        else:
            value["rarest_polarity_classes_in_slot"] = []
            value["rarest_polarity_support_in_slot"] = None
        value["slot_presence_frequency_rank"] = 1 + sum(
            other["presence_support"] > value["presence_support"] for other in per_aspect.values()
        )
        value["rarest_slot_by_presence"] = value["presence_support"] == min_presence
    return {
        "source": "full_train_only",
        "n_train": len(rows),
        "per_aspect": per_aspect,
        "aggregate_polarity_support": {
            polarity_names[label]: int(task_polarity_support[label])
            for label in POLARITY_LABELS
        },
        "aggregate_polarity_support_ascending": [
            {"label": label, "support": value} for value, label in sorted(task_polarity_values)
        ],
        "rarest_slot_presence_support": min_presence,
    }


def load_rows(
    task_id: str,
    dataset: Any,
    artifact_dir: Path,
    expected_seed: int | None = None,
    *,
    require_metadata: bool = True,
) -> list[dict[str, Any]]:
    metadata_path = artifact_dir / "metadata.json"
    prediction_path = artifact_dir / "validation_predictions.json"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Incomplete validation artifact: {artifact_dir}")
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("status") != "completed":
            raise ValueError(f"Artifact is not completed: {artifact_dir}")
        if metadata.get("include_test") is not False:
            raise ValueError(f"Artifact is not validation-only: {artifact_dir}")
        if metadata.get("task_id") != task_id:
            raise ValueError(f"Task mismatch in {metadata_path}")
        if expected_seed is not None and int(metadata.get("seed", -1)) != expected_seed:
            raise ValueError(f"Seed mismatch in {metadata_path}")
    elif require_metadata:
        raise FileNotFoundError(f"Missing metadata for validation artifact: {artifact_dir}")
    prediction_hash_before = sha256_file(prediction_path)
    payload = read_json(prediction_path)
    if sha256_file(prediction_path) != prediction_hash_before:
        raise RuntimeError(f"Validation predictions changed while being read: {prediction_path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No rows in {prediction_path}")
    examples = list(dataset.validation.examples)
    # Frozen validation artifacts have existed in both model-space (OTHERS=1)
    # and source-space (OTHERS=4) serialization.  Compare semantic rows so
    # either representation is accepted while IDs and all other labels remain
    # exact.
    expected_gold = list(dataset.validation.labels)
    if len(rows) != len(examples):
        raise ValueError(f"{task_id}: {artifact_dir} row count does not match validation")
    output: list[dict[str, Any]] = []
    for index, (row, example, expected) in enumerate(zip(rows, examples, expected_gold)):
        if not isinstance(row, dict) or not {"id", "gold", "prediction"}.issubset(row):
            raise ValueError(f"Malformed row {index} in {prediction_path}")
        if row["id"] != example.id:
            raise ValueError(f"{task_id}: validation ID mismatch at row {index}")
        serialized_gold = as_int_row(row["gold"], task_id=task_id, field="gold")
        expected_tuple = semantic_row(task_id, expected, field="source gold")
        if semantic_row(task_id, serialized_gold, field="gold") != expected_tuple:
            raise ValueError(f"{task_id}: serialized gold mismatch at row {index}")
        output.append(
            {
                "id": example.id,
                "gold": semantic_row(task_id, row["gold"], field="gold"),
                "prediction": semantic_row(task_id, row["prediction"], field="prediction"),
            }
        )
    return output


def selected_teacher_path(task_id: str) -> Path:
    selection_path = REPO_ROOT / "artifacts/e2/.orchestrator/stage0_teacher_selection.json"
    if selection_path.is_file():
        selection = read_json(selection_path)
        selected = selection.get("selected", {}).get(task_id, {})
        artifact_dir = selected.get("artifact_dir") if isinstance(selected, dict) else None
        if artifact_dir:
            return Path(str(artifact_dir)).resolve()
    # This fallback is the input contract recorded by the earlier teacher-gap
    # diagnostic, so the analysis remains usable if the selection file moves.
    gap_path = REPO_ROOT / "artifacts/diagnostics/teacher_gap.json"
    gap = read_json(gap_path)
    record = gap.get("input_prediction_files", {}).get(f"{task_id}.teacher")
    if isinstance(record, dict):
        return Path(str(record["path"])).resolve().parent
    if isinstance(record, str):
        return Path(record).resolve().parent
    raise FileNotFoundError(f"No selected teacher path for {task_id}")


def e8_path(task_id: str, seed: int) -> Path:
    return REPO_ROOT / "artifacts/e8" / task_id / f"seed_{seed}"


def nc_path(task_id: str, seed: int) -> Path:
    method_root = REPO_ROOT / "artifacts/e2" / ("stageA" if seed == 42 else "stageB")
    return method_root / "nc" / task_id / f"coef_{str(E2_NC_COEFFICIENT[task_id]).replace('.', 'p')}" / f"seed_{seed}"


def e8_ready(artifact_dir: Path) -> bool:
    """Require a completed metadata record and all 30 epoch records."""

    metadata_path = artifact_dir / "metadata.json"
    curves_path = artifact_dir / "epoch_curves.json"
    prediction_path = artifact_dir / "validation_predictions.json"
    if not metadata_path.is_file() or not curves_path.is_file() or not prediction_path.is_file():
        return False
    metadata = read_json(metadata_path)
    if metadata.get("status") != "completed" or metadata.get("include_test") is not False:
        return False
    epochs = read_json(curves_path).get("epochs")
    return isinstance(epochs, list) and len(epochs) == 30


def system_analysis(task_id: str, dataset: Any, rows: Sequence[Mapping[str, Any]], train: Mapping[str, Any]) -> dict[str, Any]:
    names = list(dataset.aspect_names)
    pair_mask = tuple(bool(value) for value in dataset.pair_scored_aspects)
    per_aspect: dict[str, Any] = {}
    counts_by_aspect: list[dict[str, int]] = []
    for index, name in enumerate(names):
        gold = [int(row["gold"][index]) for row in rows]
        prediction = [int(row["prediction"][index]) for row in rows]
        counts = empty_counts()
        for left, right in zip(gold, prediction):
            add_counts(counts, left, right, pair_mask[index])
        counts_by_aspect.append(counts)
        polarity_names = (
            {"1": "positive", "2": "negative", "3": "neutral"}
            if task_id == "visfd"
            else {"1": "negative", "2": "neutral", "3": "positive"}
        )
        per_aspect[name] = {
            "pair_scored": pair_mask[index],
            "errors": with_rates(counts),
            "detection_f1": prf(
                counts["detection_tp"], counts["predicted_present"], counts["gold_present"]
            )["f1"],
            "pair_f1": (
                prf(counts["pair_tp"], counts["pair_predicted"], counts["pair_gold"])["f1"]
                if pair_mask[index]
                else None
            ),
            "polarity_support_and_f1": (
                {
                    polarity_names[str(label)]: {
                        "label_id": label,
                        **class_metrics(gold, prediction, label),
                    }
                    for label in POLARITY_LABELS
                }
                if pair_mask[index]
                else None
            ),
            "train_frequency": dict(train["per_aspect"][name]),
        }
    aggregate = with_rates(merge_counts(counts_by_aspect))
    raw_gold = [tuple(row["gold"]) for row in rows]
    raw_prediction = [tuple(row["prediction"]) for row in rows]
    audit = evaluate_task(
        task_id=task_id,
        gold=raw_gold,
        predictions=raw_prediction,
        aspect_names=names,
        aspect_class_counts=list(dataset.aspect_class_counts),
        split="validation",
    )
    metrics = {
        "selection_f1": float(audit["selection_f1"]),
        "selection_metric": audit["selection_metric"],
        "detection_micro_precision": float(audit["detection_micro_precision"]),
        "detection_micro_recall": float(audit["detection_micro_recall"]),
        "detection_micro_f1": float(audit["detection_micro_f1"]),
        "pair_micro_precision": float(audit["pair_micro_precision"]),
        "pair_micro_recall": float(audit["pair_micro_recall"]),
        "pair_micro_f1": float(audit["pair_micro_f1"]),
    }
    return {
        "n_validation": len(rows),
        "metrics": metrics,
        "error_decomposition": aggregate,
        "per_aspect": per_aspect,
    }


def teacher_transition(task_id: str, dataset: Any, teacher: Sequence[Mapping[str, Any]], e8: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = list(dataset.aspect_names)
    pair_mask = tuple(bool(value) for value in dataset.pair_scored_aspects)
    detection_total = Counter()
    pair_total = Counter()
    polarity_total = Counter()
    per_aspect: dict[str, Any] = {}
    for index, name in enumerate(names):
        detection = Counter()
        positive_pair = Counter()
        polarity = Counter()
        teacher_pair_tp = 0
        e8_pair_tp = 0
        for teacher_row, e8_row in zip(teacher, e8):
            gold = int(teacher_row["gold"][index])
            teacher_prediction = int(teacher_row["prediction"][index])
            e8_prediction = int(e8_row["prediction"][index])
            teacher_detection_ok = (teacher_prediction > 0) == (gold > 0)
            e8_detection_ok = (e8_prediction > 0) == (gold > 0)
            if teacher_detection_ok and e8_detection_ok:
                detection["both_correct"] += 1
            elif not teacher_detection_ok and not e8_detection_ok:
                detection["both_wrong"] += 1
            elif teacher_detection_ok:
                detection["teacher_correct_e8_wrong"] += 1
            else:
                detection["teacher_wrong_e8_correct"] += 1
            if pair_mask[index] and gold > 0:
                teacher_hit = teacher_prediction == gold
                e8_hit = e8_prediction == gold
                positive_pair["eligible_gold_positive_pairs"] += 1
                if teacher_hit:
                    teacher_pair_tp += 1
                if e8_hit:
                    e8_pair_tp += 1
                if teacher_hit and e8_hit:
                    positive_pair["both_hits"] += 1
                elif not teacher_hit and not e8_hit:
                    positive_pair["neither_hit"] += 1
                elif teacher_hit:
                    positive_pair["teacher_hit_e8_miss"] += 1
                else:
                    positive_pair["teacher_miss_e8_hit"] += 1
                if teacher_prediction > 0 and e8_prediction > 0:
                    polarity["eligible_both_present"] += 1
                    teacher_polarity_ok = teacher_prediction == gold
                    e8_polarity_ok = e8_prediction == gold
                    if teacher_polarity_ok and e8_polarity_ok:
                        polarity["both_correct"] += 1
                    elif not teacher_polarity_ok and not e8_polarity_ok:
                        polarity["both_wrong"] += 1
                    elif teacher_polarity_ok:
                        polarity["teacher_correct_e8_wrong"] += 1
                    else:
                        polarity["teacher_wrong_e8_correct"] += 1
        positive_pair["teacher_pair_tp"] = teacher_pair_tp
        positive_pair["e8_pair_tp"] = e8_pair_tp
        positive_pair["teacher_pair_tp_gap"] = teacher_pair_tp - e8_pair_tp
        for key, value in detection.items():
            detection_total[key] += value
        for key, value in positive_pair.items():
            pair_total[key] += value
        for key, value in polarity.items():
            polarity_total[key] += value
        per_aspect[name] = {
            "pair_scored": pair_mask[index],
            "detection": dict(detection),
            "positive_pair": dict(positive_pair) if pair_mask[index] else {"excluded_others_slot": True},
            "conditional_polarity": dict(polarity) if pair_mask[index] else {"excluded_others_slot": True},
        }
    pair_total["teacher_pair_tp_gap"] = pair_total["teacher_pair_tp"] - pair_total["e8_pair_tp"]
    return {
        "definitions": {
            "detection": "all aspect slots; correct iff presence (label > 0) agrees, including ViSFD OTHERS",
            "positive_pair": "only pair-scored slots with gold present; hit iff exact gold polarity is predicted",
            "conditional_polarity": "pair-scored gold-present slots where both teacher and E8 predict present; exact polarity correctness",
        },
        "aggregate": {
            "detection": dict(detection_total),
            "positive_pair": dict(pair_total),
            "conditional_polarity": dict(polarity_total),
        },
        "per_aspect": per_aspect,
    }


def teacher_gap_contributors(
    teacher_analysis: Mapping[str, Any],
    e8_analysis: Mapping[str, Any],
    transitions: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aspect, teacher_row in teacher_analysis["per_aspect"].items():
        e8_row = e8_analysis["per_aspect"][aspect]
        if not teacher_row["pair_scored"]:
            continue
        transition = transitions["per_aspect"][aspect]["positive_pair"]
        tc = teacher_row["errors"]
        ec = e8_row["errors"]
        # Pair FP/FN include spurious/missed errors and one of each for every
        # wrong-polarity cell.  Keeping all terms visible prevents a TP-only
        # story from being mistaken for a causal attribution.
        rows.append(
            {
                "aspect": aspect,
                "teacher_pair_tp": int(tc["pair_tp"]),
                "e8_pair_tp": int(ec["pair_tp"]),
                "teacher_pair_tp_gap": int(tc["pair_tp"] - ec["pair_tp"]),
                "teacher_hit_e8_miss": int(transition.get("teacher_hit_e8_miss", 0)),
                "teacher_miss_e8_hit": int(transition.get("teacher_miss_e8_hit", 0)),
                "teacher_errors": {
                    key: int(tc.get(key, 0))
                    for key in ("missed_aspect", "spurious_aspect", "polarity_error")
                },
                "e8_errors": {
                    key: int(ec.get(key, 0)) for key in ("missed_aspect", "spurious_aspect", "polarity_error")
                },
                "teacher_pair_fp": int(tc.get("pair_fp", 0)),
                "teacher_pair_fn": int(tc.get("pair_fn", 0)),
                "e8_pair_fp": int(ec.get("pair_fp", 0)),
                "e8_pair_fn": int(ec.get("pair_fn", 0)),
            }
        )
    return sorted(rows, key=lambda row: (-row["teacher_pair_tp_gap"], row["aspect"]))


def compare_e8_nc(e8: Mapping[str, Any], nc: Mapping[str, Any]) -> dict[str, Any]:
    metric_keys = ("selection_f1", "detection_micro_f1", "pair_micro_f1", "pair_micro_precision", "pair_micro_recall")
    error_keys = ("missed_aspect", "spurious_aspect", "polarity_error", "pair_tp", "pair_fp", "pair_fn")
    e8_metrics = e8["metrics"]
    nc_metrics = nc["metrics"]
    e8_errors = e8["error_decomposition"]
    nc_errors = nc["error_decomposition"]
    by_aspect = []
    for aspect, e8_row in e8["per_aspect"].items():
        nc_row = nc["per_aspect"][aspect]
        by_aspect.append(
            {
                "aspect": aspect,
                "pair_f1_e8": e8_row["pair_f1"],
                "pair_f1_nc": nc_row["pair_f1"],
                "delta_pair_f1_e8_minus_nc": (
                    e8_row["pair_f1"] - nc_row["pair_f1"]
                    if e8_row["pair_f1"] is not None and nc_row["pair_f1"] is not None
                    else None
                ),
                "error_delta_e8_minus_nc": {
                    key: int(e8_row["errors"].get(key, 0)) - int(nc_row["errors"].get(key, 0))
                    for key in error_keys
                },
            }
        )
    return {
        "metrics_delta_e8_minus_nc": {
            key: float(e8_metrics[key]) - float(nc_metrics[key]) for key in metric_keys
        },
        "error_delta_e8_minus_nc": {
            key: int(e8_errors[key]) - int(nc_errors[key]) for key in error_keys
        },
        "error_reduction_nc_minus_e8": {
            key: int(nc_errors[key]) - int(e8_errors[key]) for key in error_keys
        },
        "by_aspect": sorted(
            by_aspect,
            key=lambda row: (
                -(row["delta_pair_f1_e8_minus_nc"] or 0.0),
                row["aspect"],
            ),
        ),
    }


def cross_seed_summary(seed_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    seeds = sorted(int(seed) for seed in seed_rows)
    per_seed = {
        str(seed): {
            "pair_f1_e8": seed_rows[str(seed)]["e8"]["metrics"]["pair_micro_f1"],
            "pair_f1_nc": seed_rows[str(seed)]["e2_nc"]["metrics"]["pair_micro_f1"],
            "delta_pair_f1_e8_minus_nc": seed_rows[str(seed)]["comparison"]["metrics_delta_e8_minus_nc"]["pair_micro_f1"],
            "missed_delta_e8_minus_nc": seed_rows[str(seed)]["comparison"]["error_delta_e8_minus_nc"]["missed_aspect"],
            "spurious_delta_e8_minus_nc": seed_rows[str(seed)]["comparison"]["error_delta_e8_minus_nc"]["spurious_aspect"],
            "polarity_delta_e8_minus_nc": seed_rows[str(seed)]["comparison"]["error_delta_e8_minus_nc"]["polarity_error"],
            "top_pair_f1_gain_aspect": next(
                (row["aspect"] for row in seed_rows[str(seed)]["comparison"]["by_aspect"] if row["delta_pair_f1_e8_minus_nc"] is not None),
                None,
            ),
        }
        for seed in seeds
    }
    if len(seeds) < 2:
        return {"available_seeds": seeds, "per_seed": per_seed, "repeat_status": "not_available"}
    return {
        "available_seeds": seeds,
        "per_seed": per_seed,
        "repeat_status": "assessed",
        "repeat_flags": {
            "pair_f1_gain_all_available_seeds": all(value["delta_pair_f1_e8_minus_nc"] > 0 for value in per_seed.values()),
            "spurious_reduced_all_available_seeds": all(value["spurious_delta_e8_minus_nc"] < 0 for value in per_seed.values()),
            "missed_reduced_all_available_seeds": all(value["missed_delta_e8_minus_nc"] < 0 for value in per_seed.values()),
            "polarity_errors_reduced_all_available_seeds": all(value["polarity_delta_e8_minus_nc"] < 0 for value in per_seed.values()),
            "top_pair_f1_gain_aspect_values": sorted({value["top_pair_f1_gain_aspect"] for value in per_seed.values()}),
        },
    }


def build_report() -> dict[str, Any]:
    input_files: dict[str, Path] = {}
    task_reports: dict[str, Any] = {}
    for task_id in TASKS:
        dataset = load_task(task_id, REPO_ROOT, include_test=False)
        train = train_frequency(task_id, dataset)
        teacher_dir = selected_teacher_path(task_id)
        teacher_rows = load_rows(task_id, dataset, teacher_dir, expected_seed=42)
        input_files[f"{task_id}.teacher"] = teacher_dir / "validation_predictions.json"
        seed_reports: dict[str, Any] = {}
        for seed in SEEDS:
            e8_dir = e8_path(task_id, seed)
            nc_dir = nc_path(task_id, seed)
            e8_complete = e8_ready(e8_dir)
            nc_complete = (nc_dir / "validation_predictions.json").is_file()
            if not e8_complete:
                if seed == 42:
                    raise FileNotFoundError(f"Missing required E8 validation artifact: {e8_dir}")
                continue
            if not nc_complete:
                continue
            e8_rows = load_rows(
                task_id,
                dataset,
                e8_dir,
                expected_seed=seed,
                require_metadata=True,
            )
            nc_rows = load_rows(task_id, dataset, nc_dir, expected_seed=seed, require_metadata=True)
            input_files[f"{task_id}.e8.seed{seed}"] = e8_dir / "validation_predictions.json"
            input_files[f"{task_id}.e2.nc.seed{seed}"] = nc_dir / "validation_predictions.json"
            e8_analysis = system_analysis(task_id, dataset, e8_rows, train)
            nc_analysis = system_analysis(task_id, dataset, nc_rows, train)
            seed_reports[str(seed)] = {
                "e8": {"artifact_dir": str(e8_dir), **e8_analysis},
                "e2_nc": {"artifact_dir": str(nc_dir), **nc_analysis},
                "comparison": compare_e8_nc(e8_analysis, nc_analysis),
            }
            if seed == 42:
                teacher_analysis = system_analysis(task_id, dataset, teacher_rows, train)
                transitions = teacher_transition(task_id, dataset, teacher_rows, e8_rows)
                seed_reports[str(seed)]["teacher"] = {
                    "artifact_dir": str(teacher_dir),
                    **teacher_analysis,
                }
                seed_reports[str(seed)]["teacher_vs_e8"] = transitions
                seed_reports[str(seed)]["teacher_pair_gap_contributors"] = teacher_gap_contributors(
                    teacher_analysis, e8_analysis, transitions
                )
        task_reports[task_id] = {
            "selection": {
                "e8": "NC plus LLRD, seed-specific saved validation artifact",
                "e2_nc_lambda": E2_NC_COEFFICIENT[task_id],
                "teacher": "selected E2 teacher, seed 42",
            },
            "train_frequency": train,
            "per_seed": seed_reports,
            "cross_seed_consistency": cross_seed_summary(seed_reports),
        }
    return {
        "protocol": "e8-validation-error-analysis-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "test_access": False,
        "include_test": False,
        "scope": {
            "tasks": list(TASKS),
            "required_seed": 42,
            "optional_e8_confirmation_seeds": [43, 44],
            "all_comparisons_use_same_validation_ids_and_gold": True,
        },
        "definitions": {
            "detection": "presence is label > 0; ViSFD OTHERS model 0/1 maps to semantic raw 0/4 and is included",
            "pair": "positive exact aspect-polarity hit over pair_scored_aspects; ViSFD OTHERS is excluded",
            "missed_aspect": "gold present and prediction absent",
            "spurious_aspect": "gold absent and prediction present",
            "wrong_polarity": "gold and prediction both present but exact pair label differs; one pair FP and one pair FN",
            "pair_f1": "micro F1 from pooled exact positive pairs, not flattened multiclass label accuracy",
            "teacher_gap": "descriptive teacher pair TP difference and paired transitions; no causal attribution",
            "train_rarity": "support and rare-class fields are computed from full TRAIN labels only",
        },
        "source": {
            "data_loader": str(REPO_ROOT / "src/goibert/data.py"),
            "metrics": str(REPO_ROOT / "src/goibert/metrics.py"),
            "teacher_selection": str(REPO_ROOT / "artifacts/e2/.orchestrator/stage0_teacher_selection.json"),
            "e2_stage_selection": str(REPO_ROOT / "artifacts/e2/.orchestrator/stageA_selection.json"),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "input_prediction_files": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in sorted(input_files.items())
        },
        "tasks": task_reports,
        "limits": [
            "Teacher is one validation-selected E2 seed-42 run; no teacher variance is estimated.",
            "E8 and E2 coefficients/checkpoints are compared conditional on their locked validation choices.",
            "Teacher is a larger encoder fine-tuned separately; this does not reproduce the original Bami paper setup.",
            "Argmax validation rows cannot establish why an error occurred or prove a distillation bottleneck.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/e8/error_analysis",
        help="Directory for the JSON report (default: artifacts/e8/error_analysis)",
    )
    args = parser.parse_args()
    report = build_report()
    output = args.output_dir / "e8_validation_error_analysis.json"
    write_json(output, report)
    print(json.dumps({"output": str(output), "tasks": list(TASKS), "test_access": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
