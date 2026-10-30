"""Pure-Python metrics with explicit fixed-label and ABSA scopes.

The BamiBERT paper reports F1 for ABSA but does not publish the complete
evaluator or averaging convention.  These functions therefore expose the
measured quantities with explicit names.  They do not return a generic
``classification_f1`` that could silently be mistaken for the paper value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(tp: int, predicted: int, gold: int) -> dict[str, float]:
    precision = _ratio(tp, predicted)
    recall = _ratio(tp, gold)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * tp, predicted + gold),
    }


def _validate_scalar_labels(values: Sequence[int], labels: Sequence[int], name: str) -> None:
    allowed = set(labels)
    if not allowed:
        raise ValueError("At least one fixed label ID is required")
    unknown = sorted({int(value) for value in values}.difference(allowed))
    if unknown:
        raise ValueError(f"{name} contains labels outside the fixed set {tuple(labels)!r}: {unknown}")


def classification_metrics(
    gold: Sequence[int],
    predicted: Sequence[int],
    labels: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Accuracy and macro F1 over an explicit fixed label set.

    A class with no support or no predicted examples contributes zero to the
    macro average, matching the usual ``zero_division=0`` convention.  The
    function never discovers the class set from ``gold`` or ``predicted``.
    """

    if len(gold) != len(predicted):
        raise ValueError(f"gold/predicted length mismatch: {len(gold)} != {len(predicted)}")
    fixed_labels = tuple(int(label) for label in labels)
    if len(set(fixed_labels)) != len(fixed_labels):
        raise ValueError(f"Duplicate fixed labels: {fixed_labels!r}")
    _validate_scalar_labels(gold, fixed_labels, "gold")
    _validate_scalar_labels(predicted, fixed_labels, "predicted")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    for label in fixed_labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        gold_count = sum(1 for value in gold if value == label)
        predicted_count = sum(1 for value in predicted if value == label)
        fp = predicted_count - tp
        fn = gold_count - tp
        score = _prf(tp, predicted_count, gold_count)
        f1_values.append(score["f1"])
        precision_values.append(score["precision"])
        recall_values.append(score["recall"])
        per_class[str(label)] = {
            "support": gold_count,
            "predicted": predicted_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **score,
        }
    return {
        "n": len(gold),
        "labels": list(fixed_labels),
        "accuracy": _ratio(sum(g == p for g, p in zip(gold, predicted)), len(gold)),
        "macro_precision": sum(precision_values) / len(precision_values),
        "macro_recall": sum(recall_values) / len(recall_values),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
    }


def _validate_absa_matrix(
    matrix: Sequence[Sequence[int]], name: str, aspect_count: int, allowed: set[int]
) -> None:
    for row_index, row in enumerate(matrix):
        if len(row) != aspect_count:
            raise ValueError(
                f"{name}[{row_index}] has {len(row)} aspects; expected {aspect_count}"
            )
        unknown = sorted(set(int(value) for value in row).difference(allowed))
        if unknown:
            raise ValueError(f"{name}[{row_index}] has labels outside {sorted(allowed)}: {unknown}")


def _absa_counts(
    gold: Sequence[Sequence[int]],
    predicted: Sequence[Sequence[int]],
    aspect_index: int | None,
    pair_scored: bool,
) -> dict[str, int]:
    gold_values = (
        [row[aspect_index] for row in gold]
        if aspect_index is not None
        else [value for row in gold for value in row]
    )
    predicted_values = (
        [row[aspect_index] for row in predicted]
        if aspect_index is not None
        else [value for row in predicted for value in row]
    )
    gold_present = [value > 0 for value in gold_values]
    predicted_present = [value > 0 for value in predicted_values]
    detection_tp = sum(g and p for g, p in zip(gold_present, predicted_present))
    if pair_scored:
        pair_tp = sum(
            g and p and gold_value == predicted_value
            for g, p, gold_value, predicted_value in zip(
                gold_present, predicted_present, gold_values, predicted_values
            )
        )
        gold_count = sum(gold_present)
        predicted_count = sum(predicted_present)
    else:
        pair_tp = 0
        gold_count = 0
        predicted_count = 0
    return {
        "detection_tp": int(detection_tp),
        "pair_tp": int(pair_tp),
        "gold_count": int(gold_count),
        "predicted_count": int(predicted_count),
    }


def _aggregate_counts(counts: Sequence[dict[str, int]]) -> dict[str, int]:
    return {
        key: sum(item[key] for item in counts)
        for key in ("detection_tp", "pair_tp", "gold_count", "predicted_count")
    }


def absa_metrics(
    gold: Sequence[Sequence[int]],
    predicted: Sequence[Sequence[int]],
    aspect_names: Sequence[str] | None = None,
    pair_scored_aspects: Sequence[bool] | None = None,
    *,
    allowed_labels: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Return explicit detection and aspect-polarity pair metrics.

    Detection treats every positive code (``>0``) as an aspect.  Pair
    classification requires the exact non-zero code at the same aspect slot;
    it is end-to-end because an absent/present mistake is also a pair error.
    Micro scores pool cells before computing P/R/F1.  Macro scores average the
    per-aspect P/R/F1, with OTHERS excluded from pair macro/micro when its mask
    entry is false.  No flattened multiclass macro F1 is emitted.
    """

    if len(gold) != len(predicted):
        raise ValueError(f"gold/predicted row mismatch: {len(gold)} != {len(predicted)}")
    if not gold:
        raise ValueError("ABSA metric input cannot be empty")
    aspect_count = len(gold[0])
    if aspect_count == 0:
        raise ValueError("ABSA metric input must contain at least one aspect")
    names = tuple(aspect_names or (f"aspect_{i}" for i in range(aspect_count)))
    if len(names) != aspect_count:
        raise ValueError(f"aspect_names has {len(names)} entries; expected {aspect_count}")
    if len(set(names)) != aspect_count:
        raise ValueError("aspect_names must be unique")
    if pair_scored_aspects is None:
        pair_mask = tuple(True for _ in range(aspect_count))
    else:
        pair_mask = tuple(bool(value) for value in pair_scored_aspects)
        if len(pair_mask) != aspect_count:
            raise ValueError("pair_scored_aspects length does not match aspect count")
    allowed = {int(value) for value in allowed_labels}
    _validate_absa_matrix(gold, "gold", aspect_count, allowed)
    _validate_absa_matrix(predicted, "predicted", aspect_count, allowed)

    per_aspect: dict[str, Any] = {}
    aspect_counts: list[dict[str, int]] = []
    pair_aspect_counts: list[dict[str, int]] = []
    for index, name in enumerate(names):
        counts = _absa_counts(gold, predicted, index, pair_mask[index])
        aspect_counts.append(counts)
        if pair_mask[index]:
            pair_aspect_counts.append(counts)
        detection = _prf(
            counts["detection_tp"],
            sum(row[index] > 0 for row in predicted),
            sum(row[index] > 0 for row in gold),
        )
        pair = _prf(
            counts["pair_tp"],
            counts["predicted_count"],
            counts["gold_count"],
        )
        per_aspect[name] = {
            "pair_scored": pair_mask[index],
            "detection": detection,
            "pair": pair if pair_mask[index] else None,
            "detection_tp": counts["detection_tp"],
            "gold_present": sum(row[index] > 0 for row in gold),
            "predicted_present": sum(row[index] > 0 for row in predicted),
        }

    detection_micro_counts = _aggregate_counts(aspect_counts)
    # Pool presence denominators directly; pair counts intentionally exclude
    # detection-only slots such as ViSFD OTHERS.
    detection_micro_predicted = sum(
        value > 0 for row in predicted for value in row
    )
    detection_micro_gold = sum(value > 0 for row in gold for value in row)
    detection_micro = _prf(
        detection_micro_counts["detection_tp"],
        detection_micro_predicted,
        detection_micro_gold,
    )
    detection_macro_values = [
        _prf(
            counts["detection_tp"],
            sum(row[index] > 0 for row in predicted),
            sum(row[index] > 0 for row in gold),
        )
        for index, counts in enumerate(aspect_counts)
    ]
    pair_micro_counts = _aggregate_counts(pair_aspect_counts)
    pair_micro = _prf(
        pair_micro_counts["pair_tp"],
        pair_micro_counts["predicted_count"],
        pair_micro_counts["gold_count"],
    )
    pair_macro_values = [
        _prf(counts["pair_tp"], counts["predicted_count"], counts["gold_count"])
        for counts in pair_aspect_counts
    ]

    def average(values: Sequence[dict[str, float]], key: str) -> float:
        return sum(value[key] for value in values) / len(values) if values else 0.0

    result: dict[str, Any] = {
        "n": len(gold),
        "aspect_count": aspect_count,
        "aspect_names": list(names),
        "pair_scored_aspects": list(pair_mask),
        "metric_scope": (
            "detection=all positive aspect slots; pair=end-to-end exact aspect-polarity "
            "slots over pair_scored_aspects; no flattened multiclass macro"
        ),
        "detection_micro_precision": detection_micro["precision"],
        "detection_micro_recall": detection_micro["recall"],
        "detection_micro_f1": detection_micro["f1"],
        "detection_macro_precision": average(detection_macro_values, "precision"),
        "detection_macro_recall": average(detection_macro_values, "recall"),
        "detection_macro_f1": average(detection_macro_values, "f1"),
        "pair_micro_precision": pair_micro["precision"],
        "pair_micro_recall": pair_micro["recall"],
        "pair_micro_f1": pair_micro["f1"],
        "pair_macro_precision": average(pair_macro_values, "precision"),
        "pair_macro_recall": average(pair_macro_values, "recall"),
        "pair_macro_f1": average(pair_macro_values, "f1"),
        # Explicit long-form aliases ease integration with code that calls the
        # second subtask "classification" without hiding its pair semantics.
        "classification_pair_micro_f1": pair_micro["f1"],
        "classification_pair_macro_f1": average(pair_macro_values, "f1"),
        "per_aspect": per_aspect,
    }
    return result


def evaluate_task(
    *,
    task_id: str,
    gold: Sequence[Any],
    predictions: Sequence[Any],
    aspect_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    aspect_class_counts: Sequence[int] | None = None,
    split: str = "validation",
) -> dict[str, Any]:
    """Evaluate one runner task and expose its selection metric explicitly.

    The runner passes model-space labels.  For ViSFD, the binary OTHERS model
    slot is converted back to raw code 4 before metrics; every other task is
    evaluated using its fixed source IDs.  ``split`` is recorded for audit and
    does not alter metric formulas.
    """

    if len(gold) != len(predictions):
        raise ValueError(f"gold/predictions length mismatch: {len(gold)} != {len(predictions)}")
    if task_id == "vsfc_sentiment":
        result = classification_metrics(
            [int(value) for value in gold],
            [int(value) for value in predictions],
            labels=(0, 1, 2),
        )
        result.update(
            {
                "task_id": task_id,
                "split": split,
                "selection_f1": result["macro_f1"],
                "selection_metric": "fixed_label_macro_f1",
            }
        )
        return result
    if task_id == "vsfc_topic":
        result = classification_metrics(
            [int(value) for value in gold],
            [int(value) for value in predictions],
            labels=(0, 1, 2, 3),
        )
        result.update(
            {
                "task_id": task_id,
                "split": split,
                "selection_f1": result["macro_f1"],
                "selection_metric": "fixed_label_macro_f1",
            }
        )
        return result
    if task_id not in {"visfd", "absa_restaurant", "absa_hotel"}:
        raise ValueError(f"Unsupported task ID {task_id!r}")
    if not gold:
        raise ValueError("ABSA evaluation requires at least one row")
    resolved_aspects = tuple(aspect_names or ())
    if not resolved_aspects:
        raise ValueError(f"{task_id} evaluation requires aspect_names")
    gold_rows = [tuple(int(value) for value in row) for row in gold]
    predicted_rows = [tuple(int(value) for value in row) for row in predictions]
    if task_id == "visfd":
        # The model head is 4-class for ten polarity slots and binary for the
        # OTHERS slot.  A raw code-4 prediction is also accepted for locked
        # evaluator callers that already performed the inverse mapping.
        others_index = resolved_aspects.index("OTHERS")
        normalized: list[tuple[int, ...]] = []
        for row in predicted_rows:
            if len(row) != len(resolved_aspects):
                raise ValueError("ViSFD prediction row width does not match aspect_names")
            values = list(row)
            others = values[others_index]
            if others == 1:
                values[others_index] = 4
            elif others not in (0, 4):
                raise ValueError(f"ViSFD OTHERS prediction must be model 0/1 or raw 0/4, got {others}")
            normalized.append(tuple(values))
        predicted_rows = normalized
        pair_mask = tuple(name != "OTHERS" for name in resolved_aspects)
    else:
        pair_mask = tuple(True for _ in resolved_aspects)
    result = absa_metrics(
        gold_rows,
        predicted_rows,
        aspect_names=resolved_aspects,
        pair_scored_aspects=pair_mask,
    )
    result.update(
        {
            "task_id": task_id,
            "split": split,
            "selection_f1": result["pair_micro_f1"],
            "selection_metric": "end_to_end_pair_micro_f1",
        }
    )
    return result
