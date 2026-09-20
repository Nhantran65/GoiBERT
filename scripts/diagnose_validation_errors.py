#!/usr/bin/env python3
"""Read-only error diagnosis for the locked E1/E2 validation predictions.

This script deliberately loads only the train and validation splits.  It does
not construct a model, open a checkpoint, or call either the training or test
evaluator.  The prediction files are the outputs written by the frozen E1/E2
validation loops; their row IDs and serialized gold labels are checked against
the source validation split before any counts are produced.

The main analysis is for ViSFD and UIT-ABSA Restaurant.  The other three
selected tasks are included as a compact consistency readout so the diagnosis
can be compared with the complete E2 selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from goibert.data import load_task  # noqa: E402
from goibert.metrics import evaluate_task  # noqa: E402


SEEDS = (42, 43, 44)
E1_SELECTED_LR = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}
E2_SELECTED = {
    "vsfc_sentiment": {"method": "nc", "coefficient": 0.1, "coefficient_name": "lambda"},
    "vsfc_topic": {"method": "kd", "coefficient": 0.75, "coefficient_name": "alpha"},
    "visfd": {"method": "nc", "coefficient": 0.5, "coefficient_name": "lambda"},
    "absa_restaurant": {"method": "nc", "coefficient": 0.1, "coefficient_name": "lambda"},
    "absa_hotel": {"method": "nc", "coefficient": 0.1, "coefficient_name": "lambda"},
}
TASKS = tuple(E1_SELECTED_LR)
FOCUS_TASKS = ("visfd", "absa_restaurant")
ABSA_TASKS = {"visfd", "absa_restaurant", "absa_hotel"}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_equal(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _coef_directory(coefficient: float) -> str:
    return f"coef_{str(coefficient).replace('.', 'p')}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _prediction_path(path: Path) -> Path:
    resolved = path / "validation_predictions.json"
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing validation predictions: {resolved}")
    return resolved


def _load_e1_records() -> dict[str, list[dict[str, Any]]]:
    payload = _load_json(REPO_ROOT / "artifacts" / "e1" / ".orchestrator" / "approved_records.json")
    approved = payload.get("approved")
    if not isinstance(approved, dict):
        raise ValueError("E1 approved records have no approved mapping")
    return {
        str(stage): [record for record in records if isinstance(record, dict)]
        for stage, records in approved.items()
        if isinstance(records, list)
    }


def _find_e1_artifact(task_id: str, seed: int, records: Mapping[str, Sequence[Mapping[str, Any]]]) -> Path:
    stage = "screening" if seed == 42 else "shortlist"
    selected_lr = E1_SELECTED_LR[task_id]
    candidates = []
    for record in records.get(stage, ()):
        if (
            record.get("task_id") == task_id
            and int(record.get("seed", -1)) == seed
            and _float_equal(record.get("learning_rate"), selected_lr)
        ):
            candidates.append(Path(str(record["artifact_dir"])))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one E1 {stage} artifact for {task_id} seed {seed} lr {selected_lr}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _find_e2_artifact(task_id: str, seed: int) -> Path:
    choice = E2_SELECTED[task_id]
    method = str(choice["method"])
    coefficient = float(choice["coefficient"])
    coef_dir = _coef_directory(coefficient)
    if seed == 42:
        return REPO_ROOT / "artifacts" / "e2" / "stageA" / method / task_id / coef_dir / f"seed_{seed}"
    return REPO_ROOT / "artifacts" / "e2" / "stageB" / method / task_id / coef_dir / f"seed_{seed}"


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(_prediction_path(path))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No prediction rows in {path}")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not {"id", "gold", "prediction"}.issubset(row):
            raise ValueError(f"Malformed prediction row {index} in {path}")
        normalized.append(row)
    return normalized


def _as_tuple(value: Any) -> tuple[int, ...] | int:
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    if isinstance(value, tuple):
        return tuple(int(item) for item in value)
    return int(value)


def _model_to_raw(task_id: str, row: Sequence[int]) -> tuple[int, ...]:
    values = [int(item) for item in row]
    if task_id == "visfd":
        if len(values) != 11:
            raise ValueError(f"ViSFD row width is {len(values)}, expected 11")
        others = values[5]
        if others not in (0, 1, 4):
            raise ValueError(f"Unexpected ViSFD OTHERS model value {others}")
        values[5] = 4 if others else 0
    return tuple(values)


def _raw_to_model(task_id: str, row: Sequence[int]) -> tuple[int, ...]:
    values = [int(item) for item in row]
    if task_id == "visfd":
        if len(values) != 11:
            raise ValueError(f"ViSFD row width is {len(values)}, expected 11")
        others = values[5]
        if others not in (0, 4):
            raise ValueError(f"Unexpected ViSFD OTHERS raw value {others}")
        values[5] = 1 if others == 4 else 0
    return tuple(values)


def _require_validation_alignment(task_id: str, dataset: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    examples = dataset.validation.examples
    if len(rows) != len(examples):
        raise ValueError(f"{task_id}: prediction rows {len(rows)} != validation rows {len(examples)}")
    expected_model_gold = dataset.validation.labels
    for index, (row, example, expected) in enumerate(zip(rows, examples, expected_model_gold)):
        if row["id"] != example.id:
            raise ValueError(f"{task_id}: row {index} ID mismatch {row['id']!r} != {example.id!r}")
        actual_gold = _as_tuple(row["gold"])
        expected_tuple = _as_tuple(expected)
        if actual_gold != expected_tuple:
            raise ValueError(
                f"{task_id}: row {index} serialized gold does not match model-space source labels"
            )


def _prf(tp: int, predicted: int, support: int) -> dict[str, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / support if support else 0.0
    f1 = 2 * tp / (predicted + support) if predicted + support else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _percent(value: float) -> float:
    return 100.0 * float(value)


def _length_bucket(text: str) -> str:
    count = len(text.split())
    if count <= 10:
        return "short_0_10_whitespace_tokens"
    if count <= 20:
        return "medium_11_20_whitespace_tokens"
    return "long_21plus_whitespace_tokens"


def _empty_error_counts() -> dict[str, int]:
    return {
        "cells": 0,
        "gold_present": 0,
        "predicted_present": 0,
        "detection_tp": 0,
        "missed_aspect": 0,
        "spurious_aspect": 0,
        "both_present": 0,
        "pair_both_present": 0,
        "polarity_correct": 0,
        "polarity_error": 0,
        "absent_correct": 0,
        "pair_tp": 0,
        "pair_gold": 0,
        "pair_predicted": 0,
    }


def _add_error(counts: dict[str, int], gold: int, pred: int, pair_scored: bool) -> None:
    gold_present = gold > 0
    predicted_present = pred > 0
    counts["cells"] += 1
    counts["gold_present"] += int(gold_present)
    counts["predicted_present"] += int(predicted_present)
    if gold_present and predicted_present:
        counts["detection_tp"] += 1
        counts["both_present"] += 1
        if pair_scored:
            counts["pair_both_present"] += 1
            if gold == pred:
                counts["polarity_correct"] += 1
            else:
                counts["polarity_error"] += 1
            counts["pair_gold"] += 1
            counts["pair_predicted"] += 1
            if gold == pred:
                counts["pair_tp"] += 1
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


def _merge_counts(items: Iterable[Mapping[str, int]]) -> dict[str, int]:
    result = _empty_error_counts()
    for item in items:
        for key in result:
            result[key] += int(item.get(key, 0))
    return result


def _with_rates(counts: Mapping[str, int]) -> dict[str, Any]:
    result = {key: int(value) for key, value in counts.items()}
    result.update(
        {
            "detection_precision": counts["detection_tp"] / counts["predicted_present"]
            if counts["predicted_present"]
            else 0.0,
            "detection_recall": counts["detection_tp"] / counts["gold_present"]
            if counts["gold_present"]
            else 0.0,
            "pair_f1": _prf(
                counts["pair_tp"], counts["pair_predicted"], counts["pair_gold"]
            )["f1"],
            "polarity_accuracy_when_both_present": counts["polarity_correct"]
            / counts["pair_both_present"]
            if counts["pair_both_present"]
            else 0.0,
        }
    )
    return result


def _class_metrics(gold_values: Sequence[int], pred_values: Sequence[int], labels: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in labels:
        support = sum(value == label for value in gold_values)
        predicted = sum(value == label for value in pred_values)
        tp = sum(gold == label and pred == label for gold, pred in zip(gold_values, pred_values))
        score = _prf(tp, predicted, support)
        result[str(label)] = {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "fp": predicted - tp,
            "fn": support - tp,
            **score,
        }
    return result


def _train_aspect_frequency(dataset: Any) -> dict[str, dict[str, Any]]:
    names = list(dataset.aspect_names)
    # ``labels_for_model`` already maps ViSFD OTHERS to 1; normalize it back
    # to raw code 4 so the frequency table has the source-level convention.
    train_rows = [_model_to_raw(dataset.task_id, row) for row in dataset.labels_for_model("train")]
    presence = [sum(row[index] > 0 for row in train_rows) for index in range(len(names))]
    sorted_counts = sorted(presence)
    q25 = sorted_counts[(len(sorted_counts) - 1) // 4]
    q75 = sorted_counts[(3 * (len(sorted_counts) - 1)) // 4]
    result: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        classes = Counter(row[index] for row in train_rows)
        if presence[index] <= q25:
            bucket = "rare_train_q1"
        elif presence[index] >= q75:
            bucket = "frequent_train_q4"
        else:
            bucket = "middle_train_q2_q3"
        result[name] = {
            "train_presence_support": presence[index],
            "train_absent_support": len(train_rows) - presence[index],
            "train_absent_fraction": (len(train_rows) - presence[index]) / len(train_rows),
            "train_class_support": {str(key): int(classes[key]) for key in sorted(classes)},
            "train_frequency_bucket": bucket,
            "train_q25_presence_threshold": q25,
            "train_q75_presence_threshold": q75,
        }
    return result


def _analyze_absa_seed(
    task_id: str,
    dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    train_frequency: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = list(dataset.aspect_names)
    pair_mask = [name != "OTHERS" for name in names] if task_id == "visfd" else [True] * len(names)
    gold_rows = [_model_to_raw(task_id, _as_tuple(row["gold"])) for row in rows]
    pred_rows = [_model_to_raw(task_id, _as_tuple(row["prediction"])) for row in rows]
    texts = [example.text for example in dataset.validation.examples]
    if len(gold_rows) != len(texts):
        raise ValueError("ABSA text/prediction row count mismatch")

    per_aspect_counts: dict[str, dict[str, int]] = {}
    per_aspect_classes: dict[str, Any] = {}
    for index, name in enumerate(names):
        counts = _empty_error_counts()
        gold_values = [row[index] for row in gold_rows]
        pred_values = [row[index] for row in pred_rows]
        for gold, pred in zip(gold_values, pred_values):
            _add_error(counts, gold, pred, pair_mask[index])
        per_aspect_counts[name] = counts
        class_labels = (0, 1, 2, 3) if name != "OTHERS" else (0, 4)
        per_aspect_classes[name] = {
            "class_names": (
                {"0": "absent", "1": "positive", "2": "negative", "3": "neutral"}
                if task_id == "visfd" and name != "OTHERS"
                else {"0": "absent", "1": "negative", "2": "neutral", "3": "positive"}
                if task_id != "visfd"
                else {"0": "absent", "4": "others"}
            ),
            "metrics": _class_metrics(gold_values, pred_values, class_labels),
            "train_frequency": dict(train_frequency[name]),
        }

    def grouped_counts(group_fn: Any) -> dict[str, Any]:
        groups: dict[str, list[dict[str, int]]] = defaultdict(list)
        for row_index, (gold_row, pred_row) in enumerate(zip(gold_rows, pred_rows)):
            group = str(group_fn(row_index, gold_row, pred_row))
            counts = _empty_error_counts()
            for index in range(len(names)):
                _add_error(counts, gold_row[index], pred_row[index], pair_mask[index])
            groups[group].append(counts)
        return {key: _with_rates(_merge_counts(values)) for key, values in sorted(groups.items())}

    by_frequency: dict[str, list[dict[str, int]]] = defaultdict(list)
    for name in names:
        by_frequency[str(train_frequency[name]["train_frequency_bucket"])].append(per_aspect_counts[name])
    frequency_summary = {
        key: _with_rates(_merge_counts(values)) for key, values in sorted(by_frequency.items())
    }

    # ``evaluate_task`` expects raw ViSFD gold and model-space predictions;
    # call it again with the exact serialized representation for an audit
    # metric.  This avoids relying on the diagnostic's normalized rows.
    metric = evaluate_task(
        task_id=task_id,
        gold=[_model_to_raw(task_id, _as_tuple(row["gold"])) for row in rows],
        predictions=[_as_tuple(row["prediction"]) for row in rows],
        aspect_names=names,
        aspect_class_counts=list(dataset.aspect_class_counts),
        split="validation",
    )
    length_summary = grouped_counts(lambda index, _g, _p: _length_bucket(texts[index]))
    return {
        "metrics": {
            "selection_f1": metric["selection_f1"],
            "selection_metric": metric["selection_metric"],
            "detection_micro_f1": metric["detection_micro_f1"],
            "detection_micro_precision": metric["detection_micro_precision"],
            "detection_micro_recall": metric["detection_micro_recall"],
            "pair_micro_f1": metric["pair_micro_f1"],
            "pair_micro_precision": metric["pair_micro_precision"],
            "pair_micro_recall": metric["pair_micro_recall"],
        },
        "error_decomposition": _with_rates(_merge_counts(per_aspect_counts.values())),
        "per_aspect": {
            name: {
                "errors": _with_rates(per_aspect_counts[name]),
                **per_aspect_classes[name],
            }
            for name in names
        },
        "by_train_frequency": frequency_summary,
        "by_length": length_summary,
        "validation_length": {
            "whitespace_token_count_min": min(len(text.split()) for text in texts),
            "whitespace_token_count_max": max(len(text.split()) for text in texts),
            "bucket_definition": {
                "short_0_10_whitespace_tokens": "0-10",
                "medium_11_20_whitespace_tokens": "11-20",
                "long_21plus_whitespace_tokens": "21+",
            },
        },
    }


def _analyze_scalar_seed(task_id: str, dataset: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gold = [int(row["gold"]) for row in rows]
    pred = [int(row["prediction"]) for row in rows]
    metric = evaluate_task(task_id=task_id, gold=gold, predictions=pred, split="validation")
    class_labels = tuple(range(len(dataset.label_names)))
    class_metrics = _class_metrics(gold, pred, class_labels)
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for example, expected, actual in zip(dataset.validation.examples, gold, pred):
        grouped[_length_bucket(example.text)].append((expected, actual))
    by_length = {}
    for key, pairs in sorted(grouped.items()):
        g = [item[0] for item in pairs]
        p = [item[1] for item in pairs]
        m = evaluate_task(task_id=task_id, gold=g, predictions=p, split="validation")
        by_length[key] = {"n": len(g), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    return {
        "metrics": {
            "selection_f1": metric["selection_f1"],
            "selection_metric": metric["selection_metric"],
            "accuracy": metric["accuracy"],
            "macro_f1": metric["macro_f1"],
        },
        "per_class": class_metrics,
        "by_length": by_length,
    }


def _load_tokenizer_lengths(dataset: Any) -> dict[str, Any]:
    """Measure validation truncation with the local tokenizer, without a model."""

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            REPO_ROOT / "models" / "goiturn-s-gd", local_files_only=True
        )
        texts = [example.text for example in dataset.validation.examples]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        lengths = [len(ids) for ids in encoded["input_ids"]]
        return {
            "available": True,
            "tokenizer": "models/goiturn-s-gd",
            "max_length_contract": 256,
            "true_token_count_min": min(lengths),
            "true_token_count_max": max(lengths),
            "rows_over_256": sum(value > 256 for value in lengths),
            "token_lengths_measured_without_truncation": True,
            "rows_would_truncate_at_256": sum(value > 256 for value in lengths),
            "any_would_truncate_at_256": any(value > 256 for value in lengths),
        }
    except Exception as error:  # pragma: no cover - the report records the limitation
        return {
            "available": False,
            "token_lengths_measured_without_truncation": False,
            "rows_would_truncate_at_256": None,
            "any_would_truncate_at_256": None,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _summarize_task(
    task_id: str,
    dataset: Any,
    paths: Mapping[str, Mapping[int, Path]],
    train_frequency: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    alignments: dict[str, Any] = {}
    for seed in SEEDS:
        seed_key = str(seed)
        e1_path = paths["e1"][seed]
        e2_path = paths["e2"][seed]
        e1_rows = _load_prediction_rows(e1_path)
        e2_rows = _load_prediction_rows(e2_path)
        _require_validation_alignment(task_id, dataset, e1_rows)
        _require_validation_alignment(task_id, dataset, e2_rows)
        if [row["id"] for row in e1_rows] != [row["id"] for row in e2_rows]:
            raise ValueError(f"{task_id} seed {seed}: E1/E2 prediction IDs differ")
        if task_id in ABSA_TASKS:
            assert train_frequency is not None
            e1_analysis = _analyze_absa_seed(task_id, dataset, e1_rows, train_frequency)
            e2_analysis = _analyze_absa_seed(task_id, dataset, e2_rows, train_frequency)
        else:
            e1_analysis = _analyze_scalar_seed(task_id, dataset, e1_rows)
            e2_analysis = _analyze_scalar_seed(task_id, dataset, e2_rows)
        per_seed[seed_key] = {
            "e1": {"artifact_dir": str(e1_path), **e1_analysis},
            "e2": {"artifact_dir": str(e2_path), **e2_analysis},
        }
        alignments[seed_key] = {
            "row_count": len(e1_rows),
            "e1_ids_match_source": True,
            "e2_ids_match_source": True,
            "e1_e2_ids_match": True,
            "serialized_gold_matches_source_raw_space": True,
        }

    result: dict[str, Any] = {
        "selection": {
            "e1_learning_rate": E1_SELECTED_LR[task_id],
            "e2_method": E2_SELECTED[task_id]["method"],
            "e2_coefficient": E2_SELECTED[task_id]["coefficient"],
            "e2_coefficient_name": E2_SELECTED[task_id]["coefficient_name"],
        },
        "alignment": alignments,
        "per_seed": per_seed,
        "validation_token_lengths": _load_tokenizer_lengths(dataset),
    }
    if train_frequency is not None:
        result["train_frequency"] = train_frequency
    return result


def build_report() -> dict[str, Any]:
    records = _load_e1_records()
    tasks: dict[str, Any] = {}
    for task_id in TASKS:
        dataset = load_task(task_id, REPO_ROOT, include_test=False)
        paths = {
            "e1": {seed: _find_e1_artifact(task_id, seed, records) for seed in SEEDS},
            "e2": {seed: _find_e2_artifact(task_id, seed) for seed in SEEDS},
        }
        train_frequency = _train_aspect_frequency(dataset) if task_id in ABSA_TASKS else None
        tasks[task_id] = _summarize_task(task_id, dataset, paths, train_frequency)
    return {
        "protocol": "validation-error-diagnosis-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "test_access": False,
        "include_test": False,
        "scope": {
            "tasks": list(TASKS),
            "focus_tasks": list(FOCUS_TASKS),
            "seeds": list(SEEDS),
            "e1": "selected supervised E1 learning rate per task",
            "e2": "validation-selected E2 method/coefficient per task",
        },
        "definitions": {
            "absa_presence": "positive iff label > 0; ViSFD OTHERS is raw code 4 and remains detection-positive",
            "viSFD_prediction_mapping": "serialized/model OTHERS 1 is mapped to raw code 4 for error comparison; OTHERS is excluded from pair/polarity counts",
            "missed_aspect": "gold present and prediction absent",
            "spurious_aspect": "gold absent and prediction present",
            "polarity_error": "both present but exact label differs; OTHERS is not treated as a polarity label",
            "train_rarity": "aspect presence support computed on TRAIN only; rare <= train presence Q1, frequent >= train presence Q4, middle otherwise",
            "length": "validation text length uses whitespace tokens: short 0-10, medium 11-20, long 21+",
            "truncation": "measured separately with the local GoiBERT tokenizer, no model inference",
        },
        "tasks": tasks,
        "source": {
            "e1_approved_records": str(REPO_ROOT / "artifacts/e1/.orchestrator/approved_records.json"),
            "e2_stage_selection": str(REPO_ROOT / "artifacts/e2/.orchestrator/stageA_selection.json"),
            "e1_approved_records_sha256": _sha256(REPO_ROOT / "artifacts/e1/.orchestrator/approved_records.json"),
            "e2_stage_selection_sha256": _sha256(REPO_ROOT / "artifacts/e2/.orchestrator/stageA_selection.json"),
            "data_loader": str(REPO_ROOT / "src/goibert/data.py"),
            "metrics": str(REPO_ROOT / "src/goibert/metrics.py"),
            "script": str(Path(__file__).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "diagnostics" / "validation_errors.json",
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "test_access": False, "tasks": list(report["tasks"])}, indent=2))


if __name__ == "__main__":
    main()
