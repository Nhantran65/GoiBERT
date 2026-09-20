#!/usr/bin/env python3
"""Validation-only teacher/student error diagnosis for E2.

This script deliberately reads saved validation predictions and metadata.  It
does not construct a model, open a checkpoint, load the test split, or run
inference.  It compares the frozen E1 seed-42 student with every E2 KD/NC
seed-42 candidate at the validation-selected coefficient, and summarizes the
same selected candidates over confirmation seeds 43--44.

The transition counts have two meanings:

* categorical tasks: a row is correct when its predicted class equals gold;
* ABSA tasks: detection correctness compares presence (label > 0), while a
  pair hit is a positive aspect-polarity match (gold > 0 and prediction equals
  gold).  ViSFD's OTHERS slot is included in detection and excluded from pair
  counts, matching the metric contract.

No test artifacts are read by this module.
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


TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
E1_SELECTED_LR = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}
SEEDS = (42, 43, 44)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number_close(left: float | int | None, right: float | int | None) -> bool:
    if left is None or right is None:
        return left == right
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)


def artifact_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path / "validation_predictions.json")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid validation prediction payload: {path}")
    return rows


def artifact_metadata(path: Path) -> dict[str, Any]:
    metadata = read_json(path / "metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError(f"Artifact is not completed: {path}")
    if metadata.get("include_test") is not False:
        raise ValueError(f"Diagnostic input opened test or omitted include_test=false: {path}")
    # Some selected validation artifacts were later evaluated by the separate
    # locked test gate. The diagnostic intentionally does not open those files.
    return metadata


def prediction_map(path: Path, task_id: str) -> dict[str, tuple[Any, Any]]:
    rows = artifact_rows(path)
    values: dict[str, tuple[Any, Any]] = {}
    for row in rows:
        example_id = str(row["id"])
        if example_id in values:
            raise ValueError(f"Duplicate validation ID {example_id} in {path}")
        gold = row["gold"]
        prediction = row["prediction"]
        if task_id in {"vsfc_sentiment", "vsfc_topic"}:
            gold = int(gold)
            prediction = int(prediction)
        else:
            gold = tuple(int(value) for value in gold)
            prediction = tuple(int(value) for value in prediction)
        values[example_id] = (gold, prediction)
    return values


def normalize_absa_row(task_id: str, row: Sequence[int]) -> tuple[int, ...]:
    values = [int(value) for value in row]
    if task_id == "visfd":
        # Saved predictions are in model space (OTHERS 0/1), while gold is in
        # source space (OTHERS 0/4).  Convert both to source-space comparison.
        if len(values) != 11:
            raise ValueError(f"Unexpected ViSFD row width: {len(values)}")
        others = values[5]
        if others == 1:
            values[5] = 4
        elif others not in (0, 4):
            raise ValueError(f"Invalid ViSFD OTHERS value: {others}")
    return tuple(values)


def comparable(task_id: str, value: Any) -> Any:
    if task_id in {"vsfc_sentiment", "vsfc_topic"}:
        return int(value)
    return normalize_absa_row(task_id, value)


def load_validation_contract(repo_root: Path, task_id: str) -> Any:
    # Import the frozen loader only; it is called with include_test=False.
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from goibert.data import load_task  # type: ignore

    return load_task(task_id, repo_root, include_test=False)


def validate_alignment(repo_root: Path, task_id: str, path: Path) -> tuple[Any, dict[str, tuple[Any, Any]]]:
    metadata = artifact_metadata(path)
    if metadata.get("task_id") != task_id:
        raise ValueError(f"Task mismatch in {path}: {metadata.get('task_id')} != {task_id}")
    if int(metadata.get("seed")) != 42 and "stageB" not in str(path):
        # Stage A and E1 inputs used for the direct teacher comparison are
        # seed42.  Stage B is checked separately by its caller.
        raise ValueError(f"Unexpected seed in seed42 diagnostic input: {path}")
    bundle = load_validation_contract(repo_root, task_id)
    expected_ids = set(bundle.validation.ids)
    values = prediction_map(path, task_id)
    if set(values) != expected_ids:
        missing = sorted(expected_ids.difference(values))[:5]
        extra = sorted(set(values).difference(expected_ids))[:5]
        raise ValueError(f"Validation IDs do not align for {path}; missing={missing}, extra={extra}")
    return bundle, values


def metrics_for(task_id: str, bundle: Any, values: Mapping[str, tuple[Any, Any]]) -> dict[str, Any]:
    src = bundle
    if task_id in {"vsfc_sentiment", "vsfc_topic"}:
        gold = [int(values[example.id][0]) for example in bundle.validation]
        prediction = [int(values[example.id][1]) for example in bundle.validation]
    else:
        gold = [tuple(int(value) for value in values[example.id][0]) for example in bundle.validation]
        prediction = [tuple(int(value) for value in values[example.id][1]) for example in bundle.validation]
    from goibert.metrics import evaluate_task  # type: ignore

    return evaluate_task(
        task_id=task_id,
        gold=gold,
        predictions=prediction,
        aspect_names=list(getattr(src, "aspect_names", ())),
        class_names=list(getattr(src, "class_names", ())),
        aspect_class_counts=list(getattr(src, "aspect_class_counts", ())),
        split="validation",
    )


def task_texts(bundle: Any) -> dict[str, str]:
    return {example.id: example.text for example in bundle.validation}


def categorical_transition(
    task_id: str,
    gold_student: Mapping[str, tuple[Any, Any]],
    gold_teacher: Mapping[str, tuple[Any, Any]],
    *,
    bundle: Any,
    max_examples: int = 5,
) -> dict[str, Any]:
    teacher_correct_student_wrong: list[str] = []
    teacher_wrong_student_correct: list[str] = []
    both_correct = both_wrong = 0
    for example in bundle.validation:
        example_id = example.id
        gold = int(gold_teacher[example_id][0])
        teacher_prediction = int(gold_teacher[example_id][1])
        student_prediction = int(gold_student[example_id][1])
        teacher_ok = teacher_prediction == gold
        student_ok = student_prediction == gold
        if teacher_ok and student_ok:
            both_correct += 1
        elif not teacher_ok and not student_ok:
            both_wrong += 1
        elif teacher_ok:
            teacher_correct_student_wrong.append(example_id)
        else:
            teacher_wrong_student_correct.append(example_id)
    texts = task_texts(bundle)

    def examples(ids: Iterable[str]) -> list[dict[str, Any]]:
        output = []
        for example_id in list(ids)[:max_examples]:
            gold = int(gold_teacher[example_id][0])
            output.append(
                {
                    "id": example_id,
                    "text": texts[example_id],
                    "gold": gold,
                    "teacher_prediction": int(gold_teacher[example_id][1]),
                    "student_prediction": int(gold_student[example_id][1]),
                }
            )
        return output

    total = len(bundle.validation)
    return {
        "n_rows": total,
        "teacher_correct_student_wrong": len(teacher_correct_student_wrong),
        "teacher_wrong_student_correct": len(teacher_wrong_student_correct),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_student_gain_on_teacher": len(teacher_wrong_student_correct)
        - len(teacher_correct_student_wrong),
        "teacher_correct_student_wrong_examples": examples(teacher_correct_student_wrong),
        "teacher_wrong_student_correct_examples": examples(teacher_wrong_student_correct),
    }


def absa_transition(
    task_id: str,
    student: Mapping[str, tuple[Any, Any]],
    teacher: Mapping[str, tuple[Any, Any]],
    *,
    bundle: Any,
    max_examples: int = 5,
) -> dict[str, Any]:
    aspect_names = list(bundle.aspect_names)
    pair_mask = list(bundle.pair_scored_aspects or (True for _ in aspect_names))
    n_slots = len(aspect_names)
    aggregate = {
        "detection": Counter(),
        "positive_pair": Counter(),
    }
    per_aspect: list[dict[str, Any]] = []
    examples_by_kind: dict[str, list[dict[str, Any]]] = {
        "detection_teacher_correct_student_wrong": [],
        "detection_teacher_wrong_student_correct": [],
        "pair_teacher_correct_student_wrong": [],
        "pair_teacher_wrong_student_correct": [],
    }
    texts = task_texts(bundle)

    def add_example(kind: str, example_id: str, slot: int, gold: int, t: int, s: int) -> None:
        if len(examples_by_kind[kind]) >= max_examples:
            return
        examples_by_kind[kind].append(
            {
                "id": example_id,
                "text": texts[example_id],
                "aspect": aspect_names[slot],
                "gold": gold,
                "teacher_prediction": t,
                "student_prediction": s,
            }
        )

    slot_rows: list[dict[str, Any]] = []
    for slot, aspect in enumerate(aspect_names):
        detection = Counter()
        pair = Counter()
        for example in bundle.validation:
            example_id = example.id
            gold_row = normalize_absa_row(task_id, teacher[example_id][0])
            teacher_row = normalize_absa_row(task_id, teacher[example_id][1])
            student_row = normalize_absa_row(task_id, student[example_id][1])
            gold = int(gold_row[slot])
            t = int(teacher_row[slot])
            s = int(student_row[slot])
            t_detection_ok = (t > 0) == (gold > 0)
            s_detection_ok = (s > 0) == (gold > 0)
            if t_detection_ok and s_detection_ok:
                detection["both_correct"] += 1
            elif not t_detection_ok and not s_detection_ok:
                detection["both_wrong"] += 1
            elif t_detection_ok:
                detection["teacher_correct_student_wrong"] += 1
                add_example("detection_teacher_correct_student_wrong", example_id, slot, gold, t, s)
            else:
                detection["teacher_wrong_student_correct"] += 1
                add_example("detection_teacher_wrong_student_correct", example_id, slot, gold, t, s)

            if pair_mask[slot]:
                # A positive pair hit is the unit counted by pair-micro F1.
                t_pair_ok = gold > 0 and t == gold
                s_pair_ok = gold > 0 and s == gold
                if t_pair_ok and s_pair_ok:
                    pair["both_hits"] += 1
                elif not t_pair_ok and not s_pair_ok:
                    pair["neither_hit"] += 1
                elif t_pair_ok:
                    pair["teacher_hit_student_miss"] += 1
                    add_example("pair_teacher_correct_student_wrong", example_id, slot, gold, t, s)
                else:
                    pair["teacher_miss_student_hit"] += 1
                    add_example("pair_teacher_wrong_student_correct", example_id, slot, gold, t, s)
            else:
                pair["excluded_others_slot"] += 1
        for key, value in detection.items():
            aggregate["detection"][key] += value
        for key, value in pair.items():
            aggregate["positive_pair"][key] += value
        slot_rows.append(
            {
                "aspect": aspect,
                "pair_scored": bool(pair_mask[slot]),
                "n_rows": len(bundle.validation),
                "detection": dict(detection),
                "positive_pair": dict(pair),
            }
        )

    return {
        "n_rows": len(bundle.validation),
        "n_aspects": n_slots,
        "definitions": {
            "detection_correct": "presence agrees: (label > 0) is equal for gold and prediction; ViSFD OTHERS is included",
            "positive_pair_hit": "gold is a positive aspect and predicted source-space label equals gold; ViSFD OTHERS is excluded",
            "teacher_correct_student_wrong": "teacher satisfies the named unit and student does not",
            "teacher_wrong_student_correct": "teacher does not satisfy the named unit and student does",
        },
        "aggregate": {key: dict(value) for key, value in aggregate.items()},
        "per_aspect": slot_rows,
        "examples": examples_by_kind,
    }


def transition(
    task_id: str,
    student: Mapping[str, tuple[Any, Any]],
    teacher: Mapping[str, tuple[Any, Any]],
    *,
    bundle: Any,
) -> dict[str, Any]:
    if task_id in {"vsfc_sentiment", "vsfc_topic"}:
        return {
            "unit": "validation row exact class",
            "counts": categorical_transition(task_id, student, teacher, bundle=bundle),
        }
    return absa_transition(task_id, student, teacher, bundle=bundle)


def e1_to_e2_stratified_by_teacher(
    task_id: str,
    e1: Mapping[str, tuple[Any, Any]],
    e2: Mapping[str, tuple[Any, Any]],
    teacher: Mapping[str, tuple[Any, Any]],
    *,
    bundle: Any,
) -> dict[str, Any]:
    """Count E1->E2 repairs/breaks separately on teacher-correct/wrong units."""

    def empty() -> dict[str, int]:
        return {"eligible": 0, "e1_correct_e2_wrong": 0, "e1_wrong_e2_correct": 0}

    if task_id in {"vsfc_sentiment", "vsfc_topic"}:
        output = {"teacher_correct": empty(), "teacher_wrong": empty()}
        for example in bundle.validation:
            key = example.id
            gold = int(teacher[key][0])
            teacher_ok = int(teacher[key][1]) == gold
            e1_ok = int(e1[key][1]) == gold
            e2_ok = int(e2[key][1]) == gold
            bucket = output["teacher_correct" if teacher_ok else "teacher_wrong"]
            bucket["eligible"] += 1
            if e1_ok and not e2_ok:
                bucket["e1_correct_e2_wrong"] += 1
            elif not e1_ok and e2_ok:
                bucket["e1_wrong_e2_correct"] += 1
        return {"unit": "validation row exact class", "by_teacher_status": output}

    output = {
        unit: {"teacher_correct": empty(), "teacher_wrong": empty()}
        for unit in ("detection", "positive_pair")
    }
    for example in bundle.validation:
        key = example.id
        gold_row = normalize_absa_row(task_id, teacher[key][0])
        teacher_row = normalize_absa_row(task_id, teacher[key][1])
        e1_row = normalize_absa_row(task_id, e1[key][1])
        e2_row = normalize_absa_row(task_id, e2[key][1])
        for index, scored in enumerate(bundle.pair_scored_aspects or ()):
            units = ["detection"]
            if scored:
                units.append("positive_pair")
            for unit in units:
                gold = int(gold_row[index])
                teacher_value = int(teacher_row[index])
                e1_value = int(e1_row[index])
                e2_value = int(e2_row[index])
                if unit == "detection":
                    teacher_ok = (teacher_value > 0) == (gold > 0)
                    e1_ok = (e1_value > 0) == (gold > 0)
                    e2_ok = (e2_value > 0) == (gold > 0)
                else:
                    teacher_ok = gold > 0 and teacher_value == gold
                    e1_ok = gold > 0 and e1_value == gold
                    e2_ok = gold > 0 and e2_value == gold
                bucket = output[unit]["teacher_correct" if teacher_ok else "teacher_wrong"]
                bucket["eligible"] += 1
                if e1_ok and not e2_ok:
                    bucket["e1_correct_e2_wrong"] += 1
                elif not e1_ok and e2_ok:
                    bucket["e1_wrong_e2_correct"] += 1
    return {
        "definitions": {
            "teacher_correct": "the teacher satisfies the unit (row, detection, or positive pair)",
            "teacher_wrong": "the teacher does not satisfy the unit",
            "e1_correct_e2_wrong": "E2 breaks a unit E1 got right",
            "e1_wrong_e2_correct": "E2 repairs a unit E1 got wrong",
        },
        "by_unit": output,
    }


def scan_e1_paths(repo_root: Path, task_id: str, seed: int, learning_rate: float) -> Path:
    candidates: list[Path] = []
    for prediction_path in sorted((repo_root / "artifacts" / "e1").rglob("validation_predictions.json")):
        path = prediction_path.parent
        if "smoke" in path.parts or ".orchestrator" not in path.parts:
            continue
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = read_json(metadata_path)
        if (
            metadata.get("task_id") == task_id
            and int(metadata.get("seed", -1)) == seed
            and number_close(metadata.get("learning_rate"), learning_rate)
            and metadata.get("include_test") is False
            and metadata.get("status") == "completed"
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one canonical E1 path for {task_id} seed={seed} lr={learning_rate}; "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def candidate_record_paths(repo_root: Path, task_id: str, seed: int, learning_rate: float) -> list[Path]:
    """Return all completed stage records at one LR, without smoke artifacts."""
    paths: list[Path] = []
    records = read_json(repo_root / "artifacts" / "e2" / ".orchestrator" / "approved_records.json")
    for record in records.get("approved", {}).get("stage0_teacher", []):
        if (
            record.get("task_id") == task_id
            and int(record.get("seed", -1)) == seed
            and number_close(record.get("learning_rate"), learning_rate)
        ):
            path = Path(record["artifact_dir"]).resolve()
            if path not in paths:
                paths.append(path)
    return paths


def selected_teacher_path(repo_root: Path, task_id: str) -> Path:
    selection = read_json(repo_root / "artifacts" / "e2" / ".orchestrator" / "stage0_teacher_selection.json")
    record = selection["selected"][task_id]
    path = Path(record["artifact_dir"]).resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def selected_e2_path(repo_root: Path, method: str, task_id: str, coefficient: float, seed: int) -> Path:
    if seed == 42:
        path = repo_root / "artifacts" / "e2" / "stageA" / method / task_id / coefficient_dir(coefficient) / "seed_42"
    else:
        path = repo_root / "artifacts" / "e2" / "stageB" / method / task_id / coefficient_dir(coefficient) / f"seed_{seed}"
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def coefficient_dir(value: float) -> str:
    token = f"{value:g}".replace(".", "p")
    return f"coef_{token}"


def metric_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "selection_f1",
        "selection_metric",
        "macro_f1",
        "accuracy",
        "detection_micro_f1",
        "pair_micro_f1",
        "classification_pair_micro_f1",
        "n",
    )
    return {name: metrics[name] for name in names if name in metrics}


def average_metric_dicts(metrics: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = (
        "selection_f1",
        "macro_f1",
        "accuracy",
        "detection_micro_f1",
        "pair_micro_f1",
        "classification_pair_micro_f1",
    )
    output: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in metrics if key in row]
        if values:
            output[key] = sum(values) / len(values)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output; defaults to artifacts/diagnostics/teacher_gap.json",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = (args.output or repo_root / "artifacts" / "diagnostics" / "teacher_gap.json").resolve()

    # Read the locked coefficient decisions; this file is validation-only and
    # explicitly records that test access was false.
    stage_a_selection = read_json(repo_root / "artifacts" / "e2" / ".orchestrator" / "stageA_selection.json")
    if stage_a_selection.get("test_access") is not False or stage_a_selection.get("locked") is not True:
        raise RuntimeError("Stage-A coefficient selection is not a locked validation-only record")

    results: dict[str, Any] = {}
    input_files: dict[str, str] = {}
    for task_id in TASKS:
        bundle = load_validation_contract(repo_root, task_id)
        teacher_path = selected_teacher_path(repo_root, task_id)
        teacher_meta = artifact_metadata(teacher_path)
        if int(teacher_meta.get("seed")) != 42:
            raise RuntimeError(f"Selected teacher is not seed42: {teacher_path}")
        teacher_values = prediction_map(teacher_path, task_id)
        expected_ids = set(bundle.validation.ids)
        if set(teacher_values) != expected_ids:
            raise RuntimeError(f"Teacher validation IDs do not match data: {teacher_path}")
        input_files[f"{task_id}.teacher"] = str(teacher_path / "validation_predictions.json")

        e1_paths = {seed: scan_e1_paths(repo_root, task_id, seed, E1_SELECTED_LR[task_id]) for seed in SEEDS}
        e1_values = {seed: prediction_map(path, task_id) for seed, path in e1_paths.items()}
        for seed, path in e1_paths.items():
            artifact_metadata(path)
            if set(e1_values[seed]) != expected_ids:
                raise RuntimeError(f"E1 validation IDs do not match data: {path}")
            input_files[f"{task_id}.e1.seed{seed}"] = str(path / "validation_predictions.json")

        kd_coefficient = float(stage_a_selection["selected_kd_alpha"][task_id])
        nc_coefficient = float(stage_a_selection["selected_nc_lambda"][task_id])
        e2_paths: dict[str, dict[int, Path]] = {"kd": {}, "nc": {}}
        e2_values: dict[str, dict[int, dict[str, tuple[Any, Any]]]] = {"kd": {}, "nc": {}}
        e2_metrics: dict[str, dict[int, dict[str, Any]]] = {"kd": {}, "nc": {}}
        for method, coefficient in (("kd", kd_coefficient), ("nc", nc_coefficient)):
            for seed in SEEDS:
                path = selected_e2_path(repo_root, method, task_id, coefficient, seed)
                artifact_metadata(path)
                values = prediction_map(path, task_id)
                if set(values) != expected_ids:
                    raise RuntimeError(f"E2 validation IDs do not match data: {path}")
                e2_paths[method][seed] = path
                e2_values[method][seed] = values
                e2_metrics[method][seed] = metrics_for(task_id, bundle, values)
                input_files[f"{task_id}.e2.{method}.seed{seed}"] = str(path / "validation_predictions.json")

        teacher_metrics = metrics_for(task_id, bundle, teacher_values)
        e1_metrics = {seed: metrics_for(task_id, bundle, values) for seed, values in e1_values.items()}
        method_rows: dict[str, Any] = {}
        for method in ("kd", "nc"):
            method_rows[method] = {
                "coefficient": kd_coefficient if method == "kd" else nc_coefficient,
                "seed_metrics": {str(seed): metric_subset(e2_metrics[method][seed]) for seed in SEEDS},
                "mean_seed42_44": average_metric_dicts([e2_metrics[method][seed] for seed in SEEDS]),
                "seed42_transition_vs_teacher": transition(
                    task_id, e2_values[method][42], teacher_values, bundle=bundle
                ),
                "seed42_transition_vs_e1": transition(
                    task_id, e2_values[method][42], e1_values[42], bundle=bundle
                ),
                "seed42_e1_to_e2_stratified_by_teacher": e1_to_e2_stratified_by_teacher(
                    task_id, e1_values[42], e2_values[method][42], teacher_values, bundle=bundle
                ),
            }

        # The direct transition is paired on exactly the same validation rows;
        # teacher is single-seed, while the student summary below is over the
        # three preselected E2 confirmation seeds.
        teacher_candidate_paths: list[Path] = []
        for candidate_lr in (1e-5, 2e-5, 5e-5):
            teacher_candidate_paths.extend(
                candidate_record_paths(repo_root, task_id, 42, float(candidate_lr))
            )

        results[task_id] = {
            "validation_n": len(bundle.validation),
            "teacher": {
                "learning_rate": float(teacher_meta["learning_rate"]),
                "artifact_dir": str(teacher_path),
                "metrics": metric_subset(teacher_metrics),
                "candidate_lrs": [
                    {
                        "learning_rate": float(read_json(path / "metadata.json")["learning_rate"]),
                        "artifact_dir": str(path),
                        "metrics": metric_subset(metrics_for(task_id, bundle, prediction_map(path, task_id))),
                    }
                    for path in sorted(teacher_candidate_paths)
                ],
            },
            "e1": {
                "selected_learning_rate": E1_SELECTED_LR[task_id],
                "seed_metrics": {str(seed): metric_subset(e1_metrics[seed]) for seed in SEEDS},
                "mean_seed42_44": average_metric_dicts([e1_metrics[seed] for seed in SEEDS]),
                "seed42_artifact_dir": str(e1_paths[42]),
                "seed42_transition_vs_teacher": transition(
                    task_id, e1_values[42], teacher_values, bundle=bundle
                ),
            },
            "e2": method_rows,
            "teacher_minus_student_seed42": {
                method: {
                    key: float(teacher_metrics[key]) - float(e2_metrics[method][42][key])
                    for key in ("selection_f1", "detection_micro_f1", "pair_micro_f1", "macro_f1")
                    if key in teacher_metrics and key in e2_metrics[method][42]
                }
                for method in ("kd", "nc")
            },
        }

    script_path = Path(__file__).resolve()
    payload = {
        "protocol": "teacher-student-validation-diagnosis-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "test_access": False,
        "test_access_reason": "The loader was called only with include_test=False; no test prediction artifact was read.",
        "script_sha256": sha256_file(script_path),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "e1_selected_learning_rates": E1_SELECTED_LR,
        "input_prediction_files": {
            key: {"path": value, "sha256": sha256_file(Path(value))}
            for key, value in sorted(input_files.items())
        },
        "results": results,
        "limits": [
            "Teacher metrics and direct transition counts use one selected teacher seed (42); teacher variance is not estimated.",
            "E2 coefficient choices are validation-selected and the transition analysis is conditional on those choices.",
            "Argmax prediction transitions do not measure soft-logit calibration or whether KD transferred teacher uncertainty.",
            "ABSA positive_pair counts deliberately ignore true-negative slots because pair micro-F1 scores positive aspect-polarity hits; detection counts include negatives through presence correctness.",
        ],
    }
    write_json(output, payload)
    print(json.dumps({"output": str(output), "tasks": list(TASKS), "test_access": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
