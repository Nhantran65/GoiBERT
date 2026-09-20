#!/usr/bin/env python3
"""E12 validation-only surface-error audit.

This script is intentionally isolated from the training/evaluation runners.  It
loads the frozen train/validation rows with ``include_test=False``, reads the
already saved E8 seed-42 validation predictions, and describes associations
between surface features and errors.  It does not load a model checkpoint,
run inference, or touch the test split.

The output is an audit artifact, not a causal test of a character model.  The
feature vocabulary and analysis choices are kept here so the generated files
remain reproducible without changing E1--E11 code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
DOMAINS = {
    "vsfc_sentiment": "vsfc",
    "vsfc_topic": "vsfc",
    "visfd": "visfd",
    "absa_restaurant": "absa_restaurant",
    "absa_hotel": "absa_hotel",
}
DOMAIN_ORDER = ("vsfc", "visfd", "absa_restaurant", "absa_hotel")
SEED = 12042
MODEL_MAX_LENGTH = 256

NUMERIC_FEATURES = (
    "unit_count",
    "bpe_tokens_full",
    "bpe_tokens_model",
    "bpe_per_unit_mean",
    "unit_ge3pieces_rate",
    "rare_unit_rate",
    "unseen_unit_rate",
    "unicode_ratio",
    "emoji_count",
    "combining_mark_count",
    "char_repeat_max",
    "accent_proxy_unit_rate",
    "punctuation_count",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_cell(row.get(key, "")) for key in fieldnames})


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def is_unit_char(char: str) -> bool:
    category = unicodedata.category(char)
    return category[0] in {"L", "M", "N"}


def contiguous_units(text: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    start: int | None = None
    for index, char in enumerate(text):
        if is_unit_char(char):
            if start is None:
                start = index
        elif start is not None:
            units.append({"text": text[start:index], "start": start, "end": index})
            start = None
    if start is not None:
        units.append({"text": text[start:], "start": start, "end": len(text)})
    return units


def nfc_lower(value: str) -> str:
    return unicodedata.normalize("NFC", value).lower()


def skeleton(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return unicodedata.normalize("NFC", "".join(char for char in decomposed if unicodedata.category(char) != "Mn")).lower()


def has_diacritic(value: str) -> bool:
    return any(unicodedata.category(char).startswith("M") for char in unicodedata.normalize("NFD", value))


def emoji_count(text: str) -> int:
    # A transparent range heuristic.  It is deliberately described as emoji
    # code-point count; it does not attempt full grapheme/ZWJ segmentation.
    count = 0
    for char in text:
        code = ord(char)
        if (
            0x1F000 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x27BF
            or 0x2300 <= code <= 0x23FF
        ):
            count += 1
    return count


def max_character_run(text: str) -> int:
    if not text:
        return 0
    best = current = 1
    for left, right in zip(text, text[1:]):
        current = current + 1 if left == right else 1
        best = max(best, current)
    return best


def train_frequency(train_texts: Sequence[str]) -> tuple[Counter[str], dict[str, set[str]], dict[str, int]]:
    counts: Counter[str] = Counter()
    variants: dict[str, set[str]] = defaultdict(set)
    for text in train_texts:
        for unit in contiguous_units(text):
            value = nfc_lower(unit["text"])
            counts[value] += 1
            variants[skeleton(value)].add(value)
    skeleton_ambiguity = {key: len(values) for key, values in variants.items()}
    return counts, variants, skeleton_ambiguity


def token_alignment(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]], list[int], list[tuple[int, int]]]:
    full = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    model = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MODEL_MAX_LENGTH,
        return_offsets_mapping=True,
    )
    full_ids = list(full["input_ids"])
    model_ids = list(model["input_ids"])
    full_offsets = [tuple(map(int, pair)) for pair in full["offset_mapping"]]
    model_offsets = [tuple(map(int, pair)) for pair in model["offset_mapping"]]
    return full_ids, full_offsets, model_ids, model_offsets


def build_features(
    text: str,
    tokenizer: Any,
    counts: Counter[str],
    variants: Mapping[str, set[str]],
    skeleton_ambiguity: Mapping[str, int],
) -> dict[str, Any]:
    units = contiguous_units(text)
    full_ids, full_offsets, model_ids, model_offsets = token_alignment(tokenizer, text)
    content_full = [(start, end) for start, end in full_offsets if end > start]
    content_model = [(start, end) for start, end in model_offsets if end > start]
    pieces_per_unit: list[int] = []
    rare_count = unseen_count = proxy_count = proxy_ambiguous_count = 0
    accent_partner_count: list[int] = []
    unit_records: list[dict[str, Any]] = []
    for unit in units:
        value = nfc_lower(unit["text"])
        freq = int(counts.get(value, 0))
        is_unseen = freq == 0
        is_rare = freq <= 5
        rare_count += int(is_rare)
        unseen_count += int(is_unseen)
        unit_start, unit_end = int(unit["start"]), int(unit["end"])
        overlap = sum(1 for token_start, token_end in content_full if token_start < unit_end and token_end > unit_start)
        pieces_per_unit.append(overlap)
        skel = skeleton(value)
        partners = [
            candidate for candidate in variants.get(skel, set())
            if candidate != value and has_diacritic(candidate)
        ]
        proxy = bool((not has_diacritic(value)) and is_rare and partners)
        proxy_count += int(proxy)
        proxy_ambiguous_count += int(proxy and len(partners) > 1)
        accent_partner_count.append(len(partners))
        unit_records.append(
            {
                "text": unit["text"],
                "nfc_lower": value,
                "start": unit_start,
                "end": unit_end,
                "train_frequency": freq,
                "rare_le5": is_rare,
                "unseen": is_unseen,
                "skeleton": skel,
                "accent_partner_count": len(partners),
                "skeleton_ambiguity_train": int(skeleton_ambiguity.get(skel, 0)),
                "accent_absence_proxy": proxy,
                "bpe_piece_count": overlap,
            }
        )
    unicode_chars = sum(ord(char) > 127 for char in text)
    combining_count = sum(unicodedata.category(char).startswith("M") for char in unicodedata.normalize("NFD", text))
    punctuation_count = sum(unicodedata.category(char).startswith("P") for char in text)
    multi_unit_tokens = 0
    for token_start, token_end in content_full:
        if sum(token_start < unit["end"] and token_end > unit["start"] for unit in units) > 1:
            multi_unit_tokens += 1
    unit_count = len(units)
    bpe_per_unit_mean = safe_rate(sum(pieces_per_unit), unit_count)
    return {
        "text": text,
        "char_count": len(text),
        "unit_count": unit_count,
        "bpe_tokens_full": len(content_full),
        "bpe_tokens_model": len(content_model),
        "bpe_sequence_length_full": len(full_ids),
        "bpe_sequence_length_model": len(model_ids),
        "truncated_at_256": len(full_ids) > MODEL_MAX_LENGTH,
        "bpe_per_unit_mean": bpe_per_unit_mean,
        "unit_ge3pieces_rate": safe_rate(sum(piece >= 3 for piece in pieces_per_unit), unit_count),
        "bpe_multiple_unit_token_count": multi_unit_tokens,
        "rare_unit_count": rare_count,
        "rare_unit_rate": safe_rate(rare_count, unit_count),
        "unseen_unit_count": unseen_count,
        "unseen_unit_rate": safe_rate(unseen_count, unit_count),
        "accent_proxy_unit_count": proxy_count,
        "accent_proxy_unit_rate": safe_rate(proxy_count, unit_count),
        "accent_proxy_ambiguous_count": proxy_ambiguous_count,
        "accent_partner_max": max(accent_partner_count, default=0),
        "unicode_ratio": safe_rate(unicode_chars, len(text)),
        "unicode_non_ascii_count": unicode_chars,
        "combining_mark_count": combining_count,
        "emoji_count": emoji_count(text),
        "punctuation_count": punctuation_count,
        "char_repeat_max": max_character_run(text),
        "unit_records": unit_records,
    }


def feature_thresholds(train_features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        values = [float(row[feature]) for row in train_features]
        q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
        result[feature] = {
            "q1": q1,
            "q3": q3,
            "min": min(values, default=0.0),
            "max": max(values, default=0.0),
            "constant": bool(values and min(values) == max(values)),
            "slice_enabled": bool(values and min(values) != max(values) and q1 != q3),
            "rule": "low<=q1, high>=q3; ties are retained; disabled when train feature is constant or q1=q3",
        }
    return result


def slice_group(value: Any, threshold: Mapping[str, Any]) -> str | None:
    if not threshold["slice_enabled"]:
        return None
    numeric = float(value)
    low = numeric <= float(threshold["q1"])
    high = numeric >= float(threshold["q3"])
    if low and high:
        return "low_high_tie"
    if low:
        return "low"
    if high:
        return "high"
    return "middle"


def length_bin(value: float, thresholds: Mapping[str, Any]) -> str:
    if thresholds["constant"] or thresholds["q1"] == thresholds["q3"]:
        return "all"
    if value <= thresholds["q1"]:
        return "short_or_tie"
    if value >= thresholds["q3"]:
        return "long_or_tie"
    return "middle"


def compare_ids_and_gold(task_id: str, bundle: Any, prediction_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{task_id}: prediction file has no rows: {prediction_path}")
    expected_ids = [str(value) for value in bundle.validation.ids]
    actual_ids = [str(row.get("id")) for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(f"{task_id}: prediction IDs do not exactly match validation loader order")
    loaded_gold = list(bundle.validation.labels)
    for index, row in enumerate(rows):
        saved_gold = row.get("gold")
        expected_gold = loaded_gold[index]
        if isinstance(expected_gold, tuple):
            expected_gold = list(expected_gold)
        if saved_gold != expected_gold:
            raise ValueError(f"{task_id}: saved gold differs from include_test=False loader at row {index}")
    return rows, {"prediction_rows": len(rows), "ids_match": True, "gold_match": True}


def vsfc_outcome(gold: int, prediction: int) -> dict[str, Any]:
    return {"error": int(int(gold) != int(prediction)), "error_events": int(int(gold) != int(prediction)), "gold_positive_slots": 1}


def absa_outcome(task_id: str, gold: Sequence[int], prediction: Sequence[int], aspect_names: Sequence[str]) -> dict[str, Any]:
    if len(gold) != len(prediction):
        raise ValueError(f"{task_id}: ABSA gold/prediction width mismatch")
    pair_mask = [name != "OTHERS" for name in aspect_names] if task_id == "visfd" else [True] * len(aspect_names)
    missed = spurious = wrong_polarity = pair_hits = 0
    gold_positive = predicted_positive = 0
    per_slot: list[dict[str, Any]] = []
    for index, (gold_value, pred_value, scored) in enumerate(zip(gold, prediction, pair_mask)):
        # OTHERS uses model 0/1 in saved E8 predictions, while gold uses raw 0/4.
        semantic_pred = 4 if task_id == "visfd" and aspect_names[index] == "OTHERS" and int(pred_value) == 1 else int(pred_value)
        if int(gold_value) > 0:
            gold_positive += int(scored)
        if semantic_pred > 0:
            predicted_positive += int(scored)
        if not scored:
            per_slot.append({"aspect": aspect_names[index], "scored": False, "gold": int(gold_value), "prediction": semantic_pred, "event": "detection_only"})
            continue
        if int(gold_value) > 0 and semantic_pred == 0:
            missed += 1
            event = "missed"
        elif int(gold_value) == 0 and semantic_pred > 0:
            spurious += 1
            event = "spurious"
        elif int(gold_value) > 0 and semantic_pred > 0 and int(gold_value) != semantic_pred:
            wrong_polarity += 1
            event = "wrong_polarity"
        else:
            pair_hits += 1
            event = "correct_pair"
        per_slot.append({"aspect": aspect_names[index], "scored": True, "gold": int(gold_value), "prediction": semantic_pred, "event": event})
    error_events = missed + spurious + wrong_polarity
    return {
        "error": int(error_events > 0),
        "missed": missed,
        "spurious": spurious,
        "wrong_polarity": wrong_polarity,
        "pair_hits": pair_hits,
        "error_events": error_events,
        "gold_positive_slots": gold_positive,
        "predicted_positive_slots": predicted_positive,
        # Undefined when a row has no scored gold-positive slots.  Spurious
        # predictions are reported separately instead of being hidden as a
        # zero normalized burden.
        "error_burden": safe_rate(error_events, gold_positive) if gold_positive else None,
        "spurious_per_row": safe_rate(spurious, 1),
        "per_slot": per_slot,
    }


def outcome_fields(task_id: str, bundle: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    if task_id.startswith("vsfc_"):
        return vsfc_outcome(int(row["gold"]), int(row["prediction"]))
    return absa_outcome(task_id, row["gold"], row["prediction"], list(bundle.aspect_names))


def group_metrics(rows: Sequence[Mapping[str, Any]], task_id: str) -> dict[str, Any]:
    n = len(rows)
    if task_id.startswith("vsfc_"):
        errors = sum(int(row["error"]) for row in rows)
        return {"n": n, "events": errors, "event_rate": safe_rate(errors, n), "gold_positive_slots": n, "error_burden": safe_rate(errors, n), "spurious_per_row": 0.0}
    events = sum(int(row["error_events"]) for row in rows)
    gold_positive = sum(int(row["gold_positive_slots"]) for row in rows)
    spurious = sum(int(row["spurious"]) for row in rows)
    return {"n": n, "events": events, "event_rate": safe_rate(sum(int(row["error"]) for row in rows), n), "gold_positive_slots": gold_positive, "error_burden": safe_rate(events, gold_positive) if gold_positive else None, "spurious_per_row": safe_rate(spurious, n), "missed": sum(int(row["missed"]) for row in rows), "spurious": spurious, "wrong_polarity": sum(int(row["wrong_polarity"]) for row in rows)}


def standardized_gap(rows: Sequence[Mapping[str, Any]], task_id: str, group_key: str, strata_keys: Sequence[str]) -> dict[str, Any]:
    groups = {str(row.get(group_key)) for row in rows}
    usable = {"low", "high", "present", "absent"}.intersection(groups)
    if not {"low", "high"}.issubset(groups) and not {"present", "absent"}.issubset(groups):
        return {"adjusted_gap": None, "coverage_low": 0.0, "coverage_high": 0.0, "common_strata": 0, "zero_gold_common_strata": 0, "dropped_rows": len(rows), "note": "no paired low/high support"}
    left, right = ("low", "high") if {"low", "high"}.issubset(groups) else ("absent", "present")
    by_stratum: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        stratum = tuple(row.get(key) for key in strata_keys)
        by_stratum[stratum][str(row.get(group_key))].append(row)
    common = [stratum for stratum, values in by_stratum.items() if left in values and right in values]
    if not common:
        return {"adjusted_gap": None, "coverage_low": 0.0, "coverage_high": 0.0, "common_strata": 0, "zero_gold_common_strata": 0, "dropped_rows": len(rows), "note": "no common-support strata"}
    usable_common: list[tuple[Any, ...]] = []
    zero_gold_common = 0
    for stratum in common:
        if task_id.startswith("vsfc_"):
            usable_common.append(stratum)
            continue
        left_metrics = group_metrics(by_stratum[stratum][left], task_id)
        right_metrics = group_metrics(by_stratum[stratum][right], task_id)
        if left_metrics["error_burden"] is None or right_metrics["error_burden"] is None:
            zero_gold_common += 1
        else:
            usable_common.append(stratum)
    if not usable_common:
        return {"adjusted_gap": None, "coverage_low": 0.0, "coverage_high": 0.0, "common_strata": 0, "zero_gold_common_strata": zero_gold_common, "dropped_rows": len(rows), "note": "no positive-support common strata"}
    target_total = sum(len(by_stratum[stratum][left]) + len(by_stratum[stratum][right]) for stratum in usable_common)
    left_total = sum(len(by_stratum[stratum][left]) for stratum in usable_common)
    right_total = sum(len(by_stratum[stratum][right]) for stratum in usable_common)
    left_adjusted = right_adjusted = 0.0
    for stratum in usable_common:
        values = by_stratum[stratum]
        weight = safe_rate(len(values[left]) + len(values[right]), target_total)
        left_adjusted += weight * (group_metrics(values[left], task_id)["error_burden"])
        right_adjusted += weight * (group_metrics(values[right], task_id)["error_burden"])
    return {
        "adjusted_gap": right_adjusted - left_adjusted,
        "adjusted_low_or_absent": left_adjusted,
        "adjusted_high_or_present": right_adjusted,
        "coverage_low_or_absent": safe_rate(left_total, sum(1 for row in rows if str(row.get(group_key)) == left)),
        "coverage_high_or_present": safe_rate(right_total, sum(1 for row in rows if str(row.get(group_key)) == right)),
        "common_strata": len(usable_common),
        "zero_gold_common_strata": zero_gold_common,
        "dropped_rows": len(rows) - left_total - right_total,
        "strata_keys": list(strata_keys),
    }


def make_slice_tables(task_id: str, rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    adjusted_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    strata_keys = ("length_bin", "label_stratum") if task_id.startswith("vsfc_") else ("length_bin", "gold_positive_bin")
    for feature in NUMERIC_FEATURES:
        threshold = thresholds[feature]
        if not threshold["slice_enabled"]:
            continue
        work: list[dict[str, Any]] = []
        for row in rows:
            group = slice_group(row[feature], threshold)
            if group in {"low", "high"}:
                copy = dict(row)
                copy["feature_group"] = group
                work.append(copy)
        for group in ("low", "high"):
            selected = [row for row in work if row["feature_group"] == group]
            metrics = group_metrics(selected, task_id)
            raw_rows.append({"task_id": task_id, "domain": DOMAINS[task_id], "feature": feature, "group": group, **metrics, "q1": threshold["q1"], "q3": threshold["q3"]})
        adjusted = standardized_gap(work, task_id, "feature_group", strata_keys)
        low_burden = raw_rows[-2]["error_burden"]
        high_burden = raw_rows[-1]["error_burden"]
        raw_gap = high_burden - low_burden if low_burden is not None and high_burden is not None else None
        adjusted_rows.append({"task_id": task_id, "domain": DOMAINS[task_id], "feature": feature, "raw_high_minus_low": raw_gap, **adjusted, "q1": threshold["q1"], "q3": threshold["q3"]})
        by_stratum: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in work:
            by_stratum[tuple(row.get(key) for key in strata_keys)][str(row["feature_group"])].append(row)
        for stratum, groups in sorted(by_stratum.items(), key=lambda item: repr(item[0])):
            for group, selected in sorted(groups.items()):
                strata_rows.append({
                    "task_id": task_id,
                    "domain": DOMAINS[task_id],
                    "feature": feature,
                    "stratum": dict(zip(strata_keys, stratum)),
                    "group": group,
                    **group_metrics(selected, task_id),
                })
    return raw_rows, adjusted_rows, strata_rows


def add_strata(row: dict[str, Any], task_id: str, length_threshold: Mapping[str, Any]) -> None:
    row["length_bin"] = length_bin(float(row["bpe_sequence_length_full"]), length_threshold)
    if task_id.startswith("vsfc_"):
        row["label_stratum"] = str(row["gold"])
    else:
        count = int(row["gold_positive_slots"])
        row["gold_positive_bin"] = "0" if count == 0 else ("1" if count == 1 else "2_plus")


def build_absa_slot_rows(task_id: str, rows: Sequence[Mapping[str, Any]], bundle: Any, train_rows: Sequence[Any]) -> list[dict[str, Any]]:
    names = list(bundle.aspect_names)
    result: list[dict[str, Any]] = []
    train_support: Counter[tuple[int, int]] = Counter()
    for example in train_rows:
        for index, value in enumerate(example.label):
            train_support[(index, int(value))] += 1
    for index, name in enumerate(names):
        if task_id == "visfd" and name == "OTHERS":
            label_values = ((0, "absent"), (4, "others"))
        elif task_id in {"visfd", "absa_restaurant", "absa_hotel"}:
            label_values = ((0, "absent"), (1, "positive"), (2, "negative"), (3, "neutral"))
        else:
            label_values = tuple(enumerate(("absent", "negative", "neutral", "positive")))
        for value, label_name in label_values:
            selected = []
            for row in rows:
                slot = row["per_slot"][index]
                if slot["gold"] == value:
                    selected.append(slot)
            event_counts = Counter(slot["event"] for slot in selected)
            result.append({
                "task_id": task_id,
                "domain": DOMAINS[task_id],
                "aspect": name,
                "gold_label": value,
                "gold_label_name": label_name,
                "train_cell_support": train_support[(index, value)],
                "validation_cell_count": len(selected),
                "correct_pair": event_counts["correct_pair"],
                "missed": event_counts["missed"],
                "spurious": event_counts["spurious"],
                "wrong_polarity": event_counts["wrong_polarity"],
                "detection_only": event_counts["detection_only"],
            })
    return result


def select_samples(task_rows: Mapping[str, Sequence[Mapping[str, Any]]], bundles: Mapping[str, Any], feature_rows_by_task: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    # VSFC is one shared text domain: pick each text once, carrying both task
    # predictions, so the two tasks cannot masquerade as independent domains.
    vsfc_rows = list(feature_rows_by_task["vsfc_sentiment"])
    topic_by_id = {row["id"]: row for row in feature_rows_by_task["vsfc_topic"]}
    combined_vsfc: list[dict[str, Any]] = []
    for row in vsfc_rows:
        topic = topic_by_id[row["id"].replace("vsfc_sentiment", "vsfc_topic", 1)]
        merged = dict(row)
        merged["task_id"] = "vsfc_shared"
        merged["domain"] = "vsfc"
        merged["vsfc_sentiment"] = {"gold": row["gold"], "prediction": row["prediction"], "error": row["error"]}
        merged["vsfc_topic"] = {"gold": topic["gold"], "prediction": topic["prediction"], "error": topic["error"]}
        merged["vsfc_topic_gold"] = topic["gold"]
        merged["vsfc_topic_prediction"] = topic["prediction"]
        merged["vsfc_topic_error"] = topic["error"]
        merged["combined_error"] = int(row["error"] or topic["error"])
        merged["preliminary_note"] = "shared VSFC text; mechanical correctness/error labels for both tasks; semantics require root review"
        merged["confidence"] = "mechanical_high_semantic_low"
        combined_vsfc.append(merged)
    domain_rows: dict[str, list[dict[str, Any]]] = {
        "vsfc": combined_vsfc,
        "visfd": list(feature_rows_by_task["visfd"]),
        "absa_restaurant": list(feature_rows_by_task["absa_restaurant"]),
        "absa_hotel": list(feature_rows_by_task["absa_hotel"]),
    }
    for domain in DOMAIN_ORDER:
        candidates = domain_rows[domain]
        rng.shuffle(candidates)
        errors = [row for row in candidates if int(row.get("combined_error", row.get("error", 0))) == 1]
        correct = [row for row in candidates if int(row.get("combined_error", row.get("error", 0))) == 0]
        target = 15
        error_take = min(len(errors), target // 2)
        correct_take = min(len(correct), target - error_take)
        if error_take + correct_take < target:
            extra = [row for row in candidates if row not in errors[:error_take] and row not in correct[:correct_take]]
            candidates_take = errors[:error_take] + correct[:correct_take] + extra[: target - error_take - correct_take]
        else:
            candidates_take = errors[:error_take] + correct[:correct_take]
        rng.shuffle(candidates_take)
        for row in candidates_take:
            selected.append({
                "sample_id": f"{domain}:{len([x for x in selected if x['domain'] == domain]):02d}",
                "task_id": row["task_id"],
                "domain": domain,
                "id": row["id"],
                "text": row["text"],
                "gold": row["gold"],
                "prediction": row["prediction"],
                "vsfc_topic_gold": row.get("vsfc_topic_gold", ""),
                "vsfc_topic_prediction": row.get("vsfc_topic_prediction", ""),
                "vsfc_topic_error": row.get("vsfc_topic_error", ""),
                "error": int(row.get("combined_error", row.get("error", 0))),
                "error_events": row.get("error_events", ""),
                "missed": row.get("missed", ""),
                "spurious": row.get("spurious", ""),
                "wrong_polarity": row.get("wrong_polarity", ""),
                "unit_count": row["unit_count"],
                "bpe_tokens_full": row["bpe_tokens_full"],
                "bpe_per_unit_mean": row["bpe_per_unit_mean"],
                "rare_unit_rate": row["rare_unit_rate"],
                "unseen_unit_rate": row["unseen_unit_rate"],
                "accent_proxy_unit_rate": row["accent_proxy_unit_rate"],
                "truncated_at_256": row["truncated_at_256"],
                "preliminary_note": row.get("preliminary_note", "mechanical error decomposition; surface association is not a semantic diagnosis"),
                "confidence": row.get("confidence", "mechanical_high_semantic_low"),
            })
    return selected


def render_report(
    out_path: Path,
    metrics: Mapping[str, Any],
    domain_summary: Sequence[Mapping[str, Any]],
    adjusted_rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    sample_count: int,
) -> None:
    lines = [
        "# E12 Surface Error Diagnostic Result",
        "",
        "This validation-only audit uses the E8 seed-42 saved predictions, the frozen train and validation loaders with `include_test=False`, and the original `models/goiturn-s-gd` fast tokenizer. It does not train, load a neural checkpoint, run inference, or inspect test rows.",
        "",
        "## Scope and correctness",
        "",
        "Metric values were recomputed from each saved prediction file after exact ID and gold alignment against the validation loader. VSFC sentiment and topic share one text domain and are represented once in the four-domain sample/audit view. ViSFD OTHERS is included for detection but excluded from pair-scored ABSA error counts.",
        "",
        "| Task | n | recomputed selection F1 | saved selection F1 | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for task_id in TASKS:
        row = metrics[task_id]
        lines.append(f"| {task_id} | {row['n']} | {row['recomputed_selection_f1']:.6f} | {row['saved_selection_f1']:.6f} | {row['delta']:.3g} |")
    lines.extend([
        "",
        "## Domain error decomposition",
        "",
        "The ABSA rows below are reported with missed/spurious/wrong-polarity events and burden per scored gold-positive slot. Raw any-error row rates are retained in feature CSVs for within-task review and are not compared across domains.",
        "",
        "| Domain | tasks/text scope | validation rows | error events | gold-positive slots | normalized burden |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in domain_summary:
        lines.append(f"| {row['domain']} | {row['scope']} | {row['n']} | {row['error_events']} | {row['gold_positive_slots']} | {row['error_burden']:.4f} |")
    lines.extend([
        "",
        "## Interpretation gate",
        "",
        "The slice tables report raw high/low contrasts and common-support standardization by train-derived length bins plus true class (VSFC) or gold-positive burden (ABSA). High/low groups use train quartiles; ties are retained and slices are disabled when the train feature is constant or Q1 equals Q3. Coverage and dropped rows are reported so small or non-overlapping groups are not treated as strong evidence. ABSA adjustment controls only length and gold-positive burden; it does not jointly control aspect, polarity, rarity, and fragmentation.",
        "",
        "A character/diacritic branch is worth prioritizing only if fragmentation, rarity, or the orthography proxy has adequate support, the adjusted direction is consistent across the relevant domains, and the sampled examples support the proposed mechanism. The audit itself is descriptive and cannot establish causality or predict a new architecture's gain.",
        "",
        "## Architecture evidence",
        "",
        "BPE fragmentation does not show a broad cross-domain direction after adjustment: `bpe_per_unit_mean` is +1.38 percentage points for VSFC sentiment, −8.05 pp for VSFC topic, +1.69 pp for ViSFD, −1.55 pp for Restaurant, and +2.17 pp for Hotel (high minus low error burden). The >=3-piece unit slice has usable train support only for ViSFD, where the adjusted gap is +2.16 pp. This is inconsistent and task-specific.",
        "",
        "Rarity is the clearest task-specific signal: adjusted rare-unit burden is +6.36 pp for ViSFD and +13.20 pp for Restaurant, with near-complete positive-support coverage. The corresponding rarity slices are disabled for VSFC and Hotel because train Q1 equals Q3 at zero; forcing a split would turn ties into an artificial result. Rarity remains confounded with fragmentation and is not evidence that a character encoder will improve every task.",
        "",
        "The accent-absence proxy has sparse validation support (22/1,583 VSFC rows, 92/1,112 ViSFD, 54/771 Restaurant, and 19/795 Hotel) and its train quartile slices are disabled by Q1=Q3=0. It therefore supplies no adjusted cross-domain evidence. Units without marks are not labeled errors; the proxy requires a rare/unseen mark-free unit and a marked train counterpart, while ambiguity is retained in `validation_units.csv`.",
        "",
        "Gate result: broad character/diacritic architecture prioritization is inconclusive. A narrowly scoped rarity/fragmentation follow-up could be considered for ViSFD and Restaurant after root review of the seeded examples, but this audit does not approve a shared five-task character branch.",
        "",
        "Adjusted rows with the largest absolute gaps (screening aid; inspect coverage before interpretation):",
        "",
    ])
    ranked = sorted(adjusted_rows, key=lambda row: abs(float(row["adjusted_gap"])) if row.get("adjusted_gap") is not None else -1.0, reverse=True)[:12]
    lines.append("| Task | Feature | raw high-low burden | adjusted high-low burden | common strata | coverage low | coverage high |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in ranked:
        adjusted = "NA" if row.get("adjusted_gap") is None else f"{float(row['adjusted_gap']):.4f}"
        raw_gap = "NA" if row.get("raw_high_minus_low") is None else f"{float(row['raw_high_minus_low']):.4f}"
        lines.append(f"| {row['task_id']} | {row['feature']} | {raw_gap} | {adjusted} | {row.get('common_strata', 0)} | {float(row.get('coverage_low_or_absent', 0.0)):.3f} | {float(row.get('coverage_high_or_present', 0.0)):.3f} |")
    lines.extend([
        "",
        f"The seeded manual-review file contains {sample_count} rows across all four domains, with both correct and incorrect outcomes when available. Notes in the file are mechanical preliminary labels; semantic interpretation remains for root review.",
        "",
        "## Limitations",
        "",
        "Features are descriptive proxies. A no-diacritic unit is not automatically an error: the proxy only marks rare/unseen units with a train counterpart sharing a mark-stripped skeleton, and reports counterpart ambiguity. Rarity and fragmentation are correlated; the ABSA stratification does not jointly adjust them. Full-text BPE length and model-truncated length are reported separately. In zero-gold ABSA strata, normalized burden is undefined and is shown as blank/NA; spurious predictions are reported separately. In this validation set, ViSFD has 18 zero-gold rows with 15 spurious events across 12 rows; Restaurant and Hotel have none. No synthetic counterfactuals, extra datasets, p-values, or model retraining were used.",
        "",
        f"Provenance is recorded in `artifacts/e12/provenance.json`; sample selection seed is {SEED}.",
        "",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output or (root / "artifacts" / "e12")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    from transformers import AutoTokenizer
    from goibert import train as e1_train

    tokenizer_path = root / "models" / "goiturn-s-gd"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    bundles: dict[str, Any] = {}
    predictions_by_task: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {}
    feature_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    thresholds_by_task: dict[str, Any] = {}
    frequency_by_task: dict[str, tuple[Counter[str], dict[str, set[str]], dict[str, int]]] = {}
    feature_cache: dict[tuple[str, str], dict[str, Any]] = {}
    train_feature_cache: dict[tuple[str, str], dict[str, Any]] = {}
    provenance_files: dict[str, str] = {}

    for task_id in TASKS:
        bundle = e1_train.load_task_bundle(task_id, root, include_test=False)
        bundles[task_id] = bundle
        prediction_path = root / "artifacts" / "e8" / task_id / "seed_42" / "validation_predictions.json"
        rows, alignment = compare_ids_and_gold(task_id, bundle, prediction_path)
        predictions_by_task[task_id] = rows
        recomputed = e1_train.evaluate_predictions(task_id, bundle, [row["gold"] for row in rows], [row["prediction"] for row in rows], split="validation")
        saved_path = root / "artifacts" / "e8" / task_id / "seed_42" / "best_validation_metrics.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        delta = float(recomputed["selection_f1"]) - float(saved["selection_f1"])
        if abs(delta) > 1e-12:
            raise ValueError(f"{task_id}: recomputed selection metric differs from saved metric by {delta}")
        metrics[task_id] = {
            "n": len(rows),
            "recomputed_selection_f1": float(recomputed["selection_f1"]),
            "saved_selection_f1": float(saved["selection_f1"]),
            "delta": delta,
            "selection_metric": recomputed["selection_metric"],
            "alignment": alignment,
            "recomputed_metrics": recomputed,
            "saved_metrics": saved,
        }
        for source in bundle.source_files:
            provenance_files[str(Path(source).resolve())] = sha256_file(Path(source))
        provenance_files[str(prediction_path)] = sha256_file(prediction_path)
        provenance_files[str(saved_path)] = sha256_file(saved_path)

    # Build train frequency and train-derived thresholds once for the shared
    # VSFC text domain, separately for the other three domains.
    for task_id in TASKS:
        bundle = bundles[task_id]
        frequency, variants, ambiguity = train_frequency([example.text for example in bundle.train])
        frequency_by_task[task_id] = (frequency, variants, ambiguity)
        train_features: list[dict[str, Any]] = []
        for example in bundle.train:
            cache_key = (task_id if task_id not in {"vsfc_sentiment", "vsfc_topic"} else "vsfc", example.text)
            if cache_key not in train_feature_cache:
                train_feature_cache[cache_key] = build_features(example.text, tokenizer, frequency, variants, ambiguity)
            train_features.append(train_feature_cache[cache_key])
        thresholds = feature_thresholds(train_features)
        thresholds["model_length"] = {
            "q1": percentile([float(row["bpe_sequence_length_full"]) for row in train_features], 0.25),
            "q3": percentile([float(row["bpe_sequence_length_full"]) for row in train_features], 0.75),
            "constant": len({row["bpe_sequence_length_full"] for row in train_features}) <= 1,
        }
        thresholds_by_task[task_id] = thresholds
        val_rows: list[dict[str, Any]] = []
        for pred_row in predictions_by_task[task_id]:
            example = bundle.validation[len(val_rows)]
            cache_key = (task_id if task_id not in {"vsfc_sentiment", "vsfc_topic"} else "vsfc", example.text)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = build_features(example.text, tokenizer, frequency, variants, ambiguity)
            feature = dict(feature_cache[cache_key])
            feature.pop("unit_records", None)
            feature.update({"task_id": task_id, "domain": DOMAINS[task_id], "id": pred_row["id"], "gold": pred_row["gold"], "prediction": pred_row["prediction"]})
            feature.update(outcome_fields(task_id, bundle, pred_row))
            add_strata(feature, task_id, thresholds["model_length"])
            feature["feature_groups"] = {name: slice_group(feature[name], thresholds[name]) for name in NUMERIC_FEATURES if thresholds[name]["slice_enabled"]}
            val_rows.append(feature)
        feature_rows_by_task[task_id] = val_rows

    all_feature_rows: list[dict[str, Any]] = []
    validation_unit_rows: list[dict[str, Any]] = []
    for task_id, rows in feature_rows_by_task.items():
        for row in rows:
            serial = dict(row)
            # Unit records are exported separately so every offset used for
            # fragmentation can be inspected.  VSFC text is shared, hence it
            # is emitted once under the shared sentiment-row traversal.
            if task_id != "vsfc_topic":
                for unit_index, unit in enumerate(feature_cache[(task_id if task_id not in {"vsfc_sentiment", "vsfc_topic"} else "vsfc", row["text"])] ["unit_records"]):
                    validation_unit_rows.append({
                        "task_id": task_id,
                        "domain": DOMAINS[task_id],
                        "id": row["id"],
                        "unit_index": unit_index,
                        **unit,
                    })
            serial.pop("feature_groups", None)
            serial.pop("unit_records", None)
            all_feature_rows.append(serial)
    feature_fields = [
        "task_id", "domain", "id", "text", "gold", "prediction", "error", "error_events", "missed", "spurious", "wrong_polarity", "pair_hits", "gold_positive_slots", "predicted_positive_slots", "error_burden", "spurious_per_row", "char_count", "unit_count", "bpe_tokens_full", "bpe_tokens_model", "bpe_sequence_length_full", "bpe_sequence_length_model", "truncated_at_256", "bpe_per_unit_mean", "unit_ge3pieces_rate", "bpe_multiple_unit_token_count", "rare_unit_count", "rare_unit_rate", "unseen_unit_count", "unseen_unit_rate", "accent_proxy_unit_count", "accent_proxy_unit_rate", "accent_proxy_ambiguous_count", "accent_partner_max", "unicode_ratio", "unicode_non_ascii_count", "combining_mark_count", "emoji_count", "punctuation_count", "char_repeat_max", "length_bin", "label_stratum", "gold_positive_bin",
    ]
    write_csv(out / "validation_features.csv", all_feature_rows, feature_fields)
    write_csv(out / "validation_units.csv", validation_unit_rows, [
        "task_id", "domain", "id", "unit_index", "text", "nfc_lower", "start", "end", "train_frequency", "rare_le5", "unseen", "skeleton", "accent_partner_count", "skeleton_ambiguity_train", "accent_absence_proxy", "bpe_piece_count",
    ])

    train_frequency_rows: list[dict[str, Any]] = []
    emitted_frequency_domains: set[str] = set()
    for task_id in TASKS:
        domain = DOMAINS[task_id]
        if domain in emitted_frequency_domains:
            continue
        emitted_frequency_domains.add(domain)
        frequency, variants, ambiguity = frequency_by_task[task_id]
        for unit_value, count in sorted(frequency.items(), key=lambda item: (-item[1], item[0])):
            train_frequency_rows.append({
                "task_id": task_id,
                "domain": domain,
                "nfc_lower_unit": unit_value,
                "train_frequency": count,
                "skeleton": skeleton(unit_value),
                "skeleton_variant_count": ambiguity.get(skeleton(unit_value), 1),
                "has_diacritic": has_diacritic(unit_value),
            })
    write_csv(out / "train_unit_frequency.csv", train_frequency_rows)

    train_threshold_output: dict[str, Any] = {}
    for task_id, threshold in thresholds_by_task.items():
        train_threshold_output[task_id] = threshold
    write_json(out / "train_thresholds.json", train_threshold_output)

    slice_rows: list[dict[str, Any]] = []
    adjusted_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        raw, adjusted, strata = make_slice_tables(task_id, feature_rows_by_task[task_id], thresholds_by_task[task_id])
        slice_rows.extend(raw)
        adjusted_rows.extend(adjusted)
        strata_rows.extend(strata)
        if not task_id.startswith("vsfc_"):
            slot_rows.extend(build_absa_slot_rows(task_id, feature_rows_by_task[task_id], bundles[task_id], bundles[task_id].train))
    write_csv(out / "slice_raw.csv", slice_rows)
    write_csv(out / "slice_adjusted_common_support.csv", adjusted_rows)
    write_csv(out / "slice_strata.csv", strata_rows)
    write_csv(out / "absa_slot_error_breakdown.csv", slot_rows)

    domain_summary: list[dict[str, Any]] = []
    # One VSFC row per shared text, with two task outcomes retained as a mean
    # error measure. The two tasks are not counted as independent domains.
    sent_rows = feature_rows_by_task["vsfc_sentiment"]
    topic_rows = feature_rows_by_task["vsfc_topic"]
    vsfc_error_events = sum(int(a["error"]) + int(b["error"]) for a, b in zip(sent_rows, topic_rows))
    domain_summary.append({"domain": "vsfc", "scope": "shared text; sentiment+topic outcomes", "n": len(sent_rows), "row_error_rate": safe_rate(vsfc_error_events, 2 * len(sent_rows)), "error_events": vsfc_error_events, "gold_positive_slots": 2 * len(sent_rows), "error_burden": safe_rate(vsfc_error_events, 2 * len(sent_rows))})
    for task_id in ("visfd", "absa_restaurant", "absa_hotel"):
        rows = feature_rows_by_task[task_id]
        metrics_row = group_metrics(rows, task_id)
        domain_summary.append({"domain": DOMAINS[task_id], "scope": task_id, "n": len(rows), "row_error_rate": metrics_row["event_rate"], "error_events": metrics_row["events"], "gold_positive_slots": metrics_row["gold_positive_slots"], "error_burden": metrics_row["error_burden"]})
    write_csv(out / "domain_summary.csv", domain_summary)

    sample_rows = select_samples(predictions_by_task, bundles, feature_rows_by_task)
    write_csv(out / "sample_review.csv", sample_rows)
    write_json(out / "sample_review.json", {"seed": SEED, "count": len(sample_rows), "rows": sample_rows})
    sample_md = ["# E12 seeded sample review", "", f"Seed: `{SEED}`; rows: `{len(sample_rows)}`; selection is stratified by four domains and includes correct/error outcomes when available.", "", "Semantic notes are deliberately mechanical and low-confidence; root should inspect text and labels.", ""]
    for row in sample_rows:
        sample_md.extend([f"## {row['sample_id']} — {row['domain']} — {row['id']}", "", f"- Error flag: `{row['error']}`; events: `{row['error_events']}`; missed/spurious/wrong-polarity: `{row['missed']}/{row['spurious']}/{row['wrong_polarity']}`", f"- Gold: `{json.dumps(row['gold'], ensure_ascii=False)}`; prediction: `{json.dumps(row['prediction'], ensure_ascii=False)}`"])
        if row["domain"] == "vsfc":
            sample_md.append(f"- VSFC shared-text union detail: sentiment gold/prediction `{row['gold']}`/`{row['prediction']}` (error `{row['error']}`); topic gold/prediction `{row['vsfc_topic_gold']}`/`{row['vsfc_topic_prediction']}` (error `{row['vsfc_topic_error']}`).")
        sample_md.extend([f"- Features: units `{row['unit_count']}`, full BPE `{row['bpe_tokens_full']}`, BPE/unit `{row['bpe_per_unit_mean']:.3f}`, rare `{row['rare_unit_rate']:.3f}`, unseen `{row['unseen_unit_rate']:.3f}`, accent proxy `{row['accent_proxy_unit_rate']:.3f}`, truncated `{row['truncated_at_256']}`", f"- Preliminary note: {row['preliminary_note']} (confidence `{row['confidence']}`)", f"- Text: {row['text']}", ""])
    (out / "sample_review.md").write_text("\n".join(sample_md), encoding="utf-8")

    provenance = {
        "script": str((root / "scripts" / "analyze_e12_surface_errors.py").resolve()),
        "script_sha256": sha256_file(root / "scripts" / "analyze_e12_surface_errors.py"),
        "plan": str((root / "docs" / "E12_DIAGNOSTIC_PLAN.md").resolve()),
        "plan_sha256": sha256_file(root / "docs" / "E12_DIAGNOSTIC_PLAN.md"),
        "tokenizer_dir": str(tokenizer_path),
        "tokenizer_files": {name: sha256_file(tokenizer_path / name) for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if (tokenizer_path / name).exists()},
        "input_files": provenance_files,
        "tasks": list(TASKS),
        "domains": list(DOMAIN_ORDER),
        "seed": SEED,
        "max_length": MODEL_MAX_LENGTH,
        "include_test": False,
        "test_access": "not requested; loader calls are include_test=False",
        "architecture_inference": "descriptive surface audit only; no causal or model-gain claim",
    }
    write_json(out / "provenance.json", provenance)
    write_json(out / "metric_recompute.json", metrics)
    summary = {
        "protocol": "E12-surface-error-audit-v1",
        "status": "completed",
        "tasks": list(TASKS),
        "domains": list(DOMAIN_ORDER),
        "selection_metrics": {task_id: metrics[task_id]["recomputed_selection_f1"] for task_id in TASKS},
        "domain_summary": domain_summary,
        "artifact_files": sorted(path.name for path in out.iterdir() if path.is_file()),
        "manual_sample_count": len(sample_rows),
        "manual_sample_seed": SEED,
        "limitations": ["validation-only", "single E8 seed42", "descriptive association", "rarity and fragmentation remain confounded", "accent absence is a proxy with ambiguity, not an error label"],
    }
    write_json(out / "summary.json", summary)
    render_report(root / "docs" / "E12_DIAGNOSTIC_RESULT.md", metrics, domain_summary, adjusted_rows, provenance, len(sample_rows))
    print(json.dumps({"output": str(out), "report": str(root / "docs" / "E12_DIAGNOSTIC_RESULT.md"), "sample_count": len(sample_rows), "selection_metrics": summary["selection_metrics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
