#!/usr/bin/env python3
"""Run the reviewed E13A binding diagnostic on saved E8/E11 checkpoints.

This is an inference-only diagnostic over 30 root-reviewed pairs (60 text
occurrences).  It never loads the test split, never trains, and never changes
E1--E11 code or checkpoints.  Expectations are explicit numeric source codes
and do not come from ``TaskDataset.label_names``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
MODELS = ("e8", "e11_signed", "e11_convex")
MAX_LENGTH = 256
BATCH_SIZE = 4


def pair(task_id: str, pair_id: str, kind: str, text_a: str, text_b: str, expected_a: Mapping[str, int], expected_b: Mapping[str, int]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "pair_id": pair_id,
        "kind": kind,
        "texts": {"A": text_a, "B": text_b},
        "expected": {"A": dict(expected_a), "B": dict(expected_b)},
    }


PAIRS = [
    pair("vsfc_sentiment", "S1", "paraphrase", "Thầy giảng bài rất dễ hiểu.", "Bài giảng của thầy rất dễ hiểu.", {"label": 2}, {"label": 2}),
    pair("vsfc_sentiment", "S2", "negation", "Thầy giảng bài dễ hiểu.", "Thầy giảng bài không dễ hiểu.", {"label": 2}, {"label": 0}),
    pair("vsfc_sentiment", "S3", "negated_negative", "Thầy dạy rất khó hiểu.", "Thầy dạy không hề khó hiểu.", {"label": 0}, {"label": 2}),
    pair("vsfc_sentiment", "S4", "desire_scope", "Mong cô tiếp tục cách dạy dễ hiểu này.", "Mong cô thay đổi cách dạy khó hiểu này.", {"label": 2}, {"label": 0}),
    pair("vsfc_sentiment", "S5", "speaker_contrast", "Có người nói cô dạy khó hiểu, nhưng tôi thấy cô dạy rất dễ hiểu.", "Có người nói cô dạy dễ hiểu, nhưng tôi thấy cô dạy rất khó hiểu.", {"label": 2}, {"label": 0}),
    pair("vsfc_sentiment", "S6", "reported_disagreement_control", "Tôi thấy cô dạy rất dễ hiểu, dù có người chê cô dạy khó hiểu.", "Dù có người chê cô dạy khó hiểu, tôi vẫn thấy cô dạy rất dễ hiểu.", {"label": 2}, {"label": 2}),
    pair("vsfc_topic", "T1", "paraphrase", "Máy chiếu trong phòng học bị hỏng.", "Phòng học có máy chiếu bị hỏng.", {"label": 2}, {"label": 2}),
    pair("vsfc_topic", "T2", "entity_context", "Máy chiếu trong phòng học bị hỏng.", "Máy chiếu trong phòng học do thầy phụ trách bị hỏng.", {"label": 2}, {"label": 2}),
    pair("vsfc_topic", "T3", "target_change", "Giảng viên thường xuyên vắng mặt trong buổi thực hành.", "Các buổi thực hành bị cắt khỏi chương trình đào tạo.", {"label": 0}, {"label": 1}),
    pair("vsfc_topic", "T4", "negation_preserves_topic", "Giảng viên dạy nhiệt tình.", "Giảng viên dạy không nhiệt tình.", {"label": 0}, {"label": 0}),
    pair("vsfc_topic", "T5", "target_change", "Nội dung chương trình đào tạo quá nặng.", "Máy chiếu của phòng học quá mờ.", {"label": 1}, {"label": 2}),
    pair("vsfc_topic", "T6", "event_paraphrase", "Giáo viên không đến lớp trong buổi thực hành.", "Buổi thực hành diễn ra nhưng giáo viên vắng mặt.", {"label": 0}, {"label": 0}),
    pair("visfd", "V1", "swap_binding", "Pin dùng rất lâu nhưng camera chụp ảnh rất xấu.", "Camera chụp ảnh rất đẹp nhưng pin hết rất nhanh.", {"BATTERY": 1, "CAMERA": 2}, {"BATTERY": 2, "CAMERA": 1}),
    pair("visfd", "V2", "clause_reorder", "Pin dùng rất lâu nhưng camera chụp ảnh rất xấu.", "Camera chụp ảnh rất xấu nhưng pin dùng rất lâu.", {"BATTERY": 1, "CAMERA": 2}, {"BATTERY": 1, "CAMERA": 2}),
    pair("visfd", "V3", "accessory_target", "Pin của điện thoại bị hỏng, còn sạc dự phòng được tặng thì dùng tốt.", "Pin của điện thoại dùng tốt, còn sạc dự phòng được tặng thì bị hỏng.", {"BATTERY": 2, "SER&ACC": 1}, {"BATTERY": 1, "SER&ACC": 2}),
    pair("visfd", "V4", "negation", "Camera chụp ảnh đẹp.", "Camera chụp ảnh không đẹp.", {"CAMERA": 1}, {"CAMERA": 2}),
    pair("visfd", "V5", "added_other_target", "Pin máy dùng rất lâu.", "Pin máy dùng rất lâu, nhưng nhân viên phục vụ thiếu nhiệt tình.", {"BATTERY": 1}, {"BATTERY": 1, "SER&ACC": 2}),
    pair("visfd", "V6", "paraphrase", "Nhân viên bán hàng rất nhiệt tình.", "Người bán hàng phục vụ rất tận tình.", {"SER&ACC": 1}, {"SER&ACC": 1}),
    pair("absa_restaurant", "R1", "swap_binding", "Món ăn ngon nhưng nhân viên phục vụ rất bất lịch sự.", "Món ăn dở nhưng nhân viên phục vụ rất lịch sự.", {"FOOD#QUALITY": 1, "SERVICE#GENERAL": 2}, {"FOOD#QUALITY": 2, "SERVICE#GENERAL": 1}),
    pair("absa_restaurant", "R2", "clause_reorder", "Món ăn ngon nhưng nhân viên phục vụ rất bất lịch sự.", "Nhân viên phục vụ rất bất lịch sự nhưng món ăn ngon.", {"FOOD#QUALITY": 1, "SERVICE#GENERAL": 2}, {"FOOD#QUALITY": 1, "SERVICE#GENERAL": 2}),
    pair("absa_restaurant", "R3", "role_paraphrase", "Nhân viên phục vụ rất thân thiện.", "Anh quản lý phục vụ khách rất thân thiện.", {"SERVICE#GENERAL": 1}, {"SERVICE#GENERAL": 1}),
    pair("absa_restaurant", "R4", "negation", "Đồ uống có hương vị ngon.", "Đồ uống có hương vị không ngon.", {"DRINKS#QUALITY": 1}, {"DRINKS#QUALITY": 2}),
    pair("absa_restaurant", "R5", "added_other_target", "Không gian quán rất sạch sẽ.", "Không gian quán rất sạch sẽ nhưng món ăn có mùi hôi.", {"AMBIENCE#GENERAL": 1}, {"AMBIENCE#GENERAL": 1, "FOOD#QUALITY": 2}),
    pair("absa_restaurant", "R6", "target_change", "Giá món ăn quá đắt.", "Giá đồ uống quá đắt.", {"FOOD#PRICES": 2}, {"DRINKS#PRICES": 2}),
    pair("absa_hotel", "H1", "swap_binding", "Phòng rất sạch nhưng nhân viên phục vụ bất lịch sự.", "Phòng rất bẩn nhưng nhân viên phục vụ lịch sự.", {"ROOMS#CLEANLINESS": 1, "SERVICE#GENERAL": 2}, {"ROOMS#CLEANLINESS": 2, "SERVICE#GENERAL": 1}),
    pair("absa_hotel", "H2", "clause_reorder", "Phòng rất sạch nhưng nhân viên phục vụ bất lịch sự.", "Nhân viên phục vụ bất lịch sự nhưng phòng rất sạch.", {"ROOMS#CLEANLINESS": 1, "SERVICE#GENERAL": 2}, {"ROOMS#CLEANLINESS": 1, "SERVICE#GENERAL": 2}),
    pair("absa_hotel", "H3", "paraphrase", "Phòng ở được vệ sinh sạch sẽ.", "Phòng ốc rất sạch sẽ.", {"ROOMS#CLEANLINESS": 1}, {"ROOMS#CLEANLINESS": 1}),
    pair("absa_hotel", "H4", "negation", "Nhân viên lễ tân lịch sự.", "Nhân viên lễ tân không lịch sự.", {"SERVICE#GENERAL": 1}, {"SERVICE#GENERAL": 2}),
    pair("absa_hotel", "H5", "added_other_target", "Bữa sáng rất ngon.", "Bữa sáng rất ngon nhưng phòng ngủ rất bẩn.", {"FOOD&DRINKS#QUALITY": 1}, {"FOOD&DRINKS#QUALITY": 1, "ROOMS#CLEANLINESS": 2}),
    pair("absa_hotel", "H6", "target_change", "Giá phòng quá đắt.", "Giá bữa sáng quá đắt.", {"ROOMS#PRICES": 2}, {"FOOD&DRINKS#PRICES": 2}),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(f"Checkpoint lacks model state: {path}")
    return payload


def raw_prediction(task_id: str, prediction: Sequence[int], aspect_names: Sequence[str]) -> list[int]:
    values = [int(value) for value in prediction]
    if task_id == "visfd":
        index = list(aspect_names).index("OTHERS")
        values[index] = 4 if values[index] else 0
    return values


def model_prediction(task_id: str, logits: torch.Tensor, aspect_names: Sequence[str], counts: Sequence[int]) -> tuple[list[int], list[int]]:
    if not counts:
        prediction = [int(logits.argmax(dim=-1).item())]
        return prediction, prediction
    if logits.ndim == 1 and len(set(counts)) != 1:
        values = []
        offset = 0
        for width in counts:
            values.append(int(logits[offset : offset + int(width)].argmax().item()))
            offset += int(width)
    elif logits.ndim == 1:
        if len(counts) != 1:
            raise ValueError("Flat ABSA logits with multiple equal-width slots require explicit split metadata")
        values = [int(logits.argmax().item())]
    else:
        values = [int(slot.argmax().item()) for slot in logits]
    return values, raw_prediction(task_id, values, aspect_names)


def load_bundle(task_id: str) -> Any:
    from goibert import train as e1_train

    return e1_train.load_task_bundle(task_id, ROOT, include_test=False)


def build_model(task_id: str, model_name: str):
    from transformers import AutoTokenizer
    from goibert_e2.modeling import build_e2_student
    from goibert_e11.modeling import build_e11_student

    bundle = load_bundle(task_id)
    counts = tuple(bundle.aspect_class_counts)
    num_aspects = int(bundle.num_aspects)
    num_labels = int(bundle.num_labels)
    if model_name == "e8":
        model, spec = build_e2_student(ROOT / "models" / "goiturn-s-gd", num_classes=num_labels, num_aspects=num_aspects, aspect_class_counts=counts if num_aspects else None, attention_implementation="sdpa")
        checkpoint_path = ROOT / "artifacts" / "e8" / task_id / "seed_42" / "best_checkpoint.pt"
    else:
        variant = "signed" if model_name == "e11_signed" else "convex"
        model, spec = build_e11_student(ROOT / "models" / "goiturn-s-gd", variant=variant, num_classes=num_labels, num_aspects=num_aspects, aspect_class_counts=counts if num_aspects else None, attention_implementation="sdpa", warmup_steps=1)
        checkpoint_path = ROOT / "artifacts" / "e11" / variant / task_id / "seed_42" / "best_checkpoint.pt"
    payload = state_dict(checkpoint_path)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, bundle, checkpoint_path


def infer_model(model: Any, task_id: str, bundle: Any, occurrences: Sequence[Mapping[str, Any]], tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    from goibert.train import make_autocast

    texts = [str(row["text"]) for row in occurrences]
    counts = tuple(bundle.aspect_class_counts)
    aspect_names = list(bundle.aspect_names)
    output: list[dict[str, Any]] = []
    precision = "bf16" if device.type == "cuda" else "fp32"
    with torch.inference_mode(), make_autocast(precision, device):
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start : start + BATCH_SIZE]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}
            logits = model(encoded["input_ids"], encoded["attention_mask"])
            for index in range(len(batch_texts)):
                row_logits = logits[index].detach().float().cpu()
                model_values, raw_values = model_prediction(task_id, row_logits, aspect_names, counts)
                output.append({"model_prediction": model_values, "raw_prediction": raw_values, "logits": row_logits.tolist()})
    return output


def expected_slots(pair_spec: Mapping[str, Any], side: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in pair_spec["expected"][side].items()}


def score_pair(pair_spec: Mapping[str, Any], model_occurrences: Mapping[str, Mapping[str, Any]], bundle: Any, model_name: str) -> dict[str, Any]:
    names = list(bundle.aspect_names)
    is_scalar = not names
    expected_a, expected_b = expected_slots(pair_spec, "A"), expected_slots(pair_spec, "B")
    pred_a = model_occurrences[f"{pair_spec['pair_id']}:A"]["raw_prediction"]
    pred_b = model_occurrences[f"{pair_spec['pair_id']}:B"]["raw_prediction"]
    if is_scalar:
        expected_a = {"__scalar__": expected_a["label"]}
        expected_b = {"__scalar__": expected_b["label"]}
        map_a = {"__scalar__": int(pred_a[0])}
        map_b = {"__scalar__": int(pred_b[0])}
    else:
        map_a = {name: int(pred_a[index]) for index, name in enumerate(names)}
        map_b = {name: int(pred_b[index]) for index, name in enumerate(names)}
    shared = sorted(set(expected_a).intersection(expected_b))
    only_a = sorted(set(expected_a).difference(expected_b))
    only_b = sorted(set(expected_b).difference(expected_a))
    changed_shared = sorted(name for name in shared if expected_a[name] != expected_b[name])
    label_preserving_pair = bool(shared) and not changed_shared and not only_a and not only_b
    def side_score(expected: Mapping[str, int], predicted: Mapping[str, int]) -> dict[str, Any]:
        details = {slot: {"expected": code, "prediction": predicted[slot], "correct": predicted[slot] == code} for slot, code in expected.items()}
        return {"expected_count": len(expected), "correct_count": sum(item["correct"] for item in details.values()), "all_correct": all(item["correct"] for item in details.values()), "slots": details}
    score_a, score_b = side_score(expected_a, map_a), side_score(expected_b, map_b)
    shared_rows = []
    for slot in shared:
        shared_rows.append({"slot": slot, "expected_a": expected_a[slot], "expected_b": expected_b[slot], "prediction_a": map_a[slot], "prediction_b": map_b[slot], "expected_change": expected_a[slot] != expected_b[slot], "prediction_change": map_a[slot] != map_b[slot], "a_correct": map_a[slot] == expected_a[slot], "b_correct": map_b[slot] == expected_b[slot]})
    return {"pair_id": pair_spec["pair_id"], "task_id": pair_spec["task_id"], "kind": pair_spec["kind"], "model": model_name, "side_a": score_a, "side_b": score_b, "pair_both_correct": score_a["all_correct"] and score_b["all_correct"], "shared_slots": shared_rows, "only_a": only_a, "only_b": only_b, "changed_shared_slots": changed_shared, "label_preserving_pair": label_preserving_pair}


def task_metrics(pair_specs: Sequence[Mapping[str, Any]], scored_pairs: Sequence[Mapping[str, Any]], model_name: str, model_occurrences: Mapping[str, Mapping[str, Any]], e8_occurrences: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    relevant = [row for row in scored_pairs]
    expected_total = sum(row["side_a"]["expected_count"] + row["side_b"]["expected_count"] for row in relevant)
    correct_total = sum(row["side_a"]["correct_count"] + row["side_b"]["correct_count"] for row in relevant)
    label_rows = [slot for row in relevant if row["label_preserving_pair"] for slot in row["shared_slots"]]
    changed_rows = [slot for row in relevant for slot in row["shared_slots"] if slot["expected_change"]]
    false_flip_a_to_b = sum(int(slot["a_correct"] and not slot["b_correct"] and slot["prediction_change"]) for slot in label_rows)
    false_flip_b_to_a = sum(int(slot["b_correct"] and not slot["a_correct"] and slot["prediction_change"]) for slot in label_rows)
    transitions_correct = sum(int(slot["a_correct"] and slot["b_correct"]) for slot in changed_rows)
    result = {
        "model": model_name,
        "pair_count": len(relevant),
        "expected_slot_count": expected_total,
        "correct_expected_slot_count": correct_total,
        "expected_slot_accuracy": correct_total / expected_total if expected_total else None,
        "pair_both_correct_count": sum(int(row["pair_both_correct"]) for row in relevant),
        "label_preserving_pair_count": sum(int(row["label_preserving_pair"]) for row in relevant),
        "label_preserving_shared_slot_count": len(label_rows),
        "label_preserving_false_flip_a_correct_b_wrong": false_flip_a_to_b,
        "label_preserving_false_flip_b_correct_a_wrong": false_flip_b_to_a,
        "expected_change_pair_count": sum(int(bool(row["changed_shared_slots"])) for row in relevant),
        "expected_change_shared_slot_count": len(changed_rows),
        "expected_change_transition_correct_count": transitions_correct,
        "expected_change_transition_accuracy": transitions_correct / len(changed_rows) if changed_rows else None,
    }
    if e8_occurrences is not None and model_name != "e8":
        regress_den = regress_num = gain_num = 0
        for row in relevant:
            pair_spec = next(spec for spec in pair_specs if spec["pair_id"] == row["pair_id"])
            for side in ("A", "B"):
                expected = expected_slots(pair_spec, side)
                e8_pred = e8_occurrences[f"{pair_spec['pair_id']}:{side}"]["raw_prediction"]
                pred = model_occurrences[f"{pair_spec['pair_id']}:{side}"]["raw_prediction"]
                if not bundle_for_task(pair_spec["task_id"]).aspect_names:
                    e8_map = {"__scalar__": int(e8_pred[0])}; p_map = {"__scalar__": int(pred[0])}; expected = {"__scalar__": expected["label"]}
                else:
                    names = list(bundle_for_task(pair_spec["task_id"]).aspect_names)
                    e8_map = {name: int(e8_pred[index]) for index, name in enumerate(names)}; p_map = {name: int(pred[index]) for index, name in enumerate(names)}
                for slot, code in expected.items():
                    if e8_map[slot] == code:
                        regress_den += 1
                        regress_num += int(p_map[slot] != code)
                    if e8_map[slot] != code and p_map[slot] == code:
                        gain_num += 1
        result.update({"regression_conditioned_e8_correct_numerator": regress_num, "regression_conditioned_e8_correct_denominator": regress_den, "regression_conditioned_e8_correct_rate": regress_num / regress_den if regress_den else None, "correction_e8_wrong_to_candidate_right": gain_num})
    return result


_BUNDLES: dict[str, Any] = {}
def bundle_for_task(task_id: str) -> Any:
    return _BUNDLES[task_id]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu"))
    args = parser.parse_args()
    out = ROOT / "artifacts" / "e13a"
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
    for task_id in TASKS:
        _BUNDLES[task_id] = load_bundle(task_id)
    assert len(PAIRS) == 30 and {p["task_id"] for p in PAIRS} == set(TASKS)
    occurrences = [{"occurrence_id": f"{p['pair_id']}:{side}", "pair_id": p["pair_id"], "task_id": p["task_id"], "side": side, "text": p["texts"][side], "expected": p["expected"][side]} for p in PAIRS for side in ("A", "B")]
    # Exact duplicate accounting is restricted to train/validation only.
    text_locations: dict[str, list[str]] = defaultdict(list)
    for task_id, bundle in _BUNDLES.items():
        for split_name in ("train", "validation"):
            for index, text_value in enumerate(bundle.splits[split_name].texts):
                text_locations[text_value].append(f"{task_id}:{split_name}:{index}")
    duplicate_occurrences = {row["occurrence_id"]: text_locations.get(row["text"], []) for row in occurrences if text_locations.get(row["text"])}
    if args.device == "auto":
        if torch.cuda.is_available():
            free_by_device = [(int(torch.cuda.mem_get_info(index)[0]), index) for index in range(torch.cuda.device_count())]
            device = torch.device(f"cuda:{max(free_by_device)[1]}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested {device} but CUDA is unavailable")
    spec_payload = {"protocol": "E13A-reviewed-binding-diagnostic-v1", "status": "root_reviewed_before_inference", "pair_count": len(PAIRS), "occurrence_count": len(occurrences), "numeric_maps": {"vsfc_sentiment": {"0": "negative", "1": "neutral", "2": "positive"}, "vsfc_topic": {"0": "lecturer", "1": "training_program", "2": "facility", "3": "others"}, "absa": {"0": "absent", "1": "positive", "2": "negative", "3": "neutral"}, "visfd_others": "raw code 4, not scored"}, "pairs": PAIRS, "unique_text_count": len({row["text"] for row in occurrences}), "duplicate_train_validation_matches": duplicate_occurrences, "test_access": "not requested", "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE}
    (out / "spec.json").write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_results: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for model_name in MODELS:
        model_results: dict[str, Any] = {}
        for task_id in TASKS:
            model, bundle, checkpoint_path = build_model(task_id, model_name)
            model.to(device)
            task_occurrences = [row for row in occurrences if row["task_id"] == task_id]
            inferred = infer_model(model, task_id, bundle, task_occurrences, tokenizer, device)
            by_occ = {occ["occurrence_id"]: {**occ, **pred} for occ, pred in zip(task_occurrences, inferred)}
            pair_scores = [score_pair(p, by_occ, bundle, model_name) for p in PAIRS if p["task_id"] == task_id]
            e8_occ = None
            if model_name != "e8":
                e8_occ = all_results["e8"][task_id]["occurrences"]
            metrics = task_metrics([p for p in PAIRS if p["task_id"] == task_id], pair_scores, model_name, by_occ, e8_occ)
            model_results[task_id] = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "occurrences": by_occ, "pair_scores": pair_scores, "metrics": metrics}
            input_hashes[str(checkpoint_path)] = sha256_file(checkpoint_path)
            partial = {"protocol": "E13A-reviewed-binding-diagnostic-v1", "status": "partial", "device": str(device), "completed_models": list(all_results), "completed_tasks_for_model": list(model_results), "models": {**all_results, model_name: model_results}}
            (out / "partial_results.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_results[model_name] = model_results
    output = {"protocol": "E13A-reviewed-binding-diagnostic-v1", "status": "completed", "device": str(device), "pair_count": len(PAIRS), "occurrence_count": len(occurrences), "unique_text_count": len({row["text"] for row in occurrences}), "duplicate_train_validation_matches": duplicate_occurrences, "models": all_results, "input_checkpoint_hashes": input_hashes, "source_hashes": {"script": sha256_file(Path(__file__)), "spec": sha256_file(ROOT / "docs" / "E13A_BINDING_DIAGNOSTIC_SPEC.md"), "tokenizer": {name: sha256_file(ROOT / "models" / "goiturn-s-gd" / name) for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")}, "backbone_config": sha256_file(ROOT / "models" / "goiturn-s-gd" / "config.json")}, "test_access": "not requested; all task bundles loaded include_test=False"}
    (out / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out / "results.json"), "device": str(device), "pairs": len(PAIRS), "occurrences": len(occurrences), "unique_texts": output["unique_text_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
