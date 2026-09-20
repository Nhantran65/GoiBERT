"""Build the reviewed-before-inference E11 diagnostic pair artifact.

The pairs are synthetic, short, and outside the frozen train/test files.  The
task maps and slot names are read from ``goibert.data``; expectations are
explicitly scoped to the aspect slots stated in each pair.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goibert.data import load_task  # noqa: E402


PAIR_DEFS: list[dict[str, Any]] = [
    {"task_id": "vsfc_sentiment", "kind": "surface_paraphrase", "base": "giảng viên rất nhiệt tình và dễ hiểu .", "variant": "thầy cô rất tận tâm và giảng dễ hiểu .", "base_expectations": [{"label_id": 2, "label_name": "positive"}], "variant_expectations": [{"label_id": 2, "label_name": "positive"}]},
    {"task_id": "vsfc_sentiment", "kind": "surface_paraphrase", "base": "bài giảng rõ ràng và chi tiết .", "variant": "nội dung trình bày mạch lạc và đầy đủ .", "base_expectations": [{"label_id": 2, "label_name": "positive"}], "variant_expectations": [{"label_id": 2, "label_name": "positive"}]},
    {"task_id": "vsfc_sentiment", "kind": "surface_paraphrase", "base": "giảng viên dạy khó hiểu .", "variant": "cô giảng không dễ hiểu .", "base_expectations": [{"label_id": 0, "label_name": "negative"}], "variant_expectations": [{"label_id": 0, "label_name": "negative"}]},
    {"task_id": "vsfc_sentiment", "kind": "negation_or_opposition", "base": "giảng viên nhiệt tình .", "variant": "giảng viên không nhiệt tình .", "base_expectations": [{"label_id": 2, "label_name": "positive"}], "variant_expectations": [{"label_id": 0, "label_name": "negative"}]},
    {"task_id": "vsfc_sentiment", "kind": "negation_or_opposition", "base": "bài giảng rõ ràng .", "variant": "bài giảng không rõ ràng .", "base_expectations": [{"label_id": 2, "label_name": "positive"}], "variant_expectations": [{"label_id": 0, "label_name": "negative"}]},
    {"task_id": "vsfc_sentiment", "kind": "negation_or_opposition", "base": "môn học bình thường .", "variant": "môn học không có gì nổi bật .", "base_expectations": [{"label_id": 1, "label_name": "neutral"}], "variant_expectations": [{"label_id": 1, "label_name": "neutral"}]},
    {"task_id": "vsfc_topic", "kind": "surface_paraphrase", "base": "giáo trình chưa cụ thể .", "variant": "tài liệu môn học chưa rõ ràng .", "base_expectations": [{"label_id": 1, "label_name": "training_program"}], "variant_expectations": [{"label_id": 1, "label_name": "training_program"}]},
    {"task_id": "vsfc_topic", "kind": "surface_paraphrase", "base": "giảng viên rất nhiệt tình .", "variant": "thầy cô dạy rất tận tâm .", "base_expectations": [{"label_id": 0, "label_name": "lecturer"}], "variant_expectations": [{"label_id": 0, "label_name": "lecturer"}]},
    {"task_id": "vsfc_topic", "kind": "surface_paraphrase", "base": "phòng học nhỏ .", "variant": "cơ sở vật chất phòng học chật .", "base_expectations": [{"label_id": 2, "label_name": "facility"}], "variant_expectations": [{"label_id": 2, "label_name": "facility"}]},
    {"task_id": "vsfc_topic", "kind": "negation_or_opposition", "base": "giảng viên nhiệt tình .", "variant": "giảng viên không nhiệt tình .", "base_expectations": [{"label_id": 0, "label_name": "lecturer"}], "variant_expectations": [{"label_id": 0, "label_name": "lecturer"}]},
    {"task_id": "vsfc_topic", "kind": "negation_or_opposition", "base": "giáo trình rõ ràng .", "variant": "giáo trình không rõ ràng .", "base_expectations": [{"label_id": 1, "label_name": "training_program"}], "variant_expectations": [{"label_id": 1, "label_name": "training_program"}]},
    {"task_id": "visfd", "kind": "surface_paraphrase", "base": "pin dùng được hai ngày .", "variant": "thời lượng pin kéo dài khoảng hai ngày .", "base_expectations": [{"slot": "BATTERY", "label_id": 1, "label_name": "positive"}], "variant_expectations": [{"slot": "BATTERY", "label_id": 1, "label_name": "positive"}]},
    {"task_id": "visfd", "kind": "surface_paraphrase", "base": "camera chụp ảnh rất đẹp .", "variant": "camera cho ảnh sắc nét .", "base_expectations": [{"slot": "CAMERA", "label_id": 1, "label_name": "positive"}], "variant_expectations": [{"slot": "CAMERA", "label_id": 1, "label_name": "positive"}]},
    {"task_id": "visfd", "kind": "surface_paraphrase", "base": "máy chạy mượt khi chơi game .", "variant": "chơi game trên máy rất mượt .", "base_expectations": [{"slot": "PERFORMANCE", "label_id": 1, "label_name": "positive"}], "variant_expectations": [{"slot": "PERFORMANCE", "label_id": 1, "label_name": "positive"}]},
    {"task_id": "visfd", "kind": "negation_or_opposition", "base": "pin rất trâu .", "variant": "pin không trâu .", "base_expectations": [{"slot": "BATTERY", "label_id": 1, "label_name": "positive"}], "variant_expectations": [{"slot": "BATTERY", "label_id": 2, "label_name": "negative"}]},
    {"task_id": "visfd", "kind": "negation_or_opposition", "base": "camera đẹp .", "variant": "camera không đẹp .", "base_expectations": [{"slot": "CAMERA", "label_id": 1, "label_name": "positive"}], "variant_expectations": [{"slot": "CAMERA", "label_id": 2, "label_name": "negative"}]},
    {"task_id": "absa_restaurant", "kind": "surface_paraphrase", "base": "món ăn rất ngon .", "variant": "đồ ăn có hương vị tuyệt vời .", "base_expectations": [{"slot": "FOOD#QUALITY", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "FOOD#QUALITY", "label_id": 3, "label_name": "positive"}]},
    {"task_id": "absa_restaurant", "kind": "surface_paraphrase", "base": "nhân viên phục vụ thân thiện .", "variant": "người phục vụ rất niềm nở .", "base_expectations": [{"slot": "SERVICE#GENERAL", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "SERVICE#GENERAL", "label_id": 3, "label_name": "positive"}]},
    {"task_id": "absa_restaurant", "kind": "negation_or_opposition", "base": "đồ uống ngon .", "variant": "đồ uống không ngon .", "base_expectations": [{"slot": "DRINKS#QUALITY", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "DRINKS#QUALITY", "label_id": 1, "label_name": "negative"}]},
    {"task_id": "absa_restaurant", "kind": "negation_or_opposition", "base": "món ăn ngon .", "variant": "món ăn không ngon .", "base_expectations": [{"slot": "FOOD#QUALITY", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "FOOD#QUALITY", "label_id": 1, "label_name": "negative"}]},
    {"task_id": "absa_restaurant", "kind": "aspect_addition", "base": "món ăn ngon .", "variant": "món ăn ngon và nhân viên phục vụ thân thiện .", "base_expectations": [{"slot": "FOOD#QUALITY", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "FOOD#QUALITY", "label_id": 3, "label_name": "positive"}, {"slot": "SERVICE#GENERAL", "label_id": 3, "label_name": "positive"}], "note": "Evaluate only explicitly stated slots; added service clause may add a second aspect."},
    {"task_id": "absa_hotel", "kind": "surface_paraphrase", "base": "phòng sạch sẽ .", "variant": "phòng ốc được vệ sinh tươm tất .", "base_expectations": [{"slot": "FACILITIES#CLEANLINESS", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "FACILITIES#CLEANLINESS", "label_id": 3, "label_name": "positive"}]},
    {"task_id": "absa_hotel", "kind": "surface_paraphrase", "base": "vị trí khách sạn thuận tiện .", "variant": "khách sạn nằm ở nơi dễ đi lại .", "base_expectations": [{"slot": "LOCATION#GENERAL", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "LOCATION#GENERAL", "label_id": 3, "label_name": "positive"}]},
    {"task_id": "absa_hotel", "kind": "negation_or_opposition", "base": "bữa sáng ngon .", "variant": "bữa sáng không ngon .", "base_expectations": [{"slot": "FOOD&DRINKS#QUALITY", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "FOOD&DRINKS#QUALITY", "label_id": 1, "label_name": "negative"}]},
    {"task_id": "absa_hotel", "kind": "negation_or_opposition", "base": "nhân viên lịch sự .", "variant": "nhân viên không lịch sự .", "base_expectations": [{"slot": "SERVICE#GENERAL", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "SERVICE#GENERAL", "label_id": 1, "label_name": "negative"}]},
    {"task_id": "absa_hotel", "kind": "aspect_addition", "base": "phòng sạch sẽ .", "variant": "phòng sạch sẽ và nhân viên lịch sự .", "base_expectations": [{"slot": "FACILITIES#CLEANLINESS", "label_id": 3, "label_name": "positive"}], "variant_expectations": [{"slot": "FACILITIES#CLEANLINESS", "label_id": 3, "label_name": "positive"}, {"slot": "SERVICE#GENERAL", "label_id": 3, "label_name": "positive"}], "note": "Evaluate only explicitly stated slots; added service clause may add a second aspect."},
]


def _mapping(task: Any) -> dict[str, Any]:
    return {
        "label_names": list(task.label_names),
        "label_ids": {name: index for index, name in enumerate(task.label_names)},
        "aspect_names": list(task.aspect_names),
        "slot_label_names": [list(values) for values in task.slot_label_names],
        "slot_label_ids": [{name: index for index, name in enumerate(values)} for values in task.slot_label_names],
        "label_policy": task.label_policy,
    }


def main() -> int:
    by_task: dict[str, Any] = {}
    for pair in PAIR_DEFS:
        by_task.setdefault(pair["task_id"], load_task(pair["task_id"], ROOT, include_test=False))
    for pair in PAIR_DEFS:
        task = by_task[pair["task_id"]]
        if pair["task_id"] in {"vsfc_sentiment", "vsfc_topic"}:
            for side in ("base_expectations", "variant_expectations"):
                for expectation in pair[side]:
                    if expectation["label_name"] != task.label_names[expectation["label_id"]]:
                        raise ValueError(f"Label map mismatch in {pair}")
        else:
            names = set(task.aspect_names)
            for side in ("base_expectations", "variant_expectations"):
                for expectation in pair[side]:
                    if expectation["slot"] not in names:
                        raise ValueError(f"Unknown slot {expectation['slot']!r} for {pair['task_id']}")
                    slot = task.aspect_names.index(expectation["slot"])
                    if task.slot_label_names[slot][expectation["label_id"]] != expectation["label_name"]:
                        raise ValueError(f"Slot label map mismatch in {pair}")
    artifact = {
        "protocol": "E11-diagnostic-pairs-v1",
        "status": "semantic_review_pending",
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_split": "synthetic_out_of_corpus",
        "train_injection": False,
        "test_loaded": False,
        "pair_count": len(PAIR_DEFS),
        "task_label_mappings": {task_id: _mapping(task) for task_id, task in by_task.items()},
        "pairs": [{"pair_id": f"e11diag-{index:03d}", **pair, "expectation_scope": "explicit_slots_only" if pair["task_id"] not in {"vsfc_sentiment", "vsfc_topic"} else "whole_task_label"} for index, pair in enumerate(PAIR_DEFS, start=1)],
        "review_notes": ["Root should review semantic clarity before inference.", "ABSA aspect-addition pairs intentionally score only explicitly stated slots.", "These pairs are directional diagnostics and do not estimate benchmark prevalence."],
    }
    output = ROOT / "artifacts" / "e11" / "diagnostic_pairs.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "pair_count": len(PAIR_DEFS), "status": artifact["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
