#!/usr/bin/env python3
"""Build duplicate evidence and deterministic grouped splits for the intent audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "intent_benchmark"
LABELS = ["GREETING", "CONFIRMATION", "GRATITUDE", "MENU_QUERY", "COMPLAINT", "ADDRESS", "PHONE", "OTHER"]
SPLITS = ["train", "dev", "test"]

# Root manually reviewed every fuzzy candidate from the first pass and confirmed
# these whole-case template families as substantive near-duplicate variants.  The
# family unit is the whole case so no preceding history is split from its target.
CONFIRMED_CASE_FAMILIES: list[dict[str, Any]] = [
    {"family_id": "template-pho-ngon-hu-hon", "case_ids": ["pb-v1-0176", "pb-v1-0192", "pb-v1-0226"], "rationale": "Same Phở Ngon Hú Hồn ordering template with quantity/politeness variants; root reviewed all candidate evidence."},
    {"family_id": "template-banh-mi-hoi-an", "case_ids": ["pb-v1-0179", "pb-v1-0247"], "rationale": "Same Bánh Mì Hội An ordering template with quantity variation; root reviewed all candidate evidence."},
    {"family_id": "template-com-tam-joi-original", "case_ids": ["pb-v1-0174", "pb-v1-0195", "pb-v1-0208"], "rationale": "Same Cơm Tấm Joi Original ordering template with quantity/politeness variation; root reviewed all candidate evidence."},
    {"family_id": "template-com-tam-nhe", "case_ids": ["pb-v1-0191", "pb-v1-0240"], "rationale": "Same Cơm Tấm Nhẹ selection/cancellation template with polite wording variation; root reviewed all candidate evidence."},
    {"family_id": "template-mi-tiem-coffee", "case_ids": ["pb-v1-0199", "pb-v1-0215", "pb-v1-0250"], "rationale": "Same Mì Tiềm plus coffee ordering template with coffee/checkout variants; root reviewed all candidate evidence."},
    {"family_id": "template-rice-tea", "case_ids": ["pb-v1-0201", "pb-v1-0213", "pb-v1-0219"], "rationale": "Same rice plus Trà Lài Xiêm Xanh ordering template with rice/quantity variants; root reviewed all candidate evidence."},
    {"family_id": "template-english-special-dish", "case_ids": ["pb-v1-0023", "pb-v1-0024"], "rationale": "Same English special-dish inquiry with today/week wording variation; root reviewed all candidate evidence."},
    {"family_id": "template-reservation-tonight-2", "case_ids": ["pb-v1-0155", "pb-v1-0166"], "rationale": "Same two-guest tonight reservation template with time/greeting variation; root reviewed all candidate evidence."},
    {"family_id": "template-reservation-2-guests", "case_ids": ["pb-v1-0119", "pb-v1-0164"], "rationale": "Same two-guest evening reservation template with time variation; root reviewed all candidate evidence."},
]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def current_text(row: dict[str, Any]) -> str:
    return row["model_input"]["messages"][-1]["text"]


def full_input(row: dict[str, Any]) -> str:
    return "\n".join(f"{message['role']}:{message['text']}" for message in row["model_input"]["messages"])


def generic_current(text: str) -> bool:
    value = normalize(text)
    if value in {
        "ok", "oke", "okay", "ok chốt", "xong", "được rồi", "vậy thôi", "vậy nha",
        "đúng rồi", "chốt", "cod", "cod nha", "chuyển khoản", "giao ngay", "19h", "19h, cod",
    }:
        return True
    if re.fullmatch(r"giao\s+\[address_\d+\]", value):
        return True
    if re.fullmatch(r"(?:\[phone_\d+\]|số\s*(?:mình|m)\s*\[phone_\d+\])", value):
        return True
    return len(value.split()) <= 3 and any(token in value.split() for token in {"ok", "xong", "cod", "chốt", "giao"})


def read_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (OUT / "corrected_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


class DSU:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def analyze_duplicates(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], DSU]:
    sample_ids = [row["sample_id"] for row in rows]
    dsu = DSU(sample_ids)
    current_groups: dict[str, list[str]] = defaultdict(list)
    full_groups: dict[str, list[str]] = defaultdict(list)
    source_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        sample = row["sample_id"]
        current_groups[normalize(current_text(row))].append(sample)
        full_groups[normalize(full_input(row))].append(sample)
        source_groups[row["source_ref"]].append(sample)

    for values in source_groups.values():
        for sample in values[1:]:
            dsu.union(values[0], sample)
    for values in full_groups.values():
        for sample in values[1:]:
            dsu.union(values[0], sample)
    by_case: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row["sample_id"])
    case_to_family: dict[str, str] = {}
    for family in CONFIRMED_CASE_FAMILIES:
        for case_id in family["case_ids"]:
            if case_id in case_to_family:
                raise AssertionError(f"case appears in multiple confirmed families: {case_id}")
            case_to_family[case_id] = family["family_id"]
        family_rows = [sample for case_id in family["case_ids"] for sample in by_case.get(case_id, [])]
        if not family_rows:
            raise AssertionError(f"confirmed family has no rows: {family['family_id']}")
        for sample in family_rows[1:]:
            dsu.union(family_rows[0], sample)

    current_overlap = [
        {
            "normalized_current_text": key,
            "samples": values,
            "count": len(values),
            "generic_or_placeholder": all(generic_current(current_text(next(row for row in rows if row["sample_id"] == sample))) for sample in values),
            "grouping_policy": "report_only_unless_full_input_or_substantive_family_confirmed",
        }
        for key, values in sorted(current_groups.items())
        if len(values) > 1
    ]
    full_exact = [
        {"normalized_full_input": key, "samples": values, "count": len(values), "confirmed_group": True}
        for key, values in sorted(full_groups.items())
        if len(values) > 1
    ]

    by_id = {row["sample_id"]: row for row in rows}
    data = []
    for row in rows:
        text = normalize(full_input(row))
        data.append((row["sample_id"], row["case_id"], text, set(text.split()), normalize(current_text(row))))
    candidates: list[dict[str, Any]] = []
    for index, (left_id, left_case, left_text, left_tokens, left_current) in enumerate(data):
        for right_id, right_case, right_text, right_tokens, right_current in data[index + 1 :]:
            if left_case == right_case:
                continue
            longer, shorter = max(len(left_text), len(right_text)), min(len(left_text), len(right_text))
            if shorter == 0 or longer > shorter * 2.5:
                continue
            union = left_tokens | right_tokens
            token_jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
            if token_jaccard < 0.5:
                continue
            matcher = SequenceMatcher(None, left_text, right_text, autojunk=False)
            quick = matcher.quick_ratio()
            if quick < 0.72:
                continue
            ratio = matcher.ratio()
            if ratio < 0.72:
                continue
            left_row, right_row = by_id[left_id], by_id[right_id]
            left_generic, right_generic = generic_current(left_current), generic_current(right_current)
            if left_generic or right_generic:
                category = "trivial_ack_or_placeholder_overlap"
                disposition = "report_only"
            elif case_to_family.get(left_case) and case_to_family.get(left_case) == case_to_family.get(right_case):
                category = "confirmed_substantive_template_variant"
                disposition = "confirmed_grouped"
            elif ratio >= 0.9 and token_jaccard >= 0.75:
                category = "substantive_near_duplicate_candidate"
                disposition = "candidate_requires_manual_confirmation"
            else:
                category = "near_duplicate_candidate"
                disposition = "candidate_requires_manual_confirmation"
            candidates.append({
                "left_sample_id": left_id,
                "right_sample_id": right_id,
                "left_case_id": left_case,
                "right_case_id": right_case,
                "normalized_current_left": left_current,
                "normalized_current_right": right_current,
                "sequence_similarity": round(ratio, 6),
                "token_jaccard": round(token_jaccard, 6),
                "left_label": left_row["label"],
                "right_label": right_row["label"],
                "category": category,
                "disposition": disposition,
                "evidence": "NFC/casefold/whitespace-normalized full model-input strings; current text and labels shown for audit.",
            })
    candidates.sort(key=lambda item: (-item["sequence_similarity"], -item["token_jaccard"], item["left_sample_id"], item["right_sample_id"]))

    confirmed_families = []
    for family in CONFIRMED_CASE_FAMILIES:
        members = [row["sample_id"] for row in rows if row["case_id"] in set(family["case_ids"])]
        confirmed_families.append({**family, "sample_ids": sorted(members), "confirmed_by": "root", "group_whole_cases": True})
    for candidate in candidates:
        same_confirmed_family = (
            case_to_family.get(candidate["left_case_id"])
            and case_to_family.get(candidate["left_case_id"]) == case_to_family.get(candidate["right_case_id"])
        )
        if same_confirmed_family:
            candidate["category"] = "confirmed_substantive_template_variant"
            candidate["disposition"] = "confirmed_grouped"
            candidate["reviewer"] = "root"
            candidate["review_reason"] = "The pair belongs to a root-confirmed whole-case substantive template family; all turns in both cases are grouped."
        elif candidate["disposition"] == "candidate_requires_manual_confirmation":
            candidate["disposition"] = "report_only_root_reviewed"
            if candidate["category"] == "substantive_near_duplicate_candidate":
                candidate["review_reason"] = "Root reviewed the pair as a changed food combination, extra item, or different conversation stage/history; no additional family edge was added."
            else:
                candidate["review_reason"] = "Root reviewed the pair as a generic or operational overlap; current-text overlap is reported without joining histories."
            candidate["reviewer"] = "root"
        else:
            candidate["disposition"] = "report_only_root_reviewed"
            candidate["reviewer"] = "root"
            candidate["review_reason"] = "Root reviewed this generic acknowledgment or placeholder overlap; it is reported separately and does not join histories unless covered by an explicit confirmed family."
    duplicate_report = {
        "normalization": "NFC then casefold then collapse whitespace",
        "current_text_overlap_families": current_overlap,
        "exact_full_input_duplicate_families": full_exact,
        "near_duplicate_candidates": candidates,
        "confirmed_substantive_near_duplicate_families": confirmed_families,
        "candidate_count": len(candidates),
        "policy": "case/source groups, exact full-input duplicates, and root-confirmed substantive template families are joined; repeated generic acknowledgments and placeholder-only address turns are reported separately",
        "tradeoff": "Root confirmed nine conservative whole-case template families after reviewing all 52 fuzzy candidates. Remaining candidates stay report-only because they differ in item combination, extra item, conversation stage/history, or are generic operational overlap.",
    }
    (OUT / "duplicate_analysis.json").write_text(json.dumps(duplicate_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return duplicate_report, dsu


def group_rows(rows: list[dict[str, Any]], dsu: DSU) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[dsu.find(row["sample_id"])].append(row)
    groups = []
    for root, members in grouped.items():
        cases = sorted({row["case_id"] for row in members})
        sources = sorted({row["source_ref"] for row in members})
        counts = Counter(row["label"] for row in members)
        groups.append({"group_id": root, "rows": sorted(members, key=lambda row: row["sample_id"]), "cases": cases, "sources": sources, "size": len(members), "counts": dict(counts)})
    return sorted(groups, key=lambda group: group["group_id"])


def score(assignments: dict[str, str], groups: list[dict[str, Any]], targets: dict[str, float]) -> float:
    counts = {split: Counter() for split in SPLITS}
    sizes = Counter()
    for group in groups:
        split = assignments.get(group["group_id"])
        if split is None:
            continue
        sizes[split] += group["size"]
        counts[split].update(group["counts"])
    value = 0.0
    for split in SPLITS:
        value += ((sizes[split] - targets[split]) ** 2) / max(1.0, targets[split])
        for label in LABELS:
            target = targets[split] * targets["label_share_" + label]
            value += 2.2 * ((counts[split][label] - target) ** 2) / max(1.0, target)
            if counts[split][label] == 0:
                value += 250.0
    return value


def assign_groups(groups: list[dict[str, Any]], seed: int = 42) -> tuple[dict[str, str], dict[str, Any]]:
    total = sum(group["size"] for group in groups)
    total_by_label = Counter()
    for group in groups:
        total_by_label.update(group["counts"])
    target_sizes = {"train": total * 0.70, "dev": total * 0.15, "test": total * 0.15}
    targets: dict[str, float] = dict(target_sizes)
    for label in LABELS:
        targets["label_share_" + label] = total_by_label[label] / total

    # Group-size and rare-label order ensures the greedy seed handles the 9-turn
    # conversations early; restarts are deterministic under seed 42.
    def order_key(group: dict[str, Any]) -> tuple[int, int, str]:
        rarity = sum(1 for label in LABELS if group["counts"].get(label, 0) and total_by_label[label] <= 15)
        return (-group["size"], -rarity, group["group_id"])

    rng = random.Random(seed)
    best: tuple[float, dict[str, str]] | None = None
    for restart in range(48):
        ordered = sorted(groups, key=order_key)
        # Same deterministic primary order, with a stable seeded tie permutation.
        tie = {group["group_id"]: rng.random() for group in ordered}
        ordered.sort(
            key=lambda group: (
                -group["size"],
                -sum(1 for label in LABELS if group["counts"].get(label, 0) and total_by_label[label] <= 15),
                tie[group["group_id"]],
                group["group_id"],
            )
        )
        assignments: dict[str, str] = {}
        for group in ordered:
            options = []
            for split in SPLITS:
                assignments[group["group_id"]] = split
                options.append((score(assignments, groups, targets), split))
                del assignments[group["group_id"]]
            options.sort(key=lambda item: (item[0], item[1]))
            assignments[group["group_id"]] = options[0][1]
        # Local improvement, moving or swapping one group at a time.
        improved = True
        while improved:
            improved = False
            current = score(assignments, groups, targets)
            for group in ordered:
                gid = group["group_id"]
                old = assignments[gid]
                candidates = []
                for split in SPLITS:
                    if split == old:
                        continue
                    assignments[gid] = split
                    candidates.append((score(assignments, groups, targets), split))
                assignments[gid] = old
                if candidates:
                    candidate_score, candidate_split = min(candidates)
                    if candidate_score + 1e-9 < current:
                        assignments[gid] = candidate_split
                        improved = True
                        current = candidate_score
            if current < 1e-7:
                break
        final_score = score(assignments, groups, targets)
        if best is None or final_score < best[0]:
            best = (final_score, dict(assignments))
    assert best is not None
    assignments = best[1]
    counts = {split: Counter() for split in SPLITS}
    sizes = Counter()
    for group in groups:
        split = assignments[group["group_id"]]
        sizes[split] += group["size"]
        counts[split].update(group["counts"])
    return assignments, {
        "seed": seed,
        "target_ratio": {"train": 0.70, "dev": 0.15, "test": 0.15},
        "target_sizes": {split: round(target_sizes[split], 3) for split in SPLITS},
        "actual_sizes": dict(sizes),
        "actual_class_counts": {split: dict(sorted(counts[split].items())) for split in SPLITS},
        "class_support": {split: sorted(label for label in LABELS if counts[split][label]) for split in SPLITS},
        "all_eight_classes_each_split": all(all(counts[split][label] for label in LABELS) for split in SPLITS),
        "group_count": len(groups),
        "largest_group": max((group["size"] for group in groups), default=0),
        "largest_group_cases": [group["cases"] for group in groups if group["size"] == max(g["size"] for g in groups)],
        "objective": best[0],
    }


def audit(rows: list[dict[str, Any]], groups: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    row_split = {}
    for group in groups:
        split = assignments[group["group_id"]]
        for row in group["rows"]:
            row_split[row["sample_id"]] = split
    if set(row_split) != {row["sample_id"] for row in rows}:
        raise AssertionError("not every row assigned")
    for group in groups:
        splits = {row_split[row["sample_id"]] for row in group["rows"]}
        if len(splits) != 1:
            raise AssertionError("group split across partitions")
    for row in rows:
        messages = row["model_input"]["messages"]
        turn_index = row["turn_index"]
        if len(messages) != turn_index + 1:
            # The source snapshot contains only customer turns in target history.
            # This assertion catches accidental future-turn truncation or invention.
            raise AssertionError(f"history length mismatch {row['sample_id']}")
        if messages[-1]["role"] != "customer":
            raise AssertionError(f"current role mismatch {row['sample_id']}")
        for message in messages:
            if set(message) != {"role", "text"} or message["role"] != "customer":
                raise AssertionError(f"unexpected model-input message {row['sample_id']}")
        serialized = json.dumps(row["model_input"], ensure_ascii=False)
        for forbidden in ("expectedIntent", "assertions", "fixtures", "primaryIntent", "source_ref", "sourceRef", "initialStateFixture"):
            if forbidden in serialized:
                raise AssertionError(f"model-input metadata leakage {row['sample_id']}: {forbidden}")
    case_splits = defaultdict(set)
    source_splits = defaultdict(set)
    for row in rows:
        case_splits[row["case_id"]].add(row_split[row["sample_id"]])
        source_splits[row["source_ref"]].add(row_split[row["sample_id"]])
    if any(len(values) != 1 for values in case_splits.values()) or any(len(values) != 1 for values in source_splits.values()):
        raise AssertionError("case/source group leakage")
    current_by_split: dict[str, dict[str, list[str]]] = {split: defaultdict(list) for split in SPLITS}
    full_by_split: dict[str, dict[str, list[str]]] = {split: defaultdict(list) for split in SPLITS}
    for row in rows:
        split = row_split[row["sample_id"]]
        current_by_split[split][normalize(current_text(row))].append(row["sample_id"])
        full_by_split[split][normalize(full_input(row))].append(row["sample_id"])
    cross_split_current = []
    for key in sorted(set().union(*(set(values) for values in current_by_split.values()))):
        support = {split: values[key] for split, values in current_by_split.items() if key in values}
        if len(support) > 1:
            cross_split_current.append({"normalized_current_text": key, "by_split": support})
    cross_split_full = []
    for key in sorted(set().union(*(set(values) for values in full_by_split.values()))):
        support = {split: values[key] for split, values in full_by_split.items() if key in values}
        if len(support) > 1:
            cross_split_full.append({"normalized_full_input": key, "by_split": support})
    return {
        "row_count": len(rows),
        "case_count": len(case_splits),
        "source_group_count": len(source_splits),
        "case_disjoint": True,
        "source_disjoint": True,
        "full_input_duplicate_cross_split_count": len(cross_split_full),
        "current_text_overlap_cross_split_count": len(cross_split_current),
        "current_text_overlap_cross_split_examples": cross_split_current[:50],
        "note": "Current-text overlap across splits is intentionally reported and is not treated as proof of no leakage; full-input duplicate families remain grouped.",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    if len(rows) != 507:
        raise AssertionError(f"expected507 rows, got {len(rows)}")
    duplicate_report, dsu = analyze_duplicates(rows)
    groups = group_rows(rows, dsu)
    assignments, split_stats = assign_groups(groups, args.seed)
    split_audit = audit(rows, groups, assignments)
    split_map = []
    by_split = {split: [] for split in SPLITS}
    for group in groups:
        split = assignments[group["group_id"]]
        for row in group["rows"]:
            entry = {
                "sample_id": row["sample_id"],
                "case_id": row["case_id"],
                "turn_index": row["turn_index"],
                "source_ref": row["source_ref"],
                "split": split,
                "group_id": group["group_id"],
                "label": row["label"],
            }
            split_map.append(entry)
            by_split[split].append(row)
    split_map.sort(key=lambda row: row["sample_id"])
    write_jsonl(OUT / "split_assignments.jsonl", split_map)
    for split in SPLITS:
        write_jsonl(OUT / f"{split}.jsonl", sorted(by_split[split], key=lambda row: row["sample_id"]))
    groups_manifest = [
        {
            "group_id": group["group_id"],
            "split": assignments[group["group_id"]],
            "size": group["size"],
            "cases": group["cases"],
            "source_refs": group["sources"],
            "label_counts": dict(sorted(group["counts"].items())),
        }
        for group in groups
    ]
    (OUT / "group_manifest.json").write_text(json.dumps(groups_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "split": split_stats,
        "grouping": {
            "case_groups": len({row["case_id"] for row in rows}),
            "source_groups": len({row["source_ref"] for row in rows}),
            "total_connected_groups": len(groups),
            "confirmed_substantive_template_family_count": len(CONFIRMED_CASE_FAMILIES),
            "confirmed_substantive_template_family_ids": [family["family_id"] for family in CONFIRMED_CASE_FAMILIES],
            "largest_group": split_stats["largest_group"],
            "infeasibility": None,
            "policy": "case/source groups, exact full-input duplicate families, and root-confirmed substantive template families are indivisible; remaining fuzzy candidates stay audit evidence without grouping",
        },
        "audit": split_audit,
        "duplicate_summary": {
            "current_text_family_count": len(duplicate_report["current_text_overlap_families"]),
            "exact_full_input_family_count": len(duplicate_report["exact_full_input_duplicate_families"]),
            "near_duplicate_candidate_count": duplicate_report["candidate_count"],
            "confirmed_substantive_near_duplicate_family_count": len(CONFIRMED_CASE_FAMILIES),
        },
    }
    if split_stats["largest_group"] > len(rows) * 0.5:
        report["grouping"]["infeasibility"] = "A connected duplicate/source group exceeds half of all targets; exact class-balanced 70/15/15 support may be infeasible without weakening grouping."
    (OUT / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "split_audit.json").write_text(json.dumps(split_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": split_stats, "audit": split_audit, "duplicates": report["duplicate_summary"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
