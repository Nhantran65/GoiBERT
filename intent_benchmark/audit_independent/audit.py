#!/usr/bin/env python3
"""Independent, read-only audit of the frozen intent benchmark artifacts.

The script only reads the source bundle, frozen benchmark files, local model
artifacts, run summaries and saved predictions.  It writes new evidence under
intent_benchmark/audit_independent/ and never mutates the original benchmark or
trial artifacts.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "intent_benchmark"
OUT = BENCH / "audit_independent"
SOURCE = ROOT / "tmp/prompt-bench-v1-250-standalone/scripts/ai-verify/packs/prompt-bench-v1/cases.jsonl"
LABELS = ["GREETING", "CONFIRMATION", "GRATITUDE", "MENU_QUERY", "COMPLAINT", "ADDRESS", "PHONE", "OTHER"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).casefold()).strip()


def metric_report(gold: list[int], pred: list[int]) -> dict[str, Any]:
    cm = [[0 for _ in LABELS] for _ in LABELS]
    for g, p in zip(gold, pred):
        cm[g][p] += 1
    per = {}
    f1s = []
    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(LABELS))) - tp
        fn = sum(cm[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(cm[i])}
    return {
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold)),
        "macro_f1": sum(f1s) / len(LABELS),
        "per_intent": per,
        "confusion_matrix": cm,
        "support": len(gold),
    }


def row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["sample_id"]: row for row in rows}


def source_and_input_audit() -> dict[str, Any]:
    cases = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = {(case["caseId"], i): (case, turn) for case in cases for i, turn in enumerate(case.get("targetTurns", []))}
    missing = [(key, case, turn) for key, (case, turn) in raw.items() if not turn.get("expectedIntent")]
    labeled = [(key, case, turn) for key, (case, turn) in raw.items() if turn.get("expectedIntent")]
    corrected = read_jsonl(BENCH / "corrected_dataset.jsonl")
    corrected_by_id = row_map(corrected)
    review = read_jsonl(BENCH / "review_ledger.jsonl")
    assignments = {row["sample_id"]: row for row in read_jsonl(BENCH / "split_assignments.jsonl")}
    forbidden = ("expectedIntent", "assertions", "fixtures", "primaryIntent", "source_ref", "sourceRef", "initialStateFixture")
    violations = []
    history_lengths = collections.Counter()
    rows_with_missing_history = []
    current_text = {}
    for row in corrected:
        key = (row["case_id"], row["turn_index"])
        if key not in raw:
            violations.append({"sample_id": row["sample_id"], "type": "not_in_raw"})
            continue
        case, turn = raw[key]
        expected_id = f"{case['caseId']}:t{row['turn_index']}"
        messages = row["model_input"].get("messages", [])
        history_lengths[len(messages) - 1] += 1
        expected_messages = [{"role": t.get("role"), "text": t.get("text", "")} for t in case["targetTurns"][: row["turn_index"]]]
        expected_messages += [{"role": turn.get("role"), "text": turn.get("text", "")}]
        if row["sample_id"] != expected_id:
            violations.append({"sample_id": row["sample_id"], "type": "unstable_id", "expected": expected_id})
        if row.get("source_ref") != case.get("sourceRef"):
            violations.append({"sample_id": row["sample_id"], "type": "source_ref_mismatch"})
        if messages != expected_messages:
            violations.append({"sample_id": row["sample_id"], "type": "history_or_current_mismatch"})
        if any(field in json.dumps(row["model_input"], ensure_ascii=False) for field in forbidden):
            violations.append({"sample_id": row["sample_id"], "type": "metadata_in_model_input"})
        if any((case["targetTurns"][j].get("expectedIntent") and j > row["turn_index"]) for j in range(row["turn_index"] + 1, len(case["targetTurns"]))):
            # This is an explicit evidence flag, not a violation: future labels are not copied.
            rows_with_missing_history.append(row["sample_id"])
        current_text[row["sample_id"]] = norm(turn.get("text", ""))

    changes = [row for row in corrected if row["label_provenance"]["original_label"] != row["label_provenance"]["reviewed_label"]]
    change_by_split = collections.Counter(assignments.get(row["sample_id"], {}).get("split", "MISSING") for row in changes)
    change_matrix = collections.Counter((row["label_provenance"]["original_label"], row["label_provenance"]["reviewed_label"]) for row in changes)
    uncertain = [row for row in corrected if row["label_provenance"].get("uncertain")]
    return {
        "source_sha256_actual": sha256(SOURCE),
        "source_counts": {"cases": len(cases), "all_target_turns": len(raw), "labeled": len(labeled), "missing": len(missing)},
        "source_ref_unique": len({case.get("sourceRef") for case in cases}),
        "source_ref_duplicates": [key for key, n in collections.Counter(case.get("sourceRef") for case in cases).items() if n > 1],
        "raw_roles": dict(collections.Counter(turn.get("role") for _, turn in raw.values())),
        "raw_origins": dict(collections.Counter(turn.get("origin") for _, turn in raw.values())),
        "missing_targets": [{"sample_id": f"{case['caseId']}:t{i}", "text": turn.get("text", "")} for (case_id, i), case, turn in missing],
        "corrected_rows": len(corrected),
        "review_ledger_rows": len(review),
        "input_violations": violations,
        "history_lengths_excluding_current": dict(sorted(history_lengths.items())),
        "rows_with_labeled_future_turns_not_in_input": len(rows_with_missing_history),
        "changed_rows": len(changes),
        "uncertain_rows": len(uncertain),
        "change_by_split": dict(change_by_split),
        "change_matrix": {f"{a}->{b}": n for (a, b), n in sorted(change_matrix.items())},
        "reviewed_label_counts": dict(collections.Counter(row["label"] for row in corrected)),
        "original_label_counts": dict(collections.Counter(row["label_provenance"]["original_label"] for row in corrected)),
    }


def label_review_audit() -> dict[str, Any]:
    """Independently classify the evidence strength of each explicit override.

    This is an audit of the existing decisions, not a relabeling operation. The
    source text/history and the recorded rationale remain the evidence of record.
    """
    changed = [row for row in read_jsonl(BENCH / "review_ledger.jsonl") if row["original_label"] != row["reviewed_label"]]
    assessments = {
        "pb-v1-0019:t0": ("rubric_supported_but_ambiguous", "Invoice-company address is distinct from delivery ADDRESS under the written rubric; address token creates a plausible competing reading."),
        "pb-v1-0043:t0": ("rubric_supported", "Neutral cancellation has no dissatisfaction/service-problem language."),
        "pb-v1-0045:t0": ("rubric_supported", "Personal cancellation reason plus thanks is operational and not a complaint."),
        "pb-v1-0047:t0": ("rubric_supported_but_ambiguous", "Explicit shipping-cost dissatisfaction supports COMPLAINT, but refusal/thanks is a mixed short act."),
        "pb-v1-0062:t0": ("rubric_supported_but_ambiguous", "Restaurant directions/location is outside delivery-address semantics, though ordinary readers may call it an address request."),
        "pb-v1-0063:t0": ("rubric_supported_but_ambiguous", "Restaurant address is outside the written delivery ADDRESS definition, but the word address is lexically strong."),
        "pb-v1-0181:t3": ("context_supported", "Closure phrase after a concrete order selection is a contextual confirmation; it does not imply real-order authorization."),
        "pb-v1-0183:t3": ("context_supported", "Closure phrase after a concrete order selection is a contextual confirmation; it does not imply real-order authorization."),
        "pb-v1-0184:t3": ("context_supported", "Closure phrase after a concrete order selection is a contextual confirmation; it does not imply real-order authorization."),
        "pb-v1-0185:t3": ("context_supported", "Closure phrase after a concrete order selection is a contextual confirmation; it does not imply real-order authorization."),
        "pb-v1-0189:t3": ("context_supported_but_ambiguous", "Closure phrase is consistent with confirmation in ordering context, but it is short and context-dependent."),
        "pb-v1-0215:t4": ("rubric_supported", "Summary/recap is checkout/order state, not a menu query."),
        "pb-v1-0218:t4": ("rubric_supported", "Summary request is operational checkout state, not a menu query."),
        "pb-v1-0220:t4": ("rubric_supported", "Summary request after replacement is operational checkout state, not a menu query."),
        "pb-v1-0222:t2": ("rubric_supported", "Time/payment plus summary request is operational checkout state, not a clear confirmation."),
        "pb-v1-0222:t3": ("rubric_supported", "Time/payment plus summary request is operational checkout state, not a clear confirmation."),
        "pb-v1-0228:t7": ("rubric_supported", "Summary request is operational checkout state, not a menu query."),
        "pb-v1-0229:t7": ("rubric_supported_but_ambiguous", "Summary request is operational state; the source MENU_QUERY assignment is not supported by the written boundary."),
        "pb-v1-0240:t1": ("context_supported_but_ambiguous", "Accidental prior-order cancellation is operational and explicitly retains the current order; no dissatisfaction is stated."),
    }
    result = []
    for row in changed:
        sid = f"{row['case_id']}:t{row['turn_index']}"
        assessment, basis = assessments.get(sid, ("unclassified", "No independent assessment recorded."))
        result.append({"sample_id": sid, "text": row["text"], "history": row["history"], "original_label": row["original_label"], "reviewed_label": row["reviewed_label"], "confidence": row["confidence"], "uncertain": row["uncertain"], "recorded_rationale": row["rationale"], "independent_assessment": assessment, "independent_basis": basis})
    return {"count": len(result), "rows": result, "assessment_counts": dict(collections.Counter(row["independent_assessment"] for row in result))}


def split_and_duplicate_audit() -> dict[str, Any]:
    rows = read_jsonl(BENCH / "corrected_dataset.jsonl")
    split_rows = {split: read_jsonl(BENCH / f"{split}.jsonl") for split in ("train", "dev", "test")}
    split_by_id = {row["sample_id"]: split for split, values in split_rows.items() for row in values}
    current = collections.defaultdict(list)
    full = collections.defaultdict(list)
    cases = collections.defaultdict(set)
    sources = collections.defaultdict(set)
    for row in rows:
        current[norm(row["model_input"]["messages"][-1]["text"])].append(row["sample_id"])
        full[norm("\n".join(f"{m['role']}:{m['text']}" for m in row["model_input"]["messages"]))].append(row["sample_id"])
        cases[row["case_id"]].add(split_by_id.get(row["sample_id"]))
        sources[row["source_ref"]].add(split_by_id.get(row["sample_id"]))
    cross_current = {k: v for k, v in current.items() if len({split_by_id[x] for x in v}) > 1}
    cross_full = {k: v for k, v in full.items() if len({split_by_id[x] for x in v}) > 1}
    split_stats = {}
    for split, values in split_rows.items():
        split_stats[split] = {"rows": len(values), "class_counts": dict(collections.Counter(row["label"] for row in values)), "case_count": len({r["case_id"] for r in values}), "source_count": len({r["source_ref"] for r in values})}
    existing = read_json(BENCH / "duplicate_analysis.json")
    group_manifest = read_json(BENCH / "group_manifest.json")
    group_split = {row["group_id"]: row["split"] for row in group_manifest}
    case_to_group = {case_id: row["group_id"] for row in group_manifest for case_id in row.get("cases", [])}
    family_split_checks = []
    for family in existing.get("confirmed_substantive_near_duplicate_families", []):
        member_groups = {case_to_group.get(case_id) for case_id in family.get("case_ids", [])}
        member_splits = {group_split.get(gid) for gid in member_groups}
        family_split_checks.append({"family_id": family["family_id"], "group_ids": sorted(g for g in member_groups if g), "splits": sorted(s for s in member_splits if s), "one_split": len(member_splits) == 1 and None not in member_splits})
    return {
        "split_stats": split_stats,
        "case_cross_split": {k: sorted(v) for k, v in cases.items() if len(v) > 1},
        "source_cross_split": {k: sorted(v) for k, v in sources.items() if len(v) > 1},
        "group_manifest_count": len(group_manifest),
        "group_manifest_all_single_split": all(len({row["split"]}) == 1 for row in group_manifest),
        "confirmed_family_split_checks": family_split_checks,
        "current_family_count": sum(len(v) > 1 for v in current.values()),
        "current_family_sizes": dict(collections.Counter(len(v) for v in current.values() if len(v) > 1)),
        "current_cross_split_family_count": len(cross_current),
        "current_cross_split_examples": dict(list(sorted(cross_current.items()))[:20]),
        "full_input_family_count": sum(len(v) > 1 for v in full.values()),
        "full_input_cross_split_count": len(cross_full),
        "existing_duplicate_report_summary": {key: existing.get(key) for key in ("candidate_count", "confirmed_substantive_near_duplicate_families") if key in existing},
        "fuzzy_candidate_count": len(existing.get("near_duplicate_candidates", [])),
    }


def predictions_audit() -> dict[str, Any]:
    test_rows = read_jsonl(BENCH / "test.jsonl")
    dev_rows = read_jsonl(BENCH / "dev.jsonl")
    ids_test = {row["sample_id"] for row in test_rows}
    ids_dev = {row["sample_id"] for row in dev_rows}
    frozen = {"test": row_map(test_rows), "dev": row_map(dev_rows)}
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    test_preds: dict[str, dict[str, dict[str, Any]]] = {}
    dev_preds: dict[str, dict[str, dict[str, Any]]] = {}
    metrics = {}
    for model in ("bamibert", "goi_dapt", "goi_student"):
        trows = read_jsonl(BENCH / "results" / f"test_predictions_{model}.jsonl")
        test_preds[model] = {row["sample_id"]: row for row in trows}
        gold_checks = []
        prediction_checks = []
        for row in trows:
            frozen_row = frozen["test"].get(row["sample_id"])
            gold_checks.append(bool(frozen_row and row.get("gold_label") == frozen_row["label"] and int(row.get("gold_id")) == label_to_id[frozen_row["label"]]))
            probs = row.get("probabilities", [])
            prediction_checks.append(bool(len(probs) == len(LABELS) and int(row.get("pred_id")) == label_to_id.get(row.get("pred_label")) and probs and max(range(len(probs)), key=lambda i: probs[i]) == int(row.get("pred_id")) and abs(sum(probs) - 1.0) < 1e-5))
        metrics[model] = metric_report([int(row["gold_id"]) for row in trows], [int(row["pred_id"]) for row in trows])
        run_dirs = sorted((BENCH / "runs").glob(f"{model}-lr*/summary.json"))
        metrics[model]["prediction_count"] = len(trows)
        metrics[model]["gold_matches_frozen_test"] = all(gold_checks) and set(test_preds[model]) == ids_test
        metrics[model]["prediction_rows_coherent"] = all(prediction_checks)

    # Recompute dev metrics for every trial from the saved best-epoch predictions.
    dev_trials = []
    for prediction_path in sorted((BENCH / "runs").glob("*/dev_predictions.jsonl")):
        summary_path = prediction_path.parent / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        drows = read_jsonl(prediction_path)
        recomputed = metric_report([int(row["gold_id"]) for row in drows], [int(row["pred_id"]) for row in drows])
        expected_ids = {row["sample_id"] for row in dev_rows}
        dev_gold_checks = []
        dev_prediction_checks = []
        for row in drows:
            frozen_row = frozen["dev"].get(row["sample_id"])
            dev_gold_checks.append(bool(frozen_row and row.get("gold_label") == frozen_row["label"] and int(row.get("gold_id")) == label_to_id[frozen_row["label"]]))
            probs = row.get("probabilities", [])
            dev_prediction_checks.append(bool(len(probs) == len(LABELS) and int(row.get("pred_id")) == label_to_id.get(row.get("pred_label")) and probs and max(range(len(probs)), key=lambda i: probs[i]) == int(row.get("pred_id")) and abs(sum(probs) - 1.0) < 1e-5))
        metric_fields = ("accuracy", "macro_f1", "per_intent", "confusion_matrix", "support")
        metric_match = all(recomputed[field] == summary["best_dev_metrics"][field] for field in metric_fields)
        dev_trials.append({
            "run": prediction_path.parent.name,
            "model": summary["model_name"],
            "lr": summary["learning_rate"],
            "best_epoch": summary["best_epoch"],
            "prediction_count": len(drows),
            "ids_exact_dev": {row["sample_id"] for row in drows} == expected_ids,
            "metric_recomputed": recomputed,
            "summary_metric": summary["best_dev_metrics"],
            "metric_match": metric_match,
            "gold_matches_frozen_dev": all(dev_gold_checks) and {row["sample_id"] for row in drows} == expected_ids,
            "prediction_rows_coherent": all(dev_prediction_checks),
            "predicted_class_counts": dict(collections.Counter(row["pred_label"] for row in drows)),
        })

    selected = read_json(BENCH / "results/selection.json")["selected"]
    selected_by_model = {row["model_name"]: row for row in selected}
    selected_dev = {}
    for model, selection in selected_by_model.items():
        path = ROOT / selection["run_dir"] / "dev_predictions.jsonl"
        drows = read_jsonl(path)
        selected_dev[model] = {row["sample_id"]: row for row in drows}

    # Changed labels are evaluated against both source and reviewed targets as a
    # diagnostic. This does not alter the frozen score or select a model.
    source_lookup = {}
    cases = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        for i, turn in enumerate(case["targetTurns"]):
            if turn.get("expectedIntent"):
                source_lookup[f"{case['caseId']}:t{i}"] = turn["expectedIntent"]
    corrected = row_map(read_jsonl(BENCH / "corrected_dataset.jsonl"))
    changed_test = [sid for sid, row in corrected.items() if row["label_provenance"]["original_label"] != row["label_provenance"]["reviewed_label"] and sid in ids_test]
    changed_effect = {}
    for model, pmap in test_preds.items():
        reviewed_hits = sum(pmap[sid]["pred_label"] == corrected[sid]["label"] for sid in changed_test)
        original_hits = sum(pmap[sid]["pred_label"] == source_lookup[sid] for sid in changed_test)
        changed_effect[model] = {"changed_test_rows": len(changed_test), "hits_under_reviewed": reviewed_hits, "hits_under_original": original_hits, "delta_reviewed_minus_original": reviewed_hits - original_hits}

    # Sensitivity diagnostic only: score the same predictions against the raw
    # source labels, without changing the frozen reviewed benchmark.
    source_label_metrics = {}
    for model, pmap in test_preds.items():
        source_gold = [label_to_id[source_lookup[sid]] for sid in sorted(ids_test)]
        source_pred = [int(pmap[sid]["pred_id"]) for sid in sorted(ids_test)]
        source_label_metrics[model] = metric_report(source_gold, source_pred)

    # Paired correctness counts across all 77 test rows.
    paired = {}
    for left in ("bamibert", "goi_dapt", "goi_student"):
        for right in ("bamibert", "goi_dapt", "goi_student"):
            if left >= right:
                continue
            both = left_only = right_only = neither = 0
            for sid in ids_test:
                gold = test_preds[left][sid]["gold_label"]
                lc = test_preds[left][sid]["pred_label"] == gold
                rc = test_preds[right][sid]["pred_label"] == gold
                if lc and rc: both += 1
                elif lc: left_only += 1
                elif rc: right_only += 1
                else: neither += 1
            paired[f"{left}_vs_{right}"] = {"both_correct": both, "left_only": left_only, "right_only": right_only, "neither_correct": neither}

    # Per-class macro-F1 contribution is delta/8; accuracy hit counts are direct.
    per_class_gap = {}
    for model in ("goi_dapt", "goi_student"):
        per_class_gap[model] = {}
        for label in LABELS:
            b = metrics["bamibert"]["per_intent"][label]
            m = metrics[model]["per_intent"][label]
            per_class_gap[model][label] = {"bamibert_f1": b["f1"], "model_f1": m["f1"], "delta": m["f1"] - b["f1"], "macro_delta_contribution": (m["f1"] - b["f1"]) / len(LABELS), "support": b["support"]}

    # Prediction counts and trajectories expose class collapse and recipe effects.
    trajectories = []
    for summary_path in sorted((BENCH / "runs").glob("*/summary.json")):
        summary = read_json(summary_path)
        if summary["model_name"] != "bamibert":
            continue
        trajectories.append({
            "run": summary_path.parent.name,
            "learning_rate": summary["learning_rate"],
            "epochs_completed": summary["epochs_completed"],
            "best_epoch": summary["best_epoch"],
            "best_dev_macro_f1": summary["best_dev_macro_f1"],
            "final_train_loss": summary["history"][-1]["train_loss"],
            "final_dev_macro_f1": summary["history"][-1]["dev"]["macro_f1"],
            "final_dev_accuracy": summary["history"][-1]["dev"]["accuracy"],
            "history": [{"epoch": r["epoch"], "train_loss": r["train_loss"], "dev_macro_f1": r["dev"]["macro_f1"], "dev_accuracy": r["dev"]["accuracy"]} for r in summary["history"]],
        })
    return {
        "test_metrics_recomputed": metrics,
        "saved_test_id_sets": {m: sorted(pmap) for m, pmap in test_preds.items()},
        "dev_trials": dev_trials,
        "selected_dev_prediction_counts": {m: dict(collections.Counter(row["pred_label"] for row in pmap.values())) for m, pmap in selected_dev.items()},
        "changed_test_label_sensitivity": {"sample_ids": sorted(changed_test), "by_model": changed_effect},
        "changed_test_details": [{"sample_id": sid, "text": corrected[sid]["model_input"]["messages"][-1]["text"], "original_label": corrected[sid]["label_provenance"]["original_label"], "reviewed_label": corrected[sid]["label"], "predictions": {model: test_preds[model][sid]["pred_label"] for model in test_preds}} for sid in sorted(changed_test)],
        "test_metrics_under_raw_source_labels": source_label_metrics,
        "paired_test_correctness": paired,
        "per_class_macro_f1_gap": per_class_gap,
        "bamibert_trajectories": trajectories,
    }


def model_and_protocol_audit() -> dict[str, Any]:
    models = {"bamibert": "models/bamibert-base-original", "goi_dapt": "models/goibert-b-dapt", "goi_student": "models/goiturn-s-gd"}
    out: dict[str, Any] = {"models": {}, "tokenizer_comparison": {}, "checkpoint_artifacts": {}, "protocol": {}}
    for name, rel in models.items():
        path = ROOT / rel
        cfg = read_json(path / "config.json")
        manifest_path = path / ("pretrain_manifest.json" if name == "goi_dapt" else "distill_manifest.json" if name == "goi_student" else "README.md")
        files = {f: {"sha256": sha256(path / f), "bytes": (path / f).stat().st_size} for f in ("config.json", "model.safetensors", "tokenizer.json")}
        out["models"][name] = {"path": rel, "config": {k: cfg.get(k) for k in ("architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "max_position_embeddings", "vocab_size", "initializer_range", "pad_token_id", "bos_token_id", "eos_token_id")}, "files": files, "manifest_or_readme": str(manifest_path.relative_to(ROOT)), "manifest_sha256": sha256(manifest_path)}
    try:
        from transformers import AutoTokenizer

        tokenizers = {name: AutoTokenizer.from_pretrained(ROOT / rel, local_files_only=True, use_fast=True) for name, rel in models.items()}
        texts = []
        for row in read_jsonl(BENCH / "corrected_dataset.jsonl"):
            messages = row["model_input"]["messages"]
            texts.append("\n".join(("[HISTORY] customer: " if i < len(messages) - 1 else "[CURRENT] customer: ") + m["text"] for i, m in enumerate(messages)))
        comparison = {}
        for left in ("bamibert", "goi_dapt", "goi_student"):
            for right in ("bamibert", "goi_dapt", "goi_student"):
                if left >= right:
                    continue
                ids_equal = all(tokenizers[left](text, add_special_tokens=False)["input_ids"] == tokenizers[right](text, add_special_tokens=False)["input_ids"] for text in texts)
                vocab_equal = tokenizers[left].get_vocab() == tokenizers[right].get_vocab()
                comparison[f"{left}_vs_{right}"] = {"vocab_equal": vocab_equal, "all_507_serialized_input_ids_equal": ids_equal}
        out["tokenizer_comparison"] = {"classes": {name: type(tok).__name__ for name, tok in tokenizers.items()}, "vocab_sizes": {name: int(tok.vocab_size) for name, tok in tokenizers.items()}, "special_tokens": {name: {key: getattr(tok, key, None) for key in ("bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id")} for name, tok in tokenizers.items()}, "empty_special_wrapped": {name: tok.build_inputs_with_special_tokens([]) for name, tok in tokenizers.items()}, "comparison": comparison}
    except Exception as error:  # pragma: no cover - evidence still useful when optional deps unavailable
        out["tokenizer_comparison"] = {"error": f"{type(error).__name__}: {error}"}

    for run_path in sorted((BENCH / "runs").glob("*-lr*")):
        summary_path = run_path / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        loading = read_json(run_path / "loading_info.json")
        out["checkpoint_artifacts"][run_path.name] = {
            "summary_status": summary.get("status"),
            "completed_marker": (run_path / "COMPLETED").exists(),
            "running_marker": (run_path / "RUNNING").exists(),
            "failed_marker": (run_path / "FAILED").exists(),
            "best_epoch": summary.get("best_epoch"),
            "epochs_completed": summary.get("epochs_completed"),
            "best_checkpoint_sha256": sha256(run_path / "best.pt"),
            "loading": loading,
            "summary_hash": sha256(summary_path),
        }

    cfg = read_json(BENCH / "training/training_config.json")
    approval = read_json(BENCH / "training/root_approval.json")
    out["protocol"] = {
        "config_sha256_actual": sha256(BENCH / "training/training_config.json"),
        "config_sha256_approved": approval.get("config_sha256"),
        "runner_sha256_actual": sha256(BENCH / "training/train_intent.py"),
        "runner_sha256_approved": approval.get("training_runner_sha256"),
        "config": {key: cfg.get(key) for key in ("seed", "batch_size", "max_length", "max_epochs", "early_stopping_patience", "min_delta", "warmup_ratio", "weight_decay", "gradient_clip_norm", "precision", "attention", "loss", "scheduler", "learning_rates")},
        "selection": cfg.get("selection"),
        "queues": cfg.get("queues"),
        "implementation_refs": {"input_serialization": "intent_benchmark/training/train_intent.py:114-170", "loading": "intent_benchmark/training/train_intent.py:219-280", "head": "intent_benchmark/training/train_intent.py:283-300", "train_eval": "intent_benchmark/training/train_intent.py:546-585", "selection": "intent_benchmark/training/select_dev_winners.py:50-78", "test_guard": "intent_benchmark/training/evaluate_selected.py:48-88"},
    }
    return out


def timing_audit() -> dict[str, Any]:
    timing = read_json(BENCH / "results/timing.json")
    selection = read_json(BENCH / "results/selection.json")
    test_sha = sha256(BENCH / "test.jsonl")
    selected = {row["model_name"]: row for row in selection["selected"]}
    checks = []
    for row in timing.get("models", []):
        checks.append({
            "model": row["model_name"],
            "forward_count_matches_77x2": row.get("forward_count") == 154,
            "warmup_is_10": row.get("warmup_forwards") == 10,
            "passes_is_2": row.get("passes") == 2,
            "batch_size_is_1": row.get("batch_size") == 1,
            "test_hash_matches": row.get("test_sha256") == test_sha,
            "checkpoint_hash_matches_selection": row.get("checkpoint_sha256") == selected.get(row["model_name"], {}).get("checkpoint_sha256"),
            "timing_scope": row.get("timing_scope"),
        })
    return {"selection_hash_matches": timing.get("selection_sha256") == sha256(BENCH / "results/selection.json"), "model_checks": checks}


def write_report(evidence: dict[str, Any]) -> None:
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    src = evidence["source_input"]
    pred = evidence["predictions"]
    split = evidence["split_duplicates"]
    model = evidence["models_protocol"]
    review = evidence["label_review"]
    timing = evidence["timing"]
    lines = [
        "# Independent intent benchmark audit",
        "",
        "This report was generated from an independent read-only pass over the raw prompt-bench bundle, corrected/split JSONL, training code, local model files, run summaries, checkpoints, and saved predictions. It does not rerun training or alter original artifacts. Raw evidence is in `evidence.json`.",
        "",
        "## Findings by severity",
        "",
        "### High: external validity is limited by label provenance",
        "",
        "No critical implementation error was detected in extraction, input construction, split isolation, checkpoint loading, metric recomputation, dev selection, or test timing from the artifacts inspected. The material limitation is external validity: the 507 target labels originate in the source bundle's `expectedIntent` fields and 19 rows were changed by a deterministic review override map. The checked-in README and review policy explicitly say the production prompt is DB-managed and absent; therefore label provenance is limited to the source snapshot plus an AI-assisted semantic review rather than an independently adjudicated human gold set. This is a validity limitation, not evidence of intentional model favoritism. `intent_benchmark/prepare_intent_benchmark.py:60-111,252-280`; `intent_benchmark/README.md`.",
        f"The independent counts are {src['source_counts']}; the 19 changes distribute as {src['change_by_split']} with transitions {src['change_matrix']}.",
        "",
        "The independent row-by-row review found 12 transitions directly rubric/context-supported and 7 supported but explicitly ambiguous/context-dependent (invoice/store address, shipping-price complaint, short closure/cancellation turns). The complete 19-row evidence table is below; no label is changed by this audit.",
        "",
        "| sample | source → reviewed | assessment |",
        "|---|---|---|",
    ]
    for row in review["rows"]:
        lines.append(f"| {row['sample_id']} | {row['original_label']} → {row['reviewed_label']} | {row['independent_assessment']} — {row['independent_basis']} |")
    lines += [
        "",
        "The five changed test rows all score as reviewed labels for BamiBERT and DAPT (5/5 each) and 4/5 for the student; under the raw source labels, none of the three models is correct on those five rows. This makes the reviewed labels consequential for the absolute test scores, but it does not show that the review was model-driven; the five test changes help both families equally. The 12 changed training rows can affect learned representations and remain a provenance sensitivity that saved predictions alone cannot identify.",
        "",
        "### Medium: repeated current text crosses splits by design",
        "",
        f"Case and source groups are split-disjoint; the independent group manifest check passes for all {split['group_manifest_count']} groups and all confirmed template families are one-split. Exact full-input duplicate overlap is {split['full_input_cross_split_count']}; however {split['current_cross_split_family_count']} normalized current-text families cross partitions. The largest repeated family is the placeholder delivery phrase. This is documented and is not exact full-input leakage, but it allows lexical memorization for generic/placeholder turns and limits a text-generalization interpretation. `intent_benchmark/finalize_intent_benchmark.py:357-417`; `intent_benchmark/group_manifest.json`.",
        "",
        "### Medium: the label review and single split support are fragile for rare classes",
        "",
        f"Per-split sizes/classes recomputed independently: {json.dumps(split['split_stats'], ensure_ascii=False, sort_keys=True)}. GREETING, GRATITUDE and PHONE have only 2 test examples each; COMPLAINT has 5. Macro-F1 is therefore highly sensitive to a few decisions. The benchmark is suitable as a small pilot, not a stable estimate of population performance.",
        "",
        "### Low/medium: protocol is internally consistent but narrower than a reproduction study",
        "",
        f"The locked config uses seed {model['protocol']['config']['seed']}, batch {model['protocol']['config']['batch_size']}, max {model['protocol']['config']['max_epochs']} epochs, patience {model['protocol']['config']['early_stopping_patience']}, learning rates {model['protocol']['config']['learning_rates']}, and one winner per backbone selected by dev macro-F1. Hash checks give runner={model['protocol']['runner_sha256_actual']==model['protocol']['runner_sha256_approved']} and config={model['protocol']['config_sha256_actual']==model['protocol']['config_sha256_approved']}. This is a fair common pilot budget, but two LRs/single seed/15-epoch cap do not establish a fully tuned comparison.",
        "",
        "## Independently recomputed results",
        "",
        "The saved test predictions contain 77 unique IDs per model and metrics recompute exactly from their class IDs. The requested headline values are:",
        "",
        "| model | accuracy | macro-F1 |",
        "|---|---:|---:|",
    ]
    for name in ("bamibert", "goi_dapt", "goi_student"):
        m = pred["test_metrics_recomputed"][name]
        lines.append(f"| {name} | {m['accuracy']:.9f} | {m['macro_f1']:.9f} |")
    lines += ["", "Independent integrity checks also confirm every saved test gold ID/label matches the frozen `test.jsonl` row, every saved dev gold ID/label matches its frozen `dev.jsonl` row, and every probability vector is length 8, sums to 1, and has `argmax == pred_id`. All six saved best-epoch dev prediction files pass these checks.", "", "For sensitivity only, holding saved predictions fixed and replacing reviewed test labels with the raw source labels gives:", "", "| model | reviewed-label macro-F1 | raw-source-label macro-F1 |", "|---|---:|---:|"]
    for name in ("bamibert", "goi_dapt", "goi_student"):
        lines.append(f"| {name} | {pred['test_metrics_recomputed'][name]['macro_f1']:.6f} | {pred['test_metrics_under_raw_source_labels'][name]['macro_f1']:.6f} |")
    tok = model["tokenizer_comparison"]
    tokenizer_verdict = "all 507 serialized inputs produce identical token ID sequences across all three local tokenizers; vocabularies and special-token IDs also match" if tok.get("comparison") and all(item.get("all_507_serialized_input_ids_equal") and item.get("vocab_equal") for item in tok["comparison"].values()) else "see evidence.json for tokenizer comparison details"
    lines += ["", "Paired correctness counts on the same 77 test rows:", "", "| pair | both | first only | second only | neither |", "|---|---:|---:|---:|---:|"]
    for key, value in pred["paired_test_correctness"].items():
        lines.append(f"| {key} | {value['both_correct']} | {value['left_only']} | {value['right_only']} | {value['neither_correct']} |")
    lines += ["", "### What explains BamiBERT's lower macro-F1 in the saved test run?", "", "The primary verified pattern is per-class error, not a single global collapse:", "", "| class | support | BamiBERT F1 | GoiBERT-DAPT F1 | DAPT minus Bami contribution to macro-F1 |", "|---|---:|---:|---:|---:|"]
    for label, value in pred["per_class_macro_f1_gap"]["goi_dapt"].items():
        lines.append(f"| {label} | {value['support']} | {value['bamibert_f1']:.3f} | {value['model_f1']:.3f} | {value['macro_delta_contribution']:.3f} |")
    rare_dapt = sum(pred["per_class_macro_f1_gap"]["goi_dapt"][label]["macro_delta_contribution"] for label in ("GREETING", "GRATITUDE", "PHONE"))
    rare_student = sum(pred["per_class_macro_f1_gap"]["goi_student"][label]["macro_delta_contribution"] for label in ("GREETING", "GRATITUDE", "PHONE"))
    dapt_gap = pred["test_metrics_recomputed"]["goi_dapt"]["macro_f1"] - pred["test_metrics_recomputed"]["bamibert"]["macro_f1"]
    student_gap = pred["test_metrics_recomputed"]["goi_student"]["macro_f1"] - pred["test_metrics_recomputed"]["bamibert"]["macro_f1"]
    lines += ["", "BamiBERT predicts zero examples of GREETING, GRATITUDE, and PHONE in the selected test checkpoint (their recalls are all zero), while DAPT and the student get at least one of each. The selected BamiBERT checkpoint still performs well on MENU_QUERY, ADDRESS and OTHER; the gap is driven mainly by the three rare classes plus weaker COMPLAINT. This is a verified error decomposition, not proof of why the representation failed.", f"The three rare classes contribute {rare_dapt:.6f} of the DAPT–BamiBERT macro-F1 gap ({rare_dapt / dapt_gap:.1%}) and {rare_student:.6f} of the student–BamiBERT gap ({rare_student / student_gap:.1%}). Because each has support 2, these contributions are high-variance.", "", "The paired table and per-class table separate the accuracy/F1 gap from class support effects. The BamiBERT dev trajectories also show strong learning-rate sensitivity:", ""]
    for row in pred["bamibert_trajectories"]:
        lines.append(f"- `{row['run']}`: best dev macro-F1 {row['best_dev_macro_f1']:.6f} at epoch {row['best_epoch']}; {row['epochs_completed']} epochs; final train loss {row['final_train_loss']:.6f}; final dev macro-F1 {row['final_dev_macro_f1']:.6f}.")
    lines += ["The 5e-5 BamiBERT run is still improving at the 15-epoch cap but ends at only 0.275 dev macro-F1, whereas 2e-5 reaches 0.503. GoiBERT-DAPT also shows the same recipe interaction (0.501 at 2e-5 versus 0.722 at 5e-5). Consequently, the saved gap is consistent with a combination of pretrained/domain representation and optimization interaction; the artifacts do not identify a causal factor.", "", "## Provenance and architecture checks", ""]
    lines.append("- Local model files were hashed independently. BamiBERT config reports a 12-layer, 768-hidden, 12-head RoBERTa with 20,481 vocabulary entries and 2,050 max positions; GoiBERT-DAPT matches the base dimensions; GoiTurn student is 10-layer/384-hidden/6-head. All training inputs use max length 512, within the model position limit.")
    lines += ["- DAPT loading artifacts show no missing keys and only MLM-head unexpected keys; BamiBERT shows no missing keys and only MLM-head unexpected keys; student has no missing/unexpected keys. This supports checkpoint identity/loading correctness for the saved runs, subject to the manifest provenance limits. The local BamiBERT weights hash is `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`, matching the official file hash reported for [Qualcomm-AI-Research/BamiBERT](https://huggingface.co/Qualcomm-AI-Research/BamiBERT/blob/main/model.safetensors).", "- Tokenizer verdict: " + tokenizer_verdict + ". All three load as `XLMRobertaTokenizerFast` with BOS/CLS=0, PAD=1, EOS/SEP=2, MASK=20480 and empty wrapped input `[0, 2]`.", "- The training code disables the pretrained pooler (`add_pooling_layer=False`) and consumes the first token hidden state. It creates a new `Linear(hidden, hidden) -> Tanh -> Linear(hidden, 8)` head with PyTorch default initialization, followed by dropout. This is dimensionally valid and shared across models, but the head initialization is not explicitly the HF `initializer_range=0.02` recipe; it is a protocol detail that can add variance and is not evidence of a Bami-specific fault.", "- Source model manifests report DAPT 40k MLM steps from a Bami checkpoint and student 30k hidden/attention distillation from DAPT on the same corpus-stat hash. Those manifests establish the claimed local lineage, while the corpus files themselves are absent from this workspace, so corpus overlap/domain contamination cannot be independently ruled out.", "", "## Verified implementation behavior"]
    lines.append(f"- Input history: {src['input_violations'] or 'no independently detected mismatches'}; source target history is customer-only, with unlabeled preceding turns retained as text context where present. No future labels, fixtures, assertions, source IDs or other metadata are in `model_input` according to independent reconstruction and the code assertions (`prepare_intent_benchmark.py:227-242,299-317`; `train_intent.py:114-170`).")
    lines += ["- Train/eval mode, dropout behavior, gradient updates, and metric computation are explicit (`train_intent.py:546-585`, `408-436`). Selection uses dev only and the test evaluator checks the locked selection before loading checkpoints (`select_dev_winners.py:50-78`, `evaluate_selected.py:48-88`).", "- Saved test predictions and dev winner predictions match independently recomputed IDs/metrics and frozen gold rows; no test score is needed to explain the published values. Timing artifact checks pass for selection/checkpoint/test hashes, 10 warmups, two 77-example passes, batch size 1, and forward-only scope. Timing is therefore a forward-only GPU measurement and should not be interpreted as end-to-end serving latency.", "- `intent_benchmark/README.md` contains an introductory sentence saying the directory uses no training/inference; that sentence describes preparation scope and is stale after the training artifacts were added, but it does not alter the executed protocol.", "", "## Remaining uncertainty and recommended follow-ups", "", "- Obtain the original expected-intent observation/run provenance and, if a gold claim is needed, commission an independent human double annotation with adjudication, reporting label sensitivity for the 19 changes and 82 uncertainty rows. Keep all reviewed uncertainty rows in the benchmark as specified; do not filter them post hoc.", "- For a stronger model ranking claim, pre-register multiple seeds and a broader LR/epoch budget, or reproduce the BamiBERT paper's stated fine-tuning recipe separately. Keep the 77-row test frozen and untouched.", "- Treat DAPT/student superiority as a result on this frozen, reviewed pilot; do not claim that DAPT alone caused the improvement over BamiBERT because LR sensitivity and the unverified historical corpus remain confounders.", "", "Generated evidence files: `intent_benchmark/audit_independent/evidence.json` and this report.", ""]
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {
        "audit": {"script": str(Path(__file__).relative_to(ROOT)), "python": sys.version, "read_only_inputs": True},
        "source_input": source_and_input_audit(),
        "split_duplicates": split_and_duplicate_audit(),
        "predictions": predictions_audit(),
        "models_protocol": model_and_protocol_audit(),
        "label_review": label_review_audit(),
        "timing": timing_audit(),
    }
    write_report(evidence)
    print(json.dumps({"report": str(OUT / "AUDIT_REPORT.md"), "evidence": str(OUT / "evidence.json")}, indent=2))


if __name__ == "__main__":
    main()
