#!/usr/bin/env python3
"""Inference-only controlled aspect/polarity intervention diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from goibert import train as tr  # noqa: E402
from reader_operating_point import load_checkpoint  # noqa: E402
from shadow_reader import FullCapture  # noqa: E402


MANIFEST = ROOT / "docs" / "CONTROLLED_ASPECT_INTERVENTIONS.json"
OUT = ROOT / "artifacts" / "research_diagnostics" / "controlled_aspect_interventions"
MODELS = ("student_e8", "bami_e14")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n", encoding="utf-8")
    tmp.replace(path)


def field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def aspect_indices(bundle: Any) -> dict[str, int]:
    names = list(field(bundle, "aspect_names", ()))
    if not names:
        names = list(field(bundle, "slot_names", ()))
    return {str(name): index for index, name in enumerate(names)}


def build_inputs(manifest: dict[str, Any], task: str, bundle: Any, tokenizer: Any) -> list[dict[str, Any]]:
    indices = aspect_indices(bundle)
    inputs: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        if pair["task"] != task:
            continue
        pair_id = pair["id"]
        a_name, b_name = pair["a"]["aspect"], pair["b"]["aspect"]
        a_idx, b_idx = indices[a_name], indices[b_name]
        def add(kind: str, text: str, expected: dict[str, int], **meta: Any) -> None:
            input_id = f"{pair_id}:{kind}:{len(inputs):03d}"
            token_count = len(tokenizer(text, truncation=True, max_length=256)["input_ids"])
            inputs.append({"id": input_id, "pair_id": pair_id, "task": task, "kind": kind, "text": text,
                           "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                           "target_slots": sorted(int(x) for x in expected), "expected_labels": {str(k): int(v) for k, v in expected.items()},
                           "token_count": token_count, **meta})
        for side, item, index in (("A", pair["a"], a_idx), ("B", pair["b"], b_idx)):
            add(f"standalone_{side}_positive", item["positive"], {index: 1}, template="standalone", side=side, polarity_a=1 if side == "A" else None, polarity_b=1 if side == "B" else None)
            add(f"standalone_{side}_negative", item["negative"], {index: 2}, template="standalone", side=side, polarity_a=2 if side == "A" else None, polarity_b=2 if side == "B" else None)
        for pa in (1, 2):
            for pb in (1, 2):
                a_text = pair["a"]["positive"] if pa == 1 else pair["a"]["negative"]
                b_text = pair["b"]["positive"] if pb == 1 else pair["b"]["negative"]
                for order in ("AB", "BA"):
                    for connector in manifest["connectors"]:
                        text = (a_text + connector + b_text) if order == "AB" else (b_text + connector + a_text)
                        add(f"factorial_A{pa}_B{pb}_{order}_{'dot' if connector == '. ' else 'and'}", text,
                            {a_idx: pa, b_idx: pb}, template="factorial", order=order, connector=connector, polarity_a=pa, polarity_b=pb, a_slot=a_idx, b_slot=b_idx)
        filler = pair["filler"]
        for side, item, index in (("A", pair["a"], a_idx), ("B", pair["b"], b_idx)):
            for polarity in (1, 2):
                clause = item["positive"] if polarity == 1 else item["negative"]
                for order in ("target_filler", "filler_target"):
                    text = clause + ". " + filler if order == "target_filler" else filler + ". " + clause
                    add(f"filler_{side}_{polarity}_{order}", text, {index: polarity}, template="filler", order=order, side=side, polarity=polarity)
    return inputs


@torch.inference_mode()
def infer(model: Any, tokenizer: Any, inputs: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    capture = FullCapture(model)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(inputs), 16):
        chunk = inputs[start:start + 16]
        enc = tokenizer([row["text"] for row in chunk], padding=True, truncation=True, max_length=256, return_tensors="pt")
        capture.clear(); capture.mask2d = enc["attention_mask"]
        with tr.make_autocast("bf16", device):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu()
        predictions = tr._predictions(logits, aspect_class_counts=model.aspect_class_counts)
        flat = logits.reshape(logits.shape[0], -1)
        for index, item in enumerate(chunk):
            rows.append({**item, "logits": [float(x) for x in flat[index]], "prediction": [int(x) for x in predictions[index]]})
    capture.restore()
    return rows


def target_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def correct(row: dict[str, Any], slot: int) -> bool:
        return int(row["prediction"][slot]) == int(row["expected_labels"][str(slot)])
    standalone = [row for row in rows if row["template"] == "standalone"]
    factorial = [row for row in rows if row["template"] == "factorial"]
    filler = [row for row in rows if row["template"] == "filler"]
    def accuracy(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = sum(len(row["target_slots"]) for row in subset)
        good = sum(correct(row, slot) for row in subset for slot in row["target_slots"])
        return {"correct": good, "denominator": total, "accuracy": good / max(total, 1)}
    joint_good = sum(all(correct(row, slot) for slot in row["target_slots"]) for row in factorial)
    joint = {"correct": joint_good, "denominator": len(factorial), "accuracy": joint_good / max(len(factorial), 1)}
    missed = wrong = 0
    for row in factorial:
        for slot in row["target_slots"]:
            pred = int(row["prediction"][slot]); gold = int(row["expected_labels"][str(slot)])
            missed += pred == 0
            wrong += pred > 0 and pred != gold
    switches = {}
    for pair_id in sorted({row["pair_id"] for row in rows}):
        for side in ("A", "B"):
            s = [row for row in standalone if row["pair_id"] == pair_id and row.get("side") == side]
            if len(s) == 2:
                switches[f"{pair_id}:{side}"] = {"positive_correct": correct(s[0], s[0]["target_slots"][0]), "negative_correct": correct(s[1], s[1]["target_slots"][0]), "both_correct": all(correct(row, row["target_slots"][0]) for row in s)}
    change = {"A": {"total": 0, "changed": 0, "correct_to_wrong": 0, "wrong_to_correct": 0}, "B": {"total": 0, "changed": 0, "correct_after_flip": 0, "both_correct": 0, "sensitivity": 0, "correct_to_wrong": 0, "wrong_to_correct": 0}}
    by_condition = defaultdict(lambda: {"total": 0, "changed": 0, "correct_to_wrong": 0, "wrong_to_correct": 0, "correct_after_flip": 0})
    for pair_id in sorted({row["pair_id"] for row in factorial}):
        for order in ("AB", "BA"):
            for connector in (". ", " và "):
                for pa in (1, 2):
                    base = next((r for r in factorial if r["pair_id"] == pair_id and r["order"] == order and r["connector"] == connector and r["polarity_a"] == pa and r["polarity_b"] == 1), None)
                    flip = next((r for r in factorial if r["pair_id"] == pair_id and r["order"] == order and r["connector"] == connector and r["polarity_a"] == pa and r["polarity_b"] == 2), None)
                    if base and flip:
                        slot = int(base["a_slot"]); change["A"]["total"] += 1; p0, p1 = int(base["prediction"][slot]), int(flip["prediction"][slot]); change["A"]["changed"] += p0 != p1
                        change["A"]["correct_to_wrong"] += p0 == pa and p1 != pa; change["A"]["wrong_to_correct"] += p0 != pa and p1 == pa
                        key = f"A{pa}:{order}:{connector}"; by_condition[key]["total"] += 1; by_condition[key]["changed"] += p0 != p1; by_condition[key]["correct_to_wrong"] += p0 == pa and p1 != pa; by_condition[key]["wrong_to_correct"] += p0 != pa and p1 == pa
                        bslot = int(base["b_slot"]); change["B"]["total"] += 1; q0, q1 = int(base["prediction"][bslot]), int(flip["prediction"][bslot]); change["B"]["changed"] += q0 != q1
                        change["B"]["correct_after_flip"] += q0 == 1 and q1 == 2; change["B"]["both_correct"] += q0 == 1 and q1 == 2; change["B"]["sensitivity"] += q1 == 2; change["B"]["correct_to_wrong"] += q0 == 1 and q1 != 2; change["B"]["wrong_to_correct"] += q0 != 1 and q1 == 2
    self_flip = {"total": 0, "changed": 0, "correct_after_flip": 0, "both_correct": 0, "sensitivity": 0, "correct_to_wrong": 0, "wrong_to_correct": 0}
    for pair_id in sorted({row["pair_id"] for row in factorial}):
        for order in ("AB", "BA"):
            for connector in (". ", " và "):
                for pb in (1, 2):
                    positive = next((r for r in factorial if r["pair_id"] == pair_id and r["order"] == order and r["connector"] == connector and r["polarity_a"] == 1 and r["polarity_b"] == pb), None)
                    negative = next((r for r in factorial if r["pair_id"] == pair_id and r["order"] == order and r["connector"] == connector and r["polarity_a"] == 2 and r["polarity_b"] == pb), None)
                    if not positive or not negative: continue
                    slot = int(positive["a_slot"]); p0, p1 = int(positive["prediction"][slot]), int(negative["prediction"][slot]); self_flip["total"] += 1; self_flip["changed"] += p0 != p1; self_flip["correct_after_flip"] += p0 == 1 and p1 == 2; self_flip["both_correct"] += p0 == 1 and p1 == 2; self_flip["sensitivity"] += p1 == 2; self_flip["correct_to_wrong"] += p0 == 1 and p1 != 2; self_flip["wrong_to_correct"] += p0 != 1 and p1 == 2
    filler_change = {"total": 0, "changed": 0, "error_increase": 0}
    for row in filler:
        side = row["side"]; pol = int(row["polarity"]); slot = int(row["target_slots"][0]); stand = next((x for x in standalone if x["pair_id"] == row["pair_id"] and x.get("side") == side and int(x["expected_labels"][str(slot)]) == pol), None)
        if stand:
            filler_change["total"] += 1; a, b = int(stand["prediction"][slot]), int(row["prediction"][slot]); filler_change["changed"] += a != b; filler_change["error_increase"] += (a == pol and b != pol)
    conditional_good = conditional_total = 0
    standalone_lookup = {(row["pair_id"], row.get("side"), int(next(iter(row["expected_labels"].values())))): row for row in standalone}
    for row in factorial:
        a_row = standalone_lookup.get((row["pair_id"], "A", int(row["polarity_a"])))
        b_row = standalone_lookup.get((row["pair_id"], "B", int(row["polarity_b"])))
        if a_row and b_row and correct(a_row, int(a_row["target_slots"][0])) and correct(b_row, int(b_row["target_slots"][0])):
            conditional_total += 1; conditional_good += all(correct(row, slot) for slot in row["target_slots"])
    order_swap = {"A": {"total": 0, "changed": 0, "correct_to_wrong": 0, "wrong_to_correct": 0}, "B": {"total": 0, "changed": 0, "correct_to_wrong": 0, "wrong_to_correct": 0}}
    for pair_id in sorted({row["pair_id"] for row in factorial}):
        for pa in (1, 2):
            for pb in (1, 2):
                for connector in (". ", " và "):
                    ab = next((r for r in factorial if r["pair_id"] == pair_id and r["polarity_a"] == pa and r["polarity_b"] == pb and r["connector"] == connector and r["order"] == "AB"), None)
                    ba = next((r for r in factorial if r["pair_id"] == pair_id and r["polarity_a"] == pa and r["polarity_b"] == pb and r["connector"] == connector and r["order"] == "BA"), None)
                    if not ab or not ba: continue
                    for side, slot, expected in (("A", int(ab["a_slot"]), pa), ("B", int(ab["b_slot"]), pb)):
                        p0, p1 = int(ab["prediction"][slot]), int(ba["prediction"][slot]); order_swap[side]["total"] += 1; order_swap[side]["changed"] += p0 != p1; order_swap[side]["correct_to_wrong"] += p0 == expected and p1 != expected; order_swap[side]["wrong_to_correct"] += p0 != expected and p1 == expected
    return {"standalone_target_accuracy": accuracy(standalone), "factorial_target_accuracy": accuracy(factorial), "factorial_joint_accuracy": joint,
            "factorial_target_missed": missed, "factorial_target_wrong_polarity": wrong, "switches": switches,
            "A_B_flip_effect": change, "A_self_flip_with_B_held_fixed": self_flip, "A_B_flip_effect_by_A_polarity_order_connector": dict(by_condition), "order_swap_effect": order_swap,
            "filler_vs_standalone": filler_change,
            "conditional_factorial_on_standalone_correct": {"correct": conditional_good, "denominator": conditional_total, "accuracy": conditional_good / max(conditional_total, 1)}}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda:1"); args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")); OUT.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256(MANIFEST); write_json(OUT / "input_manifest.json", {"manifest": manifest, "manifest_sha256": manifest_hash, "protocol": "controlled-aspect-interventions-v1"})
    reports = []; started = time.time()
    for model_name in MODELS:
        for task in ("absa_restaurant", "absa_hotel", "visfd"):
            model, tokenizer, bundle, counts, checkpoint, _ = load_checkpoint(model_name, task, torch.device(args.device))
            inputs = build_inputs(manifest, task, bundle, tokenizer)
            if len(inputs) != 56: raise RuntimeError(f"expected 56 inputs for {task}, got {len(inputs)}")
            rows = infer(model, tokenizer, inputs, torch.device(args.device)); report = {"protocol": "controlled-aspect-interventions-v1", "created_at": datetime.now(timezone.utc).isoformat(), "model": model_name, "task": task, "inputs": len(rows), "manifest_sha256": manifest_hash, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint), "rows": rows, "metrics": target_stats(rows), "runtime_seconds": time.time() - started}
            write_json(OUT / f"{model_name}__{task}.json", report); reports.append(report); del model; torch.cuda.empty_cache()
    summary = {"protocol": "controlled-aspect-interventions-v1", "created_at": datetime.now(timezone.utc).isoformat(), "validation_only": True, "test_access": False, "training": False, "manifest_sha256": manifest_hash, "reports": [{"model": r["model"], "task": r["task"], "inputs": r["inputs"], "metrics": r["metrics"]} for r in reports]}
    write_json(OUT / "summary.json", summary)
    lines = ["# Controlled aspect interventions", "", "Root-authored validation-only fixture; metrics score explicitly annotated target slots only.", "", "| Model | Task | Inputs | Standalone target acc | Factorial joint acc | A changes after B flip | A correct→wrong |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in reports:
        m=r["metrics"]; c=m["A_B_flip_effect"]["A"]; lines.append(f"| {r['model']} | {r['task']} | {r['inputs']} | {m['standalone_target_accuracy']['accuracy']:.3f} | {m['factorial_joint_accuracy']['accuracy']:.3f} | {c['changed']}/{c['total']} | {c['correct_to_wrong']} |")
    (OUT / "FACTUAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
