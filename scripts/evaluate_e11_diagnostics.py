"""Evaluate the reviewed E11 diagnostic pairs on saved checkpoints only."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers import AutoTokenizer  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e2.modeling import build_e2_student  # noqa: E402
from goibert_e11.modeling import build_e11_student, iter_e11_attentions  # noqa: E402


TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
VARIANTS = ("signed", "convex")


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _model_predictions(logits: torch.Tensor, counts: tuple[int, ...]) -> list[list[int] | int]:
    if not counts:
        return logits.argmax(dim=-1).detach().cpu().tolist()
    if logits.ndim == 3:
        return logits.argmax(dim=-1).detach().cpu().tolist()
    output = []
    for row in logits.detach().cpu():
        slots = []
        offset = 0
        for width in counts:
            slots.append(int(row[offset : offset + width].argmax()))
            offset += width
        output.append(slots)
    return output


def _margin(logits: torch.Tensor, expected: int) -> float:
    values = logits.detach().float().cpu()
    target = float(values[expected])
    other = float(torch.cat((values[:expected], values[expected + 1 :])).max()) if values.numel() > 1 else target
    return target - other


def _load_models(task_id: str, variant: str, device: torch.device):
    contract = e2_train._task_contract(task_id, ROOT, smoke=False)
    kwargs = {
        "num_classes": contract["num_classes"],
        "num_aspects": contract["num_aspects"],
        "aspect_class_counts": contract["aspect_class_counts"] if contract["num_aspects"] else None,
        "attention_implementation": "sdpa",
    }
    baseline, _ = build_e2_student(ROOT / "models" / "goiturn-s-gd", **kwargs)
    baseline_state = torch.load(ROOT / "artifacts" / "e8" / task_id / "seed_42" / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    baseline.load_state_dict(baseline_state["model"], strict=True)
    candidate, _ = build_e11_student(ROOT / "models" / "goiturn-s-gd", variant=variant, warmup_steps=1, **kwargs)
    candidate_state = torch.load(ROOT / "artifacts" / "e11" / variant / task_id / "seed_42" / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    candidate.load_state_dict(candidate_state["model"], strict=True)
    baseline.to(device).eval()
    candidate.to(device).eval()
    return contract, baseline, candidate


def main() -> int:
    diagnostic = _load(ROOT / "artifacts" / "e11" / "diagnostic_pairs.json")
    if diagnostic.get("train_injection") or diagnostic.get("test_loaded"):
        raise RuntimeError("Diagnostic artifact violates train/test isolation")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)
    output: dict[str, Any] = {"protocol": "E11-diagnostic-inference-v1", "source_diagnostic": str(ROOT / "artifacts" / "e11" / "diagnostic_pairs.json"), "source_split": diagnostic.get("source_split"), "test_loaded": False, "device": str(device), "status": "completed", "tasks": {}}
    pairs = diagnostic["pairs"]
    for task_id in TASKS:
        task_pairs = [pair for pair in pairs if pair["task_id"] == task_id]
        contract, baseline, _ = _load_models(task_id, "signed", device)
        counts = tuple(contract["aspect_class_counts"])
        texts = [text for pair in task_pairs for text in (pair["base"], pair["variant"])]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items() if isinstance(value, torch.Tensor)}
        with torch.inference_mode():
            base_logits = baseline(encoded["input_ids"], encoded["attention_mask"])
        base_predictions = _model_predictions(base_logits, counts)
        task_result: dict[str, Any] = {"pair_count": len(task_pairs), "variants": {}}
        del baseline
        for variant in VARIANTS:
            _, _, candidate = _load_models(task_id, variant, device)
            with torch.inference_mode():
                candidate_logits = candidate(encoded["input_ids"], encoded["attention_mask"])
            candidate_predictions = _model_predictions(candidate_logits, counts)
            module = next(iter(iter_e11_attentions(candidate)))
            correction_norm = float(module.last_correction_norm.item())
            observations = []
            base_correct = 0
            changed = 0
            candidate_correct = 0
            margin_deltas = []
            row_index = 0
            for pair in task_pairs:
                pair_observations = []
                for side in ("base", "variant"):
                    expectations = pair[f"{side}_expectations"]
                    if not counts:
                        expected = expectations[0]["label_id"]
                        bpred = int(base_predictions[row_index])
                        cpred = int(candidate_predictions[row_index])
                        bmargin = _margin(base_logits[row_index], expected)
                        cmargin = _margin(candidate_logits[row_index], expected)
                        b_ok = bpred == expected
                        c_ok = cpred == expected
                        base_correct += int(b_ok)
                        candidate_correct += int(c_ok)
                        changed += int(bpred != cpred)
                        margin_deltas.append(cmargin - bmargin)
                        pair_observations.append({"side": side, "expected": expected, "base_prediction": bpred, "candidate_prediction": cpred, "base_correct": b_ok, "candidate_correct": c_ok, "margin_delta": cmargin - bmargin})
                    else:
                        slot_rows = []
                        base_row = base_predictions[row_index]
                        candidate_row = candidate_predictions[row_index]
                        for expectation in expectations:
                            slot = contract["bundle"].aspect_names.index(expectation["slot"])
                            expected = expectation["label_id"]
                            bslot = base_logits[row_index, slot] if base_logits.ndim == 3 else base_logits[row_index, sum(counts[:slot]) : sum(counts[: slot + 1])]
                            cslot = candidate_logits[row_index, slot] if candidate_logits.ndim == 3 else candidate_logits[row_index, sum(counts[:slot]) : sum(counts[: slot + 1])]
                            bpred = int(base_row[slot])
                            cpred = int(candidate_row[slot])
                            bmargin = _margin(bslot, expected)
                            cmargin = _margin(cslot, expected)
                            b_ok = bpred == expected
                            c_ok = cpred == expected
                            base_correct += int(b_ok)
                            candidate_correct += int(c_ok)
                            changed += int(bpred != cpred)
                            margin_deltas.append(cmargin - bmargin)
                            slot_rows.append({"slot": expectation["slot"], "expected": expected, "base_prediction": bpred, "candidate_prediction": cpred, "base_correct": b_ok, "candidate_correct": c_ok, "margin_delta": cmargin - bmargin})
                        pair_observations.append({"side": side, "slots": slot_rows})
                    row_index += 1
                observations.append({"pair_id": pair["pair_id"], "kind": pair["kind"], "observations": pair_observations})
            task_result["variants"][variant] = {"base_correct_count": base_correct, "candidate_correct_count": candidate_correct, "changed_count": changed, "flip_rate_conditional_base_correct": changed / base_correct if base_correct else None, "mean_margin_delta": sum(margin_deltas) / len(margin_deltas) if margin_deltas else None, "relative_correction_norm_batch": correction_norm, "gate_values": next(iter(iter_e11_attentions(candidate))).gate_values().detach().cpu().tolist(), "observations": observations}
            del candidate
        output["tasks"][task_id] = task_result
    destination = ROOT / "artifacts" / "e11" / "diagnostic_results.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(destination), "status": output["status"], "device": str(device)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
