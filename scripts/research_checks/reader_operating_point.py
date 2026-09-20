#!/usr/bin/env python3
"""Validation-only C4 reader operating-point extraction.

This file copies the clause rule and masked-CLS forward needed by the prior
review without importing scripts that execute model batteries at import time.
It never trains, touches test data, or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import types
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import data as data_mod  # noqa: E402
from goibert import train as tr  # noqa: E402
from goibert_e14.modeling import build_bamibert_classifier  # noqa: E402
from goibert_e2.modeling import build_e2_student  # noqa: E402


OUT = ROOT / "artifacts" / "research_diagnostics" / "reader_operating_point"
TASKS = ("absa_restaurant", "absa_hotel", "visfd")
MODELS = ("student_e8", "bami_e14")
MAX_LEN = 256
REFERENCE = ROOT / "artifacts" / "independent_review" / "battery" / "clause_readers.json"
MARKERS = [
    " nhưng ", " còn ", " tuy nhiên ", " mà ", " và ", " nhưg ", " nhg ",
    " bù lại ", " được cái ", " chỉ có ", " chỉ mỗi ", " mỗi tội ", " ngoại trừ ",
    " trừ ", " tuy ", " song ",
]
SPLIT_RE = re.compile(r"([.,;:!?\n]+|" + "|".join(re.escape(m) for m in MARKERS) + r")", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def clause_spans(text: str, min_chars: int = 8) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    pos = 0
    for match in SPLIT_RE.finditer(text):
        segment = (pos, match.start())
        pos = match.end()
        if segment[1] > segment[0]:
            pieces.append(segment)
    if pos < len(text):
        pieces.append((pos, len(text)))
    merged: list[tuple[int, int]] = []
    for start, end in pieces:
        if not text[start:end].strip():
            continue
        if merged and end - start < min_chars:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged or [(0, len(text))]


class MaskedCLS:
    """Copied from the prior fork diagnostic; masks only the CLS key row."""

    def __init__(self, model: Any):
        self.model = model
        self.on = False
        self.mask2d: torch.Tensor | None = None
        self.keep: torch.Tensor | None = None
        self.originals: dict[int, Any] = {}
        for index, layer in enumerate(model.encoder.encoder.layer):
            attention = layer.attention.self
            self.originals[index] = attention.forward
            attention.forward = types.MethodType(self._make_forward(), attention)

    def restore(self) -> None:
        for index, original in self.originals.items():
            self.model.encoder.encoder.layer[index].attention.self.forward = original

    def _make_forward(self):
        state = self

        def forward(sa, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            batch, length, _ = hidden_states.shape
            heads, dim = sa.num_attention_heads, sa.attention_head_size
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                x = hidden_states.float()
                query = sa.query(x).view(batch, length, heads, dim).transpose(1, 2)
                key = sa.key(x).view(batch, length, heads, dim).transpose(1, 2)
                value = sa.value(x).view(batch, length, heads, dim).transpose(1, 2)
                valid = state.mask2d.to(x.device).bool()
                scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim)
                scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
                if state.on:
                    keep = state.keep.to(x.device).bool()
                    cls_scores = scores[:, :, 0, :].masked_fill(~keep[:, None, :], float("-inf"))
                    scores = scores.clone()
                    scores[:, :, 0, :] = cls_scores
                probs = torch.softmax(scores, dim=-1)
                context = torch.matmul(probs, value).transpose(1, 2).reshape(batch, length, heads * dim)
            return (context.to(hidden_states.dtype),)

        return forward


def checkpoint_path(kind: str, task: str) -> Path:
    if kind == "student_e8":
        return ROOT / "artifacts" / "e8" / task / "seed_42" / "best_checkpoint.pt"
    return ROOT / "artifacts" / "e14" / "trials" / task / "lr_5e-05" / "seed_42" / "best_checkpoint.pt"


def load_checkpoint(kind: str, task: str, device: torch.device):
    bundle = tr.load_task_bundle(task, ROOT, include_test=False)
    aspects = tr.task_num_aspects(bundle)
    counts = tr.task_aspect_class_counts(bundle) if aspects else None
    if kind == "student_e8":
        model_dir = ROOT / "models" / "goiturn-s-gd"
        model, _ = build_e2_student(model_dir, num_classes=tr.task_num_classes(bundle), num_aspects=aspects,
                                    aspect_class_counts=counts, attention_implementation="sdpa")
        checkpoint = checkpoint_path(kind, task)
    else:
        model_dir = ROOT / "models" / "bamibert-base-original"
        model, _ = build_bamibert_classifier(model_dir, num_classes=tr.task_num_classes(bundle), num_aspects=aspects,
                                             aspect_class_counts=counts, attention_implementation="sdpa")
        checkpoint = checkpoint_path(kind, task)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    return model, tokenizer, bundle, counts, checkpoint, model_dir


def slot_slices(counts: Any, total_width: int) -> list[tuple[int, int]]:
    if not counts:
        return [(0, total_width)]
    result: list[tuple[int, int]] = []
    offset = 0
    for width in counts:
        result.append((offset, offset + int(width)))
        offset += int(width)
    if offset != total_width:
        raise RuntimeError(f"slot width mismatch: counts={counts}, logits={total_width}")
    return result


@torch.inference_mode()
def extract_readers(model: Any, tokenizer: Any, texts: list[str], device: torch.device,
                    masking: MaskedCLS, batch_size: int = 16):
    all_logits: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    row_offsets = [0]
    reader_meta: list[list[dict[str, Any]]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        encoded = tokenizer(chunk, padding=True, truncation=True, max_length=MAX_LEN,
                             return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")
        attention = encoded["attention_mask"]
        rows: list[int] = []
        keeps: list[torch.Tensor] = []
        metas: list[list[dict[str, Any]]] = [[] for _ in chunk]
        for row, text in enumerate(chunk):
            valid_length = int(attention[row].sum())
            full_keep = torch.zeros(encoded["input_ids"].shape[1], dtype=torch.bool)
            full_keep[:valid_length] = True
            rows.append(row)
            keeps.append(full_keep)
            metas[row].append({"reader_index": 0, "kind": "full", "span": [0, len(text)],
                               "keep_token_mask": full_keep.tolist(), "valid_token_count": valid_length})
            spans = clause_spans(text)
            if len(spans) >= 2:
                for span_start, span_end in spans:
                    keep = torch.zeros_like(full_keep)
                    keep[0] = True
                    keep[valid_length - 1] = True
                    for token in range(1, valid_length - 1):
                        offset_start, offset_end = offsets[row, token].tolist()
                        if offset_end > span_start and offset_start < span_end:
                            keep[token] = True
                    if int(keep.sum()) >= 4:
                        reader_index = len(metas[row])
                        rows.append(row)
                        keeps.append(keep)
                        metas[row].append({"reader_index": reader_index, "kind": "clause", "span": [span_start, span_end],
                                           "keep_token_mask": keep.tolist(), "valid_token_count": valid_length})
        input_ids = encoded["input_ids"][rows]
        masks = attention[rows]
        keep_matrix = torch.stack(keeps)
        logits_parts: list[torch.Tensor] = []
        for offset in range(0, len(rows), 32):
            model_mask = masks[offset:offset + 32]
            model_input = input_ids[offset:offset + 32]
            masking.mask2d = model_mask
            masking.keep = keep_matrix[offset:offset + 32]
            masking.on = True
            with tr.make_autocast("bf16", device):
                logits_parts.append(model(model_input.to(device), model_mask.to(device)).float().cpu())
            masking.on = False
        logits = torch.cat(logits_parts, 0)
        if logits.ndim == 3:
            # Uniform ABSA slot schemas are returned as [reader, aspect, class].
            # Cache a flat slot-major vector so variable-width ViSFD uses the
            # same representation.
            logits = logits.reshape(logits.shape[0], -1)
        probs = torch.empty_like(logits)
        # Each aspect slot has its own softmax width; do not softmax across slots.
        widths = getattr(model, "aspect_class_counts", None)
        slices = slot_slices(widths, int(logits.shape[1]))
        for left, right in slices:
            probs[:, left:right] = torch.softmax(logits[:, left:right], dim=-1)
        cursor = 0
        for row, meta in enumerate(metas):
            count = len(meta)
            all_logits.append(logits[cursor:cursor + count])
            all_probs.append(probs[cursor:cursor + count])
            cursor += count
            row_offsets.append(row_offsets[-1] + count)
        reader_meta.extend(metas)
    return torch.cat(all_logits, 0), torch.cat(all_probs, 0), row_offsets, reader_meta


def polarity_argmax_from_flat(values: torch.Tensor, slices: list[tuple[int, int]]) -> list[int]:
    result: list[int] = []
    for left, right in slices:
        slot = values[left:right]
        if slot.numel() == 2:
            result.append(int(torch.argmax(slot)))
            continue
        positive = slot[1:4]
        result.append(int(torch.argmax(positive)) + 1)
    return result


def prediction_from_flat(values: torch.Tensor, slices: list[tuple[int, int]], *, delta: float = 0.0) -> list[int]:
    result: list[int] = []
    for left, right in slices:
        slot = values[left:right]
        if slot.numel() == 2:
            result.append(int(torch.argmax(slot)))
            continue
        positive = slot[1:4]
        choice = int(torch.argmax(positive)) + 1
        result.append(choice if float(positive.max() - slot[0]) > delta else 0)
    return result


def select_c4(flat_probs: torch.Tensor, flat_logits: torch.Tensor, slices: list[tuple[int, int]], reader_count: int):
    predictions: list[int] = []
    margins: list[float] = []
    selected: list[int] = []
    ties: list[int] = []
    cstars: list[int] = []
    for slot_index, (left, right) in enumerate(slices):
        masses = [float((1.0 - flat_probs[row, left]).item()) for row in range(reader_count)]
        best_mass = max(masses)
        tied = [row for row, mass in enumerate(masses) if mass == best_mass]
        reader = tied[0]
        logits = flat_logits[reader, left:right]
        positive = logits[1:4]
        if right - left > 2:
            cstar = int(torch.argmax(positive)) + 1
            margin = float(positive.max() - logits[0])
        else:
            cstar = int(torch.argmax(logits))
            margin = float(logits[1] - logits[0])
        if right - left == 2:
            prediction = int(torch.argmax(logits))
        else:
            prediction = cstar if margin > 0.0 else 0
        predictions.append(prediction)
        cstars.append(cstar)
        margins.append(margin)
        selected.append(reader)
        ties.append(len(tied))
    return predictions, cstars, margins, selected, ties


def curve(slot_records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_count = sum(int(item["gold"] > 0) for item in slot_records)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for item in slot_records:
        grouped[float(item["margin"])].append(item)
    points: list[dict[str, Any]] = [{"delta": None, "accepted": 0, "tp": 0, "precision": 0.0,
                                     "recall": 0.0, "f1": 0.0, "grouped_equal_margin": True}]
    accepted = 0
    true_positive = 0
    for margin in sorted(grouped, reverse=True):
        group = grouped[margin]
        accepted += len(group)
        true_positive += sum(int(item["gold"] > 0 and item["cstar"] == item["gold"]) for item in group)
        precision = true_positive / max(accepted, 1)
        recall = true_positive / max(gold_count, 1)
        points.append({"delta": math.nextafter(margin, -math.inf), "margin_group": margin,
                       "accepted": accepted, "tp": true_positive, "precision": precision,
                       "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-30),
                       "grouped_equal_margin": True})
    return {"gold_count": gold_count, "points": points}


def delta0(slot_records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_count = sum(int(item["gold"] > 0) for item in slot_records)
    accepted = [item for item in slot_records if float(item["margin"]) > 0.0]
    tp = sum(int(item["gold"] > 0 and item["cstar"] == item["gold"]) for item in accepted)
    precision = tp / max(len(accepted), 1)
    recall = tp / max(gold_count, 1)
    return {"gold_count": gold_count, "accepted": len(accepted), "tp": tp,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-30)}


def best_precision_at_recall(points: list[dict[str, Any]], recall: float) -> dict[str, Any]:
    eligible = [point for point in points if point["recall"] + 1e-15 >= recall]
    if not eligible:
        return {"precision": None, "recall": None, "f1": None, "delta": None}
    best = max(eligible, key=lambda point: (point["precision"], point["recall"], -(point["accepted"])))
    return {key: best.get(key) for key in ("precision", "recall", "f1", "delta", "accepted", "tp")}


def max_f1(points: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(points, key=lambda point: (point["f1"], point["recall"], -(point["accepted"])))
    return {key: best.get(key) for key in ("precision", "recall", "f1", "delta", "accepted", "tp")}


def validate_curve_fixture() -> None:
    """Ensure a negative global margin remains available to the PR curve."""
    result = curve([{"gold": 1, "cstar": 1, "margin": -0.25}])
    if max(point["f1"] for point in result["points"]) < 0.999:
        raise AssertionError("negative-margin PR fixture failed: curve dropped a valid cstar")


def error_summary(rows: list[dict[str, Any]], scored: list[int], key: str) -> dict[str, Any]:
    missed = wrong = spurious = 0
    single = multi = mixed = nonmixed = 0
    bins: dict[str, dict[str, int]] = defaultdict(lambda: {"rows": 0, "gold_aspects": 0, "missed": 0, "wrong_polarity": 0, "spurious": 0})
    for row in rows:
        gold = row["gold"]
        pred = row[key]
        positive_gold = [aspect for aspect in scored if gold[aspect] > 0]
        single += len(positive_gold) == 1
        multi += len(positive_gold) >= 2
        if len(positive_gold) >= 2:
            (mixed if len({gold[a] for a in positive_gold}) >= 2 else nonmixed).__class__
            if len({gold[a] for a in positive_gold}) >= 2:
                mixed += 1
            else:
                nonmixed += 1
        reader_count = int(row["reader_count"])
        group = "1" if reader_count == 1 else "2" if reader_count == 2 else "3-4" if reader_count <= 4 else ">=5"
        entry = bins[group]
        entry["rows"] += 1
        for aspect in scored:
            g, p = gold[aspect], pred[aspect]
            entry["gold_aspects"] += int(g > 0)
            missed += int(g > 0 and p == 0)
            wrong += int(g > 0 and p > 0 and p != g)
            spurious += int(g == 0 and p > 0)
            entry["missed"] += int(g > 0 and p == 0)
            entry["wrong_polarity"] += int(g > 0 and p > 0 and p != g)
            entry["spurious"] += int(g == 0 and p > 0)
    return {"missed": missed, "wrong_polarity": wrong, "spurious": spurious,
            "single_gold_rows": single, "multi_gold_rows": multi,
            "mixed_polarity_multi_rows": mixed, "nonmixed_polarity_multi_rows": nonmixed,
            "reader_count_bins": dict(bins)}


def build_report(kind: str, task: str, model: Any, tokenizer: Any, bundle: Any, counts: Any,
                 checkpoint: Path, model_dir: Path, device: torch.device) -> dict[str, Any]:
    validation = tr.split_field(bundle, "validation")
    texts = list(tr.split_texts(validation))
    ids = [str(value) for value in tr.split_ids(validation)]
    gold = [list(value) if isinstance(value, (list, tuple)) else int(value) for value in tr.model_labels(bundle, "validation", validation)]
    expected_ids = [str(value) for value in tr.split_ids(validation)]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise RuntimeError(f"validation ID contract failed for {kind}/{task}")
    if task == "visfd":
        scored = [index for index in range(tr.task_num_aspects(bundle)) if index != data_mod.VISFD_OTHERS_ASPECT_INDEX]
    else:
        scored = list(range(tr.task_num_aspects(bundle)))
    row_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    masking = MaskedCLS(model)
    logits, probs, offsets, metadata = extract_readers(model, tokenizer, texts, device, masking)
    masking.restore()
    slices = slot_slices(counts, int(logits.shape[1]))
    row_records: list[dict[str, Any]] = []
    slot_records_full: list[dict[str, Any]] = []
    slot_records_c4: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    concentration: dict[str, dict[str, int]] = defaultdict(lambda: {"selected_slots": 0, "selected_gold_positive_slots": 0, "correct_slots": 0})
    tie_counts = 0
    for row_index, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
        row_logits, row_probs = logits[left:right], probs[left:right]
        full_prediction = prediction_from_flat(row_logits[0], slices)
        full_cstar = polarity_argmax_from_flat(row_logits[0], slices)
        full_margin = []
        for slot_left, slot_right in slices:
            slot = row_logits[0, slot_left:slot_right]
            full_margin.append(float(slot[1:].max() - slot[0]) if slot_right - slot_left > 2 else float(slot[1] - slot[0]))
        c4_prediction, c4_cstar, c4_margin, selected_reader, ties = select_c4(row_probs, row_logits, slices, right - left)
        tie_counts += sum(value > 1 for value in ties)
        for slot_index, reader_index in enumerate(selected_reader):
            key = str(reader_index)
            concentration[key]["selected_slots"] += 1
            concentration[key]["selected_gold_positive_slots"] += int(gold[row_index][slot_index] > 0)
            concentration[key]["correct_slots"] += int(gold[row_index][slot_index] > 0 and c4_prediction[slot_index] == gold[row_index][slot_index])
        row_records.append({"row_index": row_index, "id": ids[row_index], "text_sha256": row_hashes[row_index],
                            "gold": gold[row_index], "full_prediction": full_prediction, "c4_prediction": c4_prediction,
                            "reader_count": right - left, "selected_reader": selected_reader, "ties": ties,
                            "full_margin": full_margin, "c4_margin": c4_margin})
        for aspect in scored:
            g = int(gold[row_index][aspect])
            full_record = {"gold": g, "cstar": full_cstar[aspect], "margin": full_margin[aspect]}
            c4_record = {"gold": g, "cstar": c4_cstar[aspect], "margin": c4_margin[aspect]}
            slot_records_full.append(full_record)
            slot_records_c4.append(c4_record)
            fp, cp = full_prediction[aspect], c4_prediction[aspect]
            if fp == 0 and g > 0 and cp == g:
                label = "missed_corrected"
            elif fp == 0 and g == 0 and cp > 0:
                label = "newly_spurious"
            elif fp == g and g > 0 and cp != g:
                label = "correct_polarity_broken"
            elif fp > 0 and fp != g and cp == g and g > 0:
                label = "wrong_polarity_fixed"
            elif fp == cp:
                label = "same_pred"
            else:
                label = "other_change"
            transitions.append({"row_index": row_index, "id": ids[row_index], "text_sha256": row_hashes[row_index],
                                "aspect": aspect, "gold": g, "full_prediction": fp, "c4_prediction": cp,
                                "full_margin": full_margin[aspect], "c4_margin": c4_margin[aspect],
                                "selected_reader": selected_reader[aspect], "reader_count": right - left, "transition": label})
    full_curve = curve(slot_records_full)
    c4_curve = curve(slot_records_c4)
    full_delta0 = delta0(slot_records_full)
    c4_delta0 = delta0(slot_records_c4)
    reference_key = f"{kind}__{task}"
    reference = read_json(REFERENCE).get(reference_key, {}) if REFERENCE.is_file() else {}
    reference_full = reference.get("baseline_full", {}).get("pair_f1")
    reference_c4 = reference.get("clauses_maxpresent", {}).get("pair_f1")
    ref_check = {
        "reference_path": str(REFERENCE),
        "reference_baseline_pair_f1_percent": reference_full,
        "reference_c4_pair_f1_percent": reference_c4,
        "actual_baseline_pair_f1_percent": 100 * full_delta0["f1"],
        "actual_c4_pair_f1_percent": 100 * c4_delta0["f1"],
        "baseline_abs_delta_pp": None if reference_full is None else abs(100 * full_delta0["f1"] - reference_full),
        "c4_abs_delta_pp": None if reference_c4 is None else abs(100 * c4_delta0["f1"] - reference_c4),
        "within_0_1pp": (reference_full is None or abs(100 * full_delta0["f1"] - reference_full) <= 0.1)
                         and (reference_c4 is None or abs(100 * c4_delta0["f1"] - reference_c4) <= 0.1),
    }
    report = {
        "protocol": "C4-reader-operating-point-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": kind,
        "task": task,
        "validation_only": True,
        "include_test": False,
        "device": str(device),
        "rows": len(texts),
        "unique_ids": len(set(ids)),
        "known_count_pass": len(texts) == {"absa_restaurant": 771, "absa_hotel": 795, "visfd": 1112}[task],
        "ids_exact_frozen_validation": ids == expected_ids,
        "gold_source": "frozen goibert validation split",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "model_dir": str(model_dir.relative_to(ROOT)),
        "reference_source_hashes": {"clause_readers.py": sha256(ROOT / "scripts" / "independent_review" / "clause_readers.py"),
                                    "fork_test.py": sha256(ROOT / "scripts" / "independent_review" / "fork_test.py"),
                                    "new_reader_script": sha256(Path(__file__))},
        "precision": "outer BF16 autocast; manual attention FP32; model eval/inference_mode",
        "scored_aspects": scored,
        "reader_policy": "reader0 full; C4 reader argmax present probability mass with first-reader tie; positive polarity argmax and global margin delta",
        "reader_count": {"mean": sum(item["reader_count"] for item in row_records) / len(row_records),
                         "rows_with_clauses": sum(item["reader_count"] > 1 for item in row_records),
                         "ties_across_slots": tie_counts},
        "delta0": {"baseline_full": full_delta0, "c4_maxpresent": c4_delta0},
        "pr_curves": {"baseline_full": full_curve, "c4_maxpresent": c4_curve},
        "pr_operating_comparisons": {
            "baseline_best_precision_at_or_above_c4_delta0_recall": best_precision_at_recall(full_curve["points"], c4_delta0["recall"]),
            "c4_best_precision_at_or_above_baseline_delta0_recall": best_precision_at_recall(c4_curve["points"], full_delta0["recall"]),
            "baseline_oracle_validation_max_f1": max_f1(full_curve["points"]),
            "c4_oracle_validation_max_f1": max_f1(c4_curve["points"]),
        },
        "native_delta0_errors": {"baseline_full": error_summary(row_records, scored, "full_prediction"),
                                  "c4_maxpresent": error_summary(row_records, scored, "c4_prediction")},
        "reader_selection_concentration": dict(concentration),
        "reference_delta0_check": ref_check,
        "runtime": {"elapsed_s": time.time() - start_time,
                     "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                     "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None},
        "cache_files": {"index": str(OUT / "cache" / f"{kind}__{task}__index.json"),
                        "tensor": str(OUT / "cache" / f"{kind}__{task}__readers.pt"),
                        "transitions_csv": str(OUT / "transitions" / f"{kind}__{task}.csv")},
    }
    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": logits, "probabilities": probs, "row_offsets": offsets,
                "slices": slices, "provenance": {"checkpoint_sha256": report["checkpoint_sha256"], "script_sha256": report["reference_source_hashes"]["new_reader_script"]}},
               cache_dir / f"{kind}__{task}__readers.pt")
    write_json(cache_dir / f"{kind}__{task}__index.json", {"provenance": report["reference_source_hashes"], "rows": [
        {"id": ids[i], "gold": gold[i], "text_sha256": row_hashes[i], "fullreader_index": 0, "readers": metadata[i]}
        for i in range(len(ids))
    ]})
    transition_path = OUT / "transitions" / f"{kind}__{task}.csv"
    transition_path.parent.mkdir(parents=True, exist_ok=True)
    with transition_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transitions[0]))
        writer.writeheader(); writer.writerows(transitions)
    return report


def write_summary(reports: list[dict[str, Any]]) -> None:
    write_json(OUT / "summary.json", {"protocol": "C4-reader-operating-point-diagnostic-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                                       "validation_only": True, "completed": [f"{r['model']}__{r['task']}" for r in reports],
                                       "reports": reports})
    lines = ["# C4 reader operating-point extraction", "", "Validation-only factual export; no threshold is selected for benchmark use.", "", "| Model | Task | n | Baseline F1@δ0 | C4 F1@δ0 | Baseline recall | C4 recall | Reference check |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for report in reports:
        base, c4 = report["delta0"]["baseline_full"], report["delta0"]["c4_maxpresent"]
        lines.append(f"| {report['model']} | {report['task']} | {report['rows']} | {100*base['f1']:.3f}% | {100*c4['f1']:.3f}% | {100*base['recall']:.3f}% | {100*c4['recall']:.3f}% | {report['reference_delta0_check']['within_0_1pp']} |")
    (OUT / "FACTUAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reusable_report(path: Path, kind: str, task: str) -> bool:
    try:
        report = read_json(path)
        return (
            report.get("protocol") == "C4-reader-operating-point-diagnostic-v1"
            and report.get("model") == kind
            and report.get("task") == task
            and report.get("reference_source_hashes", {}).get("new_reader_script") == sha256(Path(__file__))
            and report.get("checkpoint_sha256") == sha256(checkpoint_path(kind, task))
            and bool(report.get("validation_only"))
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    validate_curve_fixture()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    reports: list[dict[str, Any]] = []
    for kind in args.models:
        for task in args.tasks:
            output = OUT / f"{kind}__{task}.json"
            if output.is_file() and not args.force and reusable_report(output, kind, task):
                reports.append(json.loads(output.read_text(encoding="utf-8")))
                print(f"[reuse] {kind} {task}", flush=True)
                continue
            model, tokenizer, bundle, counts, checkpoint, model_dir = load_checkpoint(kind, task, device)
            try:
                report = build_report(kind, task, model, tokenizer, bundle, counts, checkpoint, model_dir, device)
                write_json(output, report)
                reports.append(report)
                print(f"[done] {kind} {task} rows={report['rows']} baseline_f1={report['delta0']['baseline_full']['f1']:.6f} c4_f1={report['delta0']['c4_maxpresent']['f1']:.6f}", flush=True)
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            write_summary(reports)
    write_summary(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
