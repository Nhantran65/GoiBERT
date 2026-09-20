#!/usr/bin/env python3
"""Cross-fitted reader combiner diagnostic on frozen E8 validation caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from goibert import train as tr  # noqa: E402
from reader_operating_point import load_checkpoint, slot_slices, MaskedCLS  # noqa: E402
from shadow_reader import build_reader_batch  # noqa: E402


OUT = ROOT / "artifacts" / "research_diagnostics" / "reader_learnability"
READER_OUT = ROOT / "artifacts" / "research_diagnostics" / "reader_operating_point"
TASKS = ("absa_restaurant", "absa_hotel")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n", encoding="utf-8")
    tmp.replace(path)


def feature_fixture() -> bool:
    rng = np.arange(4 * 4, dtype=np.float64).reshape(4, 4)
    features = make_features(rng[0:1])
    return features.shape == (1, 21) and np.isfinite(features).all()


def make_features(reader_logits: np.ndarray) -> np.ndarray:
    if reader_logits.ndim != 2 or reader_logits.shape[1] != 4:
        raise ValueError(f"expected readers x 4 logits, got {reader_logits.shape}")
    full = reader_logits[0]
    mean = reader_logits.mean(axis=0)
    maximum = reader_logits.max(axis=0)
    std = reader_logits.std(axis=0)
    fractions = np.array([(reader_logits.argmax(axis=1) == cls).mean() for cls in range(4)], dtype=np.float64)
    return np.concatenate([full, mean, maximum, std, fractions, [math.log(reader_logits.shape[0])]])[None, :]


def deterministic_random_masks(task: str, row_id: str, metadata: list[dict[str, Any]]) -> list[list[bool]]:
    output = []
    for reader_index, item in enumerate(metadata):
        original = list(item["keep_token_mask"])
        if reader_index == 0:
            output.append(original); continue
        valid = int(item["valid_token_count"])
        wanted = [pos for pos in range(1, valid - 1) if original[pos]]
        pool = list(range(1, valid - 1))
        seed = int(hashlib.sha256(f"{task}|{row_id}|{reader_index}".encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        selected = set(rng.sample(pool, len(wanted)))
        replacement = [False] * len(original); replacement[0] = True; replacement[valid - 1] = True
        for pos in selected: replacement[pos] = True
        output.append(replacement)
    return output


@torch.inference_mode()
def run_random_logits(task: str, tokenizer: Any, model: Any, texts: list[str], ids: list[str], clause_metadata: list[list[dict[str, Any]]], device: torch.device, counts: Any):
    masking = MaskedCLS(model); pieces = []; offsets = [0]; random_index = []
    for start in range(0, len(texts), 16):
        chunk = texts[start:start + 16]; chunk_ids = ids[start:start + 16]; metadata = clause_metadata[start:start + 16]
        encoded, attention, rows, keeps, _ = build_reader_batch(tokenizer, chunk)
        random_masks = [deterministic_random_masks(task, row_id, metadata[row]) for row, row_id in enumerate(chunk_ids)]
        flat_masks = torch.tensor([mask for row in random_masks for mask in row], dtype=torch.bool)
        masking.mask2d = attention[rows]; masking.keep = flat_masks; masking.on = True
        with tr.make_autocast("bf16", device):
            logits = model(encoded["input_ids"][rows].to(device), attention[rows].to(device)).float().cpu()
        masking.on = False
        logits = logits.reshape(logits.shape[0], -1) if logits.ndim == 3 else logits
        pieces.append(logits); cursor = 0
        for row, metadata_row in enumerate(metadata):
            count = len(metadata_row); offsets.append(offsets[-1] + count); random_index.append([mask for mask in random_masks[row]])
            cursor += count
    masking.restore()
    return torch.cat(pieces, 0), offsets, random_index


def validate_provenance(task: str, bundle: Any, ids: list[str], gold: list[list[int]], metadata: list[list[dict[str, Any]]], checkpoint: Path) -> dict[str, Any]:
    cache = torch.load(READER_OUT / "cache" / f"student_e8__{task}__readers.pt", map_location="cpu", weights_only=False)
    index = json.loads((READER_OUT / "cache" / f"student_e8__{task}__index.json").read_text())
    report = json.loads((READER_OUT / f"student_e8__{task}.json").read_text())
    if report["checkpoint_sha256"] != sha256(checkpoint): raise RuntimeError("reader checkpoint hash mismatch")
    if [str(row["id"]) for row in index["rows"]] != ids or [row["gold"] for row in index["rows"]] != gold: raise RuntimeError("reader cache IDs/gold mismatch")
    if [row["readers"] for row in index["rows"]] != metadata: raise RuntimeError("reader cache masks mismatch")
    return {"cache": cache, "index": index, "report": report}


def metrics(pred: np.ndarray, gold: np.ndarray) -> dict[str, Any]:
    gold_positive = gold > 0; pred_positive = pred > 0
    pair_tp = int(np.sum(gold_positive & (pred == gold))); pair_pred = int(np.sum(pred_positive)); pair_gold = int(np.sum(gold_positive))
    det_tp = int(np.sum(gold_positive & pred_positive)); det_pred = int(np.sum(pred_positive)); det_gold = int(np.sum(gold_positive))
    def prf(tp: int, p: int, g: int):
        precision = tp / max(p, 1); recall = tp / max(g, 1); return {"tp": tp, "predicted_positive": p, "gold_positive": g, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-30)}
    return {"pair": prf(pair_tp, pair_pred, pair_gold), "detection": prf(det_tp, det_pred, det_gold),
            "missed": int(np.sum(gold_positive & ~pred_positive)), "spurious": int(np.sum(~gold_positive & pred_positive)),
            "wrong_polarity": int(np.sum(gold_positive & pred_positive & (pred != gold))),
            "gold_neutral_missed": {"count": int(np.sum((gold == 3) & (pred == 0))), "denominator": int(np.sum(gold == 3))},
            "gold_positive_missed": {"count": int(np.sum((gold == 1) & (pred == 0))), "denominator": int(np.sum(gold == 1))},
            "gold_negative_missed": {"count": int(np.sum((gold == 2) & (pred == 0))), "denominator": int(np.sum(gold == 2))}}


def fit_crossfit(features: np.ndarray, gold: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n, aspects = gold.shape; pred = np.zeros_like(gold); probs = np.zeros((n, aspects, 4), dtype=np.float64); fit_info = []
    for aspect in range(aspects):
        for fold in range(3):
            train = folds != fold; test = folds == fold; classes = sorted(set(int(x) for x in gold[train, aspect]))
            info = {"aspect": aspect, "fold": fold, "classes": classes, "train_support": {str(c): int(np.sum(gold[train, aspect] == c)) for c in classes}, "constant": len(classes) < 2, "converged": None, "warning": None}
            if len(classes) < 2:
                constant = classes[0] if classes else 0; pred[test, aspect] = constant; probs[test, aspect, constant] = 1.0
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42, multi_class="multinomial"))
                    estimator.fit(features[train, aspect], gold[train, aspect]); p = estimator.predict_proba(features[test, aspect]); classes_fit = estimator[-1].classes_.astype(int)
                    test_indices = np.where(test)[0]
                    for column, cls in enumerate(classes_fit):
                        probs[test_indices, aspect, cls] = p[:, column]
                    pred[test, aspect] = classes_fit[p.argmax(axis=1)]
                    info["converged"] = True; info["warning"] = [str(x.message) for x in caught]
            fit_info.append(info)
    return pred, probs, fit_info


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda:1"); args = parser.parse_args()
    if not feature_fixture(): raise RuntimeError("21-feature fixture failed")
    OUT.mkdir(parents=True, exist_ok=True); reports = []; start = time.time()
    for task in TASKS:
        device = torch.device(args.device); model, tokenizer, bundle, counts, checkpoint, _ = load_checkpoint("student_e8", task, device)
        validation = tr.split_field(bundle, "validation"); texts = list(tr.split_texts(validation)); ids = [str(x) for x in tr.split_ids(validation)]; gold = [list(x) for x in tr.model_labels(bundle, "validation", validation)]
        index = json.loads((READER_OUT / "cache" / f"student_e8__{task}__index.json").read_text()); metadata = [row["readers"] for row in index["rows"]]; provenance = validate_provenance(task, bundle, ids, gold, metadata, checkpoint); cache = provenance["cache"]; original_logits = cache["logits"].float(); offsets = [int(x) for x in cache["row_offsets"]]
        random_logits_flat, random_offsets, random_masks = run_random_logits(task, tokenizer, model, texts, ids, metadata, device, counts)
        if random_offsets != offsets: raise RuntimeError("random reader offsets mismatch")
        slices = slot_slices(counts, int(original_logits.shape[1])); n = len(texts); aspects = len(gold[0]); feature_sets = {"full_only": np.zeros((n, aspects, 21)), "clause": np.zeros((n, aspects, 21)), "random": np.zeros((n, aspects, 21))}
        def reader_matrix(logits, left, right, aspect): return logits[left:right, slices[aspect][0]:slices[aspect][1]].numpy()
        for row, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
            random_matrix = random_logits_flat[left:right]; clause_matrix = original_logits[left:right]
            for aspect in range(aspects):
                count = right - left
                feature_sets["full_only"][row, aspect] = make_features(reader_matrix(clause_matrix, 0, 1, aspect))[0]
                feature_sets["clause"][row, aspect] = make_features(reader_matrix(clause_matrix, 0, count, aspect))[0]
                feature_sets["random"][row, aspect] = make_features(reader_matrix(random_matrix, 0, count, aspect))[0]
        gold_np = np.asarray(gold, dtype=int); fold_ids = KFold(n_splits=3, shuffle=True, random_state=42).split(np.arange(n)); folds = np.zeros(n, dtype=int)
        for fold, (_, test) in enumerate(fold_ids): folds[test] = fold
        arms = {}; raw_rows = [{"id": ids[i], "gold": gold[i], "fold": int(folds[i])} for i in range(n)]
        for arm, features in feature_sets.items():
            pred, probs, fit_info = fit_crossfit(features, gold_np, folds); fold_metrics = [metrics(pred[folds == fold], gold_np[folds == fold]) for fold in range(3)]
            arms[arm] = {"features_shape": list(features.shape), "feature_config": "full logits4 + mean4 + max4 + std4 + argmax fractions4 + log(reader_count) = 21", "oof_prediction": pred.tolist(), "oof_probabilities": probs.tolist(), "fit_info": fit_info, "pooled_metrics": metrics(pred, gold_np), "fold_metrics": fold_metrics}
        report = {"protocol": "reader-learnability-diagnostic-v1", "created_at": datetime.now(timezone.utc).isoformat(), "model": "student_e8", "task": task, "validation_only": True, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint), "reader_cache_provenance": provenance["report"]["reference_source_hashes"], "rows": raw_rows, "random_masks": random_masks, "arms": arms, "runtime_seconds": time.time() - start}
        write_json(OUT / f"student_e8__{task}.json", report); reports.append(report); del model; torch.cuda.empty_cache()
    summary = {"protocol": "reader-learnability-diagnostic-v1", "created_at": datetime.now(timezone.utc).isoformat(), "validation_only": True, "training": "combiner cross-fit only; encoder frozen", "test_access": False, "feature_fixture_pass": True, "reports": [{"task": r["task"], "arms": {k: v["pooled_metrics"] for k, v in r["arms"].items()}} for r in reports]}
    write_json(OUT / "summary.json", summary)
    lines = ["# Reader learnability diagnostic", "", "Exploratory 3-fold validation cross-fit; frozen encoder and no test access.", "", "| Task | Arm | Pair F1 | Detection F1 | Missed | Spurious | Wrong polarity |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in reports:
        for arm, value in r["arms"].items():
            m=value["pooled_metrics"]; lines.append(f"| {r['task']} | {arm} | {m['pair']['f1']:.3f} | {m['detection']['f1']:.3f} | {m['missed']} | {m['spurious']} | {m['wrong_polarity']} |")
    (OUT / "FACTUAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
