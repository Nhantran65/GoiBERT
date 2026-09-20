#!/usr/bin/env python3
"""CPU-only numerical certificate for ABSA output-head decision geometry.

Only checkpoint head tensors are loaded.  No encoder construction, forward
pass, test split, training, or model mutation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "research_diagnostics" / "decoder_feasibility"
TASKS = ("absa_restaurant", "absa_hotel", "visfd")
MODELS = ("student_e8", "bami_e14")
TASK_ASPECTS = {"absa_restaurant": 12, "absa_hotel": 34, "visfd": 11}
VISFD_OTHERS_INDEX = 5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def checkpoint_path(model: str, task: str) -> Path:
    if model == "student_e8":
        return ROOT / "artifacts" / "e8" / task / "seed_42" / "best_checkpoint.pt"
    return ROOT / "artifacts" / "e14" / "trials" / task / "lr_5e-05" / "seed_42" / "best_checkpoint.pt"


def load_head(model: str, task: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path]:
    checkpoint = checkpoint_path(model, task)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = state["model"]
    weight_key = next((key for key in model_state if key.endswith("head.output.weight")), None)
    bias_key = next((key for key in model_state if key.endswith("head.output.bias")), None)
    if weight_key is None or bias_key is None:
        raise KeyError(f"head output tensors not found in {checkpoint}")
    weight = model_state[weight_key].detach().cpu().numpy().astype(np.float64, copy=True)
    bias = model_state[bias_key].detach().cpu().numpy().astype(np.float64, copy=True)
    metadata = {"weight_key": weight_key, "bias_key": bias_key, "weight_shape": list(weight.shape),
                "bias_shape": list(bias.shape), "hidden_dim": int(weight.shape[1]),
                "checkpoint_sha256": sha256(checkpoint), "checkpoint": str(checkpoint.relative_to(ROOT))}
    return weight, bias, metadata, checkpoint


def build_margins(weight: np.ndarray, bias: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    aspects = TASK_ASPECTS[task]
    if task == "visfd":
        widths = [4] * aspects
        widths[VISFD_OTHERS_INDEX] = 2
    else:
        widths = [4] * aspects
    if sum(widths) != weight.shape[0]:
        raise ValueError(f"head width mismatch {task}: expected {sum(widths)}, got {weight.shape[0]}")
    rows: list[np.ndarray] = []
    betas: list[float] = []
    labels: list[dict[str, Any]] = []
    offset = 0
    for aspect, width in enumerate(widths):
        if task == "visfd" and aspect == VISFD_OTHERS_INDEX:
            offset += width
            continue
        for cls in range(1, width):
            rows.append(weight[offset + cls] - weight[offset])
            betas.append(float(bias[offset + cls] - bias[offset]))
            labels.append({"aspect": aspect, "class": cls, "offset": offset, "width": width, "scored": True})
        offset += width
    return np.stack(rows, axis=0), np.asarray(betas, dtype=np.float64), labels


def rank_info(matrix: np.ndarray) -> dict[str, Any]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    standard_tol = max(matrix.shape) * np.finfo(np.float64).eps * max(largest, 1.0)
    return {
        "rows": int(matrix.shape[0]), "columns": int(matrix.shape[1]),
        "singular_values": singular.tolist(),
        "standard_tolerance": float(standard_tol),
        "rank_standard": int(np.sum(singular > standard_tol)),
        "rank_relative_1e-8": int(np.sum(singular > largest * 1e-8)) if largest else 0,
        "rank_relative_1e-10": int(np.sum(singular > largest * 1e-10)) if largest else 0,
    }


def solve_center(matrix: np.ndarray, beta: np.ndarray) -> dict[str, Any]:
    rows, columns = matrix.shape
    objective = np.zeros(columns + 1, dtype=np.float64)
    objective[-1] = 1.0
    inequalities = np.zeros((2 * columns, columns + 1), dtype=np.float64)
    for index in range(columns):
        inequalities[2 * index, index] = 1.0
        inequalities[2 * index, -1] = -1.0
        inequalities[2 * index + 1, index] = -1.0
        inequalities[2 * index + 1, -1] = -1.0
    equality = np.zeros((rows, columns + 1), dtype=np.float64)
    equality[:, :columns] = matrix
    result = linprog(objective, A_ub=inequalities, b_ub=np.zeros(2 * columns),
                     A_eq=equality, b_eq=-beta, bounds=[(None, None)] * columns + [(0.0, None)], method="highs")
    if not result.success:
        return {"status": "inconclusive", "lp_success": False, "lp_message": result.message,
                "lp_status": int(result.status), "interior_solution": False}
    vector = np.asarray(result.x[:columns], dtype=np.float64)
    t_value = float(result.x[-1])
    return {"status": "solved", "lp_success": True, "lp_message": result.message,
            "lp_status": int(result.status), "t": t_value, "v0_max_abs": float(np.max(np.abs(vector))),
            "v0": vector.tolist(), "interior_solution": bool(t_value < 1.0),
            "center_residual_inf": float(np.max(np.abs(matrix @ vector + beta)))}


def certificate(matrix: np.ndarray, beta: np.ndarray) -> dict[str, Any]:
    rank = rank_info(matrix)
    center = solve_center(matrix, beta)
    result: dict[str, Any] = {"rank": rank, "center_lp": center, "certificate_status": "inconclusive"}
    full_row_rank = rank["rank_standard"] == rank["rows"]
    if center.get("status") != "solved" or not full_row_rank or not center.get("interior_solution"):
        result["inconclusive_reason"] = {
            "full_row_rank_standard": full_row_rank,
            "center_lp_solved": center.get("status") == "solved",
            "center_interior": center.get("interior_solution", False),
        }
        return result
    right_inverse = np.linalg.pinv(matrix, rcond=1e-12)
    residual = matrix @ right_inverse - np.eye(matrix.shape[0], dtype=np.float64)
    right_inverse_inf = float(np.linalg.norm(right_inverse, ord=np.inf))
    residual_max_entry = float(np.max(np.abs(residual)))
    residual_inf = float(np.linalg.norm(residual, ord=np.inf))
    room = 1.0 - float(center["v0_max_abs"])
    epsilon = 0.5 * room / max(right_inverse_inf, np.finfo(np.float64).tiny)
    center_error = float(center["center_residual_inf"])
    error_bound = center_error + epsilon * residual_inf
    margin_lower_bound = epsilon - 2.0 * error_bound
    result.update({"right_inverse_shape": list(right_inverse.shape), "right_inverse_inf_norm": right_inverse_inf,
                   "right_inverse_residual_inf": residual_inf, "right_inverse_residual_max_entry": residual_max_entry,
                   "box_room": room, "epsilon": epsilon, "certificate_error_bound_inf": error_bound,
                   "certificate_margin_lower_bound": margin_lower_bound,
                   "epsilon_right_inverse_inf": epsilon * right_inverse_inf,
                   "residual_tolerance": 1e-8, "certificate_tolerance": 1e-8})
    residual_ok = residual_inf <= 1e-8 and center_error <= 1e-8
    margin_ok = epsilon > 0 and epsilon * right_inverse_inf < room
    rigorous_margin_ok = margin_lower_bound > 0.0
    if residual_ok and margin_ok and rigorous_margin_ok:
        examples = []
        row_count = matrix.shape[0]
        for mode in range(5):
            if mode == 0:
                u = -np.ones(row_count, dtype=np.float64)
            elif mode <= 3:
                u = np.zeros(row_count, dtype=np.float64)
                u[mode - 1 :: 3] = 1.0
            else:
                u = np.where(np.arange(row_count) % 4 == 0, 1.0, -1.0)
            vector = np.asarray(center["v0"], dtype=np.float64) + epsilon * (right_inverse @ u)
            examples.append({"mode": mode, "u_inf": float(np.max(np.abs(u))), "v_inf": float(np.max(np.abs(vector))),
                             "box_interior": bool(np.max(np.abs(vector)) < 1.0),
                             "margin_residual_inf": float(np.max(np.abs(matrix @ vector + beta - epsilon * u)))})
        result["example_checks"] = examples
        result["certificate_status"] = "numerical_certificate" if all(x["box_interior"] and x["margin_residual_inf"] <= 1e-7 for x in examples) else "inconclusive"
    else:
        result["inconclusive_reason"] = {"residual_ok": residual_ok, "margin_ok": margin_ok, "rigorous_margin_ok": rigorous_margin_ok}
    return result


def fixtures() -> dict[str, Any]:
    interior = certificate(np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64))
    deficient = certificate(np.asarray([[1.0, 0.0], [2.0, 0.0]]), np.zeros(2, dtype=np.float64))
    outside = certificate(np.eye(2, dtype=np.float64), np.asarray([2.0, 0.0], dtype=np.float64))
    return {"full_row_rank_center_interior": interior, "rank_deficient": deficient,
            "center_outside_box": outside,
            "fixture_pass": interior["certificate_status"] == "numerical_certificate"
            and deficient["certificate_status"] == "inconclusive"
            and outside["certificate_status"] == "inconclusive"}


def run_one(model: str, task: str) -> dict[str, Any]:
    weight, bias, metadata, checkpoint = load_head(model, task)
    matrix, beta, labels = build_margins(weight, bias, task)
    result = certificate(matrix, beta)
    result.update({"protocol": "decoder-feasibility-diagnostic-v1", "created_at": now(), "model": model, "task": task,
                   "validation_only": True, "encoder_forward": False, "checkpoint": metadata,
                   "head_output_weight_key": metadata["weight_key"], "head_output_bias_key": metadata["bias_key"],
                   "head_output_weight_shape": metadata["weight_shape"], "head_output_bias_shape": metadata["bias_shape"],
                   "margin_rows": len(labels), "scored_aspects": TASK_ASPECTS[task] - (1 if task == "visfd" else 0),
                   "visfd_others_excluded": task == "visfd", "margin_labels": labels,
                   "numerical_dtype": "float64", "pseudoinverse_rcond": 1e-12})
    return result


def reusable(path: Path, model: str, task: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("protocol") == "decoder-feasibility-diagnostic-v1" and value.get("model") == model and value.get("task") == task and value.get("checkpoint", {}).get("checkpoint_sha256") == sha256(checkpoint_path(model, task))
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fixture_result = fixtures()
    if not fixture_result["fixture_pass"]:
        raise RuntimeError("CPU decoder fixtures failed")
    reports = []
    for model in args.models:
        for task in args.tasks:
            path = OUT / f"{model}__{task}.json"
            if path.is_file() and not args.force and reusable(path, model, task):
                report = json.loads(path.read_text(encoding="utf-8"))
                print(f"[reuse] {model} {task}", flush=True)
            else:
                report = run_one(model, task)
                write_json(path, report)
                print(f"[done] {model} {task} status={report['certificate_status']}", flush=True)
            reports.append(report)
    summary = {"protocol": "decoder-feasibility-diagnostic-v1", "created_at": now(), "validation_only": True,
               "encoder_forward": False, "training": False, "test_access": False, "fixtures": fixture_result,
               "reports": reports}
    write_json(OUT / "summary.json", summary)
    lines = ["# Decoder feasibility diagnostic", "", "CPU-only checkpoint-head analysis; no encoder forward, training, or test access.", "", "| Model | Task | Margin rows | Rank | Rows | LP t | Certificate status |", "|---|---|---:|---:|---:|---:|---|"]
    for report in reports:
        lines.append(f"| {report['model']} | {report['task']} | {report['margin_rows']} | {report['rank']['rank_standard']} | {report['rank']['rows']} | {report['center_lp'].get('t', 'NA')} | {report['certificate_status']} |")
    (OUT / "FACTUAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
