"""Summarise the aspect-pair pilot: reference T vs arms A/B, deltas, curves, alignment checks, per-aspect F1.

Writes artifacts/cafebert_aspect_pair/summary.json.  Recomputes every F1 from the saved regrouped prediction rows with
the unchanged evaluator instead of trusting stored numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ARMS, NUM_ASPECTS, TASK_ID, e1_train  # noqa: E402

from goibert.data import RESTAURANT_ASPECT_NAMES  # noqa: E402


def load_run(path: Path) -> dict[str, Any]:
    out = {"dir": str(path.relative_to(common.ROOT))}
    for name in ("status.json", "metadata.json", "epoch_curves.json", "best_validation_metrics.json", "validation_predictions.json"):
        file = path / name
        out[name.split(".")[0]] = common.read_json(file) if file.exists() else None
    return out


def per_aspect(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for slot, name in enumerate(RESTAURANT_ASPECT_NAMES):
        gold = [r["gold"][slot] for r in rows]; pred = [r["prediction"][slot] for r in rows]
        tp = sum(g > 0 and p > 0 and g == p for g, p in zip(gold, pred)); gc = sum(g > 0 for g in gold); pc = sum(p > 0 for p in pred)
        prec = tp / pc if pc else 0.0; rec = tp / gc if gc else 0.0
        result[name] = {"gold": gc, "predicted": pc, "tp": tp, "f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0}
    return result


def main() -> int:
    contract = common.task_contract()
    bundle = contract["bundle"]
    valid = common.reviews_of(contract["validation"])
    runs = {"T": load_run(common.REFERENCE_RUN), **{arm: load_run(common.run_dir(arm)) for arm in ARMS}}
    summary: dict[str, Any] = {"protocol": common.PROTOCOL, "generated_at": common.now(), "runs": {}}
    f1 = {}
    for name, run in runs.items():
        if run["validation_predictions"] is None:
            summary["runs"][name] = {"status": "missing", "dir": run["dir"]}
            continue
        rows = run["validation_predictions"]["rows"]
        aligned = [r["id"] for r in rows] == [r.id for r in valid] and [list(r["gold"]) for r in rows] == [list(r.gold) for r in valid]
        recomputed = common.rows_f1(bundle, rows)
        curves = run["epoch_curves"]["epochs"] if run["epoch_curves"] else []
        meta = run["metadata"] or {}
        f1[name] = recomputed
        summary["runs"][name] = {
            "dir": run["dir"], "status": (run["status"] or {}).get("status"), "best_epoch": run["validation_predictions"]["epoch"],
            "recomputed_pair_micro_f1": recomputed, "stored_best_f1": (run["status"] or {}).get("best_selection_f1"),
            "stored_equals_recomputed": abs(recomputed - float((run["status"] or {}).get("best_selection_f1", -1))) < 1e-9,
            "rows_aligned_with_loader": aligned, "metrics": {k: v for k, v in (run["best_validation_metrics"] or {}).items() if isinstance(v, (int, float))},
            "curve_f1": [round(100 * e["selection_f1"], 4) for e in curves], "mean_f1_epochs_10_30": (sum(e["selection_f1"] for e in curves[9:30]) / max(len(curves[9:30]), 1)) if len(curves) >= 10 else None,
            "elapsed_minutes": (meta.get("elapsed_seconds") or 0) / 60, "epoch_seconds_mean": (sum(e["epoch_seconds"] for e in curves) / max(len(curves), 1)) if curves else None,
            "peak_gpu_allocated_gib": (meta.get("peak_gpu_memory_allocated_bytes") or 0) / 2**30, "device": (run["status"] or {}).get("device"),
            "total_parameters": (meta.get("model") or {}).get("total_parameters"), "head_init_sha256": (meta.get("model") or {}).get("head_init_sha256"),
            "review_order_sha256_all_epochs": meta.get("review_order_sha256_all_epochs"), "postrun_checks": meta.get("postrun_checks"),
            "per_aspect": per_aspect(rows), "predicted_non_absent": sum(p > 0 for r in rows for p in r["prediction"]), "gold_non_absent": sum(g > 0 for r in rows for g in r["gold"]),
        }
    if all(k in f1 for k in ("T", "A", "B")):
        summary["deltas_pp"] = {"A_minus_T": 100 * (f1["A"] - f1["T"]), "B_minus_T": 100 * (f1["B"] - f1["T"]), "B_minus_A": 100 * (f1["B"] - f1["A"])}
        best = max(f1["A"], f1["B"]); gain = 100 * (best - f1["T"])
        summary["gate"] = {"max_arm_minus_T_pp": gain, "band": ">=0.5 confirm-worthy" if gain >= 0.5 else ("0.2-0.5 unclear" if gain >= 0.2 else "<0.2 or negative: no signal")}
        a_rows = runs["A"]["validation_predictions"]["rows"]; b_rows = runs["B"]["validation_predictions"]["rows"]; t_rows = runs["T"]["validation_predictions"]["rows"]
        slots = len(a_rows) * NUM_ASPECTS
        summary["agreement"] = {
            "slots": slots,
            "A_vs_B_slot_agreement": sum(a["prediction"][s] == b["prediction"][s] for a, b in zip(a_rows, b_rows) for s in range(NUM_ASPECTS)) / slots,
            "A_vs_T_slot_agreement": sum(a["prediction"][s] == t["prediction"][s] for a, t in zip(a_rows, t_rows) for s in range(NUM_ASPECTS)) / slots,
            "B_vs_T_slot_agreement": sum(b["prediction"][s] == t["prediction"][s] for b, t in zip(b_rows, t_rows) for s in range(NUM_ASPECTS)) / slots,
            "same_head_init_A_B": summary["runs"]["A"]["head_init_sha256"] == summary["runs"]["B"]["head_init_sha256"],
            "same_review_order_A_B": summary["runs"]["A"]["review_order_sha256_all_epochs"] == summary["runs"]["B"]["review_order_sha256_all_epochs"],
        }
    preflight = common.CHECK_ROOT / "preflight.json"
    summary["preflight_all_checks_ok"] = common.read_json(preflight).get("all_checks_ok") if preflight.exists() else None
    common.write_json(common.ARTIFACT_ROOT / "summary.json", summary)
    for name, run in summary["runs"].items():
        if "recomputed_pair_micro_f1" in run:
            print(f"{name}: F1 {100*run['recomputed_pair_micro_f1']:.4f} @ epoch {run['best_epoch']} ({run['status']}, {run['elapsed_minutes']:.1f} min)")
    print("deltas:", summary.get("deltas_pp")); print("gate:", summary.get("gate")); print("agreement:", summary.get("agreement"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
