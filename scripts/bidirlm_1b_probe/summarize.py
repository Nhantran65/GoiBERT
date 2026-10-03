#!/usr/bin/env python3
"""Summarize the five validation-only B1B runs and their postrun checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PHASE2 = C.ART.parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write_curves(rows):
    """Standalone SVG with validation F1 and train loss by epoch for every completed cell."""
    drawn = [r for r in rows if r["curve_f1"] and r["curve_loss"]]
    if not drawn:
        return None
    colors = ["#1967d2", "#9c27b0", "#e67e22", "#168a6d", "#c0392b"]
    panels = [("Validation F1 (%)", "curve_f1", 100), ("Train loss", "curve_loss", 1)]
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="660" viewBox="0 0 1120 660">',
           '<rect width="1120" height="660" fill="#ffffff"/>',
           '<text x="560" y="34" text-anchor="middle" font-family="Arial" font-size="21" font-weight="bold">BidirLM-1B validation probe — seed 42</text>']
    for pi, (title, field, scale) in enumerate(panels):
        x0 = 62 + pi*548
        y0, width, height = 82, 460, 440
        values = [v*scale for r in drawn for v in r[field]]
        lo, hi = min(values), max(values)
        margin = max((hi-lo)*.08, .5 if field == "curve_f1" else .02)
        lo, hi = max(0, lo-margin), hi+margin
        if hi <= lo:
            hi = lo+1
        out += [f'<text x="{x0+width/2}" y="66" text-anchor="middle" font-family="Arial" font-size="17">{escape(title)}</text>',
                f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#fcfcfd" stroke="#8a99a8"/>']
        for tick in range(6):
            y = y0+height*(1-tick/5)
            val = lo+(hi-lo)*tick/5
            out += [f'<line x1="{x0}" y1="{y:.2f}" x2="{x0+width}" y2="{y:.2f}" stroke="#e3e8ee"/>',
                    f'<text x="{x0-7}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{val:.2f}</text>']
        for ep in (1, 5, 10, 15, 20, 25, 30):
            x = x0+(ep-1)/29*width
            out.append(f'<text x="{x:.2f}" y="{y0+height+18}" text-anchor="middle" font-family="Arial" font-size="11">{ep}</text>')
        out.append(f'<text x="{x0+width/2}" y="{y0+height+39}" text-anchor="middle" font-family="Arial" font-size="12">Epoch</text>')
        for ri, row in enumerate(drawn):
            seq = row[field]
            points = " ".join(f"{x0+i/29*width:.2f},{y0+height-(v*scale-lo)/(hi-lo)*height:.2f}" for i, v in enumerate(seq))
            out.append(f'<polyline points="{points}" fill="none" stroke="{colors[ri%len(colors)]}" stroke-width="2.2"/>')
    for i, row in enumerate(drawn):
        x = 70+(i%3)*350
        y = 574+(i//3)*30
        label = f"{row['task']} · LR {row['lr']:g}"
        out += [f'<line x1="{x}" y1="{y-5}" x2="{x+28}" y2="{y-5}" stroke="{colors[i%len(colors)]}" stroke-width="3"/>',
                f'<text x="{x+36}" y="{y}" font-family="Arial" font-size="13">{escape(label)}</text>']
    out.append('</svg>')
    path = PHASE2 / "curves.svg"
    path.write_text("\n".join(out)+"\n")
    return str(path)


def summarize():
    rows = []
    for task in C.TASKS:
        for lr in C.LR_GRID:
            path = C.arm_dir("B1B", task, lr, C.SEED)
            meta = read(path / "metadata.json")
            if not meta:
                continue
            curves = (read(path / "epoch_curves.json") or {}).get("epochs", [])
            rows.append({"task": task, "lr": lr, "seed": C.SEED, "directory": str(path),
                         "status": meta["status"], "val_f1": meta["best_selection_f1"],
                         "best_epoch": meta["best_epoch"], "minutes": meta["elapsed_seconds"]/60,
                         "gpu_hours": meta["elapsed_seconds"]/3600,
                         "peak_gpu_gb": meta["peak_gpu_memory_allocated_bytes"]/1e9,
                         "micro_batch": meta["micro_batch_size"], "effective_batch": meta["effective_batch_size"],
                         "train_count": meta["train_count"], "validation_count": meta["validation_count"],
                         "postrun_checks": meta["postrun_checks"],
                         "curve_f1": [e["selection_f1"] for e in curves],
                         "curve_loss": [e["train_loss"] for e in curves],
                         "epoch_seconds": [e["epoch_seconds"] for e in curves],
                         "total_optimizer_steps": meta["total_steps"],
                         "examples_per_epoch_second_including_validation":
                            (meta["train_count"] * len(curves) / sum(e["epoch_seconds"] for e in curves)) if curves else None,
                         "optimizer_steps_per_epoch_second_including_validation":
                            (meta["total_steps"] / sum(e["epoch_seconds"] for e in curves)) if curves else None})
    restaurant = [r for r in rows if r["task"] == "absa_restaurant" and r["status"] == "completed"]
    selected = sorted(restaurant, key=lambda r: (-r["val_f1"], r["lr"]))[0] if restaurant else None
    best_lr = selected["lr"] if selected else None
    best = {"absa_restaurant": selected}
    if best_lr is not None:
        for task in ("visfd", "vsfc_topic"):
            best[task] = next((r for r in rows if r["task"] == task and r["lr"] == best_lr and r["status"] == "completed"), None)
    baseline = read(C.ROOT / "artifacts" / "base_encoder_baseline" / "summary.json")
    stage1 = read(C.ROOT / "artifacts" / "bidirlm_vi_mntp" / "summary.json")
    reference = {}
    for task in C.TASKS:
        reference[task] = {
            "bami": baseline["references"]["bami_e14"][task]["f1"],
            "mmbert": baseline["models"]["mmbert_base"]["selected"][task]["f1"],
            "cafe": baseline["references"]["cafe_fixed_2e-5"][task]["f1"],
            "O": next(r["val_f1"] for r in stage1["rows"] if r["arm"] == "O" and r["task"] == task and r["lr"] == 1e-5),
            "P_vi": next(r["val_f1"] for r in stage1["rows"] if r["arm"] == "P_vi" and r["task"] == task and r["lr"] == 1e-5),
        }
    expected = 5
    all_checks = all(r["status"] == "completed" and all(r["postrun_checks"].get(k) for k in
                     ("reload_matches_best", "prediction_file_matches_best", "reload_predictions_identical", "ids_stable_and_complete"))
                     for r in rows)
    budget = read(PHASE2 / "budget_estimate.json")
    if len(rows) < expected:
        gate = "BUDGET_STOP" if budget and budget.get("decision") == "STOP" else "INCOMPLETE"
    elif not all_checks:
        gate = "CHECK_FAIL"
    elif best["absa_restaurant"]["val_f1"] >= .772 and best["vsfc_topic"]["val_f1"] >= .812:
        gate = "PASS"
    else:
        gate = "CLOSE"
    preflight = read(PHASE2 / "preflight.json")
    summary = {"protocol": "bidirlm-1b-probe-ft-v1", "created_at": C.now(), "rows": rows,
               "restaurant_best_lr": best_lr,
               "best_selected": {k: ({"val_f1": v["val_f1"], "lr": v["lr"], "best_epoch": v["best_epoch"]} if v else None)
                                 for k, v in best.items()},
               "reference": reference, "gate": gate, "required_runs": expected,
               "budget": budget,
               "completed_runs": len(rows), "postrun_checks_all_pass": all_checks,
               "ft_gpu_hours": sum(r["gpu_hours"] for r in rows),
               "preflight_gpu_hours_conservative": (preflight or {}).get("elapsed_seconds", 0)/3600,
               "phase2_gpu_hours_conservative": (sum(r["gpu_hours"] for r in rows) + (preflight or {}).get("elapsed_seconds", 0)/3600)}
    summary["curves_svg"] = write_curves(rows)
    (PHASE2 / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    return summary


if __name__ == "__main__":
    result = summarize()
    if "--best-lr" in sys.argv:
        print(f"{result['restaurant_best_lr']:g}" if result["restaurant_best_lr"] is not None else "")
    else:
        print(json.dumps({"gate": result["gate"], "best_lr": result["restaurant_best_lr"],
                          "runs": result["completed_runs"], "phase2_gpu_hours": result["phase2_gpu_hours_conservative"]}))
