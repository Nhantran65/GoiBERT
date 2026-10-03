#!/usr/bin/env python3
"""Collect the stage-1 screen cells into artifacts/bidirlm_vi_mntp/summary.json and print O/P_vi tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TASK_LABELS = {"vsfc_sentiment": "Sentiment", "vsfc_topic": "Topic", "visfd": "ViSFD",
               "absa_restaurant": "Restaurant", "absa_hotel": "Hotel"}
REFERENCE = {
    "bamibert": {"absa_restaurant": 0.747442, "visfd": 0.843811, "vsfc_topic": 0.795279},
    "mmbert_base": {"absa_restaurant": 0.756262, "visfd": 0.836390, "vsfc_topic": 0.811808},
    "cafebert": {"absa_restaurant": 0.777252, "visfd": 0.854714, "vsfc_topic": 0.817359},
}
STAGE0_B2 = {"absa_restaurant": {"1e-05": 0.7572, "2e-05": 0.7513, "5e-05": 0.7389}, "visfd": {"1e-05": 0.8485},
             "vsfc_topic": {"1e-05": 0.8074}}
STAGE0_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "qwen3_encoder_probe" / "B2"
STAGE0_O_CELLS = {("O", "1e-05"): STAGE0_ROOT, ("O", "2e-05"): STAGE0_ROOT}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def cell(arm: str, task: str, lr: float, seed: int = 42):
    directory = C.arm_dir(arm, task, lr, seed)
    status = read_json(directory / "status.json")
    metadata = read_json(directory / "metadata.json")
    reused = False
    if (arm, f"{lr:g}") in STAGE0_O_CELLS and metadata is None:
        # arm O is stage-0 arm B2 (same checkpoint, same recipe): reuse those cells for LR 1e-5 / 2e-5
        directory = STAGE0_O_CELLS[(arm, f"{lr:g}")] / task / f"lr_{lr:.0e}" / f"seed_{seed}"
        status = read_json(directory / "status.json")
        metadata = read_json(directory / "metadata.json")
        reused = True
    if status is None and metadata is None:
        return None
    record = {"arm": arm, "task": task, "lr": lr, "seed": seed, "dir": str(directory),
              "status": (metadata or status or {}).get("status", "unknown"),
              "stage0_reuse": reused}
    if metadata:
        checks = metadata.get("postrun_checks", {})
        record.update({"val_f1": metadata.get("best_selection_f1"), "best_epoch": metadata.get("best_epoch"),
                       "minutes": (metadata.get("elapsed_seconds") or 0.0) / 60.0,
                       "peak_gpu_gb": (metadata.get("peak_gpu_memory_allocated_bytes") or 0) / 1e9,
                       "micro_batch": metadata.get("micro_batch_size"), "train_count": metadata.get("train_count"),
                       "checks_ok": all(checks.get(k) for k in ("reload_matches_best", "prediction_file_matches_best",
                                                                 "reload_predictions_identical", "ids_stable_and_complete")) if checks else None,
                       "curve_f1": [r.get("selection_f1") for r in (read_json(directory / "epoch_curves.json") or {}).get("epochs", [])],
                       "curve_loss": [r.get("train_loss") for r in (read_json(directory / "epoch_curves.json") or {}).get("epochs", [])],
                       "epoch_seconds": [r.get("epoch_seconds") for r in (read_json(directory / "epoch_curves.json") or {}).get("epochs", [])]})
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best-lr-per-arm", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    rows = []
    for arm in ("O", "P_vi"):
        for task in C.TASKS:
            for lr in C.LR_GRID:
                record = cell(arm, task, lr)
                if record:
                    rows.append(record)
    best = {}
    for arm in ("O", "P_vi"):
        candidates = [r for r in rows if r["arm"] == arm and r["task"] == "absa_restaurant"
                      and r["status"] == "completed" and r.get("val_f1") is not None]
        best[arm] = sorted(candidates, key=lambda r: (-r["val_f1"], r["lr"]))[0]["lr"] if candidates else None
    if args.best_lr_per_arm:
        for arm, lr in best.items():
            if lr is not None:
                print(f"{arm} {lr:g}")
        return 0
    by_key = {(r["arm"], r["task"], f"{r['lr']:g}"): r for r in rows}
    deltas = {}
    if best.get("O") and best.get("P_vi"):
        for task in C.TASKS:
            o = by_key.get(("O", task, f"{best['O']:g}"))
            p = by_key.get(("P_vi", task, f"{best['P_vi']:g}"))
            if o and p and o.get("val_f1") is not None and p.get("val_f1") is not None:
                deltas[task] = p["val_f1"] - o["val_f1"]
    summary = {"protocol": "bidirlm-vi-mntp-stage1-screen-v1", "created_at": C.now(), "rows": rows,
               "best_restaurant_lr_per_arm": best, "deltas_p_vi_minus_o": deltas, "reference": REFERENCE,
               "stage0_b2_reference": STAGE0_B2,
               "gpu_hours_recorded": sum((r.get("minutes") or 0) for r in rows) / 60.0}
    C.ART.mkdir(parents=True, exist_ok=True)
    (C.ART.parent / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(f"cells {len(rows)}  gpu-hours {summary['gpu_hours_recorded']:.1f}  best Restaurant LR {best}")
    print(f"{'arm':>5} {'task':<16} " + " ".join(f"{lr:>7g}" for lr in C.LR_GRID) + "   best")
    for arm in ("O", "P_vi"):
        for task in C.TASKS:
            vals = []
            for lr in C.LR_GRID:
                r = by_key.get((arm, task, f"{lr:g}"))
                vals.append(f"{100*r['val_f1']:>7.2f}" if r and r.get("val_f1") is not None else f"{'--':>7}")
            cells = [by_key.get((arm, task, f"{lr:g}")) for lr in C.LR_GRID]
            top = max((c["val_f1"] for c in cells if c and c.get("val_f1") is not None), default=None)
            print(f"{arm:>5} {TASK_LABELS[task]:<16} " + " ".join(vals) + (f"   {100*top:>6.2f}" if top else "      --"))
    for task, delta in deltas.items():
        print(f"delta P_vi - O ({TASK_LABELS[task]}): {100*delta:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
