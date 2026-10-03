#!/usr/bin/env python3
"""Collect probe cells into artifacts/qwen3_encoder_probe/summary.json and print the arm x task x LR table.

``--best-lr-per-arm`` prints "arm lr" lines for the launcher's wave 2; ``--best-arm`` prints the single
best bidirectional arm at its best Restaurant LR (used for the seed-43 wave).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TASK_LABELS = {"absa_restaurant": "Restaurant", "visfd": "ViSFD", "vsfc_topic": "Topic"}
REFERENCE = {
    "bamibert": {"absa_restaurant": 0.747442, "visfd": 0.843811, "vsfc_topic": 0.795279},
    "mmbert_base": {"absa_restaurant": 0.756262, "visfd": 0.836390, "vsfc_topic": 0.811808},
    "cafebert": {"absa_restaurant": 0.777252, "visfd": 0.854714, "vsfc_topic": 0.817359},
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def cell(arm: str, task: str, lr: float, seed: int) -> dict[str, Any] | None:
    directory = common.arm_dir(arm, task, lr, seed)
    status = read_json(directory / "status.json")
    metadata = read_json(directory / "metadata.json")
    if status is None and metadata is None:
        return None
    record: dict[str, Any] = {"arm": arm, "task": task, "lr": lr, "seed": seed, "dir": str(directory),
                             "status": (metadata or status or {}).get("status", "unknown")}
    if metadata:
        checks = metadata.get("postrun_checks", {})
        record.update({
            "val_f1": metadata.get("best_selection_f1"), "best_epoch": metadata.get("best_epoch"),
            "minutes": (metadata.get("elapsed_seconds") or 0.0) / 60.0,
            "peak_gpu_gb": (metadata.get("peak_gpu_memory_allocated_bytes") or 0) / 1e9,
            "micro_batch": metadata.get("micro_batch_size"), "epochs": metadata.get("epochs"),
            "checks_ok": all(checks.get(k) for k in ("reload_matches_best", "prediction_file_matches_best",
                                                     "reload_predictions_identical", "ids_stable_and_complete")) if checks else None,
            "train_count": metadata.get("train_count"), "validation_count": metadata.get("validation_count"),
            "epoch_seconds": [row.get("epoch_seconds") for row in (read_json(directory / "epoch_curves.json") or {}).get("epochs", [])],
            "curve_f1": [row.get("selection_f1") for row in (read_json(directory / "epoch_curves.json") or {}).get("epochs", [])],
            "curve_loss": [row.get("train_loss") for row in (read_json(directory / "epoch_curves.json") or {}).get("epochs", [])],
        })
    else:
        record.update({"val_f1": None, "best_epoch": None, "minutes": (status or {}).get("elapsed_seconds", 0) / 60.0,
                       "checks_ok": None, "error": (status or {}).get("error")})
    return record


def collect(seeds: tuple[int, ...] = (42, 43)) -> list[dict[str, Any]]:
    rows = []
    for arm in ("C", "B0", "B2", "B1"):
        for task in common.TASKS:
            for lr in common.LR_GRID:
                for seed in seeds:
                    record = cell(arm, task, lr, seed)
                    if record:
                        rows.append(record)
    return rows


def best_lr_per_arm(rows: list[dict[str, Any]], task: str = "absa_restaurant", seed: int = 42,
                    arms: tuple[str, ...] = ("C", "B0", "B2")) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for arm in arms:
        candidates = [r for r in rows if r["arm"] == arm and r["task"] == task and r["seed"] == seed
                      and r["status"] == "completed" and r.get("val_f1") is not None]
        if not candidates:
            out[arm] = None
            continue
        # max validation F1; ties -> smaller LR (baseline protocol)
        out[arm] = sorted(candidates, key=lambda r: (-r["val_f1"], r["lr"]))[0]["lr"]
    return out


def pick(rows: list[dict[str, Any]], arm: str, task: str, lr: float, seed: int = 42) -> dict[str, Any] | None:
    for row in rows:
        if row["arm"] == arm and row["task"] == task and abs(row["lr"] - lr) < 1e-12 and row["seed"] == seed:
            return row
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best-lr-per-arm", action="store_true")
    ap.add_argument("--best-arm", action="store_true")
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    rows = collect()
    lrs = best_lr_per_arm(rows)

    if args.best_lr_per_arm:
        for arm, lr in lrs.items():
            if lr is not None:
                print(f"{arm} {lr:g}")
        return 0

    if args.best_arm:
        bidirectional = {a: (pick(rows, a, "absa_restaurant", lrs[a]) if lrs.get(a) else None) for a in ("B0", "B2", "B1")}
        scored = [(a, r["val_f1"], lrs[a]) for a, r in bidirectional.items() if r and r.get("val_f1") is not None]
        if not scored:
            print("")
            return 1
        best = sorted(scored, key=lambda t: (-t[1], t[0]))[0]
        print(f"{best[0]} {best[2]:g}")
        return 0

    by_arm_task_lr = {}
    for row in rows:
        if row["seed"] != 42:  # seed 43 is the noise-floor cell, kept separately
            continue
        by_arm_task_lr.setdefault(row["arm"], {}).setdefault(row["task"], {})[f"{row['lr']:g}"] = {
            k: row.get(k) for k in ("seed", "status", "val_f1", "best_epoch", "minutes", "peak_gpu_gb", "checks_ok", "micro_batch")
        }
    seed43_rows = [r for r in rows if r["seed"] == 43]
    # deltas on Restaurant, each arm at its own best Restaurant LR (plan §8 diagnostics)
    deltas: dict[str, Any] = {}
    restaurant = {arm: (pick(rows, arm, "absa_restaurant", lrs[arm]) if lrs.get(arm) else None) for arm in ("C", "B0", "B1", "B2")}

    def f1(arm: str, task: str = "absa_restaurant") -> float | None:
        lr = lrs.get(arm)
        if lr is None:
            return None
        row = pick(rows, arm, task, lr)
        return row.get("val_f1") if row else None

    if f1("C") is not None and f1("B0") is not None:
        deltas["B0_minus_C_restaurant"] = f1("B0") - f1("C")
    if f1("B0") is not None and f1("B2") is not None:
        deltas["B2_minus_B0_restaurant"] = f1("B2") - f1("B0")
    if f1("B0") is not None and f1("B1") is not None:
        deltas["B1_minus_B0_restaurant"] = f1("B1") - f1("B0")
    seed42_row = pick(rows, "B0", "absa_restaurant", lrs["B0"]) if lrs.get("B0") else None
    seed43_row = pick(rows, "B0", "absa_restaurant", lrs["B0"], seed=43) if lrs.get("B0") else None
    if seed42_row and seed43_row and seed43_row.get("val_f1") is not None:
        deltas["seed43_minus_seed42_restaurant_best_arm_B0"] = seed43_row["val_f1"] - seed42_row["val_f1"]

    gpu_hours = sum((row.get("minutes") or 0.0) for row in rows) / 60.0
    reference_table = {task: {model: REFERENCE[model][task] for model in REFERENCE} for task in common.TASKS}
    summary = {"protocol": common.PROTOCOL, "created_at": common.now(), "rows": rows,
               "by_arm_task_lr": by_arm_task_lr, "best_restaurant_lr_per_arm": lrs,
               "best_arm_restaurant": {a: (r.get("val_f1") if r else None) for a, r in restaurant.items()},
               "seed43_rows": seed43_rows, "deltas": deltas, "reference": reference_table, "runs": len(rows),
               "gpu_hours_recorded": gpu_hours}
    common.ART.mkdir(parents=True, exist_ok=True)
    (common.ART / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    print(f"cells: {len(rows)}  recorded GPU-hours: {gpu_hours:.1f}")
    print("best Restaurant LR per arm: " + ", ".join(f"{a}={lrs[a]:g}" if lrs.get(a) else f"{a}=-" for a in ("C", "B0", "B1", "B2")))
    header = f"{'arm':>4} {'task':<16} " + " ".join(f"{lr:>7g}" for lr in common.LR_GRID) + "   best"
    print(header)
    for arm in ("C", "B0", "B2", "B1"):
        for task in common.TASKS:
            cells = by_arm_task_lr.get(arm, {}).get(task, {})
            values = []
            for lr in common.LR_GRID:
                record = cells.get(f"{lr:g}")
                values.append(f"{100*record['val_f1']:>7.2f}" if record and record.get("val_f1") is not None else f"{'--':>7}")
            best = max((r["val_f1"] for r in cells.values() if r.get("val_f1") is not None), default=None)
            print(f"{arm:>4} {TASK_LABELS[task]:<16} " + " ".join(values) + (f"   {100*best:>6.2f}" if best is not None else "      --"))
    for key, value in deltas.items():
        print(f"delta {key}: {100*value:+.2f}")
    for row in seed43_rows:
        print(f"seed43 {row['arm']}/{TASK_LABELS[row['task']]}/lr{row['lr']:g}: "
              + (f"{100*row['val_f1']:.2f} (best epoch {row['best_epoch']}, {row['minutes']:.1f} min)" if row.get("val_f1") is not None else f"status={row['status']}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
