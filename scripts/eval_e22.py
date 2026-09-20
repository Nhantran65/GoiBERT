#!/usr/bin/env python3
"""E22 evaluation: 3-seed e19-vs-d1_composite comparison (absa_hotel, visfd,
absa_restaurant) plus the Restaurant d1_composite-vs-d1_uniform sampling
ablation (seeds 42-44), written to ``docs/E22_RESULT.md``.

Reuses ``scripts/eval_e20.py``'s helpers verbatim: ``train_pair_frequency``
and ``error_decomposition_groups`` (both scored-aspects-aware as of the E22
fix, so ViSFD's binary OTHERS slot is correctly excluded from k/mixed/
rare-pair grouping and from missed/wrong/spurious tallies -- see the module
docstring changes in ``eval_e20.py``) and the seed-aware path helpers
(``_metrics_path`` / ``_predictions_path`` / ``_metadata_path``, which try
``seed_<seed>`` then fall back to ``seed_<seed>__rerun1``).

Deliberately lighter than ``eval_e20.evaluate_arm_task``: it reads only
``best_validation_metrics.json`` / ``validation_predictions.json`` /
``metadata.json`` (pair F1, group tables) and never loads a checkpoint, so it
needs no GPU and no ``models/goibert-b-dapt`` forward pass -- fork
contamination / composite battery are out of scope for this report.

Tolerant of missing trials throughout: any task/arm/seed cell without a
``best_validation_metrics.json``/``validation_predictions.json`` yet is
reported as ``status: "missing"`` (shown as "-" in the tables) instead of
raising, so this can be run before, during, or after
``scripts/e22_launch.sh`` finishes the queue. Nothing here modifies ``src/``,
``models/``, or any existing artifact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path("/home/kiettqa/GoiBERT")
sys.path.insert(0, str(ROOT / "scripts"))

import eval_e20 as E  # noqa: E402 -- reuse train_pair_frequency / error_decomposition_groups / path helpers

SEEDS = (42, 43, 44)
E19_VS_D1_TASKS = ("absa_hotel", "visfd", "absa_restaurant")
UNIFORM_TASK = "absa_restaurant"
RARE_KEY = "k>=2|rare_pair(<=5)"
MIXED_KEY = "k>=2|mixed"


# ---------------------------------------------------------------------------
# Per (task, arm, seed) cell: metrics + group table only, no checkpoint load
# ---------------------------------------------------------------------------
def evaluate_cell(
    task: str, arm: str, seed: int, pair_freq: dict[tuple[int, int], int], scored: Sequence[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {"task": task, "arm": arm, "seed": seed, "status": "ok", "notes": []}
    metrics_path = E._metrics_path(arm, task, seed)
    pred_path = E._predictions_path(arm, task, seed)
    meta_path = E._metadata_path(arm, task, seed)

    if not metrics_path.is_file() or not pred_path.is_file():
        result["status"] = "missing"
        result["notes"].append(f"missing metrics/predictions: {metrics_path} / {pred_path}")
        return result

    metrics = json.load(open(metrics_path))
    result["pair_f1"] = metrics.get("selection_f1")
    result["detection_micro_f1"] = metrics.get("detection_micro_f1")
    if meta_path.is_file():
        meta = json.load(open(meta_path))
        result["best_epoch"] = meta.get("best_epoch")
        result["elapsed_seconds"] = meta.get("elapsed_seconds")

    try:
        result["error_groups"] = E.error_decomposition_groups(pred_path, pair_freq, scored=scored)
    except Exception as exc:  # noqa: BLE001
        result["notes"].append(f"error_decomposition failed: {exc!r}")

    return result


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else None
    return mean, sd


def _f1_pct(cell: dict[str, Any]) -> float | None:
    f1 = cell.get("pair_f1")
    return round(100 * f1, 2) if isinstance(f1, (int, float)) else None


# ---------------------------------------------------------------------------
# 3-seed paired comparison table (arm_a vs arm_b, e.g. e19 vs d1_composite)
# ---------------------------------------------------------------------------
def paired_table(task: str, arm_a: str, arm_b: str, cells: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    a_vals: list[float] = []
    b_vals: list[float] = []
    diffs: list[float] = []
    for seed in SEEDS:
        ca = cells[seed][arm_a]
        cb = cells[seed][arm_b]
        fa = _f1_pct(ca)
        fb = _f1_pct(cb)
        diff = round(fb - fa, 2) if fa is not None and fb is not None else None
        rows.append(
            {
                "seed": seed,
                f"{arm_a}_status": ca["status"],
                f"{arm_a}_f1": fa,
                f"{arm_a}_best_epoch": ca.get("best_epoch"),
                f"{arm_b}_status": cb["status"],
                f"{arm_b}_f1": fb,
                f"{arm_b}_best_epoch": cb.get("best_epoch"),
                "paired_diff": diff,
            }
        )
        if fa is not None:
            a_vals.append(fa)
        if fb is not None:
            b_vals.append(fb)
        if diff is not None:
            diffs.append(diff)
    a_mean, a_sd = _mean_sd(a_vals)
    b_mean, b_sd = _mean_sd(b_vals)
    diff_mean, diff_sd = _mean_sd(diffs)
    return {
        "task": task,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "rows": rows,
        "n_seeds_both_present": len(diffs),
        "arm_a_mean_f1": round(a_mean, 2) if a_mean is not None else None,
        "arm_a_sd_f1": round(a_sd, 2) if a_sd is not None else None,
        "arm_b_mean_f1": round(b_mean, 2) if b_mean is not None else None,
        "arm_b_sd_f1": round(b_sd, 2) if b_sd is not None else None,
        "mean_paired_diff": round(diff_mean, 2) if diff_mean is not None else None,
        "sd_paired_diff": round(diff_sd, 2) if diff_sd is not None else None,
    }


def group_pct(cell: dict[str, Any], key: str, field: str) -> float | None:
    groups = cell.get("error_groups")
    if not groups or key not in groups:
        return None
    return groups[key].get(field)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def format_paired_table(pt: dict[str, Any]) -> str:
    arm_a, arm_b = pt["arm_a"], pt["arm_b"]
    lines = [
        f"| seed | {arm_a} F1 | {arm_a} epoch | {arm_b} F1 | {arm_b} epoch | paired diff ({arm_b} - {arm_a}) |",
        "|---|---|---|---|---|---|",
    ]
    for row in pt["rows"]:
        lines.append(
            f"| {row['seed']} | {row[f'{arm_a}_f1'] if row[f'{arm_a}_f1'] is not None else '-'} "
            f"({row[f'{arm_a}_status']}) | {row.get(f'{arm_a}_best_epoch','-')} | "
            f"{row[f'{arm_b}_f1'] if row[f'{arm_b}_f1'] is not None else '-'} ({row[f'{arm_b}_status']}) | "
            f"{row.get(f'{arm_b}_best_epoch','-')} | "
            f"{row['paired_diff'] if row['paired_diff'] is not None else '-'} |"
        )
    lines.append("")
    lines.append(
        f"**Mean +/- sd:** {arm_a} = {pt['arm_a_mean_f1']}"
        f"{' +/- ' + str(pt['arm_a_sd_f1']) if pt['arm_a_sd_f1'] is not None else ''}"
        f" (n={sum(1 for r in pt['rows'] if r[f'{arm_a}_f1'] is not None)}); "
        f"{arm_b} = {pt['arm_b_mean_f1']}"
        f"{' +/- ' + str(pt['arm_b_sd_f1']) if pt['arm_b_sd_f1'] is not None else ''}"
        f" (n={sum(1 for r in pt['rows'] if r[f'{arm_b}_f1'] is not None)})\n"
    )
    lines.append(
        f"**Mean paired diff ({arm_b} - {arm_a}):** "
        f"{pt['mean_paired_diff'] if pt['mean_paired_diff'] is not None else '-'} pts "
        f"(sd={pt['sd_paired_diff'] if pt['sd_paired_diff'] is not None else '-'}, "
        f"n={pt['n_seeds_both_present']}/{len(SEEDS)} seeds present in both arms)\n"
    )
    return "\n".join(lines)


def format_group_rows(task: str, arms: Sequence[str], cells: dict[int, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "| seed | arm | rare_pair(<=5) missed% | rare_pair(<=5) wrong% | mixed missed% | mixed wrong% |",
        "|---|---|---|---|---|---|",
    ]
    for seed in SEEDS:
        for arm in arms:
            cell = cells[seed][arm]
            if cell["status"] != "ok":
                lines.append(f"| {seed} | {arm} | - | - | - | - |")
                continue
            lines.append(
                f"| {seed} | {arm} | {group_pct(cell, RARE_KEY, 'missed_pct') if group_pct(cell, RARE_KEY, 'missed_pct') is not None else '-'} "
                f"| {group_pct(cell, RARE_KEY, 'wrong_pct') if group_pct(cell, RARE_KEY, 'wrong_pct') is not None else '-'} "
                f"| {group_pct(cell, MIXED_KEY, 'missed_pct') if group_pct(cell, MIXED_KEY, 'missed_pct') is not None else '-'} "
                f"| {group_pct(cell, MIXED_KEY, 'wrong_pct') if group_pct(cell, MIXED_KEY, 'wrong_pct') is not None else '-'} |"
            )
    return "\n".join(lines) + "\n"


def write_report(out_path: Path, e19_d1: dict[str, dict[str, Any]], restaurant_uniform: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# E22 Result -- ViSFD/Hotel d1_composite multi-seed + Restaurant d1_uniform ablation\n")
    lines.append(
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. Produced by `scripts/eval_e22.py`, "
        "reusing `scripts/eval_e20.py`'s `train_pair_frequency` / `error_decomposition_groups` (scored-aspects-"
        "aware; ViSFD's binary OTHERS slot excluded) and seed-aware path helpers. Pair F1 = `selection_f1` at the "
        "best validation epoch (never test). A `-` cell means that trial has not completed yet "
        "(`scripts/e22_launch.sh` still running or not yet launched).\n"
    )

    lines.append("\n## 1. e19 vs d1_composite, 3 seeds (42/43/44), per task\n")
    for task in E19_VS_D1_TASKS:
        pt = e19_d1[task]["paired"]
        cells = e19_d1[task]["cells"]
        lines.append(f"\n### {task}\n")
        lines.append(format_paired_table(pt))
        lines.append("\n#### Group rows (rare-pair / mixed), per seed\n")
        lines.append(format_group_rows(task, ("e19", "d1_composite"), cells))

    lines.append("\n## 2. Restaurant: d1_composite vs d1_uniform (sampling ablation), seeds 42-44\n")
    lines.append(
        "d1_uniform = same composite count/pool/boundary-assert recipe as d1_composite, but aspect pairs are "
        "drawn uniformly over valid (a,b) pairs instead of ∝ 1/(1+pair_freq). See `scripts/e22_preflight.json` "
        "for the pair-use-count uniformity check (stdev comparison) that validated the sampler before launch.\n"
    )
    lines.append(format_paired_table(restaurant_uniform["paired"]))
    lines.append("\n### Group rows (rare-pair / mixed), per seed\n")
    lines.append(format_group_rows(UNIFORM_TASK, ("d1_composite", "d1_uniform"), restaurant_uniform["cells"]))

    lines.append("\n## Sources\n")
    lines.append(
        "- e19: `artifacts/e19/<task>/seed_<seed>`.\n"
        "- d1_composite / d1_uniform: `artifacts/e20/<arm>/<task>/seed_<seed>` "
        "(or `seed_<seed>__rerun1` for the pre-fix seed-42 trials -- resolved automatically by "
        "`eval_e20._artifact_dir`).\n"
        "- Launch queue / logs: `scripts/e22_launch.sh`, `artifacts/e22/logs/`.\n"
        "- Preflight: `artifacts/e22/preflight.json`.\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "E22_RESULT.md")
    args = parser.parse_args()

    e19_d1: dict[str, dict[str, Any]] = {}
    for task in E19_VS_D1_TASKS:
        bundle = E.tr.load_task_bundle(task, ROOT, include_test=False)
        scored = E.scored_aspects(task, bundle)
        pair_freq = E.train_pair_frequency(task, scored=scored)
        cells: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            cells[seed] = {
                "e19": evaluate_cell(task, "e19", seed, pair_freq, scored),
                "d1_composite": evaluate_cell(task, "d1_composite", seed, pair_freq, scored),
            }
        paired = paired_table(task, "e19", "d1_composite", cells)
        e19_d1[task] = {"cells": cells, "paired": paired}

    ubundle = E.tr.load_task_bundle(UNIFORM_TASK, ROOT, include_test=False)
    uscored = E.scored_aspects(UNIFORM_TASK, ubundle)
    upair_freq = E.train_pair_frequency(UNIFORM_TASK, scored=uscored)
    ucells: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        ucells[seed] = {
            "d1_composite": evaluate_cell(UNIFORM_TASK, "d1_composite", seed, upair_freq, uscored),
            "d1_uniform": evaluate_cell(UNIFORM_TASK, "d1_uniform", seed, upair_freq, uscored),
        }
    restaurant_uniform = {"cells": ucells, "paired": paired_table(UNIFORM_TASK, "d1_composite", "d1_uniform", ucells)}

    write_report(args.out, e19_d1, restaurant_uniform)

    raw_path = ROOT / "artifacts" / "e22" / "eval_results.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump({"e19_vs_d1_composite": e19_d1, "restaurant_d1_composite_vs_d1_uniform": restaurant_uniform}, handle, indent=2, default=str)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
