#!/usr/bin/env python3
"""Recompute validation F1 from saved predictions and apply the frozen pilot gate."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "cafe_oof_kd")]
import common as C  # noqa: E402
from goibert import train as T  # noqa: E402

ART = ROOT / "artifacts" / "bami_trunc_kd"
TASKS = ("vsfc_sentiment", "absa_restaurant")
ARMS = ((12, "kd"), (10, "ce"), (10, "kd"), (8, "ce"), (8, "kd"))


def read_run(path: Path, task: str, bundle) -> dict:
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("status") != "completed" or meta.get("include_test") is not False:
        raise ValueError(f"Incomplete or test-including run: {path}")
    if meta.get("postrun_checks") and not all(meta["postrun_checks"].values()):
        raise ValueError(f"Postrun check failed: {path}")
    rows = json.loads((path / "validation_predictions.json").read_text())["rows"]
    valid_ids = list(T.split_ids(T.split_field(bundle, "validation")))
    if [r["id"] for r in rows] != valid_ids:
        raise ValueError(f"Validation ID mismatch: {path}")
    score = T.selection_score(T.evaluate_predictions(task, bundle, [r["gold"] for r in rows],
                                                    [r["prediction"] for r in rows], split="validation"))
    if not math.isclose(score, float(meta["best_selection_f1"]), abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"Recomputed F1 differs from metadata: {path}")
    return {"selection_f1": score, "best_epoch": meta["best_epoch"], "path": str(path.relative_to(ROOT))}


def stats(values: dict[str, dict]) -> dict:
    scores = [values[str(s)]["selection_f1"] for s in (42, 43, 44)]
    return {"seeds": values, "mean": sum(scores) / 3, "range": [min(scores), max(scores)]}


def main():
    launcher = json.loads((ART / "launcher_summary.json").read_text())
    stopped = launcher.get("stopped_reason")
    table = {}
    for task in TASKS:
        bundle = T.load_task_bundle(task, ROOT, include_test=False)
        lr = "2e-05" if task == "vsfc_sentiment" else "5e-05"
        base_paths = {42: ROOT / "artifacts" / "e14" / "trials" / task / f"lr_{lr}" / "seed_42",
                      43: C.ART / "bami" / task / "seed_43", 44: C.ART / "bami" / task / "seed_44"}
        table[task] = {"Bami": stats({str(s): read_run(path, task, bundle) for s, path in base_paths.items()})}
        reproduction = ART / "runs" / "L12_ce" / task / "seed_42"
        table[task]["L12_ce_reproduction"] = read_run(reproduction, task, bundle)
        if stopped:
            continue
        for layers, arm in ARMS:
            key = f"L{layers}_{arm}"
            table[task][key] = stats({str(s): read_run(ART / "runs" / key / task / f"seed_{s}", task, bundle)
                                      for s in (42, 43, 44)})
    if stopped:
        comparison = {task: {"reproduction": table[task]["L12_ce_reproduction"]["selection_f1"],
                             "e14_seed_42": table[task]["Bami"]["seeds"]["42"]["selection_f1"],
                             "delta_points": 100 * (table[task]["L12_ce_reproduction"]["selection_f1"] -
                                                    table[task]["Bami"]["seeds"]["42"]["selection_f1"])}
                      for task in TASKS}
        C.write_json(ART / "summary.json", {"status": "stopped_after_reproduction", "table": table,
                                            "reproduction": comparison, "launcher": launcher})
        C.write_json(ART / "gate.json", {"status": "not_evaluated", "pilot_pass": None,
                                         "smallest_passing_depth": None, "reason": stopped,
                                         "reproduction": comparison,
                                         "rule": "both task means >= Bami task mean - 0.005"})
        print(json.dumps({"status": "not_evaluated", "reason": stopped, "reproduction": comparison}, ensure_ascii=False, indent=2))
        return
    deltas = {}
    for task in TASKS:
        t = table[task]
        baseline = t["Bami"]["mean"]
        deltas[task] = {"L12_kd_minus_Bami": t["L12_kd"]["mean"] - baseline}
        for layers in (10, 8):
            ce, kd = t[f"L{layers}_ce"]["mean"], t[f"L{layers}_kd"]["mean"]
            deltas[task][f"L{layers}_kd_minus_ce"] = kd - ce
            deltas[task][f"L{layers}_ce_minus_Bami"] = ce - baseline
            deltas[task][f"L{layers}_kd_minus_Bami"] = kd - baseline
    gates = {}
    for layers in (10, 8):
        by_task = {}
        for task in TASKS:
            got, base = table[task][f"L{layers}_kd"]["mean"], table[task]["Bami"]["mean"]
            by_task[task] = {"mean_kd": got, "mean_Bami": base, "threshold": base - 0.005,
                             "pass": got >= base - 0.005}
        gates[f"L{layers}"] = {"tasks": by_task, "pass": all(v["pass"] for v in by_task.values())}
    passing = [k for k in (8, 10) if gates[f"L{k}"]["pass"]]
    gate = {"pilot_pass": bool(passing), "smallest_passing_depth": min(passing) if passing else None,
            "depths": gates, "rule": "both task means >= Bami task mean - 0.005"}
    C.write_json(ART / "summary.json", {"table": table, "deltas": deltas,
                                        "launcher": launcher})
    C.write_json(ART / "gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
