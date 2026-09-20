#!/usr/bin/env python3
"""Collect O/M/G/CafeBERT validation results, deltas, curves and the §6 screening gate -> artifacts/cafe_bami_repdistill_pilot/summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TASKS = ("absa_restaurant", "vsfc_topic", "visfd")
E14_LR = {"absa_restaurant": 5e-5, "vsfc_topic": 1e-5, "visfd": 5e-5}
CAFE_SOURCES = {"absa_restaurant": common.ROOT / "artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42/metadata.json",
                "vsfc_topic": common.ROOT / "artifacts/sealion_benchmark/cafe_runs/vsfc_topic/lr_2e-05/seed_42/metadata.json",
                "visfd": common.ROOT / "artifacts/sealion_benchmark/cafe_runs/visfd/lr_2e-05/seed_42/metadata.json"}


def e14_o(task: str) -> dict:
    grid = common.read_json(common.ROOT / "artifacts/e14/validation_grid.json")["grid"]
    row = [r for r in grid if r["task_id"] == task and abs(r["learning_rate"] - E14_LR[task]) < 1e-12][0]
    curves = common.read_json(Path(row["artifact_dir"]) / "epoch_curves.json")["epochs"]
    return {"f1": row["selection_f1"], "best_epoch": row["best_epoch"], "metric": row["selection_metric"], "source": row["artifact_dir"], "curve": [e["selection_f1"] for e in curves]}


def cafe(task: str) -> dict:
    meta = common.read_json(CAFE_SOURCES[task])
    curves = common.read_json(CAFE_SOURCES[task].parent / "epoch_curves.json")["epochs"]
    return {"f1": meta["best_selection_f1"], "best_epoch": meta["best_epoch"], "source": str(CAFE_SOURCES[task].parent.relative_to(common.ROOT)), "curve": [e["selection_f1"] for e in curves]}


def arm(a: str, task: str) -> dict | None:
    d = common.ART_DIR / "ft" / a / task
    if not (d / "metadata.json").exists():
        return None
    meta = common.read_json(d / "metadata.json")
    curve = [e["selection_f1"] for e in common.read_json(d / "epoch_curves.json")["epochs"]]
    return {"f1": meta["best_selection_f1"], "best_epoch": meta["best_epoch"], "status": meta["status"], "postrun_checks": meta["postrun_checks"], "head_init_sha256": meta["head_init_sha256"],
            "train_order_sha256": meta.get("train_order_epoch1_sha256"), "encoder_sha256": meta["encoder_sha256"], "elapsed_min": meta["elapsed_seconds"] / 60, "curve": curve, "source": str(d.relative_to(common.ROOT))}


def curve_stats(curve: list[float]) -> dict:
    if not curve:
        return {}
    best = max(curve); top3 = sorted(curve, reverse=True)[:3]
    return {"best": best, "top3_mean": sum(top3) / len(top3), "last5_mean": sum(curve[-5:]) / len(curve[-5:]), "epochs_within_0.5_of_best": sum(c >= best - 0.005 for c in curve), "n_epochs": len(curve)}


def main() -> int:
    out: dict = {"protocol": common.PROTOCOL, "created_at": common.now(), "tasks": {}, "cpt": {}}
    for a in ("M", "G"):
        s = common.ART_DIR / "cpt" / a / "summary.json"
        if s.exists():
            j = common.read_json(s)
            out["cpt"][a] = {k: j.get(k) for k in ("status", "steps_run", "tokens_processed", "tokens_processed_matches_schedule", "budget_error_batches", "passes_over_train", "wall_time_s", "gpu_hours", "tokens_per_s", "peak_gpu_memory_gb",
                                                    "mlm_first_50_mean", "mlm_last_50_mean", "rel_first_50_mean", "rel_last_50_mean", "masked_fraction")}
            out["cpt"][a]["holdout_mlm"] = j["holdout"]; out["cpt"][a]["export"] = j["export"]; out["cpt"][a]["batch_hashes"] = j["batch_hashes"]
            out["cpt"][a]["grad_ratio_min_max"] = [min(r["ratio"] for r in j["grad_ratios"]), max(r["ratio"] for r in j["grad_ratios"])] if j.get("grad_ratios") else None
    if "M" in out["cpt"] and "G" in out["cpt"]:
        out["cpt"]["same_batches_and_masks"] = out["cpt"]["M"]["batch_hashes"] == out["cpt"]["G"]["batch_hashes"]
        out["cpt"]["same_token_count"] = out["cpt"]["M"]["tokens_processed"] == out["cpt"]["G"]["tokens_processed"]
    gate_rows = []
    for t in TASKS:
        o, c, m, g = e14_o(t), cafe(t), arm("M", t), arm("G", t)
        row = {"O": o, "Cafe": c, "M": m, "G": g, "metric": o["metric"]}
        if m and g:
            row["deltas_pp"] = {"G-M": 100 * (g["f1"] - m["f1"]), "G-O": 100 * (g["f1"] - o["f1"]), "G-Cafe": 100 * (g["f1"] - c["f1"]), "M-O": 100 * (m["f1"] - o["f1"])}
            row["same_head_init"] = m["head_init_sha256"] == g["head_init_sha256"]; row["same_train_order"] = m["train_order_sha256"] == g["train_order_sha256"]
            row["curve_stats"] = {k: curve_stats(v["curve"]) for k, v in (("O", o), ("M", m), ("G", g), ("Cafe", c))}
            ref = max(o["f1"], m["f1"])
            gate_rows.append({"task": t, "G_minus_maxOM_pp": 100 * (g["f1"] - ref), "wins_by_0.5": g["f1"] - ref >= 0.005, "not_below_O_by_0.5": g["f1"] >= o["f1"] - 0.005})
        out["tasks"][t] = row
    if len(gate_rows) == 3:
        wins = [r for r in gate_rows if r["wins_by_0.5"]]
        others_ok = all(r["not_below_O_by_0.5"] for r in gate_rows if not r["wins_by_0.5"])
        key_win = any(r["task"] in ("absa_restaurant", "vsfc_topic") for r in wins)
        out["gate_6"] = {"rows": gate_rows, "wins": [r["task"] for r in wins], "n_wins": len(wins), "restaurant_or_topic_among_wins": key_win, "remaining_not_below_O_by_0.5": others_ok,
                         "pass": len(wins) >= 2 and key_win and others_ok, "rule": "G > max(O,M) by >= 0.5 pp on >= 2/3 tasks incl. Restaurant or Topic; remaining task not below O by > 0.5 pp (screening heuristic, not significance)"}
    common.write_json(common.ART_DIR / "summary.json", out)
    for t, r in out["tasks"].items():
        print(t, {k: (round(100 * r[k]["f1"], 2), r[k]["best_epoch"]) for k in ("O", "M", "G", "Cafe") if r.get(k)}, r.get("deltas_pp"))
    print("gate_6:", out.get("gate_6", {}).get("pass"), out.get("gate_6", {}).get("wins"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
