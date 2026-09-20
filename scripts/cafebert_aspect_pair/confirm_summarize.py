"""Post-run verification + summary for the seed-confirmation round.

For each seed-43 run: seed really reached RNG/head-init/sampler (hashes equal checks/pre_launch.json), T and A share the
per-epoch review order, 6600 optimizer steps, best-epoch checkpoint reloads to the same F1, F1 recomputed from the saved
(regrouped) predictions with the unchanged evaluator, IDs/gold equal the loader.  Then paired deltas across seeds 42/43.

Writes artifacts/cafebert_aspect_pair_confirmation/seed43/summary.json.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as ap  # noqa: E402
from common import e1_train  # noqa: E402

import torch  # noqa: E402

ROOT = ap.ROOT
CONFIRM = ROOT / "artifacts" / "cafebert_aspect_pair_confirmation" / "seed43"
RUNS = {
    (42, "T"): ap.REFERENCE_RUN, (42, "A"): ap.run_dir("A"),
    (43, "T"): CONFIRM / "T_absa_restaurant_seed43", (43, "A"): CONFIRM / "A_absa_restaurant_seed43",
}


def load(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"dir": str(path.relative_to(ROOT))}
    for name in ("status", "metadata", "epoch_curves", "best_validation_metrics", "validation_predictions", "seed_checks"):
        f = path / f"{name}.json"
        out[name] = ap.read_json(f) if f.exists() else None
    return out


def main() -> int:
    contract = ap.task_contract()
    bundle = contract["bundle"]
    valid = ap.reviews_of(contract["validation"])
    pre = ap.read_json(CONFIRM / "checks" / "pre_launch.json")
    runs = {k: load(p) for k, p in RUNS.items()}
    summary: dict[str, Any] = {"generated_at": ap.now(), "runs": {}, "pre_launch_ok": pre["ok"]}
    f1: dict[tuple[int, str], float] = {}
    for (seed, arm), run in runs.items():
        key = f"{arm}_seed{seed}"
        if run["validation_predictions"] is None:
            summary["runs"][key] = {"status": "missing", "dir": run["dir"]}
            continue
        rows = run["validation_predictions"]["rows"]
        meta, status, curves = run["metadata"] or {}, run["status"] or {}, (run["epoch_curves"] or {}).get("epochs", [])
        recomputed = ap.rows_f1(bundle, rows)
        f1[(seed, arm)] = recomputed
        entry: dict[str, Any] = {
            "dir": run["dir"], "status": status.get("status"), "pid": status.get("pid"), "device": status.get("device"),
            "seed_in_metadata": meta.get("seed"), "best_epoch": run["validation_predictions"]["epoch"], "recomputed_pair_micro_f1": recomputed,
            "stored_best_f1": status.get("best_selection_f1"), "stored_equals_recomputed": abs(recomputed - float(status.get("best_selection_f1", -1))) < 1e-9,
            "rows_aligned_with_loader": [r["id"] for r in rows] == [r.id for r in valid] and [list(r["gold"]) for r in rows] == [list(r.gold) for r in valid],
            "rows": len(rows), "global_step_final": curves[-1]["global_step"] if curves else None, "epochs_run": len(curves),
            "total_steps": meta.get("total_steps"), "warmup_steps": meta.get("warmup_steps"), "lr_final": curves[-1]["learning_rate"] if curves else None,
            "postrun_checks": meta.get("postrun_checks"), "elapsed_minutes": (meta.get("elapsed_seconds") or 0) / 60,
            "epoch_seconds_mean": (sum(e["epoch_seconds"] for e in curves) / len(curves)) if curves else None,
            "peak_gpu_allocated_gib": (meta.get("peak_gpu_memory_allocated_bytes") or 0) / 2**30, "curve_f1": [round(100 * e["selection_f1"], 4) for e in curves],
            "metrics": {k: v for k, v in (run["best_validation_metrics"] or {}).items() if isinstance(v, (int, float))},
            "checkpoint_sha256": meta.get("checkpoint_sha256"), "started_at": meta.get("started_at"), "finished_at": meta.get("finished_at"),
        }
        if seed == 43:
            sc = run["seed_checks"] or {}
            ckpt = torch.load(RUNS[(seed, arm)] / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            entry["seed_propagation"] = {
                "seed_checks": {k: v for k, v in sc.items() if not isinstance(v, list)},
                "head_init_equals_expected": sc.get("head_init_sha256") == pre["head_init"]["expected"][arm]["43"],
                "review_order_equals_expected_all_epochs": sc.get("review_order_sha256_per_epoch") == pre["review_order"]["expected_per_epoch"],
                "checkpoint_metadata_seed": ckpt.get("metadata", {}).get("seed"), "checkpoint_epoch": ckpt.get("epoch"), "checkpoint_global_step": ckpt.get("global_step"),
            }
            del ckpt
        summary["runs"][key] = entry
    if all(k in f1 for k in [(43, "T"), (43, "A")]):
        t43, a43 = runs[(43, "T")]["seed_checks"], runs[(43, "A")]["seed_checks"]
        summary["seed43_T_A_same_review_order"] = (t43 or {}).get("review_order_sha256_per_epoch") == (a43 or {}).get("review_order_sha256_per_epoch")
        summary["seed43_T_A_same_review_order_all_epochs_hash"] = (t43 or {}).get("review_order_sha256_all_epochs")
    deltas = {}
    for seed in (42, 43):
        if (seed, "T") in f1 and (seed, "A") in f1:
            deltas[f"seed{seed}"] = {"T": f1[(seed, "T")], "A": f1[(seed, "A")], "A_minus_T_pp": 100 * (f1[(seed, "A")] - f1[(seed, "T")])}
    if len(deltas) == 2:
        d = [deltas["seed42"]["A_minus_T_pp"], deltas["seed43"]["A_minus_T_pp"]]
        deltas["mean_paired_delta_pp"] = sum(d) / 2
        deltas["both_seeds_positive"] = all(v > 0 for v in d)
        band = "both positive, mean >= 0.5: strong enough to consider extending" if all(v > 0 for v in d) and sum(d) / 2 >= 0.5 else (
            "both positive, mean < 0.5: small gain, weigh 12x encoder cost" if all(v > 0 for v in d) else "seed43 did not gain: replication not confirmed")
        deltas["reading"] = band
        deltas["mean_T"] = (f1[(42, "T")] + f1[(43, "T")]) / 2; deltas["mean_A"] = (f1[(42, "A")] + f1[(43, "A")]) / 2
    summary["deltas"] = deltas
    for key in ("T_seed43", "A_seed43"):
        e = summary["runs"].get(key, {})
        e["all_checks_ok"] = bool(e.get("status") == "completed" and e.get("stored_equals_recomputed") and e.get("rows_aligned_with_loader") and e.get("global_step_final") == 6600
                                  and e.get("total_steps") == 6600 and e.get("warmup_steps") == 660 and e.get("epochs_run") == 30 and e.get("seed_in_metadata") == 43
                                  and all((e.get("postrun_checks") or {}).values()) and e.get("seed_propagation", {}).get("head_init_equals_expected")
                                  and e.get("seed_propagation", {}).get("review_order_equals_expected_all_epochs") and e.get("seed_propagation", {}).get("checkpoint_metadata_seed") == 43)
    ap.write_json(CONFIRM / "summary.json", summary)
    for key, e in summary["runs"].items():
        if "recomputed_pair_micro_f1" in e:
            print(f"{key}: F1 {100*e['recomputed_pair_micro_f1']:.4f} @ epoch {e['best_epoch']} ({e['status']}, {e['elapsed_minutes']:.1f} min, {e['device']}, pid {e['pid']}) checks_ok={e.get('all_checks_ok')}")
    print("same review order T/A seed43:", summary.get("seed43_T_A_same_review_order"))
    print("deltas:", summary["deltas"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
