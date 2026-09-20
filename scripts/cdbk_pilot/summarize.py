"""Collect the pilot arms into artifacts/cdbk_pilot/summary.json and print the T/B/K table for the report."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ARMS, ARTIFACT_ROOT, ROOT  # noqa: E402

E14_B = ROOT / "artifacts" / "e14" / "trials" / "absa_restaurant" / "lr_5e-05" / "seed_42"


def load_run(path: Path) -> dict | None:
    meta_path = path / "metadata.json"
    if not meta_path.is_file():
        status = common.read_json(path / "status.json") if (path / "status.json").is_file() else {"status": "missing"}
        return {"status": status.get("status"), "path": str(path.relative_to(ROOT))}
    meta = common.read_json(meta_path)
    curves = common.read_json(path / "epoch_curves.json")["epochs"]
    best = meta["best_validation_metrics"]
    return {
        "status": meta.get("status"), "path": str(path.relative_to(ROOT)), "arm": meta.get("arm"), "init": meta.get("init"),
        "learning_rate": meta.get("learning_rate"), "seed": meta.get("seed"), "epochs": meta.get("epochs"),
        "best_epoch": meta.get("best_epoch"), "selection_f1": meta.get("best_selection_f1"),
        "pair_micro_f1": best.get("pair_micro_f1"), "pair_macro_f1": best.get("pair_macro_f1"), "detection_micro_f1": best.get("detection_micro_f1"),
        "elapsed_minutes": (meta.get("elapsed_seconds") or 0) / 60, "device": meta.get("runtime", {}).get("device"),
        "peak_gpu_gb": (meta.get("peak_gpu_memory_allocated_bytes") or 0) / 1e9, "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "postrun_checks": meta.get("postrun_checks"), "kd": meta.get("kd"), "source_hash": meta.get("source_hash"),
        "curve": [{"epoch": r["epoch"], "train_loss": r["train_loss"], "train_ce": r.get("train_ce"), "train_kd": r.get("train_kd"), "selection_f1": r["selection_f1"], "epoch_seconds": r["epoch_seconds"]} for r in curves],
        "final_epoch_f1": curves[-1]["selection_f1"] if curves else None,
        "final_train_loss": curves[-1]["train_loss"] if curves else None,
    }


def main() -> int:
    runs = {arm: load_run(common.run_dir(arm)) for arm in ARMS}
    e14 = None
    if (E14_B / "metadata.json").is_file():
        m = common.read_json(E14_B / "metadata.json")
        e14 = {"path": str(E14_B.relative_to(ROOT)), "selection_f1": m["best_selection_f1"], "best_epoch": m["best_epoch"], "elapsed_minutes": m["elapsed_seconds"] / 60, "checkpoint_sha256": m["checkpoint_sha256"]}
    f = lambda arm: runs[arm].get("selection_f1") if runs.get(arm) else None  # noqa: E731
    deltas = {}
    if f("T") is not None and f("B") is not None:
        deltas["T_minus_B_pp"] = 100 * (f("T") - f("B"))
    if f("K") is not None and f("B") is not None:
        deltas["K_minus_B_pp"] = 100 * (f("K") - f("B"))
    if f("K") is not None and f("T") is not None:
        deltas["K_minus_T_pp"] = 100 * (f("K") - f("T"))
    if e14 and f("B") is not None:
        deltas["B_minus_E14B_pp"] = 100 * (f("B") - e14["selection_f1"])
    checks = {name: (common.read_json(ARTIFACT_ROOT / "checks" / f"{name}.json").get("all_ok") if (ARTIFACT_ROOT / "checks" / f"{name}.json").is_file() else None) for name in ("loss_checks", "cache_checks")}
    summary = {"protocol": common.PROTOCOL, "task": common.TASK_ID, "seed": common.SEED, "runs": runs, "e14_reference_B": e14, "deltas_pp": deltas, "checks_all_ok": checks, "created_at": common.now()}
    common.write_json(ARTIFACT_ROOT / "summary.json", summary)
    print(f"| Arm | Init | Loss | LR | Best epoch | Val pair micro-F1 | Runtime (min) | Status |")
    print(f"|---|---|---|---|---|---|---|---|")
    for arm, r in runs.items():
        if r and r.get("selection_f1") is not None:
            print(f"| {arm} | {r['init']} | {'CE+KD' if ARMS[arm]['kd'] else 'CE'} | {ARMS[arm]['learning_rate']:.0e} | {r['best_epoch']} | {100*r['selection_f1']:.2f} | {r['elapsed_minutes']:.1f} | {r['status']} |")
        else:
            print(f"| {arm} | {ARMS[arm]['init']} | {'CE+KD' if ARMS[arm]['kd'] else 'CE'} | {ARMS[arm]['learning_rate']:.0e} | – | – | – | {r.get('status') if r else 'missing'} |")
    if e14:
        print(f"| E14 B (ref) | bamibert | CE | 5e-05 | {e14['best_epoch']} | {100*e14['selection_f1']:.2f} | {e14['elapsed_minutes']:.1f} | reference |")
    for k, v in deltas.items():
        print(f"{k}: {v:+.2f}")
    print("checks:", checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
