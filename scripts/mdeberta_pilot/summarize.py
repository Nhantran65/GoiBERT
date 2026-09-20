"""Build the compact validation-only mDeBERTa pilot readout."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mdeberta_pilot import common  # noqa: E402


BASELINES = {
    "vsfc_sentiment": (0.863252, 0.876924),
    "vsfc_topic": (0.795328, 0.817446),
    "visfd": (0.843834, 0.854665),
    "absa_restaurant": (0.747418, 0.777197),
    "absa_hotel": (0.732362, 0.733438),
}


def main() -> int:
    rows = []
    for task_id in common.TASKS:
        path = common.run_dir(task_id) / "metadata.json"
        metadata = common.read_json(path) if path.is_file() else {}
        score = metadata.get("best_selection_f1")
        bami, cafe = BASELINES[task_id]
        rows.append(
            {
                "task": task_id,
                "status": metadata.get("status", common.read_json(common.run_dir(task_id) / "status.json").get("status", "pending") if (common.run_dir(task_id) / "status.json").is_file() else "pending"),
                "mdeberta_validation_f1": score,
                "best_epoch": metadata.get("best_epoch"),
                "runtime_seconds": metadata.get("elapsed_seconds"),
                "bami_e14_validation_f1": bami,
                "cafe_validation_f1": cafe,
                "delta_vs_bami": score - bami if isinstance(score, (int, float)) else None,
                "delta_vs_cafe": score - cafe if isinstance(score, (int, float)) else None,
                "reload_matches_best": (metadata.get("postrun_checks") or {}).get("reload_matches_best"),
            }
        )
    common.write_json(common.ARTIFACT_ROOT / "summary.json", {"protocol": common.PROTOCOL, "rows": rows})
    with (common.ARTIFACT_ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = {
        "vsfc_sentiment": "VSFC sentiment",
        "vsfc_topic": "VSFC topic",
        "visfd": "ViSFD",
        "absa_restaurant": "ABSA restaurant",
        "absa_hotel": "ABSA hotel",
    }
    lines = [
        "# mDeBERTa-v3-base five-task pilot",
        "",
        "Validation-only screen: seed 42, LR 2e-5, 30 epochs, effective batch 32, max length 256. Test was not opened.",
        "",
        "| Task | Bami E14 val F1 (%) | Cafe val F1 (%) | mDeBERTa val F1 (%) | Δ Bami (pp) | Δ Cafe (pp) | Best epoch | Runtime (min) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(value: object) -> str:
            return f"{100*float(value):.4f}" if isinstance(value, (int, float)) else "pending"

        runtime = row["runtime_seconds"]
        runtime_text = f"{float(runtime)/60:.1f}" if isinstance(runtime, (int, float)) else "pending"
        lines.append(
            f"| {labels[row['task']]} | {100*row['bami_e14_validation_f1']:.4f} | {100*row['cafe_validation_f1']:.4f} | "
            f"{fmt(row['mdeberta_validation_f1'])} | {fmt(row['delta_vs_bami'])} | {fmt(row['delta_vs_cafe'])} | "
            f"{row['best_epoch'] if row['best_epoch'] is not None else 'pending'} | {runtime_text} | {row['status']} |"
        )
    lines += [
        "",
        "The model uses the native DebertaV2 ContextPooler and classifier: first hidden state, pooler dropout, native pooler dense/activation, classifier dropout, then task logits. ABSA uses independent per-slot CE; ViSFD `OTHERS` is binary and excluded from pair scoring. Encoder weights load as FP32 master parameters with BF16 autocast and native eager attention.",
        "",
    ]
    (common.ROOT / "docs" / "MDEBERTA_PILOT_RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", common.ARTIFACT_ROOT / "summary.json", common.ROOT / "docs" / "MDEBERTA_PILOT_RESULT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
