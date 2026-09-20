#!/usr/bin/env python3
"""Recompute controlled-intervention paired metrics from saved logits/predictions only."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from controlled_aspect_interventions import OUT, target_stats, write_json  # noqa: E402


def main() -> int:
    reports = []
    for path in sorted(OUT.glob("student_e8__*.json")) + sorted(OUT.glob("bami_e14__*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        report["metrics"] = target_stats(report["rows"])
        write_json(path, report)
        reports.append(report)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary["reports"] = [{"model": r["model"], "task": r["task"], "inputs": r["inputs"], "metrics": r["metrics"]} for r in reports]
    write_json(OUT / "summary.json", summary)
    lines = ["# Controlled aspect interventions", "", "Root-authored validation-only fixture; metrics score explicitly annotated target slots only.", "", "| Model | Task | Inputs | Standalone target acc | Factorial joint acc | A changes after B flip | A correct→wrong |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in reports:
        m = r["metrics"]; c = m["A_B_flip_effect"]["A"]
        lines.append(f"| {r['model']} | {r['task']} | {r['inputs']} | {m['standalone_target_accuracy']['accuracy']:.3f} | {m['factorial_joint_accuracy']['accuracy']:.3f} | {c['changed']}/{c['total']} | {c['correct_to_wrong']} |")
    (OUT / "FACTUAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
