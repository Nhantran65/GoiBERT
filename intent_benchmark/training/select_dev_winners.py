#!/usr/bin/env python3
"""Lock one dev-selected trial per backbone before any test evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "intent_benchmark"
RUNS = BENCHMARK / "runs"
RESULTS = BENCHMARK / "results"
CONFIG = BENCHMARK / "training/training_config.json"
APPROVAL = BENCHMARK / "training/root_approval.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    summaries = []
    for run_dir in sorted(RUNS.iterdir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed" or not (run_dir / "COMPLETED").exists():
            raise RuntimeError(f"trial is not terminal/completed: {run_dir}")
        summaries.append((run_dir, summary, summary_path))
    if len(summaries) != 6:
        raise RuntimeError(f"expected six completed trial summaries, found {len(summaries)}")
    if any((run_dir / "FAILED").exists() or (run_dir / "RUNNING").exists() for run_dir, _, _ in summaries):
        raise RuntimeError("a trial still has RUNNING or FAILED marker")
    if sha256(BENCHMARK / "training/train_intent.py") != approval["training_runner_sha256"]:
        raise RuntimeError("training runner changed after root approval")
    if sha256(CONFIG) != approval["config_sha256"]:
        raise RuntimeError("training config changed after root approval")

    by_model = defaultdict(list)
    for run_dir, summary, summary_path in summaries:
        by_model[summary["model_name"]].append((run_dir, summary, summary_path))
    if set(by_model) != set(config["models"]):
        raise RuntimeError(f"unexpected model groups: {sorted(by_model)}")
    selected = []
    for model_name in sorted(by_model):
        candidates = by_model[model_name]
        # Maximize dev macro-F1; exact ties choose the lower learning rate.
        winner = sorted(candidates, key=lambda item: (-float(item[1]["best_dev_macro_f1"]), float(item[1]["learning_rate"])))[0]
        run_dir, summary, summary_path = winner
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"missing checkpoint: {checkpoint}")
        selected.append({
            "model_name": model_name,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "learning_rate": float(summary["learning_rate"]),
            "seed": int(summary["seed"]),
            "best_epoch": int(summary["best_epoch"]),
            "best_dev_macro_f1": float(summary["best_dev_macro_f1"]),
            "best_dev_metrics": summary["best_dev_metrics"],
            "checkpoint_sha256": sha256(checkpoint),
            "summary_sha256": sha256(summary_path),
            "candidate_dev_scores": [
                {"run_dir": str(item[0].relative_to(ROOT)), "learning_rate": float(item[1]["learning_rate"]), "best_dev_macro_f1": float(item[1]["best_dev_macro_f1"]), "best_epoch": int(item[1]["best_epoch"])}
                for item in sorted(candidates, key=lambda item: float(item[1]["learning_rate"]))
            ],
        })

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_status": "locked",
        "locked_at_unix": time.time(),
        "selection_scope": "one winner per backbone; exactly three test evaluations permitted",
        "criteria": {
            "metric": "dev_macro_f1_fixed_eight_labels",
            "within_trial": "strict improvement; earlier epoch retained on tie",
            "across_learning_rates": "exact dev tie chooses 2e-5",
            "test_used_for_selection": False,
        },
        "approved_hashes": {
            "training_runner_sha256": approval["training_runner_sha256"],
            "config_sha256": approval["config_sha256"],
            "train_sha256": sha256(BENCHMARK / "train.jsonl"),
            "dev_sha256": sha256(BENCHMARK / "dev.jsonl"),
            "label_map_sha256": sha256(BENCHMARK / "label_map.json"),
        },
        "selected": selected,
    }
    (RESULTS / "selection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
