"""Freeze the E1 test checkpoint selection from completed validation records.

This is an audit helper only.  It does not train, inspect test metrics, or
choose a checkpoint from test data.  The five task learning rates and seeds
are fixed by the E1 gate before this script is run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from goibert.data import load_task
from goibert.train import FINAL_SEEDS, TASK_IDS, bundle_provenance, read_json, sha256_file, source_hash


ROOT = Path(__file__).resolve().parents[1]
RECORD_ROOT = ROOT / "artifacts" / "e1" / ".orchestrator" / "attempts"
OUT = ROOT / "artifacts" / "e1" / "test_selection_manifest.json"
SELECTED_LR = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}


def _metadata(task_id: str, seed: int, learning_rate: float) -> tuple[Path, dict]:
    candidates = []
    for path in sorted(RECORD_ROOT.glob(f"{task_id}__*/metadata.json")):
        row = read_json(path)
        if (
            row.get("task_id") == task_id
            and int(row.get("seed", -1)) == seed
            and float(row.get("learning_rate", -1.0)) == learning_rate
            and row.get("status") == "completed"
            and int(row.get("epochs", -1)) == 30
            and row.get("include_test") is False
        ):
            candidates.append((path, row))
    # Seed 42 is the screening record; seeds 43/44 are shortlist records;
    # seeds 45/46 are final records.  A unique identity is required.
    expected_stage = "screening" if seed == 42 else "shortlist" if seed in (43, 44) else "final"
    candidates = [(p, r) for p, r in candidates if r.get("stage") == expected_stage]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one completed {task_id}/{expected_stage}/seed{seed}; found {len(candidates)}")
    metadata_path, row = candidates[0]
    return metadata_path.parent, row


def _file_hashes(path: Path) -> dict[str, str]:
    return {str(p.relative_to(path)): sha256_file(p) for p in sorted(path.rglob("*")) if p.is_file()}


def main() -> None:
    current_source_hash = source_hash(ROOT)
    selections = []
    for task_id in TASK_IDS:
        bundle = load_task(task_id, ROOT, include_test=True)
        test_provenance = bundle_provenance(bundle, ROOT)
        for seed in FINAL_SEEDS:
            artifact_dir, metadata = _metadata(task_id, seed, SELECTED_LR[task_id])
            checkpoint = artifact_dir / "best_checkpoint.pt"
            metadata_path = artifact_dir / "metadata.json"
            if not checkpoint.is_file() or not metadata_path.is_file():
                raise RuntimeError(f"Incomplete selected artifact: {artifact_dir}")
            if metadata.get("source_hash") != current_source_hash:
                raise RuntimeError(f"Source hash mismatch in {metadata_path}")
            if metadata.get("backbone_hashes") is None or metadata.get("tokenizer_hashes") is None:
                raise RuntimeError(f"Missing model/tokenizer provenance in {metadata_path}")
            if metadata.get("data_provenance", {}).get("evaluator_code_hash") != test_provenance.get("evaluator_code_hash"):
                raise RuntimeError(f"Evaluator hash mismatch in {metadata_path}")
            selections.append(
                {
                    "task_id": task_id,
                    "stage": metadata["stage"],
                    "seed": seed,
                    "learning_rate": metadata["learning_rate"],
                    "best_epoch": metadata["best_epoch"],
                    "artifact_dir": str(artifact_dir.resolve()),
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "metadata_path": str(metadata_path.resolve()),
                    "metadata_sha256": sha256_file(metadata_path),
                    "source_hash": current_source_hash,
                    "data_provenance": test_provenance,
                    "backbone_hashes": metadata["backbone_hashes"],
                    "tokenizer_hashes": metadata["tokenizer_hashes"],
                    "precision": metadata["runtime"]["precision"],
                    "attention_implementation": metadata.get("attention_implementation", "sdpa"),
                    "max_length": metadata["max_length"],
                    "effective_batch_size": metadata["effective_batch_size"],
                    "micro_batch_size": metadata["micro_batch_size"],
                    "epochs": metadata["epochs"],
                    "model": metadata["model"],
                    "selection_basis": "best validation selection_f1; LR fixed before test access",
                }
            )
    if len(selections) != 25 or len({(r["task_id"], r["seed"]) for r in selections}) != 25:
        raise RuntimeError("Selection manifest does not contain exactly 25 unique task/seed records")
    payload = {
        "protocol": "E1-test-gate-v1",
        "manifest_id": "E1-test-selection-20260908-v1",
        "locked": True,
        "created_at": "2026-09-08T00:00:00Z",
        "source_hash": current_source_hash,
        "checkpoint_count": len(selections),
        "tasks": list(TASK_IDS),
        "seeds": list(FINAL_SEEDS),
        "selected_learning_rates": SELECTED_LR,
        "paper_metric_equivalence": "unresolved; internal test metrics are reported with explicit evaluator names",
        "test_access_policy": "each locked checkpoint evaluated once; no test-based selection or tuning",
        "selections": selections,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUT), "count": len(selections), "source_hash": current_source_hash}, indent=2))


if __name__ == "__main__":
    main()
