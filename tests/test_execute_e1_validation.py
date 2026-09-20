from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.execute_e1_validation import (
    FROZEN_SOURCE_HASH,
    REPO_ROOT,
    StateStore,
    TASK_IDS,
    TrialSpec,
    OrchestrationBlocked,
    _wait_for_pilot,
    candidate_records,
    metadata_matches,
    run_specs,
    select_final_lrs,
    select_top_two_lrs,
)


def _metadata(spec: TrialSpec, *, score: float = 0.5, source_hash: str = FROZEN_SOURCE_HASH) -> dict:
    return {
        "task_id": spec.task_id,
        "stage": spec.stage,
        "learning_rate": spec.learning_rate,
        "seed": spec.seed,
        "status": "completed",
        "source_hash": source_hash,
        "epochs": 30,
        "effective_batch_size": 32,
        "micro_batch_size": 32,
        "attention_implementation": "sdpa",
        "include_test": False,
        "runtime": {"precision": "bf16"},
        "best_epoch": 3,
        "best_selection_f1": score,
    }


def _write_complete_artifact(path: Path, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    import json

    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for name in (
        "resolved_config.json",
        "epoch_curves.json",
        "best_validation_metrics.json",
        "validation_predictions.json",
    ):
        (path / name).write_text("{}", encoding="utf-8")
    (path / "best_checkpoint.pt").write_bytes(b"checkpoint")


class DurableValidationTests(unittest.TestCase):
    def test_exact_contract_rejects_smoke_or_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            spec = TrialSpec("vsfc_sentiment", "screening", 2e-5, 42, str(artifact))
            metadata = _metadata(spec)
            _write_complete_artifact(artifact, metadata)
            self.assertTrue(metadata_matches(metadata, artifact, spec))
            metadata["source_hash"] = "wrong"
            self.assertFalse(metadata_matches(metadata, artifact, spec))
            metadata["source_hash"] = FROZEN_SOURCE_HASH
            metadata["smoke"] = True
            self.assertFalse(metadata_matches(metadata, artifact, spec))

    def test_candidate_collision_does_not_cherry_pick(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = TrialSpec("vsfc_sentiment", "screening", 2e-5, 42, str(root / "expected"))
            _write_complete_artifact(root / "a", _metadata(spec, score=0.5))
            _write_complete_artifact(root / "b", _metadata(spec, score=0.6))
            from scripts.execute_e1_validation import _candidate_record

            with self.assertRaises(RuntimeError):
                _candidate_record(root, spec)

    def test_top_two_and_final_require_complete_seed_sets(self) -> None:
        screening = []
        for task_index, task in enumerate(TASK_IDS):
            for lr_index, lr in enumerate((1e-5, 2e-5, 5e-5, 1e-4, 2e-4)):
                screening.append(
                    {
                        "task_id": task,
                        "stage": "screening",
                        "learning_rate": lr,
                        "seed": 42,
                        "best_selection_f1": (lr_index + task_index) / 10,
                    }
                )
        top_two = select_top_two_lrs(screening)
        self.assertEqual(set(top_two[TASK_IDS[0]]), {1e-4, 2e-4})
        shortlist = []
        for task in TASK_IDS:
            for lr in top_two[task]:
                for seed in (43, 44):
                    shortlist.append(
                        {
                            "task_id": task,
                            "stage": "shortlist",
                            "learning_rate": lr,
                            "seed": seed,
                            "best_selection_f1": 0.5 if seed != 44 else 0.7,
                        }
                    )
        selected = select_final_lrs(screening, shortlist, top_two)
        self.assertEqual(set(selected), set(TASK_IDS))

    def test_reuse_is_added_to_approved_state_without_gpu_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = TrialSpec("vsfc_sentiment", "screening", 2e-5, 42, str(root / "expected"))
            metadata = _metadata(spec)
            _write_complete_artifact(Path(spec.expected_dir), metadata)
            store = StateStore(root / ".orchestrator", FROZEN_SOURCE_HASH)
            records = run_specs(
                store,
                [spec],
                output_root=root,
                repo_root=REPO_ROOT,
                devices=["cuda:0"],
                poll_seconds=0.01,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(len(store.state["approved"]["screening"]), 1)

    def test_pilot_wait_rejects_completed_wrong_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            spec = TrialSpec("vsfc_sentiment", "screening", 2e-5, 42, str(path))
            metadata = _metadata(spec)
            metadata["epochs"] = 10
            _write_complete_artifact(path, metadata)
            with self.assertRaises(OrchestrationBlocked):
                _wait_for_pilot(path, path, 0.01)


if __name__ == "__main__":
    unittest.main()
