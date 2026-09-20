from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.execute_e2_validation import (
    ALL_STAGEB_SEEDS,
    CATEGORICAL_TASKS,
    E1_DEPENDENCY_HASH,
    EPOCHS,
    KD_ALPHAS,
    MICRO_BATCH_SIZE,
    PRECISION,
    PROTOCOL,
    REVISION,
    STAGE0_SEED,
    STUDENT_LEARNING_RATE,
    TASK_IDS,
    TEACHER_LEARNING_RATES,
    TrialSpec,
    OrchestrationBlocked,
    StateStore,
    _select_coefficients,
    _stable_cache_id_hash,
    build_command,
    cache_specs,
    candidate_records,
    e2_config_hash,
    e2_code_hash,
    metadata_matches,
    orchestrator_hash,
    run_specs,
    select_teachers,
    stagea_kd_specs,
    stagea_nc_specs,
    stageb_specs,
    teacher_specs,
)


HASHES = {
    "e1_dependency_hash": E1_DEPENDENCY_HASH,
    "e2_code_hash": "e2-code-test",
    "e2_config_hash": "e2-config-test",
    "orchestrator_hash": "orchestrator-test",
}


def _write_train_artifact(path: Path, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "best_checkpoint.pt").write_bytes(b"checkpoint")
    for name in (
        "resolved_config.json",
        "epoch_curves.json",
        "best_validation_metrics.json",
        "validation_predictions.json",
    ):
        (path / name).write_text("{}", encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _metadata(spec: TrialSpec, *, score: float = 0.5, best_epoch: int = 3, smoke: bool = False) -> dict:
    mode = {"teacher": "teacher", "ce_replay": "ce", "kd": "kd", "nc": "nc"}.get(spec.method)
    hyper = {"alpha": spec.coefficient if spec.method == "kd" else None,
             "consistency_lambda": spec.coefficient if spec.method == "nc" else None}
    return {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "stage": spec.stage,
        "method": spec.method,
        "coefficient": spec.coefficient,
        "task_id": spec.task_id,
        "mode": mode,
        "seed": spec.seed,
        "learning_rate": spec.learning_rate,
        "epochs": EPOCHS,
        "effective_batch_size": 32,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "include_test": False,
        "source_hashes": {
            "e1_dependency": E1_DEPENDENCY_HASH,
            "e2": HASHES["e2_code_hash"],
            "e2_config": HASHES["e2_config_hash"],
            "orchestrator": HASHES["orchestrator_hash"],
        },
        "runtime": {"precision": PRECISION},
        "attention_implementation": "sdpa",
        "hyperparameters": hyper,
        "best_epoch": best_epoch,
        "best_selection_f1": score,
        "status": "completed",
        "smoke": smoke,
    }


class E2OrchestrationTests(unittest.TestCase):
    def test_locked_grid_counts_and_sentinel_learning_rates(self) -> None:
        root = Path("/tmp") / "e2-grid-test"
        self.assertEqual(len(teacher_specs(root)), 15)
        self.assertEqual(len(cache_specs(root)), 5)
        self.assertEqual(len(stagea_kd_specs(root)), 15)
        self.assertEqual(len(stagea_nc_specs(root)), 10)
        self.assertEqual(len(stageb_specs(root, {task: 0.5 for task in TASK_IDS}, {task: 0.1 for task in TASK_IDS})), 20)
        self.assertEqual(
            {spec.learning_rate for spec in stagea_kd_specs(root) if spec.task_id in CATEGORICAL_TASKS},
            {1e-5},
        )
        self.assertEqual(
            {spec.learning_rate for spec in stagea_nc_specs(root) if spec.task_id not in CATEGORICAL_TASKS},
            {1e-4},
        )
        self.assertTrue(all(spec.expected_dir.endswith("teacher_logits.pt") for spec in cache_specs(root)))

    def test_teacher_selection_is_validation_only_and_lower_lr_breaks_tie(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for task in TASK_IDS:
                for lr in TEACHER_LEARNING_RATES:
                    spec = TrialSpec("stage0_teacher", task, "teacher", lr, STAGE0_SEED, None, str(root / task / str(lr)))
                    artifact = Path(spec.expected_dir)
                    artifact.mkdir(parents=True)
                    (artifact / "best_checkpoint.pt").write_bytes(f"{task}-{lr}".encode())
                    records.append({
                        "stage": "stage0_teacher",
                        "method": "teacher",
                        "task_id": task,
                        "seed": STAGE0_SEED,
                        "learning_rate": lr,
                        "best_epoch": 9 if lr == 1e-5 else 2,
                        "best_selection_f1": 0.8 if lr != 1e-4 else 0.7,
                        "artifact_dir": str(artifact),
                    })
            selected = select_teachers(records)
            for task in TASK_IDS:
                self.assertEqual(selected[task]["learning_rate"], 1e-5)
                self.assertEqual(selected[task]["teacher_strength_is_gate"], False)
                self.assertTrue(Path(selected[task]["checkpoint"]).is_file())

    def test_coefficient_selection_requires_grid_and_breaks_tie_low(self) -> None:
        rows = []
        for task in TASK_IDS:
            for alpha in KD_ALPHAS:
                rows.append({
                    "stage": "stagea_kd",
                    "method": "kd",
                    "task_id": task,
                    "seed": 42,
                    "coefficient": alpha,
                    "best_selection_f1": 0.7 if alpha != 0.75 else 0.8,
                })
        selected = _select_coefficients(rows, stage="stagea_kd", method="kd", candidates=KD_ALPHAS)
        self.assertEqual(set(selected.values()), {0.75})
        rows[-1]["best_selection_f1"] = 0.7
        rows[-2]["best_selection_f1"] = 0.7
        rows[-3]["best_selection_f1"] = 0.7
        selected = _select_coefficients(rows, stage="stagea_kd", method="kd", candidates=KD_ALPHAS)
        self.assertEqual(selected[TASK_IDS[-1]], 0.25)
        with self.assertRaises(OrchestrationBlocked):
            _select_coefficients(rows[:-1], stage="stagea_kd", method="kd", candidates=KD_ALPHAS)

    def test_commands_match_canonical_runner_and_cache_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher_ckpt = root / "teacher.pt"
            teacher_ckpt.write_bytes(b"teacher")
            selected = {"vsfc_sentiment": {"checkpoint": str(teacher_ckpt)}}
            kd = stagea_kd_specs(root, tasks=("vsfc_sentiment",))[0]
            command = build_command(kd, repo_root=Path("/repo"), output_root=root, device="cuda:0", hashes=HASHES, selected_teachers=selected)
            self.assertEqual(command[0:3], [sys.executable, "-m", "goibert_e2.train"])
            self.assertIn("--mode", command)
            self.assertIn("kd", command)
            self.assertIn("--alpha", command)
            self.assertIn("--cache-path", command)
            self.assertIn("--stage", command)
            self.assertIn("--method", command)
            selected_all = {task: {"learning_rate": 2e-5, "checkpoint": str(teacher_ckpt)} for task in TASK_IDS}
            cache = cache_specs(root, selected_all)[0]
            cache_command = build_command(cache, repo_root=Path("/repo"), output_root=root, device="cuda:0", hashes=HASHES, selected_teachers=selected_all)
            self.assertEqual(cache_command[0:3], [sys.executable, "-m", "goibert_e2.train"])
            self.assertIn("--mode", cache_command)
            self.assertIn("cache", cache_command)
            self.assertEqual(cache_command[cache_command.index("--cache-path") + 1], cache.expected_dir)

    def test_matching_completed_artifact_reused_and_smoke_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = stagea_kd_specs(root, tasks=("vsfc_sentiment",))[0]
            metadata = _metadata(spec)
            _write_train_artifact(Path(spec.expected_dir), metadata)
            found = candidate_records(root, spec, e2_code_hash_value=HASHES["e2_code_hash"], e2_config_hash_value=HASHES["e2_config_hash"], orchestrator_hash_value=HASHES["orchestrator_hash"])
            self.assertEqual(len(found), 1)
            metadata["smoke"] = True
            (Path(spec.expected_dir) / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(OrchestrationBlocked):
                candidate_records(root, spec, e2_code_hash_value=HASHES["e2_code_hash"], e2_config_hash_value=HASHES["e2_config_hash"], orchestrator_hash_value=HASHES["orchestrator_hash"])

    def test_failed_identity_cannot_be_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = stagea_nc_specs(root, tasks=("vsfc_sentiment",))[0]
            store = StateStore(root / ".orchestrator", HASHES)
            store.state["failed"].append({"spec": {**spec.__dict__}, "reason": "OOM"})
            store.save()
            with patch("scripts.execute_e2_validation.free_memory_gib", return_value=99.0):
                with self.assertRaises(OrchestrationBlocked):
                    run_specs(store, [spec], output_root=root, repo_root=Path("/repo"), devices=("cuda:0",), poll_seconds=0.0, hashes=HASHES)

    def test_cache_metadata_rejects_stale_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = cache_specs(root)[0]
            summary = {
                "task_id": spec.task_id,
                "split": "train",
                "source_ids": ["a", "b"],
                "source_ids_hash": _stable_cache_id_hash(["a", "b"]),
                "e1_dependency_source_hash": E1_DEPENDENCY_HASH,
                "source_hashes": {
                    "e1_dependency": E1_DEPENDENCY_HASH,
                    "e2": HASHES["e2_code_hash"],
                    "e2_config": HASHES["e2_config_hash"],
                    "orchestrator": HASHES["orchestrator_hash"],
                },
            }
            cache_path = Path(spec.expected_dir)
            with patch("scripts.execute_e2_validation._load_cache_summary", return_value=summary):
                self.assertTrue(
                    metadata_matches(
                        summary,
                        cache_path,
                        spec,
                        e2_code_hash=HASHES["e2_code_hash"],
                        e2_config_hash=HASHES["e2_config_hash"],
                        orchestrator_hash=HASHES["orchestrator_hash"],
                    )
                )
                summary["source_hashes"]["e2"] = "stale"
                self.assertFalse(
                    metadata_matches(
                        summary,
                        cache_path,
                        spec,
                        e2_code_hash=HASHES["e2_code_hash"],
                        e2_config_hash=HASHES["e2_config_hash"],
                        orchestrator_hash=HASHES["orchestrator_hash"],
                    )
                )


if __name__ == "__main__":
    unittest.main()
