from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from scripts import execute_e2_recovery_v2 as recovery
from scripts import execute_e2_validation as frozen


HASHES = {
    "e1_dependency_hash": frozen.E1_DEPENDENCY_HASH,
    "e2_code_hash": "e2-code-fixture",
    "e2_config_hash": "e2-config-fixture",
    "orchestrator_hash": "scheduler-fixture",
}


def write_train_artifact(path: Path, spec: frozen.TrialSpec, *, complete: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if complete:
        metadata = {
            "protocol": frozen.PROTOCOL,
            "revision": frozen.REVISION,
            "stage": spec.stage,
            "method": spec.method,
            "coefficient": spec.coefficient,
            "task_id": spec.task_id,
            "mode": {"teacher": "teacher", "kd": "kd", "nc": "nc"}[spec.method],
            "seed": spec.seed,
            "learning_rate": spec.learning_rate,
            "epochs": frozen.EPOCHS,
            "effective_batch_size": frozen.EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": frozen.TEACHER_MICRO_BATCH_SIZE if spec.method == "teacher" else frozen.MICRO_BATCH_SIZE,
            "include_test": False,
            "source_hashes": {
                "e1_dependency": frozen.E1_DEPENDENCY_HASH,
                "e2": HASHES["e2_code_hash"],
                "e2_config": HASHES["e2_config_hash"],
                "orchestrator": HASHES["orchestrator_hash"],
            },
            "runtime": {"precision": frozen.PRECISION},
            "attention_implementation": frozen.ATTENTION_IMPLEMENTATION,
            "hyperparameters": {
                "alpha": spec.coefficient if spec.method == "kd" else None,
                "consistency_lambda": spec.coefficient if spec.method == "nc" else None,
            },
            "best_epoch": 3,
            "best_selection_f1": 0.5,
            "status": "completed",
            "smoke": False,
        }
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (path / "best_checkpoint.pt").write_bytes(b"checkpoint")
        for name in (
            "resolved_config.json",
            "epoch_curves.json",
            "best_validation_metrics.json",
            "validation_predictions.json",
        ):
            (path / name).write_text("{}", encoding="utf-8")
    else:
        (path / "status.json").write_text(json.dumps({"status": "running", "epoch": 21}), encoding="utf-8")
        (path / "epoch_curves.json").write_text(json.dumps({"epochs": [1, 2, 21]}), encoding="utf-8")


def add_job(store: frozen.StateStore, spec: frozen.TrialSpec, pid: int) -> dict:
    job = {
        "spec": asdict(spec),
        "output_dir": spec.expected_dir,
        "pid": pid,
        "log_path": str(store.root / "logs" / f"{spec.slug()}.log"),
        "status": "running",
    }
    Path(job["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["log_path"]).write_text("runner fixture\n", encoding="utf-8")
    store.state.setdefault("jobs", []).append(job)
    store.save()
    return job


class E2RecoveryV2Tests(unittest.TestCase):
    def test_adapter_checks_e1_then_calls_frozen_three_key_api(self) -> None:
        calls = []

        def frozen_validator(*args, **kwargs):
            calls.append(kwargs)
            return True

        original = getattr(frozen, "_E2_RECOVERY_V2_FROZEN_METADATA_MATCHES", None)
        frozen._E2_RECOVERY_V2_FROZEN_METADATA_MATCHES = frozen_validator
        try:
            spec = frozen.teacher_specs(Path("/tmp"), tasks=("vsfc_topic",))[0]
            self.assertTrue(
                recovery.metadata_matches_compat(
                    frozen,
                    {},
                    Path("/tmp/artifact"),
                    spec,
                    **HASHES,
                )
            )
            self.assertEqual(set(calls[0]), {"e2_code_hash", "e2_config_hash", "orchestrator_hash"})
            self.assertFalse(
                recovery.metadata_matches_compat(
                    frozen,
                    {},
                    Path("/tmp/artifact"),
                    spec,
                    e1_dependency_hash="wrong-e1",
                    e2_code_hash=HASHES["e2_code_hash"],
                    e2_config_hash=HASHES["e2_config_hash"],
                    orchestrator_hash=HASHES["orchestrator_hash"],
                )
            )
        finally:
            if original is None:
                delattr(frozen, "_E2_RECOVERY_V2_FROZEN_METADATA_MATCHES")
            else:
                frozen._E2_RECOVERY_V2_FROZEN_METADATA_MATCHES = original

    def test_reconcile_adopts_complete_and_archives_dead_partial_for_explicit_fresh_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "e2"
            store = frozen.StateStore(output / ".orchestrator", HASHES, memory_guards={"student": 5, "teacher": 5, "nc": 5})
            complete = frozen.teacher_specs(output, tasks=("vsfc_topic",))[0]
            partial = frozen.teacher_specs(output, tasks=("visfd",))[0]
            write_train_artifact(Path(complete.expected_dir), complete, complete=True)
            write_train_artifact(Path(partial.expected_dir), partial, complete=False)
            add_job(store, complete, 99999991)
            add_job(store, partial, 99999992)
            recovery.install_metadata_compat(frozen)
            with patch.object(frozen, "_pid_alive", return_value=False):
                report = recovery.reconcile_stale_jobs_v2(
                    frozen,
                    repo_root=root,
                    output_root=output,
                    hashes=HASHES,
                    memory_guards={"student": 5, "teacher": 5, "nc": 5},
                    allow_fresh_rerun=True,
                )
            self.assertEqual(len(report["completed_adopted"]), 1)
            self.assertEqual(len(report["interrupted_archived"]), 1)
            self.assertFalse(Path(partial.expected_dir).exists())
            archive = Path(report["interrupted_archived"][0]["archive_dir"])
            self.assertTrue((archive / "artifact" / "status.json").is_file())
            state = json.loads((output / ".orchestrator" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual([job["status"] for job in state["jobs"]], ["completed", "interrupted"])
            self.assertEqual(len(state["interrupted_attempts"]), 1)
            self.assertEqual(state["failed"], [])

    def test_live_child_blocks_before_any_archive_or_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifacts" / "e2"
            store = frozen.StateStore(output / ".orchestrator", HASHES, memory_guards={"student": 5, "teacher": 5, "nc": 5})
            spec = frozen.teacher_specs(output, tasks=("vsfc_topic",))[0]
            write_train_artifact(Path(spec.expected_dir), spec, complete=False)
            add_job(store, spec, os.getpid())
            recovery.install_metadata_compat(frozen)
            with self.assertRaises(recovery.RecoveryBlocked):
                recovery.reconcile_stale_jobs_v2(
                    frozen,
                    repo_root=root,
                    output_root=output,
                    hashes=HASHES,
                    memory_guards={"student": 5, "teacher": 5, "nc": 5},
                    allow_fresh_rerun=True,
                )
            self.assertTrue(Path(spec.expected_dir).is_dir())

    def test_cache_and_student_kd_nc_contracts_use_same_adapter_signature(self) -> None:
        recovery.install_metadata_compat(frozen)
        calls = []

        original = getattr(frozen, "_E2_RECOVERY_V2_FROZEN_METADATA_MATCHES")

        def probe(*args, **kwargs):
            calls.append(kwargs)
            return True

        frozen._E2_RECOVERY_V2_FROZEN_METADATA_MATCHES = probe
        try:
            root = Path("/tmp/e2-recovery-v2-contract")
            specs = [
                frozen.cache_specs(root)[0],
                frozen.stagea_kd_specs(root, tasks=("vsfc_topic",))[0],
                frozen.stagea_nc_specs(root, tasks=("visfd",))[0],
            ]
            for spec in specs:
                self.assertTrue(
                    recovery.metadata_matches_compat(
                        frozen,
                        {},
                        root,
                        spec,
                        **HASHES,
                    )
                )
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(set(call) == {"e2_code_hash", "e2_config_hash", "orchestrator_hash"} for call in calls))
        finally:
            frozen._E2_RECOVERY_V2_FROZEN_METADATA_MATCHES = original

    def test_durable_child_uses_new_session_and_completes(self) -> None:
        recovery.install_metadata_compat(frozen)
        with recovery.durable_children(frozen):
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(0.02)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.assertEqual(os.getsid(child.pid), child.pid)
            self.assertEqual(child.wait(timeout=2), 0)


if __name__ == "__main__":
    unittest.main()
