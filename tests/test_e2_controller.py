from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.continue_e2_after_pilots import (
    RELEASE,
    ControllerBlocked,
    canonical_artifact_path,
    expected_ce,
    expected_nc,
    expected_teacher,
    metadata_ok,
    validate_ce_delta,
    wait_pilot,
)
from scripts.run_e2_ce_replay_compat import main as ce_replay_compat_main


REQUIRED = (
    "metadata.json",
    "resolved_config.json",
    "epoch_curves.json",
    "best_checkpoint.pt",
    "best_validation_metrics.json",
    "validation_predictions.json",
)


def write_artifact(path: Path, *, mode: str, stage: str, method: str, lr: float, micro: int, score: float = 0.845) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": "E2-training-ablation-v1",
        "revision": "e2-20260908-r1",
        "stage": stage,
        "method": method,
        "mode": mode,
        "task_id": "vsfc_sentiment",
        "seed": 42,
        "learning_rate": lr,
        "coefficient": None if mode == "ce" else lr,
        "epochs": 30,
        "effective_batch_size": 32,
        "micro_batch_size": micro,
        "include_test": False,
        "smoke": False,
        "attention_implementation": "sdpa",
        "runtime": {"precision": "bf16"},
        "source_hashes": RELEASE,
        "status": "completed",
        "best_epoch": 4,
        "best_selection_f1": score,
    }
    for name in REQUIRED:
        target = path / name
        if name == "metadata.json":
            target.write_text(json.dumps(metadata), encoding="utf-8")
        elif name == "best_checkpoint.pt":
            target.write_bytes(b"checkpoint")
        else:
            target.write_text("{}", encoding="utf-8")


class ContinueE2ControllerTests(unittest.TestCase):
    def test_completed_teacher_and_ce_metadata_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = expected_teacher(root, 2e-5)
            ce = expected_ce(root)
            write_artifact(teacher, mode="teacher", stage="stage0_teacher", method="teacher", lr=2e-5, micro=8)
            write_artifact(ce, mode="ce", stage="stagea_ce_replay", method="ce_replay", lr=1e-5, micro=32)
            self.assertEqual(metadata_ok(teacher, stage="stage0_teacher", method="teacher", mode="teacher", lr=2e-5, micro=8, coefficient=2e-5)["status"], "completed")
            self.assertEqual(metadata_ok(ce, stage="stagea_ce_replay", method="ce_replay", mode="ce", lr=1e-5, micro=32, coefficient=None)["status"], "completed")

    def test_completed_nc_artifact_accepts_collision_safe_canonical_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = expected_nc(root, "vsfc_sentiment", 0.1)
            target = canonical.with_name("seed_42__rerun1")
            write_artifact(target, mode="nc", stage="stagea_nc", method="nc", lr=1e-5, micro=32)
            metadata_path = target / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["coefficient"] = 0.1
            metadata["hyperparameters"] = {"consistency_lambda": 0.1}
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.symlink_to(target.name)
            resolved = canonical_artifact_path(canonical, canonical, "Additional NC pilot")
            self.assertEqual(resolved, target.resolve())
            self.assertEqual(
                metadata_ok(
                    resolved,
                    stage="stagea_nc",
                    method="nc",
                    mode="nc",
                    lr=1e-5,
                    micro=32,
                    coefficient=0.1,
                )["status"],
                "completed",
            )

    def test_missing_or_failed_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = expected_teacher(Path(directory), 1e-5)
            write_artifact(path, mode="teacher", stage="stage0_teacher", method="teacher", lr=1e-5, micro=8)
            (path / "best_checkpoint.pt").unlink()
            with self.assertRaises(ControllerBlocked):
                metadata_ok(path, stage="stage0_teacher", method="teacher", mode="teacher", lr=1e-5, micro=8, coefficient=1e-5)
            write_artifact(path, mode="teacher", stage="stage0_teacher", method="teacher", lr=1e-5, micro=8)
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["status"] = "failed"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ControllerBlocked):
                metadata_ok(path, stage="stage0_teacher", method="teacher", mode="teacher", lr=1e-5, micro=8, coefficient=1e-5)

    def test_ce_delta_threshold_is_strictly_validation_only(self) -> None:
        self.assertAlmostEqual(validate_ce_delta(0.845, 0.8401), 0.0049)
        with self.assertRaises(ControllerBlocked):
            validate_ce_delta(0.8451, 0.84)

    def test_dead_pid_with_incomplete_artifact_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_42"
            with self.assertRaises(ControllerBlocked):
                wait_pilot({"pid": 99999999}, path, lambda _: (_ for _ in ()).throw(ControllerBlocked("incomplete")), 0.01, 0.001)

    def test_dead_pid_with_completed_artifact_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = expected_teacher(Path(directory), 1e-5)
            write_artifact(path, mode="teacher", stage="stage0_teacher", method="teacher", lr=1e-5, micro=8)
            metadata = wait_pilot(
                {"pid": 99999999},
                path,
                lambda value: metadata_ok(value, stage="stage0_teacher", method="teacher", mode="teacher", lr=1e-5, micro=8, coefficient=1e-5),
                0.0,
                0.0,
                completion_grace_seconds=0.0,
            )
            self.assertEqual(metadata["status"], "completed")

    def test_dead_pid_reports_validator_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_42"
            with self.assertRaisesRegex(ControllerBlocked, "validator: incomplete"):
                wait_pilot(
                    {"pid": 99999999},
                    path,
                    lambda _: (_ for _ in ()).throw(ControllerBlocked("incomplete")),
                    0.0,
                    0.0,
                    completion_grace_seconds=0.0,
                )

    def test_ce_compat_wrapper_runs_real_scheduler_branch_and_reuses_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "artifacts" / "e2"
            ce_path = output_root / "stageA" / "ce_replay" / "vsfc_sentiment" / "seed_42"
            write_artifact(ce_path, mode="ce", stage="stagea_ce_replay", method="ce_replay", lr=1e-5, micro=32)
            gate = root / "smoke_gate.json"
            gate.write_text(json.dumps({"status": "passed", "test_access": False}), encoding="utf-8")
            code = ce_replay_compat_main(
                [
                    "--repo-root",
                    str(Path(__file__).resolve().parents[1]),
                    "--output-root",
                    str(output_root),
                    "--root-smoke-gate",
                    str(gate),
                    "--devices",
                    "cuda:0",
                    "--poll-seconds",
                    "0",
                    "--student-memory-guard-gib",
                    "5",
                    "--teacher-memory-guard-gib",
                    "5",
                    "--nc-memory-guard-gib",
                    "5",
                    "--execute",
                ]
            )
            self.assertEqual(code, 0)
            approved = json.loads((output_root / ".orchestrator" / "stagea_ce_replay_approved.json").read_text(encoding="utf-8"))
            self.assertEqual(approved["approved_count"], 1)
            state = json.loads((output_root / ".orchestrator" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(state["reused"]), 1)
            self.assertEqual(state["jobs"], [])


if __name__ == "__main__":
    unittest.main()
