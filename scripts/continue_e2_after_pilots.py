"""Gate the two E2 pilot runs, then hand control to the frozen scheduler."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "E2-training-ablation-v1"
REVISION = "e2-20260908-r1"
RELEASE = {
    "e1_dependency": "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873",
    "e2": "efb81ac232de2f0ae4287688ad0125a1d84fb1e689e2b1566ef26d8d6e2b60b5",
    "e2_config": "45e6c36196fe95ae0eb9193e9197224bdf2980515c15ddf01998267b668b61b8",
    "orchestrator": "f626112d163a4c9ed476d9811bb74447b337d23bd6e87d9e105ab8e0a66807b8",
}
REQUIRED = ("metadata.json", "resolved_config.json", "epoch_curves.json", "best_checkpoint.pt", "best_validation_metrics.json", "validation_predictions.json")
PILOT_COMPLETION_GRACE_SECONDS = 30.0
class ControllerBlocked(RuntimeError):
    pass
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
def lr_slug(value: float) -> str:
    return f"{float(value):.0e}"
def expected_teacher(root: Path, lr: float) -> Path:
    return root / "stage0_teacher" / "vsfc_sentiment" / lr_slug(lr) / "seed_42"
def expected_ce(root: Path) -> Path:
    return root / "stageA" / "ce_replay" / "vsfc_sentiment" / "seed_42"
def expected_nc(root: Path, task: str, coefficient: float, seed: int = 42) -> Path:
    return root / "stageA" / "nc" / task / f"coef_{coefficient:g}".replace(".", "p") / f"seed_{seed}"
def release_hashes(repo_root: Path) -> None:
    from scripts.execute_e2_validation import source_hashes
    actual = source_hashes(repo_root)
    expected = {
        "e1_dependency_hash": RELEASE["e1_dependency"],
        "e2_code_hash": RELEASE["e2"],
        "e2_config_hash": RELEASE["e2_config"],
        "orchestrator_hash": RELEASE["orchestrator"],
    }
    if actual != expected:
        raise ControllerBlocked(f"Frozen E2 release drifted: expected {RELEASE}, found {actual}")
def check_gate(path: Path) -> None:
    gate = read_json(path)
    if not isinstance(gate, dict) or gate.get("status") != "passed" or gate.get("test_access") is not False:
        raise ControllerBlocked(f"Smoke gate is not passed: {path}")
    gate_keys = {"e1_dependency": "e1_dependency_hash", "e2": "e2_code_hash", "e2_config": "e2_config_hash", "orchestrator": "orchestrator_hash"}
    for key, field in gate_keys.items():
        if gate.get(field) != RELEASE[key]:
            raise ControllerBlocked(f"Smoke gate hash mismatch for {key}")
def metadata_ok(path: Path, *, stage: str, method: str, mode: str, lr: float, micro: int, coefficient: float | None) -> dict[str, Any]:
    # The canonical NC directory may be a symlink to a collision-safe rerun
    # directory (seed_42__rerun1).  Canonical identity is checked by the
    # caller after resolving the manifest path; metadata validation must not
    # reject a valid target solely because of its basename.
    if not path.is_dir() or any((path / name).is_file() is False for name in REQUIRED):
        raise ControllerBlocked(f"Pilot artifact is incomplete: {path}")
    metadata = read_json(path / "metadata.json")
    if not isinstance(metadata, dict):
        raise ControllerBlocked(f"Malformed pilot metadata: {path}")
    expected = {"protocol": PROTOCOL, "revision": REVISION, "stage": stage, "method": method, "mode": mode, "task_id": "vsfc_sentiment", "seed": 42, "learning_rate": lr, "coefficient": coefficient, "epochs": 30, "effective_batch_size": 32, "micro_batch_size": micro, "include_test": False, "smoke": False, "attention_implementation": "sdpa"}
    for key, value in expected.items():
        actual = metadata.get(key)
        if isinstance(value, float):
            if not isinstance(actual, (int, float)) or abs(float(actual) - value) > 1e-12:
                raise ControllerBlocked(f"Pilot metadata mismatch {key} at {path}: {actual!r}")
        elif actual != value:
            raise ControllerBlocked(f"Pilot metadata mismatch {key} at {path}: {actual!r}")
    runtime = metadata.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("precision") != "bf16":
        raise ControllerBlocked(f"Pilot precision is not bf16: {path}")
    hashes = metadata.get("source_hashes")
    if not isinstance(hashes, dict) or any(hashes.get(key) != value for key, value in RELEASE.items()):
        raise ControllerBlocked(f"Pilot source hashes are not the frozen release: {path}")
    if metadata.get("status") != "completed" or not isinstance(metadata.get("best_epoch"), int) or not isinstance(metadata.get("best_selection_f1"), (int, float)):
        raise ControllerBlocked(f"Pilot is not a completed validation artifact: {path}")
    if any((path / name).exists() for name in ("test_metrics.json", "test_predictions.json", "test_evaluation_metadata.json")):
        raise ControllerBlocked(f"Pilot contains test artifacts: {path}")
    return metadata
def wait_pilot(
    entry: Mapping[str, Any],
    path: Path,
    validator: Any,
    timeout: float,
    poll: float,
    completion_grace_seconds: float = PILOT_COMPLETION_GRACE_SECONDS,
) -> dict[str, Any]:
    started = time.monotonic()
    pid = int(entry["pid"]) if entry.get("pid") is not None else None
    pid_ended_at: float | None = None
    last_error: Exception | None = None
    while True:
        metadata = None
        try:
            metadata = validator(path)
            last_error = None
        except (ControllerBlocked, OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        alive = pid_alive(pid)
        if metadata is not None and not alive:
            return metadata
        if pid is None and metadata is None:
            detail = f"; validator: {last_error}" if last_error else ""
            raise ControllerBlocked(f"Pilot has no live PID and no completed artifact: {path}{detail}")
        if not alive and pid is not None:
            if pid_ended_at is None:
                pid_ended_at = time.monotonic()
            if time.monotonic() - pid_ended_at >= max(0.0, completion_grace_seconds):
                detail = f"; validator: {last_error}" if last_error else ""
                raise ControllerBlocked(f"Pilot PID ended before a valid completed artifact: {path}{detail}")
        else:
            pid_ended_at = None
        if timeout > 0 and time.monotonic() - started >= timeout:
            raise ControllerBlocked(f"Timed out waiting for pilot: {path}")
        time.sleep(poll)
def e1_reference(repo_root: Path, manifest_path: Path | None = None) -> float:
    manifest = read_json(manifest_path or (repo_root / "artifacts" / "e1" / ".orchestrator" / "approved_records.json"))
    rows = manifest.get("approved", {})
    if isinstance(rows, dict):
        rows = sum(rows.values(), [])
    matches = [row for row in rows if row.get("task_id") == "vsfc_sentiment" and row.get("seed") == 42 and abs(float(row.get("learning_rate", -1)) - 1e-5) <= 1e-12]
    if len(matches) != 1:
        raise ControllerBlocked(f"Expected exactly one E1 seed-42/LR-1e-5 reference, found {len(matches)}")
    row = matches[0]
    path = Path(row["artifact_dir"])
    metadata = metadata_ok_e1(path, approved_row=row)
    return float(metadata["best_selection_f1"])
def validate_ce_delta(ce_score: float, e1_score: float, tolerance: float = 0.005) -> float:
    delta = abs(float(ce_score) - float(e1_score))
    if delta > tolerance:
        raise ControllerBlocked(f"CE replay differs from E1 seed42/LR1e-5 by {delta:.6f} (> {tolerance:.3f})")
    return delta
def metadata_ok_e1(path: Path, *, approved_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_dir() or any(not (path / name).is_file() for name in REQUIRED):
        raise ControllerBlocked(f"E1 reference artifact is incomplete: {path}")
    metadata = read_json(path / "metadata.json")
    if metadata.get("source_hash") != RELEASE["e1_dependency"] or metadata.get("status") != "completed" or metadata.get("include_test") is not False or metadata.get("smoke") is True:
        raise ControllerBlocked(f"E1 reference metadata is not the approved supervised record: {path}")
    if approved_row is not None:
        for key in ("artifact_dir", "task_id", "seed", "learning_rate"):
            if key not in approved_row:
                raise ControllerBlocked(f"E1 approved row is missing {key}: {path}")
        if Path(str(approved_row["artifact_dir"])).resolve() != path.resolve():
            raise ControllerBlocked(f"E1 approved row points to a different artifact: {path}")
        if metadata.get("task_id") != approved_row["task_id"] or metadata.get("seed") != approved_row["seed"]:
            raise ControllerBlocked(f"E1 approved row identity mismatch: {path}")
        if not isinstance(metadata.get("learning_rate"), (int, float)) or abs(float(metadata["learning_rate"]) - float(approved_row["learning_rate"])) > 1e-12:
            raise ControllerBlocked(f"E1 approved row learning-rate mismatch: {path}")
    # The E1 test evaluation was intentionally performed later in the same
    # artifact directory.  This gate reads only the validation metadata above;
    # test files are not opened or used here.
    return metadata


def canonical_artifact_path(path: Path, expected: Path, label: str) -> Path:
    """Resolve an artifact (including a canonical symlink) and verify identity."""

    try:
        actual = path.resolve(strict=True)
        target = expected.resolve(strict=True)
    except OSError as exc:
        raise ControllerBlocked(f"{label} artifact path is unavailable: {path}: {exc}") from exc
    if actual != target:
        raise ControllerBlocked(f"{label} artifact is not at its canonical E2 path: {actual}")
    return actual


def scheduler_command(args: argparse.Namespace, label: str) -> tuple[list[str], str | None]:
    """Build a frozen-scheduler command, using the CE CLI compatibility shim."""

    common = [
        "--repo-root",
        str(args.repo_root),
        "--output-root",
        str(args.output_root),
        "--root-smoke-gate",
        str(args.root_smoke_gate),
        "--devices",
        *args.devices,
        "--poll-seconds",
        str(args.poll_seconds),
        "--student-memory-guard-gib",
        "5",
        "--teacher-memory-guard-gib",
        "5",
        "--nc-memory-guard-gib",
        "5",
        "--execute",
    ]
    if label == "stagea_ce_replay":
        wrapper = args.repo_root / "scripts" / "run_e2_ce_replay_compat.py"
        if not wrapper.is_file():
            raise ControllerBlocked(f"CE compatibility wrapper is missing: {wrapper}")
        return [sys.executable, str(wrapper), *common], sha256_file(wrapper)
    return [
        sys.executable,
        str(args.repo_root / "scripts" / "execute_e2_validation.py"),
        "--stage",
        label,
        *common,
    ], None


def run_scheduler(args: argparse.Namespace, state: dict[str, Any], label: str) -> None:
    controller_dir = args.output_root / ".controller"
    log_path = controller_dir / f"scheduler_{label}.log"
    command, wrapper_hash = scheduler_command(args, label)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=args.repo_root,
            env={**os.environ, "PYTHONPATH": os.pathsep.join((str(args.repo_root / "src"), str(args.repo_root)))},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        scheduler_state = {"label": label, "pid": process.pid, "command": command, "log": str(log_path), "status": "running", "started_at": now()}
        if wrapper_hash is not None:
            scheduler_state["compat_wrapper"] = str(args.repo_root / "scripts" / "run_e2_ce_replay_compat.py")
            scheduler_state["compat_wrapper_sha256"] = wrapper_hash
        state["scheduler"] = scheduler_state
        write_json(controller_dir / "state.json", state)
        code = process.wait()
    state["scheduler"].update({"status": "completed" if code == 0 else "failed", "returncode": code, "ended_at": now()})
    write_json(controller_dir / "state.json", state)
    if code != 0:
        raise ControllerBlocked(f"Frozen scheduler failed at stage {label}; see {log_path}")
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "e2")
    parser.add_argument("--root-smoke-gate", type=Path, default=ROOT / "artifacts" / "e2" / "preflight" / "smoke_gate.json")
    parser.add_argument("--e1-reference", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.output_root = args.output_root.resolve()
    args.pilot_manifest = args.pilot_manifest.resolve()
    args.root_smoke_gate = args.root_smoke_gate.resolve()
    controller_dir = args.output_root / ".controller"
    controller_hash = sha256_file(Path(__file__).resolve())
    state_path = controller_dir / "state.json"
    manifest_hash = sha256_file(args.pilot_manifest) if args.pilot_manifest.is_file() else ""
    if state_path.is_file():
        previous = read_json(state_path)
        if previous.get("status") == "completed":
            return 0
        if previous.get("status") in {"running", "blocked"}:
            previous_pid = previous.get("controller_pid")
            scheduler_pid = (previous.get("scheduler") or {}).get("pid")
            live_previous = pid_alive(int(previous_pid)) if previous_pid else False
            live_scheduler = pid_alive(int(scheduler_pid)) if scheduler_pid else False
            if live_previous or live_scheduler:
                raise ControllerBlocked("An earlier pilot controller or scheduler is still running")
            raise ControllerBlocked("An earlier controller stopped; refusing an automatic rerun")
    state = {"protocol": "E2-pilot-controller-v1", "revision": REVISION, "controller_pid": os.getpid(), "controller_hash": controller_hash, "manifest": str(args.pilot_manifest), "manifest_sha256": manifest_hash, "started_at": now(), "status": "running"}
    write_json(state_path, state)
    try:
        if not args.pilot_manifest.is_file():
            raise ControllerBlocked(f"Pilot manifest is missing: {args.pilot_manifest}")
        release_hashes(args.repo_root)
        check_gate(args.root_smoke_gate)
        manifest = read_json(args.pilot_manifest)
        if manifest.get("protocol") != "E2-pilot-manifest-v1":
            raise ControllerBlocked("Pilot manifest protocol mismatch")
        teachers = manifest.get("teacher_pilots")
        if not isinstance(teachers, list) or {float(row.get("learning_rate")) for row in teachers} != {1e-5, 2e-5, 5e-5} or len(teachers) != 3:
            raise ControllerBlocked("Pilot manifest must contain exactly the three VSFC teacher LRs")
        for row in teachers:
            path = Path(row["artifact_dir"])
            lr = float(row["learning_rate"])
            path = canonical_artifact_path(path, expected_teacher(args.output_root, lr), "Teacher pilot")
            wait_pilot(row, path, lambda p, lr=lr: metadata_ok(p, stage="stage0_teacher", method="teacher", mode="teacher", lr=lr, micro=8, coefficient=lr), args.wait_timeout_seconds, args.poll_seconds)
        ce = manifest.get("ce_replay")
        if not isinstance(ce, dict):
            raise ControllerBlocked("Pilot manifest has no ce_replay entry")
        ce_path = canonical_artifact_path(Path(ce["artifact_dir"]), expected_ce(args.output_root), "CE replay")
        ce_metadata = wait_pilot(ce, ce_path, lambda p: metadata_ok(p, stage="stagea_ce_replay", method="ce_replay", mode="ce", lr=1e-5, micro=32, coefficient=None), args.wait_timeout_seconds, args.poll_seconds)
        nc_pilots = manifest.get("additional_nc_pilots", manifest.get("additional_pilots", []))
        if not isinstance(nc_pilots, list):
            raise ControllerBlocked("additional_nc_pilots must be a list")
        seen_nc: set[tuple[str, float, float, int]] = set()
        nc_results = []
        for row in nc_pilots:
            task = str(row.get("task_id", "vsfc_sentiment"))
            lr = float(row.get("learning_rate", 1e-5))
            coefficient = float(row["coefficient"])
            seed = int(row.get("seed", 42))
            identity = (task, lr, coefficient, seed)
            if identity in seen_nc:
                raise ControllerBlocked(f"Duplicate additional NC pilot: {identity}")
            seen_nc.add(identity)
            if seed != 42:
                raise ControllerBlocked(f"Additional NC pilot must use seed 42: {identity}")
            path = canonical_artifact_path(Path(row["artifact_dir"]), expected_nc(args.output_root, task, coefficient, seed), "Additional NC pilot")
            nc_results.append(wait_pilot(row, path, lambda p, task=task, lr=lr, coefficient=coefficient: metadata_ok(p, stage="stagea_nc", method="nc", mode="nc", lr=lr, micro=32, coefficient=coefficient), args.wait_timeout_seconds, args.poll_seconds))
        ref = e1_reference(args.repo_root, args.e1_reference.resolve() if args.e1_reference else None)
        delta = validate_ce_delta(float(ce_metadata["best_selection_f1"]), ref)
        state["pilot_validation"] = {"teacher_lrs": [1e-5, 2e-5, 5e-5], "ce_best_selection_f1": ce_metadata["best_selection_f1"], "additional_nc_count": len(nc_results), "additional_nc_scores": [row["best_selection_f1"] for row in nc_results], "e1_reference_f1": ref, "absolute_delta": delta, "validated_at": now()}
        write_json(controller_dir / "pilot_validation.json", state["pilot_validation"])
        write_json(controller_dir / "state.json", state)
        run_scheduler(args, state, "stagea_ce_replay")
        run_scheduler(args, state, "all")
        state["status"] = "completed"
        state["ended_at"] = now()
        write_json(controller_dir / "state.json", state)
        return 0
    except (ControllerBlocked, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        state["status"] = "blocked"
        state["error"] = str(exc)
        state["ended_at"] = now()
        write_json(controller_dir / "state.json", state)
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
if __name__ == "__main__":
    raise SystemExit(main())
