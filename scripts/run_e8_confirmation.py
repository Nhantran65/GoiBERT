"""Confirm the frozen E8 NC+LLRD recipe on ViSFD and UIT-ABSA Restaurant.

This module is intentionally an adapter around ``run_e8_nc_llrd``.  The
frozen E8 module remains the only training implementation: this file only
sets the process-local seed, isolates orchestration paths, coordinates the
four confirmation workers, locks validation-selected checkpoints, and runs
the explicit test gate with the frozen E1 evaluator primitives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# This must be set before importing torch through either E8 or E1.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer, __version__ as TRANSFORMERS_VERSION  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert.modeling import build_classifier  # noqa: E402

import run_e8_nc_llrd as frozen_e8  # noqa: E402


CONFIRM_ROOT = ROOT / "artifacts" / "e8" / "confirmation"
STATE_DIR = CONFIRM_ROOT / "state"
LOG_DIR = CONFIRM_ROOT / "logs"
MANIFEST_PATH = CONFIRM_ROOT / "manifest.json"
SUMMARY_PATH = CONFIRM_ROOT / "summary.json"
SUMMARY_CSV_PATH = CONFIRM_ROOT / "summary.csv"
TEST_ACCESS_LOG_PATH = CONFIRM_ROOT / "test_access_log.json"
RESULT_DOC = ROOT / "docs" / "E8_CONFIRMATION_RESULT.md"
BACKBONE = ROOT / "models" / "goiturn-s-gd"

TASKS = ("visfd", "absa_restaurant")
SEEDS = (43, 44)
ALL_RUNS = tuple((task, seed) for seed in SEEDS for task in TASKS)
EXPECTED_TEST_COUNTS = {"visfd": 2224, "absa_restaurant": 1938}
METRIC_NAMES = {
    "visfd": ("pair_micro_f1", "detection_micro_f1"),
    "absa_restaurant": ("pair_micro_f1", "detection_micro_f1"),
}
TASK_LABELS = {"visfd": "ViSFD", "absa_restaurant": "UIT-ABSA Restaurant"}
PAPER_TARGETS = {
    "visfd": {"pair_micro_f1": 84.24, "detection_micro_f1": 89.14},
    "absa_restaurant": {"pair_micro_f1": 74.89, "detection_micro_f1": 88.01},
}
ORIGINAL_E8_SOURCE_HASH = "089726d5d0294d2835ff6b4212bf30e6e47f246e84f50213536c0cbe4ccfd437"
CONFIRMATION_PROTOCOL = "E8-confirmation-NC-LLRD-v1"
TEST_PROTOCOL = "E8-confirmation-test-gate-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    return frozen_e8._json_default(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_hash() -> str:
    return _sha256(Path(__file__).resolve())


def _is_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _artifact_dir(task_id: str, seed: int) -> Path:
    return ROOT / "artifacts" / "e8" / task_id / f"seed_{seed}"


def _isolated_roots(seed: int) -> tuple[Path, Path, Path]:
    root = CONFIRM_ROOT / f"seed_{seed}"
    return root / ".orchestrator", root / ".orchestrator" / "logs", root / ".orchestrator" / "workers"


def _configure_frozen(seed: int) -> None:
    """Override only imported module globals for this process."""

    orchestrator, logs, workers = _isolated_roots(seed)
    frozen_e8.SEED = int(seed)
    frozen_e8.ARTIFACT_ROOT = ROOT / "artifacts" / "e8"
    frozen_e8.ORCH_ROOT = orchestrator
    frozen_e8.LOG_ROOT = logs
    frozen_e8.WORKER_ROOT = workers
    for path in (orchestrator, logs, workers):
        path.mkdir(parents=True, exist_ok=True)


def _gpu_snapshot() -> dict[str, dict[str, int]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    snapshot: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3:
            snapshot[fields[0]] = {"free_mib": int(float(fields[1])), "used_mib": int(float(fields[2]))}
    if not snapshot:
        raise RuntimeError(f"nvidia-smi returned no GPU rows: {output!r}")
    return snapshot


def _eligible_devices() -> tuple[list[str], dict[str, dict[str, int]]]:
    snapshot = _gpu_snapshot()
    devices = [
        f"cuda:{index}"
        for index in frozen_e8.PREFERRED_GPU_INDICES
        if index in snapshot and snapshot[index]["free_mib"] >= frozen_e8.GPU_GUARD_MIB
    ]
    if not devices:
        raise RuntimeError(f"No preferred GPU has {frozen_e8.GPU_GUARD_MIB} MiB free: {snapshot}")
    return devices, snapshot


def _assert_no_active_or_duplicate() -> None:
    """Fail closed before launching any new confirmation process."""

    state_path = CONFIRM_ROOT / "controller_state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("status") == "running" and state.get("controller_pid") != os.getpid() and _is_alive(state.get("controller_pid")):
            raise RuntimeError(f"Confirmation controller already active: PID {state.get('controller_pid')}")
    for task_id, seed in ALL_RUNS:
        artifact = _artifact_dir(task_id, seed)
        if artifact.exists():
            raise RuntimeError(f"Confirmation artifact path already exists; refusing duplicate: {artifact}")
        _, _, workers = _isolated_roots(seed)
        worker_path = workers / f"{task_id}.json"
        if worker_path.is_file():
            worker = _read_json(worker_path)
            if worker.get("status") == "running" and _is_alive(worker.get("pid")):
                raise RuntimeError(f"Confirmation worker already active: {task_id}, seed {seed}")


def _static_preflight() -> dict[str, Any]:
    source = frozen_e8._source_hashes()
    e8_hash = frozen_e8.e8_source_hash()
    if e8_hash != ORIGINAL_E8_SOURCE_HASH:
        raise RuntimeError(f"Frozen E8 source hash changed: {e8_hash}")
    checks = {
        "frozen_e8_source_hash": e8_hash == ORIGINAL_E8_SOURCE_HASH,
        "frozen_e1_dependency": source["e1_dependency"] == frozen_e8.E1_SOURCE_HASH,
        "frozen_e2_code": source["e2"] == frozen_e8.E2_CODE_HASH,
        "frozen_e2_config": source["e2_config"] == frozen_e8.E2_CONFIG_HASH,
        "tasks_locked": TASKS == ("visfd", "absa_restaurant"),
        "seeds_locked": SEEDS == (43, 44),
        "recipe_locked": {
            "epochs": frozen_e8.EPOCHS,
            "effective_batch_size": frozen_e8.EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": frozen_e8.MICRO_BATCH_SIZE,
            "max_length": frozen_e8.MAX_LENGTH,
            "gamma": frozen_e8.GAMMA,
            "attention_implementation": "sdpa",
            "precision": "bf16",
        },
        # Existing seed-42 E8 artifacts are the frozen read-only baseline;
        # only a seed-42 directory inside this confirmation namespace would
        # indicate an accidental duplicate confirmation run.
        "no_seed42_confirmation_dirs": not (CONFIRM_ROOT / "seed_42").exists(),
    }
    if not all(value if isinstance(value, bool) else True for value in checks.values()):
        raise RuntimeError(f"Confirmation static preflight failed: {checks}")
    devices, snapshot = _eligible_devices()
    payload = {
        "protocol": CONFIRMATION_PROTOCOL,
        "status": "passed",
        "created_at": _now(),
        "adapter_hash": _adapter_hash(),
        "original_e8_source_hash": e8_hash,
        "e2_source_hashes": source,
        "eligible_devices": devices,
        "gpu_snapshot": snapshot,
        "checks": checks,
        "worker_count": len(ALL_RUNS),
        "test_counts": EXPECTED_TEST_COUNTS,
    }
    _write_json(CONFIRM_ROOT / "preflight.json", payload)
    return payload


def _annotate_confirmation(task_id: str, seed: int, device_name: str, artifact: Path) -> None:
    """Add confirmation provenance after the frozen E8 worker completes."""

    fields = {
        "confirmation_protocol": CONFIRMATION_PROTOCOL,
        "confirmation_seed": seed,
        "confirmation_adapter": str(Path(__file__).resolve().relative_to(ROOT)),
        "confirmation_adapter_hash": _adapter_hash(),
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
        "seed_override": {
            "frozen_module": "run_e8_nc_llrd.SEED",
            "value": seed,
            "process_local": True,
            "source_file_unchanged": True,
        },
        "orchestration_path_override": {
            "orchestrator_root": str(_isolated_roots(seed)[0]),
            "worker_root": str(_isolated_roots(seed)[2]),
            "log_root": str(_isolated_roots(seed)[1]),
            "artifact_root": str((ROOT / "artifacts" / "e8").resolve()),
        },
        "runtime_override": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED"),
            "python": sys.executable,
            "host_cuda_required": True,
            "cpu_fallback": False,
        },
        "confirmation_task": task_id,
        "confirmation_device": device_name,
        "confirmation_artifact": str(artifact.resolve()),
    }
    metadata_path = artifact / "metadata.json"
    resolved_path = artifact / "resolved_config.json"
    metadata = _read_json(metadata_path)
    resolved = _read_json(resolved_path)
    metadata.update(fields)
    resolved.update(fields)
    _write_json(metadata_path, metadata)
    _write_json(resolved_path, resolved)
    checkpoint = artifact / "best_checkpoint.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Completed E8 checkpoint missing model state: {checkpoint}")
    checkpoint_metadata = dict(state.get("metadata", {}))
    checkpoint_metadata.update(fields)
    state["metadata"] = checkpoint_metadata
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(checkpoint)
    metadata = _read_json(metadata_path)
    metadata["checkpoint_sha256_actual"] = _sha256(checkpoint)
    _write_json(metadata_path, metadata)


def run_worker(task_id: str, seed: int, device_name: str, output_dir: Path) -> dict[str, Any]:
    if task_id not in TASKS or seed not in SEEDS:
        raise ValueError(f"Confirmation task/seed not locked: {task_id}, {seed}")
    if output_dir != _artifact_dir(task_id, seed):
        raise ValueError(f"Output path must be seed-isolated confirmation path: {output_dir}")
    _configure_frozen(seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise RuntimeError(f"Refusing duplicate confirmation output: {output_dir}")
    # run_worker is the exact frozen E2 NC + E7 LLRD recipe.  It sees the
    # overridden SEED and roots only within this process.
    result = frozen_e8.run_worker(task_id, device_name, output_dir)
    _annotate_confirmation(task_id, seed, device_name, output_dir)
    status_path = frozen_e8.WORKER_ROOT / f"{task_id}.json"
    status = _read_json(status_path)
    status.update({
        "confirmation_protocol": CONFIRMATION_PROTOCOL,
        "confirmation_seed": seed,
        "confirmation_adapter_hash": _adapter_hash(),
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
        "artifact_dir": str(output_dir.resolve()),
    })
    _write_json(status_path, status)
    return result


def _worker_command(task_id: str, seed: int, device_name: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--task-id",
        task_id,
        "--seed",
        str(seed),
        "--device",
        device_name,
    ]


def _launch_detached_controller() -> dict[str, Any]:
    CONFIRM_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = CONFIRM_ROOT / "controller_state.json"
    _assert_no_active_or_duplicate()
    command = [sys.executable, str(Path(__file__).resolve()), "--controller"]
    log_path = LOG_DIR / "controller.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    log_handle.write(json.dumps({"event": "detached_controller_launch", "time": _now(), "command": command}) + "\n")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    log_handle.close()
    payload = {
        "protocol": CONFIRMATION_PROTOCOL,
        "status": "running",
        "controller_pid": process.pid,
        "started_at": _now(),
        "command": command,
        "log_path": str(log_path.resolve()),
        "adapter_hash": _adapter_hash(),
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
    }
    _write_json(state_path, payload)
    return payload


def _controller_state() -> Path:
    return CONFIRM_ROOT / "controller_state.json"


def _write_controller_state(payload: Mapping[str, Any]) -> None:
    _write_json(_controller_state(), dict(payload))


def _controller() -> int:
    CONFIRM_ROOT.mkdir(parents=True, exist_ok=True)
    _assert_no_active_or_duplicate()
    preflight = _static_preflight()
    devices, snapshot = _eligible_devices()
    # Run the known frozen E8 optimizer/loss preflight once on this seed's
    # isolated root before launching any full trial.
    _configure_frozen(SEEDS[0])
    known_preflight = frozen_e8.run_preflight(devices[0])
    if known_preflight.get("status") != "passed":
        raise RuntimeError("Frozen E8 optimizer/loss preflight failed; no workers launched")
    jobs: dict[str, dict[str, Any]] = {
        f"{task}__seed{seed}": {
            "task_id": task,
            "seed": seed,
            "status": "queued",
            "artifact_dir": str(_artifact_dir(task, seed).resolve()),
        }
        for task, seed in ALL_RUNS
    }
    state: dict[str, Any] = {
        "protocol": CONFIRMATION_PROTOCOL,
        "status": "running",
        "controller_pid": os.getpid(),
        "started_at": _now(),
        "adapter_hash": _adapter_hash(),
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
        "preflight": str((CONFIRM_ROOT / "preflight.json").resolve()),
        "known_frozen_preflight": str((frozen_e8.ORCH_ROOT / "preflight.json").resolve()),
        "gpu_snapshot_at_launch": snapshot,
        "eligible_devices": devices,
        "jobs": jobs,
    }
    _write_controller_state(state)
    controller_log = (LOG_DIR / "controller.log").open("a", encoding="utf-8", buffering=1)
    running: dict[str, tuple[subprocess.Popen[str], str, Any]] = {}
    queue = list(ALL_RUNS)
    free_devices = list(devices)
    try:
        while queue or running:
            while queue and free_devices:
                task_id, seed = queue.pop(0)
                key = f"{task_id}__seed{seed}"
                device = free_devices.pop(0)
                log_path = LOG_DIR / f"{task_id}__seed{seed}.log"
                command = _worker_command(task_id, seed, device)
                handle = log_path.open("a", encoding="utf-8", buffering=1)
                handle.write(json.dumps({"event": "launch", "time": _now(), "command": command}) + "\n")
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    },
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                handle.close()
                running[key] = (process, device, (task_id, seed))
                jobs[key].update({
                    "status": "running",
                    "device": device,
                    "pid": process.pid,
                    "log_path": str(log_path.resolve()),
                    "command": command,
                    "launched_at": _now(),
                })
                controller_log.write(json.dumps({"event": "launch", "key": key, "pid": process.pid, "device": device}) + "\n")
                _write_controller_state(state)
            finished: list[str] = []
            for key, (process, device, pair) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                task_id, seed = pair
                jobs[key].update({
                    "status": "completed" if return_code == 0 else "failed",
                    "returncode": return_code,
                    "finished_at": _now(),
                })
                controller_log.write(json.dumps({"event": "finish", "key": key, "returncode": return_code}) + "\n")
                finished.append(key)
                free_devices.append(device)
            for key in finished:
                running.pop(key)
            _write_controller_state(state)
            failed = [key for key, job in jobs.items() if job.get("status") == "failed"]
            if failed:
                state.update({"status": "failed", "finished_at": _now(), "failed_jobs": failed})
                _write_controller_state(state)
                raise RuntimeError(f"Confirmation workers failed: {failed}")
            if running:
                time.sleep(15)
    except Exception:
        state.update({"status": "failed", "finished_at": _now(), "traceback": traceback.format_exc()})
        _write_controller_state(state)
        raise
    finally:
        controller_log.close()
    state.update({"status": "training_completed", "finished_at": _now()})
    _write_controller_state(state)
    # The test gate is deliberately a separate controller phase and starts
    # only after all four training processes have completed.
    build_manifest()
    run_tests()
    state.update({"status": "completed", "completed_at": _now(), "manifest": str(MANIFEST_PATH.resolve()), "summary": str(SUMMARY_PATH.resolve())})
    _write_controller_state(state)
    return 0


def _split_snapshot(split: Any) -> dict[str, Any]:
    ids = list(e1_train.split_ids(split))
    texts = list(e1_train.split_texts(split))
    labels = list(e1_train.split_labels(split))
    rows = [{"id": i, "text": t, "label": y} for i, t, y in zip(ids, texts, labels)]
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {
        "count": len(ids),
        "ids_sha256": hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "labels_sha256": hashlib.sha256(json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "rows_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _manifest_row(task_id: str, seed: int) -> dict[str, Any]:
    artifact = _artifact_dir(task_id, seed)
    paths = {
        "checkpoint": artifact / "best_checkpoint.pt",
        "metadata": artifact / "metadata.json",
        "resolved_config": artifact / "resolved_config.json",
        "validation_predictions": artifact / "validation_predictions.json",
        "best_validation_metrics": artifact / "best_validation_metrics.json",
        "status": artifact / "status.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"Incomplete confirmation artifact for {task_id} seed {seed}: {artifact}")
    metadata = _read_json(paths["metadata"])
    status = _read_json(paths["status"])
    expected = {
        "task_id": task_id,
        "seed": seed,
        "status": "completed",
        "method": "nc_plus_llrd",
        "mode": "nc",
        "coefficient": frozen_e8.NC_LAMBDAS[task_id],
        "epochs": 30,
        "best_epoch": metadata.get("best_epoch"),
        "max_length": 256,
        "micro_batch_size": 32,
        "effective_batch_size": 32,
        "attention_implementation": "sdpa",
        "include_test": False,
        "confirmation_seed": seed,
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
    }
    if status.get("status") != "completed":
        raise RuntimeError(f"Worker status is not completed: {task_id}, seed {seed}")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Confirmation metadata mismatch {task_id}, seed {seed}: {key}={metadata.get(key)!r}, expected {value!r}")
    if metadata.get("runtime", {}).get("precision") != "bf16":
        raise RuntimeError(f"Confirmation precision mismatch: {task_id}, seed {seed}")
    validation_rows = _read_json(paths["validation_predictions"]).get("rows")
    if not isinstance(validation_rows, list) or not validation_rows:
        raise RuntimeError(f"Missing validation predictions: {task_id}, seed {seed}")
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
    validation = e1_train.split_field(bundle, "validation")
    if validation is None:
        raise RuntimeError(f"Missing validation split: {task_id}")
    expected_ids = [str(value) for value in e1_train.split_ids(validation)]
    actual_ids = [str(row.get("id")) for row in validation_rows]
    if actual_ids != expected_ids:
        raise RuntimeError(f"Validation prediction IDs/order mismatch: {task_id}, seed {seed}")
    model_state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(model_state, Mapping) or "model" not in model_state:
        raise RuntimeError(f"Checkpoint model state missing: {paths['checkpoint']}")
    return {
        "task_id": task_id,
        "task": TASK_LABELS[task_id],
        "seed": seed,
        "method": metadata["method"],
        "mode": metadata["mode"],
        "coefficient": metadata["coefficient"],
        "learning_rate": metadata["learning_rate"],
        "best_epoch": metadata["best_epoch"],
        "best_selection_f1": metadata["best_selection_f1"],
        "artifact_dir": str(artifact.resolve()),
        "checkpoint_path": str(paths["checkpoint"].resolve()),
        "metadata_path": str(paths["metadata"].resolve()),
        "resolved_config_path": str(paths["resolved_config"].resolve()),
        "validation_predictions_path": str(paths["validation_predictions"].resolve()),
        "best_validation_metrics_path": str(paths["best_validation_metrics"].resolve()),
        "checkpoint_sha256": _sha256(paths["checkpoint"]),
        "metadata_sha256": _sha256(paths["metadata"]),
        "resolved_config_sha256": _sha256(paths["resolved_config"]),
        "validation_predictions_sha256": _sha256(paths["validation_predictions"]),
        "best_validation_metrics_sha256": _sha256(paths["best_validation_metrics"]),
        "source_hashes": metadata.get("source_hashes"),
        "e8_source_hash": metadata.get("e8_source_hash"),
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
        "adapter_hash": _adapter_hash(),
        "model": metadata["model"],
        "backbone_hashes": metadata["backbone_hashes"],
        "tokenizer_hashes": metadata["tokenizer_hashes"],
        "data_provenance": metadata["data_provenance"],
        "precision": metadata["runtime"]["precision"],
        "attention_implementation": metadata["attention_implementation"],
        "max_length": metadata["max_length"],
        "micro_batch_size": metadata["micro_batch_size"],
        "effective_batch_size": metadata["effective_batch_size"],
        "validation_row_count": len(validation_rows),
        "validation_split_snapshot": _split_snapshot(validation),
        "selection_basis": "best validation selection_f1 only; test not loaded during training",
    }


def build_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        manifest = _read_json(MANIFEST_PATH)
        if manifest.get("locked") is not True or manifest.get("checkpoint_count") != 4:
            raise RuntimeError(f"Malformed confirmation manifest: {MANIFEST_PATH}")
        return manifest
    _assert_all_training_complete()
    rows = [_manifest_row(task, seed) for task, seed in ALL_RUNS]
    manifest = {
        "protocol": CONFIRMATION_PROTOCOL,
        "test_protocol": TEST_PROTOCOL,
        "manifest_id": "E8-confirmation-manifest-20260910-seeds43-44-v1",
        "created_at": _now(),
        "locked": True,
        "selection_locked_before_test": True,
        "checkpoint_count": len(rows),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "method": "nc_plus_llrd",
        "mode": "nc",
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
        "adapter_hash": _adapter_hash(),
        "selection_basis": "validation-only best checkpoint per fixed task/seed; test split excluded",
        "test_reselection": False,
        "rows": rows,
    }
    _write_json(MANIFEST_PATH, manifest)
    return manifest


def _assert_all_training_complete() -> None:
    for task_id, seed in ALL_RUNS:
        artifact = _artifact_dir(task_id, seed)
        status_path = artifact / "status.json"
        metadata_path = artifact / "metadata.json"
        checkpoint_path = artifact / "best_checkpoint.pt"
        if not all(path.is_file() for path in (status_path, metadata_path, checkpoint_path)):
            raise RuntimeError(f"Training artifact incomplete: {artifact}")
        if _read_json(status_path).get("status") != "completed":
            raise RuntimeError(f"Training status is not completed: {artifact}")


def _runtime(device: torch.device) -> dict[str, Any]:
    value = e1_train._runtime_metadata(device, "bf16")
    value.update({
        "torch_version": torch.__version__,
        "transformers_version": TRANSFORMERS_VERSION,
        "cuda_available": bool(torch.cuda.is_available()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "attention_implementation": "sdpa",
        "cpu_fallback": False,
    })
    return value


def _assert_finite(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{where}.{key}")
    elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
        raise RuntimeError(f"Non-finite value in {where}: {value}")


def _test_paths(artifact: Path) -> tuple[Path, Path, Path]:
    return (
        artifact / "e8_test_metrics.json",
        artifact / "e8_test_predictions.json",
        artifact / "e8_test_evaluation_metadata.json",
    )


def _evaluate_one(row: Mapping[str, Any], manifest_sha256: str, device_name: str) -> dict[str, Any]:
    task_id = str(row["task_id"])
    seed = int(row["seed"])
    artifact = Path(str(row["artifact_dir"]))
    paths = _test_paths(artifact)
    present = tuple(path.exists() for path in paths)
    if any(present):
        if not all(present):
            raise RuntimeError(f"Partial test output set; refusing overwrite: {artifact}")
        raise RuntimeError(f"Test outputs already exist; refusing duplicate inference: {artifact}")
    if _sha256(Path(str(row["checkpoint_path"]))) != row["checkpoint_sha256"]:
        raise RuntimeError(f"Locked checkpoint hash changed before test: {task_id}, seed {seed}")
    if _sha256(Path(str(row["metadata_path"]))) != row["metadata_sha256"]:
        raise RuntimeError(f"Locked metadata hash changed before test: {task_id}, seed {seed}")
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=True)
    train = e1_train.split_field(bundle, "train")
    test = e1_train.split_field(bundle, "test")
    if train is None or test is None or len(test) != EXPECTED_TEST_COUNTS[task_id]:
        raise RuntimeError(f"Frozen test contract failed: {task_id}")
    device = e1_train.resolve_device(device_name)
    if device.type != "cuda":
        raise RuntimeError("Confirmation test requires CUDA; CPU fallback forbidden")
    precision = e1_train.resolve_precision("bf16", device)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True, use_fast=True)
    lengths = e1_train.inspect_train_lengths(tokenizer, list(e1_train.split_texts(train)))
    if lengths.max_length != 256:
        raise RuntimeError(f"Train-only max length changed: {task_id}")
    counts = e1_train.task_aspect_class_counts(bundle)
    model, model_spec = build_classifier(
        BACKBONE,
        num_classes=e1_train.task_num_classes(bundle),
        num_aspects=e1_train.task_num_aspects(bundle),
        aspect_class_counts=counts if e1_train.task_num_aspects(bundle) else None,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    if model_spec.to_dict() != row["model"]:
        raise RuntimeError(f"Model specification changed: {task_id}, seed {seed}")
    state = torch.load(Path(str(row["checkpoint_path"])), map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Checkpoint missing model state: {task_id}, seed {seed}")
    model.load_state_dict(state["model"], strict=True)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Non-finite checkpoint parameter: {task_id}:{name}")
    model.to(device)
    model.eval()
    loader = e1_train._make_loader(
        tokenizer,
        test,
        max_length=256,
        batch_size=32,
        shuffle=False,
        seed=seed,
        labels=e1_train.model_labels(bundle, "test", test),
        gold_labels=e1_train.split_labels(test),
    )
    started = time.time()
    # run_test is evaluation-only and disables noise by construction; E8
    # checkpoint selection has already been locked in the manifest.
    metrics, prediction_rows = e1_train.run_test(model, loader, device, precision, task_id, bundle)
    if len(prediction_rows) != EXPECTED_TEST_COUNTS[task_id]:
        raise RuntimeError(f"Unexpected test prediction count: {task_id}, seed {seed}")
    expected_ids = list(e1_train.split_ids(test))
    expected_gold = list(e1_train.split_labels(test))
    actual_ids = [item["id"] for item in prediction_rows]
    actual_gold = [item["gold"] for item in prediction_rows]
    if actual_ids != expected_ids or actual_gold != expected_gold:
        raise RuntimeError(f"Test IDs/golds changed or reordered: {task_id}, seed {seed}")
    _assert_finite(metrics, f"{task_id}.seed{seed}.metrics")
    runtime = _runtime(device)
    evaluation = {
        "protocol": TEST_PROTOCOL,
        "status": "completed",
        "evaluated_at_utc": _now(),
        "task_id": task_id,
        "seed": seed,
        "method": row["method"],
        "mode": row["mode"],
        "coefficient": row["coefficient"],
        "best_epoch": row["best_epoch"],
        "artifact_dir": str(artifact.resolve()),
        "checkpoint_sha256": row["checkpoint_sha256"],
        "metadata_sha256": row["metadata_sha256"],
        "manifest_sha256": manifest_sha256,
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
        "confirmation_adapter_hash": _adapter_hash(),
        "source_hashes": row["source_hashes"],
        "backbone_hashes": row["backbone_hashes"],
        "tokenizer_hashes": row["tokenizer_hashes"],
        "data_provenance": row["data_provenance"],
        "model": model_spec.to_dict(),
        "runtime": runtime,
        "contract": {
            "mode": "inference_only",
            "teacher_inference": False,
            "noise_inference": False,
            "dropout": "disabled via model.eval()",
            "strict_state_dict": True,
            "max_length": 256,
            "micro_batch_size": 32,
            "precision": "bf16",
            "attention_implementation": "sdpa",
            "seed_for_loader": seed,
        },
        "n_test": len(prediction_rows),
        "elapsed_seconds": time.time() - started,
    }
    _write_json(paths[0], metrics)
    _write_json(paths[1], {"rows": prediction_rows})
    _write_json(paths[2], evaluation)
    checkpoint_after = _sha256(Path(str(row["checkpoint_path"])))
    if checkpoint_after != row["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint changed during test: {task_id}, seed {seed}")
    del model
    torch.cuda.empty_cache()
    return {"task_id": task_id, "seed": seed, "status": "evaluated", "prediction_count": len(prediction_rows), "metrics": metrics, "paths": [str(path.resolve()) for path in paths], "evaluation": evaluation}


def _metric_from_validation(row: Mapping[str, Any], metric: str) -> float:
    metrics = _read_json(Path(str(row["best_validation_metrics_path"])))
    return float(metrics[metric])


def _load_existing_seed42(task_id: str) -> dict[str, Any]:
    artifact = _artifact_dir(task_id, 42)
    metadata = _read_json(artifact / "metadata.json")
    val = _read_json(artifact / "best_validation_metrics.json")
    test = _read_json(artifact / "e8_test_metrics.json")
    return {
        "task_id": task_id,
        "seed": 42,
        "artifact_dir": str(artifact.resolve()),
        "best_epoch": metadata["best_epoch"],
        "metadata": metadata,
        "validation_metrics": val,
        "test_metrics": test,
        "checkpoint_sha256": _sha256(artifact / "best_checkpoint.pt"),
        "metadata_sha256": _sha256(artifact / "metadata.json"),
    }


def _diagnostic_baseline_paths(task_id: str, seed: int, mode: str) -> tuple[Path, Path]:
    diagnostics = _read_json(ROOT / "artifacts" / "diagnostics" / "teacher_gap.json")
    key = f"{task_id}.e1.seed{seed}" if mode == "e1" else f"{task_id}.e2.nc.seed{seed}"
    path_value = diagnostics["input_prediction_files"][key]["path"]
    prediction_path = Path(path_value)
    artifact = prediction_path.parent
    metadata_path = artifact / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    return artifact, metadata_path


def _baseline_metric(task_id: str, seed: int, mode: str, metric: str, split: str) -> float:
    artifact, metadata_path = _diagnostic_baseline_paths(task_id, seed, mode)
    metadata = _read_json(metadata_path)
    if int(metadata.get("seed", -1)) != seed or metadata.get("task_id") != task_id:
        raise RuntimeError(f"Baseline metadata mismatch: {task_id}, seed {seed}, {mode}")
    if split == "validation":
        name = "best_validation_metrics.json"
    else:
        name = "test_metrics.json" if mode == "e1" else "e2_test_metrics.json"
    path = artifact / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return float(_read_json(path)[metric])


def _mean_sd(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize empty values")
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return {"n": len(values), "mean": mean, "sample_sd": sd}


def _build_summary(manifest: Mapping[str, Any], test_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confirmation_rows = list(manifest["rows"])
    test_by_key = {(str(item["task_id"]), int(item["seed"])): item for item in test_results}
    rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    paired: list[dict[str, Any]] = []
    for task_id in TASKS:
        aggregate[task_id] = {"validation": {}, "test": {}}
        for metric in METRIC_NAMES[task_id]:
            val_values = []
            test_values = []
            for seed in (42, 43, 44):
                if seed == 42:
                    seed42 = _load_existing_seed42(task_id)
                    val = float(seed42["validation_metrics"][metric])
                    test = float(seed42["test_metrics"][metric])
                else:
                    match = next(item for item in confirmation_rows if item["task_id"] == task_id and int(item["seed"]) == seed)
                    val = _metric_from_validation(match, metric)
                    test = float(test_by_key[(task_id, seed)]["metrics"][metric])
                val_values.append(val)
                test_values.append(test)
            aggregate[task_id]["validation"][metric] = _mean_sd(val_values)
            aggregate[task_id]["test"][metric] = _mean_sd(test_values)
            for seed in (42, 43, 44):
                if seed == 42:
                    e8_val = float(_load_existing_seed42(task_id)["validation_metrics"][metric])
                    e8_test = float(_load_existing_seed42(task_id)["test_metrics"][metric])
                else:
                    match = next(item for item in confirmation_rows if item["task_id"] == task_id and int(item["seed"]) == seed)
                    e8_val = _metric_from_validation(match, metric)
                    e8_test = float(test_by_key[(task_id, seed)]["metrics"][metric])
                paired_row = next((item for item in paired if item["task_id"] == task_id and item["metric"] == metric), None)
                if paired_row is None:
                    paired_row = {"task_id": task_id, "task": TASK_LABELS[task_id], "metric": metric, "seeds": []}
                    paired.append(paired_row)
                paired_row["seeds"].append({
                    "seed": seed,
                    "e8_validation": e8_val,
                    "e8_test": e8_test,
                    "e1_validation": _baseline_metric(task_id, seed, "e1", metric, "validation"),
                    "e1_test": _baseline_metric(task_id, seed, "e1", metric, "test"),
                    "e2_nc_validation": _baseline_metric(task_id, seed, "e2", metric, "validation"),
                    "e2_nc_test": _baseline_metric(task_id, seed, "e2", metric, "test"),
                    "delta_e8_minus_e1_validation": e8_val - _baseline_metric(task_id, seed, "e1", metric, "validation"),
                    "delta_e8_minus_e1_test": e8_test - _baseline_metric(task_id, seed, "e1", metric, "test"),
                    "delta_e8_minus_e2_nc_validation": e8_val - _baseline_metric(task_id, seed, "e2", metric, "validation"),
                    "delta_e8_minus_e2_nc_test": e8_test - _baseline_metric(task_id, seed, "e2", metric, "test"),
                })
                # The next pass creates compact mean deltas without choosing a
                # method or seed from test results.
            deltas = paired[-1]["seeds"]
            paired[-1]["mean_delta_e8_minus_e1_validation"] = sum(x["delta_e8_minus_e1_validation"] for x in deltas) / 3
            paired[-1]["mean_delta_e8_minus_e1_test"] = sum(x["delta_e8_minus_e1_test"] for x in deltas) / 3
            paired[-1]["mean_delta_e8_minus_e2_nc_validation"] = sum(x["delta_e8_minus_e2_nc_validation"] for x in deltas) / 3
            paired[-1]["mean_delta_e8_minus_e2_nc_test"] = sum(x["delta_e8_minus_e2_nc_test"] for x in deltas) / 3
            paired[-1]["positive_delta_count_e8_minus_e2_nc_test"] = sum(x["delta_e8_minus_e2_nc_test"] > 0 for x in deltas)
    for row in confirmation_rows:
        result = test_by_key[(str(row["task_id"]), int(row["seed"]))]
        rows.append({
            "task_id": row["task_id"],
            "task": row["task"],
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "best_validation_selection_f1": row["best_selection_f1"],
            "validation_metrics": _read_json(Path(str(row["best_validation_metrics_path"]))),
            "test_metrics": result["metrics"],
            "test_prediction_count": result["prediction_count"],
            "test_output_paths": result["paths"],
        })
    return {
        "protocol": CONFIRMATION_PROTOCOL,
        "summary_id": "E8-confirmation-summary-20260910-seeds42-44-v1",
        "status": "completed",
        "created_at": _now(),
        "original_e8_source_hash": ORIGINAL_E8_SOURCE_HASH,
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "training_runs": rows,
        "aggregate_seed42_43_44": aggregate,
        "paired_deltas": paired,
        "paper_targets_percent": PAPER_TARGETS,
        "paper_target_caveat": "Targets use the cited paper's averaging definitions; local evaluator averaging equivalence remains a caveat.",
        "llrd_multiseed_scope": "E7 has seed 42 only; no multi-seed superiority claim over LLRD is made.",
        "test_selection": "No test-based selection, tuning, or reselection; all checkpoints were locked by validation before test inference.",
    }


def _write_summary_csv(summary: Mapping[str, Any]) -> None:
    fields = ["task_id", "task", "seed", "best_epoch", "metric", "validation", "test", "paper_target_percent"]
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["training_runs"]:
            for metric in METRIC_NAMES[row["task_id"]]:
                writer.writerow({
                    "task_id": row["task_id"],
                    "task": row["task"],
                    "seed": row["seed"],
                    "best_epoch": row["best_epoch"],
                    "metric": metric,
                    "validation": row["validation_metrics"][metric],
                    "test": row["test_metrics"][metric],
                    "paper_target_percent": PAPER_TARGETS[row["task_id"]][metric],
                })


def _write_result_doc(summary: Mapping[str, Any]) -> None:
    lines = [
        "# E8 confirmation result",
        "",
        "The frozen E8 NC+LLRD recipe was confirmed on ViSFD and UIT-ABSA Restaurant with fresh students and seeds 43 and 44. Each best checkpoint was selected by validation only, locked in a four-row manifest, and evaluated once on the test split with CUDA BF16, SDPA, max length 256, batch size 32, strict model loading, and dropout disabled.",
        "",
        "| Task | Seed | Best epoch | Pair F1 validation | Pair F1 test | Detection F1 test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["training_runs"]:
        lines.append(
            f"| {row['task']} | {row['seed']} | {row['best_epoch']} | {row['validation_metrics']['pair_micro_f1']:.4f} | {row['test_metrics']['pair_micro_f1']:.4f} | {row['test_metrics']['detection_micro_f1']:.4f} |"
        )
    lines.extend([
        "",
        "Aggregate validation and test means use seeds 42, 43, and 44 with sample standard deviation. Paired per-seed E8 deltas against the already completed E1 CE and E2 NC runs are recorded in `summary.json`; the E8-versus-E2 NC positive-count is descriptive with three seeds and is not a significance test.",
        "",
        "Paper-target comparisons are retained as secondary context: ViSFD pair F1 84.24 and detection F1 89.14; UIT-ABSA Restaurant pair F1 74.89 and detection F1 88.01. The local evaluator's averaging definitions may not be equivalent to the paper's reported values.",
        "",
        "E7 LLRD has only seed 42 in the existing evidence, so this confirmation makes no multi-seed superiority claim over E7.",
        "",
        f"Manifest: `{MANIFEST_PATH}` (SHA-256 `{summary['manifest_sha256']}`). CSV: `{SUMMARY_CSV_PATH}`.",
    ])
    RESULT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tests() -> dict[str, Any]:
    manifest = build_manifest()
    if not manifest.get("locked") or manifest.get("checkpoint_count") != 4:
        raise RuntimeError("Confirmation test gate requires locked four-checkpoint manifest")
    manifest_sha256 = _sha256(MANIFEST_PATH)
    devices, snapshot = _eligible_devices()
    results: list[dict[str, Any]] = []
    for row in manifest["rows"]:
        results.append(_evaluate_one(row, manifest_sha256, devices[0]))
    if len(results) != 4 or any(item["prediction_count"] != EXPECTED_TEST_COUNTS[item["task_id"]] for item in results):
        raise RuntimeError("Confirmation test gate did not complete all four test passes")
    access = {
        "protocol": TEST_PROTOCOL,
        "access_log_id": "E8-confirmation-test-access-20260910-seeds43-44-v1",
        "started_at_utc": results[0]["evaluation"]["evaluated_at_utc"],
        "completed_at_utc": _now(),
        "manifest_path": str(MANIFEST_PATH.resolve()),
        "manifest_sha256": manifest_sha256,
        "policy": "four locked E8 confirmation checkpoints, one test pass each; no test selection or tuning",
        "device": devices[0],
        "gpu_snapshot": snapshot,
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "results": results,
    }
    _write_json(TEST_ACCESS_LOG_PATH, access)
    summary = _build_summary(manifest, results)
    _write_json(SUMMARY_PATH, summary)
    _write_summary_csv(summary)
    _write_result_doc(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run locked E8 ViSFD/Restaurant seed confirmation")
    parser.add_argument("--launch", action="store_true", help="start durable detached controller")
    parser.add_argument("--controller", action="store_true", help="run durable controller")
    parser.add_argument("--worker", action="store_true", help="run one isolated frozen E8 worker")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.launch:
            print(json.dumps(_launch_detached_controller(), indent=2), flush=True)
            return 0
        if args.controller:
            return _controller()
        if args.worker:
            if args.task_id is None or args.seed is None:
                raise ValueError("--worker requires --task-id and --seed")
            result = run_worker(args.task_id, args.seed, args.device, _artifact_dir(args.task_id, args.seed))
            print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
            return 0
        if args.build_manifest:
            print(json.dumps(build_manifest(), ensure_ascii=False, indent=2), flush=True)
            return 0
        if args.run_tests:
            print(json.dumps(run_tests(), ensure_ascii=False, indent=2), flush=True)
            return 0
        raise ValueError("Choose --launch, --controller, --worker, --build-manifest, or --run-tests")
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
