"""Adopt an active E9 worker and schedule the remaining fixed tasks on GPUs 1/2/3.

This small recovery controller exists so a controller restart can expand from
the original GPU2-only launch without restarting or duplicating its detached
worker.  It invokes the already launched E9 worker entry point directly and
does not change model/training code or the experiment recipe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_e9_label_attention import (  # noqa: E402
    ARTIFACT_ROOT,
    EPOCHS,
    EFFECTIVE_BATCH_SIZE,
    MICRO_BATCH_SIZE,
    PROTOCOL,
    REVISION,
    ROOT as E9_ROOT,
    SEED,
    TASKS,
    _artifact_candidates,
    _completed_artifact,
    _gpu_snapshot,
    _now,
    _pid_alive,
    _read_json,
    _summary_from_artifacts,
    _write_json,
    source_hashes,
)


DEVICES = ("cuda:1", "cuda:2", "cuda:3")
GPU_GUARD_MIB = 5 * 1024
WORKER_ROOT = ARTIFACT_ROOT / ".orchestrator" / "workers"
LOG_ROOT = ARTIFACT_ROOT / ".orchestrator" / "logs"
STATE_PATH = ARTIFACT_ROOT / ".orchestrator" / "state.json"


def _eligible_devices() -> tuple[list[str], dict[str, dict[str, int]]]:
    snapshot = _gpu_snapshot()
    eligible = [
        device
        for device in DEVICES
        if snapshot.get(device.split(":", 1)[1], {}).get("free_mib", 0) >= GPU_GUARD_MIB
    ]
    if not eligible:
        raise RuntimeError(f"No GPU1/2/3 satisfies {GPU_GUARD_MIB} MiB guard: {snapshot}")
    return eligible, snapshot


def _active_workers() -> dict[str, dict[str, Any]]:
    active: dict[str, dict[str, Any]] = {}
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    for task_id in TASKS:
        path = WORKER_ROOT / f"{task_id}.json"
        if not path.is_file():
            continue
        try:
            row = _read_json(path)
        except Exception:
            continue
        if row.get("status") == "running" and _pid_alive(row.get("pid")):
            active[task_id] = {
                "task_id": task_id,
                "pid": int(row["pid"]),
                "device": row.get("device"),
                "status": "running",
                "artifact_dir": row.get("output_dir"),
                "adopted": True,
            }
    return active


def _completed() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for task_id in TASKS:
        path = _completed_artifact(task_id)
        if path is not None:
            result[task_id] = path
    return result


def _write_state(
    *,
    values: dict[str, Any],
    snapshot: dict[str, dict[str, int]],
    eligible: list[str],
    jobs: dict[str, dict[str, Any]],
    status: str,
    started_at: str,
    finished_at: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": status,
        "controller_pid": os.getpid(),
        "started_at": started_at,
        "source_hashes": values,
        "preferred_gpu_indices": [1, 2, 3],
        "eligible_devices": eligible,
        "gpu_snapshot_at_launch": snapshot,
        "tasks": list(TASKS),
        "jobs": jobs,
        "controller_mode": "parallel_recovery_adopt_existing_worker",
        "preflight": str(ARTIFACT_ROOT / ".orchestrator" / "preflight.json"),
    }
    if finished_at is not None:
        payload["finished_at"] = finished_at
    _write_json(STATE_PATH, payload)


def run(*, poll_seconds: float = 15.0) -> dict[str, Any]:
    values = source_hashes()
    eligible, snapshot = _eligible_devices()
    completed = _completed()
    active = _active_workers()
    overlap = set(completed).intersection(active)
    if overlap:
        raise RuntimeError(f"Task marked completed and running simultaneously: {sorted(overlap)}")
    if len(completed) == len(TASKS):
        return _summary_from_artifacts(completed, values)

    jobs: dict[str, dict[str, Any]] = {
        task_id: {
            "task_id": task_id,
            "status": "completed",
            "artifact_dir": str(path),
            "reused_existing": True,
        }
        for task_id, path in completed.items()
    }
    jobs.update(active)
    started_at = _now()
    _write_state(
        values=values,
        snapshot=snapshot,
        eligible=eligible,
        jobs=jobs,
        status="running",
        started_at=started_at,
    )
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    controller_log = (LOG_ROOT / "parallel_controller.log").open("a", encoding="utf-8", buffering=1)
    controller_log.write(json.dumps({"event": "adopt", "time": _now(), "active": active}) + "\n")

    queue = [task_id for task_id in TASKS if task_id not in completed and task_id not in active]
    free_devices = [device for device in eligible if device not in {row.get("device") for row in active.values()}]
    running: dict[str, dict[str, Any]] = {
        task_id: {"pid": row["pid"], "device": row.get("device"), "popen": None, "adopted": True}
        for task_id, row in active.items()
    }

    def launch(task_id: str, device: str) -> None:
        output_dir = ARTIFACT_ROOT / task_id / f"seed_{SEED}"
        log_path = LOG_ROOT / f"{task_id}__seed{SEED}.log"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_e9_label_attention.py"),
            "--task-id",
            task_id,
            "--device",
            device,
            "--output-dir",
            str(output_dir),
        ]
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        handle.write(json.dumps({"event": "parallel_launch", "time": _now(), "command": command}) + "\n")
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
        running[task_id] = {"pid": process.pid, "device": device, "popen": process, "adopted": False}
        jobs[task_id] = {
            "task_id": task_id,
            "pid": process.pid,
            "device": device,
            "artifact_dir": str(output_dir),
            "log_path": str(log_path),
            "status": "running",
            "launched_at": _now(),
            "command": command,
        }
        controller_log.write(json.dumps({"event": "launch", "task_id": task_id, "pid": process.pid, "device": device}) + "\n")
        _write_state(values=values, snapshot=snapshot, eligible=eligible, jobs=jobs, status="running", started_at=started_at)

    try:
        while queue or running:
            while queue and free_devices:
                launch(queue.pop(0), free_devices.pop(0))
            finished: list[str] = []
            for task_id, row in list(running.items()):
                process = row.get("popen")
                worker_path = WORKER_ROOT / f"{task_id}.json"
                if process is not None:
                    return_code = process.poll()
                    if return_code is None:
                        continue
                    if return_code != 0:
                        raise RuntimeError(f"E9 worker {task_id} failed with return code {return_code}")
                    worker = _read_json(worker_path)
                    if worker.get("status") != "completed":
                        raise RuntimeError(f"E9 worker {task_id} exited without completed status")
                    artifact = Path(str(worker["artifact_dir"]))
                else:
                    if worker_path.is_file():
                        worker = _read_json(worker_path)
                        if worker.get("status") == "completed":
                            artifact = Path(str(worker["artifact_dir"]))
                        elif worker.get("status") == "failed":
                            raise RuntimeError(f"Adopted E9 worker {task_id} failed: {worker.get('error')}")
                        elif not _pid_alive(row["pid"]):
                            raise RuntimeError(f"Adopted E9 worker {task_id} PID {row['pid']} exited without status")
                        else:
                            continue
                    elif not _pid_alive(row["pid"]):
                        raise RuntimeError(f"Adopted E9 worker {task_id} PID {row['pid']} exited")
                    else:
                        continue
                completed[task_id] = artifact
                jobs[task_id].update({"status": "completed", "artifact_dir": str(artifact), "finished_at": _now()})
                finished.append(task_id)
                free_devices.append(str(row["device"]))
            for task_id in finished:
                running.pop(task_id)
            if finished:
                _write_state(values=values, snapshot=snapshot, eligible=eligible, jobs=jobs, status="running", started_at=started_at)
            if running:
                time.sleep(poll_seconds)
    finally:
        controller_log.close()
    summary = _summary_from_artifacts(completed, values)
    _write_state(values=values, snapshot=snapshot, eligible=eligible, jobs=jobs, status="completed", started_at=started_at, finished_at=_now())
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, default=str), flush=True)
