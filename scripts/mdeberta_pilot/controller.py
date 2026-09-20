"""Durable five-worker controller for the mDeBERTa pilot.

The controller serializes only plain JSON state.  Child process/file handles
stay in memory, which keeps restart inspection safe and avoids the historical
Popen serialization/adoption failure mode.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mdeberta_pilot import common  # noqa: E402


GPUS = ("0", "1", "2", "3")


def job_key(task_id: str) -> str:
    return f"mdeberta:{task_id}:lr_{common.LEARNING_RATE:.0e}:seed_{common.SEED}"


def status_path(task_id: str) -> Path:
    return common.run_dir(task_id) / "status.json"


def read_status(task_id: str) -> dict[str, Any]:
    path = status_path(task_id)
    if not path.is_file():
        return {}
    try:
        return common.read_json(path)
    except Exception:
        return {}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--max-shared-free-mib", type=int, default=18000)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    common.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tasks = list(common.TASKS)
    state_path = common.ARTIFACT_ROOT / "controller_state.json"
    active: dict[str, dict[str, Any]] = {}
    interrupted = False

    def write_state(state: str) -> None:
        jobs = []
        for task_id in tasks:
            status = read_status(task_id)
            jobs.append(
                {
                    "key": job_key(task_id),
                    "task": task_id,
                    "learning_rate": common.LEARNING_RATE,
                    "seed": common.SEED,
                    "status": status.get("status", "pending"),
                    "pid": status.get("pid"),
                    "gpu": status.get("gpu"),
                }
            )
        common.write_json(
            state_path,
            {
                "protocol": common.PROTOCOL,
                "status": state,
                "pid": os.getpid(),
                "started_at_unix": started,
                "updated_at": common.now(),
                "tasks": jobs,
                "active": [
                    {key: value for key, value in record.items() if key in {"task", "gpu", "pid", "started_at", "log_path"}}
                    for record in active.values()
                ],
                "gpu_snapshot": common.gpu_snapshot(),
            },
        )

    def stop(_signal: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    write_state("planned")
    if args.once:
        print(json.dumps(common.read_json(state_path), indent=2))
        return 0

    def launch(task_id: str, gpu: str) -> None:
        output_dir = common.run_dir(task_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train.log"
        log = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--task",
            task_id,
            "--device",
            "cuda:0",
            "--micro-batch",
            str(args.micro_batch),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(common.ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        common.write_json(
            output_dir / "status.json",
            {
                "status": "running",
                "protocol": common.PROTOCOL,
                "task_id": task_id,
                "learning_rate": common.LEARNING_RATE,
                "seed": common.SEED,
                "pid": process.pid,
                "gpu": gpu,
                "device": "cuda:0",
                "started_at": common.now(),
            },
        )
        active[task_id] = {"task": task_id, "gpu": gpu, "pid": process.pid, "started_at": common.now(), "process": process, "log": log, "log_path": str(log_path)}
        print(f"launched {job_key(task_id)} gpu={gpu} pid={process.pid}", flush=True)

    # Reuse only genuinely running workers on controller restart.  A stale
    # status is never adopted and is left for the worker's owner to inspect.
    for task_id in tasks:
        status = read_status(task_id)
        pid = int(status.get("pid", 0) or 0)
        if status.get("status") == "running" and pid and alive(pid):
            active[task_id] = {"task": task_id, "gpu": str(status.get("gpu", "unknown")), "pid": pid, "started_at": status.get("started_at"), "adopted": True}

    write_state("running")
    # Start one worker per physical GPU.  The fifth worker is launched only
    # after a live memory profile confirms a safe shared device.
    next_index = 0
    shared_launched = False
    initial_launch_unix: float | None = None
    while not interrupted:
        for task_id, record in list(active.items()):
            process = record.get("process")
            if process is not None and process.poll() is None:
                continue
            status = read_status(task_id)
            if process is None and status.get("status") == "running" and alive(int(record["pid"])):
                continue
            if process is not None and status.get("status") == "running":
                common.write_json(
                    status_path(task_id),
                    {"status": "failed", "protocol": common.PROTOCOL, "task_id": task_id, "pid": record["pid"], "gpu": record["gpu"], "finished_at": common.now(), "error": f"worker exited with code {process.returncode}"},
                )
            log = record.get("log")
            if log is not None:
                log.close()
            del active[task_id]

        pending = [task_id for task_id in tasks if task_id not in active and not status_path(task_id).exists()]
        # Fill unused GPU slots first.
        for gpu in GPUS:
            if not pending:
                break
            if sum(str(record.get("gpu")) == gpu for record in active.values()) >= 1:
                continue
            task_id = pending.pop(0)
            launch(task_id, gpu)
            if initial_launch_unix is None:
                initial_launch_unix = time.time()
        # Profile after initial launch and share the device with most free VRAM.
        # Give the first four workers time to load the 1.33 GB checkpoint and
        # reach a real training heartbeat before considering GPU sharing.
        heartbeats = []
        for record in active.values():
            task = record["task"]
            heartbeat_path = common.run_dir(task) / "heartbeat.json"
            if heartbeat_path.is_file():
                try:
                    heartbeats.append(common.read_json(heartbeat_path))
                except Exception:
                    pass
        settled = bool(initial_launch_unix and time.time() - initial_launch_unix >= 60)
        training_heartbeat = sum(1 for heartbeat in heartbeats if heartbeat.get("status") == "running" and "batch" in heartbeat) >= 1
        if pending and not shared_launched and settled and training_heartbeat:
            snapshot = common.gpu_snapshot()
            candidates = [
                (int(values["free_mib"]), gpu)
                for gpu, values in snapshot.items()
                if gpu in GPUS and int(values.get("free_mib", 0)) >= args.max_shared_free_mib
            ]
            if candidates:
                _, gpu = max(candidates)
                task_id = pending.pop(0)
                launch(task_id, gpu)
                shared_launched = True
        write_state("running")
        if not active and not pending and all(read_status(task_id).get("status") in {"completed", "completed_with_check_failure", "failed"} for task_id in tasks):
            break
        if pending and not active:
            # A non-running stale status should never be silently overwritten.
            raise RuntimeError(f"Pending jobs blocked by existing status files: {pending}")
        time.sleep(max(2, args.poll_seconds))
    write_state("stopped" if interrupted else "completed")
    return 0


if __name__ == "__main__":
    if "--worker" in sys.argv:
        from scripts.mdeberta_pilot.train import main as worker_main

        sys.argv.remove("--worker")
        raise SystemExit(worker_main())
    raise SystemExit(main())
