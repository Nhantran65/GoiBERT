"""Durable E3 screening orchestrator.

The script owns only the two approved seed-42 validation trials.  It never
loads test data.  ``--plan`` records the launch contract; ``--execute`` runs
the two worker processes in detached sessions and records their PIDs/logs.
The workers themselves write status and heartbeat files inside each artifact.
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
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from goibert.train import _json_default, read_json, sha256_file  # noqa: E402
from goibert_e3.train import (  # noqa: E402
    LEARNING_RATE,
    MAX_LENGTH,
    MICRO_BATCH_SIZE,
    PROTOCOL,
    REVISION,
    SCREENING_SEED,
    EPOCHS,
    source_hashes,
)


ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "e3"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
TASKS = ("visfd", "absa_restaurant")
GPU_GUARD_GIB = 5.0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gpu_free_memory() -> dict[str, float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    values: dict[str, float] = {}
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) == 2:
            values[pieces[0]] = float(pieces[1]) / 1024.0
    return values


def _choose_devices() -> dict[str, str]:
    free = _gpu_free_memory()
    eligible = sorted(
        ((memory, index) for index, memory in free.items() if memory >= GPU_GUARD_GIB),
        reverse=True,
    )
    if len(eligible) < 2:
        raise RuntimeError(
            f"Need two GPUs with >= {GPU_GUARD_GIB:.1f} GiB free; observed {free}"
        )
    # Prefer the user's requested GPU2, then the next most free device.  A
    # worker receives the physical index directly, so CUDA_VISIBLE_DEVICES is
    # left untouched and runtime metadata reports the actual device.
    ordered = [item for item in eligible if item[1] == "2"] + [item for item in eligible if item[1] != "2"]
    selected = [index for _, index in ordered[:2]]
    return {TASKS[0]: f"cuda:{selected[0]}", TASKS[1]: f"cuda:{selected[1]}"}


def _trial_dir(task: str) -> Path:
    base = ARTIFACT_ROOT / task / f"seed_{SCREENING_SEED}"
    if not base.exists():
        return base
    for index in range(1, 10_000):
        candidate = base.with_name(f"{base.name}__rerun{index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not allocate E3 artifact directory for {task}")


def _plan_payload() -> dict[str, Any]:
    hashes = source_hashes(REPO_ROOT)
    return {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "created_at": _now(),
        "locked": True,
        "tasks": list(TASKS),
        "seed": SCREENING_SEED,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "effective_batch_size": MICRO_BATCH_SIZE,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "precision": "bf16",
        "attention_implementation": "sdpa",
        "include_test": False,
        "source_hashes": hashes,
        "e1_dependency_source_hash": hashes["e1_dependency"],
        "worker_contract": "fresh goiturn-s-gd + E1 head; CE_E1 + 0.5 pooled conditional polarity CE",
        "artifacts": {
            task: str(ARTIFACT_ROOT / task / f"seed_{SCREENING_SEED}") for task in TASKS
        },
    }


def write_plan() -> dict[str, Any]:
    payload = _plan_payload()
    _write_json(ORCH_ROOT / "plan.json", payload)
    return payload


def _launch_worker(task: str, device: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    artifact_dir = _trial_dir(task)
    log_path = LOG_ROOT / f"{task}__seed{SCREENING_SEED}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "goibert_e3.train",
        "--task-id",
        task,
        "--repo-root",
        str(REPO_ROOT),
        "--output-dir",
        str(artifact_dir),
        "--seed",
        str(SCREENING_SEED),
        "--device",
        device,
        "--precision",
        "bf16",
        "--attention-implementation",
        "sdpa",
        "--epochs",
        str(EPOCHS),
        "--expected-source-hash",
        str(plan["source_hashes"]["e3_source"]),
    ]
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    log_handle.write(json.dumps({"event": "launch", "time": _now(), "command": command}) + "\n")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")},
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # The parent may close its descriptor immediately; the detached child has
    # its own inherited descriptor and continues writing to the log.
    log_handle.close()
    return {
        "task_id": task,
        "pid": process.pid,
        "device": device,
        "artifact_dir": str(artifact_dir),
        "log_path": str(log_path),
        "command": command,
        "launched_at": _now(),
        "status": "running",
    }


def execute(*, wait: bool = False, poll_seconds: float = 20.0) -> dict[str, Any]:
    plan = read_json(ORCH_ROOT / "plan.json") if (ORCH_ROOT / "plan.json").is_file() else write_plan()
    if plan.get("protocol") != PROTOCOL or plan.get("locked") is not True:
        raise RuntimeError("E3 plan is missing or not locked")
    existing_state_path = ORCH_ROOT / "state.json"
    if existing_state_path.is_file():
        existing_state = read_json(existing_state_path)
        if existing_state.get("status") in {"running", "completed"}:
            raise RuntimeError("E3 trials already launched; inspect artifacts/e3/.orchestrator/state.json before rerunning")
    devices = _choose_devices()
    jobs = [_launch_worker(task, devices[task], plan) for task in TASKS]
    state = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "started_at": _now(),
        "source_hashes": plan["source_hashes"],
        "jobs": jobs,
        "status": "running",
    }
    _write_json(ORCH_ROOT / "state.json", state)
    if wait:
        while True:
            statuses: list[str] = []
            for job in jobs:
                status_path = Path(job["artifact_dir"]) / "status.json"
                if status_path.is_file():
                    try:
                        statuses.append(str(read_json(status_path).get("status", "unknown")))
                    except (OSError, ValueError, json.JSONDecodeError):
                        statuses.append("unknown")
                else:
                    statuses.append("starting")
            if all(status in {"completed", "failed"} for status in statuses):
                state["status"] = "completed" if all(status == "completed" for status in statuses) else "failed"
                state["finished_at"] = _now()
                state["worker_statuses"] = statuses
                _write_json(ORCH_ROOT / "state.json", state)
                break
            time.sleep(max(1.0, poll_seconds))
    return state


def status() -> dict[str, Any]:
    path = ORCH_ROOT / "state.json"
    if not path.is_file():
        return {"status": "not_started", "protocol": PROTOCOL}
    payload = read_json(path)
    for job in payload.get("jobs", []):
        status_path = Path(job["artifact_dir"]) / "status.json"
        if status_path.is_file():
            try:
                job["worker_status"] = read_json(status_path)
            except (OSError, ValueError, json.JSONDecodeError):
                job["worker_status"] = {"status": "unreadable"}
        alive = False
        pid = int(job.get("pid", -1))
        try:
            os.kill(pid, 0)
            alive = True
        except (ProcessLookupError, PermissionError):
            alive = False
        job["pid_alive"] = alive
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan/launch durable E3 validation trials")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--status", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.plan:
        print(json.dumps(write_plan(), ensure_ascii=False, indent=2, default=_json_default))
    elif args.execute:
        print(json.dumps(execute(wait=args.wait, poll_seconds=args.poll_seconds), ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(json.dumps(status(), ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
