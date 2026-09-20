#!/usr/bin/env python3
"""Durable, source-bound recovery controller for the three E17 Hotel trials.

This helper intentionally lives outside the frozen E17 runner.  It archives
the interrupted Hotel artifacts, verifies the approved source binding, then
launches only the three fresh Hotel workers sequentially on a rechecked GPU.
Workers still execute the frozen ``scripts/run_e17.py --worker`` path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "e17"
APPROVAL_PATH = ARTIFACT_ROOT / "root_approval.json"
RUNNER = ROOT / "scripts" / "run_e17.py"
PYTHON = ROOT / ".venv-e0-cuda" / "bin" / "python"
TASK = "absa_hotel"
ARMS = ("baseline", "full_context", "cavity")
PROFILE_PATH = ARTIFACT_ROOT / "profile.json"
RESERVE_HEADROOM_MIB = 4 * 1024
GUARD_MIB = 5 * 1024


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def directory_hashes(root: Path) -> dict[str, str]:
    suffixes = {".json", ".safetensors", ".txt", ".model", ".vocab"}
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    }


def current_source_binding() -> dict[str, Any]:
    # Importing the frozen E2 module is safe here: the helper never patches it.
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from goibert_e2 import train as e2_train  # type: ignore

    source_paths = [
        RUNNER,
        ROOT / "src" / "goibert_e17" / "modeling.py",
        ROOT / "src" / "goibert_e17" / "optim.py",
        ROOT / "src" / "goibert_e17" / "__init__.py",
        ROOT / "configs" / "e17" / "protocol.json",
        ROOT / "src" / "goibert_e2" / "train.py",
        ROOT / "src" / "goibert_e2" / "modeling.py",
        ROOT / "src" / "goibert_e7" / "optim.py",
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in source_paths
    }
    e7_root = ROOT / "src" / "goibert_e7"
    e7_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(e7_root.rglob("*.py"))
    }
    approval = read_json(APPROVAL_PATH)
    recipe = approval["source_binding"]["recipe"]
    return {
        "protocol": approval["source_binding"]["protocol"],
        "revision": approval["source_binding"]["revision"],
        "runner_sha256": sha256(RUNNER),
        "source_hashes": source_hashes,
        "backbone": {
            "path": str((ROOT / "models" / "goiturn-s-gd").resolve()),
            "files": directory_hashes(ROOT / "models" / "goiturn-s-gd"),
        },
        "frozen_e2_source_hashes": e2_train.source_hashes(ROOT),
        "frozen_e7_files": e7_hashes,
        "recipe": recipe,
    }


def verify_source_binding() -> dict[str, Any]:
    approval = read_json(APPROVAL_PATH)
    expected = approval.get("source_binding")
    actual = current_source_binding()
    if expected != actual:
        raise RuntimeError(
            "source binding mismatch; refusing recovery mutation/launch\n"
            + json.dumps({"expected": expected, "actual": actual}, indent=2, sort_keys=True)
        )
    return {
        "verified_at": now(),
        "approval_path": str(APPROVAL_PATH),
        "approval_sha256": sha256(APPROVAL_PATH),
        "source_binding": actual,
    }


def host_process_snapshot() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid,ppid,stat,lstart,args"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return {"available": False, "error": repr(exc), "rows": []}
    rows = [line for line in output.splitlines() if "scripts/run_e17.py" in line or "recover_e17_hotel.py" in line]
    return {"available": True, "rows": rows}


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    rows: dict[str, dict[str, Any]] = {}
    for line in output.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 6:
            rows[fields[0]] = {
                "name": fields[1],
                "total_mib": int(float(fields[2])),
                "used_mib": int(float(fields[3])),
                "free_mib": int(float(fields[4])),
                "utilization_gpu": int(float(fields[5])),
            }
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return {"captured_at": now(), "rows": rows}


def required_free_mib() -> int:
    profile = read_json(PROFILE_PATH)
    if profile.get("status") != "passed":
        raise RuntimeError(f"profile gate is not passed: {profile.get('status')!r}")
    rows = profile.get("rows", [])
    peak = max(int(row["peak_reserved_bytes"]) for row in rows)
    return math.ceil(peak / (1024 * 1024)) + RESERVE_HEADROOM_MIB


def choose_gpu(snapshot: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    required = required_free_mib()
    candidates = []
    for index, row in snapshot["rows"].items():
        if int(row["free_mib"]) >= max(required, GUARD_MIB):
            candidates.append((int(row["free_mib"]), index, row))
    if not candidates:
        raise RuntimeError(f"no GPU meets guard/profile headroom; required_free_mib={required}")
    _, index, row = max(candidates)
    return f"cuda:{index}", {
        "device": f"cuda:{index}",
        "free_mib": int(row["free_mib"]),
        "utilization_gpu": int(row["utilization_gpu"]),
        "required_free_mib": required,
        "guard_mib": GUARD_MIB,
        "pass": True,
    }


def archive_partial(run_root: Path) -> dict[str, Any]:
    if run_root.exists():
        raise RuntimeError(f"recovery directory already exists: {run_root}")
    binding = verify_source_binding()
    host = host_process_snapshot()
    active_old = [row for row in host.get("rows", []) if "--worker" in row or "--run-all" in row]
    if active_old:
        raise RuntimeError("live E17 runner detected; refusing archive/recovery: " + repr(active_old))

    run_root.mkdir(parents=True, exist_ok=False)
    partial_root = run_root / "partial"
    partial_root.mkdir(parents=True, exist_ok=False)
    archived: list[dict[str, Any]] = []
    for arm in ARMS:
        source = ARTIFACT_ROOT / arm / TASK / "seed_42"
        target = partial_root / arm / TASK / "seed_42"
        if not source.is_dir():
            raise RuntimeError(f"missing partial artifact directory: {source}")
        target.parent.mkdir(parents=True, exist_ok=False)
        shutil.move(str(source), str(target))
        archived.append({"source": str(source), "archive": str(target)})
    log_root = run_root / "partial" / "logs"
    log_root.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        source = ARTIFACT_ROOT / "logs" / f"{arm}__{TASK}.log"
        target = log_root / source.name
        if source.is_file():
            shutil.move(str(source), str(target))
            archived.append({"source": str(source), "archive": str(target)})
    state_source = ARTIFACT_ROOT / "controller_state.json"
    state_target = run_root / "controller_state_snapshot.json"
    shutil.copy2(state_source, state_target)
    archived.append({"source": str(state_source), "archive": str(state_target)})
    dispatch_source = ARTIFACT_ROOT / "source_manifest_dispatch.json"
    if dispatch_source.is_file():
        dispatch_target = run_root / "source_manifest_dispatch_snapshot.json"
        shutil.copy2(dispatch_source, dispatch_target)
        archived.append({"source": str(dispatch_source), "archive": str(dispatch_target)})

    manifest = {
        "protocol": "E17-recipient-excluded-keys-screening-v1",
        "recovery_kind": "fresh_restart_after_nonresumable_interruption",
        "created_at": now(),
        "task": TASK,
        "arms": list(ARMS),
        "seed": 42,
        "epochs": 30,
        "reason_for_fresh_restart": "Frozen E2 loop declares resumable=false and fresh_rerun_only; best_checkpoint is best-epoch state, not a latest checkpoint/resume contract.",
        "archived": archived,
        "source_verification": binding,
        "host_process_snapshot": host,
        "old_controller_state": read_json(state_source),
        "policy": "archive partials; run only these three Hotel identities; preserve 12 completed trials; no test/multiseed/grid/source changes",
    }
    write_json(run_root / "recovery_manifest.json", manifest)
    return manifest


def update_state(path: Path, state: dict[str, Any]) -> None:
    state["heartbeat_at"] = now()
    write_json(path, state)


def run_controller(run_root: Path, device: str | None = None) -> None:
    verification = verify_source_binding()
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "controller_state.json"
    state: dict[str, Any] = {
        "protocol": "E17-recipient-excluded-keys-screening-v1",
        "recovery_kind": "fresh_restart_after_nonresumable_interruption",
        "status": "running",
        "created_at": now(),
        "controller_pid": os.getpid(),
        "source_verification": verification,
        "task": TASK,
        "arms": list(ARMS),
        "seed": 42,
        "epochs": 30,
        "jobs": [],
        "queue_remaining": [{"arm": arm, "task_id": TASK} for arm in ARMS],
        "worker_policy": "sequential; one worker/GPU; source-bound frozen runner",
    }
    update_state(state_path, state)
    for index, arm in enumerate(ARMS):
        snapshot = gpu_snapshot()
        selected_device, gate = choose_gpu(snapshot)
        if device is not None and selected_device != device:
            requested_index = device.split(":", 1)[-1]
            row = snapshot["rows"].get(requested_index)
            if row is None:
                raise RuntimeError(f"requested recovery device absent: {device}")
            required = required_free_mib()
            if int(row["free_mib"]) < max(required, GUARD_MIB):
                raise RuntimeError(f"requested recovery device fails VRAM gate: {device} row={row}")
            selected_device = device
            gate = {
                "device": device,
                "free_mib": int(row["free_mib"]),
                "utilization_gpu": int(row["utilization_gpu"]),
                "required_free_mib": required,
                "guard_mib": GUARD_MIB,
                "pass": True,
            }
        output_dir = ARTIFACT_ROOT / arm / TASK / "seed_42"
        log_path = run_root / "logs" / f"{arm}__{TASK}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            str(PYTHON),
            str(RUNNER),
            "--worker",
            "--arm",
            arm,
            "--task-id",
            TASK,
            "--device",
            selected_device,
            "--output-dir",
            str(output_dir),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        job = {
            "arm": arm,
            "task_id": TASK,
            "device": selected_device,
            "pid": process.pid,
            "command": command,
            "status": "running",
            "started_at": now(),
            "log": str(log_path),
            "output_dir": str(output_dir),
            "vram_gate": gate,
            "gpu_snapshot_at_launch": snapshot,
        }
        state["jobs"].append(job)
        state["active_pid"] = process.pid
        state["active_arm"] = arm
        state["queue_remaining"] = [{"arm": value, "task_id": TASK} for value in ARMS[index + 1 :]]
        update_state(state_path, state)
        log_handle.close()
        while True:
            code = process.poll()
            if code is not None:
                break
            update_state(state_path, state)
            time.sleep(15)
        job["returncode"] = int(code)
        job["status"] = "completed" if code == 0 else "failed"
        job["finished_at"] = now()
        state["active_pid"] = None
        state["active_arm"] = None
        update_state(state_path, state)
        if code != 0:
            state["status"] = "failed"
            state["failure"] = {"arm": arm, "returncode": int(code)}
            update_state(state_path, state)
            raise RuntimeError(f"Hotel worker failed: {job}")
    state["status"] = "completed"
    state["finished_at"] = now()
    state["queue_remaining"] = []
    update_state(state_path, state)


def launch_controller(run_root: Path, device: str | None = None) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "controller.log"
    log_handle = log_path.open("w", encoding="utf-8")
    command = [str(PYTHON), str(Path(__file__).resolve()), "--controller", "--run-root", str(run_root)]
    if device:
        command.extend(["--device", device])
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    log_handle.close()
    result = {"launched_at": now(), "launcher_pid": os.getpid(), "controller_pid": process.pid, "command": command, "log": str(log_path)}
    write_json(run_root / "launch.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_source_binding(), indent=2, sort_keys=True))
    elif args.prepare:
        print(json.dumps(archive_partial(args.run_root), indent=2, sort_keys=True))
    elif args.launch:
        print(json.dumps(launch_controller(args.run_root, args.device), indent=2, sort_keys=True))
    elif args.controller:
        run_controller(args.run_root, args.device)
    else:
        parser.error("choose --verify, --prepare, --launch, or --controller")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
