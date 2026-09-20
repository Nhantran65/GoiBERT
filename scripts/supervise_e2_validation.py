"""Keep the authorized E2 validation queue alive across agent/terminal turns.

This is a small runtime supervisor around ``execute_e2_recovery_v2.py``.  It
does not alter the frozen E2 recipe or scheduler.  The child is launched in a
new session, its output is written to a durable log, and a JSON heartbeat is
updated independently of the scheduler's legacy state file.

If this supervisor is restarted while the recovery child is still alive, it
attaches to that child instead of launching a duplicate queue.  A stale child
PID is never trusted on its own: its command line is checked before attach.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "E2-runtime-supervisor-v3"
REVISION = "e2-20260908-r1"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def pid_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def snapshot(output_root: Path) -> dict[str, Any]:
    """Summarize scheduler state without treating it as a live heartbeat."""

    state_path = output_root / ".orchestrator" / "state.json"
    state = read_json(state_path)
    if not isinstance(state, dict):
        return {"state_path": str(state_path), "state_available": False}
    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        jobs = []
    counts: dict[str, int] = {}
    for job in jobs:
        if isinstance(job, dict):
            status = str(job.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
    approved = state.get("approved")
    approved_counts = {
        str(stage): len(rows) if isinstance(rows, list) else 0
        for stage, rows in approved.items()
    } if isinstance(approved, dict) else {}
    return {
        "state_path": str(state_path),
        "state_available": True,
        "job_counts": counts,
        "approved_counts": approved_counts,
        "last_state_update": state.get("updated_at"),
    }


def build_command(
    *,
    repo_root: Path,
    output_root: Path,
    root_smoke_gate: Path,
    devices: Sequence[str],
    poll_seconds: float,
) -> list[str]:
    return [
        sys.executable,
        str(repo_root / "scripts" / "execute_e2_recovery_v2.py"),
        "--stage",
        "all",
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(output_root),
        "--root-smoke-gate",
        str(root_smoke_gate),
        "--devices",
        *devices,
        "--poll-seconds",
        str(poll_seconds),
        "--student-memory-guard-gib",
        "5",
        "--teacher-memory-guard-gib",
        "5",
        "--nc-memory-guard-gib",
        "5",
        "--execute",
    ]


def valid_existing_child(status: Any) -> tuple[int, str] | None:
    if not isinstance(status, dict):
        return None
    pid = status.get("child_pid")
    if not isinstance(pid, int) or not pid_alive(pid):
        return None
    command = pid_cmdline(pid)
    if "execute_e2_recovery_v2.py" not in command:
        raise RuntimeError(
            f"status points to live pid {pid}, but command is not the E2 recovery child: {command!r}"
        )
    return pid, command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "e2")
    parser.add_argument(
        "--root-smoke-gate",
        type=Path,
        default=ROOT / "artifacts" / "e2" / "preflight" / "smoke_gate.json",
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    controller_root = output_root / ".controller"
    controller_root.mkdir(parents=True, exist_ok=True)
    status_path = controller_root / "recovery_v3_status.json"
    lock_path = controller_root / "recovery_v3.lock"
    log_path = controller_root / "recovery_v3_child.log"

    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("BLOCKED: another E2 v3 supervisor holds the lock", file=sys.stderr)
        return 2

    previous = read_json(status_path)
    existing = valid_existing_child(previous)
    requested_command = build_command(
        repo_root=repo_root,
        output_root=output_root,
        root_smoke_gate=args.root_smoke_gate.resolve(),
        devices=args.devices,
        poll_seconds=args.poll_seconds,
    )
    command: list[str]
    child: subprocess.Popen[str] | None = None
    child_pid: int
    attached = False
    if existing is not None:
        child_pid, existing_command = existing
        # Keep the actual child command visible.  It may belong to a previous
        # supervisor revision; the new no-fresh-rerun policy is recorded as
        # ``next_launch_command`` and applies if this child ever exits.
        command = shlex.split(existing_command)
        attached = True
    else:
        command = requested_command
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        log.write(f"\n[{now()}] supervisor launch\n$ {' '.join(command)}\n")
        log.flush()
        child = subprocess.Popen(
            command,
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": os.pathsep.join((str(repo_root / "src"), str(repo_root)))},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        child_pid = child.pid

    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "running",
        "supervisor_pid": os.getpid(),
        "child_pid": child_pid,
        "child_attached": attached,
        "command": command,
        "child_cmdline_actual": pid_cmdline(child_pid).strip(),
        "next_launch_command": requested_command,
        "child_log": str(log_path),
        "started_at": (previous.get("started_at") if attached and isinstance(previous, dict) else now()),
        "heartbeat_at": now(),
        "include_test": False,
        "supervisor_source_sha256": sha256_file(Path(__file__).resolve()),
        "previous_supervisor_source_sha256": (
            previous.get("supervisor_source_sha256") if isinstance(previous, dict) else None
        ),
        "recovery_script_sha256": sha256_file(repo_root / "scripts" / "execute_e2_recovery_v2.py"),
        "snapshot": snapshot(output_root),
    }
    write_json(status_path, payload)

    try:
        while True:
            if child is not None:
                return_code = child.poll()
            else:
                return_code = None if pid_alive(child_pid) else 0
                if return_code is None:
                    time.sleep(max(1.0, args.heartbeat_seconds))
                    payload["heartbeat_at"] = now()
                    payload["snapshot"] = snapshot(output_root)
                    write_json(status_path, payload)
                    continue
                # An attached child may have exited before this supervisor
                # attached.  The recovery scheduler's final state/log is the
                # source of the exact return code in that rare case.
                return_code = 0

            payload["heartbeat_at"] = now()
            payload["snapshot"] = snapshot(output_root)
            if return_code is None:
                write_json(status_path, payload)
                time.sleep(max(1.0, args.heartbeat_seconds))
                continue
            payload.update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "returncode": return_code,
                    "ended_at": now(),
                    "heartbeat_at": now(),
                    "snapshot": snapshot(output_root),
                }
            )
            write_json(status_path, payload)
            return int(return_code)
    finally:
        try:
            if child is not None:
                # The child owns its own session and is intentionally left
                # alive if this supervisor exits unexpectedly.
                pass
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
