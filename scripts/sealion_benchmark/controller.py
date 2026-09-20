"""Durable two GPU controller for the SEA-LION/CafeBERT benchmark queue."""

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
from scripts.sealion_benchmark import common  # noqa: E402


GPUS = ("0", "1", "2", "3")
GPU_CAPACITY = {"0": 1, "1": 1, "2": 2, "3": 2}


def jobs() -> list[dict[str, Any]]:
    result = []
    for task_id in common.TASKS:
        for lr in common.LEARNING_RATES:
            result.append({"model_kind": "sealion", "task": task_id, "learning_rate": lr})
    # Restaurant CafeBERT is already complete in cdbk_pilot and is reused.
    for task_id in common.TASKS:
        if task_id != "absa_restaurant":
            result.append({"model_kind": "cafebert", "task": task_id, "learning_rate": 2e-5})
    return result


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def status_path(job: dict[str, Any]) -> Path:
    return common.run_dir(job["model_kind"], job["task"], job["learning_rate"]) / "status.json"


def job_key(job: dict[str, Any]) -> str:
    return f"{job['model_kind']}:{job['task']}:lr_{job['learning_rate']:.0e}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--once", action="store_true", help="write queue state and exit")
    args = parser.parse_args()
    common.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    queue = jobs()
    state_path = common.ARTIFACT_ROOT / "controller_state.json"
    controller_started = time.time()
    active: dict[str, dict[str, Any]] = {}
    adopted: set[str] = set()
    # Adoption: controller restart leaves worker processes running.  Attach
    # to their status files and wait for completion instead of launching a
    # duplicate run.
    for job in queue:
        key = job_key(job); path = status_path(job)
        if path.is_file():
            try:
                status = common.read_json(path)
            except Exception:
                status = {}
            pid = int(status.get("pid", 0) or 0)
            if status.get("status") == "running" and pid and alive(pid):
                gpu = status.get("gpu")
                if gpu is None:
                    resolved = common.run_dir(job["model_kind"], job["task"], job["learning_rate"]) / "resolved_config.json"
                    if resolved.is_file():
                        gpu = common.read_json(resolved).get("runtime", {}).get("cuda_visible_devices")
                active[key] = {"job": job, "gpu": str(gpu or "unknown"), "pid": pid, "adopted": True, "started_at": status.get("started_at")}
                adopted.add(key)

    def write_state(status: str) -> None:
        payload = {
            "protocol": common.PROTOCOL, "status": status, "pid": os.getpid(), "started_at": controller_started,
            "updated_at": common.now(), "gpus": list(GPUS), "gpu_capacity": GPU_CAPACITY, "queue_total": len(queue),
            "jobs": [{"key": job_key(j), **j, "status": (common.read_json(status_path(j)).get("status") if status_path(j).is_file() else "pending"), "pid": (common.read_json(status_path(j)).get("pid") if status_path(j).is_file() else None)} for j in queue],
            # Popen and file handles are intentionally excluded; controller
            # state must remain JSON-readable for restart/adoption.
            "active": [{k: value.get(k) for k in ("job", "gpu", "pid", "adopted", "started_at")} for value in active.values()],
            "adopted": sorted(adopted),
        }
        common.write_json(state_path, payload)

    def completed(job: dict[str, Any]) -> bool:
        p = status_path(job)
        if not p.is_file():
            return False
        try:
            value = common.read_json(p)
        except Exception:
            return False
        return value.get("status") == "completed"

    def terminal(job: dict[str, Any]) -> bool:
        p = status_path(job)
        if not p.is_file():
            return False
        try:
            return common.read_json(p).get("status") in {"completed", "completed_with_check_failure", "failed"}
        except Exception:
            return False

    def launch(job: dict[str, Any], gpu: str) -> None:
        key = job_key(job); output_dir = common.run_dir(job["model_kind"], job["task"], job["learning_rate"])
        output_dir.mkdir(parents=True, exist_ok=True)
        log = (output_dir / "train.log").open("a", encoding="utf-8")
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu; env["PYTHONUNBUFFERED"] = "1"; env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model-kind", job["model_kind"], "--task", job["task"], "--learning-rate", str(job["learning_rate"]), "--device", "cuda:0", "--micro-batch", str(args.micro_batch)]
        proc = subprocess.Popen(cmd, cwd=str(common.ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[key] = {"job": job, "gpu": gpu, "pid": proc.pid, "adopted": False, "started_at": common.now(), "process": proc, "log": log}
        common.write_json(output_dir / "status.json", {"status": "running", "model_kind": job["model_kind"], "task_id": job["task"], "learning_rate": job["learning_rate"], "pid": proc.pid, "gpu": gpu, "started_at": common.now()})
        print(f"launched {key} gpu={gpu} pid={proc.pid}", flush=True)

    # Worker mode is handled in this same durable entrypoint so the command
    # line and process ownership stay inspectable.
    if getattr(args, "worker", False):
        raise AssertionError("worker mode parsed below")
    if args.once:
        write_state("planned"); print(json.dumps({"jobs": queue, "adopted": list(adopted)}, indent=2)); return 0
    interrupted = False
    def stop(_sig: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    write_state("running")
    next_job = 0
    while not interrupted:
        # Reap finished children and verify adopted workers via status files.
        for key, record in list(active.items()):
            proc = record.get("process")
            if proc is not None and proc.poll() is None:
                continue
            job = record["job"]; path = status_path(job)
            finished = common.read_json(path).get("status") if path.is_file() else None
            if proc is None and finished == "running" and alive(int(record["pid"])):
                continue
            if proc is not None and finished == "running":
                common.write_json(path, {"status": "failed", "model_kind": job["model_kind"], "task_id": job["task"], "learning_rate": job["learning_rate"], "pid": record["pid"], "finished_at": common.now(), "error": f"worker exited with code {proc.returncode}"})
            if record.get("log") is not None:
                record["log"].close()
            del active[key]
            print(f"finished {key} status={finished}", flush=True)
        # User-authorized device sharing. Leave all unrelated processes intact.
        for gpu in GPUS:
            current = [r for r in active.values() if str(r["gpu"]) == gpu]
            if len(current) >= GPU_CAPACITY[gpu]:
                continue
            pending = [j for j in queue if job_key(j) not in active and not status_path(j).exists()]
            if not pending:
                break
            job = pending[0]
            free = common.gpu_snapshot().get(gpu, {}).get("free_mib", 0)
            needed = 24576 if job["model_kind"] == "cafebert" else 18432
            if free < needed or (job["model_kind"] == "cafebert" and current):
                continue
            launch(job, gpu)
        write_state("running")
        if not active and all(terminal(job) for job in queue):
            break
        time.sleep(max(args.poll_seconds, 2))
    write_state("stopped" if interrupted else "completed")
    return 0


if __name__ == "__main__":
    # argparse worker branch is kept outside the controller parser to avoid
    # exposing implementation details in normal queue usage.
    if "--worker" in sys.argv:
        from scripts.sealion_benchmark.train import main as worker_main
        sys.argv.remove("--worker")
        raise SystemExit(worker_main())
    raise SystemExit(main())
