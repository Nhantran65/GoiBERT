#!/usr/bin/env python3
"""Detached, resumable validation pilot scheduler; at most one own worker per GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "cafe_oof_kd"))
import common as C  # noqa: E402

ART = ROOT / "artifacts" / "bami_trunc_kd"
PY = str(ROOT / ".venv-e0-cuda" / "bin" / "python")
GPUS = tuple(int(x) for x in os.environ.get("BTK_GPUS", "0,1,2,3").split(","))
BUDGET_H = 8.0
MIN_FREE_MIB = 12_000
TASKS = ("vsfc_sentiment", "absa_restaurant")


def jobs():
    specs = [(12, "ce", task, 42) for task in TASKS]
    specs += [(layers, arm, task, seed) for layers, arm in ((12, "kd"), (10, "ce"), (10, "kd"), (8, "ce"), (8, "kd"))
              for task in TASKS for seed in (42, 43, 44)]
    return [{"name": f"L{k}_{arm}_{task}_s{seed}", "layers": k, "arm": arm, "task": task, "seed": seed,
             "out": ART / "runs" / f"L{k}_{arm}" / task / f"seed_{seed}"} for k, arm, task, seed in specs]


def free_mib():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(a): int(b) for a, b in (line.split(",") for line in raw.strip().splitlines())}


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def completed(job):
    path = job["out"] / "metadata.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("status") == "completed"


def adopted(job):
    path = job["out"] / "status.json"
    if not path.exists():
        return None
    st = json.loads(path.read_text())
    if st.get("status") == "running" and pid_alive(st.get("pid")):
        return int(st["pid"]), int(st["gpu"]), float(st.get("started_at_unix", time.time()))
    return None


def reproduction():
    details = {}
    for task in TASKS:
        job = next(j for j in jobs() if j["layers"] == 12 and j["arm"] == "ce" and j["task"] == task)
        if not completed(job):
            return None
        got = json.loads((job["out"] / "metadata.json").read_text())["best_selection_f1"]
        lr = "2e-05" if task == "vsfc_sentiment" else "5e-05"
        ref = ROOT / "artifacts" / "e14" / "trials" / task / f"lr_{lr}" / "seed_42" / "metadata.json"
        expected = json.loads(ref.read_text())["best_selection_f1"]
        details[task] = {"new": got, "e14": expected, "delta_points": 100 * (got - expected),
                         "within_0_3_points": abs(got - expected) <= 0.003}
    C.write_json(ART / "reproduction_gate.json", details)
    return all(d["within_0_3_points"] for d in details.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-accepted-reproduction", action="store_true",
                        help="Root reviewed E14 reproduction and explicitly authorized continuing after its numerical gate")
    args = parser.parse_args()
    if not GPUS or len(set(GPUS)) != len(GPUS) or any(g not in (0, 1, 2, 3) for g in GPUS):
        raise ValueError("BTK_GPUS must contain unique indices from 0,1,2,3")
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    all_jobs = jobs()
    queue = [j for j in all_jobs if not completed(j)]
    active, external, failed = {}, {}, []
    prior = ART / "launcher_summary.json"
    used_h = float(json.loads(prior.read_text()).get("used_gpu_hours", 0)) if prior.exists() else 0.0
    for job in list(queue):
        state = adopted(job)
        if state:
            pid, gpu, started = state
            if gpu in external:
                raise RuntimeError(f"Two adopted jobs claim GPU {gpu}")
            external[gpu] = (pid, job, started)
            queue.remove(job)
            print(f"adopt {job['name']} GPU{gpu} pid={pid}", flush=True)
        elif job["out"].exists() and any(job["out"].iterdir()):
            failed.append({"job": job["name"], "reason": "incomplete_directory_requires_archive"})
            queue.remove(job)
    stopped_reason = None
    while queue or active or external:
        for gpu, (pid, job, started) in list(external.items()):
            if pid_alive(pid):
                continue
            external.pop(gpu)
            used_h += (time.time() - started) / 3600
            if not completed(job):
                failed.append({"job": job["name"], "reason": "adopted_worker_failed"})
            print(f"adopted {'done' if completed(job) else 'FAILED'} {job['name']} GPU{gpu}", flush=True)
        for gpu, (proc, job, started, fh) in list(active.items()):
            if proc.poll() is None:
                continue
            fh.close(); active.pop(gpu)
            used_h += (time.time() - started) / 3600
            if proc.returncode or not completed(job):
                failed.append({"job": job["name"], "reason": f"worker_rc_{proc.returncode}"})
            print(f"{'done' if completed(job) else 'FAILED'} {job['name']} GPU{gpu} rc={proc.returncode} used_h={used_h:.8f}", flush=True)
        gate = reproduction()
        if gate is None and any(f["job"].startswith("L12_ce_") for f in failed):
            stopped_reason = "L12-CE reproduction worker failed"
            queue.clear()
        if gate is False and not args.root_accepted_reproduction:
            stopped_reason = "L12-CE reproduction differs from E14 by >0.3 F1 points"
            queue.clear()
        if used_h + sum((time.time() - v[2]) / 3600 for v in active.values()) + sum((time.time() - v[2]) / 3600 for v in external.values()) >= BUDGET_H:
            stopped_reason = "8 GPU-hour budget reached"
            queue.clear()
        free = free_mib()
        for gpu in GPUS:
            if gpu in active or gpu in external or not queue:
                continue
            # Strict phase barrier: both reproduction jobs must finish and pass before any experimental arm.
            if gate is None:
                ready = [j for j in queue if j["layers"] == 12 and j["arm"] == "ce"]
            elif gate is True or (gate is False and args.root_accepted_reproduction):
                ready = queue
            else:
                ready = []
            if not ready or free.get(gpu, 0) < MIN_FREE_MIB:
                continue
            job = ready[0]; queue.remove(job)
            fh = (ART / "logs" / f"{job['name']}.log").open("a", encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": "src", "HF_HUB_OFFLINE": "1",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"}
            cmd = [PY, "scripts/bami_trunc_kd/train.py", "--task", job["task"], "--layers", str(job["layers"]),
                   "--arm", job["arm"], "--seed", str(job["seed"])]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            active[gpu] = (proc, job, time.time(), fh)
            print(f"launch {job['name']} GPU{gpu} pid={proc.pid}", flush=True)
            time.sleep(20)
            free = free_mib()
        C.write_json(ART / "launcher_state.json", {"status": "running", "used_gpu_hours_finished": used_h,
                     "root_accepted_reproduction": args.root_accepted_reproduction,
                     "active": {str(g): v[1]["name"] for g, v in active.items()},
                     "adopted": {str(g): v[1]["name"] for g, v in external.items()},
                     "queued": [j["name"] for j in queue], "failed": failed, "stopped_reason": stopped_reason})
        if queue or active or external:
            time.sleep(60)
    summary = {"used_gpu_hours": used_h, "failed": failed, "stopped_reason": stopped_reason,
               "completed": sum(completed(j) for j in all_jobs), "expected": len(all_jobs), "gpus": list(GPUS),
               "root_accepted_reproduction": args.root_accepted_reproduction}
    C.write_json(ART / "launcher_summary.json", summary)
    print(f"ALL DONE {summary}", flush=True)
    return 0 if not failed and not stopped_reason and summary["completed"] == len(all_jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
