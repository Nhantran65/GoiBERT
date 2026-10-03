#!/usr/bin/env python3
"""Schedule teacher folds, soft-label builds, student CE/KD and Bami seeds on GPUs 0/1/2 (one own job per GPU)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PY = str(C.ROOT / ".venv-e0-cuda" / "bin" / "python")
GPUS = tuple(int(g) for g in __import__("os").environ.get("OOF_GPUS", "0,1,2").split(","))
BUDGET_H = 14.0
LOGS = C.ART / "logs"


def jobs():
    out = []
    for t in C.TASKS:
        for k in range(C.N_FOLDS):
            out.append({"name": f"teacher_{t}_f{k}", "kind": "teacher", "task": t, "need_mib": 20480,
                        "cmd": ["scripts/cafe_oof_kd/teacher_fold.py", "--task", t, "--fold", str(k)],
                        "done": C.ART / "teacher" / t / f"fold_{k}" / "oof_summary.json"})
    for t in C.TASKS:
        for s in C.SEEDS:
            out.append({"name": f"kd_{t}_s{s}", "kind": "kd", "task": t, "need_mib": 10240,
                        "cmd": ["scripts/cafe_oof_kd/student.py", "--task", t, "--arm", "kd", "--seed", str(s)],
                        "done": C.ART / "student" / "kd" / t / f"seed_{s}" / "metadata.json"})
    for t in C.TASKS:
        for s in C.SEEDS:
            out.append({"name": f"ce_{t}_s{s}", "kind": "ce", "task": t, "need_mib": 10240,
                        "cmd": ["scripts/cafe_oof_kd/student.py", "--task", t, "--arm", "ce", "--seed", str(s)],
                        "done": C.ART / "student" / "ce" / t / f"seed_{s}" / "metadata.json"})
        for s in (43, 44):
            out.append({"name": f"bami_{t}_s{s}", "kind": "bami", "task": t, "need_mib": 10240,
                        "cmd": ["scripts/cafe_oof_kd/bami_seed.py", "--task", t, "--seed", str(s)],
                        "done": C.ART / "bami" / t / f"seed_{s}" / "metadata.json"})
    return out


def free_mib():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(a): int(b) for a, b in (line.split(",") for line in raw.strip().splitlines())}


def pid_alive(pid) -> bool:
    try:
        __import__("os").kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def running_elsewhere(job):
    """(pid, gpu) if this job was started by an earlier launcher and is still training."""
    st = job["done"].parent / "status.json"
    if not st.exists():
        return None
    d = json.loads(st.read_text())
    if d.get("status") != "running" or not pid_alive(d.get("pid")):
        return None
    return int(d["pid"]), int(d["gpu"])


def soft_ready(task):
    return (C.ART / "soft" / f"{task}.npz").exists()


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    queue = [j for j in jobs() if not j["done"].exists()]
    active, failed, used_h = {}, [], 0.0
    external = {}  # gpu -> (pid, job) for jobs adopted from a previous launcher
    for j in list(queue):
        r = running_elsewhere(j)
        if r:
            external[r[1]] = (r[0], j); queue.remove(j)
            print(f"{time.strftime('%H:%M:%S')} adopt running {j['name']} GPU{r[1]} pid={r[0]}", flush=True)
    while queue or active or external:
        for gpu, (pid, job) in list(external.items()):
            if pid_alive(pid):
                continue
            del external[gpu]
            ok = job["done"].exists()
            print(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} adopted {job['name']} GPU{gpu}", flush=True)
            if not ok:
                failed.append(job["name"])
        for gpu, (proc, job, t0, fh) in list(active.items()):
            if proc.poll() is None:
                continue
            fh.close(); del active[gpu]
            used_h += (time.time() - t0) / 3600
            ok = proc.returncode == 0 and job["done"].exists()
            print(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} {job['name']} rc={proc.returncode} GPU{gpu} used={used_h:.2f}h", flush=True)
            if not ok:
                failed.append(job["name"])
        for t in C.TASKS:  # build soft labels once all folds of a task are finished
            if not soft_ready(t) and all((C.ART / "teacher" / t / f"fold_{k}" / "oof_summary.json").exists() for k in range(C.N_FOLDS)):
                rc = subprocess.call([PY, "scripts/cafe_oof_kd/build_soft.py", "--task", t], cwd=C.ROOT,
                                     stdout=open(LOGS / f"build_soft_{t}.log", "w"), stderr=subprocess.STDOUT)
                print(f"{time.strftime('%H:%M:%S')} build_soft {t} rc={rc}", flush=True)
                if rc != 0:
                    failed.append(f"build_soft_{t}")
                    queue = [j for j in queue if not (j["kind"] == "kd" and j["task"] == t)]
        if failed and any(n.startswith("teacher") for n in failed):
            queue = [j for j in queue if not (j["kind"] in ("teacher", "kd") and j["task"] in {n.split("_f")[0][8:] for n in failed if n.startswith("teacher")})]
        if used_h >= BUDGET_H:
            print(f"STOP: budget {used_h:.2f}h >= {BUDGET_H}h; not launching more", flush=True)
            queue = []
        free = free_mib()
        for gpu in GPUS:
            if gpu in active or gpu in external or not queue:
                continue
            ready = [j for j in queue if j["kind"] != "kd" or soft_ready(j["task"])]
            ready = [j for j in ready if free.get(gpu, 0) >= j["need_mib"]]
            if not ready:
                continue
            job = ready[0]; queue.remove(job)
            fh = open(LOGS / f"{job['name']}.log", "a")
            env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": "src", "HF_HUB_OFFLINE": "1"}
            proc = subprocess.Popen([PY, *job["cmd"]], cwd=C.ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            active[gpu] = (proc, job, time.time(), fh)
            print(f"{time.strftime('%H:%M:%S')} launch {job['name']} GPU{gpu} pid={proc.pid}", flush=True)
            time.sleep(20)  # let the job allocate before the next free-memory read
            free = free_mib()
        time.sleep(60)
    C.write_json(C.ART / "launcher_summary.json", {"used_gpu_hours": used_h, "failed": failed})
    print(f"ALL DONE used={used_h:.2f}h failed={failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
