#!/usr/bin/env python3
"""Finish only missing Hotel/Sentiment seed-42 cells on free GPUs 2/3.

One benchmark process per GPU. GPUs occupied by other compute processes are
never selected. This launcher is intentionally independent from the original
three-task run_grid.sh.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "xlmr_large_baseline"
LOG = ART / "logs_five_task_extension"
PY = ROOT / ".venv-e0-cuda" / "bin" / "python"
JOBS = [(task, lr) for lr in (2e-5, 5e-5) for task in ("vsfc_sentiment", "absa_hotel")]
GPU_IDS = (2, 3)


def cell_dir(task: str, lr: float) -> Path:
    return ART / "xlmr_large" / task / f"lr_{lr:.0e}" / "seed_42"


def done(task: str, lr: float) -> bool:
    meta = cell_dir(task, lr) / "metadata.json"
    return meta.exists() and json.loads(meta.read_text()).get("status") == "completed"


def free_gpus() -> set[int]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    busy = {line.strip() for line in apps if line.strip()}
    ready = set()
    for line in rows:
        index, uuid, free = (part.strip() for part in line.split(",", 2))
        if int(index) in GPU_IDS and int(free) >= 20000 and uuid not in busy:
            ready.add(int(index))
    return ready


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    queue = [(task, lr) for task, lr in JOBS if not done(task, lr)]
    active: dict[int, tuple[subprocess.Popen, str, float]] = {}
    while queue or active:
        for gpu, (proc, task, lr) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[gpu]
            if code or not done(task, lr):
                print(f"FAILED {task} lr={lr:.0e} GPU={gpu} exit={code}", flush=True)
                return 1
            print(f"DONE {task} lr={lr:.0e} GPU={gpu}", flush=True)
        for gpu in sorted(free_gpus() - active.keys()):
            if not queue:
                break
            task, lr = queue.pop(0)
            if done(task, lr):
                continue
            path = LOG / f"{task}_lr_{lr:.0e}_seed_42.log"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = "src"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            command = [str(PY), "scripts/xlmr_large_baseline/finetune.py", "--model", "xlmr_large", "--task", task, "--lr", str(lr), "--seed", "42", "--device", "cuda:0", "--micro-batch", "32"]
            with path.open("w") as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (proc, task, lr)
            print(f"START {task} lr={lr:.0e} GPU={gpu} pid={proc.pid}", flush=True)
        if queue or active:
            time.sleep(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
