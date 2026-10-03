#!/usr/bin/env python3
"""Complete the approved five-task grid after all four GPUs were authorized.

The Sentiment 2e-5 job was already launched on GPU 2. This script launches
only the three other missing cells, with one new benchmark on each of GPUs
0, 1, and 3. Existing processes are left untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "xlmr_large_baseline"
PY = ROOT / ".venv-e0-cuda" / "bin" / "python"
LOG = ART / "logs_five_task_extension"
NEW_JOBS = [("absa_hotel", 2e-5, 3), ("vsfc_sentiment", 5e-5, 0), ("absa_hotel", 5e-5, 1)]


def cell_dir(task: str, lr: float) -> Path:
    return ART / "xlmr_large" / task / f"lr_{lr:.0e}" / "seed_42"


def done(task: str, lr: float) -> bool:
    meta = cell_dir(task, lr) / "metadata.json"
    return meta.exists() and json.loads(meta.read_text()).get("status") == "completed"


def free_memory(gpu: int) -> int:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    for line in rows:
        index, free = (part.strip() for part in line.split(",", 1))
        if int(index) == gpu:
            return int(free)
    raise RuntimeError(f"GPU {gpu} unavailable")


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    active = []
    for task, lr, gpu in NEW_JOBS:
        if done(task, lr):
            print(f"SKIP completed {task} lr={lr:.0e}", flush=True)
            continue
        free = free_memory(gpu)
        if free < 20000:
            raise RuntimeError(f"GPU {gpu} only has {free} MiB free; refusing {task} lr={lr:.0e}")
        path = LOG / f"{task}_lr_{lr:.0e}_seed_42.log"
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH="src", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        command = [str(PY), "scripts/xlmr_large_baseline/finetune.py", "--model", "xlmr_large", "--task", task, "--lr", str(lr), "--seed", "42", "--device", "cuda:0", "--micro-batch", "32"]
        with path.open("w") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        active.append((proc, task, lr, gpu))
        print(f"START {task} lr={lr:.0e} GPU={gpu} pid={proc.pid} free_before={free}MiB", flush=True)
    while active:
        for item in list(active):
            proc, task, lr, gpu = item
            code = proc.poll()
            if code is None:
                continue
            active.remove(item)
            if code != 0 or not done(task, lr):
                print(f"FAILED {task} lr={lr:.0e} GPU={gpu} exit={code}", flush=True)
                return 1
            print(f"DONE {task} lr={lr:.0e} GPU={gpu}", flush=True)
        if active:
            time.sleep(15)
    while not done("vsfc_sentiment", 2e-5):
        status = cell_dir("vsfc_sentiment", 2e-5) / "status.json"
        if status.exists() and json.loads(status.read_text()).get("status") in ("failed", "completed_with_check_failure"):
            print("FAILED vsfc_sentiment lr=2e-05 GPU=2", flush=True)
            return 1
        time.sleep(15)
    print("DONE vsfc_sentiment lr=2e-05 GPU=2", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
