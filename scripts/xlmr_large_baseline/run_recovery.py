#!/usr/bin/env python3
"""Retry two CUDA-failed cells in separate artifact dirs on GPUs 1 and 2."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "xlmr_large_baseline"
PY = ROOT / ".venv-e0-cuda" / "bin" / "python"
JOBS = [("absa_hotel", 2e-5, 1), ("vsfc_sentiment", 5e-5, 2)]


def cell_dir(task: str, lr: float, recovery: bool = False) -> Path:
    base = ART / "recovery" if recovery else ART / "xlmr_large"
    return base / task / f"lr_{lr:.0e}" / "seed_42"


def status(task: str, lr: float, recovery: bool = False) -> str | None:
    path = cell_dir(task, lr, recovery) / "status.json"
    return json.loads(path.read_text()).get("status") if path.exists() else None


def free_memory(gpu: int) -> int:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
    for line in rows:
        index, free = (part.strip() for part in line.split(",", 1))
        if int(index) == gpu:
            return int(free)
    raise RuntimeError(f"GPU {gpu} unavailable")


def main() -> int:
    logs = ART / "logs_five_task_extension"
    logs.mkdir(parents=True, exist_ok=True)
    active = []
    for task, lr, gpu in JOBS:
        if status(task, lr, recovery=True) == "completed":
            print(f"SKIP completed recovery {task} lr={lr:.0e}", flush=True)
            continue
        if gpu == 2:
            while status("vsfc_sentiment", 2e-5) == "running":
                time.sleep(15)
            if status("vsfc_sentiment", 2e-5) != "completed":
                raise RuntimeError("Sentiment 2e-5 did not complete on GPU 2; refusing to stack recovery")
        free = free_memory(gpu)
        if free < 20000:
            raise RuntimeError(f"GPU {gpu} only has {free} MiB free; refusing recovery {task}")
        out = cell_dir(task, lr, recovery=True)
        if out.joinpath("best_checkpoint.pt").exists() or out.joinpath("metadata.json").exists():
            raise FileExistsError(out)
        log_path = logs / f"recovery_{task}_lr_{lr:.0e}_seed_42.log"
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH="src", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [str(PY), "scripts/xlmr_large_baseline/finetune.py", "--model", "xlmr_large", "--task", task, "--lr", str(lr), "--seed", "42", "--device", "cuda:0", "--micro-batch", "32", "--out-dir", str(out)]
        with log_path.open("w") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        active.append((proc, task, lr, gpu))
        print(f"START recovery {task} lr={lr:.0e} GPU={gpu} pid={proc.pid} free_before={free}MiB", flush=True)
    while active:
        for item in list(active):
            proc, task, lr, gpu = item
            code = proc.poll()
            if code is None:
                continue
            active.remove(item)
            if code or status(task, lr, recovery=True) != "completed":
                print(f"FAILED recovery {task} lr={lr:.0e} GPU={gpu} exit={code}", flush=True)
                return 1
            print(f"DONE recovery {task} lr={lr:.0e} GPU={gpu}", flush=True)
        if active:
            time.sleep(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
