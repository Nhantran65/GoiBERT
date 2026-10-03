#!/usr/bin/env python3
"""Validation-only three-task screen, one pilot job on each of GPUs 0 and 1."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "cafe_cross_head"
PY = ROOT / ".venv-e0-cuda" / "bin" / "python"
TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
ARMS = ("mha", "diagonal", "full")
JOBS = [(task, arm) for task in TASKS for arm in ARMS]
GPUS = (0, 1)


def output_dir(task: str, arm: str) -> Path:
    return ART / arm / task / "lr_2e-05" / "seed_42"


def completed(task: str, arm: str) -> bool:
    path = output_dir(task, arm) / "metadata.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "completed"


def gpu_free_mib() -> dict[int, int]:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
    return {int(i.strip()): int(free.strip()) for i, free in (row.split(",", 1) for row in rows)}


def main() -> int:
    log_dir = ART / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = [(task, arm) for task, arm in JOBS if not completed(task, arm)]
    active: dict[int, tuple[subprocess.Popen, str, str]] = {}
    failures = []
    while queue or active:
        for gpu, (proc, task, arm) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[gpu]
            if code or not completed(task, arm):
                failures.append((task, arm, gpu, code))
                print(f"FAILED {task}/{arm} GPU={gpu} exit={code}", flush=True)
            else:
                print(f"DONE {task}/{arm} GPU={gpu}", flush=True)
        if not failures:
            free = gpu_free_mib()
            for gpu in GPUS:
                if not queue or gpu in active or free.get(gpu, 0) < 20000:
                    continue
                task, arm = queue.pop(0)
                out = output_dir(task, arm)
                if any((out / name).exists() for name in ("epoch_curves.json", "best_checkpoint.pt", "metadata.json")):
                    failures.append((task, arm, gpu, "partial artifact requires review"))
                    print(f"REFUSE partial artifact {out}", flush=True)
                    break
                env = dict(os.environ)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH="src", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
                cmd = [str(PY), "scripts/cafe_cross_head/train.py", "--model-kind", arm, "--task", task, "--learning-rate", "2e-5", "--device", "cuda:0", "--micro-batch", "32"]
                log_path = log_dir / f"{task}_{arm}_seed42.log"
                with log_path.open("w") as handle:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[gpu] = (proc, task, arm)
                print(f"START {task}/{arm} GPU={gpu} pid={proc.pid} free_before={free[gpu]}MiB", flush=True)
        if queue or active:
            time.sleep(15)
        if failures and not active:
            break
    if failures:
        print(f"SCREEN INCOMPLETE: {failures}", flush=True)
        return 1
    print("SCREEN COMPLETE: 9/9 cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
