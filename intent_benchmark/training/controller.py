#!/usr/bin/env python3
"""Detached sequential trial controller for one authorized GPU queue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "intent_benchmark/training/training_config.json"
TRAIN = ROOT / "intent_benchmark/training/train_intent.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, choices=[0, 1])
    parser.add_argument("--runs-root", type=Path, default=ROOT / "intent_benchmark/runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    queue = config["queues"][f"gpu{args.gpu}"]
    args.runs_root.mkdir(parents=True, exist_ok=True)
    controller_dir = args.runs_root / f"controller-gpu{args.gpu}"
    controller_dir.mkdir(parents=True, exist_ok=True)
    (controller_dir / "controller.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    events_path = controller_dir / "events.jsonl"
    with events_path.open("a", encoding="utf-8") as events:
        for trial in queue:
            model_name, lr_text = trial.split("@", 1)
            run_name = f"{model_name}-lr{lr_text.replace('-', 'm').replace('.', 'p')}"
            run_dir = args.runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            event = {"event": "queued", "trial": trial, "gpu": args.gpu, "run_dir": str(run_dir), "time": time.time()}
            events.write(json.dumps(event) + "\n")
            events.flush()
            if args.dry_run:
                continue
            log_path = run_dir / "worker.log"
            command = [
                sys.executable,
                str(TRAIN),
                "trial",
                "--model",
                model_name,
                "--lr",
                lr_text,
                "--seed",
                str(config["seed"]),
                "--gpu",
                str(args.gpu),
                "--run-dir",
                str(run_dir),
            ]
            start = {"event": "started", "trial": trial, "gpu": args.gpu, "command": command, "time": time.time()}
            events.write(json.dumps(start) + "\n")
            events.flush()
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                (run_dir / "worker.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
                heartbeat_path = controller_dir / "heartbeat.json"
                last_heartbeat = 0.0
                while True:
                    return_code = process.poll()
                    now = time.time()
                    if return_code is not None:
                        break
                    if now - last_heartbeat >= 15.0:
                        heartbeat_path.write_text(
                            json.dumps(
                                {
                                    "controller_pid": os.getpid(),
                                    "trial": trial,
                                    "gpu": args.gpu,
                                    "worker_pid": process.pid,
                                    "status": "waiting_for_worker",
                                    "updated_at": now,
                                },
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                        last_heartbeat = now
                    time.sleep(1.0)
            finished = {"event": "completed" if return_code == 0 else "failed", "trial": trial, "gpu": args.gpu, "pid": process.pid, "return_code": return_code, "time": time.time()}
            events.write(json.dumps(finished) + "\n")
            events.flush()
            if return_code != 0:
                (run_dir / "FAILED").write_text(json.dumps(finished, sort_keys=True) + "\n", encoding="utf-8")
                (run_dir / "RUNNING").unlink(missing_ok=True)
                raise SystemExit(return_code)
    (controller_dir / "completed").write_text(str(time.time()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
