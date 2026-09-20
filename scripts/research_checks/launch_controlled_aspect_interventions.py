#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv-e0-cuda" / "bin" / "python"
SCRIPT = Path(__file__).resolve().parent / "controlled_aspect_interventions.py"
OUT = ROOT / "artifacts" / "research_diagnostics" / "controlled_aspect_interventions"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", required=True); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "controlled_aspect_interventions.log"
    command = [str(PYTHON), str(SCRIPT), "--device", args.device]
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                   text=True, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"}, start_new_session=True)
    launch = {"launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "launcher_pid": os.getpid(), "pid": process.pid, "device": args.device, "command": command, "log": str(log)}
    (OUT / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(launch, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
