#!/usr/bin/env python3
"""Run the authorized one-batch smoke check for all three local backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_intent import run_smoke_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in ("goi_dapt", "goi_student", "bamibert"):
        result = run_smoke_model(model_name, args.gpu, args.output)
        results.append(result)
        print(json.dumps({"event": "smoke", **result}, ensure_ascii=False, sort_keys=True), flush=True)
        if not result["smoke_pass"]:
            raise SystemExit(f"smoke failed for {model_name}")
    summary = {
        "gpu": args.gpu,
        "models": results,
        "all_pass": all(result["smoke_pass"] for result in results),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "smoke_complete", "gpu": args.gpu, "all_pass": summary["all_pass"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
