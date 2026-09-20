"""Controller-owned compatibility entrypoint for the frozen CE replay branch.

The frozen scheduler already implements ``stagea_ce_replay`` in
``run_orchestration`` but its public CLI choices omit that stage.  This small
wrapper parses the scheduler's common runtime arguments and constructs the
same Namespace that the scheduler would have received, without changing the
frozen scheduler source or its provenance hash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--root-smoke-gate", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=float, required=True)
    parser.add_argument("--student-memory-guard-gib", type=float, required=True)
    parser.add_argument("--teacher-memory-guard-gib", type=float, required=True)
    parser.add_argument("--nc-memory-guard-gib", type=float, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    return parser


def scheduler_namespace(parsed: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        stage="stagea_ce_replay",
        repo_root=parsed.repo_root.resolve(),
        output_root=parsed.output_root.resolve(),
        devices=list(parsed.devices),
        poll_seconds=float(parsed.poll_seconds),
        student_memory_guard_gib=float(parsed.student_memory_guard_gib),
        teacher_memory_guard_gib=float(parsed.teacher_memory_guard_gib),
        nc_memory_guard_gib=float(parsed.nc_memory_guard_gib),
        root_smoke_gate=parsed.root_smoke_gate.resolve(),
        execute=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    from scripts import execute_e2_validation as scheduler

    try:
        return int(scheduler.run_orchestration(scheduler_namespace(parsed)))
    except scheduler.OrchestrationBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
