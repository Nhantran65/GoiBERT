"""Fine-tune D1 arms with the M1 runner: `--arm TD` (CE, lr 2e-5) or `--arm KD_D --teacher-cache <TD cache>` (CE+KD, lr 5e-5).
Everything else (loop, checks, outputs) is scripts/cdbk_pilot/finetune.py; see common_d1.install()."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_d1  # noqa: E402

common_d1.install()
import finetune  # noqa: E402  (scripts/cdbk_pilot)

finetune.PROTOCOL = common_d1.PROTOCOL_D1

if __name__ == "__main__":
    raise SystemExit(finetune.main())
