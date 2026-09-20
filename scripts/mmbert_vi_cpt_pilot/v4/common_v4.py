"""Shared contract for pilot v4 (mmBERT + Vietnamese informal speech from the ViSoBERT corpus).

v4 keeps the v3 plan's arms/recipe/gate/fine-tunes and replaces the (failed) HPLT informal bucket with the ViSoBERT
comment corpus.  **Licence:** `phucdev/ViSoBERT` is research-only / CC BY-NC 4.0, so the P_inf and C_inf checkpoints of
this pilot are `_NC_research_only`: they must not be released or used commercially.  P_web and O contain no NC data.
Nothing in v1-v3 is modified; v4 writes only under its own data/artifacts directories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V4_DIR = Path(__file__).resolve().parent
PILOT_DIR = V4_DIR.parent
for extra in (PILOT_DIR, PILOT_DIR / "v2", PILOT_DIR / "v3"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import common  # noqa: E402        (v1 shared contract: paths, hashing, tokenizer, benchmarks)
import common_v2 as C2  # noqa: E402
import common_v3 as C3  # noqa: E402

ROOT = common.ROOT
DATA_V4 = ROOT / "data" / "mmbert_vi_cpt_pilot" / "v4"
ART_V4 = ROOT / "artifacts" / "mmbert_vi_cpt_pilot" / "v4"
SEED = common.SEED

VISOBERT_REPO = "phucdev/ViSoBERT"
VISOBERT_REVISION = "74dbc64e92c627c70670dd91344b776c8500a104"
VISOBERT_LICENSE_CARD = "uitnlp/visobert card: research purposes only; SEACrowd/visobert: CC BY-NC 4.0"
LICENSE_NOTE = ("ViSoBERT corpus is research-only / CC BY-NC 4.0. Checkpoints derived from it (P_inf, C_inf) are "
                "research-only, must not be released or used commercially, and their directories carry the suffix "
                "`_NC_research_only`. P_web and O contain no NC data.")
NC_SUFFIX = "_NC_research_only"

ARMS_NC = ("P_inf", "C_inf")
ARMS = ("P_web", "P_inf", "C_inf")
ARMS_WEIGHTS = {"P_web": {"general": 0.95, "wiki": 0.05},
                "P_inf": {"visobert": 0.30, "general": 0.65, "wiki": 0.05}}
C_INF_STAGES = {"stage1": (0.40, 0.10), "stage2": (0.30, 0.20), "stage3": (0.30, 0.667)}   # (token share, informal share)


def write_json(path: Path, payload) -> None:
    common.write_json(path, payload)


def read_json(path: Path):
    return common.read_json(path)


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)
