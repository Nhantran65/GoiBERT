"""Shared contract for pilot v3 ("tiếng Việt văn nói", docs/MMBERT_VI_CPT_PILOT_PLAN_V3_INFORMAL.md).

v3 reuses the v2 pool (v1 filters/dedup/decontam + vá 1 scoring), the v2 spun-filter model files and the v2 audit
labels, and adds (a) a spam-filter fix so that genuine informal Vietnamese is not dropped, (b) an informal-register
bucket and (c) the v1 §4-§5 CPT/fine-tune recipe.  Nothing in v1/v2 is modified: all v3 code lives in this directory,
all v3 outputs under `data/mmbert_vi_cpt_pilot/v3/` and `artifacts/mmbert_vi_cpt_pilot/v3/`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

V3_DIR = Path(__file__).resolve().parent
PILOT_DIR = V3_DIR.parent
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))
V2_DIR = PILOT_DIR / "v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

import common  # noqa: E402        (v1 shared contract)
import common_v2 as C2  # noqa: E402  (v2 paths, host lists, pool table, part_texts)
import spun_filter as S  # noqa: E402  (v2 S1 LM + S2 model)

ROOT = common.ROOT
DATA_V3 = ROOT / "data" / "mmbert_vi_cpt_pilot" / "v3"
ART_V3 = ROOT / "artifacts" / "mmbert_vi_cpt_pilot" / "v3"
SEED = common.SEED

TEENCODE_MAP_PATH = ROOT / "data" / "cpt" / "teencode_map.json"

# Informal markers (plan §3.2 measurement definition)
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u2764]")
PRONOUNS = {"mình", "tui", "tớ", "em", "tao", "t", "m", "e", "mk", "tụi mình", "bọn mình", "chúng mình", "tụi tui",
            "chị", "anh", "các bác", "mọi người", "mn", "ae"}
PRONOUN_RX = re.compile(r"(?<![\w])(?:" + "|".join(sorted(map(re.escape, PRONOUNS), key=len, reverse=True)) + r")(?![\w])", re.I)


def load_teencode() -> dict[str, str]:
    return json.loads(TEENCODE_MAP_PATH.read_text(encoding="utf-8"))


def teencode_pattern(teencode: dict[str, str]) -> re.Pattern:
    keys = sorted(teencode, key=len, reverse=True)
    return re.compile(r"(?<![\w])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![\w])", re.I)


def informal_features(text: str, teen_rx: re.Pattern) -> tuple[int, int, int, bool]:
    """(teencode count, emoji count, first-person pronoun count, informal cue) for one document."""
    n_teen = len(teen_rx.findall(text))
    n_emoji = len(EMOJI.findall(text))
    n_pron = len(PRONOUN_RX.findall(text))
    cue = (n_teen >= 3) or (n_emoji >= 1) or (n_pron >= 3 and n_teen >= 1)
    return n_teen, n_emoji, n_pron, cue


# ------------------------------------------------------------------ v1 pool + ID register
def id_register_by_pool_index() -> dict[int, float]:
    """`web-register.ID` for every v1 pool document (written by v1 stage 2 into pool.jsonl)."""
    out: dict[int, float] = {}
    for i, line in enumerate((common.DATA_DIR / "pool" / "pool.jsonl").open(encoding="utf-8")):
        r = json.loads(line)
        out[i] = float(r.get("id_reg") or 0.0)
    return out


def write_json(path: Path, payload) -> None:
    common.write_json(path, payload)


def read_json(path: Path):
    return common.read_json(path)


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)
