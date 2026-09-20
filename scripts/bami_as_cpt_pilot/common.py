"""Shared contract for the BamiBERT aspect-sentiment CPT pilot (docs/BAMIBERT_ASPECT_SENTIMENT_CPT_PILOT_PLAN.md).

Paths, pinned sources, hashing, text normalisation and the BamiBERT tokenizer contract used by every stage
(data gate -> lexicon -> CPT R/S -> fine-tune -> summary).  Nothing here touches benchmark labels.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import platform
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROTOCOL = "BamiBERT-AS-CPT-pilot-v1"
SEED = 42
SCRIPT_DIR = ROOT / "scripts" / "bami_as_cpt_pilot"
DATA_DIR = ROOT / "data" / "bami_as_cpt_pilot"
RAW_DIR = DATA_DIR / "raw"
ART_DIR = ROOT / "artifacts" / "bami_as_cpt_pilot"
MODEL_DIR = ROOT / "models" / "bamibert-base-original"
BAMI_HF_REPO = "Qualcomm-AI-Research/BamiBERT"
BAMI_HF_COMMIT = "57bc1340debbe4e348ec549047a763caebe4a977"
BAMI_WEIGHTS_SHA256 = "c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627"  # E14

V3_REPO = "phamluan/vn_msa_sar_v3"
V3_REVISION = "6f3f17b390dcf1bf411804e1bc8246bf88ea1bf4"
V3_TRAIN_FILE = "data/train-00000-of-00001.parquet"
V3_TRAIN_SHA256 = "c7fe8ab15d5023c3a618a9fa94bfa3ea4db61fb02dcb1fdccb9c2f6d36601ce1"  # LFS sha from the Hub listing
V3_LOCAL = RAW_DIR / "vn_msa_sar_v3" / V3_TRAIN_FILE
V3_LICENSE = "CC BY 4.0 (dataset card); attribution: phamluan/vn_msa_sar_v3 @ 6f3f17b3"

FW2_REPO = "HuggingFaceFW/fineweb-2"
FW2_COMMIT = "af9c13333eb981300149d5ca60a8e9d659b276b9"  # from data/raw/fineweb2_vie/.cache/huggingface/download/*.metadata (Pilot B download)
FW2_SHARD = "data/vie_Latn/train/000_00000.parquet"
FW2_SHARD_SHA256 = "337df3c387a7d5bc854059605acb9382e3887dc7ab0da958ecfe3f73d6219acf"
FW2_LOCAL = ROOT / "data" / "raw" / "fineweb2_vie" / FW2_SHARD
FW2_LICENSE = "ODC-By 1.0 (dataset card); underlying page rights unresolved -> pilot does not redistribute raw text"

MAX_TOKENS = 256           # BamiBERT tokens per CPT sequence (incl. specials)
CPT_TOKEN_BUDGET = 20_000_000
MAX_PASSES = 5
MIN_TRAIN_TOKENS = 4_000_000
MIN_DOMAIN_TOKENS = 500_000
MANUAL_AUDIT_MIN_FRACTION = 0.70
SHINGLE = 8

_WS = re.compile(r"\s+")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize(text: str) -> str:
    """HTML unescape, NFC, collapse whitespace; keep diacritics / casing."""
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    return _WS.sub(" ", text).strip()


def key_of(text: str) -> str:
    return normalize(text).casefold()


def shingles(key: str, n: int = SHINGLE) -> list[str]:
    words = key.split()
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def directory_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(SCRIPT_DIR.glob("*.py")):
        digest.update(path.name.encode()); digest.update(sha256_file(path).encode("ascii"))
    for rel in ("goibert/train.py", "goibert/data.py", "goibert/metrics.py", "goibert_e14/modeling.py"):
        path = SRC / rel
        if path.is_file():
            digest.update(rel.encode()); digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=_json_default)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def runtime(device: Any = None) -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "hostname": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    try:
        import torch
        import transformers
        info.update({"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda})
        if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(device)
    except Exception as error:  # noqa: BLE001
        info["torch_error"] = repr(error)
    return info


def gpu_snapshot() -> dict[str, dict[str, int]]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": repr(error)}  # type: ignore[return-value]
    result = {}
    for line in out.strip().splitlines():
        f = [v.strip() for v in line.split(",")]
        if len(f) >= 4:
            result[f[0]] = {"free_mib": int(float(f[1])), "used_mib": int(float(f[2])), "util_pct": int(float(f[3]))}
    return result


def load_bami_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    tok.model_max_length = int(1e30)
    if len(tok) != 20481 or tok.mask_token_id != 20480 or tok.pad_token_id != 1 or tok.bos_token_id != 0 or tok.eos_token_id != 2:
        raise RuntimeError(f"unexpected BamiBERT tokenizer contract: vocab={len(tok)} mask={tok.mask_token_id} pad={tok.pad_token_id}")
    return tok


def benchmark_texts() -> dict[str, list[str]]:
    """Text columns only of all 5 benchmarks x 3 splits; label columns are never read (same reader as D1)."""
    import csv
    out: dict[str, list[str]] = {}
    for split in ("train", "dev", "test"):
        out[f"vsfc/{split}"] = [l.rstrip("\n") for l in (ROOT / "data/raw/uit-vsfc" / split / "sents.txt").open(encoding="utf-8")]
    for split in ("Train", "Dev", "Test"):
        with (ROOT / "data/raw/uit-visfd" / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
            out[f"visfd/{split}"] = [row["comment"] for row in csv.DictReader(fh)]
        for dom in ("Restaurant", "Hotel"):
            with (ROOT / "data/raw/uit-absa-public-candidate/dataset" / f"UIT-ABSA-{dom}" / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
                out[f"absa_{dom.lower()}/{split}"] = [row["Review"] for row in csv.DictReader(fh)]
    return out
