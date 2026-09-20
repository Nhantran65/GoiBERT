"""Shared contract for the CafeBERT -> BamiBERT representation-distillation pilot
(docs/CAFEBERT_TO_BAMIBERT_REP_DISTILL_PILOT_PLAN.md).

Pinned sources, paths, hashing, text normalisation, the two tokenizers, and the character-span alignment used by
the data gate (§3), the alignment gate (§4) and the relational objective (§5).  Benchmark labels are never read.
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

PROTOCOL = "CafeBERT-BamiBERT-repdistill-pilot-v2"   # v2 = §5 gradient gate amended (plan Phụ lục A); objective, budget, corpus unchanged
SEED = 42
SCRIPT_DIR = ROOT / "scripts" / "cafe_bami_repdistill_pilot"
DATA_DIR = ROOT / "data" / "cafe_bami_repdistill_pilot"
ART_DIR = ROOT / "artifacts" / "cafe_bami_repdistill_pilot"

STUDENT_DIR = ROOT / "models" / "bamibert-base-original"
STUDENT_HF_REPO = "Qualcomm-AI-Research/BamiBERT"
STUDENT_HF_COMMIT = "57bc1340debbe4e348ec549047a763caebe4a977"
STUDENT_WEIGHTS_SHA256 = "c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627"
TEACHER_HF_REPO = "uitnlp/CafeBERT"
TEACHER_HF_COMMIT = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"
TEACHER_TOKENIZER_REPO = "FacebookAI/xlm-roberta-large"
TEACHER_TOKENIZER_COMMIT = "c23d21b0620b635a76227c604d44e43a9f0ee389"

FW2_REPO = "HuggingFaceFW/fineweb-2"
FW2_COMMIT = "af9c13333eb981300149d5ca60a8e9d659b276b9"
FW2_SHARD = "data/vie_Latn/train/000_00000.parquet"
FW2_SHARD_SHA256 = "337df3c387a7d5bc854059605acb9382e3887dc7ab0da958ecfe3f73d6219acf"   # verified in the AS-CPT pilot manifest
FW2_LOCAL = ROOT / "data" / "raw" / "fineweb2_vie" / FW2_SHARD
FW2_LICENSE = "ODC-By 1.0 (dataset card); page-level rights of the crawled sites are not resolved by this pilot; raw web text is not redistributed"

MAX_TOKENS = 256            # both tokenizers, including specials
MIN_UNITS, MAX_UNITS = 8, 96
TOKEN_BUDGET = 30_000_000   # non-pad BamiBERT input tokens per arm
MIN_TRAIN_TOKENS = 31_000_000
GLOBAL_BATCH = 128
LAYER_PAIRS = ((12, 6), (24, 12))   # (teacher hidden_states index, student hidden_states index); index 0 = embeddings
REL_TEMPERATURE = 0.2
REL_LAMBDA = 1.0
MASK_PROB = 0.15
SHINGLE = 8

_WS = re.compile(r"\s+")
UNIT = re.compile(r"[^\W_]+", re.U)          # contiguous letters/digits by Unicode char position; whitespace/punct are boundaries
CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
VI_CHARS = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]")
VI_COMMON = frozenset("và của là có không được cho với các những này để trong một mình rất cũng nhưng tôi thì đã bị hơi ở về từ ra vào khá quá nhiều còn khi đến mà chỉ thấy đi vì người năm như sẽ đó nên theo trên đang hay tại sau đây làm việc".split())


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize(text: str) -> str:
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


def is_vietnamese(text: str) -> bool:
    if CJK.search(text):
        return False
    words = [w.lower() for w in UNIT.findall(text)]
    if not words:
        return False
    dia = len(VI_CHARS.findall(text)) / max(1, len(text))
    common = sum(w in VI_COMMON for w in words) / len(words)
    return dia >= 0.02 or common >= 0.15


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
        digest.update(line.encode("utf-8")); digest.update(b"\n")
    return digest.hexdigest()


def sha256_array(arr) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


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
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def runtime(device: Any = None) -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "hostname": platform.node(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    try:
        import torch, transformers
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


# --------------------------------------------------------------------------- tokenizers / models

def snapshot_dir(repo: str, commit: str) -> Path:
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    if not path.is_dir():
        raise FileNotFoundError(f"missing local HF snapshot {repo}@{commit}: {path}")
    return path


def snapshot_hashes(path: Path) -> dict[str, str]:
    return {p.name: sha256_file(p.resolve()) for p in sorted(path.iterdir()) if p.resolve().is_file()}


def load_student_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT_DIR, local_files_only=True, use_fast=True)
    tok.model_max_length = int(1e30)
    if len(tok) != 20481 or tok.mask_token_id != 20480 or tok.pad_token_id != 1 or tok.bos_token_id != 0 or tok.eos_token_id != 2:
        raise RuntimeError(f"unexpected BamiBERT tokenizer contract: vocab={len(tok)} mask={tok.mask_token_id} pad={tok.pad_token_id}")
    return tok


def load_teacher_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEACHER_TOKENIZER_REPO, revision=TEACHER_TOKENIZER_COMMIT, local_files_only=True, use_fast=True)
    tok.model_max_length = int(1e30)
    if len(tok) != 250002 or tok.mask_token_id != 250001 or tok.pad_token_id != 1 or tok.bos_token_id != 0 or tok.eos_token_id != 2:
        raise RuntimeError(f"unexpected XLM-R tokenizer contract: vocab={len(tok)} mask={tok.mask_token_id} pad={tok.pad_token_id}")
    teacher_spm = snapshot_dir(TEACHER_HF_REPO, TEACHER_HF_COMMIT) / "sentencepiece.bpe.model"
    tok_spm = snapshot_dir(TEACHER_TOKENIZER_REPO, TEACHER_TOKENIZER_COMMIT) / "sentencepiece.bpe.model"
    if sha256_file(teacher_spm.resolve()) != sha256_file(tok_spm.resolve()):
        raise RuntimeError("CafeBERT sentencepiece model differs from the pinned XLM-R-large tokenizer")
    return tok


def tokenizer_manifest() -> dict[str, Any]:
    student_files = {k: v for k, v in directory_hashes(STUDENT_DIR).items() if Path(k).name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}}
    tdir = snapshot_dir(TEACHER_TOKENIZER_REPO, TEACHER_TOKENIZER_COMMIT)
    return {"student": {"source": str(STUDENT_DIR.relative_to(ROOT)), "hf_repo": STUDENT_HF_REPO, "hf_commit": STUDENT_HF_COMMIT, "files": student_files},
            "teacher": {"source": TEACHER_TOKENIZER_REPO, "hf_commit": TEACHER_TOKENIZER_COMMIT, "files": snapshot_hashes(tdir),
                        "note": "uitnlp/CafeBERT ships only sentencepiece.bpe.model (identical to XLM-R-large); the XLM-R-large fast tokenizer is used as in CDBK"}}


def load_student_mlm(attn: str = "sdpa"):
    """RobertaForMaskedLM with the pretrained MLM head; returns (model, checks)."""
    import torch
    from transformers import RobertaForMaskedLM
    if sha256_file(STUDENT_DIR / "model.safetensors") != STUDENT_WEIGHTS_SHA256:
        raise RuntimeError("student weights sha256 differs from the E14-pinned value")
    model, info = RobertaForMaskedLM.from_pretrained(STUDENT_DIR, local_files_only=True, attn_implementation=attn, output_loading_info=True)
    emb = model.roberta.embeddings.word_embeddings.weight
    dec = model.lm_head.decoder.weight
    cfg = model.config
    checks = {
        "architecture": cfg.architectures, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "vocab": cfg.vocab_size,
        "missing_keys": list(info.get("missing_keys", [])), "unexpected_keys": list(info.get("unexpected_keys", [])), "mismatched_keys": [str(k) for k in info.get("mismatched_keys", [])],
        "mlm_head_pretrained": not any(k.startswith("lm_head") for k in info.get("missing_keys", [])),
        "decoder_tied_to_embeddings": bool(dec.data_ptr() == emb.data_ptr()), "pooler_absent": not any("pooler" in n for n, _ in model.named_parameters()),
        "parameters_total": sum(p.numel() for p in model.parameters()), "parameters_encoder": sum(p.numel() for p in model.roberta.parameters()),
        "weights_sha256": STUDENT_WEIGHTS_SHA256,
    }
    unexpected = [k for k in checks["unexpected_keys"] if "pooler" not in k]
    if (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.vocab_size) != (12, 768, 12, 20481) or checks["missing_keys"] or unexpected or checks["mismatched_keys"] or not checks["decoder_tied_to_embeddings"]:
        raise RuntimeError(f"student MLM does not load cleanly: {checks}")
    return model, checks


def load_teacher(attn: str = "sdpa"):
    """Frozen CafeBERT encoder (XLMRobertaModel, no pooler, no task head), eval mode; returns (model, checks)."""
    import torch
    from transformers import XLMRobertaModel
    model, info = XLMRobertaModel.from_pretrained(TEACHER_HF_REPO, revision=TEACHER_HF_COMMIT, local_files_only=True, add_pooling_layer=False, attn_implementation=attn, output_loading_info=True)
    cfg = model.config
    unexpected = [k for k in info.get("unexpected_keys", []) if not (k.startswith("lm_head") or "pooler" in k)]
    checks = {"architecture": cfg.architectures, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads, "vocab": cfg.vocab_size,
              "missing_keys": list(info.get("missing_keys", [])), "unexpected_keys_dropped": list(info.get("unexpected_keys", [])), "mismatched_keys": [str(k) for k in info.get("mismatched_keys", [])],
              "parameters_total": sum(p.numel() for p in model.parameters()), "hf_commit": TEACHER_HF_COMMIT,
              "weights_sha256": sha256_file((snapshot_dir(TEACHER_HF_REPO, TEACHER_HF_COMMIT) / "model.safetensors").resolve())}
    if (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads) != (24, 1024, 16) or checks["missing_keys"] or unexpected or checks["mismatched_keys"]:
        raise RuntimeError(f"teacher does not load cleanly: {checks}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    checks["frozen"] = not any(p.requires_grad for p in model.parameters())
    return model, checks


# --------------------------------------------------------------------------- spans / alignment

def unit_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of contiguous letter/digit runs in NFC text (Unicode-aware)."""
    return [(m.start(), m.end()) for m in UNIT.finditer(text)]


def span_token_index(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    """For each token offset (start,end) return the index of the span it intersects, or -1 (specials/pad/no overlap).

    A token intersecting several spans (rare: tokenizer merges across a boundary) is assigned to the span with the largest
    overlap.  Spans are assumed sorted and non-overlapping."""
    out = [-1] * len(offsets)
    j = 0
    for t, (a, b) in enumerate(offsets):
        if b <= a:                      # special tokens have (0,0)
            continue
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        best, best_ov = -1, 0
        k = j
        while k < len(spans) and spans[k][0] < b:
            ov = min(b, spans[k][1]) - max(a, spans[k][0])
            if ov > best_ov:
                best, best_ov = k, ov
            k += 1
        out[t] = best
    return out


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
