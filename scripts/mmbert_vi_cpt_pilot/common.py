"""Shared contract for the mmBERT-base + Vietnamese CPT pilot (docs/MMBERT_VI_CPT_PILOT_PLAN.md).

Pinned sources, paths, hashing, Vietnamese text normalisation/content checks, the mmBERT (Gemma 256K) tokenizer and the
`ModernBertForMaskedLM` loader used by the data gate (§3), the preflight (§4), the three CPT arms (§5) and the
fine-tunes (§6).  Benchmark labels are never read: only the text columns of train/dev/test are used, for decontamination.

Helpers marked COPIED are copies of helpers from earlier scripts in this repo; the source file sha256 is recorded by
`helper_provenance()` and written into every manifest.
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

PROTOCOL = "mmbert-vi-cpt-pilot-v1"
SEED = 42

SCRIPT_DIR = ROOT / "scripts" / "mmbert_vi_cpt_pilot"
DATA_DIR = ROOT / "data" / "mmbert_vi_cpt_pilot"
ART_DIR = ROOT / "artifacts" / "mmbert_vi_cpt_pilot"
HPLT_DIR = ROOT / "data" / "raw" / "hplt3_vie"
WIKI_DIR = ROOT / "data" / "raw" / "wikipedia_vi"
FW2_LOCAL = ROOT / "data" / "raw" / "fineweb2_vie" / "data" / "vie_Latn" / "train" / "000_00000.parquet"

# --------------------------------------------------------------------------- pinned sources (plan §3)
HPLT_REPO = "HPLT/HPLT3.0"
HPLT_HF_REVISION = "3394d6ba8dae4da834e3b11771daf95028a960b1"
HPLT_BASE_URL = "https://data.hplt-project.org/three/sorted/vie_Latn"
HPLT_LICENSE = ("CC0 1.0 for the HPLT packaging (dataset card); the pilot does not own the rights to the source pages, "
                "does not resolve page-level rights, and does not redistribute raw text")
WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.vi"
WIKI_LICENSE = "CC BY-SA 4.0 (Wikipedia); attribution kept in the manifest, raw text not redistributed"
FW2_REPO = "HuggingFaceFW/fineweb-2"
FW2_COMMIT = "af9c13333eb981300149d5ca60a8e9d659b276b9"
FW2_SHARD_REL = "data/vie_Latn/train/000_00000.parquet"
FW2_SHARD_SHA256 = "337df3c387a7d5bc854059605acb9382e3887dc7ab0da958ecfe3f73d6219acf"
FW2_LICENSE = "ODC-By 1.0 (dataset card); page-level rights of the crawled sites are not resolved by this pilot; raw web text is not redistributed"

MODEL_REPO = "jhu-clsp/mmBERT-base"
MODEL_COMMIT = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
MODEL_ARCH = "ModernBertForMaskedLM"
MODEL_WEIGHTS = "pytorch_model.bin"
MODEL_PARAMS_TOTAL_M = 307.0
MODEL_PARAMS_NON_EMBED_M = 110.3


def detect_attn_implementation() -> tuple[str, bool]:
    """flash_attention_2 when it is importable, otherwise sdpa; the choice is recorded and shared by every arm."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2", True
    except Exception:  # noqa: BLE001
        return "sdpa", False


ATTN_IMPL, FLASH_AVAILABLE = detect_attn_implementation()

# --------------------------------------------------------------------------- recipe constants (plan §4/§5/§6)
MAX_LEN = 512                 # packed sequence length for CPT
GLOBAL_BATCH = 128            # sequences per optimizer step
TOKEN_BUDGET = 250_000_000    # non-pad tokens per arm
STEPS = round(TOKEN_BUDGET / (GLOBAL_BATCH * MAX_LEN))   # 3815 steps = 250 019 840 non-pad tokens (error << 1 batch)
TOKENS_SCHEDULED = STEPS * GLOBAL_BATCH * MAX_LEN
MASK_PROB = 0.15
CPT_LR = 5e-5
CPT_BETAS = (0.9, 0.98)
CPT_EPS = 1e-6
CPT_WEIGHT_DECAY = 0.01
CPT_CLIP = 1.0
WARMUP_FRACTION = 0.06
DECAY_START_FRACTION = 0.70
VAL_SEED = 1234

FT_EPOCHS = 30
FT_EFFECTIVE_BATCH = 32
FT_MAX_LENGTH = 256
FT_LR = {"absa_restaurant": 5e-5, "visfd": 5e-5, "vsfc_topic": 2e-5}
FT_TASKS = ("absa_restaurant", "visfd", "vsfc_topic")

ARMS = ("P_web", "P_mix", "C_mix")
O_SEED42 = {"absa_restaurant": 0.75626204238921, "visfd": 0.836390977443609, "vsfc_topic": 0.811808487044734}
REFERENCE = {
    "Cafe": {"absa_restaurant": 0.7772, "visfd": 0.8547, "vsfc_topic": 0.8174},
    "Bami": {"absa_restaurant": 0.7474, "visfd": 0.8438, "vsfc_topic": 0.7953},
}

# register mixture (plan §2/§3): general bins 9-10 65 %, register 20 %, bin-8 general 10 %, Wikipedia 5 %
MIX_TARGET = {"general": 0.65, "register": 0.20, "bin8": 0.10, "wiki": 0.05}
WEB_TARGET = {"general": 0.95, "wiki": 0.05}

# --------------------------------------------------------------------------- text utilities
_WS = re.compile(r"\s+")
UNIT = re.compile(r"[^\W_]+", re.U)
CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")
VI_CHARS = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]")
VI_COMMON = frozenset("và của là có không được cho với các những này để trong một mình rất cũng nhưng tôi thì đã bị hơi ở về từ ra vào khá quá nhiều còn khi đến mà chỉ thấy đi vì người năm như sẽ đó nên theo trên đang hay tại sau đây làm việc".split())
SHINGLE = 8
SHINGLE_MOD = 16          # keep shingles whose hash % SHINGLE_MOD == 0 (plan §3)
SHINGLE_BASE = 0x100000001B3          # polynomial rolling-hash base (mod 2**64); odd => invertible
_WORD_HASH_CACHE: dict[str, int] = {}
_POW_CACHE: dict[str, Any] = {"n": 0, "b": None, "binv": None}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize(text: str) -> str:
    """HTML-unescape, NFC, collapse whitespace; diacritics and spelling kept (COPIED: repdistill common.normalize)."""
    return _WS.sub(" ", unicodedata.normalize("NFC", html.unescape(text or ""))).strip()


def key_of(text: str) -> str:
    return normalize(text).casefold()


def words(key: str) -> list[str]:
    return key.split()


def shingles(key: str, n: int = SHINGLE) -> list[str]:
    w = words(key)
    if len(w) < n:
        return []
    return [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]


def sampled_shingle_hashes(key: str, n: int = SHINGLE, mod: int = SHINGLE_MOD) -> list[int]:
    """Sampled (hash % mod == 0) 64-bit shingle hashes of a text key; see shingle_hashes()."""
    arr = shingle_hashes(key, n)
    return [int(x) for x in arr[arr % mod == 0]]


def _word_hash(word: str) -> int:
    cached = _WORD_HASH_CACHE.get(word)
    if cached is None:
        cached = int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "big")
        if len(_WORD_HASH_CACHE) < 2_000_000:
            _WORD_HASH_CACHE[word] = cached
    return cached


def shingle_hashes(key: str, n: int = SHINGLE):
    """All n-word shingle hashes of `key` as a uint64 numpy array.

    Canonical hash = polynomial rolling hash of the per-word 64-bit (blake2b) hashes, base SHINGLE_BASE, mod 2**64
    (vectorised; used for both the near-duplicate check and the benchmark decontamination check).
    """
    import numpy as np
    w = key.split()
    if len(w) < n:
        return np.zeros(0, dtype=np.uint64)
    a = np.fromiter((_word_hash(x) for x in w), dtype=np.uint64, count=len(w))
    b_pow, binv_pow = _powers(len(w))
    with np.errstate(over="ignore"):
        c = np.cumsum((a * binv_pow).astype(np.uint64), dtype=np.uint64)
        c = np.concatenate([np.zeros(1, dtype=np.uint64), c])
        return ((c[n:] - c[:-n]) * b_pow[n - 1:]).astype(np.uint64)


def _powers(length: int):
    """Cached [1, B, B^2, ...] and [1, B^-1, B^-2, ...] modulo 2**64 as uint64 arrays."""
    import numpy as np
    if _POW_CACHE["n"] < length:
        size = max(length, 1 << 16)
        b = np.multiply.accumulate(np.full(size, np.uint64(SHINGLE_BASE), dtype=np.uint64))
        binv = np.multiply.accumulate(np.full(size, np.uint64(pow(SHINGLE_BASE, -1, 1 << 64)), dtype=np.uint64))
        _POW_CACHE.update({"n": size, "b": b, "binv": binv})
    return _POW_CACHE["b"][:length], _POW_CACHE["binv"][:length]


def is_vietnamese(text: str) -> bool:
    """COPIED: scripts/cafe_bami_repdistill_pilot/common.py::is_vietnamese (plan §3 content check)."""
    if CJK.search(text):
        return False
    dia = len(VI_CHARS.findall(text)) / max(1, len(text))
    if dia >= 0.02:                      # identical short-circuit to `dia >= 0.02 or common >= 0.15`
        return True
    w = [x.lower() for x in UNIT.findall(text)]
    if not w:
        return False
    common = sum(x in VI_COMMON for x in w) / len(w)
    return common >= 0.15


def n_units(text: str) -> int:
    return len(UNIT.findall(text))


def doc_id_of(text: str, source: str) -> str:
    return hashlib.sha1(f"{source}:{key_of(text)}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- hashing / io
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def sha256_array(arr: Any) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, default=_json_default)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def now_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# --------------------------------------------------------------------------- tokenizer / model
def snapshot_dir(repo: str, commit: str) -> Path:
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    if not path.is_dir():
        raise FileNotFoundError(f"missing local HF snapshot {repo}@{commit}: {path}")
    return path


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_COMMIT, local_files_only=True, use_fast=True)
    tok.model_max_length = MAX_LEN
    if len(tok) != 256000 or tok.pad_token_id != 0 or tok.eos_token_id != 1 or tok.bos_token_id != 2 or tok.unk_token_id != 3 or tok.mask_token_id != 4:
        raise RuntimeError(f"unexpected mmBERT tokenizer contract: vocab={len(tok)} pad={tok.pad_token_id} eos={tok.eos_token_id} mask={tok.mask_token_id}")
    return tok


def special_ids(tok) -> list[int]:
    return sorted(set(int(x) for x in tok.all_special_ids))


def tokenizer_manifest() -> dict[str, Any]:
    d = snapshot_dir(MODEL_REPO, MODEL_COMMIT)
    names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json", MODEL_WEIGHTS}
    return {"repo": MODEL_REPO, "commit": MODEL_COMMIT, "class": "PreTrainedTokenizerFast (Gemma 256K)",
            "vocab_size": 256000, "special_ids": {"pad": 0, "eos": 1, "bos": 2, "unk": 3, "mask": 4}, "sep_token_id": 1,
            "files": {p.name: sha256_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name in names}}


def load_mlm(attn: str):
    """ModernBertForMaskedLM with pretrained MLM head; returns (model, checks). Plan §4: clean load, tied decoder."""
    from transformers import ModernBertForMaskedLM
    model, info = ModernBertForMaskedLM.from_pretrained(MODEL_REPO, revision=MODEL_COMMIT, local_files_only=True,
                                                        attn_implementation=attn, output_loading_info=True)
    cfg = model.config
    emb = model.model.embeddings.tok_embeddings.weight
    dec = model.decoder.weight
    head_missing = [k for k in info.get("missing_keys", []) if k.startswith("head") or k.startswith("decoder")]
    checks = {"architecture": cfg.architectures, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "heads": cfg.num_attention_heads,
              "vocab": cfg.vocab_size, "max_position_embeddings": cfg.max_position_embeddings, "tie_word_embeddings": bool(cfg.tie_word_embeddings),
              "missing_keys": list(info.get("missing_keys", [])), "unexpected_keys": list(info.get("unexpected_keys", [])),
              "mismatched_keys": [str(k) for k in info.get("mismatched_keys", [])], "mlm_head_pretrained": not head_missing,
              "decoder_tied_to_embeddings": bool(dec.data_ptr() == emb.data_ptr()),
              "parameters_total": sum(p.numel() for p in model.parameters()),
              "parameters_non_embedding": sum(p.numel() for p in model.parameters()) - int(emb.numel()),
              "weights_sha256": sha256_file(snapshot_dir(MODEL_REPO, MODEL_COMMIT) / MODEL_WEIGHTS)}
    if (cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size) != (22, 768, 256000) or checks["missing_keys"] or checks["unexpected_keys"] or checks["mismatched_keys"] or not checks["decoder_tied_to_embeddings"]:
        raise RuntimeError(f"MLM model does not load cleanly: {checks}")
    return model, checks


def load_encoder(path_or_repo: str, attn: str):
    """ModernBertModel encoder (no head) from a local export dir or from the pinned repo; returns (model, info)."""
    from transformers import ModernBertModel
    if path_or_repo == MODEL_REPO:
        return ModernBertModel.from_pretrained(MODEL_REPO, revision=MODEL_COMMIT, local_files_only=True,
                                               attn_implementation=attn, output_loading_info=True)
    return ModernBertModel.from_pretrained(path_or_repo, local_files_only=True, attn_implementation=attn, output_loading_info=True)


def helper_provenance() -> dict[str, str]:
    """sha256 of the earlier repo scripts this pilot copies helpers from (plan §7: copy with recorded hash)."""
    files = {
        "scripts/cafe_bami_repdistill_pilot/common.py": "normalize/key_of/shingles/is_vietnamese/benchmark_texts copies",
        "scripts/cafe_bami_repdistill_pilot/cpt_train.py": "CPT loop / masking / export pattern",
        "scripts/base_encoder_baseline/finetune.py": "E14 fine-tune loop (recipe template)",
        "scripts/base_encoder_baseline/common.py": "pinned mmBERT model/E14 head builder",
        "src/goibert/train.py": "fine-tune data/optimiser/validation helpers (imported, not copied)",
        "src/goibert_e14/modeling.py": "E14 CLS head (imported, not copied)",
    }
    out = {}
    for rel, note in files.items():
        p = ROOT / rel
        out[rel] = {"sha256": sha256_file(p) if p.is_file() else None, "use": note}
    return out


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(SCRIPT_DIR.glob("*.py")) + sorted(SCRIPT_DIR.glob("*.sh")):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def runtime(device: Any = None) -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "hostname": platform.node(),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch_threads": os.environ.get("OMP_NUM_THREADS")}
    try:
        import torch, transformers
        info.update({"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda})
        if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(device)
    except Exception as error:  # noqa: BLE001
        info["torch_error"] = repr(error)
    return info


def gpu_snapshot() -> dict[str, Any]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": repr(error)}
    rows = {}
    for line in out.strip().splitlines():
        f = [v.strip() for v in line.split(",")]
        if len(f) >= 5:
            rows[f[0]] = {"name": f[1], "free_mib": int(float(f[2])), "used_mib": int(float(f[3])), "util_pct": int(float(f[4]))}
    return rows


def memory_gb() -> dict[str, float]:
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = float(v.strip().split()[0]) / 1e6
        return {"total_gb": mem.get("MemTotal", 0.0), "available_gb": mem.get("MemAvailable", 0.0)}
    except OSError:
        return {}


# --------------------------------------------------------------------------- benchmarks (text only)
def benchmark_texts() -> dict[str, list[str]]:
    """COPIED: scripts/cafe_bami_repdistill_pilot/common.py::benchmark_texts — text columns of all 5 benchmarks x
    train/dev/test; label columns are never read (plan §3 decontamination)."""
    import csv
    out: dict[str, list[str]] = {}
    for split in ("train", "dev", "test"):
        p = ROOT / "data/raw/uit-vsfc" / split / "sents.txt"
        out[f"vsfc/{split}"] = [l.rstrip("\n") for l in p.open(encoding="utf-8")]
    for split in ("Train", "Dev", "Test"):
        with (ROOT / "data/raw/uit-visfd" / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
            out[f"visfd/{split}"] = [row["comment"] for row in csv.DictReader(fh)]
        for dom in ("Restaurant", "Hotel"):
            with (ROOT / "data/raw/uit-absa-public-candidate/dataset" / f"UIT-ABSA-{dom}" / f"{split}.csv").open(encoding="utf-8", newline="") as fh:
                out[f"absa_{dom.lower()}/{split}"] = [row["Review"] for row in csv.DictReader(fh)]
    return out


def benchmark_shingle_array():
    """Sorted unique uint64 set of ALL 8-word shingle hashes of every benchmark text (decontamination index)."""
    import numpy as np
    bench = benchmark_texts()
    parts = []
    for name, rows in bench.items():
        for r in rows:
            k = key_of(r)
            if k:
                parts.append(shingle_hashes(k))
    arr = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint64)
    return np.unique(arr)


def benchmark_key_index() -> tuple[dict[str, str], dict[str, Any]]:
    """Exact key hashes of every benchmark text (first-hit provenance) plus counts."""
    bench = benchmark_texts()
    keys: dict[str, str] = {}
    row_counts = {}
    for name, rows in bench.items():
        row_counts[name] = len(rows)
        for r in rows:
            k = key_of(r)
            if k:
                keys.setdefault(hashlib.sha1(k.encode()).hexdigest(), name)
    stats = {"rows_per_split": row_counts, "unique_keys": len(keys), "key_set_sha256": sha256_lines(sorted(keys))}
    return keys, stats
