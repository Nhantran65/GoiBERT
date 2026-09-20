"""Shared contract for the CafeBERT -> BamiBERT KD pilot (docs/CAFEBERT_DAPT_BAMIBERT_KD_PLAN.md, Milestone 1).

Three arms on absa_restaurant, seed 42, train/validation only:

* T: CafeBERT (uitnlp/CafeBERT, XLM-R-large backbone, H=1024), CE, lr 2e-5.
* B: original BamiBERT (models/bamibert-base-original, H=768), CE, lr 5e-5.
* K: original BamiBERT, (1-alpha) CE + alpha * T**2 KL(teacher || student) from a T logit cache, lr 5e-5.

Everything else is the frozen E14 recipe: 30 epochs, effective batch 32, max_len 256, AdamW wd 0.01,
10 % warmup + linear decay, clip 1.0, bf16 autocast + SDPA, CLS -> dropout .1 -> dense H -> tanh ->
dropout .1 -> 48 logits, best epoch by validation pair micro-F1.  The student arms build their model
through the exact E14 code path so B and K share head init, data order and optimizer.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer, XLMRobertaModel  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e14.modeling import BamiBERTClassifier, build_bamibert_classifier  # noqa: E402

PROTOCOL = "CDBK-pilot-M1-restaurant-v1"
ARTIFACT_ROOT = ROOT / "artifacts" / "cdbk_pilot"
RUN_ROOT = ARTIFACT_ROOT / "runs"
CACHE_ROOT = ARTIFACT_ROOT / "teacher_cache"
CHECK_ROOT = ARTIFACT_ROOT / "checks"

TASK_ID = "absa_restaurant"
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
GRADIENT_CLIP_NORM = 1.0
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5

STUDENT_DIR = ROOT / "models" / "bamibert-base-original"
STUDENT_HF_REPO = "Qualcomm-AI-Research/BamiBERT"
STUDENT_HF_COMMIT = "57bc1340debbe4e348ec549047a763caebe4a977"  # as recorded by E14
TEACHER_HF_REPO = "uitnlp/CafeBERT"
TEACHER_HF_COMMIT = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"
TEACHER_TOKENIZER_REPO = "FacebookAI/xlm-roberta-large"
TEACHER_TOKENIZER_COMMIT = "c23d21b0620b635a76227c604d44e43a9f0ee389"

ARMS: dict[str, dict[str, Any]] = {
    "T": {"init": "cafebert", "learning_rate": 2e-5, "kd": False, "head_hidden": 1024},
    "B": {"init": "bamibert", "learning_rate": 5e-5, "kd": False, "head_hidden": 768},
    "K": {"init": "bamibert", "learning_rate": 5e-5, "kd": True, "head_hidden": 768},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return e1_train._json_default(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def source_hash() -> str:
    paths = [
        *sorted((ROOT / "scripts" / "cdbk_pilot").glob("*.py")),
        SRC / "goibert" / "train.py",
        SRC / "goibert" / "data.py",
        SRC / "goibert" / "metrics.py",
        SRC / "goibert_e14" / "modeling.py",
        SRC / "goibert_e2" / "losses.py",
        SRC / "goibert_e2" / "cache.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.resolve().relative_to(ROOT)).encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def runtime(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": PRECISION,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "hostname": platform.node(),
    }


def gpu_snapshot() -> dict[str, dict[str, int]]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": repr(error)}  # type: ignore[return-value]
    result: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) >= 4:
            result[fields[0]] = {"free_mib": int(float(fields[1])), "used_mib": int(float(fields[2])), "util_pct": int(float(fields[3]))}
    return result


# --------------------------------------------------------------------------- data


def task_contract() -> dict[str, Any]:
    bundle = e1_train.load_task_bundle(TASK_ID, ROOT, include_test=False)
    counts = e1_train.task_aspect_class_counts(bundle)
    num_aspects = e1_train.task_num_aspects(bundle)
    if num_aspects != 12 or tuple(counts) != (4,) * 12:
        raise RuntimeError(f"Restaurant contract mismatch: {num_aspects} slots, counts {counts}")
    return {
        "bundle": bundle,
        "train": e1_train.split_field(bundle, "train"),
        "validation": e1_train.split_field(bundle, "validation"),
        "num_aspects": num_aspects,
        "aspect_class_counts": tuple(counts),
        "num_classes": e1_train.task_num_classes(bundle),
    }


# --------------------------------------------------------------------------- tokenizer / model


def _snapshot_dir(repo: str, commit: str) -> Path:
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    if not path.is_dir():
        raise FileNotFoundError(f"Missing local HF snapshot for {repo}@{commit}: {path}")
    return path


def _snapshot_hashes(path: Path) -> dict[str, str]:
    # snapshot entries are symlinks into blobs/; hash the resolved file content
    return {p.name: sha256_file(p.resolve()) for p in sorted(path.iterdir()) if p.resolve().is_file()}


def load_tokenizer(init: str) -> tuple[Any, dict[str, Any]]:
    if init == "bamibert":
        tokenizer = AutoTokenizer.from_pretrained(STUDENT_DIR, local_files_only=True, use_fast=True)
        hashes = directory_hashes(STUDENT_DIR)
        manifest = {
            "source": str(STUDENT_DIR.relative_to(ROOT)),
            "files": {k: v for k, v in hashes.items() if Path(k).name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"}},
        }
    elif init == "cafebert":
        tokenizer = AutoTokenizer.from_pretrained(
            TEACHER_TOKENIZER_REPO, revision=TEACHER_TOKENIZER_COMMIT, local_files_only=True, use_fast=True
        )
        tok_dir = _snapshot_dir(TEACHER_TOKENIZER_REPO, TEACHER_TOKENIZER_COMMIT)
        teacher_dir = _snapshot_dir(TEACHER_HF_REPO, TEACHER_HF_COMMIT)
        tok_hashes = _snapshot_hashes(tok_dir)
        teacher_spm = teacher_dir / "sentencepiece.bpe.model"
        teacher_spm_hash = sha256_file(teacher_spm.resolve()) if teacher_spm.exists() else None
        manifest = {
            "source": TEACHER_TOKENIZER_REPO,
            "hf_commit": TEACHER_TOKENIZER_COMMIT,
            "files": tok_hashes,
            "cafebert_sentencepiece_sha256": teacher_spm_hash,
            "sentencepiece_identical_to_cafebert": teacher_spm_hash == tok_hashes.get("sentencepiece.bpe.model"),
            "reason": "uitnlp/CafeBERT ships only sentencepiece.bpe.model (identical to XLM-R-large); the XLM-R-large fast tokenizer is used, as in the CafeBERT probe",
        }
        if not manifest["sentencepiece_identical_to_cafebert"]:
            raise RuntimeError("CafeBERT sentencepiece model differs from XLM-R-large tokenizer; refusing to proceed")
    else:
        raise ValueError(init)
    manifest["tokenizer_class"] = type(tokenizer).__name__
    manifest["vocab_size"] = int(len(tokenizer))
    return tokenizer, manifest


def build_model(init: str, contract: Mapping[str, Any]) -> tuple[BamiBERTClassifier, dict[str, Any]]:
    """Return (model, spec).  Student path == E14 builder so head init / RNG consumption match E14."""

    num_classes = int(contract["num_classes"])
    num_aspects = int(contract["num_aspects"])
    counts = tuple(contract["aspect_class_counts"])
    if init == "bamibert":
        model, spec = build_bamibert_classifier(
            STUDENT_DIR, num_classes=num_classes, num_aspects=num_aspects, aspect_class_counts=counts,
            attention_implementation=ATTENTION_IMPLEMENTATION,
        )
        spec_dict = spec.to_dict()
        spec_dict.update({"hf_commit": STUDENT_HF_COMMIT, "local_path": str(STUDENT_DIR.relative_to(ROOT)),
                          "weight_file_sha256": sha256_file(STUDENT_DIR / "model.safetensors")})
        return model, spec_dict
    if init == "cafebert":
        encoder = XLMRobertaModel.from_pretrained(
            TEACHER_HF_REPO, revision=TEACHER_HF_COMMIT, local_files_only=True, add_pooling_layer=False,
            attn_implementation=ATTENTION_IMPLEMENTATION,
        )
        config = encoder.config
        expected = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}
        actual = {name: int(getattr(config, name)) for name in expected}
        if actual != expected:
            raise RuntimeError(f"CafeBERT encoder architecture mismatch: expected {expected}, got {actual}")
        model = BamiBERTClassifier(
            encoder, num_labels=num_classes, num_aspects=num_aspects, aspect_class_counts=counts,
            head_hidden_size=int(config.hidden_size),
        )
        spec_dict = model.spec().to_dict()
        # BamiBERTClassifier.spec() hardcodes the BamiBERT repo name; record the true source.
        snapshot = _snapshot_dir(TEACHER_HF_REPO, TEACHER_HF_COMMIT)
        spec_dict.update({
            "backbone": TEACHER_HF_REPO, "hf_commit": TEACHER_HF_COMMIT, "encoder_class": type(encoder).__name__,
            "weight_file_sha256": sha256_file((snapshot / "model.safetensors").resolve()),
            "config_name_or_path": getattr(config, "_name_or_path", None),
            "note": "head class reused from goibert_e14.modeling with hidden 1024; pooler dropped; MLM head dropped",
        })
        return model, spec_dict
    raise ValueError(init)


def make_loaders(tokenizer: Any, contract: Mapping[str, Any], micro_batch_size: int):
    bundle = contract["bundle"]
    train_split, valid_split = contract["train"], contract["validation"]
    train_loader = e1_train._make_loader(
        tokenizer, train_split, max_length=MAX_LENGTH, batch_size=micro_batch_size, shuffle=True, seed=SEED,
        labels=e1_train.model_labels(bundle, "train", train_split), gold_labels=e1_train.split_labels(train_split),
    )
    valid_loader = e1_train._make_loader(
        tokenizer, valid_split, max_length=MAX_LENGTH, batch_size=micro_batch_size, shuffle=False, seed=SEED,
        labels=e1_train.model_labels(bundle, "validation", valid_split), gold_labels=e1_train.split_labels(valid_split),
    )
    return train_loader, valid_loader


def ordered_train_loader(tokenizer: Any, contract: Mapping[str, Any], batch_size: int):
    bundle, train_split = contract["bundle"], contract["train"]
    return e1_train._make_loader(
        tokenizer, train_split, max_length=MAX_LENGTH, batch_size=batch_size, shuffle=False, seed=SEED,
        labels=e1_train.model_labels(bundle, "train", train_split), gold_labels=e1_train.split_labels(train_split),
    )


def run_dir(arm: str) -> Path:
    return RUN_ROOT / f"{arm}_{TASK_ID}_seed{SEED}"


def load_model_checkpoint(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Checkpoint missing model state: {path}")
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    return {k: v for k, v in state.items() if k != "model"}
