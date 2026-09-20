"""Shared contracts for the mDeBERTa validation-only pilot.

The data loader, loss, and evaluator are the frozen ``src.goibert`` E1
implementations.  This package owns only the pinned mDeBERTa checkpoint
adapter and durable pilot paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
MODEL_REPO = "microsoft/mdeberta-v3-base"
MODEL_COMMIT = "a0484667b22365f84929a935b5e50a51f71f159d"
MODEL_DIR = ROOT / "models" / "mdeberta-v3-base"
ARTIFACT_ROOT = ROOT / "artifacts" / "mdeberta_pilot"
RUN_ROOT = ARTIFACT_ROOT / "runs"
TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
SEED = 42
LEARNING_RATE = 2e-5
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
GRADIENT_CLIP_NORM = 1.0
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "eager"
PROTOCOL = "mdeberta-v3-base-five-task-pilot-v1"

# The pilot intentionally downloads only these original model/tokenizer files.
# DeBERTa-v3 generator and TensorFlow files are excluded.
DOWNLOAD_FILES = (
    "config.json",
    "pytorch_model.bin",
    "spm.model",
    "tokenizer_config.json",
)

import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from goibert import train as e1_train  # noqa: E402


def now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value)!r}")


def write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".cache"
    }


def source_hash() -> str:
    paths = [
        *sorted((ROOT / "scripts" / "mdeberta_pilot").glob("*.py")),
        ROOT / "src" / "goibert" / "train.py",
        ROOT / "src" / "goibert" / "data.py",
        ROOT / "src" / "goibert" / "metrics.py",
        ROOT / "docs" / "MDEBERTA_PILOT_PLAN.md",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def runtime(device: torch.device) -> dict[str, Any]:
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": PRECISION,
        "attention_implementation_requested": ATTENTION_IMPLEMENTATION,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "hostname": platform.node(),
    }


def gpu_snapshot() -> dict[str, dict[str, int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": repr(error)}  # type: ignore[return-value]
    result: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            result[fields[0]] = {
                "free_mib": int(float(fields[1])),
                "used_mib": int(float(fields[2])),
                "util_pct": int(float(fields[3])),
            }
    return result


def load_tokenizer() -> tuple[Any, dict[str, Any]]:
    required = [MODEL_DIR / name for name in DOWNLOAD_FILES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pinned mDeBERTa files: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        use_fast=False,
    )
    manifest = {
        "source": str(MODEL_DIR.relative_to(ROOT)),
        "hf_repo": MODEL_REPO,
        "hf_commit": MODEL_COMMIT,
        "files": {name: sha256_file(MODEL_DIR / name) for name in DOWNLOAD_FILES},
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": int(len(tokenizer)),
        "special_tokens": {
            key: getattr(tokenizer, key, None)
            for key in ("bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id")
        },
        "preprocessing": "official mDeBERTa tokenizer; raw Vietnamese text; no external word segmentation",
    }
    return tokenizer, manifest


class NativeMDebertaClassifier(nn.Module):
    """Adapter that exposes native HF logits to the frozen E1 train loop."""

    def __init__(self, model: nn.Module, *, num_labels: int, num_aspects: int, aspect_class_counts: Sequence[int], loading_info: Mapping[str, Any]):
        super().__init__()
        self.model = model
        self.num_labels = int(num_labels)
        self.num_aspects = int(num_aspects)
        self.aspect_class_counts = tuple(int(value) for value in aspect_class_counts)
        self.loading_info = dict(loading_info)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, token_type_ids: Tensor | None = None) -> Tensor:
        # Native DebertaV2 does not use token_type_ids (type_vocab_size=0), but
        # retaining the adapter argument keeps the E1 call contract unchanged.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            return_dict=True,
        )
        return outputs.logits

    def spec(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "backbone": MODEL_REPO,
            "local_path": str(MODEL_DIR.relative_to(ROOT)),
            "hf_commit": MODEL_COMMIT,
            "encoder_class": type(self.model.deberta).__name__,
            "classifier_class": type(self.model).__name__,
            "model_type": config.model_type,
            "hidden_size": int(config.hidden_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "intermediate_size": int(config.intermediate_size),
            "vocab_size": int(config.vocab_size),
            "pooling": "native DebertaV2 ContextPooler: hidden_states[:,0]",
            "pooler_dropout": float(config.pooler_dropout),
            "pooler_hidden_act": str(config.pooler_hidden_act),
            "classifier_dropout": float(getattr(config, "cls_dropout", None) if getattr(config, "cls_dropout", None) is not None else config.hidden_dropout_prob),
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "num_labels": self.num_labels,
            "num_aspects": self.num_aspects,
            "aspect_class_counts": list(self.aspect_class_counts),
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "encoder_weight_file_sha256": sha256_file(MODEL_DIR / "pytorch_model.bin"),
            "loading_info": self.loading_info,
        }


def build_model(contract: Mapping[str, Any]) -> tuple[NativeMDebertaClassifier, dict[str, Any]]:
    num_labels = int(contract["num_classes"])
    num_aspects = int(contract["num_aspects"])
    counts = tuple(int(value) for value in contract["aspect_class_counts"])
    model, loading_info = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    config = model.config
    if config.model_type != "deberta-v2" or int(config.hidden_size) != 768 or int(config.num_hidden_layers) != 12:
        raise RuntimeError(f"Unexpected mDeBERTa architecture: {config.to_dict()}")
    # ``output_loading_info`` contains lists of names and mismatched tuples;
    # normalize to plain JSON-friendly values at the boundary.
    normalized_info = {
        "missing_keys": list(loading_info.get("missing_keys", [])),
        "unexpected_keys": list(loading_info.get("unexpected_keys", [])),
        "mismatched_keys": [list(value) for value in loading_info.get("mismatched_keys", [])],
        "error_msgs": list(loading_info.get("error_msgs", [])),
    }
    if normalized_info["error_msgs"]:
        raise RuntimeError(f"mDeBERTa load errors: {normalized_info['error_msgs']}")
    # The only intentionally fresh weights are the native ContextPooler and
    # task classifier.  A missing/mismatched encoder key would mean the
    # pretrained checkpoint was not actually transferred.
    bad_missing = [
        key
        for key in normalized_info["missing_keys"]
        if key.startswith("deberta.")
    ]
    bad_mismatch = [
        key
        for key in normalized_info["mismatched_keys"]
        if key and str(key[0]).startswith("deberta.")
    ]
    bad_unexpected = [
        key
        for key in normalized_info["unexpected_keys"]
        if key.startswith("deberta.") and key != "deberta.embeddings.word_embeddings._weight"
    ]
    if bad_missing or bad_mismatch or bad_unexpected:
        raise RuntimeError(
            "mDeBERTa pretrained encoder load incomplete: "
            f"missing={bad_missing[:12]}, mismatched={bad_mismatch[:12]}, unexpected={bad_unexpected[:12]}"
        )
    first_parameter = next(model.parameters())
    if first_parameter.dtype != torch.float32:
        raise RuntimeError(f"Expected FP32 master parameters after load, got {first_parameter.dtype}")
    wrapped = NativeMDebertaClassifier(
        model,
        num_labels=num_labels,
        num_aspects=num_aspects,
        aspect_class_counts=counts,
        loading_info=normalized_info,
    )
    return wrapped, wrapped.spec()


def task_contract(task_id: str, *, include_test: bool = False) -> dict[str, Any]:
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=include_test)
    return {
        "bundle": bundle,
        "train": e1_train.split_field(bundle, "train"),
        "validation": e1_train.split_field(bundle, "validation"),
        "num_aspects": e1_train.task_num_aspects(bundle),
        "aspect_class_counts": tuple(e1_train.task_aspect_class_counts(bundle)),
        "num_classes": e1_train.task_num_classes(bundle),
    }


def make_loaders(tokenizer: Any, contract: Mapping[str, Any], micro_batch_size: int):
    bundle = contract["bundle"]
    train_split, valid_split = contract["train"], contract["validation"]
    train_loader = e1_train._make_loader(
        tokenizer,
        train_split,
        max_length=MAX_LENGTH,
        batch_size=micro_batch_size,
        shuffle=True,
        seed=SEED,
        labels=e1_train.model_labels(bundle, "train", train_split),
        gold_labels=e1_train.split_labels(train_split),
    )
    valid_loader = e1_train._make_loader(
        tokenizer,
        valid_split,
        max_length=MAX_LENGTH,
        batch_size=micro_batch_size,
        shuffle=False,
        seed=SEED,
        labels=e1_train.model_labels(bundle, "validation", valid_split),
        gold_labels=e1_train.split_labels(valid_split),
    )
    return train_loader, valid_loader


def run_dir(task_id: str) -> Path:
    return RUN_ROOT / task_id / f"lr_{LEARNING_RATE:.0e}" / f"seed_{SEED}"


def load_model_checkpoint(path: Path, model: nn.Module) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Checkpoint missing model state: {path}")
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    return {key: value for key, value in state.items() if key != "model"}
