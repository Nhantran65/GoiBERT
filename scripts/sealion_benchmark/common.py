"""Frozen SEA-LION/CafeBERT benchmark contract.

This module intentionally reuses the audited E1 data loader, metrics, and
optimizer.  The only new model adapter is a ModernBERT encoder plus the same
CLS-style task head used by the local BamiBERT/CafeBERT baselines.  For
ModernBERT, the first token is the tokenizer's ``<bos>`` token (ID 2); it is
recorded as first-token/BOS in every run manifest.
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
from torch import Tensor, nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402
from transformers import get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e14.modeling import BamiBERTClassifier  # noqa: E402

MODEL_DIR = ROOT / "models" / "sea-lion-modernbert-300m"
MODEL_REPO = "aisingapore/SEA-LION-ModernBERT-300M"
MODEL_COMMIT = "e3d70b810769bf86ae9041d1943e0ecd10b62b31"
ARTIFACT_ROOT = ROOT / "artifacts" / "sealion_benchmark"
RUN_ROOT = ARTIFACT_ROOT / "runs"
PROTOCOL = "SEA-LION-ModernBERT-300M-five-task-v1"
SEED = 42
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
LEARNING_RATES = (2e-5, 5e-5)
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
GRADIENT_CLIP_NORM = 1.0
HEAD_DROPOUT = 0.1
HEAD_HIDDEN_SIZE = 768

# The recipe for new CafeBERT runs is frozen to the completed T restaurant
# pilot.  The restaurant run itself is reused and never overwritten.
CAFE_RUN_ROOT = ARTIFACT_ROOT / "cafe_runs"
CAFE_TASKS = TASKS[0:2] + TASKS[2:]
CAFE_REPO = "uitnlp/CafeBERT"
CAFE_COMMIT = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"
CAFE_TOKENIZER_REPO = "FacebookAI/xlm-roberta-large"
CAFE_TOKENIZER_COMMIT = "c23d21b0620b635a76227c604d44e43a9f0ee389"
CAFE_EXISTING_RUN = ROOT / "artifacts" / "cdbk_pilot" / "runs" / "T_absa_restaurant_seed42"


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
    tmp.replace(path)


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
    # Include the adapter itself and the frozen loader/evaluator/optimizer
    # implementations.  Do not include generated artifacts or historical
    # source directories that can change during a run.
    paths = [
        *sorted((ROOT / "scripts" / "sealion_benchmark").glob("*.py")),
        SRC / "goibert" / "train.py",
        SRC / "goibert" / "data.py",
        SRC / "goibert" / "metrics.py",
        SRC / "goibert_e14" / "modeling.py",
        ROOT / "scripts" / "cdbk_pilot" / "common.py",
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


def _snapshot_dir(repo: str, commit: str) -> Path:
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    if not path.is_dir():
        raise FileNotFoundError(f"Missing local HF snapshot for {repo}@{commit}: {path}")
    return path


def load_sealion_tokenizer() -> tuple[Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    special = {k: getattr(tokenizer, k, None) for k in ("bos_token_id", "eos_token_id", "cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id")}
    if special["bos_token_id"] != 2 or special["pad_token_id"] != 0:
        raise RuntimeError(f"Unexpected SEA-LION tokenizer special IDs: {special}")
    manifest = {
        "source": str(MODEL_DIR.relative_to(ROOT)), "hf_repo": MODEL_REPO, "hf_commit": MODEL_COMMIT,
        "files": {k: v for k, v in directory_hashes(MODEL_DIR).items() if Path(k).name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}},
        "tokenizer_class": type(tokenizer).__name__, "vocab_size": int(len(tokenizer)),
        "special_tokens": special, "first_token_head": "input_ids[:,0] = <bos> (token ID 2); ModernBERT native classifier_pooling=mean is not used",
        "token_type_ids": "not emitted and not passed to ModernBERT",
    }
    return tokenizer, manifest


def load_cafe_tokenizer() -> tuple[Any, dict[str, Any]]:
    from scripts.cdbk_pilot import common as cdbk_common
    tokenizer, manifest = cdbk_common.load_tokenizer("cafebert")
    manifest = dict(manifest)
    manifest["benchmark_adapter"] = "scripts/cdbk_pilot/common.py via sealion_benchmark"
    return tokenizer, manifest


class ClassificationHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, *, hidden_size: int = HEAD_HIDDEN_SIZE, dropout: float = HEAD_DROPOUT) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        first = hidden_states[:, 0]
        return self.output(self.dropout(self.activation(self.dense(self.input_dropout(first)))))


class SeaLionClassifier(nn.Module):
    """ModernBERT encoder with the shared first-token/BOS task head."""

    def __init__(self, encoder: nn.Module, *, num_labels: int, num_aspects: int, aspect_class_counts: Sequence[int]) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.num_aspects = int(num_aspects)
        self.aspect_class_counts = tuple(int(v) for v in aspect_class_counts)
        self.head = ClassificationHead(int(encoder.config.hidden_size), self.num_labels)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, token_type_ids: Tensor | None = None) -> Tensor:
        # ModernBertModel does not accept token_type_ids.  The parameter is
        # kept in the adapter signature so the audited runner can call both
        # families through the same path; it is deliberately ignored.
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = self.head(outputs.last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return logits

    def spec(self) -> dict[str, Any]:
        config = self.encoder.config
        return {
            "backbone": MODEL_REPO, "local_path": str(MODEL_DIR.relative_to(ROOT)), "hf_commit": MODEL_COMMIT,
            "encoder_class": type(self.encoder).__name__, "model_type": config.model_type,
            "hidden_size": int(config.hidden_size), "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads), "intermediate_size": int(config.intermediate_size),
            "head_hidden_size": HEAD_HIDDEN_SIZE, "head_dropout": HEAD_DROPOUT,
            "pooling": "first-token/BOS", "native_classifier_pooling": getattr(config, "classifier_pooling", None),
            "num_labels": self.num_labels, "num_aspects": self.num_aspects,
            "aspect_class_counts": list(self.aspect_class_counts),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "encoder_weight_file_sha256": sha256_file(MODEL_DIR / "model.safetensors"),
        }


def build_model(model_kind: str, contract: Mapping[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    num_classes = int(contract["num_classes"])
    num_aspects = int(contract["num_aspects"])
    counts = tuple(contract["aspect_class_counts"])
    if model_kind == "sealion":
        encoder = AutoModel.from_pretrained(
            MODEL_DIR, local_files_only=True, attn_implementation=ATTENTION_IMPLEMENTATION,
        )
        config = encoder.config
        expected = {"hidden_size": 768, "num_hidden_layers": 22, "num_attention_heads": 12, "intermediate_size": 1152, "vocab_size": 262144}
        actual = {name: int(getattr(config, name)) for name in expected}
        if actual != expected:
            raise RuntimeError(f"SEA-LION architecture mismatch: expected {expected}, got {actual}")
        model = SeaLionClassifier(encoder, num_labels=num_classes, num_aspects=num_aspects, aspect_class_counts=counts)
        return model, model.spec()
    if model_kind == "cafebert":
        from scripts.cdbk_pilot import common as cdbk_common
        model, spec = cdbk_common.build_model("cafebert", contract)
        return model, {**spec, "benchmark_adapter": "scripts/cdbk_pilot/common.py via sealion_benchmark"}
    raise ValueError(model_kind)


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


def test_loader(tokenizer: Any, contract: Mapping[str, Any], micro_batch_size: int):
    bundle = contract["bundle"]
    split = e1_train.split_field(bundle, "test")
    return e1_train._make_loader(
        tokenizer, split, max_length=MAX_LENGTH, batch_size=micro_batch_size, shuffle=False, seed=SEED,
        labels=e1_train.model_labels(bundle, "test", split), gold_labels=e1_train.split_labels(split),
    )


def run_dir(model_kind: str, task_id: str, learning_rate: float) -> Path:
    root = RUN_ROOT if model_kind == "sealion" else CAFE_RUN_ROOT
    lr_name = f"lr_{learning_rate:.0e}".replace("-", "-")
    return root / task_id / lr_name / f"seed_{SEED}"


def load_model_checkpoint(path: Path, model: nn.Module) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Checkpoint missing model state: {path}")
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    return {k: v for k, v in state.items() if k != "model"}

