"""Pinned BidirLM-1B fine-tune contract with native bidirectional attention."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
for _path in (SRC, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PROTOCOL = "bidirlm-1b-probe-ft-v1"
ART = ROOT / "artifacts" / "bidirlm_1b_probe" / "phase2" / "ft"
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTN = "eager"  # custom 1B code uses its own sdpa_attention_forward on this HF-supported path
ATTENTION_DROPOUT = 0.1
LR_GRID = (5e-6, 1e-5, 2e-5)
TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
HEAD_DROPOUT = 0.1
EOS_TOKEN_ID = 1
BOS_TOKEN_ID = 2  # prepended by modeling_bidirlm.py itself
MASK_TOKEN_ID = 4

MODELS: dict[str, dict[str, Any]] = {
    "bidirlm_1b": {
        "repo": "BidirLM/BidirLM-1B-Base",
        "branch": "transformers-v4",
        "commit": "f98e971f0f0bfcf88a7a8e0b74914497930f51f3",
        "weights": "model.safetensors",
        "family": "bidirlm",
        "license": "apache-2.0; Gemma foundation under Gemma Terms of Use",
        "trust_remote_code": True,
        "local": False,
        "expected": {"num_hidden_layers": 26, "hidden_size": 1152, "num_attention_heads": 4, "num_key_value_heads": 1, "vocab_size": 262144},
    },
}

ARMS: dict[str, dict[str, Any]] = {
    "B1B": {"model": "bidirlm_1b", "attention": "bidirectional", "mask": "native", "pooling": "mean", "micro_batch": 16,
            "role": "BidirLM-1B-Base, Gemma3-1B foundation"},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pinned_snapshot(key: str) -> Path:
    m = MODELS[key]
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "hub" / f"models--{m['repo'].replace('/', '--')}" / "snapshots" / m["commit"]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def model_source(key: str) -> Path:
    return MODELS[key]["local_dir"] if MODELS[key]["local"] else pinned_snapshot(key)


def model_manifest(key: str) -> dict[str, Any]:
    m = MODELS[key]
    d = model_source(key)
    files = {p.name: sha256_file(p.resolve()) for p in sorted(d.iterdir()) if p.resolve().is_file()}
    return {"key": key, "repo": m["repo"], "branch": m["branch"], "commit": m["commit"], "family": m["family"], "license": m["license"],
            "trust_remote_code": m["trust_remote_code"], "snapshot": str(d), "files": files,
            "weights_sha256": files.get(m["weights"])}


def load_tokenizer(arm: str):
    """BidirLM Gemma tokenizer; model prepends BOS and no EOS is appended."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(pinned_snapshot("bidirlm_1b")), local_files_only=True, use_fast=True)


def set_attention_dropout(encoder: Any, value: float) -> int:
    encoder.config.attention_dropout = float(value)
    touched = 0
    for module in encoder.modules():
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = float(value)
            touched += 1
    return touched


def pool_hidden(hidden: Any, attention_mask: Any, mode: str):
    import torch

    mask = attention_mask.to(hidden.dtype)
    if mode == "mean":
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    if mode == "last":
        lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
        index = lengths.to(hidden.device).long().view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return torch.gather(hidden, 1, index).squeeze(1)
    raise ValueError(f"unknown pooling {mode!r}")


def _classifier_class():
    from torch import nn

    from goibert_e14.modeling import ABSA_NUM_CLASSES, ClassificationHead

    class _BidirLMClassifier(nn.Module):
        def __init__(self, encoder, *, arm: str, num_labels: int, num_aspects: int, aspect_class_counts: Any) -> None:
            super().__init__()
            counts = tuple(int(v) for v in (aspect_class_counts or ()))
            if counts and sum(counts) != num_labels:
                raise ValueError("num_labels must equal sum(aspect_class_counts)")
            self.arm = arm
            self.encoder = encoder
            self.num_labels = int(num_labels)
            self.num_aspects = int(num_aspects)
            self.aspect_class_counts = counts or ((ABSA_NUM_CLASSES,) * self.num_aspects)
            self.pooling = ARMS[arm]["pooling"]
            hidden = int(encoder.config.hidden_size)
            self.head = ClassificationHead(hidden, self.num_labels, dropout=HEAD_DROPOUT, hidden_size=hidden)
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = pool_hidden(outputs.last_hidden_state, attention_mask, self.pooling)
            logits = self.head(pooled.unsqueeze(1))
            if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
                logits = logits.reshape(logits.shape[0], self.num_aspects, ABSA_NUM_CLASSES)
            return logits

    return _BidirLMClassifier


def load_encoder(arm: str):
    import torch
    from transformers import AutoModelForMaskedLM

    key = ARMS[arm]["model"]
    m = MODELS[key]
    source = str(model_source(key))
    full, info = AutoModelForMaskedLM.from_pretrained(source, output_loading_info=True, local_files_only=True,
                                                     attn_implementation=ATTN, dtype=torch.float32,
                                                     trust_remote_code=m["trust_remote_code"])
    encoder = full.model
    cfg = encoder.config
    for name, value in m["expected"].items():
        if int(getattr(cfg, name)) != int(value):
            raise RuntimeError(f"{key}: config {name}={getattr(cfg, name)} != {value}")
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"{key} did not load cleanly: {info}")
    del full
    return encoder, info


def build_model(arm: str, *, num_classes: int, num_aspects: int, aspect_class_counts: Any):
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known {sorted(ARMS)}")
    encoder, info = load_encoder(arm)
    dropout_modules = set_attention_dropout(encoder, ATTENTION_DROPOUT)
    model = _classifier_class()(encoder, arm=arm, num_labels=num_classes, num_aspects=num_aspects,
                                aspect_class_counts=aspect_class_counts)
    embedding = encoder.get_input_embeddings().weight
    enc_total = sum(p.numel() for p in encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    key = ARMS[arm]["model"]
    spec = {
        "protocol": PROTOCOL, "arm": arm, "arm_role": ARMS[arm]["role"], "arm_attention": ARMS[arm]["attention"],
        "arm_mask_mode": ARMS[arm]["mask"], "pooling": ARMS[arm]["pooling"], "attention_dropout": ATTENTION_DROPOUT,
        "attention_dropout_modules": dropout_modules, "backbone": key, "backbone_repo": MODELS[key]["repo"],
        "hf_commit": MODELS[key]["commit"], "encoder_class": type(encoder).__name__,
        "hidden_size": int(encoder.config.hidden_size), "num_hidden_layers": int(encoder.config.num_hidden_layers),
        "num_attention_heads": int(encoder.config.num_attention_heads), "num_key_value_heads": int(encoder.config.num_key_value_heads),
        "vocab_size": int(encoder.config.vocab_size), "encoder_parameters": int(enc_total),
        "embedding_parameters": int(embedding.numel()), "encoder_non_embedding_parameters": int(enc_total - embedding.numel()),
        "total_parameters_with_head": int(total), "num_labels": int(num_classes), "num_aspects": int(num_aspects),
        "aspect_class_counts": list(model.aspect_class_counts), "dropped_keys_on_load": list(info.get("unexpected_keys", [])),
        "attention_implementation": ATTN, "dtype": "float32 master weights + bf16 autocast",
    }
    return model, spec


def arm_dir(arm: str, task: str, lr: float, seed: int) -> Path:
    return ART / arm / task / f"lr_{lr:.0e}".replace("+", "") / f"seed_{seed}"
