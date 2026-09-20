"""Base-encoder baseline (docs/BASE_ENCODER_BASELINE_PLAN.md): pinned models, E14 recipe constants, model builder."""

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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROTOCOL = "base-encoder-baseline-v1"
ART = ROOT / "artifacts" / "base_encoder_baseline"
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTN = "sdpa"
LR_GRID = (1e-5, 2e-5, 5e-5)
TASKS = ("absa_restaurant", "vsfc_topic", "visfd", "absa_hotel", "vsfc_sentiment")
MODELS: dict[str, dict[str, Any]] = {
    "mmbert_base": {"repo": "jhu-clsp/mmBERT-base", "commit": "c5955035435e2bf121cde7f3c8863ef52ff35d82", "weights": "pytorch_model.bin", "family": "modernbert", "expected": {"num_hidden_layers": 22, "hidden_size": 768}},
    "xlmr_base": {"repo": "FacebookAI/xlm-roberta-base", "commit": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", "weights": "model.safetensors", "family": "xlm-roberta", "expected": {"num_hidden_layers": 12, "hidden_size": 768}},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot_dir(key: str) -> Path:
    m = MODELS[key]
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    p = home / "hub" / f"models--{m['repo'].replace('/', '--')}" / "snapshots" / m["commit"]
    if not p.is_dir():
        raise FileNotFoundError(p)
    return p


def model_manifest(key: str) -> dict[str, Any]:
    m = MODELS[key]; d = snapshot_dir(key)
    files = {p.name: sha256_file(p.resolve()) for p in sorted(d.iterdir()) if p.resolve().is_file()}
    return {"key": key, "repo": m["repo"], "commit": m["commit"], "family": m["family"], "snapshot": str(d), "files": files, "weights_sha256": files.get(m["weights"])}


def load_tokenizer(key: str):
    from transformers import AutoTokenizer
    m = MODELS[key]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["commit"], local_files_only=True, use_fast=True)
    return tok


def build_model(key: str, *, num_classes: int, num_aspects: int, aspect_class_counts):
    """AutoModel encoder (no pooler) + E14 ClassificationHead with hidden = encoder hidden; returns (model, spec dict)."""
    from transformers import AutoModel
    from goibert_e14.modeling import BamiBERTClassifier
    m = MODELS[key]
    kwargs: dict[str, Any] = {"revision": m["commit"], "local_files_only": True, "attn_implementation": ATTN}
    if m["family"] == "xlm-roberta":
        kwargs["add_pooling_layer"] = False
    encoder, info = AutoModel.from_pretrained(m["repo"], output_loading_info=True, **kwargs)
    cfg = encoder.config
    for k, v in m["expected"].items():
        if int(getattr(cfg, k)) != v:
            raise RuntimeError(f"{key}: config {k}={getattr(cfg, k)} != {v}")
    unexpected = [k for k in info.get("unexpected_keys", []) if not (k.startswith("lm_head") or k.startswith("decoder") or k.startswith("head") or "pooler" in k)]
    if info.get("missing_keys") or unexpected or info.get("mismatched_keys"):
        raise RuntimeError(f"{key} did not load cleanly: {info}")
    model = BamiBERTClassifier(encoder, num_labels=num_classes, num_aspects=num_aspects, aspect_class_counts=aspect_class_counts if num_aspects else None, head_hidden_size=int(cfg.hidden_size))
    emb = encoder.get_input_embeddings().weight
    total = sum(p.numel() for p in model.parameters()); enc_total = sum(p.numel() for p in encoder.parameters())
    spec = model.spec().to_dict()
    spec.update({"backbone": m["repo"], "hf_commit": m["commit"], "encoder_class": type(encoder).__name__, "encoder_parameters": enc_total, "embedding_parameters": int(emb.numel()),
                 "encoder_non_embedding_parameters": enc_total - int(emb.numel()), "total_parameters_with_head": total, "dropped_keys_on_load": list(info.get("unexpected_keys", [])), "vocab_size": int(cfg.vocab_size)})
    return model, spec
