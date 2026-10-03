"""Step 2 contract (docs/VOCAB_TRIM_STUDENT_PLAN.md §2): four candidates on the trimmed Vietnamese vocabulary.

Copy of ``scripts/base_encoder_baseline/common.py`` with the registry swapped to the trimmed models and a
``keep_layers`` option (bottom-N layers, Sajjad et al. 2020).  All candidates load the *trimmed* fast
tokenizer from ``artifacts/vocab_trim_student/tokenizer_vi`` (plan step 1.2B).
"""

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

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROTOCOL = "vocab-trim-student-v1"
ART = ROOT / "artifacts" / "vocab_trim_student"
TRIMMED = ROOT / "models" / "trimmed"
TOKENIZER_DIR = ART / "tokenizer_vi"
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTN = "sdpa"
TASKS = ("absa_restaurant", "vsfc_topic", "visfd")  # no Hotel/Sentiment in this task
LR_GRID_NARROW = (2e-5, 5e-5, 1e-4)   # M12 / M6 (hidden 384)
LR_GRID_WIDE = (1e-5, 2e-5, 5e-5)     # B8 / B12 (hidden 768)
MODELS: dict[str, dict[str, Any]] = {
    "m12": {"local_dir": TRIMMED / "mminilmv2_l12_vi", "repo": "nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large",
            "commit": "d828558d1a570cbbb5e62a8dbf85c8f18bf7982a", "family": "xlm-roberta", "lr_grid": LR_GRID_NARROW,
            "expected": {"num_hidden_layers": 12, "hidden_size": 384}},
    "m6": {"local_dir": TRIMMED / "mminilmv2_l6_vi", "repo": "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large",
           "commit": "160deb78aca30f63754e512a93337ce8013a32ca", "family": "xlm-roberta", "lr_grid": LR_GRID_NARROW,
           "expected": {"num_hidden_layers": 6, "hidden_size": 384}},
    "b8": {"local_dir": TRIMMED / "xlmr_base_vi", "repo": "FacebookAI/xlm-roberta-base",
           "commit": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", "family": "xlm-roberta", "lr_grid": LR_GRID_WIDE,
           "expected": {"num_hidden_layers": 12, "hidden_size": 768}, "keep_layers": 8},
    "b12": {"local_dir": TRIMMED / "xlmr_base_vi", "repo": "FacebookAI/xlm-roberta-base",
            "commit": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", "family": "xlm-roberta", "lr_grid": LR_GRID_WIDE,
            "expected": {"num_hidden_layers": 12, "hidden_size": 768}},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_manifest(key: str) -> dict[str, Any]:
    m = MODELS[key]
    directory = m["local_dir"]
    files = {p.name: sha256_file(p.resolve()) for p in sorted(directory.iterdir()) if p.resolve().is_file()}
    return {"key": key, "source_repo": m["repo"], "source_commit": m["commit"], "trimmed_dir": str(directory),
            "files": files, "weights_sha256": files.get("model.safetensors")}


def load_tokenizer(key: str | None = None):
    from transformers import AutoTokenizer

    # the trimmed tokenizer is shipped inside every trimmed model directory (a bare tokenizer dir has no config.json,
    # which makes ``AutoTokenizer.from_pretrained`` unable to infer the class)
    return AutoTokenizer.from_pretrained(str(TRIMMED / "xlmr_base_vi"), local_files_only=True, use_fast=True)


def build_model(key: str, *, num_classes: int, num_aspects: int, aspect_class_counts, keep_layers: int | None = None):
    from transformers import AutoModel

    from goibert_e14.modeling import BamiBERTClassifier

    m = MODELS[key]
    kwargs: dict[str, Any] = {"local_files_only": True, "attn_implementation": ATTN}
    if m["family"] == "xlm-roberta":
        kwargs["add_pooling_layer"] = False
    encoder, info = AutoModel.from_pretrained(str(m["local_dir"]), output_loading_info=True, **kwargs)
    cfg = encoder.config
    for name, value in m["expected"].items():
        if int(getattr(cfg, name)) != int(value):
            raise RuntimeError(f"{key}: config {name}={getattr(cfg, name)} != {value}")
    unexpected = [k for k in info.get("unexpected_keys", []) if not (k.startswith("lm_head") or k.startswith("decoder") or k.startswith("head") or "pooler" in k)]
    if info.get("missing_keys") or unexpected or info.get("mismatched_keys"):
        raise RuntimeError(f"{key} did not load cleanly: {info}")
    layers_kept = int(cfg.num_hidden_layers)
    if keep_layers:
        if keep_layers > int(cfg.num_hidden_layers):
            raise ValueError(f"{key}: keep_layers={keep_layers} > {cfg.num_hidden_layers}")
        encoder.encoder.layer = encoder.encoder.layer[:keep_layers]  # bottom layers (Sajjad et al. 2020)
        encoder.config.num_hidden_layers = keep_layers
        layers_kept = keep_layers
    model = BamiBERTClassifier(encoder, num_labels=num_classes, num_aspects=num_aspects,
                               aspect_class_counts=aspect_class_counts if num_aspects else None,
                               head_hidden_size=int(cfg.hidden_size))
    emb = encoder.get_input_embeddings().weight
    total = sum(p.numel() for p in model.parameters())
    enc_total = sum(p.numel() for p in encoder.parameters())
    spec = model.spec().to_dict()
    spec.update({"backbone": key, "source_repo": m["repo"], "source_commit": m["commit"],
                 "trimmed_dir": str(m["local_dir"].relative_to(ROOT)), "encoder_class": type(encoder).__name__,
                 "layers_kept": layers_kept, "encoder_parameters": enc_total, "embedding_parameters": int(emb.numel()),
                 "encoder_non_embedding_parameters": enc_total - int(emb.numel()), "total_parameters_with_head": total,
                 "dropped_keys_on_load": list(info.get("unexpected_keys", [])), "vocab_size": int(cfg.vocab_size)})
    return model, spec


def out_dir_for(key: str, task: str, lr: float, seed: int = SEED) -> Path:
    return ART / key / task / f"lr_{lr:.0e}".replace("+", "") / f"seed_{seed}"
