"""Qwen3-0.6B-Base bidirectional-read probe (docs/QWEN3_ENCODER_PROBE_PLAN.md).

Pinned models, arm registry, bidirectional mask, Qwen3 EOS post-processor, pooling
and the E14 head wiring.  Stage 0 probe: validation-only, no test split.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
for _path in (SRC, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PROTOCOL = "qwen3-encoder-probe-v1"
ART = ROOT / "artifacts" / "qwen3_encoder_probe"
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTN = "sdpa"
ATTENTION_DROPOUT = 0.1  # differs from E14 (Qwen3/BidirLM default 0.0); Gemma Encoder: 10% dropout optimal
LR_GRID = (1e-5, 2e-5, 5e-5)
TASKS = ("absa_restaurant", "vsfc_topic", "visfd")
HEAD_DROPOUT = 0.1
EOS_TOKEN = "<|endoftext|>"
EOS_TOKEN_ID = 151643
IM_START_TOKEN_ID = 151644  # hard-coded prepend inside modeling_bidirlm.py
MASK_TOKEN = "<mask>"
MASK_TOKEN_ID = 151669

MODELS: dict[str, dict[str, Any]] = {
    "qwen3_base": {
        "repo": "Qwen/Qwen3-0.6B-Base",
        "commit": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "weights": "model.safetensors",
        "family": "qwen3",
        "license": "apache-2.0",
        "trust_remote_code": False,
        "expected": {"num_hidden_layers": 28, "hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 151936},
    },
    "bidirlm_base": {
        "repo": "BidirLM/BidirLM-0.6B-Base",
        "commit": "29794508bbb4e378b4789eebff42ae4c2d47d524",
        "weights": "model.safetensors",
        "family": "bidirlm",
        "license": "apache-2.0",
        "trust_remote_code": True,
        "expected": {"num_hidden_layers": 28, "hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 151936},
    },
}

EXPECTED_TOTAL_PARAMETERS = 596_049_920  # 596.0M
EXPECTED_NON_EMBEDDING_PARAMETERS = 440_467_456  # 440.5M, embedding 151936x1024 = 155.6M, tied

# Arms (plan §4).  ``mask`` selects how the padding/direction mask is fed to the encoder:
#   "2d"        -> original causal decoder with the plain 2D padding mask
#   "4d_bidir"  -> is_causal=False everywhere + explicit 4D bool mask (True = may attend)
#   "native"    -> BidirLM custom code (packed varlen, always bidirectional), 2D mask in
ARMS: dict[str, dict[str, Any]] = {
    "C": {"model": "qwen3_base", "attention": "causal", "mask": "2d", "pooling": "last", "micro_batch": 32,
          "role": "control: Qwen3 original one-directional read"},
    "B0": {"model": "qwen3_base", "attention": "bidirectional", "mask": "4d_bidir", "pooling": "mean", "micro_batch": 32,
           "role": "Qwen3-0.6B-Base fine-tuned bidirectional"},
    "B1": {"model": "mntp_export", "attention": "bidirectional", "mask": "4d_bidir", "pooling": "mean", "micro_batch": 32,
           "role": "Qwen3-0.6B-Base + short Vietnamese MNTP (stage 4, optional)"},
    "B2": {"model": "bidirlm_base", "attention": "bidirectional", "mask": "native", "pooling": "mean", "micro_batch": 16,
           "role": "BidirLM-0.6B-Base (30B-token MNTP adaptation of Qwen3-0.6B-Base)"},
}
MNTP_EXPORT_DIR = ART / "mntp" / "encoder"


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
    m = MODELS[key]
    d = snapshot_dir(key)
    files = {p.name: sha256_file(p.resolve()) for p in sorted(d.iterdir()) if p.resolve().is_file()}
    return {
        "key": key, "repo": m["repo"], "commit": m["commit"], "family": m["family"], "license": m["license"],
        "trust_remote_code": m["trust_remote_code"], "snapshot": str(d), "files": files,
        "weights_sha256": files.get(m["weights"]),
    }


def with_eos_post_processor(tokenizer: Any):
    """Qwen3 does not append BOS/EOS; the probe appends EOS to every string (plan §3)."""
    from tokenizers.processors import TemplateProcessing

    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"$A {EOS_TOKEN}",
        pair=f"$A {EOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[(EOS_TOKEN, EOS_TOKEN_ID)],
    )
    return tokenizer


def tokenizer_key_for(arm: str) -> str:
    return ARMS[arm]["model"] if ARMS[arm]["model"] != "mntp_export" else "qwen3_base"


def load_tokenizer(arm: str):
    """C/B0/B1: Qwen3 tokenizer + EOS post-processor.  B2: BidirLM tokenizer as shipped (it prepends <|im_start|> itself)."""
    from transformers import AutoTokenizer

    key = tokenizer_key_for(arm)
    m = MODELS[key]
    tok = AutoTokenizer.from_pretrained(
        m["repo"], revision=m["commit"], local_files_only=True, use_fast=True,
        trust_remote_code=m["trust_remote_code"],
    )
    if ARMS[arm]["mask"] != "native":
        with_eos_post_processor(tok)
    return tok


def bidir_mask(attention_mask: Any):
    """4D bool mask (B,1,L,L), True = the key may be attended.

    Rows of padded positions attend themselves only, so no row is fully masked.
    transformers 4.57 returns an already-4D mask from ``create_causal_mask`` unchanged.
    """
    import torch

    am = attention_mask.to(torch.bool)
    b, length = am.shape
    key_ok = am[:, None, None, :].expand(b, 1, length, length)
    eye = torch.eye(length, dtype=torch.bool, device=am.device).reshape(1, 1, length, length)
    return key_ok | eye


def disable_causal(encoder: Any) -> int:
    """Set ``is_causal = False`` on every module carrying the attribute (plan §3a)."""
    touched = 0
    for module in encoder.modules():
        if hasattr(module, "is_causal"):
            module.is_causal = False
            touched += 1
    return touched


def set_attention_dropout(encoder: Any, value: float) -> int:
    """Set attention dropout on the config *and* on every module that cached it at construction time."""
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
    import torch
    from torch import nn

    from goibert_e14.modeling import ABSA_NUM_CLASSES, ClassificationHead

    class _ProbeClassifier(nn.Module):
        def __init__(self, encoder, *, arm: str, num_labels: int, num_aspects: int, aspect_class_counts: Any) -> None:
            super().__init__()
            if num_labels <= 0:
                raise ValueError("num_labels must be positive")
            counts = tuple(int(v) for v in (aspect_class_counts or ()))
            if counts and sum(counts) != num_labels:
                raise ValueError("num_labels must equal sum(aspect_class_counts)")
            self.arm = arm
            self.encoder = encoder
            self.num_labels = int(num_labels)
            self.num_aspects = int(num_aspects)
            self.aspect_class_counts = counts or ((ABSA_NUM_CLASSES,) * self.num_aspects)
            self.mask_mode = ARMS[arm]["mask"]
            self.pooling = ARMS[arm]["pooling"]
            hidden = int(encoder.config.hidden_size)
            self.head = ClassificationHead(hidden, self.num_labels, dropout=HEAD_DROPOUT, hidden_size=hidden)
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(True)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            if self.mask_mode == "4d_bidir":
                mask = bidir_mask(attention_mask)
            else:
                mask = attention_mask
            outputs = self.encoder(input_ids=input_ids, attention_mask=mask, return_dict=True)
            pooled = pool_hidden(outputs.last_hidden_state, attention_mask, self.pooling)
            # E14 head expects [:, 0] of its input; keep the head contract by unsqueezing the pooled vector
            logits = self.head(pooled.unsqueeze(1))
            if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
                logits = logits.reshape(logits.shape[0], self.num_aspects, ABSA_NUM_CLASSES)
            return logits

    return _ProbeClassifier


def load_encoder(arm: str):
    """Load the pinned encoder, check architecture + parameter counts and loading info."""
    import torch
    from transformers import AutoModel

    key = ARMS[arm]["model"]
    # master weights stay fp32 (plan §4: bf16 autocast + master fp32); Qwen3 ships a bfloat16 config dtype
    kwargs: dict[str, Any] = {"output_loading_info": True, "local_files_only": True, "attn_implementation": ATTN,
                              "dtype": torch.float32}
    if key == "mntp_export":
        if not MNTP_EXPORT_DIR.is_dir():
            raise FileNotFoundError(f"arm {arm} needs the stage-4 MNTP export at {MNTP_EXPORT_DIR}")
        source: Any = str(MNTP_EXPORT_DIR)
        expected = MODELS["qwen3_base"]
    else:
        source = MODELS[key]["repo"]
        kwargs["revision"] = MODELS[key]["commit"]
        kwargs["trust_remote_code"] = MODELS[key]["trust_remote_code"]
        expected = MODELS[key]
    encoder, info = AutoModel.from_pretrained(source, **kwargs)
    cfg = encoder.config
    for name, value in expected["expected"].items():
        if int(getattr(cfg, name)) != int(value):
            raise RuntimeError(f"{key}: config {name}={getattr(cfg, name)} != {value}")
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith("lm_head")]
    if info.get("missing_keys") or unexpected or info.get("mismatched_keys"):
        raise RuntimeError(f"{key} did not load cleanly: {info}")
    return encoder, info


def build_model(arm: str, *, num_classes: int, num_aspects: int, aspect_class_counts: Any):
    """Encoder + E14 ClassificationHead(logits over pooled.unsqueeze(1)); returns (model, spec)."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known {sorted(ARMS)}")
    encoder, info = load_encoder(arm)
    dropout_modules = set_attention_dropout(encoder, ATTENTION_DROPOUT)
    touched = disable_causal(encoder) if ARMS[arm]["attention"] == "bidirectional" else 0
    if ARMS[arm]["attention"] == "bidirectional" and touched == 0 and ARMS[arm]["mask"] != "native":
        raise RuntimeError(f"{arm}: no module exposed is_causal; bidirectional switch did not apply")
    model = _classifier_class()(encoder, arm=arm, num_labels=num_classes, num_aspects=num_aspects,
                                aspect_class_counts=aspect_class_counts)
    embedding = encoder.get_input_embeddings().weight
    enc_total = sum(p.numel() for p in encoder.parameters())
    total = sum(p.numel() for p in model.parameters())
    key = ARMS[arm]["model"]
    spec = {
        "protocol": PROTOCOL, "arm": arm, "arm_role": ARMS[arm]["role"], "arm_attention": ARMS[arm]["attention"],
        "arm_mask_mode": ARMS[arm]["mask"], "pooling": ARMS[arm]["pooling"], "attention_dropout": ATTENTION_DROPOUT,
        "attention_dropout_modules": dropout_modules,
        "is_causal_modules_set_false": touched, "backbone": key,
        "backbone_repo": MODELS[tokenizer_key_for(arm)]["repo"], "hf_commit": MODELS[tokenizer_key_for(arm)]["commit"],
        "encoder_class": type(encoder).__name__, "hidden_size": int(encoder.config.hidden_size),
        "num_hidden_layers": int(encoder.config.num_hidden_layers), "num_attention_heads": int(encoder.config.num_attention_heads),
        "num_key_value_heads": int(encoder.config.num_key_value_heads), "vocab_size": int(encoder.config.vocab_size),
        "encoder_parameters": int(enc_total), "embedding_parameters": int(embedding.numel()),
        "encoder_non_embedding_parameters": int(enc_total - embedding.numel()), "total_parameters_with_head": int(total),
        "num_labels": int(num_classes), "num_aspects": int(num_aspects), "aspect_class_counts": list(model.aspect_class_counts),
        "dropped_keys_on_load": list(info.get("unexpected_keys", [])),
        "attention_implementation": ATTN, "dtype": "float32 master weights + bf16 autocast",
    }
    return model, spec


def arm_dir(arm: str, task: str, lr: float, seed: int) -> Path:
    return ART / arm / task / f"lr_{lr:.0e}".replace("+", "") / f"seed_{seed}"
