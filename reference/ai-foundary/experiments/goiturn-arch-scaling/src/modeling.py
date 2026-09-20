"""Construct BamiBERT-family classifiers and initialize distilled embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from transformers import RobertaConfig, RobertaModel


VOCAB_SIZE = 20_481
TYPE_VOCAB_SIZE = 1
WORD_EMBEDDING_KEY = "roberta.embeddings.word_embeddings.weight"


class ClassificationHead(nn.Module):
    """Shared first-token classification head for sentiment and multi-slot ABSA."""

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states[:, 0]
        return self.output(self.dropout(torch.tanh(self.dense(pooled))))


class RobertaClassifier(nn.Module):
    """RoBERTa encoder plus the experiment's shared classification head."""

    def __init__(self, encoder: RobertaModel, output_size: int, n_aspects: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(encoder.config.hidden_size, output_size)
        self.n_aspects = n_aspects

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head(hidden)
        if self.n_aspects:
            return logits.reshape(-1, self.n_aspects, 4)
        return logits


def load_architectures(experiment_root: Path) -> dict[str, dict[str, int]]:
    path = experiment_root / "configs" / "variants.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_config(arch: dict[str, int]) -> RobertaConfig:
    return RobertaConfig(
        vocab_size=VOCAB_SIZE,
        type_vocab_size=TYPE_VOCAB_SIZE,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        **arch,
    )


def parameter_counts(model: RobertaClassifier) -> tuple[int, int]:
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    head = sum(parameter.numel() for parameter in model.head.parameters())
    return encoder, head


def expected_encoder_parameters(arch: dict[str, int]) -> int:
    """Return the closed-form embeddings+encoder count, excluding pooler and head."""
    hidden = arch["hidden_size"]
    intermediate = arch["intermediate_size"]
    layers = arch["num_hidden_layers"]
    positions = arch["max_position_embeddings"]
    embeddings = VOCAB_SIZE * hidden + positions * hidden + TYPE_VOCAB_SIZE * hidden + 2 * hidden
    per_layer = 4 * hidden * hidden + 2 * hidden * intermediate + 9 * hidden + intermediate
    return embeddings + layers * per_layer


def _standardized_pca_projection(
    checkpoint_path: Path, hidden_size: int, cache_path: Path
) -> torch.Tensor:
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        if WORD_EMBEDDING_KEY not in checkpoint.keys():
            raise KeyError(f"{WORD_EMBEDDING_KEY!r} is missing from {checkpoint_path}")
        teacher = checkpoint.get_tensor(WORD_EMBEDDING_KEY).float()
    if teacher.shape != (VOCAB_SIZE, 768):
        raise ValueError(f"Unexpected BamiBERT word-embedding shape: {tuple(teacher.shape)}")
    centered = teacher - teacher.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=hidden_size, center=False, niter=2)
    projected = centered @ components
    projected = projected / projected.std(dim=0, unbiased=False).clamp_min(1e-12)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projected, cache_path)
    return projected


def initialize_pca_embeddings(
    model: RobertaClassifier, checkpoint_path: Path, cache_dir: Path
) -> None:
    """Project BamiBERT words with PCA and match the random-init std per column."""
    word_embeddings = model.encoder.embeddings.word_embeddings.weight
    cache_path = cache_dir / f"bamibert_pca_{word_embeddings.shape[1]}.pt"
    projected = _standardized_pca_projection(
        checkpoint_path, word_embeddings.shape[1], cache_path
    )
    target_std = word_embeddings.detach().float().std(dim=0, unbiased=False).cpu()
    initialized = projected * target_std
    with torch.no_grad():
        word_embeddings.copy_(initialized.to(dtype=word_embeddings.dtype, device=word_embeddings.device))


def build_model(
    model_name: str,
    mode: str,
    n_aspects: int,
    repo_root: Path,
    experiment_root: Path,
) -> tuple[RobertaClassifier, dict[str, Any], int, int]:
    """Build a pooler-free encoder so printed counts match the experiment contract."""
    architectures = load_architectures(experiment_root)
    if model_name not in architectures:
        raise ValueError(f"Unknown architecture {model_name!r}")
    arch = architectures[model_name]
    config = make_config(arch)
    # ROCm SDPA kernels give NaN forwards with pretrained-scale weights (LUMI
    # MI250X, torch 2.5.1+rocm6.2, 2026-08-30) - use eager attention there.
    # GOITURN_ATTN=eager|sdpa overrides.
    attn = os.environ.get("GOITURN_ATTN") or (
        "eager" if getattr(torch.version, "hip", None) is not None else "sdpa"
    )
    config._attn_implementation = attn
    print(f"Attention implementation: {attn}", flush=True)
    checkpoint_dir = repo_root / "models" / "bamibert"
    checkpoint_file = checkpoint_dir / "model.safetensors"
    if model_name == "baseline":
        if mode != "finetune":
            raise ValueError("baseline supports only --mode finetune")
        encoder = RobertaModel.from_pretrained(
            checkpoint_dir, config=config, local_files_only=True, add_pooling_layer=False
        )
    else:
        if mode not in {"scratch", "distill"}:
            raise ValueError("variants support only --mode scratch or distill")
        encoder = RobertaModel(config, add_pooling_layer=False)
    output_size = 3 if n_aspects == 0 else n_aspects * 4
    model = RobertaClassifier(encoder, output_size, n_aspects)
    if mode == "distill":
        initialize_pca_embeddings(model, checkpoint_file, experiment_root / "results" / "pca")
    encoder_count, head_count = parameter_counts(model)
    expected = expected_encoder_parameters(arch)
    if encoder_count != expected:
        raise RuntimeError(f"{model_name} encoder count {encoder_count:,} != expected {expected:,}")
    print(
        f"Built {model_name}: encoder+embeddings={encoder_count:,}; head={head_count:,}",
        flush=True,
    )
    return model, dict(arch), encoder_count, head_count
