"""Construct the classic RoBERTa GoiBERT family and initialize PCA embeddings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel


VOCAB_SIZE = 20_481
TYPE_VOCAB_SIZE = 1
TEACHER_HIDDEN_SIZE = 768


@dataclass
class ClassifierOutput:
    """Classifier logits plus optional encoder internals used by distillation."""

    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]
    attentions: tuple[torch.Tensor, ...]


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
    """Pooler-free RoBERTa encoder plus the experiment's classification head."""

    def __init__(self, encoder: RobertaModel, output_size: int, n_aspects: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = ClassificationHead(encoder.config.hidden_size, output_size)
        self.n_aspects = n_aspects

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> torch.Tensor | ClassifierOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        logits = self.head(outputs.last_hidden_state)
        if self.n_aspects:
            logits = logits.reshape(-1, self.n_aspects, 4)
        if output_hidden_states or output_attentions:
            return ClassifierOutput(
                logits=logits,
                hidden_states=tuple(outputs.hidden_states or ()),
                attentions=tuple(outputs.attentions or ()),
            )
        return logits


def load_architectures(experiment_root: Path) -> dict[str, dict[str, int]]:
    path = experiment_root / "configs" / "family.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_config(architecture: dict[str, int], require_attentions: bool = False) -> RobertaConfig:
    config = RobertaConfig(
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
        use_cache=False,
        **architecture,
    )
    # Distillation needs materialized attention probabilities. Eager attention is
    # the portable default for that path and for ROCm; GOITURN_ATTN still overrides.
    attention_implementation = os.environ.get("GOITURN_ATTN") or (
        "eager"
        if require_attentions or getattr(torch.version, "hip", None) is not None
        else "sdpa"
    )
    config._attn_implementation = attention_implementation
    print(f"Attention implementation: {attention_implementation}", flush=True)
    return config


def expected_encoder_parameters(architecture: dict[str, int]) -> int:
    """Return the exact embeddings-and-encoder count, excluding pooler and head."""
    hidden = architecture["hidden_size"]
    intermediate = architecture["intermediate_size"]
    layers = architecture["num_hidden_layers"]
    positions = architecture["max_position_embeddings"]
    embeddings = VOCAB_SIZE * hidden + positions * hidden + TYPE_VOCAB_SIZE * hidden + 2 * hidden
    per_layer = 4 * hidden * hidden + 2 * hidden * intermediate + 9 * hidden + intermediate
    return embeddings + layers * per_layer


def parameter_counts(model: RobertaClassifier) -> dict[str, int]:
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    head = sum(parameter.numel() for parameter in model.head.parameters())
    return {"encoder_and_embeddings": encoder, "head": head, "model_total": encoder + head}


def _standardized_pca_projection(
    checkpoint_dir: Path,
    hidden_size: int,
    cache_path: Path,
) -> torch.Tensor:
    if cache_path.is_file():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    teacher = RobertaModel.from_pretrained(
        checkpoint_dir, local_files_only=True, add_pooling_layer=False
    )
    teacher_words = teacher.embeddings.word_embeddings.weight.detach().float().cpu()
    del teacher
    if teacher_words.shape != (VOCAB_SIZE, TEACHER_HIDDEN_SIZE):
        raise ValueError(f"Unexpected BamiBERT embedding shape: {tuple(teacher_words.shape)}")
    centered = teacher_words - teacher_words.mean(dim=0, keepdim=True)
    _, _, components = torch.pca_lowrank(centered, q=hidden_size, center=False, niter=2)
    projected = centered @ components
    projected = projected / projected.std(dim=0, unbiased=False).clamp_min(1e-12)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projected, cache_path)
    return projected


def initialize_pca_embeddings(
    model: RobertaClassifier,
    repo_root: Path,
    experiment_root: Path,
) -> None:
    """Project BamiBERT word embeddings and match the random-init column scale."""
    word_embeddings = model.encoder.embeddings.word_embeddings.weight
    hidden_size = word_embeddings.shape[1]
    local_cache = experiment_root / "results" / "pca" / f"bamibert_pca_{hidden_size}.pt"
    shared_cache = (
        repo_root
        / "experiments"
        / "goiturn-arch-scaling"
        / "results"
        / "pca"
        / f"bamibert_pca_{hidden_size}.pt"
    )
    cache_path = shared_cache if shared_cache.is_file() else local_cache
    projected = _standardized_pca_projection(
        repo_root / "models" / "bamibert", hidden_size, cache_path
    )
    target_std = word_embeddings.detach().float().std(dim=0, unbiased=False).cpu()
    initialized = projected * target_std
    with torch.no_grad():
        word_embeddings.copy_(
            initialized.to(dtype=word_embeddings.dtype, device=word_embeddings.device)
        )


def full_architecture(architecture: dict[str, int]) -> dict[str, Any]:
    return {
        **architecture,
        "vocab_size": VOCAB_SIZE,
        "type_vocab_size": TYPE_VOCAB_SIZE,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "position_embedding_type": "absolute",
        "hidden_act": "gelu",
    }


def build_model(
    model_name: str,
    n_aspects: int,
    repo_root: Path,
    experiment_root: Path,
    *,
    pretrained: bool,
    init_dir: Path | None = None,
    pca_init: bool = False,
    require_attentions: bool = False,
) -> tuple[RobertaClassifier, dict[str, Any], dict[str, int]]:
    """Build one family member and print exact ``p.numel()`` counts."""
    architectures = load_architectures(experiment_root)
    if model_name not in architectures:
        raise ValueError(f"Unknown family model {model_name!r}")
    if pretrained != (model_name == "goibert_b"):
        raise ValueError("Only goibert_b may load BamiBERT pretrained weights")
    if init_dir is not None and pca_init:
        raise ValueError("A checkpoint initialization cannot be combined with PCA initialization")
    architecture = architectures[model_name]
    config = make_config(architecture, require_attentions=require_attentions)
    if init_dir is not None:
        encoder = RobertaModel.from_pretrained(
            init_dir,
            config=config,
            local_files_only=True,
            add_pooling_layer=False,
        )
    elif pretrained:
        encoder = RobertaModel.from_pretrained(
            repo_root / "models" / "bamibert",
            config=config,
            local_files_only=True,
            add_pooling_layer=False,
        )
    else:
        encoder = RobertaModel(config, add_pooling_layer=False)
    output_size = 3 if n_aspects == 0 else n_aspects * 4
    model = RobertaClassifier(encoder, output_size, n_aspects)
    if pca_init:
        initialize_pca_embeddings(model, repo_root, experiment_root)
    counts = parameter_counts(model)
    expected = expected_encoder_parameters(architecture)
    if counts["encoder_and_embeddings"] != expected:
        raise RuntimeError(
            f"{model_name} encoder count {counts['encoder_and_embeddings']:,} != {expected:,}"
        )
    print(
        f"Built {model_name}: encoder+embeddings={counts['encoder_and_embeddings']:,}; "
        f"head={counts['head']:,}; total={counts['model_total']:,}",
        flush=True,
    )
    return model, full_architecture(architecture), counts
