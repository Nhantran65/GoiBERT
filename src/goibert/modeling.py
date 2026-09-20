"""Model construction for the E1 supervised baseline.

This module has no task-specific data or metric knowledge.  It only owns the
contract that every trial starts from a fresh ``goiturn-s-gd`` encoder and a
new, randomly initialised classification head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import RobertaModel


HEAD_DROPOUT = 0.1
HEAD_HIDDEN_SIZE = 384
ABSA_NUM_CLASSES = 4


@dataclass(frozen=True)
class ModelSpec:
    """Stable model metadata written into every trial artifact."""

    backbone: str
    hidden_size: int
    head_hidden_size: int
    head_dropout: float
    num_labels: int
    num_aspects: int
    aspect_class_counts: tuple[int, ...]
    total_parameters: int
    trainable_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClassificationHead(nn.Module):
    """CLS -> dropout(0.1) -> dense(384) -> tanh -> dropout(0.1) -> logits.

    The shape is deliberately fixed for E1 so that LR comparisons only vary
    the declared hyperparameter.  Uniform ABSA tasks use four classes per
    aspect.  A task may pass variable slot widths (for example 10x4 + 1x2
    when OTHERS is binary); those logits stay flat and are split by helpers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        dropout: float = HEAD_DROPOUT,
        hidden_size: int = HEAD_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # RoBERTa's first position is the <s>/CLS representation.
        pooled = hidden_states[:, 0]
        return self.output(
            self.dropout(self.activation(self.dense(self.input_dropout(pooled))))
        )


class GoiBERTClassifier(nn.Module):
    """A fully trainable GoiTurn-S-GD encoder with a fresh task head."""

    def __init__(
        self,
        encoder: RobertaModel,
        *,
        num_labels: int,
        num_aspects: int = 0,
        head_dropout: float = HEAD_DROPOUT,
        head_hidden_size: int = HEAD_HIDDEN_SIZE,
        backbone_name: str = "goiturn-s-gd",
        aspect_class_counts: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        if num_aspects < 0:
            raise ValueError("num_aspects must be non-negative")
        counts = tuple(int(value) for value in (aspect_class_counts or ()))
        if any(value <= 0 for value in counts):
            raise ValueError("aspect_class_counts must contain positive class counts")
        if counts and len(counts) != num_aspects:
            raise ValueError("aspect_class_counts length must equal num_aspects")
        if counts and sum(counts) != num_labels:
            raise ValueError("num_labels must equal sum(aspect_class_counts)")
        if num_aspects and not counts and num_labels != num_aspects * ABSA_NUM_CLASSES:
            raise ValueError(
                "Uniform ABSA classifier output must be num_aspects * 4; "
                f"got num_labels={num_labels}, num_aspects={num_aspects}"
            )
        self.encoder = encoder
        self.num_labels = num_labels
        self.num_aspects = num_aspects
        self.aspect_class_counts = counts or ((ABSA_NUM_CLASSES,) * num_aspects)
        self.head = ClassificationHead(
            encoder.config.hidden_size,
            num_labels,
            dropout=head_dropout,
            hidden_size=head_hidden_size,
        )
        # This explicit call documents the E1 rule and protects against a
        # future helper accidentally freezing the loaded backbone.
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        self._backbone_name = backbone_name
        self._head_dropout = head_dropout
        self._head_hidden_size = head_hidden_size

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        encoder_kwargs: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs, return_dict=True)
        logits = self.head(outputs.last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, ABSA_NUM_CLASSES)
        return logits

    def spec(self) -> ModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return ModelSpec(
            backbone=self._backbone_name,
            hidden_size=int(self.encoder.config.hidden_size),
            head_hidden_size=self._head_hidden_size,
            head_dropout=self._head_dropout,
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
        )


def build_classifier(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    local_files_only: bool = True,
    attention_implementation: str | None = None,
) -> tuple[GoiBERTClassifier, ModelSpec]:
    """Load one independent encoder and attach a newly initialised head.

    ``from_pretrained`` is deliberately called inside this function for every
    trial.  Callers must not pass an already fine-tuned model or state dict.
    """

    if not backbone_dir.is_dir():
        raise FileNotFoundError(f"GoiBERT backbone directory does not exist: {backbone_dir}")
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "add_pooling_layer": False,
    }
    if attention_implementation is not None:
        # Select the attention backend at construction time.  Mutating the
        # config after ``from_pretrained`` is too late for all Transformers
        # versions used by this project.
        kwargs["attn_implementation"] = attention_implementation
    encoder = RobertaModel.from_pretrained(backbone_dir, **kwargs)
    if aspect_class_counts is not None and num_classes != sum(aspect_class_counts):
        raise ValueError(
            f"num_classes={num_classes} does not match aspect_class_counts sum="
            f"{sum(aspect_class_counts)}"
        )
    model = GoiBERTClassifier(
        encoder,
        num_labels=num_classes if not num_aspects else num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
    )
    return model, model.spec()
