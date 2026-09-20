"""BamiBERT original encoder plus the frozen E14 task-head contract.

The adapter is deliberately isolated from the frozen E1--E13 modules.  It
loads the official ``RobertaModel`` encoder (there is no MLM head in the
downstream model), checks the expected 12-layer/768-hidden architecture, and
attaches a fresh standard CLS head with hidden width 768.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import RobertaModel


HEAD_DROPOUT = 0.1
HEAD_HIDDEN_SIZE = 768
ABSA_NUM_CLASSES = 4


@dataclass(frozen=True)
class BamiBERTModelSpec:
    backbone: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
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
    """CLS -> dropout(.1) -> dense(768) -> tanh -> dropout(.1) -> logits."""

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
        pooled = hidden_states[:, 0]
        return self.output(self.dropout(self.activation(self.dense(self.input_dropout(pooled)))))


class BamiBERTClassifier(nn.Module):
    def __init__(
        self,
        encoder: RobertaModel,
        *,
        num_labels: int,
        num_aspects: int = 0,
        aspect_class_counts: Sequence[int] | None = None,
        head_dropout: float = HEAD_DROPOUT,
        head_hidden_size: int = HEAD_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        if num_aspects < 0:
            raise ValueError("num_aspects must be non-negative")
        counts = tuple(int(value) for value in (aspect_class_counts or ()))
        if counts and len(counts) != num_aspects:
            raise ValueError("aspect_class_counts length must equal num_aspects")
        if counts and sum(counts) != num_labels:
            raise ValueError("num_labels must equal sum(aspect_class_counts)")
        if num_aspects and not counts and num_labels != num_aspects * ABSA_NUM_CLASSES:
            raise ValueError("uniform ABSA output size mismatch")
        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.num_aspects = int(num_aspects)
        self.aspect_class_counts = counts or ((ABSA_NUM_CLASSES,) * num_aspects)
        self.head = ClassificationHead(
            int(encoder.config.hidden_size),
            self.num_labels,
            dropout=head_dropout,
            hidden_size=head_hidden_size,
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        self._head_dropout = float(head_dropout)
        self._head_hidden_size = int(head_hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        kwargs: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs, return_dict=True)
        logits = self.head(outputs.last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, ABSA_NUM_CLASSES)
        return logits

    def spec(self) -> BamiBERTModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        config = self.encoder.config
        return BamiBERTModelSpec(
            backbone="Qualcomm-AI-Research/BamiBERT",
            hidden_size=int(config.hidden_size),
            num_hidden_layers=int(config.num_hidden_layers),
            num_attention_heads=int(config.num_attention_heads),
            intermediate_size=int(config.intermediate_size),
            head_hidden_size=self._head_hidden_size,
            head_dropout=self._head_dropout,
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
        )


def build_bamibert_classifier(
    model_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    local_files_only: bool = True,
    attention_implementation: str | None = "sdpa",
) -> tuple[BamiBERTClassifier, BamiBERTModelSpec]:
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "add_pooling_layer": False,
    }
    if attention_implementation is not None:
        kwargs["attn_implementation"] = attention_implementation
    encoder = RobertaModel.from_pretrained(model_dir, **kwargs)
    config = encoder.config
    expected = {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072}
    actual = {name: int(getattr(config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Official BamiBERT encoder architecture mismatch: expected {expected}, got {actual}")
    model = BamiBERTClassifier(
        encoder,
        num_labels=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
    )
    return model, model.spec()
