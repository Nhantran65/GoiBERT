"""E2 model construction.

The student constructor delegates to the frozen E1 constructor.  This is
deliberate: E2 must start every student run from the original
``goiturn-s-gd`` checkpoint with the same E1 head and with every encoder
parameter trainable.  Only the DAPT teacher gets the E2-specific 768-wide
task head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import RobertaModel

from goibert.modeling import (
    HEAD_DROPOUT,
    ClassificationHead,
    GoiBERTClassifier,
    build_classifier,
)


@dataclass(frozen=True)
class E2ModelSpec:
    """Serializable model identity used in E2 metadata."""

    role: str
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


class E2TeacherClassifier(nn.Module):
    """DAPT teacher encoder plus a fresh task classification head.

    ``goibert-b-dapt`` is a ``RobertaForMaskedLM`` checkpoint.  Loading it as
    ``RobertaModel(add_pooling_layer=False)`` discards the MLM decoder and the
    unused pooler while retaining the exact encoder weights.  The head uses a
    768-wide hidden layer, matching the teacher representation width.
    """

    def __init__(
        self,
        encoder: RobertaModel,
        *,
        num_labels: int,
        num_aspects: int = 0,
        aspect_class_counts: Sequence[int] | None = None,
        head_dropout: float = HEAD_DROPOUT,
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        counts = tuple(int(value) for value in (aspect_class_counts or ()))
        if num_aspects < 0:
            raise ValueError("num_aspects must be non-negative")
        if counts and len(counts) != num_aspects:
            raise ValueError("aspect_class_counts length must equal num_aspects")
        if counts and sum(counts) != num_labels:
            raise ValueError("num_labels must equal sum(aspect_class_counts)")
        if num_aspects and not counts:
            raise ValueError("ABSA teacher requires an explicit slot schema")
        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.num_aspects = int(num_aspects)
        self.aspect_class_counts = counts
        self.head = ClassificationHead(
            int(encoder.config.hidden_size),
            num_labels,
            dropout=head_dropout,
            hidden_size=int(encoder.config.hidden_size),
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        self._head_dropout = float(head_dropout)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        kwargs: dict[str, Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs, return_dict=True)
        logits = self.head(output.last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return logits

    def spec(self) -> E2ModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return E2ModelSpec(
            role="teacher",
            backbone="goibert-b-dapt",
            hidden_size=int(self.encoder.config.hidden_size),
            head_hidden_size=int(self.encoder.config.hidden_size),
            head_dropout=self._head_dropout,
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
        )


def build_e2_student(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    attention_implementation: str = "sdpa",
) -> tuple[GoiBERTClassifier, Any]:
    """Build a fresh E1-identical student for every E2 run."""

    model, spec = build_classifier(
        backbone_dir,
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts if num_aspects else None,
        local_files_only=True,
        attention_implementation=attention_implementation,
    )
    return model, spec


def build_e2_teacher(
    teacher_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    attention_implementation: str = "sdpa",
) -> tuple[E2TeacherClassifier, E2ModelSpec]:
    """Load the local DAPT encoder and attach the standard 768-wide head."""

    if not teacher_dir.is_dir():
        raise FileNotFoundError(f"Teacher directory does not exist: {teacher_dir}")
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "add_pooling_layer": False,
    }
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation
    encoder = RobertaModel.from_pretrained(teacher_dir, **kwargs)
    if int(encoder.config.hidden_size) != 768:
        raise ValueError(
            "E2 teacher contract requires hidden_size=768; "
            f"found {encoder.config.hidden_size}"
        )
    model = E2TeacherClassifier(
        encoder,
        num_labels=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
    )
    return model, model.spec()

