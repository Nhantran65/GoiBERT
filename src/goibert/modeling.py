"""Task models: an encoder with a freshly initialised head.

* :class:`SequenceClassifier` puts a head on the first (``<s>``) token:
  dropout -> dense -> tanh -> dropout -> logits.  It serves single-label
  classification, sentence-pair NLI and ABSA (one output slot per aspect).
* :class:`TokenClassifier` puts dropout -> linear on every token (NER).

Head widths: 384 for the GoiBERT students, 768 for BamiBERT and 1024 for
CafeBERT.  Parameter names are ``encoder.*`` and ``head.*``; the LLRD
optimizer relies on them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from torch import Tensor, nn
from transformers import AutoModel, RobertaModel

HEAD_DROPOUT = 0.1
STUDENT_HEAD_HIDDEN_SIZE = 384
ABSA_NUM_CLASSES = 4

CAFEBERT_REPO = "uitnlp/CafeBERT"
CAFEBERT_REVISION = "af76fcf2a04096b2b54b348a3e4eb48253c93c5d"
CAFEBERT_TOKENIZER_REPO = "FacebookAI/xlm-roberta-large"
CAFEBERT_TOKENIZER_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"


class ClassificationHead(nn.Module):
    """First token -> dropout -> dense -> tanh -> dropout -> logits."""

    def __init__(self, input_size: int, output_size: int, *, hidden_size: int, dropout: float = HEAD_DROPOUT) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        pooled = hidden_states[:, 0]
        return self.output(self.dropout(self.activation(self.dense(self.input_dropout(pooled)))))


class SequenceClassifier(nn.Module):
    """A fully trainable encoder with a new classification head.

    For ABSA tasks the output is one block of classes per aspect slot.  When
    every slot has four classes the logits are reshaped to
    ``[batch, aspects, 4]``; otherwise they stay flat as
    ``[batch, sum(slot widths)]``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_labels: int,
        head_hidden_size: int,
        num_aspects: int = 0,
        aspect_class_counts: Sequence[int] | None = None,
        head_dropout: float = HEAD_DROPOUT,
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
            raise ValueError(f"Uniform ABSA output must be num_aspects * 4, got {num_labels}")
        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.num_aspects = int(num_aspects)
        self.aspect_class_counts = counts or ((ABSA_NUM_CLASSES,) * num_aspects)
        self.head = ClassificationHead(
            int(encoder.config.hidden_size), self.num_labels, hidden_size=head_hidden_size, dropout=head_dropout
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, token_type_ids: Tensor | None = None) -> Tensor:
        kwargs: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        logits = self.head(self.encoder(**kwargs, return_dict=True).last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, ABSA_NUM_CLASSES)
        return logits


class TokenClassifier(nn.Module):
    """Encoder with a per-token dropout -> linear head (NER)."""

    def __init__(self, encoder: nn.Module, num_labels: int, *, dropout: float = HEAD_DROPOUT) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(int(encoder.config.hidden_size), num_labels))

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor | None = None) -> Tensor:
        kwargs: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self.head(self.encoder(**kwargs, return_dict=True).last_hidden_state)


def load_encoder(source: str | Path, *, kind: str, local_files_only: bool = False) -> nn.Module:
    """Load an encoder without pooler, with SDPA attention.

    ``kind`` is ``student`` or ``bamibert`` (a local or Hub RoBERTa
    checkpoint) or ``cafebert`` (``uitnlp/CafeBERT`` at a pinned revision).
    """

    kwargs: dict[str, Any] = {"add_pooling_layer": False, "attn_implementation": "sdpa", "local_files_only": local_files_only}
    if kind == "cafebert":
        encoder = AutoModel.from_pretrained(CAFEBERT_REPO, revision=CAFEBERT_REVISION, **kwargs)
        expected = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}
    elif kind in ("student", "bamibert"):
        encoder = RobertaModel.from_pretrained(source, **kwargs)
        expected = (
            {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072}
            if kind == "bamibert"
            else {}
        )
    else:
        raise ValueError(f"Unknown encoder kind {kind!r}")
    actual = {name: int(getattr(encoder.config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"{kind} encoder architecture mismatch: expected {expected}, got {actual}")
    return encoder


def head_hidden_size(kind: str, encoder: nn.Module) -> int:
    return STUDENT_HEAD_HIDDEN_SIZE if kind == "student" else int(encoder.config.hidden_size)


def build_model(
    source: str | Path,
    *,
    kind: str,
    task_type: str,
    num_labels: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    local_files_only: bool = False,
) -> nn.Module:
    """Load the encoder, then attach a newly initialised head for ``task_type``.

    ``task_type`` is ``sequence`` or ``token``.
    """

    encoder = load_encoder(source, kind=kind, local_files_only=local_files_only)
    if task_type == "token":
        return TokenClassifier(encoder, num_labels)
    return SequenceClassifier(
        encoder,
        num_labels=num_labels,
        head_hidden_size=head_hidden_size(kind, encoder),
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts if num_aspects else None,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
