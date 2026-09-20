"""Label-aware attention model for the isolated E9 experiment.

E9 keeps the original GoiTurn-S-GD encoder and the original classifier output
matrix shape.  The readout changes from one CLS vector to one learned query
per VSFC class or ABSA aspect.  Each query is mapped to its corresponding
row (or contiguous row group) of the original output ``Linear`` layer; this
avoids introducing an unnecessary class-by-class projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, RobertaModel

from goibert.modeling import ABSA_NUM_CLASSES, HEAD_DROPOUT, HEAD_HIDDEN_SIZE


E9_QUERY_STD = 0.02
E9_ATTENTION_HIDDEN_SIZE = HEAD_HIDDEN_SIZE


def _canonical_counts(
    num_labels: int,
    num_aspects: int,
    aspect_class_counts: Sequence[int] | None,
) -> tuple[int, ...]:
    counts = tuple(int(value) for value in (aspect_class_counts or ()))
    if num_aspects < 0:
        raise ValueError("num_aspects must be non-negative")
    if any(value <= 0 for value in counts):
        raise ValueError("aspect_class_counts must contain positive values")
    if counts and len(counts) != num_aspects:
        raise ValueError("aspect_class_counts length must equal num_aspects")
    if counts and sum(counts) != num_labels:
        raise ValueError("num_labels must equal sum(aspect_class_counts)")
    if num_aspects and not counts:
        if num_labels != num_aspects * ABSA_NUM_CLASSES:
            raise ValueError("uniform ABSA output must be num_aspects * 4")
        return (ABSA_NUM_CLASSES,) * num_aspects
    return counts


def content_attention_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    special_token_ids: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """Return ``(attention_allowed, content_mask)`` with CLS empty fallback.

    ``attention_allowed`` has at least one true position per row.  It contains
    only non-padding, non-special positions for normal rows; rows with no
    content token contain position zero (CLS) only.
    """

    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask shapes must match")
    if input_ids.shape[1] == 0:
        raise ValueError("sequence length must be positive")
    valid = attention_mask.to(dtype=torch.bool)
    special = torch.zeros_like(valid)
    for token_id in special_token_ids:
        special |= input_ids.eq(int(token_id))
    content = valid & ~special
    allowed = content.clone()
    empty = ~content.any(dim=1)
    if empty.any():
        allowed[empty] = False
        allowed[empty, 0] = True
    return allowed, content


class LabelAwareAttentionHead(nn.Module):
    """Learned-query readout with the E1 dense/dropout/output structure."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        num_aspects: int,
        aspect_class_counts: Sequence[int] | None,
        special_token_ids: Sequence[int],
        dropout: float = HEAD_DROPOUT,
        hidden_size: int = HEAD_HIDDEN_SIZE,
        query_size: int = E9_ATTENTION_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        if input_size != query_size:
            raise ValueError(
                f"E9 query size must equal encoder hidden size ({query_size}), got {input_size}"
            )
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        counts = _canonical_counts(output_size, num_aspects, aspect_class_counts)
        query_count = num_aspects if num_aspects else output_size
        if query_count <= 0:
            raise ValueError("E9 requires at least one attention query")
        self.num_aspects = int(num_aspects)
        self.num_labels = int(output_size)
        self.aspect_class_counts = counts
        self.query_size = int(query_size)
        self.query_count = int(query_count)
        self.special_token_ids = tuple(int(value) for value in special_token_ids)
        if not self.special_token_ids:
            raise ValueError("E9 requires tokenizer.all_special_ids")

        # Keep every new trainable parameter under head.*.  Queries use the
        # requested N(0, 0.02) initializer; dense/output preserve E1 module
        # names and dimensions, including output rows.
        self.queries = nn.Parameter(torch.empty(query_count, query_size))
        nn.init.normal_(self.queries, mean=0.0, std=E9_QUERY_STD)
        self.input_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self._head_dropout = float(dropout)
        self._head_hidden_size = int(hidden_size)

    def _attention_weights(
        self, hidden_states: Tensor, input_ids: Tensor, attention_mask: Tensor
    ) -> Tensor:
        allowed, _ = content_attention_mask(
            input_ids, attention_mask, self.special_token_ids
        )
        # Scores and softmax are explicitly FP32.  Cast only the normalized
        # weights back to the encoder dtype for the context reduction.
        hidden_float = hidden_states.float()
        query_float = self.queries.float()
        # The caller wraps the full model in BF16 autocast.  Disabling it for
        # this block makes the FP32 score and softmax contract explicit even
        # on CUDA kernels that otherwise autocast einsum/softmax.
        with torch.autocast(
            device_type=hidden_states.device.type,
            enabled=False,
        ):
            scores = torch.einsum("bsh,qh->bqs", hidden_float, query_float)
            scores = scores / (float(self.query_size) ** 0.5)
            scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
        return weights

    def _contexts(self, hidden_states: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        weights = self._attention_weights(hidden_states, input_ids, attention_mask)
        return torch.einsum("bqs,bsh->bqh", weights.to(hidden_states.dtype), hidden_states)

    def _logits(self, transformed: Tensor) -> Tensor:
        if not self.num_aspects:
            # One output row per class/query.  This is equivalent to applying
            # each row of the old output Linear to its corresponding query.
            return torch.einsum("bqh,qh->bq", transformed, self.output.weight) + self.output.bias

        pieces: list[Tensor] = []
        offset = 0
        for query_index, width in enumerate(self.aspect_class_counts):
            rows = self.output.weight[offset : offset + width]
            bias = self.output.bias[offset : offset + width]
            pieces.append(torch.nn.functional.linear(transformed[:, query_index, :], rows, bias))
            offset += width
        flat = torch.cat(pieces, dim=-1)
        if len(set(self.aspect_class_counts)) == 1:
            return flat.reshape(flat.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return flat

    def forward(self, hidden_states: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("hidden states and token masks have incompatible shapes")
        context = self._contexts(hidden_states, input_ids, attention_mask)
        cls = hidden_states[:, 0, :].unsqueeze(1)
        readout = (cls + context) / 2
        hidden = self.input_dropout(readout)
        hidden = self.dense(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self._logits(hidden)

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "name": "label_aware_attention",
            "query_count": self.query_count,
            "query_target": "aspect" if self.num_aspects else "class",
            "query_hidden_size": self.query_size,
            "query_initializer": "normal(std=0.02)",
            "attention": {
                "formula": "softmax_fp32(q @ h / sqrt(384))",
                "mask": "attention_mask and not tokenizer.all_special_ids",
                "empty_fallback": "CLS as sole token",
                "context_weights_cast_to_hidden_dtype": True,
            },
            "readout": "(CLS + weighted_context) / 2",
            "head": "dropout0.1 -> dense384 -> tanh -> dropout0.1",
            "output": {
                "layout": "E1 output Linear rows",
                "vsfc": "query q uses output row q",
                "absa": "aspect q uses contiguous output rows for that aspect",
            },
            "num_labels": self.num_labels,
            "num_aspects": self.num_aspects,
            "aspect_class_counts": list(self.aspect_class_counts),
            "special_token_ids": list(self.special_token_ids),
        }


class E9GoiBERTClassifier(nn.Module):
    """Fully trainable original encoder with E9 label-aware readout."""

    def __init__(
        self,
        encoder: RobertaModel,
        *,
        num_labels: int,
        num_aspects: int = 0,
        aspect_class_counts: Sequence[int] | None = None,
        special_token_ids: Sequence[int],
        head_dropout: float = HEAD_DROPOUT,
        head_hidden_size: int = HEAD_HIDDEN_SIZE,
        backbone_name: str = "goiturn-s-gd",
    ) -> None:
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.num_aspects = int(num_aspects)
        self.aspect_class_counts = _canonical_counts(
            self.num_labels, self.num_aspects, aspect_class_counts
        )
        self.head = LabelAwareAttentionHead(
            int(encoder.config.hidden_size),
            self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts if self.num_aspects else None,
            special_token_ids=special_token_ids,
            dropout=head_dropout,
            hidden_size=head_hidden_size,
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        self._backbone_name = backbone_name
        self._head_dropout = float(head_dropout)
        self._head_hidden_size = int(head_hidden_size)

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
        outputs = self.encoder(**kwargs, return_dict=True)
        return self.head(outputs.last_hidden_state, input_ids, attention_mask)

    def spec(self) -> "E9ModelSpec":
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return E9ModelSpec(
            backbone=self._backbone_name,
            hidden_size=int(self.encoder.config.hidden_size),
            head_hidden_size=self._head_hidden_size,
            head_dropout=self._head_dropout,
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
            architecture=self.head.architecture_metadata(),
        )


@dataclass(frozen=True)
class E9ModelSpec:
    """Serializable model identity used in E9 checkpoint metadata."""

    backbone: str
    hidden_size: int
    head_hidden_size: int
    head_dropout: float
    num_labels: int
    num_aspects: int
    aspect_class_counts: tuple[int, ...]
    total_parameters: int
    trainable_parameters: int
    architecture: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        # Keep JSON metadata canonical and avoid tuple/list mismatches when a
        # checkpoint is loaded through JSON and reconstructed by the builder.
        return {
            "backbone": self.backbone,
            "hidden_size": self.hidden_size,
            "head_hidden_size": self.head_hidden_size,
            "head_dropout": self.head_dropout,
            "num_labels": self.num_labels,
            "num_aspects": self.num_aspects,
            "aspect_class_counts": list(self.aspect_class_counts),
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "architecture": _jsonable(self.architecture),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def build_e9_student(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    local_files_only: bool = True,
    attention_implementation: str = "sdpa",
) -> tuple[E9GoiBERTClassifier, E9ModelSpec]:
    """Load a fresh original encoder and attach E9's fresh head."""

    if not backbone_dir.is_dir():
        raise FileNotFoundError(f"GoiBERT backbone directory does not exist: {backbone_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        backbone_dir,
        local_files_only=local_files_only,
        use_fast=True,
    )
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "add_pooling_layer": False,
    }
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation
    encoder = RobertaModel.from_pretrained(backbone_dir, **kwargs)
    if int(encoder.config.hidden_size) != E9_ATTENTION_HIDDEN_SIZE:
        raise ValueError(
            f"E9 requires encoder hidden size {E9_ATTENTION_HIDDEN_SIZE}; "
            f"found {encoder.config.hidden_size}"
        )
    model = E9GoiBERTClassifier(
        encoder,
        num_labels=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
        special_token_ids=tokenizer.all_special_ids,
        backbone_name="goiturn-s-gd",
    )
    return model, model.spec()
