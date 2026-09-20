"""Mean-pooling adapter for the isolated E5 screening experiment.

E5 keeps the E1 encoder and classification-head parameters exactly intact.  It
only replaces the first-token vector with the arithmetic mean of non-padding,
non-special hidden states.  The original CLS vector is an explicit fallback
when an input contains no valid content token.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer

from goibert.modeling import GoiBERTClassifier, ModelSpec, build_classifier


E5_POOLING_NAME = "content_mean_excluding_padding_and_special_tokens"
E5_FALLBACK_NAME = "first_token_cls_when_no_content_tokens"


def mean_pool_hidden_states(
    hidden_states: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    special_token_ids: Sequence[int],
) -> tuple[Tensor, Tensor]:
    """Return ``(pooled, valid_content_counts)`` for a padded batch.

    ``attention_mask`` excludes padding.  Special IDs are then excluded from
    the remaining positions.  The count is clamped only for division; rows
    with zero content positions are replaced by their original CLS vector.
    """

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must have shape [batch, sequence]")
    if input_ids.shape != attention_mask.shape or input_ids.shape[:2] != hidden_states.shape[:2]:
        raise ValueError("hidden states and token masks have incompatible shapes")

    valid = attention_mask.to(dtype=torch.bool)
    special = torch.zeros_like(valid)
    for token_id in special_token_ids:
        special |= input_ids.eq(int(token_id))
    content = valid & ~special
    counts = content.sum(dim=1, keepdim=True)
    weights = content.unsqueeze(-1).to(dtype=hidden_states.dtype)
    summed = (hidden_states * weights).sum(dim=1)
    denominator = counts.to(dtype=hidden_states.dtype).clamp_min(1)
    pooled = summed / denominator
    fallback = hidden_states[:, 0]
    pooled = torch.where(counts.gt(0), pooled, fallback)
    return pooled, counts.squeeze(1)


class MeanPooledGoiBERTClassifier(GoiBERTClassifier):
    """E1 classifier with the same modules and a different pooling operation.

    The constructor deliberately does not call ``GoiBERTClassifier.__init__``:
    doing so would create and randomly initialise a second head.  Instead it
    reuses the already-created E1 encoder and head, preserving both state keys
    and the exact head RNG draw.
    """

    def __init__(self, base: GoiBERTClassifier, special_token_ids: Sequence[int]) -> None:
        nn.Module.__init__(self)
        self.encoder = base.encoder
        self.head = base.head
        self.num_labels = base.num_labels
        self.num_aspects = base.num_aspects
        self.aspect_class_counts = base.aspect_class_counts
        self._backbone_name = base._backbone_name
        self._head_dropout = base._head_dropout
        self._head_hidden_size = base._head_hidden_size
        self.special_token_ids = tuple(int(value) for value in special_token_ids)
        if not self.special_token_ids:
            raise ValueError("E5 requires tokenizer.all_special_ids")

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
        pooled, _ = mean_pool_hidden_states(
            outputs.last_hidden_state,
            input_ids,
            attention_mask,
            self.special_token_ids,
        )
        # This is exactly ClassificationHead.forward after its ``[:, 0]``
        # selection, including the same modules and ordering.
        hidden = self.head.input_dropout(pooled)
        hidden = self.head.dense(hidden)
        hidden = self.head.activation(hidden)
        hidden = self.head.dropout(hidden)
        logits = self.head.output(hidden)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return logits

    def pooling_metadata(self) -> dict[str, Any]:
        return {
            "name": E5_POOLING_NAME,
            "excluded": ["padding", "tokenizer.all_special_ids"],
            "special_token_ids": list(self.special_token_ids),
            "fallback": E5_FALLBACK_NAME,
        }


def build_mean_pool_classifier(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    local_files_only: bool = True,
    attention_implementation: str | None = None,
) -> tuple[MeanPooledGoiBERTClassifier, ModelSpec]:
    """Build E1's fresh classifier, then wrap its exact encoder and head."""

    tokenizer = AutoTokenizer.from_pretrained(
        backbone_dir,
        local_files_only=local_files_only,
        use_fast=True,
    )
    base, spec = build_classifier(
        backbone_dir,
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
        local_files_only=local_files_only,
        attention_implementation=attention_implementation,
    )
    model = MeanPooledGoiBERTClassifier(base, tokenizer.all_special_ids)
    # The wrapper has the same parameter set as the base classifier.  Keep
    # this assertion close to construction so a future E1 change cannot pass
    # silently with a mismatched checkpoint contract.
    if model.spec().to_dict() != spec.to_dict():
        raise RuntimeError("E5 wrapper changed the E1 model specification")
    return model, spec
