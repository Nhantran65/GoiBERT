"""Matched eager attention in CafeBERT's top four layers.

MHA, diagonal and full cross-head arms share the same implementation. The
operator acts on finite QK scores before the additive padding mask and on
attention probabilities before dropout, as in Talking-Heads Attention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaSelfAttention

TOP_LAYERS = tuple(range(20, 24))
ARMS = ("mha", "diagonal", "full")


class CrossHeadSelfAttention(XLMRobertaSelfAttention):
    def __init__(self, config, arm: str):
        super().__init__(config)
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        heads = self.num_attention_heads
        if heads != 16 or self.position_embedding_type != "absolute" or self.is_decoder:
            raise ValueError("pilot requires CafeBERT encoder with 16 heads and absolute positions")
        if arm == "diagonal":
            self.logit_mix = nn.Parameter(torch.ones(heads))
            self.weight_mix = nn.Parameter(torch.ones(heads))
        elif arm == "full":
            self.logit_mix = nn.Parameter(torch.eye(heads))
            self.weight_mix = nn.Parameter(torch.eye(heads))

    def _mix(self, tensor: Tensor, matrix: Tensor) -> Tensor:
        if self.arm == "diagonal":
            return tensor * matrix.view(1, -1, 1, 1)
        return torch.einsum("bhqk,ho->boqk", tensor, matrix)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[Tensor, ...]:
        if encoder_hidden_states is not None or encoder_attention_mask is not None or past_key_value is not None:
            raise ValueError("pilot supports encoder self-attention only")
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if self.arm != "mha":
            scores = self._mix(scores, self.logit_mix)
        # Mask only after head mixing: multiplying -inf by an off-diagonal zero
        # would otherwise yield NaNs.
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = nn.functional.softmax(scores, dim=-1)
        if self.arm != "mha":
            probs = self._mix(probs, self.weight_mix)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask
        context = torch.matmul(probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.all_head_size)
        return (context, probs) if output_attentions else (context,)


def retrofit(encoder: nn.Module, arm: str) -> dict[str, object]:
    if arm not in ARMS:
        raise ValueError(arm)
    layers = encoder.encoder.layer
    if len(layers) != 24:
        raise ValueError(f"expected 24 layers, got {len(layers)}")
    loaded = []
    for index in TOP_LAYERS:
        old = layers[index].attention.self
        new = CrossHeadSelfAttention(encoder.config, arm)
        info = new.load_state_dict(old.state_dict(), strict=False)
        expected_missing = {"logit_mix", "weight_mix"} if arm != "mha" else set()
        if set(info.missing_keys) != expected_missing or info.unexpected_keys:
            raise RuntimeError(f"attention {index} state mismatch: {info}")
        layers[index].attention.self = new
        loaded.append(index)
    return {"arm": arm, "top_layers": loaded, "extra_parameters": sum(p.numel() for layer in (layers[i] for i in TOP_LAYERS) for name, p in layer.attention.self.named_parameters() if name in ("logit_mix", "weight_mix"))}
