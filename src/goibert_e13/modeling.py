"""E13 centered pair-value residual model.

The experiment keeps the original GoiTurn-S-GD encoder and E8 CLS head.  A
single auxiliary message path is inserted into the attention block at index
6.  The original Q/K/V projections and SDPA path remain intact; the
auxiliary path reuses the original attention routing and adds a pairwise
value residual after the attention output projection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaModel,
    RobertaSdpaSelfAttention,
)

from goibert.modeling import HEAD_DROPOUT, HEAD_HIDDEN_SIZE, ClassificationHead
from goibert_e2.modeling import build_e2_student


INSERTION_LAYER_INDEX = 6
PAIR_RANK = 64
PAIR_CHUNK_SIZE = 64
PAIR_INIT_STD = 0.02
VARIANTS = ("centered", "uncentered")
HIDDEN_SIZE = 384
NUM_HEADS = 6
HEAD_DIM = 64


@dataclass(frozen=True)
class E13ModelSpec:
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
    architecture: str
    variant: str
    insertion_layer_index: int
    pair_rank: int
    pair_chunk_size: int
    pair_activation: str
    added_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class E13SdpaSelfAttention(RobertaSdpaSelfAttention):
    """Original SDPA self-attention plus a chunked pair-value message.

    The module deliberately retains the inherited ``query``, ``key`` and
    ``value`` parameter names.  ``U``, ``W`` and ``P`` are the only added
    trainables and are bias-free.  ``P`` starts at zero, so the first forward
    has the original function while still constructing the branch and
    retaining gradients to ``P``.
    """

    def __init__(
        self,
        source: RobertaSdpaSelfAttention,
        *,
        variant: str,
        config: Any,
        chunk_size: int = PAIR_CHUNK_SIZE,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown E13 variant {variant!r}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        # The caller wraps construction in fork_rng.  Initialising the old
        # inherited module and loading the source state preserves all original
        # attention parameter names and values.
        super().__init__(config)
        self.load_state_dict(source.state_dict(), strict=True)
        hidden = int(config.hidden_size)
        if hidden != HIDDEN_SIZE:
            raise ValueError(f"E13 is locked to hidden_size=384, found {hidden}")
        if int(config.num_attention_heads) != NUM_HEADS:
            raise ValueError(
                f"E13 is locked to six heads, found {config.num_attention_heads}"
            )
        if int(config.hidden_size // config.num_attention_heads) != HEAD_DIM:
            raise ValueError("E13 requires 64-dimensional attention heads")
        self.variant = str(variant)
        self.chunk_size = int(chunk_size)
        self.U = nn.Linear(hidden, PAIR_RANK, bias=False)
        self.W = nn.Linear(hidden, PAIR_RANK, bias=False)
        self.P = nn.Linear(PAIR_RANK, hidden, bias=False)
        nn.init.normal_(self.U.weight, mean=0.0, std=PAIR_INIT_STD)
        nn.init.normal_(self.W.weight, mean=0.0, std=PAIR_INIT_STD)
        nn.init.zeros_(self.P.weight)
        # Set by E13Classifier.forward from the original 2D input mask.  It
        # lets the branch suppress query padding rows while direct module
        # calls can still use the additive key mask supplied by Transformers.
        self._query_mask: Tensor | None = None
        self.collect_diagnostics = False
        self.register_buffer("last_delta_norm", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "last_delta_relative_norm", torch.zeros((), dtype=torch.float32)
        )

    @property
    def added_parameter_count(self) -> int:
        return int(self.U.weight.numel() + self.W.weight.numel() + self.P.weight.numel())

    def set_query_mask(self, attention_mask: Tensor | None) -> None:
        if attention_mask is None:
            self._query_mask = None
            return
        if attention_mask.ndim != 2:
            raise ValueError("E13 query mask must be [batch, sequence]")
        self._query_mask = attention_mask

    def set_diagnostics(self, enabled: bool) -> None:
        self.collect_diagnostics = bool(enabled)

    def _attention_scores(self, query: Tensor, key: Tensor, attention_mask: Tensor | None) -> Tensor:
        """Compute original QK scores in fp32 for the auxiliary routing map."""

        # The routing map is required to stay fp32 even under an outer BF16
        # autocast context.  This keeps the softmax numerics auditable and
        # avoids silently promoting only part of the expression.
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
            scores = scores / math.sqrt(float(self.attention_head_size))
            if attention_mask is not None:
                # Transformers supplies [B,1,1,S] for the SDPA backend.  The
                # singleton head dimension broadcasts over our [B,H,Q,S]
                # scores.
                scores = scores + attention_mask.float()
            elif self._query_mask is not None:
                key_mask = self._query_mask.to(device=scores.device, dtype=torch.bool)
                scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        return scores

    def _pair_message(
        self,
        hidden_states: Tensor,
        original_query: Tensor,
        original_key: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Return ``P sum_j mean_heads(A_ij) r_ij`` with fixed query chunks."""

        # U/W execute under the caller's autocast, then the routing map and
        # softmax are explicitly promoted to fp32 as required by E13.
        aux_query = self.U(hidden_states)
        aux_key = self.W(hidden_states)
        scores = self._attention_scores(original_query, original_key, attention_mask)
        with torch.autocast(device_type=scores.device.type, enabled=False):
            mean_attention = torch.softmax(scores, dim=-1).mean(dim=1)
        batch_size, seq_len, _ = hidden_states.shape
        outputs: list[Tensor] = []
        for start in range(0, seq_len, self.chunk_size):
            stop = min(start + self.chunk_size, seq_len)
            q_chunk = aux_query[:, start:stop]
            # The temporary is bounded by query chunks; its autograd graph is
            # retained as usual so profiling can expose the true backward
            # memory cost.  The runner may switch to checkpointing only if a
            # real profile requires it.
            pair = q_chunk.unsqueeze(2) + aux_key.unsqueeze(1)
            interaction = torch.nn.functional.gelu(pair, approximate="none")
            if self.variant == "centered":
                interaction = interaction - torch.nn.functional.gelu(
                    q_chunk, approximate="none"
                ).unsqueeze(2)
                interaction = interaction - torch.nn.functional.gelu(
                    aux_key, approximate="none"
                ).unsqueeze(1)
            weighted = (
                mean_attention[:, start:stop]
                .to(dtype=interaction.dtype)
                .unsqueeze(-1)
                * interaction
            )
            summary = weighted.sum(dim=2)
            outputs.append(self.P(summary))
        delta = torch.cat(outputs, dim=1)
        query_mask = self._query_mask
        if query_mask is not None:
            delta = delta * query_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        if self.collect_diagnostics:
            with torch.no_grad():
                delta_norm = delta.float().norm()
                self.last_delta_norm.copy_(delta_norm)
                # This is a diagnostic only; the route output is not detached.
                self.last_delta_relative_norm.copy_(
                    delta_norm / hidden_states.float().norm().clamp_min(1e-12)
                )
        if delta.shape != (batch_size, seq_len, int(self.all_head_size)):
            raise RuntimeError(f"Unexpected E13 delta shape {tuple(delta.shape)}")
        return delta

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Tensor, ...]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, ...]:
        if (
            encoder_hidden_states is not None
            or encoder_attention_mask is not None
            or past_key_value is not None
            or head_mask is not None
            or output_attentions
        ):
            raise ValueError("E13 supports encoder self-attention with SDPA only")
        bsz, tgt_len, _ = hidden_states.size()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        if (
            self.require_contiguous_qkv
            and query_layer.device.type == "cuda"
            and attention_mask is not None
        ):
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()
        # This is the original SDPA invocation, including its original
        # attention dropout behavior.  The auxiliary map below has no dropout.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )
        delta = self._pair_message(hidden_states, query_layer, key_layer, attention_mask)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.all_head_size)
        return attn_output, delta


class E13RobertaAttention(nn.Module):
    """Roberta attention wrapper adding delta after Wo and before dropout."""

    def __init__(self, source: RobertaAttention, *, variant: str, config: Any) -> None:
        super().__init__()
        self.self = E13SdpaSelfAttention(source.self, variant=variant, config=config)
        # Reuse the original output module so Wo/dropout/LN parameters and
        # names are unchanged.
        self.output = source.output
        self.pruned_heads = source.pruned_heads

    def prune_heads(self, heads: Any) -> None:
        raise NotImplementedError("E13 does not support pruning after construction")

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[Tensor, ...]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Tensor, ...]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        if len(self_outputs) != 2:
            raise RuntimeError("E13 self-attention did not return context and delta")
        context, delta = self_outputs
        projected = self.output.dense(context)
        projected = projected + delta
        projected = self.output.dropout(projected)
        attention_output = self.output.LayerNorm(projected + hidden_states)
        return (attention_output,)


class E13Classifier(nn.Module):
    """E1/E8-compatible classifier carrying the patched encoder and old head."""

    def __init__(
        self,
        base: nn.Module,
        *,
        variant: str,
        insertion_layer_index: int = INSERTION_LAYER_INDEX,
        pair_chunk_size: int = PAIR_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown E13 variant {variant!r}")
        if insertion_layer_index != INSERTION_LAYER_INDEX:
            raise ValueError("E13 insertion layer is locked to block index 6")
        self.encoder = base.encoder
        self.head = base.head
        self.num_labels = int(base.num_labels)
        self.num_aspects = int(base.num_aspects)
        self.aspect_class_counts = tuple(base.aspect_class_counts)
        self._backbone_name = str(base._backbone_name)
        self._head_dropout = float(base._head_dropout)
        self._head_hidden_size = int(base._head_hidden_size)
        self.variant = str(variant)
        self.insertion_layer_index = int(insertion_layer_index)
        attention = self.encoder.encoder.layer[insertion_layer_index].attention
        if not isinstance(attention, E13RobertaAttention):
            raise TypeError("E13 classifier requires an E13 attention replacement")
        # Keep the requested chunk size visible in the constructor identity.
        attention.self.chunk_size = int(pair_chunk_size)

    @property
    def e13_attention(self) -> E13RobertaAttention:
        """Return the patched attention without registering a second module."""

        attention = self.encoder.encoder.layer[self.insertion_layer_index].attention
        if not isinstance(attention, E13RobertaAttention):
            raise TypeError("E13 attention was replaced after construction")
        return attention

    @property
    def pair_module(self) -> E13SdpaSelfAttention:
        return self.e13_attention.self

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        self.pair_module.set_query_mask(attention_mask)
        kwargs: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        try:
            output = self.encoder(**kwargs, return_dict=True)
        finally:
            self.pair_module.set_query_mask(None)
        logits = self.head(output.last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return logits

    def spec(self) -> E13ModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return E13ModelSpec(
            role="student",
            backbone=self._backbone_name,
            hidden_size=int(self.encoder.config.hidden_size),
            head_hidden_size=self._head_hidden_size,
            head_dropout=self._head_dropout,
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
            architecture="original_sdpa_plus_block6_pair_value_residual",
            variant=self.variant,
            insertion_layer_index=self.insertion_layer_index,
            pair_rank=PAIR_RANK,
            pair_chunk_size=self.pair_module.chunk_size,
            pair_activation="gelu_exact",
            added_parameter_count=self.pair_module.added_parameter_count,
        )


def build_e13_student(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    variant: str,
    attention_implementation: str = "sdpa",
    pair_chunk_size: int = PAIR_CHUNK_SIZE,
) -> tuple[E13Classifier, E13ModelSpec]:
    """Build a fresh E1 student, then patch only block index 6."""

    if attention_implementation != "sdpa":
        raise ValueError("E13 is locked to the original SDPA backend")
    base, _ = build_e2_student(
        backbone_dir,
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts if num_aspects else None,
        attention_implementation=attention_implementation,
    )
    layer = base.encoder.encoder.layer[INSERTION_LAYER_INDEX]
    source_attention = layer.attention
    # No random draw from replacement construction may perturb the fresh head
    # or the subsequent E8 training stream.
    with torch.random.fork_rng(devices=[]):
        layer.attention = E13RobertaAttention(
            source_attention,
            variant=variant,
            config=base.encoder.config,
        )
        layer.attention.self.chunk_size = int(pair_chunk_size)
    model = E13Classifier(
        base,
        variant=variant,
        insertion_layer_index=INSERTION_LAYER_INDEX,
        pair_chunk_size=pair_chunk_size,
    )
    expected_added = 3 * HIDDEN_SIZE * PAIR_RANK
    if model.pair_module.added_parameter_count != expected_added:
        raise AssertionError("E13 added parameter count is not exactly 73,728")
    return model, model.spec()
