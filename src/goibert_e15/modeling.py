"""E15 centered/uncentered pair-value residual on original BamiBERT.

The module is intentionally isolated from E13's 384-wide student constants.
It patches only BamiBERT attention block index 7, keeps the original SDPA
path and output/dropout/LN sequence, and adds the rank-64 pair branch after
the original attention output projection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaSdpaSelfAttention,
)


INSERTION_LAYER_INDEX = 7
PAIR_RANK = 64
PAIR_CHUNK_SIZE = 64
PAIR_INIT_STD = 0.02
VARIANTS = ("centered", "uncentered")
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = 64
HEAD_DROPOUT = 0.1
HEAD_HIDDEN_SIZE = 768
ABSA_NUM_CLASSES = 4


@dataclass(frozen=True)
class E15ModelSpec:
    role: str
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
    architecture: str
    variant: str
    insertion_layer_index: int
    pair_rank: int
    pair_chunk_size: int
    pair_activation: str
    added_parameter_count: int

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


class E15SdpaSelfAttention(RobertaSdpaSelfAttention):
    """Original SDPA plus a mean-head-routed pair-value message."""

    def __init__(
        self,
        source: RobertaSdpaSelfAttention,
        *,
        variant: str,
        config: Any,
        chunk_size: int = PAIR_CHUNK_SIZE,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown E15 variant {variant!r}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        super().__init__(config)
        self.load_state_dict(source.state_dict(), strict=True)
        hidden = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        if hidden != HIDDEN_SIZE or heads != NUM_HEADS:
            raise ValueError(f"E15 requires hidden={HIDDEN_SIZE}, heads={NUM_HEADS}; got {hidden}/{heads}")
        if hidden // heads != HEAD_DIM:
            raise ValueError("E15 requires 64-dimensional attention heads")
        self.variant = str(variant)
        self.chunk_size = int(chunk_size)
        self.U = nn.Linear(hidden, PAIR_RANK, bias=False)
        self.W = nn.Linear(hidden, PAIR_RANK, bias=False)
        self.P = nn.Linear(PAIR_RANK, hidden, bias=False)
        nn.init.normal_(self.U.weight, mean=0.0, std=PAIR_INIT_STD)
        nn.init.normal_(self.W.weight, mean=0.0, std=PAIR_INIT_STD)
        nn.init.zeros_(self.P.weight)
        self._query_mask: Tensor | None = None
        self.collect_diagnostics = False
        self.register_buffer("last_delta_norm", torch.zeros((), dtype=torch.float32))
        self.register_buffer("last_delta_relative_norm", torch.zeros((), dtype=torch.float32))

    @property
    def added_parameter_count(self) -> int:
        return int(self.U.weight.numel() + self.W.weight.numel() + self.P.weight.numel())

    def set_query_mask(self, attention_mask: Tensor | None) -> None:
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("E15 query mask must be [batch, sequence]")
        self._query_mask = attention_mask

    def set_diagnostics(self, enabled: bool) -> None:
        self.collect_diagnostics = bool(enabled)

    def _attention_scores(self, query: Tensor, key: Tensor, attention_mask: Tensor | None) -> Tensor:
        # This context is deliberately disabled even when the caller is under
        # BF16 autocast: routing scores and softmax must be auditable FP32.
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
            scores = scores / math.sqrt(float(self.attention_head_size))
            if attention_mask is not None:
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
        aux_query = self.U(hidden_states)
        aux_key = self.W(hidden_states)
        scores = self._attention_scores(original_query, original_key, attention_mask)
        with torch.autocast(device_type=scores.device.type, enabled=False):
            mean_attention = torch.softmax(scores, dim=-1).mean(dim=1)
        batch_size, seq_len, _ = hidden_states.shape
        outputs: list[Tensor] = []
        q_gelu: Tensor | None = None
        k_gelu: Tensor | None = None
        if self.variant == "centered":
            q_gelu = torch.nn.functional.gelu(aux_query, approximate="none")
            k_gelu = torch.nn.functional.gelu(aux_key, approximate="none")
        for start in range(0, seq_len, self.chunk_size):
            stop = min(start + self.chunk_size, seq_len)
            q_chunk = aux_query[:, start:stop]
            interaction = torch.nn.functional.gelu(q_chunk.unsqueeze(2) + aux_key.unsqueeze(1), approximate="none")
            if q_gelu is not None and k_gelu is not None:
                interaction = interaction - q_gelu[:, start:stop].unsqueeze(2) - k_gelu.unsqueeze(1)
            weighted = mean_attention[:, start:stop].to(dtype=interaction.dtype).unsqueeze(-1) * interaction
            outputs.append(self.P(weighted.sum(dim=2)))
        delta = torch.cat(outputs, dim=1)
        if self._query_mask is not None:
            delta = delta * self._query_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        if self.collect_diagnostics:
            with torch.no_grad():
                norm = delta.float().norm()
                self.last_delta_norm.copy_(norm)
                self.last_delta_relative_norm.copy_(norm / hidden_states.float().norm().clamp_min(1e-12))
        if delta.shape != (batch_size, seq_len, int(self.all_head_size)):
            raise RuntimeError(f"Unexpected E15 delta shape {tuple(delta.shape)}")
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
        if encoder_hidden_states is not None or encoder_attention_mask is not None or past_key_value is not None or head_mask is not None or output_attentions:
            raise ValueError("E15 supports encoder self-attention with SDPA only")
        bsz, tgt_len, _ = hidden_states.size()
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()
        context = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )
        delta = self._pair_message(hidden_states, query_layer, key_layer, attention_mask)
        context = context.transpose(1, 2).reshape(bsz, tgt_len, self.all_head_size)
        return context, delta


class E15RobertaAttention(nn.Module):
    """Add delta after Wo and before BamiBERT's shared output dropout/LN."""

    def __init__(self, source: RobertaAttention, *, variant: str, config: Any) -> None:
        super().__init__()
        self.self = E15SdpaSelfAttention(source.self, variant=variant, config=config)
        self.output = source.output
        self.pruned_heads = source.pruned_heads

    def prune_heads(self, heads: Any) -> None:
        raise NotImplementedError("E15 does not support pruning after construction")

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
        context, delta = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        projected = self.output.dense(context) + delta
        projected = self.output.dropout(projected)
        return (self.output.LayerNorm(projected + hidden_states),)


class BamiBERTE15Classifier(nn.Module):
    def __init__(
        self,
        encoder: RobertaModel,
        *,
        num_labels: int,
        num_aspects: int = 0,
        aspect_class_counts: Sequence[int] | None = None,
        variant: str,
        pair_chunk_size: int = PAIR_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown E15 variant {variant!r}")
        counts = tuple(int(value) for value in (aspect_class_counts or ()))
        if counts and len(counts) != num_aspects:
            raise ValueError("aspect_class_counts length must equal num_aspects")
        if counts and sum(counts) != num_labels:
            raise ValueError("num_labels must equal sum(aspect_class_counts)")
        if num_aspects and not counts and num_labels != num_aspects * ABSA_NUM_CLASSES:
            raise ValueError("uniform ABSA output size mismatch")
        if int(encoder.config.hidden_size) != HIDDEN_SIZE:
            raise ValueError("E15 classifier requires hidden_size=768")
        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.num_aspects = int(num_aspects)
        self.aspect_class_counts = counts or ((ABSA_NUM_CLASSES,) * num_aspects)
        self.head = ClassificationHead(HIDDEN_SIZE, self.num_labels)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        self.variant = variant
        self.insertion_layer_index = INSERTION_LAYER_INDEX
        self._head_dropout = HEAD_DROPOUT
        self._head_hidden_size = HEAD_HIDDEN_SIZE
        attention = self.encoder.encoder.layer[INSERTION_LAYER_INDEX].attention
        if not isinstance(attention, E15RobertaAttention):
            raise TypeError("E15 classifier requires an E15 attention replacement")
        attention.self.chunk_size = int(pair_chunk_size)

    @property
    def e15_attention(self) -> E15RobertaAttention:
        attention = self.encoder.encoder.layer[self.insertion_layer_index].attention
        if not isinstance(attention, E15RobertaAttention):
            raise TypeError("E15 attention was replaced after construction")
        return attention

    @property
    def pair_module(self) -> E15SdpaSelfAttention:
        return self.e15_attention.self

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, token_type_ids: Tensor | None = None) -> Tensor:
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

    def spec(self) -> E15ModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        config = self.encoder.config
        return E15ModelSpec(
            role="bamibert_original_e15",
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
            architecture="original_bamibert_sdpa_plus_block7_pair_value_residual",
            variant=self.variant,
            insertion_layer_index=self.insertion_layer_index,
            pair_rank=PAIR_RANK,
            pair_chunk_size=self.pair_module.chunk_size,
            pair_activation="gelu_exact",
            added_parameter_count=self.pair_module.added_parameter_count,
        )


def build_e15_classifier(
    model_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    variant: str,
    attention_implementation: str = "sdpa",
    pair_chunk_size: int = PAIR_CHUNK_SIZE,
) -> tuple[BamiBERTE15Classifier, E15ModelSpec]:
    if attention_implementation != "sdpa":
        raise ValueError("E15 is locked to the SDPA backend")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    encoder = RobertaModel.from_pretrained(
        model_dir,
        local_files_only=True,
        add_pooling_layer=False,
        attn_implementation=attention_implementation,
    )
    config = encoder.config
    expected = {"hidden_size": HIDDEN_SIZE, "num_hidden_layers": 12, "num_attention_heads": NUM_HEADS, "intermediate_size": 3072}
    actual = {name: int(getattr(config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Official BamiBERT architecture mismatch: expected {expected}, got {actual}")
    head = ClassificationHead(HIDDEN_SIZE, int(num_classes))
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    source_attention = encoder.encoder.layer[INSERTION_LAYER_INDEX].attention
    # Replacement draws are isolated so E14's fresh head initialization and
    # downstream training RNG stream remain exact under the same seed.
    with torch.random.fork_rng(devices=[]):
        replacement = E15RobertaAttention(source_attention, variant=variant, config=config)
    encoder.encoder.layer[INSERTION_LAYER_INDEX].attention = replacement
    model = BamiBERTE15Classifier.__new__(BamiBERTE15Classifier)
    nn.Module.__init__(model)
    model.encoder = encoder
    model.num_labels = int(num_classes)
    model.num_aspects = int(num_aspects)
    counts = tuple(int(value) for value in (aspect_class_counts or ()))
    if counts and len(counts) != num_aspects:
        raise ValueError("aspect_class_counts length must equal num_aspects")
    if counts and sum(counts) != num_classes:
        raise ValueError("num_classes must equal sum(aspect_class_counts)")
    if num_aspects and not counts and num_classes != num_aspects * ABSA_NUM_CLASSES:
        raise ValueError("uniform ABSA output size mismatch")
    model.aspect_class_counts = counts or ((ABSA_NUM_CLASSES,) * num_aspects)
    model.head = head
    model.variant = variant
    model.insertion_layer_index = INSERTION_LAYER_INDEX
    model._head_dropout = HEAD_DROPOUT
    model._head_hidden_size = HEAD_HIDDEN_SIZE
    replacement.self.chunk_size = int(pair_chunk_size)
    expected_added = 3 * HIDDEN_SIZE * PAIR_RANK
    if model.pair_module.added_parameter_count != expected_added:
        raise AssertionError("E15 added parameter count is not exactly 147,456")
    return model, model.spec()
