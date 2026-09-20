"""E11 block-9 two-map attention with signed/convex controls.

The module keeps the frozen E1/E8 student and head intact.  Only the
self-attention module at encoder block index 8 is replaced.  Q1/K1/V and the
attention output projection retain their original parameter names; Q2/K2 and
per-head gate logits are the only new trainables.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from transformers.models.roberta.modeling_roberta import RobertaSdpaSelfAttention

from goibert.modeling import HEAD_DROPOUT, HEAD_HIDDEN_SIZE, ModelSpec, GoiBERTClassifier
from goibert_e2.modeling import build_e2_student


INSERTION_LAYER_INDEX = 8
NUM_HEADS = 6
QK_NOISE_STD = 0.01
GATE_INIT = 0.1
GATE_INIT_LOGIT = math.log(GATE_INIT / (1.0 - GATE_INIT))
VARIANTS = ("signed", "convex")


@dataclass(frozen=True)
class E11ModelSpec:
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
    variant: str
    insertion_layer_index: int
    qk_noise_std: float
    gate_init: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class E11TwoMapSdpaSelfAttention(RobertaSdpaSelfAttention):
    """Two SDPA maps sharing V, with a checkpointed optimizer-step gate."""

    def __init__(
        self,
        source: RobertaSdpaSelfAttention,
        *,
        variant: str,
        warmup_steps: int = 1,
        qk_noise_std: float = QK_NOISE_STD,
        config: Any | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown E11 variant {variant!r}")
        # Constructing the replacement and added modules under fork_rng keeps
        # old head/model initialization and the training RNG stream unchanged.
        with torch.random.fork_rng(devices=[]):
            super().__init__(config if config is not None else source)
            with torch.no_grad():
                self.query.load_state_dict(source.query.state_dict())
                self.key.load_state_dict(source.key.state_dict())
                self.value.load_state_dict(source.value.state_dict())
        self.variant = variant
        self.qk_noise_std = float(qk_noise_std)
        with torch.random.fork_rng(devices=[]):
            self.q2 = nn.Linear(self.all_head_size, self.all_head_size, bias=True)
            self.k2 = nn.Linear(self.all_head_size, self.all_head_size, bias=True)
            with torch.no_grad():
                self.q2.weight.copy_(self.query.weight + torch.randn_like(self.query.weight) * self.qk_noise_std)
                self.q2.bias.copy_(self.query.bias)
                self.k2.weight.copy_(self.key.weight + torch.randn_like(self.key.weight) * self.qk_noise_std)
                self.k2.bias.copy_(self.key.bias)
            self.gate_logits = nn.Parameter(
                torch.full((self.num_attention_heads,), float(GATE_INIT_LOGIT))
            )
        self.register_buffer("optimizer_step", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "warmup_steps",
            torch.tensor(max(1, int(warmup_steps)), dtype=torch.long),
        )
        self.register_buffer("last_correction_norm", torch.zeros((), dtype=torch.float32))

    def set_optimizer_step(self, step: int) -> None:
        if step < 0:
            raise ValueError("optimizer step must be non-negative")
        self.optimizer_step.fill_(int(step))

    def set_warmup_steps(self, warmup_steps: int) -> None:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        self.warmup_steps.fill_(int(warmup_steps))

    def gate_values(self) -> Tensor:
        ramp = min(1.0, float(self.optimizer_step.item()) / max(1, int(self.warmup_steps.item())))
        return torch.sigmoid(self.gate_logits) * ramp

    def effective_weights(self) -> Tensor:
        """Return per-map effective weights [heads, 2]."""

        lam = self.gate_values().detach()
        if self.variant == "signed":
            return torch.stack((1.0 + lam, -lam), dim=-1)
        return torch.stack((1.0 - lam, lam), dim=-1)

    def _sdpa(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        head_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, ...]:
        if encoder_hidden_states is not None or past_key_value is not None or head_mask is not None or output_attentions:
            raise ValueError("E11 two-map attention supports encoder self-attention only")
        # Exact old path at t=0 is useful for parity and avoids consuming a
        # second dropout stream before the first optimizer step.
        if int(self.optimizer_step.item()) == 0:
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        bsz, tgt_len, _ = hidden_states.size()
        q1 = self.transpose_for_scores(self.query(hidden_states))
        q2 = self.transpose_for_scores(self.q2(hidden_states))
        k1 = self.transpose_for_scores(self.key(hidden_states))
        k2 = self.transpose_for_scores(self.k2(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        o1 = self._sdpa(q1, k1, v, attention_mask)
        o2 = self._sdpa(q2, k2, v, attention_mask)
        lam = self.gate_values().to(dtype=o1.dtype, device=o1.device).view(1, -1, 1, 1)
        if self.variant == "signed":
            context = o1 + lam * (o1 - o2)
        else:
            context = o1 + lam * (o2 - o1)
        with torch.no_grad():
            denom = o1.detach().float().norm().clamp_min(1e-12)
            self.last_correction_norm.copy_((context.detach().float() - o1.detach().float()).norm() / denom)
        context = context.transpose(1, 2).reshape(bsz, tgt_len, self.all_head_size)
        return (context,)


class E11Classifier(nn.Module):
    """E1-compatible classifier carrying the patched encoder and old head."""

    def __init__(self, base: GoiBERTClassifier, *, variant: str) -> None:
        super().__init__()
        self.encoder = base.encoder
        self.head = base.head
        self.num_labels = int(base.num_labels)
        self.num_aspects = int(base.num_aspects)
        self.aspect_class_counts = tuple(base.aspect_class_counts)
        self._backbone_name = str(base._backbone_name)
        self._head_dropout = float(base._head_dropout)
        self._head_hidden_size = int(base._head_hidden_size)
        self.variant = variant

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
        output = self.encoder(**kwargs, return_dict=True)
        logits = self.head(output.last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return logits

    def spec(self) -> E11ModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return E11ModelSpec(
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
            variant=self.variant,
            insertion_layer_index=INSERTION_LAYER_INDEX,
            qk_noise_std=QK_NOISE_STD,
            gate_init=GATE_INIT,
        )

    def architecture_metadata(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "insertion_layer_index": INSERTION_LAYER_INDEX,
            "attention": "two SDPA maps, shared V and output projection",
            "signed_formula": "O1 + lambda_h * (O1 - O2)",
            "convex_formula": "O1 + lambda_h * (O2 - O1)",
            "q2_k2_initialization": "Q1/K1 copied plus Normal(0, 0.01) under fork_rng; biases copied",
            "lambda_schedule": "min(1, optimizer_step / warmup_steps) * sigmoid(gate_logits)",
            "gate_init": GATE_INIT,
            "qk_noise_std": QK_NOISE_STD,
            "warmup_steps": int(next(iter_e11_attentions(self)).warmup_steps.item()),
        }

    def set_optimizer_step(self, step: int) -> None:
        for module in iter_e11_attentions(self):
            module.set_optimizer_step(step)

    def set_warmup_steps(self, warmup_steps: int) -> None:
        for module in iter_e11_attentions(self):
            module.set_warmup_steps(warmup_steps)


def iter_e11_attentions(model: nn.Module) -> Iterable[E11TwoMapSdpaSelfAttention]:
    for module in model.modules():
        if isinstance(module, E11TwoMapSdpaSelfAttention):
            yield module


def build_e11_student(
    backbone_dir: Path,
    *,
    variant: str,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    attention_implementation: str = "sdpa",
    warmup_steps: int = 1,
) -> tuple[E11Classifier, E11ModelSpec]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown E11 variant {variant!r}")
    base, _ = build_e2_student(
        backbone_dir,
        num_classes=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
        attention_implementation=attention_implementation,
    )
    layer = base.encoder.encoder.layer[INSERTION_LAYER_INDEX]
    source = layer.attention.self
    if not isinstance(source, RobertaSdpaSelfAttention):
        raise TypeError(f"Expected RobertaSdpaSelfAttention at block {INSERTION_LAYER_INDEX}, got {type(source)!r}")
    layer.attention.self = E11TwoMapSdpaSelfAttention(
        source,
        variant=variant,
        warmup_steps=warmup_steps,
        config=base.encoder.config,
    )
    model = E11Classifier(base, variant=variant)
    return model, model.spec()
