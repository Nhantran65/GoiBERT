"""Identity-gated FFN variants for the original BamiBERT encoder.

Only encoder block index 7 is replaced.  The replacement keeps the original
attention, FFN parameter names, dropout, residual and LayerNorm path while
capturing the attention projection output needed by ``attention_update_gate``.
The two gates are deliberately small in scope: one zero-initialized
3072x768 matrix and no bias, with the same 32-bit correction path for both
variants.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaAttention, RobertaLayer, RobertaSelfOutput


INSERTION_LAYER_INDEX = 7
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 3072
NUM_HEADS = 12
HEAD_DROPOUT = 0.1
HEAD_HIDDEN_SIZE = 768
ABSA_NUM_CLASSES = 4
GATE_PARAMETER_COUNT = INTERMEDIATE_SIZE * HIDDEN_SIZE
ARMS = ("baseline", "attention_update_gate", "ffn_input_gate")


@dataclass(frozen=True)
class E16ModelSpec:
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
    arm: str
    insertion_layer_index: int
    gate_shape: tuple[int, int] | None
    gate_bias: bool
    gate_init: str
    calibration_sc: float
    calibration_sx: float
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


class E16SelfOutput(RobertaSelfOutput):
    """Original attention output with a per-forward ``c`` capture.

    ``c`` is exactly ``Wo(attention_context)`` before the original output
    dropout, residual and LayerNorm.  It is cleared by the target layer after
    its FFN has consumed it, so a graph is never retained across batches.
    """

    def __init__(self, source: RobertaSelfOutput) -> None:
        nn.Module.__init__(self)
        self.dense = source.dense
        self.dropout = source.dropout
        self.LayerNorm = source.LayerNorm
        self.last_c: Tensor | None = None

    def clear_capture(self) -> None:
        self.last_c = None

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        c = self.dense(hidden_states)
        self.last_c = c
        return self.LayerNorm(self.dropout(c) + input_tensor)


class E16RobertaAttention(nn.Module):
    """Attention wrapper retaining all original child parameter names."""

    def __init__(self, source: RobertaAttention) -> None:
        super().__init__()
        self.self = source.self
        self.output = E16SelfOutput(source.output)
        self.pruned_heads = source.pruned_heads

    def prune_heads(self, heads: Any) -> None:
        raise NotImplementedError("E16 does not support pruning after construction")

    def clear_capture(self) -> None:
        self.output.clear_capture()

    @property
    def last_c(self) -> Tensor | None:
        return self.output.last_c

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
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class E16TargetLayer(RobertaLayer):
    """Block 7 with the target FFN fusion and the original residual path."""

    def __init__(
        self,
        source: RobertaLayer,
        *,
        arm: str,
        sc: float,
        sx: float,
        config: Any,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"Unknown E16 arm {arm!r}")
        # Avoid constructing randomly initialized modules: all original child
        # modules are moved into this wrapper and retain their exact weights.
        nn.Module.__init__(self)
        self.chunk_size_feed_forward = source.chunk_size_feed_forward
        self.seq_len_dim = source.seq_len_dim
        self.attention = E16RobertaAttention(source.attention)
        self.is_decoder = source.is_decoder
        self.add_cross_attention = source.add_cross_attention
        if self.add_cross_attention and hasattr(source, "crossattention"):
            self.crossattention = source.crossattention
        self.intermediate = source.intermediate
        self.output = source.output
        self.arm = str(arm)
        self.sc = float(sc)
        self.sx = float(sx)
        self.register_buffer("calibration_sc", torch.tensor(float(sc), dtype=torch.float32))
        self.register_buffer("calibration_sx", torch.tensor(float(sx), dtype=torch.float32))
        if self.arm == "baseline":
            self.gate = None
        else:
            self.gate = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
            nn.init.zeros_(self.gate.weight)
        self.collect_diagnostics = False
        self.last_probe: dict[str, float] = {}
        self.last_dtypes: dict[str, str] = {}

    @property
    def added_parameter_count(self) -> int:
        return 0 if self.gate is None else int(self.gate.weight.numel())

    def set_diagnostics(self, enabled: bool) -> None:
        self.collect_diagnostics = bool(enabled)

    def clear_capture(self) -> None:
        self.attention.clear_capture()

    def _fused_ffn(self, x: Tensor, c: Tensor | None) -> Tensor:
        # f0 is intentionally obtained through the original BF16 autocast
        # path.  The correction and the final fusion are always FP32.
        z = self.intermediate.dense(x)
        z = self.intermediate.intermediate_act_fn(z)
        f0 = self.output.dense(z)
        if self.gate is None:
            fused = f0.float()
            self.last_probe = {"s_rms": 0.0, "s_abs_p95": 0.0, "correction_norm": 0.0, "f0_norm": float(fused.detach().float().norm().item())}
        else:
            source = c if self.arm == "attention_update_gate" else x
            if source is None:
                raise RuntimeError("E16 gate source was not captured for target FFN")
            scalar = self.calibration_sc if self.arm == "attention_update_gate" else self.calibration_sx
            # No autocast is allowed in the gate/correction path.  This also
            # guarantees that W2 is explicitly FP32 for F.linear.
            with torch.autocast(device_type=x.device.type, enabled=False):
                v = source.float() / scalar.float().clamp_min(1e-6)
                s = torch.tanh(self.gate(v))
                delta = torch.nn.functional.linear(z.float() * s, self.output.dense.weight.float(), bias=None)
                fused = f0.float() + delta
            if self.collect_diagnostics:
                with torch.no_grad():
                    values = s.detach().float()
                    self.last_probe = {
                        "s_rms": float(values.square().mean().sqrt().item()),
                        "s_abs_p95": float(values.abs().flatten().quantile(0.95).item()),
                        "correction_norm": float(delta.detach().float().norm().item()),
                        "f0_norm": float(f0.detach().float().norm().item()),
                        "delta_relative_norm": float(delta.detach().float().norm().item() / f0.detach().float().norm().clamp_min(1e-12).item()),
                    }
                    self.last_dtypes = {"gate_source": str(v.dtype), "gate_signal": str(s.dtype), "correction": str(delta.dtype), "fused": str(fused.dtype)}
        return self.output.LayerNorm(self.output.dropout(fused) + x)

    def feed_forward_chunk(self, attention_output: Tensor) -> Tensor:
        c = self.attention.last_c
        return self._fused_ffn(attention_output, c)

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
        self.clear_capture()
        try:
            self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            attention_output = self_attention_outputs[0]
            if self.is_decoder:
                outputs = self_attention_outputs[1:-1]
                present_key_value = self_attention_outputs[-1]
            else:
                outputs = self_attention_outputs[1:]
            cross_attn_present_key_value = None
            if self.is_decoder and encoder_hidden_states is not None:
                if not hasattr(self, "crossattention"):
                    raise ValueError("E16 decoder layer has no cross attention")
                cross_past = past_key_value[-2:] if past_key_value is not None else None
                cross_outputs = self.crossattention(
                    attention_output, attention_mask, head_mask, encoder_hidden_states,
                    encoder_attention_mask, cross_past, output_attentions,
                )
                attention_output = cross_outputs[0]
                outputs = outputs + cross_outputs[1:-1]
                cross_attn_present_key_value = cross_outputs[-1]
                present_key_value = present_key_value + cross_attn_present_key_value
            layer_output = self.feed_forward_chunk(attention_output)
            outputs = (layer_output,) + outputs
            if self.is_decoder:
                outputs = outputs + (present_key_value,)
            return outputs
        finally:
            # The local tensor has already been consumed by the FFN.  Drop it
            # before returning so no autograd graph survives the batch.
            self.clear_capture()


class BamiBERTE16Classifier(nn.Module):
    def __init__(
        self,
        encoder: RobertaModel,
        *,
        num_labels: int,
        num_aspects: int,
        aspect_class_counts: Sequence[int] | None,
        arm: str,
        sc: float,
        sx: float,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown E16 arm {arm!r}")
        counts = tuple(int(value) for value in (aspect_class_counts or ()))
        if counts and len(counts) != num_aspects:
            raise ValueError("aspect_class_counts length must equal num_aspects")
        if counts and sum(counts) != num_labels:
            raise ValueError("num_labels must equal sum(aspect_class_counts)")
        if num_aspects and not counts and num_labels != num_aspects * ABSA_NUM_CLASSES:
            raise ValueError("uniform ABSA output size mismatch")
        if int(encoder.config.hidden_size) != HIDDEN_SIZE or int(encoder.config.intermediate_size) != INTERMEDIATE_SIZE:
            raise ValueError("E16 requires hidden_size=768 and intermediate_size=3072")
        self.encoder = encoder
        self.num_labels = int(num_labels)
        self.num_aspects = int(num_aspects)
        self.aspect_class_counts = counts or ((ABSA_NUM_CLASSES,) * num_aspects)
        self.head = ClassificationHead(HIDDEN_SIZE, self.num_labels)
        self.arm = str(arm)
        self.insertion_layer_index = INSERTION_LAYER_INDEX
        self.calibration_sc = float(sc)
        self.calibration_sx = float(sx)
        self.register_buffer("shared_sc", torch.tensor(float(sc), dtype=torch.float32))
        self.register_buffer("shared_sx", torch.tensor(float(sx), dtype=torch.float32))
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(True)
        source_layer = self.encoder.encoder.layer[INSERTION_LAYER_INDEX]
        replacement = E16TargetLayer(source_layer, arm=arm, sc=sc, sx=sx, config=encoder.config)
        self.encoder.encoder.layer[INSERTION_LAYER_INDEX] = replacement

    @property
    def target_layer(self) -> E16TargetLayer:
        value = self.encoder.encoder.layer[self.insertion_layer_index]
        if not isinstance(value, E16TargetLayer):
            raise TypeError("E16 target layer was replaced after construction")
        return value

    @property
    def gate(self) -> nn.Linear | None:
        return self.target_layer.gate

    def set_diagnostics(self, enabled: bool) -> None:
        self.target_layer.set_diagnostics(enabled)

    def diagnostics(self) -> dict[str, float]:
        return dict(self.target_layer.last_probe)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, *, token_type_ids: Tensor | None = None) -> Tensor:
        kwargs: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs, return_dict=True)
        logits = self.head(outputs.last_hidden_state)
        if self.num_aspects and len(set(self.aspect_class_counts)) == 1:
            logits = logits.reshape(logits.shape[0], self.num_aspects, self.aspect_class_counts[0])
        return logits

    def spec(self) -> E16ModelSpec:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        config = self.encoder.config
        return E16ModelSpec(
            role="bamibert_original_e16",
            backbone="Qualcomm-AI-Research/BamiBERT",
            hidden_size=int(config.hidden_size),
            num_hidden_layers=int(config.num_hidden_layers),
            num_attention_heads=int(config.num_attention_heads),
            intermediate_size=int(config.intermediate_size),
            head_hidden_size=HEAD_HIDDEN_SIZE,
            head_dropout=HEAD_DROPOUT,
            num_labels=self.num_labels,
            num_aspects=self.num_aspects,
            aspect_class_counts=self.aspect_class_counts,
            total_parameters=total,
            trainable_parameters=trainable,
            architecture="original_bamibert_sdpa_plus_block7_identity_gated_ffn",
            arm=self.arm,
            insertion_layer_index=INSERTION_LAYER_INDEX,
            gate_shape=(INTERMEDIATE_SIZE, HIDDEN_SIZE) if self.gate is not None else None,
            gate_bias=False,
            gate_init="zero" if self.gate is not None else "none",
            calibration_sc=self.calibration_sc,
            calibration_sx=self.calibration_sx,
            added_parameter_count=0 if self.gate is None else GATE_PARAMETER_COUNT,
        )


def build_e16_classifier(
    model_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    arm: str,
    calibration_sc: float,
    calibration_sx: float,
    attention_implementation: str = "sdpa",
) -> tuple[BamiBERTE16Classifier, E16ModelSpec]:
    if attention_implementation != "sdpa":
        raise ValueError("E16 is locked to the SDPA backend")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not (math.isfinite(float(calibration_sc)) and math.isfinite(float(calibration_sx)) and calibration_sc >= 1e-6 and calibration_sx >= 1e-6):
        raise ValueError("E16 calibration scalars must be finite and >= 1e-6")
    encoder = RobertaModel.from_pretrained(
        model_dir,
        local_files_only=True,
        add_pooling_layer=False,
        attn_implementation=attention_implementation,
    )
    config = encoder.config
    expected = {"hidden_size": HIDDEN_SIZE, "num_hidden_layers": 12, "num_attention_heads": NUM_HEADS, "intermediate_size": INTERMEDIATE_SIZE}
    actual = {name: int(getattr(config, name)) for name in expected}
    if actual != expected:
        raise RuntimeError(f"Official BamiBERT architecture mismatch: expected {expected}, got {actual}")
    # Head construction remains in the same position as the original E14
    # builder. Replacement modules contain no random initialization except a
    # deterministic zero gate, preserving the fresh-head RNG stream.
    model = BamiBERTE16Classifier.__new__(BamiBERTE16Classifier)
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
    model.head = ClassificationHead(HIDDEN_SIZE, int(num_classes))
    # Explicitly preserve the post-head RNG state even if future constructor
    # edits add a random draw in a replacement wrapper.
    post_head_state = torch.get_rng_state()
    replacement = E16TargetLayer(model.encoder.encoder.layer[INSERTION_LAYER_INDEX], arm=arm, sc=calibration_sc, sx=calibration_sx, config=config)
    model.encoder.encoder.layer[INSERTION_LAYER_INDEX] = replacement
    torch.set_rng_state(post_head_state)
    model.arm = arm
    model.insertion_layer_index = INSERTION_LAYER_INDEX
    model.calibration_sc = float(calibration_sc)
    model.calibration_sx = float(calibration_sx)
    model.register_buffer("shared_sc", torch.tensor(float(calibration_sc), dtype=torch.float32))
    model.register_buffer("shared_sx", torch.tensor(float(calibration_sx), dtype=torch.float32))
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    if model.gate is not None and tuple(model.gate.weight.shape) != (INTERMEDIATE_SIZE, HIDDEN_SIZE):
        raise AssertionError("E16 gate shape mismatch")
    return model, model.spec()
