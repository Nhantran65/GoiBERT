"""Function-preserving FFN expansion for the E4 screening experiment.

The original GoiTurn-S-GD encoder is loaded for every trial.  Each Roberta
FFN is widened from 1536 to 3072 units by duplicating the input projection and
splitting the output projection into two cancelling halves.  A small,
deterministic antisymmetric perturbation breaks the duplicated-neuron
symmetry without changing the initial function in exact arithmetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from transformers import RobertaModel

from goibert.modeling import GoiBERTClassifier


ORIGINAL_INTERMEDIATE_SIZE = 1536
EXPANDED_INTERMEDIATE_SIZE = 3072
CONVERSION_SEED = 314159
CONVERSION_NOISE_SCALE = 0.01
EXPANDED_BACKBONE_NAME = "goiturn-s-gd-ffn3072"


class E4ModelSpec:
    """Small adapter that keeps E1's ``to_dict`` contract and adds E4 fields."""

    def __init__(self, base_spec: Any, conversion: dict[str, Any]) -> None:
        self._base_spec = base_spec
        self._conversion = conversion

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self._base_spec.to_dict())
        payload.update(
            {
                "architecture_variant": EXPANDED_BACKBONE_NAME,
                "source_architecture": "goiturn-s-gd",
                "ffn_expansion": self._conversion,
            }
        )
        return payload


def _new_linear_like(
    old: nn.Linear,
    *,
    in_features: int,
    out_features: int,
) -> nn.Linear:
    """Allocate a replacement linear layer without consuming global RNG."""

    # The caller wraps expansion in fork_rng.  Keeping construction here
    # explicit makes it difficult to accidentally replace a layer outside
    # that protected scope later.
    return nn.Linear(
        in_features,
        out_features,
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )


def expand_roberta_ffn(
    encoder: RobertaModel,
    *,
    conversion_seed: int = CONVERSION_SEED,
    noise_scale: float = CONVERSION_NOISE_SCALE,
) -> dict[str, Any]:
    """Expand every encoder FFN and return an auditable conversion summary.

    This function must be called inside ``torch.random.fork_rng`` by the
    model builder.  The perturbation generator is local to CPU and therefore
    does not consume either the process RNG or CUDA RNG used by E1 training.
    """

    config = encoder.config
    old_size = int(config.intermediate_size)
    if old_size != ORIGINAL_INTERMEDIATE_SIZE:
        raise ValueError(
            f"E4 expects intermediate_size={ORIGINAL_INTERMEDIATE_SIZE}, found {old_size}"
        )
    hidden_size = int(config.hidden_size)
    if len(encoder.encoder.layer) != int(config.num_hidden_layers):
        raise ValueError("Encoder layer count does not match its config")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(conversion_seed))
    layer_records: list[dict[str, Any]] = []

    for layer_index, layer in enumerate(encoder.encoder.layer):
        old_intermediate = layer.intermediate.dense
        old_output = layer.output.dense
        if not isinstance(old_intermediate, nn.Linear) or not isinstance(old_output, nn.Linear):
            raise TypeError(f"Layer {layer_index} FFN projections are not Linear modules")
        if tuple(old_intermediate.weight.shape) != (old_size, hidden_size):
            raise ValueError(f"Unexpected intermediate shape at layer {layer_index}")
        if tuple(old_output.weight.shape) != (hidden_size, old_size):
            raise ValueError(f"Unexpected output shape at layer {layer_index}")
        if old_intermediate.bias is None or old_output.bias is None:
            raise ValueError("E4 requires bias terms in both FFN projections")

        # Copy source tensors before replacing modules.  The model is loaded
        # on CPU in the builder, but using the source dtype/device here keeps
        # the conversion valid if that assumption changes.
        w1 = old_intermediate.weight.detach().clone()
        b1 = old_intermediate.bias.detach().clone()
        w2 = old_output.weight.detach().clone()
        b2 = old_output.bias.detach().clone()
        w2_std = float(w2.float().std(unbiased=False).item())
        noise = torch.randn(
            w2.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).mul(noise_scale * w2_std)
        noise = noise.to(device=w2.device, dtype=w2.dtype)

        expanded_intermediate = _new_linear_like(
            old_intermediate,
            in_features=hidden_size,
            out_features=EXPANDED_INTERMEDIATE_SIZE,
        )
        expanded_output = _new_linear_like(
            old_output,
            in_features=EXPANDED_INTERMEDIATE_SIZE,
            out_features=hidden_size,
        )
        with torch.no_grad():
            expanded_intermediate.weight.copy_(torch.cat((w1, w1), dim=0))
            expanded_intermediate.bias.copy_(torch.cat((b1, b1), dim=0))
            expanded_output.weight.copy_(
                torch.cat((w2.mul(0.5) + noise, w2.mul(0.5) - noise), dim=1)
            )
            expanded_output.bias.copy_(b2)
        layer.intermediate.dense = expanded_intermediate
        layer.output.dense = expanded_output
        layer_records.append(
            {
                "layer": layer_index,
                "source_w1_shape": list(w1.shape),
                "source_w2_shape": list(w2.shape),
                "expanded_w1_shape": list(expanded_intermediate.weight.shape),
                "expanded_w2_shape": list(expanded_output.weight.shape),
                "source_w2_std_fp32": w2_std,
                "noise_std": float(noise.float().std(unbiased=False).item()),
            }
        )

    config.intermediate_size = EXPANDED_INTERMEDIATE_SIZE
    return {
        "conversion": "ffn_duplicate_split_antisymmetric_noise",
        "source_intermediate_size": old_size,
        "expanded_intermediate_size": EXPANDED_INTERMEDIATE_SIZE,
        "conversion_seed": int(conversion_seed),
        "noise_scale_relative_to_w2_std": float(noise_scale),
        "generator": "local_cpu_single_generator_advanced_across_layers",
        "layers": layer_records,
    }


def build_expanded_classifier(
    backbone_dir: Path,
    *,
    num_classes: int,
    num_aspects: int = 0,
    aspect_class_counts: Sequence[int] | None = None,
    local_files_only: bool = True,
    attention_implementation: str | None = None,
) -> tuple[GoiBERTClassifier, Any]:
    """Load a fresh student, expand its FFNs, and attach the E1 head.

    Expansion occurs under ``fork_rng`` so construction of replacement Linear
    modules leaves the global RNG at exactly the point where E1 constructs its
    fresh classification head.  This preserves the E1 head/shuffle seed
    comparison while changing only the encoder capacity.
    """

    if not backbone_dir.is_dir():
        raise FileNotFoundError(backbone_dir)
    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "add_pooling_layer": False,
    }
    if attention_implementation is not None:
        kwargs["attn_implementation"] = attention_implementation
    encoder = RobertaModel.from_pretrained(backbone_dir, **kwargs)
    with torch.random.fork_rng(devices=[]):
        conversion = expand_roberta_ffn(encoder)
    if int(encoder.config.intermediate_size) != EXPANDED_INTERMEDIATE_SIZE:
        raise RuntimeError("FFN expansion did not update the encoder config")
    if aspect_class_counts is not None and num_classes != sum(aspect_class_counts):
        raise ValueError(
            f"num_classes={num_classes} does not match aspect_class_counts sum="
            f"{sum(aspect_class_counts)}"
        )
    model = GoiBERTClassifier(
        encoder,
        num_labels=num_classes,
        num_aspects=num_aspects,
        aspect_class_counts=aspect_class_counts,
        backbone_name=EXPANDED_BACKBONE_NAME,
    )
    return model, E4ModelSpec(model.spec(), conversion)

