"""AdamW optimizers used for task fine-tuning.

* :func:`build_adamw` is plain AdamW with two groups (weight decay on matrices,
  none on biases and LayerNorm weights).  It is used for the CafeBERT teachers
  and the BamiBERT baseline.
* :func:`build_layerwise_optimizer` adds layer-wise learning-rate decay (LLRD)
  and is used for the GoiBERT students.  Every trainable parameter belongs to
  exactly one group, and each region is split into a decay and a no-decay
  group so the weight-decay rule is the same as in :func:`build_adamw`.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import nn


WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPSILON = 1e-8
DEFAULT_GAMMA = 0.8


def _is_no_decay(name: str, layer_norm_names: set[str] = frozenset()) -> bool:
    """Biases and LayerNorm weights are not decayed."""

    if name in layer_norm_names:
        return True
    leaf = name.rsplit(".", 1)[-1]
    lowered = name.lower()
    return leaf == "bias" or (
        leaf == "weight"
        and ("layernorm" in lowered or "layer_norm" in lowered or ".norm." in lowered)
    )


def _layer_norm_weight_names(model: nn.Module) -> set[str]:
    return {
        f"{module_name}.weight" if module_name else "weight"
        for module_name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
    }


def build_adamw(model: nn.Module, learning_rate: float) -> torch.optim.AdamW:
    """Plain AdamW: one decayed group and one non-decayed group."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    layer_norm_weights = _layer_norm_weight_names(model)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name in layer_norm_weights or _is_no_decay(name) else decay).append(parameter)
    if not decay or not no_decay:
        raise RuntimeError("Expected both decayed and non-decayed parameter groups")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=BETAS,
        eps=EPSILON,
    )


def _region_for_name(name: str, *, num_layers: int) -> tuple[str, int]:
    """Return ``(region, exponent)`` for one classifier parameter name."""

    if name.startswith("head."):
        return "head", 0
    if name.startswith("encoder.embeddings."):
        return "embeddings", num_layers
    prefix = "encoder.encoder.layer."
    if name.startswith(prefix):
        layer_text = name[len(prefix) :].split(".", 1)[0]
        if layer_text.isdigit():
            layer = int(layer_text)
            if 0 <= layer < num_layers:
                return f"block_{layer:02d}", num_layers - 1 - layer
    raise ValueError(f"Cannot assign parameter to embeddings, an encoder block or the head: {name}")


def _group_name(region: str, decay: bool) -> str:
    return f"{region}_{'decay' if decay else 'no_decay'}"


def build_layerwise_optimizer(
    model: nn.Module,
    base_lr: float,
    *,
    num_layers: int,
    gamma: float = DEFAULT_GAMMA,
    weight_decay: float = WEIGHT_DECAY,
    betas: tuple[float, float] = BETAS,
    eps: float = EPSILON,
) -> torch.optim.AdamW:
    """AdamW with layer-wise learning-rate decay.

    With ``L = num_layers``, encoder block ``l`` (0-based) uses
    ``base_lr * gamma ** (L - 1 - l)``, the embeddings use
    ``base_lr * gamma ** L`` and the task head uses ``base_lr``.  The learning
    rate scheduler scales every group by the same factor, so these ratios hold
    throughout training.
    """

    if not math.isfinite(float(base_lr)) or base_lr <= 0:
        raise ValueError(f"base_lr must be finite and positive, got {base_lr!r}")
    if not math.isfinite(float(gamma)) or not (0 < gamma <= 1):
        raise ValueError(f"gamma must be finite and in (0, 1], got {gamma!r}")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    layer_norm_names = _layer_norm_weight_names(model)
    named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named_parameters:
        raise ValueError("The model has no trainable parameters")

    assignments: dict[str, list[tuple[str, nn.Parameter]]] = {}
    for name, parameter in named_parameters:
        region, _ = _region_for_name(name, num_layers=num_layers)
        decay = not _is_no_decay(name, layer_norm_names)
        assignments.setdefault(_group_name(region, decay), []).append((name, parameter))

    regions = ["embeddings"] + [f"block_{layer:02d}" for layer in range(num_layers)] + ["head"]
    expected_group_names = [_group_name(region, decay) for region in regions for decay in (True, False)]

    groups: list[dict[str, Any]] = []
    for region in regions:
        if region == "embeddings":
            exponent = num_layers
        elif region == "head":
            exponent = 0
        else:
            exponent = num_layers - 1 - int(region.split("_")[1])
        scale = float(gamma**exponent)
        group_lr = float(base_lr * scale)
        for decay in (True, False):
            group_name = _group_name(region, decay)
            entries = assignments.get(group_name, [])
            if not entries:
                raise ValueError(f"Expected a non-empty parameter group {group_name}")
            groups.append(
                {
                    "params": [parameter for _, parameter in entries],
                    "lr": group_lr,
                    "initial_lr": group_lr,
                    "weight_decay": float(weight_decay if decay else 0.0),
                    "group_name": group_name,
                    "region": region,
                    "lr_scale": scale,
                    "param_names": [name for name, _ in entries],
                }
            )

    optimizer = torch.optim.AdamW(groups, lr=float(base_lr), betas=betas, eps=float(eps))
    _check_groups(model, optimizer, expected_group_names)
    return optimizer


def _check_groups(model: nn.Module, optimizer: torch.optim.Optimizer, expected_group_names: Iterable[str]) -> None:
    """Every trainable parameter must appear in exactly one group."""

    trainable = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in trainable:
                raise AssertionError("Optimizer contains a parameter that is not a model trainable")
            if id(parameter) in seen:
                raise AssertionError(f"Parameter appears in several groups: {trainable[id(parameter)]}")
            seen.add(id(parameter))
    if seen != set(trainable):
        missing = [name for pid, name in trainable.items() if pid not in seen]
        raise AssertionError(f"Optimizer is missing trainable parameters: {missing}")
    names = [group["group_name"] for group in optimizer.param_groups]
    if names != list(expected_group_names):
        raise AssertionError(f"Unexpected parameter group order: {names}")
