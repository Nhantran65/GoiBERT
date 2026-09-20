"""Optimizer adapter for E7 layer-wise learning-rate decay.

The E1 model and training loop stay frozen.  This module only constructs the
AdamW parameter groups used by E7.  Every trainable parameter belongs to one
and only one group.  Groups are split into decay and no-decay variants so the
E1 bias/LayerNorm weight-decay contract is preserved exactly.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
from torch import nn


E7_DEFAULT_GAMMA = 0.8
E7_NUM_LAYERS = 10
E7_WEIGHT_DECAY = 0.01
E7_BETAS = (0.9, 0.999)
E7_EPSILON = 1e-8


def _is_no_decay(name: str, layer_norm_names: set[str]) -> bool:
    """Match the frozen E1 no-decay predicate, including LayerNorm names."""

    if name in layer_norm_names:
        return True
    leaf = name.rsplit(".", 1)[-1]
    lowered = name.lower()
    # Keep the same precedence as src/goibert/train.py::_is_no_decay.
    return leaf == "bias" or (
        leaf == "weight"
        and ("layernorm" in lowered or "layer_norm" in lowered or ".norm." in lowered)
    )


def _region_for_name(name: str, *, num_layers: int) -> tuple[str, int]:
    """Return ``(region, exponent)`` for one E1 classifier parameter name."""

    if name.startswith("head."):
        return "head", 0
    if name.startswith("encoder.embeddings."):
        return "embeddings", num_layers
    prefix = "encoder.encoder.layer."
    if name.startswith(prefix):
        rest = name[len(prefix) :]
        layer_text = rest.split(".", 1)[0]
        if layer_text.isdigit():
            layer = int(layer_text)
            if 0 <= layer < num_layers:
                return f"block_{layer:02d}", num_layers - 1 - layer
    raise ValueError(
        "E7 could not assign parameter to embeddings, encoder block, or head: "
        f"{name}"
    )


def _group_name(region: str, decay: bool) -> str:
    return f"{region}_{'decay' if decay else 'no_decay'}"


def build_layerwise_optimizer(
    model: nn.Module,
    base_lr: float,
    *,
    gamma: float = E7_DEFAULT_GAMMA,
    num_layers: int = E7_NUM_LAYERS,
    weight_decay: float = E7_WEIGHT_DECAY,
    betas: tuple[float, float] = E7_BETAS,
    eps: float = E7_EPSILON,
) -> torch.optim.AdamW:
    """Construct E7's AdamW with layer-wise learning-rate decay.

    ``encoder.encoder.layer.0`` receives ``base_lr * gamma**9`` and
    ``encoder.encoder.layer.9`` plus the classification head receive
    ``base_lr``.  Embeddings receive ``base_lr * gamma**10``.  The same
    scheduler multiplier is later applied to all groups by E1's frozen
    ``get_linear_schedule_with_warmup`` call, so these ratios remain fixed.
    """

    if not math.isfinite(float(base_lr)) or base_lr <= 0:
        raise ValueError(f"base_lr must be finite and positive, got {base_lr!r}")
    if not math.isfinite(float(gamma)) or not (0 < gamma <= 1):
        raise ValueError(f"gamma must be finite and in (0, 1], got {gamma!r}")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    layer_norm_names = {
        f"{module_name}.weight" if module_name else "weight"
        for module_name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
    }
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("E7 requires at least one trainable parameter")

    assignments: dict[str, list[tuple[str, nn.Parameter]]] = {}
    for name, parameter in named_parameters:
        region, exponent = _region_for_name(name, num_layers=num_layers)
        decay = not _is_no_decay(name, layer_norm_names)
        group_name = _group_name(region, decay)
        assignments.setdefault(group_name, []).append((name, parameter))

    expected_regions = ["embeddings"] + [f"block_{layer:02d}" for layer in range(num_layers)] + ["head"]
    expected_group_names = [
        _group_name(region, decay)
        for region in expected_regions
        for decay in (True, False)
    ]
    missing_regions = [
        region
        for region in expected_regions
        if not any(name.startswith(f"{region}_") for name in assignments)
    ]
    if missing_regions:
        raise ValueError(f"E7 found no trainable parameters for regions: {missing_regions}")

    groups: list[dict[str, Any]] = []
    for region in expected_regions:
        if region == "embeddings":
            exponent = num_layers
        elif region == "head":
            exponent = 0
        else:
            layer = int(region.split("_")[1])
            exponent = num_layers - 1 - layer
        scale = float(gamma**exponent)
        group_lr = float(base_lr * scale)
        for decay in (True, False):
            group_name = _group_name(region, decay)
            entries = assignments.get(group_name, [])
            if not entries:
                # A missing decay half is usually a model-layout bug.  Fail
                # closed instead of silently changing the E7 parameterization.
                raise ValueError(f"E7 expected non-empty parameter group {group_name}")
            params = [parameter for _, parameter in entries]
            groups.append(
                {
                    "params": params,
                    "lr": group_lr,
                    "initial_lr": group_lr,
                    "weight_decay": float(weight_decay if decay else 0.0),
                    "e7_group_name": group_name,
                    "e7_region": region,
                    "e7_lr_scale": scale,
                    "e7_exponent": exponent,
                    "e7_decay": decay,
                    # Retaining names in the optimizer state makes coverage
                    # auditable after the checkpoint is saved.
                    "e7_param_names": [name for name, _ in entries],
                }
            )

    optimizer = torch.optim.AdamW(
        groups,
        lr=float(base_lr),
        betas=betas,
        eps=float(eps),
    )
    validate_optimizer_groups(
        model,
        optimizer,
        base_lr=base_lr,
        gamma=gamma,
        num_layers=num_layers,
        weight_decay=weight_decay,
        expected_group_names=expected_group_names,
    )
    return optimizer


def describe_optimizer_groups(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Return JSON-safe group metadata, including every parameter name."""

    rows: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        rows.append(
            {
                "index": index,
                "name": group.get("e7_group_name"),
                "region": group.get("e7_region"),
                "decay": bool(group.get("e7_decay")),
                "lr": float(group["lr"]),
                "initial_lr": float(group.get("initial_lr", group["lr"])),
                "lr_scale": float(group.get("e7_lr_scale", 1.0)),
                "exponent": int(group.get("e7_exponent", 0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_count": len(group["params"]),
                "parameter_names": list(group.get("e7_param_names", [])),
            }
        )
    return rows


def validate_optimizer_groups(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    gamma: float = E7_DEFAULT_GAMMA,
    num_layers: int = E7_NUM_LAYERS,
    weight_decay: float = E7_WEIGHT_DECAY,
    expected_group_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fail closed and return coverage/endpoint checks for preflight."""

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    ids_to_names = {id(parameter): name for name, parameter in trainable.items()}
    seen: list[int] = []
    group_rows = describe_optimizer_groups(optimizer)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id not in ids_to_names:
                raise AssertionError("optimizer contains a parameter outside model trainables")
            if parameter_id in seen:
                raise AssertionError(f"parameter appears in multiple E7 groups: {ids_to_names[parameter_id]}")
            seen.append(parameter_id)
    expected_ids = {id(parameter) for parameter in trainable.values()}
    if set(seen) != expected_ids:
        missing = [name for name, parameter in trainable.items() if id(parameter) not in set(seen)]
        raise AssertionError(f"E7 optimizer missed trainable parameters: {missing}")
    actual_names = [str(row["name"]) for row in group_rows]
    if expected_group_names is not None and actual_names != list(expected_group_names):
        raise AssertionError(f"Unexpected E7 group order/names: {actual_names}")

    expected_scales: dict[str, float] = {"embeddings": gamma**num_layers, "head": 1.0}
    expected_scales.update(
        {f"block_{layer:02d}": gamma ** (num_layers - 1 - layer) for layer in range(num_layers)}
    )
    for row in group_rows:
        region = str(row["region"])
        expected_scale = expected_scales[region]
        if not math.isclose(float(row["lr_scale"]), expected_scale, rel_tol=1e-12, abs_tol=1e-15):
            raise AssertionError(f"E7 scale mismatch for {region}: {row['lr_scale']} vs {expected_scale}")
        expected_lr = float(base_lr * expected_scale)
        if not math.isclose(float(row["lr"]), expected_lr, rel_tol=1e-12, abs_tol=1e-15):
            raise AssertionError(f"E7 LR mismatch for {region}: {row['lr']} vs {expected_lr}")
        expected_wd = weight_decay if row["decay"] else 0.0
        if not math.isclose(float(row["weight_decay"]), expected_wd, rel_tol=0, abs_tol=1e-15):
            raise AssertionError(f"E7 weight decay mismatch for {row['name']}")

    return {
        "all_trainable": len(seen) == len(expected_ids),
        "parameter_count": len(expected_ids),
        "group_count": len(group_rows),
        "duplicate_parameter_count": len(seen) - len(set(seen)),
        "group_names": actual_names,
        "endpoint_scales": {
            "embeddings": expected_scales["embeddings"],
            "block_00": expected_scales["block_00"],
            "block_09": expected_scales["block_09"],
            "head": expected_scales["head"],
        },
        "groups": group_rows,
    }

