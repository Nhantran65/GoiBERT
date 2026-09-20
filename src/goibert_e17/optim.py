"""E17 optimizer adapter over the frozen E7 layer-wise schedule.

The frozen E7 validator treats every non-bias parameter as decayed.  E17 eta
is a scalar vector parameter and is explicitly moved to block-06 no-decay
while retaining the block-06 gamma^3 learning rate.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from goibert_e7 import optim as e7_optim


def build_e17_optimizer(
    model: nn.Module,
    base_lr: float,
    *,
    gamma: float = 0.8,
    num_layers: int = 10,
    weight_decay: float = 0.01,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    optimizer = e7_optim.build_layerwise_optimizer(
        model,
        base_lr,
        gamma=gamma,
        num_layers=num_layers,
        weight_decay=weight_decay,
    )
    expected_names = [
        f"{region}_{kind}"
        for region in ("embeddings", *[f"block_{index:02d}" for index in range(num_layers)], "head")
        for kind in ("decay", "no_decay")
    ]
    eta = getattr(model, "eta", None)
    if eta is None:
        coverage = e7_optim.validate_optimizer_groups(
            model,
            optimizer,
            base_lr=base_lr,
            gamma=gamma,
            num_layers=num_layers,
            weight_decay=weight_decay,
            expected_group_names=expected_names,
        )
        coverage.update({
            "eta_present": False,
            "eta_group": None,
            "eta_weight_decay": None,
            "eta_learning_rate": None,
            "eta_lr_scale": None,
        })
        return optimizer, coverage
    eta_id = id(eta)
    decay_group = None
    no_decay_group = None
    for group in optimizer.param_groups:
        names = list(group.get("e7_param_names", []))
        if eta_id in {id(parameter) for parameter in group["params"]}:
            if group.get("e7_decay"):
                decay_group = group
            else:
                no_decay_group = group
        if group.get("e7_group_name") == "block_06_decay":
            decay_group = group
        if group.get("e7_group_name") == "block_06_no_decay":
            no_decay_group = group
    if decay_group is None or no_decay_group is None:
        raise AssertionError("E17 eta groups were not found")
    decay_group["params"] = [parameter for parameter in decay_group["params"] if id(parameter) != eta_id]
    decay_names = [name for name in decay_group.get("e7_param_names", []) if name != "encoder.encoder.layer.6.attention.self.eta"]
    decay_group["e7_param_names"] = decay_names
    if eta_id not in {id(parameter) for parameter in no_decay_group["params"]}:
        no_decay_group["params"].append(eta)
    no_decay_names = list(no_decay_group.get("e7_param_names", []))
    if "encoder.encoder.layer.6.attention.self.eta" not in no_decay_names:
        no_decay_names.append("encoder.encoder.layer.6.attention.self.eta")
    no_decay_group["e7_param_names"] = no_decay_names

    coverage = e7_optim.validate_optimizer_groups(
        model,
        optimizer,
        base_lr=base_lr,
        gamma=gamma,
        num_layers=num_layers,
        weight_decay=weight_decay,
        expected_group_names=expected_names,
    )
    eta_group = next(row for row in coverage["groups"] if row["name"] == "block_06_no_decay")
    eta_lr = float(base_lr * gamma ** (num_layers - 1 - 6))
    if eta_group["weight_decay"] != 0.0 or not math.isclose(eta_group["lr"], eta_lr, rel_tol=1e-12, abs_tol=1e-15):
        raise AssertionError("E17 eta must use block-06 gamma^3 LR with zero weight decay")
    coverage.update(
        {
            "eta_present": True,
            "eta_group": "block_06_no_decay",
            "eta_weight_decay": 0.0,
            "eta_learning_rate": eta_lr,
            "eta_lr_scale": gamma ** (num_layers - 1 - 6),
            "eta_parameter_name": "encoder.encoder.layer.6.attention.self.eta",
        }
    )
    return optimizer, coverage
