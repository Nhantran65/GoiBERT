"""E7 layer-wise learning-rate-decay experiment adapters."""

from .optim import (
    E7_DEFAULT_GAMMA,
    E7_NUM_LAYERS,
    build_layerwise_optimizer,
    describe_optimizer_groups,
    validate_optimizer_groups,
)

__all__ = [
    "E7_DEFAULT_GAMMA",
    "E7_NUM_LAYERS",
    "build_layerwise_optimizer",
    "describe_optimizer_groups",
    "validate_optimizer_groups",
]
