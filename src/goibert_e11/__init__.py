"""E11 signed versus convex two-map attention adaptation."""

from .modeling import (
    E11Classifier,
    E11ModelSpec,
    INSERTION_LAYER_INDEX,
    VARIANTS,
    build_e11_student,
    iter_e11_attentions,
)

__all__ = [
    "E11Classifier",
    "E11ModelSpec",
    "INSERTION_LAYER_INDEX",
    "VARIANTS",
    "build_e11_student",
    "iter_e11_attentions",
]
